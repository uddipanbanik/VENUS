#pragma once

// Choose and record the optional single-GPU subhalo field clocks.
// Inputs are the common mesh, a current particle state, two field observations,
// and the elapsed interval. The gravitational bound uses resolved cell force;
// the field-change clock uses a complete mass-weighted receiver measure in
// its supported EdS setup and the supplied maximum-norm clock elsewhere.
// Outputs are selected timescales and optional diagnostic files. Observation
// leaves particles, collision ages, and random counters unchanged.

#include "receiver_sums.hpp"
#include "venus_gpu/vlasov_poisson.hpp"
#include "ev/weighted_pic_timestep_policy.hpp"
#include "ev/weighted_vlasov_pic_event_scheduler.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace venus_gpu_receiver {
// Compare the imposed maximum step with a fraction of the shortest
// resolved gravitational time, [abs(dPhi/dr)/r + H_*^2]^(-1/2). H_* is the
// prescribed 2/[3(t+12)] numerical frequency envelope in both backgrounds.
inline double gravitational_timestep(const venus_gpu::FieldHost& field,
        const venus_gpu::RadialMesh& mesh, double time, double maximum,
        double fraction) {
    require(field.shell_mass.size() == mesh.cells()
        && field.phi_r.size() == mesh.cells()
        && std::isfinite(time) && time > -12.,
        "Invalid gravitational cell clocks.");
    const double deposited = field.diagnostics.deposited_mesh_mass;
    require(std::isfinite(deposited) && deposited > 0,
        "Gravitational clock requires a positive source ledger.");
    // Cells below this fraction of deposited source mass do not set this
    // orbit bound. Their particles still deposit mass and follow the push.
    const double floor = 1e-12 * deposited;
    const double hubble = (2./3.)/(time+12.);
    double result = maximum;
    for (std::size_t i=0; i<mesh.cells(); ++i) {
        require(std::isfinite(field.shell_mass[i]) && field.shell_mass[i]>=0
            && std::isfinite(mesh.radius_center[i]) && mesh.radius_center[i]>0
            && std::isfinite(field.phi_r[i]),"Invalid gravitational cell.");
        if (!(field.shell_mass[i]>floor)) continue;
        result = std::min(result,ev::weighted_pic_timestep_policy::force_h_clock(
            std::abs(field.phi_r[i])/mesh.radius_center[i],hubble*hubble,
            fraction,maximum,true));
    }
    require(std::isfinite(result) && result>0,"Invalid gravitational timestep.");
    return result;
}

class CoupledReceiverClocks {
public:
    CoupledReceiverClocks(std::string directory,bool trace)
        : directory_(std::move(directory)),trace_(trace) {
        if (trace_) {
            history_.open(directory_+"/receiver_clock_history.csv");
            decisions_.open(directory_+"/receiver_cadence_history.csv");
            require(bool(history_) && bool(decisions_),"Cannot open receiver diagnostics.");
            history_ << "observation,phase,step,time,dt,maximum_time,selected_time,fallback,"
                "resolved_neff,physical_mass,physical_mass_square,hidden_mass,hidden_count,"
                "hidden_sample_count,hidden_samples_complete,hidden_mass_residual,total_mass_residual,potential_relative,force_relative,noise,"
                "phi0_square,phi1_square,delta_phi_square,force0_square,force1_square,delta_force_square,"
                "previous_inner_phi,current_inner_phi,previous_inner_mass,current_inner_mass,"
                "cumulative_receiver_seconds,cumulative_trace_seconds\n";
            decisions_ << "step,time,dt,age,deadline,controlling,accept,apply_now,overdue,retry\n";
        }
    }

    // Compare fields at fixed current receiver positions. This is a
    // temporal field observation, separate from moving the particle orbits.
    double observe(const venus_gpu::WpicState& state,
            const venus_gpu::FieldHost& previous,const venus_gpu::FieldHost& current,
            const venus_gpu::RadialMesh& mesh,const venus_gpu::VlasovPoissonOptions& options,
            double previous_inner_phi,double current_inner_phi,double dt,
            double maximum_time,const char* phase,std::int64_t step,double time) {
        const auto start=std::chrono::steady_clock::now();
        // The hidden-mass continuation and receiver normalization below
        // require this explicit background, deposition, and central-mass setup.
        const bool supported=options.central_mass==0 && options.gravity>0
            && options.particle_shape==venus_gpu::ParticleShape::tsc
            && options.carry_active_mass_inside_inner_face
            && !options.subtract_homogeneous_background
            && options.expansion.enabled && options.expansion.alpha==2./3.
            && options.expansion.time_offset==12.;
        MixedClock result;
        DeviceReceiverResult receiver;
        const auto cells=mesh.cells();
        // Other supported solver setups retain the caller's maximum-norm
        // variation time, including the static-background branch.
        if (!supported) {
            result.maximum_fallback=true;
            result.measured.timescale=maximum_time;
        } else {
            require(cells>0 && mesh.edge.size()==cells+1 && mesh.q_center.size()==cells
                && previous.source_mass_edge.size()==cells+1
                && current.source_mass_edge.size()==cells+1,
                "Receiver field/source dimensions differ.");
            InnerField before{mesh.edge.front(),previous_inner_phi,
                previous.source_mass_edge.front(),options.gravity};
            InnerField after{mesh.edge.front(),current_inner_phi,
                current.source_mass_edge.front(),options.gravity};
            // Bound diagnostic sample transfers while keeping the complete
            // device reduction for every observation.
            const std::size_t samples=trace_ && observation_<17 ? 64 : 0;
            receiver=receivers_.observe(state,mesh.edge.back(),mesh.q_center.front(),
                before,after,samples);
            try {
                result=finish_clock(previous.phi,previous.phi_r,current.phi,current.phi_r,
                    current.shell_mass,current.effective_marker_number,dt,
                    current.diagnostics.hidden_inner_mass,maximum_time,receiver.sum,true);
            } catch (...) {
                // Preserve the first cause after a bounded read-only sample.
                try {
                    receiver=receivers_.observe(state,mesh.edge.back(),mesh.q_center.front(),before,after,64);
                    cache_fields(previous,current,mesh,receiver.first_hidden_samples);
                    std::ofstream failure(directory_+"/receiver_failure_summary.json");
                    failure<<std::setprecision(21)
                        <<"{\"diagnostic_only\":true,\"phase\":\""<<phase
                        <<"\",\"step\":"<<step<<",\"time_argument\":"<<time<<",\"dt\":"<<dt
                        <<",\"expected_hidden_mass\":"<<current.diagnostics.hidden_inner_mass
                        <<",\"receiver_hidden_mass\":"<<receiver.sum.hidden_mass
                        <<",\"hidden_mass_residual\":"<<receiver.sum.hidden_mass-current.diagnostics.hidden_inner_mass
                        <<",\"physical_mass\":"<<receiver.sum.physical_mass
                        <<",\"active_count\":"<<receiver.sum.active_count
                        <<",\"hidden_count\":"<<receiver.sum.hidden_count
                        <<",\"hidden_samples\":"<<receiver.first_hidden_samples.size()
                        <<",\"hidden_samples_complete\":"<<(receiver.first_hidden_samples.size()==receiver.sum.hidden_count?"true":"false")
                        <<",\"observation\":"<<observation_<<"}\n";
                    failure.flush();
                } catch (...) {}
                throw;
            }
        }
        receiver_seconds_+=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count()-receiver.sample_seconds;
        trace_seconds_+=receiver.sample_seconds;
        // Record the actual field pair, closure residuals, and measured time
        // used by the cadence policy, so a rejected trial can be inspected.
        if (trace_) {
            const auto trace_start=std::chrono::steady_clock::now();
            history_ << std::setprecision(21) << observation_ << ',' << phase << ','
                << step << ',' << time << ',' << dt << ',' << maximum_time << ','
                << result.measured.timescale << ',' << result.maximum_fallback << ','
                << result.measured.effective_marker_number << ','
                << receiver.sum.physical_mass << ',' << receiver.sum.physical_mass2 << ','
                << receiver.sum.hidden_mass << ',' << receiver.sum.hidden_count << ','
                << receiver.first_hidden_samples.size() << ','
                << (receiver.first_hidden_samples.size()==receiver.sum.hidden_count) << ','
                << result.hidden_mass_residual << ',' << result.total_mass_residual << ','
                << result.measured.potential_relative_change << ','
                << result.measured.force_relative_change << ','
                << result.measured.relative_noise_floor << ','
                << receiver.sum.phi0_square << ',' << receiver.sum.phi1_square << ','
                << receiver.sum.phi_difference_square << ','
                << receiver.sum.force0_square << ',' << receiver.sum.force1_square << ','
                << receiver.sum.force_difference_square << ','
                << previous_inner_phi << ',' << current_inner_phi << ','
                << (previous.source_mass_edge.empty()?0:previous.source_mass_edge.front()) << ','
                << (current.source_mass_edge.empty()?0:current.source_mass_edge.front()) << ','
                << receiver_seconds_ << ',' << trace_seconds_ << '\n';
            if (observation_<17 && supported)
                cache_fields(previous,current,mesh,receiver.first_hidden_samples);
            history_.flush();
            require(bool(history_),"Receiver history write failed.");
            trace_seconds_+=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-trace_start).count();
        }
        ++observation_;
        return result.measured.timescale;
    }

    // The caller supplies its cadence decision after comparing the carried
    // collision age with the newly selected endpoint deadline.
    void decision(std::int64_t step,double time,double dt,double age,
            const ev::WeightedPICEndpointCadenceDecision& x) {
        if (!trace_) return;
        const auto start=std::chrono::steady_clock::now();
        decisions_ << std::setprecision(17) << step << ',' << time << ',' << dt << ','
            << age << ',' << x.selected_interval << ',' << x.controlling_timescale << ','
            << x.accept_step << ',' << x.apply_collision_now << ',' << x.unresolved_overdue
            << ',' << x.retry_interval << '\n';
        decisions_.flush();
        require(bool(decisions_),"Receiver cadence history write failed.");
        trace_seconds_+=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
    }

    // Finish writes observational work totals after all cadence decisions.
    void finish() {
        if (!trace_) return;
        std::ofstream output(directory_+"/receiver_timing.json");
        output << std::setprecision(17)
            << "{\"observations\":" << observation_
            << ",\"receiver_reduction_and_clock_seconds\":" << receiver_seconds_
            << ",\"sampling_and_trace_seconds\":" << trace_seconds_ << "}\n";
        output.flush();
        require(bool(output),"Receiver final timing write failed.");
    }

private:
    // Cache a complete radial field pair and a bounded set of interior
    // receivers; these files describe the observer inputs at one epoch.
    void cache_fields(const venus_gpu::FieldHost& a,const venus_gpu::FieldHost& b,
            const venus_gpu::RadialMesh& mesh,const std::vector<ReceiverSample>& samples) {
        std::ostringstream name;
        name << directory_ << "/receiver_pair_" << std::setw(5) << std::setfill('0') << observation_;
        std::ofstream field(name.str()+".csv"),hidden(name.str()+"_hidden.csv");
        require(bool(field)&&bool(hidden),"Cannot open bounded receiver cache.");
        field << "cell,r,left,right,mass,neff,phi0,phi1,force0,force1\n" << std::setprecision(17);
        for (std::size_t i=0;i<mesh.cells();++i)
            field << i << ',' << mesh.radius_center[i] << ',' << mesh.edge[i] << ','
                << mesh.edge[i+1] << ',' << b.shell_mass[i] << ','
                << b.effective_marker_number[i] << ',' << a.phi[i] << ',' << b.phi[i]
                << ',' << a.phi_r[i] << ',' << b.phi_r[i] << '\n';
        hidden << "index,id,r,mass,hidden_weight,phi0,phi1,force0,force1\n" << std::setprecision(17);
        for (const auto& x:samples)
            hidden << x.index << ',' << x.id << ',' << x.radius << ',' << x.mass << ','
                << x.hidden_weight << ',' << x.previous_potential << ','
                << x.current_potential << ',' << x.previous_force << ',' << x.current_force << '\n';
        field.flush();hidden.flush();
        require(bool(field)&&bool(hidden),"Bounded receiver cache write failed.");
    }
    std::string directory_;
    bool trace_;
    DeviceReceivers receivers_;
    std::ofstream history_,decisions_;
    std::size_t observation_=0;
    double receiver_seconds_=0,trace_seconds_=0;
};
} // namespace venus_gpu_receiver
