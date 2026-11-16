#pragma once

// Evaluate and sample a specified classical elastic scalar interaction.
// The amplitude uses hbar=c=1; squared energies are Mandelstam invariants
// and cross sections have units of inverse energy squared. Inputs include the
// local event, material masses and a reciprocal squared amplitude with a
// certified global bound. Outputs are differential areas, conservative local
// scattering proposals or the gain-minus-loss collision integrand.
// These local material interactions are separate from gravitational subhalo
// scattering and its effective cross sections.

#include "coordinate_geometry.hpp"
#include "local_dsmc.hpp"
#include <functional>

namespace rba {

// Curvature and background arguments are scalar model inputs at this
// event. Their coefficients specify the chosen interaction; covariance
// alone supplies no universal curvature-dependent material cross section.
struct ScalarFields {
    double ricci = 0.0;           // E^2
    double kretschmann = 0.0;     // E^4
    double background = 0.0;     // A prescribed dimensionless scalar.
};

struct Event {
    Vector coordinates;
    ScalarFields scalars;
};

// With signature (-,+,+,+), s=-(p1+p2)^2, t=-(p1-p3)^2 and
// u=-(p1-p4)^2. An elastic pair obeys s+t+u=2m1^2+2m2^2.
struct Mandelstam {
    double s = 0.0;
    double t = 0.0;
    double u = 0.0;
    double mass1 = 0.0;
    double mass2 = 0.0;
};

// The final-state factor counts identical outgoing states once. The
// separate Poisson sampler already counts each incoming unordered pair once.
enum class FinalStates : unsigned int {
    distinguishable = 1,
    identical_scalar = 2
};

struct AmplitudeModel {
    std::string name;
    FinalStates final_states = FinalStates::distinguishable;
    bool exchange_symmetrized = false;
    bool isotropic = false;
    bool inverse_reciprocal = false;
    // Pure reciprocal callback on collision phase space. Dependence on
    // coordinate labels must be transformed by composition under chart changes.
    std::function<double(const Event&,const Mandelstam&)> squared;
    // Certified supremum of |M|^2/s over every reachable elastic state at x,
    // for material masses >= minimum_mass. The caller supplies its proof/model.
    std::function<double(const Event&,double)> squared_over_s_bound;
};

// The full outgoing solid angle needs a divisor of 2! for identical
// scalars and 1 for distinguishable partners.
inline double symmetry_divisor(const AmplitudeModel& model) {
    rb::require(model.final_states == FinalStates::distinguishable
                || model.final_states == FinalStates::identical_scalar,
                "Unknown final-state counting convention.");
    return static_cast<unsigned int>(model.final_states);
}

// Coordinate components are contracted with the metric at the same event.
// The overload below gives the identical scalar invariants in a tetrad.
inline Mandelstam invariants(const Geometry& geometry, Vector p1, Vector p2,
                            Vector p3, Vector p4, double mass1, double mass2) {
    Vector sum{};
    Vector transfer{};
    Vector exchange{};
    for (int i = 0; i < 4; ++i) {
        sum[i] = p1[i]+p2[i];
        transfer[i] = p1[i]-p3[i];
        exchange[i] = p1[i]-p4[i];
    }
    return {-dot(geometry.metric,sum,sum),-dot(geometry.metric,transfer,transfer),
            -dot(geometry.metric,exchange,exchange),mass1,mass2};
}

inline Mandelstam invariants(rb::Four p1, rb::Four p2, rb::Four p3, rb::Four p4,
                            double mass1, double mass2) {
    return {-rb::minkowski(p1+p2,p1+p2),-rb::minkowski(p1-p3,p1-p3),
            -rb::minkowski(p1-p4,p1-p4),mass1,mass2};
}

// Check the massive elastic channel before invoking the model callback.
// The callback returns |M|^2, and reciprocal gain/loss uses the same model
// for incoming and reversed outgoing momenta.
inline double amplitude_squared(const AmplitudeModel& model, const Event& point,
                                const Mandelstam& v) {
    rb::require(!model.name.empty() && static_cast<bool>(model.squared)
                && static_cast<bool>(model.squared_over_s_bound) && model.inverse_reciprocal,
                "The elastic gain-loss model requires a declared reciprocal amplitude.");
    for (double x : {point.scalars.ricci,point.scalars.kretschmann,point.scalars.background,
                     v.s,v.t,v.u,v.mass1,v.mass2}) {
        rb::require(std::isfinite(x),"Nonfinite amplitude input.");
    }
    rb::require(v.mass1 > 0.0 && v.mass2 > 0.0 && v.s > 0.0,"Invalid massive channel.");
    for (double x : point.coordinates) {
        rb::require(std::isfinite(x),"Nonfinite amplitude event coordinate.");
    }
    const double scale = v.s+2*v.mass1*v.mass1+2*v.mass2*v.mass2;
    rb::require(std::isfinite(scale),"The material mass/invariant scale is numerically unresolved.");
    const double tolerance = 1.0e-10*scale;
    rb::require(v.s+tolerance >= (v.mass1+v.mass2)*(v.mass1+v.mass2)
                && v.t <= tolerance
                && std::abs(v.s+v.t+v.u-2*v.mass1*v.mass1-2*v.mass2*v.mass2) <= tolerance,
                "The supplied invariants are outside the elastic channel.");
    symmetry_divisor(model);
    const double value = model.squared(point,v);
    rb::require(value >= 0.0 && std::isfinite(value),"An amplitude square must be nonnegative and finite.");
    // For identical scalar final states the declared amplitude must already
    // include exchange symmetry. Checking t<->u prevents applying the counting
    // factor to a distinguishable amplitude by mistake.
    if (model.final_states == FinalStates::identical_scalar) {
        rb::require(v.mass1 == v.mass2 && model.exchange_symmetrized,
                    "Identical final states require a declared exchange-symmetrized scalar amplitude.");
        Mandelstam swapped = v;
        std::swap(swapped.t,swapped.u);
        const double exchanged = model.squared(point,swapped);
        rb::require(exchanged >= 0.0 && std::isfinite(exchanged)
                    && std::abs(exchanged-value) <= 1.0e-12*std::max({value,exchanged,1.0e-300}),
                    "The declared identical-scalar amplitude violates t-u symmetry.");
    }
    return value;
}

// Elastic channel: q_out/q_in=1. Full-sphere counting includes S_f=2! when
// the two outgoing particles are identical. Incoming pair counting is separate.
inline double differential_cross_section(const AmplitudeModel& model, const Event& point,
                                         const Mandelstam& v) {
    return amplitude_squared(model,point,v)/(64.0*rb::pi*rb::pi*v.s*symmetry_divisor(model));
}

// Integrate a certified full-sphere differential envelope. The model
// callback supplies the bound over all reachable invariant energies and angles.
inline double cross_section_envelope(const AmplitudeModel& model, const Event& point,
                                    double minimum_mass) {
    rb::require(minimum_mass > 0.0 && std::isfinite(minimum_mass)
                && !model.name.empty() && static_cast<bool>(model.squared)
                && static_cast<bool>(model.squared_over_s_bound) && model.inverse_reciprocal,
                "Missing reciprocal amplitude envelope/model.");
    rb::require(model.final_states != FinalStates::identical_scalar || model.exchange_symmetrized,
                "Identical-scalar amplitudes require explicit exchange symmetrization.");
    const double bound = model.squared_over_s_bound(point,minimum_mass);
    rb::require(bound >= 0.0 && std::isfinite(bound),"Invalid global differential envelope.");
    // Integrating the differential bound over 4pi produces a total-area
    // envelope. Its validity must cover every reachable energy and angle,
    // including states generated by earlier accepted collisions.
    return bound/(16.0*rb::pi*symmetry_divisor(model));
}

// A chosen tree-level scalar contact benchmark. For identical scalars its
// L_int=-lambda_eff(x)*phi^4/4! gives M=-i lambda_eff. Optional curvature and
// background coefficients define a supplied local EFT interaction; they are
// not a universal curvature correction derived from covariance.
inline AmplitudeModel scalar_contact_benchmark(
    double lambda, FinalStates states = FinalStates::identical_scalar,
    double curvature_scale = 1.0, double ricci_coefficient = 0.0,
    double kretschmann_coefficient = 0.0, double background_coefficient = 0.0) {
    rb::require(lambda >= 0.0 && std::isfinite(lambda) && curvature_scale > 0.0
                && std::isfinite(curvature_scale) && std::isfinite(ricci_coefficient)
                && std::isfinite(kretschmann_coefficient) && std::isfinite(background_coefficient),
                "Invalid contact benchmark parameters.");
    // The supplied scalar combination defines lambda_eff at the event.
    // Ricci and Kretschmann terms are divided by the declared energy scale
    // to keep each contribution dimensionless.
    const auto coupling = [=](const Event& x) {
        const double scale2 = curvature_scale*curvature_scale;
        const double scale4 = scale2*scale2;
        rb::require(scale2 > 0.0 && std::isfinite(scale2) && scale4 > 0.0 && std::isfinite(scale4),
                    "The benchmark curvature scale is numerically unresolved.");
        const double value = lambda+ricci_coefficient*x.scalars.ricci/scale2
            +kretschmann_coefficient*x.scalars.kretschmann/scale4
            +background_coefficient*x.scalars.background;
        rb::require(value >= 0.0 && std::isfinite(value),"The supplied quartic coupling is outside its benchmark domain.");
        return value;
    };
    AmplitudeModel result;
    result.name = "specified_scalar_contact_natural_units";
    result.final_states = states;
    result.exchange_symmetrized = states == FinalStates::identical_scalar;
    result.isotropic = true;
    result.inverse_reciprocal = true;
    result.squared = [=](const Event& x, const Mandelstam&) {
        const double value = coupling(x);
        return value*value;
    };
    // For constant angular amplitude, s>=4m_min^2 supplies a conservative
    // energy envelope. The sampler checks this bound before accepting a draw.
    result.squared_over_s_bound = [=](const Event& x, double mmin) {
        const double value = coupling(x);
        const double threshold = 4.0*mmin*mmin;
        rb::require(threshold > 0.0 && std::isfinite(threshold),
                    "The contact threshold is numerically unresolved.");
        return value*value/threshold;
    };
    return result;
}

// An isotropic differential area is constant over outgoing direction.
// Multiplication by 4pi gives its physical total cross section.
inline double contact_total_cross_section(const AmplitudeModel& model, const Event& point,
                                         double s, double mass1, double mass2) {
    rb::require(model.isotropic,"A full-sphere value requires a declared isotropic model.");
    return 4.0*rb::pi*differential_cross_section(model,point,{s,0.0,2*mass1*mass1+2*mass2*mass2-s,mass1,mass2});
}

struct Proposal {
    rb::PairOutput output;
    double angular_acceptance = 0.0;
};

// A uniform CM direction is a proposal. The angular acceptance below
// weights it by the actual differential area relative to the certified bound.
inline Proposal propose_amplitude_event(
    const rb::Particle& first, const rb::Particle& second, rb::Random& random,
    const Event& point, const AmplitudeModel& model, double sigma_envelope) {
    rb::require(sigma_envelope > 0.0 && std::isfinite(sigma_envelope),"Positive envelope required.");
    const auto g = rb::pair_geometry(first.material_mass,first.momentum,
                                     second.material_mass,second.momentum);
    const rb::Four direction = rb::boost({0.0,random.unit_direction()},-1.0*g.cm_velocity);
    const auto output = rb::scatter_with_direction(first.material_mass,first.momentum,
                                                   second.material_mass,second.momentum,direction);
    const auto v = invariants(first.momentum,second.momentum,output.first,output.second,
                              first.material_mass,second.material_mass);
    // Reversing the same elastic quadruple tests microscopic reciprocity.
    // This is the detailed-balance condition needed by the classical
    // gain-minus-loss operator, distinct from angular isotropy.
    const auto reversed = invariants(output.first,output.second,first.momentum,second.momentum,
                                     first.material_mass,second.material_mass);
    const double forward = amplitude_squared(model,point,v);
    const double inverse = amplitude_squared(model,point,reversed);
    rb::require(std::abs(forward-inverse) <= 1.0e-10*std::max({forward,inverse,1.0e-300}),
                "The supplied amplitude fails inverse-process reciprocity.");
    // A direction drawn uniformly on the CM sphere has proposal density
    // 1/(4pi). This ratio thins that proposal to the actual differential area.
    const double probability = 4.0*rb::pi*differential_cross_section(model,point,v)/sigma_envelope;
    rb::require(probability >= 0.0 && probability <= 1.0,
                "The certified amplitude envelope was exceeded; transaction refused.");
    return {output,probability};
}

// Joint angle/rate thinning supports bounded anisotropic scalar callbacks.
// It samples the actual d_sigma(x,s,t), not uniform physical scattering angles.
// Internal shared event process. Public entry points below impose either the
// single identical species or the fixed distinguishable two-packet channel.
// Complete model/context/channel restart binding is not implemented here;
// Mechanical serialize_cell alone is not a complete amplitude restart format.
// Event, geometry, volume and lapse are frozen for this local suboperator.
// Its internal Poisson clock does not resolve an evolving physical worldline.
namespace amplitude_detail {
inline void advance_population(rb::CellState& caller, double dt,
                                   const Event& point, const AmplitudeModel& model,
                                   rb::CellControls controls) {
    rb::require(controls.cross_section == 0.0,"Amplitude mode sets its own physical cross section.");
    double minimum_mass = std::numeric_limits<double>::infinity();
    for (const auto& p : caller.particles) {
        minimum_mass = std::min(minimum_mass,p.material_mass);
        rb::require(model.final_states != FinalStates::identical_scalar
                    || p.material_mass == caller.particles.front().material_mass,
                    "An identical-scalar channel requires a single material rest mass.");
    }
    const double sigma_envelope = caller.particles.empty() ? 0.0
        : cross_section_envelope(model,point,minimum_mass);
    controls.cross_section = sigma_envelope;
    rb::validate_cell(caller,controls);
    rb::require(dt >= 0.0 && std::isfinite(dt),"Invalid amplitude collision interval.");
    if (dt == 0.0) {
        return;
    }
    // Use a private copy of particles, clock and RNG. A failed envelope or
    // conditioning check therefore preserves the incoming physical state.
    rb::CellState s = caller;
    if (s.number_weight_majorant == 0.0) {
        for (const auto& p : s.particles) {
            s.number_weight_majorant = std::max(s.number_weight_majorant,p.number_weight);
        }
    }
    const double end = s.coordinate_time+dt;
    rb::require(end > s.coordinate_time && std::isfinite(end),"Unresolved amplitude event clock.");
    std::uint64_t proposals = 0;
    while (s.coordinate_time < end && s.particles.size() >= 2 && sigma_envelope > 0.0) {
        const double n = static_cast<double>(s.particles.size());
        // The total-area envelope and v_M<=2 define the proposal Poisson rate.
        // Actual flux, unequal statistical weights and angular dependence are
        // restored together by the acceptance probability below.
        const double rate = controls.lapse*sigma_envelope/controls.proper_volume
                            *s.number_weight_majorant*n*(n-1.0);
        rb::require(rate > 0.0 && std::isfinite(rate),"Invalid amplitude proposal rate.");
        if (s.remaining_optical_depth < 0.0) {
            s.remaining_optical_depth = -std::log(s.random.uniform());
        }
        const double remaining = end-s.coordinate_time;
        const double wait = s.remaining_optical_depth/rate;
        // Retain the unused exponential optical depth when the next proposal
        // falls outside this frozen interval. Subdividing an interval then keeps
        // the carried waiting-time information.
        if (wait > remaining) {
            s.remaining_optical_depth -= rate*remaining;
            break;
        }
        rb::require(++proposals <= controls.maximum_proposals,"Amplitude proposal cap exceeded.");
        rb::require(s.coordinate_time+wait > s.coordinate_time || wait == 0.0,
                    "Amplitude event clock resolution exhausted.");
        s.coordinate_time += wait;
        ++s.proposals;
        std::size_t i = rb::random_index(s.random,s.particles.size());
        std::size_t j = rb::random_index(s.random,s.particles.size()-1);
        if (j >= i) {
            ++j;
        }
        // The restricted distinguishable pair stores partner 1/2 in slots 0/1.
        // Keep those physical roles fixed while retaining both selection draws.
        if (model.final_states == FinalStates::distinguishable && i > j) {
            std::swap(i,j);
        }
        const auto first = s.particles[i];
        const auto second = s.particles[j];
        const Proposal proposed = propose_amplitude_event(first,second,s.random,point,model,sigma_envelope);
        const double probability = std::max(first.number_weight,second.number_weight)/s.number_weight_majorant
                                   *0.5*rb::moller_flux_speed(first.momentum,second.momentum)
                                   *proposed.angular_acceptance;
        rb::require(probability >= 0.0 && probability <= 1.0,"Amplitude joint thinning bound failed.");
        if (s.random.uniform() < probability) {
            rb::require(s.particles.size()+(first.number_weight != second.number_weight)
                            <= controls.maximum_markers,"Amplitude split population cap exceeded.");
            // Scatter the smaller represented number from each partner. Keep the
            // heavier packet remainder at its incoming momentum and give it a new ID,
            // so number and weighted four-momentum budgets remain well defined.
            const double q = std::min(first.number_weight,second.number_weight);
            s.particles[i].number_weight = q;
            s.particles[j].number_weight = q;
            s.particles[i].momentum = proposed.output.first;
            s.particles[j].momentum = proposed.output.second;
            if (first.number_weight != second.number_weight) {
                rb::require(s.next_id < std::numeric_limits<std::uint64_t>::max(),"Amplitude marker ID overflow.");
                auto remainder = first.number_weight > second.number_weight ? first : second;
                remainder.number_weight -= q;
                remainder.id = s.next_id++;
                s.particles.push_back(remainder);
            }
            const double separation = rb::norm(first.position-second.position);
            s.spacelike_exchanges += separation > 0.0;
            s.maximum_exchange_distance = std::max(s.maximum_exchange_distance,separation);
            ++s.accepted;
        }
        s.remaining_optical_depth = -std::log(s.random.uniform());
    }
    s.coordinate_time = end;
    caller = std::move(s);
}
}  // namespace amplitude_detail

// This population entry admits one identical scalar species. A general
// mixed-species reaction network needs its own channel bookkeeping.
inline void advance_amplitude_cell(rb::CellState& state, double dt,
                                  const Event& point, const AmplitudeModel& model,
                                  rb::CellControls controls) {
    rb::require(model.final_states == FinalStates::identical_scalar,
                "A general population currently supports one identical scalar species.");
    amplitude_detail::advance_population(state,dt,point,model,controls);
}

// The distinguishable entry has exactly two equal-weight partner
// packets. Fixed slots preserve the physical partner labels in |M|^2.
inline void advance_distinguishable_pair(rb::CellState& state, double dt,
                                        const Event& point, const AmplitudeModel& model,
                                        rb::CellControls controls) {
    rb::require(model.final_states == FinalStates::distinguishable && state.particles.size() == 2
                && state.particles[0].number_weight == state.particles[1].number_weight,
                "The distinguishable channel requires exactly one equal-weight packet of each partner.");
    amplitude_detail::advance_population(state,dt,point,model,controls);
}

// Evaluate the local classical collision integrand on a conserving
// elastic momentum quadruple. The scalar distribution values are ordered
// as incoming f1,f2 and outgoing f3,f4; the result carries their gain-loss
// factor f3*f4-f1*f2, invariant flux and differential cross section.
inline double differential_collision_kernel(const Geometry& geometry,
    const std::array<Vector,4>& momenta, double mass1, double mass2,
    const Event& point, const AmplitudeModel& model, const std::array<double,4>& scalar_f) {
    validate_geometry(geometry);
    rb::require(geometry.position == point.coordinates,"The amplitude and momenta must refer to the same event.");
    for (double f : scalar_f) {
        rb::require(f >= 0.0 && std::isfinite(f),"Invalid classical scalar distribution.");
    }
    for (int i = 0; i < 4; ++i) {
        rb::validate_momentum(i%2 ? mass2 : mass1,to_tetrad(geometry,momenta[i]));
    }
    const rb::Four a = to_tetrad(geometry,momenta[0]);
    const rb::Four b = to_tetrad(geometry,momenta[1]);
    const rb::Four c = to_tetrad(geometry,momenta[2]);
    const rb::Four d = to_tetrad(geometry,momenta[3]);
    const rb::Four residual = a+b-c-d;
    rb::require(std::max(std::abs(residual.t),rb::norm(residual.p)) <= 1.0e-11*(a.t+b.t),
                "The collision quadruple violates local four-momentum conservation.");
    const auto v = invariants(geometry,momenta[0],momenta[1],momenta[2],momenta[3],mass1,mass2);
    const rb::Four p1 = to_tetrad(geometry,momenta[0]);
    const rb::Four p2 = to_tetrad(geometry,momenta[1]);
    const auto reversed = invariants(c,d,a,b,mass1,mass2);
    const double forward = amplitude_squared(model,point,v);
    const double inverse = amplitude_squared(model,point,reversed);
    rb::require(std::abs(forward-inverse) <= 1.0e-10*std::max({forward,inverse,1.0e-300}),
                "The collision kernel requires inverse-process reciprocity.");
    return rb::invariant_flux(p1,p2)*differential_cross_section(model,point,v)
           *(scalar_f[2]*scalar_f[3]-scalar_f[0]*scalar_f[1]);
}

}  // namespace rba
