#pragma once

// Apply local material collisions to weighted spherical Einstein--Vlasov
// markers on a Schwarzschild time slice. The metric is
//  ds^2=-alpha^2 dt^2+A^2 dr^2+r^2 dOmega^2, with c=1.
// Inputs are represented marker rest masses, specific momenta, a physical
// material mass and the reconstructed metric. Outputs retain event radii
// and source labels while updating momenta, positive remainder packets,
// conservation diagnostics and the persistent collision clock.
// The proper-volume and lapse factors here belong to a spacelike slice.

#include "local_dsmc.hpp"
#include "ev/weighted_einstein_vlasov_pic_schwarzschild.hpp"

namespace rb {

struct SphericalCellGeometry {
    double inner_radius = 0.0;
    double outer_radius = 0.0;
    double proper_volume = 0.0;
    double proper_radial_width = 0.0;
    double mean_lapse = 0.0;
    // Ranges over the eight quadrature samples, not rigorous cell extrema.
    double lapse_relative_range = 0.0;
    double radial_metric_relative_range = 0.0;
};

// Quadrature of the maintained reconstructed metric on one spacelike cell.
// The homogeneous-cell approximation uses <alpha>_V / V for its rate.
inline SphericalCellGeometry sc_cell_geometry(
    const ev::WeightedPICFieldGrid& grid,
    const ev::WeightedEinsteinVlasovPICField& field,
    std::size_t index) {
    require(index < grid.radius.size(),"Invalid SC collision cell.");
    SphericalCellGeometry result;
    result.inner_radius = grid.mesh.edge[index];
    result.outer_radius = grid.mesh.edge[index+1];
    // Eight-point Gauss--Legendre nodes integrate the reconstructed radial
    // metric inside this cell. The same interpolation used by transport
    // supplies alpha=exp(mu) and A=exp(lambda) at each quadrature point.
    const std::array<double,8> x = {{
        -0.9602898564975363,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,
         0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363}};
    const std::array<double,8> weight = {{
        0.1012285362903763,0.2223810344533745,0.3137066458778873,0.3626837833783620,
        0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763}};
    const double middle = 0.5*(result.inner_radius+result.outer_radius);
    const double half = 0.5*(result.outer_radius-result.inner_radius);
    double lapse_min = std::numeric_limits<double>::infinity();
    double lapse_max = 0.0;
    double A_min = std::numeric_limits<double>::infinity();
    double A_max = 0.0;
    double lapse_volume = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double radius = middle+half*x[k];
        const ev::MetricSample metric = ev::weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(
            radius,grid,field.metric);
        const double lapse = std::exp(metric.mu);
        const double A = std::exp(metric.lambda);
        require(lapse > 0.0 && A > 0.0 && std::isfinite(lapse) && std::isfinite(A),
                "Collision geometry requires a regular spacelike SC metric.");
        // The proper shell measure is 4pi r^2 A dr. Weighting alpha with this
        // measure gives the mean lapse multiplying the homogeneous-cell rate.
        // The separate A dr integral reports the proper radial width.
        const double dv = 4.0*pi*radius*radius*A*half*weight[k];
        result.proper_volume += dv;
        result.proper_radial_width += A*half*weight[k];
        lapse_volume += lapse*dv;
        lapse_min = std::min(lapse_min,lapse);
        lapse_max = std::max(lapse_max,lapse);
        A_min = std::min(A_min,A);
        A_max = std::max(A_max,A);
    }
    require(result.proper_volume > 0.0,"Collision cell has zero proper volume.");
    result.mean_lapse = lapse_volume/result.proper_volume;
    result.lapse_relative_range = (lapse_max-lapse_min)/lapse_min;
    result.radial_metric_relative_range = (A_max-A_min)/A_min;
    return result;
}

// At fixed r, w is the local radial momentum per material mass and
// sqrt(2eta)/r is its tangential magnitude. A random azimuth supplies the
// unrecorded spherical orientation while preserving the material mass shell.
inline Four lift_sc_marker(const ev::WeightedPICMarker& marker,
                           double material_mass, double azimuth) {
    require(marker.state == ev::WeightedPICMarkerState::active
            && marker.mass > 0.0 && std::isfinite(marker.mass)
            && marker.r >= 0.0 && std::isfinite(marker.r)
            && marker.eta >= 0.0 && std::isfinite(marker.eta)
            && marker.internal_energy == 0.0 && marker.internal_velocity_dispersion == 0.0,
            "The material elastic adapter requires active particles without unresolved subhalo internal energy/dispersion.");
    require(marker.r > 0.0 || marker.eta == 0.0,"A regular center requires zero angular momentum.");
    const double transverse = marker.r > 0.0 ? std::sqrt(2.0*marker.eta)/marker.r : 0.0;
    return on_shell(material_mass,material_mass*Vec3{
        marker.w,transverse*std::cos(azimuth),transverse*std::sin(azimuth)});
}

// Replace w and eta from the outgoing local physical momentum at the
// same radius. Angular momentum changes during scattering; its new value
// comes from eta=r^2 |p_perp/m_chi|^2/2.
inline void project_sc_momentum(ev::WeightedPICMarker& marker,
                               Four momentum, double material_mass) {
    validate_momentum(material_mass,momentum);
    marker.w = momentum.p.x/material_mass;
    const double transverse_squared =
        (momentum.p.y*momentum.p.y+momentum.p.z*momentum.p.z)/(material_mass*material_mass);
    marker.eta = 0.5*marker.r*marker.r*transverse_squared;
    require(marker.r > 0.0 || transverse_squared == 0.0,
            "A point at the spherical coordinate center needs a Cartesian momentum chart.");
}

struct SCState {
    std::vector<ev::WeightedPICMarker> markers;
    Random random;
    double coordinate_time = 0.0;
    double remaining_optical_depth = -1.0;
    double number_weight_majorant = 0.0;
    std::uint64_t next_id = 1;
    std::uint64_t proposals = 0;
    std::uint64_t accepted = 0;
    std::uint64_t radially_nonlocal_exchanges = 0;
    double largest_radial_coordinate_separation = 0.0;
    double largest_local_four_momentum_residual = 0.0;
};

struct SCControls {
    double material_mass = 1.0;
    CellControls cell;
};

// One spherical momentum population or one explicitly approximate radial
// collision cell. Angular orientations are sampled at the same representative
// angular point for each proposal, as required by spherical symmetry.
// A common radius gives a local distribution collision, not a claim that
// independently tracked particles at different angular positions meet.
inline void advance_sc_cell(SCState& caller, double dt, const SCControls& controls) {
    const CellControls& c = controls.cell;
    require(controls.material_mass > 0.0 && std::isfinite(controls.material_mass)
            && dt >= 0.0 && std::isfinite(dt),"Invalid spherical collision controls.");
    // Use the shared cell validator to check positive weights, future-shell
    // momenta, IDs and the explicit spatial approximation before sampling.
    // The validation map uses represented number equal to marker mass/m_chi.
    CellState validation;
    validation.coordinate_time = caller.coordinate_time;
    validation.remaining_optical_depth = caller.remaining_optical_depth;
    validation.number_weight_majorant = caller.number_weight_majorant;
    validation.next_id = caller.next_id;
    for (const auto& m : caller.markers) {
        validation.particles.push_back({controls.material_mass,m.mass/controls.material_mass,
                                       lift_sc_marker(m,controls.material_mass,0.0),{m.r,0.0,0.0},m.id});
    }
    // Here the separation guard is in radial coordinate units. A physical
    // cell's proper width is reported separately by sc_cell_geometry().
    validate_cell(validation,c);
    if (dt == 0.0) {
        return;
    }
    // All event draws and splits occur on a copy. Only a fully successful
    // frozen collision interval replaces the caller state.
    SCState s = caller;
    if (s.number_weight_majorant == 0.0) {
        for (const auto& m : s.markers) {
            s.number_weight_majorant = std::max(s.number_weight_majorant,m.mass/controls.material_mass);
        }
    }
    const double end = s.coordinate_time+dt;
    require(std::isfinite(end) && end > s.coordinate_time,"Invalid SC event clock advance.");
    std::uint64_t proposed = 0;
    while (s.coordinate_time < end && s.markers.size() >= 2 && c.cross_section > 0.0) {
        const double qmax = s.number_weight_majorant;
        const double n = static_cast<double>(s.markers.size());
        // The unordered pair count and v_M<=2 envelope give n(n-1). The
        // lapse supplies d tau=alpha dt; dividing by proper volume converts
        // represented partner counts to local density.
        const double rate = c.lapse*c.cross_section/c.proper_volume*qmax*n*(n-1.0);
        require(rate > 0.0 && std::isfinite(rate),"Invalid SC proposal rate.");
        if (s.remaining_optical_depth < 0.0) {
            s.remaining_optical_depth = -std::log(s.random.uniform());
        }
        const double wait = s.remaining_optical_depth/rate;
        // Carry the unspent optical depth across the caller interval boundary.
        // A new waiting-time draw occurs after a proposal, including rejection.
        if (wait > end-s.coordinate_time) {
            s.remaining_optical_depth -= rate*(end-s.coordinate_time);
            break;
        }
        require(++proposed <= c.maximum_proposals,"SC collision proposal budget exceeded.");
        require(s.coordinate_time+wait > s.coordinate_time || wait == 0.0,
                "SC event interval is below clock resolution.");
        s.coordinate_time += wait;
        ++s.proposals;
        const std::size_t i = random_index(s.random,s.markers.size());
        std::size_t j = random_index(s.random,s.markers.size()-1);
        if (j >= i) {
            ++j;
        }
        const auto a = s.markers[i];
        const auto b = s.markers[j];
        // Sample the two tangential azimuths independently at the representative
        // angular event. The particles remain spherical distribution samples,
        // with an explicitly approximate finite radial separation if admitted.
        const Four p1 = lift_sc_marker(a,controls.material_mass,2.0*pi*s.random.uniform());
        const Four p2 = lift_sc_marker(b,controls.material_mass,2.0*pi*s.random.uniform());
        const double probability = std::max(a.mass,b.mass)/controls.material_mass/qmax
                                   *0.5*moller_flux_speed(p1,p2);
        require(probability >= 0.0 && probability <= 1.0+1.0e-14,"SC flux majorant failed.");
        if (s.random.uniform() < probability) {
            require(s.markers.size()+(a.mass != b.mass) <= c.maximum_markers,
                    "SC conservative splitting marker budget exceeded.");
            const PairOutput output = scatter_isotropic(
                controls.material_mass,p1,controls.material_mass,p2,s.random);
            const Four residual = output.first+output.second-p1-p2;
            s.largest_local_four_momentum_residual = std::max(
                s.largest_local_four_momentum_residual,
                std::max(std::abs(residual.t),norm(residual.p))/(p1.t+p2.t));
            // The smaller represented rest mass scatters from each packet. A
            // positive unequal-weight remainder preserves its incoming momentum
            // and auxiliary marker properties under a newly assigned identity.
            const double scattered_mass = std::min(a.mass,b.mass);
            s.markers[i].mass = scattered_mass;
            s.markers[j].mass = scattered_mass;
            project_sc_momentum(s.markers[i],output.first,controls.material_mass);
            project_sc_momentum(s.markers[j],output.second,controls.material_mass);
            if (a.mass != b.mass) {
                require(s.next_id < std::numeric_limits<std::uint64_t>::max(),"SC marker ID overflow.");
                auto remainder = a.mass > b.mass ? a : b;
                remainder.mass -= scattered_mass;
                remainder.id = s.next_id++;
                s.markers.push_back(remainder);
            }
            const double distance = std::abs(a.r-b.r);
            s.largest_radial_coordinate_separation = std::max(s.largest_radial_coordinate_separation,distance);
            s.radially_nonlocal_exchanges += distance > 0.0;
            ++s.accepted;
        }
        s.remaining_optical_depth = -std::log(s.random.uniform());
    }
    s.coordinate_time = end;
    caller = std::move(s);
}

struct Stress {
    double number = 0.0;
    double rest_mass = 0.0;
    double energy = 0.0;
    double radial_current = 0.0;
    double radial_pressure = 0.0;
    double tangential_pressure_sum = 0.0;
};

// Shell-integrated proper moments. Divide by proper volume for densities.
// Existing EV mass is invariant represented rest mass; p/m_chi=(epsilon,w,...).
// These are shell-integrated proper moments: sum W, sum W*epsilon,
// sum W*w and pressure sums W*w^2/epsilon. Divide by the proper
// volume supplied above to form the corresponding local densities.
inline Stress sc_proper_moments(const std::vector<ev::WeightedPICMarker>& markers,
                               double material_mass = 1.0) {
    require(material_mass > 0.0 && std::isfinite(material_mass),"Invalid material particle mass.");
    Stress result;
    for (const auto& m : markers) {
        const Four p = lift_sc_marker(m,1.0,0.0);
        result.number += m.mass/material_mass;
        result.rest_mass += m.mass;
        result.energy += m.mass*p.t;
        result.radial_current += m.mass*m.w;
        result.radial_pressure += m.mass*m.w*m.w/p.t;
        result.tangential_pressure_sum += m.mass*dot(Vec3{0,p.p.y,p.p.z},Vec3{0,p.p.y,p.p.z})/p.t;
    }
    return result;
}

// Save the shared stochastic cell together with the full SC marker
// properties. Inner-boundary phase and source labels must survive a
// collision restart even though this local substep changes only momenta.
inline std::string serialize_sc(const SCState& s, const SCControls& c) {
    SCState validation = s;
    advance_sc_cell(validation,0.0,c);
    CellState base;
    base.random = s.random;
    base.coordinate_time = s.coordinate_time;
    base.remaining_optical_depth = s.remaining_optical_depth;
    base.number_weight_majorant = s.number_weight_majorant;
    base.next_id = s.next_id;
    base.proposals = s.proposals;
    base.accepted = s.accepted;
    base.spacelike_exchanges = s.radially_nonlocal_exchanges;
    base.maximum_exchange_distance = s.largest_radial_coordinate_separation;
    for (const auto& m : s.markers) {
        base.particles.push_back({c.material_mass,m.mass/c.material_mass,
            lift_sc_marker(m,c.material_mass,0.0),{m.r,0,0},m.id});
    }
    using namespace restart_detail;
    std::string result = "RBSCS002";
    put_double(result,c.material_mass);
    const std::string common = serialize_cell(base,c.cell);
    put_u64(result,common.size());
    result += common;
    put_double(result,s.largest_local_four_momentum_residual);
    for (const auto& m : s.markers) {
        for (double x : {m.mass,m.r,m.w,m.eta,m.inner_boundary_speed,
                         m.remaining_inner_time,m.internal_energy,m.internal_velocity_dispersion}) {
            put_double(result,x);
        }
        put_u64(result,m.id);
        put_u64(result,m.source_cell);
        put_u64(result,static_cast<std::uint64_t>(m.state));
    }
    put_u64(result,checksum(result));
    return result;
}

// Restore the shared RNG/optical depth first, then each SC marker.
// A checksum and exact canonical re-serialization bind the whole result
// to its material mass, collision controls and record layout.
inline SCState deserialize_sc(const std::string& bytes, const SCControls& c) {
    using namespace restart_detail;
    require(bytes.size() >= 32 && bytes.substr(0,8) == "RBSCS002","Unknown SC collision restart.");
    Reader tail{bytes,bytes.size()-8};
    require(checksum(bytes.substr(0,bytes.size()-8)) == tail.u64(),"SC restart checksum failed.");
    Reader r{bytes,8};
    require(r.real() == c.material_mass,"SC restart material mass changed.");
    const std::uint64_t common_size = r.u64();
    require(common_size <= bytes.size()-r.offset,"Truncated SC restart common state.");
    const CellState common = deserialize_cell(bytes.substr(r.offset,common_size),c.cell);
    r.offset += common_size;
    SCState s;
    s.random = common.random;
    s.coordinate_time = common.coordinate_time;
    s.remaining_optical_depth = common.remaining_optical_depth;
    s.number_weight_majorant = common.number_weight_majorant;
    s.next_id = common.next_id;
    s.proposals = common.proposals;
    s.accepted = common.accepted;
    s.radially_nonlocal_exchanges = common.spacelike_exchanges;
    s.largest_radial_coordinate_separation = common.maximum_exchange_distance;
    s.largest_local_four_momentum_residual = r.real();
    for (std::size_t i = 0; i < common.particles.size(); ++i) {
        ev::WeightedPICMarker m;
        m.mass = r.real();
        m.r = r.real();
        m.w = r.real();
        m.eta = r.real();
        m.inner_boundary_speed = r.real();
        m.remaining_inner_time = r.real();
        m.internal_energy = r.real();
        m.internal_velocity_dispersion = r.real();
        m.id = r.u64();
        m.source_cell = r.u64();
        const std::uint64_t state = r.u64();
        require(state == 0,"SC elastic restart contains an unsupported marker state.");
        m.state = static_cast<ev::WeightedPICMarkerState>(state);
        s.markers.push_back(m);
    }
    require(r.offset+8 == bytes.size() && serialize_sc(s,c) == bytes,
            "SC collision restart correspondence failed.");
    return s;
}

// This adapter uses the spacelike proper-cell measure. The EF material
// path instead uses spherical_collisions.hpp and its derived null-shell
// flux factors; the two measures have distinct coordinate domains.
inline void require_spacelike_slice(bool advanced_null_slice) {
    require(!advanced_null_slice,
            "Advanced-null collisions require a separately derived flux measure and adapter.");
}

}  // namespace rb
