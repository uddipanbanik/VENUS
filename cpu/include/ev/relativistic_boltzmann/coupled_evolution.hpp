#pragma once

// Advance material particles with their self-consistent gravitational field,
// then apply elastic collisions over the accepted streaming interval.
// SC uses spacelike proper-cell geometry; EF uses the null crossing measure.
// Inputs are a complete marker state, one persistent clock per radial cell,
// the field grid and split controls. Outputs include the updated state,
// reconstructed field, encounter counts and collision-induced ADM diagnostic.
// The sequence is a first-order operator split with frozen endpoint geometry.

#include "eb_restart.hpp"
#include "schwarzschild_collision_adapter.hpp"
#include "ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp"
#include <map>

namespace eb {

struct SplitControls {
    Model model;
    bool allow_radial_cell_approximation = false;
    std::size_t maximum_markers_per_cell = 4096;
    std::uint64_t maximum_proposals_per_cell = 1000000;
};

// Marker transport and stochastic clocks share one coordinate time.
// next_id is global across cells, so unequal-weight remainders receive
// unique identities as cells are visited in radial order.
template<class Marker> struct CoupledState {
    std::vector<Marker> markers;
    std::vector<Cell> collision_clocks;
    double time = 0;
    std::uint64_t next_id = 1;
};

using SCState = CoupledState<ev::WeightedPICMarker>;
using EFState = CoupledState<ev::eddington_finkelstein::Marker>;

struct SCStep {
    ev::WeightedEinsteinVlasovPICStepResult streaming;
    ev::WeightedEinsteinVlasovPICField field;
    std::uint64_t collisions = 0;
    std::uint64_t nonlocal_collisions = 0;
    double collision_adm_change = 0;
};

struct EFStep {
    ev::eddington_finkelstein::StepResult streaming;
    ev::eddington_finkelstein::Field field;
    std::uint64_t collisions = 0;
    std::uint64_t nonlocal_collisions = 0;
    double regular_center_reconstructed_adm = 0;
    double collision_adm_change = 0;
};

// Bin the radius after the accepted streaming update in [edge_i,edge_{i+1}).
// The marker ID and its source-cell label keep their separate identity.
inline std::size_t radial_cell(double r, const ev::WeightedPICFieldGrid& grid) {
    const auto upper = std::upper_bound(grid.mesh.edge.begin(),grid.mesh.edge.end(),r);
    rb::require(upper != grid.mesh.edge.begin() && r < grid.mesh.edge.back(),
                "Matter reached the isolated collision-domain boundary.");
    return static_cast<std::size_t>(upper-grid.mesh.edge.begin()-1);
}

// Package the slice-dependent volume and time factor together with the
// explicit finite radial-cell policy. Their meanings are established by
// the SC metric quadrature or EF coordinate geometry at the call site.
inline Geometry collision_geometry(Slice slice, double volume, double factor,
                                   double width, const SplitControls& c) {
    Geometry g;
    g.slice = slice;
    g.volume = volume;
    g.time_factor = factor;
    g.allow_radial_cell_approximation = c.allow_radial_cell_approximation;
    g.maximum_radial_separation = width;
    g.maximum_markers = c.maximum_markers_per_cell;
    g.maximum_proposals = c.maximum_proposals_per_cell;
    return g;
}

// The clock records are persistent, while each step reassigns the current
// population to cells. A co-located cell is point local. A finite radial cell
// is an explicitly selected homogeneous collision approximation.
template<class Marker>
inline void prepare_clocks(CoupledState<Marker>& s, const ev::WeightedPICFieldGrid& grid) {
    rb::require(s.collision_clocks.size() == grid.radius.size(),"One persistent collision clock per cell is required.");
    for (auto& clock : s.collision_clocks) {
        rb::require(clock.coordinate_time == s.time,"The material and collision clocks disagree.");
        // Refill packet membership from the transported population below.
        // Random state, residual optical depth and the carried majorant persist.
        clock.packets.clear();
    }
}

// First-order EV streaming -> frozen collision suboperator -> Einstein
// constraints. Finite radial cells use a homogeneous local approximation.
inline SCStep advance_sc(SCState& caller, double dt, const ev::WeightedPICFieldGrid& grid,
                         const ev::WeightedEinsteinVlasovPICOptions& evolution,
                         const SplitControls& c) {
    rb::require(evolution.require_vacuum_outer_boundary,"SC evolution requires the isolated boundary guard.");
    validate(Cell{},c.model,Geometry{});
    // Copy the entire coupled state before transport and collision sampling.
    // A later boundary, metric or collision guard therefore preserves the
    // caller's complete state rather than a partly updated cell population.
    SCState s = caller;
    // The collisionless SC path returns the accepted transport result and
    // synchronizes the clock epochs without consuming collision randomness.
    SCStep out;
    out.streaming = ev::advance_weighted_einstein_vlasov_pic_self_consistent(s.markers,dt,grid,evolution);
    out.field = out.streaming.endpoint_field;
    if (c.model.cross_section == 0 || out.streaming.accepted_dt == 0) {
        s.time += out.streaming.accepted_dt;
        for (auto& clock : s.collision_clocks) clock.coordinate_time = s.time;
        caller = std::move(s);
        return out;
    }
    rb::require(!out.streaming.horizon_threshold_reached,"SC evolution reached its polar-areal stopping threshold.");
    prepare_clocks(s,grid);
    // ancestor_id refers to this interval's incoming record. Splitting gives
    // a remainder its own ID while preserving its auxiliary marker properties.
    std::map<std::uint64_t,ev::WeightedPICMarker> incoming;
    for (const auto& marker : s.markers) {
        rb::require(marker.state == ev::WeightedPICMarkerState::active
                    && marker.internal_energy == 0 && marker.internal_velocity_dispersion == 0,
                    "Material SIDM requires active particles without unresolved subhalo internal energy/dispersion.");
        rb::require(incoming.emplace(marker.id,marker).second,"Duplicate SC marker ID.");
        s.collision_clocks[radial_cell(marker.r,grid)].packets.push_back(
            {marker.mass,marker.r,marker.w,marker.eta,marker.id,marker.source_cell,marker.id});
    }
    s.markers.clear();
    for (std::size_t i = 0; i < s.collision_clocks.size(); ++i) {
        auto& cell = s.collision_clocks[i];
        cell.next_id = s.next_id;
        // Migrating packets can exceed a prior cell's carried majorant.
        for (const auto& p : cell.packets)
            cell.number_majorant = std::max(cell.number_majorant,p.rest_weight/c.model.material_mass);
        // Eight-point radial quadrature gives proper volume and mean lapse.
        // Collision exposure is the accepted interval, including a shortened step.
        const auto geometry = rb::sc_cell_geometry(grid,out.streaming.endpoint_field,i);
        const auto g = collision_geometry(Slice::schwarzschild,geometry.proper_volume,
            geometry.mean_lapse,geometry.outer_radius-geometry.inner_radius,c);
        const auto before = cell.collisions;
        const auto nonlocal = cell.radially_nonlocal_collisions;
        advance_cell(cell,out.streaming.accepted_dt,c.model,g);
        out.collisions += cell.collisions-before;
        out.nonlocal_collisions += cell.radially_nonlocal_collisions-nonlocal;
        s.next_id = cell.next_id;
        for (const auto& p : cell.packets) {
            ev::WeightedPICMarker marker = incoming.at(p.ancestor_id);
            marker.mass = p.rest_weight;
            marker.r = p.r;
            marker.w = p.radial;
            marker.eta = p.eta;
            marker.id = p.id;
            marker.source_cell = p.source_id;
            s.markers.push_back(marker);
        }
        cell.packets.clear();
    }
    // Recompute the radial constraints after every cell has committed its
    // outgoing momenta. Each local collision used the streaming endpoint metric.
    out.field = ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
        s.markers,grid,evolution.shape,evolution.threads);
    rb::require(out.field.metric.min_gap > grid.parameters.gap_stop,"Postcollision SC constraint gap reached its stop.");
    // Local elastic four-momentum conservation and global ADM conservation
    // are different checks. Reconstructing the field measures the ADM change
    // associated with this frozen collision substep.
    out.collision_adm_change = out.field.adm_mass-out.streaming.endpoint_field.adm_mass;
    s.time += out.streaming.accepted_dt;
    caller = std::move(s);
    return out;
}

inline EFStep advance_ef(EFState& caller, double dv, const ev::WeightedPICFieldGrid& grid,
                         const ev::eddington_finkelstein::EvolutionOptions& evolution,
                         const SplitControls& c) {
    namespace ef = ev::eddington_finkelstein;
    rb::require(evolution.require_vacuum_outer_boundary && !evolution.enable_excision,
                "The material EF module preserves every particle and requires the isolated boundary guard.");
    validate(Cell{},c.model,Geometry{});
    // The EF transaction preserves the incoming markers, RNG and clocks
    // until streaming, all cell collisions and the postcollision constraints
    // have succeeded.
    EFState s = caller;
    EFStep out;
    out.streaming = ef::advance_self_consistent(s.markers,dv,grid,evolution);
    out.field = out.streaming.endpoint_field;
    if (c.model.cross_section == 0 || out.streaming.accepted_dt == 0) {
        s.time += out.streaming.accepted_dt;
        for (auto& clock : s.collision_clocks) clock.coordinate_time = s.time;
        caller = std::move(s);
        return out;
    }
    prepare_clocks(s,grid);
    // Retain auxiliary properties through ancestry while assigning new IDs
    // to split remainders across the complete cell traversal.
    std::map<std::uint64_t,ef::Marker> incoming;
    for (const auto& marker : s.markers) {
        rb::require(marker.state == ef::MarkerState::active,"The retained material EF population must stay active.");
        rb::require(incoming.emplace(marker.id,marker).second,"Duplicate EF marker ID.");
        s.collision_clocks[radial_cell(marker.r,grid)].packets.push_back(
            {marker.rest_weight,marker.r,marker.p_r,marker.eta,marker.id,marker.source_id,marker.id});
    }
    s.markers.clear();
    for (std::size_t i = 0; i < s.collision_clocks.size(); ++i) {
        auto& cell = s.collision_clocks[i];
        cell.next_id = s.next_id;
        for (const auto& p : cell.packets)
            cell.number_majorant = std::max(cell.number_majorant,p.rest_weight/c.model.material_mass);
        // The EF null-slice rate uses coordinate shell volume V0 and center b.
        // The momentum factors in advance_cell supply the null crossing measure.
        const double b = out.streaming.endpoint_field.metric.b[i];
        const double V0 = grid.mesh.shell_volume[i];
        const auto g = collision_geometry(Slice::advanced_null,V0,b,
            grid.mesh.edge[i+1]-grid.mesh.edge[i],c);
        const auto before = cell.collisions;
        const auto nonlocal = cell.radially_nonlocal_collisions;
        advance_cell(cell,out.streaming.accepted_dt,c.model,g);
        out.collisions += cell.collisions-before;
        out.nonlocal_collisions += cell.radially_nonlocal_collisions-nonlocal;
        s.next_id = cell.next_id;
        for (const auto& p : cell.packets) {
            ef::Marker marker = incoming.at(p.ancestor_id);
            marker.rest_weight = p.rest_weight;
            marker.r = p.r;
            marker.p_r = p.radial;
            marker.eta = p.eta;
            marker.id = p.id;
            marker.source_id = p.source_id;
            s.markers.push_back(marker);
        }
        cell.packets.clear();
    }
    ef::ConstraintOptions constraint;
    constraint.enable_excision = false;
    // Independent regular-center reconstruction exposes any collision/sampling
    // ADM residual before the isolated fixed-ADM boundary is reapplied.
    // With no specified outer ADM value, this reconstruction integrates from
    // the regular center and measures the mass implied by the outgoing source.
    const auto regular = ef::deposit_and_solve_field(s.markers,grid,evolution.shape,evolution.threads,constraint);
    out.regular_center_reconstructed_adm = regular.metric.adm_mass;
    out.collision_adm_change = regular.metric.adm_mass-out.streaming.conserved_adm_mass;
    // The following solve instead fixes the outer ADM integration constant
    // to the streaming value. Its inner mass condition is therefore a distinct
    // boundary choice; the regular reconstruction above remains the source
    // normalization diagnostic.
    constraint.fixed_adm_mass = out.streaming.conserved_adm_mass;
    // After every cell has returned its outgoing momentum, solve the radial
    // constraints with the declared boundary choice. The clock advances by
    // the actual accepted streaming interval, including a shortened step.
    out.field = ef::deposit_and_solve_field(s.markers,grid,evolution.shape,evolution.threads,constraint);
    s.time += out.streaming.accepted_dt;
    caller = std::move(s);
    return out;
}

}  // namespace eb
