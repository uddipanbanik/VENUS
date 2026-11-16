#pragma once

// Approximate the reversible spherical Boltzmann operator by summing an active
// velocity-cell block and sampling its complement.  Inputs are shell f(w,eta),
// cell geometry, angular/cross-section controls and optional persistent caches.
// Outputs are a positive finite-time shell update and activity, sampling-error,
// projection, entropy and locality diagnostics.
//
// The dense block is closed under radial velocity reversal and sums all its
// discrete angular channels.  Complement draws carry their inverse proposal
// probability.  A draw and its radial reverse share one outgoing pair stencil;
// the reaction uses geometric forward/reverse activities and conservative pair
// moments.  Repeated draws add coefficients to the same graph edge before the
// finite-time reaction solve.  Maxwellian cancellation follows each retained
// edge's invariant exponents in the resolved entropy-floor regime.

#include "ev/sampled_reversible_boltzmann.hpp"

#include <unordered_map>

namespace ev {

// A projection depends on the incoming cells and discrete angular indices.
// The variant distinguishes a directly constructed stencil from its reversal.
struct BPStreamedProjectionKey {
    int first = 0;
    int second = 0;
    int pair_phi = 0;
    int scatter_mu = 0;
    int scatter_phi = 0;
    int projection_variant = 0;

    bool operator==(const BPStreamedProjectionKey& other) const {
        return first == other.first
            && second == other.second
            && pair_phi == other.pair_phi
            && scatter_mu == other.scatter_mu
            && scatter_phi == other.scatter_phi
            && projection_variant == other.projection_variant;
    }
};

struct BPStreamedProjectionKeyHash {
    std::size_t operator()(const BPStreamedProjectionKey& key) const {
        std::uint64_t value = static_cast<std::uint64_t>(key.first);
        value = bp_sampled_splitmix64(
            value ^ (static_cast<std::uint64_t>(key.second) << 17));
        value = bp_sampled_splitmix64(
            value ^ (static_cast<std::uint64_t>(key.pair_phi) << 7)
            ^ (static_cast<std::uint64_t>(key.scatter_mu) << 23)
            ^ (static_cast<std::uint64_t>(key.scatter_phi) << 37)
            ^ (static_cast<std::uint64_t>(key.projection_variant) << 47));
        return static_cast<std::size_t>(value);
    }
};

struct BPStreamedPairKey {
    int first = 0;
    int second = 0;
    int pair_phi = 0;

    bool operator==(const BPStreamedPairKey& other) const {
        return first == other.first
            && second == other.second
            && pair_phi == other.pair_phi;
    }
};

struct BPStreamedPairKeyHash {
    std::size_t operator()(const BPStreamedPairKey& key) const {
        std::uint64_t value = static_cast<std::uint64_t>(key.first);
        value = bp_sampled_splitmix64(
            value ^ (static_cast<std::uint64_t>(key.second) << 19));
        value = bp_sampled_splitmix64(
            value ^ (static_cast<std::uint64_t>(key.pair_phi) << 41));
        return static_cast<std::size_t>(value);
    }
};

// A pilot ranks important reversal-pair orbits; this optional CDF supplements
// the full-support complement proposal through an explicitly weighted mixture.
struct BPStreamedGuidedProposal {
    double fraction = 0.0;
    std::vector<BPStreamedPairKey> pair;
    std::vector<double> cdf;
    std::unordered_map<
        BPStreamedPairKey,
        double,
        BPStreamedPairKeyHash> probability;
};

// Cache only geometry-dependent products and omission status.  Current f
// determines reaction activity each time the cached stencil is evaluated.
struct BPStreamedProjectionRecord {
    bool represented = false;
    bool outside_domain = false;
    unsigned outside_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
    double maximum_projection_error = 0.0;
    std::vector<BPReversibleSpecies> direct_species;
    std::vector<BPReversibleSpecies> radial_reverse_species;
};

struct BPStreamedProjectionCache {
    std::unordered_map<
        BPStreamedProjectionKey,
        BPStreamedProjectionRecord,
        BPStreamedProjectionKeyHash> record;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;

    void clear(std::size_t expected_entries = 0) {
        record.clear();
        hits = 0;
        misses = 0;
        if (expected_entries > record.bucket_count()) {
            record.reserve(expected_entries);
        }
    }
};

// The dense graph can be reused while its selected cell set and the outer
// workspace configuration agree.  Its activities are refreshed from current f.
struct BPStreamedDenseGraphCache {
    std::vector<int> dense_cells;
    BPReversibleGraph graph;
    bool valid = false;

    void clear() {
        dense_cells.clear();
        graph = BPReversibleGraph{};
        valid = false;
    }
};

// Reusing a sampled graph is an explicit approximation controlled by the
// volume-weighted L1 change from reference_f; its earlier error estimates are
// retained as provenance for the cached graph.
struct BPStreamedOperatorGraphCache {
    bool valid = false;
    bool converged = false;
    std::vector<double> reference_f;
    BPReversibleGraph graph;
    int samples_per_batch = 0;
    int adaptive_rounds = 0;
    std::uint64_t dense_reactions = 0;
    std::uint64_t complement_reactions = 0;
    double rhs_relative_error = 0.0;
    double rhs_gross_error = 0.0;
    double heat_relative_error = 0.0;
    double heat_gross_error = 0.0;
    double energy_invariant_error = 0.0;

    void clear() {
        valid = false;
        converged = false;
        reference_f.clear();
        graph = BPReversibleGraph{};
        samples_per_batch = 0;
        adaptive_rounds = 0;
        dense_reactions = 0;
        complement_reactions = 0;
        rhs_relative_error = 0.0;
        rhs_gross_error = 0.0;
        heat_relative_error = 0.0;
        heat_gross_error = 0.0;
        energy_invariant_error = 0.0;
    }
};

// Separate shell caches prevent workers from modifying the same records.
// Geometry, quadrature, physical law and sampling controls govern invalidation.
struct BPStreamedCollisionWorkspace {
    std::vector<BPStreamedProjectionCache> shell_projection_cache;
    std::vector<BPStreamedDenseGraphCache> shell_dense_graph_cache;
    std::vector<BPStreamedOperatorGraphCache> shell_operator_graph_cache;
    std::vector<double> radius;
    std::vector<double> radial_velocity_edge;
    std::vector<double> eta_edge;
    int candidate_radius = 0;
    int pair_phi_points = 0;
    int scatter_mu_points = 0;
    int scatter_phi_points = 0;
    int dense_velocity_cells = 0;
    double projection_tolerance = 0.0;
    std::string cross_section;
    std::string kernel;
    double collision_strength = 0.0;
    double sigma0 = 0.0;
    double sigma_u0 = 0.0;
    double sigma_n = 0.0;
    double yukawa_alpha = 0.0;
    double yukawa_mchi = 0.0;
    double yukawa_mphi = 0.0;
    std::uint64_t operator_configuration_hash = 0;
    std::size_t maximum_persistent_projection_records = 262144;

    void clear() {
        shell_projection_cache.clear();
        shell_dense_graph_cache.clear();
        shell_operator_graph_cache.clear();
        radius.clear();
        radial_velocity_edge.clear();
        eta_edge.clear();
    }
};

// Hash literal parameter bits so changes in proposal or convergence policy
// invalidate a graph built with different coefficients or admission controls.
inline std::uint64_t bp_streamed_operator_configuration_hash(
    const Params& p) {
    std::uint64_t hash = UINT64_C(0x6a09e667f3bcc909);
    const auto add_integer = [&hash](std::uint64_t value) {
        hash = bp_sampled_splitmix64(hash ^ value);
    };
    const auto add_double = [&add_integer](double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        add_integer(bits);
    };
    add_integer(static_cast<std::uint64_t>(p.bp_sampled_dense_cells));
    add_integer(static_cast<std::uint64_t>(
        p.bp_sampled_min_samples_per_batch));
    add_integer(static_cast<std::uint64_t>(
        p.bp_sampled_max_samples_per_batch));
    add_integer(static_cast<std::uint64_t>(
        p.bp_sampled_replicates_per_estimate));
    add_integer(static_cast<std::uint64_t>(
        p.bp_sampled_outgoing_angle_samples));
    add_integer(static_cast<std::uint64_t>(
        p.bp_sampled_guided_pilot_samples));
    add_integer(static_cast<std::uint64_t>(p.bp_sampled_guided_pairs));
    add_double(p.bp_sampled_guided_fraction);
    add_integer(p.bp_sampled_fixed_geometric_sequence ? 1 : 0);
    add_integer(p.bp_sampled_reuse_operator_graph ? 1 : 0);
    add_double(p.bp_sampled_operator_refresh_l1_fraction);
    add_integer(p.bp_sampled_ntc_pair_thinning ? 1 : 0);
    add_integer(static_cast<std::uint64_t>(p.bp_sampled_ntc_pilot_samples));
    add_double(p.bp_sampled_ntc_quantile);
    add_integer(static_cast<std::uint64_t>(p.bp_sampled_energy_strata));
    add_integer(static_cast<std::uint64_t>(p.bp_sampled_L_strata));
    add_double(p.bp_sampled_tail_fraction);
    add_double(p.bp_sampled_stratified_fraction);
    add_double(p.bp_sampled_equilibrium_fraction);
    add_double(p.bp_sampled_phase_density_fraction);
    add_double(p.bp_sampled_stratum_uniform_fraction);
    add_double(p.bp_sampled_uniform_fraction);
    add_double(p.bp_sampled_tail_power);
    add_double(p.bp_sampled_rhs_tolerance);
    add_double(p.bp_sampled_rhs_floor_fraction);
    add_double(p.bp_sampled_heat_tolerance);
    add_double(p.bp_sampled_heat_floor_fraction);
    add_double(p.bp_sampled_activity_boost_max);
    add_integer(p.bp_sampled_seed);
    // These controls either change the sampled reaction graph itself or decide
    // whether an unconverged estimate is admissible.  Include them so a caller
    // that updates collision parameters cannot reuse a graph built under a
    // physically different softened-gravity kernel or sampling contract.
    add_double(p.vp_gravity);
    add_double(p.bp_subhalo_mass);
    add_double(p.bp_subhalo_radius);
    add_double(p.bp_coulomb_log);
    add_double(p.bp_reversible_entropy_floor);
    add_integer(p.bp_sampled_fail_on_unconverged ? 1 : 0);
    return hash;
}

// Reuse requires the same mesh, shell radii, projection rules and physical
// collision law; clear all dependent caches when any comparison changes.
inline void bp_streamed_prepare_collision_workspace(
    BPStreamedCollisionWorkspace& workspace,
    const std::vector<double>& radius,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    const std::uint64_t operator_configuration_hash =
        bp_streamed_operator_configuration_hash(p);
    const bool compatible = workspace.radius == radius
        && workspace.radial_velocity_edge == geometry.radial_velocity.edge
        && workspace.eta_edge == geometry.eta.edge
        && workspace.candidate_radius == p.bp_reversible_candidate_radius
        && workspace.pair_phi_points == p.bp_boltzmann_pair_phi_points
        && workspace.scatter_mu_points == p.bp_boltzmann_scatter_mu_points
        && workspace.scatter_phi_points == p.bp_boltzmann_scatter_phi_points
        && workspace.dense_velocity_cells == p.bp_sampled_dense_cells
        && workspace.projection_tolerance
            == p.bp_reversible_projection_tolerance
        && workspace.cross_section == p.bp_cross_section
        && workspace.kernel == bp_fa_kernel_name(p)
        && workspace.collision_strength == p.bp_collision_strength
        && workspace.sigma0 == p.bp_sigma0
        && workspace.sigma_u0 == p.bp_sigma_u0
        && workspace.sigma_n == p.bp_sigma_n
        && workspace.yukawa_alpha == p.bp_yukawa_alpha
        && workspace.yukawa_mchi == p.bp_yukawa_mchi
        && workspace.yukawa_mphi == p.bp_yukawa_mphi
        && workspace.operator_configuration_hash
            == operator_configuration_hash;
    if (!compatible) {
        workspace.clear();
        workspace.radius = radius;
        workspace.radial_velocity_edge = geometry.radial_velocity.edge;
        workspace.eta_edge = geometry.eta.edge;
        workspace.candidate_radius = p.bp_reversible_candidate_radius;
        workspace.pair_phi_points = p.bp_boltzmann_pair_phi_points;
        workspace.scatter_mu_points = p.bp_boltzmann_scatter_mu_points;
        workspace.scatter_phi_points = p.bp_boltzmann_scatter_phi_points;
        workspace.dense_velocity_cells = p.bp_sampled_dense_cells;
        workspace.projection_tolerance = p.bp_reversible_projection_tolerance;
        workspace.cross_section = p.bp_cross_section;
        workspace.kernel = bp_fa_kernel_name(p);
        workspace.collision_strength = p.bp_collision_strength;
        workspace.sigma0 = p.bp_sigma0;
        workspace.sigma_u0 = p.bp_sigma_u0;
        workspace.sigma_n = p.bp_sigma_n;
        workspace.yukawa_alpha = p.bp_yukawa_alpha;
        workspace.yukawa_mchi = p.bp_yukawa_mchi;
        workspace.yukawa_mphi = p.bp_yukawa_mphi;
        workspace.operator_configuration_hash =
            operator_configuration_hash;
    }
    if (workspace.shell_projection_cache.size() != radius.size()) {
        workspace.shell_projection_cache.clear();
        workspace.shell_projection_cache.resize(radius.size());
    }
    if (workspace.shell_dense_graph_cache.size() != radius.size()) {
        workspace.shell_dense_graph_cache.clear();
        workspace.shell_dense_graph_cache.resize(radius.size());
    }
    if (workspace.shell_operator_graph_cache.size() != radius.size()) {
        workspace.shell_operator_graph_cache.clear();
        workspace.shell_operator_graph_cache.resize(radius.size());
    }
}

// One shell supplies velocity volumes, proposal probabilities, angular tables
// and rigorous sigma*g majorants for the subsequent draws.
struct BPStreamedShellContext {
    double radius = 0.0;
    double maximum_relative_speed = 0.0;
    double u_sigma_majorant = 0.0;
    std::array<double, 4> component_u_sigma_majorant{{0.0, 0.0, 0.0, 0.0}};
    std::vector<double> volume;
    BPSampledProposal proposal;
    std::vector<double> pair_phi_cos;
    std::vector<double> pair_phi_sin;
    std::vector<double> scatter_mu;
    std::vector<double> scatter_mu_weight;
    std::vector<double> scatter_phi_cos;
    std::vector<double> scatter_phi_sin;
    std::string kernel;
    std::vector<int> dense_cells;
    std::vector<unsigned char> dense_mask;
    double complement_probability = 1.0;
    BPSampledComplementProposal complement_proposal;
    BPReversibleGraph dense_graph;
    std::vector<double> dense_rhs;
    std::vector<double> dense_gross_rhs;
};

// Retain unnormalized importance-weighted sums so an adaptive extension can
// append new draws and divide by the final draw count exactly once.
struct BPStreamedBatch {
    std::vector<double> rhs_sum;
    std::vector<double> gross_rhs_sum;
    std::uint64_t attempted_draws = 0;
    std::uint64_t retained_draws = 0;
    std::uint64_t represented_channels = 0;
    double represented_activity_sum = 0.0;
    double omitted_domain_activity_sum = 0.0;
    double omitted_projection_activity_sum = 0.0;
    double maximum_projection_error = 0.0;
    double maximum_event_rhs = 0.0;
    int maximum_event_rhs_cell = -1;
    BPStreamedProjectionKey maximum_event_key;
    std::unordered_map<
        BPStreamedProjectionKey,
        double,
        BPStreamedProjectionKeyHash> reaction_coefficient_sum;
    std::unordered_map<
        BPStreamedProjectionKey,
        double,
        BPStreamedProjectionKeyHash> omitted_domain_coefficient_sum;
    std::unordered_map<
        BPStreamedProjectionKey,
        double,
        BPStreamedProjectionKeyHash> omitted_projection_coefficient_sum;
};

struct BPStreamedOperatorEstimate {
    std::vector<double> rhs;
    std::vector<double> gross_rhs;
    BPSampledBatchError error;
    int samples_per_batch = 0;
    int adaptive_rounds = 0;
    bool converged = false;
    std::uint64_t controller_attempted_draws = 0;
    std::uint64_t controller_retained_draws = 0;
    std::uint64_t controller_represented_channels = 0;
    BPStreamedBatch first;
    BPStreamedBatch second;
    std::vector<BPStreamedBatch> additional;
};

struct BPStreamedShellResult {
    BPReversibleShellResult step;
    int samples_per_batch = 0;
    int adaptive_rounds = 0;
    bool converged = true;
    std::uint64_t attempted_draws = 0;
    std::uint64_t retained_draws = 0;
    std::uint64_t represented_channels = 0;
    std::uint64_t dense_reactions = 0;
    std::uint64_t complement_reactions = 0;
    bool reused_operator_graph = false;
    double operator_reference_l1 = 0.0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    double rhs_relative_error = 0.0;
    double rhs_gross_error = 0.0;
    double heat_relative_error = 0.0;
    double heat_gross_error = 0.0;
    double energy_invariant_error = 0.0;
    double context_seconds = 0.0;
    double estimate_seconds = 0.0;
    double materialize_seconds = 0.0;
    double advance_seconds = 0.0;
    int stage1_positivity_rejections = 0;
    int stage2_positivity_rejections = 0;
};

inline void bp_streamed_prepare_dense_partition(
    const std::vector<double>& shell_f,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    BPStreamedShellContext& context,
    BPStreamedDenseGraphCache* persistent_cache = nullptr);

// Build the proposal and reversal-closed dense partition at the actual shell
// radius.  Maximum relative speeds bound sigma*g over each proposal support.
inline BPStreamedShellContext bp_streamed_make_shell_context(
    const std::vector<double>& shell_f,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    BPStreamedDenseGraphCache* dense_graph_cache = nullptr) {
    BPStreamedShellContext context;
    context.radius = radius;
    BPReversibleGraph volume_graph;
    bp_sampled_initialize_graph_volumes(
        volume_graph, radius, geometry, p);
    context.volume = std::move(volume_graph.velocity_cell_volume);
    context.proposal = bp_sampled_build_proposal(
        shell_f, radius, w, eta, geometry, p);
    context.dense_cells = bp_sampled_dense_velocity_cells(
        context.proposal, radius, w, eta, p);
    context.dense_mask.assign(
        static_cast<std::size_t>(p.Nw * p.NL), 0);
    for (int q : context.dense_cells) {
        context.dense_mask[static_cast<std::size_t>(q)] = 1;
    }
    context.complement_probability = bp_sampled_complement_probability(
        context.proposal, context.dense_cells);
    if (context.complement_probability > 0.0) {
        context.complement_proposal = bp_sampled_build_complement_proposal(
            context.proposal,
            context.dense_mask,
            context.complement_probability);
    }
    std::array<double, 4> maximum_speed_squared{{0.0, 0.0, 0.0, 0.0}};
    std::vector<unsigned char> active_stratum(
        context.proposal.strata.size(), 0);
    for (int stratum : context.proposal.active_strata) {
        active_stratum[static_cast<std::size_t>(stratum)] = 1;
    }
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const std::array<int, 2> indices =
            bp_reversible_velocity_indices(q, p);
        const double wr = w[static_cast<std::size_t>(indices[0])];
        const double vt_squared = 2.0
            * eta[static_cast<std::size_t>(indices[1])]
            / (radius * radius);
        const double speed_squared = wr * wr + vt_squared;
        maximum_speed_squared[3] = std::max(
            maximum_speed_squared[3], speed_squared);
        if (context.proposal.bulk_probability[static_cast<std::size_t>(q)]
            > 0.0) {
            maximum_speed_squared[0] = std::max(
                maximum_speed_squared[0], speed_squared);
        }
        if (context.proposal.tail_probability[static_cast<std::size_t>(q)]
            > 0.0) {
            maximum_speed_squared[1] = std::max(
                maximum_speed_squared[1], speed_squared);
        }
        const int stratum = context.proposal.cell_stratum[
            static_cast<std::size_t>(q)];
        if (active_stratum[static_cast<std::size_t>(stratum)]) {
            maximum_speed_squared[2] = std::max(
                maximum_speed_squared[2], speed_squared);
        }
    }
    for (int component = 0; component < 4; ++component) {
        const double maximum_relative_speed =
            2.0 * std::sqrt(maximum_speed_squared[
                static_cast<std::size_t>(component)]);
        context.component_u_sigma_majorant[
            static_cast<std::size_t>(component)] =
            bp_sampled_u_sigma_upper_bound(maximum_relative_speed, p);
    }
    context.maximum_relative_speed =
        2.0 * std::sqrt(maximum_speed_squared[3]);
    context.u_sigma_majorant = context.component_u_sigma_majorant[3];
    bp_fa_midpoint_angle_table(
        std::max(1, p.bp_boltzmann_pair_phi_points),
        context.pair_phi_cos,
        context.pair_phi_sin);
    bp_fa_gauss_legendre_mu_table(
        std::max(1, p.bp_boltzmann_scatter_mu_points),
        context.scatter_mu,
        context.scatter_mu_weight);
    bp_fa_midpoint_angle_table(
        std::max(1, p.bp_boltzmann_scatter_phi_points),
        context.scatter_phi_cos,
        context.scatter_phi_sin);
    context.kernel = bp_fa_kernel_name(p);
    bp_streamed_prepare_dense_partition(
        shell_f, w, eta, geometry, p, context, dense_graph_cache);
    return context;
}

// Draw from the normalized discrete polar cross section and uniform discrete
// scattering azimuth.  This samples the same angular rule as a full sweep.
inline std::array<int, 2> bp_streamed_discrete_scatter_indices(
    double polar_unit,
    double azimuth_unit,
    double relative_speed,
    const BPStreamedShellContext& context,
    const Params& p) {
    double total_weight = 0.0;
    for (std::size_t q = 0; q < context.scatter_mu.size(); ++q) {
        total_weight += context.scatter_mu_weight[q]
            * bp_fa_scatter_weight_raw(
                context.scatter_mu[q], relative_speed, p, context.kernel);
    }
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw std::runtime_error(
            "streamed Boltzmann polar quadrature has zero weight.");
    }
    const double clipped_polar = std::max(
        0.0, std::min(std::nextafter(1.0, 0.0), polar_unit));
    const double target = clipped_polar * total_weight;
    double cumulative = 0.0;
    int mu_index = static_cast<int>(context.scatter_mu.size()) - 1;
    for (int q = 0; q < static_cast<int>(context.scatter_mu.size()); ++q) {
        cumulative += context.scatter_mu_weight[static_cast<std::size_t>(q)]
            * bp_fa_scatter_weight_raw(
                context.scatter_mu[static_cast<std::size_t>(q)],
                relative_speed,
                p,
                context.kernel);
        if (target < cumulative) {
            mu_index = q;
            break;
        }
    }
    const double clipped_azimuth = std::max(
        0.0, std::min(std::nextafter(1.0, 0.0), azimuth_unit));
    const int phi_index = std::min(
        static_cast<int>(context.scatter_phi_cos.size()) - 1,
        static_cast<int>(
            clipped_azimuth
            * static_cast<double>(context.scatter_phi_cos.size())));
    return {mu_index, phi_index};
}

// Reflect radial velocity indices while preserving eta and pair weights.
// Merge coincident reflected products into one conservative stencil.
inline std::vector<BPReversibleProductPair>
bp_streamed_reverse_products(
    const std::vector<BPReversibleProductPair>& products,
    const BPSampledProposal& proposal) {
    std::vector<BPReversibleProductPair> reversed;
    reversed.reserve(products.size());
    for (const BPReversibleProductPair& product : products) {
        const int first = proposal.time_reverse_cell[
            static_cast<std::size_t>(product.first)];
        const int second = proposal.time_reverse_cell[
            static_cast<std::size_t>(product.second)];
        if (first < 0 || second < 0) {
            throw std::runtime_error(
                "streamed Boltzmann product grid is not closed under radial reversal.");
        }
        reversed.push_back(BPReversibleProductPair{
            std::min(first, second),
            std::max(first, second),
            product.weight,
        });
    }
    std::sort(
        reversed.begin(),
        reversed.end(),
        [](const BPReversibleProductPair& first,
           const BPReversibleProductPair& second) {
            return std::tie(first.first, first.second)
                < std::tie(second.first, second.second);
        });
    std::size_t count = 0;
    for (const BPReversibleProductPair& product : reversed) {
        if (count > 0
            && reversed[count - 1].first == product.first
            && reversed[count - 1].second == product.second) {
            reversed[count - 1].weight += product.weight;
        } else {
            reversed[count++] = product;
        }
    }
    reversed.resize(count);
    return reversed;
}

// Compute a joint outgoing stencil once and cache both its direct reaction
// and radial reverse, including unsupported-domain/projection classifications.
inline const BPStreamedProjectionRecord& bp_streamed_projection_record(
    int first,
    int second,
    int pair_phi,
    int mu_index,
    int scatter_phi_index,
    const BPFAVec3& outgoing_first,
    const BPFAVec3& outgoing_second,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const BPStreamedShellContext& context,
    BPStreamedProjectionCache& cache,
    int projection_variant = 0) {
    const BPStreamedProjectionKey key{
        first,
        second,
        pair_phi,
        mu_index,
        scatter_phi_index,
        projection_variant};
    const auto found = cache.record.find(key);
    if (found != cache.record.end()) {
        ++cache.hits;
        return found->second;
    }
    ++cache.misses;
    BPStreamedProjectionRecord value;
    BPReversibleProjection projection;
    value.represented = bp_reversible_project_outgoing_pair(
        outgoing_first,
        outgoing_second,
        context.radius,
        w,
        eta,
        geometry,
        p,
        projection,
        &value.outside_domain,
        &value.outside_domain_mask);
    if (value.represented) {
        value.maximum_projection_error = std::max({
            std::abs(projection.mass_error),
            std::abs(projection.momentum_error),
            std::abs(projection.energy_error),
        });
        value.direct_species = bp_reversible_compress_reaction(
            first, second, projection.products);
        const int reverse_first = context.proposal.time_reverse_cell[
            static_cast<std::size_t>(first)];
        const int reverse_second = context.proposal.time_reverse_cell[
            static_cast<std::size_t>(second)];
        value.radial_reverse_species = bp_reversible_compress_reaction(
            reverse_first,
            reverse_second,
            bp_streamed_reverse_products(
                projection.products, context.proposal));
    }
    return cache.record.emplace(key, std::move(value)).first->second;
}

// The radial reverse exchanges the two cached species lists and shares the
// direct projection's admissibility and moment-error record.
inline const BPStreamedProjectionRecord&
bp_streamed_radial_reverse_projection_record(
    const BPStreamedProjectionRecord& direct,
    int reverse_first,
    int reverse_second,
    int pair_phi,
    int mu_index,
    int scatter_phi_index,
    BPStreamedProjectionCache& cache) {
    const BPStreamedProjectionKey key{
        reverse_first,
        reverse_second,
        pair_phi,
        mu_index,
        scatter_phi_index,
        1};
    const auto found = cache.record.find(key);
    if (found != cache.record.end()) {
        ++cache.hits;
        return found->second;
    }
    ++cache.misses;
    BPStreamedProjectionRecord value;
    value.represented = direct.represented;
    value.outside_domain = direct.outside_domain;
    value.outside_domain_mask = direct.outside_domain_mask;
    value.maximum_projection_error = direct.maximum_projection_error;
    value.direct_species = direct.radial_reverse_species;
    value.radial_reverse_species = direct.direct_species;
    return cache.record.emplace(key, std::move(value)).first->second;
}

// Positive finite coefficients enter either the represented graph or its
// explicit omission ledger, preserving the reason for every unsupported edge.
inline void bp_streamed_add_cached_reaction(
    BPReversibleGraph& graph,
    int first,
    int second,
    const BPStreamedProjectionRecord& projection,
    double coefficient) {
    if (!(coefficient > 0.0) || !std::isfinite(coefficient)) {
        return;
    }
    if (!projection.represented) {
        BPReversibleOmittedChannel channel{
            first,
            second,
            coefficient,
            projection.outside_domain_mask,
        };
        if (projection.outside_domain) {
            graph.omitted_domain_channels.push_back(channel);
        } else {
            graph.omitted_projection_channels.push_back(channel);
        }
        return;
    }
    BPReversibleReaction reaction;
    reaction.incoming_first = first;
    reaction.incoming_second = second;
    reaction.coefficient = coefficient;
    reaction.species = projection.direct_species;
    graph.reactions.push_back(std::move(reaction));
    graph.maximum_projection_error = std::max(
        graph.maximum_projection_error,
        projection.maximum_projection_error);
}

// Sum all discrete channels of the dense block, processing one member of each
// radial-reversal orbit and adding its paired stencil with symmetry weights.
inline void bp_streamed_prepare_dense_partition(
    const std::vector<double>& shell_f,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    BPStreamedShellContext& context,
    BPStreamedDenseGraphCache* persistent_cache) {
    context.dense_graph.velocity_cell_volume = context.volume;
    if (context.dense_cells.empty()) {
        context.dense_rhs.assign(shell_f.size(), 0.0);
        context.dense_gross_rhs.assign(shell_f.size(), 0.0);
        return;
    }

    if (persistent_cache != nullptr
        && persistent_cache->valid
        && persistent_cache->dense_cells == context.dense_cells) {
        context.dense_graph = persistent_cache->graph;
        bp_sampled_finalize_graph_activity(context.dense_graph, shell_f);
        context.dense_rhs = bp_reversible_shell_rhs(
            shell_f,
            context.dense_graph,
            p,
            &context.dense_gross_rhs);
        return;
    }

    BPStreamedProjectionCache cache;
    const int npair_phi = static_cast<int>(context.pair_phi_cos.size());
    const int nmu = static_cast<int>(context.scatter_mu.size());
    const int npsi = static_cast<int>(context.scatter_phi_cos.size());
    cache.clear(
        context.dense_cells.size()
        * context.dense_cells.size()
        * static_cast<std::size_t>(npair_phi * nmu * npsi));
    for (int first : context.dense_cells) {
        for (int second : context.dense_cells) {
            const int reverse_first = context.proposal.time_reverse_cell[
                static_cast<std::size_t>(first)];
            const int reverse_second = context.proposal.time_reverse_cell[
                static_cast<std::size_t>(second)];
            if (reverse_first < 0 || reverse_second < 0
                || !context.dense_mask[
                    static_cast<std::size_t>(reverse_first)]
                || !context.dense_mask[
                    static_cast<std::size_t>(reverse_second)]) {
                throw std::runtime_error(
                    "streamed Boltzmann exact block is not closed under radial reversal.");
            }
            if (std::tie(reverse_first, reverse_second)
                < std::tie(first, second)) {
                continue;
            }
            const bool self_reversed =
                reverse_first == first && reverse_second == second;
            const std::size_t reaction_begin =
                context.dense_graph.reactions.size();
            for (int pair_phi = 0; pair_phi < npair_phi; ++pair_phi) {
                BPFAVec3 center_of_mass;
                BPFAVec3 incoming_direction;
                std::array<BPFAVec3, 2> basis;
                double relative_speed = 0.0;
                double sigma = 0.0;
                if (!bp_sampled_pair_kinematics(
                        first,
                        second,
                        pair_phi,
                        context.radius,
                        w,
                        eta,
                        context.pair_phi_cos,
                        context.pair_phi_sin,
                        p,
                        center_of_mass,
                        incoming_direction,
                        basis,
                        relative_speed,
                        sigma)) {
                    continue;
                }
                double polar_sum = 0.0;
                for (int im = 0; im < nmu; ++im) {
                    polar_sum += context.scatter_mu_weight[
                        static_cast<std::size_t>(im)]
                        * bp_fa_scatter_weight_raw(
                            context.scatter_mu[static_cast<std::size_t>(im)],
                            relative_speed,
                            p,
                            context.kernel);
                }
                if (!(polar_sum > 0.0)) {
                    continue;
                }
                const double direct_pair_coefficient =
                    0.25
                    * context.volume[static_cast<std::size_t>(first)]
                    * context.volume[static_cast<std::size_t>(second)]
                    * relative_speed * sigma
                    / static_cast<double>(npair_phi);
                const double reverse_pair_coefficient =
                    0.25
                    * context.volume[static_cast<std::size_t>(reverse_first)]
                    * context.volume[static_cast<std::size_t>(reverse_second)]
                    * relative_speed * sigma
                    / static_cast<double>(npair_phi);
                for (int im = 0; im < nmu; ++im) {
                    const double polar_probability =
                        context.scatter_mu_weight[
                            static_cast<std::size_t>(im)]
                        * bp_fa_scatter_weight_raw(
                            context.scatter_mu[static_cast<std::size_t>(im)],
                            relative_speed,
                            p,
                            context.kernel)
                        / polar_sum;
                    for (int ipsi = 0; ipsi < npsi; ++ipsi) {
                        BPFAVec3 outgoing_first;
                        BPFAVec3 outgoing_second;
                        bp_sampled_outgoing_velocities(
                            center_of_mass,
                            incoming_direction,
                            basis,
                            relative_speed,
                            context.scatter_mu[
                                static_cast<std::size_t>(im)],
                            context.scatter_phi_cos[
                                static_cast<std::size_t>(ipsi)],
                            context.scatter_phi_sin[
                                static_cast<std::size_t>(ipsi)],
                            outgoing_first,
                            outgoing_second);
                        const BPStreamedProjectionRecord& projection =
                            bp_streamed_projection_record(
                                first,
                                second,
                                pair_phi,
                                im,
                                ipsi,
                                outgoing_first,
                                outgoing_second,
                                w,
                                eta,
                                geometry,
                                p,
                                context,
                                cache,
                                0);
                        const BPStreamedProjectionRecord& reverse_projection =
                            bp_streamed_radial_reverse_projection_record(
                                projection,
                                reverse_first,
                                reverse_second,
                                pair_phi,
                                im,
                                ipsi,
                                cache);
                        const double angular_probability =
                            polar_probability / static_cast<double>(npsi);
                        const double symmetry_weight =
                            self_reversed ? 0.5 : 1.0;
                        bp_streamed_add_cached_reaction(
                            context.dense_graph,
                            first,
                            second,
                            projection,
                            symmetry_weight
                                * direct_pair_coefficient
                                * angular_probability);
                        bp_streamed_add_cached_reaction(
                            context.dense_graph,
                            reverse_first,
                            reverse_second,
                            reverse_projection,
                            symmetry_weight
                                * reverse_pair_coefficient
                                * angular_probability);
                    }
                }
            }
            bp_sampled_coalesce_reaction_suffix(
                context.dense_graph, reaction_begin);
        }
    }
    bp_sampled_finalize_graph_activity(context.dense_graph, shell_f);
    context.dense_rhs = bp_reversible_shell_rhs(
        shell_f,
        context.dense_graph,
        p,
        &context.dense_gross_rhs);
    if (persistent_cache != nullptr) {
        persistent_cache->dense_cells = context.dense_cells;
        persistent_cache->graph = context.dense_graph;
        persistent_cache->valid = true;
    }
}

// Both directions use the current f; geometric product powers reproduce the
// projected pair's conserved Maxwellian exponent within the log-floor regime.
inline std::pair<double, double> bp_streamed_channel_activities(
    const std::vector<double>& f,
    int first,
    int second,
    const std::vector<BPReversibleSpecies>& species,
    double floor) {
    const double forward = f[static_cast<std::size_t>(first)]
        * f[static_cast<std::size_t>(second)];
    bool reverse_zero = false;
    double log_reverse = 0.0;
    for (const BPReversibleSpecies& value : species) {
        if (!(value.product_order > 0.0)) {
            continue;
        }
        const double product = f[static_cast<std::size_t>(value.index)];
        if (!(product > 0.0)) {
            reverse_zero = true;
            break;
        }
        log_reverse += value.product_order
            * std::log(std::max(product, floor));
    }
    return {forward, reverse_zero ? 0.0 : std::exp(log_reverse)};
}

// Use one lexicographic representative for a pair and its radial reversal.
inline BPStreamedPairKey bp_streamed_canonical_pair_key(
    int first,
    int second,
    int pair_phi,
    const BPSampledProposal& proposal) {
    const int reverse_first = proposal.time_reverse_cell[
        static_cast<std::size_t>(first)];
    const int reverse_second = proposal.time_reverse_cell[
        static_cast<std::size_t>(second)];
    if (reverse_first < 0 || reverse_second < 0) {
        throw std::runtime_error(
            "streamed Boltzmann pair is not closed under radial reversal.");
    }
    if (std::tie(reverse_first, reverse_second)
        < std::tie(first, second)) {
        return BPStreamedPairKey{
            reverse_first, reverse_second, pair_phi};
    }
    return BPStreamedPairKey{first, second, pair_phi};
}

// Accumulate net and gross two-way activity separately.  Net cancellation can
// be strong near equilibrium while gross rates still measure sampling work.
inline void bp_streamed_accumulate_reaction(
    const std::vector<double>& f,
    int first,
    int second,
    const std::vector<BPReversibleSpecies>& species,
    double coefficient,
    const std::vector<double>& volume,
    double entropy_floor,
    BPStreamedBatch& batch,
    const BPStreamedProjectionKey* projection_key = nullptr) {
    const std::pair<double, double> activity =
        bp_streamed_channel_activities(
            f, first, second, species, entropy_floor);
    const double flux = coefficient * (activity.first - activity.second);
    const double gross = coefficient * (activity.first + activity.second);
    for (const BPReversibleSpecies& value : species) {
        const std::size_t q = static_cast<std::size_t>(value.index);
        batch.rhs_sum[q] += value.net_population_change * flux / volume[q];
        batch.gross_rhs_sum[q] +=
            std::abs(value.net_population_change) * gross / volume[q];
        const double event_rhs = std::abs(
            value.net_population_change * flux / volume[q]);
        if (event_rhs > batch.maximum_event_rhs) {
            batch.maximum_event_rhs = event_rhs;
            batch.maximum_event_rhs_cell = value.index;
            if (projection_key != nullptr) {
                batch.maximum_event_key = *projection_key;
            }
        }
    }
    batch.represented_activity_sum += coefficient * activity.first;
    ++batch.represented_channels;
}

// Importance-sample the complement, adding the same conservative reaction for
// each accepted draw and its radial reverse.  Keep raw sums for later averaging.
inline void bp_streamed_accumulate_batch_range(
    const std::vector<double>& f,
    const BPStreamedShellContext& context,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    std::uint64_t seed,
    std::uint64_t draw_begin,
    std::uint64_t draw_count,
    BPStreamedProjectionCache& cache,
    BPStreamedBatch& batch,
    bool retain_reactions = false,
    const BPStreamedGuidedProposal* guided_proposal = nullptr) {
    if (batch.rhs_sum.empty()) {
        batch.rhs_sum.assign(f.size(), 0.0);
        batch.gross_rhs_sum.assign(f.size(), 0.0);
    }
    if (!(context.complement_probability > 0.0)) {
        return;
    }
    const int angle_samples = std::max(
        1, p.bp_sampled_outgoing_angle_samples);
    const bool exact_angular_sweep =
        angle_samples
        == static_cast<int>(context.scatter_mu.size()
            * context.scatter_phi_cos.size());
    const double target_pair_weight =
        1.0 / static_cast<double>(context.proposal.pair_phi_points);
    for (std::uint64_t local = 0; local < draw_count; ++local) {
        const std::uint64_t draw = draw_begin + local;
        ++batch.attempted_draws;
        int first = 0;
        int second = 0;
        int pair_phi = 0;
        const bool use_guide = guided_proposal != nullptr
            && guided_proposal->fraction > 0.0
            && !guided_proposal->pair.empty()
            && bp_sampled_shifted_halton(draw, 10, seed)
                < guided_proposal->fraction;
        if (use_guide) {
            const int selected = bp_sampled_cdf_index(
                guided_proposal->cdf,
                bp_sampled_shifted_halton(draw, 11, seed));
            const BPStreamedPairKey& key = guided_proposal->pair[
                static_cast<std::size_t>(selected)];
            first = key.first;
            second = key.second;
            pair_phi = key.pair_phi;
        } else {
            std::tie(first, second, pair_phi) =
                bp_sampled_draw_complement_state(
                    context.proposal,
                    context.complement_proposal,
                    draw,
                    seed);
        }
        if (context.dense_mask[static_cast<std::size_t>(first)]
            && context.dense_mask[static_cast<std::size_t>(second)]) {
            throw std::runtime_error(
                "streamed Boltzmann complement draw entered the exact block.");
        }
        int reverse_first = context.proposal.time_reverse_cell[
            static_cast<std::size_t>(first)];
        int reverse_second = context.proposal.time_reverse_cell[
            static_cast<std::size_t>(second)];
        if (reverse_first < 0 || reverse_second < 0) {
            throw std::runtime_error(
                "streamed Boltzmann velocity grid is not closed under radial reversal.");
        }
        if (context.dense_mask[static_cast<std::size_t>(reverse_first)]
            && context.dense_mask[static_cast<std::size_t>(reverse_second)]) {
            throw std::runtime_error(
                "streamed Boltzmann complement is not closed under radial reversal.");
        }
        // Both members of a radial-reversal orbit must use the same finite
        // angular stencil.  Canonicalizing before projection makes the
        // antithetic orbit estimator independent of which member was drawn.
        if (std::tie(reverse_first, reverse_second)
            < std::tie(first, second)) {
            std::swap(first, reverse_first);
            std::swap(second, reverse_second);
        }
        const double proposal_probability =
            bp_sampled_joint_proposal_probability(
                context.proposal, first, second, pair_phi);
        if (!(proposal_probability > 0.0)) {
            throw std::runtime_error(
                "streamed Boltzmann proposal lost full channel support.");
        }
        const double reverse_probability =
            bp_sampled_joint_proposal_probability(
                context.proposal,
                reverse_first,
                reverse_second,
                pair_phi);
        const double orbit_probability =
            proposal_probability + reverse_probability;
        if (!(orbit_probability > 0.0)) {
            throw std::runtime_error(
                "streamed Boltzmann radial-reversal orbit has zero probability.");
        }
        const BPStreamedPairKey pair_key{first, second, pair_phi};
        const double base_pair_probability =
            orbit_probability / context.complement_probability;
        double guided_pair_probability = 0.0;
        double guided_fraction = 0.0;
        if (guided_proposal != nullptr && !guided_proposal->pair.empty()) {
            guided_fraction = guided_proposal->fraction;
            const auto guided = guided_proposal->probability.find(pair_key);
            if (guided != guided_proposal->probability.end()) {
                guided_pair_probability = guided->second;
            }
        }
        // The inverse weight uses the actual mixture probability of the
        // reversal orbit, including complement conditioning and guide draws.
        const double sampling_probability =
            (1.0 - guided_fraction) * base_pair_probability
            + guided_fraction * guided_pair_probability;
        if (!(sampling_probability > 0.0)
            || !std::isfinite(sampling_probability)) {
            throw std::runtime_error(
                "streamed Boltzmann guided proposal has zero pair probability.");
        }

        BPFAVec3 center_of_mass;
        BPFAVec3 incoming_direction;
        std::array<BPFAVec3, 2> basis;
        double relative_speed = 0.0;
        double sigma = 0.0;
        if (!bp_sampled_pair_kinematics(
                first,
                second,
                pair_phi,
                context.radius,
                w,
                eta,
                context.pair_phi_cos,
                context.pair_phi_sin,
                p,
                center_of_mass,
                incoming_direction,
                basis,
                relative_speed,
                sigma)) {
            continue;
        }
        const double pair_activity = relative_speed * sigma;
        if (!(pair_activity > 0.0)) {
            continue;
        }
        // Optional rejection thinning uses a rigorous global sigma*g bound.
        // Inverse acceptance restores the original channel coefficient.
        double inverse_ntc_acceptance = 1.0;
        if (p.bp_sampled_ntc_pair_thinning) {
            // Conditioning changes the component mixture.  The global bound
            // remains rigorous for every component of the complement.
            const double majorant = context.u_sigma_majorant;
            if (!(majorant > 0.0)
                || pair_activity
                    > majorant
                        * (1.0 + 64.0
                            * std::numeric_limits<double>::epsilon())) {
                throw std::runtime_error(
                    "streamed Boltzmann u*sigma majorant was violated.");
            }
            const double acceptance = std::min(
                1.0, pair_activity / majorant);
            const double acceptance_unit = bp_sampled_shifted_halton(
                draw,
                6,
                seed ^ UINT64_C(0xd1b54a32d192ed03));
            if (acceptance_unit >= acceptance) {
                continue;
            }
            inverse_ntc_acceptance = 1.0 / acceptance;
        }
        ++batch.retained_draws;
        const double direct_pair_coefficient =
            0.25
            * context.volume[static_cast<std::size_t>(first)]
            * context.volume[static_cast<std::size_t>(second)]
            * pair_activity * inverse_ntc_acceptance
            * target_pair_weight / sampling_probability;
        const double reverse_pair_coefficient =
            0.25
            * context.volume[static_cast<std::size_t>(reverse_first)]
            * context.volume[static_cast<std::size_t>(reverse_second)]
            * pair_activity * inverse_ntc_acceptance
            * target_pair_weight / sampling_probability;

        // A full angular sweep uses quadrature weights directly.  Otherwise
        // sampled angular draws each contribute the reciprocal sample count.
        for (int angle = 0; angle < angle_samples; ++angle) {
            std::array<int, 2> scatter{0, 0};
            double angular_probability =
                1.0 / static_cast<double>(angle_samples);
            if (exact_angular_sweep) {
                scatter[0] = angle
                    % static_cast<int>(context.scatter_mu.size());
                scatter[1] = angle
                    / static_cast<int>(context.scatter_mu.size());
                double polar_sum = 0.0;
                for (std::size_t im = 0;
                     im < context.scatter_mu.size();
                     ++im) {
                    polar_sum += context.scatter_mu_weight[im]
                        * bp_fa_scatter_weight_raw(
                            context.scatter_mu[im],
                            relative_speed,
                            p,
                            context.kernel);
                }
                angular_probability =
                    context.scatter_mu_weight[
                        static_cast<std::size_t>(scatter[0])]
                    * bp_fa_scatter_weight_raw(
                        context.scatter_mu[
                            static_cast<std::size_t>(scatter[0])],
                        relative_speed,
                        p,
                        context.kernel)
                    / polar_sum
                    / static_cast<double>(
                        context.scatter_phi_cos.size());
            } else {
                const std::array<double, 2> units =
                    bp_sampled_stratified_outgoing_units(
                        draw,
                        angle,
                        angle_samples,
                        p.bp_boltzmann_scatter_mu_points,
                        seed);
                scatter = bp_streamed_discrete_scatter_indices(
                    units[0], units[1], relative_speed, context, p);
            }
            BPFAVec3 outgoing_first;
            BPFAVec3 outgoing_second;
            bp_sampled_outgoing_velocities(
                center_of_mass,
                incoming_direction,
                basis,
                relative_speed,
                context.scatter_mu[static_cast<std::size_t>(scatter[0])],
                context.scatter_phi_cos[static_cast<std::size_t>(scatter[1])],
                context.scatter_phi_sin[static_cast<std::size_t>(scatter[1])],
                outgoing_first,
                outgoing_second);
            const BPStreamedProjectionRecord& projection =
                bp_streamed_projection_record(
                    first,
                    second,
                    pair_phi,
                    scatter[0],
                    scatter[1],
                    outgoing_first,
                    outgoing_second,
                    w,
                    eta,
                    geometry,
                    p,
                    context,
                    cache,
                    0);
            const BPStreamedProjectionRecord& reverse_projection =
                bp_streamed_radial_reverse_projection_record(
                    projection,
                    reverse_first,
                    reverse_second,
                    pair_phi,
                    scatter[0],
                    scatter[1],
                    cache);
            const double direct_coefficient =
                direct_pair_coefficient * angular_probability;
            const double reverse_coefficient =
                reverse_pair_coefficient * angular_probability;
            const BPStreamedProjectionKey direct_key{
                first,
                second,
                pair_phi,
                scatter[0],
                scatter[1],
                0};
            const BPStreamedProjectionKey reverse_key{
                reverse_first,
                reverse_second,
                pair_phi,
                scatter[0],
                scatter[1],
                1};
            if (!projection.represented) {
                const double direct_omitted = direct_coefficient
                    * f[static_cast<std::size_t>(first)]
                    * f[static_cast<std::size_t>(second)];
                if (projection.outside_domain) {
                    batch.omitted_domain_activity_sum += direct_omitted;
                } else {
                    batch.omitted_projection_activity_sum += direct_omitted;
                }
                if (retain_reactions) {
                    auto& omitted = projection.outside_domain
                        ? batch.omitted_domain_coefficient_sum
                        : batch.omitted_projection_coefficient_sum;
                    omitted[direct_key] += direct_coefficient;
                }
            } else {
                batch.maximum_projection_error = std::max(
                    batch.maximum_projection_error,
                    projection.maximum_projection_error);
                bp_streamed_accumulate_reaction(
                    f,
                    first,
                    second,
                    projection.direct_species,
                    direct_coefficient,
                    context.volume,
                    p.bp_reversible_entropy_floor,
                    batch,
                    &direct_key);
                if (retain_reactions) {
                    batch.reaction_coefficient_sum[direct_key]
                        += direct_coefficient;
                }
            }
            if (!reverse_projection.represented) {
                const double reverse_omitted = reverse_coefficient
                    * f[static_cast<std::size_t>(reverse_first)]
                    * f[static_cast<std::size_t>(reverse_second)];
                if (reverse_projection.outside_domain) {
                    batch.omitted_domain_activity_sum += reverse_omitted;
                } else {
                    batch.omitted_projection_activity_sum += reverse_omitted;
                }
                if (retain_reactions) {
                    auto& omitted = reverse_projection.outside_domain
                        ? batch.omitted_domain_coefficient_sum
                        : batch.omitted_projection_coefficient_sum;
                    omitted[reverse_key] += reverse_coefficient;
                }
            } else {
                batch.maximum_projection_error = std::max(
                    batch.maximum_projection_error,
                    reverse_projection.maximum_projection_error);
                bp_streamed_accumulate_reaction(
                    f,
                    reverse_first,
                    reverse_second,
                    reverse_projection.direct_species,
                    reverse_coefficient,
                    context.volume,
                    p.bp_reversible_entropy_floor,
                    batch,
                    &reverse_key);
                if (retain_reactions) {
                    batch.reaction_coefficient_sum[reverse_key]
                        += reverse_coefficient;
                }
            }
        }
    }
}

// Rank pilot pairs by gross population change plus a normalized radial heat
// moment change, then build a finite guide mixed with the full-support proposal.
inline BPStreamedGuidedProposal bp_streamed_build_guided_proposal(
    const std::vector<double>& f,
    const BPStreamedShellContext& context,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const Params& p,
    const BPStreamedBatch& pilot,
    const BPStreamedProjectionCache& cache) {
    BPStreamedGuidedProposal guide;
    if (!(p.bp_sampled_guided_fraction > 0.0)
        || p.bp_sampled_guided_pairs <= 0
        || pilot.reaction_coefficient_sum.empty()) {
        return guide;
    }
    double maximum_heat_weight = 1.0;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        maximum_heat_weight = std::max(
            maximum_heat_weight,
            std::abs(
                bp_reversible_radial_velocity(q, w, p)
                * bp_reversible_energy(q, context.radius, w, eta, p)));
    }
    std::unordered_map<
        BPStreamedPairKey,
        double,
        BPStreamedPairKeyHash> score;
    score.reserve(pilot.reaction_coefficient_sum.size());
    for (const auto& entry : pilot.reaction_coefficient_sum) {
        const auto projection = cache.record.find(entry.first);
        if (projection == cache.record.end()
            || !projection->second.represented
            || !(entry.second > 0.0)) {
            continue;
        }
        const std::pair<double, double> activity =
            bp_streamed_channel_activities(
                f,
                entry.first.first,
                entry.first.second,
                projection->second.direct_species,
                p.bp_reversible_entropy_floor);
        double population_change = 0.0;
        double heat_change = 0.0;
        for (const BPReversibleSpecies& species
             : projection->second.direct_species) {
            population_change += std::abs(species.net_population_change);
            heat_change += species.net_population_change
                * bp_reversible_radial_velocity(species.index, w, p)
                * bp_reversible_energy(
                    species.index, context.radius, w, eta, p);
        }
        const double moment_weight = population_change
            + std::abs(heat_change) / maximum_heat_weight;
        const double gross_score = entry.second
            * (activity.first + activity.second)
            * moment_weight;
        if (!(gross_score > 0.0) || !std::isfinite(gross_score)) {
            continue;
        }
        const BPStreamedPairKey key = bp_streamed_canonical_pair_key(
            entry.first.first,
            entry.first.second,
            entry.first.pair_phi,
            context.proposal);
        score[key] += gross_score;
    }
    std::vector<std::pair<BPStreamedPairKey, double>> ranked;
    ranked.reserve(score.size());
    for (const auto& entry : score) {
        ranked.push_back(entry);
    }
    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const auto& first, const auto& second) {
            if (first.second != second.second) {
                return first.second > second.second;
            }
            return std::tie(
                first.first.first,
                first.first.second,
                first.first.pair_phi)
                < std::tie(
                    second.first.first,
                    second.first.second,
                    second.first.pair_phi);
        });
    if (ranked.size()
        > static_cast<std::size_t>(p.bp_sampled_guided_pairs)) {
        ranked.resize(static_cast<std::size_t>(p.bp_sampled_guided_pairs));
    }
    double total_score = 0.0;
    for (const auto& entry : ranked) {
        total_score += entry.second;
    }
    if (!(total_score > 0.0) || !std::isfinite(total_score)) {
        return guide;
    }
    guide.fraction = p.bp_sampled_guided_fraction;
    guide.pair.reserve(ranked.size());
    guide.cdf.reserve(ranked.size());
    guide.probability.reserve(ranked.size());
    double cumulative = 0.0;
    for (const auto& entry : ranked) {
        const double probability = entry.second / total_score;
        guide.pair.push_back(entry.first);
        guide.probability.emplace(entry.first, probability);
        cumulative += probability;
        guide.cdf.push_back(cumulative);
    }
    guide.cdf.back() = 1.0;
    return guide;
}

// Add the dense contribution once and divide only the sampled sum by draws.
inline std::vector<double> bp_streamed_normalized_values(
    const std::vector<double>& sum,
    int samples,
    const std::vector<double>& exact_part) {
    if (sum.size() != exact_part.size()) {
        throw std::runtime_error(
            "streamed Boltzmann exact and sampled arrays have different sizes.");
    }
    std::vector<double> value = exact_part;
    if (samples <= 0) {
        return value;
    }
    const double inverse = 1.0 / static_cast<double>(samples);
    for (std::size_t q = 0; q < sum.size(); ++q) {
        value[q] += inverse * sum[q];
    }
    return value;
}

// Choose sample count using separate controller draws, then assess independent
// production replicates with distribution and heat-moment error criteria.
// Extensions keep existing draw ranges and append only the missing samples.
inline BPStreamedOperatorEstimate bp_streamed_estimate_shell_operator(
    const std::vector<double>& f,
    const BPStreamedShellContext& context,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    std::uint64_t first_seed,
    std::uint64_t second_seed,
    BPStreamedProjectionCache& cache,
    int starting_samples = 0,
    bool adapt_samples = true,
    bool retain_reactions = false) {
    BPStreamedOperatorEstimate estimate;
    int samples = starting_samples > 0
        ? starting_samples
        : p.bp_sampled_min_samples_per_batch;
    samples = std::max(
        p.bp_sampled_min_samples_per_batch,
        std::min(samples, p.bp_sampled_max_samples_per_batch));
    auto batch_error = [&](const BPStreamedBatch& first,
                           const BPStreamedBatch& second,
                           int count) {
        const std::vector<double> first_rhs =
            bp_streamed_normalized_values(
                first.rhs_sum, count, context.dense_rhs);
        const std::vector<double> second_rhs =
            bp_streamed_normalized_values(
                second.rhs_sum, count, context.dense_rhs);
        const std::vector<double> first_gross =
            bp_streamed_normalized_values(
                first.gross_rhs_sum, count, context.dense_gross_rhs);
        const std::vector<double> second_gross =
            bp_streamed_normalized_values(
                second.gross_rhs_sum, count, context.dense_gross_rhs);
        return bp_sampled_batch_error(
            first_rhs,
            second_rhs,
            context.volume,
            context.radius,
            w,
            eta,
            p,
            &first_gross,
            &second_gross);
    };

    if (!(context.complement_probability > 0.0)) {
        estimate.rhs = context.dense_rhs;
        estimate.gross_rhs = context.dense_gross_rhs;
        estimate.samples_per_batch = 0;
        estimate.converged = true;
        return estimate;
    }

    // Guide pilots select a proposal, controller pairs choose its sample
    // count, and independent production batches estimate the final operator.
    BPStreamedBatch guide_pilot;
    if (p.bp_sampled_guided_pilot_samples > 0
        && p.bp_sampled_guided_pairs > 0
        && p.bp_sampled_guided_fraction > 0.0) {
        const int pilot_replicates = std::max(
            2,
            std::min(4, p.bp_sampled_replicates_per_estimate));
        const std::uint64_t guide_seed = bp_sampled_splitmix64(
            first_seed ^ UINT64_C(0x243f6a8885a308d3));
        int assigned_pilot_samples = 0;
        for (int replicate = 0;
             replicate < pilot_replicates;
             ++replicate) {
            const int remaining =
                p.bp_sampled_guided_pilot_samples
                - assigned_pilot_samples;
            const int count = remaining
                / (pilot_replicates - replicate);
            assigned_pilot_samples += count;
            if (count <= 0) {
                continue;
            }
            bp_streamed_accumulate_batch_range(
                f,
                context,
                w,
                eta,
                geometry,
                p,
                bp_sampled_splitmix64(
                    guide_seed
                    ^ UINT64_C(0x510e527fade682d1)
                    ^ static_cast<std::uint64_t>(replicate)),
                0,
                static_cast<std::uint64_t>(count),
                cache,
                guide_pilot,
                true,
                nullptr);
        }
        estimate.controller_attempted_draws +=
            guide_pilot.attempted_draws;
        estimate.controller_retained_draws +=
            guide_pilot.retained_draws;
        estimate.controller_represented_channels +=
            guide_pilot.represented_channels;
    }
    const BPStreamedGuidedProposal guided_proposal =
        bp_streamed_build_guided_proposal(
            f, context, w, eta, p, guide_pilot, cache);

    if (adapt_samples && samples < p.bp_sampled_max_samples_per_batch) {
        BPStreamedBatch controller_first;
        BPStreamedBatch controller_second;
        const std::uint64_t controller_first_seed = bp_sampled_splitmix64(
            first_seed ^ UINT64_C(0x3c6ef372fe94f82b));
        const std::uint64_t controller_second_seed = bp_sampled_splitmix64(
            second_seed ^ UINT64_C(0xa54ff53a5f1d36f1));
        int accumulated = 0;
        while (true) {
            const int extension = samples - accumulated;
            bp_streamed_accumulate_batch_range(
                f,
                context,
                w,
                eta,
                geometry,
                p,
                controller_first_seed,
                static_cast<std::uint64_t>(accumulated),
                static_cast<std::uint64_t>(extension),
                cache,
                controller_first,
                false,
                &guided_proposal);
            bp_streamed_accumulate_batch_range(
                f,
                context,
                w,
                eta,
                geometry,
                p,
                controller_second_seed,
                static_cast<std::uint64_t>(accumulated),
                static_cast<std::uint64_t>(extension),
                cache,
                controller_second,
                false,
                &guided_proposal);
            accumulated = samples;
            const BPSampledBatchError controller_error = batch_error(
                controller_first, controller_second, samples);
            const std::uint64_t minimum_represented =
                static_cast<std::uint64_t>(std::min(
                    p.bp_sampled_ntc_pilot_samples, samples));
            const bool controller_converged =
                controller_first.represented_channels
                    >= minimum_represented
                && controller_second.represented_channels
                    >= minimum_represented
                // The controller and production batches are independent.
                // Leave a fourfold margin so a production batch does not
                // routinely fail after the controller selected its size.
                && controller_error.rhs
                    <= 0.25 * p.bp_sampled_rhs_tolerance
                && controller_error.heat
                    <= 0.25 * p.bp_sampled_heat_tolerance;
            if (controller_converged
                || samples >= p.bp_sampled_max_samples_per_batch) {
                break;
            }
            samples = std::min(
                p.bp_sampled_max_samples_per_batch, 2 * samples);
            ++estimate.adaptive_rounds;
        }
        estimate.controller_attempted_draws +=
            controller_first.attempted_draws
            + controller_second.attempted_draws;
        estimate.controller_retained_draws +=
            controller_first.retained_draws
            + controller_second.retained_draws;
        estimate.controller_represented_channels +=
            controller_first.represented_channels
            + controller_second.represented_channels;
    }

    // Replicate disagreement estimates sampling error on net RHS and heat
    // transport, with gross-activity floors near cancellation.
    const int production_batches = std::max(
        2, p.bp_sampled_replicates_per_estimate);
    estimate.additional.resize(
        static_cast<std::size_t>(production_batches - 2));
    std::vector<BPStreamedBatch*> batch;
    batch.reserve(static_cast<std::size_t>(production_batches));
    batch.push_back(&estimate.first);
    batch.push_back(&estimate.second);
    for (BPStreamedBatch& value : estimate.additional) {
        batch.push_back(&value);
    }
    std::vector<std::uint64_t> production_seed(
        static_cast<std::size_t>(production_batches), 0);
    std::vector<std::vector<double>> rhs_replicates;
    std::vector<std::vector<double>> gross_replicates;
    rhs_replicates.reserve(batch.size());
    gross_replicates.reserve(batch.size());
    for (int replicate = 0; replicate < production_batches; ++replicate) {
        const std::uint64_t replicate_seed = replicate == 0
            ? first_seed
            : (replicate == 1
                ? second_seed
                : bp_sampled_splitmix64(
                    first_seed
                    ^ UINT64_C(0x1f83d9abfb41bd6b)
                    ^ static_cast<std::uint64_t>(replicate)));
        production_seed[static_cast<std::size_t>(replicate)] =
            replicate_seed;
        bp_streamed_accumulate_batch_range(
            f,
            context,
            w,
            eta,
            geometry,
            p,
            replicate_seed,
            0,
            static_cast<std::uint64_t>(samples),
            cache,
            *batch[static_cast<std::size_t>(replicate)],
            retain_reactions,
            &guided_proposal);
    }

    bool enough_represented_channels = false;
    auto evaluate_production_batches = [&]() {
        rhs_replicates.clear();
        gross_replicates.clear();
        enough_represented_channels = true;
        const std::uint64_t minimum_represented =
            static_cast<std::uint64_t>(std::min(
                p.bp_sampled_ntc_pilot_samples, samples));
        for (BPStreamedBatch* current : batch) {
            enough_represented_channels = enough_represented_channels
                && current->represented_channels >= minimum_represented;
            rhs_replicates.push_back(bp_streamed_normalized_values(
                current->rhs_sum, samples, context.dense_rhs));
            gross_replicates.push_back(bp_streamed_normalized_values(
                current->gross_rhs_sum,
                samples,
                context.dense_gross_rhs));
        }
        estimate.error = bp_sampled_replicated_batch_error(
            rhs_replicates,
            context.volume,
            context.radius,
            w,
            eta,
            p,
            &gross_replicates);
        estimate.converged =
            enough_represented_channels
            && estimate.error.rhs <= p.bp_sampled_rhs_tolerance
            && estimate.error.heat <= p.bp_sampled_heat_tolerance;
    };
    evaluate_production_batches();
    while (adapt_samples
           && !estimate.converged
           && samples < p.bp_sampled_max_samples_per_batch) {
        const int next_samples = std::min(
            p.bp_sampled_max_samples_per_batch, 2 * samples);
        const int extension = next_samples - samples;
        for (int replicate = 0;
             replicate < production_batches;
             ++replicate) {
            bp_streamed_accumulate_batch_range(
                f,
                context,
                w,
                eta,
                geometry,
                p,
                production_seed[static_cast<std::size_t>(replicate)],
                static_cast<std::uint64_t>(samples),
                static_cast<std::uint64_t>(extension),
                cache,
                *batch[static_cast<std::size_t>(replicate)],
                retain_reactions,
                &guided_proposal);
        }
        samples = next_samples;
        ++estimate.adaptive_rounds;
        evaluate_production_batches();
    }
    estimate.rhs.resize(f.size(), 0.0);
    estimate.gross_rhs.resize(f.size(), 0.0);
    const double inverse_batches =
        1.0 / static_cast<double>(production_batches);
    for (int replicate = 0; replicate < production_batches; ++replicate) {
        for (std::size_t q = 0; q < f.size(); ++q) {
            estimate.rhs[q] += inverse_batches
                * rhs_replicates[static_cast<std::size_t>(replicate)][q];
            estimate.gross_rhs[q] += inverse_batches
                * gross_replicates[static_cast<std::size_t>(replicate)][q];
        }
    }
    estimate.samples_per_batch = samples;
    return estimate;
}

// Velocity volumes weight the local H functional used to assess relaxation.
inline double bp_streamed_shell_entropy(
    const std::vector<double>& f,
    const std::vector<double>& volume,
    double floor) {
    double entropy = 0.0;
    for (std::size_t q = 0; q < f.size(); ++q) {
        if (!(f[q] >= 0.0) || !std::isfinite(f[q])) {
            throw std::runtime_error(
                "streamed Boltzmann entropy requires a nonnegative DF.");
        }
        if (f[q] > 0.0) {
            entropy += volume[q] * f[q]
                * (std::log(std::max(f[q], floor)) - 1.0);
        }
    }
    return entropy;
}

// Measure the distribution change in the same d^3v measure as the cached
// operator, normalized by the larger current/reference represented population.
inline double bp_streamed_operator_reference_l1(
    const std::vector<double>& f,
    const BPStreamedOperatorGraphCache& cache) {
    if (!cache.valid
        || cache.reference_f.size() != f.size()
        || cache.graph.velocity_cell_volume.size() != f.size()) {
        return std::numeric_limits<double>::infinity();
    }
    long double difference = 0.0L;
    long double reference_mass = 0.0L;
    long double current_mass = 0.0L;
    for (std::size_t q = 0; q < f.size(); ++q) {
        const long double volume =
            cache.graph.velocity_cell_volume[q];
        difference += volume * std::abs(f[q] - cache.reference_f[q]);
        reference_mass += volume * cache.reference_f[q];
        current_mass += volume * f[q];
    }
    const long double scale = std::max({
        reference_mass,
        current_mass,
        std::numeric_limits<long double>::min(),
    });
    return static_cast<double>(difference / scale);
}

// Sum repeated sampled coefficients, sort represented keys deterministically,
// and average over production batches and draws.  The dense graph is added
// once; omitted channels retain the same sampling normalization.
inline BPReversibleGraph bp_streamed_materialize_reaction_graph(
    const std::vector<double>& f,
    const BPStreamedOperatorEstimate& estimate,
    const BPStreamedShellContext& context,
    const BPStreamedProjectionCache& cache) {
    std::unordered_map<
        BPStreamedProjectionKey,
        double,
        BPStreamedProjectionKeyHash> coefficient_sum =
            estimate.first.reaction_coefficient_sum;
    coefficient_sum.reserve(
        coefficient_sum.size()
        + estimate.second.reaction_coefficient_sum.size());
    for (const auto& entry : estimate.second.reaction_coefficient_sum) {
        coefficient_sum[entry.first] += entry.second;
    }
    for (const BPStreamedBatch& batch : estimate.additional) {
        coefficient_sum.reserve(
            coefficient_sum.size()
            + batch.reaction_coefficient_sum.size());
        for (const auto& entry : batch.reaction_coefficient_sum) {
            coefficient_sum[entry.first] += entry.second;
        }
    }
    using CoefficientMap = std::unordered_map<
        BPStreamedProjectionKey,
        double,
        BPStreamedProjectionKeyHash>;
    CoefficientMap omitted_domain_sum =
        estimate.first.omitted_domain_coefficient_sum;
    CoefficientMap omitted_projection_sum =
        estimate.first.omitted_projection_coefficient_sum;
    const auto accumulate_coefficients = [](
        CoefficientMap& destination,
        const CoefficientMap& source) {
        destination.reserve(destination.size() + source.size());
        for (const auto& entry : source) {
            destination[entry.first] += entry.second;
        }
    };
    accumulate_coefficients(
        omitted_domain_sum,
        estimate.second.omitted_domain_coefficient_sum);
    accumulate_coefficients(
        omitted_projection_sum,
        estimate.second.omitted_projection_coefficient_sum);
    for (const BPStreamedBatch& batch : estimate.additional) {
        accumulate_coefficients(
            omitted_domain_sum, batch.omitted_domain_coefficient_sum);
        accumulate_coefficients(
            omitted_projection_sum,
            batch.omitted_projection_coefficient_sum);
    }
    std::vector<std::pair<BPStreamedProjectionKey, double>> ordered;
    ordered.reserve(coefficient_sum.size());
    for (const auto& entry : coefficient_sum) {
        ordered.push_back(entry);
    }
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto& first, const auto& second) {
            const BPStreamedProjectionKey& a = first.first;
            const BPStreamedProjectionKey& b = second.first;
            return std::tie(
                a.first,
                a.second,
                a.pair_phi,
                a.scatter_mu,
                a.scatter_phi,
                a.projection_variant)
                < std::tie(
                    b.first,
                    b.second,
                    b.pair_phi,
                    b.scatter_mu,
                    b.scatter_phi,
                    b.projection_variant);
        });
    BPReversibleGraph graph = context.dense_graph;
    graph.reactions.reserve(graph.reactions.size() + ordered.size());
    const int production_batches =
        2 + static_cast<int>(estimate.additional.size());
    const double normalization = estimate.samples_per_batch > 0
        ? 1.0
            / (static_cast<double>(production_batches)
               * static_cast<double>(estimate.samples_per_batch))
        : 0.0;
    for (const auto& entry : ordered) {
        const double coefficient = normalization * entry.second;
        if (!(coefficient > 0.0) || !std::isfinite(coefficient)) {
            continue;
        }
        const auto found = cache.record.find(entry.first);
        if (found == cache.record.end() || !found->second.represented) {
            throw std::runtime_error(
                "streamed Boltzmann retained reaction lost its projection cache record.");
        }
        BPReversibleReaction reaction;
        reaction.incoming_first = entry.first.first;
        reaction.incoming_second = entry.first.second;
        reaction.coefficient = coefficient;
        reaction.species = found->second.direct_species;
        graph.reactions.push_back(std::move(reaction));
        graph.maximum_projection_error = std::max(
            graph.maximum_projection_error,
            found->second.maximum_projection_error);
    }
    const auto append_omitted = [&cache, normalization](
        const CoefficientMap& sum,
        std::vector<BPReversibleOmittedChannel>& destination) {
        destination.reserve(destination.size() + sum.size());
        for (const auto& entry : sum) {
            const double coefficient = normalization * entry.second;
            if (!(coefficient > 0.0) || !std::isfinite(coefficient)) {
                continue;
            }
            const auto found = cache.record.find(entry.first);
            if (found == cache.record.end() || found->second.represented) {
                throw std::runtime_error(
                    "streamed Boltzmann omitted channel lost its projection record.");
            }
            destination.push_back(BPReversibleOmittedChannel{
                entry.first.first,
                entry.first.second,
                coefficient,
                found->second.outside_domain_mask,
            });
        }
    };
    append_omitted(
        omitted_domain_sum, graph.omitted_domain_channels);
    append_omitted(
        omitted_projection_sum, graph.omitted_projection_channels);
    bp_sampled_finalize_graph_activity(graph, f);
    return graph;
}

// Reuse an admitted graph or estimate a new one, then integrate its coupled
// reversible reactions for dt.  Report sampling cost separately from evolution.
inline BPStreamedShellResult bp_streamed_advance_shell(
    const std::vector<double>& initial,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    std::uint64_t shell_salt,
    BPStreamedProjectionCache& cache,
    BPStreamedDenseGraphCache* dense_graph_cache = nullptr,
    BPStreamedOperatorGraphCache* operator_graph_cache = nullptr,
    int initial_samples = 0,
    bool retain_projection_cache = false) {
    BPStreamedShellResult result;
    const auto context_begin = std::chrono::steady_clock::now();
    const double reference_l1 = operator_graph_cache != nullptr
        ? bp_streamed_operator_reference_l1(initial, *operator_graph_cache)
        : std::numeric_limits<double>::infinity();
    // This path reuses the existing sampled edge set inside its prescribed
    // distribution-change interval; reaction populations still evolve freshly.
    if (p.bp_sampled_reuse_operator_graph
        && operator_graph_cache != nullptr
        && operator_graph_cache->valid
        && reference_l1 <= p.bp_sampled_operator_refresh_l1_fraction) {
        result.context_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - context_begin).count();
        result.samples_per_batch = operator_graph_cache->samples_per_batch;
        result.adaptive_rounds = operator_graph_cache->adaptive_rounds;
        result.converged = operator_graph_cache->converged;
        result.rhs_relative_error =
            operator_graph_cache->rhs_relative_error;
        result.rhs_gross_error =
            operator_graph_cache->rhs_gross_error;
        result.heat_relative_error =
            operator_graph_cache->heat_relative_error;
        result.heat_gross_error =
            operator_graph_cache->heat_gross_error;
        result.energy_invariant_error =
            operator_graph_cache->energy_invariant_error;
        result.dense_reactions =
            operator_graph_cache->dense_reactions;
        result.complement_reactions =
            operator_graph_cache->complement_reactions;
        result.represented_channels =
            operator_graph_cache->graph.reactions.size();
        result.reused_operator_graph = true;
        result.operator_reference_l1 = reference_l1;
        const auto advance_begin = std::chrono::steady_clock::now();
        result.step = bp_reversible_advance_shell(
            initial, operator_graph_cache->graph, dt, p);
        result.advance_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - advance_begin).count();
        return result;
    }
    BPStreamedShellContext context = bp_streamed_make_shell_context(
        initial, radius, w, eta, geometry, p, dense_graph_cache);
    // Multiply in size_t after capping each positive factor.  Performing this
    // product in int could overflow for a syntactically valid large parameter
    // file and turn a modest reserve hint into a huge allocation request.
    const std::size_t expected_cache_entries = std::min<std::size_t>(
        static_cast<std::size_t>(262144),
        static_cast<std::size_t>(p.bp_sampled_max_samples_per_batch)
            * static_cast<std::size_t>(
                std::max(1, p.bp_sampled_outgoing_angle_samples)));
    if (!retain_projection_cache) {
        cache.clear(expected_cache_entries);
    } else if (expected_cache_entries > cache.record.bucket_count()) {
        cache.record.reserve(expected_cache_entries);
    }
    result.context_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - context_begin).count();
    const std::uint64_t state_seed = p.bp_sampled_fixed_geometric_sequence
        ? bp_sampled_splitmix64(p.bp_sampled_seed ^ shell_salt)
        : bp_sampled_shell_hash(
            initial, radius, p.bp_sampled_seed ^ shell_salt);
    const std::uint64_t cache_hits_before = cache.hits;
    const std::uint64_t cache_misses_before = cache.misses;
    const auto estimate_begin = std::chrono::steady_clock::now();
    BPStreamedOperatorEstimate estimate =
        bp_streamed_estimate_shell_operator(
            initial,
            context,
            w,
            eta,
            geometry,
            p,
            bp_sampled_splitmix64(
                state_seed ^ UINT64_C(0x6a09e667f3bcc909)),
            bp_sampled_splitmix64(
                state_seed ^ UINT64_C(0xbb67ae8584caa73b)),
            cache,
            initial_samples,
            true,
            true);
    result.estimate_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - estimate_begin).count();
    if (!estimate.converged && p.bp_sampled_fail_on_unconverged) {
        std::ostringstream message;
        message
            << "streamed Boltzmann failed its independent-batch tolerance at r="
            << radius
            << ": rhs_error=" << estimate.error.rhs
            << ", heat_error=" << estimate.error.heat
            << ", samples_per_batch=" << estimate.samples_per_batch;
        throw std::runtime_error(message.str());
    }
    const auto materialize_begin = std::chrono::steady_clock::now();
    BPReversibleGraph graph = bp_streamed_materialize_reaction_graph(
        initial, estimate, context, cache);
    result.materialize_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - materialize_begin).count();
    result.cache_hits = cache.hits - cache_hits_before;
    result.cache_misses = cache.misses - cache_misses_before;
    result.dense_reactions = context.dense_graph.reactions.size();
    result.complement_reactions = graph.reactions.size()
        - context.dense_graph.reactions.size();
    const BPReversibleGraph* active_graph = &graph;
    if (p.bp_sampled_reuse_operator_graph
        && operator_graph_cache != nullptr) {
        operator_graph_cache->reference_f = initial;
        operator_graph_cache->graph = std::move(graph);
        operator_graph_cache->samples_per_batch =
            estimate.samples_per_batch;
        operator_graph_cache->adaptive_rounds = estimate.adaptive_rounds;
        operator_graph_cache->dense_reactions = result.dense_reactions;
        operator_graph_cache->complement_reactions =
            result.complement_reactions;
        operator_graph_cache->rhs_relative_error = estimate.error.rhs;
        operator_graph_cache->rhs_gross_error = estimate.error.rhs_gross;
        operator_graph_cache->heat_relative_error = estimate.error.heat;
        operator_graph_cache->heat_gross_error = estimate.error.heat_gross;
        operator_graph_cache->energy_invariant_error =
            estimate.error.energy_invariant;
        operator_graph_cache->converged = estimate.converged;
        operator_graph_cache->valid = true;
        active_graph = &operator_graph_cache->graph;
    }
    const auto advance_begin = std::chrono::steady_clock::now();
    result.step = bp_reversible_advance_shell(initial, *active_graph, dt, p);
    result.advance_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - advance_begin).count();
    result.samples_per_batch = estimate.samples_per_batch;
    result.adaptive_rounds = estimate.adaptive_rounds;
    result.converged = estimate.converged;
    result.rhs_relative_error = estimate.error.rhs;
    result.rhs_gross_error = estimate.error.rhs_gross;
    result.heat_relative_error = estimate.error.heat;
    result.heat_gross_error = estimate.error.heat_gross;
    result.energy_invariant_error = estimate.error.energy_invariant;
    result.attempted_draws = estimate.first.attempted_draws
        + estimate.second.attempted_draws
        + estimate.controller_attempted_draws;
    result.retained_draws = estimate.first.retained_draws
        + estimate.second.retained_draws
        + estimate.controller_retained_draws;
    for (const BPStreamedBatch& batch : estimate.additional) {
        result.attempted_draws += batch.attempted_draws;
        result.retained_draws += batch.retained_draws;
    }
    result.represented_channels = active_graph->reactions.size();
    result.operator_reference_l1 = 0.0;
    result.step.maximum_projection_error = std::max({
        result.step.maximum_projection_error,
        estimate.first.maximum_projection_error,
        estimate.second.maximum_projection_error,
    });
    for (const BPStreamedBatch& batch : estimate.additional) {
        result.step.maximum_projection_error = std::max(
            result.step.maximum_projection_error,
            batch.maximum_projection_error);
    }
    const int production_batches =
        2 + static_cast<int>(estimate.additional.size());
    const double inverse_samples = estimate.samples_per_batch > 0
        ? 1.0
            / (static_cast<double>(production_batches)
               * static_cast<double>(estimate.samples_per_batch))
        : 0.0;
    double omitted_domain_sum =
        estimate.first.omitted_domain_activity_sum
        + estimate.second.omitted_domain_activity_sum;
    double omitted_projection_sum =
        estimate.first.omitted_projection_activity_sum
        + estimate.second.omitted_projection_activity_sum;
    for (const BPStreamedBatch& batch : estimate.additional) {
        omitted_domain_sum += batch.omitted_domain_activity_sum;
        omitted_projection_sum += batch.omitted_projection_activity_sum;
    }
    result.step.omitted_domain_activity =
        context.dense_graph.omitted_domain_activity
        + inverse_samples * omitted_domain_sum;
    result.step.omitted_projection_activity =
        context.dense_graph.omitted_projection_activity
        + inverse_samples * omitted_projection_sum;
    const double attempted = result.step.represented_activity
        + result.step.omitted_domain_activity
        + result.step.omitted_projection_activity;
    result.step.omitted_fraction = attempted > 0.0
        ? (result.step.omitted_domain_activity
           + result.step.omitted_projection_activity) / attempted
        : 0.0;
    return result;
}

/// Applies the streamed reversible operator independently in each radial cell.
inline BPCollisionStepInfo apply_bp_streamed_reversible_boltzmann_collision_step(
    std::vector<double>& f,
    std::vector<double>* tracer_f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    double dt,
    const Params& p,
    const MpiContext& mpi,
    BPStreamedCollisionWorkspace* persistent_workspace = nullptr) {
    BPCollisionStepInfo info;
    if (!using_bp_streamed_reversible_boltzmann(p)
        || !(dt > 0.0)
        || p.bp_collision_strength == 0.0) {
        return info;
    }
    if (mpi.size != 1) {
        throw std::runtime_error(
            "boltzmann_reversible_streamed is OpenMP-only and never uses MPI.");
    }
    if (tracer_f != nullptr && !tracer_f->empty()) {
        throw std::runtime_error(
            "boltzmann_reversible_streamed passive-tracer collisions are not implemented.");
    }
    const PhaseSpaceCellGeometry geometry =
        make_cell_average_eta_geometry(make_positive_radial_cell_axis(p), p);
    validate_cell_average_eta_views(
        geometry,
        r,
        w,
        L,
        wq,
        Lq,
        p,
        "boltzmann_reversible_streamed");
    validate_finite_nonnegative_distribution(
        f,
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL,
        "boltzmann_reversible_streamed input");
    const std::vector<double> eta = geometry.eta.center;
    // The shell plan bounds inactive collision activity and measures radial
    // cell width relative to the local RMS mean free path before any update.
    const BPSampledShellPlan shell_plan =
        bp_sampled_build_shell_plan(f, r, w, eta, geometry, p);
    info.minimum_mean_free_path = shell_plan.minimum_mean_free_path;
    info.maximum_radial_cell_width_over_mean_free_path =
        shell_plan.maximum_radial_cell_width_over_mean_free_path;
    info.nonlocal_collision_cells = shell_plan.nonlocal_collision_cells;
    if (p.bp_collision_fail_on_nonlocal_cells
        && shell_plan.nonlocal_collision_cells > 0) {
        const int i = shell_plan.maximum_locality_shell;
        std::ostringstream message;
        message
            << "boltzmann_reversible_streamed radial collision cell is "
            << "nonlocal: Delta r / lambda_rms="
            << shell_plan.maximum_radial_cell_width_over_mean_free_path
            << " exceeds bp_collision_locality_max_ratio="
            << p.bp_collision_locality_max_ratio;
        if (i >= 0) {
            message
                << " at shell " << i
                << ", r=" << r[static_cast<std::size_t>(i)]
                << ", Delta r="
                << geometry.radial.width[static_cast<std::size_t>(i)]
                << ", lambda_rms="
                << shell_plan.mean_free_path[static_cast<std::size_t>(i)];
        }
        message << ". Refine the radial grid before continuing.";
        throw std::runtime_error(message.str());
    }

    std::vector<BPStreamedShellResult> shell_result(
        static_cast<std::size_t>(p.Nr));
    const int available_threads = effective_threads(p, p.Nr);
    BPStreamedCollisionWorkspace local_workspace;
    BPStreamedCollisionWorkspace& workspace = persistent_workspace != nullptr
        ? *persistent_workspace : local_workspace;
    bp_streamed_prepare_collision_workspace(workspace, r, geometry, p);
    // Parallelize independent radial shells and use one worker inside each
    // shell to avoid nested thread teams and shared shell-cache mutation.
    Params shell_p = p;
    shell_p.threads = 1;
    parallel_for_dynamic_chunks(
        0,
        p.Nr,
        available_threads,
        1,
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                BPStreamedProjectionCache& cache =
                    workspace.shell_projection_cache[
                        static_cast<std::size_t>(i)];
                BPStreamedDenseGraphCache& dense_graph_cache =
                    workspace.shell_dense_graph_cache[
                        static_cast<std::size_t>(i)];
                BPStreamedOperatorGraphCache& operator_graph_cache =
                    workspace.shell_operator_graph_cache[
                        static_cast<std::size_t>(i)];
                std::vector<double> shell_f(
                    static_cast<std::size_t>(p.Nw * p.NL), 0.0);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        shell_f[static_cast<std::size_t>(
                            bp_reversible_velocity_index(j, k, p))] =
                            f[idx(i, j, k, p)];
                    }
                }
                BPStreamedShellResult& shell =
                    shell_result[static_cast<std::size_t>(i)];
                if (shell_plan.inactive[static_cast<std::size_t>(i)]) {
                    shell.step.f = std::move(shell_f);
                    shell.step.omitted_inactive_activity =
                        shell_plan.activity_upper_bound[
                            static_cast<std::size_t>(i)];
                    shell.step.omitted_fraction = 1.0;
                    continue;
                }
                try {
                    shell = bp_streamed_advance_shell(
                        shell_f,
                        r[static_cast<std::size_t>(i)],
                        w,
                        eta,
                        geometry,
                        dt,
                        shell_p,
                        static_cast<std::uint64_t>(i),
                        cache,
                        &dense_graph_cache,
                        &operator_graph_cache,
                        shell_plan.initial_samples[static_cast<std::size_t>(i)],
                        true);
                } catch (const std::exception& error) {
                    std::ostringstream message;
                    message
                        << "boltzmann_reversible_streamed failed at shell "
                        << i << ", r=" << r[static_cast<std::size_t>(i)]
                        << ": " << error.what();
                    throw std::runtime_error(message.str());
                }
            }
        });
    // Bound persistent projection storage after shell work.  Clearing this
    // geometry cache changes future work cost while retaining the distribution.
    std::size_t retained_projection_records = 0;
    for (BPStreamedProjectionCache& cache :
         workspace.shell_projection_cache) {
        if (retained_projection_records + cache.record.size()
            <= workspace.maximum_persistent_projection_records) {
            retained_projection_records += cache.record.size();
            continue;
        }
        std::unordered_map<
            BPStreamedProjectionKey,
            BPStreamedProjectionRecord,
            BPStreamedProjectionKeyHash>{}.swap(cache.record);
        cache.hits = 0;
        cache.misses = 0;
    }
    bp_reversible_validate_omission_budgets(
        p.Nr,
        geometry,
        p,
        [&](int i) -> const BPReversibleShellResult& {
            return shell_result[static_cast<std::size_t>(i)].step;
        });

    // Publish shell updates after omission checks and reduce physical
    // activity, sampling effort and error diagnostics in shell order.
    double minimum_dt = std::numeric_limits<double>::infinity();
    std::vector<double> shell_mass(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> shell_collision_timescale(
        static_cast<std::size_t>(p.Nr),
        std::numeric_limits<double>::infinity());
    info.sampled_min_samples_per_batch = std::numeric_limits<int>::max();
    for (int i = 0; i < p.Nr; ++i) {
        const BPStreamedShellResult& shell =
            shell_result[static_cast<std::size_t>(i)];
        long double number_density = 0.0L;
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t q = static_cast<std::size_t>(
                    bp_reversible_velocity_index(j, k, p));
                f[idx(i, j, k, p)] = shell.step.f[q];
                const double velocity_cell_volume = 2.0 * PI
                    * geometry.radial_velocity.width[
                        static_cast<std::size_t>(j)]
                    * geometry.eta.width[static_cast<std::size_t>(k)]
                    / (r[static_cast<std::size_t>(i)]
                       * r[static_cast<std::size_t>(i)]);
                number_density += shell.step.f[q] * velocity_cell_volume;
            }
        }
        const double physical_shell_volume = 4.0 * PI
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.width[static_cast<std::size_t>(i)];
        shell_mass[static_cast<std::size_t>(i)] = static_cast<double>(
            number_density * physical_shell_volume);
        const double collision_activity =
            shell.step.represented_activity
            + shell.step.omitted_domain_activity
            + shell.step.omitted_projection_activity
            + shell.step.omitted_inactive_activity;
        // Activity counts pair events; twice that activity gives the rate
        // of participating particles used in this shell collision time.
        if (number_density > 0.0L && collision_activity > 0.0) {
            shell_collision_timescale[static_cast<std::size_t>(i)] =
                static_cast<double>(number_density)
                / (2.0 * collision_activity);
        }
        info.substeps = std::max(info.substeps, shell.step.substeps);
        info.explicit_substeps =
            std::max(info.explicit_substeps, shell.step.substeps);
        info.reversible_reaction_updates += shell.step.reaction_updates;
        info.max_shell_remap_skipped_fraction = std::max(
            info.max_shell_remap_skipped_fraction,
            shell.step.omitted_fraction);
        const double shell_volume =
            4.0 * PI
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.width[static_cast<std::size_t>(i)];
        info.remap_attempted_event_density += shell_volume * (
            shell.step.represented_activity
            + shell.step.omitted_domain_activity
            + shell.step.omitted_projection_activity
            + shell.step.omitted_inactive_activity);
        info.remap_skipped_event_density += shell_volume * (
            shell.step.omitted_domain_activity
            + shell.step.omitted_projection_activity
            + shell.step.omitted_inactive_activity);
        info.remap_skipped_domain_event_density +=
            shell_volume * shell.step.omitted_domain_activity;
        info.remap_skipped_projection_event_density +=
            shell_volume * shell.step.omitted_projection_activity;
        info.remap_skipped_inactive_event_density +=
            shell_volume * shell.step.omitted_inactive_activity;
        info.maximum_projection_error = std::max(
            info.maximum_projection_error,
            shell.step.maximum_projection_error);
        info.sampled_dense_reactions += shell.dense_reactions;
        info.sampled_complement_reactions += shell.complement_reactions;
        info.sampled_complement_draws += shell.attempted_draws;
        info.sampled_retained_collision_draws += shell.retained_draws;
        info.sampled_projection_cache_hits += shell.cache_hits;
        info.sampled_projection_cache_misses += shell.cache_misses;
        info.sampled_reused_operator_shells +=
            shell.reused_operator_graph ? 1 : 0;
        info.sampled_max_operator_reference_l1 = std::max(
            info.sampled_max_operator_reference_l1,
            shell.operator_reference_l1);
        if (shell.samples_per_batch > 0) {
            info.sampled_min_samples_per_batch = std::min(
                info.sampled_min_samples_per_batch,
                shell.samples_per_batch);
        }
        info.sampled_max_samples_per_batch = std::max(
            info.sampled_max_samples_per_batch,
            shell.samples_per_batch);
        info.sampled_max_adaptive_rounds = std::max(
            info.sampled_max_adaptive_rounds,
            shell.adaptive_rounds);
        info.sampled_unconverged_shells += shell.converged ? 0 : 1;
        info.sampled_inactive_shells +=
            shell_plan.inactive[static_cast<std::size_t>(i)] ? 1 : 0;
        info.sampled_max_rhs_relative_error = std::max(
            info.sampled_max_rhs_relative_error,
            shell.rhs_relative_error);
        info.sampled_max_rhs_gross_error = std::max(
            info.sampled_max_rhs_gross_error,
            shell.rhs_gross_error);
        info.sampled_max_heat_relative_error = std::max(
            info.sampled_max_heat_relative_error,
            shell.heat_relative_error);
        info.sampled_max_heat_gross_error = std::max(
            info.sampled_max_heat_gross_error,
            shell.heat_gross_error);
        info.sampled_max_energy_invariant_error = std::max(
            info.sampled_max_energy_invariant_error,
            shell.energy_invariant_error);
        info.sampled_context_cpu_seconds += shell.context_seconds;
        info.sampled_estimate_cpu_seconds += shell.estimate_seconds;
        info.sampled_materialize_cpu_seconds += shell.materialize_seconds;
        info.sampled_advance_cpu_seconds += shell.advance_seconds;
        info.sampled_max_shell_seconds = std::max(
            info.sampled_max_shell_seconds,
            shell.context_seconds + shell.estimate_seconds
                + shell.materialize_seconds + shell.advance_seconds);
        minimum_dt = std::min(minimum_dt, shell.step.minimum_dt_limit);
    }
    if (info.sampled_min_samples_per_batch
        == std::numeric_limits<int>::max()) {
        info.sampled_min_samples_per_batch = 0;
    }
    info.collision_dt = dt;
    info.explicit_dt_limit = minimum_dt;
    info.collision_time_estimate =
        bp_sampled_mass_weighted_collision_timescale(
            shell_mass,
            shell_collision_timescale,
            p.bp_reversible_active_shell_floor_fraction);
    return info;
}

}  // namespace ev
