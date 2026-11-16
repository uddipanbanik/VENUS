#pragma once

// Construct a reversible spherical Boltzmann reaction graph from a dense
// velocity-cell block and importance samples of the remaining pair space.
// Inputs are nonnegative shell f(w,eta), cell volumes, cross sections, angular
// rules and sampling tolerances.  Outputs are updated f and measured sampling,
// omitted-activity, locality, positivity and entropy diagnostics.
//
// The dense block sums its discrete angular channels.  The complement uses
// shifted low-discrepancy replicates with explicit proposal probabilities;
// inverse probabilities recover each sampled channel's physical coefficient.
// Every represented channel uses the joint conservative pair projection and
// geometric reverse activity of the reversible operator.  Sampling changes the
// channel weights while preserving each retained reaction's invariant vector.
// Replicate disagreement controls the sampled RHS and radial heat moment.

#include "ev/reversible_finite_angle_boltzmann.hpp"

#include <cstring>
#include <tuple>

namespace ev {

// A stratum groups cells by energy rank, radial direction and eta index range.
// Its CDF draws a cell conditional on choosing that group.
struct BPSampledStratum {
    std::vector<int> cells;
    std::vector<double> cdf;
};

// Store each mixture component separately so the draw and the inverse-weight
// calculation use the same complete joint probability, including overlap.
struct BPSampledProposal {
    int velocity_cells = 0;
    int pair_phi_points = 0;
    int relative_speed_bins = 0;
    double bulk_fraction = 0.0;
    double tail_fraction = 0.0;
    double stratified_fraction = 0.0;
    double equilibrium_fraction = 0.0;
    double phase_density_fraction = 0.0;
    double uniform_fraction = 0.0;
    std::vector<double> bulk_probability;
    std::vector<double> bulk_cdf;
    std::vector<double> tail_probability;
    std::vector<double> tail_cdf;
    std::vector<double> equilibrium_probability;
    std::vector<double> equilibrium_cdf;
    std::vector<double> phase_density_probability;
    std::vector<double> phase_density_cdf;
    std::vector<int> time_reverse_cell;
    std::vector<int> cell_stratum;
    std::vector<double> within_stratum_probability;
    std::vector<BPSampledStratum> strata;
    std::vector<int> active_strata;
    std::vector<int> catalog_relative_bin;
    std::vector<double> catalog_probability;
    std::vector<std::vector<int>> relative_bin_catalog;
    std::vector<std::vector<double>> relative_bin_catalog_cdf;
};

// Split a cell distribution into dense and outside support, retaining the
// dense probability needed to condition a pair outside the dense-by-dense block.
struct BPSampledMaskedDistribution {
    double dense_probability = 0.0;
    std::vector<int> all_cells;
    std::vector<double> all_cdf;
    std::vector<int> dense_cells;
    std::vector<double> dense_cdf;
    std::vector<int> outside_cells;
    std::vector<double> outside_cdf;
};

struct BPSampledComplementProposal {
    double probability = 0.0;
    std::vector<int> component_kind;
    std::vector<double> component_cdf;
    BPSampledMaskedDistribution bulk;
    BPSampledMaskedDistribution tail;
    BPSampledMaskedDistribution equilibrium;
    BPSampledMaskedDistribution phase_density;
    BPSampledMaskedDistribution uniform;
    std::vector<BPSampledMaskedDistribution> strata;
    std::vector<int> stratified_catalog;
    std::vector<double> stratified_catalog_cdf;
};

struct BPSampledBatch {
    BPReversibleGraph graph;
    std::uint64_t accepted_samples = 0;
    std::uint64_t attempted_draws = 0;
    std::uint64_t retained_collision_draws = 0;
};

struct BPSampledShellGraph {
    BPReversibleGraph graph;
    int dense_velocity_cells = 0;
    std::uint64_t dense_reactions = 0;
    std::uint64_t complement_reactions = 0;
    std::uint64_t complement_draws = 0;
    std::uint64_t retained_collision_draws = 0;
    int samples_per_batch = 0;
    int adaptive_rounds = 0;
    bool converged = true;
    double rhs_relative_error = 0.0;
    double rhs_gross_error = 0.0;
    double heat_relative_error = 0.0;
    double heat_gross_error = 0.0;
    double energy_invariant_error = 0.0;
    bool inactive = false;
    bool ntc_fallback = false;
};

struct BPSampledShellResult {
    BPReversibleShellResult step;
    BPSampledShellGraph sampling;
};

// Shell work follows measured activity.  Explicit upper/lower activity budgets
// bound inactive shells, while RMS mean free paths quantify spatial locality.
struct BPSampledShellPlan {
    std::vector<int> initial_samples;
    std::vector<double> activity_upper_bound;
    std::vector<double> mean_free_path;
    std::vector<double> radial_cell_width_over_mean_free_path;
    std::vector<unsigned char> inactive;
    double weighted_inactive_upper_bound = 0.0;
    double weighted_activity_lower_bound = 0.0;
    double minimum_mean_free_path = std::numeric_limits<double>::infinity();
    double maximum_radial_cell_width_over_mean_free_path = 0.0;
    int maximum_locality_shell = -1;
    int nonlocal_collision_cells = 0;
};

inline double bp_sampled_u_sigma_upper_bound(
    double maximum_relative_speed,
    const Params& p);

// Thin small sigma*g contributions with a capped acceptance probability and
// compensate retained draws by its reciprocal.  A pilot threshold controls
// work distribution; values above that threshold are retained with weight one.
inline double bp_sampled_ntc_inverse_acceptance(
    double pair_activity,
    double majorant,
    double unit_draw) {
    if (!(pair_activity > 0.0)) {
        return 0.0;
    }
    if (!(majorant > 0.0)) {
        return 1.0;
    }
    const double acceptance =
        std::min(1.0, pair_activity / majorant);
    if (unit_draw >= acceptance) {
        return 0.0;
    }
    return 1.0 / acceptance;
}

// Mix persistent seed/counter bits reproducibly for independent shifted draws.
inline std::uint64_t bp_sampled_splitmix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

// Convert the high 53 bits to a binary64 unit variate on [0,1).
inline double bp_sampled_unit_from_u64(std::uint64_t value) {
    return static_cast<double>(value >> 11)
        * (1.0 / static_cast<double>(UINT64_C(1) << 53));
}

// Reverse base digits to construct one low-discrepancy coordinate.
inline double bp_sampled_radical_inverse(std::uint64_t index, int base) {
    double value = 0.0;
    double factor = 1.0 / static_cast<double>(base);
    while (index > 0) {
        value += factor * static_cast<double>(index % static_cast<std::uint64_t>(base));
        index /= static_cast<std::uint64_t>(base);
        factor /= static_cast<double>(base);
    }
    return value;
}

// A seed-dependent modulo-one shift gives each replicate its own sequence
// while preserving the draw index used by adaptive extensions and workers.
inline double bp_sampled_shifted_halton(
    std::uint64_t index,
    int dimension,
    std::uint64_t seed) {
    static constexpr int primes[16] = {
        2, 3, 5, 7, 11, 13, 17, 19,
        23, 29, 31, 37, 41, 43, 47, 53,
    };
    const double shift = bp_sampled_unit_from_u64(
        bp_sampled_splitmix64(seed + static_cast<std::uint64_t>(dimension)));
    const double raw = bp_sampled_radical_inverse(
        index + 1, primes[dimension % 16]) + shift;
    return raw - std::floor(raw);
}

// Invert the continuous polar law: isotropic mu is uniform on [-1,1], and
// the Born kernel uses its analytic angular CDF.
inline double bp_sampled_scatter_mu_from_unit(
    double unit,
    double relative_speed,
    const Params& p,
    const std::string& kernel) {
    const double clipped_unit =
        std::max(0.0, std::min(1.0, unit));
    if (kernel != "yukawa_born") {
        return std::fma(2.0, clipped_unit, -1.0);
    }
    // This API uses the increasing CDF; the shared inverse uses its upper tail.
    return bp_fa_yukawa_born_mu_from_unit(
        relative_speed, p, 1.0 - clipped_unit);
}

// Select the normalized discrete polar quadrature and uniform azimuth table.
// This samples the angular discretization used by the dense reaction graph.
inline void bp_sampled_discrete_scatter_direction(
    double polar_unit,
    double azimuth_unit,
    double relative_speed,
    const Params& p,
    const std::string& kernel,
    const std::vector<double>& mu,
    const std::vector<double>& mu_weight,
    const std::vector<double>& psi_cos,
    const std::vector<double>& psi_sin,
    double& scatter_mu,
    double& scatter_psi_cos,
    double& scatter_psi_sin) {
    if (mu.empty()
        || mu.size() != mu_weight.size()
        || psi_cos.empty()
        || psi_cos.size() != psi_sin.size()) {
        throw std::runtime_error(
            "sampled Boltzmann discrete angular table is invalid.");
    }
    double total_weight = 0.0;
    for (std::size_t q = 0; q < mu.size(); ++q) {
        total_weight +=
            mu_weight[q]
            * bp_fa_scatter_weight_raw(
                mu[q], relative_speed, p, kernel);
    }
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw std::runtime_error(
            "sampled Boltzmann discrete polar quadrature has zero weight.");
    }
    const double target =
        std::max(0.0, std::min(
            std::nextafter(1.0, 0.0), polar_unit))
        * total_weight;
    double cumulative = 0.0;
    std::size_t polar_index = mu.size() - 1;
    for (std::size_t q = 0; q < mu.size(); ++q) {
        cumulative +=
            mu_weight[q]
            * bp_fa_scatter_weight_raw(
                mu[q], relative_speed, p, kernel);
        if (target < cumulative) {
            polar_index = q;
            break;
        }
    }
    const double clipped_azimuth =
        std::max(0.0, std::min(
            std::nextafter(1.0, 0.0), azimuth_unit));
    const std::size_t azimuth_index = std::min(
        psi_cos.size() - 1,
        static_cast<std::size_t>(
            clipped_azimuth * static_cast<double>(psi_cos.size())));
    scatter_mu = mu[polar_index];
    scatter_psi_cos = psi_cos[azimuth_index];
    scatter_psi_sin = psi_sin[azimuth_index];
}

// Tile the unit polar/azimuth square into equal-area strata, then apply shared
// pair-dependent shifts so each outgoing sample covers one tile.
inline std::array<double, 2> bp_sampled_stratified_outgoing_units(
    std::uint64_t pair_draw,
    int angle,
    int angle_samples,
    int polar_strata_hint,
    std::uint64_t seed) {
    if (angle_samples < 1 || angle < 0 || angle >= angle_samples) {
        throw std::runtime_error(
            "sampled Boltzmann outgoing-angle index is invalid.");
    }
    int polar_strata = std::min(
        angle_samples, std::max(1, polar_strata_hint));
    while (polar_strata > 1
           && angle_samples % polar_strata != 0) {
        --polar_strata;
    }
    const int azimuth_strata = angle_samples / polar_strata;
    const int polar_index = angle % polar_strata;
    const int azimuth_index = angle / polar_strata;
    const double polar_shift =
        bp_sampled_shifted_halton(
            pair_draw,
            8,
            seed ^ UINT64_C(0xa24baed4963ee407));
    const double azimuth_shift =
        bp_sampled_shifted_halton(
            pair_draw,
            9,
            seed ^ UINT64_C(0x9fb21c651e98df25));
    return {
        (static_cast<double>(polar_index) + polar_shift)
            / static_cast<double>(polar_strata),
        (static_cast<double>(azimuth_index) + azimuth_shift)
            / static_cast<double>(azimuth_strata),
    };
}

// Bind the shifted sequence to the exact shell radius, distribution bits and
// input seed, making repeated evaluation of the same shell reproducible.
inline std::uint64_t bp_sampled_shell_hash(
    const std::vector<double>& shell_f,
    double radius,
    std::uint64_t seed) {
    std::uint64_t hash = bp_sampled_splitmix64(seed);
    auto mix_double = [&](double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
        std::memcpy(&bits, &value, sizeof(bits));
        hash = bp_sampled_splitmix64(hash ^ bits);
    };
    mix_double(radius);
    for (double value : shell_f) {
        mix_double(value);
    }
    return hash;
}

// Normalize finite positive total weight; the empty/zero-weight distribution
// uses its specified uniform fallback over the supplied cells.
inline std::vector<double> bp_sampled_normalize_probability(
    const std::vector<double>& weights) {
    std::vector<double> probability(weights.size(), 0.0);
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total > 0.0 && std::isfinite(total)) {
        for (std::size_t q = 0; q < weights.size(); ++q) {
            probability[q] = weights[q] / total;
        }
    } else if (!weights.empty()) {
        const double uniform = 1.0 / static_cast<double>(weights.size());
        std::fill(probability.begin(), probability.end(), uniform);
    }
    return probability;
}

// Form a cumulative distribution and close its last endpoint at unity.
inline std::vector<double> bp_sampled_probability_cdf(
    const std::vector<double>& probability) {
    std::vector<double> cdf(probability.size(), 0.0);
    double sum = 0.0;
    for (std::size_t q = 0; q < probability.size(); ++q) {
        sum += probability[q];
        cdf[q] = sum;
    }
    if (!cdf.empty()) {
        cdf.back() = 1.0;
    }
    return cdf;
}

// Locate the selected discrete cell with the same cumulative endpoint rule
// for every mixture component.
inline int bp_sampled_cdf_index(
    const std::vector<double>& cdf,
    double unit_value) {
    if (cdf.empty()) {
        throw std::runtime_error("sampled Boltzmann proposal has an empty CDF.");
    }
    const auto it = std::lower_bound(cdf.begin(), cdf.end(), unit_value);
    return (it == cdf.end())
        ? static_cast<int>(cdf.size()) - 1
        : static_cast<int>(std::distance(cdf.begin(), it));
}

// Encode/decode a stratum pair and its incoming tangential azimuth.
inline int bp_sampled_catalog_index(
    int first_stratum,
    int second_stratum,
    int pair_phi,
    int stratum_count,
    int pair_phi_points) {
    return ((first_stratum * stratum_count) + second_stratum)
        * pair_phi_points + pair_phi;
}

inline std::tuple<int, int, int> bp_sampled_decode_catalog_index(
    int encoded,
    int stratum_count,
    int pair_phi_points) {
    const int pair_phi = encoded % pair_phi_points;
    const int pair = encoded / pair_phi_points;
    return std::make_tuple(
        pair / stratum_count,
        pair % stratum_count,
        pair_phi);
}

// Build proposal components from population f*d^3v, phase density f, a local
// Maxwellian estimate, energy-enhanced tails and uniform support.  These are
// sampling distributions; the actual collision law enters importance weights.
inline BPSampledProposal bp_sampled_build_proposal(
    const std::vector<double>& shell_f,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    BPSampledProposal proposal;
    proposal.velocity_cells = p.Nw * p.NL;
    proposal.pair_phi_points = std::max(1, p.bp_boltzmann_pair_phi_points);
    proposal.tail_fraction = p.bp_sampled_tail_fraction;
    proposal.stratified_fraction = p.bp_sampled_stratified_fraction;
    proposal.equilibrium_fraction = p.bp_sampled_equilibrium_fraction;
    proposal.phase_density_fraction = p.bp_sampled_phase_density_fraction;
    proposal.uniform_fraction = p.bp_sampled_uniform_fraction;
    proposal.bulk_fraction =
        1.0 - proposal.tail_fraction
        - proposal.stratified_fraction
        - proposal.equilibrium_fraction
        - proposal.phase_density_fraction
        - proposal.uniform_fraction;
    const int cells = proposal.velocity_cells;
    std::vector<double> volume(static_cast<std::size_t>(cells), 0.0);
    std::vector<double> population(static_cast<std::size_t>(cells), 0.0);
    std::vector<double> energy(static_cast<std::size_t>(cells), 0.0);
    double total_population = 0.0;
    double total_radial_momentum = 0.0;
    double total_energy = 0.0;
    for (int q = 0; q < cells; ++q) {
        const std::array<int, 2> indices = bp_reversible_velocity_indices(q, p);
        volume[static_cast<std::size_t>(q)] =
            2.0 * PI
            * geometry.radial_velocity.width[static_cast<std::size_t>(indices[0])]
            * geometry.eta.width[static_cast<std::size_t>(indices[1])]
            / (radius * radius);
        population[static_cast<std::size_t>(q)] =
            volume[static_cast<std::size_t>(q)]
            * shell_f[static_cast<std::size_t>(q)];
        energy[static_cast<std::size_t>(q)] =
            bp_reversible_energy(q, radius, w, eta, p);
        total_population += population[static_cast<std::size_t>(q)];
        total_radial_momentum += population[static_cast<std::size_t>(q)]
            * w[static_cast<std::size_t>(indices[0])];
        total_energy += population[static_cast<std::size_t>(q)]
            * energy[static_cast<std::size_t>(q)];
    }
    proposal.bulk_probability =
        bp_sampled_normalize_probability(population);
    proposal.bulk_cdf =
        bp_sampled_probability_cdf(proposal.bulk_probability);
    proposal.phase_density_probability =
        bp_sampled_normalize_probability(shell_f);
    proposal.phase_density_cdf =
        bp_sampled_probability_cdf(
            proposal.phase_density_probability);
    const double mean_energy =
        (total_population > 0.0)
        ? total_energy / total_population
        : 1.0;
    const double mean_radial_velocity = total_population > 0.0
        ? total_radial_momentum / total_population
        : 0.0;
    // This moment-based Maxwellian is a proposal, with its exponential
    // shifted by the largest exponent to avoid overflow in normalization.
    const double temperature = std::max(
        1.0e-12,
        (2.0 / 3.0) * std::max(
            0.0,
            mean_energy
                - 0.5 * mean_radial_velocity * mean_radial_velocity));
    const double beta = mean_radial_velocity / temperature;
    const double gamma = 1.0 / temperature;
    std::vector<double> equilibrium_weight(
        static_cast<std::size_t>(cells), 0.0);
    double maximum_log_equilibrium =
        -std::numeric_limits<double>::infinity();
    for (int q = 0; q < cells; ++q) {
        const double radial_velocity = bp_reversible_radial_velocity(q, w, p);
        maximum_log_equilibrium = std::max(
            maximum_log_equilibrium,
            beta * radial_velocity
                - gamma * energy[static_cast<std::size_t>(q)]);
    }
    for (int q = 0; q < cells; ++q) {
        const double radial_velocity = bp_reversible_radial_velocity(q, w, p);
        equilibrium_weight[static_cast<std::size_t>(q)] =
            volume[static_cast<std::size_t>(q)]
            * std::exp(
                beta * radial_velocity
                - gamma * energy[static_cast<std::size_t>(q)]
                - maximum_log_equilibrium);
    }
    proposal.equilibrium_probability =
        bp_sampled_normalize_probability(equilibrium_weight);
    proposal.equilibrium_cdf =
        bp_sampled_probability_cdf(proposal.equilibrium_probability);
    std::vector<double> tail_weight(static_cast<std::size_t>(cells), 0.0);
    for (int q = 0; q < cells; ++q) {
        const double scaled_energy =
            energy[static_cast<std::size_t>(q)]
            / std::max(mean_energy, 1.0e-300);
        tail_weight[static_cast<std::size_t>(q)] =
            population[static_cast<std::size_t>(q)]
            * std::pow(1.0 + scaled_energy, p.bp_sampled_tail_power);
    }
    proposal.tail_probability =
        bp_sampled_normalize_probability(tail_weight);
    proposal.tail_cdf =
        bp_sampled_probability_cdf(proposal.tail_probability);
    proposal.time_reverse_cell.assign(static_cast<std::size_t>(cells), -1);
    for (int q = 0; q < cells; ++q) {
        const std::array<int, 2> indices =
            bp_reversible_velocity_indices(q, p);
        const int reverse_j = p.Nw - 1 - indices[0];
        const double velocity_scale = std::max({
            1.0,
            std::abs(w[static_cast<std::size_t>(indices[0])]),
            std::abs(w[static_cast<std::size_t>(reverse_j)]),
        });
        if (std::abs(
                w[static_cast<std::size_t>(indices[0])]
                + w[static_cast<std::size_t>(reverse_j)])
            <= 64.0 * std::numeric_limits<double>::epsilon()
                * velocity_scale) {
            proposal.time_reverse_cell[static_cast<std::size_t>(q)] =
                bp_reversible_velocity_index(reverse_j, indices[1], p);
        }
    }

    // Equal-count energy ranks, radial direction and eta-index bins define
    // strata.  Within each stratum, mix population with uniform cell support.
    std::vector<int> energy_order(static_cast<std::size_t>(cells), 0);
    std::iota(energy_order.begin(), energy_order.end(), 0);
    std::stable_sort(
        energy_order.begin(),
        energy_order.end(),
        [&](int first, int second) {
            return energy[static_cast<std::size_t>(first)]
                < energy[static_cast<std::size_t>(second)];
        });
    std::vector<int> energy_bin(static_cast<std::size_t>(cells), 0);
    for (int rank = 0; rank < cells; ++rank) {
        energy_bin[static_cast<std::size_t>(
            energy_order[static_cast<std::size_t>(rank)])] =
            std::min(
                p.bp_sampled_energy_strata - 1,
                rank * p.bp_sampled_energy_strata / std::max(1, cells));
    }
    const int stratum_count =
        p.bp_sampled_energy_strata * 2 * p.bp_sampled_L_strata;
    proposal.strata.resize(static_cast<std::size_t>(stratum_count));
    proposal.cell_stratum.assign(static_cast<std::size_t>(cells), 0);
    proposal.within_stratum_probability.assign(
        static_cast<std::size_t>(cells), 0.0);
    for (int q = 0; q < cells; ++q) {
        const std::array<int, 2> indices = bp_reversible_velocity_indices(q, p);
        const int sign = (w[static_cast<std::size_t>(indices[0])] < 0.0) ? 0 : 1;
        const int L_bin = std::min(
            p.bp_sampled_L_strata - 1,
            indices[1] * p.bp_sampled_L_strata / std::max(1, p.NL));
        const int stratum =
            (energy_bin[static_cast<std::size_t>(q)] * 2 + sign)
            * p.bp_sampled_L_strata + L_bin;
        proposal.cell_stratum[static_cast<std::size_t>(q)] = stratum;
        proposal.strata[static_cast<std::size_t>(stratum)].cells.push_back(q);
    }
    std::vector<double> representative_w(
        static_cast<std::size_t>(stratum_count), 0.0);
    std::vector<double> representative_vt(
        static_cast<std::size_t>(stratum_count), 0.0);
    std::vector<double> stratum_population(
        static_cast<std::size_t>(stratum_count), 0.0);
    for (int stratum = 0; stratum < stratum_count; ++stratum) {
        BPSampledStratum& entry =
            proposal.strata[static_cast<std::size_t>(stratum)];
        if (entry.cells.empty()) {
            continue;
        }
        std::vector<double> weights(entry.cells.size(), 0.0);
        double weight_sum = 0.0;
        for (std::size_t local = 0; local < entry.cells.size(); ++local) {
            const int q = entry.cells[local];
            weights[local] = population[static_cast<std::size_t>(q)];
            weight_sum += weights[local];
        }
        stratum_population[static_cast<std::size_t>(stratum)] =
            weight_sum;
        if (!(weight_sum > 0.0)) {
            std::fill(weights.begin(), weights.end(), 1.0);
            weight_sum = static_cast<double>(weights.size());
        } else {
            proposal.active_strata.push_back(stratum);
        }
        std::vector<double> conditional(weights.size(), 0.0);
        for (std::size_t local = 0; local < entry.cells.size(); ++local) {
            const double mass_probability = weights[local] / weight_sum;
            const double uniform_probability =
                1.0 / static_cast<double>(entry.cells.size());
            conditional[local] =
                (1.0 - p.bp_sampled_stratum_uniform_fraction)
                    * mass_probability
                + p.bp_sampled_stratum_uniform_fraction
                    * uniform_probability;
            const int q = entry.cells[local];
            proposal.within_stratum_probability[
                static_cast<std::size_t>(q)] = conditional[local];
            const std::array<int, 2> indices =
                bp_reversible_velocity_indices(q, p);
            representative_w[static_cast<std::size_t>(stratum)] +=
                conditional[local] * w[static_cast<std::size_t>(indices[0])];
            representative_vt[static_cast<std::size_t>(stratum)] +=
                conditional[local]
                * std::sqrt(2.0 * eta[static_cast<std::size_t>(indices[1])])
                / radius;
        }
        entry.cdf = bp_sampled_probability_cdf(conditional);
    }

    // Representative stratum velocities rank catalog entries by relative
    // speed; exact cell kinematics are evaluated after drawing an entry.
    std::vector<double> pair_phi_cos;
    std::vector<double> pair_phi_sin;
    bp_fa_midpoint_angle_table(
        proposal.pair_phi_points, pair_phi_cos, pair_phi_sin);
    struct CatalogSpeed {
        int encoded = 0;
        double speed = 0.0;
        double activity_weight = 0.0;
    };
    std::vector<CatalogSpeed> catalog;
    for (int first_stratum : proposal.active_strata) {
        const BPFAVec3 first{
            representative_w[static_cast<std::size_t>(first_stratum)],
            representative_vt[static_cast<std::size_t>(first_stratum)],
            0.0,
        };
        for (int second_stratum : proposal.active_strata) {
            for (int iphi = 0; iphi < proposal.pair_phi_points; ++iphi) {
                const BPFAVec3 second{
                    representative_w[static_cast<std::size_t>(second_stratum)],
                    representative_vt[static_cast<std::size_t>(second_stratum)]
                        * pair_phi_cos[static_cast<std::size_t>(iphi)],
                    representative_vt[static_cast<std::size_t>(second_stratum)]
                        * pair_phi_sin[static_cast<std::size_t>(iphi)],
                };
                const double speed = bp_fa_norm(bp_fa_sub(first, second));
                catalog.push_back(CatalogSpeed{
                    bp_sampled_catalog_index(
                        first_stratum,
                        second_stratum,
                        iphi,
                        stratum_count,
                        proposal.pair_phi_points),
                    speed,
                    stratum_population[
                        static_cast<std::size_t>(first_stratum)]
                        * stratum_population[
                            static_cast<std::size_t>(second_stratum)]
                        * speed * bp_fa_total_cross_section(speed, p),
                });
            }
        }
    }
    std::stable_sort(
        catalog.begin(),
        catalog.end(),
        [](const CatalogSpeed& first, const CatalogSpeed& second) {
            return first.speed < second.speed;
        });
    proposal.relative_speed_bins = std::min(
        std::max(2, p.bp_sampled_energy_strata),
        std::max(1, static_cast<int>(catalog.size())));
    proposal.relative_bin_catalog.resize(
        static_cast<std::size_t>(proposal.relative_speed_bins));
    proposal.relative_bin_catalog_cdf.resize(
        static_cast<std::size_t>(proposal.relative_speed_bins));
    proposal.catalog_relative_bin.assign(
        static_cast<std::size_t>(
            stratum_count * stratum_count * proposal.pair_phi_points),
        -1);
    proposal.catalog_probability.assign(
        proposal.catalog_relative_bin.size(), 0.0);
    std::vector<std::vector<double>> relative_bin_weight(
        static_cast<std::size_t>(proposal.relative_speed_bins));
    for (std::size_t rank = 0; rank < catalog.size(); ++rank) {
        const int bin = std::min(
            proposal.relative_speed_bins - 1,
            static_cast<int>(
                rank * static_cast<std::size_t>(proposal.relative_speed_bins)
                / std::max<std::size_t>(1, catalog.size())));
        proposal.relative_bin_catalog[static_cast<std::size_t>(bin)].push_back(
            catalog[rank].encoded);
        relative_bin_weight[static_cast<std::size_t>(bin)].push_back(
            catalog[rank].activity_weight);
        proposal.catalog_relative_bin[
            static_cast<std::size_t>(catalog[rank].encoded)] = bin;
    }
    for (int bin = 0; bin < proposal.relative_speed_bins; ++bin) {
        const std::vector<double> conditional =
            bp_sampled_normalize_probability(
                relative_bin_weight[static_cast<std::size_t>(bin)]);
        proposal.relative_bin_catalog_cdf[static_cast<std::size_t>(bin)] =
            bp_sampled_probability_cdf(conditional);
        const std::vector<int>& entries =
            proposal.relative_bin_catalog[static_cast<std::size_t>(bin)];
        for (std::size_t local = 0; local < entries.size(); ++local) {
            proposal.catalog_probability[
                static_cast<std::size_t>(entries[local])] =
                conditional[local]
                / static_cast<double>(proposal.relative_speed_bins);
        }
    }
    return proposal;
}

// Sum every mixture contribution to this exact incoming cell pair and angle.
// Using the total probability accounts for overlap between proposal components.
inline double bp_sampled_joint_proposal_probability(
    const BPSampledProposal& proposal,
    int first,
    int second,
    int pair_phi) {
    const double inv_phi =
        1.0 / static_cast<double>(proposal.pair_phi_points);
    const double inverse_cells =
        1.0 / static_cast<double>(proposal.velocity_cells);
    double probability =
        proposal.bulk_fraction
            * proposal.bulk_probability[static_cast<std::size_t>(first)]
            * proposal.bulk_probability[static_cast<std::size_t>(second)]
            * inv_phi
        + proposal.tail_fraction
            * proposal.tail_probability[static_cast<std::size_t>(first)]
            * proposal.tail_probability[static_cast<std::size_t>(second)]
            * inv_phi
        + proposal.equilibrium_fraction
            * proposal.equilibrium_probability[
                static_cast<std::size_t>(first)]
            * proposal.equilibrium_probability[
                static_cast<std::size_t>(second)]
            * inv_phi
        + proposal.phase_density_fraction
            * proposal.phase_density_probability[
                static_cast<std::size_t>(first)]
            * proposal.phase_density_probability[
                static_cast<std::size_t>(second)]
            * inv_phi
        + proposal.uniform_fraction
            * inverse_cells * inverse_cells * inv_phi;
    if (proposal.stratified_fraction > 0.0) {
        const int first_stratum =
            proposal.cell_stratum[static_cast<std::size_t>(first)];
        const int second_stratum =
            proposal.cell_stratum[static_cast<std::size_t>(second)];
        const int encoded = bp_sampled_catalog_index(
            first_stratum,
            second_stratum,
            pair_phi,
            static_cast<int>(proposal.strata.size()),
            proposal.pair_phi_points);
        const int bin = proposal.catalog_relative_bin[
            static_cast<std::size_t>(encoded)];
        if (bin >= 0) {
            probability +=
                proposal.stratified_fraction
                * proposal.catalog_probability[
                    static_cast<std::size_t>(encoded)]
                * proposal.within_stratum_probability[
                    static_cast<std::size_t>(first)]
                * proposal.within_stratum_probability[
                    static_cast<std::size_t>(second)];
        }
    }
    return probability;
}

// Select one component, then its conditional cells and incoming azimuth.
// The stratified component draws a speed-bin catalog before its two cells.
inline std::tuple<int, int, int> bp_sampled_draw_joint_state(
    const BPSampledProposal& proposal,
    std::uint64_t draw,
    std::uint64_t seed) {
    const double component = bp_sampled_shifted_halton(draw, 0, seed);
    const double first_unit = bp_sampled_shifted_halton(draw, 1, seed);
    const double second_unit = bp_sampled_shifted_halton(draw, 2, seed);
    const double phi_unit = bp_sampled_shifted_halton(draw, 3, seed);
    if (component < proposal.bulk_fraction) {
        return std::make_tuple(
            bp_sampled_cdf_index(proposal.bulk_cdf, first_unit),
            bp_sampled_cdf_index(proposal.bulk_cdf, second_unit),
            std::min(
                proposal.pair_phi_points - 1,
                static_cast<int>(phi_unit * proposal.pair_phi_points)));
    }
    if (component < proposal.bulk_fraction + proposal.tail_fraction) {
        return std::make_tuple(
            bp_sampled_cdf_index(proposal.tail_cdf, first_unit),
            bp_sampled_cdf_index(proposal.tail_cdf, second_unit),
            std::min(
                proposal.pair_phi_points - 1,
                static_cast<int>(phi_unit * proposal.pair_phi_points)));
    }
    if (component
        < proposal.bulk_fraction
            + proposal.tail_fraction
            + proposal.equilibrium_fraction) {
        return std::make_tuple(
            bp_sampled_cdf_index(
                proposal.equilibrium_cdf, first_unit),
            bp_sampled_cdf_index(
                proposal.equilibrium_cdf, second_unit),
            std::min(
                proposal.pair_phi_points - 1,
                static_cast<int>(phi_unit * proposal.pair_phi_points)));
    }
    if (component
        < proposal.bulk_fraction
            + proposal.tail_fraction
            + proposal.equilibrium_fraction
            + proposal.phase_density_fraction) {
        return std::make_tuple(
            bp_sampled_cdf_index(
                proposal.phase_density_cdf, first_unit),
            bp_sampled_cdf_index(
                proposal.phase_density_cdf, second_unit),
            std::min(
                proposal.pair_phi_points - 1,
                static_cast<int>(phi_unit * proposal.pair_phi_points)));
    }
    if (component
        < proposal.bulk_fraction
            + proposal.tail_fraction
            + proposal.equilibrium_fraction
            + proposal.phase_density_fraction
            + proposal.stratified_fraction) {
        const double bin_unit = bp_sampled_shifted_halton(draw, 4, seed);
        const double entry_unit = bp_sampled_shifted_halton(draw, 5, seed);
        const int bin = std::min(
            proposal.relative_speed_bins - 1,
            static_cast<int>(bin_unit * proposal.relative_speed_bins));
        const std::vector<int>& entries =
            proposal.relative_bin_catalog[static_cast<std::size_t>(bin)];
        const int encoded = entries[static_cast<std::size_t>(
            bp_sampled_cdf_index(
                proposal.relative_bin_catalog_cdf[
                    static_cast<std::size_t>(bin)],
                entry_unit))];
        int first_stratum = 0;
        int second_stratum = 0;
        int pair_phi = 0;
        std::tie(first_stratum, second_stratum, pair_phi) =
            bp_sampled_decode_catalog_index(
                encoded,
                static_cast<int>(proposal.strata.size()),
                proposal.pair_phi_points);
        const BPSampledStratum& first_entry =
            proposal.strata[static_cast<std::size_t>(first_stratum)];
        const BPSampledStratum& second_entry =
            proposal.strata[static_cast<std::size_t>(second_stratum)];
        return std::make_tuple(
            first_entry.cells[static_cast<std::size_t>(
                bp_sampled_cdf_index(first_entry.cdf, first_unit))],
            second_entry.cells[static_cast<std::size_t>(
                bp_sampled_cdf_index(second_entry.cdf, second_unit))],
            pair_phi);
    }
    const int uniform_first = std::min(
        proposal.velocity_cells - 1,
        static_cast<int>(first_unit * proposal.velocity_cells));
    const int uniform_second = std::min(
        proposal.velocity_cells - 1,
        static_cast<int>(second_unit * proposal.velocity_cells));
    const int pair_phi = std::min(
        proposal.pair_phi_points - 1,
        static_cast<int>(phi_unit * proposal.pair_phi_points));
    return std::make_tuple(
        uniform_first, uniform_second, pair_phi);
}

// Preserve the original component probabilities while building conditional
// all/dense/outside CDFs; reject invalid weights before normalizing them.
inline BPSampledMaskedDistribution bp_sampled_mask_distribution(
    const std::vector<int>& cells,
    const std::vector<double>& weights,
    const std::vector<unsigned char>& dense_mask) {
    if (cells.size() != weights.size()) {
        throw std::runtime_error(
            "sampled Boltzmann masked distribution has incompatible arrays.");
    }
    BPSampledMaskedDistribution distribution;
    distribution.all_cells = cells;
    std::vector<double> all_weights;
    std::vector<double> dense_weights;
    std::vector<double> outside_weights;
    all_weights.reserve(cells.size());
    dense_weights.reserve(cells.size());
    outside_weights.reserve(cells.size());
    double total_weight = 0.0;
    double dense_weight = 0.0;
    for (std::size_t local = 0; local < cells.size(); ++local) {
        const int cell = cells[local];
        if (cell < 0
            || static_cast<std::size_t>(cell) >= dense_mask.size()) {
            throw std::runtime_error(
                "sampled Boltzmann masked distribution has an invalid cell.");
        }
        const double weight = weights[local];
        if (!(weight >= 0.0) || !std::isfinite(weight)) {
            throw std::runtime_error(
                "sampled Boltzmann masked distribution has an invalid weight.");
        }
        total_weight += weight;
        all_weights.push_back(weight);
        if (dense_mask[static_cast<std::size_t>(cell)]) {
            distribution.dense_cells.push_back(cell);
            dense_weights.push_back(weight);
            dense_weight += weight;
        } else {
            distribution.outside_cells.push_back(cell);
            outside_weights.push_back(weight);
        }
    }
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw std::runtime_error(
            "sampled Boltzmann masked distribution has zero total weight.");
    }
    distribution.dense_probability = std::max(
        0.0, std::min(1.0, dense_weight / total_weight));
    distribution.all_cdf = bp_sampled_probability_cdf(
        bp_sampled_normalize_probability(all_weights));
    if (!dense_weights.empty() && dense_weight > 0.0) {
        distribution.dense_cdf = bp_sampled_probability_cdf(
            bp_sampled_normalize_probability(dense_weights));
    }
    const double outside_weight = total_weight - dense_weight;
    if (!outside_weights.empty() && outside_weight > 0.0) {
        distribution.outside_cdf = bp_sampled_probability_cdf(
            bp_sampled_normalize_probability(outside_weights));
    }
    return distribution;
}

inline int bp_sampled_draw_masked_cell(
    const std::vector<int>& cells,
    const std::vector<double>& cdf,
    double unit_value) {
    const int local = bp_sampled_cdf_index(cdf, unit_value);
    return cells[static_cast<std::size_t>(local)];
}

// The complement has two disjoint categories: first outside/second anywhere,
// or first dense/second outside.  Their weights sum to 1-p_dense1*p_dense2.
inline std::pair<int, int> bp_sampled_draw_conditioned_pair(
    const BPSampledMaskedDistribution& first,
    const BPSampledMaskedDistribution& second,
    double category_unit,
    double first_unit,
    double second_unit) {
    const double dense_first = first.dense_probability;
    const double dense_second = second.dense_probability;
    const double complement_probability =
        1.0 - dense_first * dense_second;
    if (!(complement_probability > 0.0)) {
        throw std::runtime_error(
            "sampled Boltzmann conditioned pair has empty support.");
    }
    const double first_outside_probability =
        (1.0 - dense_first) / complement_probability;
    if (category_unit < first_outside_probability
        || !(dense_first > 0.0)) {
        return {
            bp_sampled_draw_masked_cell(
                first.outside_cells, first.outside_cdf, first_unit),
            bp_sampled_draw_masked_cell(
                second.all_cells, second.all_cdf, second_unit),
        };
    }
    return {
        bp_sampled_draw_masked_cell(
            first.dense_cells, first.dense_cdf, first_unit),
        bp_sampled_draw_masked_cell(
            second.outside_cells, second.outside_cdf, second_unit),
    };
}

// Reweight each mixture component by its available complement probability,
// including each stratified catalog entry.  Compare the resulting normalization
// with explicit dense-block subtraction before using the conditional sampler.
inline BPSampledComplementProposal bp_sampled_build_complement_proposal(
    const BPSampledProposal& proposal,
    const std::vector<unsigned char>& dense_mask,
    double enumerated_complement_probability) {
    if (dense_mask.size()
        != static_cast<std::size_t>(proposal.velocity_cells)) {
        throw std::runtime_error(
            "sampled Boltzmann complement mask has an invalid size.");
    }
    std::vector<int> all_cells(
        static_cast<std::size_t>(proposal.velocity_cells), 0);
    std::iota(all_cells.begin(), all_cells.end(), 0);

    BPSampledComplementProposal complement;
    complement.bulk = bp_sampled_mask_distribution(
        all_cells, proposal.bulk_probability, dense_mask);
    complement.tail = bp_sampled_mask_distribution(
        all_cells, proposal.tail_probability, dense_mask);
    complement.equilibrium = bp_sampled_mask_distribution(
        all_cells, proposal.equilibrium_probability, dense_mask);
    complement.phase_density = bp_sampled_mask_distribution(
        all_cells, proposal.phase_density_probability, dense_mask);
    std::vector<double> uniform_weight(
        all_cells.size(), 1.0);
    complement.uniform = bp_sampled_mask_distribution(
        all_cells, uniform_weight, dense_mask);

    complement.strata.reserve(proposal.strata.size());
    for (const BPSampledStratum& stratum : proposal.strata) {
        if (stratum.cells.empty()) {
            complement.strata.emplace_back();
            continue;
        }
        std::vector<double> weights;
        weights.reserve(stratum.cells.size());
        for (int cell : stratum.cells) {
            weights.push_back(
                proposal.within_stratum_probability[
                    static_cast<std::size_t>(cell)]);
        }
        complement.strata.push_back(
            bp_sampled_mask_distribution(
                stratum.cells, weights, dense_mask));
    }

    std::vector<double> stratified_catalog_weight;
    if (proposal.stratified_fraction > 0.0) {
        const int stratum_count =
            static_cast<int>(proposal.strata.size());
        for (const std::vector<int>& bin : proposal.relative_bin_catalog) {
            if (bin.empty()) {
                continue;
            }
            for (int encoded : bin) {
                int first_stratum = 0;
                int second_stratum = 0;
                int pair_phi = 0;
                std::tie(first_stratum, second_stratum, pair_phi) =
                    bp_sampled_decode_catalog_index(
                        encoded,
                        stratum_count,
                        proposal.pair_phi_points);
                (void)pair_phi;
                const double available =
                    1.0
                    - complement.strata[
                        static_cast<std::size_t>(first_stratum)]
                          .dense_probability
                        * complement.strata[
                            static_cast<std::size_t>(second_stratum)]
                              .dense_probability;
                if (available > 0.0) {
                    complement.stratified_catalog.push_back(encoded);
                    stratified_catalog_weight.push_back(
                        proposal.catalog_probability[
                            static_cast<std::size_t>(encoded)]
                            * available);
                }
            }
        }
        if (!stratified_catalog_weight.empty()) {
            complement.stratified_catalog_cdf =
                bp_sampled_probability_cdf(
                    bp_sampled_normalize_probability(
                        stratified_catalog_weight));
        }
    }

    const double bulk_available =
        1.0
        - complement.bulk.dense_probability
            * complement.bulk.dense_probability;
    const double tail_available =
        1.0
        - complement.tail.dense_probability
            * complement.tail.dense_probability;
    const double uniform_available =
        1.0
        - complement.uniform.dense_probability
            * complement.uniform.dense_probability;
    const double equilibrium_available =
        1.0
        - complement.equilibrium.dense_probability
            * complement.equilibrium.dense_probability;
    const double phase_density_available =
        1.0
        - complement.phase_density.dense_probability
            * complement.phase_density.dense_probability;
    const double stratified_available = std::accumulate(
        stratified_catalog_weight.begin(),
        stratified_catalog_weight.end(),
        0.0);
    const std::array<double, 6> component_weight{
        proposal.bulk_fraction * bulk_available,
        proposal.tail_fraction * tail_available,
        proposal.stratified_fraction * stratified_available,
        proposal.uniform_fraction * uniform_available,
        proposal.equilibrium_fraction * equilibrium_available,
        proposal.phase_density_fraction * phase_density_available,
    };
    std::vector<double> positive_component_weight;
    for (int component = 0; component < 6; ++component) {
        if (component_weight[static_cast<std::size_t>(component)] > 0.0) {
            complement.component_kind.push_back(component);
            positive_component_weight.push_back(
                component_weight[static_cast<std::size_t>(component)]);
        }
    }
    complement.probability = std::accumulate(
        positive_component_weight.begin(),
        positive_component_weight.end(),
        0.0);
    complement.component_cdf = bp_sampled_probability_cdf(
        bp_sampled_normalize_probability(positive_component_weight));
    const double probability_scale = std::max({
        1.0,
        std::abs(complement.probability),
        std::abs(enumerated_complement_probability),
    });
    if (std::abs(
            complement.probability
            - enumerated_complement_probability)
        > 5.0e-12 * probability_scale) {
        throw std::runtime_error(
            "sampled Boltzmann conditional complement probability does not match the full proposal.");
    }
    return complement;
}

// Draw directly from the conditioned mixture, avoiding rejection of pairs
// inside the dense block while retaining the same complement probability.
inline std::tuple<int, int, int> bp_sampled_draw_complement_state(
    const BPSampledProposal& proposal,
    const BPSampledComplementProposal& complement,
    std::uint64_t draw,
    std::uint64_t seed) {
    if (!(complement.probability > 0.0)) {
        throw std::runtime_error(
            "sampled Boltzmann complement proposal has empty support.");
    }
    const double component_unit =
        bp_sampled_shifted_halton(draw, 0, seed);
    const double first_unit =
        bp_sampled_shifted_halton(draw, 1, seed);
    const double second_unit =
        bp_sampled_shifted_halton(draw, 2, seed);
    const double phi_unit =
        bp_sampled_shifted_halton(draw, 3, seed);
    const double category_unit =
        bp_sampled_shifted_halton(draw, 7, seed);
    const int component_local = bp_sampled_cdf_index(
        complement.component_cdf, component_unit);
    const int component = complement.component_kind[
        static_cast<std::size_t>(component_local)];

    const BPSampledMaskedDistribution* distribution = nullptr;
    if (component == 0) {
        distribution = &complement.bulk;
    } else if (component == 1) {
        distribution = &complement.tail;
    } else if (component == 3) {
        distribution = &complement.uniform;
    } else if (component == 4) {
        distribution = &complement.equilibrium;
    } else if (component == 5) {
        distribution = &complement.phase_density;
    }
    if (distribution != nullptr) {
        const std::pair<int, int> pair =
            bp_sampled_draw_conditioned_pair(
                *distribution,
                *distribution,
                category_unit,
                first_unit,
                second_unit);
        const int pair_phi = std::min(
            proposal.pair_phi_points - 1,
            static_cast<int>(
                phi_unit * proposal.pair_phi_points));
        return std::make_tuple(pair.first, pair.second, pair_phi);
    }

    const double catalog_unit =
        bp_sampled_shifted_halton(draw, 4, seed);
    const int catalog_local = bp_sampled_cdf_index(
        complement.stratified_catalog_cdf, catalog_unit);
    const int encoded = complement.stratified_catalog[
        static_cast<std::size_t>(catalog_local)];
    int first_stratum = 0;
    int second_stratum = 0;
    int pair_phi = 0;
    std::tie(first_stratum, second_stratum, pair_phi) =
        bp_sampled_decode_catalog_index(
            encoded,
            static_cast<int>(proposal.strata.size()),
            proposal.pair_phi_points);
    const std::pair<int, int> pair = bp_sampled_draw_conditioned_pair(
        complement.strata[static_cast<std::size_t>(first_stratum)],
        complement.strata[static_cast<std::size_t>(second_stratum)],
        category_unit,
        first_unit,
        second_unit);
    return std::make_tuple(pair.first, pair.second, pair_phi);
}

// At fixed r the spherical velocity measure is 2*pi*dw*deta/r^2.
inline void bp_sampled_initialize_graph_volumes(
    BPReversibleGraph& graph,
    double radius,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    graph.velocity_cell_volume.assign(
        static_cast<std::size_t>(p.Nw * p.NL), 0.0);
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            graph.velocity_cell_volume[static_cast<std::size_t>(
                bp_reversible_velocity_index(j, k, p))] =
                2.0 * PI
                * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
                * geometry.eta.width[static_cast<std::size_t>(k)]
                / (radius * radius);
        }
    }
}

// Apply the conservative whole-pair projection and retain its exact omission
// category when finite domain or positive moment constraints prevent a stencil.
inline void bp_sampled_add_projected_channel(
    BPReversibleGraph& graph,
    int first_index,
    int second_index,
    const BPFAVec3& outgoing_first,
    const BPFAVec3& outgoing_second,
    double coefficient,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    if (!(coefficient > 0.0) || !std::isfinite(coefficient)) {
        return;
    }
    BPReversibleProjection projection;
    bool outside_domain = false;
    unsigned outside_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
    if (!bp_reversible_project_outgoing_pair(
            outgoing_first,
            outgoing_second,
            radius,
            w,
            eta,
            geometry,
            p,
            projection,
            &outside_domain,
            &outside_domain_mask)) {
        BPReversibleOmittedChannel channel{
            first_index,
            second_index,
            coefficient,
            outside_domain_mask,
        };
        if (outside_domain) {
            graph.omitted_domain_channels.push_back(channel);
        } else {
            graph.omitted_projection_channels.push_back(channel);
        }
        return;
    }
    BPReversibleReaction reaction;
    reaction.incoming_first = first_index;
    reaction.incoming_second = second_index;
    reaction.coefficient = coefficient;
    reaction.species = bp_reversible_compress_reaction(
        first_index,
        second_index,
        projection.products);
    graph.reactions.push_back(std::move(reaction));
    graph.maximum_projection_error = std::max(
        graph.maximum_projection_error,
        std::max({
            std::abs(projection.mass_error),
            std::abs(projection.momentum_error),
            std::abs(projection.energy_error),
        }));
}

// Identical incoming cells and species powers define the same reversible edge;
// their physical coefficients can be added without changing that edge.
inline bool bp_sampled_same_reaction_stencil(
    const BPReversibleReaction& first,
    const BPReversibleReaction& second) {
    if (first.incoming_first != second.incoming_first
        || first.incoming_second != second.incoming_second
        || first.species.size() != second.species.size()) {
        return false;
    }
    for (std::size_t i = 0; i < first.species.size(); ++i) {
        const BPReversibleSpecies& a = first.species[i];
        const BPReversibleSpecies& b = second.species[i];
        if (a.index != b.index
            || a.product_order != b.product_order
            || a.net_population_change != b.net_population_change) {
            return false;
        }
    }
    return true;
}

// Merge repeated newly appended stencils so one finite-time reaction carries
// their summed coefficient.  Earlier graph entries keep their order.
inline void bp_sampled_coalesce_reaction_suffix(
    BPReversibleGraph& graph,
    std::size_t first_reaction) {
    if (first_reaction >= graph.reactions.size()) {
        return;
    }
    std::vector<BPReversibleReaction> merged;
    merged.reserve(graph.reactions.size() - first_reaction);
    for (std::size_t i = first_reaction;
         i < graph.reactions.size();
         ++i) {
        BPReversibleReaction& reaction = graph.reactions[i];
        auto existing = std::find_if(
            merged.begin(),
            merged.end(),
            [&](const BPReversibleReaction& candidate) {
                return bp_sampled_same_reaction_stencil(
                    candidate, reaction);
            });
        if (existing == merged.end()) {
            merged.push_back(std::move(reaction));
        } else {
            existing->coefficient += reaction.coefficient;
        }
    }
    graph.reactions.resize(first_reaction);
    graph.reactions.insert(
        graph.reactions.end(),
        std::make_move_iterator(merged.begin()),
        std::make_move_iterator(merged.end()));
}

// Construct two local Cartesian velocities from w, eta and tangential azimuth;
// their center of mass and relative direction define elastic scattering.
inline bool bp_sampled_pair_kinematics(
    int first_index,
    int second_index,
    int pair_phi,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const std::vector<double>& pair_phi_cos,
    const std::vector<double>& pair_phi_sin,
    const Params& p,
    BPFAVec3& center_of_mass,
    BPFAVec3& incoming_direction,
    std::array<BPFAVec3, 2>& basis,
    double& relative_speed,
    double& total_cross_section) {
    const std::array<int, 2> first_grid =
        bp_reversible_velocity_indices(first_index, p);
    const std::array<int, 2> second_grid =
        bp_reversible_velocity_indices(second_index, p);
    const double first_vt =
        std::sqrt(2.0 * eta[static_cast<std::size_t>(first_grid[1])]) / radius;
    const double second_vt =
        std::sqrt(2.0 * eta[static_cast<std::size_t>(second_grid[1])]) / radius;
    const BPFAVec3 first{
        w[static_cast<std::size_t>(first_grid[0])],
        first_vt,
        0.0,
    };
    const BPFAVec3 second{
        w[static_cast<std::size_t>(second_grid[0])],
        second_vt * pair_phi_cos[static_cast<std::size_t>(pair_phi)],
        second_vt * pair_phi_sin[static_cast<std::size_t>(pair_phi)],
    };
    const BPFAVec3 relative = bp_fa_sub(first, second);
    relative_speed = bp_fa_norm(relative);
    if (!(relative_speed > 0.0)) {
        return false;
    }
    total_cross_section = bp_fa_total_cross_section(relative_speed, p);
    if (!(total_cross_section > 0.0)) {
        return false;
    }
    center_of_mass = bp_fa_scale(bp_fa_add(first, second), 0.5);
    incoming_direction = bp_fa_scale(relative, 1.0 / relative_speed);
    basis = bp_fa_perpendicular_basis_unit(incoming_direction);
    return true;
}

// Rotate the relative velocity at fixed magnitude, then restore the common
// center-of-mass velocity, preserving the continuous pair kinetic invariants.
inline void bp_sampled_outgoing_velocities(
    const BPFAVec3& center_of_mass,
    const BPFAVec3& incoming_direction,
    const std::array<BPFAVec3, 2>& basis,
    double relative_speed,
    double cosine,
    double psi_cosine,
    double psi_sine,
    BPFAVec3& outgoing_first,
    BPFAVec3& outgoing_second) {
    const double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
    const BPFAVec3 outgoing_direction =
        bp_fa_scattered_relative_direction_from_basis(
            incoming_direction,
            basis[0],
            basis[1],
            cosine,
            sine,
            psi_cosine,
            psi_sine);
    const BPFAVec3 outgoing_relative =
        bp_fa_scale(outgoing_direction, relative_speed);
    outgoing_first =
        bp_fa_add(center_of_mass, bp_fa_scale(outgoing_relative, 0.5));
    outgoing_second =
        bp_fa_sub(center_of_mass, bp_fa_scale(outgoing_relative, 0.5));
}

// Evaluate current population-dependent represented and omitted event activity.
inline void bp_sampled_finalize_graph_activity(
    BPReversibleGraph& graph,
    const std::vector<double>& shell_f) {
    const BPReversibleActivityBudget budget =
        bp_reversible_activity_budget(shell_f, graph);
    graph.represented_activity = budget.represented;
    graph.omitted_domain_activity = budget.omitted_domain;
    graph.omitted_projection_activity = budget.omitted_projection;
    graph.omitted_activity = budget.omitted();
}

// Rank cells by a population-weighted collision-activity proxy and add radial
// reversal partners together within the prescribed dense-cell count.
inline std::vector<int> bp_sampled_dense_velocity_cells(
    const BPSampledProposal& proposal,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const Params& p) {
    std::vector<int> order(static_cast<std::size_t>(proposal.velocity_cells), 0);
    std::vector<double> activity_score(
        static_cast<std::size_t>(proposal.velocity_cells), 0.0);
    std::iota(order.begin(), order.end(), 0);
    for (int q = 0; q < proposal.velocity_cells; ++q) {
        const double energy =
            bp_reversible_energy(q, radius, w, eta, p);
        const double speed = std::sqrt(std::max(0.0, 2.0 * energy));
        const double collision_speed = std::max(speed, p.bp_sigma_u0);
        activity_score[static_cast<std::size_t>(q)] =
            proposal.bulk_probability[static_cast<std::size_t>(q)]
            * collision_speed
            * bp_fa_total_cross_section(collision_speed, p);
    }
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](int first, int second) {
            return activity_score[static_cast<std::size_t>(first)]
                > activity_score[static_cast<std::size_t>(second)];
        });
    const int dense_limit = std::min(
        p.bp_sampled_dense_cells, proposal.velocity_cells);
    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(dense_limit));
    std::vector<unsigned char> selected_mask(
        static_cast<std::size_t>(proposal.velocity_cells), 0);
    for (int q : order) {
        if (static_cast<int>(selected.size()) >= dense_limit) {
            break;
        }
        if (selected_mask[static_cast<std::size_t>(q)]) {
            continue;
        }
        const int reverse =
            proposal.time_reverse_cell[static_cast<std::size_t>(q)];
        const bool add_reverse =
            reverse >= 0
            && reverse != q
            && !selected_mask[static_cast<std::size_t>(reverse)];
        const int required = add_reverse ? 2 : 1;
        if (static_cast<int>(selected.size()) + required > dense_limit) {
            continue;
        }
        selected.push_back(q);
        selected_mask[static_cast<std::size_t>(q)] = 1;
        if (add_reverse) {
            selected.push_back(reverse);
            selected_mask[static_cast<std::size_t>(reverse)] = 1;
        }
    }
    return selected;
}

// Subtract the full mixture probability of every dense-block pair/angle.
inline double bp_sampled_complement_probability(
    const BPSampledProposal& proposal,
    const std::vector<int>& dense_cells) {
    double dense_probability = 0.0;
    for (int first : dense_cells) {
        for (int second : dense_cells) {
            for (int iphi = 0; iphi < proposal.pair_phi_points; ++iphi) {
                dense_probability += bp_sampled_joint_proposal_probability(
                    proposal, first, second, iphi);
            }
        }
    }
    const double complement = std::max(0.0, 1.0 - dense_probability);
    return complement
        <= 64.0 * std::numeric_limits<double>::epsilon()
        ? 0.0
        : complement;
}

inline void bp_sampled_move_scaled_graph(
    BPReversibleGraph& destination,
    BPReversibleGraph& source,
    double scale);

// Enumerate dense incoming pairs and their angular quadrature, adding distinct
// radial-reverse partners together.  Workers build private ordered graph ranges.
inline BPReversibleGraph bp_sampled_build_dense_graph(
    const std::vector<double>& shell_f,
    const std::vector<int>& dense_cells,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    BPReversibleGraph graph;
    bp_sampled_initialize_graph_volumes(graph, radius, geometry, p);
    const int npair_phi = std::max(1, p.bp_boltzmann_pair_phi_points);
    const int nmu = std::max(1, p.bp_boltzmann_scatter_mu_points);
    const int npsi = std::max(1, p.bp_boltzmann_scatter_phi_points);
    std::vector<double> pair_phi_cos;
    std::vector<double> pair_phi_sin;
    std::vector<double> psi_cos;
    std::vector<double> psi_sin;
    std::vector<double> mu;
    std::vector<double> mu_weight;
    bp_fa_midpoint_angle_table(npair_phi, pair_phi_cos, pair_phi_sin);
    bp_fa_midpoint_angle_table(npsi, psi_cos, psi_sin);
    bp_fa_gauss_legendre_mu_table(nmu, mu, mu_weight);
    const std::string kernel = bp_fa_kernel_name(p);
    std::vector<unsigned char> dense_mask(
        static_cast<std::size_t>(p.Nw * p.NL), 0);
    for (int q : dense_cells) {
        dense_mask[static_cast<std::size_t>(q)] = 1;
    }
    std::vector<std::array<int, 2>> pair_tasks;
    pair_tasks.reserve(dense_cells.size() * dense_cells.size());
    for (int first : dense_cells) {
        for (int second : dense_cells) {
            const std::array<int, 2> first_indices =
                bp_reversible_velocity_indices(first, p);
            const std::array<int, 2> second_indices =
                bp_reversible_velocity_indices(second, p);
            const int reverse_first = bp_reversible_velocity_index(
                p.Nw - 1 - first_indices[0], first_indices[1], p);
            const int reverse_second = bp_reversible_velocity_index(
                p.Nw - 1 - second_indices[0], second_indices[1], p);
            const bool reverse_in_dense =
                dense_mask[static_cast<std::size_t>(reverse_first)]
                && dense_mask[static_cast<std::size_t>(reverse_second)];
            if (reverse_in_dense && std::tie(first, second)
                > std::tie(reverse_first, reverse_second)) {
                continue;
            }
            pair_tasks.push_back({first, second});
        }
    }

    const int useful_ranges = std::max(
        1, static_cast<int>((pair_tasks.size() + 63) / 64));
    const int workers = effective_threads(p, useful_ranges);
    std::vector<BPReversibleGraph> partial(
        static_cast<std::size_t>(workers));
    parallel_for_chunks(
        0,
        static_cast<int>(pair_tasks.size()),
        workers,
        [&](int lo, int hi, int tid) {
            BPReversibleGraph& local_graph =
                partial[static_cast<std::size_t>(tid)];
            bp_sampled_initialize_graph_volumes(
                local_graph, radius, geometry, p);
            const std::size_t maximum_local_reactions =
                static_cast<std::size_t>(hi - lo)
                * static_cast<std::size_t>(npair_phi)
                * static_cast<std::size_t>(nmu)
                * static_cast<std::size_t>(npsi)
                * 2;
            local_graph.reactions.reserve(maximum_local_reactions);
            std::vector<double> normalized_mu(
                static_cast<std::size_t>(nmu), 0.0);
            for (int task = lo; task < hi; ++task) {
            const int first =
                pair_tasks[static_cast<std::size_t>(task)][0];
            const int second =
                pair_tasks[static_cast<std::size_t>(task)][1];
            const std::array<int, 2> first_indices =
                bp_reversible_velocity_indices(first, p);
            const std::array<int, 2> second_indices =
                bp_reversible_velocity_indices(second, p);
            const int reverse_first = bp_reversible_velocity_index(
                p.Nw - 1 - first_indices[0], first_indices[1], p);
            const int reverse_second = bp_reversible_velocity_index(
                p.Nw - 1 - second_indices[0], second_indices[1], p);
            const bool has_distinct_reverse =
                dense_mask[static_cast<std::size_t>(reverse_first)]
                && dense_mask[static_cast<std::size_t>(reverse_second)]
                && (reverse_first != first || reverse_second != second);
            const std::size_t pair_reaction_begin =
                local_graph.reactions.size();
            for (int iphi = 0; iphi < npair_phi; ++iphi) {
                BPFAVec3 center_of_mass;
                BPFAVec3 incoming_direction;
                std::array<BPFAVec3, 2> basis;
                double relative_speed = 0.0;
                double sigma = 0.0;
                if (!bp_sampled_pair_kinematics(
                        first,
                        second,
                        iphi,
                        radius,
                        w,
                        eta,
                        pair_phi_cos,
                        pair_phi_sin,
                        p,
                        center_of_mass,
                        incoming_direction,
                        basis,
                        relative_speed,
                        sigma)) {
                    continue;
                }
                std::fill(
                    normalized_mu.begin(), normalized_mu.end(), 0.0);
                double mu_sum = 0.0;
                for (int im = 0; im < nmu; ++im) {
                    normalized_mu[static_cast<std::size_t>(im)] =
                        mu_weight[static_cast<std::size_t>(im)]
                        * bp_fa_scatter_weight_raw(
                            mu[static_cast<std::size_t>(im)],
                            relative_speed,
                            p,
                            kernel);
                    mu_sum += normalized_mu[static_cast<std::size_t>(im)];
                }
                if (!(mu_sum > 0.0)) {
                    continue;
                }
                for (int im = 0; im < nmu; ++im) {
                    const double angular_mu =
                        normalized_mu[static_cast<std::size_t>(im)] / mu_sum;
                    for (int ipsi = 0; ipsi < npsi; ++ipsi) {
                        BPFAVec3 outgoing_first;
                        BPFAVec3 outgoing_second;
                        bp_sampled_outgoing_velocities(
                            center_of_mass,
                            incoming_direction,
                            basis,
                            relative_speed,
                            mu[static_cast<std::size_t>(im)],
                            psi_cos[static_cast<std::size_t>(ipsi)],
                            psi_sin[static_cast<std::size_t>(ipsi)],
                            outgoing_first,
                            outgoing_second);
                        const double coefficient =
                            0.25
                            * local_graph.velocity_cell_volume[
                                static_cast<std::size_t>(first)]
                            * local_graph.velocity_cell_volume[
                                static_cast<std::size_t>(second)]
                            * relative_speed * sigma
                            / static_cast<double>(npair_phi)
                            * angular_mu
                            / static_cast<double>(npsi);
                        bp_sampled_add_projected_channel(
                            local_graph,
                            first,
                            second,
                            outgoing_first,
                            outgoing_second,
                            coefficient,
                            radius,
                            w,
                            eta,
                            geometry,
                            p);
                        if (has_distinct_reverse) {
                            bp_sampled_add_projected_channel(
                                local_graph,
                                reverse_first,
                                reverse_second,
                                bp_fa_scale(outgoing_first, -1.0),
                                bp_fa_scale(outgoing_second, -1.0),
                                coefficient,
                                radius,
                                w,
                                eta,
                                geometry,
                                p);
                        }
                    }
                }
            }
            bp_sampled_coalesce_reaction_suffix(
                local_graph, pair_reaction_begin);
            }
        });
    for (BPReversibleGraph& value : partial) {
        bp_sampled_move_scaled_graph(graph, value, 1.0);
        }
    bp_sampled_finalize_graph_activity(graph, shell_f);
    return graph;
}

// A separate pilot quantile sets a practical thinning scale.  The subsequent
// acceptance/reciprocal rule remains valid on either side of this threshold.
inline double bp_sampled_ntc_activity_threshold(
    const BPSampledProposal& proposal,
    const BPSampledComplementProposal& complement_proposal,
    double complement_probability,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const Params& p,
    std::uint64_t seed) {
    if (!p.bp_sampled_ntc_pair_thinning
        || !(complement_probability > 0.0)) {
        return 0.0;
    }
    std::vector<double> pair_phi_cos;
    std::vector<double> pair_phi_sin;
    bp_fa_midpoint_angle_table(
        proposal.pair_phi_points,
        pair_phi_cos,
        pair_phi_sin);
    const int pilot_samples =
        std::max(1, p.bp_sampled_ntc_pilot_samples);
    std::vector<double> activity;
    activity.reserve(static_cast<std::size_t>(pilot_samples));
    const std::uint64_t pilot_seed =
        bp_sampled_splitmix64(
            seed ^ UINT64_C(0x94d049bb133111eb));
    std::uint64_t draws = 0;
    while (activity.size() < static_cast<std::size_t>(pilot_samples)) {
        int first = 0;
        int second = 0;
        int iphi = 0;
        std::tie(first, second, iphi) =
            bp_sampled_draw_complement_state(
                proposal,
                complement_proposal,
                draws,
                pilot_seed);
        ++draws;
        BPFAVec3 center_of_mass;
        BPFAVec3 incoming_direction;
        std::array<BPFAVec3, 2> basis;
        double relative_speed = 0.0;
        double sigma = 0.0;
        const bool valid =
            bp_sampled_pair_kinematics(
                first,
                second,
                iphi,
                radius,
                w,
                eta,
                pair_phi_cos,
                pair_phi_sin,
                p,
                center_of_mass,
                incoming_direction,
                basis,
                relative_speed,
                sigma);
        activity.push_back(
            valid ? relative_speed * sigma : 0.0);
    }
    const std::size_t quantile_index =
        std::min(
            activity.size() - 1,
            static_cast<std::size_t>(
                std::floor(
                    p.bp_sampled_ntc_quantile
                    * static_cast<double>(activity.size() - 1))));
    std::nth_element(
        activity.begin(),
        activity.begin() + static_cast<std::ptrdiff_t>(quantile_index),
        activity.end());
    return activity[quantile_index];
}

// Build a finite complement graph with actual joint probabilities, retained
// thinning weights and angular samples.  Count all conditioned draws, including
// zero-rate and thinned draws, in the estimator normalization.
inline BPSampledBatch bp_sampled_build_complement_batch_serial(
    const std::vector<double>& shell_f,
    const BPSampledProposal& proposal,
    const BPSampledComplementProposal& complement_proposal,
    const std::vector<unsigned char>& dense_mask,
    double complement_probability,
    int sample_count,
    std::uint64_t seed,
    std::uint64_t draw_offset,
    double ntc_activity_threshold,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    int normalization_sample_count = 0) {
    BPSampledBatch batch;
    bp_sampled_initialize_graph_volumes(batch.graph, radius, geometry, p);
    if (sample_count <= 0 || !(complement_probability > 0.0)) {
        return batch;
    }
    const int npair_phi = proposal.pair_phi_points;
    const int normalization_samples = normalization_sample_count > 0
        ? normalization_sample_count
        : sample_count;
    if (normalization_samples < sample_count) {
        throw std::runtime_error(
            "sampled Boltzmann batch normalization is smaller than its sample range.");
    }
    std::vector<double> pair_phi_cos;
    std::vector<double> pair_phi_sin;
    bp_fa_midpoint_angle_table(npair_phi, pair_phi_cos, pair_phi_sin);
    std::vector<double> scatter_mu;
    std::vector<double> scatter_mu_weight;
    std::vector<double> scatter_psi_cos;
    std::vector<double> scatter_psi_sin;
    bp_fa_gauss_legendre_mu_table(
        std::max(1, p.bp_boltzmann_scatter_mu_points),
        scatter_mu,
        scatter_mu_weight);
    bp_fa_midpoint_angle_table(
        std::max(1, p.bp_boltzmann_scatter_phi_points),
        scatter_psi_cos,
        scatter_psi_sin);
    const std::string kernel = bp_fa_kernel_name(p);
    if (p.bp_sampled_ntc_pair_thinning
        && (!(ntc_activity_threshold >= 0.0)
            || !std::isfinite(ntc_activity_threshold))) {
        throw std::runtime_error(
            "sampled Boltzmann NTC activity threshold is not finite.");
    }
    while (batch.accepted_samples < static_cast<std::uint64_t>(sample_count)) {
        int first = 0;
        int second = 0;
        int iphi = 0;
        const std::uint64_t sequence_draw =
            draw_offset + batch.attempted_draws;
        std::tie(first, second, iphi) =
            bp_sampled_draw_complement_state(
                proposal,
                complement_proposal,
                sequence_draw,
                seed);
        const std::uint64_t angular_draw = sequence_draw;
        ++batch.attempted_draws;
        if (dense_mask[static_cast<std::size_t>(first)]
            && dense_mask[static_cast<std::size_t>(second)]) {
            throw std::runtime_error(
                "sampled Boltzmann conditioned complement draw entered the dense block.");
        }
        ++batch.accepted_samples;
        const double proposal_probability =
            bp_sampled_joint_proposal_probability(
                proposal, first, second, iphi);
        if (!(proposal_probability > 0.0)) {
            throw std::runtime_error(
                "sampled Boltzmann full-support proposal assigned zero probability.");
        }
        BPFAVec3 center_of_mass;
        BPFAVec3 incoming_direction;
        std::array<BPFAVec3, 2> basis;
        double relative_speed = 0.0;
        double sigma = 0.0;
        if (!bp_sampled_pair_kinematics(
                first,
                second,
                iphi,
                radius,
                w,
                eta,
                pair_phi_cos,
                pair_phi_sin,
                p,
                center_of_mass,
                incoming_direction,
                basis,
                relative_speed,
                sigma)) {
            continue;
        }
        const double pair_activity = relative_speed * sigma;
        double inverse_ntc_acceptance = 1.0;
        if (p.bp_sampled_ntc_pair_thinning
            && ntc_activity_threshold > 0.0) {
            const double acceptance_draw =
                bp_sampled_shifted_halton(
                    sequence_draw,
                    6,
                    seed ^ UINT64_C(0xd1b54a32d192ed03));
            inverse_ntc_acceptance =
                bp_sampled_ntc_inverse_acceptance(
                    pair_activity,
                    ntc_activity_threshold,
                    acceptance_draw);
            if (!(inverse_ntc_acceptance > 0.0)) {
                continue;
            }
        }
        ++batch.retained_collision_draws;
        const double pair_coefficient =
            0.25
            * batch.graph.velocity_cell_volume[static_cast<std::size_t>(first)]
            * batch.graph.velocity_cell_volume[static_cast<std::size_t>(second)]
            * pair_activity
            * inverse_ntc_acceptance;
        const double target_pair_weight =
            1.0 / static_cast<double>(npair_phi);
        const int reverse_first =
            proposal.time_reverse_cell[static_cast<std::size_t>(first)];
        const int reverse_second =
            proposal.time_reverse_cell[static_cast<std::size_t>(second)];
        const bool reverse_is_complement =
            reverse_first >= 0
            && reverse_second >= 0
            && !(dense_mask[static_cast<std::size_t>(reverse_first)]
                 && dense_mask[static_cast<std::size_t>(reverse_second)]);
        // Pair a draw with its radial reverse and divide by their combined
        // proposal probability, using the same outgoing angular sample.
        if (reverse_is_complement) {
            const double reverse_probability =
                bp_sampled_joint_proposal_probability(
                    proposal, reverse_first, reverse_second, iphi);
            const double paired_importance_weight =
                complement_probability
                * target_pair_weight
                / (static_cast<double>(normalization_samples)
                   * (proposal_probability + reverse_probability));
            const double reverse_pair_coefficient =
                0.25
                * batch.graph.velocity_cell_volume[
                    static_cast<std::size_t>(reverse_first)]
                * batch.graph.velocity_cell_volume[
                    static_cast<std::size_t>(reverse_second)]
                * pair_activity
                * inverse_ntc_acceptance;
            for (int angle = 0;
                 angle < p.bp_sampled_outgoing_angle_samples;
                 ++angle) {
                const std::array<double, 2> scatter =
                    bp_sampled_stratified_outgoing_units(
                        angular_draw,
                        angle,
                        p.bp_sampled_outgoing_angle_samples,
                        p.bp_boltzmann_scatter_mu_points,
                        seed);
                double sampled_mu = 0.0;
                double sampled_psi_cos = 0.0;
                double sampled_psi_sin = 0.0;
                bp_sampled_discrete_scatter_direction(
                    scatter[0],
                    scatter[1],
                    relative_speed,
                    p,
                    kernel,
                    scatter_mu,
                    scatter_mu_weight,
                    scatter_psi_cos,
                    scatter_psi_sin,
                    sampled_mu,
                    sampled_psi_cos,
                    sampled_psi_sin);
                BPFAVec3 outgoing_first;
                BPFAVec3 outgoing_second;
                bp_sampled_outgoing_velocities(
                    center_of_mass,
                    incoming_direction,
                    basis,
                    relative_speed,
                    sampled_mu,
                    sampled_psi_cos,
                    sampled_psi_sin,
                    outgoing_first,
                    outgoing_second);
                const double angular_importance =
                    paired_importance_weight
                    / static_cast<double>(
                        p.bp_sampled_outgoing_angle_samples);
                bp_sampled_add_projected_channel(
                    batch.graph,
                    first,
                    second,
                    outgoing_first,
                    outgoing_second,
                    pair_coefficient * angular_importance,
                    radius,
                    w,
                    eta,
                    geometry,
                    p);
                bp_sampled_add_projected_channel(
                    batch.graph,
                    reverse_first,
                    reverse_second,
                    bp_fa_scale(outgoing_first, -1.0),
                    bp_fa_scale(outgoing_second, -1.0),
                    reverse_pair_coefficient * angular_importance,
                    radius,
                    w,
                    eta,
                    geometry,
                    p);
            }
        } else {
            // When a reverse partner belongs outside this sampled support,
            // use the directly drawn channel's own proposal probability.
            const double importance_weight =
                complement_probability
                * target_pair_weight
                / (static_cast<double>(normalization_samples)
                   * proposal_probability);
            for (int angle = 0;
                 angle < p.bp_sampled_outgoing_angle_samples;
                 ++angle) {
                const std::array<double, 2> scatter =
                    bp_sampled_stratified_outgoing_units(
                        angular_draw,
                        angle,
                        p.bp_sampled_outgoing_angle_samples,
                        p.bp_boltzmann_scatter_mu_points,
                        seed);
                double sampled_mu = 0.0;
                double sampled_psi_cos = 0.0;
                double sampled_psi_sin = 0.0;
                bp_sampled_discrete_scatter_direction(
                    scatter[0],
                    scatter[1],
                    relative_speed,
                    p,
                    kernel,
                    scatter_mu,
                    scatter_mu_weight,
                    scatter_psi_cos,
                    scatter_psi_sin,
                    sampled_mu,
                    sampled_psi_cos,
                    sampled_psi_sin);
                BPFAVec3 outgoing_first;
                BPFAVec3 outgoing_second;
                bp_sampled_outgoing_velocities(
                    center_of_mass,
                    incoming_direction,
                    basis,
                    relative_speed,
                    sampled_mu,
                    sampled_psi_cos,
                    sampled_psi_sin,
                    outgoing_first,
                    outgoing_second);
                bp_sampled_add_projected_channel(
                    batch.graph,
                    first,
                    second,
                    outgoing_first,
                    outgoing_second,
                    pair_coefficient * importance_weight
                        / static_cast<double>(
                            p.bp_sampled_outgoing_angle_samples),
                    radius,
                    w,
                    eta,
                    geometry,
                    p);
            }
        }
    }
    bp_sampled_finalize_graph_activity(batch.graph, shell_f);
    return batch;
}

/// Builds one deterministic complement estimate using contiguous sample ranges.
///
/// Every range keeps the serial low-discrepancy sequence index and uses the
/// total batch size in its importance normalization.  Concatenating ranges in
/// thread-index order therefore gives the same ordered reaction list as the
/// serial builder, apart from roundoff in independently accumulated activity
/// diagnostics, which are recomputed after the merge.
inline BPSampledBatch bp_sampled_build_complement_batch(
    const std::vector<double>& shell_f,
    const BPSampledProposal& proposal,
    const BPSampledComplementProposal& complement_proposal,
    const std::vector<unsigned char>& dense_mask,
    double complement_probability,
    int sample_count,
    std::uint64_t seed,
    std::uint64_t draw_offset,
    double ntc_activity_threshold,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    const int useful_ranges = std::max(
        1, (sample_count + 255) / 256);
    const int workers = effective_threads(p, useful_ranges);
    if (workers == 1) {
        return bp_sampled_build_complement_batch_serial(
            shell_f,
            proposal,
            complement_proposal,
            dense_mask,
            complement_probability,
            sample_count,
            seed,
            draw_offset,
            ntc_activity_threshold,
            radius,
            w,
            eta,
            geometry,
            p);
    }

    std::vector<BPSampledBatch> partial(
        static_cast<std::size_t>(workers));
    parallel_for_chunks(
        0,
        sample_count,
        workers,
        [&](int lo, int hi, int tid) {
            partial[static_cast<std::size_t>(tid)] =
                bp_sampled_build_complement_batch_serial(
                    shell_f,
                    proposal,
                    complement_proposal,
                    dense_mask,
                    complement_probability,
                    hi - lo,
                    seed,
                    draw_offset + static_cast<std::uint64_t>(lo),
                    ntc_activity_threshold,
                    radius,
                    w,
                    eta,
                    geometry,
                    p,
                    sample_count);
        });

    BPSampledBatch batch;
    bp_sampled_initialize_graph_volumes(
        batch.graph, radius, geometry, p);
    for (BPSampledBatch& value : partial) {
        batch.accepted_samples += value.accepted_samples;
        batch.attempted_draws += value.attempted_draws;
        batch.retained_collision_draws +=
            value.retained_collision_draws;
        bp_sampled_move_scaled_graph(batch.graph, value.graph, 1.0);
    }
    bp_sampled_finalize_graph_activity(batch.graph, shell_f);
    return batch;
}

inline std::vector<double> bp_sampled_add_rhs(
    const std::vector<double>& first,
    const std::vector<double>& second) {
    if (first.size() != second.size()) {
        throw std::runtime_error("sampled Boltzmann RHS vectors have different sizes.");
    }
    std::vector<double> sum(first.size(), 0.0);
    for (std::size_t q = 0; q < sum.size(); ++q) {
        sum[q] = first[q] + second[q];
    }
    return sum;
}

// Error measures compare replicate estimates of net distribution change and
// the radial energy-flux moment; gross rates set scales near cancellation.
struct BPSampledBatchError {
    double rhs = 0.0;
    double rhs_gross = 0.0;
    double heat = 0.0;
    double heat_gross = 0.0;
    double energy_invariant = 0.0;
};

// Estimate variance of the replicate mean with 1/[R*(R-1)].  Integrate its
// velocity-cell norm using d^3v; heat uses energy*w and the energy residual
// checks each replicate against the gross energy-change scale.
inline BPSampledBatchError bp_sampled_replicated_batch_error(
    const std::vector<std::vector<double>>& rhs_replicates,
    const std::vector<double>& cell_volume,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const Params& p,
    const std::vector<std::vector<double>>* gross_replicates = nullptr) {
    if (rhs_replicates.size() < 2) {
        throw std::runtime_error(
            "sampled Boltzmann error estimate requires at least two replicas.");
    }
    const std::size_t cells = rhs_replicates.front().size();
    for (const std::vector<double>& rhs : rhs_replicates) {
        if (rhs.size() != cells) {
            throw std::runtime_error(
                "sampled Boltzmann RHS replicas have incompatible sizes.");
        }
    }
    if (gross_replicates != nullptr) {
        if (gross_replicates->size() != rhs_replicates.size()) {
            throw std::runtime_error(
                "sampled Boltzmann gross RHS replica count is incompatible.");
        }
        for (const std::vector<double>& gross : *gross_replicates) {
            if (gross.size() != cells) {
                throw std::runtime_error(
                    "sampled Boltzmann gross RHS replicas have incompatible sizes.");
            }
        }
    }
    if (cell_volume.size() != cells) {
        throw std::runtime_error(
            "sampled Boltzmann cell volumes have incompatible size.");
    }
    BPSampledBatchError error;
    const std::size_t count = rhs_replicates.size();
    const double inverse_count = 1.0 / static_cast<double>(count);
    const double inverse_standard_error_denom =
        1.0 / (
            static_cast<double>(count)
            * static_cast<double>(count - 1));
    double rhs_variance_of_mean = 0.0;
    double rhs_mean_square = 0.0;
    double rhs_gross_mean_square = 0.0;
    std::vector<double> heat_replicate(count, 0.0);
    std::vector<double> energy_replicate(count, 0.0);
    double heat_gross = 0.0;
    double energy_gross = 0.0;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const double volume = cell_volume[static_cast<std::size_t>(q)];
        const double energy =
            bp_reversible_energy(q, radius, w, eta, p);
        const double radial_velocity =
            bp_reversible_radial_velocity(q, w, p);
        double mean = 0.0;
        for (const std::vector<double>& rhs : rhs_replicates) {
            mean += inverse_count * rhs[static_cast<std::size_t>(q)];
        }
        double variance_numerator = 0.0;
        for (const std::vector<double>& rhs : rhs_replicates) {
            const double difference =
                rhs[static_cast<std::size_t>(q)] - mean;
            variance_numerator += difference * difference;
        }
        rhs_variance_of_mean +=
            volume * variance_numerator
            * inverse_standard_error_denom;
        rhs_mean_square += volume * mean * mean;
        if (gross_replicates != nullptr) {
            double gross_mean = 0.0;
            for (const std::vector<double>& gross : *gross_replicates) {
                gross_mean +=
                    inverse_count * gross[static_cast<std::size_t>(q)];
            }
            rhs_gross_mean_square +=
                volume * gross_mean * gross_mean;
            heat_gross +=
                volume * energy * std::abs(radial_velocity)
                * gross_mean;
            energy_gross += volume * energy * gross_mean;
        } else {
            double mean_absolute_rhs = 0.0;
            for (const std::vector<double>& rhs : rhs_replicates) {
                mean_absolute_rhs +=
                    inverse_count
                    * std::abs(rhs[static_cast<std::size_t>(q)]);
            }
            heat_gross += volume * energy * std::abs(radial_velocity)
                * mean_absolute_rhs;
            energy_gross += volume * energy * mean_absolute_rhs;
        }
        for (std::size_t replicate = 0; replicate < count; ++replicate) {
            const double value =
                rhs_replicates[replicate][static_cast<std::size_t>(q)];
            heat_replicate[replicate] +=
                volume * energy * radial_velocity * value;
            energy_replicate[replicate] +=
                volume * energy * value;
        }
    }
    const double rhs_scale =
        std::sqrt(std::max(0.0, rhs_mean_square));
    const double rhs_gross_scale =
        (gross_replicates != nullptr)
        ? std::sqrt(std::max(0.0, rhs_gross_mean_square))
        : 0.0;
    const double rhs_standard_error =
        std::sqrt(std::max(0.0, rhs_variance_of_mean));
    error.rhs =
        rhs_standard_error
        / std::max({
            rhs_scale,
            p.bp_sampled_rhs_floor_fraction * rhs_gross_scale,
            1.0e-300,
        });
    error.rhs_gross =
        rhs_standard_error / std::max(rhs_gross_scale, 1.0e-300);
    double mean_heat = 0.0;
    for (double value : heat_replicate) {
        mean_heat += inverse_count * value;
    }
    double heat_variance_numerator = 0.0;
    for (double value : heat_replicate) {
        const double difference = value - mean_heat;
        heat_variance_numerator += difference * difference;
    }
    const double heat_standard_error =
        std::sqrt(std::max(
            0.0,
            heat_variance_numerator
                * inverse_standard_error_denom));
    const double heat_scale = std::max({
        std::abs(mean_heat),
        p.bp_sampled_heat_floor_fraction * heat_gross,
        1.0e-300,
    });
    error.heat = heat_standard_error / heat_scale;
    error.heat_gross =
        heat_standard_error / std::max(heat_gross, 1.0e-300);
    for (double value : energy_replicate) {
        error.energy_invariant = std::max(
            error.energy_invariant,
            std::abs(value) / std::max(energy_gross, 1.0e-300));
    }
    return error;
}

// The two-batch interface uses the same replicate estimator and optional
// gross-rate normalization as the general multi-replicate calculation.
inline BPSampledBatchError bp_sampled_batch_error(
    const std::vector<double>& rhs_first,
    const std::vector<double>& rhs_second,
    const std::vector<double>& cell_volume,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const Params& p,
    const std::vector<double>* gross_first = nullptr,
    const std::vector<double>* gross_second = nullptr) {
    if ((gross_first == nullptr) != (gross_second == nullptr)) {
        throw std::runtime_error(
            "sampled Boltzmann gross RHS vectors have incompatible sizes.");
    }
    const std::vector<std::vector<double>> rhs_replicates{
        rhs_first,
        rhs_second,
    };
    if (gross_first == nullptr) {
        return bp_sampled_replicated_batch_error(
            rhs_replicates,
            cell_volume,
            radius,
            w,
            eta,
            p);
    }
    const std::vector<std::vector<double>> gross_replicates{
        *gross_first,
        *gross_second,
    };
    return bp_sampled_replicated_batch_error(
        rhs_replicates,
        cell_volume,
        radius,
        w,
        eta,
        p,
        &gross_replicates);
}

// Append represented and omitted coefficients with one common scale while
// moving their stencils, preserving the graph's full activity bookkeeping.
inline void bp_sampled_move_scaled_graph(
    BPReversibleGraph& destination,
    BPReversibleGraph& source,
    double scale) {
    auto reserve_append = [](auto& values, std::size_t additional) {
        const std::size_t required = values.size() + additional;
        if (required <= values.capacity()) {
            return;
        }
        const std::size_t grown = values.capacity() > 0
            ? 2 * values.capacity()
            : std::size_t{8};
        values.reserve(std::max(required, grown));
    };
    reserve_append(destination.reactions, source.reactions.size());
    for (BPReversibleReaction& reaction : source.reactions) {
        reaction.coefficient *= scale;
        destination.reactions.push_back(std::move(reaction));
    }
    reserve_append(
        destination.omitted_domain_channels,
        source.omitted_domain_channels.size());
    for (BPReversibleOmittedChannel& channel :
         source.omitted_domain_channels) {
        channel.coefficient *= scale;
        destination.omitted_domain_channels.push_back(std::move(channel));
    }
    reserve_append(
        destination.omitted_projection_channels,
        source.omitted_projection_channels.size());
    for (BPReversibleOmittedChannel& channel :
         source.omitted_projection_channels) {
        channel.coefficient *= scale;
        destination.omitted_projection_channels.push_back(std::move(channel));
    }
    destination.maximum_projection_error = std::max(
        destination.maximum_projection_error,
        source.maximum_projection_error);
}

// Scale represented and omitted coefficients together to preserve the same
// estimator normalization and omission fraction.
inline void bp_sampled_scale_graph(
    BPReversibleGraph& graph,
    double scale) {
    for (BPReversibleReaction& reaction : graph.reactions) {
        reaction.coefficient *= scale;
    }
    for (BPReversibleOmittedChannel& channel :
         graph.omitted_domain_channels) {
        channel.coefficient *= scale;
    }
    for (BPReversibleOmittedChannel& channel :
         graph.omitted_projection_channels) {
        channel.coefficient *= scale;
    }
}

// Combine two already-normalized estimates in proportion to their draw
// counts, producing the mean over the complete accumulated sample range.
inline void bp_sampled_append_batch(
    BPSampledBatch& accumulated,
    BPSampledBatch& extension,
    std::uint64_t accumulated_samples,
    std::uint64_t extension_samples) {
    const std::uint64_t total_samples =
        accumulated_samples + extension_samples;
    if (total_samples == 0) {
        return;
    }
    const double accumulated_weight =
        static_cast<double>(accumulated_samples)
        / static_cast<double>(total_samples);
    const double extension_weight =
        static_cast<double>(extension_samples)
        / static_cast<double>(total_samples);
    bp_sampled_scale_graph(accumulated.graph, accumulated_weight);
    bp_sampled_move_scaled_graph(
        accumulated.graph, extension.graph, extension_weight);
    accumulated.accepted_samples += extension.accepted_samples;
    accumulated.attempted_draws += extension.attempted_draws;
    accumulated.retained_collision_draws +=
        extension.retained_collision_draws;
}

// Combine the deterministic block with independent complement replicates.
// Increase sample counts until RHS/heat criteria pass or the declared maximum
// is reached, then form their average graph for finite-time integration.
inline BPSampledShellGraph bp_sampled_build_shell_graph(
    const std::vector<double>& shell_f,
    double radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    int initial_samples = 0,
    std::uint64_t shell_salt = 0,
    const BPSampledProposal* prepared_proposal = nullptr) {
    validate_finite_nonnegative_distribution(
        shell_f,
        static_cast<std::size_t>(p.Nw * p.NL),
        "boltzmann_reversible_sampled graph input");
    validate_cell_average_eta_geometry(
        geometry, p, "boltzmann_reversible_sampled graph");
    BPSampledShellGraph result;
    BPSampledProposal local_proposal;
    if (prepared_proposal == nullptr) {
        local_proposal =
            bp_sampled_build_proposal(
                shell_f, radius, w, eta, geometry, p);
        prepared_proposal = &local_proposal;
    }
    const BPSampledProposal& proposal = *prepared_proposal;
    if (proposal.velocity_cells != p.Nw * p.NL
        || proposal.pair_phi_points
            != std::max(1, p.bp_boltzmann_pair_phi_points)) {
        throw std::runtime_error(
            "sampled Boltzmann prepared proposal does not match the velocity grid.");
    }
    std::vector<int> dense_cells =
        bp_sampled_dense_velocity_cells(
            proposal, radius, w, eta, p);
    result.dense_velocity_cells =
        static_cast<int>(dense_cells.size());
    const double complement_probability =
        bp_sampled_complement_probability(proposal, dense_cells);
    BPReversibleGraph dense_graph =
        bp_sampled_build_dense_graph(
            shell_f, dense_cells, radius, w, eta, geometry, p);
    result.dense_reactions = dense_graph.reactions.size();
    if (!(complement_probability > 0.0)) {
        result.graph = std::move(dense_graph);
        result.samples_per_batch = 0;
        return result;
    }
    std::vector<unsigned char> dense_mask(
        static_cast<std::size_t>(p.Nw * p.NL), 0);
    for (int q : dense_cells) {
        dense_mask[static_cast<std::size_t>(q)] = 1;
    }
    const BPSampledComplementProposal complement_proposal =
        bp_sampled_build_complement_proposal(
            proposal,
            dense_mask,
            complement_probability);
    std::vector<double> dense_gross_rhs;
    const std::vector<double> dense_rhs =
        bp_reversible_shell_rhs(
            shell_f, dense_graph, p, &dense_gross_rhs);
    int samples = (initial_samples > 0)
        ? initial_samples
        : p.bp_sampled_min_samples_per_batch;
    samples = std::max(
        p.bp_sampled_min_samples_per_batch,
        std::min(samples, p.bp_sampled_max_samples_per_batch));
    const std::uint64_t state_seed =
        bp_sampled_shell_hash(
            shell_f,
            radius,
            p.bp_sampled_seed ^ shell_salt);
    double ntc_activity_threshold =
        bp_sampled_ntc_activity_threshold(
            proposal,
            complement_proposal,
            complement_probability,
            radius,
            w,
            eta,
            p,
            state_seed);
    const int replicates = std::max(
        1, p.bp_sampled_replicates_per_estimate);
    std::vector<std::uint64_t> first_seeds;
    std::vector<std::uint64_t> second_seeds;
    std::vector<BPSampledBatch> batches_first;
    std::vector<BPSampledBatch> batches_second;
    first_seeds.reserve(static_cast<std::size_t>(replicates));
    second_seeds.reserve(static_cast<std::size_t>(replicates));
    batches_first.reserve(static_cast<std::size_t>(replicates));
    batches_second.reserve(static_cast<std::size_t>(replicates));
    for (int replicate = 0; replicate < replicates; ++replicate) {
        const std::uint64_t replicate_salt =
            (replicate == 0)
            ? UINT64_C(0)
            : bp_sampled_splitmix64(
                static_cast<std::uint64_t>(replicate));
        const std::uint64_t first_seed =
            bp_sampled_splitmix64(
                state_seed
                ^ UINT64_C(0x6a09e667f3bcc909)
                ^ replicate_salt);
        const std::uint64_t second_seed =
            bp_sampled_splitmix64(
                state_seed
                ^ UINT64_C(0xbb67ae8584caa73b)
                ^ replicate_salt);
        first_seeds.push_back(first_seed);
        second_seeds.push_back(second_seed);
    }
    auto rebuild_batches =
        [&](int batch_samples, double activity_threshold) {
            batches_first.clear();
            batches_second.clear();
            for (int replicate = 0; replicate < replicates; ++replicate) {
                batches_first.push_back(
                    bp_sampled_build_complement_batch(
                        shell_f,
                        proposal,
                        complement_proposal,
                        dense_mask,
                        complement_probability,
                        batch_samples,
                        first_seeds[static_cast<std::size_t>(replicate)],
                        0,
                        activity_threshold,
                        radius,
                        w,
                        eta,
                        geometry,
                        p));
                batches_second.push_back(
                    bp_sampled_build_complement_batch(
                        shell_f,
                        proposal,
                        complement_proposal,
                        dense_mask,
                        complement_probability,
                        batch_samples,
                        second_seeds[static_cast<std::size_t>(replicate)],
                        0,
                        activity_threshold,
                        radius,
                        w,
                        eta,
                        geometry,
                        p));
            }
        };
    const int starting_samples = samples;
    std::uint64_t discarded_attempted_draws = 0;
    std::uint64_t discarded_retained_draws = 0;
    rebuild_batches(samples, ntc_activity_threshold);
    auto append_replicate_rhs =
        [&](const std::vector<BPSampledBatch>& batches,
            std::vector<std::vector<double>>& rhs_replicates,
            std::vector<std::vector<double>>& gross_replicates) {
            for (const BPSampledBatch& batch : batches) {
                std::vector<double> complement_gross_rhs;
                const std::vector<double> complement_rhs =
                    bp_reversible_shell_rhs(
                        shell_f,
                        batch.graph,
                        p,
                        &complement_gross_rhs);
                rhs_replicates.push_back(
                    bp_sampled_add_rhs(
                        dense_rhs, complement_rhs));
                gross_replicates.push_back(
                    bp_sampled_add_rhs(
                        dense_gross_rhs, complement_gross_rhs));
            }
        };
    while (true) {
        std::vector<std::vector<double>> rhs_replicates;
        std::vector<std::vector<double>> gross_replicates;
        rhs_replicates.reserve(static_cast<std::size_t>(2 * replicates));
        gross_replicates.reserve(static_cast<std::size_t>(2 * replicates));
        append_replicate_rhs(
            batches_first, rhs_replicates, gross_replicates);
        append_replicate_rhs(
            batches_second, rhs_replicates, gross_replicates);
        const BPSampledBatchError error =
            bp_sampled_replicated_batch_error(
                rhs_replicates,
                dense_graph.velocity_cell_volume,
                radius,
                w,
                eta,
                p,
                &gross_replicates);
        bool observed_activity =
            dense_graph.represented_activity
                + dense_graph.omitted_domain_activity
                + dense_graph.omitted_projection_activity
            > 0.0;
        auto observe_batches =
            [&](const std::vector<BPSampledBatch>& batches) {
                for (const BPSampledBatch& batch : batches) {
                    observed_activity = observed_activity
                        || batch.graph.represented_activity
                                + batch.graph.omitted_domain_activity
                                + batch.graph.omitted_projection_activity
                            > 0.0;
                }
            };
        observe_batches(batches_first);
        observe_batches(batches_second);
        result.rhs_relative_error = error.rhs;
        result.rhs_gross_error = error.rhs_gross;
        result.heat_relative_error = error.heat;
        result.heat_gross_error = error.heat_gross;
        result.energy_invariant_error = error.energy_invariant;
        result.samples_per_batch = samples;
        result.converged =
            observed_activity
            && error.rhs <= p.bp_sampled_rhs_tolerance
            && error.heat <= p.bp_sampled_heat_tolerance;
        if (result.converged) {
            break;
        }
        if (samples >= p.bp_sampled_max_samples_per_batch) {
            if (p.bp_sampled_ntc_pair_thinning
                && ntc_activity_threshold > 0.0
                && !result.ntc_fallback) {
                for (const BPSampledBatch& batch : batches_first) {
                    discarded_attempted_draws += batch.attempted_draws;
                    discarded_retained_draws +=
                        batch.retained_collision_draws;
                }
                for (const BPSampledBatch& batch : batches_second) {
                    discarded_attempted_draws += batch.attempted_draws;
                    discarded_retained_draws +=
                        batch.retained_collision_draws;
                }
                // Retry the estimator without thinning after its first
                // maximum-sample failure; retain discarded work in diagnostics.
                ntc_activity_threshold = 0.0;
                result.ntc_fallback = true;
                samples = starting_samples;
                rebuild_batches(samples, ntc_activity_threshold);
                ++result.adaptive_rounds;
                continue;
            }
            break;
        }
        const int next_samples = std::min(
            p.bp_sampled_max_samples_per_batch, 2 * samples);
        const int extension_samples = next_samples - samples;
        for (int replicate = 0; replicate < replicates; ++replicate) {
            BPSampledBatch first_extension =
                bp_sampled_build_complement_batch(
                    shell_f,
                    proposal,
                    complement_proposal,
                    dense_mask,
                    complement_probability,
                    extension_samples,
                    first_seeds[static_cast<std::size_t>(replicate)],
                    batches_first[static_cast<std::size_t>(replicate)]
                        .attempted_draws,
                    ntc_activity_threshold,
                    radius,
                    w,
                    eta,
                    geometry,
                    p);
            BPSampledBatch second_extension =
                bp_sampled_build_complement_batch(
                    shell_f,
                    proposal,
                    complement_proposal,
                    dense_mask,
                    complement_probability,
                    extension_samples,
                    second_seeds[static_cast<std::size_t>(replicate)],
                    batches_second[static_cast<std::size_t>(replicate)]
                        .attempted_draws,
                    ntc_activity_threshold,
                    radius,
                    w,
                    eta,
                    geometry,
                    p);
            bp_sampled_append_batch(
                batches_first[static_cast<std::size_t>(replicate)],
                first_extension,
                static_cast<std::uint64_t>(samples),
                static_cast<std::uint64_t>(extension_samples));
            bp_sampled_append_batch(
                batches_second[static_cast<std::size_t>(replicate)],
                second_extension,
                static_cast<std::uint64_t>(samples),
                static_cast<std::uint64_t>(extension_samples));
        }
        samples = next_samples;
        ++result.adaptive_rounds;
    }
    if (!result.converged && p.bp_sampled_fail_on_unconverged) {
        std::ostringstream message;
        message
            << "boltzmann_reversible_sampled failed its two-estimate tolerance at r="
            << radius
            << ": rhs_error=" << result.rhs_relative_error
            << ", rhs_gross_error=" << result.rhs_gross_error
            << ", heat_error=" << result.heat_relative_error
            << ", heat_gross_error=" << result.heat_gross_error
            << ", samples_per_batch=" << result.samples_per_batch;
        throw std::runtime_error(message.str());
    }
    result.graph = std::move(dense_graph);
    result.complement_draws = discarded_attempted_draws;
    result.retained_collision_draws = discarded_retained_draws;
    const double graph_weight =
        0.5 / static_cast<double>(replicates);
    for (int replicate = 0; replicate < replicates; ++replicate) {
        BPSampledBatch& first =
            batches_first[static_cast<std::size_t>(replicate)];
        BPSampledBatch& second =
            batches_second[static_cast<std::size_t>(replicate)];
        result.complement_reactions +=
            first.graph.reactions.size()
            + second.graph.reactions.size();
        result.complement_draws +=
            first.attempted_draws + second.attempted_draws;
        result.retained_collision_draws +=
            first.retained_collision_draws
            + second.retained_collision_draws;
        bp_sampled_move_scaled_graph(
            result.graph, first.graph, graph_weight);
        bp_sampled_move_scaled_graph(
            result.graph, second.graph, graph_weight);
    }
    bp_sampled_finalize_graph_activity(result.graph, shell_f);
    return result;
}

// Bound sigma(g)*g over 0<=g<=g_max using endpoints and the physical law
// stationary point, then round the result outward for proposal-rate use.
inline double bp_sampled_u_sigma_upper_bound(
    double maximum_relative_speed,
    const Params& p) {
    const double umax = std::max(0.0, maximum_relative_speed);
    if (!(umax > 0.0)) {
        return 0.0;
    }
    auto evaluate = [&](double speed) {
        const double u = std::max(0.0, std::min(speed, umax));
        return u * bp_fa_total_cross_section(u, p);
    };
    double bound = std::max(evaluate(umax), evaluate(std::min(umax, 1.0e-14)));
    if (p.bp_cross_section == "constant") {
        return std::nextafter(bound, std::numeric_limits<double>::infinity());
    }
    if (p.bp_cross_section == "powerlaw") {
        if (p.bp_sigma_n > 1.0) {
            if (p.bp_sigma_u0 > 0.0) {
                const double critical =
                    p.bp_sigma_u0 / std::sqrt(p.bp_sigma_n - 1.0);
                bound = std::max(bound, evaluate(critical));
            } else {
                bound = std::max(bound, evaluate(std::min(umax, 1.0e-14)));
            }
        }
    } else if (p.bp_cross_section == "softened_gravity") {
        const double numerator =
            4.0 * PI * p.vp_gravity * p.vp_gravity
            * p.bp_subhalo_mass * p.bp_subhalo_mass
            * std::max(p.bp_coulomb_log, 0.0);
        const double sigma_saturation =
            (p.bp_sigma0 > 0.0)
            ? p.bp_sigma0
            : (2.0 * PI * p.bp_subhalo_radius * p.bp_subhalo_radius);
        if (numerator > 0.0 && sigma_saturation > 0.0) {
            const double critical =
                std::pow(
                    numerator / (3.0 * sigma_saturation),
                    0.25);
            bound = std::max(bound, evaluate(critical));
        }
    } else if (p.bp_cross_section == "yukawa_born") {
        if (p.bp_yukawa_mchi > 0.0 && p.bp_yukawa_mphi > 0.0) {
            bound = std::max(
                bound,
                evaluate(p.bp_yukawa_mphi / p.bp_yukawa_mchi));
        }
    } else {
        return std::numeric_limits<double>::infinity();
    }
    return std::nextafter(
        bound * (1.0 + 128.0 * std::numeric_limits<double>::epsilon()),
        std::numeric_limits<double>::infinity());
}

// Estimate shell work and spatial locality from actual velocity populations.
// A subset of positive pair activities supplies a lower bound; a sigma*g
// envelope supplies an upper bound for the inactive-activity budget.
inline BPSampledShellPlan bp_sampled_build_shell_plan(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    BPSampledShellPlan plan;
    plan.initial_samples.assign(
        static_cast<std::size_t>(p.Nr),
        p.bp_sampled_min_samples_per_batch);
    plan.activity_upper_bound.assign(
        static_cast<std::size_t>(p.Nr), 0.0);
    plan.mean_free_path.assign(
        static_cast<std::size_t>(p.Nr),
        std::numeric_limits<double>::infinity());
    plan.radial_cell_width_over_mean_free_path.assign(
        static_cast<std::size_t>(p.Nr), 0.0);
    plan.inactive.assign(static_cast<std::size_t>(p.Nr), 0);
    std::vector<double> activity(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> activity_lower(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> weighted_upper(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> weighted_lower(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> positive_activity;
    const int pair_phi_points =
        std::max(1, p.bp_boltzmann_pair_phi_points);
    std::vector<double> pair_phi_cos;
    std::vector<double> pair_phi_sin;
    bp_fa_midpoint_angle_table(
        pair_phi_points, pair_phi_cos, pair_phi_sin);
    for (int i = 0; i < p.Nr; ++i) {
        const double radius = r[static_cast<std::size_t>(i)];
        double number_density = 0.0;
        double speed_moment = 0.0;
        double radial_momentum = 0.0;
        double speed_squared_moment = 0.0;
        double maximum_speed = 0.0;
        std::vector<double> cell_population(
            static_cast<std::size_t>(p.Nw * p.NL), 0.0);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const int q = bp_reversible_velocity_index(j, k, p);
                const double cell_volume =
                    2.0 * PI
                    * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
                    * geometry.eta.width[static_cast<std::size_t>(k)]
                    / (radius * radius);
                const double population =
                    cell_volume * f[idx(i, j, k, p)];
                const double speed =
                    std::sqrt(std::max(
                        0.0,
                        2.0 * bp_reversible_energy(
                            q, radius, w, eta, p)));
                number_density += population;
                speed_moment += population * speed;
                radial_momentum +=
                    population * w[static_cast<std::size_t>(j)];
                speed_squared_moment += population * speed * speed;
                if (population > 0.0) {
                    maximum_speed = std::max(maximum_speed, speed);
                }
                cell_population[static_cast<std::size_t>(q)] = population;
            }
        }
        const double mean_speed =
            (number_density > 0.0)
            ? speed_moment / number_density
            : 0.0;
        const double mean_radial_velocity =
            (number_density > 0.0)
            ? radial_momentum / number_density
            : 0.0;
        // Subtract radial bulk motion before estimating the pair RMS speed.
        // The locality diagnostic uses lambda=1/[n*sigma(g_rms)] at this speed.
        const double peculiar_speed_squared =
            (number_density > 0.0)
            ? std::max(
                0.0,
                speed_squared_moment / number_density
                    - mean_radial_velocity * mean_radial_velocity)
            : 0.0;
        const double relative_rms_speed =
            std::sqrt(2.0 * peculiar_speed_squared);
        const double locality_cross_section =
            bp_fa_total_cross_section(
                std::max(relative_rms_speed, 1.0e-14), p);
        const double mean_free_path =
            (number_density > 0.0 && locality_cross_section > 0.0)
            ? 1.0 / (number_density * locality_cross_section)
            : std::numeric_limits<double>::infinity();
        plan.mean_free_path[static_cast<std::size_t>(i)] =
            mean_free_path;
        plan.radial_cell_width_over_mean_free_path[
            static_cast<std::size_t>(i)] =
            std::isfinite(mean_free_path) && mean_free_path > 0.0
            ? geometry.radial.width[static_cast<std::size_t>(i)]
                / mean_free_path
            : 0.0;
        activity[static_cast<std::size_t>(i)] =
            number_density * number_density * mean_speed
            * bp_fa_total_cross_section(mean_speed, p);
        if (activity[static_cast<std::size_t>(i)] > 0.0) {
            positive_activity.push_back(activity[static_cast<std::size_t>(i)]);
        }
        const double upper =
            0.25 * number_density * number_density
            * bp_sampled_u_sigma_upper_bound(2.0 * maximum_speed, p);
        plan.activity_upper_bound[static_cast<std::size_t>(i)] = upper;
        std::vector<int> population_order(
            static_cast<std::size_t>(p.Nw * p.NL), 0);
        std::iota(population_order.begin(), population_order.end(), 0);
        std::stable_sort(
            population_order.begin(),
            population_order.end(),
            [&](int first, int second) {
                return cell_population[static_cast<std::size_t>(first)]
                    > cell_population[static_cast<std::size_t>(second)];
            });
        const int lower_cells = std::min(
            static_cast<int>(population_order.size()),
            std::max(4, std::min(32, p.bp_sampled_dense_cells)));
        double lower = 0.0;
        for (int a = 0; a < lower_cells; ++a) {
            const int first =
                population_order[static_cast<std::size_t>(a)];
            const double first_population =
                cell_population[static_cast<std::size_t>(first)];
            if (!(first_population > 0.0)) {
                break;
            }
            for (int b = 0; b < lower_cells; ++b) {
                const int second =
                    population_order[static_cast<std::size_t>(b)];
                const double second_population =
                    cell_population[static_cast<std::size_t>(second)];
                if (!(second_population > 0.0)) {
                    break;
                }
                for (int pair_phi = 0;
                     pair_phi < pair_phi_points;
                     ++pair_phi) {
                    BPFAVec3 center_of_mass;
                    BPFAVec3 incoming_direction;
                    std::array<BPFAVec3, 2> basis;
                    double relative_speed = 0.0;
                    double total_cross_section = 0.0;
                    if (!bp_sampled_pair_kinematics(
                            first,
                            second,
                            pair_phi,
                            radius,
                            w,
                            eta,
                            pair_phi_cos,
                            pair_phi_sin,
                            p,
                            center_of_mass,
                            incoming_direction,
                            basis,
                            relative_speed,
                            total_cross_section)) {
                        continue;
                    }
                    lower +=
                        0.25 * first_population * second_population
                        * relative_speed * total_cross_section
                        / static_cast<double>(pair_phi_points);
                }
            }
        }
        activity_lower[static_cast<std::size_t>(i)] = lower;
        const double shell_volume =
            4.0 * PI
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.width[static_cast<std::size_t>(i)];
        weighted_upper[static_cast<std::size_t>(i)] =
            shell_volume * upper;
        weighted_lower[static_cast<std::size_t>(i)] =
            shell_volume * lower;
    }
    double median = 0.0;
    if (!positive_activity.empty()) {
        const std::size_t middle = positive_activity.size() / 2;
        std::nth_element(
            positive_activity.begin(),
            positive_activity.begin() + static_cast<std::ptrdiff_t>(middle),
            positive_activity.end());
        median = positive_activity[middle];
    }
    if (median > 0.0) {
        for (int i = 0; i < p.Nr; ++i) {
            const double raw_boost = std::pow(
                std::max(
                    1.0,
                    activity[static_cast<std::size_t>(i)] / median),
                0.25);
            const double boost = std::min(
                p.bp_sampled_activity_boost_max, raw_boost);
            int level = p.bp_sampled_min_samples_per_batch;
            while (level < p.bp_sampled_max_samples_per_batch
                   && static_cast<double>(level)
                        < boost * p.bp_sampled_min_samples_per_batch) {
                level = std::min(
                    p.bp_sampled_max_samples_per_batch, 2 * level);
            }
            plan.initial_samples[static_cast<std::size_t>(i)] = level;
        }
    }
    const double maximum_weighted_lower =
        *std::max_element(weighted_lower.begin(), weighted_lower.end());
    const double maximum_local_upper =
        *std::max_element(
            plan.activity_upper_bound.begin(),
            plan.activity_upper_bound.end());
    plan.weighted_activity_lower_bound =
        std::accumulate(weighted_lower.begin(), weighted_lower.end(), 0.0);
    const auto finalize_locality = [&]() {
        plan.minimum_mean_free_path =
            std::numeric_limits<double>::infinity();
        plan.maximum_radial_cell_width_over_mean_free_path = 0.0;
        plan.maximum_locality_shell = -1;
        plan.nonlocal_collision_cells = 0;
        for (int i = 0; i < p.Nr; ++i) {
            if (plan.inactive[static_cast<std::size_t>(i)] != 0) {
                continue;
            }
            const double mean_free_path =
                plan.mean_free_path[static_cast<std::size_t>(i)];
            const double ratio =
                plan.radial_cell_width_over_mean_free_path[
                    static_cast<std::size_t>(i)];
            plan.minimum_mean_free_path = std::min(
                plan.minimum_mean_free_path, mean_free_path);
            if (ratio
                > plan.maximum_radial_cell_width_over_mean_free_path) {
                plan.maximum_radial_cell_width_over_mean_free_path = ratio;
                plan.maximum_locality_shell = i;
            }
            plan.nonlocal_collision_cells +=
                ratio > p.bp_collision_locality_max_ratio ? 1 : 0;
        }
    };
    if (!(maximum_weighted_lower > 0.0)
        || !(maximum_local_upper > 0.0)
        || !(plan.weighted_activity_lower_bound > 0.0)) {
        finalize_locality();
        return plan;
    }
    std::vector<int> inactive_candidates;
    for (int i = 0; i < p.Nr; ++i) {
        if (std::isfinite(weighted_upper[static_cast<std::size_t>(i)])
            && plan.activity_upper_bound[static_cast<std::size_t>(i)]
                < p.bp_reversible_active_shell_floor_fraction
                    * maximum_local_upper
            && weighted_upper[static_cast<std::size_t>(i)]
                < p.bp_reversible_active_shell_floor_fraction
                    * maximum_weighted_lower) {
            inactive_candidates.push_back(i);
        }
    }
    std::stable_sort(
        inactive_candidates.begin(),
        inactive_candidates.end(),
        [&](int first, int second) {
            return weighted_upper[static_cast<std::size_t>(first)]
                < weighted_upper[static_cast<std::size_t>(second)];
        });
    // Inactive upper activity may consume only one tenth of the configured
    // omission fraction times a positive global activity lower bound.
    const double inactive_budget =
        0.1 * p.bp_boltzmann_remap_max_skipped_fraction
        * plan.weighted_activity_lower_bound;
    for (int i : inactive_candidates) {
        const double candidate =
            weighted_upper[static_cast<std::size_t>(i)];
        if (plan.weighted_inactive_upper_bound + candidate
            <= inactive_budget) {
            plan.inactive[static_cast<std::size_t>(i)] = 1;
            plan.weighted_inactive_upper_bound += candidate;
        }
    }
    finalize_locality();
    return plan;
}

// Expose the shell work plan's starting sample counts to other collision paths.
inline std::vector<int> bp_sampled_shell_initial_samples(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    return bp_sampled_build_shell_plan(
        f, r, w, eta, geometry, p).initial_samples;
}

// Sort finite shell collision times and omit complete fastest shells only
// while their cumulative mass fits the declared fraction.  The next shell
// sets the clock; all shell populations remain represented in the state.
inline double bp_sampled_mass_weighted_collision_timescale(
    const std::vector<double>& shell_mass,
    const std::vector<double>& shell_timescale,
    double ignored_mass_fraction) {
    if (shell_mass.size() != shell_timescale.size()
        || !(ignored_mass_fraction >= 0.0)
        || !(ignored_mass_fraction < 1.0)
        || !std::isfinite(ignored_mass_fraction)) {
        throw std::runtime_error(
            "Invalid sampled Boltzmann collision-timescale data.");
    }
    long double total_mass = 0.0L;
    std::vector<std::size_t> order;
    order.reserve(shell_mass.size());
    for (std::size_t i = 0; i < shell_mass.size(); ++i) {
        if (!(shell_mass[i] >= 0.0) || !std::isfinite(shell_mass[i])
            || !(shell_timescale[i] > 0.0)
            || std::isnan(shell_timescale[i])) {
            throw std::runtime_error(
                "Sampled Boltzmann collision timescale is nonphysical.");
        }
        total_mass += shell_mass[i];
        if (shell_mass[i] > 0.0 && std::isfinite(shell_timescale[i])) {
            order.push_back(i);
        }
    }
    if (!(total_mass > 0.0L) || order.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](std::size_t first, std::size_t second) {
            return shell_timescale[first] != shell_timescale[second]
                ? shell_timescale[first] < shell_timescale[second]
                : first < second;
        });
    const long double ignored_budget = ignored_mass_fraction * total_mass;
    long double ignored_mass = 0.0L;
    for (std::size_t i : order) {
        const long double candidate = shell_mass[i];
        if (ignored_mass + candidate <= ignored_budget) {
            ignored_mass += candidate;
            continue;
        }
        return shell_timescale[i];
    }
    return shell_timescale[order.back()];
}

/// Applies the adaptive sampled reversible operator independently in each shell.
inline BPCollisionStepInfo apply_bp_sampled_reversible_boltzmann_collision_step(
    std::vector<double>& f,
    std::vector<double>* tracer_f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    double dt,
    const Params& p,
    const MpiContext& mpi) {
    BPCollisionStepInfo info;
    if (!using_bp_sampled_reversible_boltzmann(p)
        || !(dt > 0.0)
        || p.bp_collision_strength == 0.0) {
        return info;
    }
    if (mpi.size != 1) {
        throw std::runtime_error(
            "boltzmann_reversible_sampled is OpenMP-only and never uses MPI.");
    }
    if (tracer_f != nullptr && !tracer_f->empty()) {
        throw std::runtime_error(
            "boltzmann_reversible_sampled passive-tracer collisions are not implemented.");
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
        "boltzmann_reversible_sampled");
    validate_finite_nonnegative_distribution(
        f,
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL,
        "boltzmann_reversible_sampled input");
    const std::vector<double> eta = geometry.eta.center;
    // Check radial width/mean-free-path locality before constructing any
    // collision update; inactive activity has its own conservative budget.
    const BPSampledShellPlan shell_plan =
        bp_sampled_build_shell_plan(f, r, w, eta, geometry, p);
    info.minimum_mean_free_path = shell_plan.minimum_mean_free_path;
    info.maximum_radial_cell_width_over_mean_free_path =
        shell_plan.maximum_radial_cell_width_over_mean_free_path;
    info.nonlocal_collision_cells =
        shell_plan.nonlocal_collision_cells;
    if (p.bp_collision_fail_on_nonlocal_cells
        && shell_plan.nonlocal_collision_cells > 0) {
        const int i = shell_plan.maximum_locality_shell;
        std::ostringstream message;
        message
            << "boltzmann_reversible_sampled radial collision cell is "
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
        message
            << ". Refine the radial grid before continuing.";
        throw std::runtime_error(message.str());
    }
    std::vector<BPSampledShellResult> shell_result(
        static_cast<std::size_t>(p.Nr));
    const int available_threads = effective_threads(p, p.Nr);
    Params shell_p = p;
    shell_p.threads = available_threads;
    std::vector<std::vector<double>> shell_input(
        static_cast<std::size_t>(p.Nr));
    std::vector<BPSampledProposal> shell_proposal(
        static_cast<std::size_t>(p.Nr));
    parallel_for_dynamic_chunks(
        0,
        p.Nr,
        available_threads,
        1,
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                std::vector<double>& shell_f =
                    shell_input[static_cast<std::size_t>(i)];
                shell_f.assign(
                    static_cast<std::size_t>(p.Nw * p.NL), 0.0);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        shell_f[static_cast<std::size_t>(
                            bp_reversible_velocity_index(j, k, p))] =
                            f[idx(i, j, k, p)];
                    }
                }
                if (!shell_plan.inactive[static_cast<std::size_t>(i)]) {
                    shell_proposal[static_cast<std::size_t>(i)] =
                        bp_sampled_build_proposal(
                            shell_f,
                            r[static_cast<std::size_t>(i)],
                            w,
                            eta,
                            geometry,
                            p);
                }
            }
        });
    parallel_for_dynamic_chunks(
        0,
        p.Nr,
        available_threads,
        1,
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                std::vector<double>& shell_f =
                    shell_input[static_cast<std::size_t>(i)];
                BPSampledShellResult& shell =
                    shell_result[static_cast<std::size_t>(i)];
                if (shell_plan.inactive[static_cast<std::size_t>(i)]) {
                    shell.step.f = std::move(shell_f);
                    shell.step.omitted_inactive_activity =
                        shell_plan.activity_upper_bound[
                            static_cast<std::size_t>(i)];
                    shell.step.omitted_fraction = 1.0;
                    shell.sampling.inactive = true;
                    continue;
                }
                shell.sampling = bp_sampled_build_shell_graph(
                    shell_f,
                    r[static_cast<std::size_t>(i)],
                    w,
                    eta,
                    geometry,
                    shell_p,
                    shell_plan.initial_samples[static_cast<std::size_t>(i)],
                    static_cast<std::uint64_t>(i),
                    &shell_proposal[static_cast<std::size_t>(i)]);
                shell_proposal[static_cast<std::size_t>(i)] =
                    BPSampledProposal{};
                try {
                    shell.step = bp_reversible_advance_shell(
                        shell_f, shell.sampling.graph, dt, shell_p);
                } catch (const std::exception& error) {
                    std::ostringstream message;
                    message
                        << "boltzmann_reversible_sampled failed at shell "
                        << i << ", r="
                        << r[static_cast<std::size_t>(i)]
                        << ": " << error.what();
                    throw std::runtime_error(message.str());
                }
                shell.sampling.graph = BPReversibleGraph{};
            }
        });
    // Shell results remain private until the global and active-shell omitted
    // activity fractions pass, then their distributions are published to f.
    bp_reversible_validate_omission_budgets(
        p.Nr,
        geometry,
        p,
        [&](int i) -> const BPReversibleShellResult& {
            return shell_result[static_cast<std::size_t>(i)].step;
        });
    double minimum_dt = std::numeric_limits<double>::infinity();
    std::vector<double> shell_mass(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> shell_collision_timescale(
        static_cast<std::size_t>(p.Nr),
        std::numeric_limits<double>::infinity());
    info.sampled_min_samples_per_batch = std::numeric_limits<int>::max();
    for (int i = 0; i < p.Nr; ++i) {
        const BPSampledShellResult& shell =
            shell_result[static_cast<std::size_t>(i)];
        long double number_density = 0.0L;
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t q = static_cast<std::size_t>(
                    bp_reversible_velocity_index(j, k, p));
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
        if (number_density > 0.0L && collision_activity > 0.0) {
            shell_collision_timescale[static_cast<std::size_t>(i)] =
                static_cast<double>(number_density)
                / (2.0 * collision_activity);
        }
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                f[idx(i, j, k, p)] =
                    shell.step.f[static_cast<std::size_t>(
                        bp_reversible_velocity_index(j, k, p))];
            }
        }
        info.substeps = std::max(info.substeps, shell.step.substeps);
        info.explicit_substeps =
            std::max(info.explicit_substeps, shell.step.substeps);
        info.reversible_reaction_updates += shell.step.reaction_updates;
        info.reversible_explicit_fast_path_fallbacks +=
            static_cast<std::uint64_t>(
                shell.step.explicit_fast_path_fallbacks);
        info.reversible_implicit_reaction_updates +=
            shell.step.implicit_reaction_updates;
        info.reversible_implicit_iterations +=
            shell.step.implicit_iterations;
        info.max_shell_remap_skipped_fraction = std::max(
            info.max_shell_remap_skipped_fraction,
            shell.step.omitted_fraction);
        const double shell_volume =
            4.0 * PI
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.width[static_cast<std::size_t>(i)];
        info.remap_attempted_event_density +=
            shell_volume
            * (shell.step.represented_activity
               + shell.step.omitted_domain_activity
               + shell.step.omitted_projection_activity
               + shell.step.omitted_inactive_activity);
        info.remap_skipped_event_density +=
            shell_volume
            * (shell.step.omitted_domain_activity
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
        info.sampled_dense_reactions += shell.sampling.dense_reactions;
        info.sampled_complement_reactions +=
            shell.sampling.complement_reactions;
        info.sampled_complement_draws += shell.sampling.complement_draws;
        info.sampled_retained_collision_draws +=
            shell.sampling.retained_collision_draws;
        info.sampled_ntc_fallback_shells +=
            shell.sampling.ntc_fallback ? 1 : 0;
        if (shell.sampling.samples_per_batch > 0) {
            info.sampled_min_samples_per_batch = std::min(
                info.sampled_min_samples_per_batch,
                shell.sampling.samples_per_batch);
        }
        info.sampled_max_samples_per_batch = std::max(
            info.sampled_max_samples_per_batch,
            shell.sampling.samples_per_batch);
        info.sampled_max_adaptive_rounds = std::max(
            info.sampled_max_adaptive_rounds,
            shell.sampling.adaptive_rounds);
        info.sampled_unconverged_shells += shell.sampling.converged ? 0 : 1;
        info.sampled_inactive_shells += shell.sampling.inactive ? 1 : 0;
        info.sampled_max_rhs_relative_error = std::max(
            info.sampled_max_rhs_relative_error,
            shell.sampling.rhs_relative_error);
        info.sampled_max_rhs_gross_error = std::max(
            info.sampled_max_rhs_gross_error,
            shell.sampling.rhs_gross_error);
        info.sampled_max_heat_relative_error = std::max(
            info.sampled_max_heat_relative_error,
            shell.sampling.heat_relative_error);
        info.sampled_max_heat_gross_error = std::max(
            info.sampled_max_heat_gross_error,
            shell.sampling.heat_gross_error);
        info.sampled_max_energy_invariant_error = std::max(
            info.sampled_max_energy_invariant_error,
            shell.sampling.energy_invariant_error);
        minimum_dt = std::min(
            minimum_dt, shell.step.minimum_dt_limit);
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
