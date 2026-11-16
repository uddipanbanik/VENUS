#pragma once

// Evaluate local elastic Boltzmann collisions on a spherical (w,eta) velocity
// grid, with eta=L^2/2 and tangential speed sqrt(2*eta)/r.  Inputs are finite
// nonnegative cell distributions, their velocity volumes and angular quadrature.
// Outputs are updated distributions and positivity, entropy and omission ledgers.
//
// Each continuous outgoing pair is represented by positive grid-pair weights
// that reproduce pair number, radial momentum and kinetic energy to the stated
// projection tolerance.  These weights also define a geometric reverse activity.
// Its logarithm is a weighted sum of log(f), so conserved Maxwellian exponents
// give equal forward/reverse activities in the resolved entropy-floor regime.
// Reaction extents change all participating cells together.  Their common
// admissible interval enforces positivity; shell acceptance also checks the
// discrete H functional, integral f*(log(f)-1) d^3v.

#include "ev/finite_angle_boltzmann.hpp"
#include "ev/kinetic_invariant_ledger.hpp"

#include <map>
#include <numeric>

namespace ev {

// A product stencil is a convex mixture of whole pairs; each weight creates
// one unit of population in each of its two velocity cells.
struct BPReversibleProductPair {
    int first = 0;
    int second = 0;
    double weight = 0.0;
};

// product_order is the exponent in the geometric reverse activity.  The net
// change subtracts incoming multiplicity from that product population.
struct BPReversibleSpecies {
    int index = 0;
    double product_order = 0.0;
    double net_population_change = 0.0;
};

struct BPReversibleReaction {
    int incoming_first = 0;
    int incoming_second = 0;
    double coefficient = 0.0;
    std::vector<BPReversibleSpecies> species;
};

struct BPReversibleOmittedChannel {
    int incoming_first = 0;
    int incoming_second = 0;
    double coefficient = 0.0;
    unsigned domain_mask = 0;
};

enum BPReversibleDomainMask : unsigned {
    BP_REVERSIBLE_DOMAIN_NONE = 0U,
    BP_REVERSIBLE_DOMAIN_RADIAL_VELOCITY = 1U,
    BP_REVERSIBLE_DOMAIN_ANGULAR_MOMENTUM = 2U,
};

struct BPReversibleProjection {
    std::vector<BPReversibleProductPair> products;
    double mass_error = 0.0;
    double momentum_error = 0.0;
    double energy_error = 0.0;
};

// The graph stores geometry and rate coefficients separately from f, allowing
// repeated time integration with current forward/reverse populations.
struct BPReversibleGraph {
    std::vector<BPReversibleReaction> reactions;
    std::vector<BPReversibleOmittedChannel> omitted_domain_channels;
    std::vector<BPReversibleOmittedChannel> omitted_projection_channels;
    std::vector<double> velocity_cell_volume;
    double represented_activity = 0.0;
    double omitted_activity = 0.0;
    double omitted_domain_activity = 0.0;
    double omitted_projection_activity = 0.0;
    double omitted_inactive_activity = 0.0;
    double maximum_projection_error = 0.0;
};

struct BPReversibleShellResult {
    std::vector<double> f;
    int substeps = 0;
    int entropy_rejections = 0;
    int explicit_fast_path_fallbacks = 0;
    std::uint64_t reaction_updates = 0;
    std::uint64_t implicit_reaction_updates = 0;
    std::uint64_t implicit_iterations = 0;
    double minimum_dt_limit = std::numeric_limits<double>::infinity();
    double omitted_fraction = 0.0;
    double maximum_projection_error = 0.0;
    double represented_activity = 0.0;
    double omitted_domain_activity = 0.0;
    double omitted_projection_activity = 0.0;
    double omitted_inactive_activity = 0.0;
};

// Flatten only the velocity plane within one radial shell.
inline int bp_reversible_velocity_index(int j, int k, const Params& p) {
    return j * p.NL + k;
}

inline std::array<int, 2> bp_reversible_velocity_indices(int q, const Params& p) {
    return {q / p.NL, q % p.NL};
}

// Specific kinetic energy includes the tangential contribution eta/r^2.
inline double bp_reversible_energy(
    int velocity_index,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const Params& p) {
    const std::array<int, 2> indices =
        bp_reversible_velocity_indices(velocity_index, p);
    const double wr = w[static_cast<std::size_t>(indices[0])];
    return 0.5 * wr * wr
        + eta[static_cast<std::size_t>(indices[1])] / (ri * ri);
}

inline double bp_reversible_radial_velocity(
    int velocity_index,
    const std::vector<double>& w,
    const Params& p) {
    return w[static_cast<std::size_t>(velocity_index / p.NL)];
}

inline bool bp_reversible_solve_3x3(
    double matrix[3][3],
    double rhs[3],
    double solution[3]) {
    return bp_fa_solve_3x3(matrix, rhs, solution);
}

// Admit points inside the physical velocity faces.  Near a boundary, the
// outermost cell center supplies the initial candidate for joint moment repair.
inline bool bp_reversible_center_bracket(
    const std::vector<double>& centers,
    const std::vector<double>& edges,
    double value,
    int& lower,
    int& upper,
    double& fraction) {
    if (centers.size() < 2 || edges.size() != centers.size() + 1) {
        return false;
    }
    const double lower_face = edges.front();
    const double upper_face = edges.back();
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(lower_face), std::abs(upper_face), std::abs(value)});
    if (value < lower_face - tolerance || value > upper_face + tolerance) {
        return false;
    }
    if (value <= centers.front()) {
        lower = 0;
        upper = 0;
        fraction = 0.0;
        return true;
    }
    if (value >= centers.back()) {
        lower = static_cast<int>(centers.size()) - 1;
        upper = lower;
        fraction = 0.0;
        return true;
    }
    const auto it = std::upper_bound(centers.begin(), centers.end(), value);
    upper = static_cast<int>(std::distance(centers.begin(), it));
    lower = upper - 1;
    fraction = (value - centers[static_cast<std::size_t>(lower)])
        / (centers[static_cast<std::size_t>(upper)]
           - centers[static_cast<std::size_t>(lower)]);
    return true;
}

struct BPReversibleNodeCandidate {
    int index = 0;
    double base_weight = 0.0;
};

// Start from bilinear interpolation in w and eta, then add neighboring centers
// so a joint pair stencil can satisfy momentum and energy simultaneously.
inline bool bp_reversible_node_candidates(
    double wr,
    double eta_value,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const std::vector<double>& w_edge,
    const std::vector<double>& eta_edge,
    const Params& p,
    int candidate_radius,
    std::vector<BPReversibleNodeCandidate>& candidates,
    unsigned* outside_domain_mask = nullptr) {
    if (outside_domain_mask != nullptr) {
        *outside_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
    }
    int j0 = 0;
    int j1 = 0;
    int k0 = 0;
    int k1 = 0;
    double tw = 0.0;
    double te = 0.0;
    const bool radial_velocity_inside =
        bp_reversible_center_bracket(w, w_edge, wr, j0, j1, tw);
    const bool angular_momentum_inside =
        bp_reversible_center_bracket(
            eta, eta_edge, eta_value, k0, k1, te);
    if (!radial_velocity_inside || !angular_momentum_inside) {
        if (outside_domain_mask != nullptr) {
            if (!radial_velocity_inside) {
                *outside_domain_mask |=
                    BP_REVERSIBLE_DOMAIN_RADIAL_VELOCITY;
            }
            if (!angular_momentum_inside) {
                *outside_domain_mask |=
                    BP_REVERSIBLE_DOMAIN_ANGULAR_MOMENTUM;
            }
        }
        return false;
    }
    const int radius = std::max(1, candidate_radius);
    const int j_begin = std::max(0, j0 - radius);
    const int j_end = std::min(p.Nw - 1, j1 + radius);
    const int k_begin = std::max(0, k0 - radius);
    const int k_end = std::min(p.NL - 1, k1 + radius);
    std::map<int, double> base;
    const int js[2] = {j0, j1};
    const int ks[2] = {k0, k1};
    const double jw[2] = {
        (j0 == j1) ? 1.0 : (1.0 - tw),
        (j0 == j1) ? 0.0 : tw,
    };
    const double kw[2] = {
        (k0 == k1) ? 1.0 : (1.0 - te),
        (k0 == k1) ? 0.0 : te,
    };
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            if (jw[a] * kw[b] > 0.0) {
                base[bp_reversible_velocity_index(js[a], ks[b], p)] += jw[a] * kw[b];
            }
        }
    }
    candidates.clear();
    for (int j = j_begin; j <= j_end; ++j) {
        for (int k = k_begin; k <= k_end; ++k) {
            const int index = bp_reversible_velocity_index(j, k, p);
            candidates.push_back(BPReversibleNodeCandidate{index, base[index]});
        }
    }
    return true;
}

struct BPReversiblePairCandidate {
    int first = 0;
    int second = 0;
    double base_weight = 0.0;
    double momentum = 0.0;
    double energy = 0.0;
};

// Correct the base pair weights under three linear constraints: unit weight,
// total radial velocity and total kinetic energy.  An active-set solve fixes
// the most negative weight to zero and repeats over the remaining candidates.
inline bool bp_reversible_solve_projection_subset(
    const std::vector<BPReversiblePairCandidate>& pairs,
    const std::vector<std::size_t>& selected,
    double target_momentum,
    double target_energy,
    double momentum_scale,
    double energy_scale,
    double tolerance,
    std::vector<double>& alpha) {
    if (selected.size() < 3) {
        return false;
    }
    std::vector<unsigned char> free(selected.size(), 1);
    std::vector<double> local_alpha(selected.size(), 0.0);
    bool solved = false;
    for (std::size_t active_iteration = 0;
         active_iteration < selected.size();
         ++active_iteration) {
        double gram[3][3] = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
        };
        double base_constraint[3] = {0.0, 0.0, 0.0};
        int free_count = 0;
        for (std::size_t local = 0; local < selected.size(); ++local) {
            if (!free[local]) {
                continue;
            }
            ++free_count;
            const BPReversiblePairCandidate& pair =
                pairs[selected[local]];
            const double column[3] = {
                1.0,
                pair.momentum / momentum_scale,
                pair.energy / energy_scale,
            };
            for (int a = 0; a < 3; ++a) {
                base_constraint[a] += pair.base_weight * column[a];
                for (int b = 0; b < 3; ++b) {
                    gram[a][b] += column[a] * column[b];
                }
            }
        }
        if (free_count < 3) {
            break;
        }
        double rhs[3] = {
            1.0 - base_constraint[0],
            target_momentum / momentum_scale - base_constraint[1],
            target_energy / energy_scale - base_constraint[2],
        };
        double lambda[3] = {0.0, 0.0, 0.0};
        if (!bp_reversible_solve_3x3(gram, rhs, lambda)) {
            break;
        }
        double most_negative = 0.0;
        std::size_t most_negative_local = selected.size();
        for (std::size_t local = 0; local < selected.size(); ++local) {
            if (!free[local]) {
                local_alpha[local] = 0.0;
                continue;
            }
            const BPReversiblePairCandidate& pair =
                pairs[selected[local]];
            const double column[3] = {
                1.0,
                pair.momentum / momentum_scale,
                pair.energy / energy_scale,
            };
            local_alpha[local] = pair.base_weight
                + column[0] * lambda[0]
                + column[1] * lambda[1]
                + column[2] * lambda[2];
            if (local_alpha[local] < most_negative) {
                most_negative = local_alpha[local];
                most_negative_local = local;
            }
        }
        if (most_negative_local == selected.size()
            || most_negative >= -tolerance) {
            solved = true;
            break;
        }
        free[most_negative_local] = 0;
        local_alpha[most_negative_local] = 0.0;
    }
    if (!solved) {
        return false;
    }
    alpha.assign(pairs.size(), 0.0);
    double represented_mass = 0.0;
    double represented_momentum = 0.0;
    double represented_energy = 0.0;
    for (std::size_t local = 0; local < selected.size(); ++local) {
        if (local_alpha[local] > tolerance) {
            const std::size_t q = selected[local];
            alpha[q] = local_alpha[local];
            represented_mass += local_alpha[local];
            represented_momentum +=
                local_alpha[local] * pairs[q].momentum;
            represented_energy +=
                local_alpha[local] * pairs[q].energy;
        }
    }
    const double normalized_error = std::max({
        std::abs(represented_mass - 1.0),
        std::abs(represented_momentum - target_momentum) / momentum_scale,
        std::abs(represented_energy - target_energy) / energy_scale,
    });
    return normalized_error <= 32.0 * tolerance;
}

struct BPReversibleHullPoint {
    double momentum = 0.0;
    double energy = 0.0;
    std::size_t pair_index = 0;
};

inline double bp_reversible_hull_cross(
    const BPReversibleHullPoint& first,
    const BPReversibleHullPoint& second,
    const BPReversibleHullPoint& third) {
    return (second.momentum - first.momentum)
            * (third.energy - first.energy)
        - (second.energy - first.energy)
            * (third.momentum - first.momentum);
}

/// Finds a nonnegative three-point moment stencil when the local active-set
/// projection needs a wider fallback.  In the two-dimensional (P,E) moment
/// plane, every point in a convex hull has a representation using at most
/// three vertices.
inline bool bp_reversible_solve_projection_hull(
    const std::vector<BPReversiblePairCandidate>& pairs,
    double target_momentum,
    double target_energy,
    double momentum_scale,
    double energy_scale,
    double tolerance,
    std::vector<double>& alpha,
    const std::vector<std::size_t>* selected = nullptr) {
    if (pairs.empty() || (selected != nullptr && selected->empty())) {
        return false;
    }
    std::vector<BPReversibleHullPoint> points;
    points.reserve(selected == nullptr ? pairs.size() : selected->size());
    auto append_point = [&](std::size_t q) {
        if (q >= pairs.size()) {
            throw std::runtime_error(
                "boltzmann_reversible hull subset index is out of range.");
        }
        points.push_back(BPReversibleHullPoint{
            pairs[q].momentum / momentum_scale,
            pairs[q].energy / energy_scale,
            q,
        });
    };
    if (selected == nullptr) {
        for (std::size_t q = 0; q < pairs.size(); ++q) {
            append_point(q);
        }
    } else {
        for (std::size_t q : *selected) {
            append_point(q);
        }
    }
    std::stable_sort(
        points.begin(),
        points.end(),
        [&](const BPReversibleHullPoint& first,
            const BPReversibleHullPoint& second) {
            if (first.momentum != second.momentum) {
                return first.momentum < second.momentum;
            }
            if (first.energy != second.energy) {
                return first.energy < second.energy;
            }
            const BPReversiblePairCandidate& first_pair =
                pairs[first.pair_index];
            const BPReversiblePairCandidate& second_pair =
                pairs[second.pair_index];
            if (first_pair.base_weight != second_pair.base_weight) {
                return first_pair.base_weight > second_pair.base_weight;
            }
            if (first_pair.first != second_pair.first) {
                return first_pair.first < second_pair.first;
            }
            return first_pair.second < second_pair.second;
        });
    points.erase(
        std::unique(
            points.begin(),
            points.end(),
            [](const BPReversibleHullPoint& first,
               const BPReversibleHullPoint& second) {
                return first.momentum == second.momentum
                    && first.energy == second.energy;
            }),
        points.end());

    double coordinate_scale = 1.0;
    for (const BPReversibleHullPoint& point : points) {
        coordinate_scale = std::max({
            coordinate_scale,
            std::abs(point.momentum),
            std::abs(point.energy),
        });
    }
    const double target_p = target_momentum / momentum_scale;
    const double target_e = target_energy / energy_scale;
    coordinate_scale = std::max({
        coordinate_scale,
        std::abs(target_p),
        std::abs(target_e),
    });
    const double cross_tolerance =
        512.0 * std::numeric_limits<double>::epsilon()
        * coordinate_scale * coordinate_scale;

    std::vector<BPReversibleHullPoint> lower;
    std::vector<BPReversibleHullPoint> upper;
    lower.reserve(points.size());
    upper.reserve(points.size());
    for (const BPReversibleHullPoint& point : points) {
        while (lower.size() >= 2
               && bp_reversible_hull_cross(
                      lower[lower.size() - 2], lower.back(), point)
                      <= cross_tolerance) {
            lower.pop_back();
        }
        lower.push_back(point);
    }
    for (auto point = points.rbegin(); point != points.rend(); ++point) {
        while (upper.size() >= 2
               && bp_reversible_hull_cross(
                      upper[upper.size() - 2], upper.back(), *point)
                      <= cross_tolerance) {
            upper.pop_back();
        }
        upper.push_back(*point);
    }
    std::vector<BPReversibleHullPoint> hull;
    if (!lower.empty()) {
        lower.pop_back();
    }
    if (!upper.empty()) {
        upper.pop_back();
    }
    hull.reserve(lower.size() + upper.size());
    hull.insert(hull.end(), lower.begin(), lower.end());
    hull.insert(hull.end(), upper.begin(), upper.end());
    if (hull.empty()) {
        hull.push_back(points.front());
    }

    alpha.assign(pairs.size(), 0.0);
    auto accept_weights = [&](const std::array<std::size_t, 3>& indices,
                              std::array<double, 3> weights) {
        const double weight_tolerance = std::max(
            32.0 * tolerance,
            2048.0 * std::numeric_limits<double>::epsilon());
        for (double weight : weights) {
            if (weight < -weight_tolerance || !std::isfinite(weight)) {
                return false;
            }
        }
        double sum = 0.0;
        for (double& weight : weights) {
            weight = std::max(0.0, weight);
            sum += weight;
        }
        if (!(sum > 0.0) || !std::isfinite(sum)) {
            return false;
        }
        std::fill(alpha.begin(), alpha.end(), 0.0);
        for (std::size_t q = 0; q < weights.size(); ++q) {
            alpha[indices[q]] += weights[q] / sum;
        }
        double represented_mass = 0.0;
        double represented_momentum = 0.0;
        double represented_energy = 0.0;
        for (std::size_t q = 0; q < alpha.size(); ++q) {
            if (!(alpha[q] > 0.0)) {
                continue;
            }
            represented_mass += alpha[q];
            represented_momentum += alpha[q] * pairs[q].momentum;
            represented_energy += alpha[q] * pairs[q].energy;
        }
        const double normalized_error = std::max({
            std::abs(represented_mass - 1.0),
            std::abs(represented_momentum - target_momentum)
                / momentum_scale,
            std::abs(represented_energy - target_energy) / energy_scale,
        });
        return normalized_error <= 32.0 * tolerance;
    };

    if (hull.size() == 1) {
        return accept_weights(
            {hull[0].pair_index, hull[0].pair_index, hull[0].pair_index},
            {1.0, 0.0, 0.0});
    }
    if (hull.size() == 2) {
        const double dp = hull[1].momentum - hull[0].momentum;
        const double de = hull[1].energy - hull[0].energy;
        const double fraction =
            (std::abs(dp) >= std::abs(de) && dp != 0.0)
            ? (target_p - hull[0].momentum) / dp
            : (de != 0.0)
                ? (target_e - hull[0].energy) / de
                : 0.0;
        return accept_weights(
            {hull[0].pair_index, hull[1].pair_index, hull[1].pair_index},
            {1.0 - fraction, fraction, 0.0});
    }

    const BPReversibleHullPoint target{target_p, target_e, 0};
    for (std::size_t q = 1; q + 1 < hull.size(); ++q) {
        const BPReversibleHullPoint& first = hull[0];
        const BPReversibleHullPoint& second = hull[q];
        const BPReversibleHullPoint& third = hull[q + 1];
        const double denominator =
            bp_reversible_hull_cross(first, second, third);
        if (!(denominator > cross_tolerance)) {
            continue;
        }
        const double first_weight =
            bp_reversible_hull_cross(target, second, third)
            / denominator;
        const double second_weight =
            bp_reversible_hull_cross(target, third, first)
            / denominator;
        const double third_weight =
            bp_reversible_hull_cross(target, first, second)
            / denominator;
        if (accept_weights(
                {first.pair_index, second.pair_index, third.pair_index},
                {first_weight, second_weight, third_weight})) {
            return true;
        }
    }
    return false;
}

// Form candidate whole pairs at the shell radius.  The feasible target lies
// in their momentum-energy convex hull; widen the support if the local solve
// needs additional positive weights, then check all three represented moments.
inline bool bp_reversible_project_outgoing_pair_at_radius(
    const BPFAVec3& first_velocity,
    const BPFAVec3& second_velocity,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    int candidate_radius,
    BPReversibleProjection& projection,
    bool* outside_domain = nullptr,
    unsigned* outside_domain_mask = nullptr) {
    if (outside_domain != nullptr) {
        *outside_domain = false;
    }
    if (outside_domain_mask != nullptr) {
        *outside_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
    }
    const double eta_first =
        0.5 * ri * ri
        * std::max(0.0, first_velocity.y * first_velocity.y
                           + first_velocity.z * first_velocity.z);
    const double eta_second =
        0.5 * ri * ri
        * std::max(0.0, second_velocity.y * second_velocity.y
                           + second_velocity.z * second_velocity.z);
    std::vector<BPReversibleNodeCandidate> first_nodes;
    std::vector<BPReversibleNodeCandidate> second_nodes;
    unsigned first_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
    unsigned second_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
    const bool first_inside = bp_reversible_node_candidates(
            first_velocity.x,
            eta_first,
            w,
            eta,
            geometry.radial_velocity.edge,
            geometry.eta.edge,
            p,
            candidate_radius,
            first_nodes,
            &first_domain_mask);
    const bool second_inside = bp_reversible_node_candidates(
            second_velocity.x,
            eta_second,
            w,
            eta,
            geometry.radial_velocity.edge,
            geometry.eta.edge,
            p,
            candidate_radius,
            second_nodes,
            &second_domain_mask);
    if (!first_inside || !second_inside) {
        if (outside_domain != nullptr) {
            *outside_domain = true;
        }
        if (outside_domain_mask != nullptr) {
            *outside_domain_mask = first_domain_mask | second_domain_mask;
        }
        return false;
    }
    std::vector<BPReversiblePairCandidate> pairs;
    pairs.reserve(first_nodes.size() * second_nodes.size());
    for (const BPReversibleNodeCandidate& first : first_nodes) {
        for (const BPReversibleNodeCandidate& second : second_nodes) {
            BPReversiblePairCandidate pair;
            pair.first = first.index;
            pair.second = second.index;
            pair.base_weight = first.base_weight * second.base_weight;
            pair.momentum =
                bp_reversible_radial_velocity(pair.first, w, p)
                + bp_reversible_radial_velocity(pair.second, w, p);
            pair.energy =
                bp_reversible_energy(pair.first, ri, w, eta, p)
                + bp_reversible_energy(pair.second, ri, w, eta, p);
            pairs.push_back(pair);
        }
    }
    const double target_momentum = first_velocity.x + second_velocity.x;
    const double target_energy =
        0.5 * (
            bp_fa_dot(first_velocity, first_velocity)
            + bp_fa_dot(second_velocity, second_velocity));
    const double momentum_scale = std::max(1.0, std::abs(target_momentum));
    const double energy_scale = std::max(1.0, std::abs(target_energy));
    std::vector<std::size_t> selected;
    std::vector<std::size_t> zero_base;
    selected.reserve(std::min<std::size_t>(pairs.size(), 512));
    zero_base.reserve(pairs.size());
    for (std::size_t q = 0; q < pairs.size(); ++q) {
        if (pairs[q].base_weight > 0.0) {
            selected.push_back(q);
        } else {
            zero_base.push_back(q);
        }
    }
    std::stable_sort(
        zero_base.begin(),
        zero_base.end(),
        [&](std::size_t first, std::size_t second) {
            const double first_momentum =
                (pairs[first].momentum - target_momentum) / momentum_scale;
            const double first_energy =
                (pairs[first].energy - target_energy) / energy_scale;
            const double second_momentum =
                (pairs[second].momentum - target_momentum) / momentum_scale;
            const double second_energy =
                (pairs[second].energy - target_energy) / energy_scale;
            const double first_distance =
                first_momentum * first_momentum
                + first_energy * first_energy;
            const double second_distance =
                second_momentum * second_momentum
                + second_energy * second_energy;
            if (first_distance != second_distance) {
                return first_distance < second_distance;
            }
            if (pairs[first].first != pairs[second].first) {
                return pairs[first].first < pairs[second].first;
            }
            return pairs[first].second < pairs[second].second;
        });
    std::vector<double> alpha;
    bool solved = bp_reversible_solve_projection_subset(
        pairs,
        selected,
        target_momentum,
        target_energy,
        momentum_scale,
        energy_scale,
        p.bp_reversible_projection_tolerance,
        alpha);
    static constexpr std::size_t support_limits[] = {
        32, 64, 128, 256, 512,
    };
    std::size_t consumed_zero = 0;
    for (std::size_t support_limit : support_limits) {
        if (solved || selected.size() >= pairs.size()) {
            break;
        }
        const std::size_t target_size =
            std::min(support_limit, pairs.size());
        while (selected.size() < target_size
               && consumed_zero < zero_base.size()) {
            selected.push_back(zero_base[consumed_zero++]);
        }
        solved = bp_reversible_solve_projection_hull(
            pairs,
            target_momentum,
            target_energy,
            momentum_scale,
            energy_scale,
            p.bp_reversible_projection_tolerance,
            alpha,
            &selected);
    }
    if (!solved) {
        solved = bp_reversible_solve_projection_hull(
            pairs,
            target_momentum,
            target_energy,
            momentum_scale,
            energy_scale,
            p.bp_reversible_projection_tolerance,
            alpha);
    }
    if (!solved) {
        return false;
    }

    projection = {};
    projection.products.reserve(std::min<std::size_t>(pairs.size(), 8));
    double represented_mass = 0.0;
    double represented_momentum = 0.0;
    double represented_energy = 0.0;
    for (std::size_t q = 0; q < pairs.size(); ++q) {
        if (alpha[q] <= p.bp_reversible_projection_tolerance) {
            continue;
        }
        const int first = std::min(pairs[q].first, pairs[q].second);
        const int second = std::max(pairs[q].first, pairs[q].second);
        projection.products.push_back(
            BPReversibleProductPair{first, second, alpha[q]});
        represented_mass += alpha[q];
        represented_momentum += alpha[q] * pairs[q].momentum;
        represented_energy += alpha[q] * pairs[q].energy;
    }
    std::sort(
        projection.products.begin(),
        projection.products.end(),
        [](const BPReversibleProductPair& first,
           const BPReversibleProductPair& second) {
            return std::tie(first.first, first.second)
                < std::tie(second.first, second.second);
        });
    std::size_t merged_size = 0;
    for (const BPReversibleProductPair& product : projection.products) {
        if (merged_size > 0
            && projection.products[merged_size - 1].first == product.first
            && projection.products[merged_size - 1].second == product.second) {
            projection.products[merged_size - 1].weight += product.weight;
        } else {
            projection.products[merged_size++] = product;
        }
    }
    projection.products.resize(merged_size);
    projection.mass_error = represented_mass - 1.0;
    projection.momentum_error = represented_momentum - target_momentum;
    projection.energy_error = represented_energy - target_energy;
    const double normalized_error = std::max({
        std::abs(projection.mass_error),
        std::abs(projection.momentum_error) / momentum_scale,
        std::abs(projection.energy_error) / energy_scale,
    });
    return !projection.products.empty()
        && normalized_error <= 32.0 * p.bp_reversible_projection_tolerance;
}

/// Projects one continuous outgoing pair onto a nonnegative invariant-preserving pair stencil.
inline bool bp_reversible_project_outgoing_pair(
    const BPFAVec3& first_velocity,
    const BPFAVec3& second_velocity,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    BPReversibleProjection& projection,
    bool* outside_domain = nullptr,
    unsigned* outside_domain_mask = nullptr) {
    if (outside_domain != nullptr) {
        *outside_domain = false;
    }
    if (outside_domain_mask != nullptr) {
        *outside_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
    }
    auto velocity_less = [](const BPFAVec3& first, const BPFAVec3& second) {
        if (first.x != second.x) {
            return first.x < second.x;
        }
        if (first.y != second.y) {
            return first.y < second.y;
        }
        return first.z < second.z;
    };
    auto ordered_pair = [&](BPFAVec3 first, BPFAVec3 second) {
        if (velocity_less(second, first)) {
            std::swap(first, second);
        }
        return std::array<BPFAVec3, 2>{first, second};
    };
    auto pair_less = [&](const std::array<BPFAVec3, 2>& first,
                         const std::array<BPFAVec3, 2>& second) {
        if (velocity_less(first[0], second[0])) {
            return true;
        }
        if (velocity_less(second[0], first[0])) {
            return false;
        }
        return velocity_less(first[1], second[1]);
    };
    const double velocity_scale = std::max({
        1.0,
        std::abs(w.front()),
        std::abs(w.back()),
    });
    const bool symmetric_velocity_grid =
        std::abs(w.front() + w.back())
        <= 64.0 * std::numeric_limits<double>::epsilon() * velocity_scale;
    const std::array<BPFAVec3, 2> original_pair =
        ordered_pair(first_velocity, second_velocity);
    const std::array<BPFAVec3, 2> reversed_pair =
        ordered_pair(
            bp_fa_scale(first_velocity, -1.0),
            bp_fa_scale(second_velocity, -1.0));
    // Choose one representative of velocity reversal on the symmetric grid.
    // Reflect its stencil back afterward; self-reversed pairs use an equal
    // mixture to preserve that symmetry in the discrete projection.
    const bool reverse_projection =
        symmetric_velocity_grid && pair_less(original_pair, reversed_pair);
    const bool self_reverse_projection =
        symmetric_velocity_grid
        && !pair_less(original_pair, reversed_pair)
        && !pair_less(reversed_pair, original_pair);
    const std::array<BPFAVec3, 2>& canonical_pair =
        reverse_projection ? reversed_pair : original_pair;
    const int maximum_radius =
        std::max(1, p.bp_reversible_candidate_radius);
    for (int radius = 1; radius <= maximum_radius; ++radius) {
        bool trial_outside_domain = false;
        unsigned trial_domain_mask = BP_REVERSIBLE_DOMAIN_NONE;
        BPReversibleProjection trial;
        if (bp_reversible_project_outgoing_pair_at_radius(
                canonical_pair[0],
                canonical_pair[1],
                ri,
                w,
                eta,
                geometry,
                p,
                radius,
                trial,
                &trial_outside_domain,
                &trial_domain_mask)) {
            if (reverse_projection || self_reverse_projection) {
                std::vector<BPReversibleProductPair> mapped;
                mapped.reserve(
                    (self_reverse_projection ? 2 : 1)
                    * trial.products.size());
                for (const BPReversibleProductPair& product
                     : trial.products) {
                    const std::array<int, 2> first_indices =
                        bp_reversible_velocity_indices(product.first, p);
                    const std::array<int, 2> second_indices =
                        bp_reversible_velocity_indices(product.second, p);
                    const int reverse_first =
                        bp_reversible_velocity_index(
                            p.Nw - 1 - first_indices[0],
                            first_indices[1],
                            p);
                    const int reverse_second =
                        bp_reversible_velocity_index(
                            p.Nw - 1 - second_indices[0],
                            second_indices[1],
                            p);
                    const std::pair<int, int> reversed_product{
                        std::min(reverse_first, reverse_second),
                        std::max(reverse_first, reverse_second),
                    };
                    if (self_reverse_projection) {
                        const std::pair<int, int> original_product{
                            std::min(product.first, product.second),
                            std::max(product.first, product.second),
                        };
                        mapped.push_back(BPReversibleProductPair{
                            original_product.first,
                            original_product.second,
                            0.5 * product.weight,
                        });
                        mapped.push_back(BPReversibleProductPair{
                            reversed_product.first,
                            reversed_product.second,
                            0.5 * product.weight,
                        });
                    } else {
                        mapped.push_back(BPReversibleProductPair{
                            reversed_product.first,
                            reversed_product.second,
                            product.weight,
                        });
                    }
                }
                std::sort(
                    mapped.begin(),
                    mapped.end(),
                    [](const BPReversibleProductPair& first,
                       const BPReversibleProductPair& second) {
                        return std::tie(first.first, first.second)
                            < std::tie(second.first, second.second);
                    });
                std::size_t mapped_size = 0;
                for (const BPReversibleProductPair& product : mapped) {
                    if (mapped_size > 0
                        && mapped[mapped_size - 1].first == product.first
                        && mapped[mapped_size - 1].second == product.second) {
                        mapped[mapped_size - 1].weight += product.weight;
                    } else {
                        mapped[mapped_size++] = product;
                    }
                }
                mapped.resize(mapped_size);
                trial.products = std::move(mapped);
                double represented_mass = 0.0;
                double represented_momentum = 0.0;
                double represented_energy = 0.0;
                for (const BPReversibleProductPair& product
                     : trial.products) {
                    represented_mass += product.weight;
                    represented_momentum += product.weight * (
                        bp_reversible_radial_velocity(product.first, w, p)
                        + bp_reversible_radial_velocity(product.second, w, p));
                    represented_energy += product.weight * (
                        bp_reversible_energy(product.first, ri, w, eta, p)
                        + bp_reversible_energy(product.second, ri, w, eta, p));
                }
                const double target_momentum =
                    first_velocity.x + second_velocity.x;
                const double target_energy = 0.5 * (
                    bp_fa_dot(first_velocity, first_velocity)
                    + bp_fa_dot(second_velocity, second_velocity));
                trial.mass_error = represented_mass - 1.0;
                trial.momentum_error =
                    represented_momentum - target_momentum;
                trial.energy_error = represented_energy - target_energy;
                const double normalized_error = std::max({
                    std::abs(trial.mass_error),
                    std::abs(trial.momentum_error)
                        / std::max(1.0, std::abs(target_momentum)),
                    std::abs(trial.energy_error)
                        / std::max(1.0, std::abs(target_energy)),
                });
                if (normalized_error
                    > 32.0 * p.bp_reversible_projection_tolerance) {
                    continue;
                }
            }
            projection = std::move(trial);
            return true;
        }
        if (trial_outside_domain) {
            if (outside_domain != nullptr) {
                *outside_domain = true;
            }
            if (outside_domain_mask != nullptr) {
                *outside_domain_mask = trial_domain_mask;
            }
            return false;
        }
    }
    return false;
}

// Channel activity is coefficient*f_a*f_b, evaluated from the current shell.
inline double bp_reversible_channel_activity(
    const std::vector<double>& shell_f,
    int first,
    int second,
    double coefficient) {
    return coefficient
        * shell_f[static_cast<std::size_t>(first)]
        * shell_f[static_cast<std::size_t>(second)];
}

// Combine repeated velocity indices before evaluating a reaction so one cell
// receives its complete product exponent and net population change once.
inline std::vector<BPReversibleSpecies> bp_reversible_compress_reaction(
    int incoming_first,
    int incoming_second,
    const std::vector<BPReversibleProductPair>& products) {
    std::vector<BPReversibleSpecies> species;
    species.reserve(2 * products.size() + 2);
    for (const BPReversibleProductPair& product : products) {
        species.push_back(BPReversibleSpecies{
            product.first, product.weight, product.weight});
        species.push_back(BPReversibleSpecies{
            product.second, product.weight, product.weight});
    }
    species.push_back(BPReversibleSpecies{incoming_first, 0.0, -1.0});
    species.push_back(BPReversibleSpecies{incoming_second, 0.0, -1.0});
    std::sort(
        species.begin(),
        species.end(),
        [](const BPReversibleSpecies& first,
           const BPReversibleSpecies& second) {
            return first.index < second.index;
        });
    std::size_t compressed_size = 0;
    for (const BPReversibleSpecies& value : species) {
        if (compressed_size > 0
            && species[compressed_size - 1].index == value.index) {
            species[compressed_size - 1].product_order += value.product_order;
            species[compressed_size - 1].net_population_change +=
                value.net_population_change;
        } else {
            species[compressed_size++] = value;
        }
    }
    species.resize(compressed_size);
    return species;
}

// Omitted channels retain their coefficients and current activity.  This
// separates finite-domain loss from failure of the moment projection.
struct BPReversibleActivityBudget {
    double represented = 0.0;
    double omitted_domain = 0.0;
    double omitted_projection = 0.0;
    double omitted_inactive = 0.0;

    double omitted() const {
        return omitted_domain + omitted_projection + omitted_inactive;
    }

    double omitted_fraction() const {
        const double attempted = represented + omitted();
        return (attempted > 0.0) ? omitted() / attempted : 0.0;
    }
};

// Re-evaluate population-dependent channel weights as the shell relaxes.
inline BPReversibleActivityBudget bp_reversible_activity_budget(
    const std::vector<double>& shell_f,
    const BPReversibleGraph& graph) {
    BPReversibleActivityBudget budget;
    budget.omitted_inactive = graph.omitted_inactive_activity;
    for (const BPReversibleReaction& reaction : graph.reactions) {
        budget.represented += bp_reversible_channel_activity(
            shell_f,
            reaction.incoming_first,
            reaction.incoming_second,
            reaction.coefficient);
    }
    for (const BPReversibleOmittedChannel& channel : graph.omitted_domain_channels) {
        budget.omitted_domain += bp_reversible_channel_activity(
            shell_f,
            channel.incoming_first,
            channel.incoming_second,
            channel.coefficient);
    }
    for (const BPReversibleOmittedChannel& channel : graph.omitted_projection_channels) {
        budget.omitted_projection += bp_reversible_channel_activity(
            shell_f,
            channel.incoming_first,
            channel.incoming_second,
            channel.coefficient);
    }
    return budget;
}

/// Builds the symmetric weak-form reaction graph for one radial shell.
inline BPReversibleGraph bp_reversible_build_shell_graph(
    const std::vector<double>& shell_f,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    BPReversibleGraph graph;
    const int velocity_cells = p.Nw * p.NL;
    validate_finite_nonnegative_distribution(
        shell_f,
        static_cast<std::size_t>(velocity_cells),
        "boltzmann_reversible graph input");
    validate_cell_average_eta_geometry(
        geometry, p, "boltzmann_reversible graph");
    if (!(ri >= geometry.radial.edge.front())
        || !(ri <= geometry.radial.edge.back())
        || w != geometry.radial_velocity.center
        || eta != geometry.eta.center) {
        throw std::runtime_error(
            "boltzmann_reversible graph coordinates do not match its cell geometry.");
    }
    if (velocity_cells > p.bp_reversible_max_velocity_cells) {
        throw std::runtime_error(
            "boltzmann_reversible full reaction graph exceeds "
            "bp_reversible_max_velocity_cells. Increase the explicit reference limit "
            "only after checking its O((Nw NL)^2) memory and runtime cost.");
    }
    // At fixed r, d^3v=2*pi*dw*deta/r^2.  The same cell volumes enter
    // reaction coefficients, population changes and entropy.
    graph.velocity_cell_volume.assign(
        static_cast<std::size_t>(velocity_cells), 0.0);
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            const int q = bp_reversible_velocity_index(j, k, p);
            graph.velocity_cell_volume[static_cast<std::size_t>(q)] =
                2.0 * PI
                * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
                * geometry.eta.width[static_cast<std::size_t>(k)]
                / (ri * ri);
        }
    }
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
    const double inv_pair_phi = 1.0 / static_cast<double>(npair_phi);
    const double inv_psi = 1.0 / static_cast<double>(npsi);

    // Enumerate ordered incoming pairs and normalized angular quadrature.
    // Keep unsupported domain/projection rates in separate omission ledgers.
    std::map<std::array<int, 3>, double> omitted_domain_coefficients;
    std::map<std::pair<int, int>, double> omitted_projection_coefficients;
    for (int a_index = 0; a_index < velocity_cells; ++a_index) {
        const std::array<int, 2> a_grid =
            bp_reversible_velocity_indices(a_index, p);
        const double a_wr = w[static_cast<std::size_t>(a_grid[0])];
        const double a_vt =
            std::sqrt(2.0 * eta[static_cast<std::size_t>(a_grid[1])]) / ri;
        const BPFAVec3 first{a_wr, a_vt, 0.0};
        for (int b_index = 0; b_index < velocity_cells; ++b_index) {
            const std::array<int, 2> b_grid =
                bp_reversible_velocity_indices(b_index, p);
            const double b_wr = w[static_cast<std::size_t>(b_grid[0])];
            const double b_vt =
                std::sqrt(2.0 * eta[static_cast<std::size_t>(b_grid[1])]) / ri;
            for (int iphi = 0; iphi < npair_phi; ++iphi) {
                const BPFAVec3 second{
                    b_wr,
                    b_vt * pair_phi_cos[static_cast<std::size_t>(iphi)],
                    b_vt * pair_phi_sin[static_cast<std::size_t>(iphi)],
                };
                const BPFAVec3 relative = bp_fa_sub(first, second);
                const double relative_speed = bp_fa_norm(relative);
                if (!(relative_speed > 0.0)) {
                    continue;
                }
                const double sigma = bp_fa_total_cross_section(relative_speed, p);
                if (!(sigma > 0.0)) {
                    continue;
                }
                std::vector<double> normalized_mu(static_cast<std::size_t>(nmu), 0.0);
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
                const BPFAVec3 center_of_mass =
                    bp_fa_scale(bp_fa_add(first, second), 0.5);
                const BPFAVec3 incoming_direction =
                    bp_fa_scale(relative, 1.0 / relative_speed);
                const std::array<BPFAVec3, 2> basis =
                    bp_fa_perpendicular_basis_unit(incoming_direction);
                for (int im = 0; im < nmu; ++im) {
                    const double cosine = mu[static_cast<std::size_t>(im)];
                    const double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
                    const double angular_mu =
                        normalized_mu[static_cast<std::size_t>(im)] / mu_sum;
                    for (int ipsi = 0; ipsi < npsi; ++ipsi) {
                        const BPFAVec3 outgoing_direction =
                            bp_fa_scattered_relative_direction_from_basis(
                                incoming_direction,
                                basis[0],
                                basis[1],
                                cosine,
                                sine,
                                psi_cos[static_cast<std::size_t>(ipsi)],
                                psi_sin[static_cast<std::size_t>(ipsi)]);
                        const BPFAVec3 outgoing_relative =
                            bp_fa_scale(outgoing_direction, relative_speed);
                        const BPFAVec3 outgoing_first =
                            bp_fa_add(center_of_mass, bp_fa_scale(outgoing_relative, 0.5));
                        const BPFAVec3 outgoing_second =
                            bp_fa_sub(center_of_mass, bp_fa_scale(outgoing_relative, 0.5));
                        const double angular_weight =
                            inv_pair_phi * angular_mu * inv_psi;
                        const double first_volume =
                            graph.velocity_cell_volume[
                                static_cast<std::size_t>(a_index)];
                        const double second_volume =
                            graph.velocity_cell_volume[
                                static_cast<std::size_t>(b_index)];
                        // The symmetric weak-form normalization is one quarter
                        // of the ordered pair measure times sigma*g and the
                        // normalized incoming/scattering angular weights.
                        const double coefficient =
                            0.25 * first_volume * second_volume
                            * relative_speed * sigma * angular_weight;
                        BPReversibleProjection projection;
                        bool outside_domain = false;
                        unsigned outside_domain_mask =
                            BP_REVERSIBLE_DOMAIN_NONE;
                        if (!bp_reversible_project_outgoing_pair(
                                outgoing_first,
                                outgoing_second,
                                ri,
                                w,
                                eta,
                                geometry,
                                p,
                                projection,
                                &outside_domain,
                                &outside_domain_mask)) {
                            if (outside_domain) {
                                omitted_domain_coefficients[{
                                    a_index,
                                    b_index,
                                    static_cast<int>(outside_domain_mask),
                                }] += coefficient;
                            } else {
                                omitted_projection_coefficients[{
                                    a_index,
                                    b_index,
                                }] += coefficient;
                            }
                            continue;
                        }
                        BPReversibleReaction reaction;
                        reaction.incoming_first = a_index;
                        reaction.incoming_second = b_index;
                        reaction.coefficient = coefficient;
                        reaction.species = bp_reversible_compress_reaction(
                            a_index,
                            b_index,
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
                }
            }
        }
    }
    for (const auto& entry : omitted_domain_coefficients) {
        graph.omitted_domain_channels.push_back(
            BPReversibleOmittedChannel{
                entry.first[0],
                entry.first[1],
                entry.second,
                static_cast<unsigned>(entry.first[2]),
            });
    }
    for (const auto& entry : omitted_projection_coefficients) {
        graph.omitted_projection_channels.push_back(
            BPReversibleOmittedChannel{
                entry.first.first,
                entry.first.second,
                entry.second,
            });
    }
    const BPReversibleActivityBudget budget =
        bp_reversible_activity_budget(shell_f, graph);
    graph.represented_activity = budget.represented;
    graph.omitted_domain_activity = budget.omitted_domain;
    graph.omitted_projection_activity = budget.omitted_projection;
    graph.omitted_activity = budget.omitted();
    return graph;
}

// Sum conservative reaction fluxes into df/dt.  Fixed reaction blocks own
// private arrays and reduce in block order, preserving a deterministic sum
// across worker scheduling.  gross_rhs records the total two-way activity.
inline std::vector<double> bp_reversible_shell_rhs(
    const std::vector<double>& shell_f,
    const BPReversibleGraph& graph,
    const Params& p,
    std::vector<double>* gross_rhs = nullptr) {
    validate_finite_nonnegative_distribution(
        shell_f,
        graph.velocity_cell_volume.size(),
        "boltzmann_reversible RHS input");
    std::vector<double> rhs(shell_f.size(), 0.0);
    if (graph.velocity_cell_volume.size() != shell_f.size()) {
        throw std::runtime_error(
            "boltzmann_reversible reaction graph has incompatible cell volumes.");
    }
    const double floor = p.bp_reversible_entropy_floor;
    if (gross_rhs != nullptr) {
        gross_rhs->assign(shell_f.size(), 0.0);
    }
    constexpr std::size_t reaction_block_size = 8192;
    const std::size_t block_count = std::max<std::size_t>(
        1,
        (graph.reactions.size() + reaction_block_size - 1)
            / reaction_block_size);
    std::vector<std::vector<double>> block_rhs(
        block_count,
        std::vector<double>(shell_f.size(), 0.0));
    std::vector<std::vector<double>> block_gross;
    if (gross_rhs != nullptr) {
        block_gross.assign(
            block_count,
            std::vector<double>(shell_f.size(), 0.0));
    }
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(block_count),
        effective_threads(p, static_cast<int>(block_count)),
        1,
        [&](int blo, int bhi, int) {
            for (int block = blo; block < bhi; ++block) {
                std::vector<double>& local_rhs =
                    block_rhs[static_cast<std::size_t>(block)];
                std::vector<double>* local_gross = gross_rhs != nullptr
                    ? &block_gross[static_cast<std::size_t>(block)]
                    : nullptr;
                const std::size_t reaction_begin =
                    static_cast<std::size_t>(block) * reaction_block_size;
                const std::size_t reaction_end = std::min(
                    graph.reactions.size(),
                    reaction_begin + reaction_block_size);
                for (std::size_t reaction_index = reaction_begin;
                     reaction_index < reaction_end;
                     ++reaction_index) {
                    const BPReversibleReaction& reaction =
                        graph.reactions[reaction_index];
                    const double first = shell_f[static_cast<std::size_t>(
                        reaction.incoming_first)];
                    const double second = shell_f[static_cast<std::size_t>(
                        reaction.incoming_second)];
                    const double forward_activity = first * second;
                    // The geometric product has logarithm sum(a_q*log(f_q)).
                    // Its difference from f_a*f_b drives relaxation along the
                    // same conservative reaction vector in either direction.
                    bool zero_reverse = false;
                    double log_reverse = 0.0;
                    for (const BPReversibleSpecies& species :
                         reaction.species) {
                        if (!(species.product_order > 0.0)) {
                            continue;
                        }
                        const double product_value =
                            shell_f[static_cast<std::size_t>(species.index)];
                        if (!(product_value > 0.0)) {
                            zero_reverse = true;
                            break;
                        }
                        log_reverse += species.product_order
                            * std::log(std::max(product_value, floor));
                    }
                    const double reverse_activity = zero_reverse
                        ? 0.0
                        : std::exp(log_reverse);
                    const double flux = reaction.coefficient
                        * (forward_activity - reverse_activity);
                    const double gross_flux = reaction.coefficient
                        * (forward_activity + reverse_activity);
                    for (const BPReversibleSpecies& species :
                         reaction.species) {
                        const std::size_t q =
                            static_cast<std::size_t>(species.index);
                        local_rhs[q] +=
                            species.net_population_change * flux
                            / graph.velocity_cell_volume[q];
                        if (local_gross != nullptr) {
                            (*local_gross)[q] +=
                                std::abs(species.net_population_change)
                                * gross_flux
                                / graph.velocity_cell_volume[q];
                        }
                    }
                }
            }
        });
    parallel_for_chunks(
        0,
        static_cast<int>(shell_f.size()),
        effective_threads(p, static_cast<int>(shell_f.size())),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                double value = 0.0;
                double gross_value = 0.0;
                for (std::size_t block = 0; block < block_count; ++block) {
                    value += block_rhs[block][static_cast<std::size_t>(q)];
                    if (gross_rhs != nullptr) {
                        gross_value +=
                            block_gross[block][static_cast<std::size_t>(q)];
                    }
                }
                rhs[static_cast<std::size_t>(q)] = value;
                if (gross_rhs != nullptr) {
                    (*gross_rhs)[static_cast<std::size_t>(q)] = gross_value;
                }
            }
        });
    return rhs;
}

// The largest fractional net loss rate bounds an explicit positive update.
inline double bp_reversible_positive_dt_limit(
    const std::vector<double>& f,
    const std::vector<double>& rhs,
    double cfl) {
    double maximum_loss = 0.0;
    for (std::size_t q = 0; q < f.size(); ++q) {
        if (rhs[q] < 0.0 && f[q] > 0.0) {
            maximum_loss = std::max(maximum_loss, -rhs[q] / f[q]);
        }
    }
    return (maximum_loss > 0.0)
        ? cfl / maximum_loss
        : std::numeric_limits<double>::infinity();
}

inline bool bp_reversible_nonnegative(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
        return value >= 0.0 && std::isfinite(value);
    });
}

// Compute the discrete H functional with each cell's d^3v volume.  Zero
// populations contribute their limiting value; positive logs use the floor.
inline double bp_reversible_shell_entropy(
    const std::vector<double>& values,
    const BPReversibleGraph& graph,
    double floor) {
    if (values.size() != graph.velocity_cell_volume.size()) {
        throw std::runtime_error(
            "boltzmann_reversible entropy received incompatible cell volumes.");
    }
    double entropy = 0.0;
    for (std::size_t q = 0; q < values.size(); ++q) {
        const double value = values[q];
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(
                "boltzmann_reversible entropy requires finite nonnegative values.");
        }
        if (value > 0.0) {
            entropy += graph.velocity_cell_volume[q]
                * value * (std::log(std::max(value, floor)) - 1.0);
        }
    }
    return entropy;
}

struct BPReversibleReactionEvaluation {
    double flux = 0.0;
    double forward_activity = 0.0;
    double reverse_activity = 0.0;
    double derivative = 0.0;
    bool derivative_is_finite = false;
    bool valid = false;
};

// At trial extent x, each f_q changes by nu_q*x/volume_q.  Evaluate both
// activities and their derivative along this one invariant-preserving line.
inline BPReversibleReactionEvaluation bp_reversible_evaluate_reaction(
    const std::vector<double>& f,
    const BPReversibleReaction& reaction,
    const std::vector<double>& volume,
    double extent,
    double floor) {
    BPReversibleReactionEvaluation evaluation;
    double incoming_first = 0.0;
    double incoming_second = 0.0;
    double incoming_first_derivative = 0.0;
    double incoming_second_derivative = 0.0;
    double log_reverse = 0.0;
    double log_reverse_derivative = 0.0;
    bool reverse_is_zero = false;
    for (const BPReversibleSpecies& species : reaction.species) {
        const std::size_t q = static_cast<std::size_t>(species.index);
        const double derivative =
            species.net_population_change / volume[q];
        const double value = std::fma(derivative, extent, f[q]);
        if (!(value >= 0.0) || !std::isfinite(value)) {
            return evaluation;
        }
        if (species.index == reaction.incoming_first) {
            incoming_first = value;
            incoming_first_derivative = derivative;
        }
        if (species.index == reaction.incoming_second) {
            incoming_second = value;
            incoming_second_derivative = derivative;
        }
        if (species.product_order > 0.0) {
            if (!(value > 0.0)) {
                reverse_is_zero = true;
            } else {
                log_reverse += species.product_order
                    * std::log(std::max(value, floor));
                log_reverse_derivative +=
                    species.product_order * derivative / value;
            }
        }
    }
    const double forward = incoming_first * incoming_second;
    const double forward_derivative =
        incoming_first_derivative * incoming_second
        + incoming_first * incoming_second_derivative;
    const double reverse =
        reverse_is_zero ? 0.0 : std::exp(log_reverse);
    evaluation.forward_activity = reaction.coefficient * forward;
    evaluation.reverse_activity = reaction.coefficient * reverse;
    evaluation.flux = reaction.coefficient * (forward - reverse);
    if (!reverse_is_zero) {
        evaluation.derivative = reaction.coefficient
            * (forward_derivative - reverse * log_reverse_derivative);
        evaluation.derivative_is_finite =
            std::isfinite(evaluation.derivative);
    }
    evaluation.valid = std::isfinite(evaluation.flux);
    return evaluation;
}

// Intersect the nonnegative-population intervals of every participating cell.
// Both forward and reverse directions are constrained by their consumed cells.
inline std::pair<double, double> bp_reversible_extent_bounds(
    const std::vector<double>& f,
    const BPReversibleReaction& reaction,
    const std::vector<double>& volume) {
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();
    for (const BPReversibleSpecies& species : reaction.species) {
        const std::size_t q = static_cast<std::size_t>(species.index);
        if (species.net_population_change > 0.0) {
            const long double candidate =
                -static_cast<long double>(f[q])
                * static_cast<long double>(volume[q])
                / static_cast<long double>(
                    species.net_population_change);
            lower = std::max(
                lower,
                static_cast<double>(candidate));
        } else if (species.net_population_change < 0.0) {
            const long double candidate =
                static_cast<long double>(f[q])
                * static_cast<long double>(volume[q])
                / static_cast<long double>(
                    -species.net_population_change);
            upper = std::min(
                upper,
                static_cast<double>(candidate));
        }
    }
    return {lower, upper};
}

inline bool bp_reversible_extent_is_strictly_feasible(
    double extent,
    const std::pair<double, double>& bounds) {
    return extent > bounds.first && extent < bounds.second
        && std::isfinite(extent);
}

// A geometric midpoint resolves brackets spanning many positive decades;
// ordinary bisection handles intervals crossing zero or with similar scales.
inline double bp_reversible_bracket_midpoint(
    double lower,
    double upper) {
    const double minimum_positive =
        std::numeric_limits<double>::denorm_min();
    if (lower >= 0.0) {
        const double positive_lower =
            (lower > 0.0) ? lower : minimum_positive;
        if (upper / positive_lower > 4.0) {
            return std::exp(
                0.5 * (
                    std::log(positive_lower)
                    + std::log(upper)));
        }
    } else if (upper <= 0.0) {
        const double positive_upper =
            (upper < 0.0) ? -upper : minimum_positive;
        const double positive_lower = -lower;
        if (positive_lower / positive_upper > 4.0) {
            return -std::exp(
                0.5 * (
                    std::log(positive_lower)
                    + std::log(positive_upper)));
        }
    }
    return lower + 0.5 * (upper - lower);
}

/// Integrates one reversible reaction.  Heun's method handles nonstiff
/// channels.  A bracketed backward-Euler solve handles channels whose
/// fractional reverse population becomes stiff near an empty cell.
inline double bp_reversible_reaction_extent(
    const std::vector<double>& f,
    const BPReversibleReaction& reaction,
    const std::vector<double>& volume,
    double dt,
    double floor,
    std::uint64_t* implicit_updates = nullptr,
    std::uint64_t* implicit_iterations = nullptr) {
    if (!(dt > 0.0) || !(reaction.coefficient > 0.0)) {
        return 0.0;
    }
    const BPReversibleReactionEvaluation initial =
        bp_reversible_evaluate_reaction(
            f, reaction, volume, 0.0, floor);
    if (!initial.valid || initial.flux == 0.0) {
        if (!initial.valid) {
            throw std::runtime_error(
                "boltzmann_reversible reaction has a nonfinite initial flux.");
        }
        return 0.0;
    }
    const std::pair<double, double> bounds =
        bp_reversible_extent_bounds(f, reaction, volume);

    // Use the two-stage explicit extent when it stays positive and on the
    // initial side of equilibrium; otherwise solve the implicit scalar extent.
    const double euler_extent = dt * initial.flux;
    if (bp_reversible_extent_is_strictly_feasible(
            euler_extent, bounds)) {
        const BPReversibleReactionEvaluation euler =
            bp_reversible_evaluate_reaction(
                f, reaction, volume, euler_extent, floor);
        if (euler.valid) {
            const double heun_extent =
                0.5 * dt * (initial.flux + euler.flux);
            if (bp_reversible_extent_is_strictly_feasible(
                    heun_extent, bounds)
                && heun_extent * initial.flux >= 0.0) {
                const BPReversibleReactionEvaluation heun =
                    bp_reversible_evaluate_reaction(
                        f, reaction, volume, heun_extent, floor);
                if (heun.valid
                    && heun.flux * initial.flux >= 0.0) {
                    return heun_extent;
                }
            }
        }
    }

    double lower = 0.0;
    double upper = 0.0;
    if (implicit_updates != nullptr) {
        ++(*implicit_updates);
    }
    if (initial.flux > 0.0) {
        if (bounds.second == 0.0) {
            return 0.0;
        }
        if (!(bounds.second > 0.0) || !std::isfinite(bounds.second)) {
            throw std::runtime_error(
                "boltzmann_reversible reaction has no finite positive extent bound.");
        }
        lower = 0.0;
        upper = bounds.second;
    } else {
        if (bounds.first == 0.0) {
            return 0.0;
        }
        if (!(bounds.first < 0.0) || !std::isfinite(bounds.first)) {
            throw std::runtime_error(
                "boltzmann_reversible reaction has no finite negative extent bound.");
        }
        lower = bounds.first;
        upper = 0.0;
    }
    double extent = std::clamp(euler_extent, lower, upper);
    if (!bp_reversible_extent_is_strictly_feasible(extent, bounds)
        || extent == lower || extent == upper) {
        extent = bp_reversible_bracket_midpoint(lower, upper);
    }
    const double extent_scale = std::max({
        std::abs(lower),
        std::abs(upper),
        std::abs(euler_extent),
        std::numeric_limits<double>::min(),
    });
    double last_residual = std::numeric_limits<double>::quiet_NaN();
    double last_width = upper - lower;
    int invalid_trials = 0;
    for (int iteration = 0; iteration < 512; ++iteration) {
        if (implicit_iterations != nullptr) {
            ++(*implicit_iterations);
        }
        const BPReversibleReactionEvaluation trial =
            bp_reversible_evaluate_reaction(
                f, reaction, volume, extent, floor);
        if (!trial.valid) {
            ++invalid_trials;
            if (initial.flux > 0.0) {
                upper = extent;
            } else {
                lower = extent;
            }
            const double candidate =
                bp_reversible_bracket_midpoint(lower, upper);
            if (candidate == extent
                || candidate == lower
                || candidate == upper) {
                return std::nextafter(extent, 0.0);
            }
            extent = candidate;
            continue;
        }
        // Backward Euler requires x=dt*flux(f+nu*x/V).  Keep the root
        // bracket inside the positive interval and safeguard Newton steps.
        const double residual = extent - dt * trial.flux;
        if (residual > 0.0) {
            upper = extent;
        } else {
            lower = extent;
        }
        const double width = upper - lower;
        last_residual = residual;
        last_width = width;
        const double bracket_scale = std::max({
            std::abs(lower),
            std::abs(upper),
            std::numeric_limits<double>::min(),
        });
        if (width <= 1024.0 * std::numeric_limits<double>::epsilon()
                * bracket_scale
            || std::abs(residual)
                <= 1024.0 * std::numeric_limits<double>::epsilon()
                    * extent_scale) {
            return extent;
        }
        double candidate =
            bp_reversible_bracket_midpoint(lower, upper);
        if (trial.derivative_is_finite) {
            const double residual_derivative =
                1.0 - dt * trial.derivative;
            if (residual_derivative > 0.0
                && std::isfinite(residual_derivative)) {
                const double newton =
                    extent - residual / residual_derivative;
                const double safeguard_margin = 0.25 * width;
                if (newton > lower + safeguard_margin
                    && newton < upper - safeguard_margin
                    && std::isfinite(newton)) {
                    candidate = newton;
                }
            }
        }
        if (candidate == extent
            || candidate == lower
            || candidate == upper) {
            return extent;
        }
        extent = candidate;
    }
    std::ostringstream message;
    message
        << "boltzmann_reversible implicit reaction extent did not converge:"
        << " flux0=" << initial.flux
        << ", coefficient=" << reaction.coefficient
        << ", dt=" << dt
        << ", lower=" << lower
        << ", upper=" << upper
        << ", extent=" << extent
        << ", residual=" << last_residual
        << ", width=" << last_width
        << ", scale=" << extent_scale
        << ", invalid_trials=" << invalid_trials;
    throw std::runtime_error(message.str());
}

// Apply the common extent through the reaction's invariant-preserving vector
// and reject any nonfinite or negative population before storing that cell.
inline void bp_reversible_apply_extent(
    std::vector<double>& f,
    const BPReversibleReaction& reaction,
    const std::vector<double>& volume,
    double extent) {
    for (const BPReversibleSpecies& species : reaction.species) {
        const std::size_t q = static_cast<std::size_t>(species.index);
        const double updated = std::fma(
            species.net_population_change / volume[q],
            extent,
            f[q]);
        if (!(updated >= 0.0) || !std::isfinite(updated)) {
            std::ostringstream message;
            message
                << "boltzmann_reversible implicit reaction produced "
                << "an invalid population in velocity cell "
                << species.index << ": " << updated;
            throw std::runtime_error(message.str());
        }
        f[q] = updated;
    }
}

/// Reference explicit integrator retained for focused convergence tests.
inline BPReversibleShellResult bp_reversible_advance_shell_explicit_ssprk2(
    const std::vector<double>& initial,
    const BPReversibleGraph& graph,
    double dt,
    const Params& p,
    const std::vector<double>* first_rhs = nullptr) {
    BPReversibleShellResult result;
    result.f = initial;
    validate_finite_nonnegative_distribution(
        initial,
        graph.velocity_cell_volume.size(),
        "boltzmann_reversible step input");
    result.maximum_projection_error = graph.maximum_projection_error;
    const BPReversibleActivityBudget initial_budget =
        bp_reversible_activity_budget(initial, graph);
    result.represented_activity = initial_budget.represented;
    result.omitted_domain_activity = initial_budget.omitted_domain;
    result.omitted_projection_activity = initial_budget.omitted_projection;
    result.omitted_inactive_activity = initial_budget.omitted_inactive;
    result.omitted_fraction = initial_budget.omitted_fraction();
    double remaining = dt;
    while (remaining > 1.0e-14 * std::max(1.0, dt)) {
        if (result.substeps >= p.bp_collision_max_substeps) {
            throw std::runtime_error(
                "boltzmann_reversible exceeded bp_collision_max_substeps.");
        }
        const std::vector<double> rhs0 =
            (result.substeps == 0 && first_rhs != nullptr)
            ? *first_rhs
            : bp_reversible_shell_rhs(result.f, graph, p);
        const double limit0 =
            bp_reversible_positive_dt_limit(result.f, rhs0, p.bp_collision_cfl);
        double subdt = std::min(remaining, limit0);
        bool candidate_reduced = false;
        if (!(subdt > 0.0) || !std::isfinite(subdt)) {
            if (std::isinf(limit0)) {
                break;
            }
            throw std::runtime_error("boltzmann_reversible found an invalid positivity timestep.");
        }
        bool accepted = false;
        std::string last_rejection = "none";
        double last_minimum_value = std::numeric_limits<double>::infinity();
        double last_entropy_excess = 0.0;
        const double entropy_before = bp_reversible_shell_entropy(
            result.f, graph, p.bp_reversible_entropy_floor);
        for (int retry = 0; retry < 32; ++retry) {
            std::vector<double> stage1(result.f.size(), 0.0);
            for (std::size_t q = 0; q < result.f.size(); ++q) {
                stage1[q] = result.f[q] + subdt * rhs0[q];
            }
            if (!bp_reversible_nonnegative(stage1)) {
                last_rejection = "stage1_nonnegative";
                last_minimum_value = *std::min_element(
                    stage1.begin(), stage1.end());
                subdt *= 0.5;
                candidate_reduced = true;
                continue;
            }
            const std::vector<double> rhs1 =
                bp_reversible_shell_rhs(stage1, graph, p);
            std::vector<double> stage2(result.f.size(), 0.0);
            for (std::size_t q = 0; q < result.f.size(); ++q) {
                stage2[q] = 0.5 * result.f[q]
                    + 0.5 * (stage1[q] + subdt * rhs1[q]);
            }
            if (!bp_reversible_nonnegative(stage2)) {
                last_rejection = "stage2_nonnegative";
                last_minimum_value = *std::min_element(
                    stage2.begin(), stage2.end());
                subdt *= 0.5;
                candidate_reduced = true;
                continue;
            }
            const double entropy_after = bp_reversible_shell_entropy(
                stage2, graph, p.bp_reversible_entropy_floor);
            const double entropy_tolerance =
                p.bp_reversible_entropy_tolerance
                * std::max({1.0, std::abs(entropy_before), std::abs(entropy_after)});
            if (entropy_after > entropy_before + entropy_tolerance) {
                ++result.entropy_rejections;
                last_rejection = "entropy";
                last_entropy_excess =
                    entropy_after - entropy_before - entropy_tolerance;
                subdt *= 0.5;
                candidate_reduced = true;
                continue;
            }
            result.f = std::move(stage2);
            result.reaction_updates +=
                2 * static_cast<std::uint64_t>(graph.reactions.size());
            const BPReversibleActivityBudget stage_budget =
                bp_reversible_activity_budget(result.f, graph);
            result.omitted_fraction = std::max(
                result.omitted_fraction,
                stage_budget.omitted_fraction());
            accepted = true;
            break;
        }
        if (!accepted) {
            std::ostringstream message;
            message
                << "boltzmann_reversible could not find an admissible "
                << "SSPRK2 substep: reason=" << last_rejection
                << ", remaining_dt=" << remaining
                << ", initial_dt_limit=" << limit0
                << ", final_trial_dt=" << subdt
                << ", minimum_trial_value=" << last_minimum_value
                << ", entropy_excess=" << last_entropy_excess;
            throw std::runtime_error(message.str());
        }
        const double effective_limit =
            candidate_reduced ? std::min(limit0, subdt) : limit0;
        result.minimum_dt_limit = std::min(
            result.minimum_dt_limit, effective_limit);
        remaining = std::max(0.0, remaining - subdt);
        ++result.substeps;
    }
    return result;
}

/// Advances one shell with conservative reaction extents and symmetric
/// forward/reverse ordering.
inline BPReversibleShellResult bp_reversible_advance_shell(
    const std::vector<double>& initial,
    const BPReversibleGraph& graph,
    double dt,
    const Params& p) {
    BPReversibleShellResult result;
    result.f = initial;
    validate_finite_nonnegative_distribution(
        initial,
        graph.velocity_cell_volume.size(),
        "boltzmann_reversible implicit step input");
    result.maximum_projection_error = graph.maximum_projection_error;
    const BPReversibleActivityBudget initial_budget =
        bp_reversible_activity_budget(initial, graph);
    result.represented_activity = initial_budget.represented;
    result.omitted_domain_activity = initial_budget.omitted_domain;
    result.omitted_projection_activity = initial_budget.omitted_projection;
    result.omitted_inactive_activity = initial_budget.omitted_inactive;
    result.omitted_fraction = initial_budget.omitted_fraction();
    if (!(dt > 0.0) || graph.reactions.empty()) {
        return result;
    }
    if (p.bp_reversible_explicit_fast_path) {
        const std::vector<double> initial_rhs =
            bp_reversible_shell_rhs(initial, graph, p);
        const double explicit_limit =
            bp_reversible_positive_dt_limit(
                initial, initial_rhs, p.bp_collision_cfl);
        if (std::isinf(explicit_limit)
            || (explicit_limit > 0.0
                && dt <= explicit_limit)) {
            try {
                return bp_reversible_advance_shell_explicit_ssprk2(
                    initial, graph, dt, p, &initial_rhs);
            } catch (const std::runtime_error&) {
                // The first Euler bound does not guarantee positivity of the
                // second SSPRK stage.  The explicit routine advances a private
                // copy, so the original state remains available for the
                // conservative implicit reaction sweep below.
                result.explicit_fast_path_fallbacks = 1;
            }
        }
    }
    const double entropy_before = bp_reversible_shell_entropy(
        result.f, graph, p.bp_reversible_entropy_floor);
    // A forward half-sweep followed by the reverse ordering balances the
    // ordering of coupled reactions.  Each extent sees the populations left
    // by the preceding reaction, followed by a whole-shell entropy check.
    const double half_dt = 0.5 * dt;
    for (const BPReversibleReaction& reaction : graph.reactions) {
        ++result.reaction_updates;
        const double extent = bp_reversible_reaction_extent(
            result.f,
            reaction,
            graph.velocity_cell_volume,
            half_dt,
            p.bp_reversible_entropy_floor,
            &result.implicit_reaction_updates,
            &result.implicit_iterations);
        bp_reversible_apply_extent(
            result.f, reaction, graph.velocity_cell_volume, extent);
    }
    for (auto reaction = graph.reactions.rbegin();
         reaction != graph.reactions.rend();
         ++reaction) {
        ++result.reaction_updates;
        const double extent = bp_reversible_reaction_extent(
            result.f,
            *reaction,
            graph.velocity_cell_volume,
            half_dt,
            p.bp_reversible_entropy_floor,
            &result.implicit_reaction_updates,
            &result.implicit_iterations);
        bp_reversible_apply_extent(
            result.f, *reaction, graph.velocity_cell_volume, extent);
    }
    validate_finite_nonnegative_distribution(
        result.f,
        graph.velocity_cell_volume.size(),
        "boltzmann_reversible implicit step output");
    const double entropy_after = bp_reversible_shell_entropy(
        result.f, graph, p.bp_reversible_entropy_floor);
    const double entropy_tolerance =
        p.bp_reversible_entropy_tolerance
        * std::max({1.0, std::abs(entropy_before), std::abs(entropy_after)});
    if (entropy_after > entropy_before + entropy_tolerance) {
        std::ostringstream message;
        message
            << "boltzmann_reversible implicit reaction sweep increased "
            << "the discrete entropy functional by "
            << entropy_after - entropy_before;
        throw std::runtime_error(message.str());
    }
    const BPReversibleActivityBudget final_budget =
        bp_reversible_activity_budget(result.f, graph);
    result.omitted_fraction = std::max(
        result.omitted_fraction,
        final_budget.omitted_fraction());
    result.substeps = 2;
    result.minimum_dt_limit = std::numeric_limits<double>::infinity();
    return result;
}

// Bound the omitted fraction globally and within shells carrying appreciable
// activity.  Radial weights follow this operator's 4*pi*r_center^2*dr rule.
template <typename ShellAccessor>
inline void bp_reversible_validate_omission_budgets(
    int shell_count,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    ShellAccessor&& shell_at) {
    double maximum_weighted_activity = 0.0;
    double global_attempted_activity = 0.0;
    double global_omitted_activity = 0.0;
    std::vector<double> weighted_activity(
        static_cast<std::size_t>(shell_count), 0.0);
    for (int i = 0; i < shell_count; ++i) {
        const BPReversibleShellResult& shell = shell_at(i);
        const double radius =
            geometry.radial.center[static_cast<std::size_t>(i)];
        const double shell_volume =
            4.0 * PI * radius * radius
            * geometry.radial.width[static_cast<std::size_t>(i)];
        const double attempted =
            shell.represented_activity
            + shell.omitted_domain_activity
            + shell.omitted_projection_activity
            + shell.omitted_inactive_activity;
        const double omitted =
            shell.omitted_domain_activity
            + shell.omitted_projection_activity
            + shell.omitted_inactive_activity;
        weighted_activity[static_cast<std::size_t>(i)] =
            shell_volume * attempted;
        maximum_weighted_activity = std::max(
            maximum_weighted_activity,
            weighted_activity[static_cast<std::size_t>(i)]);
        global_attempted_activity += shell_volume * attempted;
        global_omitted_activity += shell_volume * omitted;
    }
    const double global_fraction =
        (global_attempted_activity > 0.0)
        ? global_omitted_activity / global_attempted_activity
        : 0.0;
    if (global_fraction > p.bp_boltzmann_remap_max_skipped_fraction) {
        std::ostringstream message;
        message
            << "boltzmann_reversible globally omitted "
            << global_fraction
            << " of the volume-integrated collision activity; allowed "
            << p.bp_boltzmann_remap_max_skipped_fraction
            << ". Enlarge w_max/L_max or the projection candidate radius.";
        throw std::runtime_error(message.str());
    }
    const double active_floor =
        p.bp_reversible_active_shell_floor_fraction
        * maximum_weighted_activity;
    for (int i = 0; i < shell_count; ++i) {
        const BPReversibleShellResult& shell = shell_at(i);
        if (weighted_activity[static_cast<std::size_t>(i)] >= active_floor
            && shell.omitted_fraction
                > p.bp_boltzmann_remap_max_skipped_fraction) {
            std::ostringstream message;
            message
                << "boltzmann_reversible omitted "
                << shell.omitted_fraction
                << " of collision activity in active shell r="
                << geometry.radial.center[static_cast<std::size_t>(i)]
                << "; allowed "
                << p.bp_boltzmann_remap_max_skipped_fraction
                << ". Enlarge w_max/L_max or the projection candidate radius.";
            throw std::runtime_error(message.str());
        }
    }
}

/// Applies the reversible finite-angle operator independently in every radial shell.
inline BPCollisionStepInfo apply_bp_reversible_boltzmann_collision_step(
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
    if (!using_bp_reversible_boltzmann(p)
        || !(dt > 0.0)
        || p.bp_collision_strength == 0.0) {
        return info;
    }
    if (mpi.size != 1) {
        throw std::runtime_error("boltzmann_reversible is OpenMP-only and never uses MPI.");
    }
    if (tracer_f != nullptr && !tracer_f->empty()) {
        throw std::runtime_error(
            "boltzmann_reversible passive-tracer collisions are not implemented.");
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
        "boltzmann_reversible");
    validate_finite_nonnegative_distribution(
        f,
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL,
        "boltzmann_reversible input");
    const std::vector<double> eta = geometry.eta.center;
    // Each radial shell is local in velocity space.  Compute private shell
    // results first and validate omission budgets before publishing them to f.
    std::vector<BPReversibleShellResult> shell_result(static_cast<std::size_t>(p.Nr));
    parallel_for_dynamic_chunks(
        0,
        p.Nr,
        effective_threads(p, p.Nr),
        1,
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                std::vector<double> shell_f(static_cast<std::size_t>(p.Nw * p.NL), 0.0);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        shell_f[static_cast<std::size_t>(
                            bp_reversible_velocity_index(j, k, p))] =
                            f[idx(i, j, k, p)];
                    }
                }
                const BPReversibleGraph graph = bp_reversible_build_shell_graph(
                    shell_f,
                    r[static_cast<std::size_t>(i)],
                    w,
                    eta,
                    geometry,
                    p);
                shell_result[static_cast<std::size_t>(i)] =
                    bp_reversible_advance_shell(shell_f, graph, dt, p);
            }
        });
    bp_reversible_validate_omission_budgets(
        p.Nr,
        geometry,
        p,
        [&](int i) -> const BPReversibleShellResult& {
            return shell_result[static_cast<std::size_t>(i)];
        });
    double minimum_dt = std::numeric_limits<double>::infinity();
    for (int i = 0; i < p.Nr; ++i) {
        const BPReversibleShellResult& shell =
            shell_result[static_cast<std::size_t>(i)];
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                f[idx(i, j, k, p)] = shell.f[static_cast<std::size_t>(
                    bp_reversible_velocity_index(j, k, p))];
            }
        }
        info.substeps = std::max(info.substeps, shell.substeps);
        info.explicit_substeps = std::max(info.explicit_substeps, shell.substeps);
        info.reversible_reaction_updates += shell.reaction_updates;
        info.reversible_explicit_fast_path_fallbacks +=
            static_cast<std::uint64_t>(shell.explicit_fast_path_fallbacks);
        info.reversible_implicit_reaction_updates +=
            shell.implicit_reaction_updates;
        info.reversible_implicit_iterations += shell.implicit_iterations;
        info.max_shell_remap_skipped_fraction = std::max(
            info.max_shell_remap_skipped_fraction, shell.omitted_fraction);
        const double shell_volume =
            4.0 * PI
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.width[static_cast<std::size_t>(i)];
        info.remap_attempted_event_density +=
            shell_volume
            * (shell.represented_activity
               + shell.omitted_domain_activity
               + shell.omitted_projection_activity
               + shell.omitted_inactive_activity);
        info.remap_skipped_event_density +=
            shell_volume
            * (shell.omitted_domain_activity
               + shell.omitted_projection_activity
               + shell.omitted_inactive_activity);
        info.remap_skipped_domain_event_density +=
            shell_volume * shell.omitted_domain_activity;
        info.remap_skipped_projection_event_density +=
            shell_volume * shell.omitted_projection_activity;
        info.remap_skipped_inactive_event_density +=
            shell_volume * shell.omitted_inactive_activity;
        info.maximum_projection_error = std::max(
            info.maximum_projection_error,
            shell.maximum_projection_error);
        minimum_dt = std::min(minimum_dt, shell.minimum_dt_limit);
    }
    info.collision_dt = dt;
    info.explicit_dt_limit = minimum_dt;
    info.collision_time_estimate =
        info.explicit_dt_limit / std::max(p.bp_collision_cfl, 1.0e-300);
    return info;
}

}  // namespace ev
