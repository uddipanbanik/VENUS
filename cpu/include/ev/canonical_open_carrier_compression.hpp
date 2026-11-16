#pragma once

// Reduce the number of explicit open-orbit carriers by positive quadrature.
// A carrier is a numerical mass element with a continuous particle state;
// positive quadrature represents the same constrained moments with fewer
// weighted input points.
// Inputs are particle states, phase-space cells, a frozen potential, selected
// CIC/TSC deposition, and size/weight/error controls. Group nearby phase-space
// points sharing their deposition stencil, then vary only their positive
// masses along directions that preserve the specified moment constraints.
// Outputs are retained input points with revised weights and full before/after
// moment, effective-count, and reduction diagnostics. Positions, velocities,
// eta, and specific internal states stay at their original sampled values.
// Constraints include actual deposited moments and first/second Hamiltonian
// moments; these define the quantities this representation change preserves.
// A poorly conditioned or unhelpful local reduction retains its original group.

#include "ev/phase_space_cells.hpp"
#include "ev/parallel.hpp"
#include "ev/weighted_vlasov_pic.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <tuple>

namespace ev {

// Controls set when to reduce, which physical neighborhoods may mix, and
// how tightly their deposited/energy moments and maximum weights are retained.
struct CanonicalOpenCarrierCompressionOptions {
    std::size_t trigger_markers = 65536;
    // Absolute floor on the total packet count, including inactive boundary
    // carriers.  Complete-group reductions retain at least this population.
    std::size_t minimum_total_markers = 0;
    int phase_subcells = 1;
    unsigned int threads = 1;
    bool preserve_source_cell = false;
    WeightedPICParticleShape particle_shape = WeightedPICParticleShape::cic;
    double minimum_reduction_fraction = 0.0;
    double relative_moment_tolerance = 2.0e-11;
    // Both endpoint choices preserve the constrained moments before
    // roundoff.  This option changes only which positive endpoint is chosen;
    // accepted reductions satisfy their moment constraints directly.
    bool minimize_retained_weight_variance = true;
    // Do not let a quadrature point become heavier than this multiple of the
    // heaviest input point in its local group.  A finite bound converts the
    // reduction into a bounded positive quadrature: points that hit the upper
    // bound are retained and the remaining affine reduction continues.  The
    // default preserves the original upper bound on numerical packet mass.
    double maximum_local_mass_amplification = 1.0;
};

// Separate the change in representation size and unequal-weight effective
// population from any physical capture, escape, or collision event.
struct CanonicalOpenCarrierCompressionDiagnostics {
    bool applied = false;
    std::size_t active_before = 0;
    std::size_t active_after = 0;
    std::size_t groups = 0;
    std::size_t groups_retained_for_conditioning = 0;
    std::size_t groups_retained_for_marker_floor = 0;
    double active_mass_before = 0.0;
    double active_mass_after = 0.0;
    double maximum_relative_moment_residual = 0.0;
    double maximum_attempted_group_residual = 0.0;
    double minimum_positive_mass_before =
        std::numeric_limits<double>::infinity();
    double maximum_mass_before = 0.0;
    double minimum_positive_mass_after =
        std::numeric_limits<double>::infinity();
    double maximum_mass_after = 0.0;
    double effective_sample_size_before = 0.0;
    double effective_sample_size_after = 0.0;
    double elapsed_seconds = 0.0;
};

namespace canonical_open_compression_detail {

// Each shape slot constrains mass, radial momentum, both kinetic moments,
// radial energy flux, eta, and internal energy. Two internal-dispersion and
// two Hamiltonian moments supplement those deposited constraints.
constexpr std::size_t deposited_moment_count = 7;
constexpr std::size_t maximum_shape_points = 3;
constexpr std::size_t internal_state_feature_count = 2;
constexpr std::size_t hamiltonian_feature_count = 2;
constexpr std::size_t feature_count =
    deposited_moment_count * maximum_shape_points
    + internal_state_feature_count
    + hamiltonian_feature_count;
using Feature = std::array<long double, feature_count>;

// A group shares local (r,w,eta) subcells and identical stencil cell IDs.
// The optional source ID additionally preserves explicit source ownership.
struct Key {
    int radial_cell = 0;
    int radial_subcell = 0;
    int radial_velocity_cell = 0;
    int radial_velocity_subcell = 0;
    int eta_cell = 0;
    int eta_subcell = 0;
    std::array<std::size_t, maximum_shape_points> shape_cell{};
    int shape_count = 1;
    std::uint64_t source_cell = 0;

    bool operator<(const Key& other) const {
        return std::tie(
                   radial_cell,
                   radial_subcell,
                   radial_velocity_cell,
                   radial_velocity_subcell,
                   eta_cell,
                   eta_subcell,
                   shape_cell,
                   shape_count,
                   source_cell)
            < std::tie(
                   other.radial_cell,
                   other.radial_subcell,
                   other.radial_velocity_cell,
                   other.radial_velocity_subcell,
                   other.eta_cell,
                   other.eta_subcell,
                   other.shape_cell,
                   other.shape_count,
                   other.source_cell);
    }
};

// Define deterministic record ordering for group reduction and the final
// retained sequence, including all physical and boundary-state tie breakers.
inline bool marker_identity_less(
    const WeightedPICMarker& first,
    const WeightedPICMarker& second) {
    return std::tie(
               first.id,
               first.source_cell,
               first.state,
               first.r,
               first.w,
               first.eta,
               first.mass,
               first.inner_boundary_speed,
               first.remaining_inner_time,
               first.internal_energy,
               first.internal_velocity_dispersion)
        < std::tie(
               second.id,
               second.source_cell,
               second.state,
               second.r,
               second.w,
               second.eta,
               second.mass,
               second.inner_boundary_speed,
               second.remaining_inner_time,
               second.internal_energy,
               second.internal_velocity_dispersion);
}

// Preserve distinct below/above-grid keys for open carriers. A point on
// the final face belongs to the final represented cell.
inline int containing_cell_with_overflow(
    double value,
    const std::vector<double>& edge) {
    if (value < edge.front()) {
        return -1;
    }
    if (value > edge.back()) {
        return static_cast<int>(edge.size()) - 1;
    }
    if (value == edge.back()) {
        return static_cast<int>(edge.size()) - 2;
    }
    return static_cast<int>(
        std::upper_bound(edge.begin(), edge.end(), value) - edge.begin()) - 1;
}

inline int subcell_index(
    double value,
    int cell,
    const std::vector<double>& edge,
    int subcells) {
    if (cell < 0 || cell + 1 >= static_cast<int>(edge.size())) {
        return 0;
    }
    const double fraction = std::clamp(
        (value - edge[static_cast<std::size_t>(cell)])
            / (edge[static_cast<std::size_t>(cell + 1)]
               - edge[static_cast<std::size_t>(cell)]),
        0.0,
        std::nextafter(1.0, 0.0));
    return std::min(
        subcells - 1, static_cast<int>(fraction * subcells));
}

// Inside the first face, split the spherical central region by r^3
// fraction; represented radial cells use their ordinary local coordinate.
inline int radial_subcell_index(
    double radius,
    int cell,
    const std::vector<double>& edge,
    int subcells) {
    if (cell < 0) {
        const double fraction = std::clamp(
            std::pow(radius / edge.front(), 3),
            0.0,
            std::nextafter(1.0, 0.0));
        return std::min(
            subcells - 1, static_cast<int>(fraction * subcells));
    }
    return subcell_index(radius, cell, edge, subcells);
}

template <class PotentialAtRadius>
// Construct each point's moment contribution per unit particle mass.
// Multiplying this vector by its adjustable mass gives the conserved
// linear constraints used by the positive quadrature reduction.
inline Feature feature(
    const WeightedPICMarker& marker,
    const WeightedPICRadialMesh& mesh,
    WeightedPICParticleShape particle_shape,
    PotentialAtRadius& potential_at_radius) {
    const long double w = marker.w;
    const long double w2 = w * w;
    const long double vt2 = marker.r > 0.0
        ? 2.0L * marker.eta
            / (static_cast<long double>(marker.r) * marker.r)
        : 0.0L;
    const long double energy_flux = 0.5L * (w2 + vt2) * w;
    const long double eta = marker.eta;
    const long double sigma = marker.internal_velocity_dispersion;
    const long double specific_internal =
        static_cast<long double>(marker.internal_energy) / marker.mass;
    // H=Phi+(w^2+v_t^2)/2 is specific orbital energy in the frozen field.
    // Its first and second mass moments constrain the open-orbit energy spread.
    const long double hamiltonian = potential_at_radius(marker.r)
        + 0.5L * (w2 + vt2);
    const std::array<long double, deposited_moment_count> moment{
        1.0L,
        w,
        w2,
        vt2,
        energy_flux,
        eta,
        specific_internal,
    };
    Feature result{};
    auto assign_shape_features = [&](const auto& stencil) {
        for (std::size_t d = 0; d < deposited_moment_count; ++d) {
            for (int slot = 0; slot < stencil.count; ++slot) {
                result[d * maximum_shape_points
                       + static_cast<std::size_t>(slot)] =
                    moment[d]
                    * static_cast<long double>(
                        stencil.weight[static_cast<std::size_t>(slot)]);
            }
        }
    };
    if (marker.r < mesh.edge.front()) {
        // Origin-parity deposition hands the cloud continuously from the
        // resolved face stencil to the hidden central carrier.  Slot zero is
        // that carrier; the remaining slots are the frozen face stencil.
        // Constraining them separately prevents representation compression
        // from changing gravity when retained unequal-mass markers sample a
        // different part of the handoff.
        if (particle_shape == WeightedPICParticleShape::cic) {
            const auto stencil =
                weighted_pic_cic_stencil_with_inner_carrier(marker.r, mesh);
            if (stencil.resolved.count + 1
                > static_cast<int>(maximum_shape_points)) {
                throw std::runtime_error(
                    "CIC inner-carrier compression stencil is too large.");
            }
            for (std::size_t d = 0; d < deposited_moment_count; ++d) {
                result[d * maximum_shape_points] =
                    moment[d] * stencil.hidden_weight;
                for (int slot = 0; slot < stencil.resolved.count; ++slot) {
                    result[d * maximum_shape_points
                           + 1 + static_cast<std::size_t>(slot)] =
                        moment[d] * stencil.resolved.weight[
                            static_cast<std::size_t>(slot)];
                }
            }
        } else {
            const auto stencil =
                weighted_pic_tsc_stencil_with_inner_carrier(marker.r, mesh);
            if (stencil.resolved.count + 1
                > static_cast<int>(maximum_shape_points)) {
                throw std::runtime_error(
                    "TSC inner-carrier compression stencil is too large.");
            }
            for (std::size_t d = 0; d < deposited_moment_count; ++d) {
                result[d * maximum_shape_points] =
                    moment[d] * stencil.hidden_weight;
                for (int slot = 0; slot < stencil.resolved.count; ++slot) {
                    result[d * maximum_shape_points
                           + 1 + static_cast<std::size_t>(slot)] =
                        moment[d] * stencil.resolved.weight[
                            static_cast<std::size_t>(slot)];
                }
            }
        }
    } else if (particle_shape == WeightedPICParticleShape::cic) {
        assign_shape_features(weighted_pic_cic_stencil(marker.r, mesh));
    } else {
        assign_shape_features(weighted_pic_tsc_stencil(marker.r, mesh));
    }
    constexpr std::size_t internal_offset =
        deposited_moment_count * maximum_shape_points;
    constexpr std::size_t hamiltonian_offset =
        internal_offset + internal_state_feature_count;
    // Groups are already local in (r,w,eta), so these two exact constraints
    // protect the lower moments of the marker-carried internal state without
    // tripling them over the deposition stencil.
    result[internal_offset] = sigma;
    result[internal_offset + 1] = w * sigma;
    result[hamiltonian_offset] = hamiltonian;
    result[hamiltonian_offset + 1] =
        hamiltonian * hamiltonian;
    return result;
}

// With one more point than feature constraints, find a vector of weight
// changes leaving all feature sums fixed. Row scaling improves conditioning
// before pivoted elimination and a single free-coordinate back substitution.
inline std::array<long double, feature_count + 1> null_vector(
    const std::array<Feature, feature_count + 1>& point) {
    constexpr std::size_t rows = feature_count;
    constexpr std::size_t columns = feature_count + 1;
    std::array<std::array<long double, columns>, rows> matrix{};
    for (std::size_t row = 0; row < rows; ++row) {
        long double scale = 0.0L;
        for (std::size_t column = 0; column < columns; ++column) {
            scale = std::max(scale, std::abs(point[column][row]));
        }
        if (!(scale > 0.0L)) {
            scale = 1.0L;
        }
        for (std::size_t column = 0; column < columns; ++column) {
            matrix[row][column] = point[column][row] / scale;
        }
    }

    // Select representable pivots in extended precision; dependent moment
    // constraints leave additional free columns in the nullspace solve.
    std::array<int, rows> pivot_column{};
    pivot_column.fill(-1);
    std::array<unsigned char, columns> is_pivot{};
    std::size_t rank = 0;
    const long double pivot_tolerance = 4096.0L
        * std::numeric_limits<long double>::epsilon();
    for (std::size_t column = 0; column < columns && rank < rows; ++column) {
        std::size_t pivot = rank;
        long double maximum = 0.0L;
        for (std::size_t row = rank; row < rows; ++row) {
            const long double candidate = std::abs(matrix[row][column]);
            if (candidate > maximum) {
                maximum = candidate;
                pivot = row;
            }
        }
        if (!(maximum > pivot_tolerance)) {
            continue;
        }
        std::swap(matrix[rank], matrix[pivot]);
        const long double divisor = matrix[rank][column];
        for (std::size_t q = column; q < columns; ++q) {
            matrix[rank][q] /= divisor;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            if (row == rank) {
                continue;
            }
            const long double multiplier = matrix[row][column];
            if (multiplier == 0.0L) {
                continue;
            }
            for (std::size_t q = column; q < columns; ++q) {
                matrix[row][q] -= multiplier * matrix[rank][q];
            }
        }
        pivot_column[rank] = static_cast<int>(column);
        is_pivot[column] = 1;
        ++rank;
    }

    std::size_t free_column = columns;
    for (std::size_t column = columns; column-- > 0;) {
        if (is_pivot[column] == 0) {
            free_column = column;
            break;
        }
    }
    if (free_column == columns) {
        throw std::runtime_error(
            "Open-carrier quadrature reduction found no null direction.");
    }
    std::array<long double, columns> result{};
    result[free_column] = 1.0L;
    for (std::size_t reverse = rank; reverse-- > 0;) {
        const std::size_t pivot = static_cast<std::size_t>(
            pivot_column[reverse]);
        long double sum = 0.0L;
        for (std::size_t column = pivot + 1; column < columns; ++column) {
            sum += matrix[reverse][column] * result[column];
        }
        result[pivot] = -sum;
    }
    return result;
}

struct ReducedGroup {
    std::vector<WeightedPICMarker> marker;
    double attempted_residual = 0.0;
    bool retained_for_conditioning = false;
};

template <class PotentialAtRadius>
// Process one neighborhood in small active sets. A null-direction endpoint
// either removes a zero-weight point or fixes a point at the mass ceiling;
// the remaining active subset then continues with the next input point.
inline ReducedGroup reduce_group(
    const std::vector<WeightedPICMarker>& marker,
    const std::vector<std::size_t>& index,
    const WeightedPICRadialMesh& mesh,
    WeightedPICParticleShape particle_shape,
    PotentialAtRadius& potential_at_radius,
    double relative_moment_tolerance,
    bool minimize_retained_weight_variance,
    double maximum_local_mass_amplification) {
    constexpr std::size_t maximum_points = feature_count;
    ReducedGroup result;
    if (index.size() <= maximum_points) {
        result.marker.reserve(index.size());
        for (std::size_t q : index) {
            result.marker.push_back(marker[q]);
        }
        return result;
    }

    std::vector<WeightedPICMarker> active;
    std::vector<Feature> active_feature;
    std::vector<WeightedPICMarker> fixed;
    std::array<long double, feature_count> before{};
    std::array<long double, feature_count> absolute_scale{};
    double maximum_input_mass = 0.0;
    for (std::size_t source : index) {
        maximum_input_mass = std::max(
            maximum_input_mass, marker[source].mass);
    }
    // Bound retained weight relative to this actual group's heaviest input,
    // limiting the unequal-weight change entering later sampled collisions.
    const long double maximum_output_mass =
        static_cast<long double>(maximum_input_mass)
        * maximum_local_mass_amplification;
    active.reserve(maximum_points + 1);
    active_feature.reserve(maximum_points + 1);
    fixed.reserve(index.size());
    for (std::size_t source : index) {
        active.push_back(marker[source]);
        active_feature.push_back(feature(
            marker[source], mesh, particle_shape, potential_at_radius));
        for (std::size_t d = 0; d < feature_count; ++d) {
            before[d] += static_cast<long double>(marker[source].mass)
                * active_feature.back()[d];
            absolute_scale[d] += static_cast<long double>(marker[source].mass)
                * std::abs(active_feature.back()[d]);
        }
        while (active.size() > maximum_points) {
            std::array<Feature, maximum_points + 1> point{};
            for (std::size_t q = 0; q < point.size(); ++q) {
                point[q] = active_feature[q];
            }
            // Along this null direction every constrained moment is fixed
            // in exact arithmetic. Find both endpoints allowed by positive
            // masses and the configured upper mass bound.
            const auto direction = null_vector(point);
            long double positive_step =
                std::numeric_limits<long double>::infinity();
            long double negative_step =
                std::numeric_limits<long double>::infinity();
            std::size_t positive_bound = active.size();
            std::size_t negative_bound = active.size();
            bool positive_bound_is_zero = false;
            bool negative_bound_is_zero = false;
            auto update_bound = [](
                long double candidate,
                std::size_t q,
                bool is_zero,
                long double& bound,
                std::size_t& bound_index,
                bool& bound_is_zero) {
                if (candidate < bound) {
                    bound = candidate;
                    bound_index = q;
                    bound_is_zero = is_zero;
                }
            };
            for (std::size_t q = 0; q < direction.size(); ++q) {
                const long double mass = active[q].mass;
                if (direction[q] < 0.0L) {
                    update_bound(
                        mass / -direction[q],
                        q,
                        true,
                        positive_step,
                        positive_bound,
                        positive_bound_is_zero);
                    update_bound(
                        (maximum_output_mass - mass) / -direction[q],
                        q,
                        false,
                        negative_step,
                        negative_bound,
                        negative_bound_is_zero);
                } else if (direction[q] > 0.0L) {
                    update_bound(
                        (maximum_output_mass - mass) / direction[q],
                        q,
                        false,
                        positive_step,
                        positive_bound,
                        positive_bound_is_zero);
                    update_bound(
                        mass / direction[q],
                        q,
                        true,
                        negative_step,
                        negative_bound,
                        negative_bound_is_zero);
                }
            }
            const long double progress_tolerance =
                std::max(
                    8192.0L
                        * std::numeric_limits<long double>::epsilon(),
                    0.01L * relative_moment_tolerance)
                * std::max(1.0e-300L, maximum_output_mass);
            const bool positive_valid = std::isfinite(positive_step)
                && positive_step > 0.0L;
            const bool negative_valid = std::isfinite(negative_step)
                && negative_step > 0.0L;
            if (!positive_valid && !negative_valid) {
                // The current null direction is blocked by points already at
                // the upper bound.  Freeze one such point at its existing
                // mass; this changes no moment and exposes a smaller affine
                // system for the remaining points.
                std::size_t freeze = active.size();
                for (std::size_t q = 0; q < active.size(); ++q) {
                    if (maximum_output_mass - active[q].mass
                        <= progress_tolerance) {
                        freeze = q;
                        break;
                    }
                }
                if (freeze >= active.size()) {
                    throw std::runtime_error(
                        "Bounded open-carrier quadrature made no progress.");
                }
                fixed.push_back(active[freeze]);
                active.erase(
                    active.begin() + static_cast<std::ptrdiff_t>(freeze));
                active_feature.erase(
                    active_feature.begin()
                    + static_cast<std::ptrdiff_t>(freeze));
                continue;
            }
            // At fixed total mass, smaller sum(m^2) gives a larger effective
            // count. Maximum/minimum weights provide deterministic tie breaks.
            auto endpoint_score = [&](long double candidate_step) {
                long double sum_square = 0.0L;
                long double maximum = 0.0L;
                long double minimum =
                    std::numeric_limits<long double>::infinity();
                for (std::size_t q = 0; q < direction.size(); ++q) {
                    const long double mass =
                        static_cast<long double>(active[q].mass)
                        + candidate_step * direction[q];
                    if (mass < -progress_tolerance
                        || mass > maximum_output_mass
                            + progress_tolerance) {
                        return std::tuple{
                            std::numeric_limits<long double>::infinity(),
                            std::numeric_limits<long double>::infinity(),
                            0.0L};
                    }
                    const long double clipped = std::clamp(
                        mass, 0.0L, maximum_output_mass);
                    sum_square += clipped * clipped;
                    maximum = std::max(maximum, clipped);
                    if (clipped > 0.0L) {
                        minimum = std::min(minimum, clipped);
                    }
                }
                return std::tuple{sum_square, maximum, -minimum};
            };
            long double step = positive_valid
                ? positive_step : -negative_step;
            std::size_t selected_bound = positive_valid
                ? positive_bound : negative_bound;
            bool selected_bound_is_zero = positive_valid
                ? positive_bound_is_zero : negative_bound_is_zero;
            // Prefer an endpoint removing a point. When both have the same
            // removal status, use the requested weight-variance criterion or
            // the smaller feasible step.
            if (positive_valid && negative_valid) {
                const bool positive_removes = positive_bound_is_zero;
                const bool negative_removes = negative_bound_is_zero;
                if (positive_removes != negative_removes) {
                    if (positive_removes) {
                        step = positive_step;
                        selected_bound = positive_bound;
                        selected_bound_is_zero = true;
                    } else {
                        step = -negative_step;
                        selected_bound = negative_bound;
                        selected_bound_is_zero = true;
                    }
                } else if (minimize_retained_weight_variance) {
                    const auto positive_score =
                        endpoint_score(positive_step);
                    const auto negative_score =
                        endpoint_score(-negative_step);
                    if (negative_score < positive_score) {
                        step = -negative_step;
                        selected_bound = negative_bound;
                        selected_bound_is_zero =
                            negative_bound_is_zero;
                    } else {
                        step = positive_step;
                        selected_bound = positive_bound;
                        selected_bound_is_zero =
                            positive_bound_is_zero;
                    }
                } else {
                    if (positive_step <= negative_step) {
                        step = positive_step;
                        selected_bound = positive_bound;
                        selected_bound_is_zero =
                            positive_bound_is_zero;
                    } else {
                        step = -negative_step;
                        selected_bound = negative_bound;
                        selected_bound_is_zero =
                            negative_bound_is_zero;
                    }
                }
            }
            if (selected_bound >= active.size()) {
                throw std::runtime_error(
                    "Bounded open-carrier quadrature lost its limiting point.");
            }

            for (std::size_t q = 0; q < direction.size(); ++q) {
                // Internal energy is extensive: scale it with the new mass
                // so the retained point's specific internal state stays fixed.
                const long double specific_internal =
                    static_cast<long double>(active[q].internal_energy)
                    / active[q].mass;
                long double mass =
                    static_cast<long double>(active[q].mass)
                    + step * direction[q];
                if (q == selected_bound) {
                    mass = selected_bound_is_zero
                        ? 0.0L : maximum_output_mass;
                }
                if (mass < 0.0L
                    && std::abs(mass) <= progress_tolerance) {
                    mass = 0.0L;
                }
                if (mass > maximum_output_mass
                    && mass - maximum_output_mass
                        <= progress_tolerance) {
                    mass = maximum_output_mass;
                }
                if (mass < 0.0L || mass > maximum_output_mass) {
                    throw std::runtime_error(
                        "Bounded open-carrier quadrature violated a mass bound.");
                }
                active[q].mass = static_cast<double>(mass);
                active[q].internal_energy = static_cast<double>(
                    mass * specific_internal);
            }
            // A zero-weight point leaves the quadrature. A ceiling-weight
            // point is retained unchanged in the fixed set, allowing the
            // remaining bounded affine problem to make further progress.
            if (selected_bound_is_zero) {
                active.erase(
                    active.begin()
                    + static_cast<std::ptrdiff_t>(selected_bound));
                active_feature.erase(
                    active_feature.begin()
                    + static_cast<std::ptrdiff_t>(selected_bound));
            } else {
                fixed.push_back(active[selected_bound]);
                active.erase(
                    active.begin()
                    + static_cast<std::ptrdiff_t>(selected_bound));
                active_feature.erase(
                    active_feature.begin()
                    + static_cast<std::ptrdiff_t>(selected_bound));
            }
        }
    }
    // Evaluate actual retained moments, including fixed ceiling points.
    // Compare against absolute feature sums so cancellation does not create
    // a misleadingly small normalization for the relative residual.
    std::array<long double, feature_count> after{};
    result.marker.reserve(fixed.size() + active.size());
    for (const WeightedPICMarker& value : fixed) {
        const Feature value_feature = feature(
            value, mesh, particle_shape, potential_at_radius);
        for (std::size_t d = 0; d < feature_count; ++d) {
            after[d] += static_cast<long double>(value.mass)
                * value_feature[d];
        }
        result.marker.push_back(value);
    }
    for (std::size_t q = 0; q < active.size(); ++q) {
        for (std::size_t d = 0; d < feature_count; ++d) {
            after[d] += static_cast<long double>(active[q].mass)
                * active_feature[q][d];
        }
        result.marker.push_back(active[q]);
    }
    for (std::size_t d = 0; d < feature_count; ++d) {
        const long double scale = std::max({
            1.0e-300L,
            absolute_scale[d],
            std::abs(before[d]),
            std::abs(after[d]),
        });
        result.attempted_residual = std::max(
            result.attempted_residual,
            static_cast<double>(std::abs(after[d] - before[d]) / scale));
    }
    // Reject this proposed local compression if numerical conditioning
    // prevents moment closure. The physical input group remains available.
    if (result.attempted_residual > relative_moment_tolerance) {
        result.retained_for_conditioning = true;
        result.marker.clear();
        result.marker.reserve(index.size());
        for (std::size_t q : index) {
            result.marker.push_back(marker[q]);
        }
    } else {
        if (result.marker.size() >= index.size()) {
            result.marker.clear();
            result.marker.reserve(index.size());
            for (std::size_t q : index) {
                result.marker.push_back(marker[q]);
            }
        }
    }
    return result;
}

inline long double relative_residual(
    long double before,
    long double after,
    long double absolute_scale) {
    return std::abs(after - before)
        / std::max({1.0e-300L, absolute_scale, std::abs(before), std::abs(after)});
}

}  // namespace canonical_open_compression_detail

/// Reduces active open carriers to a positive local quadrature.  The feature
/// vector includes the actual CIC or TSC shape weights, so deposition of
/// density, momentum, kinetic energy, radial energy flux, angular momentum,
/// and internal energy is unchanged to roundoff.  Every retained phase-space
/// point came from the input set; no position or velocity is averaged or moved.
template <class PotentialAtRadius>
inline CanonicalOpenCarrierCompressionDiagnostics
compress_canonical_open_carriers(
    std::vector<WeightedPICMarker>& marker,
    const PhaseSpaceCellGeometry& geometry,
    const CanonicalOpenCarrierCompressionOptions& options,
    PotentialAtRadius&& potential_at_radius) {
    using namespace canonical_open_compression_detail;
    if (options.phase_subcells < 1
        || options.threads < 1
        || (options.particle_shape != WeightedPICParticleShape::cic
            && options.particle_shape != WeightedPICParticleShape::tsc)
        || !(options.minimum_reduction_fraction >= 0.0)
        || !(options.minimum_reduction_fraction < 1.0)
        || !std::isfinite(options.minimum_reduction_fraction)
        || !(options.relative_moment_tolerance > 0.0)
        || !std::isfinite(options.relative_moment_tolerance)
        || !(options.maximum_local_mass_amplification >= 1.0)
        || !std::isfinite(options.maximum_local_mass_amplification)) {
        throw std::runtime_error(
            "Invalid canonical open-carrier compression options.");
    }
    CanonicalOpenCarrierCompressionDiagnostics diagnostics;
    const WeightedPICRadialMesh mesh = make_weighted_pic_radial_mesh(geometry);
    long double squared_mass_before = 0.0L;
    long double squared_mass_after = 0.0L;
    long double total_mass_after = 0.0L;
    // Restrict reduction to positive active particles within the modeled
    // radial domain. All other records pass through unchanged.
    auto compressible = [&](const WeightedPICMarker& value) {
        return value.state == WeightedPICMarkerState::active
            && value.mass > 0.0
            && value.r >= 0.0
            && value.r <= mesh.edge.back();
    };
    for (const WeightedPICMarker& value : marker) {
        validate_weighted_pic_marker(
            value, "Canonical open-carrier compression input");
        if (compressible(value)) {
            ++diagnostics.active_before;
            diagnostics.active_mass_before += value.mass;
            squared_mass_before +=
                static_cast<long double>(value.mass) * value.mass;
            diagnostics.minimum_positive_mass_before = std::min(
                diagnostics.minimum_positive_mass_before, value.mass);
            diagnostics.maximum_mass_before = std::max(
                diagnostics.maximum_mass_before, value.mass);
        }
    }
    if (squared_mass_before > 0.0L) {
        diagnostics.effective_sample_size_before = static_cast<double>(
            static_cast<long double>(diagnostics.active_mass_before)
            * diagnostics.active_mass_before / squared_mass_before);
    }
    // Below the trigger the carrier representation and all before/after
    // diagnostics coincide, avoiding unnecessary small-set reductions.
    if (diagnostics.active_before < options.trigger_markers) {
        diagnostics.active_after = diagnostics.active_before;
        diagnostics.active_mass_after = diagnostics.active_mass_before;
        diagnostics.minimum_positive_mass_after =
            diagnostics.minimum_positive_mass_before;
        diagnostics.maximum_mass_after = diagnostics.maximum_mass_before;
        diagnostics.effective_sample_size_after =
            diagnostics.effective_sample_size_before;
        return diagnostics;
    }

    // Build deterministic local neighborhoods before parallel work. Actual
    // deposition cell IDs are part of the key so a feature slot always
    // refers to the same physical shell within its reduction group.
    std::map<Key, std::vector<std::size_t>> group;
    std::vector<WeightedPICMarker> inactive;
    inactive.reserve(marker.size() - diagnostics.active_before);
    for (std::size_t q = 0; q < marker.size(); ++q) {
        const WeightedPICMarker& value = marker[q];
        if (!compressible(value)) {
            inactive.push_back(value);
            continue;
        }
        const int radial_cell = containing_cell_with_overflow(
            value.r, geometry.radial.edge);
        const int velocity_cell = containing_cell_with_overflow(
            value.w, geometry.radial_velocity.edge);
        const int eta_cell = containing_cell_with_overflow(
            value.eta, geometry.eta.edge);
        std::array<std::size_t, maximum_shape_points> shape_cell{};
        int shape_count = 0;
        if (value.r < mesh.edge.front()) {
            // Slot zero is the hidden carrier.  Subsequent slots identify the
            // resolved cells in the same smooth handoff used by deposition.
            // The negative radial-cell key already distinguishes the virtual
            // carrier from resolved cell zero.
            if (options.particle_shape == WeightedPICParticleShape::cic) {
                const auto stencil =
                    weighted_pic_cic_stencil_with_inner_carrier(
                        value.r, mesh);
                shape_count = stencil.resolved.count + 1;
                if (shape_count
                    > static_cast<int>(maximum_shape_points)) {
                    throw std::runtime_error(
                        "CIC inner-carrier compression key is too large.");
                }
                for (int slot = 0; slot < stencil.resolved.count; ++slot) {
                    shape_cell[1 + static_cast<std::size_t>(slot)] =
                        stencil.resolved.cell[
                            static_cast<std::size_t>(slot)];
                }
            } else {
                const auto stencil =
                    weighted_pic_tsc_stencil_with_inner_carrier(
                        value.r, mesh);
                shape_count = stencil.resolved.count + 1;
                if (shape_count
                    > static_cast<int>(maximum_shape_points)) {
                    throw std::runtime_error(
                        "TSC inner-carrier compression key is too large.");
                }
                for (int slot = 0; slot < stencil.resolved.count; ++slot) {
                    shape_cell[1 + static_cast<std::size_t>(slot)] =
                        stencil.resolved.cell[
                            static_cast<std::size_t>(slot)];
                }
            }
        } else if (options.particle_shape == WeightedPICParticleShape::cic) {
            const WeightedPICCICStencil stencil =
                weighted_pic_cic_stencil(value.r, mesh);
            shape_cell = {
                stencil.cell[0],
                stencil.cell[1],
                0,
            };
            shape_count = stencil.count;
        } else {
            const WeightedPICTSCStencil stencil =
                weighted_pic_tsc_stencil(value.r, mesh);
            shape_cell = stencil.cell;
            shape_count = stencil.count;
        }
        const Key key{
            radial_cell,
            radial_subcell_index(
                value.r,
                radial_cell,
                geometry.radial.edge,
                options.phase_subcells),
            velocity_cell,
            subcell_index(
                value.w,
                velocity_cell,
                geometry.radial_velocity.edge,
                options.phase_subcells),
            eta_cell,
            subcell_index(
                value.eta,
                eta_cell,
                geometry.eta.edge,
                options.phase_subcells),
            shape_cell,
            shape_count,
            options.preserve_source_cell ? value.source_cell : 0,
        };
        group[key].push_back(q);
    }
    std::sort(inactive.begin(), inactive.end(), marker_identity_less);
    for (auto& item : group) {
        std::sort(
            item.second.begin(),
            item.second.end(),
            [&](std::size_t first, std::size_t second) {
                return marker_identity_less(marker[first], marker[second]);
            });
    }

    std::array<long double, feature_count> before{};
    std::array<long double, feature_count> after{};
    std::array<long double, feature_count> absolute_scale{};
    std::vector<WeightedPICMarker> compressed;
    compressed.reserve(inactive.size() + feature_count * group.size());
    compressed.insert(
        compressed.end(),
        std::make_move_iterator(inactive.begin()),
        std::make_move_iterator(inactive.end()));
    if (group.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Open-carrier compression has too many independent groups.");
    }
    const auto reduction_begin = std::chrono::steady_clock::now();
    std::vector<const std::vector<std::size_t>*> group_index;
    group_index.reserve(group.size());
    for (const auto& item : group) {
        group_index.push_back(&item.second);
    }
    // Independent groups can be reduced in separate worker chunks. Final
    // assembly below follows the same ordered group sequence on the host.
    std::vector<ReducedGroup> reduced_group(group.size());
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(group.size()),
        static_cast<int>(options.threads),
        1,
        [&](int begin, int end, int) {
            for (int q = begin; q < end; ++q) {
                reduced_group[static_cast<std::size_t>(q)] = reduce_group(
                    marker,
                    *group_index[static_cast<std::size_t>(q)],
                    mesh,
                    options.particle_shape,
                    potential_at_radius,
                    options.relative_moment_tolerance,
                    options.minimize_retained_weight_variance,
                    options.maximum_local_mass_amplification);
            }
        });

    diagnostics.groups = group.size();
    // Choose complete reduced groups without crossing the total marker
    // floor. Largest savings are considered first with a stable group tie break.
    std::vector<unsigned char> use_reduced_group(group.size(), 0);
    std::vector<std::size_t> reduction_candidate;
    reduction_candidate.reserve(group.size());
    for (std::size_t q = 0; q < group_index.size(); ++q) {
        if (reduced_group[q].marker.size() > group_index[q]->size()) {
            throw std::runtime_error(
                "Open-carrier compression increased a local group.");
        }
        if (reduced_group[q].marker.size() < group_index[q]->size()) {
            reduction_candidate.push_back(q);
        }
    }
    std::sort(
        reduction_candidate.begin(),
        reduction_candidate.end(),
        [&](std::size_t first, std::size_t second) {
            const std::size_t first_saving =
                group_index[first]->size() - reduced_group[first].marker.size();
            const std::size_t second_saving =
                group_index[second]->size() - reduced_group[second].marker.size();
            return first_saving != second_saving
                ? first_saving > second_saving
                : first < second;
        });
    std::size_t selected_total_markers = marker.size();
    for (std::size_t q : reduction_candidate) {
        const std::size_t saving =
            group_index[q]->size() - reduced_group[q].marker.size();
        if (selected_total_markers >= saving
            && selected_total_markers - saving
                >= options.minimum_total_markers) {
            use_reduced_group[q] = 1;
            selected_total_markers -= saving;
        } else {
            ++diagnostics.groups_retained_for_marker_floor;
        }
    }
    for (std::size_t q = 0; q < group_index.size(); ++q) {
        diagnostics.maximum_attempted_group_residual = std::max(
            diagnostics.maximum_attempted_group_residual,
            reduced_group[q].attempted_residual);
        if (reduced_group[q].retained_for_conditioning) {
            ++diagnostics.groups_retained_for_conditioning;
        }
        for (std::size_t index : *group_index[q]) {
            const Feature value = feature(
                marker[index],
                mesh,
                options.particle_shape,
                potential_at_radius);
            for (std::size_t d = 0; d < feature_count; ++d) {
                before[d] += static_cast<long double>(marker[index].mass)
                    * value[d];
                absolute_scale[d] += static_cast<long double>(marker[index].mass)
                    * std::abs(value[d]);
            }
        }
        // Recompute each installed contribution from its retained point
        // and mass, closing the global feature and unequal-weight ledgers.
        const auto append_marker = [&](const WeightedPICMarker& value_marker) {
            const Feature value = feature(
                value_marker,
                mesh,
                options.particle_shape,
                potential_at_radius);
            for (std::size_t d = 0; d < feature_count; ++d) {
                after[d] += static_cast<long double>(value_marker.mass)
                    * value[d];
            }
            squared_mass_after +=
                static_cast<long double>(value_marker.mass)
                * value_marker.mass;
            total_mass_after += value_marker.mass;
            diagnostics.minimum_positive_mass_after = std::min(
                diagnostics.minimum_positive_mass_after, value_marker.mass);
            diagnostics.maximum_mass_after = std::max(
                diagnostics.maximum_mass_after, value_marker.mass);
            compressed.push_back(value_marker);
        };
        if (use_reduced_group[q] != 0) {
            for (const WeightedPICMarker& value_marker :
                 reduced_group[q].marker) {
                append_marker(value_marker);
            }
        } else {
            for (std::size_t index : *group_index[q]) {
                append_marker(marker[index]);
            }
        }
    }
    for (std::size_t d = 0; d < feature_count; ++d) {
        diagnostics.maximum_relative_moment_residual = std::max(
            diagnostics.maximum_relative_moment_residual,
            static_cast<double>(relative_residual(
                before[d], after[d], absolute_scale[d])));
    }
    diagnostics.active_after = compressed.size()
        - (marker.size() - diagnostics.active_before);
    diagnostics.active_mass_after = static_cast<double>(total_mass_after);
    if (squared_mass_after > 0.0L) {
        diagnostics.effective_sample_size_after = static_cast<double>(
            static_cast<long double>(diagnostics.active_mass_after)
            * diagnostics.active_mass_after / squared_mass_after);
    }
    // The assembled population must meet the same moment tolerance as
    // each accepted neighborhood before replacing the caller's records.
    if (diagnostics.maximum_relative_moment_residual
        > options.relative_moment_tolerance) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "Canonical open-carrier compression failed moment closure: "
                << "residual="
                << diagnostics.maximum_relative_moment_residual
                << ", tolerance=" << options.relative_moment_tolerance
                << '.';
        throw std::runtime_error(message.str());
    }
    // A caller may require a useful minimum size reduction. Insufficient
    // savings return the original representation with its original diagnostics.
    const double reduction_fraction = diagnostics.active_before > 0
        ? 1.0
            - static_cast<double>(diagnostics.active_after)
                / static_cast<double>(diagnostics.active_before)
        : 0.0;
    if (reduction_fraction < options.minimum_reduction_fraction) {
        diagnostics.active_after = diagnostics.active_before;
        diagnostics.active_mass_after = diagnostics.active_mass_before;
        diagnostics.minimum_positive_mass_after =
            diagnostics.minimum_positive_mass_before;
        diagnostics.maximum_mass_after = diagnostics.maximum_mass_before;
        diagnostics.effective_sample_size_after =
            diagnostics.effective_sample_size_before;
        diagnostics.elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - reduction_begin).count();
        return diagnostics;
    }
    std::sort(compressed.begin(), compressed.end(), marker_identity_less);
    // Publish only the fully checked, deterministically ordered quadrature.
    marker = std::move(compressed);
    diagnostics.applied = true;
    diagnostics.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_begin).count();
    return diagnostics;
}

}  // namespace ev
