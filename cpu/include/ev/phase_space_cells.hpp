#pragma once

// Canonical cell geometry and positive quadrature for spherical kinetic theory.
// The stored distribution is averaged over dr dw deta, where w is radial velocity
// and eta=L^2/2. Angular integration then gives d^3v=2*pi*dw*deta/r^2 and the
// full spherical phase measure is 8*pi^2*dr*dw*deta. Faces, widths and arithmetic
// centres define the same control volumes for transport, collisions and budgets.
// Inputs are domain/grid parameters or validated array views. Outputs include
// quadrature rules, contiguous MPI slabs and compatible L and Lq arrays.

#include "ev/grids.hpp"

#include <array>
#include <string_view>

namespace ev {

/// One cell-centered axis and its control-volume faces and widths.
struct PhaseSpaceAxis {
    std::vector<double> center;
    std::vector<double> edge;
    std::vector<double> width;
};

/// Shared (r,w,eta) geometry used by transport, collisions, and ledgers.
struct PhaseSpaceCellGeometry {
    PhaseSpaceAxis radial;
    PhaseSpaceAxis radial_velocity;
    PhaseSpaceAxis eta;
    std::vector<double> L;
    std::vector<double> legacy_Lq;
};

inline PhaseSpaceAxis make_positive_radial_cell_axis(const Params& p);
inline PhaseSpaceAxis make_uniform_cell_axis(
    double lower, double upper, int cells);
inline PhaseSpaceAxis make_symmetric_stretched_cell_axis(
    double limit, double stretch, int cells);
inline std::vector<double> eta_centers_from_L(const std::vector<double>& L);

/// Gauss--Legendre nodes and weights on [-1,1] for cell-average quadrature.
inline const std::pair<std::vector<double>, std::vector<double>>&
phase_space_gauss_legendre_rule(int points) {
    using Rule = std::pair<std::vector<double>, std::vector<double>>;
    if (points < 1 || points > 12) {
        throw std::runtime_error(
            "Cell-average Gauss--Legendre quadrature supports one through twelve points.");
    }
    static const std::array<Rule, 13> rule = [] {
        std::array<Rule, 13> built;
        built[1] = {{0.0}, {2.0}};
        built[2] = {
            {
                -0.577350269189625764509148780501957456,
                0.577350269189625764509148780501957456,
            },
            {1.0, 1.0},
        };
        built[3] = {
            {
                -0.774596669241483377035853079956479922,
                0.0,
                0.774596669241483377035853079956479922,
            },
            {
                0.555555555555555555555555555555555556,
                0.888888888888888888888888888888888889,
                0.555555555555555555555555555555555556,
            },
        };
        built[4] = {
            {
                -0.861136311594052575223946488892809505,
                -0.339981043584856264802665759103244687,
                0.339981043584856264802665759103244687,
                0.861136311594052575223946488892809505,
            },
            {
                0.347854845137453857373063949221999407,
                0.652145154862546142626936050778000593,
                0.652145154862546142626936050778000593,
                0.347854845137453857373063949221999407,
            },
        };
        for (int count = 5; count <= 12; ++count) {
            std::vector<double> node(static_cast<std::size_t>(count), 0.0);
            std::vector<double> weight(static_cast<std::size_t>(count), 0.0);
            const int roots = (count + 1) / 2;
            // Legendre roots occur in symmetric pairs. Newton iteration and its derivative
            // give each positive quadrature weight; reflected nodes preserve that symmetry.
            for (int root = 0; root < roots; ++root) {
                double z = std::cos(
                    PI * (static_cast<double>(root) + 0.75)
                    / (static_cast<double>(count) + 0.5));
                double derivative = 0.0;
                for (int iteration = 0; iteration < 64; ++iteration) {
                    double previous = 1.0;
                    double current = z;
                    for (int degree = 2; degree <= count; ++degree) {
                        const double next =
                            ((2.0 * degree - 1.0) * z * current
                             - (degree - 1.0) * previous)
                            / static_cast<double>(degree);
                        previous = current;
                        current = next;
                    }
                    derivative = static_cast<double>(count)
                        * (z * current - previous) / (z * z - 1.0);
                    const double next_z = z - current / derivative;
                    if (std::abs(next_z - z)
                        <= 8.0 * std::numeric_limits<double>::epsilon()
                            * std::max(1.0, std::abs(z))) {
                        z = next_z;
                        break;
                    }
                    z = next_z;
                }
                double previous = 1.0;
                double current = z;
                for (int degree = 2; degree <= count; ++degree) {
                    const double next =
                        ((2.0 * degree - 1.0) * z * current
                         - (degree - 1.0) * previous)
                        / static_cast<double>(degree);
                    previous = current;
                    current = next;
                }
                derivative = static_cast<double>(count)
                    * (z * current - previous) / (z * z - 1.0);
                const double root_weight =
                    2.0 / ((1.0 - z * z) * derivative * derivative);
                const int left = root;
                const int right = count - 1 - root;
                node[static_cast<std::size_t>(left)] = -z;
                node[static_cast<std::size_t>(right)] = z;
                weight[static_cast<std::size_t>(left)] = root_weight;
                weight[static_cast<std::size_t>(right)] = root_weight;
            }
            built[static_cast<std::size_t>(count)] = {
                std::move(node), std::move(weight)};
        }
        return built;
    }();
    return rule[static_cast<std::size_t>(points)];
}

/// Left Gauss--Radau nodes and weights on [-1,1].  The rule contains x=-1
/// and is exact through degree 2N-2.  Canonical eta atlases use it only in the
/// first eta cell so positive interpolation covers eta=0 without duplicating
/// every internal cell face.
inline const std::pair<std::vector<double>, std::vector<double>>&
phase_space_gauss_radau_left_rule(int points) {
    using Rule = std::pair<std::vector<double>, std::vector<double>>;
    if (points < 1 || points > 12) {
        throw std::runtime_error(
            "Cell-average Gauss--Radau quadrature supports one through twelve points.");
    }
    static const std::array<Rule, 13> rule = [] {
        std::array<Rule, 13> built;
        built[1] = {{-1.0}, {2.0}};
        auto legendre = [](int degree, double x) {
            if (degree == 0) {
                return 1.0;
            }
            double previous = 1.0;
            double current = x;
            for (int order = 2; order <= degree; ++order) {
                const double next =
                    ((2.0 * order - 1.0) * x * current
                     - (order - 1.0) * previous)
                    / static_cast<double>(order);
                previous = current;
                current = next;
            }
            return current;
        };
        for (int count = 2; count <= 12; ++count) {
            std::vector<double> node;
            node.reserve(static_cast<std::size_t>(count));
            node.push_back(-1.0);
            // The left endpoint is fixed. Interior roots of P_N+P_(N-1) complete the
            // Radau rule, giving positive interpolation support at eta=0.
            auto radau_polynomial = [&](double x) {
                return legendre(count, x) + legendre(count - 1, x);
            };
            constexpr int scan_intervals = 16384;
            double left = -1.0 + 1.0e-10;
            double f_left = radau_polynomial(left);
            for (int interval = 1;
                 interval <= scan_intervals
                    && node.size() < static_cast<std::size_t>(count);
                 ++interval) {
                const double right = -1.0
                    + 2.0 * static_cast<double>(interval)
                        / static_cast<double>(scan_intervals);
                const double f_right = radau_polynomial(right);
                if (f_left * f_right < 0.0) {
                    double bracket_left = left;
                    double bracket_right = right;
                    double bracket_f_left = f_left;
                    for (int iteration = 0; iteration < 96; ++iteration) {
                        const double middle = 0.5
                            * (bracket_left + bracket_right);
                        const double f_middle = radau_polynomial(middle);
                        if (bracket_f_left * f_middle <= 0.0) {
                            bracket_right = middle;
                        } else {
                            bracket_left = middle;
                            bracket_f_left = f_middle;
                        }
                    }
                    node.push_back(0.5 * (bracket_left + bracket_right));
                }
                left = right;
                f_left = f_right;
            }
            if (node.size() != static_cast<std::size_t>(count)) {
                throw std::runtime_error(
                    "Failed to construct the Gauss--Radau quadrature nodes.");
            }
            std::vector<double> weight(static_cast<std::size_t>(count), 0.0);
            weight.front() = 2.0
                / (static_cast<double>(count) * static_cast<double>(count));
            for (int q = 1; q < count; ++q) {
                const double x = node[static_cast<std::size_t>(q)];
                const double polynomial = legendre(count - 1, x);
                weight[static_cast<std::size_t>(q)] = (1.0 - x)
                    / (static_cast<double>(count) * static_cast<double>(count)
                       * polynomial * polynomial);
            }
            built[static_cast<std::size_t>(count)] = {
                std::move(node), std::move(weight)};
        }
        return built;
    }();
    return rule[static_cast<std::size_t>(points)];
}

/// Right Gauss--Radau rule, obtained by reflecting the left rule.
inline const std::pair<std::vector<double>, std::vector<double>>&
phase_space_gauss_radau_right_rule(int points) {
    using Rule = std::pair<std::vector<double>, std::vector<double>>;
    if (points < 1 || points > 12) {
        throw std::runtime_error(
            "Cell-average Gauss--Radau quadrature supports one through twelve points.");
    }
    static const std::array<Rule, 13> rule = [] {
        std::array<Rule, 13> built;
        for (int count = 1; count <= 12; ++count) {
            const auto& left = phase_space_gauss_radau_left_rule(count);
            std::vector<double> node(static_cast<std::size_t>(count), 0.0);
            std::vector<double> weight(static_cast<std::size_t>(count), 0.0);
            for (int q = 0; q < count; ++q) {
                const std::size_t source = static_cast<std::size_t>(count - 1 - q);
                node[static_cast<std::size_t>(q)] = -left.first[source];
                weight[static_cast<std::size_t>(q)] = left.second[source];
            }
            built[static_cast<std::size_t>(count)] = {
                std::move(node), std::move(weight)};
        }
        return built;
    }();
    return rule[static_cast<std::size_t>(points)];
}

/// Builds widths after validating a strictly increasing face array.
inline std::vector<double> phase_space_widths(const std::vector<double>& edges) {
    if (edges.size() < 2) {
        throw std::runtime_error("A cell axis needs at least one control volume.");
    }
    std::vector<double> widths(edges.size() - 1, 0.0);
    for (std::size_t q = 0; q + 1 < edges.size(); ++q) {
        widths[q] = edges[q + 1] - edges[q];
        if (!(widths[q] > 0.0) || !std::isfinite(widths[q])) {
            throw std::runtime_error("Phase-space cell faces must be finite and strictly increasing.");
        }
    }
    return widths;
}

inline bool phase_space_nearly_equal(double first, double second) {
    const double scale = std::max({1.0, std::abs(first), std::abs(second)});
    return std::abs(first - second)
        <= 128.0 * std::numeric_limits<double>::epsilon() * scale;
}

/// Validates one cell axis, including its arithmetic cell centroids.
inline void validate_phase_space_axis(
    const PhaseSpaceAxis& axis,
    int cells,
    std::string_view name) {
    if (cells < 1
        || static_cast<int>(axis.center.size()) != cells
        || static_cast<int>(axis.width.size()) != cells
        || static_cast<int>(axis.edge.size()) != cells + 1) {
        throw std::runtime_error(
            std::string(name) + " phase-space axis has inconsistent dimensions.");
    }
    for (int q = 0; q < cells; ++q) {
        const double left = axis.edge[static_cast<std::size_t>(q)];
        const double right = axis.edge[static_cast<std::size_t>(q + 1)];
        const double width = axis.width[static_cast<std::size_t>(q)];
        const double center = axis.center[static_cast<std::size_t>(q)];
        if (!std::isfinite(left) || !std::isfinite(right)
            || !std::isfinite(width) || !std::isfinite(center)
            || !(right > left)
            || !phase_space_nearly_equal(width, right - left)
            || !phase_space_nearly_equal(center, 0.5 * (left + right))) {
            throw std::runtime_error(
                std::string(name)
                + " phase-space axis violates its face/width/centroid contract.");
        }
    }
}

/// Validates the shared positive-domain (r,w,eta) cell representation.
// Validate the coordinate measure together with the domain. In particular
// L*Lq must equal the eta-cell width used by every conservative mass ledger.
inline void validate_cell_average_eta_geometry(
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    std::string_view context = "cell-average eta") {
    validate_phase_space_axis(geometry.radial, p.Nr, "radial");
    validate_phase_space_axis(geometry.radial_velocity, p.Nw, "radial-velocity");
    validate_phase_space_axis(geometry.eta, p.NL, "eta");
    if (geometry.L.size() != static_cast<std::size_t>(p.NL)
        || geometry.legacy_Lq.size() != static_cast<std::size_t>(p.NL)) {
        throw std::runtime_error(
            std::string(context) + " geometry has inconsistent angular arrays.");
    }
    if (geometry.radial.edge.front() != p.r_min
        || geometry.radial.edge.back() != p.r_max
        || !phase_space_nearly_equal(
            geometry.radial_velocity.edge.front(), -p.w_max)
        || !phase_space_nearly_equal(
            geometry.radial_velocity.edge.back(), p.w_max)
        || geometry.eta.edge.front()
            < -128.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, 0.5 * p.L_max * p.L_max)
        || geometry.eta.edge.back()
            > 0.5 * p.L_max * p.L_max
                + 128.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, 0.5 * p.L_max * p.L_max)) {
        throw std::runtime_error(
            std::string(context) + " geometry does not match the requested domain.");
    }
    for (int k = 0; k < p.NL; ++k) {
        const double eta = geometry.eta.center[static_cast<std::size_t>(k)];
        const double L = geometry.L[static_cast<std::size_t>(k)];
        const double Lq = geometry.legacy_Lq[static_cast<std::size_t>(k)];
        if (!(L > 0.0) || !std::isfinite(L) || !std::isfinite(Lq)
            || !phase_space_nearly_equal(L * L, 2.0 * eta)
            || !phase_space_nearly_equal(
                L * Lq,
                geometry.eta.width[static_cast<std::size_t>(k)])) {
            throw std::runtime_error(
                std::string(context)
                + " geometry violates eta=L^2/2 or L dL=deta.");
        }
    }
}

/// Validates external array views against the shared cell geometry.
inline void validate_cell_average_eta_views(
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p,
    std::string_view context) {
    validate_cell_average_eta_geometry(geometry, p, context);
    if (r != geometry.radial.center
        || w != geometry.radial_velocity.center
        || L != geometry.L
        || wq != geometry.radial_velocity.width
        || Lq != geometry.legacy_Lq) {
        throw std::runtime_error(
            std::string(context)
            + " array views do not match the canonical cell-average geometry.");
    }
}

/// Reconstructs the local eta faces from the L/Lq views and validates them.
///
/// This supports an MPI eta slab, whose first eta face need not be zero, while
/// still enforcing continuity and L dL=deta inside the slab.
inline PhaseSpaceCellGeometry cell_average_eta_geometry_from_views(
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p,
    std::string_view context) {
    PhaseSpaceCellGeometry geometry;
    geometry.radial = make_positive_radial_cell_axis(p);
    geometry.radial_velocity = make_symmetric_stretched_cell_axis(
        p.w_max, p.w_stretch, p.Nw);
    geometry.L = L;
    geometry.legacy_Lq = Lq;
    geometry.eta.center = eta_centers_from_L(L);
    geometry.eta.width.resize(static_cast<std::size_t>(p.NL), 0.0);
    geometry.eta.edge.resize(static_cast<std::size_t>(p.NL + 1), 0.0);
    if (L.size() != static_cast<std::size_t>(p.NL)
        || Lq.size() != static_cast<std::size_t>(p.NL)) {
        throw std::runtime_error(
            std::string(context) + " has inconsistent angular array sizes.");
    }
    for (int k = 0; k < p.NL; ++k) {
        geometry.eta.width[static_cast<std::size_t>(k)] =
            L[static_cast<std::size_t>(k)] * Lq[static_cast<std::size_t>(k)];
    }
    // Recover the local left face from its first centroid and width. A distributed
    // angular slab can start at positive eta while retaining adjacent cell faces.
    geometry.eta.edge.front() =
        geometry.eta.center.front() - 0.5 * geometry.eta.width.front();
    for (int k = 0; k < p.NL; ++k) {
        geometry.eta.edge[static_cast<std::size_t>(k + 1)] =
            geometry.eta.edge[static_cast<std::size_t>(k)]
            + geometry.eta.width[static_cast<std::size_t>(k)];
    }
    validate_cell_average_eta_views(
        geometry, r, w, L, wq, Lq, p, context);
    return geometry;
}

/// Rejects malformed kinetic states instead of silently interpreting them as vacuum.
inline void validate_finite_nonnegative_distribution(
    const std::vector<double>& f,
    std::size_t expected_size,
    std::string_view context) {
    if (f.size() != expected_size) {
        throw std::runtime_error(
            std::string(context) + " distribution has the wrong size.");
    }
    for (double value : f) {
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(
                std::string(context)
                + " distribution must be finite and nonnegative.");
        }
    }
}

/// Constructs positive radial cells on [r_min,r_max].
inline PhaseSpaceAxis make_positive_radial_cell_axis(const Params& p) {
    PhaseSpaceAxis axis;
    axis.edge = positive_radial_cell_edges(p);
    axis.center.resize(static_cast<std::size_t>(p.Nr), 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        const double left = axis.edge[static_cast<std::size_t>(i)];
        const double right = axis.edge[static_cast<std::size_t>(i + 1)];
        // The canonical spherical kinetic measure is dr dw d( L^2/2 ).
        // Therefore a radial cell average is located at its arithmetic
        // dr-centroid even when its faces are logarithmically spaced.
        axis.center[static_cast<std::size_t>(i)] = 0.5 * (left + right);
    }
    axis.width = phase_space_widths(axis.edge);
    return axis;
}

/// Builds uniformly spaced cell centers and faces on a closed interval.
inline PhaseSpaceAxis make_uniform_cell_axis(double lower, double upper, int cells) {
    if (cells < 1 || !(upper > lower)) {
        throw std::runtime_error("Uniform cell axes need cells >= 1 and upper > lower.");
    }
    PhaseSpaceAxis axis;
    axis.edge.resize(static_cast<std::size_t>(cells + 1), 0.0);
    axis.center.resize(static_cast<std::size_t>(cells), 0.0);
    const double width = (upper - lower) / static_cast<double>(cells);
    for (int q = 0; q <= cells; ++q) {
        axis.edge[static_cast<std::size_t>(q)] = lower + width * static_cast<double>(q);
    }
    for (int q = 0; q < cells; ++q) {
        axis.center[static_cast<std::size_t>(q)] =
            0.5 * (axis.edge[static_cast<std::size_t>(q)]
                   + axis.edge[static_cast<std::size_t>(q + 1)]);
    }
    axis.width.assign(static_cast<std::size_t>(cells), width);
    return axis;
}

/// Builds symmetric sinh-stretched cells with extra resolution near w=0.
inline PhaseSpaceAxis make_symmetric_stretched_cell_axis(
    double limit,
    double stretch,
    int cells) {
    if (!(limit > 0.0) || !(stretch >= 0.0) || !std::isfinite(stretch)
        || cells < 2) {
        throw std::runtime_error(
            "Stretched cell axes need a positive limit, nonnegative stretch, "
            "and at least two cells.");
    }
    if (stretch == 0.0) {
        return make_uniform_cell_axis(-limit, limit, cells);
    }
    PhaseSpaceAxis axis;
    axis.edge.resize(static_cast<std::size_t>(cells + 1), 0.0);
    axis.center.resize(static_cast<std::size_t>(cells), 0.0);
    const double denominator = std::sinh(stretch);
    for (int q = 0; q <= cells; ++q) {
        const double coordinate =
            -1.0 + 2.0 * static_cast<double>(q) / static_cast<double>(cells);
        axis.edge[static_cast<std::size_t>(q)] =
            limit * std::sinh(stretch * coordinate) / denominator;
    }
    axis.edge.front() = -limit;
    axis.edge.back() = limit;
    axis.width = phase_space_widths(axis.edge);
    for (int q = 0; q < cells; ++q) {
        axis.center[static_cast<std::size_t>(q)] =
            0.5 * (axis.edge[static_cast<std::size_t>(q)]
                   + axis.edge[static_cast<std::size_t>(q + 1)]);
    }
    return axis;
}

/// Constructs the canonical eta grid and compatibility L/Lq arrays.
// Resolve low angular momenta with a power grid. The composite option joins
// that central power grid to logarithmic outer eta cells at the requested join.
inline PhaseSpaceAxis make_eta_axis(const Params& p) {
    const double L_max = p.L_max;
    const int cells = p.NL;
    if (!(L_max > 0.0)) {
        throw std::runtime_error("The eta cell grid requires L_max > 0.");
    }
    if (p.kinetic_eta_grid == "uniform") {
        return make_uniform_cell_axis(0.0, 0.5 * L_max * L_max, cells);
    }
    if ((p.kinetic_eta_grid != "power"
         && p.kinetic_eta_grid != "composite_power")
        || !(p.kinetic_eta_power >= 1.0)
        || !std::isfinite(p.kinetic_eta_power)) {
        throw std::runtime_error(
            "The eta cell grid requires kinetic_eta_grid="
            "uniform|power|composite_power and "
            "kinetic_eta_power >= 1.");
    }
    PhaseSpaceAxis axis;
    axis.edge.resize(static_cast<std::size_t>(cells + 1), 0.0);
    axis.center.resize(static_cast<std::size_t>(cells), 0.0);
    const double eta_max = 0.5 * L_max * L_max;
    if (p.kinetic_eta_grid == "power") {
        for (int q = 0; q <= cells; ++q) {
            const double fraction =
                static_cast<double>(q) / static_cast<double>(cells);
            axis.edge[static_cast<std::size_t>(q)] =
                eta_max * std::pow(fraction, p.kinetic_eta_power);
        }
    } else {
        const int inner_cells = p.kinetic_eta_inner_cells;
        const double eta_join = p.kinetic_eta_join;
        if (inner_cells < 1 || inner_cells >= cells
            || !(eta_join > 0.0) || !(eta_join < eta_max)
            || !std::isfinite(eta_join)) {
            throw std::runtime_error(
                "The composite eta grid requires 1 <= inner_cells < NL "
                "and 0 < eta_join < L_max^2/2.");
        }
        for (int q = 0; q <= inner_cells; ++q) {
            const double fraction = static_cast<double>(q)
                / static_cast<double>(inner_cells);
            axis.edge[static_cast<std::size_t>(q)] =
                eta_join * std::pow(fraction, p.kinetic_eta_power);
        }
        const int outer_cells = cells - inner_cells;
        const double ratio = eta_max / eta_join;
        for (int q = 1; q <= outer_cells; ++q) {
            const double fraction = static_cast<double>(q)
                / static_cast<double>(outer_cells);
            axis.edge[static_cast<std::size_t>(inner_cells + q)] =
                eta_join * std::pow(ratio, fraction);
        }
        axis.edge[static_cast<std::size_t>(inner_cells)] = eta_join;
        axis.edge.back() = eta_max;
    }
    axis.width = phase_space_widths(axis.edge);
    for (int q = 0; q < cells; ++q) {
        axis.center[static_cast<std::size_t>(q)] =
            0.5 * (axis.edge[static_cast<std::size_t>(q)]
                   + axis.edge[static_cast<std::size_t>(q + 1)]);
    }
    return axis;
}

/// Builds the geometry from an already constructed positive radial cell axis.
inline PhaseSpaceCellGeometry make_cell_average_eta_geometry(
    const PhaseSpaceAxis& radial_axis,
    const Params& p) {
    if (!using_cell_average_eta_grid(p)
        || static_cast<int>(radial_axis.center.size()) != p.Nr) {
        throw std::runtime_error("Invalid radial axis for the cell-average eta geometry.");
    }
    PhaseSpaceCellGeometry geometry;
    geometry.radial = radial_axis;
    geometry.radial_velocity = make_symmetric_stretched_cell_axis(
        p.w_max, p.w_stretch, p.Nw);
    geometry.eta = make_eta_axis(p);
    geometry.L.resize(static_cast<std::size_t>(p.NL), 0.0);
    // The compatibility arrays obey L=sqrt(2*eta) and Lq=Delta_eta/L. Their product
    // therefore reproduces the canonical angular measure exactly in the stored grid.
    geometry.legacy_Lq.resize(static_cast<std::size_t>(p.NL), 0.0);
    for (int k = 0; k < p.NL; ++k) {
        const double eta = geometry.eta.center[static_cast<std::size_t>(k)];
        geometry.L[static_cast<std::size_t>(k)] = std::sqrt(2.0 * eta);
        geometry.legacy_Lq[static_cast<std::size_t>(k)] =
            geometry.eta.width[static_cast<std::size_t>(k)]
            / geometry.L[static_cast<std::size_t>(k)];
    }
    validate_cell_average_eta_geometry(
        geometry, p, "constructed cell-average eta");
    return geometry;
}

/// Extracts a contiguous cell slab while retaining its bounding faces.
inline PhaseSpaceAxis slice_phase_space_axis(
    const PhaseSpaceAxis& global,
    int offset,
    int count) {
    if (offset < 0 || count < 1
        || offset + count > static_cast<int>(global.center.size())) {
        throw std::runtime_error("Invalid phase-space axis slab.");
    }
    PhaseSpaceAxis local;
    local.center.assign(
        global.center.begin() + offset,
        global.center.begin() + offset + count);
    local.edge.assign(
        global.edge.begin() + offset,
        global.edge.begin() + offset + count + 1);
    local.width.assign(
        global.width.begin() + offset,
        global.width.begin() + offset + count);
    return local;
}

/// Recovers eta cell centers from the equivalent angular-momentum centers.
inline std::vector<double> eta_centers_from_L(const std::vector<double>& L) {
    std::vector<double> eta(L.size(), 0.0);
    for (std::size_t k = 0; k < L.size(); ++k) {
        eta[k] = 0.5 * L[k] * L[k];
    }
    return eta;
}

/// Tests uniformity using a scale-aware tolerance.
inline bool phase_space_axis_is_uniform(
    const std::vector<double>& widths,
    double relative_tolerance = 128.0 * std::numeric_limits<double>::epsilon()) {
    if (widths.empty()) {
        return false;
    }
    const double reference = widths.front();
    for (double width : widths) {
        if (std::abs(width - reference)
            > relative_tolerance * std::max({1.0, std::abs(reference), std::abs(width)})) {
            return false;
        }
    }
    return true;
}

/// Returns the canonical per-cell d r d w d eta measure.
// Integrate a cell-average distribution with the coordinate cell volume.
// The global 8*pi^2 factor enters physical spherical budgets in the caller.
inline double canonical_cell_measure(
    const PhaseSpaceCellGeometry& geometry,
    int i,
    int j,
    int k) {
    return geometry.radial.width[static_cast<std::size_t>(i)]
        * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
        * geometry.eta.width[static_cast<std::size_t>(k)];
}

/// Returns the local physical velocity measure d^3v=2 pi r^-2 dw deta.
// At fixed r, integrate the azimuthal velocity angle to obtain 2*pi/r^2.
// Multiplying this measure by f gives a local density contribution.
inline double local_velocity_cell_measure(
    const PhaseSpaceCellGeometry& geometry,
    int i,
    int j,
    int k) {
    const double ri = geometry.radial.center[static_cast<std::size_t>(i)];
    return 2.0 * PI
        * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
        * geometry.eta.width[static_cast<std::size_t>(k)]
        / (ri * ri);
}

}  // namespace ev
