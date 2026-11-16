#pragma once

// Construct the spherical phase-space grids and their integration measures.
// Inputs are the requested radial, radial-velocity and angular-momentum mesh
// parameters. Outputs include nodes, physical control-volume faces, cell
// widths and nodal trapezoid weights. Storage orders L fastest, then w, then r:
// q=(i*Nw+j)*NL+k. Nodal quadrature and cell-average geometry have separate
// measures and therefore separate face/width constructors.

#include "ev/common.hpp"

namespace ev {

/// Converts three-dimensional (r,w,L) indices into the row-major one-dimensional storage index.
std::size_t idx(int i, int j, int k, const Params& p) {
    return (static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)) * p.NL
        + static_cast<std::size_t>(k);
}

/// Builds n uniformly spaced grid points between two endpoints.
std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> x(n);
    const double dx = (b - a) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = a + dx * static_cast<double>(i);
    }
    return x;
}

/// Builds n logarithmically spaced positive grid points between two endpoints.
std::vector<double> logspace(double a, double b, int n) {
    std::vector<double> x(n);
    const double la = std::log(a);
    const double lb = std::log(b);
    const double dx = (lb - la) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(la + dx * static_cast<double>(i));
    }
    return x;
}

/// Converts a radial-grid name into the compact integer code stored in restart files.
// Stable integer codes identify mesh geometry in binary records. The
// inverse functions reject unknown codes so a restart retains its grid
// interpretation rather than silently choosing a different discretization.
int radial_grid_code(const std::string& name) {
    if (name == "log") {
        return 0;
    }
    if (name == "linear") {
        return 1;
    }
    if (name == "piecewise-linear") {
        return 2;
    }
    if (name == "custom") {
        return 3;
    }
    throw std::runtime_error("radial_grid must be 'log', 'linear', 'piecewise-linear', or 'custom'.");
}

/// Converts a restart-file radial-grid code back into a human-readable grid name.
std::string radial_grid_name(int code) {
    if (code == 0) {
        return "log";
    }
    if (code == 1) {
        return "linear";
    }
    if (code == 2) {
        return "piecewise-linear";
    }
    if (code == 3) {
        return "custom";
    }
    throw std::runtime_error("Restart file has unknown radial grid code.");
}

/// Converts the phase-space representation into the restart metadata code.
// The stored representation distinguishes samples f(r,w,L) from
// cell averages in the canonical eta=L^2/2 coordinate.
int kinetic_grid_code(const std::string& name) {
    if (name == "legacy_nodal") {
        return 0;
    }
    if (name == "cell_average_eta") {
        return 1;
    }
    throw std::runtime_error("Unknown kinetic_grid restart representation.");
}

/// Recovers the phase-space representation from restart metadata.
std::string kinetic_grid_name(int code) {
    if (code == 0) {
        return "legacy_nodal";
    }
    if (code == 1) {
        return "cell_average_eta";
    }
    throw std::runtime_error("Restart file has unknown kinetic grid code.");
}

/// Converts the canonical eta-cell mapping into restart metadata.
int kinetic_eta_grid_code(const std::string& name) {
    if (name == "uniform") {
        return 0;
    }
    if (name == "power") {
        return 1;
    }
    if (name == "composite_power") {
        return 2;
    }
    throw std::runtime_error("Unknown kinetic_eta_grid restart representation.");
}

/// Recovers the canonical eta-cell mapping from restart metadata.
std::string kinetic_eta_grid_name(int code) {
    if (code == 0) {
        return "uniform";
    }
    if (code == 1) {
        return "power";
    }
    if (code == 2) {
        return "composite_power";
    }
    throw std::runtime_error("Restart file has unknown kinetic eta grid code.");
}

/// Tests whether the requested initial data should be read from an EVICRWL1 file.
bool using_binary_initial_data(const Params& p) {
    return p.initial_data == "binary_rwl" || p.initial_data == "binary" || p.initial_data == "file";
}

// Grid construction.  Log grids resolve small r strongly but tighten the CFL
// condition; piecewise-linear grids give a controlled inner refinement without
// making the innermost spacing quite so punishing.
/// Constructs a radial grid with a refined inner linear segment and coarser outer segment.
std::vector<double> piecewise_linear_grid(const Params& p) {
    const double split = p.inner_refine_radius;
    const int inner_cells = std::clamp(p.inner_refine_cells, 1, p.Nr - 2);
    const int inner_points = inner_cells + 1;
    const int outer_cells = p.Nr - inner_points;
    if (outer_cells < 1) {
        throw std::runtime_error("piecewise-linear grid needs at least one outer cell.");
    }

    std::vector<double> r(p.Nr);
    const double inner_dr = (split - p.r_min) / static_cast<double>(inner_cells);
    for (int i = 0; i <= inner_cells; ++i) {
        r[i] = p.r_min + inner_dr * static_cast<double>(i);
    }
    // Keep the refinement join as one shared node. The remaining cells span
    // from that physical join to the outer boundary without duplicating it.
    const double outer_dr = (p.r_max - split) / static_cast<double>(outer_cells);
    for (int q = 1; q <= outer_cells; ++q) {
        r[inner_cells + q] = split + outer_dr * static_cast<double>(q);
    }
    return r;
}

/// Dispatches to the selected radial-grid constructor after reading user parameters.
std::vector<double> make_radial_grid(const Params& p) {
    if (p.radial_grid == "log") {
        return logspace(p.r_min, p.r_max, p.Nr);
    }
    if (p.radial_grid == "linear") {
        return linspace(p.r_min, p.r_max, p.Nr);
    }
    if (p.radial_grid == "piecewise-linear") {
        return piecewise_linear_grid(p);
    }
    if (p.radial_grid == "custom") {
        if (!p.restart_from.empty()) {
            if (p.custom_radial_coordinate == "log") {
                return logspace(p.r_min, p.r_max, p.Nr);
            }
            if (p.custom_radial_coordinate == "linear") {
                return linspace(p.r_min, p.r_max, p.Nr);
            }
        }
        throw std::runtime_error("A custom radial grid must be loaded from initial_data_file.");
    }
    throw std::runtime_error("radial_grid must be 'log', 'linear', 'piecewise-linear', or 'custom'.");
}

/// Returns the interpolation/CFL coordinate associated with one physical radius.
// Interpolation and CFL distances use the selected coordinate. On a
// logarithmic radial mesh, dx=dr/r, while the force and moments still
// refer to the physical radius r.
double radial_coordinate(double rp, const Params& p) {
    if (p.radial_grid == "log" || (p.radial_grid == "custom" && p.custom_radial_coordinate == "log")) {
        return std::log(rp);
    }
    return rp;
}

/// True when the selected interpolation coordinate is log(r).
bool using_log_radial_coordinate(const Params& p) {
    return p.radial_grid == "log" || (p.radial_grid == "custom" && p.custom_radial_coordinate == "log");
}

/// Maps a physical radial grid into the coordinate used for interpolation and CFL estimates.
std::vector<double> radial_coordinate_grid(const std::vector<double>& r, const Params& p) {
    std::vector<double> x(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) {
        x[i] = radial_coordinate(r[i], p);
    }
    return x;
}

/// Builds a sinh-stretched momentum grid concentrated around the chosen center.
std::vector<double> stretched_w_grid(double w_max, double stretch, double center, int n) {
    if (stretch <= 0.0) {
        return linspace(-w_max, w_max, n);
    }
    std::vector<double> w(n);
    // Map a uniform logical coordinate through sinh. The two branches
    // place the requested center at u=0 while preserving both momentum
    // endpoints even when the center is displaced from zero.
    const double denom = std::sinh(stretch);
    for (int i = 0; i < n; ++i) {
        const double u = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1);
        if (u < 0.0) {
            w[i] = center - (center + w_max) * std::sinh(stretch * (-u)) / denom;
        } else {
            w[i] = center + (w_max - center) * std::sinh(stretch * u) / denom;
        }
    }
    return w;
}

/// Computes trapezoidal quadrature weights for a nonuniform one-dimensional grid.
// For nodal data, each interior weight spans half the two neighboring
// intervals. Endpoint half weights complete the trapezoidal integral.
std::vector<double> quadrature_weights(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    if (n < 2) {
        throw std::runtime_error("Need at least two points for trapezoid quadrature.");
    }
    std::vector<double> w(n, 0.0);
    w.front() = 0.5 * (x[1] - x[0]);
    w.back() = 0.5 * (x[n - 1] - x[n - 2]);
    for (int i = 1; i < n - 1; ++i) {
        w[i] = 0.5 * (x[i + 1] - x[i - 1]);
    }
    return w;
}

/// Converts cell-center coordinates into monotone control-volume edges.
// Nodal control volumes end at the first and last supplied nodes.
// Interior faces lie halfway between adjacent nodes. A genuine Nr-cell
// representation uses its separately constructed Nr+1 physical faces.
std::vector<double> control_volume_edges(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    if (n < 2) {
        throw std::runtime_error("Need at least two points for control-volume edges.");
    }
    std::vector<double> edges(static_cast<std::size_t>(n + 1), 0.0);
    edges.front() = x.front();
    edges.back() = x.back();
    for (int i = 1; i < n; ++i) {
        edges[static_cast<std::size_t>(i)] = 0.5 * (x[static_cast<std::size_t>(i - 1)] + x[static_cast<std::size_t>(i)]);
    }
    return edges;
}

/// Builds the positive radial faces of a finite logarithmic or linear domain.
///
/// The conservative kinetic representation is cell based.  Thus Nr denotes
/// the number of radial control volumes and the returned array contains Nr+1
/// faces spanning [r_min,r_max].  In particular, r=0 is not part of a
/// logarithmic mesh and is never manufactured as an extra face.
std::vector<double> positive_radial_cell_edges(const Params& p) {
    if (p.Nr < 2 || !(p.r_min > 0.0) || !(p.r_max > p.r_min)
        || !(p.radial_grid == "log" || p.radial_grid == "linear")) {
        throw std::runtime_error("Invalid parameters for the positive radial cell grid.");
    }
    std::vector<double> edges(static_cast<std::size_t>(p.Nr + 1), 0.0);
    if (p.radial_grid == "log") {
        const double log_min = std::log(p.r_min);
        const double log_max = std::log(p.r_max);
        for (int q = 0; q <= p.Nr; ++q) {
            const double fraction =
                static_cast<double>(q) / static_cast<double>(p.Nr);
            edges[static_cast<std::size_t>(q)] =
                std::exp(log_min + fraction * (log_max - log_min));
        }
    } else {
        for (int q = 0; q <= p.Nr; ++q) {
            const double fraction =
                static_cast<double>(q) / static_cast<double>(p.Nr);
            edges[static_cast<std::size_t>(q)] =
                p.r_min + fraction * (p.r_max - p.r_min);
        }
    }
    // Preserve the user-specified physical boundaries bit-for-bit.  Computing
    // exp(log(r_min)) can otherwise move the first face by one ulp and make a
    // characteristic evaluated at r_min appear spuriously out of domain.
    edges.front() = p.r_min;
    edges.back() = p.r_max;
    return edges;
}

// Select physical faces from the declared representation before using
// them for conservative fluxes or phase-space integration.
std::vector<double> radial_control_volume_edges(const std::vector<double>& r, const Params& p) {
    if (using_cell_average_eta_grid(p)) {
        if (static_cast<int>(r.size()) != p.Nr) {
            throw std::runtime_error("Radial center count does not match the cell-average grid.");
        }
        return positive_radial_cell_edges(p);
    }
    std::vector<double> edges = control_volume_edges(r);
    return edges;
}

/// Returns the physical radial widths of the conservative phase-space cells.
std::vector<double> radial_control_volume_widths(const std::vector<double>& r, const Params& p) {
    const std::vector<double> edges = radial_control_volume_edges(r, p);
    std::vector<double> widths(r.size(), 0.0);
    for (std::size_t i = 0; i < r.size(); ++i) {
        widths[i] = edges[i + 1] - edges[i];
        if (!(widths[i] > 0.0)) {
            throw std::runtime_error("Conservative radial cells must have positive widths.");
        }
    }
    return widths;
}

/// Finds the smallest adjacent spacing in a monotone grid.
double min_spacing(const std::vector<double>& x) {
    double out = std::numeric_limits<double>::infinity();
    for (int i = 1; i < static_cast<int>(x.size()); ++i) {
        out = std::min(out, x[i] - x[i - 1]);
    }
    return out;
}
}  // namespace ev
