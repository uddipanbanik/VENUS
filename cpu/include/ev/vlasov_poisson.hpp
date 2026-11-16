#pragma once

#include "ev/central_drift.hpp"

// VENUS CPU weak-field Vlasov/Boltzmann-Poisson module.
// This header handles static and expanding backgrounds, spherical Poisson
// solves, VP semi-Lagrangian/finite-volume advection, and VP inflow examples.
//
// Field and transport order:
//   1. Evaluate the static or FLRW background and the physical/comoving
//      characteristic equations.
//   2. Integrate cell mass to enclosed mass and solve spherical Poisson gravity.
//   3. Provide raw, Hamiltonian, conservative-Hamiltonian, energy-shell, and
//      expanding-physical-velocity interpolation for nodal pullback.
//   4. Advance explicit conservative finite-volume fluxes when selected.
//   5. Supply outer-inflow states and boundary/escape diagnostics.
//   6. Dispatch the cell-integrated SLDG modules from solver_evolution.hpp.
//
// The variable w changes interpretation between static and expanding modes;
// keep every interpolation, force, timestep, and output label consistent with
// the convention stated by VPBackground.

#include "ev/advection.hpp"
#include "ev/phase_space_cells.hpp"

namespace ev {

/// FLRW background factors used by the weak-field Vlasov-Poisson module.
// Newtonian spherical gravity uses physical radius r and radial peculiar
// velocity w=dr/dt-H*r. Static background has a=1,H=0. The expanding power-law
// background supplies H and a separately from the self-consistent peculiar field.
struct VPBackground {
    double tau = 1.0;
    double a = 1.0;
    double H = 0.0;
    double rho_bar = 0.0;
};

/// Stores the spherical peculiar potential reconstructed from Poisson's equation.
struct VlasovPoissonField {
    std::vector<double> source_mass;
    std::vector<double> phi;
    std::vector<double> phi_r;
    std::vector<double> radial_edge;
    std::vector<double> source_mass_edge;
    std::vector<double> source_line_density;
    std::vector<double> source_line_density_linear;
    std::vector<double> source_line_density_quadratic;
    std::vector<double> phi_edge;
    VPBackground background;
    double matter_mass = 0.0;
    double outer_source_mass = 0.0;
    double hidden_carrier_mass = 0.0;
    double fixed_central_mass = 0.0;
    double min_gap = 1.0;
};

/// One fixed-L projection of the distribution onto frozen VP energy shells.
struct EnergyShellTable {
    std::vector<double> h;
    std::vector<double> f;
};

/// Holds radial potential data interpolated at one VP characteristic footpoint.
struct VPSample {
    double phi = 0.0;
    double phi_r = 0.0;
    int radial_cell = -1;
};

/// Evaluates a(t), H(t), and rho_bar(t) for the selected static or expanding VP background.
VPBackground vp_background(double t, const Params& p) {
    VPBackground bg;
    bg.tau = t + p.vp_time_offset;
    if (!using_expanding_vp(p)) {
        return bg;
    }
    bg.H = p.vp_alpha / bg.tau;
    bg.a = std::pow(bg.tau / p.vp_time_offset, p.vp_alpha);
    bg.rho_bar = 3.0 * bg.H * bg.H / (8.0 * PI * p.vp_gravity);
    return bg;
}

/// Integrates f to form density, radial/tangential pressure, and peculiar current.
// Integrate f with local velocity measure 2*pi*L*dL*dw/r^2. Density uses
// unit mass weight; p and pt are radial and summed tangential velocity second
// moments, and j is peculiar radial mass current.
Moments compute_vp_moments(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    const std::size_t expected =
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    if (using_cell_average_eta_grid(p)) {
        (void)cell_average_eta_geometry_from_views(
            r, w, L, wq, Lq, p, "cell-average Poisson moments");
        validate_finite_nonnegative_distribution(
            f, expected, "cell-average Poisson moments");
    } else if (f.size() != expected
               || r.size() != static_cast<std::size_t>(p.Nr)
               || w.size() != static_cast<std::size_t>(p.Nw)
               || L.size() != static_cast<std::size_t>(p.NL)
               || wq.size() != w.size()
               || Lq.size() != L.size()) {
        throw std::runtime_error(
            "Vlasov--Poisson moments received incompatible grid arrays.");
    }
    Moments out;
    out.rho.assign(p.Nr, 0.0);
    out.p.assign(p.Nr, 0.0);
    out.pt.assign(p.Nr, 0.0);
    out.j.assign(p.Nr, 0.0);

    const int work = (p.NL > 0) ? p.Nr * p.Nw : 0;
    const int nthreads = effective_threads(p, work);
    const std::size_t stride = static_cast<std::size_t>(p.Nr);
    std::vector<double> rho_partial(static_cast<std::size_t>(nthreads) * stride, 0.0);
    std::vector<double> p_partial(static_cast<std::size_t>(nthreads) * stride, 0.0);
    std::vector<double> pt_partial(static_cast<std::size_t>(nthreads) * stride, 0.0);
    std::vector<double> j_partial(static_cast<std::size_t>(nthreads) * stride, 0.0);
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            const std::size_t base = static_cast<std::size_t>(tid) * stride;
            for (int q = lo; q < hi; ++q) {
                const int i = q / p.Nw;
                const int j = q - i * p.Nw;
                double srho = 0.0;
                double sp = 0.0;
                double spt = 0.0;
                double sj = 0.0;
                for (int k = 0; k < p.NL; ++k) {
                    const double weight = wq[j] * Lq[k] * L[k] * f[idx(i, j, k, p)];
                    const double vt = L[k] / r[i];
                    srho += weight;
                    sp += w[j] * w[j] * weight;
                    spt += vt * vt * weight;
                    sj += w[j] * weight;
                }
                rho_partial[base + static_cast<std::size_t>(i)] += srho;
                p_partial[base + static_cast<std::size_t>(i)] += sp;
                pt_partial[base + static_cast<std::size_t>(i)] += spt;
                j_partial[base + static_cast<std::size_t>(i)] += sj;
            }
        });

    for (int i = 0; i < p.Nr; ++i) {
        double srho = 0.0;
        double sp = 0.0;
        double spt = 0.0;
        double sj = 0.0;
        for (int tid = 0; tid < nthreads; ++tid) {
            const std::size_t q = static_cast<std::size_t>(tid) * stride + static_cast<std::size_t>(i);
            srho += rho_partial[q];
            sp += p_partial[q];
            spt += pt_partial[q];
            sj += j_partial[q];
        }
        const double factor = 2.0 * PI / (r[i] * r[i]);
        out.rho[i] = factor * srho;
        out.p[i] = factor * sp;
        out.pt[i] = factor * spt;
        out.j[i] = factor * sj;
    }
    return out;
}

/// Forms global Newtonian moments by summing local L-slab contributions over MPI ranks.
Moments compute_global_vp_moments(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& local_p,
    const MpiContext& mpi) {
    Moments out = compute_vp_moments(f, r, w, L, wq, Lq, local_p);
    mpi_allreduce_sum_vector(out.rho, mpi);
    mpi_allreduce_sum_vector(out.p, mpi);
    mpi_allreduce_sum_vector(out.pt, mpi);
    mpi_allreduce_sum_vector(out.j, mpi);
    return out;
}

/// Computes matter mass with the same inner-origin convention used by Poisson.
double vp_matter_mass(const Moments& moments, const std::vector<double>& r, const Params& p) {
    if (using_conservative_phase_space_cells(p)) {
        const std::vector<double> widths = radial_control_volume_widths(r, p);
        double mass = 0.0;
        for (int i = 0; i < static_cast<int>(r.size()); ++i) {
            mass += 4.0 * PI * r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)]
                * moments.rho[static_cast<std::size_t>(i)] * widths[static_cast<std::size_t>(i)];
        }
        return mass;
    }
    double mass = using_origin_parity_inner(p)
        ? 4.0 * PI * moments.rho.front() * r.front() * r.front() * r.front() / 3.0
        : 0.0;
    for (int i = 1; i < static_cast<int>(r.size()); ++i) {
        const double y0 = r[i - 1] * r[i - 1] * moments.rho[i - 1];
        const double y1 = r[i] * r[i] * moments.rho[i];
        mass += 4.0 * PI * 0.5 * (y0 + y1) * (r[i] - r[i - 1]);
    }
    return mass;
}

inline bool vp_solve_3x3(
    std::array<std::array<long double, 3>, 3> matrix,
    std::array<long double, 3> rhs,
    std::array<long double, 3>& solution) {
    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row) {
            if (std::abs(matrix[static_cast<std::size_t>(row)]
                                  [static_cast<std::size_t>(column)])
                > std::abs(matrix[static_cast<std::size_t>(pivot)]
                                    [static_cast<std::size_t>(column)])) {
                pivot = row;
            }
        }
        const long double scale = std::max({
            1.0L,
            std::abs(matrix[static_cast<std::size_t>(pivot)][0]),
            std::abs(matrix[static_cast<std::size_t>(pivot)][1]),
            std::abs(matrix[static_cast<std::size_t>(pivot)][2]),
        });
        if (std::abs(matrix[static_cast<std::size_t>(pivot)]
                           [static_cast<std::size_t>(column)])
            <= 256.0L * std::numeric_limits<long double>::epsilon() * scale) {
            return false;
        }
        if (pivot != column) {
            std::swap(
                matrix[static_cast<std::size_t>(pivot)],
                matrix[static_cast<std::size_t>(column)]);
            std::swap(
                rhs[static_cast<std::size_t>(pivot)],
                rhs[static_cast<std::size_t>(column)]);
        }
        const long double diagonal = matrix[static_cast<std::size_t>(column)]
            [static_cast<std::size_t>(column)];
        for (int row = column + 1; row < 3; ++row) {
            const long double factor = matrix[static_cast<std::size_t>(row)]
                [static_cast<std::size_t>(column)] / diagonal;
            for (int q = column; q < 3; ++q) {
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(q)] -= factor
                    * matrix[static_cast<std::size_t>(column)]
                            [static_cast<std::size_t>(q)];
            }
            rhs[static_cast<std::size_t>(row)] -=
                factor * rhs[static_cast<std::size_t>(column)];
        }
    }
    for (int row = 2; row >= 0; --row) {
        long double value = rhs[static_cast<std::size_t>(row)];
        for (int q = row + 1; q < 3; ++q) {
            value -= matrix[static_cast<std::size_t>(row)]
                           [static_cast<std::size_t>(q)]
                * solution[static_cast<std::size_t>(q)];
        }
        solution[static_cast<std::size_t>(row)] = value
            / matrix[static_cast<std::size_t>(row)]
                    [static_cast<std::size_t>(row)];
    }
    return true;
}

// Fit a quadratic line density from three neighbouring cell averages.
// Enforce the central average algebraically, then scale its deviations from
// that average until the polynomial is nonnegative throughout the cell.
inline std::array<double, 3> vp_positive_p2_line_mass(
    int cell,
    const std::vector<double>& edge,
    const std::vector<double>& average) {
    const int cells = static_cast<int>(average.size());
    if (cells < 3 || cell < 0 || cell >= cells) {
        return {average[static_cast<std::size_t>(cell)], 0.0, 0.0};
    }
    const int stencil_begin = std::clamp(cell - 1, 0, cells - 3);
    const long double origin = edge[static_cast<std::size_t>(cell)];
    std::array<std::array<long double, 3>, 3> matrix{};
    std::array<long double, 3> rhs{};
    for (int row = 0; row < 3; ++row) {
        const int sample = stencil_begin + row;
        const long double left =
            edge[static_cast<std::size_t>(sample)] - origin;
        const long double right =
            edge[static_cast<std::size_t>(sample + 1)] - origin;
        matrix[static_cast<std::size_t>(row)] = {
            1.0L,
            0.5L * (left + right),
            (left * left + left * right + right * right) / 3.0L,
        };
        rhs[static_cast<std::size_t>(row)] =
            average[static_cast<std::size_t>(sample)];
    }
    std::array<long double, 3> coefficient{};
    if (!vp_solve_3x3(matrix, rhs, coefficient)) {
        return {average[static_cast<std::size_t>(cell)], 0.0, 0.0};
    }
    const long double width = edge[static_cast<std::size_t>(cell + 1)]
        - edge[static_cast<std::size_t>(cell)];
    const long double mean = average[static_cast<std::size_t>(cell)];
    // Enforce the central-cell average algebraically before positivity
    // scaling, so roundoff in the 3x3 solve cannot perturb shell mass.
    coefficient[0] = mean - 0.5L * width * coefficient[1]
        - width * width * coefficient[2] / 3.0L;
    auto evaluate = [&](long double x) {
        return coefficient[0] + x * (coefficient[1] + x * coefficient[2]);
    };
    long double minimum = std::min(evaluate(0.0L), evaluate(width));
    if (coefficient[2] > 0.0L) {
        const long double vertex = -0.5L * coefficient[1] / coefficient[2];
        if (vertex > 0.0L && vertex < width) {
            minimum = std::min(minimum, evaluate(vertex));
        }
    }
    if (minimum < 0.0L) {
        const long double denominator = mean - minimum;
        const long double theta = denominator > 0.0L
            ? std::clamp(mean / denominator, 0.0L, 1.0L)
            : 0.0L;
        coefficient[0] = mean
            + theta * (coefficient[0] - mean);
        coefficient[1] *= theta;
        coefficient[2] *= theta;
    }
    return {
        static_cast<double>(coefficient[0]),
        static_cast<double>(coefficient[1]),
        static_cast<double>(coefficient[2]),
    };
}

/// Solves spherical Poisson after interpreting rho as total density or as a pre-subtracted contrast.
// Gauss law integrates the selected density source to enclosed source mass.
// Hidden central matter and a prescribed point mass enter its inner boundary;
// the peculiar potential is then matched to the exterior -G*M_source/r.
VlasovPoissonField solve_vlasov_poisson_field(
    const Moments& moments,
    const std::vector<double>& r,
    double t,
    const Params& p,
    double hidden_carrier_mass = 0.0) {
    if (!(hidden_carrier_mass >= 0.0)
        || !std::isfinite(hidden_carrier_mass)) {
        throw std::runtime_error(
            "Poisson solve received invalid delayed-return carrier mass.");
    }
    VlasovPoissonField field;
    field.background = vp_background(t, p);
    field.source_mass.assign(p.Nr, 0.0);
    field.phi.assign(p.Nr, 0.0);
    field.phi_r.assign(p.Nr, 0.0);
    field.hidden_carrier_mass = hidden_carrier_mass;
    field.fixed_central_mass = p.central_mass;
    field.matter_mass = vp_matter_mass(moments, r, p)
        + hidden_carrier_mass;

    // In an expanding total-density representation, subtract the homogeneous
    // background once. A density-contrast input already supplies the subtracted
    // source, including its central carried contribution.
    auto source_density = [&](int i) {
        double source = moments.rho[i];
        if (using_expanding_vp(p) && p.vp_poisson_source == "total_density") {
            source -= field.background.rho_bar;
        }
        return source;
    };

    if (using_conservative_phase_space_cells(p)) {
        const std::vector<double> edges = radial_control_volume_edges(r, p);
        field.radial_edge = edges;
        field.source_mass_edge.assign(static_cast<std::size_t>(p.Nr + 1), 0.0);
        field.source_line_density.assign(static_cast<std::size_t>(p.Nr), 0.0);
        field.source_line_density_linear.assign(
            static_cast<std::size_t>(p.Nr), 0.0);
        field.source_line_density_quadratic.assign(
            static_cast<std::size_t>(p.Nr), 0.0);
        std::vector<double> average_line_density(
            static_cast<std::size_t>(p.Nr), 0.0);
        for (int i = 0; i < p.Nr; ++i) {
            const double ri = r[static_cast<std::size_t>(i)];
            average_line_density[static_cast<std::size_t>(i)] =
                4.0 * PI * ri * ri * source_density(i);
        }
        // total_density includes the homogeneous background in the central
        // sphere as well as in resolved cells. density_contrast is already
        // subtracted everywhere, including the carried central mass.
        const double inner_background_mass = using_expanding_vp(p)
                && p.vp_poisson_source == "total_density"
            ? 4.0 * PI * field.background.rho_bar * edges.front()
                * edges.front() * edges.front() / 3.0
            : 0.0;
        field.source_mass_edge.front() =
            p.central_mass + hidden_carrier_mass - inner_background_mass;
        double completed_mass = field.source_mass_edge.front();
        for (int i = 0; i < p.Nr; ++i) {
            const std::array<double, 3> line =
                p.vp_poisson_reconstruction == "p2_line_mass"
                ? vp_positive_p2_line_mass(i, edges, average_line_density)
                : std::array<double, 3>{
                    average_line_density[static_cast<std::size_t>(i)],
                    0.0,
                    0.0,
                };
            field.source_line_density[static_cast<std::size_t>(i)] = line[0];
            field.source_line_density_linear[static_cast<std::size_t>(i)] =
                line[1];
            field.source_line_density_quadratic[static_cast<std::size_t>(i)] =
                line[2];
            const double left = edges[static_cast<std::size_t>(i)];
            const double center_offset = r[static_cast<std::size_t>(i)] - left;
            field.source_mass[static_cast<std::size_t>(i)] = completed_mass
                + center_offset * (
                    line[0] + center_offset
                        * (0.5 * line[1]
                           + center_offset * line[2] / 3.0));
            const double width = edges[static_cast<std::size_t>(i + 1)] - left;
            completed_mass += width * (
                line[0] + width
                    * (0.5 * line[1] + width * line[2] / 3.0));
            field.source_mass_edge[static_cast<std::size_t>(i + 1)] = completed_mass;
        }
        field.outer_source_mass = completed_mass;
    } else {
        field.source_mass[0] = p.central_mass + hidden_carrier_mass;
        if (using_origin_parity_inner(p)) {
            field.source_mass[0] += 4.0 * PI * source_density(0) * r[0] * r[0] * r[0] / 3.0;
        }
        for (int i = 1; i < p.Nr; ++i) {
            const double y0 = r[i - 1] * r[i - 1] * source_density(i - 1);
            const double y1 = r[i] * r[i] * source_density(i);
            field.source_mass[i] = field.source_mass[i - 1]
                + 4.0 * PI * 0.5 * (y0 + y1) * (r[i] - r[i - 1]);
        }
        field.outer_source_mass = field.source_mass.back();
    }

    for (int i = 0; i < p.Nr; ++i) {
        field.phi_r[i] = p.vp_gravity * field.source_mass[i] / (r[i] * r[i]);
    }
    if (using_conservative_phase_space_cells(p)) {
        // Integrate G*M(r)/r^2 analytically for the cell polynomial line density.
        // Using this same polynomial for mass, force and potential keeps their radial
        // reconstruction mutually consistent.
        auto integrated_force = [&](int i, double lower, double upper) {
            if (!(upper > lower) || !(lower > 0.0)) {
                throw std::runtime_error(
                    "Conservative Poisson potential integration requires 0 < lower < upper.");
            }
            const double left_edge =
                field.radial_edge[static_cast<std::size_t>(i)];
            const long double c0 =
                field.source_line_density[static_cast<std::size_t>(i)];
            const long double c1 =
                field.source_line_density_linear[static_cast<std::size_t>(i)];
            const long double c2 =
                field.source_line_density_quadratic[static_cast<std::size_t>(i)];
            const long double left = left_edge;
            const long double a = c0 - left * c1 + left * left * c2;
            const long double b = c1 - 2.0L * left * c2;
            const long double c = c2;
            const long double mass_offset =
                field.source_mass_edge[static_cast<std::size_t>(i)]
                - a * left - 0.5L * b * left * left
                - c * left * left * left / 3.0L;
            const long double lo = lower;
            const long double hi = upper;
            const long double integral =
                mass_offset * (1.0L / lo - 1.0L / hi)
                + a * std::log(hi / lo)
                + 0.5L * b * (hi - lo)
                + c * (hi * hi - lo * lo) / 6.0L;
            return static_cast<double>(p.vp_gravity * integral);
        };
        field.phi_edge.assign(static_cast<std::size_t>(p.Nr + 1), 0.0);
        field.phi_edge.back() =
            -p.vp_gravity * field.outer_source_mass / field.radial_edge.back();
        for (int i = p.Nr - 1; i >= 0; --i) {
            const double center = r[static_cast<std::size_t>(i)];
            const double right = field.radial_edge[static_cast<std::size_t>(i + 1)];
            field.phi[static_cast<std::size_t>(i)] =
                field.phi_edge[static_cast<std::size_t>(i + 1)]
                - integrated_force(i, center, right);
            const double left = field.radial_edge[static_cast<std::size_t>(i)];
            field.phi_edge[static_cast<std::size_t>(i)] =
                field.phi_edge[static_cast<std::size_t>(i + 1)]
                - integrated_force(i, left, right);
        }
    } else {
        field.phi.back() = -p.vp_gravity * field.source_mass.back() / r.back();
        for (int i = p.Nr - 2; i >= 0; --i) {
            const double dr = r[i + 1] - r[i];
            field.phi[i] = field.phi[i + 1]
                - 0.5 * (field.phi_r[i + 1] + field.phi_r[i]) * dr;
        }
    }
    return field;
}

/// Converts VP diagnostic fields into the existing profile-output column layout.
Metric vp_as_diagnostic_metric(const VlasovPoissonField& field, const Params& p) {
    Metric metric;
    metric.m = field.source_mass;
    metric.mu = field.phi;
    metric.lambda.assign(p.Nr, 0.0);
    metric.lambda_r.assign(p.Nr, 0.0);
    metric.mu_r = field.phi_r;
    metric.lambda_t.assign(p.Nr, 0.0);
    metric.gap.assign(p.Nr, 1.0);
    metric.min_gap = field.min_gap;
    return metric;
}

/// Returns the physical outer face for cell-average transport, or the last radial node for nodal transport.
inline double vp_radial_outer_boundary(const std::vector<double>& r, const Params& p) {
    return using_cell_average_eta_grid(p) ? p.r_max : r.back();
}

inline double vp_momentum_lower_boundary(const std::vector<double>& w, const Params& p) {
    return using_cell_average_eta_grid(p) ? -p.w_max : w.front();
}

inline double vp_momentum_upper_boundary(const std::vector<double>& w, const Params& p) {
    return using_cell_average_eta_grid(p) ? p.w_max : w.back();
}

/// Samples the VP radial force field at one physical radius.
// Sample the exact within-cell mass polynomial when available. The returned
// force and potential then derive from one source representation rather than
// independent interpolations that could give different orbit energies.
VPSample sample_vp_field(
    double rp,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const Params& p,
    int radial_cell_hint = -1) {
    (void)x_r;
    if (using_origin_parity_inner(p)) {
        rp = std::abs(rp);
    }

    if (!field.radial_edge.empty()
        && rp >= field.radial_edge.front()
        && rp <= field.radial_edge.back()) {
        int i = radial_cell_hint;
        if (i < 0
            || i >= static_cast<int>(r.size())
            || rp < field.radial_edge[static_cast<std::size_t>(i)]
            || rp > field.radial_edge[static_cast<std::size_t>(i + 1)]) {
            const auto upper_edge = std::upper_bound(
                field.radial_edge.begin(), field.radial_edge.end(), rp);
            i = std::clamp(
                static_cast<int>(
                    std::distance(field.radial_edge.begin(), upper_edge)) - 1,
                0,
                static_cast<int>(r.size()) - 1);
        }
        const double left = field.radial_edge[static_cast<std::size_t>(i)];
        const double right = field.radial_edge[static_cast<std::size_t>(i + 1)];
        const double c0 =
            field.source_line_density[static_cast<std::size_t>(i)];
        const double c1 = field.source_line_density_linear.empty()
            ? 0.0
            : field.source_line_density_linear[static_cast<std::size_t>(i)];
        const double c2 = field.source_line_density_quadratic.empty()
            ? 0.0
            : field.source_line_density_quadratic[static_cast<std::size_t>(i)];
        const double offset = rp - left;
        const double mass =
            field.source_mass_edge[static_cast<std::size_t>(i)]
            + offset * (c0 + offset * (0.5 * c1 + offset * c2 / 3.0));
        const double force = p.vp_gravity * mass / (rp * rp);
        const long double global_a = static_cast<long double>(c0)
            - static_cast<long double>(left) * c1
            + static_cast<long double>(left) * left * c2;
        const long double global_b = static_cast<long double>(c1)
            - 2.0L * left * c2;
        const long double global_c = c2;
        const long double mass_offset =
            field.source_mass_edge[static_cast<std::size_t>(i)]
            - global_a * left - 0.5L * global_b * left * left
            - global_c * left * left * left / 3.0L;
        const long double lo = rp;
        const long double hi = right;
        const double force_integral = static_cast<double>(p.vp_gravity * (
            mass_offset * (1.0L / lo - 1.0L / hi)
            + global_a * std::log(hi / lo)
            + 0.5L * global_b * (hi - lo)
            + global_c * (hi * hi - lo * lo) / 6.0L));
        return {
            field.phi_edge[static_cast<std::size_t>(i + 1)] - force_integral,
            force,
            i,
        };
    }

    if (rp < r.front()) {
        if (!using_origin_parity_inner(p)) {
            if (!field.radial_edge.empty() && rp < field.radial_edge.front()) {
                std::ostringstream message;
                message
                    << "A conservative VP characteristic was sampled at r="
                    << std::setprecision(17) << rp
                    << " below the finite radial face r_min="
                    << field.radial_edge.front()
                    << " before reflection.";
                throw std::runtime_error(message.str());
            }
            return {field.phi.front(), field.phi_r.front()};
        }
        // Continue the unresolved central sphere from its physical face.
        // A shell outside that face exerts zero force in an empty cavity.
        // Anchoring at the first center would include part of that shell
        // and introduce jumps in both force and potential at the face.
        const bool face_resolved = !field.radial_edge.empty();
        const double r0 = face_resolved ? field.radial_edge.front() : r.front();
        const double phi0 = face_resolved ? field.phi_edge.front() : field.phi.front();
        const double face_force = face_resolved
            ? p.vp_gravity * field.source_mass_edge.front() / (r0 * r0)
            : field.phi_r.front();
        const double central_force = (p.central_mass > 0.0)
            ? p.vp_gravity * p.central_mass / (r0 * r0)
            : 0.0;
        const double core_force_coefficient =
            (face_force - central_force) / r0;
        const double rc = std::max(rp, 1.0e-150);
        const double central_phi_offset = (p.central_mass > 0.0)
            ? p.vp_gravity * p.central_mass * (1.0 / r0 - 1.0 / rc)
            : 0.0;
        const double phi = phi0
            + central_phi_offset
            + 0.5 * core_force_coefficient * (rp * rp - r0 * r0);
        const double phi_r = (p.central_mass > 0.0)
            ? p.vp_gravity * p.central_mass / (rc * rc) + core_force_coefficient * rp
            : core_force_coefficient * rp;
        return {phi, phi_r};
    }

    const double outer_boundary = vp_radial_outer_boundary(r, p);
    if (rp > outer_boundary) {
        const double mass = field.outer_source_mass;
        return {
            -p.vp_gravity * mass / rp,
            p.vp_gravity * mass / (rp * rp),
        };
    }

    const auto upper = std::upper_bound(r.begin(), r.end(), rp);
    const int i = std::clamp(
        static_cast<int>(std::distance(r.begin(), upper)) - 1,
        0,
        static_cast<int>(r.size()) - 2);
    const double r0 = r[static_cast<std::size_t>(i)];
    const double r1 = r[static_cast<std::size_t>(i + 1)];
    const double dr = r1 - r0;
    const double s = std::clamp((rp - r0) / dr, 0.0, 1.0);
    const double phi0 = field.phi[static_cast<std::size_t>(i)];
    const double phi1 = field.phi[static_cast<std::size_t>(i + 1)];
    const double secant = (phi1 - phi0) / dr;
    const double derivative_limit = 3.0 * std::abs(secant);
    const double d0 = std::clamp(
        field.phi_r[static_cast<std::size_t>(i)],
        -derivative_limit,
        derivative_limit);
    const double d1 = std::clamp(
        field.phi_r[static_cast<std::size_t>(i + 1)],
        -derivative_limit,
        derivative_limit);
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    const double h10 = s3 - 2.0 * s2 + s;
    const double h01 = -2.0 * s3 + 3.0 * s2;
    const double h11 = s3 - s2;
    const double phi =
        h00 * phi0 + h10 * dr * d0 + h01 * phi1 + h11 * dr * d1;
    const double phi_r =
        ((6.0 * s2 - 6.0 * s) * phi0
            + (-6.0 * s2 + 6.0 * s) * phi1) / dr
        + (3.0 * s2 - 4.0 * s + 1.0) * d0
        + (3.0 * s2 - 2.0 * s) * d1;
    return {phi, phi_r};
}

/// Evaluates dr/dt = H r + u for the VP characteristics.
double vp_radial_velocity(double rp, double wp, const VlasovPoissonField& field) {
    return field.background.H * rp + wp;
}

/// Evaluates du/dt = L^2/r^3 - Phi_r - H u for the VP characteristics.
// Physical radial motion has centrifugal acceleration and peculiar gravity,
// while expansion adds -H*w to peculiar velocity. Eta fixes L throughout this
// collisionless characteristic.
double vp_radial_acceleration(
    double rp,
    double wp,
    double Lk,
    const VPSample& sample,
    const std::vector<double>& r,
    const VlasovPoissonField& field) {
    (void)r;
    const double rc = std::max(std::abs(rp), 1.0e-300);
    double centrifugal = 0.0;
    if (Lk != 0.0) {
        const double angular_squared = Lk * Lk;
        const double radius_cubed = rc * rc * rc;
        // Preserve ordinary-orbit arithmetic while evaluating finite
        // extreme-scale forces without 0/0 or infinity/infinity. The L=0
        // limit above has exactly zero centrifugal acceleration at r=0.
        const double tangential_velocity = Lk / rc;
        centrifugal = angular_squared > 0.0 && std::isfinite(angular_squared)
                && radius_cubed > 0.0 && std::isfinite(radius_cubed)
            ? angular_squared / radius_cubed
            : tangential_velocity * tangential_velocity / rc;
    }
    return centrifugal - sample.phi_r - field.background.H * wp;
}

/// Evaluates the frozen static-frame VP Hamiltonian at one phase-space point.
double vp_particle_hamiltonian(double rp, double wp, double Lk, double phi) {
    const double rc = std::max(rp, 1.0e-300);
    return phi + 0.5 * (wp * wp + (Lk / rc) * (Lk / rc));
}

struct VPUnregularizedAdvance {
    CharacteristicState state;
    VPSample initial_sample;
    VPSample final_sample;
    bool has_endpoint_samples = false;
};

/// Advances one VP characteristic by a signed physical interval without regularization.
// The static KDK path integrates the complete free central drift analytically
// between gravitational kicks. The other branches use the configured signed-time
// characteristic integrator in the supplied frozen field.
VPUnregularizedAdvance advance_vp_characteristic_unregularized_result(
    double ri,
    double wi,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double physical_dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan) {
    if (using_kick_drift_kick_integrator(p)) {
        const VPSample target_sample = sample_vp_field(ri, x_r, r, field, p);
        if (!using_expanding_vp(p)) {
            // Split the static Hamiltonian into Phi(r) and the complete free
            // central-motion kinetic term.  The drift below is the exact
            // radial projection of straight-line motion with conserved L;
            // it removes the explicit L^2/r^3 pericentre singularity from the
            // ordinary KDK path.  The frozen-Hamiltonian audit below still
            // sends any inaccurate orbit to the Levi--Civita integrator.
            const bool radial_origin = using_origin_parity_inner(p) && Lk == 0.0;
            if (ri == 0.0 && (!radial_origin || p.central_mass != 0.0)) {
                throw std::runtime_error(
                    "VP point KDK requires L=0 and zero central point mass at the origin.");
            }
            const double w_half =
                wi - 0.5 * physical_dt * target_sample.phi_r;
            const double radial_axis = ri + physical_dt * w_half;
            const double transverse_velocity = radial_origin ? 0.0
                : Lk / std::max(ri, 1.0e-300);
            const double tangential_axis =
                physical_dt * transverse_velocity;
            CharacteristicState y;
            y.r = std::hypot(radial_axis, tangential_axis);
            if (!std::isfinite(y.r)
                || (!(y.r > 0.0) && using_origin_parity_inner(p) && !radial_origin)
                || (radial_origin && p.central_mass != 0.0 && radial_axis <= 0.0)) {
                throw std::runtime_error(
                    "VP exact central KDK drift reached an invalid radius.");
            }
            if (radial_origin) {
                // Continue the signed Cartesian radial trajectory. An exact
                // origin endpoint stores the velocity after crossing in the
                // direction of integration, including backward tracing.
                y.w = radial_axis < 0.0 ? -w_half : w_half;
                if (radial_axis == 0.0 && physical_dt != 0.0)
                    y.w = std::copysign(std::abs(w_half), physical_dt);
            } else {
                y.w = y.r > 0.0 ? (w_half * radial_axis
                       + transverse_velocity * tangential_axis) / y.r : 0.0;
            }
            const double speed2 = w_half * w_half
                + transverse_velocity * transverse_velocity;
            const double nearest_time = speed2 > 0.0
                ? std::max(std::min(0.0, physical_dt),
                    std::min(std::max(0.0, physical_dt), -ri * w_half / speed2))
                : 0.0;
            const double nearest_radius = std::hypot(
                ri + nearest_time * w_half, nearest_time * transverse_velocity);
            const double outer = vp_radial_outer_boundary(r, p);
            if ((!using_origin_parity_inner(p) && nearest_radius <= p.r_min)
                || (using_reflecting_outer(p) && std::max(ri, y.r) >= outer)) {
                y = {ri, w_half};
                if (!advance_central_drift_with_walls(
                        y.r, y.w, Lk, physical_dt, 1.0 + std::abs(physical_dt),
                        0.0, p.r_min, !using_origin_parity_inner(p),
                        outer, using_reflecting_outer(p))) {
                    throw std::runtime_error("VP KDK wall event could not be resolved.");
                }
            }
            const VPSample dep_sample = sample_vp_field(
                y.r,
                x_r,
                r,
                field,
                p,
                target_sample.radial_cell);
            y.w -= 0.5 * physical_dt * dep_sample.phi_r;
            reflect_radial_boundaries(
                y.r,
                y.w,
                p.r_min,
                vp_radial_outer_boundary(r, p),
                p, field.background.H);
            return {y, target_sample, dep_sample, true};
        }
        const double a_target = vp_radial_acceleration(ri, wi, Lk, target_sample, r, field);
        const double w_half = wi + 0.5 * physical_dt * a_target;
        const double v_half = vp_radial_velocity(ri, w_half, field);
        CharacteristicState y{ri + physical_dt * v_half, w_half};
        reflect_radial_boundaries(y.r, y.w, p.r_min, vp_radial_outer_boundary(r, p), p, field.background.H);
        const VPSample dep_sample = sample_vp_field(
            y.r,
            x_r,
            r,
            field,
            p,
            target_sample.radial_cell);
        y.w += 0.5 * physical_dt
            * vp_radial_acceleration(y.r, y.w, Lk, dep_sample, r, field);
        reflect_radial_boundaries(y.r, y.w, p.r_min, vp_radial_outer_boundary(r, p), p, field.background.H);
        return {y, target_sample, dep_sample, true};
    }

    auto rhs = [&](const CharacteristicState& y) {
        const VPSample sample = sample_vp_field(y.r, x_r, r, field, p);
        return CharacteristicDerivative{
            vp_radial_velocity(y.r, y.w, field),
            vp_radial_acceleration(y.r, y.w, Lk, sample, r, field),
        };
    };
    auto project = [&](CharacteristicState& y) {
        reflect_radial_boundaries(
            y.r, y.w, p.r_min, vp_radial_outer_boundary(r, p), p, field.background.H);
    };

    const CharacteristicState target{ri, wi};
    if (plan.method == "runge_kutta") {
        return {
            characteristic_rk_advance(target, physical_dt, plan, rhs, project),
            {},
            {},
            false,
        };
    }
    if (plan.method == "extrapolated_midpoint") {
        return {
            characteristic_extrapolated_midpoint_advance(
                target, physical_dt, plan, rhs, project),
            {},
            {},
            false,
        };
    }
    if (using_adams_bashforth_characteristic_integrator(p)) {
        return {
            characteristic_adams_bashforth_advance(
                target, physical_dt, plan, rhs, project),
            {},
            {},
            false,
        };
    }
    throw std::runtime_error("Unknown semi-Lagrangian VP characteristic integrator: " + p.integrator);
}

CharacteristicState advance_vp_characteristic_unregularized(
    double ri,
    double wi,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double physical_dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan) {
    return advance_vp_characteristic_unregularized_result(
        ri, wi, Lk, x_r, r, field, physical_dt, p, plan).state;
}

double vp_frozen_hamiltonian_error_from_samples(
    const CharacteristicState& initial,
    const CharacteristicState& final,
    double Lk,
    const VPSample& initial_sample,
    const VPSample& final_sample) {
    const double h0 =
        vp_particle_hamiltonian(initial.r, initial.w, Lk, initial_sample.phi);
    const double h1 =
        vp_particle_hamiltonian(final.r, final.w, Lk, final_sample.phi);
    const double radius0 = std::max(initial.r, 1.0e-300);
    const double kinetic0 = 0.5 * (
        initial.w * initial.w + (Lk / radius0) * (Lk / radius0));
    const double scale = std::max(
        {std::abs(h0), std::abs(initial_sample.phi) + kinetic0, 1.0e-14});
    return std::abs(h1 - h0) / scale;
}

double vp_frozen_hamiltonian_error(
    const CharacteristicState& initial,
    const CharacteristicState& final,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const Params& p) {
    const VPSample initial_sample =
        sample_vp_field(initial.r, x_r, r, field, p);
    const VPSample final_sample =
        sample_vp_field(final.r, x_r, r, field, p);
    return vp_frozen_hamiltonian_error_from_samples(
        initial, final, Lk, initial_sample, final_sample);
}

/// Advances one VP/BP characteristic and retries stiff pericentres in LC coordinates.
// Audit ordinary static orbits against the frozen Hamiltonian. Difficult
// pericentre/wall passages receive a tighter characteristic retry. If its finite
// retry budget ends, return the best measured state and increment the unresolved
// diagnostic for the caller to inspect.
CharacteristicState advance_vp_characteristic(
    double ri,
    double wi,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double physical_dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    bool regularization_eligible = true,
    bool record_regularization_diagnostics = true) {
    const CharacteristicState initial{ri, wi};
    const VPUnregularizedAdvance ordinary_result =
        advance_vp_characteristic_unregularized_result(
        ri, wi, Lk, x_r, r, field, physical_dt, p, plan);
    const CharacteristicState& ordinary = ordinary_result.state;
    double ordinary_error = std::numeric_limits<double>::infinity();
    const bool audit_frozen_hamiltonian =
        using_vp_orbit_regularization(p) && !using_expanding_vp(p);
    if (audit_frozen_hamiltonian) {
        ordinary_error = ordinary_result.has_endpoint_samples
            ? vp_frozen_hamiltonian_error_from_samples(
                initial,
                ordinary,
                Lk,
                ordinary_result.initial_sample,
                ordinary_result.final_sample)
            : vp_frozen_hamiltonian_error(
                initial, ordinary, Lk, x_r, r, field, p);
        if (record_regularization_diagnostics) {
            vp_orbit_atomic_max_finite(
                vp_orbit_maximum_ordinary_energy_error,
                ordinary_error);
        }
    }
    const bool finite_inner_wall =
        !using_origin_parity_inner(p) && !using_expanding_vp(p);
    const VPSample initial_sample = ordinary_result.has_endpoint_samples
        ? ordinary_result.initial_sample
        : sample_vp_field(ri, x_r, r, field, p);
    bool regularized_path_stays_inside_domain = true;
    bool initially_moves_toward_inner_wall = false;
    if (finite_inner_wall) {
        const double inner_phi = !field.phi_edge.empty()
            ? field.phi_edge.front()
            : sample_vp_field(p.r_min, x_r, r, field, p).phi;
        const double initial_hamiltonian =
            vp_particle_hamiltonian(ri, wi, Lk, initial_sample.phi);
        const double inner_effective_potential =
            inner_phi + 0.5 * (Lk / p.r_min) * (Lk / p.r_min);
        const double barrier_scale = std::max({
            1.0,
            std::abs(initial_hamiltonian),
            std::abs(inner_effective_potential),
        });
        regularized_path_stays_inside_domain =
            initial_hamiltonian
            < inner_effective_potential
                - 16.0 * p.vp_orbit_energy_tolerance * barrier_scale;
        const double initial_radial_velocity =
            vp_radial_velocity(ri, wi, field);
        const double signed_radial_displacement =
            physical_dt * initial_radial_velocity;
        const double displacement_tolerance =
            64.0 * std::numeric_limits<double>::epsilon()
            * std::max({
                1.0,
                std::abs(ri),
                std::abs(signed_radial_displacement),
            });
        initially_moves_toward_inner_wall =
            signed_radial_displacement < -displacement_tolerance;
        if (std::abs(signed_radial_displacement)
            <= displacement_tolerance) {
            const double acceleration = vp_radial_acceleration(
                ri, wi, Lk, initial_sample, r, field);
            initially_moves_toward_inner_wall =
                0.5 * physical_dt * physical_dt * acceleration < 0.0;
        }
    }
    const bool can_use_levi_civita = Lk > 0.0;
    const bool can_use_wall_regularization = finite_inner_wall;
    if (!regularization_eligible
        || !using_vp_orbit_regularization(p)
        || (!can_use_levi_civita && !can_use_wall_regularization)
        || (using_reflecting_outer(p)
            && ri > 0.9 * vp_radial_outer_boundary(r, p))) {
        if (audit_frozen_hamiltonian
            && record_regularization_diagnostics) {
            vp_orbit_atomic_max_finite(
                vp_orbit_maximum_final_energy_error,
                ordinary_error);
        }
        return ordinary;
    }

    bool retry = p.vp_orbit_regularization == "always";
    if (!using_expanding_vp(p)) {
        retry = retry
            || !std::isfinite(ordinary_error)
            || ordinary_error > p.vp_orbit_energy_tolerance;
    } else if (!retry) {
        const double angular_phase =
            std::abs(physical_dt * Lk / (ri * ri));
        if (angular_phase > 0.25) {
            CharacteristicState fine = advance_vp_characteristic_unregularized(
                ri, wi, Lk, x_r, r, field, 0.5 * physical_dt, p, plan);
            fine = advance_vp_characteristic_unregularized(
                fine.r, fine.w, Lk, x_r, r, field, 0.5 * physical_dt, p, plan);
            const double local_error = std::hypot(
                (fine.r - ordinary.r) / std::max(ri, r.front()),
                (fine.w - ordinary.w) / std::max(1.0, std::abs(wi)));
            retry = local_error > p.vp_orbit_energy_tolerance;
        }
    }
    if (!retry) {
        if (record_regularization_diagnostics) {
            vp_orbit_atomic_max_finite(
                vp_orbit_maximum_final_energy_error,
                ordinary_error);
        }
        return ordinary;
    }
    auto rhs = [&](const CharacteristicState& y) {
        if (!std::isfinite(y.r) || !std::isfinite(y.w)) {
            const double invalid =
                std::numeric_limits<double>::quiet_NaN();
            return CharacteristicDerivative{invalid, invalid};
        }
        double radius = y.r;
        if (finite_inner_wall) {
            const double boundary_tolerance =
                64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, p.r_min);
            if (radius < p.r_min - boundary_tolerance) {
                const double invalid =
                    std::numeric_limits<double>::quiet_NaN();
                return CharacteristicDerivative{invalid, invalid};
            }
            radius = std::max(radius, p.r_min);
        }
        const VPSample sample = sample_vp_field(
            radius, x_r, r, field, p);
        return CharacteristicDerivative{
            vp_radial_velocity(radius, y.w, field),
            vp_radial_acceleration(radius, y.w, Lk, sample, r, field),
        };
    };

    CharacteristicState best = ordinary;
    double best_error = ordinary_error;
    double relative_tolerance = p.vp_orbit_lc_relative_tolerance;
    double absolute_tolerance = p.vp_orbit_lc_absolute_tolerance;
    bool attempted_lc = false;
    bool attempted_wall = false;
    bool prefer_wall =
        can_use_wall_regularization
        && !regularized_path_stays_inside_domain
        && initially_moves_toward_inner_wall;
    for (int refinement = 0;
         refinement <= p.vp_orbit_lc_max_refinements;
         ++refinement) {
        if (!prefer_wall && can_use_levi_civita) {
            if (!attempted_lc) {
                if (record_regularization_diagnostics) {
                    vp_orbit_lc_retries.fetch_add(
                        1, std::memory_order_relaxed);
                }
                attempted_lc = true;
            }
            RegularizedOrbitResult regularized =
                advance_radial_characteristic_levi_civita(
                    initial,
                    physical_dt,
                    rhs,
                    relative_tolerance,
                    absolute_tolerance,
                    p.vp_orbit_lc_max_steps);
            if (regularized.converged) {
                reflect_radial_boundaries(
                    regularized.state.r,
                    regularized.state.w,
                    p.r_min,
                    vp_radial_outer_boundary(r, p),
                    p, field.background.H);
                if (using_expanding_vp(p)) {
                    if (record_regularization_diagnostics) {
                        vp_orbit_lc_accepted.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    return regularized.state;
                }
                const double error = vp_frozen_hamiltonian_error(
                    initial, regularized.state, Lk, x_r, r, field, p);
                if (std::isfinite(error) && error < best_error) {
                    best = regularized.state;
                    best_error = error;
                }
                if (error <= p.vp_orbit_energy_tolerance) {
                    if (record_regularization_diagnostics) {
                        vp_orbit_atomic_max_finite(
                            vp_orbit_maximum_final_energy_error,
                            error);
                        vp_orbit_lc_accepted.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    return regularized.state;
                }
            }
            if (can_use_wall_regularization
                && !regularized_path_stays_inside_domain) {
                prefer_wall = true;
            }
        }

        if (prefer_wall && can_use_wall_regularization) {
            if (!attempted_wall) {
                if (record_regularization_diagnostics) {
                    vp_orbit_wall_retries.fetch_add(
                        1, std::memory_order_relaxed);
                }
                attempted_wall = true;
            }
            RegularizedOrbitResult regularized =
                advance_radial_characteristic_reflecting_wall(
                    initial,
                    physical_dt,
                    p.r_min,
                    rhs,
                    relative_tolerance,
                    absolute_tolerance,
                    p.vp_orbit_lc_max_steps);
            if (regularized.converged) {
                reflect_radial_boundaries(
                    regularized.state.r,
                    regularized.state.w,
                    p.r_min,
                    vp_radial_outer_boundary(r, p),
                    p, field.background.H);
                const double error = vp_frozen_hamiltonian_error(
                    initial, regularized.state, Lk, x_r, r, field, p);
                if (std::isfinite(error) && error < best_error) {
                    best = regularized.state;
                    best_error = error;
                }
                if (error <= p.vp_orbit_energy_tolerance) {
                    if (record_regularization_diagnostics) {
                        vp_orbit_atomic_max_finite(
                            vp_orbit_maximum_final_energy_error,
                            error);
                        vp_orbit_wall_accepted.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    return regularized.state;
                }
            }
        }
        relative_tolerance *= 0.1;
        absolute_tolerance *= 0.1;
    }
    if (record_regularization_diagnostics) {
        vp_orbit_atomic_max_finite(
            vp_orbit_maximum_final_energy_error,
            best_error);
        if (attempted_lc) {
            vp_orbit_lc_unresolved.fetch_add(1, std::memory_order_relaxed);
        }
        if (attempted_wall) {
            vp_orbit_wall_unresolved.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return best;
}

/// Traces one VP characteristic footpoint backward by dt with the selected method.
CharacteristicState trace_vp_characteristic_footpoint(
    double ri,
    double wi,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    bool regularization_eligible = true) {
    return advance_vp_characteristic(
        ri, wi, Lk, x_r, r, field, -dt, p, plan, regularization_eligible);
}

/// Traces a phase-space cell center forward by dt using the same frozen Hamiltonian flow.
CharacteristicState trace_vp_characteristic_arrival(
    double ri,
    double wi,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    bool regularization_eligible = true) {
    return trace_vp_characteristic_footpoint(
        ri, wi, Lk, x_r, r, field, -dt, p, plan, regularization_eligible);
}

#include "ev/vp_expanding_interpolation.hpp"

/// One local radial sample in coordinates (r,H,sign w) for fixed L.
struct VPHamiltonianRadialSample {
    double x = 0.0;
    double value = 0.0;
    bool valid = false;
};

/// Interpolates one fixed-r, fixed-L branch using H rather than w as the local coordinate.
// At fixed r and L, energy determines |w| and sign(w) selects the radial
// branch. Interpolate within that branch so a static f(H,L) is represented
// without mixing incoming and outgoing velocities.
bool interpolate_vp_fixed_radius_hamiltonian_branch(
    const std::vector<double>& f,
    int i,
    int k,
    double h,
    double sign_w,
    double Lk,
    int first_nonnegative_w,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const VlasovPoissonField& field,
    const Params& p,
    double& out) {
    const double ri = r[static_cast<std::size_t>(i)];
    const double tangential = (Lk / ri) * (Lk / ri);
    double speed2 = 2.0 * (h - field.phi[static_cast<std::size_t>(i)]) - tangential;
    const double h_scale = 1.0 + std::abs(h) + std::abs(field.phi[static_cast<std::size_t>(i)])
        + 0.5 * tangential;
    if (speed2 < -1.0e-12 * h_scale) {
        return false;
    }
    speed2 = std::max(0.0, speed2);
    const double wp = (sign_w >= 0.0 ? 1.0 : -1.0) * std::sqrt(speed2);
    if (wp < w.front() || wp > w.back()) {
        return false;
    }

    const int first_nonnegative = first_nonnegative_w;
    const int last_negative = first_nonnegative - 1;
    int j0 = -1;
    int j1 = -1;
    if (sign_w >= 0.0) {
        if (first_nonnegative >= p.Nw) {
            return false;
        }
        if (wp <= w[static_cast<std::size_t>(first_nonnegative)] || first_nonnegative == p.Nw - 1) {
            out = f[idx(i, first_nonnegative, k, p)];
            return true;
        }
        j0 = std::max(first_nonnegative, lower_cell(w, wp));
        j1 = j0 + 1;
        if (j1 >= p.Nw) {
            out = f[idx(i, p.Nw - 1, k, p)];
            return true;
        }
    } else {
        if (last_negative < 0) {
            return false;
        }
        if (wp >= w[static_cast<std::size_t>(last_negative)] || last_negative == 0) {
            out = f[idx(i, last_negative, k, p)];
            return true;
        }
        j0 = lower_cell(w, wp);
        j1 = j0 + 1;
        if (j1 > last_negative) {
            j1 = last_negative;
            j0 = j1 - 1;
        }
        if (j0 < 0) {
            out = f[idx(i, 0, k, p)];
            return true;
        }
    }

    const double h0 = vp_particle_hamiltonian(
        ri, w[static_cast<std::size_t>(j0)], Lk, field.phi[static_cast<std::size_t>(i)]);
    const double h1 = vp_particle_hamiltonian(
        ri, w[static_cast<std::size_t>(j1)], Lk, field.phi[static_cast<std::size_t>(i)]);
    if (h1 == h0) {
        out = 0.5 * (f[idx(i, j0, k, p)] + f[idx(i, j1, k, p)]);
        return true;
    }
    const double q = std::clamp((h - h0) / (h1 - h0), 0.0, 1.0);
    out = (1.0 - q) * f[idx(i, j0, k, p)] + q * f[idx(i, j1, k, p)];
    return true;
}

/// Interpolates the instantaneous f at a departure point in local (r,H,sign w) coordinates.
double interpolate_vp_hamiltonian_slice(
    const std::vector<double>& f,
    double rp,
    double wp,
    double h_target,
    int k,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    int first_nonnegative_w,
    const Params& p) {
    regularize_radial_footpoint_for_interpolation(rp, wp, r, p);
    if (rp < r.front() || rp > r.back()) {
        return 0.0;
    }

    const double phi_p = sample_vp_field(rp, x_r, r, field, p).phi;
    const double Lk = L[static_cast<std::size_t>(k)];
    const double rc = std::max(rp, 1.0e-300);
    const double tangential = (Lk / rc) * (Lk / rc);
    double corrected_speed2 = 2.0 * (h_target - phi_p) - tangential;
    const double h_scale = 1.0 + std::abs(h_target) + std::abs(phi_p) + 0.5 * tangential;
    if (corrected_speed2 < -1.0e-12 * h_scale) {
        return interpolate_slice(f, rp, wp, k, x_r, r, w, p);
    }
    corrected_speed2 = std::max(0.0, corrected_speed2);
    const double sign_w = (wp >= 0.0) ? 1.0 : -1.0;
    const double wp_corrected = sign_w * std::sqrt(corrected_speed2);
    if (wp_corrected < w.front() || wp_corrected > w.back()) {
        return 0.0;
    }

    const double xp = radial_coordinate(rp, p);
    const int i_center = lower_cell(x_r, xp);
    VPHamiltonianRadialSample left;
    VPHamiltonianRadialSample right;
    for (int di = -2; di <= 3; ++di) {
        const int ii = i_center + di;
        if (ii < 0 || ii >= p.Nr) {
            continue;
        }
        double value = 0.0;
        if (!interpolate_vp_fixed_radius_hamiltonian_branch(
                f, ii, k, h_target, sign_w, Lk, first_nonnegative_w, r, w, field, p, value)) {
            continue;
        }
        const double xi = x_r[static_cast<std::size_t>(ii)];
        if (xi <= xp) {
            if (!left.valid || xi > left.x) {
                left = {xi, value, true};
            }
        }
        if (xi >= xp) {
            if (!right.valid || xi < right.x) {
                right = {xi, value, true};
            }
        }
    }

    if (left.valid && right.valid) {
        if (right.x == left.x) {
            return 0.5 * (left.value + right.value);
        }
        const double q = std::clamp((xp - left.x) / (right.x - left.x), 0.0, 1.0);
        return (1.0 - q) * left.value + q * right.value;
    }
    if (left.valid) {
        return left.value;
    }
    if (right.valid) {
        return right.value;
    }
    return interpolate_slice(f, rp, wp_corrected, k, x_r, r, w, p);
}

/// Interpolates a binned energy-shell table, clamping only outside its sampled range.
double sample_energy_shell_table(const EnergyShellTable& table, double h) {
    if (table.h.empty()) {
        return 0.0;
    }
    if (table.h.size() == 1 || h <= table.h.front()) {
        return table.f.front();
    }
    if (h >= table.h.back()) {
        return table.f.back();
    }
    auto it = std::upper_bound(table.h.begin(), table.h.end(), h);
    int i = static_cast<int>(std::distance(table.h.begin(), it)) - 1;
    i = std::clamp(i, 0, static_cast<int>(table.h.size()) - 2);
    const double h0 = table.h[static_cast<std::size_t>(i)];
    const double h1 = table.h[static_cast<std::size_t>(i + 1)];
    if (!(h1 > h0)) {
        return 0.5 * (table.f[static_cast<std::size_t>(i)] + table.f[static_cast<std::size_t>(i + 1)]);
    }
    const double q = (h - h0) / (h1 - h0);
    return (1.0 - q) * table.f[static_cast<std::size_t>(i)]
        + q * table.f[static_cast<std::size_t>(i + 1)];
}

/// Builds per-L, binned f(H,L) tables from the current grid distribution.
// The energy-shell option constructs an arithmetic nodal reference f(H,L)
// from the current grid, then advects its residual. This is a numerical
// interpolation reference built at the current frozen field.
std::vector<EnergyShellTable> build_vp_energy_shell_tables(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    const Params& p) {
    std::vector<EnergyShellTable> tables(static_cast<std::size_t>(p.NL));
    const int bin_count = std::clamp(p.Nr * p.Nw / 4, 32, 2048);
    parallel_for_chunks(0, p.NL, effective_threads(p, p.NL),
        [&](int klo, int khi, int) {
            for (int k = klo; k < khi; ++k) {
                const double Lk = L[static_cast<std::size_t>(k)];
                double h_min = std::numeric_limits<double>::infinity();
                double h_max = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                        const double h = vp_particle_hamiltonian(
                            r[static_cast<std::size_t>(i)],
                            w[static_cast<std::size_t>(j)],
                            Lk,
                            field.phi[static_cast<std::size_t>(i)]);
                        h_min = std::min(h_min, h);
                        h_max = std::max(h_max, h);
                    }
                }
                if (!(h_max > h_min) || !std::isfinite(h_min) || !std::isfinite(h_max)) {
                    tables[static_cast<std::size_t>(k)].h.push_back(0.0);
                    tables[static_cast<std::size_t>(k)].f.push_back(0.0);
                    continue;
                }

                std::vector<double> h_sum(static_cast<std::size_t>(bin_count), 0.0);
                std::vector<double> f_sum(static_cast<std::size_t>(bin_count), 0.0);
                std::vector<int> count(static_cast<std::size_t>(bin_count), 0);
                const double inv_dh = static_cast<double>(bin_count) / (h_max - h_min);
                for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                        const double h = vp_particle_hamiltonian(
                            r[static_cast<std::size_t>(i)],
                            w[static_cast<std::size_t>(j)],
                            Lk,
                            field.phi[static_cast<std::size_t>(i)]);
                        int b = static_cast<int>((h - h_min) * inv_dh);
                        b = std::clamp(b, 0, bin_count - 1);
                        h_sum[static_cast<std::size_t>(b)] += h;
                        f_sum[static_cast<std::size_t>(b)] += f[idx(i, j, k, p)];
                        ++count[static_cast<std::size_t>(b)];
                    }
                }

                EnergyShellTable& table = tables[static_cast<std::size_t>(k)];
                for (int b = 0; b < bin_count; ++b) {
                    const int n = count[static_cast<std::size_t>(b)];
                    if (n <= 0) {
                        continue;
                    }
                    table.h.push_back(h_sum[static_cast<std::size_t>(b)] / static_cast<double>(n));
                    table.f.push_back(std::max(0.0, f_sum[static_cast<std::size_t>(b)] / static_cast<double>(n)));
                }
                if (table.h.empty()) {
                    table.h.push_back(0.0);
                    table.f.push_back(0.0);
                }
            }
        });
    return tables;
}

/// Splits f into a frozen-energy-shell baseline plus a raw residual.
std::vector<double> vp_energy_shell_residual(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    const std::vector<EnergyShellTable>& tables,
    const Params& p) {
    std::vector<double> residual(f.size(), 0.0);
    const int work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(0, work, effective_threads(p, work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double h = vp_particle_hamiltonian(
                    r[static_cast<std::size_t>(i)],
                    w[static_cast<std::size_t>(j)],
                    L[static_cast<std::size_t>(k)],
                    field.phi[static_cast<std::size_t>(i)]);
                const double f_eq =
                    sample_energy_shell_table(tables[static_cast<std::size_t>(k)], h);
                residual[static_cast<std::size_t>(q)] = f[static_cast<std::size_t>(q)] - f_eq;
            }
        });
    return residual;
}

/// Returns the inward coordinate speed prescribed at the VP inflow boundary.
double vp_inflow_coordinate_speed(double r_boundary, const Params& p) {
    if (p.vp_inflow_coord_speed > 0.0) {
        return p.vp_inflow_coord_speed;
    }
    return p.vp_inflow_vff_factor
        * std::sqrt(std::max(2.0 * p.vp_gravity * p.vp_inflow_freefall_mass / r_boundary, 0.0));
}

/// Evaluates a normalized one-dimensional Gaussian.
double gaussian_pdf(double x, double center, double sigma) {
    const double z = (x - center) / sigma;
    return std::exp(-0.5 * z * z) / (std::sqrt(2.0 * PI) * sigma);
}

/// Tangential factor normalized so (2 pi / r^2) int_0^inf L T(L) dL = 1.
double normalized_tangential_pdf(double Lk, double r_boundary, double sigma_t, double L_center) {
    const double sigma_L = std::max(r_boundary * sigma_t, 1.0e-300);
    const double z0 = L_center / sigma_L;
    const double integral = sigma_L * sigma_L * std::exp(-0.5 * z0 * z0)
        + L_center * sigma_L * std::sqrt(0.5 * PI)
            * (1.0 + std::erf(L_center / (std::sqrt(2.0) * sigma_L)));
    const double norm = r_boundary * r_boundary / (2.0 * PI * std::max(integral, 1.0e-300));
    const double z = (Lk - L_center) / sigma_L;
    return norm * std::exp(-0.5 * z * z);
}

/// Boundary distribution for a scheduled train of Gaussian VP inflow pulses at r_max.
// A pulse prescribes mass flux at the physical outer sphere. Normalize by
// the incoming physical-speed Gaussian integral and the truncated time Gaussian
// so the intended incoming mass is expressed by the boundary distribution.
double vp_outer_inflow_distribution(
    double time,
    double wp,
    double Lk,
    double r_boundary,
    const VlasovPoissonField& field,
    const Params& p) {
    if (!using_vp_outer_inflow(p)) {
        return 0.0;
    }

    const double speed = std::max(vp_inflow_coordinate_speed(r_boundary, p), 1.0e-300);
    const double u_center = -field.background.H * r_boundary - speed;
    const double shell_mass_default = p.vp_inflow_shell_mass;
    const double shell_mass_high =
        (p.vp_inflow_high_shell_mass > 0.0) ? p.vp_inflow_high_shell_mass : shell_mass_default;
    const double sigma_u_high =
        (p.vp_inflow_high_sigma_u > 0.0) ? p.vp_inflow_high_sigma_u : p.vp_inflow_sigma_u;
    const double time_normalization = std::erf(p.vp_inflow_time_cutoff / std::sqrt(2.0));

    double value = 0.0;
    for (int n = 0; n < p.vp_inflow_count; ++n) {
        const bool high_l = n >= p.vp_inflow_low_l_count;
        const double center_time = p.vp_inflow_start_time + n * p.vp_inflow_period
            + (high_l ? p.vp_inflow_high_l_delay : 0.0);
        const double dtau = time - center_time;
        if (std::abs(dtau) > p.vp_inflow_time_cutoff * p.vp_inflow_time_sigma) {
            continue;
        }

        const double mass = high_l ? shell_mass_high : shell_mass_default;
        const double sigma_u = high_l ? sigma_u_high : p.vp_inflow_sigma_u;
        // Only incoming physical velocities cross this boundary. Their
        // flux-weighted Gaussian integral is <(-u) 1_{u<0}>, which tends
        // to speed in the cold limit and remains finite for a warm inflow.
        const double z = speed / sigma_u;
        const double incoming_speed = 0.5 * speed * std::erfc(-z / std::sqrt(2.0))
            + sigma_u * std::exp(-0.5 * z * z) / std::sqrt(2.0 * PI);
        const double rho = mass * gaussian_pdf(time, center_time, p.vp_inflow_time_sigma)
            / (4.0 * PI * r_boundary * r_boundary * incoming_speed * time_normalization);
        const double sigma_t = high_l ? p.vp_inflow_high_sigma_t : p.vp_inflow_low_sigma_t;
        const double L_center = high_l ? p.vp_inflow_high_l_center : p.vp_inflow_low_l_center;
        value += rho * gaussian_pdf(wp, u_center, sigma_u)
            * normalized_tangential_pdf(Lk, r_boundary, sigma_t, L_center);
    }
    return value;
}

/// Selects the next timestep from active-cell VP characteristic speeds.
// Characteristic steps use the configured occupied-support clock, whereas
// explicit finite-volume transport checks every positive donor. The latter
// requires the actual outgoing loss coefficient to preserve positivity.
double choose_dt_vlasov_poisson(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    double radial_step,
    double min_dw,
    const Params& p,
    const MpiContext& mpi,
    const PhaseSpaceCellGeometry* geometry = nullptr) {
    const double max_f = mpi_allreduce_max(max_value_parallel(f, p), mpi);
    const double threshold = std::max(p.active_floor_abs, p.active_floor_rel * max_f);
    if (max_f <= 0.0) {
        return p.dt_max;
    }

    // Canonical cell geometry permits a per-cell mass threshold derived from
    // the configured total omission budget. It changes clock selection while
    // the complete distribution remains in field moments and transport.
    const bool mass_weighted_active_set =
        geometry != nullptr && using_cell_average_eta_grid(p);
    double cell_mass_threshold = 0.0;
    if (mass_weighted_active_set) {
        if (geometry->radial.width.size() != static_cast<std::size_t>(p.Nr)
            || geometry->radial_velocity.width.size()
                != static_cast<std::size_t>(p.Nw)
            || geometry->eta.width.size() != static_cast<std::size_t>(p.NL)) {
            throw std::runtime_error(
                "VP timestep mass-weighted active set received incompatible geometry.");
        }
        const int work = static_cast<int>(f.size());
        const int threads = effective_threads(p, work);
        std::vector<long double> partial_mass(
            static_cast<std::size_t>(threads), 0.0L);
        parallel_for_chunks(0, work, threads,
            [&](int lo, int hi, int tid) {
                long double sum = 0.0L;
                for (int q = lo; q < hi; ++q) {
                    const int i = q / (p.Nw * p.NL);
                    const int rem = q - i * p.Nw * p.NL;
                    const int j = rem / p.NL;
                    const int k = rem - j * p.NL;
                    sum += static_cast<long double>(f[static_cast<std::size_t>(q)])
                        * geometry->radial.width[static_cast<std::size_t>(i)]
                        * geometry->radial_velocity.width[static_cast<std::size_t>(j)]
                        * geometry->eta.width[static_cast<std::size_t>(k)];
                }
                partial_mass[static_cast<std::size_t>(tid)] = sum;
            });
        long double local_mass = 0.0L;
        for (long double value : partial_mass) {
            local_mass += value;
        }
        const double total_mass = mpi_allreduce_sum(
            static_cast<double>(local_mass), mpi);
        const long long global_cells = mpi_allreduce_sum_ll(
            static_cast<long long>(f.size()), mpi);
        if (total_mass > 0.0 && global_cells > 0) {
            cell_mass_threshold =
                p.vp_timestep_omitted_mass_fraction * total_mass
                / static_cast<double>(global_cells);
        }
    }

    const bool finite_volume = p.integrator == "conservative_finite_volume";
    std::vector<double> r_edges = finite_volume ? control_volume_edges(r) : std::vector<double>{};
    if (finite_volume && using_origin_parity_inner(p)) {
        r_edges.front() = 0.0;
    }
    const std::vector<double> w_edges = finite_volume ? control_volume_edges(w) : std::vector<double>{};
    const int work = static_cast<int>(f.size());
    const int nthreads = effective_threads(p, work);
    std::vector<double> local_vr(static_cast<std::size_t>(nthreads), 0.0);
    std::vector<double> local_vw(static_cast<std::size_t>(nthreads), 0.0);
    std::vector<double> local_fv_rate(static_cast<std::size_t>(nthreads), 0.0);
    std::vector<double> local_r_over_acceleration(
        static_cast<std::size_t>(nthreads),
        std::numeric_limits<double>::infinity());
    std::vector<long long> local_active(static_cast<std::size_t>(nthreads), 0);
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            double max_vr_local = 0.0;
            double max_vw_local = 0.0;
            double max_fv_rate_local = 0.0;
            double min_r_over_acceleration_local =
                std::numeric_limits<double>::infinity();
            long long active_local = 0;
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double dr_cell = finite_volume
                    ? std::max(r_edges[static_cast<std::size_t>(i + 1)]
                                   - r_edges[static_cast<std::size_t>(i)],
                               1.0e-300)
                    : 1.0;
                const double dw_cell = finite_volume
                    ? std::max(w_edges[static_cast<std::size_t>(j + 1)]
                                   - w_edges[static_cast<std::size_t>(j)],
                               1.0e-300)
                    : 1.0;
                const double fij = f[static_cast<std::size_t>(q)];
                const double represented_cell_mass = mass_weighted_active_set
                    ? fij
                        * geometry->radial.width[static_cast<std::size_t>(i)]
                        * geometry->radial_velocity.width[static_cast<std::size_t>(j)]
                        * geometry->eta.width[static_cast<std::size_t>(k)]
                    : 0.0;
                // Every positive donor contributes to the explicit FV
                // update. Its loss rate must obey CFL, including small tails.
                if (finite_volume ? !(fij > 0.0)
                    : ((mass_weighted_active_set
                        && represented_cell_mass <= cell_mass_threshold)
                       || (!mass_weighted_active_set && fij <= threshold))) {
                    continue;
                }
                ++active_local;
                const double vr = field.background.H * r[i] + w[j];
                const double vw = L[k] * L[k] / (r[i] * r[i] * r[i])
                    - field.phi_r[i] - field.background.H * w[j];
                const double vr_coord = using_log_radial_coordinate(p) ? vr / r[i] : vr;
                max_vr_local = std::max(max_vr_local, std::abs(vr_coord));
                max_vw_local = std::max(max_vw_local, std::abs(vw));
                if (std::abs(vw) > 1.0e-300) {
                    min_r_over_acceleration_local = std::min(
                        min_r_over_acceleration_local,
                        r[static_cast<std::size_t>(i)] / std::abs(vw));
                }
                if (finite_volume) {
                    const double H = field.background.H;
                    double radial_loss = std::max(0.0, -(H * r_edges[i] + w[j]))
                        + std::max(0.0, H * r_edges[i + 1] + w[j]);
                    auto wall_outgoing_average = [&](double wall, bool outer) {
                        const double drift = H * wall;
                        const double left = outer ? std::max(w_edges[j], -drift) : w_edges[j];
                        const double right = outer ? w_edges[j + 1] : std::min(w_edges[j + 1], -drift);
                        return right > left
                            ? std::abs((right - left) * (0.5 * (left + right) + drift)) / dw_cell
                            : 0.0;
                    };
                    if (i == 0) {
                        radial_loss += wall_outgoing_average(r_edges.front(), false)
                            - std::max(0.0, -(H * r_edges.front() + w[j]));
                    }
                    if (i == p.Nr - 1 && using_reflecting_outer(p)) {
                        radial_loss += wall_outgoing_average(r_edges.back(), true)
                            - std::max(0.0, H * r_edges.back() + w[j]);
                    }
                    const double force = L[k] * L[k] / (r[i] * r[i] * r[i]) - field.phi_r[i];
                    const double velocity_loss = std::max(0.0, -(force - H * w_edges[j]))
                        + std::max(0.0, force - H * w_edges[j + 1]);
                    // These are exactly the outgoing donor coefficients used
                    // by the radial and velocity fluxes. Incoming terms are
                    // nonnegative and need no additional positivity bound.
                    max_fv_rate_local = std::max(max_fv_rate_local,
                        radial_loss / dr_cell + velocity_loss / dw_cell);
                }
            }
            local_vr[static_cast<std::size_t>(tid)] = max_vr_local;
            local_vw[static_cast<std::size_t>(tid)] = max_vw_local;
            local_fv_rate[static_cast<std::size_t>(tid)] = max_fv_rate_local;
            local_r_over_acceleration[static_cast<std::size_t>(tid)] =
                min_r_over_acceleration_local;
            local_active[static_cast<std::size_t>(tid)] = active_local;
        });

    double max_vr_coord = 0.0;
    double max_vw = 0.0;
    double max_fv_rate = 0.0;
    double min_r_over_acceleration = std::numeric_limits<double>::infinity();
    long long active_count = 0;
    for (int q = 0; q < nthreads; ++q) {
        max_vr_coord = std::max(max_vr_coord, local_vr[static_cast<std::size_t>(q)]);
        max_vw = std::max(max_vw, local_vw[static_cast<std::size_t>(q)]);
        max_fv_rate = std::max(max_fv_rate, local_fv_rate[static_cast<std::size_t>(q)]);
        min_r_over_acceleration = std::min(
            min_r_over_acceleration,
            local_r_over_acceleration[static_cast<std::size_t>(q)]);
        active_count += local_active[static_cast<std::size_t>(q)];
    }
    max_vr_coord = mpi_allreduce_max(max_vr_coord, mpi);
    max_vw = mpi_allreduce_max(max_vw, mpi);
    max_fv_rate = mpi_allreduce_max(max_fv_rate, mpi);
    min_r_over_acceleration =
        mpi_allreduce_min(min_r_over_acceleration, mpi);
    active_count = mpi_allreduce_sum_ll(active_count, mpi);

    if (active_count == 0) {
        return p.dt_max;
    }
    if (finite_volume) {
        max_fv_rate = std::max(max_fv_rate, 1.0e-14);
        return std::min(p.dt_max, p.cfl / max_fv_rate);
    }
    if (p.vp_timestep_mode == "frank_dynamical") {
        if (!std::isfinite(min_r_over_acceleration)) {
            return p.dt_max;
        }
        // Frank et al. Eq. (13): the semi-Lagrangian characteristic step is
        // limited by a physical dynamical time, not by a velocity-cell
        // crossing.  The total radial acceleration includes the centrifugal
        // term; LC retry handles the resulting low-L pericentre stiffness.
        const double dynamical_dt =
            std::sqrt(2.0 * p.vp_dynamical_eta * min_r_over_acceleration);
        return std::min(p.dt_max, dynamical_dt);
    }
    max_vr_coord = std::max(max_vr_coord, 1.0e-14);
    max_vw = std::max(max_vw, 1.0e-14);
    return std::min({p.dt_max, p.cfl * radial_step / max_vr_coord, p.cfl * min_dw / max_vw});
}

/// Brackets a deposition coordinate on cell centers while preserving a partition of unity.
bool vp_conservative_deposition_bracket(
    const std::vector<double>& grid,
    double value,
    bool clamp_below,
    bool clamp_above,
    int& i0,
    int& i1,
    double& fraction) {
    if (value < grid.front()) {
        if (!clamp_below) {
            return false;
        }
        i0 = 0;
        i1 = 0;
        fraction = 0.0;
        return true;
    }
    if (value > grid.back()) {
        if (!clamp_above) {
            return false;
        }
        i0 = static_cast<int>(grid.size()) - 1;
        i1 = i0;
        fraction = 0.0;
        return true;
    }
    if (value <= grid.front()) {
        i0 = 0;
        i1 = 0;
        fraction = 0.0;
        return true;
    }
    if (value >= grid.back()) {
        i0 = static_cast<int>(grid.size()) - 1;
        i1 = i0;
        fraction = 0.0;
        return true;
    }
    i0 = lower_cell(grid, value);
    i1 = i0 + 1;
    fraction = (value - grid[static_cast<std::size_t>(i0)])
        / (grid[static_cast<std::size_t>(i1)] - grid[static_cast<std::size_t>(i0)]);
    fraction = std::clamp(fraction, 0.0, 1.0);
    return true;
}

std::vector<double> advect_vlasov_poisson(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    double dt,
    double t,
    const Params& p);

/// Forward conservative semi-Lagrangian remap on Hamiltonian characteristics.
// Push positive source mass forward along each fixed-L orbit. Deposit it
// with weights summing to one, then blend the higher-order pullback toward
// this conservative result only as required by each L-plane mass budget.
std::vector<double> advect_vlasov_poisson_hamiltonian_conservative(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    double dt,
    double t,
    const Params& p) {
    const std::vector<double> rq = radial_control_volume_widths(r, p);
    const std::vector<double> wq = quadrature_weights(w);
    const CharacteristicIntegratorPlan plan = make_characteristic_integrator_plan(p);
    std::vector<double> cell_mass(f.size(), 0.0);
    const double active_threshold = std::max(
        p.active_floor_abs,
        p.active_floor_rel * max_value_parallel(f, p));

    // L is invariant, so assigning each L plane to one worker avoids atomics.
    parallel_for_dynamic_chunks(0, p.NL, effective_threads(p, p.NL), 1,
        [&](int klo, int khi, int) {
            for (int k = klo; k < khi; ++k) {
                const double Lk = L[static_cast<std::size_t>(k)];
                for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                        const double fq = f[idx(i, j, k, p)];
                        if (!(fq > 0.0) || !std::isfinite(fq)) {
                            continue;
                        }
                        CharacteristicState arrival = trace_vp_characteristic_arrival(
                            r[static_cast<std::size_t>(i)],
                            w[static_cast<std::size_t>(j)],
                            Lk,
                            x_r,
                            r,
                            field,
                            dt,
                            p,
                            plan,
                            fq > active_threshold);

                        // Project the velocity back to the same frozen Hamiltonian shell.
                        const double phi_arrival =
                            sample_vp_field(arrival.r, x_r, r, field, p).phi;
                        const double h0 = vp_particle_hamiltonian(
                            r[static_cast<std::size_t>(i)],
                            w[static_cast<std::size_t>(j)],
                            Lk,
                            field.phi[static_cast<std::size_t>(i)]);
                        const double rc = std::max(arrival.r, r.front());
                        double wr2 = 2.0 * (h0 - phi_arrival) - (Lk / rc) * (Lk / rc);
                        const double hscale = 1.0 + std::abs(h0) + std::abs(phi_arrival)
                            + 0.5 * (Lk / rc) * (Lk / rc);
                        if (wr2 >= -1.0e-11 * hscale) {
                            wr2 = std::max(0.0, wr2);
                            arrival.w = (arrival.w >= 0.0 ? 1.0 : -1.0) * std::sqrt(wr2);
                        }

                        int ir0 = 0;
                        int ir1 = 0;
                        int jw0 = 0;
                        int jw1 = 0;
                        double tr = 0.0;
                        double tw = 0.0;
                        const bool radial_ok = vp_conservative_deposition_bracket(
                            r,
                            arrival.r,
                            using_origin_parity_inner(p),
                            using_reflecting_outer(p),
                            ir0,
                            ir1,
                            tr);
                        const bool momentum_ok = vp_conservative_deposition_bracket(
                            w, arrival.w, false, false, jw0, jw1, tw);
                        if (!radial_ok || !momentum_ok) {
                            continue;
                        }

                        const double transported_mass = fq
                            * rq[static_cast<std::size_t>(i)]
                            * wq[static_cast<std::size_t>(j)];
                        const int ir[2] = {ir0, ir1};
                        const int jw[2] = {jw0, jw1};
                        const double rw[2] = {
                            (ir0 == ir1) ? 1.0 : (1.0 - tr),
                            (ir0 == ir1) ? 0.0 : tr,
                        };
                        const double ww[2] = {
                            (jw0 == jw1) ? 1.0 : (1.0 - tw),
                            (jw0 == jw1) ? 0.0 : tw,
                        };
                        for (int a = 0; a < 2; ++a) {
                            for (int b = 0; b < 2; ++b) {
                                const double weight = rw[a] * ww[b];
                                if (weight > 0.0) {
                                    cell_mass[idx(ir[a], jw[b], k, p)] += transported_mass * weight;
                                }
                            }
                        }
                    }
                }
            }
        });

    std::vector<double> low_order(f.size(), 0.0);
    const int work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(0, work, effective_threads(p, work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const double volume = rq[static_cast<std::size_t>(i)] * wq[static_cast<std::size_t>(j)];
                if (volume > 0.0) {
                    low_order[static_cast<std::size_t>(q)] = cell_mass[static_cast<std::size_t>(q)] / volume;
                }
            }
        });

    Params high_order_p = p;
    high_order_p.vp_interpolation = "hamiltonian";
    std::vector<double> result = advect_vlasov_poisson(
        f, x_r, r, w, L, field, dt, t, high_order_p);

    // Flux-corrected remap: retain the high-order Hamiltonian result and move it
    // toward the conservative low-order result only where required to match the
    // exact per-L transported mass.  Every corrected cell stays between the two
    // positive candidates; this is not a global normalization.
    parallel_for_dynamic_chunks(0, p.NL, effective_threads(p, p.NL), 1,
        [&](int klo, int khi, int) {
            for (int k = klo; k < khi; ++k) {
                double low_total = 0.0;
                double high_total = 0.0;
                for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                        const double volume = rq[static_cast<std::size_t>(i)]
                            * wq[static_cast<std::size_t>(j)];
                        low_total += volume * low_order[idx(i, j, k, p)];
                        high_total += volume * result[idx(i, j, k, p)];
                    }
                }
                const double needed = low_total - high_total;
                const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
                    * std::max({1.0, std::abs(low_total), std::abs(high_total)});
                if (std::abs(needed) <= tolerance) {
                    continue;
                }

                double available = 0.0;
                for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                        const double volume = rq[static_cast<std::size_t>(i)]
                            * wq[static_cast<std::size_t>(j)];
                        const double difference = volume
                            * (low_order[idx(i, j, k, p)] - result[idx(i, j, k, p)]);
                        available += (needed > 0.0)
                            ? std::max(difference, 0.0)
                            : std::max(-difference, 0.0);
                    }
                }
                if (!(available > 0.0)) {
                    for (int i = 0; i < p.Nr; ++i) {
                        for (int j = 0; j < p.Nw; ++j) {
                            result[idx(i, j, k, p)] = low_order[idx(i, j, k, p)];
                        }
                    }
                    continue;
                }
                const double blend = std::min(1.0, std::abs(needed) / available);
                for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                        const std::size_t q = idx(i, j, k, p);
                        const double difference = low_order[q] - result[q];
                        if ((needed > 0.0 && difference > 0.0)
                            || (needed < 0.0 && difference < 0.0)) {
                            result[q] += blend * difference;
                        }
                    }
                }
            }
        });
    return result;
}

/// Advances f by one weak-field VP semi-Lagrangian step with the selected characteristic tracer.
std::vector<double> advect_vlasov_poisson(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    double dt,
    double t,
    const Params& p) {
    if (using_conservative_hamiltonian_transport(p)) {
        return advect_vlasov_poisson_hamiltonian_conservative(
            f, x_r, r, w, L, field, dt, t, p);
    }
    std::vector<double> f_new(f.size(), 0.0);
    const bool energy_shell_interpolation = p.vp_interpolation == "energy_shell";
    const bool hamiltonian_interpolation = p.vp_interpolation == "hamiltonian";
    const bool expanding_physical_velocity_interpolation =
        p.vp_interpolation == "expanding_physical_velocity";
    const std::vector<EnergyShellTable> energy_tables = energy_shell_interpolation
        ? build_vp_energy_shell_tables(f, r, w, L, field, p)
        : std::vector<EnergyShellTable>{};
    const CharacteristicIntegratorPlan integrator_plan = make_characteristic_integrator_plan(p);
    const double active_threshold = std::max(
        p.active_floor_abs,
        p.active_floor_rel * max_value_parallel(f, p));
    const std::vector<double> residual = energy_shell_interpolation
        ? vp_energy_shell_residual(f, r, w, L, field, energy_tables, p)
        : std::vector<double>{};
    // For an energy-shell reference, pull back only the residual and add the
    // reference at the destination Hamiltonian. Other interpolation modes pull
    // back the full distribution using their declared velocity coordinates.
    const std::vector<double>& departure_source = energy_shell_interpolation ? residual : f;
    const int first_nonnegative_w = hamiltonian_interpolation
        ? static_cast<int>(std::distance(w.begin(), std::lower_bound(w.begin(), w.end(), 0.0)))
        : 0;
    const int phase_space_work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(0, phase_space_work, effective_threads(p, phase_space_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double Lk = L[k];
                const CharacteristicState dep = trace_vp_characteristic_footpoint(
                    r[i],
                    w[j],
                    Lk,
                    x_r,
                    r,
                    field,
                    dt,
                    p,
                    integrator_plan,
                    f[static_cast<std::size_t>(q)] > active_threshold);
                double updated = 0.0;
                if (dep.r > r.back() && using_vp_outer_inflow(p)) {
                    updated = vp_outer_inflow_distribution(t, dep.w, Lk, r.back(), field, p);
                } else {
                    if (hamiltonian_interpolation) {
                        const double h_target =
                            vp_particle_hamiltonian(r[i], w[j], Lk, field.phi[i]);
                        updated = interpolate_vp_hamiltonian_slice(
                            f, dep.r, dep.w, h_target, k, x_r, r, w, L, field, first_nonnegative_w, p);
                    } else if (expanding_physical_velocity_interpolation) {
                        updated = interpolate_vp_expanding_physical_velocity_slice(
                            f, dep.r, dep.w, k, x_r, r, w, field, p);
                    } else {
                        updated = interpolate_slice(departure_source, dep.r, dep.w, k, x_r, r, w, p);
                    }
                    if (energy_shell_interpolation) {
                        const double h_target =
                            vp_particle_hamiltonian(r[i], w[j], Lk, field.phi[i]);
                        updated += sample_energy_shell_table(
                            energy_tables[static_cast<std::size_t>(k)],
                            h_target);
                    }
                }
                f_new[idx(i, j, k, p)] = std::max(0.0, updated);
            }
        });
    enforce_inner_wall_symmetry(f_new, w, p, field.background.H);
    return f_new;
}

/// Conservative first-order finite-volume VP advection in (r,v_r) for each fixed L.
// Construct one shared upwind flux per radial and velocity face, then take
// its cell divergence. Radial wall flux is rebinned after physical-velocity
// reflection; velocity-boundary outflow remains part of the open-domain budget.
std::vector<double> advect_vlasov_poisson_finite_volume(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    double dt,
    double t,
    const Params& p) {
    std::vector<double> r_edges = control_volume_edges(r);
    if (using_origin_parity_inner(p)) {
        r_edges.front() = 0.0;
    }
    const std::vector<double> w_edges = control_volume_edges(w);
    std::vector<double> flux_r(
        static_cast<std::size_t>(p.Nr + 1) * static_cast<std::size_t>(p.Nw) * static_cast<std::size_t>(p.NL),
        0.0);
    std::vector<double> flux_w(
        static_cast<std::size_t>(p.Nr) * static_cast<std::size_t>(p.Nw + 1) * static_cast<std::size_t>(p.NL),
        0.0);

    const int rface_work = (p.Nr + 1) * p.Nw * p.NL;
    parallel_for_chunks(0, rface_work, effective_threads(p, rface_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int face = q / (p.Nw * p.NL);
                const int rem = q - face * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;

                double flux = 0.0;
                if (face == 0) {
                    // Both reflecting walls are filled conservatively below.
                    flux = 0.0;
                } else if (face > 0 && face < p.Nr) {
                    const double rf = r_edges[static_cast<std::size_t>(face)];
                    const double vf = vp_radial_velocity(rf, w[static_cast<std::size_t>(j)], field);
                    const int up_i = (vf >= 0.0) ? (face - 1) : face;
                    flux = vf * f[idx(up_i, j, k, p)];
                } else if (face == p.Nr) {
                    const double rf = r_edges.back();
                    const double vf = vp_radial_velocity(rf, w[static_cast<std::size_t>(j)], field);
                    if (using_reflecting_outer(p)) {
                        flux = 0.0;
                    } else if (vf >= 0.0) {
                        flux = vf * f[idx(p.Nr - 1, j, k, p)];
                    } else if (using_vp_outer_inflow(p)) {
                        flux = vf * vp_outer_inflow_distribution(
                            t, w[static_cast<std::size_t>(j)], L[static_cast<std::size_t>(k)], r.back(), field, p);
                    }
                }

                flux_r[static_cast<std::size_t>(q)] = flux;
            }
        });

    // Integrate outgoing flux over each velocity cell and transfer it into
    // the cells reached by w'=-w-2*H*R. Rebinning the flux itself conserves
    // mass even when the reflected velocities lie between grid centers.
    // Reflections beyond the finite velocity domain leave that domain, just
    // as characteristics crossing the ordinary momentum boundary do.
    auto fill_reflecting_wall = [&](int face, int radial_i, double wall, bool outer) {
        const double drift = field.background.H * wall;
        for (int k = 0; k < p.NL; ++k) {
            for (int j = 0; j < p.Nw; ++j) {
                const double lo = outer ? std::max(w_edges[j], -drift) : w_edges[j];
                const double hi = outer ? w_edges[j + 1] : std::min(w_edges[j + 1], -drift);
                if (!(hi > lo)) {
                    continue;
                }
                const double value = f[idx(radial_i, j, k, p)];
                const auto integrated_flux = [&](double left, double right) {
                    return value * (right - left) * (0.5 * (left + right) + drift);
                };
                const std::size_t offset = static_cast<std::size_t>(face) * p.Nw * p.NL;
                flux_r[offset + j * p.NL + k] += integrated_flux(lo, hi)
                    / (w_edges[j + 1] - w_edges[j]);
                const double reflected_lo = -hi - 2.0 * drift;
                const double reflected_hi = -lo - 2.0 * drift;
                const int first = std::max(0, static_cast<int>(std::upper_bound(
                    w_edges.begin(), w_edges.end(), reflected_lo) - w_edges.begin()) - 1);
                for (int jm = first; jm < p.Nw && w_edges[jm] < reflected_hi; ++jm) {
                    const double left = std::max(reflected_lo, w_edges[jm]);
                    const double right = std::min(reflected_hi, w_edges[jm + 1]);
                    if (right > left) {
                        flux_r[offset + jm * p.NL + k] -= integrated_flux(
                            -right - 2.0 * drift, -left - 2.0 * drift)
                            / (w_edges[jm + 1] - w_edges[jm]);
                    }
                }
            }
        }
    };
    fill_reflecting_wall(0, 0, r_edges.front(), false);
    if (using_reflecting_outer(p)) {
        fill_reflecting_wall(p.Nr, p.Nr - 1, r_edges.back(), true);
    }

    const int wface_work = p.Nr * (p.Nw + 1) * p.NL;
    parallel_for_chunks(0, wface_work, effective_threads(p, wface_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / ((p.Nw + 1) * p.NL);
                const int rem = q - i * (p.Nw + 1) * p.NL;
                const int face = rem / p.NL;
                const int k = rem - face * p.NL;

                const double wf = w_edges[static_cast<std::size_t>(face)];
                const VPSample sample{
                    field.phi[static_cast<std::size_t>(i)],
                    field.phi_r[static_cast<std::size_t>(i)],
                };
                const double af = vp_radial_acceleration(
                    r[static_cast<std::size_t>(i)], wf, L[static_cast<std::size_t>(k)], sample, r, field);
                double flux = 0.0;
                if (face > 0 && face < p.Nw) {
                    const int up_j = (af >= 0.0) ? (face - 1) : face;
                    flux = af * f[idx(i, up_j, k, p)];
                } else if (face == 0) {
                    if (af < 0.0) {
                        flux = af * f[idx(i, 0, k, p)];
                    }
                } else {
                    if (af > 0.0) {
                        flux = af * f[idx(i, p.Nw - 1, k, p)];
                    }
                }

                flux_w[static_cast<std::size_t>(q)] = flux;
            }
        });

    std::vector<double> f_new(f.size(), 0.0);
    const int work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(0, work, effective_threads(p, work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;

                const double dr = r_edges[static_cast<std::size_t>(i + 1)]
                    - r_edges[static_cast<std::size_t>(i)];
                const double dw = w_edges[static_cast<std::size_t>(j + 1)]
                    - w_edges[static_cast<std::size_t>(j)];
                const double fr_right =
                    flux_r[(static_cast<std::size_t>(i + 1) * static_cast<std::size_t>(p.Nw)
                            + static_cast<std::size_t>(j))
                           * static_cast<std::size_t>(p.NL)
                           + static_cast<std::size_t>(k)];
                const double fr_left =
                    flux_r[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw)
                            + static_cast<std::size_t>(j))
                           * static_cast<std::size_t>(p.NL)
                           + static_cast<std::size_t>(k)];
                const double fw_right =
                    flux_w[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw + 1)
                            + static_cast<std::size_t>(j + 1))
                           * static_cast<std::size_t>(p.NL)
                           + static_cast<std::size_t>(k)];
                const double fw_left =
                    flux_w[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw + 1)
                            + static_cast<std::size_t>(j))
                           * static_cast<std::size_t>(p.NL)
                           + static_cast<std::size_t>(k)];
                f_new[static_cast<std::size_t>(q)] = f[static_cast<std::size_t>(q)]
                    - dt * ((fr_right - fr_left) / std::max(dr, 1.0e-300)
                            + (fw_right - fw_left) / std::max(dw, 1.0e-300));
            }
        });
    return f_new;
}

/// Estimates phase-space mass that will leave r or u boundaries under VP characteristics.
// Estimate boundary losses from forward characteristic velocities over the
// next interval. This support diagnostic uses the phase quadrature and is
// separate from the exact flux budget of a conservative transport update.
EscapeStats estimate_escapes_vlasov_poisson(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& rq,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const MpiContext& mpi) {
    EscapeStats stats;
    const double max_f = mpi_allreduce_max(max_value_parallel(f, p), mpi);
    const double threshold = std::max(p.active_floor_abs, p.active_floor_rel * max_f);
    const int work = static_cast<int>(f.size());
    const int nthreads = effective_threads(p, work);
    std::vector<EscapeStats> local(static_cast<std::size_t>(nthreads));
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            EscapeStats stats_local;
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double fij = f[static_cast<std::size_t>(q)];
                if (fij <= threshold) {
                    continue;
                }
                const double drdt = field.background.H * r[i] + w[j];
                const double dwdt = L[k] * L[k] / (r[i] * r[i] * r[i])
                    - field.phi_r[i] - field.background.H * w[j];
                double rp = r[i] + dt * drdt;
                double wp = w[j] + dt * dwdt;
                const double radial_outer = vp_radial_outer_boundary(r, p);
                reflect_radial_boundaries(rp, wp, p.r_min, radial_outer, p, field.background.H);

                const bool out_outer = !using_reflecting_outer(p) && rp > radial_outer;
                const bool out_w_low = wp < vp_momentum_lower_boundary(w, p);
                const bool out_w_high = wp > vp_momentum_upper_boundary(w, p);
                if (!out_outer && !out_w_low && !out_w_high) {
                    continue;
                }

                const double cell_mass = 8.0 * PI * PI * L[k] * fij * rq[i] * wq[j] * Lq[k];
                stats_local.total_mass += cell_mass;
                if (out_outer) {
                    stats_local.outer_mass += cell_mass;
                    ++stats_local.outer_cells;
                }
                if (out_w_low) {
                    stats_local.w_low_mass += cell_mass;
                    ++stats_local.w_low_cells;
                }
                if (out_w_high) {
                    stats_local.w_high_mass += cell_mass;
                    ++stats_local.w_high_cells;
                }
            }
            local[static_cast<std::size_t>(tid)] = stats_local;
        });
    for (const EscapeStats& s : local) {
        stats.outer_mass += s.outer_mass;
        stats.w_low_mass += s.w_low_mass;
        stats.w_high_mass += s.w_high_mass;
        stats.total_mass += s.total_mass;
        stats.outer_cells += s.outer_cells;
        stats.w_low_cells += s.w_low_cells;
        stats.w_high_cells += s.w_high_cells;
    }
    stats.outer_mass = mpi_allreduce_sum(stats.outer_mass, mpi);
    stats.w_low_mass = mpi_allreduce_sum(stats.w_low_mass, mpi);
    stats.w_high_mass = mpi_allreduce_sum(stats.w_high_mass, mpi);
    stats.total_mass = mpi_allreduce_sum(stats.total_mass, mpi);
    stats.outer_cells = mpi_allreduce_sum_ll(stats.outer_cells, mpi);
    stats.w_low_cells = mpi_allreduce_sum_ll(stats.w_low_cells, mpi);
    stats.w_high_cells = mpi_allreduce_sum_ll(stats.w_high_cells, mpi);
    return stats;
}

}  // namespace ev
