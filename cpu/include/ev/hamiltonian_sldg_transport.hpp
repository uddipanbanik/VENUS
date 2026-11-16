#pragma once

// Reconstruct and transport spherical cell averages using the local Hamiltonian.
// Inputs are f(r,w,eta), its optional carried P2 modes, a represented potential,
// phase-space geometry, and a transport interval. H=w^2/2+eta/r^2+Phi(r),
// with eta=L^2/2. Local energy-normal and orbit-tangent directions separate
// across-orbit variation from variation along a trajectory before conversion
// to the common positive P2 polynomial used by conservative SLDG remapping.
// Outputs are transported averages/modes and mass, reconstruction, and field
// diagnostics. An optional instantaneous F(H,eta) reference leaves a residual
// for remap and is added back afterward. The self-consistent entry predicts
// midpoint moments, solves their field, and uses that frozen field for the step.
// A changed collision state requires rebuilding its reconstruction/reference.

#include "ev/hamiltonian_orbit_reference.hpp"

namespace ev {

// These are averages of H and H^2 over a whole (r,w,eta) cell, retaining
// subcell kinetic and angular variance together with center energy.
struct HamiltonianSLDGCellMoments {
    std::vector<double> mean_h;
    std::vector<double> mean_h2;
};

// Store physical r/w slopes alongside the normalized P2 polynomial.
// Counters distinguish limited positive reconstructions from degenerate fits.
struct HamiltonianSLDGReconstruction {
    std::vector<std::array<double, 2>> slope;
    std::vector<SLDGSourcePolynomial> polynomial;
    long long reconstructed_cells = 0;
    long long limited_cells = 0;
    long long rank_deficient_cells = 0;
    long long phase_degenerate_cells = 0;
    long long eta_limited_cells = 0;
    double maximum_phase_gradient_fraction = 0.0;
};

struct HamiltonianSLDGQuadraticSample {
    std::array<double, 5> feature{};
    double delta_f = 0.0;
    double weight = 1.0;
};

// The field-variation cap records both Phi/abs(dPhi/dt) and the analogous
// radial-force time, including the cell controlling each resolved channel.
struct VPPotentialVariationLimit {
    double timescale = std::numeric_limits<double>::infinity();
    double timestep = std::numeric_limits<double>::infinity();
    double potential_timescale = std::numeric_limits<double>::infinity();
    double force_timescale = std::numeric_limits<double>::infinity();
    int controlling_potential_cell = -1;
    int controlling_force_cell = -1;
};

// Uniform-cell velocity moments: <w^2>=w_c^2+dw^2/12; the fourth moment
// below supplies the radial kinetic contribution to <H^2>.
inline double hamiltonian_sldg_mean_w2(
    double center,
    double width) {
    return center * center + width * width / 12.0;
}

inline double hamiltonian_sldg_mean_w4(
    double center,
    double width) {
    const double center2 = center * center;
    const double width2 = width * width;
    return center2 * center2
        + 0.5 * center2 * width2
        + width2 * width2 / 80.0;
}

/// Computes cell averages of H and H^2 with the represented radial potential.
///
/// The w and eta integrals are analytic.  Four-point Gauss--Legendre
/// quadrature integrates only the smooth represented potential in r.
inline HamiltonianSLDGCellMoments
build_hamiltonian_sldg_cell_moments(
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    const std::size_t cells =
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    HamiltonianSLDGCellMoments moments;
    moments.mean_h.assign(cells, 0.0);
    moments.mean_h2.assign(cells, 0.0);

    const auto& rule = phase_space_gauss_legendre_rule(4);
    const std::vector<double>& node = rule.first;
    const std::vector<double>& weight = rule.second;
    std::vector<std::array<double, 4>> radial_phi(
        static_cast<std::size_t>(p.Nr));
    std::vector<std::array<double, 4>> radial_inverse_r2(
        static_cast<std::size_t>(p.Nr));
    for (int i = 0; i < p.Nr; ++i) {
        const double center =
            geometry.radial.center[static_cast<std::size_t>(i)];
        const double half_width =
            0.5 * geometry.radial.width[static_cast<std::size_t>(i)];
        for (int q = 0; q < 4; ++q) {
            const double radius =
                center + half_width * node[static_cast<std::size_t>(q)];
            const VPSample sample =
                sample_vp_field(radius, x_r, r, field, p, i);
            radial_phi[static_cast<std::size_t>(i)][
                static_cast<std::size_t>(q)] = sample.phi;
            radial_inverse_r2[static_cast<std::size_t>(i)][
                static_cast<std::size_t>(q)] = 1.0 / (radius * radius);
        }
    }

    // Reuse radial potential samples across velocity/eta cells. Analytic
    // uniform-cell moments in w and eta leave only radial Gauss quadrature.
    const int work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(
        0,
        work,
        effective_threads(p, work),
        [&](int lo, int hi, int) {
            for (int flat = lo; flat < hi; ++flat) {
                const int i = flat / (p.Nw * p.NL);
                const int remainder = flat - i * p.Nw * p.NL;
                const int j = remainder / p.NL;
                const int k = remainder - j * p.NL;
                const double wc =
                    geometry.radial_velocity.center[
                        static_cast<std::size_t>(j)];
                const double dw =
                    geometry.radial_velocity.width[
                        static_cast<std::size_t>(j)];
                const double mean_w2 =
                    hamiltonian_sldg_mean_w2(wc, dw);
                const double mean_w4 =
                    hamiltonian_sldg_mean_w4(wc, dw);
                const double eta =
                    geometry.eta.center[static_cast<std::size_t>(k)];
                const double deta =
                    geometry.eta.width[static_cast<std::size_t>(k)];
                // The finite eta width contributes to <H^2> through
                // Var(eta)/r^4 even when its mean fixes the first energy moment.
                const double eta_variance = deta * deta / 12.0;
                long double mean_h = 0.0L;
                long double mean_h2 = 0.0L;
                for (int q = 0; q < 4; ++q) {
                    const long double radial_weight =
                        0.5L * weight[static_cast<std::size_t>(q)];
                    const long double inverse_r2 =
                        radial_inverse_r2[static_cast<std::size_t>(i)][
                            static_cast<std::size_t>(q)];
                    const long double base =
                        radial_phi[static_cast<std::size_t>(i)][
                            static_cast<std::size_t>(q)]
                        + static_cast<long double>(eta) * inverse_r2;
                    mean_h += radial_weight
                        * (base + 0.5L * mean_w2);
                    mean_h2 += radial_weight
                        * (base * base
                           + static_cast<long double>(eta_variance)
                               * inverse_r2 * inverse_r2
                           + base * mean_w2
                           + 0.25L * mean_w4);
                }
                const std::size_t cell = idx(i, j, k, p);
                moments.mean_h[cell] = static_cast<double>(mean_h);
                moments.mean_h2[cell] = static_cast<double>(mean_h2);
            }
        });
    return moments;
}

// Solve the small scaled normal system with pivoting. Rank-deficient
// geometry returns failure to the reconstruction fallback instead of
// amplifying unresolved phase-space directions.
inline bool hamiltonian_sldg_solve_dense_system(
    std::array<std::array<long double, 3>, 3> matrix,
    std::array<long double, 3> rhs,
    int dimension,
    std::array<long double, 3>& solution) {
    long double matrix_scale = 0.0L;
    for (int row = 0; row < dimension; ++row) {
        for (int column = 0; column < dimension; ++column) {
            matrix_scale = std::max(
                matrix_scale,
                std::abs(matrix[static_cast<std::size_t>(row)]
                                [static_cast<std::size_t>(column)]));
        }
    }
    if (!(matrix_scale > 0.0L)) {
        return false;
    }
    const long double pivot_tolerance =
        4096.0L * std::numeric_limits<long double>::epsilon()
        * matrix_scale;
    for (int column = 0; column < dimension; ++column) {
        int pivot = column;
        for (int row = column + 1; row < dimension; ++row) {
            if (std::abs(matrix[static_cast<std::size_t>(row)]
                                [static_cast<std::size_t>(column)])
                > std::abs(matrix[static_cast<std::size_t>(pivot)]
                                  [static_cast<std::size_t>(column)])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[static_cast<std::size_t>(pivot)]
                            [static_cast<std::size_t>(column)])
            <= pivot_tolerance) {
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
        const long double diagonal =
            matrix[static_cast<std::size_t>(column)]
                  [static_cast<std::size_t>(column)];
        for (int row = column + 1; row < dimension; ++row) {
            const long double factor =
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(column)]
                / diagonal;
            for (int entry = column; entry < dimension; ++entry) {
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(entry)]
                    -= factor
                        * matrix[static_cast<std::size_t>(column)]
                                [static_cast<std::size_t>(entry)];
            }
            rhs[static_cast<std::size_t>(row)]
                -= factor * rhs[static_cast<std::size_t>(column)];
        }
    }
    solution.fill(0.0L);
    for (int row = dimension - 1; row >= 0; --row) {
        long double value = rhs[static_cast<std::size_t>(row)];
        for (int column = row + 1; column < dimension; ++column) {
            value -= matrix[static_cast<std::size_t>(row)]
                           [static_cast<std::size_t>(column)]
                * solution[static_cast<std::size_t>(column)];
        }
        solution[static_cast<std::size_t>(row)] =
            value
            / matrix[static_cast<std::size_t>(row)]
                    [static_cast<std::size_t>(row)];
    }
    for (int q = 0; q < dimension; ++q) {
        if (!std::isfinite(solution[static_cast<std::size_t>(q)])) {
            return false;
        }
    }
    return true;
}

// The five-mode counterpart solves the P2 fit in extended precision,
// using a scale-relative pivot floor to detect dependent features.
inline bool hamiltonian_sldg_solve_quadratic_system(
    std::array<std::array<long double, 5>, 5> matrix,
    std::array<long double, 5> rhs,
    std::array<long double, 5>& solution) {
    long double matrix_scale = 0.0L;
    for (int row = 0; row < 5; ++row) {
        for (int column = 0; column < 5; ++column) {
            matrix_scale = std::max(
                matrix_scale,
                std::abs(matrix[static_cast<std::size_t>(row)]
                               [static_cast<std::size_t>(column)]));
        }
    }
    if (!(matrix_scale > 0.0L)) {
        return false;
    }
    const long double pivot_floor =
        4096.0L * std::numeric_limits<long double>::epsilon()
        * matrix_scale;
    for (int pivot = 0; pivot < 5; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < 5; ++row) {
            if (std::abs(matrix[static_cast<std::size_t>(row)]
                                [static_cast<std::size_t>(pivot)])
                > std::abs(matrix[static_cast<std::size_t>(best)]
                                  [static_cast<std::size_t>(pivot)])) {
                best = row;
            }
        }
        if (std::abs(matrix[static_cast<std::size_t>(best)]
                           [static_cast<std::size_t>(pivot)])
            <= pivot_floor) {
            return false;
        }
        if (best != pivot) {
            std::swap(
                matrix[static_cast<std::size_t>(best)],
                matrix[static_cast<std::size_t>(pivot)]);
            std::swap(
                rhs[static_cast<std::size_t>(best)],
                rhs[static_cast<std::size_t>(pivot)]);
        }
        const long double diagonal =
            matrix[static_cast<std::size_t>(pivot)]
                  [static_cast<std::size_t>(pivot)];
        for (int column = pivot; column < 5; ++column) {
            matrix[static_cast<std::size_t>(pivot)]
                  [static_cast<std::size_t>(column)] /= diagonal;
        }
        rhs[static_cast<std::size_t>(pivot)] /= diagonal;
        for (int row = 0; row < 5; ++row) {
            if (row == pivot) {
                continue;
            }
            const long double factor =
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(pivot)];
            for (int column = pivot; column < 5; ++column) {
                matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(column)] -=
                    factor
                    * matrix[static_cast<std::size_t>(pivot)]
                            [static_cast<std::size_t>(column)];
            }
            rhs[static_cast<std::size_t>(row)] -=
                factor * rhs[static_cast<std::size_t>(pivot)];
        }
    }
    solution = rhs;
    return true;
}

// Normalize each feature by its observed stencil scale before assembling
// weighted least-squares normal equations; restore coefficient units afterward.
inline bool hamiltonian_sldg_fit_quadratic(
    const std::array<HamiltonianSLDGQuadraticSample, 24>& sample,
    int sample_count,
    std::array<double, 5>& coefficient) {
    if (sample_count < 5) {
        return false;
    }
    std::array<double, 5> scale{};
    for (int q = 0; q < sample_count; ++q) {
        for (int entry = 0; entry < 5; ++entry) {
            scale[static_cast<std::size_t>(entry)] = std::max(
                scale[static_cast<std::size_t>(entry)],
                std::abs(sample[static_cast<std::size_t>(q)].feature[
                    static_cast<std::size_t>(entry)]));
        }
    }
    for (double value : scale) {
        if (!(value > 4096.0 * std::numeric_limits<double>::epsilon())) {
            return false;
        }
    }
    std::array<std::array<long double, 5>, 5> normal{};
    std::array<long double, 5> rhs{};
    for (int q = 0; q < sample_count; ++q) {
        const HamiltonianSLDGQuadraticSample& value =
            sample[static_cast<std::size_t>(q)];
        std::array<long double, 5> normalized{};
        for (int entry = 0; entry < 5; ++entry) {
            normalized[static_cast<std::size_t>(entry)] =
                value.feature[static_cast<std::size_t>(entry)]
                / scale[static_cast<std::size_t>(entry)];
        }
        for (int row = 0; row < 5; ++row) {
            rhs[static_cast<std::size_t>(row)] +=
                static_cast<long double>(value.weight)
                * normalized[static_cast<std::size_t>(row)]
                * value.delta_f;
            for (int column = 0; column < 5; ++column) {
                normal[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(column)] +=
                    static_cast<long double>(value.weight)
                    * normalized[static_cast<std::size_t>(row)]
                    * normalized[static_cast<std::size_t>(column)];
            }
        }
    }
    std::array<long double, 5> normalized_coefficient{};
    if (!hamiltonian_sldg_solve_quadratic_system(
            normal, rhs, normalized_coefficient)) {
        return false;
    }
    for (int entry = 0; entry < 5; ++entry) {
        coefficient[static_cast<std::size_t>(entry)] =
            static_cast<double>(normalized_coefficient[
                static_cast<std::size_t>(entry)])
            / scale[static_cast<std::size_t>(entry)];
    }
    return true;
}

struct HamiltonianSLDGRegressionSample {
    double delta_h = 0.0;
    double delta_h_curvature = 0.0;
    double delta_tau = 0.0;
    double delta_f = 0.0;
    double weight = 1.0;
};

// Fit selected energy slope, energy curvature, and orbit-tangent slope.
// Optional directions change system dimension while retaining the same
// normalized weighted least-squares construction.
inline bool hamiltonian_sldg_fit_regression(
    const std::array<HamiltonianSLDGRegressionSample, 8>& sample,
    int sample_count,
    double h_scale,
    double curvature_scale,
    double tau_scale,
    bool include_curvature,
    bool include_phase,
    std::array<double, 3>& coefficient) {
    const int dimension =
        1 + (include_curvature ? 1 : 0) + (include_phase ? 1 : 0);
    if (sample_count < dimension) {
        return false;
    }
    std::array<std::array<long double, 3>, 3> normal{};
    std::array<long double, 3> rhs{};
    for (int sample_index = 0;
         sample_index < sample_count;
         ++sample_index) {
        const HamiltonianSLDGRegressionSample& value =
            sample[static_cast<std::size_t>(sample_index)];
        std::array<long double, 3> feature{};
        int entry = 0;
        feature[static_cast<std::size_t>(entry++)] =
            value.delta_h / h_scale;
        if (include_curvature) {
            feature[static_cast<std::size_t>(entry++)] =
                value.delta_h_curvature / curvature_scale;
        }
        if (include_phase) {
            feature[static_cast<std::size_t>(entry++)] =
                value.delta_tau / tau_scale;
        }
        for (int row = 0; row < dimension; ++row) {
            rhs[static_cast<std::size_t>(row)] +=
                static_cast<long double>(value.weight)
                * feature[static_cast<std::size_t>(row)]
                * value.delta_f;
            for (int column = 0; column < dimension; ++column) {
                normal[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(column)] +=
                    static_cast<long double>(value.weight)
                    * feature[static_cast<std::size_t>(row)]
                    * feature[static_cast<std::size_t>(column)];
            }
        }
    }
    std::array<long double, 3> normalized{};
    if (!hamiltonian_sldg_solve_dense_system(
            normal, rhs, dimension, normalized)) {
        return false;
    }
    coefficient.fill(0.0);
    int entry = 0;
    coefficient[0] =
        static_cast<double>(normalized[static_cast<std::size_t>(entry++)])
        / h_scale;
    if (include_curvature) {
        coefficient[1] =
            static_cast<double>(normalized[static_cast<std::size_t>(entry++)])
            / curvature_scale;
    }
    if (include_phase) {
        coefficient[2] =
            static_cast<double>(normalized[static_cast<std::size_t>(entry++)])
            / tau_scale;
    }
    return true;
}

/// Scales only the mean-free affine part needed to keep f nonnegative.
///
/// An affine function reaches its minimum over a rectangular source cell at a
/// corner, so the four corner checks are exact.  Scaling the mean-free part
/// leaves the represented cell mass unchanged.
inline double hamiltonian_sldg_limit_positive_linear_reconstruction(
    double cell_average,
    double slope_r,
    double slope_w,
    double half_dr,
    double half_dw) {
    if (!(cell_average > 0.0)) {
        return 0.0;
    }
    const double minimum_delta =
        -std::abs(slope_r) * half_dr
        -std::abs(slope_w) * half_dw;
    if (cell_average + minimum_delta >= 0.0) {
        return 1.0;
    }
    const double theta = std::clamp(
        cell_average / (-minimum_delta), 0.0, 1.0);
    return theta > 0.0 && theta < 1.0
        ? std::nextafter(theta, 0.0)
        : theta;
}

/// Builds a general local reconstruction in (H,tau) at every phase-space cell.
// Fit neighboring cell averages, including their finite cell widths, in
// locally energy-normal/tangent coordinates. Carry the source mean exactly
// and limit only mean-free modes when the parent-cell minimum is negative.
inline HamiltonianSLDGReconstruction
build_hamiltonian_sldg_reconstruction(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    validate_finite_nonnegative_distribution(
        f,
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL,
        "Hamiltonian SLDG reconstruction");
    (void)x_r;
    (void)r;
    HamiltonianSLDGReconstruction reconstruction;
    reconstruction.slope.assign(f.size(), {0.0, 0.0});
    reconstruction.polynomial.resize(f.size());

    const int work = p.Nr * p.Nw * p.NL;
    const int threads = effective_threads(p, work);
    std::vector<HamiltonianSLDGReconstruction> partial(
        static_cast<std::size_t>(threads));
    parallel_for_chunks(
        0,
        work,
        threads,
        [&](int lo, int hi, int tid) {
            HamiltonianSLDGReconstruction& diagnostic =
                partial[static_cast<std::size_t>(tid)];
            for (int flat = lo; flat < hi; ++flat) {
                const int i = flat / (p.Nw * p.NL);
                const int remainder = flat - i * p.Nw * p.NL;
                const int j = remainder / p.NL;
                const int k = remainder - j * p.NL;
                const std::size_t cell = idx(i, j, k, p);
                const double center_f = f[cell];
                const double rc =
                    geometry.radial.center[static_cast<std::size_t>(i)];
                const double wc =
                    geometry.radial_velocity.center[
                        static_cast<std::size_t>(j)];
                const double dr =
                    geometry.radial.width[static_cast<std::size_t>(i)];
                const double dw =
                    geometry.radial_velocity.width[
                        static_cast<std::size_t>(j)];
                SLDGSourcePolynomial polynomial;
                polynomial.cell_average = center_f;
                polynomial.center_r = rc;
                polynomial.center_w = wc;
                polynomial.scale_r = dr;
                polynomial.scale_w = dw;
                reconstruction.polynomial[cell] = polynomial;
                if (!(center_f > 0.0)) {
                    continue;
                }
                if (center_f < std::numeric_limits<double>::min()) {
                    // Keep the conservative cell average, but use a constant
                    // reconstruction where double precision cannot resolve a
                    // positivity-preserving higher-order slope.
                    continue;
                }
                ++diagnostic.reconstructed_cells;
                const double eta =
                    geometry.eta.center[static_cast<std::size_t>(k)];
                // Grad H=(dPhi/dr-2*eta/r^3,w). Scale each component by
                // its cell width before defining a local orthogonal frame.
                const double h_r =
                    field.phi_r[static_cast<std::size_t>(i)]
                    - 2.0 * eta / (rc * rc * rc);
                const double scaled_h_r = h_r * dr;
                const double scaled_h_w = wc * dw;
                const double scaled_gradient_norm =
                    std::hypot(scaled_h_r, scaled_h_w);
                const double gradient_floor =
                    4096.0 * std::numeric_limits<double>::epsilon()
                    * std::max({
                        1.0,
                        std::abs(scaled_h_r),
                        std::abs(scaled_h_w),
                    });
                // At a nearly zero scaled Hamiltonian gradient the normal
                // direction is unresolved. Record that degeneracy and use
                // the stated coordinate fallback for the reconstruction fit.
                const bool phase_available =
                    scaled_gradient_norm > gradient_floor;
                if (!phase_available) {
                    ++diagnostic.phase_degenerate_cells;
                }
                const double normal_r = phase_available
                    ? scaled_h_r / scaled_gradient_norm
                    : 1.0;
                const double normal_w = phase_available
                    ? scaled_h_w / scaled_gradient_norm
                    : 0.0;
                const double tangent_r = -normal_w;
                const double tangent_w = normal_r;

                // Neighbor averages constrain a local five-mode quadratic.
                // Integrating their normalized widths accounts for a finite
                // source cell instead of treating each average as a point value.
                std::array<HamiltonianSLDGQuadraticSample, 24> samples{};
                int sample_count = 0;
                for (int di = -2; di <= 2; ++di) {
                    for (int dj = -2; dj <= 2; ++dj) {
                        if ((di == 0 && dj == 0)
                            || i + di < 0 || i + di >= p.Nr
                            || j + dj < 0 || j + dj >= p.Nw) {
                            continue;
                        }
                        const int in = i + di;
                        const int jn = j + dj;
                        const std::size_t neighbor = idx(in, jn, k, p);
                        const double mean_x =
                            (geometry.radial.center[
                                static_cast<std::size_t>(in)] - rc) / dr;
                        const double mean_y =
                            (geometry.radial_velocity.center[
                                static_cast<std::size_t>(jn)] - wc) / dw;
                        const double width_x =
                            geometry.radial.width[
                                static_cast<std::size_t>(in)] / dr;
                        const double width_y =
                            geometry.radial_velocity.width[
                                static_cast<std::size_t>(jn)] / dw;
                        const double mean_x2 =
                            mean_x * mean_x + width_x * width_x / 12.0;
                        const double mean_y2 =
                            mean_y * mean_y + width_y * width_y / 12.0;
                        const double mean_xy = mean_x * mean_y;
                        const double mean_h =
                            normal_r * mean_x + normal_w * mean_y;
                        const double mean_tau =
                            tangent_r * mean_x + tangent_w * mean_y;
                        const double mean_h2 =
                            normal_r * normal_r * mean_x2
                            + 2.0 * normal_r * normal_w * mean_xy
                            + normal_w * normal_w * mean_y2;
                        const double mean_h_tau =
                            normal_r * tangent_r * mean_x2
                            + (normal_r * tangent_w
                               + normal_w * tangent_r) * mean_xy
                            + normal_w * tangent_w * mean_y2;
                        const double mean_tau2 =
                            tangent_r * tangent_r * mean_x2
                            + 2.0 * tangent_r * tangent_w * mean_xy
                            + tangent_w * tangent_w * mean_y2;
                        HamiltonianSLDGQuadraticSample value;
                        // Energy-normal and tangent squares have parent-cell
                        // mean 1/12; subtract it so every fitted mode carries
                        // zero mean and the central mass stays independent.
                        value.feature = {
                            mean_h,
                            mean_tau,
                            mean_h2 - 1.0 / 12.0,
                            mean_h_tau,
                            mean_tau2 - 1.0 / 12.0,
                        };
                        value.delta_f = f[neighbor] - center_f;
                        value.weight =
                            1.0 / static_cast<double>(di * di + dj * dj);
                        samples[static_cast<std::size_t>(sample_count++)] =
                            value;
                    }
                }
                std::array<double, 5> coefficient{};
                if (!hamiltonian_sldg_fit_quadratic(
                        samples, sample_count, coefficient)) {
                    ++diagnostic.rank_deficient_cells;
                    continue;
                }
                // Rotate the fitted normal/tangent modes back into the
                // common normalized (r,w) P2 basis used by the conservative remap.
                polynomial.mode[0] =
                    coefficient[0] * normal_r
                    + coefficient[1] * tangent_r;
                polynomial.mode[1] =
                    coefficient[0] * normal_w
                    + coefficient[1] * tangent_w;
                polynomial.mode[2] =
                    coefficient[2] * normal_r * normal_r
                    + coefficient[3] * normal_r * tangent_r
                    + coefficient[4] * tangent_r * tangent_r;
                polynomial.mode[3] =
                    2.0 * coefficient[2] * normal_r * normal_w
                    + coefficient[3] * (
                        normal_r * tangent_w
                        + normal_w * tangent_r)
                    + 2.0 * coefficient[4] * tangent_r * tangent_w;
                polynomial.mode[4] =
                    coefficient[2] * normal_w * normal_w
                    + coefficient[3] * normal_w * tangent_w
                    + coefficient[4] * tangent_w * tangent_w;
                bool finite = true;
                for (double value : polynomial.mode) {
                    finite = finite && std::isfinite(value);
                }
                if (!finite) {
                    ++diagnostic.rank_deficient_cells;
                    continue;
                }
                // Scale higher modes against the actual polynomial minimum,
                // preserving the cell average while making the source positive.
                const double theta = polynomial.enforce_positivity();
                if (theta < 1.0 - 64.0
                        * std::numeric_limits<double>::epsilon()) {
                    ++diagnostic.limited_cells;
                }
                reconstruction.polynomial[cell] = polynomial;
                reconstruction.slope[cell] = {
                    polynomial.mode[0] / dr,
                    polynomial.mode[1] / dw,
                };
                const double total_gradient_norm =
                    std::hypot(coefficient[0], coefficient[1]);
                if (total_gradient_norm > 0.0) {
                    diagnostic.maximum_phase_gradient_fraction = std::max(
                        diagnostic.maximum_phase_gradient_fraction,
                        std::min(
                            1.0,
                            std::abs(coefficient[1])
                                / total_gradient_norm));
                }
            }
        });
    // Combine disjoint worker counters and retain the largest relative
    // orbit-tangent gradient as a reconstruction diagnostic.
    for (const HamiltonianSLDGReconstruction& value : partial) {
        reconstruction.reconstructed_cells += value.reconstructed_cells;
        reconstruction.limited_cells += value.limited_cells;
        reconstruction.rank_deficient_cells +=
            value.rank_deficient_cells;
        reconstruction.phase_degenerate_cells +=
            value.phase_degenerate_cells;
        reconstruction.maximum_phase_gradient_fraction = std::max(
            reconstruction.maximum_phase_gradient_fraction,
            value.maximum_phase_gradient_fraction);
    }
    return reconstruction;
}

/// Rebuilds source polynomials from modal data carried by the previous remap.
// Reattach previously transported modes to current cell geometry/averages.
// Validate and positively limit the complete polynomial before it is reused.
inline HamiltonianSLDGReconstruction
restore_hamiltonian_sldg_reconstruction(
    const std::vector<double>& f,
    const std::vector<std::array<double, 5>>& modal_coefficients,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    if (modal_coefficients.size() != f.size()) {
        throw std::runtime_error(
            "Hamiltonian SLDG persistent-mode array size mismatch.");
    }
    HamiltonianSLDGReconstruction reconstruction;
    reconstruction.slope.assign(f.size(), {0.0, 0.0});
    reconstruction.polynomial.resize(f.size());
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = idx(i, j, k, p);
                SLDGSourcePolynomial polynomial;
                polynomial.cell_average = f[cell];
                polynomial.center_r = geometry.radial.center[
                    static_cast<std::size_t>(i)];
                polynomial.center_w = geometry.radial_velocity.center[
                    static_cast<std::size_t>(j)];
                polynomial.scale_r = geometry.radial.width[
                    static_cast<std::size_t>(i)];
                polynomial.scale_w = geometry.radial_velocity.width[
                    static_cast<std::size_t>(j)];
                polynomial.mode = modal_coefficients[cell];
                for (double coefficient : polynomial.mode) {
                    if (!std::isfinite(coefficient)) {
                        throw std::runtime_error(
                            "Hamiltonian SLDG restart contains a non-finite P2 mode.");
                    }
                }
                if (polynomial.enforce_positivity() < 1.0) {
                    ++reconstruction.limited_cells;
                }
                reconstruction.slope[cell] = {{
                    polynomial.mode[0] / polynomial.scale_r,
                    polynomial.mode[1] / polynomial.scale_w,
                }};
                reconstruction.polynomial[cell] = polynomial;
                if (polynomial.cell_average > 0.0) {
                    ++reconstruction.reconstructed_cells;
                }
            }
        }
    }
    return reconstruction;
}

/// Reconstructs the P2(r,w) coefficients linearly across each eta cell.
/// The eta-linear term has zero cell average.  A single positivity factor is
/// applied to all six coefficient slopes only when one eta face would make the
/// full polynomial negative somewhere in its (r,w) parent cell.
inline std::vector<SLDGSourcePolynomial>
build_hamiltonian_sldg_eta_node_polynomials(
    const std::vector<SLDGSourcePolynomial>& eta_average_polynomial,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    long long* limited_cells = nullptr) {
    const std::size_t cells =
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    if (eta_average_polynomial.size() != cells) {
        throw std::runtime_error(
            "Hamiltonian eta reconstruction received the wrong cell count.");
    }
    const auto& rule = phase_space_gauss_legendre_rule(
        p.vp_sldg_eta_quadrature_points);
    std::vector<SLDGSourcePolynomial> node_polynomial(
        rule.first.size() * cells);
    long long limited = 0;

    auto coefficient = [](const SLDGSourcePolynomial& polynomial, int mode) {
        return mode < 0
            ? polynomial.cell_average
            : polynomial.mode[static_cast<std::size_t>(mode)];
    };
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = idx(i, j, k, p);
                const SLDGSourcePolynomial& base =
                    eta_average_polynomial[cell];
                std::array<double, 6> derivative{};
                if (p.NL > 1) {
                    const int left = k == 0 ? 0 : k - 1;
                    const int right = k == p.NL - 1 ? p.NL - 1 : k + 1;
                    const double span =
                        geometry.eta.center[static_cast<std::size_t>(right)]
                        - geometry.eta.center[static_cast<std::size_t>(left)];
                    if (!(span > 0.0)) {
                        throw std::runtime_error(
                            "Hamiltonian eta reconstruction has a non-positive stencil span.");
                    }
                    const SLDGSourcePolynomial& lower =
                        eta_average_polynomial[idx(i, j, left, p)];
                    const SLDGSourcePolynomial& upper =
                        eta_average_polynomial[idx(i, j, right, p)];
                    for (int mode = -1; mode < 5; ++mode) {
                        derivative[static_cast<std::size_t>(mode + 1)] =
                            (coefficient(upper, mode) - coefficient(lower, mode))
                            / span;
                    }
                }

                // An affine eta dependence updates all six P2 coefficients,
                // including the mean, while its symmetric eta average is zero.
                auto shifted = [&](double eta_delta, double theta) {
                    SLDGSourcePolynomial polynomial = base;
                    polynomial.cell_average += theta * eta_delta * derivative[0];
                    for (std::size_t mode = 0; mode < 5; ++mode) {
                        polynomial.mode[mode] +=
                            theta * eta_delta * derivative[mode + 1];
                    }
                    return polynomial;
                };
                const double half_width = 0.5 * geometry.eta.width[
                    static_cast<std::size_t>(k)];
                // Positivity on both eta faces controls every intermediate
                // eta by affine convexity. Use the full (r,w) polynomial minimum.
                auto face_minimum = [&](double theta) {
                    return std::min(
                        shifted(-half_width, theta).minimum_on_parent_cell(),
                        shifted(half_width, theta).minimum_on_parent_cell());
                };
                double theta = 1.0;
                double coefficient_scale = std::abs(base.cell_average);
                for (double value : base.mode) {
                    coefficient_scale = std::max(
                        coefficient_scale, std::abs(value));
                }
                for (double value : derivative) {
                    coefficient_scale = std::max(
                        coefficient_scale,
                        half_width * std::abs(value));
                }
                const double positivity_tolerance =
                    256.0 * std::numeric_limits<double>::epsilon()
                    * std::max(
                        coefficient_scale,
                        std::numeric_limits<double>::min());
                // One common factor scales all coefficient slopes, retaining
                // their shared eta reconstruction and central-cell averages.
                if (face_minimum(theta) < 0.0) {
                    double lower = 0.0;
                    double upper = 1.0;
                    for (int iteration = 0; iteration < 64; ++iteration) {
                        const double middle = 0.5 * (lower + upper);
                        if (face_minimum(middle) >= 0.0) {
                            lower = middle;
                        } else {
                            upper = middle;
                        }
                    }
                    theta = lower > 0.0
                        ? std::nextafter(lower, 0.0)
                        : 0.0;
                    ++limited;
                }
                for (std::size_t q_eta = 0;
                     q_eta < rule.first.size();
                     ++q_eta) {
                    SLDGSourcePolynomial polynomial = shifted(
                        half_width * rule.first[q_eta], theta);
                    const double minimum =
                        polynomial.minimum_on_parent_cell();
                    if (minimum < -positivity_tolerance) {
                        throw std::runtime_error(
                            "Hamiltonian eta reconstruction failed positivity.");
                    }
                    if (minimum < 0.0) {
                        polynomial.enforce_positivity();
                    }
                    node_polynomial[q_eta * cells + cell] = polynomial;
                }
            }
        }
    }
    if (limited_cells != nullptr) {
        *limited_cells = limited;
    }
    return node_polynomial;
}

// Use one frozen field for the conservative geometric remap. Reconstruct
// total polynomials, resolve their eta quadrature, then select full or
// orbit-reference residual transport according to the configured policy.
inline std::pair<std::vector<double>, SLDGStepDiagnostics>
advect_vlasov_poisson_hamiltonian_sldg_frozen(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    DelayedInnerCarrier* carrier = nullptr,
    const std::vector<std::array<double, 5>>* source_modal_coefficients = nullptr,
    std::vector<std::array<double, 5>>* transported_modal_coefficients = nullptr) {
    if (!using_hamiltonian_sldg_transport(p)) {
        throw std::runtime_error(
            "Hamiltonian SLDG requires vp_interpolation=hamiltonian_sldg.");
    }
    HamiltonianSLDGReconstruction reconstruction =
        source_modal_coefficients != nullptr
        ? restore_hamiltonian_sldg_reconstruction(
            f, *source_modal_coefficients, geometry, p)
        : build_hamiltonian_sldg_reconstruction(
            f, x_r, r, field, geometry, p);
    const std::vector<SLDGSourcePolynomial> eta_node_total_polynomial =
        build_hamiltonian_sldg_eta_node_polynomials(
            reconstruction.polynomial,
            geometry,
            p,
            &reconstruction.eta_limited_cells);
    // The full-distribution path supplies Hamiltonian-aligned polynomials
    // directly to the common conservative source-cell transport.
    if (p.vp_hamiltonian_reference == "none") {
        auto result =
            advect_vlasov_poisson_conservative_sldg_frozen_with_polynomials(
            f,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            dt,
            p,
            &reconstruction.polynomial,
            "Hamiltonian SLDG",
            carrier,
            transported_modal_coefficients,
            false,
            &eta_node_total_polynomial);
        result.second.hamiltonian_reconstructed_cells =
            reconstruction.reconstructed_cells;
        result.second.hamiltonian_limited_cells =
            reconstruction.limited_cells;
        result.second.hamiltonian_rank_deficient_cells =
            reconstruction.rank_deficient_cells;
        result.second.hamiltonian_phase_degenerate_cells =
            reconstruction.phase_degenerate_cells;
        result.second.hamiltonian_eta_limited_cells =
            reconstruction.eta_limited_cells;
        result.second.hamiltonian_maximum_phase_gradient_fraction =
            reconstruction.maximum_phase_gradient_fraction;
        return result;
    }

    const DelayedInnerCarrierLedger carrier_before = carrier != nullptr
        ? evaluate_delayed_inner_carrier_ledger(*carrier, geometry, p)
        : DelayedInnerCarrierLedger{};
    // Rebuild the closed-orbit reference at actual eta nodes. The same
    // field defines its energy profile and the residual characteristics.
    const HamiltonianOrbitReference reference =
        build_hamiltonian_orbit_reference_eta_quadrature(
            f,
            reconstruction.polynomial,
            x_r,
            r,
            field,
            geometry,
            p,
            &eta_node_total_polynomial);
    std::vector<std::array<double, 5>> residual_modes;
    auto result =
        advect_vlasov_poisson_conservative_sldg_frozen_with_polynomials(
            reference.residual_cell_average,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            dt,
            p,
            &reference.residual_polynomial,
            "Hamiltonian orbit-residual SLDG",
            carrier,
            transported_modal_coefficients != nullptr
                ? &residual_modes
                : nullptr,
            false,
            &reference.eta_node_residual_polynomial);
    // Restore the stationary reference after moving only its residual.
    // Their sum is the complete physical distribution at the endpoint.
    for (std::size_t q = 0; q < result.first.size(); ++q) {
        result.first[q] += reference.cell_average[q];
    }
    const int work = static_cast<int>(result.first.size());
    const int threads = effective_threads(p, work);
    std::vector<long double> deposited_mass_by_thread(
        static_cast<std::size_t>(threads), 0.0L);
    std::vector<long double> source_mass_by_thread(
        static_cast<std::size_t>(threads), 0.0L);
    std::vector<long long> limited_by_thread(
        static_cast<std::size_t>(threads), 0);
    if (transported_modal_coefficients != nullptr) {
        transported_modal_coefficients->assign(
            result.first.size(), std::array<double, 5>{});
    }
    parallel_for_chunks(
        0,
        work,
        threads,
        [&](int lo, int hi, int tid) {
            long double deposited_mass = 0.0L;
            long double source_mass = 0.0L;
            long long limited = 0;
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int remainder = q - i * p.Nw * p.NL;
                const int j = remainder / p.NL;
                const int k = remainder - j * p.NL;
                const double total = result.first[static_cast<std::size_t>(q)];
                if (!std::isfinite(total) || total < 0.0) {
                    std::ostringstream message;
                    message << "Hamiltonian orbit-reference remap produced an "
                               "invalid total cell average: q="
                            << q << ", residual="
                            << result.first[static_cast<std::size_t>(q)]
                            << ", reference="
                            << reference.cell_average[static_cast<std::size_t>(q)]
                            << ", total=" << total << '.';
                    throw std::runtime_error(message.str());
                }
                const long double physical_measure = 8.0L * PI * PI
                    * geometry.radial.width[static_cast<std::size_t>(i)]
                    * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
                    * geometry.eta.width[static_cast<std::size_t>(k)];
                source_mass += physical_measure
                    * f[static_cast<std::size_t>(q)];
                deposited_mass += physical_measure * total;
                if (transported_modal_coefficients != nullptr) {
                    // Add transported residual modes to the fixed reference
                    // modes, then check positivity of their total polynomial.
                    SLDGSourcePolynomial destination =
                        reference.polynomial[static_cast<std::size_t>(q)];
                    destination.cell_average = total;
                    for (std::size_t mode = 0; mode < 5; ++mode) {
                        destination.mode[mode] += residual_modes[
                            static_cast<std::size_t>(q)][mode];
                    }
                    if (destination.enforce_positivity() < 1.0) {
                        ++limited;
                    }
                    (*transported_modal_coefficients)[
                        static_cast<std::size_t>(q)] = destination.mode;
                }
            }
            deposited_mass_by_thread[static_cast<std::size_t>(tid)] =
                deposited_mass;
            source_mass_by_thread[static_cast<std::size_t>(tid)] = source_mass;
            limited_by_thread[static_cast<std::size_t>(tid)] = limited;
        });
    long double source_mass = 0.0L;
    long double deposited_mass = 0.0L;
    for (int tid = 0; tid < threads; ++tid) {
        source_mass += source_mass_by_thread[static_cast<std::size_t>(tid)];
        deposited_mass += deposited_mass_by_thread[static_cast<std::size_t>(tid)];
        result.second.hamiltonian_projected_limited_cells +=
            limited_by_thread[static_cast<std::size_t>(tid)];
    }
    // Close mass for the complete source/reference/residual combination
    // together with hidden carriers before interpreting any escaped amount.
    const DelayedInnerCarrierLedger carrier_after = carrier != nullptr
        ? evaluate_delayed_inner_carrier_ledger(*carrier, geometry, p)
        : DelayedInnerCarrierLedger{};
    result.second.source_mass = static_cast<double>(source_mass);
    result.second.deposited_mass = static_cast<double>(deposited_mass);
    result.second.carrier_mass_before = carrier_before.physical_mass;
    result.second.carrier_mass_after = carrier_after.physical_mass;
    result.second.escaped_mass = result.second.source_mass
        + result.second.carrier_mass_before
        - result.second.deposited_mass
        - result.second.carrier_mass_after;
    const double budget_scale = std::max({
        1.0,
        result.second.source_mass + result.second.carrier_mass_before,
        result.second.deposited_mass + result.second.carrier_mass_after,
    });
    if (result.second.escaped_mass
        < -p.vp_inner_carrier_mass_tolerance * budget_scale) {
        std::ostringstream message;
        message << "Hamiltonian orbit-reference remap created mass in its "
                   "total ledger: source="
                << std::setprecision(17) << result.second.source_mass
                << ", hidden_before=" << result.second.carrier_mass_before
                << ", deposited=" << result.second.deposited_mass
                << ", hidden_after=" << result.second.carrier_mass_after
                << ", residual=" << result.second.escaped_mass
                << ", tolerance="
                << p.vp_inner_carrier_mass_tolerance * budget_scale << '.';
        throw std::runtime_error(message.str());
    }
    result.second.carrier_mass_residual = 0.0;
    result.second.hamiltonian_reconstructed_cells =
        reconstruction.reconstructed_cells;
    result.second.hamiltonian_limited_cells =
        reconstruction.limited_cells;
    result.second.hamiltonian_rank_deficient_cells =
        reconstruction.rank_deficient_cells;
    result.second.hamiltonian_phase_degenerate_cells =
        reconstruction.phase_degenerate_cells;
    result.second.hamiltonian_eta_limited_cells =
        reconstruction.eta_limited_cells;
    result.second.hamiltonian_maximum_phase_gradient_fraction =
        reconstruction.maximum_phase_gradient_fraction;
    result.second.hamiltonian_reference_mass_fraction =
        reference.reference_mass_fraction;
    result.second.hamiltonian_residual_l1_fraction =
        reference.residual_l1_fraction;
    result.second.hamiltonian_reference_active_planes =
        reference.active_energy_planes;
    result.second.hamiltonian_minimum_positive_reference_scale =
        reference.minimum_positive_reference_scale;
    result.second.hamiltonian_positive_reference_scaled_planes =
        reference.positive_reference_scaled_planes;
    result.second.hamiltonian_positivity_redistributed_mass =
        0.0;
    result.second.hamiltonian_positivity_redistributed_fraction =
        0.0;
    return result;
}

// Predict the midpoint density/current response, solve its self-consistent
// Poisson field, and transport the full interval in that midpoint field.
inline SLDGStepResult advect_vlasov_poisson_hamiltonian_sldg(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& initial_field,
    const Moments& initial_moments,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    double t,
    const Params& p,
    const MpiContext& mpi,
    DelayedInnerCarrier* carrier = nullptr,
    const std::vector<std::array<double, 5>>* source_modal_coefficients = nullptr) {
    (void)initial_field;
    const double local_hidden_mass = carrier != nullptr
        ? evaluate_delayed_inner_carrier_ledger(
            *carrier, geometry, p).physical_mass
        : 0.0;
    // Hidden carrier mass is part of the global gravitational source, so
    // reduce it across eta owners before constructing the midpoint field.
    const double hidden_mass =
        mpi_allreduce_sum(local_hidden_mass, mpi);
    const SLDGMidpointPrediction predictor =
        predict_vlasov_poisson_midpoint_moments(
            initial_moments, geometry, 0.5 * dt, p, hidden_mass);
    VlasovPoissonField midpoint_field =
        solve_vlasov_poisson_field(
            predictor.moments,
            r,
            t + 0.5 * dt,
            p,
            predictor.hidden_mass_after);
    std::vector<std::array<double, 5>> transported_modal_coefficients;
    auto final = advect_vlasov_poisson_hamiltonian_sldg_frozen(
        f,
        x_r,
        r,
        w,
        L,
        midpoint_field,
        geometry,
        dt,
        p,
        carrier,
        source_modal_coefficients,
        &transported_modal_coefficients);
    SLDGStepResult result = finish_vlasov_poisson_sldg_step(
        std::move(final),
        std::move(midpoint_field),
        predictor,
        mpi);
    result.modal_coefficients = std::move(transported_modal_coefficients);
    result.has_modal_coefficients = true;
    return result;
}

/// Uses the discrete continuity equation and the same Poisson solve to obtain
/// the instantaneous represented dPhi/dt.
inline VlasovPoissonField vp_potential_time_derivative(
    const Moments& moments,
    const std::vector<double>& r,
    double t,
    const Params& p) {
    const std::vector<double> edge = radial_control_volume_edges(r, p);
    // Continuity gives shell mass rate as minus the difference of radial
    // mass fluxes 4*pi*r^2*j at adjacent faces.
    std::vector<double> face_flux(
        static_cast<std::size_t>(p.Nr + 1), 0.0);
    for (int face = 1; face < p.Nr; ++face) {
        const double radius = edge[static_cast<std::size_t>(face)];
        const double fraction = std::clamp(
            (radius - r[static_cast<std::size_t>(face - 1)])
                / (r[static_cast<std::size_t>(face)]
                   - r[static_cast<std::size_t>(face - 1)]),
            0.0,
            1.0);
        const double current =
            (1.0 - fraction)
                * moments.j[static_cast<std::size_t>(face - 1)]
            + fraction * moments.j[static_cast<std::size_t>(face)];
        face_flux[static_cast<std::size_t>(face)] =
            4.0 * PI * radius * radius * current;
    }
    // A delayed inner carrier exchanges signed flux with the grid. Other
    // open faces retain only their outward-loss direction; reflecting faces
    // keep the zero boundary flux initialized above.
    if (using_delayed_inner_return(p)) {
        const double radius = edge.front();
        face_flux.front() =
            4.0 * PI * radius * radius * moments.j.front();
    } else if (p.inner_boundary != "reflecting") {
        const double radius = edge.front();
        face_flux.front() = std::min(
            0.0,
            4.0 * PI * radius * radius * moments.j.front());
    }
    if (p.outer_boundary != "reflecting") {
        const double radius = edge.back();
        face_flux.back() = std::max(
            0.0,
            4.0 * PI * radius * radius * moments.j.back());
    }

    Moments derivative;
    derivative.rho.assign(static_cast<std::size_t>(p.Nr), 0.0);
    derivative.p.assign(static_cast<std::size_t>(p.Nr), 0.0);
    derivative.pt.assign(static_cast<std::size_t>(p.Nr), 0.0);
    derivative.j.assign(static_cast<std::size_t>(p.Nr), 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        const double shell_mass_rate =
            -(face_flux[static_cast<std::size_t>(i + 1)]
              - face_flux[static_cast<std::size_t>(i)]);
        const double measure =
            4.0 * PI * r[static_cast<std::size_t>(i)]
                * r[static_cast<std::size_t>(i)]
            * (edge[static_cast<std::size_t>(i + 1)]
               - edge[static_cast<std::size_t>(i)]);
        derivative.rho[static_cast<std::size_t>(i)] =
            shell_mass_rate / measure;
    }
    // Poisson is linear in source mass. Supply hidden-mass rate as the
    // central source parameter for this derivative solve, distinguishing it
    // from a physical point-mass value in an ordinary field solve.
    Params derivative_params = p;
    derivative_params.central_mass = using_delayed_inner_return(p)
        ? -face_flux.front()
        : 0.0;
    return solve_vlasov_poisson_field(
        derivative, r, t, derivative_params);
}

inline VPPotentialVariationLimit
vp_potential_variation_limit(
    const Moments& moments,
    const VlasovPoissonField& field,
    const std::vector<double>& r,
    double t,
    const Params& p) {
    VPPotentialVariationLimit limit;
    // This instantaneous continuity-based variation cap is applied to the
    // static formulation; the expanding formulation keeps an infinite cap here.
    if (using_expanding_vp(p)) {
        return limit;
    }
    const VlasovPoissonField derivative =
        vp_potential_time_derivative(
            moments, r, t, p);
    double maximum_phi = 0.0;
    double maximum_phi_rate = 0.0;
    double maximum_force = 0.0;
    double maximum_force_rate = 0.0;
    for (int i = 0; i < p.Nr; ++i) {
        maximum_phi = std::max(
            maximum_phi,
            std::abs(field.phi[static_cast<std::size_t>(i)]));
        maximum_phi_rate = std::max(
            maximum_phi_rate,
            std::abs(derivative.phi[static_cast<std::size_t>(i)]));
        maximum_force = std::max(
            maximum_force,
            std::abs(field.phi_r[static_cast<std::size_t>(i)]));
        maximum_force_rate = std::max(
            maximum_force_rate,
            std::abs(derivative.phi_r[static_cast<std::size_t>(i)]));
    }
    // Resolve both field amplitude and rate above their scale-relative
    // roundoff floors before forming a potentially restrictive ratio.
    const double phi_floor =
        4096.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, maximum_phi);
    const double phi_rate_floor =
        4096.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, maximum_phi_rate);
    const double force_floor =
        4096.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, maximum_force);
    const double force_rate_floor =
        4096.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, maximum_force_rate);
    for (int i = 0; i < p.Nr; ++i) {
        const double phi =
            std::abs(field.phi[static_cast<std::size_t>(i)]);
        const double phi_rate =
            std::abs(derivative.phi[static_cast<std::size_t>(i)]);
        if (phi > phi_floor && phi_rate > phi_rate_floor) {
            const double timescale = phi / phi_rate;
            if (timescale < limit.potential_timescale) {
                limit.potential_timescale = timescale;
                limit.controlling_potential_cell = i;
            }
        }
        const double force =
            std::abs(field.phi_r[static_cast<std::size_t>(i)]);
        const double force_rate =
            std::abs(derivative.phi_r[static_cast<std::size_t>(i)]);
        if (force > force_floor && force_rate > force_rate_floor) {
            const double timescale = force / force_rate;
            if (timescale < limit.force_timescale) {
                limit.force_timescale = timescale;
                limit.controlling_force_cell = i;
            }
        }
    }
    // The shorter potential/force time controls. vp_potential_variation_eta
    // is the configured timestep fraction, separate from orbital eta=L^2/2.
    limit.timescale =
        std::min(limit.potential_timescale, limit.force_timescale);
    if (std::isfinite(limit.timescale)) {
        limit.timestep =
            p.vp_potential_variation_eta * limit.timescale;
    }
    return limit;
}

}  // namespace ev
