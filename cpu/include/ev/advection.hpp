#pragma once

// Pointwise semi-Lagrangian transport on the spherical (r,w,L) grid.
// For each destination node, trace its characteristic backward through the supplied
// frozen metric and interpolate the old distribution at the departure point.
// Boundary helpers express either regular-origin parity or reflection at a finite
// physical wall. Separate routines enlarge the momentum box and estimate escaped
// support. Inputs are f, grid/quadrature arrays, metric, timestep and controls;
// outputs are the transported/remapped distribution and diagnostic budgets.

#include "ev/characteristic_integrators.hpp"
#include "ev/grids.hpp"
#include "ev/parallel.hpp"
#include "ev/mpi.hpp"

namespace ev {

/// Mirrors a characteristic footpoint through the finite inner reflecting wall.
// At a fixed physical wall, u=H*r+w changes sign. H=0 also covers
// static Newtonian motion and the relativistic radial momentum convention.
void reflect_inner(double& rp, double& wp, double r_min, double hubble = 0.0) {
    if (rp >= r_min) {
        return;
    }
    rp = r_min + (r_min - rp);
    wp = -wp - 2.0 * hubble * r_min;
}

/// Mirrors a characteristic footpoint through the true spherical origin.
void reflect_origin(double& rp, double& wp) {
    if (rp >= 0.0) {
        return;
    }
    rp = -rp;
    wp = -wp;
}

/// Mirrors a characteristic footpoint through a finite outer reflecting wall.
void reflect_outer(double& rp, double& wp, double r_max, double hubble = 0.0) {
    if (rp <= r_max) {
        return;
    }
    rp = r_max - (rp - r_max);
    wp = -wp - 2.0 * hubble * r_max;
}

/// True when the radial box is closed at the outer edge.
bool using_reflecting_outer(const Params& p) {
    return p.outer_boundary == "reflecting";
}

/// True when the inner boundary is the regular spherical origin rather than a wall at r_min.
bool using_origin_parity_inner(const Params& p) {
    return p.inner_boundary == "origin_parity";
}

/// Applies radial wall reflections to a characteristic footpoint.
// Apply the configured physical boundary after each coordinate reflection.
// A finite iteration limit catches steps whose footpoint remains outside the box.
void reflect_radial_boundaries(double& rp, double& wp, double r_min, double r_max, const Params& p, double hubble = 0.0) {
    for (int n = 0; n < 8; ++n) {
        if (using_origin_parity_inner(p) && rp < 0.0) {
            reflect_origin(rp, wp);
            continue;
        }
        if (!using_origin_parity_inner(p) && rp < r_min) {
            reflect_inner(rp, wp, r_min, hubble);
            continue;
        }
        if (using_reflecting_outer(p) && rp > r_max) {
            reflect_outer(rp, wp, r_max, hubble);
            continue;
        }
        break;
    }
    if (!std::isfinite(rp) || !std::isfinite(wp)
        || rp < (using_origin_parity_inner(p) ? 0.0 : r_min)
        || (using_reflecting_outer(p) && rp > r_max)) {
        throw std::runtime_error(
            "Radial reflection failed: reduce the characteristic timestep.");
    }
}

/// Maps an origin-parity footpoint into the positive radial grid used for interpolation.
void regularize_radial_footpoint_for_interpolation(
    double& rp,
    double& wp,
    const std::vector<double>& r,
    const Params& p) {
    if (using_origin_parity_inner(p)) {
        reflect_origin(rp, wp);
        if (rp < r.front()) {
            rp = r.front();
        }
    }
}

/// Maps an origin-parity radius into the positive radial grid used for profile sampling.
double regularize_radial_profile_sample(double rp, const std::vector<double>& r, const Params& p) {
    if (using_origin_parity_inner(p)) {
        rp = std::abs(rp);
        if (rp < r.front()) {
            return r.front();
        }
    }
    return rp;
}

/// Declares the helper that locates the lower interpolation cell for a grid value.
int lower_cell(const std::vector<double>& grid, double value);

/// Interpolates a one-dimensional momentum slice at a requested momentum coordinate.
double interpolate_w_value(const std::vector<double>& values, double wp, const std::vector<double>& w) {
    if (wp < w.front() || wp > w.back()) {
        return 0.0;
    }
    const int j = lower_cell(w, wp);
    const double tw = (wp - w[j]) / (w[j + 1] - w[j]);
    return (1.0 - tw) * values[j] + tw * values[j + 1];
}

/// Symmetrizes f at r_min so the finite wall approximates regular-center reflection.
void enforce_inner_wall_symmetry(std::vector<double>& f, const std::vector<double>& w, const Params& p, double hubble = 0.0) {
    // Origin parity relates points at +r and -r. At a positive first grid
    // point it permits nonzero radial current, so only a finite wall is
    // symmetrized here.
    if (using_origin_parity_inner(p)) {
        return;
    }
    std::vector<double> boundary(p.Nw);
    for (int k = 0; k < p.NL; ++k) {
        for (int j = 0; j < p.Nw; ++j) {
            boundary[j] = f[idx(0, j, k, p)];
        }
        for (int j = 0; j < p.Nw; ++j) {
            const double mirrored = interpolate_w_value(
                boundary, -w[j] - 2.0 * hubble * p.r_min, w);
            f[idx(0, j, k, p)] = 0.5 * (boundary[j] + mirrored);
        }
    }
}

/// Computes the phase-space mass represented by f using tensor-product quadrature weights.
// Angular integration supplies 8*pi^2*L dL dr dw. The VP branch weights this
// by unity; the relativistic branch reports the energy-weighted phase integral.
double distribution_mass(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& rq,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p,
    const MpiContext& mpi) {
    const int work = static_cast<int>(f.size());
    const int nthreads = effective_threads(p, work);
    std::vector<double> local(static_cast<std::size_t>(nthreads), 0.0);
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            double sum = 0.0;
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double eps = std::sqrt(
                    1.0 + w[j] * w[j] + (L[k] / r[i]) * (L[k] / r[i]));
                const double phase_weight = using_vlasov_poisson(p) ? 1.0 : eps;
                sum += 8.0 * PI * PI * phase_weight * L[k] * f[static_cast<std::size_t>(q)]
                    * rq[i] * wq[j] * Lq[k];
            }
            local[static_cast<std::size_t>(tid)] = sum;
        });
    double total = 0.0;
    for (double x : local) {
        total += x;
    }
    return mpi_allreduce_sum(total, mpi);
}

struct DistributionCollisionInvariants {
    double mass = 0.0;
    double radial_momentum = 0.0;
    double kinetic_energy = 0.0;
};

/// Computes the global VP mass, radial momentum, and kinetic energy on the
/// same phase-space quadrature used by the local finite-angle collision step.
DistributionCollisionInvariants distribution_collision_invariants(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& rq,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p,
    const MpiContext& mpi) {
    const int work = static_cast<int>(f.size());
    const int nthreads = effective_threads(p, work);
    std::vector<DistributionCollisionInvariants> local(static_cast<std::size_t>(nthreads));
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            DistributionCollisionInvariants sum;
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double vt = L[static_cast<std::size_t>(k)] / r[static_cast<std::size_t>(i)];
                const double phase_mass = 8.0 * PI * PI
                    * L[static_cast<std::size_t>(k)]
                    * f[static_cast<std::size_t>(q)]
                    * rq[static_cast<std::size_t>(i)]
                    * wq[static_cast<std::size_t>(j)]
                    * Lq[static_cast<std::size_t>(k)];
                sum.mass += phase_mass;
                sum.radial_momentum += phase_mass * w[static_cast<std::size_t>(j)];
                sum.kinetic_energy += phase_mass * 0.5
                    * (w[static_cast<std::size_t>(j)] * w[static_cast<std::size_t>(j)] + vt * vt);
            }
            local[static_cast<std::size_t>(tid)] = sum;
        });
    DistributionCollisionInvariants total;
    for (const DistributionCollisionInvariants& value : local) {
        total.mass += value.mass;
        total.radial_momentum += value.radial_momentum;
        total.kinetic_energy += value.kinetic_energy;
    }
    total.mass = mpi_allreduce_sum(total.mass, mpi);
    total.radial_momentum = mpi_allreduce_sum(total.radial_momentum, mpi);
    total.kinetic_energy = mpi_allreduce_sum(total.kinetic_energy, mpi);
    return total;
}

/// Evaluates the unit-rest-mass relativistic energy sqrt(1+w^2+L^2/r^2).
double particle_energy(double r, double w, double L) {
    return std::sqrt(1.0 + w * w + (L / r) * (L / r));
}

/// Remaps f from one w grid to another while approximately preserving the model's phase-space mass.
// Integrate overlap lengths of old and new momentum control volumes. The
// relativistic energy factor is included before dividing by the new cell measure.
void conservative_energy_remap_w(
    const std::vector<double>& old_f,
    std::vector<double>& f_new,
    const std::vector<double>& r,
    const std::vector<double>& old_w,
    const std::vector<double>& new_w,
    const std::vector<double>& L,
    const Params& p) {
    const std::vector<double> old_edges = control_volume_edges(old_w);
    const std::vector<double> new_edges = control_volume_edges(new_w);

    parallel_for_chunks(0, p.Nr, effective_threads(p, p.Nr),
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                const double ri = r[static_cast<std::size_t>(i)];
                for (int k = 0; k < p.NL; ++k) {
                    const double Lk = L[static_cast<std::size_t>(k)];
                    int old_j_start = 0;
                    for (int j = 0; j < p.Nw; ++j) {
                        const double a = new_edges[static_cast<std::size_t>(j)];
                        const double b = new_edges[static_cast<std::size_t>(j + 1)];
                        const double width = b - a;
                        if (!(width > 0.0)) {
                            continue;
                        }

                        while (old_j_start < p.Nw
                               && old_edges[static_cast<std::size_t>(old_j_start + 1)] <= a) {
                            ++old_j_start;
                        }

                        double cell_energy = 0.0;
                        for (int jo = old_j_start; jo < p.Nw; ++jo) {
                            const double old_a = old_edges[static_cast<std::size_t>(jo)];
                            const double old_b = old_edges[static_cast<std::size_t>(jo + 1)];
                            if (old_a >= b) {
                                break;
                            }
                            const double left = std::max(a, old_a);
                            const double right = std::min(b, old_b);
                            if (right <= left) {
                                continue;
                            }
                            const double weight_old = using_vlasov_poisson(p)
                                ? 1.0
                                : particle_energy(ri, old_w[static_cast<std::size_t>(jo)], Lk);
                            cell_energy += (right - left) * weight_old * old_f[idx(i, jo, k, p)];
                        }

                        if (cell_energy <= 0.0) {
                            continue;
                        }
                        const double weight_new = using_vlasov_poisson(p)
                            ? 1.0
                            : particle_energy(ri, new_w[static_cast<std::size_t>(j)], Lk);
                        f_new[idx(i, j, k, p)] = std::max(0.0, cell_energy / (width * weight_new));
                    }
                }
            }
        });
}

// Expand the momentum box when active support approaches |w|=w_max.  This is
// intentionally one-way expansion.  The remap below conserves the
// cell-integrated phase-space mass in each (r,L) slice, rather than
// interpolating point values of f and then hoping the new quadrature weights
// agree.  The relativistic branch uses an energy weight; VP uses unit weight.
/// Expands and remaps the w grid when active support approaches the momentum boundary.
bool adapt_w_grid_if_needed(
    std::vector<double>& f,
    std::vector<double>& w,
    std::vector<double>& wq,
    double& min_dw,
    const std::vector<double>& r,
    const std::vector<double>& L,
    const std::vector<double>& rq,
    const std::vector<double>& Lq,
    Params& p,
    int step,
    double t,
    std::ostream* w_grid_history,
    const MpiContext& mpi) {
    if (!p.adaptive_w || step <= 0 || step % p.adaptive_w_every != 0) {
        return false;
    }

    const double max_f = mpi_allreduce_max(max_value_parallel(f, p), mpi);
    if (max_f <= 0.0) {
        return false;
    }
    // Locate the occupied velocity extent using the configured distribution floor.
    // This threshold selects the enlargement trigger; the remap integrates all cells.
    const double threshold = std::max(p.active_floor_abs, p.active_floor_rel * max_f);
    double active_min = std::numeric_limits<double>::infinity();
    double active_max = -std::numeric_limits<double>::infinity();
    const int support_work = p.Nr * p.Nw;
    const int nthreads = effective_threads(p, support_work);
    std::vector<double> local_min(static_cast<std::size_t>(nthreads), std::numeric_limits<double>::infinity());
    std::vector<double> local_max(static_cast<std::size_t>(nthreads), -std::numeric_limits<double>::infinity());
    parallel_for_chunks(0, support_work, nthreads,
        [&](int lo, int hi, int tid) {
            double active_min_local = std::numeric_limits<double>::infinity();
            double active_max_local = -std::numeric_limits<double>::infinity();
            for (int q = lo; q < hi; ++q) {
                const int i = q / p.Nw;
                const int j = q - i * p.Nw;
                bool active = false;
                for (int k = 0; k < p.NL; ++k) {
                    if (f[idx(i, j, k, p)] > threshold) {
                        active = true;
                        break;
                    }
                }
                if (active) {
                    active_min_local = std::min(active_min_local, w[j]);
                    active_max_local = std::max(active_max_local, w[j]);
                }
            }
            local_min[static_cast<std::size_t>(tid)] = active_min_local;
            local_max[static_cast<std::size_t>(tid)] = active_max_local;
        });
    for (int q = 0; q < nthreads; ++q) {
        active_min = std::min(active_min, local_min[static_cast<std::size_t>(q)]);
        active_max = std::max(active_max, local_max[static_cast<std::size_t>(q)]);
    }
    active_min = mpi_allreduce_min(active_min, mpi);
    active_max = mpi_allreduce_max(active_max, mpi);
    if (!std::isfinite(active_min) || !std::isfinite(active_max)) {
        return false;
    }

    const double active_abs = std::max(std::abs(active_min), std::abs(active_max));
    if (active_abs < p.adaptive_w_trigger * p.w_max) {
        return false;
    }

    double new_w_max = std::max(p.w_max * p.adaptive_w_growth, active_abs * p.adaptive_w_margin);
    if (p.adaptive_w_max > 0.0) {
        new_w_max = std::min(new_w_max, p.adaptive_w_max);
    }
    if (!(new_w_max > p.w_max * (1.0 + 1.0e-12))) {
        return false;
    }

    // Keep the old distribution and quadrature until both before/after budgets
    // have been measured. The new velocity box expands while keeping the cell count.
    const double old_w_max = p.w_max;
    const std::vector<double> old_w = w;
    const std::vector<double> old_wq = wq;
    const std::vector<double> old_f = f;
    const double mass_before = distribution_mass(old_f, r, old_w, L, rq, old_wq, Lq, p, mpi);
    p.w_max = new_w_max;
    if (using_binary_initial_data(p)) {
        const double scale = p.w_max / std::max(old_w_max, 1.0e-300);
        for (int j = 0; j < p.Nw; ++j) {
            w[j] = old_w[j] * scale;
        }
    } else {
        w = stretched_w_grid(p.w_max, p.w_stretch, p.w_center, p.Nw);
    }

    std::vector<double> f_new(f.size(), 0.0);
    conservative_energy_remap_w(old_f, f_new, r, old_w, w, L, p);
    f.swap(f_new);
    wq = quadrature_weights(w);
    min_dw = min_spacing(w);
    const double mass_after = distribution_mass(f, r, w, L, rq, wq, Lq, p, mpi);
    const double rel_mass_change = (mass_before > 0.0) ? (mass_after - mass_before) / mass_before : 0.0;
    if (is_root(mpi) && w_grid_history != nullptr) {
        (*w_grid_history) << step << ',' << t << ',' << old_w_max << ',' << p.w_max << ','
                          << active_min << ',' << active_max << ',' << min_dw << ','
                          << mass_before << ',' << mass_after << ',' << rel_mass_change << '\n';
        flush_output_checked(*w_grid_history, "w_grid_history.csv");
    }
    if (is_root(mpi)) {
        std::cout << "Adaptive w remap at step=" << step << " t=" << std::setprecision(8) << t
                  << ": w_max " << old_w_max << " -> " << p.w_max
                  << " active=[" << active_min << ',' << active_max << "]"
                  << " rel_mass_change=" << rel_mass_change << "\n";
    }
    return true;
}

// Interpolation helpers for the semi-Lagrangian departure point.
/// Returns the index of the grid cell immediately below a requested coordinate.
int lower_cell(const std::vector<double>& grid, double value) {
    auto it = std::upper_bound(grid.begin(), grid.end(), value);
    int i = static_cast<int>(std::distance(grid.begin(), it)) - 1;
    return std::clamp(i, 0, static_cast<int>(grid.size()) - 2);
}

/// Interpolates a radial metric/profile array at one physical radius.
double interpolate_profile_radial(
    const std::vector<double>& values,
    double rp,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const Params& p) {
    rp = regularize_radial_profile_sample(rp, r, p);
    const double rc = std::clamp(rp, r.front(), r.back());
    const double xp = radial_coordinate(rc, p);
    const int i = lower_cell(x_r, xp);
    const double t = (xp - x_r[i]) / (x_r[i + 1] - x_r[i]);
    return (1.0 - t) * values[i] + t * values[i + 1];
}

/// Samples all metric fields needed by the characteristic equations at one radius.
MetricSample sample_metric(
    double rp,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const Metric& metric,
    const Params& p) {
    return {
        interpolate_profile_radial(metric.mu, rp, x_r, r, p),
        interpolate_profile_radial(metric.lambda, rp, x_r, r, p),
        interpolate_profile_radial(metric.mu_r, rp, x_r, r, p),
        interpolate_profile_radial(metric.lambda_t, rp, x_r, r, p),
    };
}

/// Evaluates dr/dt for one characteristic in the frozen metric.
double radial_velocity(double rp, double wp, double Lk, const MetricSample& metric, const std::vector<double>& r) {
    const double rc = std::clamp(rp, r.front(), r.back());
    const double eps = std::sqrt(1.0 + wp * wp + (Lk / rc) * (Lk / rc));
    const double fac = std::exp(metric.mu - metric.lambda);
    return fac * wp / eps;
}

/// Evaluates dw/dt for one characteristic in the frozen metric.
// The polar-areal characteristic contains metric time variation, radial
// gravitation and the centrifugal term. All terms use the same sampled metric.
double radial_acceleration(double rp, double wp, double Lk, const MetricSample& metric, const std::vector<double>& r) {
    const double rc = std::clamp(rp, r.front(), r.back());
    const double eps = std::sqrt(1.0 + wp * wp + (Lk / rc) * (Lk / rc));
    const double fac = std::exp(metric.mu - metric.lambda);
    return -metric.lambda_t * wp - fac * metric.mu_r * eps
        + fac * Lk * Lk / (rc * rc * rc * eps);
}

/// Bilinearly interpolates one fixed-L slice of f at an off-grid (r,w) point.
// Interpolate at fixed conserved angular momentum. The four bilinear weights
// are nonnegative inside the cell; an open departure outside the domain gives zero.
double interpolate_slice(
    const std::vector<double>& f,
    double rp,
    double wp,
    int k,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const Params& p) {
    regularize_radial_footpoint_for_interpolation(rp, wp, r, p);
    if (rp < r.front() || rp > r.back() || wp < w.front() || wp > w.back()) {
        return 0.0;
    }

    const double xp = radial_coordinate(rp, p);
    const int i = lower_cell(x_r, xp);
    const int j = lower_cell(w, wp);
    const double tr = (xp - x_r[i]) / (x_r[i + 1] - x_r[i]);
    const double tw = (wp - w[j]) / (w[j + 1] - w[j]);

    const double f00 = f[idx(i, j, k, p)];
    const double f10 = f[idx(i + 1, j, k, p)];
    const double f01 = f[idx(i, j + 1, k, p)];
    const double f11 = f[idx(i + 1, j + 1, k, p)];
    return (1.0 - tr) * (1.0 - tw) * f00
        + tr * (1.0 - tw) * f10
        + (1.0 - tr) * tw * f01
        + tr * tw * f11;
}

/// Traces one EV characteristic footpoint backward by dt with the selected method.
// Use negative physical time to pull the distribution back from a destination
// node. Each supported integrator applies the same configured boundary projection.
CharacteristicState trace_ev_characteristic_footpoint(
    double ri,
    double wi,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const Metric& metric,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan) {
    if (using_kick_drift_kick_integrator(p)) {
        const MetricSample target_metric = sample_metric(ri, x_r, r, metric, p);
        const double a_target = radial_acceleration(ri, wi, Lk, target_metric, r);
        const double w_half = wi - 0.5 * dt * a_target;
        const double v_half = radial_velocity(ri, w_half, Lk, target_metric, r);
        CharacteristicState y{ri - dt * v_half, w_half};
        reflect_radial_boundaries(y.r, y.w, p.r_min, r.back(), p);
        const MetricSample dep_metric = sample_metric(y.r, x_r, r, metric, p);
        y.w -= 0.5 * dt * radial_acceleration(y.r, y.w, Lk, dep_metric, r);
        reflect_radial_boundaries(y.r, y.w, p.r_min, r.back(), p);
        return y;
    }

    auto rhs = [&](const CharacteristicState& y) {
        const MetricSample sample = sample_metric(y.r, x_r, r, metric, p);
        return CharacteristicDerivative{
            radial_velocity(y.r, y.w, Lk, sample, r),
            radial_acceleration(y.r, y.w, Lk, sample, r),
        };
    };
    auto project = [&](CharacteristicState& y) {
        reflect_radial_boundaries(y.r, y.w, p.r_min, r.back(), p);
    };

    const CharacteristicState target{ri, wi};
    if (plan.method == "runge_kutta") {
        return characteristic_rk_advance(
            target, -dt, plan, rhs, project);
    }
    if (plan.method == "extrapolated_midpoint") {
        return characteristic_extrapolated_midpoint_advance(
            target, -dt, plan, rhs, project);
    }
    if (using_adams_bashforth_characteristic_integrator(p)) {
        return characteristic_adams_bashforth_advance(
            target, -dt, plan, rhs, project);
    }
    throw std::runtime_error("Unknown semi-Lagrangian EV characteristic integrator: " + p.integrator);
}

// One timestep of Vlasov advection.  The metric is frozen over the step and the
// characteristics are traced backward with the selected characteristic method.
/// Advances f by one semi-Lagrangian step with frozen metric fields.
// Independent destinations read only the old f and frozen metric. Their
// pullbacks form a new array before finite-wall symmetry is applied.
std::vector<double> advect(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const Metric& metric,
    double dt,
    const Params& p) {
    std::vector<double> f_new(f.size(), 0.0);
    const CharacteristicIntegratorPlan integrator_plan = make_characteristic_integrator_plan(p);
    const int phase_space_work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(0, phase_space_work, effective_threads(p, phase_space_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double Lk = L[k];
                const CharacteristicState dep = trace_ev_characteristic_footpoint(
                    r[i], w[j], Lk, x_r, r, metric, dt, p, integrator_plan);
                f_new[idx(i, j, k, p)] =
                    std::max(0.0, interpolate_slice(f, dep.r, dep.w, k, x_r, r, w, p));
            }
        });
    enforce_inner_wall_symmetry(f_new, w, p);
    return f_new;
}

// Boundary-loss diagnostics are estimates from forward characteristic motion.
// They are useful warnings, not exact conservative flux integrals.
/// Estimates mass that will leave r or w boundaries during the next timestep.
// A first-order forward displacement identifies support likely to leave the
// box. Its energy-weighted integral is an escape estimate for monitoring.
EscapeStats estimate_escapes(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& rq,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Metric& metric,
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
                const double fac = std::exp(metric.mu[i] - metric.lambda[i]);
                const double fij = f[static_cast<std::size_t>(q)];
                if (fij <= threshold) {
                    continue;
                }
                const double eps = std::sqrt(1.0 + w[j] * w[j] + (L[k] / r[i]) * (L[k] / r[i]));
                const double drdt = fac * w[j] / eps;
                const double dwdt = -metric.lambda_t[i] * w[j] - fac * metric.mu_r[i] * eps
                    + fac * L[k] * L[k] / (r[i] * r[i] * r[i] * eps);
                double rp = r[i] + dt * drdt;
                double wp = w[j] + dt * dwdt;
                reflect_radial_boundaries(rp, wp, p.r_min, r.back(), p);

                const bool out_outer = !using_reflecting_outer(p) && rp > r.back();
                const bool out_w_low = wp < w.front();
                const bool out_w_high = wp > w.back();
                if (!out_outer && !out_w_low && !out_w_high) {
                    continue;
                }

                const double cell_mass = 8.0 * PI * PI * eps * L[k] * fij * rq[i] * wq[j] * Lq[k];
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
