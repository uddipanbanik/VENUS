#pragma once

// Choose the Eulerian Einstein--Vlasov characteristic timestep in a frozen
// metric. The supplied activity thresholds identify the represented phase-
// space support for this scan. The code bounds crossings in the selected
// radial interpolation coordinate and in radial momentum w, then combines
// both CFL bounds with dt_max. It changes no distribution or metric state.

#include "ev/metric.hpp"

namespace ev {

/// Selects the next timestep from active-cell characteristic speeds and CFL limits.
double choose_dt(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const Metric& metric,
    double radial_step,
    double min_dw,
    const Params& p,
    const MpiContext& mpi) {
    const double max_f = mpi_allreduce_max(max_value_parallel(f, p), mpi);
    // Use the larger absolute or relative distribution threshold. This
    // configured numerical support criterion belongs to the Eulerian scan;
    // it differs from a per-marker mass criterion in a particle solver.
    const double threshold = std::max(p.active_floor_abs, p.active_floor_rel * max_f);
    if (max_f <= 0.0) {
        return p.dt_max;
    }

    const int work = static_cast<int>(f.size());
    const int nthreads = effective_threads(p, work);
    std::vector<double> local_vr(static_cast<std::size_t>(nthreads), 0.0);
    std::vector<double> local_vw(static_cast<std::size_t>(nthreads), 0.0);
    std::vector<long long> local_active(static_cast<std::size_t>(nthreads), 0);
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            double max_vr_local = 0.0;
            double max_vw_local = 0.0;
            long long active_local = 0;
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
                ++active_local;
                const double eps = std::sqrt(1.0 + w[j] * w[j] + (L[k] / r[i]) * (L[k] / r[i]));
                // The characteristic radial speed is exp(mu-lambda)*w/epsilon.
                // Momentum changes through the time-dependent metric, lapse gradient
                // and angular-momentum barrier.
                const double vr = fac * w[j] / eps;
                const double vw = -metric.lambda_t[i] * w[j] - fac * metric.mu_r[i] * eps
                    + fac * L[k] * L[k] / (r[i] * r[i] * r[i] * eps);
                // A log-radius interpolation cell has speed d(log r)/dt=(dr/dt)/r.
                // Its crossing time must use log spacing rather than physical dr.
                const double vr_coord = using_log_radial_coordinate(p) ? vr / r[i] : vr;
                max_vr_local = std::max(max_vr_local, std::abs(vr_coord));
                max_vw_local = std::max(max_vw_local, std::abs(vw));
            }
            local_vr[static_cast<std::size_t>(tid)] = max_vr_local;
            local_vw[static_cast<std::size_t>(tid)] = max_vw_local;
            local_active[static_cast<std::size_t>(tid)] = active_local;
        });

    double max_vr_coord = 0.0;
    double max_vw = 0.0;
    long long active_count = 0;
    for (int q = 0; q < nthreads; ++q) {
        max_vr_coord = std::max(max_vr_coord, local_vr[static_cast<std::size_t>(q)]);
        max_vw = std::max(max_vw, local_vw[static_cast<std::size_t>(q)]);
        active_count += local_active[static_cast<std::size_t>(q)];
    }
    // Take the fastest active characteristic over every thread and MPI
    // slab. All ranks then advance with the same global interval.
    max_vr_coord = mpi_allreduce_max(max_vr_coord, mpi);
    max_vw = mpi_allreduce_max(max_vw, mpi);
    active_count = mpi_allreduce_sum_ll(active_count, mpi);

    if (active_count == 0) {
        return p.dt_max;
    }
    max_vr_coord = std::max(max_vr_coord, 1.0e-14);
    max_vw = std::max(max_vw, 1.0e-14);
    // The radial and momentum coordinates each impose a crossing-time
    // bound. Their minimum and the configured maximum cap determine dt.
    return std::min({p.dt_max, p.cfl * radial_step / max_vr_coord, p.cfl * min_dw / max_vw});
}

}  // namespace ev
