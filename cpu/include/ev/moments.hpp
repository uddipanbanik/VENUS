#pragma once

// Integrate the Eulerian spherical distribution into Einstein source moments.
// The specific energy is epsilon=sqrt(1+w^2+L^2/r^2), in rest-mass units.
// The momentum-space measure is (2pi/r^2) L dL dw. Outputs are energy
// density rho, radial pressure p, the sum of the two tangential pressures pt,
// and radial energy current j. Local L-slab contributions are assembled
// across MPI ranks after the thread-local quadratures.

#include "ev/initial_data.hpp"
#include "ev/parallel.hpp"
#include "ev/mpi.hpp"

namespace ev {

/// Integrates f over momentum space to form rho(r), p(r), and j(r) on this rank.
Moments compute_moments(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    Moments out;
    out.rho.assign(p.Nr, 0.0);
    out.p.assign(p.Nr, 0.0);
    out.pt.assign(p.Nr, 0.0);
    out.j.assign(p.Nr, 0.0);

    // Each work item fixes a radial and radial-velocity node. A worker
    // integrates its local angular-momentum slab into private radial arrays,
    // so no two workers update the same accumulator.
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
                    const double LL = L[k];
                    const double vt = LL / r[i];
                    const double eps = std::sqrt(1.0 + w[j] * w[j] + vt * vt);
                    // The L factor is the spherical momentum Jacobian. Energy density
                    // weights by epsilon, radial pressure by w^2/epsilon, tangential
                    // pressure sum by (L/r)^2/epsilon and current by w.
                    const double weight = wq[j] * Lq[k] * LL * f[idx(i, j, k, p)];
                    srho += eps * weight;
                    sp += (w[j] * w[j] / eps) * weight;
                    spt += (vt * vt / eps) * weight;
                    sj += w[j] * weight;
                }
                rho_partial[base + static_cast<std::size_t>(i)] += srho;
                p_partial[base + static_cast<std::size_t>(i)] += sp;
                pt_partial[base + static_cast<std::size_t>(i)] += spt;
                j_partial[base + static_cast<std::size_t>(i)] += sj;
            }
        });

    // Combine thread lanes in a fixed order, then apply the common
    // 2pi/r^2 geometric factor to each source moment.
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

/// Forms global radial matter moments by summing local L-slab contributions over MPI ranks.
Moments compute_global_moments(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& local_p,
    const MpiContext& mpi) {
    Moments out = compute_moments(f, r, w, L, wq, Lq, local_p);
    // Every rank holds the same radii but a different L slab. Summing the
    // radial source vectors gives each rank the complete matter field needed
    // for the same Einstein constraint solve.
    mpi_allreduce_sum_vector(out.rho, mpi);
    mpi_allreduce_sum_vector(out.p, mpi);
    mpi_allreduce_sum_vector(out.pt, mpi);
    mpi_allreduce_sum_vector(out.j, mpi);
    return out;
}

}  // namespace ev
