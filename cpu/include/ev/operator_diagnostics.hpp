#pragma once

// Measure the increments produced by Vlasov transport and collisions.
// Inputs are before/after distribution arrays, the accepted interval and the
// phase-space quadrature. Outputs are weighted RMS increments per unit time,
// globally and in the declared radial bands, plus their ratios.
// These diagnostic reductions describe the applied operators and do not add
// a source term to the physical evolution.

#include <array>

#include "ev/grids.hpp"
#include "ev/mpi.hpp"

namespace ev {

/// Adds after-before to an accumulated operator increment.
void accumulate_operator_delta(
    std::vector<double>& total_delta,
    const std::vector<double>& before,
    const std::vector<double>& after,
    const Params& p) {
    if (total_delta.size() != before.size() || before.size() != after.size()) {
        throw std::runtime_error("Operator diagnostic delta arrays have inconsistent sizes.");
    }
    const int work = static_cast<int>(total_delta.size());
    parallel_for_chunks(0, work, effective_threads(p, work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                total_delta[static_cast<std::size_t>(q)] +=
                    after[static_cast<std::size_t>(q)] - before[static_cast<std::size_t>(q)];
            }
        });
}

/// Returns after-before as one operator increment.
std::vector<double> operator_delta(
    const std::vector<double>& before,
    const std::vector<double>& after,
    const Params& p) {
    std::vector<double> delta(before.size(), 0.0);
    accumulate_operator_delta(delta, before, after, p);
    return delta;
}

inline double safe_operator_ratio(double numerator, double denominator) {
    return (denominator > 0.0 && std::isfinite(denominator))
        ? numerator / denominator
        : 0.0;
}

inline int operator_radius_band(double r) {
    if (r < 1.0) {
        return 1;
    }
    if (r < 3.0) {
        return 2;
    }
    return 3;
}

/// Computes weighted RMS norms of the Vlasov and collision increments divided by dt.
OperatorDiagnostics compute_operator_diagnostics(
    const std::vector<double>& vlasov_delta,
    const std::vector<double>& collision_delta,
    double dt,
    const std::vector<double>& r,
    const std::vector<double>& wq,
    const std::vector<double>& L,
    const std::vector<double>& rq,
    const std::vector<double>& Lq,
    const Params& p,
    const MpiContext& mpi) {
    if (vlasov_delta.size() != collision_delta.size()) {
        throw std::runtime_error("Operator diagnostic arrays have inconsistent sizes.");
    }
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        return {};
    }

    double weight_local[4] = {0.0, 0.0, 0.0, 0.0};
    double vlasov_sq_local[4] = {0.0, 0.0, 0.0, 0.0};
    double collision_sq_local[4] = {0.0, 0.0, 0.0, 0.0};

    const int work = p.Nr * p.Nw * p.NL;
    const int nthreads = effective_threads(p, work);
    std::vector<std::array<double, 12>> partial(static_cast<std::size_t>(nthreads));

    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            double weight[4] = {0.0, 0.0, 0.0, 0.0};
            double vlasov_sq[4] = {0.0, 0.0, 0.0, 0.0};
            double collision_sq[4] = {0.0, 0.0, 0.0, 0.0};
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                // Use the same r,w,L quadrature and spherical L Jacobian to weight
                // each operator increment. The constant geometric prefactor cancels
                // between the squared norm and its weight normalization.
                const double cell_weight = rq[static_cast<std::size_t>(i)]
                    * wq[static_cast<std::size_t>(j)]
                    * Lq[static_cast<std::size_t>(k)]
                    * std::max(L[static_cast<std::size_t>(k)], 0.0);
                if (!(cell_weight > 0.0) || !std::isfinite(cell_weight)) {
                    continue;
                }
                // Divide the complete split-step increment by the accepted interval to
                // obtain an average operator magnitude, rather than interpreting a
                // raw change in f as a rate.
                const double vlasov = vlasov_delta[static_cast<std::size_t>(q)] / dt;
                const double collision = collision_delta[static_cast<std::size_t>(q)] / dt;
                if (!std::isfinite(vlasov) || !std::isfinite(collision)) {
                    continue;
                }
                const int band = operator_radius_band(r[static_cast<std::size_t>(i)]);
                weight[0] += cell_weight;
                vlasov_sq[0] += cell_weight * vlasov * vlasov;
                collision_sq[0] += cell_weight * collision * collision;
                weight[band] += cell_weight;
                vlasov_sq[band] += cell_weight * vlasov * vlasov;
                collision_sq[band] += cell_weight * collision * collision;
            }
            std::array<double, 12>& out = partial[static_cast<std::size_t>(tid)];
            for (int b = 0; b < 4; ++b) {
                out[static_cast<std::size_t>(b)] = weight[b];
                out[static_cast<std::size_t>(4 + b)] = vlasov_sq[b];
                out[static_cast<std::size_t>(8 + b)] = collision_sq[b];
            }
        });

    for (const std::array<double, 12>& part : partial) {
        for (int b = 0; b < 4; ++b) {
            weight_local[b] += part[static_cast<std::size_t>(b)];
            vlasov_sq_local[b] += part[static_cast<std::size_t>(4 + b)];
            collision_sq_local[b] += part[static_cast<std::size_t>(8 + b)];
        }
    }

    double weight[4] = {0.0, 0.0, 0.0, 0.0};
    double vlasov_sq[4] = {0.0, 0.0, 0.0, 0.0};
    double collision_sq[4] = {0.0, 0.0, 0.0, 0.0};
    // Reduce both the total quadrature weight and squared increments over
    // all MPI slabs before forming a global RMS value.
    for (int b = 0; b < 4; ++b) {
        weight[b] = mpi_allreduce_sum(weight_local[b], mpi);
        vlasov_sq[b] = mpi_allreduce_sum(vlasov_sq_local[b], mpi);
        collision_sq[b] = mpi_allreduce_sum(collision_sq_local[b], mpi);
    }

    double vlasov_rms[4] = {0.0, 0.0, 0.0, 0.0};
    double collision_rms[4] = {0.0, 0.0, 0.0, 0.0};
    for (int b = 0; b < 4; ++b) {
        if (weight[b] > 0.0) {
            vlasov_rms[b] = std::sqrt(std::max(vlasov_sq[b] / weight[b], 0.0));
            collision_rms[b] = std::sqrt(std::max(collision_sq[b] / weight[b], 0.0));
        }
    }

    // Store the full-domain and fixed radial-band results separately.
    // The collision/Vlasov ratio uses the protected denominator convention
    // above when the measured Vlasov RMS vanishes.
    OperatorDiagnostics diag;
    diag.vlasov_rms = vlasov_rms[0];
    diag.collision_rms = collision_rms[0];
    diag.collision_over_vlasov = safe_operator_ratio(collision_rms[0], vlasov_rms[0]);
    diag.vlasov_rms_r_lt_1 = vlasov_rms[1];
    diag.collision_rms_r_lt_1 = collision_rms[1];
    diag.collision_over_vlasov_r_lt_1 = safe_operator_ratio(collision_rms[1], vlasov_rms[1]);
    diag.vlasov_rms_r_1_3 = vlasov_rms[2];
    diag.collision_rms_r_1_3 = collision_rms[2];
    diag.collision_over_vlasov_r_1_3 = safe_operator_ratio(collision_rms[2], vlasov_rms[2]);
    diag.vlasov_rms_r_ge_3 = vlasov_rms[3];
    diag.collision_rms_r_ge_3 = collision_rms[3];
    diag.collision_over_vlasov_r_ge_3 = safe_operator_ratio(collision_rms[3], vlasov_rms[3]);
    return diag;
}

/// Appends operator diagnostic columns to history.csv.
void write_operator_diagnostics_csv(std::ostream& out, const OperatorDiagnostics& diag) {
    out << ',' << diag.vlasov_rms
        << ',' << diag.collision_rms
        << ',' << diag.collision_over_vlasov
        << ',' << diag.vlasov_rms_r_lt_1
        << ',' << diag.collision_rms_r_lt_1
        << ',' << diag.collision_over_vlasov_r_lt_1
        << ',' << diag.vlasov_rms_r_1_3
        << ',' << diag.collision_rms_r_1_3
        << ',' << diag.collision_over_vlasov_r_1_3
        << ',' << diag.vlasov_rms_r_ge_3
        << ',' << diag.collision_rms_r_ge_3
        << ',' << diag.collision_over_vlasov_r_ge_3;
}

}  // namespace ev
