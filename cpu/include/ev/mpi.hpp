#pragma once

// Assemble global radial data from MPI angular-momentum slabs.
// Each rank owns a contiguous part of the L or eta axis while retaining the
// radial and radial-velocity coordinates. Sum reductions combine source
// moments and counters; extrema give global timestep and guard bounds.
// Serial builds return the local value through the same calling interface.

#include "ev/common.hpp"

namespace ev {

/// Forward-declares environment-based detection of accidental multi-rank launches.
int detected_multi_rank_size();

/// Returns true on the rank responsible for user-visible output and shared files.
bool is_root(const MpiContext& mpi) {
    return mpi.rank == 0;
}

/// Synchronizes MPI ranks when MPI is enabled and is a no-op in serial builds.
void mpi_barrier(const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    (void)mpi;
#endif
}

/// Sums one double over all MPI ranks, returning the input unchanged without MPI.
double mpi_allreduce_sum(double value, const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        double out = 0.0;
        MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return out;
    }
#else
    (void)mpi;
#endif
    return value;
}

/// Computes a global maximum over all MPI ranks, or a local maximum in serial builds.
double mpi_allreduce_max(double value, const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        double out = 0.0;
        MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return out;
    }
#else
    (void)mpi;
#endif
    return value;
}

/// Computes a global minimum over all MPI ranks, or a local minimum in serial builds.
double mpi_allreduce_min(double value, const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        double out = 0.0;
        MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        return out;
    }
#else
    (void)mpi;
#endif
    return value;
}

/// Sums one integer counter over all MPI ranks, preserving serial behavior without MPI.
long long mpi_allreduce_sum_ll(long long value, const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        long long out = 0;
        MPI_Allreduce(&value, &out, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        return out;
    }
#else
    (void)mpi;
#endif
    return value;
}

/// Computes a global maximum integer counter, preserving serial behavior.
long long mpi_allreduce_max_ll(long long value, const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        long long out = 0;
        MPI_Allreduce(&value, &out, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        return out;
    }
#else
    (void)mpi;
#endif
    return value;
}

/// Replaces a vector with its rank-wise sum so all ranks share global radial data.
// All ranks must call this collective with equal vector lengths. The
// result gives each rank the complete radial source reconstructed from
// its distinct angular-momentum slab.
void mpi_allreduce_sum_vector(std::vector<double>& values, const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled && !values.empty()) {
        std::vector<double> out(values.size(), 0.0);
        MPI_Allreduce(
            values.data(),
            out.data(),
            static_cast<int>(values.size()),
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);
        values.swap(out);
    }
#else
    (void)values;
    (void)mpi;
#endif
}

/// Splits the global L grid into contiguous slabs and assigns one slab to this rank.
LDecomposition make_l_decomposition(int global_NL, const MpiContext& mpi) {
    // Distribute the remainder one cell at a time to the earliest ranks.
    // The prefix offset keeps slabs contiguous and covers each global
    // angular-momentum entry exactly once.
    const int base = global_NL / mpi.size;
    const int rem = global_NL % mpi.size;
    LDecomposition decomp;
    decomp.global_NL = global_NL;
    decomp.count = base + ((mpi.rank < rem) ? 1 : 0);
    decomp.offset = mpi.rank * base + std::min(mpi.rank, rem);
    return decomp;
}

/// Extracts the local L-slab slice of a global one-dimensional vector.
template <typename T>
// Use the same offset and count for coordinates and weights as for
// the distribution slab. This preserves its physical quadrature measure.
std::vector<T> slice_vector(const std::vector<T>& values, const LDecomposition& decomp) {
    if (decomp.count <= 0) {
        return {};
    }
    return std::vector<T>(
        values.begin() + static_cast<std::ptrdiff_t>(decomp.offset),
        values.begin() + static_cast<std::ptrdiff_t>(decomp.offset + decomp.count));
}
}  // namespace ev
