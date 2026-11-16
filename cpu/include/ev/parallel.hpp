#pragma once

// Divide CPU work among OpenMP threads or the std::thread backend.
// Inputs are an index range, a requested worker count and a function that
// writes disjoint output or thread-local accumulators. All workers finish
// before this interface returns or rethrows their first stored error.
// Physics modules own the reduction order and numerical interpretation.

#include "ev/common.hpp"

#include <exception>

namespace ev {

/// Reports which thread backend was compiled: OpenMP when available, otherwise std::thread.
std::string parallel_backend_name() {
#ifdef _OPENMP
    return "openmp";
#else
    return "std_thread";
#endif
}

/// Returns the maximum useful thread count advertised by the active backend.
int max_backend_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    return static_cast<int>(hw);
#endif
}

/// Chooses a safe number of threads for a loop with the requested amount of work.
int effective_threads(const Params& p, int work_items) {
    const int requested_threads = (p.threads > 0) ? p.threads : max_backend_threads();
    if (work_items <= 1 || requested_threads <= 1) {
        return 1;
    }
    return std::max(1, std::min(requested_threads, work_items));
}

/// Runs a one-dimensional loop range in contiguous chunks over the available threads.
template <typename Func>
// Stable contiguous chunks assign each work item to one worker. The
// last std::thread chunk runs in the caller, avoiding one extra thread.
// Each worker stores an exception for propagation after all workers join.
void parallel_for_chunks(int begin, int end, int requested_threads, Func fn) {
    const int work_items = end - begin;
    if (work_items <= 0) {
        return;
    }
    const int nthreads = std::max(1, std::min(requested_threads, work_items));
    if (nthreads == 1) {
        fn(begin, end, 0);
        return;
    }

#ifdef _OPENMP
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(nthreads));
    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int actual_threads = omp_get_num_threads();
        const int lo = begin + (work_items * tid) / actual_threads;
        const int hi = begin + (work_items * (tid + 1)) / actual_threads;
        try {
            fn(lo, hi, tid);
        } catch (...) {
            errors[static_cast<std::size_t>(tid)] = std::current_exception();
        }
    }
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
#else
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(nthreads - 1));
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(nthreads));
    auto invoke = [&](int lo, int hi, int tid) {
        try {
            fn(lo, hi, tid);
        } catch (...) {
            errors[static_cast<std::size_t>(tid)] = std::current_exception();
        }
    };
    for (int tid = 0; tid < nthreads; ++tid) {
        const int lo = begin + (work_items * tid) / nthreads;
        const int hi = begin + (work_items * (tid + 1)) / nthreads;
        if (tid + 1 == nthreads) {
            invoke(lo, hi, tid);
        } else {
            workers.emplace_back(invoke, lo, hi, tid);
        }
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
#endif
}

/// Runs a loop with dynamic chunk assignment for uneven workloads.
template <typename Func>
// Dynamic assignment balances work whose cost varies by index. A
// shared next-index counter gives each chunk to exactly one worker;
// the failed flag prevents starting further chunks after an exception.
void parallel_for_dynamic_chunks(int begin, int end, int requested_threads, int chunk_size, Func fn) {
    const int work_items = end - begin;
    if (work_items <= 0) {
        return;
    }
    const int nthreads = std::max(1, std::min(requested_threads, work_items));
    const int chunk = std::max(1, chunk_size);
    if (nthreads == 1) {
        fn(begin, end, 0);
        return;
    }

#ifdef _OPENMP
    std::atomic<bool> failed(false);
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(nthreads));
    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, chunk)
        for (int lo = begin; lo < end; lo += chunk) {
            if (failed.load(std::memory_order_relaxed)) {
                continue;
            }
            const int hi = std::min(end, lo + chunk);
            try {
                fn(lo, hi, tid);
            } catch (...) {
                errors[static_cast<std::size_t>(tid)] =
                    std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        }
    }
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
#else
    std::atomic<int> next(begin);
    std::atomic<bool> failed(false);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(nthreads - 1));
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(nthreads));
    auto worker = [&](int tid) {
        while (!failed.load(std::memory_order_relaxed)) {
            const int lo = next.fetch_add(chunk);
            if (lo >= end) {
                break;
            }
            try {
                fn(lo, std::min(end, lo + chunk), tid);
            } catch (...) {
                errors[static_cast<std::size_t>(tid)] =
                    std::current_exception();
                failed.store(true, std::memory_order_relaxed);
                break;
            }
        }
    };
    for (int tid = 0; tid < nthreads; ++tid) {
        if (tid + 1 == nthreads) {
            worker(tid);
        } else {
            workers.emplace_back(worker, tid);
        }
    }
    for (std::thread& w : workers) {
        w.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
#endif
}

/// Computes the maximum entry of a vector using the configured thread backend.
// Store one maximum per worker and reduce those values after all
// workers finish. This is used by support thresholds and timestep scans.
double max_value_parallel(const std::vector<double>& values, const Params& p) {
    if (values.empty()) {
        return -std::numeric_limits<double>::infinity();
    }
    const int nthreads = effective_threads(p, static_cast<int>(values.size()));
    if (nthreads == 1) {
        return *std::max_element(values.begin(), values.end());
    }
    std::vector<double> local(static_cast<std::size_t>(nthreads), -std::numeric_limits<double>::infinity());
    parallel_for_chunks(0, static_cast<int>(values.size()), nthreads,
        [&](int lo, int hi, int tid) {
            double m = -std::numeric_limits<double>::infinity();
            for (int q = lo; q < hi; ++q) {
                m = std::max(m, values[static_cast<std::size_t>(q)]);
            }
            local[static_cast<std::size_t>(tid)] = m;
        });
    return *std::max_element(local.begin(), local.end());
}
}  // namespace ev
