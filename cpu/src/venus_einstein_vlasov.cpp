// Evolve a spherically symmetric, collisionless distribution in general
// relativity. The input is f(r,w,L), where r is areal radius, w is radial
// momentum measured by a local static observer, and L is specific angular
// momentum. Units are G=c=1 and the particle rest mass is one.
//
// Each time step follows the same physical chain. Integrate f over momentum
// to obtain energy density, radial pressure and radial momentum density.
// Solve the radial Einstein constraints for the metric. Trace the particle
// trajectories backward through that metric, then interpolate the old f at
// their departure points to obtain the new distribution. Write radial
// profiles, global histories, phase-space samples and complete restart states.
//
// Source organization:
//   1. Runtime/state records and serial/MPI parallel helpers.
//   2. Grid construction, quadrature, and binary/analytic initial data.
//   3. Einstein-Vlasov moments, metric constraints, and timestep selection.
//   4. Characteristic integration, semi-Lagrangian pullback, and boundaries.
//   5. Profiles, phase-space output, metadata, and versioned restarts.
//   6. Configuration parsing followed by the complete evolution loop in main.
//
// This file implements the standalone grid solver. The other CPU task
// executables call the corresponding shared numerical headers in include/ev.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef EV_USE_MPI
#include <mpi.h>
#endif

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

template <std::size_t N>
bool fixed_binary_identifier_equals(
    const char (&field)[N],
    const char* expected) {
    const std::size_t expected_size = std::char_traits<char>::length(expected);
    return expected_size < N
        && std::equal(expected, expected + expected_size, field)
        && field[expected_size] == '\0';
}

// Runtime parameters.  The names intentionally match the long command-line
// options and the keys accepted by the config-file parser below.
struct Params {
    // Phase-space resolution.
    int Nr = 260;
    int Nw = 240;
    int NL = 48;

    // Radial grid and momentum-domain controls.
    double r_min = 1.0e-3;
    double r_max = 60.0;
    std::string radial_grid = "log";
    double inner_refine_radius = 0.1;
    int inner_refine_cells = 96;
    double w_max = 8.0;
    double L_max = 6.0;
    double w_stretch = 2.5;
    double w_center = 0.0;
    bool adaptive_w = false;
    int adaptive_w_every = 20;
    double adaptive_w_trigger = 0.65;
    double adaptive_w_margin = 1.8;
    double adaptive_w_growth = 1.6;
    double adaptive_w_max = 0.0;

    // Time stepping and output cadence.
    double t_end = 20.0;
    double cfl = 0.35;
    double dt_max = 0.04;
    int max_steps = 200000;
    int save_every = 20;
    int phase_every = 0;
    int phase_r_stride = 1;
    int phase_w_stride = 1;
    int restart_every = 0;
    std::string integrator = "semi_lagrangian_kick_drift_kick";
    std::string inner_boundary = "reflecting";
    std::string outer_boundary = "outflow_vacuum";
    std::string momentum_boundary = "outflow_vacuum";

    double central_mass = 0.0;

    // Initial-data source.  "analytic" keeps the built-in shell/boundary-seed
    // families.  "binary_rwl" reads a user supplied f(r,w,L) file whose header
    // carries the r, w, and L grids.
    std::string initial_data = "analytic";
    std::string initial_data_file = "initial_data/example_shell_rwl.bin";
    bool custom_ic_normalize_mass = false;
    double custom_ic_target_mass = 1.0e-2;

    // Initial data.  The shell options use a Maxwell-Juttner-like momentum
    // seed evaluated at R0 so the radial profile is not accidentally baked in.
    bool shell_initial_df = false;
    double R0 = 5.0;
    double dR = 1.0;
    bool top_hat_shell = false;
    bool top_hat_density_shell = false;
    bool gaussian_density_shell = false;
    bool double_gaussian_density_shell = false;
    double beta = 4.0;
    double amp = 1.0e-2;
    double w_in = -0.6;
    double R0_2 = 10.0;
    double dR_2 = 1.0;
    double amp_2 = 1.0e-2;
    double w_in_2 = -1.0;
    int initial_metric_iters = 4;

    double metric_floor = 1.0e-10;
    double gap_stop = 2.0e-5;
    double active_floor_rel = 1.0e-12;
    double active_floor_abs = 0.0;
    double f_floor = 0.0;

    // Parallelism and file layout.  threads=0 means derive from EV_THREADS,
    // OMP_NUM_THREADS, SLURM_CPUS_PER_TASK, or the local hardware.
    int threads = 0;
    bool allow_multi_rank_launch = false;

    std::string output_dir = "output/venus_einstein_vlasov";
    std::string restart_from;
};

struct InitialDataRWL {
    int Nr = 0;
    int Nw = 0;
    int NL = 0;
    std::vector<double> r;
    std::vector<double> w;
    std::vector<double> L;
    std::vector<double> f;
};

struct Moments {
    std::vector<double> rho;
    std::vector<double> p;
    std::vector<double> j;
};

struct Metric {
    std::vector<double> m;
    std::vector<double> lambda;
    std::vector<double> mu;
    std::vector<double> lambda_r;
    std::vector<double> mu_r;
    std::vector<double> lambda_t;
    std::vector<double> gap;
    double min_gap = 1.0;
};

struct EscapeStats {
    double outer_mass = 0.0;
    double w_low_mass = 0.0;
    double w_high_mass = 0.0;
    double total_mass = 0.0;
    long long outer_cells = 0;
    long long w_low_cells = 0;
    long long w_high_cells = 0;
};

struct MetricSample {
    double mu = 0.0;
    double lambda = 0.0;
    double mu_r = 0.0;
    double lambda_t = 0.0;
};

struct MpiContext {
    bool enabled = false;
    int rank = 0;
    int size = 1;
};

struct LDecomposition {
    int global_NL = 0;
    int offset = 0;
    int count = 0;
};

int detected_multi_rank_size();

bool is_root(const MpiContext& mpi) {
    return mpi.rank == 0;
}

void flush_output_checked(std::ostream& stream, const std::string& label) {
    stream.flush();
    if (!stream) {
        throw std::runtime_error("Failed to flush output stream: " + label);
    }
}

void mpi_barrier(const MpiContext& mpi) {
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    (void)mpi;
#endif
}

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

// Divide only the L coordinate between MPI ranks.  Each rank retains the
// entire (r,w) mesh, so a characteristic at fixed L can be followed locally.
// The remainder gives one extra L point to each of the first few ranks.
LDecomposition make_l_decomposition(int global_NL, const MpiContext& mpi) {
    const int base = global_NL / mpi.size;
    const int rem = global_NL % mpi.size;
    LDecomposition decomp;
    decomp.global_NL = global_NL;
    decomp.count = base + ((mpi.rank < rem) ? 1 : 0);
    decomp.offset = mpi.rank * base + std::min(mpi.rank, rem);
    return decomp;
}

template <typename T>
std::vector<T> slice_vector(const std::vector<T>& values, const LDecomposition& decomp) {
    if (decomp.count <= 0) {
        return {};
    }
    return std::vector<T>(
        values.begin() + static_cast<std::ptrdiff_t>(decomp.offset),
        values.begin() + static_cast<std::ptrdiff_t>(decomp.offset + decomp.count));
}

std::string parallel_backend_name() {
#ifdef _OPENMP
    return "openmp";
#else
    return "std_thread";
#endif
}

int max_backend_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    return static_cast<int>(hw);
#endif
}

int effective_threads(const Params& p, int work_items) {
    const int requested_threads = (p.threads > 0) ? p.threads : max_backend_threads();
    if (work_items <= 1 || requested_threads <= 1) {
        return 1;
    }
    return std::max(1, std::min(requested_threads, work_items));
}

// Give each worker a disjoint contiguous interval.  Numerical routines write
// their own interval or accumulate into a worker-local sum, then combine the
// sums after every worker has finished.
template <typename Func>
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
    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int actual_threads = omp_get_num_threads();
        const int lo = begin + (work_items * tid) / actual_threads;
        const int hi = begin + (work_items * (tid + 1)) / actual_threads;
        fn(lo, hi, tid);
    }
#else
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(nthreads - 1));
    for (int tid = 0; tid < nthreads; ++tid) {
        const int lo = begin + (work_items * tid) / nthreads;
        const int hi = begin + (work_items * (tid + 1)) / nthreads;
        if (tid + 1 == nthreads) {
            fn(lo, hi, tid);
        } else {
            workers.emplace_back([=, &fn]() { fn(lo, hi, tid); });
        }
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
#endif
}

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

// Store neighboring L values contiguously, followed by w and then r.  The
// same ordering is used in binary distribution files and MPI slab reads.
std::size_t idx(int i, int j, int k, const Params& p) {
    return (static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)) * p.NL
        + static_cast<std::size_t>(k);
}

// ---- Grid coordinates, cell measures, and momentum quadrature --------------
std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> x(n);
    const double dx = (b - a) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = a + dx * static_cast<double>(i);
    }
    return x;
}

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

bool using_binary_initial_data(const Params& p) {
    return p.initial_data == "binary_rwl" || p.initial_data == "binary" || p.initial_data == "file";
}

// Grid construction.  Log grids resolve small r strongly but tighten the CFL
// condition; piecewise-linear grids give a controlled inner refinement without
// making the innermost spacing quite so punishing.
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
    const double outer_dr = (p.r_max - split) / static_cast<double>(outer_cells);
    for (int q = 1; q <= outer_cells; ++q) {
        r[inner_cells + q] = split + outer_dr * static_cast<double>(q);
    }
    return r;
}

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
        throw std::runtime_error("A custom radial grid must be loaded from initial_data_file.");
    }
    throw std::runtime_error("radial_grid must be 'log', 'linear', 'piecewise-linear', or 'custom'.");
}

double radial_coordinate(double rp, const Params& p) {
    return (p.radial_grid == "log") ? std::log(rp) : rp;
}

std::vector<double> radial_coordinate_grid(const std::vector<double>& r, const Params& p) {
    std::vector<double> x(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) {
        x[i] = radial_coordinate(r[i], p);
    }
    return x;
}

// Map a uniformly spaced auxiliary coordinate through sinh.  This clusters
// momentum points around the requested center while retaining both endpoints.
std::vector<double> stretched_w_grid(double w_max, double stretch, double center, int n) {
    if (stretch <= 0.0) {
        return linspace(-w_max, w_max, n);
    }
    std::vector<double> w(n);
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

// Trapezoidal weights equal half the distance between adjacent neighbors.
// At a domain endpoint only the adjacent half interval contributes.  Using
// these widths lets the same moment sum work on a nonuniform momentum grid.
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

// Place internal control-volume faces halfway between sample points.  These
// faces define the overlap lengths used when the momentum grid is expanded.
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

double min_spacing(const std::vector<double>& x) {
    double out = std::numeric_limits<double>::infinity();
    for (int i = 1; i < static_cast<int>(x.size()); ++i) {
        out = std::min(out, x[i] - x[i - 1]);
    }
    return out;
}

template <typename T>
T read_ic_pod(std::ifstream& in, const std::filesystem::path& path) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected end of initial-data file: " + path.string());
    }
    return value;
}

void read_ic_doubles(
    std::ifstream& in,
    std::vector<double>& values,
    const std::filesystem::path& path,
    const std::string& label) {
    in.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Unexpected end of " + label + " array in " + path.string());
    }
}

bool strictly_increasing(const std::vector<double>& values) {
    if (values.empty() || !std::isfinite(values.front())) {
        return false;
    }
    for (int i = 1; i < static_cast<int>(values.size()); ++i) {
        if (!std::isfinite(values[i]) || !(values[i] > values[i - 1])) {
            return false;
        }
    }
    return true;
}

[[maybe_unused]] InitialDataRWL read_initial_data_rwl(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open binary initial-data file: " + path.string());
    }

    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVICRWL1")) {
        throw std::runtime_error(
            "Initial-data file has wrong magic header. Expected EVICRWL1: " + path.string());
    }
    const std::int32_t version = read_ic_pod<std::int32_t>(in, path);
    if (version != 1) {
        throw std::runtime_error("Unsupported initial-data file version in " + path.string());
    }
    const std::int32_t nr = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nw = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nl = read_ic_pod<std::int32_t>(in, path);
    if (nr < 3 || nw < 3 || nl < 2) {
        throw std::runtime_error("Initial-data file needs Nr >= 3, Nw >= 3, and NL >= 2.");
    }

    InitialDataRWL ic;
    ic.Nr = nr;
    ic.Nw = nw;
    ic.NL = nl;
    ic.r.assign(static_cast<std::size_t>(nr), 0.0);
    ic.w.assign(static_cast<std::size_t>(nw), 0.0);
    ic.L.assign(static_cast<std::size_t>(nl), 0.0);
    ic.f.assign(
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nw) * static_cast<std::size_t>(nl),
        0.0);
    read_ic_doubles(in, ic.r, path, "r");
    read_ic_doubles(in, ic.w, path, "w");
    read_ic_doubles(in, ic.L, path, "L");
    read_ic_doubles(in, ic.f, path, "f");

    if (!strictly_increasing(ic.r) || ic.r.front() <= 0.0) {
        throw std::runtime_error("Initial-data r grid must be positive and strictly increasing.");
    }
    if (!strictly_increasing(ic.w)) {
        throw std::runtime_error("Initial-data w grid must be strictly increasing.");
    }
    if (!strictly_increasing(ic.L) || ic.L.front() < 0.0) {
        throw std::runtime_error("Initial-data L grid must be nonnegative and strictly increasing.");
    }
    for (double value : ic.f) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("Initial-data f values must be finite and nonnegative.");
        }
    }
    return ic;
}

// ---- Binary and analytic initial distributions -----------------------------
InitialDataRWL read_initial_data_rwl_metadata(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open binary initial-data file: " + path.string());
    }

    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVICRWL1")) {
        throw std::runtime_error(
            "Initial-data file has wrong magic header. Expected EVICRWL1: " + path.string());
    }
    const std::int32_t version = read_ic_pod<std::int32_t>(in, path);
    if (version != 1) {
        throw std::runtime_error("Unsupported initial-data file version in " + path.string());
    }
    const std::int32_t nr = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nw = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nl = read_ic_pod<std::int32_t>(in, path);
    if (nr < 3 || nw < 3 || nl < 2) {
        throw std::runtime_error("Initial-data file needs Nr >= 3, Nw >= 3, and NL >= 2.");
    }

    InitialDataRWL ic;
    ic.Nr = nr;
    ic.Nw = nw;
    ic.NL = nl;
    ic.r.assign(static_cast<std::size_t>(nr), 0.0);
    ic.w.assign(static_cast<std::size_t>(nw), 0.0);
    ic.L.assign(static_cast<std::size_t>(nl), 0.0);
    read_ic_doubles(in, ic.r, path, "r");
    read_ic_doubles(in, ic.w, path, "w");
    read_ic_doubles(in, ic.L, path, "L");

    if (!strictly_increasing(ic.r) || ic.r.front() <= 0.0) {
        throw std::runtime_error("Initial-data r grid must be positive and strictly increasing.");
    }
    if (!strictly_increasing(ic.w)) {
        throw std::runtime_error("Initial-data w grid must be strictly increasing.");
    }
    if (!strictly_increasing(ic.L) || ic.L.front() < 0.0) {
        throw std::runtime_error("Initial-data L grid must be nonnegative and strictly increasing.");
    }
    return ic;
}

std::streamoff initial_data_f_offset(const InitialDataRWL& ic) {
    return static_cast<std::streamoff>(16 + 4 * sizeof(std::int32_t))
        + static_cast<std::streamoff>(ic.Nr + ic.Nw + ic.NL) * static_cast<std::streamoff>(sizeof(double));
}

// Seek directly to the contiguous L interval owned by this rank at each
// (r,w) location.  Reading only that slab avoids replicating the full 3-D
// distribution on every MPI rank.
std::vector<double> read_initial_data_rwl_local_values(
    const std::filesystem::path& path,
    const InitialDataRWL& ic,
    const LDecomposition& decomp) {
    std::vector<double> f(
        static_cast<std::size_t>(ic.Nr) * ic.Nw * decomp.count,
        0.0);
    if (decomp.count == 0) {
        return f;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open binary initial-data file: " + path.string());
    }
    const std::streamoff f0 = initial_data_f_offset(ic);
    for (int i = 0; i < ic.Nr; ++i) {
        for (int j = 0; j < ic.Nw; ++j) {
            const std::uint64_t global_base =
                (static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(ic.Nw)
                 + static_cast<std::uint64_t>(j))
                    * static_cast<std::uint64_t>(ic.NL)
                + static_cast<std::uint64_t>(decomp.offset);
            const std::streamoff offset =
                f0 + static_cast<std::streamoff>(global_base * static_cast<std::uint64_t>(sizeof(double)));
            in.seekg(offset);
            if (!in) {
                throw std::runtime_error("Failed to seek local initial-data slab in " + path.string());
            }
            double* dest = f.data()
                + (static_cast<std::size_t>(i) * ic.Nw + static_cast<std::size_t>(j))
                    * static_cast<std::size_t>(decomp.count);
            in.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(decomp.count * sizeof(double)));
            if (!in) {
                throw std::runtime_error("Unexpected end of local initial-data slab in " + path.string());
            }
            for (int k = 0; k < decomp.count; ++k) {
                const double value = dest[k];
                if (!std::isfinite(value) || value < 0.0) {
                    throw std::runtime_error("Initial-data f values must be finite and nonnegative.");
                }
            }
        }
    }
    return f;
}

std::vector<double> custom_initial_distribution_rwl(
    const InitialDataRWL& ic,
    const Params& p) {
    std::vector<double> f(static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    std::copy(ic.f.begin(), ic.f.end(), f.begin());
    return f;
}

// Parameter validation catches the most common unphysical or internally
// inconsistent runs before a large phase-space array is allocated.
void validate_params(const Params& p) {
    const bool binary_ic = using_binary_initial_data(p);
    if (!(p.initial_data == "analytic" || binary_ic)) {
        throw std::runtime_error("initial_data must be 'analytic' or 'binary_rwl'.");
    }
    if (binary_ic && p.initial_data_file.empty()) {
        throw std::runtime_error("initial_data_file is required when initial_data=binary_rwl.");
    }
    if (p.custom_ic_normalize_mass && p.custom_ic_target_mass <= 0.0) {
        throw std::runtime_error("custom_ic_target_mass must be positive when normalization is enabled.");
    }
    if (p.Nr < 3 || p.Nw < 3 || p.NL < 2) {
        throw std::runtime_error("Need Nr >= 3, Nw >= 3, and NL >= 2.");
    }
    if (p.r_min <= 0.0 || p.r_max <= p.r_min) {
        throw std::runtime_error("Require 0 < r_min < r_max.");
    }
    radial_grid_code(p.radial_grid);
    if (p.radial_grid == "piecewise-linear") {
        if (!(p.r_min < p.inner_refine_radius && p.inner_refine_radius < p.r_max)) {
            throw std::runtime_error(
                "Require r_min < inner_refine_radius < r_max for piecewise-linear grid.");
        }
        if (p.inner_refine_cells < 1 || p.inner_refine_cells > p.Nr - 2) {
            throw std::runtime_error(
                "Require 1 <= inner_refine_cells <= Nr - 2 for piecewise-linear grid.");
        }
    }
    if (p.w_max <= 0.0 || p.L_max <= 0.0) {
        throw std::runtime_error("w_max and L_max must be positive.");
    }
    if (p.w_stretch < 0.0) {
        throw std::runtime_error("w_stretch must be non-negative.");
    }
    if (!(p.w_center > -p.w_max && p.w_center < p.w_max)) {
        throw std::runtime_error("Require -w_max < w_center < w_max.");
    }
    if (std::abs(p.w_center) > 1.0e-12) {
        std::cerr
            << "Warning: the inner reflecting wall mirrors w about 0, but "
            << "--w-center is nonzero. This can under-resolve the reflected "
            << "outgoing branch. Prefer --w-center 0 for reflecting-center runs.\n";
    }
    if (p.adaptive_w) {
        if (binary_ic) {
            throw std::runtime_error(
                "adaptive_w=true is disabled for binary_rwl initial data in this release: "
                "the restart format stores w_max but not the full remapped w grid.");
        }
        if (p.adaptive_w_every < 1) {
            throw std::runtime_error("adaptive_w_every must be positive.");
        }
        if (!(p.adaptive_w_trigger > 0.0 && p.adaptive_w_trigger < 1.0)) {
            throw std::runtime_error("adaptive_w_trigger must lie between 0 and 1.");
        }
        if (p.adaptive_w_margin <= 1.0 || p.adaptive_w_growth <= 1.0) {
            throw std::runtime_error("adaptive_w_margin and adaptive_w_growth must be > 1.");
        }
        if (p.adaptive_w_max > 0.0 && p.adaptive_w_max <= p.w_max) {
            throw std::runtime_error("adaptive_w_max must exceed the initial w_max when set.");
        }
    }
    if (p.central_mass > 0.0 && p.r_min <= 2.05 * p.central_mass) {
        throw std::runtime_error(
            "For a nonzero central mass use r_min safely outside 2M, or leave "
            "central_mass=0 for the reflecting regular-center run.");
    }
    if (binary_ic && p.radial_grid != "custom") {
        throw std::runtime_error("Binary initial data must set radial_grid=custom after loading.");
    }
    if (!binary_ic && p.radial_grid == "custom") {
        throw std::runtime_error("radial_grid=custom is only available with initial_data=binary_rwl.");
    }
    if (!binary_ic && (p.shell_initial_df || p.double_gaussian_density_shell) && p.R0 <= 0.0) {
        throw std::runtime_error("Require R0 > 0 for shell initial data.");
    }
    if (!binary_ic && (p.shell_initial_df || p.double_gaussian_density_shell) && p.dR <= 0.0) {
        throw std::runtime_error("Require dR > 0 for shell initial data.");
    }
    if (!binary_ic && (p.shell_initial_df || p.double_gaussian_density_shell)
        && (p.R0 < p.r_min || p.R0 > p.r_max)) {
        throw std::runtime_error("Shell center R0 must lie inside the radial grid.");
    }
    const int shell_mode_count = (p.top_hat_shell ? 1 : 0)
        + (p.top_hat_density_shell ? 1 : 0)
        + (p.gaussian_density_shell ? 1 : 0)
        + (p.double_gaussian_density_shell ? 1 : 0);
    if (!binary_ic && shell_mode_count > 1) {
        throw std::runtime_error(
            "Choose only one shell radial mode: --top-hat-shell, "
            "--top-hat-density-shell, --gaussian-density-shell, "
            "or --double-gaussian-density-shell.");
    }
    if (!binary_ic && p.double_gaussian_density_shell) {
        if (p.R0_2 <= 0.0 || p.dR_2 <= 0.0) {
            throw std::runtime_error("Require R0_2 > 0 and dR_2 > 0 for double shell data.");
        }
        if (p.R0_2 < p.r_min || p.R0_2 > p.r_max) {
            throw std::runtime_error("Second shell center R0_2 must lie inside the radial grid.");
        }
        if (p.amp_2 < 0.0) {
            throw std::runtime_error("Require amp_2 >= 0 for double shell data.");
        }
        if (p.w_in_2 <= -p.w_max || p.w_in_2 >= p.w_max) {
            throw std::runtime_error("The second shell velocity center w_in_2 must lie inside the w grid.");
        }
    }
    if (!binary_ic && (p.top_hat_shell || p.top_hat_density_shell)
        && (p.R0 - 0.5 * p.dR < p.r_min || p.R0 + 0.5 * p.dR > p.r_max)) {
        std::cerr
            << "Warning: top-hat shell support extends outside the radial grid; "
            << "the initial shell will be truncated.\n";
    }
    if (!binary_ic && p.beta <= 0.0) {
        throw std::runtime_error("Require beta > 0 for the boundary-seed initial DF.");
    }
    if (!binary_ic && p.amp < 0.0) {
        throw std::runtime_error("Require amp >= 0 for the initial DF.");
    }
    if (!binary_ic && (p.w_in <= -p.w_max || p.w_in >= p.w_max)) {
        throw std::runtime_error("The initial DF velocity center w_in must lie inside the w grid.");
    }
    if (p.t_end < 0.0 || p.dt_max <= 0.0 || p.cfl <= 0.0) {
        throw std::runtime_error("Require t_end >= 0, dt_max > 0, and cfl > 0.");
    }
    if (p.integrator != "semi_lagrangian_kick_drift_kick") {
        throw std::runtime_error("Only integrator=semi_lagrangian_kick_drift_kick is implemented.");
    }
    if (p.inner_boundary != "reflecting") {
        throw std::runtime_error("Only inner_boundary=reflecting is implemented.");
    }
    if (p.outer_boundary != "outflow_vacuum") {
        throw std::runtime_error("Only outer_boundary=outflow_vacuum is implemented.");
    }
    if (p.momentum_boundary != "outflow_vacuum") {
        throw std::runtime_error("Only momentum_boundary=outflow_vacuum is implemented.");
    }
    if (p.threads < 1) {
        throw std::runtime_error("Require threads >= 1.");
    }
    if (p.max_steps < 0) {
        throw std::runtime_error("Require max_steps >= 0.");
    }
    if (p.metric_floor <= 0.0 || p.gap_stop <= 0.0) {
        throw std::runtime_error("Require metric_floor > 0 and gap_stop > 0.");
    }
    if (p.active_floor_rel < 0.0 || p.active_floor_abs < 0.0 || p.f_floor < 0.0) {
        throw std::runtime_error("Active-cell floors must be non-negative.");
    }
}

// Initial radial shell envelope.  Density-weighted shell modes later multiply
// this by a moment-integral correction so rho(r,t=0) follows the requested
// envelope instead of inheriting a spurious spherical-measure power law.
double shell_profile(double r, const Params& p) {
    if (p.top_hat_shell || p.top_hat_density_shell) {
        return (r >= p.R0 - 0.5 * p.dR && r <= p.R0 + 0.5 * p.dR) ? 1.0 : 0.0;
    }
    const double x = (r - p.R0) / p.dR;
    return std::exp(-0.5 * x * x);
}

double interpolate_radial_grid_value(
    const std::vector<double>& values,
    double rp,
    const std::vector<double>& r,
    const std::vector<double>& x_r,
    const Params& p) {
    const double rc = std::clamp(rp, r.front(), r.back());
    const double x = radial_coordinate(rc, p);
    auto it = std::upper_bound(x_r.begin(), x_r.end(), x);
    int i = static_cast<int>(std::distance(x_r.begin(), it)) - 1;
    i = std::clamp(i, 0, static_cast<int>(r.size()) - 2);
    const double x0 = x_r[i];
    const double x1 = x_r[i + 1];
    const double q = (x - x0) / (x1 - x0);
    return (1.0 - q) * values[i] + q * values[i + 1];
}

// Construct the prescribed momentum distribution and radial envelope.
// Density-shell modes correct for the r-dependent energy integral below,
// since multiplying f by a Gaussian alone need not give a Gaussian rho.
std::vector<double> initial_distribution(
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const std::vector<double>& mu_seed,
    const std::vector<double>& x_r,
    const Params& p) {
    std::vector<double> f(static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    if (p.double_gaussian_density_shell) {
        auto add_density_gaussian_shell = [&](double rb, double width, double amp, double w_center_shell) {
            const double eVb = std::exp(interpolate_radial_grid_value(mu_seed, rb, r, x_r, p));
            const double b = std::max(p.beta * eVb, 1.0e-300);
            const double norm = amp * std::sqrt(b / (2.0 * PI)) * b / (rb * rb);
            std::vector<double> base(static_cast<std::size_t>(p.Nw) * p.NL, 0.0);
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    const double ell_over_rb = L[k] / rb;
                    const double eps_shift = std::sqrt(
                        1.0 + (w[j] - w_center_shell) * (w[j] - w_center_shell)
                        + ell_over_rb * ell_over_rb);
                    base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)]
                        = norm * std::exp(-b * (eps_shift - 1.0));
                }
            }

            auto moment_integral = [&](double radius) {
                double out = 0.0;
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const double eps = std::sqrt(
                            1.0 + w[j] * w[j] + (L[k] / radius) * (L[k] / radius));
                        out += wq[j] * Lq[k] * L[k]
                            * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)]
                            * eps;
                    }
                }
                return out;
            };

            // rho carries a 1/r^2 factor and the momentum integral I(r).
            // Multiplication by r^2/rb^2 * I(rb)/I(r) cancels both variations
            // and leaves the requested Gaussian energy-density envelope.
            const double I_rb = moment_integral(rb);
            for (int i = 0; i < p.Nr; ++i) {
                const double x = (r[i] - rb) / width;
                double shell = std::exp(-0.5 * x * x);
                if (shell <= 0.0) {
                    continue;
                }
                const double I_r = std::max(moment_integral(r[i]), 1.0e-300);
                shell *= (r[i] * r[i] / (rb * rb)) * (I_rb / I_r);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        f[idx(i, j, k, p)] += shell
                            * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)];
                    }
                }
            }
        };

        add_density_gaussian_shell(p.R0, p.dR, p.amp, p.w_in);
        add_density_gaussian_shell(p.R0_2, p.dR_2, p.amp_2, p.w_in_2);
        return f;
    }

    if (p.shell_initial_df) {
        const double rb = p.R0;
        const double eVb = std::exp(interpolate_radial_grid_value(mu_seed, rb, r, x_r, p));
        const double b = std::max(p.beta * eVb, 1.0e-300);
        const double norm = p.amp * std::sqrt(b / (2.0 * PI)) * b / (rb * rb);
        std::vector<double> base(static_cast<std::size_t>(p.Nw) * p.NL, 0.0);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const double ell_over_rb = L[k] / rb;
                const double eps_shift = std::sqrt(
                    1.0 + (w[j] - p.w_in) * (w[j] - p.w_in) + ell_over_rb * ell_over_rb);
                base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)]
                    = norm * std::exp(-b * (eps_shift - 1.0));
            }
        }

        auto moment_integral = [&](double radius) {
            double out = 0.0;
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    const double eps = std::sqrt(1.0 + w[j] * w[j] + (L[k] / radius) * (L[k] / radius));
                    out += wq[j] * Lq[k] * L[k]
                        * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)] * eps;
                }
            }
            return out;
        };
        // For a density shell, apply the same moment correction to the
        // chosen top-hat or Gaussian envelope.  The other shell mode applies
        // its envelope directly to f.
        const bool density_weighted_shell = p.top_hat_density_shell || p.gaussian_density_shell;
        const double I_rb = density_weighted_shell ? moment_integral(rb) : 1.0;

        for (int i = 0; i < p.Nr; ++i) {
            double shell = shell_profile(r[i], p);
            if (shell > 0.0 && density_weighted_shell) {
                const double I_r = std::max(moment_integral(r[i]), 1.0e-300);
                shell *= (r[i] * r[i] / (rb * rb)) * (I_rb / I_r);
            }
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    f[idx(i, j, k, p)] = shell
                        * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)];
                }
            }
        }
        return f;
    }

    for (int i = 0; i < p.Nr; ++i) {
        const double eV = std::exp(mu_seed[i]);
        const double b = std::max(p.beta * eV, 1.0e-300);
        const double norm = p.amp * std::sqrt(b / (2.0 * PI)) * b / (r[i] * r[i]);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const double ell_over_r = L[k] / r[i];
                const double eps_shift = std::sqrt(
                    1.0 + (w[j] - p.w_in) * (w[j] - p.w_in) + ell_over_r * ell_over_r);
                f[idx(i, j, k, p)] = norm * std::exp(-b * (eps_shift - 1.0));
            }
        }
    }
    return f;
}

// Integrate over the local momentum coordinates at each radius. Spherical
// symmetry removes the tangential azimuth, leaving the measure
// (2*pi/r^2)*L*dL*dw. Multiplying it by epsilon, w^2/epsilon or w gives,
// respectively, energy density rho, radial pressure p or radial current j.
// The supplied wq and Lq arrays contain the one-dimensional quadrature weights.
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
    out.j.assign(p.Nr, 0.0);

    parallel_for_chunks(0, p.Nr, effective_threads(p, p.Nr),
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                double srho = 0.0;
                double sp = 0.0;
                double sj = 0.0;
                // One thread owns this whole radial sum, so the momentum
                // quadrature needs no concurrent updates to shared moments.
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const double LL = L[k];
                        // L/r is tangential momentum. The mass-shell relation
                        // gives epsilon=sqrt(1+w^2+(L/r)^2) for each grid point.
                        const double eps = std::sqrt(1.0 + w[j] * w[j] + (LL / r[i]) * (LL / r[i]));
                        const double weight = wq[j] * Lq[k] * LL * f[idx(i, j, k, p)];
                        srho += eps * weight;
                        sp += (w[j] * w[j] / eps) * weight;
                        sj += w[j] * weight;
                    }
                }
                // Apply the angular Jacobian after completing the sums.
                const double factor = 2.0 * PI / (r[i] * r[i]);
                out.rho[i] = factor * srho;
                out.p[i] = factor * sp;
                out.j[i] = factor * sj;
            }
        });
    return out;
}

// MPI ranks hold different L intervals of the same distribution. Add their
// partial moments before solving gravity, so every rank uses the full source.
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
    mpi_allreduce_sum_vector(out.rho, mpi);
    mpi_allreduce_sum_vector(out.p, mpi);
    mpi_allreduce_sum_vector(out.j, mpi);
    return out;
}

// Reconstruct the polar-areal metric from the three Vlasov moments.
// First integrate m_r=4*pi*r^2*rho outward. Then obtain lambda from
// exp(-2*lambda)=1-2*m/r, integrate the mu constraint inward from the outer
// Schwarzschild vacuum, and obtain lambda_t from the radial current.
Metric solve_metric(
    const Moments& moments,
    const std::vector<double>& r,
    const Params& p) {
    Metric metric;
    // The first radius is positive. With a regular center, approximate the
    // unresolved interior by a uniform sphere at the first density value.
    // An explicitly supplied central mass instead supplies the inner mass.
    const double inner_matter_mass = (p.central_mass > 0.0)
        ? 0.0
        : (4.0 * PI / 3.0) * moments.rho.front() * r.front() * r.front() * r.front();
    metric.m.assign(p.Nr, p.central_mass + inner_matter_mass);
    metric.lambda.assign(p.Nr, 0.0);
    metric.mu.assign(p.Nr, 0.0);
    metric.lambda_r.assign(p.Nr, 0.0);
    metric.mu_r.assign(p.Nr, 0.0);
    metric.lambda_t.assign(p.Nr, 0.0);
    metric.gap.assign(p.Nr, 1.0);

    // Trapezoidal quadrature adds the mass between adjacent radial points.
    for (int i = 1; i < p.Nr; ++i) {
        const double y0 = r[i - 1] * r[i - 1] * moments.rho[i - 1];
        const double y1 = r[i] * r[i] * moments.rho[i];
        metric.m[i] = metric.m[i - 1] + 4.0 * PI * 0.5 * (y0 + y1) * (r[i] - r[i - 1]);
    }

    metric.min_gap = 1.0e300;
    for (int i = 0; i < p.Nr; ++i) {
        // Keep the actual compactness gap for the stopping diagnostic.
        // The floor below only makes the metric evaluation well defined
        // while the caller decides whether the coordinate limit was reached.
        const double gap_raw = 1.0 - 2.0 * metric.m[i] / r[i];
        metric.gap[i] = gap_raw;
        metric.min_gap = std::min(metric.min_gap, gap_raw);
        const double gap = std::max(gap_raw, p.metric_floor);
        metric.lambda[i] = -0.5 * std::log(gap);
        metric.lambda_r[i] = (4.0 * PI * r[i] * moments.rho[i] - metric.m[i] / (r[i] * r[i])) / gap;
        metric.mu_r[i] = (metric.m[i] / (r[i] * r[i]) + 4.0 * PI * r[i] * moments.p[i]) / gap;
    }

    // The vacuum exterior fixes the time normalization. Integrating inward
    // from this boundary includes the gravitational redshift of the matter.
    const double outer_gap = std::max(1.0 - 2.0 * metric.m.back() / r.back(), p.metric_floor);
    metric.mu.back() = 0.5 * std::log(outer_gap);
    for (int i = p.Nr - 2; i >= 0; --i) {
        const double dr = r[i + 1] - r[i];
        metric.mu[i] = metric.mu[i + 1] - 0.5 * (metric.mu_r[i + 1] + metric.mu_r[i]) * dr;
    }

    // The Einstein current equation supplies the time derivative needed in
    // the geodesic radial-momentum equation.
    for (int i = 0; i < p.Nr; ++i) {
        metric.lambda_t[i] = -4.0 * PI * r[i] * std::exp(metric.lambda[i] + metric.mu[i]) * moments.j[i];
    }
    return metric;
}

// Resolve motion across both the radial grid and the radial-momentum grid.
// Evaluate the characteristic speeds where f exceeds the configured activity
// floor, take their maxima across threads and MPI ranks, and apply the CFL
// fraction to the corresponding cell-crossing times. This grid solver uses
// its own activity criterion; the particle solvers have separate selectors.
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
    const double threshold = std::max(p.active_floor_abs, p.active_floor_rel * max_f);
    if (max_f <= 0.0) {
        return p.dt_max;
    }

    const int nthreads = effective_threads(p, p.Nr);
    std::vector<double> local_vr(static_cast<std::size_t>(nthreads), 0.0);
    std::vector<double> local_vw(static_cast<std::size_t>(nthreads), 0.0);
    std::vector<long long> local_active(static_cast<std::size_t>(nthreads), 0);
    parallel_for_chunks(0, p.Nr, nthreads,
        [&](int ilo, int ihi, int tid) {
            double max_vr_local = 0.0;
            double max_vw_local = 0.0;
            long long active_local = 0;
            for (int i = ilo; i < ihi; ++i) {
                const double fac = std::exp(metric.mu[i] - metric.lambda[i]);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const double fij = f[idx(i, j, k, p)];
                        if (fij <= threshold) {
                            continue;
                        }
                        ++active_local;
                        const double eps = std::sqrt(1.0 + w[j] * w[j] + (L[k] / r[i]) * (L[k] / r[i]));
                        const double vr = fac * w[j] / eps;
                        const double vw = -metric.lambda_t[i] * w[j] - fac * metric.mu_r[i] * eps
                            + fac * L[k] * L[k] / (r[i] * r[i] * r[i] * eps);
                        // A logarithmic radial grid advances x=log(r), whose
                        // speed is dx/dt=(dr/dt)/r. Use that speed with its
                        // logarithmic spacing rather than a physical dr.
                        const double vr_coord = (p.radial_grid == "log") ? vr / r[i] : vr;
                        max_vr_local = std::max(max_vr_local, std::abs(vr_coord));
                        max_vw_local = std::max(max_vw_local, std::abs(vw));
                    }
                }
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
    // Every rank must accept the same interval, including a rank that owns
    // only slowly moving angular-momentum cells.
    max_vr_coord = mpi_allreduce_max(max_vr_coord, mpi);
    max_vw = mpi_allreduce_max(max_vw, mpi);
    active_count = mpi_allreduce_sum_ll(active_count, mpi);

    if (active_count == 0) {
        return p.dt_max;
    }
    max_vr_coord = std::max(max_vr_coord, 1.0e-14);
    max_vw = std::max(max_vw, 1.0e-14);
    return std::min({p.dt_max, p.cfl * radial_step / max_vr_coord, p.cfl * min_dw / max_vw});
}

// Reflect a characteristic foot that crosses the inner wall.  This implements
// the regular-center mirror condition in the finite-r_min calculation: r is
// reflected about r_min and radial momentum changes sign.
void reflect_inner(double& rp, double& wp, double r_min) {
    if (rp >= r_min) {
        return;
    }
    rp = r_min + (r_min - rp);
    wp = -wp;
}

int lower_cell(const std::vector<double>& grid, double value);

double interpolate_w_value(const std::vector<double>& values, double wp, const std::vector<double>& w) {
    if (wp < w.front() || wp > w.back()) {
        return 0.0;
    }
    const int j = lower_cell(w, wp);
    const double tw = (wp - w[j]) / (w[j + 1] - w[j]);
    return (1.0 - tw) * values[j] + tw * values[j + 1];
}

// Enforce f(r_min,w,L)=f(r_min,-w,L) by averaging each boundary value with
// its reflected partner.  Interpolation supplies the partner when the w mesh
// has no sample exactly at -w.
void enforce_inner_wall_symmetry(std::vector<double>& f, const std::vector<double>& w, const Params& p) {
    std::vector<double> boundary(p.Nw);
    for (int k = 0; k < p.NL; ++k) {
        for (int j = 0; j < p.Nw; ++j) {
            boundary[j] = f[idx(0, j, k, p)];
        }
        for (int j = 0; j < p.Nw; ++j) {
            const double mirrored = interpolate_w_value(boundary, -w[j], w);
            f[idx(0, j, k, p)] = 0.5 * (boundary[j] + mirrored);
        }
    }
}

// Integrate the energy density over areal volume.  The r^2 factors in
// 4*pi*r^2 dr and the spherical momentum measure cancel, leaving
// 8*pi^2 * epsilon * L * f dr dw dL.  This is the matter energy-mass integral
// used by this solver, with no extra proper-volume factor exp(lambda).
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
    const int nthreads = effective_threads(p, p.Nr);
    std::vector<double> local(static_cast<std::size_t>(nthreads), 0.0);
    parallel_for_chunks(0, p.Nr, nthreads,
        [&](int ilo, int ihi, int tid) {
            double sum = 0.0;
            for (int i = ilo; i < ihi; ++i) {
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const double eps = std::sqrt(
                            1.0 + w[j] * w[j] + (L[k] / r[i]) * (L[k] / r[i]));
                        sum += 8.0 * PI * PI * eps * L[k] * f[idx(i, j, k, p)]
                            * rq[i] * wq[j] * Lq[k];
                    }
                }
            }
            local[static_cast<std::size_t>(tid)] = sum;
        });
    double total = 0.0;
    for (double x : local) {
        total += x;
    }
    return mpi_allreduce_sum(total, mpi);
}

double particle_energy(double r, double w, double L) {
    return std::sqrt(1.0 + w * w + (L / r) * (L / r));
}

// Treat epsilon*f as constant inside each old momentum control volume.
// Accumulate its overlap with each new volume, then divide by the new width
// and epsilon.  The overlap sum preserves this discrete energy integral at
// each (r,L), subject to the common old/new domain.
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
                            const double eps_old = particle_energy(ri, old_w[static_cast<std::size_t>(jo)], Lk);
                            cell_energy += (right - left) * eps_old * old_f[idx(i, jo, k, p)];
                        }

                        if (cell_energy <= 0.0) {
                            continue;
                        }
                        const double eps_new = particle_energy(ri, new_w[static_cast<std::size_t>(j)], Lk);
                        f_new[idx(i, j, k, p)] = std::max(0.0, cell_energy / (width * eps_new));
                    }
                }
            }
        });
}

// Expand the momentum domain when the active distribution approaches its
// endpoints.  The new domain contains the old one.  The overlap remap
// preserves the cell-integrated epsilon*f in each (r,L) slice, and the
// diagnostic below measures its global energy-mass residual.
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
int lower_cell(const std::vector<double>& grid, double value) {
    auto it = std::upper_bound(grid.begin(), grid.end(), value);
    int i = static_cast<int>(std::distance(grid.begin(), it)) - 1;
    return std::clamp(i, 0, static_cast<int>(grid.size()) - 2);
}

double interpolate_profile_radial(
    const std::vector<double>& values,
    double rp,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const Params& p) {
    const double rc = std::clamp(rp, r.front(), r.back());
    const double xp = radial_coordinate(rc, p);
    const int i = lower_cell(x_r, xp);
    const double t = (xp - x_r[i]) / (x_r[i + 1] - x_r[i]);
    return (1.0 - t) * values[i] + t * values[i + 1];
}

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

// Convert the observer's radial speed w/epsilon to the coordinate speed
// dr/dt=exp(mu-lambda)*w/epsilon using the lapse and radial metric factor.
double radial_velocity(double rp, double wp, double Lk, const MetricSample& metric, const std::vector<double>& r) {
    const double rc = std::clamp(rp, r.front(), r.back());
    const double eps = std::sqrt(1.0 + wp * wp + (Lk / rc) * (Lk / rc));
    const double fac = std::exp(metric.mu - metric.lambda);
    return fac * wp / eps;
}

// The three contributions to dw/dt are the time-dependent radial metric,
// gravitational redshift gradient and centrifugal acceleration, in that order.
// They use the same local energy and metric interpolation as dr/dt.
double radial_acceleration(double rp, double wp, double Lk, const MetricSample& metric, const std::vector<double>& r) {
    const double rc = std::clamp(rp, r.front(), r.back());
    const double eps = std::sqrt(1.0 + wp * wp + (Lk / rc) * (Lk / rc));
    const double fac = std::exp(metric.mu - metric.lambda);
    return -metric.lambda_t * wp - fac * metric.mu_r * eps
        + fac * Lk * Lk / (rc * rc * rc * eps);
}

// Recover f at a characteristic's departure point by bilinear interpolation
// in the chosen radial coordinate and w, at fixed L. The four interpolation
// weights sum to one. A departure beyond the stored domain has vacuum f=0.
double interpolate_slice(
    const std::vector<double>& f,
    double rp,
    double wp,
    int k,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const Params& p) {
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

// One timestep of Vlasov advection.  The metric is frozen over the step and the
// characteristics are traced backward with a kick-drift-kick update.
// ---- Backward characteristic tracing and semi-Lagrangian pullback ----------
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
    const int phase_space_work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(0, phase_space_work, effective_threads(p, phase_space_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double Lk = L[k];
                const MetricSample target_metric{
                    metric.mu[i],
                    metric.lambda[i],
                    metric.mu_r[i],
                    metric.lambda_t[i],
                };
                // Start at the new-time grid point and reverse the two kicks
                // and intervening drift to locate its old-time trajectory.
                const double a_target = radial_acceleration(r[i], w[j], Lk, target_metric, r);
                const double w_half = w[j] - 0.5 * dt * a_target;
                const double v_half = radial_velocity(r[i], w_half, Lk, target_metric, r);
                double rp = r[i] - dt * v_half;
                double w_mid = w_half;
                reflect_inner(rp, w_mid, p.r_min);
                // The second kick acts at the departure radius, so gather
                // its metric there rather than reusing the arrival value.
                const MetricSample dep_metric = sample_metric(rp, x_r, r, metric, p);
                double wp = w_mid - 0.5 * dt * radial_acceleration(rp, w_mid, Lk, dep_metric, r);
                reflect_inner(rp, wp, p.r_min);
                // Vlasov transport carries f along the characteristic.
                // Interpolate its old value and preserve nonnegative f.
                f_new[idx(i, j, k, p)] = std::max(0.0, interpolate_slice(f, rp, wp, k, x_r, r, w, p));
            }
        });
    enforce_inner_wall_symmetry(f_new, w, p);
    return f_new;
}

// Estimate the material that would cross a radial or momentum boundary in
// one forward Euler trajectory step. These estimates indicate when the
// chosen domain needs attention. The transport itself is performed above.
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
    const int nthreads = effective_threads(p, p.Nr);
    std::vector<EscapeStats> local(static_cast<std::size_t>(nthreads));
    parallel_for_chunks(0, p.Nr, nthreads,
        [&](int ilo, int ihi, int tid) {
            EscapeStats stats_local;
            for (int i = ilo; i < ihi; ++i) {
                const double fac = std::exp(metric.mu[i] - metric.lambda[i]);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const double fij = f[idx(i, j, k, p)];
                        if (fij <= threshold) {
                            continue;
                        }
                        const double eps = std::sqrt(1.0 + w[j] * w[j] + (L[k] / r[i]) * (L[k] / r[i]));
                        const double drdt = fac * w[j] / eps;
                        const double dwdt = -metric.lambda_t[i] * w[j] - fac * metric.mu_r[i] * eps
                            + fac * L[k] * L[k] / (r[i] * r[i] * r[i] * eps);
                        double rp = r[i] + dt * drdt;
                        double wp = w[j] + dt * dwdt;
                        reflect_inner(rp, wp, p.r_min);

                        const bool out_outer = rp > r.back();
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

// ---- Profiles, phase output, metadata, and restart persistence -------------
// Integrate over L with its spherical Jacobian: g(r,w)=integral L*f dL.
// This projected distribution is useful for phase-space images.  The energy
// density additionally needs the epsilon moment and the factor 2*pi/r^2.
std::vector<double> phase_projection(
    const std::vector<double>& f,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& p) {
    std::vector<double> g(static_cast<std::size_t>(p.Nr) * p.Nw, 0.0);
    parallel_for_chunks(0, p.Nr, effective_threads(p, p.Nr),
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                for (int j = 0; j < p.Nw; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < p.NL; ++k) {
                        sum += L[k] * f[idx(i, j, k, p)] * Lq[k];
                    }
                    g[static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)] = sum;
                }
            }
        });
    return g;
}

std::vector<double> phase_projection_global(
    const std::vector<double>& f,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& local_p,
    const MpiContext& mpi) {
    std::vector<double> g = phase_projection(f, L, Lq, local_p);
    mpi_allreduce_sum_vector(g, mpi);
    return g;
}

void write_phase_space(
    const std::filesystem::path& path,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& g,
    const Params& p) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open phase-space output file: " + path.string());
    }
    out << std::setprecision(17);
    out << "r,w,g\n";
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            out << r[i] << ',' << w[j] << ','
                << g[static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)] << '\n';
        }
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to close phase-space output file after writing: " + path.string());
    }
}

void write_phase_snapshot_rows(
    std::ofstream& out,
    double t,
    int step,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& g,
    const Params& p) {
    out << std::setprecision(17);
    for (int i = 0; i < p.Nr; i += p.phase_r_stride) {
        for (int j = 0; j < p.Nw; j += p.phase_w_stride) {
            out << t << ',' << step << ',' << i << ',' << j << ','
                << r[i] << ',' << w[j] << ','
                << g[static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)] << '\n';
        }
    }
}

void write_profile_rows(
    std::ofstream& out,
    double t,
    int step,
    const std::vector<double>& r,
    const Moments& moments,
    const Metric& metric,
    const Params& p) {
    out << std::setprecision(17);
    for (int i = 0; i < p.Nr; ++i) {
        out << t << ',' << step << ',' << r[i] << ','
            << moments.rho[i] << ',' << moments.p[i] << ',' << moments.j[i] << ','
            << metric.m[i] << ',' << metric.mu[i] << ',' << metric.lambda[i] << ','
            << metric.gap[i] << '\n';
    }
}

template <typename T>
void write_binary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("Failed while writing restart file.");
    }
}

template <typename T>
T read_binary(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected end of restart file.");
    }
    return value;
}

bool close_enough(double a, double b, double rel = 1.0e-12) {
    return std::isfinite(a) && std::isfinite(b) && std::isfinite(rel)
        && rel >= 0.0
        && std::abs(a - b)
            <= rel * std::max({1.0, std::abs(a), std::abs(b)});
}

void validate_restart_double(const char* name, double got, double expected) {
    if (!close_enough(got, expected)) {
        throw std::runtime_error(std::string("Restart grid mismatch for ") + name);
    }
}

void validate_restart_clock(double time, std::int64_t step) {
    if (!(time >= 0.0) || !std::isfinite(time)) {
        throw std::runtime_error("Restart time must be finite and nonnegative.");
    }
    if (step < 0
        || step > static_cast<std::int64_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("Restart step is outside int range.");
    }
}

void validate_restart_distribution(
    const std::vector<double>& values,
    const Params& p) {
    const std::size_t expected =
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    if (values.size() != expected) {
        throw std::runtime_error("Restart state array has wrong size.");
    }
    for (double value : values) {
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(
                "Restart state must be finite and nonnegative.");
        }
    }
}

// Write a complete temporary file, check the stream, then rename it to the
// restart path.  Readers therefore see either the previous complete state or
// the newly completed state.  Grid parameters, physical time and f travel
// together so a restart can verify their compatibility.
void write_restart_state(
    const std::filesystem::path& path,
    const Params& p,
    double t,
    int step,
    const std::vector<double>& f) {
    validate_restart_clock(t, step);
    validate_restart_distribution(f, p);
    const std::filesystem::path tmp = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open restart file for writing: " + tmp.string());
    }

    const char magic[16] = {'E', 'V', 'L', 'O', 'G', 'R', 'S', 'T', '1', '\0'};
    out.write(magic, sizeof(magic));
    const std::int32_t version = 3;
    write_binary(out, version);
    write_binary(out, static_cast<std::int32_t>(p.Nr));
    write_binary(out, static_cast<std::int32_t>(p.Nw));
    write_binary(out, static_cast<std::int32_t>(p.NL));
    write_binary(out, static_cast<std::int32_t>(radial_grid_code(p.radial_grid)));
    write_binary(out, p.r_min);
    write_binary(out, p.r_max);
    write_binary(out, p.inner_refine_radius);
    write_binary(out, static_cast<std::int32_t>(p.inner_refine_cells));
    write_binary(out, p.w_max);
    write_binary(out, p.L_max);
    write_binary(out, p.w_stretch);
    write_binary(out, p.w_center);
    write_binary(out, p.central_mass);
    write_binary(out, t);
    write_binary(out, static_cast<std::int64_t>(step));
    write_binary(out, static_cast<std::uint64_t>(f.size()));
    out.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error("Failed while writing restart state array.");
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to close restart file after writing: " + tmp.string());
    }
    std::filesystem::rename(tmp, path);
}

[[maybe_unused]] std::vector<double> read_restart_state(
    const std::filesystem::path& path,
    const Params& p,
    double& t,
    int& step) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open restart file: " + path.string());
    }

    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVLOGRST1")) {
        throw std::runtime_error("Restart file has wrong magic header.");
    }
    const std::int32_t version = read_binary<std::int32_t>(in);
    if (version != 1 && version != 2 && version != 3) {
        throw std::runtime_error("Unsupported restart file version.");
    }
    const std::int32_t nr = read_binary<std::int32_t>(in);
    const std::int32_t nw = read_binary<std::int32_t>(in);
    const std::int32_t nl = read_binary<std::int32_t>(in);
    if (nr != p.Nr || nw != p.Nw || nl != p.NL) {
        throw std::runtime_error("Restart file dimensions do not match requested grid.");
    }
    const std::int32_t saved_radial_grid_code = (version >= 2) ? read_binary<std::int32_t>(in) : 0;
    const std::string saved_radial_grid = radial_grid_name(saved_radial_grid_code);
    if (saved_radial_grid != p.radial_grid) {
        throw std::runtime_error("Restart radial grid does not match requested radial grid.");
    }
    validate_restart_double("r_min", read_binary<double>(in), p.r_min);
    validate_restart_double("r_max", read_binary<double>(in), p.r_max);
    if (version >= 3) {
        validate_restart_double("inner_refine_radius", read_binary<double>(in), p.inner_refine_radius);
        const std::int32_t saved_inner_cells = read_binary<std::int32_t>(in);
        if (saved_inner_cells != p.inner_refine_cells) {
            throw std::runtime_error("Restart grid mismatch for inner_refine_cells");
        }
    }
    validate_restart_double("w_max", read_binary<double>(in), p.w_max);
    validate_restart_double("L_max", read_binary<double>(in), p.L_max);
    validate_restart_double("w_stretch", read_binary<double>(in), p.w_stretch);
    validate_restart_double("w_center", read_binary<double>(in), p.w_center);
    validate_restart_double("central_mass", read_binary<double>(in), p.central_mass);
    t = read_binary<double>(in);
    const std::int64_t saved_step = read_binary<std::int64_t>(in);
    validate_restart_clock(t, saved_step);
    step = static_cast<int>(saved_step);
    const std::uint64_t n_values = read_binary<std::uint64_t>(in);
    const std::uint64_t expected_values = static_cast<std::uint64_t>(p.Nr) * p.Nw * p.NL;
    if (n_values != expected_values) {
        throw std::runtime_error("Restart state array has wrong size.");
    }
    std::vector<double> f(static_cast<std::size_t>(n_values));
    in.read(reinterpret_cast<char*>(f.data()), static_cast<std::streamsize>(f.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Unexpected end of restart state array.");
    }
    validate_restart_distribution(f, p);
    return f;
}

std::filesystem::path rank_restart_path(const std::filesystem::path& manifest_path, int rank) {
    std::ostringstream suffix;
    suffix << ".rank" << std::setw(6) << std::setfill('0') << rank;
    return std::filesystem::path(manifest_path.string() + suffix.str());
}

// Each rank publishes its own complete L slab.  After all slabs are ready,
// rank zero publishes the common manifest.  The saved rank/slab descriptors
// let the reader reject an incompatible MPI decomposition.
void write_restart_state_distributed(
    const std::filesystem::path& path,
    const Params& global_p,
    const Params& local_p,
    const LDecomposition& decomp,
    const MpiContext& mpi,
    double t,
    int step,
    const std::vector<double>& f) {
    validate_restart_clock(t, step);
    validate_restart_distribution(f, local_p);
    if (mpi.size == 1) {
        write_restart_state(path, global_p, t, step, f);
        return;
    }

    const std::filesystem::path rank_path = rank_restart_path(path, mpi.rank);
    const std::filesystem::path rank_tmp = rank_path.string() + ".tmp";
    {
        std::ofstream out(rank_tmp, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open rank restart file for writing: " + rank_tmp.string());
        }
        const char rank_magic[16] = {'E', 'V', 'M', 'P', 'I', 'R', 'N', 'K', '1', '\0'};
        out.write(rank_magic, sizeof(rank_magic));
        const std::int32_t version = 1;
        write_binary(out, version);
        write_binary(out, static_cast<std::int32_t>(global_p.Nr));
        write_binary(out, static_cast<std::int32_t>(global_p.Nw));
        write_binary(out, static_cast<std::int32_t>(decomp.global_NL));
        write_binary(out, static_cast<std::int32_t>(radial_grid_code(global_p.radial_grid)));
        write_binary(out, global_p.r_min);
        write_binary(out, global_p.r_max);
        write_binary(out, global_p.inner_refine_radius);
        write_binary(out, static_cast<std::int32_t>(global_p.inner_refine_cells));
        write_binary(out, global_p.w_max);
        write_binary(out, global_p.L_max);
        write_binary(out, global_p.w_stretch);
        write_binary(out, global_p.w_center);
        write_binary(out, global_p.central_mass);
        write_binary(out, t);
        write_binary(out, static_cast<std::int64_t>(step));
        write_binary(out, static_cast<std::int32_t>(mpi.size));
        write_binary(out, static_cast<std::int32_t>(mpi.rank));
        write_binary(out, static_cast<std::int32_t>(decomp.offset));
        write_binary(out, static_cast<std::int32_t>(decomp.count));
        write_binary(out, static_cast<std::uint64_t>(f.size()));
        out.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size() * sizeof(double)));
        if (!out) {
            throw std::runtime_error("Failed while writing rank restart state array.");
        }
        out.close();
        if (!out) {
            throw std::runtime_error("Failed to close rank restart file after writing: " + rank_tmp.string());
        }
    }
    std::filesystem::rename(rank_tmp, rank_path);
    mpi_barrier(mpi);

    if (is_root(mpi)) {
        const std::filesystem::path tmp = path.string() + ".tmp";
        std::ofstream out(tmp, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open MPI restart manifest for writing: " + tmp.string());
        }
        const char magic[16] = {'E', 'V', 'M', 'P', 'I', 'R', 'S', 'T', '1', '\0'};
        out.write(magic, sizeof(magic));
        const std::int32_t version = 1;
        write_binary(out, version);
        write_binary(out, static_cast<std::int32_t>(global_p.Nr));
        write_binary(out, static_cast<std::int32_t>(global_p.Nw));
        write_binary(out, static_cast<std::int32_t>(decomp.global_NL));
        write_binary(out, static_cast<std::int32_t>(radial_grid_code(global_p.radial_grid)));
        write_binary(out, global_p.r_min);
        write_binary(out, global_p.r_max);
        write_binary(out, global_p.inner_refine_radius);
        write_binary(out, static_cast<std::int32_t>(global_p.inner_refine_cells));
        write_binary(out, global_p.w_max);
        write_binary(out, global_p.L_max);
        write_binary(out, global_p.w_stretch);
        write_binary(out, global_p.w_center);
        write_binary(out, global_p.central_mass);
        write_binary(out, t);
        write_binary(out, static_cast<std::int64_t>(step));
        write_binary(out, static_cast<std::int32_t>(mpi.size));
        out.close();
        if (!out) {
            throw std::runtime_error("Failed to close MPI restart manifest after writing: " + tmp.string());
        }
        std::filesystem::rename(tmp, path);
    }
    mpi_barrier(mpi);
    (void)local_p;
}

void validate_restart_grid_header(
    std::ifstream& in,
    std::int32_t version,
    const Params& global_p,
    const LDecomposition& decomp,
    bool has_inner_refine) {
    if (version < 1) {
        throw std::runtime_error("Unsupported restart file version.");
    }
    const std::int32_t nr = read_binary<std::int32_t>(in);
    const std::int32_t nw = read_binary<std::int32_t>(in);
    const std::int32_t nl = read_binary<std::int32_t>(in);
    if (nr != global_p.Nr || nw != global_p.Nw || nl != decomp.global_NL) {
        throw std::runtime_error("Restart file dimensions do not match requested global grid.");
    }
    const std::int32_t saved_radial_grid_code = read_binary<std::int32_t>(in);
    const std::string saved_radial_grid = radial_grid_name(saved_radial_grid_code);
    if (saved_radial_grid != global_p.radial_grid) {
        throw std::runtime_error("Restart radial grid does not match requested radial grid.");
    }
    validate_restart_double("r_min", read_binary<double>(in), global_p.r_min);
    validate_restart_double("r_max", read_binary<double>(in), global_p.r_max);
    if (has_inner_refine) {
        validate_restart_double("inner_refine_radius", read_binary<double>(in), global_p.inner_refine_radius);
        const std::int32_t saved_inner_cells = read_binary<std::int32_t>(in);
        if (saved_inner_cells != global_p.inner_refine_cells) {
            throw std::runtime_error("Restart grid mismatch for inner_refine_cells");
        }
    }
    validate_restart_double("w_max", read_binary<double>(in), global_p.w_max);
    validate_restart_double("L_max", read_binary<double>(in), global_p.L_max);
    validate_restart_double("w_stretch", read_binary<double>(in), global_p.w_stretch);
    validate_restart_double("w_center", read_binary<double>(in), global_p.w_center);
    validate_restart_double("central_mass", read_binary<double>(in), global_p.central_mass);
}

std::vector<double> read_single_restart_local_slab(
    const std::filesystem::path& path,
    const Params& global_p,
    const LDecomposition& decomp,
    double& t,
    int& step) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open restart file: " + path.string());
    }
    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVLOGRST1")) {
        throw std::runtime_error("Restart file has wrong magic header.");
    }
    const std::int32_t version = read_binary<std::int32_t>(in);
    if (version != 1 && version != 2 && version != 3) {
        throw std::runtime_error("Unsupported restart file version.");
    }
    const std::int32_t nr = read_binary<std::int32_t>(in);
    const std::int32_t nw = read_binary<std::int32_t>(in);
    const std::int32_t nl = read_binary<std::int32_t>(in);
    if (nr != global_p.Nr || nw != global_p.Nw || nl != decomp.global_NL) {
        throw std::runtime_error("Restart file dimensions do not match requested global grid.");
    }
    const std::int32_t saved_radial_grid_code = (version >= 2) ? read_binary<std::int32_t>(in) : 0;
    const std::string saved_radial_grid = radial_grid_name(saved_radial_grid_code);
    if (saved_radial_grid != global_p.radial_grid) {
        throw std::runtime_error("Restart radial grid does not match requested radial grid.");
    }
    validate_restart_double("r_min", read_binary<double>(in), global_p.r_min);
    validate_restart_double("r_max", read_binary<double>(in), global_p.r_max);
    if (version >= 3) {
        validate_restart_double("inner_refine_radius", read_binary<double>(in), global_p.inner_refine_radius);
        const std::int32_t saved_inner_cells = read_binary<std::int32_t>(in);
        if (saved_inner_cells != global_p.inner_refine_cells) {
            throw std::runtime_error("Restart grid mismatch for inner_refine_cells");
        }
    }
    validate_restart_double("w_max", read_binary<double>(in), global_p.w_max);
    validate_restart_double("L_max", read_binary<double>(in), global_p.L_max);
    validate_restart_double("w_stretch", read_binary<double>(in), global_p.w_stretch);
    validate_restart_double("w_center", read_binary<double>(in), global_p.w_center);
    validate_restart_double("central_mass", read_binary<double>(in), global_p.central_mass);
    t = read_binary<double>(in);
    const std::int64_t saved_step = read_binary<std::int64_t>(in);
    validate_restart_clock(t, saved_step);
    step = static_cast<int>(saved_step);
    const std::uint64_t n_values = read_binary<std::uint64_t>(in);
    const std::uint64_t expected_values =
        static_cast<std::uint64_t>(global_p.Nr) * global_p.Nw * decomp.global_NL;
    if (n_values != expected_values) {
        throw std::runtime_error("Restart state array has wrong size.");
    }
    const std::streamoff f0 = in.tellg();
    std::vector<double> f(
        static_cast<std::size_t>(global_p.Nr) * global_p.Nw * decomp.count,
        0.0);
    if (decomp.count == 0) {
        return f;
    }
    for (int i = 0; i < global_p.Nr; ++i) {
        for (int j = 0; j < global_p.Nw; ++j) {
            const std::uint64_t global_base =
                (static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(global_p.Nw)
                 + static_cast<std::uint64_t>(j))
                    * static_cast<std::uint64_t>(decomp.global_NL)
                + static_cast<std::uint64_t>(decomp.offset);
            const std::streamoff offset =
                f0 + static_cast<std::streamoff>(global_base * static_cast<std::uint64_t>(sizeof(double)));
            in.seekg(offset);
            if (!in) {
                throw std::runtime_error("Failed to seek local restart slab in " + path.string());
            }
            double* dest = f.data()
                + (static_cast<std::size_t>(i) * global_p.Nw + static_cast<std::size_t>(j))
                    * static_cast<std::size_t>(decomp.count);
            in.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(decomp.count * sizeof(double)));
            if (!in) {
                throw std::runtime_error("Unexpected end of local restart slab in " + path.string());
            }
            for (int k = 0; k < decomp.count; ++k) {
                if (!(dest[k] >= 0.0) || !std::isfinite(dest[k])) {
                    throw std::runtime_error(
                        "Restart state must be finite and nonnegative.");
                }
            }
        }
    }
    return f;
}

std::vector<double> read_restart_state_distributed(
    const std::filesystem::path& path,
    const Params& global_p,
    const Params& local_p,
    const LDecomposition& decomp,
    const MpiContext& mpi,
    double& t,
    int& step) {
    std::ifstream manifest(path, std::ios::binary);
    if (!manifest) {
        throw std::runtime_error("Could not open restart file: " + path.string());
    }
    char magic[16] = {};
    manifest.read(magic, sizeof(magic));
    const bool single_file_magic =
        manifest && fixed_binary_identifier_equals(magic, "EVLOGRST1");
    const bool mpi_manifest_magic =
        manifest && fixed_binary_identifier_equals(magic, "EVMPIRST1");
    manifest.close();

    if (single_file_magic) {
        return read_single_restart_local_slab(path, global_p, decomp, t, step);
    }
    if (!mpi_manifest_magic) {
        throw std::runtime_error("Restart file has wrong magic header.");
    }

    {
        std::ifstream in(path, std::ios::binary);
        in.read(magic, sizeof(magic));
        const std::int32_t version = read_binary<std::int32_t>(in);
        if (version != 1) {
            throw std::runtime_error("Unsupported MPI restart manifest version.");
        }
        validate_restart_grid_header(in, version, global_p, decomp, true);
        t = read_binary<double>(in);
        const std::int64_t saved_step = read_binary<std::int64_t>(in);
        validate_restart_clock(t, saved_step);
        step = static_cast<int>(saved_step);
        const std::int32_t saved_size = read_binary<std::int32_t>(in);
        if (saved_size != mpi.size) {
            throw std::runtime_error(
                "Distributed restart currently requires the same MPI rank count used when it was written.");
        }
    }

    const std::filesystem::path rank_path = rank_restart_path(path, mpi.rank);
    std::ifstream in(rank_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open rank restart file: " + rank_path.string());
    }
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVMPIRNK1")) {
        throw std::runtime_error("Rank restart file has wrong magic header.");
    }
    const std::int32_t version = read_binary<std::int32_t>(in);
    if (version != 1) {
        throw std::runtime_error("Unsupported rank restart file version.");
    }
    validate_restart_grid_header(in, version, global_p, decomp, true);
    const double rank_t = read_binary<double>(in);
    validate_restart_double("restart time", rank_t, t);
    const std::int64_t saved_step = read_binary<std::int64_t>(in);
    if (saved_step != step) {
        throw std::runtime_error("Rank restart step does not match manifest.");
    }
    const std::int32_t saved_size = read_binary<std::int32_t>(in);
    const std::int32_t saved_rank = read_binary<std::int32_t>(in);
    const std::int32_t saved_offset = read_binary<std::int32_t>(in);
    const std::int32_t saved_count = read_binary<std::int32_t>(in);
    if (saved_size != mpi.size || saved_rank != mpi.rank
        || saved_offset != decomp.offset || saved_count != decomp.count) {
        throw std::runtime_error("Rank restart decomposition does not match current MPI layout.");
    }
    const std::uint64_t n_values = read_binary<std::uint64_t>(in);
    const std::uint64_t expected_values =
        static_cast<std::uint64_t>(local_p.Nr) * local_p.Nw * local_p.NL;
    if (n_values != expected_values) {
        throw std::runtime_error("Rank restart state array has wrong size.");
    }
    std::vector<double> f(static_cast<std::size_t>(n_values));
    in.read(reinterpret_cast<char*>(f.data()), static_cast<std::streamsize>(f.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Unexpected end of rank restart state array.");
    }
    validate_restart_distribution(f, local_p);
    return f;
}

void write_metadata(
    const std::filesystem::path& path,
    const Params& p,
    const Params& local_p,
    const LDecomposition& decomp,
    const MpiContext& mpi) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open metadata output file: " + path.string());
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"Nr\": " << p.Nr << ",\n";
    out << "  \"Nw\": " << p.Nw << ",\n";
    out << "  \"NL\": " << p.NL << ",\n";
    out << "  \"r_min\": " << p.r_min << ",\n";
    out << "  \"r_max\": " << p.r_max << ",\n";
    out << "  \"w_max\": " << p.w_max << ",\n";
    out << "  \"L_max\": " << p.L_max << ",\n";
    out << "  \"radial_grid\": \"" << p.radial_grid << "\",\n";
    out << "  \"inner_refine_radius\": " << p.inner_refine_radius << ",\n";
    out << "  \"inner_refine_cells\": " << p.inner_refine_cells << ",\n";
    out << "  \"w_stretch\": " << p.w_stretch << ",\n";
    out << "  \"w_center\": " << p.w_center << ",\n";
    out << "  \"adaptive_w\": " << (p.adaptive_w ? "true" : "false") << ",\n";
    out << "  \"adaptive_w_every\": " << p.adaptive_w_every << ",\n";
    out << "  \"adaptive_w_trigger\": " << p.adaptive_w_trigger << ",\n";
    out << "  \"adaptive_w_margin\": " << p.adaptive_w_margin << ",\n";
    out << "  \"adaptive_w_growth\": " << p.adaptive_w_growth << ",\n";
    out << "  \"adaptive_w_max\": " << p.adaptive_w_max << ",\n";
    out << "  \"integrator\": \"" << p.integrator << "\",\n";
    out << "  \"time_integrator\": \"semi_lagrangian_kick_drift_kick_metric_frozen\",\n";
    out << "  \"inner_boundary\": \"" << p.inner_boundary << "\",\n";
    out << "  \"outer_boundary\": \"" << p.outer_boundary << "\",\n";
    out << "  \"momentum_boundary\": \"" << p.momentum_boundary << "\",\n";
    out << "  \"inner_boundary_condition\": \"mirror_r_about_r_min_and_w_about_zero_with_boundary_slice_symmetrization\",\n";
    out << "  \"phase_projection\": \"g(r,w)=int_dL_L_f\",\n";
    out << "  \"t_end\": " << p.t_end << ",\n";
    out << "  \"cfl\": " << p.cfl << ",\n";
    out << "  \"dt_max\": " << p.dt_max << ",\n";
    out << "  \"max_steps\": " << p.max_steps << ",\n";
    out << "  \"save_every\": " << p.save_every << ",\n";
    out << "  \"phase_every\": " << p.phase_every << ",\n";
    out << "  \"phase_r_stride\": " << p.phase_r_stride << ",\n";
    out << "  \"phase_w_stride\": " << p.phase_w_stride << ",\n";
    out << "  \"restart_every\": " << p.restart_every << ",\n";
    out << "  \"restart_from\": \"" << p.restart_from << "\",\n";
    out << "  \"central_mass\": " << p.central_mass << ",\n";
    out << "  \"initial_data\": \"" << p.initial_data << "\",\n";
    out << "  \"initial_data_file\": \"" << p.initial_data_file << "\",\n";
    out << "  \"custom_ic_normalize_mass\": "
        << (p.custom_ic_normalize_mass ? "true" : "false") << ",\n";
    out << "  \"custom_ic_target_mass\": " << p.custom_ic_target_mass << ",\n";
    out << "  \"shell_initial_df\": " << (p.shell_initial_df ? "true" : "false") << ",\n";
    out << "  \"R0\": " << p.R0 << ",\n";
    out << "  \"dR\": " << p.dR << ",\n";
    out << "  \"top_hat_shell\": " << (p.top_hat_shell ? "true" : "false") << ",\n";
    out << "  \"top_hat_density_shell\": " << (p.top_hat_density_shell ? "true" : "false") << ",\n";
    out << "  \"gaussian_density_shell\": " << (p.gaussian_density_shell ? "true" : "false") << ",\n";
    out << "  \"double_gaussian_density_shell\": "
        << (p.double_gaussian_density_shell ? "true" : "false") << ",\n";
    out << "  \"R0_2\": " << p.R0_2 << ",\n";
    out << "  \"dR_2\": " << p.dR_2 << ",\n";
    out << "  \"amp_2\": " << p.amp_2 << ",\n";
    out << "  \"w_in_2\": " << p.w_in_2 << ",\n";
    out << "  \"shell_radial_mode\": \""
        << (p.double_gaussian_density_shell
                ? "density_double_gaussian"
                : (p.top_hat_density_shell ? "density_tophat"
                                           : (p.gaussian_density_shell
                                                  ? "density_gaussian"
                                                  : (p.top_hat_shell ? "f_tophat" : "f_gaussian"))))
        << "\",\n";
    out << "  \"beta\": " << p.beta << ",\n";
    out << "  \"amp\": " << p.amp << ",\n";
    out << "  \"w_in\": " << p.w_in << ",\n";
    out << "  \"initial_df\": \""
        << (using_binary_initial_data(p)
                ? "binary_f_of_r_w_L"
                : (p.shell_initial_df ? "shell_boundary_seed_F0_with_R0_for_zeta_b"
                                      : "boundary_seed_F0_with_r_for_zeta"))
        << "\",\n";
    out << "  \"initial_metric_iters\": " << p.initial_metric_iters << ",\n";
    out << "  \"metric_floor\": " << p.metric_floor << ",\n";
    out << "  \"gap_stop\": " << p.gap_stop << ",\n";
    out << "  \"active_floor_rel\": " << p.active_floor_rel << ",\n";
    out << "  \"active_floor_abs\": " << p.active_floor_abs << ",\n";
    out << "  \"f_floor\": " << p.f_floor << ",\n";
    out << "  \"threads\": " << p.threads << ",\n";
    out << "  \"parallel_backend\": \"" << parallel_backend_name() << "\",\n";
    out << "  \"mpi_enabled\": " << (mpi.enabled ? "true" : "false") << ",\n";
    out << "  \"mpi_size\": " << mpi.size << ",\n";
    out << "  \"detected_multi_rank_size\": " << detected_multi_rank_size() << ",\n";
    out << "  \"allow_multi_rank_launch\": " << (p.allow_multi_rank_launch ? "true" : "false") << ",\n";
    out << "  \"decomposition\": \"L_slab\",\n";
    out << "  \"global_NL\": " << decomp.global_NL << ",\n";
    out << "  \"rank0_local_L_offset\": " << decomp.offset << ",\n";
    out << "  \"rank0_local_NL\": " << local_p.NL << ",\n";
    out << "  \"restart_layout\": \""
        << ((mpi.size > 1) ? "mpi_manifest_plus_rank_files" : "single_file") << "\"\n";
    out << "}\n";
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to close metadata output file after writing: " + path.string());
    }
}

std::string trim_copy(const std::string& s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    return s.substr(first, last - first);
}

std::string strip_inline_comment(const std::string& s) {
    bool in_single = false;
    bool in_double = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '\'' && !in_double) {
            in_single = !in_single;
        } else if (c == '"' && !in_single) {
            in_double = !in_double;
        } else if (!in_single && !in_double && (c == '#' || c == ';')) {
            return s.substr(0, i);
        }
    }
    return s;
}

std::string unquote_copy(const std::string& s) {
    std::string out = trim_copy(s);
    if (out.size() >= 2
        && ((out.front() == '"' && out.back() == '"')
            || (out.front() == '\'' && out.back() == '\''))) {
        out = out.substr(1, out.size() - 2);
    }
    return out;
}

std::string normalize_key(std::string key) {
    key = trim_copy(key);
    while (key.rfind("--", 0) == 0) {
        key.erase(0, 2);
    }
    for (char& c : key) {
        if (c == '_') {
            c = '-';
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return key;
}

bool parse_bool_value(const std::string& text, const std::string& key) {
    std::string value = normalize_key(text);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("Expected boolean value for " + key + ", got '" + text + "'.");
}

const char* getenv_nonempty(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

int parse_positive_int_text(const std::string& text, const std::string& source) {
    const std::string value = trim_copy(text);
    std::size_t pos = 0;
    int out = 0;
    try {
        out = std::stoi(value, &pos);
    } catch (const std::exception&) {
        throw std::runtime_error("Expected a positive integer for " + source + ", got '" + text + "'.");
    }
    if (pos != value.size() || out <= 0) {
        throw std::runtime_error("Expected a positive integer for " + source + ", got '" + text + "'.");
    }
    return out;
}

int env_positive_int_or_zero(const char* name) {
    const char* value = getenv_nonempty(name);
    if (value == nullptr) {
        return 0;
    }
    return parse_positive_int_text(value, std::string("environment variable ") + name);
}

int first_positive_env_int(const std::vector<std::string>& names) {
    for (const std::string& name : names) {
        const int value = env_positive_int_or_zero(name.c_str());
        if (value > 0) {
            return value;
        }
    }
    return 0;
}

void apply_environment_overrides(Params& p) {
    if (const char* output_dir = getenv_nonempty("EV_OUTPUT_DIR")) {
        p.output_dir = output_dir;
    }
    if (const char* output_dir = getenv_nonempty("EV_SOLVER_OUTPUT_DIR")) {
        p.output_dir = output_dir;
    }
    if (const char* restart_from = getenv_nonempty("EV_RESTART_FROM")) {
        p.restart_from = restart_from;
    }
    const int forced_threads = first_positive_env_int({"EV_THREADS", "EV_SOLVER_THREADS"});
    if (forced_threads > 0) {
        p.threads = forced_threads;
    }
    if (const char* allow = getenv_nonempty("EV_ALLOW_MULTI_RANK_LAUNCH")) {
        p.allow_multi_rank_launch = parse_bool_value(allow, "EV_ALLOW_MULTI_RANK_LAUNCH");
    }
}

void finalize_parallel_runtime(Params& p) {
    if (p.threads <= 0) {
        const int env_threads = first_positive_env_int({"OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"});
        p.threads = (env_threads > 0) ? env_threads : max_backend_threads();
    }
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, p.threads));
#endif
}

int detected_multi_rank_size() {
    int world_size = 1;
    for (const char* name : {
             "SLURM_NTASKS",
             "OMPI_COMM_WORLD_SIZE",
             "PMI_SIZE",
             "PMIX_SIZE",
             "MPI_LOCALNRANKS"}) {
        const int value = env_positive_int_or_zero(name);
        world_size = std::max(world_size, value);
    }
    return world_size;
}

void check_parallel_launch_safety(const Params& p) {
#ifdef EV_USE_MPI
    (void)p;
    return;
#else
    const int world_size = detected_multi_rank_size();
    if (world_size <= 1 || p.allow_multi_rank_launch) {
        return;
    }
    throw std::runtime_error(
        "Detected a multi-rank Slurm/MPI launch with " + std::to_string(world_size)
        + " ranks, but this non-MPI executable cannot coordinate a shared output directory. "
        "Build the MPI target for distributed L-slab runs, or use one task with many OpenMP threads, e.g. --ntasks=1 "
        "--cpus-per-task=$OMP_NUM_THREADS, or set EV_ALLOW_MULTI_RANK_LAUNCH=true "
        "only if every rank has a distinct output directory.");
#endif
}

// Config-file keys and values enter through one conversion table.  Normalize
// spelling first, then convert to the corresponding physical or numerical
// parameter.  Unknown keys fail here so misspellings cannot select defaults.
void apply_param(Params& p, const std::string& raw_key, const std::string& raw_value, const std::string& source) {
    const std::string key = normalize_key(raw_key);
    const std::string value = unquote_copy(raw_value);
    auto to_int = [&]() { return std::stoi(value); };
    auto to_double = [&]() { return std::stod(value); };
    auto to_bool = [&]() { return parse_bool_value(value, key); };

    if (key == "physics-model") {
        if (value != "einstein_vlasov") {
            throw std::runtime_error(
                "physics_model=" + value
                + " requires the modular VENUS VP/BP solver, not venus_einstein_vlasov.");
        }
    } else if (key == "nr") {
        p.Nr = to_int();
    } else if (key == "nw") {
        p.Nw = to_int();
    } else if (key == "nl") {
        p.NL = to_int();
    } else if (key == "r-min") {
        p.r_min = to_double();
    } else if (key == "r-max") {
        p.r_max = to_double();
    } else if (key == "radial-grid") {
        p.radial_grid = value;
    } else if (key == "inner-refine-radius") {
        p.inner_refine_radius = to_double();
    } else if (key == "inner-refine-cells") {
        p.inner_refine_cells = to_int();
    } else if (key == "w-max") {
        p.w_max = to_double();
    } else if (key == "l-max") {
        p.L_max = to_double();
    } else if (key == "w-stretch") {
        p.w_stretch = to_double();
    } else if (key == "w-center") {
        p.w_center = to_double();
    } else if (key == "adaptive-w") {
        p.adaptive_w = to_bool();
    } else if (key == "adaptive-w-every") {
        p.adaptive_w_every = std::max(1, to_int());
    } else if (key == "adaptive-w-trigger") {
        p.adaptive_w_trigger = to_double();
    } else if (key == "adaptive-w-margin") {
        p.adaptive_w_margin = to_double();
    } else if (key == "adaptive-w-growth") {
        p.adaptive_w_growth = to_double();
    } else if (key == "adaptive-w-max") {
        p.adaptive_w_max = to_double();
    } else if (key == "t-end") {
        p.t_end = to_double();
    } else if (key == "cfl") {
        p.cfl = to_double();
    } else if (key == "dt-max") {
        p.dt_max = to_double();
    } else if (key == "max-steps") {
        p.max_steps = to_int();
    } else if (key == "save-every") {
        p.save_every = std::max(1, to_int());
    } else if (key == "phase-every") {
        p.phase_every = std::max(0, to_int());
    } else if (key == "phase-r-stride") {
        p.phase_r_stride = std::max(1, to_int());
    } else if (key == "phase-w-stride") {
        p.phase_w_stride = std::max(1, to_int());
    } else if (key == "restart-every") {
        p.restart_every = std::max(0, to_int());
    } else if (key == "integrator") {
        p.integrator = value;
    } else if (key == "inner-boundary") {
        p.inner_boundary = value;
    } else if (key == "outer-boundary") {
        p.outer_boundary = value;
    } else if (key == "momentum-boundary") {
        p.momentum_boundary = value;
    } else if (key == "restart-from") {
        p.restart_from = value;
    } else if (key == "threads") {
        p.threads = to_int();
    } else if (key == "allow-multi-rank-launch") {
        p.allow_multi_rank_launch = to_bool();
    } else if (key == "central-mass") {
        p.central_mass = to_double();
    } else if (key == "initial-data" || key == "initial-data-mode") {
        p.initial_data = value;
    } else if (key == "initial-data-file") {
        p.initial_data_file = value;
    } else if (key == "custom-ic-normalize-mass" || key == "normalize-initial-mass") {
        p.custom_ic_normalize_mass = to_bool();
    } else if (key == "custom-ic-target-mass" || key == "target-initial-mass") {
        p.custom_ic_target_mass = to_double();
    } else if (key == "shell-initial-df") {
        p.shell_initial_df = to_bool();
    } else if (key == "r0") {
        p.R0 = to_double();
    } else if (key == "dr") {
        p.dR = to_double();
    } else if (key == "top-hat-shell") {
        p.top_hat_shell = to_bool();
        p.shell_initial_df = p.shell_initial_df || p.top_hat_shell;
    } else if (key == "top-hat-density-shell") {
        p.top_hat_density_shell = to_bool();
        p.shell_initial_df = p.shell_initial_df || p.top_hat_density_shell;
    } else if (key == "gaussian-density-shell") {
        p.gaussian_density_shell = to_bool();
        p.shell_initial_df = p.shell_initial_df || p.gaussian_density_shell;
    } else if (key == "double-gaussian-density-shell") {
        p.double_gaussian_density_shell = to_bool();
        if (p.double_gaussian_density_shell) {
            p.shell_initial_df = false;
        }
    } else if (key == "r0-2") {
        p.R0_2 = to_double();
    } else if (key == "dr-2") {
        p.dR_2 = to_double();
    } else if (key == "amp-2") {
        p.amp_2 = to_double();
    } else if (key == "w-in-2") {
        p.w_in_2 = to_double();
    } else if (key == "beta") {
        p.beta = to_double();
    } else if (key == "amp") {
        p.amp = to_double();
    } else if (key == "w-in") {
        p.w_in = to_double();
    } else if (key == "initial-metric-iters") {
        p.initial_metric_iters = std::max(0, to_int());
    } else if (key == "metric-floor") {
        p.metric_floor = to_double();
    } else if (key == "gap-stop") {
        p.gap_stop = to_double();
    } else if (key == "active-floor-rel") {
        p.active_floor_rel = to_double();
    } else if (key == "active-floor-abs") {
        p.active_floor_abs = to_double();
    } else if (key == "f-floor") {
        p.f_floor = to_double();
    } else if (key == "output-dir") {
        p.output_dir = value;
    } else {
        throw std::runtime_error("Unknown option '" + raw_key + "' in " + source + ".");
    }
}

void load_config_file(const std::filesystem::path& path, Params& p) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open config file: " + path.string());
    }
    const std::filesystem::path config_dir = std::filesystem::absolute(path).parent_path();
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        line = trim_copy(strip_inline_comment(line));
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            continue;
        }
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            eq = line.find(':');
        }
        if (eq == std::string::npos) {
            throw std::runtime_error(
                "Expected 'key = value' in " + path.string() + ":" + std::to_string(line_no));
        }
        const std::string key = trim_copy(line.substr(0, eq));
        const std::string value = trim_copy(line.substr(eq + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "Empty key or value in " + path.string() + ":" + std::to_string(line_no));
        }
        apply_param(p, key, value, path.string() + ":" + std::to_string(line_no));
    }

    auto resolve_path = [&](std::string& value, bool allow_empty) {
        if (value.empty()) {
            if (allow_empty) {
                return;
            }
            throw std::runtime_error("Empty path in config file: " + path.string());
        }
        std::filesystem::path candidate(value);
        if (candidate.is_relative()) {
            value = (config_dir / candidate).lexically_normal().string();
        }
    };
    resolve_path(p.initial_data_file, false);
    resolve_path(p.output_dir, false);
    resolve_path(p.restart_from, true);
}

std::filesystem::path default_config_path(const char* argv0) {
    const std::filesystem::path cwd_config = std::filesystem::current_path() / "params.ini";
    if (std::filesystem::exists(cwd_config)) {
        return cwd_config;
    }
    const std::filesystem::path exe_path = std::filesystem::absolute(argv0);
    const std::filesystem::path exe_config = exe_path.parent_path() / "params.ini";
    if (std::filesystem::exists(exe_config)) {
        return exe_config;
    }
    return {};
}

void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  The solver auto-loads params.ini from the current directory or executable directory.\n"
        << "  --config FILE, -i FILE  Read a different sectioned key=value parameter file before CLI overrides\n"
        << "  EV_CONFIG may also name the input file; CLI options override config and environment values.\n"
        << "  --Nr N --Nw N --NL N\n"
        << "  --radial-grid log|linear|piecewise-linear --r-min X --r-max X\n"
        << "  --inner-refine-radius X --inner-refine-cells N --w-max X --L-max X --w-stretch X --w-center X\n"
        << "  --adaptive-w --adaptive-w-every N --adaptive-w-trigger X --adaptive-w-margin X --adaptive-w-growth X --adaptive-w-max X\n"
        << "  --t-end X --dt-max X --cfl X --save-every N --phase-every N --max-steps N\n"
        << "  --phase-r-stride N --phase-w-stride N --restart-every N --restart-from PATH\n"
        << "  --integrator semi_lagrangian_kick_drift_kick\n"
        << "  --inner-boundary reflecting --outer-boundary outflow_vacuum --momentum-boundary outflow_vacuum\n"
        << "  --threads N  (0 means auto from EV_THREADS/OMP_NUM_THREADS/SLURM_CPUS_PER_TASK)\n"
        << "  --allow-multi-rank-launch  Non-MPI build only: bypass duplicate-rank output guard\n"
        << "  --initial-data analytic|binary_rwl --initial-data-file PATH\n"
        << "  --custom-ic-normalize-mass --custom-ic-target-mass X\n"
        << "  --central-mass X --beta X --amp X --w-in X --initial-metric-iters N\n"
        << "  --metric-floor X --gap-stop X\n"
        << "  --shell-initial-df --R0 X --dR X --top-hat-shell --top-hat-density-shell --gaussian-density-shell\n"
        << "  --double-gaussian-density-shell --R0-2 X --dR-2 X --amp-2 X --w-in-2 X\n"
        << "  --active-floor-rel X --active-floor-abs X --f-floor X --output-dir DIR\n";
}

// ---- User configuration and launch validation ------------------------------
// Load the selected parameter file first, apply environment settings, then
// apply explicit command-line values.  This precedence makes a supplied
// command-line setting the final choice for the corresponding parameter.
Params parse_args(int argc, char** argv) {
    Params p;

    bool saw_config = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        }
        if (arg == "--config" || arg == "-i" || arg == "--input" || arg == "--input-file") {
            saw_config = true;
            load_config_file(need(arg), p);
        }
    }
    if (!saw_config) {
        const char* env_config = getenv_nonempty("EV_CONFIG");
        const std::filesystem::path config_path =
            (env_config != nullptr) ? std::filesystem::path(env_config) : default_config_path(argv[0]);
        if (!config_path.empty()) {
            load_config_file(config_path, p);
        }
    }

    apply_environment_overrides(p);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--config" || arg == "-i" || arg == "--input" || arg == "--input-file") {
            (void)need(arg);
        } else if (arg == "--Nr") {
            p.Nr = std::stoi(need(arg));
        } else if (arg == "--Nw") {
            p.Nw = std::stoi(need(arg));
        } else if (arg == "--NL") {
            p.NL = std::stoi(need(arg));
        } else if (arg == "--r-min") {
            p.r_min = std::stod(need(arg));
        } else if (arg == "--r-max") {
            p.r_max = std::stod(need(arg));
        } else if (arg == "--radial-grid") {
            p.radial_grid = need(arg);
        } else if (arg == "--inner-refine-radius") {
            p.inner_refine_radius = std::stod(need(arg));
        } else if (arg == "--inner-refine-cells") {
            p.inner_refine_cells = std::stoi(need(arg));
        } else if (arg == "--w-max") {
            p.w_max = std::stod(need(arg));
        } else if (arg == "--L-max") {
            p.L_max = std::stod(need(arg));
        } else if (arg == "--w-stretch") {
            p.w_stretch = std::stod(need(arg));
        } else if (arg == "--w-center") {
            p.w_center = std::stod(need(arg));
        } else if (arg == "--adaptive-w") {
            p.adaptive_w = true;
        } else if (arg == "--adaptive-w-every") {
            p.adaptive_w_every = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--adaptive-w-trigger") {
            p.adaptive_w_trigger = std::stod(need(arg));
        } else if (arg == "--adaptive-w-margin") {
            p.adaptive_w_margin = std::stod(need(arg));
        } else if (arg == "--adaptive-w-growth") {
            p.adaptive_w_growth = std::stod(need(arg));
        } else if (arg == "--adaptive-w-max") {
            p.adaptive_w_max = std::stod(need(arg));
        } else if (arg == "--t-end") {
            p.t_end = std::stod(need(arg));
        } else if (arg == "--cfl") {
            p.cfl = std::stod(need(arg));
        } else if (arg == "--dt-max") {
            p.dt_max = std::stod(need(arg));
        } else if (arg == "--max-steps") {
            p.max_steps = std::stoi(need(arg));
        } else if (arg == "--save-every") {
            p.save_every = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--phase-every") {
            p.phase_every = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--phase-r-stride") {
            p.phase_r_stride = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--phase-w-stride") {
            p.phase_w_stride = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--restart-every") {
            p.restart_every = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--restart-from") {
            p.restart_from = need(arg);
        } else if (arg == "--integrator") {
            p.integrator = need(arg);
        } else if (arg == "--inner-boundary") {
            p.inner_boundary = need(arg);
        } else if (arg == "--outer-boundary") {
            p.outer_boundary = need(arg);
        } else if (arg == "--momentum-boundary") {
            p.momentum_boundary = need(arg);
        } else if (arg == "--threads") {
            p.threads = std::stoi(need(arg));
        } else if (arg == "--allow-multi-rank-launch") {
            p.allow_multi_rank_launch = true;
        } else if (arg == "--central-mass") {
            p.central_mass = std::stod(need(arg));
        } else if (arg == "--initial-data") {
            p.initial_data = need(arg);
        } else if (arg == "--initial-data-file") {
            p.initial_data_file = need(arg);
        } else if (arg == "--custom-ic-normalize-mass") {
            p.custom_ic_normalize_mass = true;
        } else if (arg == "--custom-ic-target-mass") {
            p.custom_ic_target_mass = std::stod(need(arg));
        } else if (arg == "--shell-initial-df") {
            p.shell_initial_df = true;
        } else if (arg == "--R0") {
            p.R0 = std::stod(need(arg));
        } else if (arg == "--dR") {
            p.dR = std::stod(need(arg));
        } else if (arg == "--top-hat-shell") {
            p.top_hat_shell = true;
            p.shell_initial_df = true;
        } else if (arg == "--top-hat-density-shell") {
            p.shell_initial_df = true;
            p.top_hat_density_shell = true;
        } else if (arg == "--gaussian-density-shell") {
            p.shell_initial_df = true;
            p.gaussian_density_shell = true;
        } else if (arg == "--double-gaussian-density-shell") {
            p.shell_initial_df = false;
            p.double_gaussian_density_shell = true;
        } else if (arg == "--R0-2") {
            p.R0_2 = std::stod(need(arg));
        } else if (arg == "--dR-2") {
            p.dR_2 = std::stod(need(arg));
        } else if (arg == "--amp-2") {
            p.amp_2 = std::stod(need(arg));
        } else if (arg == "--w-in-2") {
            p.w_in_2 = std::stod(need(arg));
        } else if (arg == "--beta") {
            p.beta = std::stod(need(arg));
        } else if (arg == "--amp") {
            p.amp = std::stod(need(arg));
        } else if (arg == "--w-in") {
            p.w_in = std::stod(need(arg));
        } else if (arg == "--initial-metric-iters") {
            p.initial_metric_iters = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--metric-floor") {
            p.metric_floor = std::stod(need(arg));
        } else if (arg == "--gap-stop") {
            p.gap_stop = std::stod(need(arg));
        } else if (arg == "--active-floor-rel") {
            p.active_floor_rel = std::stod(need(arg));
        } else if (arg == "--active-floor-abs") {
            p.active_floor_abs = std::stod(need(arg));
        } else if (arg == "--f-floor") {
            p.f_floor = std::stod(need(arg));
        } else if (arg == "--output-dir") {
            p.output_dir = need(arg);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }
    finalize_parallel_runtime(p);
    check_parallel_launch_safety(p);
    return p;
}

}  // namespace

// ---- End-to-end standalone Einstein-Vlasov evolution -----------------------
int main(int argc, char** argv) {
    MpiContext mpi;
#ifdef EV_USE_MPI
    MPI_Init(&argc, &argv);
    mpi.enabled = true;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
#endif

    int exit_code = 0;
    try {
        // ---- Configuration and parallel phase-space ownership ------------
        // MPI ranks own disjoint L slabs; OpenMP threads share work inside
        // each slab.  The physical distribution is assembled by reductions.
        Params global_p = parse_args(argc, argv);

        // ---- Initial-data geometry ---------------------------------------
        // A binary EVICRWL1 carrier supplies its own (r,w,L) grid.  Analytic
        // data instead use the grid requested by the runtime parameters.
        InitialDataRWL custom_ic;
        const bool binary_ic = using_binary_initial_data(global_p);
        if (binary_ic) {
            custom_ic = read_initial_data_rwl_metadata(global_p.initial_data_file);
            global_p.Nr = custom_ic.Nr;
            global_p.Nw = custom_ic.Nw;
            global_p.NL = custom_ic.NL;
            global_p.r_min = custom_ic.r.front();
            global_p.r_max = custom_ic.r.back();
            global_p.L_max = custom_ic.L.back();
            global_p.w_max = std::max(std::abs(custom_ic.w.front()), std::abs(custom_ic.w.back()));
            global_p.radial_grid = "custom";
        }
        validate_params(global_p);
        if (mpi.size > global_p.NL) {
            throw std::runtime_error(
                "MPI L-slab decomposition requires mpi_size <= NL. "
                "Use no more MPI ranks than angular-momentum cells, or move extra cores to OpenMP "
                "threads per rank with --cpus-per-task/OMP_NUM_THREADS.");
        }

        const LDecomposition decomp = make_l_decomposition(global_p.NL, mpi);
        Params local_p = global_p;
        local_p.NL = decomp.count;

        if (is_root(mpi)) {
            std::cout << "Cluster solver parallel backend=" << parallel_backend_name()
                      << " threads_per_rank=" << global_p.threads
                      << " mpi_size=" << mpi.size
                      << " global_NL=" << decomp.global_NL << "\n";
        }

        const std::vector<double> r = binary_ic ? custom_ic.r : make_radial_grid(global_p);
        const std::vector<double> x_r = radial_coordinate_grid(r, global_p);
        std::vector<double> w =
            binary_ic ? custom_ic.w : stretched_w_grid(global_p.w_max, global_p.w_stretch, global_p.w_center, global_p.Nw);
        const std::vector<double> L_global = binary_ic ? custom_ic.L : linspace(0.0, global_p.L_max, global_p.NL);
        const std::vector<double> Lq_global = quadrature_weights(L_global);
        const std::vector<double> L = slice_vector(L_global, decomp);
        const std::vector<double> Lq = slice_vector(Lq_global, decomp);
        const double radial_step = min_spacing(x_r);
        double min_dw = min_spacing(w);
        const std::vector<double> rq = quadrature_weights(r);
        std::vector<double> wq = quadrature_weights(w);

        // ---- Fresh distribution or versioned restart ---------------------
        // A restart restores f and physical time exactly.  Fresh analytic
        // data may iterate the metric constraint so matter and spacetime are
        // mutually consistent before the first characteristic step.
        double t = 0.0;
        int step = 0;
        std::vector<double> f;
        if (!global_p.restart_from.empty()) {
            f = read_restart_state_distributed(global_p.restart_from, global_p, local_p, decomp, mpi, t, step);
            if (is_root(mpi)) {
                std::cout << "Restarted from " << global_p.restart_from
                          << " at step=" << step << " t=" << std::setprecision(17) << t << "\n";
            }
        } else {
            if (binary_ic) {
                custom_ic.f = read_initial_data_rwl_local_values(global_p.initial_data_file, custom_ic, decomp);
                f = custom_initial_distribution_rwl(custom_ic, local_p);
                if (global_p.custom_ic_normalize_mass) {
                    const Moments moments = compute_global_moments(f, r, w, L, wq, Lq, local_p, mpi);
                    const Metric metric = solve_metric(moments, r, global_p);
                    const double matter_mass = metric.m.back() - global_p.central_mass;
                    if (!(matter_mass > 0.0) || !std::isfinite(matter_mass)) {
                        throw std::runtime_error("Cannot normalize custom IC: computed matter mass is not positive.");
                    }
                    // Optional normalization multiplies every f by the same
                    // factor.  It sets the requested matter mass while
                    // preserving the imported distribution's shape.
                    const double scale = global_p.custom_ic_target_mass / matter_mass;
                    for (double& value : f) {
                        value *= scale;
                    }
                    if (is_root(mpi)) {
                        std::cout << "Normalized custom initial data mass from "
                                  << std::setprecision(12) << matter_mass
                                  << " to " << global_p.custom_ic_target_mass << "\n";
                    }
                }
            } else {
                std::vector<double> mu_seed(global_p.Nr, 0.0);
                f = initial_distribution(r, w, L, wq, Lq, mu_seed, x_r, local_p);
                // The analytic momentum seed depends on exp(mu).  Recompute
                // its moments and metric, then evaluate the seed in that
                // metric for the requested number of consistency iterations.
                for (int iter = 0; iter < global_p.initial_metric_iters; ++iter) {
                    const Moments moments = compute_global_moments(f, r, w, L, wq, Lq, local_p, mpi);
                    const Metric metric = solve_metric(moments, r, global_p);
                    f = initial_distribution(r, w, L, wq, Lq, metric.mu, x_r, local_p);
                }
            }
        }

        // ---- Diagnostic streams and initial checkpoint -------------------
        // Profiles contain radial moments and metric fields; phase-space
        // files contain the L-integrated distribution used for visualization.
        const std::filesystem::path outdir(global_p.output_dir);
        if (is_root(mpi)) {
            std::filesystem::create_directories(outdir);
        }
        mpi_barrier(mpi);
        const std::filesystem::path restart_path = outdir / "restart_state.bin";
        if (is_root(mpi)) {
            write_metadata(outdir / "metadata.json", global_p, local_p, decomp, mpi);
        }
        {
            std::vector<double> g = phase_projection_global(f, L, Lq, local_p, mpi);
            if (is_root(mpi)) {
                write_phase_space(outdir / "phase_space_initial.csv", r, w, g, global_p);
            }
        }
        write_restart_state_distributed(restart_path, global_p, local_p, decomp, mpi, t, step, f);

        std::ofstream profiles;
        std::ofstream history;
        std::ofstream w_grid_history;
        std::ofstream phase_snapshots;
        if (is_root(mpi)) {
            profiles.open(outdir / "profiles.csv");
            if (!profiles) {
                throw std::runtime_error("Could not open profiles output file.");
            }
            profiles << "time,step,r,rho,p,j,m,mu,lambda,gap\n";
            if (!profiles) {
                throw std::runtime_error("Could not write profiles output header.");
            }
            flush_output_checked(profiles, "profiles.csv");
            history.open(outdir / "history.csv");
            if (!history) {
                throw std::runtime_error("Could not open history output file.");
            }
            history << "step,time,dt,matter_mass,max_rho,min_gap,"
                    << "last_outer_escape_est,last_w_low_escape_est,last_w_high_escape_est,"
                    << "cum_outer_escape_est,cum_w_low_escape_est,cum_w_high_escape_est,"
                    << "last_outer_cells,last_w_low_cells,last_w_high_cells,"
                    << "w_max,min_dw\n";
            history << std::setprecision(17);
            if (!history) {
                throw std::runtime_error("Could not write history output header.");
            }
            flush_output_checked(history, "history.csv");
            w_grid_history.open(outdir / "w_grid_history.csv");
            if (!w_grid_history) {
                throw std::runtime_error("Could not open w-grid history output file.");
            }
            w_grid_history
                << "step,time,old_w_max,new_w_max,active_w_min,active_w_max,min_dw,"
                << "mass_before,mass_after,rel_mass_change\n";
            w_grid_history << std::setprecision(17);
            if (!w_grid_history) {
                throw std::runtime_error("Could not write w-grid history output header.");
            }
            flush_output_checked(w_grid_history, "w_grid_history.csv");
            if (global_p.phase_every > 0) {
                phase_snapshots.open(outdir / "phase_space_snapshots.csv");
                if (!phase_snapshots) {
                    throw std::runtime_error("Could not open phase-space snapshots output file.");
                }
                phase_snapshots << "time,step,r_index,w_index,r,w,g\n";
                if (!phase_snapshots) {
                    throw std::runtime_error("Could not write phase-space snapshots output header.");
                }
                flush_output_checked(phase_snapshots, "phase_space_snapshots.csv");
            }
        }

        double last_dt = 0.0;
        EscapeStats last_escape;
        EscapeStats cumulative_escape;

        // ---- Einstein--Vlasov evolution ----------------------------------
        // Each cycle forms stress-energy moments, solves the spherical metric
        // constraints, chooses a characteristic CFL step, and pulls f back
        // along collisionless trajectories.  No collision operator is used.
        while (step <= global_p.max_steps) {
            const Moments moments = compute_global_moments(f, r, w, L, wq, Lq, local_p, mpi);
            const Metric metric = solve_metric(moments, r, global_p);
            const double matter_mass = metric.m.back() - global_p.central_mass;
            const double max_rho = *std::max_element(moments.rho.begin(), moments.rho.end());

            if (is_root(mpi) && (step % std::max(1, global_p.save_every) == 0 || t >= global_p.t_end)) {
                write_profile_rows(profiles, t, step, r, moments, metric, global_p);
                flush_output_checked(profiles, "profiles.csv");
            }
            if (global_p.phase_every > 0 && (step % global_p.phase_every == 0 || t >= global_p.t_end)) {
                std::vector<double> g = phase_projection_global(f, L, Lq, local_p, mpi);
                if (is_root(mpi)) {
                    write_phase_snapshot_rows(phase_snapshots, t, step, r, w, g, global_p);
                    flush_output_checked(phase_snapshots, "phase_space_snapshots.csv");
                }
            }
            if (is_root(mpi)) {
                history << step << ',' << t << ',' << last_dt << ',' << matter_mass << ','
                        << max_rho << ',' << metric.min_gap << ','
                        << last_escape.outer_mass << ',' << last_escape.w_low_mass << ','
                        << last_escape.w_high_mass << ','
                        << cumulative_escape.outer_mass << ',' << cumulative_escape.w_low_mass << ','
                        << cumulative_escape.w_high_mass << ','
                        << last_escape.outer_cells << ',' << last_escape.w_low_cells << ','
                        << last_escape.w_high_cells << ','
                        << global_p.w_max << ',' << min_dw << '\n';
                flush_output_checked(history, "history.csv");
            }

            if (t >= global_p.t_end) {
                break;
            }
            // The polar-areal chart becomes singular as 1-2m/r approaches
            // zero.  Stop at the declared gap while retaining the measured
            // value, separately from the floor used inside metric formulas.
            if (metric.min_gap <= global_p.gap_stop) {
                if (is_root(mpi)) {
                    std::cout << "Stopping: compactness gap reached " << metric.min_gap
                              << " at t=" << t << "\n";
                }
                break;
            }
            if (step >= global_p.max_steps) {
                if (is_root(mpi)) {
                    std::cout << "Stopping: reached max_steps=" << global_p.max_steps << ".\n";
                }
                break;
            }

            double dt = choose_dt(f, r, w, L, metric, radial_step, min_dw, local_p, mpi);
            if (t + dt > global_p.t_end) {
                dt = global_p.t_end - t;
            }
            if (!(dt > 0.0) || !std::isfinite(dt)) {
                throw std::runtime_error("Non-positive or non-finite timestep.");
            }

            // Boundary-loss estimates are diagnostics of truncated phase
            // space.  They are measured before advection and never fed back
            // as a physical source or sink.
            last_escape = estimate_escapes(f, r, w, L, rq, wq, Lq, metric, dt, local_p, mpi);
            cumulative_escape.outer_mass += last_escape.outer_mass;
            cumulative_escape.w_low_mass += last_escape.w_low_mass;
            cumulative_escape.w_high_mass += last_escape.w_high_mass;

            // Follow characteristics backward in the field of this state.
            // Only after the new f is available advance its physical clock.
            // The next loop then obtains a fresh metric from the new moments.
            f = advect(f, x_r, r, w, L, metric, dt, local_p);
            t += dt;
            last_dt = dt;
            ++step;

            adapt_w_grid_if_needed(
                f, w, wq, min_dw, r, L, rq, Lq, local_p, step, t,
                is_root(mpi) ? &w_grid_history : nullptr,
                mpi);
            global_p.w_max = local_p.w_max;

            if (global_p.restart_every > 0 && step % global_p.restart_every == 0) {
                write_restart_state_distributed(restart_path, global_p, local_p, decomp, mpi, t, step, f);
            }

            if (is_root(mpi) && step % std::max(1, 5 * global_p.save_every) == 0) {
                std::cout << "step=" << std::setw(6) << step
                          << " t=" << std::setw(11) << std::setprecision(6) << t
                          << " dt=" << std::scientific << std::setprecision(3) << dt
                          << " mass=" << matter_mass
                          << " max_rho=" << max_rho
                          << " min_gap=" << metric.min_gap << std::defaultfloat << "\n";
            }
        }

        // ---- Final restart and projected phase-space state ----------------
        write_restart_state_distributed(restart_path, global_p, local_p, decomp, mpi, t, step, f);
        {
            std::vector<double> g = phase_projection_global(f, L, Lq, local_p, mpi);
            if (is_root(mpi)) {
                write_phase_space(outdir / "phase_space_final.csv", r, w, g, global_p);
            }
        }
        if (is_root(mpi)) {
            std::cout << "Wrote C++ solver data to " << std::filesystem::absolute(outdir) << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "rank " << mpi.rank << " error: " << ex.what() << "\n";
        exit_code = 1;
#ifdef EV_USE_MPI
        if (mpi.enabled) {
            MPI_Abort(MPI_COMM_WORLD, exit_code);
        }
#endif
    }
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        MPI_Finalize();
    }
#endif
    return exit_code;
}
