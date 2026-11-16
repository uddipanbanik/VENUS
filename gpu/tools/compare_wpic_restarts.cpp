// Host-only CPU/GPU checkpoint comparator.  Inputs are two compatible WPIC
// restart paths plus optional absolute/relative tolerances.  It validates the
// discrete clocks/RNG metadata and canonical marker sequence, then reports
// bitwise mismatches and tolerance failures for every scalar field to stdout.
// Files are read but never modified; exit status is 0 for agreement, 1 for a
// numeric/discrete mismatch, and 2 for usage or malformed input.  Meaningful
// comparison assumes both checkpoints represent the same physical time and
// preserve the canonical marker-ID ordering established by production I/O.

#include "venus_gpu/restart.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

// Track exact binary64 agreement separately from the requested numerical
// tolerance. Report the particle ID carrying the largest absolute difference.
struct Metric {
    std::uint64_t bitwise_mismatches = 0;
    std::uint64_t tolerance_failures = 0;
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    std::uint64_t maximum_absolute_id = 0;
};

void observe(
    Metric& metric,
    double cpu,
    double gpu,
    std::uint64_t id,
    double absolute_tolerance,
    double relative_tolerance) {
    if (std::memcmp(&cpu, &gpu, sizeof(double)) != 0) {
        ++metric.bitwise_mismatches;
    }
    // Positive infinity is the restart sentinel for an as-yet-unmeasured
    // event timescale.  Equal sentinels agree; a finite/infinite pair must be
    // a tolerance failure rather than falling through inf/inf -> NaN below.
    if (!std::isfinite(cpu) || !std::isfinite(gpu)) {
        if (cpu == gpu) {
            return;
        }
        metric.maximum_absolute = std::numeric_limits<double>::infinity();
        metric.maximum_relative = std::numeric_limits<double>::infinity();
        metric.maximum_absolute_id = id;
        ++metric.tolerance_failures;
        return;
    }
    // Apply abs_tol + rel_tol*max(abs(a),abs(b)) to finite values. The
    // symmetric scale treats the two solver outputs on equal terms.
    const double absolute = std::abs(gpu - cpu);
    const double scale = std::max(std::abs(cpu), std::abs(gpu));
    const double relative = scale > 0.0 ? absolute / scale : 0.0;
    if (absolute > metric.maximum_absolute) {
        metric.maximum_absolute = absolute;
        metric.maximum_absolute_id = id;
    }
    metric.maximum_relative = std::max(metric.maximum_relative, relative);
    if (absolute > absolute_tolerance + relative_tolerance * scale) {
        ++metric.tolerance_failures;
    }
}

void print_metric(const char* name, const Metric& metric) {
    std::cout << std::left << std::setw(30) << name
              << " bitwise_mismatches=" << metric.bitwise_mismatches
              << " tolerance_failures=" << metric.tolerance_failures
              << " max_abs=" << metric.maximum_absolute
              << " max_rel=" << metric.maximum_relative
              << " max_abs_marker_id=" << metric.maximum_absolute_id << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 5) {
            std::cerr
                << "usage: compare_wpic_restarts CPU GPU [abs_tol [rel_tol]]\n";
            return 2;
        }
        const double absolute_tolerance = argc >= 4
            ? std::stod(argv[3]) : 5.0e-14;
        const double relative_tolerance = argc >= 5
            ? std::stod(argv[4]) : 5.0e-12;
        if (!(absolute_tolerance >= 0.0)
            || !std::isfinite(absolute_tolerance)
            || !(relative_tolerance >= 0.0)
            || !std::isfinite(relative_tolerance)) {
            throw std::runtime_error(
                "Comparison tolerances must be finite and non-negative.");
        }
        const venus_gpu::RestartState cpu = venus_gpu::read_restart(argv[1]);
        const venus_gpu::RestartState gpu = venus_gpu::read_restart(argv[2]);
        std::cout << std::setprecision(17)
                  << "absolute_tolerance=" << absolute_tolerance
                  << " relative_tolerance=" << relative_tolerance << '\n'
                  << "cpu_time=" << cpu.time << " gpu_time=" << gpu.time
                  << " cpu_step=" << cpu.step << " gpu_step=" << gpu.step
                  << " cpu_markers=" << cpu.markers.size()
                  << " gpu_markers=" << gpu.markers.size() << '\n';
        // Population size, steps, seeds, and counters are discrete identity
        // checks; a floating-point tolerance has no meaning for them.
        std::uint64_t discrete_metadata_failures = 0;
        discrete_metadata_failures +=
            cpu.markers.size() != gpu.markers.size();
        discrete_metadata_failures += cpu.step != gpu.step;
        discrete_metadata_failures +=
            cpu.initialization_seed != gpu.initialization_seed;
        discrete_metadata_failures +=
            cpu.initialization_counter != gpu.initialization_counter;
        discrete_metadata_failures +=
            cpu.collision_sampler.seed != gpu.collision_sampler.seed;
        discrete_metadata_failures +=
            cpu.collision_sampler.next_counter
            != gpu.collision_sampler.next_counter;

        // Compare the accepted epoch and carried operator clocks as well as
        // particle payloads, since they determine a future continuation.
        Metric time;
        Metric potential_age;
        Metric collision_age;
        Metric collision_timescale;
        Metric potential_timescale;
        observe(time, cpu.time, gpu.time, 0, absolute_tolerance,
                relative_tolerance);
        observe(potential_age, cpu.potential_event_age,
                gpu.potential_event_age, 0, absolute_tolerance,
                relative_tolerance);
        observe(collision_age, cpu.collision_event_age,
                gpu.collision_event_age, 0, absolute_tolerance,
                relative_tolerance);
        observe(collision_timescale, cpu.collision_timescale_estimate,
                gpu.collision_timescale_estimate, 0, absolute_tolerance,
                relative_tolerance);
        observe(potential_timescale, cpu.potential_timescale_estimate,
                gpu.potential_timescale_estimate, 0, absolute_tolerance,
                relative_tolerance);

        Metric mass;
        Metric radius;
        Metric radial_velocity;
        Metric angular_momentum;
        Metric internal_energy;
        Metric internal_dispersion;
        Metric inner_boundary_speed;
        Metric remaining_inner_time;
        std::uint64_t discrete_marker_failures = 0;
        // Compare the stored sequence directly. An ID/state mismatch marks
        // a different particle correspondence before numerical fields are used.
        const std::size_t common_markers = std::min(
            cpu.markers.size(), gpu.markers.size());
        for (std::size_t i = 0; i < common_markers; ++i) {
            const auto& a = cpu.markers[i];
            const auto& b = gpu.markers[i];
            if (a.id != b.id || a.source_cell != b.source_cell
                || a.state != b.state) {
                ++discrete_marker_failures;
                continue;
            }
            observe(mass, a.mass, b.mass, a.id, absolute_tolerance,
                    relative_tolerance);
            observe(radius, a.r, b.r, a.id, absolute_tolerance,
                    relative_tolerance);
            observe(radial_velocity, a.w, b.w, a.id, absolute_tolerance,
                    relative_tolerance);
            observe(angular_momentum, a.eta, b.eta, a.id,
                    absolute_tolerance, relative_tolerance);
            observe(internal_energy, a.internal_energy, b.internal_energy,
                    a.id, absolute_tolerance, relative_tolerance);
            observe(internal_dispersion, a.internal_velocity_dispersion,
                    b.internal_velocity_dispersion, a.id,
                    absolute_tolerance, relative_tolerance);
            observe(inner_boundary_speed, a.inner_boundary_speed,
                    b.inner_boundary_speed, a.id,
                    absolute_tolerance, relative_tolerance);
            observe(remaining_inner_time, a.remaining_inner_time,
                    b.remaining_inner_time, a.id,
                    absolute_tolerance, relative_tolerance);
        }

        print_metric("time", time);
        print_metric("potential_event_age", potential_age);
        print_metric("collision_event_age", collision_age);
        print_metric("collision_timescale", collision_timescale);
        print_metric("potential_timescale", potential_timescale);
        print_metric("marker.mass", mass);
        print_metric("marker.r", radius);
        print_metric("marker.w", radial_velocity);
        print_metric("marker.eta", angular_momentum);
        print_metric("marker.internal_energy", internal_energy);
        print_metric("marker.internal_dispersion", internal_dispersion);
        print_metric("marker.inner_boundary_speed", inner_boundary_speed);
        print_metric("marker.remaining_inner_time", remaining_inner_time);
        // Completion requires every discrete check and every scalar tolerance
        // check to pass. Malformed input has its separate exception status.
        const std::uint64_t numeric_failures =
            time.tolerance_failures + potential_age.tolerance_failures
            + collision_age.tolerance_failures
            + collision_timescale.tolerance_failures
            + potential_timescale.tolerance_failures
            + mass.tolerance_failures + radius.tolerance_failures
            + radial_velocity.tolerance_failures
            + angular_momentum.tolerance_failures
            + internal_energy.tolerance_failures
            + internal_dispersion.tolerance_failures
            + inner_boundary_speed.tolerance_failures
            + remaining_inner_time.tolerance_failures;
        std::cout << "discrete_metadata_failures="
                  << discrete_metadata_failures
                  << " discrete_marker_failures=" << discrete_marker_failures
                  << " numeric_tolerance_failures=" << numeric_failures
                  << '\n';
        return discrete_metadata_failures == 0
                && discrete_marker_failures == 0 && numeric_failures == 0
            ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "compare_wpic_restarts: " << error.what() << '\n';
        return 2;
    }
}
