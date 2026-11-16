
// Time separate operations of the spherical DSMC solver from a saved state.
// Input is a restart and optional worker count. The physical parameters select
// the declared constant-area benchmark. Outputs separate gravity/timestep scans,
// sorting, shell construction and collision cost, with their work counters.
// Each measured step begins from a copy of the same checkpoint; persistent
// work arrays are warmed and reused. The source restart is left unchanged.

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "ev/config.hpp"
#include "ev/spherical_dsmc_solver.hpp"

namespace {

using Clock = std::chrono::steady_clock;

// Average repeated wall measurements of one operation. The surrounding
// benchmark decides which physical state and workspace are restored
// between repetitions, so timing includes only the stated work.
template <class Function>
double measure_seconds(int repetitions, Function&& function) {
    const auto start = Clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        function();
    }
    const auto stop = Clock::now();
    return std::chrono::duration<double>(stop - start).count()
        / static_cast<double>(repetitions);
}

ev::Params frank_parameters(int threads) {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.vp_universe = "static";
    p.vp_gravity = 1.0;
    p.vp_poisson_source = "total_density";
    p.vp_orbit_regularization = "auto";
    p.vp_orbit_energy_tolerance = 0.01;
    p.vp_orbit_lc_relative_tolerance = 1.0e-9;
    p.vp_orbit_lc_absolute_tolerance = 1.0e-12;
    p.vp_orbit_lc_max_steps = 2048;
    p.vp_orbit_lc_max_refinements = 20;
    p.bp_collision_operator = "spherical_dsmc";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 32.354198286197665;
    p.bp_dsmc_particles = 1000000;
    p.bp_dsmc_total_mass = 1.1252580933331495;
    p.bp_dsmc_shell_particles = 4096;
    p.bp_dsmc_min_shell_particles = 32;
    p.bp_dsmc_collision_epsilon = 1.0;
    p.bp_dsmc_gravity_eta = 0.01;
    p.bp_dsmc_seed = UINT64_C(5489);
    p.dt_max = 1.0;
    p.threads = threads;
    return p;
}

void print_timing(const char* label, double seconds) {
    std::cout << std::left << std::setw(32) << label
              << std::right << std::fixed << std::setprecision(9)
              << seconds << " s\n";
}

} // namespace

int main(int argc, char** argv) {
    // ---- Decode one production checkpoint and benchmark configuration ----
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0]
                  << " RESTART_STATE_BIN [THREADS]\n";
        return 2;
    }
    const int threads = argc == 3 ? std::stoi(argv[2]) : 32;
    ev::Params p = frank_parameters(threads);
    ev::spherical_dsmc_validate_paper_exact_configuration(p);

    std::vector<ev::SphericalDSMCParticle> particles;
    std::vector<double> remainders;
    const ev::SphericalDSMCRestartHeader header =
        ev::spherical_dsmc_read_restart(argv[1], particles, &remainders);
    const double particle_mass = header.particle_mass;

    // ---- Build adaptive collision shells and physical timestep limits ----
    ev::SphericalDSMCShellWorkspace shell_workspace;
    std::vector<ev::SphericalDSMCShell> shells;
    const double shell_seconds = measure_seconds(10, [&]() {
        shells = ev::spherical_dsmc_build_shells(
            particles, particle_mass, p, shell_workspace);
    });
    const double collision_dt =
        ev::spherical_dsmc_collision_timestep(shells, p);

    double gravity_dt = 0.0;
    const double gravity_timestep_seconds = measure_seconds(10, [&]() {
        gravity_dt = ev::spherical_dsmc_gravity_timestep(
            particles, particle_mass, p);
    });
    // The benchmark interval satisfies the configured maximum, gravitational
    // and collision restrictions of the restored physical state.
    const double dt = std::min({p.dt_max, gravity_dt, collision_dt});

    std::vector<double> outer_sum;
    std::vector<double> potential;
    const double potential_seconds = measure_seconds(10, [&]() {
        ev::spherical_dsmc_orbit_audit_specific_potential(
            particles, particle_mass, p, outer_sum, potential);
    });

    ev::SphericalDSMCReductionWorkspace reduction_workspace;
    ev::SphericalDSMCStateSums sums;
    const double reduction_seconds = measure_seconds(10, [&]() {
        sums = ev::spherical_dsmc_state_sums_parallel(
            particles, particle_mass, p, reduction_workspace);
    });

    // ---- Measure post-drift disorder and radius-sorting cost --------------
    // Gravity requires radius order because enclosed mass determines the
    // spherical force.  Inversion counts quantify how far the drift moved the
    // nearly sorted input away from that order.
    std::vector<ev::SphericalDSMCParticle> drifted = particles;
    const int nthreads = ev::effective_threads(
        p, static_cast<int>(drifted.size()));
    ev::parallel_for_chunks(
        0,
        static_cast<int>(drifted.size()),
        nthreads,
        [&](int lo, int hi, int) {
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                drifted[i].vr += 0.5 * dt
                    * ev::spherical_dsmc_acceleration(
                        drifted[i], i, particle_mass, p);
                drifted[i].r += dt * drifted[i].vr;
                if (drifted[i].r < 0.0) {
                    drifted[i].r = -drifted[i].r;
                    drifted[i].vr = -drifted[i].vr;
                }
                drifted[i].r = std::max(drifted[i].r, 1.0e-300);
            }
        });
    // Count nearby radius-order inversions after a provisional drift.
    // Spherical enclosed-mass gravity requires sorted radius, so this
    // measures the actual ordering work before timing the sort algorithms.
    std::size_t adjacent_inversions = 0;
    std::size_t gap4_inversions = 0;
    std::size_t gap8_inversions = 0;
    std::size_t gap16_inversions = 0;
    std::size_t gap32_inversions = 0;
    for (std::size_t i = 1; i < drifted.size(); ++i) {
        adjacent_inversions += drifted[i].r < drifted[i - 1].r ? 1U : 0U;
        gap4_inversions += i >= 4 && drifted[i].r < drifted[i - 4].r
            ? 1U : 0U;
        gap8_inversions += i >= 8 && drifted[i].r < drifted[i - 8].r
            ? 1U : 0U;
        gap16_inversions += i >= 16 && drifted[i].r < drifted[i - 16].r
            ? 1U : 0U;
        gap32_inversions += i >= 32 && drifted[i].r < drifted[i - 32].r
            ? 1U : 0U;
    }
    std::vector<ev::SphericalDSMCParticle> parallel_drifted = drifted;
    const auto sort_start = Clock::now();
    ev::spherical_dsmc_insertion_sort_particles(drifted);
    const double insertion_sort_seconds =
        std::chrono::duration<double>(Clock::now() - sort_start).count();
    ev::SphericalDSMCSortWorkspace sort_workspace;
    const auto parallel_sort_start = Clock::now();
    ev::spherical_dsmc_sort_particles_parallel(
        parallel_drifted, p, sort_workspace, false);
    const double parallel_insertion_sort_seconds =
        std::chrono::duration<double>(Clock::now() - parallel_sort_start).count();
    if (parallel_drifted.size() != drifted.size()) {
        throw std::runtime_error("Parallel insertion-sort size changed");
    }
    // Check particle properties and identity after parallel sorting against
    // the serial sorted order. The comparison tests the sorting operation
    // before using its time in a speed assessment.
    for (std::size_t i = 0; i < drifted.size(); ++i) {
        if (parallel_drifted[i].r != drifted[i].r
            || parallel_drifted[i].vr != drifted[i].vr
            || parallel_drifted[i].L != drifted[i].L
            || parallel_drifted[i].id != drifted[i].id) {
            throw std::runtime_error(
                "Parallel insertion-sort result differs from literal serial order");
        }
    }

    // ---- Time gravity/orbit and collision operators independently --------
    ev::SphericalDSMCGravityWorkspace gravity_workspace;
    ev::SphericalDSMCDiagnostics gravity_diagnostics;
    std::vector<ev::SphericalDSMCParticle> gravity_particles = particles;
    // Prime persistent workspaces before timing.  Production reuses these
    // allocations, so charging their first resize to every step materially
    // overstates the steady-state cost for a million markers.
    gravity_diagnostics = ev::spherical_dsmc_gravity_step(
        gravity_particles,
        particle_mass,
        dt,
        p,
        gravity_workspace);
    double gravity_step_seconds = 0.0;
    // Reset particles from the identical checkpoint for every gravity timing
    // while reusing allocated workspaces, separating steady-state operation
    // cost from first-allocation overhead.
    constexpr int gravity_repetitions = 5;
    for (int repetition = 0; repetition < gravity_repetitions; ++repetition) {
        gravity_particles = particles;
        const auto gravity_start = Clock::now();
        gravity_diagnostics = ev::spherical_dsmc_gravity_step(
            gravity_particles,
            particle_mass,
            dt,
            p,
            gravity_workspace);
        gravity_step_seconds += std::chrono::duration<double>(
            Clock::now() - gravity_start).count();
    }
    gravity_step_seconds /= static_cast<double>(gravity_repetitions);

    ev::SphericalDSMCCollisionWorkspace collision_workspace;
    collision_workspace.paper_remainder_accumulator_by_id = remainders;
    ev::SphericalDSMCDiagnostics collision_diagnostics;
    std::vector<ev::SphericalDSMCParticle> collision_particles = particles;
    collision_diagnostics = ev::spherical_dsmc_apply_collisions(
        collision_particles,
        shells,
        particle_mass,
        dt,
        static_cast<int>(header.step),
        p,
        collision_workspace);
    double collision_seconds = 0.0;
    // Restore both particles and the saved per-ID stochastic remainder
    // state before each collision timing. Reusing only the particles would
    // benchmark a different carried event process.
    constexpr int collision_repetitions = 5;
    for (int repetition = 0; repetition < collision_repetitions; ++repetition) {
        collision_particles = particles;
        collision_workspace.paper_remainder_accumulator_by_id = remainders;
        const auto collision_start = Clock::now();
        collision_diagnostics = ev::spherical_dsmc_apply_collisions(
            collision_particles,
            shells,
            particle_mass,
            dt,
            static_cast<int>(header.step),
            p,
            collision_workspace);
        collision_seconds += std::chrono::duration<double>(
            Clock::now() - collision_start).count();
    }
    collision_seconds /= static_cast<double>(collision_repetitions);

    // ---- Time the complete output-free production step -------------------
    std::vector<ev::SphericalDSMCParticle> full_particles = particles;
    ev::SphericalDSMCGravityWorkspace full_gravity_workspace;
    ev::SphericalDSMCShellWorkspace full_shell_workspace;
    ev::SphericalDSMCReductionWorkspace full_reduction_workspace;
    ev::SphericalDSMCCollisionWorkspace full_collision_workspace;
    full_collision_workspace.paper_remainder_accumulator_by_id = remainders;
    ev::SphericalDSMCDiagnostics full_gravity;
    ev::SphericalDSMCDiagnostics full_collision;
    // Warm the complete steady-state path once, then time repeated steps from
    // the identical checkpoint while retaining all workspaces.
    full_gravity = ev::spherical_dsmc_gravity_step(
        full_particles,
        particle_mass,
        dt,
        p,
        full_gravity_workspace);
    (void) ev::spherical_dsmc_state_sums_parallel(
        full_particles,
        particle_mass,
        p,
        full_reduction_workspace);
    const std::vector<ev::SphericalDSMCShell> full_shells =
        ev::spherical_dsmc_build_shells(
            full_particles,
            particle_mass,
            p,
            full_shell_workspace);
    full_collision = ev::spherical_dsmc_apply_collisions(
        full_particles,
        full_shells,
        particle_mass,
        dt,
        static_cast<int>(header.step),
        p,
        full_collision_workspace);
    (void) ev::spherical_dsmc_collision_timestep_after_collisions(
        full_shells, full_collision_workspace, p);
    double full_step_seconds = 0.0;
    // Time the complete output-free sequence from the same saved state:
    // gravity, state budget, rebuilt physical shells, collisions and the
    // postcollision timestep. This complements timings of isolated pieces.
    constexpr int full_repetitions = 5;
    for (int repetition = 0; repetition < full_repetitions; ++repetition) {
        full_particles = particles;
        full_collision_workspace.paper_remainder_accumulator_by_id = remainders;
        const auto full_start = Clock::now();
        full_gravity = ev::spherical_dsmc_gravity_step(
            full_particles,
            particle_mass,
            dt,
            p,
            full_gravity_workspace);
        (void) ev::spherical_dsmc_state_sums_parallel(
            full_particles,
            particle_mass,
            p,
            full_reduction_workspace);
        const std::vector<ev::SphericalDSMCShell> repeated_shells =
            ev::spherical_dsmc_build_shells(
                full_particles,
                particle_mass,
                p,
                full_shell_workspace);
        full_collision = ev::spherical_dsmc_apply_collisions(
            full_particles,
            repeated_shells,
            particle_mass,
            dt,
            static_cast<int>(header.step),
            p,
            full_collision_workspace);
        (void) ev::spherical_dsmc_collision_timestep_after_collisions(
            repeated_shells, full_collision_workspace, p);
        full_step_seconds += std::chrono::duration<double>(
            Clock::now() - full_start).count();
    }
    full_step_seconds /= static_cast<double>(full_repetitions);

    // ---- Report timings together with physics-work counters --------------
    std::cout << std::setprecision(17)
              << "checkpoint_time=" << header.time
              << " checkpoint_step=" << header.step
              << " particles=" << particles.size()
              << " shells=" << shells.size()
              << " dt=" << dt
              << " gravity_dt=" << gravity_dt
              << " collision_dt=" << collision_dt << '\n';
    print_timing("gravity timestep scan", gravity_timestep_seconds);
    print_timing("orbit-audit potential scan", potential_seconds);
    print_timing("literal insertion after drift", insertion_sort_seconds);
    print_timing("parallel insertion + merge", parallel_insertion_sort_seconds);
    print_timing("full gravity + orbit audit", gravity_step_seconds);
    print_timing("state/energy reduction", reduction_seconds);
    print_timing("Frank cursor shell build", shell_seconds);
    print_timing("paper-exact collisions", collision_seconds);
    print_timing("full output-free step", full_step_seconds);
    std::cout << "adjacent_inversions=" << adjacent_inversions
              << " gap4_inversions=" << gap4_inversions
              << " gap8_inversions=" << gap8_inversions
              << " gap16_inversions=" << gap16_inversions
              << " gap32_inversions=" << gap32_inversions
              << " gravity_lc_retries=" << gravity_diagnostics.lc_retries
              << " gravity_lc_unresolved=" << gravity_diagnostics.lc_unresolved
              << " collision_trials=" << collision_diagnostics.collision_trials
              << " collision_accepted="
              << collision_diagnostics.collision_accepted
              << " full_lc_retries=" << full_gravity.lc_retries
              << " full_collision_trials=" << full_collision.collision_trials
              << " checksum=" << sums.energy.total << '\n';
    return gravity_diagnostics.lc_unresolved == 0
            && full_gravity.lc_unresolved == 0
        ? 0
        : 3;
}
