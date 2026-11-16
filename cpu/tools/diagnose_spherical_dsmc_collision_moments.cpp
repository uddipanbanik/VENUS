
// Measure the continuous spherical DSMC elastic kernel on simple homogeneous
// velocity distributions. Radial counter-streaming beams and a tangential ring
// have analytic small-interval collision rates and second-moment derivatives.
// The tool averages finite random realizations at several intervals and also
// compares tangential-azimuth refresh schedules. Self-gravity and radial orbit
// transport are held fixed, isolating collision sampling and conservation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ev/config.hpp"
#include "ev/spherical_dsmc_solver.hpp"

namespace {

struct Moments {
    double radial = 0.0;
    double tangential_per_component = 0.0;
    double anisotropy = 0.0;
    double speed_squared = 0.0;
};

// The anisotropy is <vr^2>-<vt^2>/2. Its relaxation tests angular
// isotropization, while <vr^2+vt^2> checks elastic kinetic energy.
Moments moments(const std::vector<ev::SphericalDSMCParticle>& particles) {
    Moments result;
    for (const ev::SphericalDSMCParticle& particle : particles) {
        const double radial = particle.vr * particle.vr;
        const double tangential = particle.L * particle.L
            / (particle.r * particle.r);
        result.radial += radial;
        result.tangential_per_component += 0.5 * tangential;
        result.speed_squared += radial + tangential;
    }
    const double inverse =
        1.0 / static_cast<double>(particles.size());
    result.radial *= inverse;
    result.tangential_per_component *= inverse;
    result.speed_squared *= inverse;
    result.anisotropy =
        result.radial - result.tangential_per_component;
    return result;
}

struct EnsembleResult {
    double accepted_rate_per_particle = 0.0;
    double trial_rate_per_particle = 0.0;
    double radial_derivative = 0.0;
    double tangential_derivative = 0.0;
    double anisotropy_derivative = 0.0;
    double speed_squared_derivative = 0.0;
    double maximum_pair_momentum_error = 0.0;
    double maximum_pair_energy_error = 0.0;
};

EnsembleResult run_case(
    const std::string& name,
    std::size_t marker_count,
    int realizations,
    double dt,
    bool radial_streams) {
    // Compare the Monte-Carlo relaxation rate of two analytically simple
    // anisotropic velocity distributions with their exact angular averages.
    ev::Params p;
    p.threads = 1;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0;
    p.bp_dsmc_seed = UINT64_C(0x42);

    constexpr double radius = 1.0;
    constexpr double speed = 1.0;
    constexpr double density = 1.0;
    std::vector<ev::SphericalDSMCParticle> initial(marker_count);
    for (std::size_t i = 0; i < marker_count; ++i) {
        initial[i].r = radius;
        initial[i].vr =
            radial_streams ? ((i & 1U) == 0U ? speed : -speed) : 0.0;
        initial[i].L = radial_streams ? 0.0 : radius * speed;
        initial[i].id = static_cast<std::uint64_t>(i);
    }
    const Moments before = moments(initial);

    ev::SphericalDSMCShell shell;
    shell.begin = 0;
    shell.end = marker_count;
    shell.inner_edge = 0.5;
    shell.outer_edge = 1.5;
    shell.density = density;
    shell.maximum_speed = speed;
    shell.maximum_speed_index = 0;
    shell.majorant_u_sigma =
        ev::dsmc_u_sigma_majorant(2.0 * speed, p);

    EnsembleResult mean;
    for (int realization = 0; realization < realizations; ++realization) {
        std::vector<ev::SphericalDSMCParticle> particles = initial;
        ev::SphericalDSMCCollisionWorkspace::ThreadVelocityCache cache;
        const ev::DSMCCollisionStats stats =
            ev::spherical_dsmc_apply_shell_collisions_lazy(
                particles,
                shell,
                dt,
                p,
                UINT64_C(0x10000000)
                    + static_cast<std::uint64_t>(2 * realization),
                UINT64_C(0x20000000)
                    + static_cast<std::uint64_t>(2 * realization + 1),
                cache);
        const Moments after = moments(particles);
        const double normalization =
            1.0 / static_cast<double>(realizations);
        mean.accepted_rate_per_particle += normalization
            * static_cast<double>(stats.accepted_pairs)
            / (static_cast<double>(marker_count) * dt);
        mean.trial_rate_per_particle += normalization
            * static_cast<double>(stats.trial_pairs)
            / (static_cast<double>(marker_count) * dt);
        mean.radial_derivative += normalization
            * (after.radial - before.radial) / dt;
        mean.tangential_derivative += normalization
            * (after.tangential_per_component
               - before.tangential_per_component) / dt;
        mean.anisotropy_derivative += normalization
            * (after.anisotropy - before.anisotropy) / dt;
        mean.speed_squared_derivative += normalization
            * (after.speed_squared - before.speed_squared) / dt;
        mean.maximum_pair_momentum_error = std::max(
            mean.maximum_pair_momentum_error,
            stats.max_pair_momentum_error);
        mean.maximum_pair_energy_error = std::max(
            mean.maximum_pair_energy_error,
            stats.max_pair_kinetic_energy_error);
    }

    // These analytic angular averages apply to the unit-density, unit-speed
    // and unit constant-area setup above in the dt->0 limit. The three finite
    // intervals expose multiple-encounter effects on the measured derivative.
    const double analytic_accepted = radial_streams
        ? 0.5
        : 2.0 / ev::PI;
    const double analytic_radial = radial_streams
        ? -2.0 / 3.0
        : 8.0 / (9.0 * ev::PI);
    const double analytic_tangential = radial_streams
        ? 1.0 / 3.0
        : -4.0 / (9.0 * ev::PI);
    const double analytic_anisotropy = radial_streams
        ? -1.0
        : 4.0 / (3.0 * ev::PI);

    auto ratio = [](double measured, double analytic) {
        return measured / analytic;
    };
    std::cout << name
              << " dt=" << dt
              << " accepted_rate=" << mean.accepted_rate_per_particle
              << " analytic=" << analytic_accepted
              << " ratio=" << ratio(
                    mean.accepted_rate_per_particle, analytic_accepted)
              << " trial_rate=" << mean.trial_rate_per_particle
              << '\n';
    std::cout << "  d<vr^2>/dt=" << mean.radial_derivative
              << " analytic=" << analytic_radial
              << " ratio=" << ratio(mean.radial_derivative, analytic_radial)
              << '\n';
    std::cout << "  d<vt_component^2>/dt="
              << mean.tangential_derivative
              << " analytic=" << analytic_tangential
              << " ratio=" << ratio(
                    mean.tangential_derivative, analytic_tangential)
              << '\n';
    std::cout << "  dA/dt=" << mean.anisotropy_derivative
              << " analytic=" << analytic_anisotropy
              << " ratio=" << ratio(
                    mean.anisotropy_derivative, analytic_anisotropy)
              << " d<v^2>/dt=" << mean.speed_squared_derivative
              << '\n';
    std::cout << "  max_pair_dp=" << mean.maximum_pair_momentum_error
              << " max_pair_dK=" << mean.maximum_pair_energy_error
              << '\n';
    return mean;
}

// The spherical particle stores tangential speed, not its azimuth.
// Compare one finite collision call with repeated short calls from the
// same initial distribution, averaging their different random realizations.
void compare_lazy_azimuth_to_frequent_refresh(
    std::size_t marker_count,
    int realizations,
    bool radial_streams) {
    // The spherical marker stores tangential speed but not its azimuth.  This
    // test checks that drawing that missing angle lazily during collisions
    // agrees with frequently rebuilding the three-dimensional velocity.
    ev::Params p;
    p.threads = 1;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0;

    constexpr double radius = 1.0;
    constexpr double speed = 1.0;
    constexpr double density = 1.0;
    constexpr double total_time = 0.5;
    constexpr int substeps = 100;
    constexpr double substep_dt =
        total_time / static_cast<double>(substeps);
    std::vector<ev::SphericalDSMCParticle> initial(marker_count);
    for (std::size_t i = 0; i < marker_count; ++i) {
        initial[i].r = radius;
        initial[i].vr =
            radial_streams ? ((i & 1U) == 0U ? speed : -speed) : 0.0;
        initial[i].L = radial_streams ? 0.0 : radius * speed;
        initial[i].id = static_cast<std::uint64_t>(i);
    }
    ev::SphericalDSMCShell shell;
    shell.begin = 0;
    shell.end = marker_count;
    shell.density = density;
    shell.maximum_speed = speed;
    shell.maximum_speed_index = 0;
    shell.majorant_u_sigma =
        ev::dsmc_u_sigma_majorant(2.0 * speed, p);

    double one_call_anisotropy = 0.0;
    double refreshed_anisotropy = 0.0;
    double one_call_energy = 0.0;
    double refreshed_energy = 0.0;
    for (int realization = 0; realization < realizations; ++realization) {
        std::vector<ev::SphericalDSMCParticle> one_call = initial;
        std::vector<ev::SphericalDSMCParticle> refreshed = initial;
        ev::SphericalDSMCCollisionWorkspace::ThreadVelocityCache cache;
        ev::spherical_dsmc_apply_shell_collisions_lazy(
            one_call,
            shell,
            total_time,
            p,
            UINT64_C(0x30000000)
                + static_cast<std::uint64_t>(2 * realization),
            UINT64_C(0x40000000)
                + static_cast<std::uint64_t>(2 * realization + 1),
            cache);
        for (int step = 0; step < substeps; ++step) {
            double maximum_speed = 0.0;
            std::size_t maximum_index = 0;
            for (std::size_t i = 0; i < refreshed.size(); ++i) {
                const double marker_speed = std::hypot(
                    refreshed[i].vr,
                    refreshed[i].L / refreshed[i].r);
                if (marker_speed > maximum_speed) {
                    maximum_speed = marker_speed;
                    maximum_index = i;
                }
            }
            shell.maximum_speed = maximum_speed;
            shell.maximum_speed_index = maximum_index;
            shell.majorant_u_sigma =
                ev::dsmc_u_sigma_majorant(2.0 * maximum_speed, p);
            ev::spherical_dsmc_apply_shell_collisions_lazy(
                refreshed,
                shell,
                substep_dt,
                p,
                UINT64_C(0x50000000)
                    + static_cast<std::uint64_t>(
                        4096 * realization + 2 * step),
                UINT64_C(0x60000000)
                    + static_cast<std::uint64_t>(
                        4096 * realization + 2 * step + 1),
                cache);
        }
        const Moments one = moments(one_call);
        const Moments many = moments(refreshed);
        const double normalization =
            1.0 / static_cast<double>(realizations);
        one_call_anisotropy += normalization * one.anisotropy;
        refreshed_anisotropy += normalization * many.anisotropy;
        one_call_energy += normalization * one.speed_squared;
        refreshed_energy += normalization * many.speed_squared;
    }
    std::cout
        << (radial_streams ? "radial beams" : "tangential ring")
        << " finite-step azimuth check, rho*majorant*dt=1:"
        << " one_call_A=" << one_call_anisotropy
        << " refresh_every_0.005_A=" << refreshed_anisotropy
        << " relative_difference="
        << (one_call_anisotropy - refreshed_anisotropy)
            / std::max(
                {std::abs(one_call_anisotropy),
                 std::abs(refreshed_anisotropy),
                 1.0e-300})
        << " one_call_<v2>=" << one_call_energy
        << " refreshed_<v2>=" << refreshed_energy
        << '\n';
}

// Uniform sampling in r^3 represents constant density in a ball.
// The shell builder must include the physical 4pi r^2 dr volume once
// when converting represented mass to the collision density.
void check_spherical_volume_normalization() {
    // Verify that shell density contains the physical 4*pi*r^2 radial volume
    // exactly once before any collision rate is interpreted.
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0e-12;
    p.bp_dsmc_shell_particles = 4096;
    p.bp_dsmc_min_shell_particles = 32;
    constexpr std::size_t marker_count = 4096;
    constexpr double radius = 1.0;
    const double volume = (4.0 * ev::PI / 3.0)
        * radius * radius * radius;
    const double marker_mass =
        volume / static_cast<double>(marker_count);
    std::vector<ev::SphericalDSMCParticle> particles(marker_count);
    for (std::size_t i = 0; i < marker_count; ++i) {
        particles[i].r = radius * std::cbrt(
            static_cast<double>(i + 1)
            / static_cast<double>(marker_count));
        particles[i].vr = (i & 1U) == 0U ? 1.0 : -1.0;
        particles[i].id = static_cast<std::uint64_t>(i);
    }
    const std::vector<ev::SphericalDSMCShell> shells =
        ev::spherical_dsmc_build_shells(particles, marker_mass, p);
    std::cout << "uniform unit-density sphere: shells=" << shells.size()
              << " measured_density=" << shells.front().density
              << " expected_density=1 ratio="
              << shells.front().density
              << " (4*pi volume included once)"
              << '\n';
}

}  // namespace

int main() {
    // ---- Geometry normalization and elastic-kernel moment tests ----------
    std::cout << std::setprecision(12);
    check_spherical_volume_normalization();
    constexpr std::size_t marker_count = 500000;
    constexpr int realizations = 16;
    for (double dt : {0.02, 0.01, 0.005}) {
        run_case(
            "counter-streaming radial beams",
            marker_count,
            realizations,
            dt,
            true);
        run_case(
            "tangential ring",
            marker_count,
            realizations,
            dt,
            false);
    }
    // ---- Finite-interval sensitivity to tangential-azimuth refresh --------
    compare_lazy_azimuth_to_frequent_refresh(250000, 12, true);
    compare_lazy_azimuth_to_frequent_refresh(250000, 12, false);
    return 0;
}
