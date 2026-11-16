#pragma once

// Elastic Boltzmann scattering of whole particles in one spherical shell.
// Inputs are three-dimensional velocities, positive physical particle masses,
// shell volume, elapsed time, cross-section parameters and a random seed. The
// no-time-counter sampler returns changed velocities and pair/shell conservation
// diagnostics. Velocities use the caller's common physical units; sigma_ij is an
// area and u*sigma_ij/volume is a pair rate. The equal-mass material sigma/m law
// and the unequal physical-mass closure are specified separately below.
// Radial transport, shell refinement and gravity are supplied by the solver.

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ev/finite_angle_boltzmann.hpp"

namespace ev {

struct DSMCVelocity {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct DSMCCollisionStats {
    std::uint64_t trial_pairs = 0;
    std::uint64_t accepted_pairs = 0;
    double majorant_u_sigma = 0.0;
    double represented_event_density = 0.0;
    double momentum_error_x = 0.0;
    double momentum_error_y = 0.0;
    double momentum_error_z = 0.0;
    double kinetic_energy_error = 0.0;
    double max_pair_momentum_error = 0.0;
    double max_pair_kinetic_energy_error = 0.0;
};

class DSMCSplitMix64 {
public:
    explicit DSMCSplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next_u64() {
        std::uint64_t z = (state_ += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }

    double uniform_open() {
        // Clamp the top midpoint, which rounds to one in double precision.
        const double value = (static_cast<double>(next_u64() >> 11) + 0.5)
            * (1.0 / 9007199254740992.0);
        return std::min(value, 0x1.fffffffffffffp-1);
    }

    // Reject the incomplete remainder of the 64-bit integer range before reducing
    // modulo n. This gives every particle index the same proposal probability.
    std::uint64_t uniform_index(std::uint64_t n) {
        if (n == 0) {
            throw std::runtime_error("DSMC uniform_index called with an empty range.");
        }
        const std::uint64_t threshold = static_cast<std::uint64_t>(-n) % n;
        for (;;) {
            const std::uint64_t value = next_u64();
            if (value >= threshold) {
                return value % n;
            }
        }
    }

private:
    std::uint64_t state_;
};

inline double dsmc_splitmix_uniform_open_at(
    std::uint64_t seed,
    std::uint64_t index) {
    std::uint64_t z = seed
        + (index + UINT64_C(1)) * UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    const double value = (static_cast<double>(z >> 11) + 0.5)
        * (1.0 / 9007199254740992.0);
    return std::min(value, 0x1.fffffffffffffp-1);
}

inline BPFAVec3 dsmc_to_vec3(const DSMCVelocity& v) {
    return {v.x, v.y, v.z};
}

inline DSMCVelocity dsmc_from_vec3(BPFAVec3 v) {
    return {v.x, v.y, v.z};
}

inline double dsmc_speed(const DSMCVelocity& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline double dsmc_kinetic_energy(const DSMCVelocity& v) {
    return 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
}

// The proposal rate needs the maximum of u*sigma(u) over the allowed speeds.
// Check its analytic interior extremum as well as the endpoint for each law.
inline double dsmc_u_sigma_majorant(double u_max, const Params& p) {
    if (!(u_max > 0.0)) {
        return 0.0;
    }

    double candidate = u_max;
    if (p.bp_cross_section == "powerlaw" && p.bp_sigma_n > 1.0
        && p.bp_sigma_u0 > 0.0) {
        candidate = std::min(
            u_max,
            p.bp_sigma_u0 / std::sqrt(p.bp_sigma_n - 1.0));
    } else if (p.bp_cross_section == "softened_gravity") {
        const double eps = p.bp_subhalo_radius;
        const double mass = p.bp_subhalo_mass;
        const double sigma_sat =
            (p.bp_sigma0 > 0.0) ? p.bp_sigma0 : (2.0 * PI * eps * eps);
        const double amplitude = 4.0 * PI * p.vp_gravity * p.vp_gravity
            * mass * mass * std::max(p.bp_coulomb_log, 0.0);
        if (sigma_sat > 0.0 && amplitude > 0.0) {
            candidate = std::min(
                u_max,
                std::pow(amplitude / (3.0 * sigma_sat), 0.25));
        }
    } else if (p.bp_cross_section == "yukawa_born"
               && p.bp_yukawa_mchi > 0.0) {
        candidate = std::min(
            u_max,
            p.bp_yukawa_mphi / p.bp_yukawa_mchi);
    }

    const double endpoint = u_max * bp_fa_total_cross_section(u_max, p);
    const double interior = candidate * bp_fa_total_cross_section(candidate, p);
    const double majorant = std::max(endpoint, interior);
    return majorant * (1.0 + 128.0 * std::numeric_limits<double>::epsilon());
}

/// Physical elastic cross section of a pair with masses `first_mass` and
/// `second_mass`.
///
/// A material input sigma/m uniquely fixes only the equal-mass area.  The
/// maintained whole-object closure
/// sigma_ij=(m_i+m_j)(sigma/m)/2 is symmetric, homogeneous under common
/// refinement, and exactly recovers that equal-mass frequency, but it is not
/// a microscopic mixed-species cross section derivable from sigma/m alone.
/// Production material-SIDM benchmarks therefore initialize equal masses.
/// Unequal physical species require their mixed cross section to be specified
/// as additional physics; unequal numerical weights require a weighted DSMC
/// algorithm and must not be reinterpreted as molecular masses.
///
/// The softened-gravity input is already a physical area calibrated at
/// `bp_subhalo_mass`; its focusing mass and homologous softening area are
/// rescaled with both encounter masses.
inline double dsmc_pair_total_cross_section(
    double relative_speed,
    double first_mass,
    double second_mass,
    const Params& p) {
    if (!(first_mass > 0.0) || !(second_mass > 0.0)
        || !std::isfinite(first_mass) || !std::isfinite(second_mass)) {
        throw std::runtime_error(
            "DSMC pair masses must be positive and finite.");
    }
    if (p.bp_cross_section != "softened_gravity") {
        return 0.5 * (first_mass + second_mass)
            * bp_fa_total_cross_section(relative_speed, p);
    }

    const double reference_mass = p.bp_subhalo_mass;
    const double reference_radius = p.bp_subhalo_radius;
    if (!(reference_mass > 0.0) || !(reference_radius > 0.0)) {
        return 0.0;
    }
    // Constant internal density gives radius proportional to mass^(1/3). The
    // softened encounter area and focusing amplitude then use both physical masses.
    const double first_radius = reference_radius
        * std::cbrt(first_mass / reference_mass);
    const double second_radius = reference_radius
        * std::cbrt(second_mass / reference_mass);
    const double pair_radius = 0.5 * (first_radius + second_radius);
    const double radius_ratio = pair_radius / reference_radius;
    const double reference_saturation =
        p.bp_sigma0 > 0.0
        ? p.bp_sigma0
        : 2.0 * PI * reference_radius * reference_radius;
    const double saturation =
        reference_saturation * radius_ratio * radius_ratio;
    const double reference_high_amplitude =
        4.0 * PI * p.vp_gravity * p.vp_gravity
        * reference_mass * reference_mass
        * std::max(p.bp_coulomb_log, 0.0);
    const double total_mass_ratio =
        (first_mass + second_mass) / (2.0 * reference_mass);
    const double high_amplitude =
        reference_high_amplitude * total_mass_ratio * total_mass_ratio;
    const double speed = std::max(std::abs(relative_speed), 1.0e-14);
    const double high = high_amplitude / std::pow(speed, 4);
    if (!(saturation > 0.0) || !(high > 0.0)) {
        return 0.0;
    }
    return p.bp_collision_strength / (1.0 / saturation + 1.0 / high);
}

/// Rigorous pair-kernel majorant for any two masses not exceeding
/// `maximum_mass`.
inline double dsmc_pair_u_sigma_majorant(
    double u_max,
    double maximum_mass,
    const Params& p) {
    if (!(u_max > 0.0) || !(maximum_mass > 0.0)) {
        return 0.0;
    }
    if (p.bp_cross_section != "softened_gravity") {
        return maximum_mass * dsmc_u_sigma_majorant(u_max, p);
    }
    const double reference_mass = p.bp_subhalo_mass;
    const double reference_radius = p.bp_subhalo_radius;
    if (!(reference_mass > 0.0) || !(reference_radius > 0.0)) {
        return 0.0;
    }
    const double pair_radius = reference_radius
        * std::cbrt(maximum_mass / reference_mass);
    const double radius_ratio = pair_radius / reference_radius;
    const double reference_saturation =
        p.bp_sigma0 > 0.0
        ? p.bp_sigma0
        : 2.0 * PI * reference_radius * reference_radius;
    const double saturation =
        reference_saturation * radius_ratio * radius_ratio;
    const double reference_high_amplitude =
        4.0 * PI * p.vp_gravity * p.vp_gravity
        * reference_mass * reference_mass
        * std::max(p.bp_coulomb_log, 0.0);
    const double high_amplitude = reference_high_amplitude
        * (maximum_mass / reference_mass)
        * (maximum_mass / reference_mass);
    double candidate = u_max;
    if (saturation > 0.0 && high_amplitude > 0.0) {
        candidate = std::min(
            u_max,
            std::pow(high_amplitude / (3.0 * saturation), 0.25));
    }
    const double endpoint = u_max * dsmc_pair_total_cross_section(
        u_max, maximum_mass, maximum_mass, p);
    const double interior = candidate * dsmc_pair_total_cross_section(
        candidate, maximum_mass, maximum_mass, p);
    return std::max(endpoint, interior)
        * (1.0 + 128.0 * std::numeric_limits<double>::epsilon());
}

inline double dsmc_sample_scatter_mu(
    double relative_speed,
    const Params& p,
    DSMCSplitMix64& rng) {
    if (bp_fa_kernel_name(p) != "yukawa_born") {
        return 2.0 * rng.uniform_open() - 1.0;
    }

    return bp_fa_yukawa_born_mu_from_unit(relative_speed, p, rng.uniform_open());
}

// The four entries are the three momentum components and kinetic energy.
// This overload has unit masses; the next overload includes each physical mass.
inline std::array<double, 4> dsmc_marker_invariants(
    const std::vector<DSMCVelocity>& velocities) {
    std::array<double, 4> result{0.0, 0.0, 0.0, 0.0};
    for (const DSMCVelocity& v : velocities) {
        result[0] += v.x;
        result[1] += v.y;
        result[2] += v.z;
        result[3] += dsmc_kinetic_energy(v);
    }
    return result;
}

inline std::array<double, 4> dsmc_marker_invariants(
    const std::vector<DSMCVelocity>& velocities,
    const std::vector<double>& masses) {
    if (velocities.size() != masses.size()) {
        throw std::runtime_error(
            "DSMC velocity and mass arrays have different sizes.");
    }
    std::array<double, 4> result{0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        const double mass = masses[i];
        if (!(mass > 0.0) || !std::isfinite(mass)) {
            throw std::runtime_error(
                "DSMC marker masses must be positive and finite.");
        }
        const DSMCVelocity& v = velocities[i];
        result[0] += mass * v.x;
        result[1] += mass * v.y;
        result[2] += mass * v.z;
        result[3] += mass * dsmc_kinetic_energy(v);
    }
    return result;
}

struct DSMCMarkerSummary {
    std::array<double, 4> invariants{0.0, 0.0, 0.0, 0.0};
    double maximum_speed = 0.0;
};

inline DSMCMarkerSummary dsmc_marker_summary(
    const std::vector<DSMCVelocity>& velocities) {
    DSMCMarkerSummary summary;
    for (const DSMCVelocity& v : velocities) {
        summary.invariants[0] += v.x;
        summary.invariants[1] += v.y;
        summary.invariants[2] += v.z;
        const double speed_squared =
            v.x * v.x + v.y * v.y + v.z * v.z;
        summary.invariants[3] += 0.5 * speed_squared;
        summary.maximum_speed = std::max(
            summary.maximum_speed,
            std::sqrt(speed_squared));
    }
    return summary;
}

inline DSMCMarkerSummary dsmc_marker_summary(
    const std::vector<DSMCVelocity>& velocities,
    const std::vector<double>& masses) {
    DSMCMarkerSummary summary;
    summary.invariants = dsmc_marker_invariants(velocities, masses);
    for (const DSMCVelocity& v : velocities) {
        summary.maximum_speed = std::max(
            summary.maximum_speed,
            dsmc_speed(v));
    }
    return summary;
}

/// Applies Bird's NTC process to complete markers of arbitrary positive mass.
///
/// `volume` is the physical shell volume.  Unordered pairs are proposed
/// uniformly, with trial rate C(N,2) K_max / volume.  The actual pair kernel is
/// K_ij=u sigma_ij.  Elastic scattering leaves both masses unchanged.
inline DSMCCollisionStats apply_dsmc_ntc_unequal_mass(
    std::vector<DSMCVelocity>& velocities,
    const std::vector<double>& masses,
    double volume,
    double dt,
    const Params& p,
    std::uint64_t seed) {
    DSMCCollisionStats stats;
    const std::size_t marker_count = velocities.size();
    if (masses.size() != marker_count) {
        throw std::runtime_error(
            "DSMC velocity and mass arrays have different sizes.");
    }
    if (!(volume > 0.0) || !std::isfinite(volume)
        || !(dt >= 0.0) || !std::isfinite(dt)) {
        throw std::runtime_error("Invalid unequal-mass DSMC shell request.");
    }
    double maximum_mass = 0.0;
    for (double mass : masses) {
        if (!(mass > 0.0) || !std::isfinite(mass)) {
            throw std::runtime_error(
                "DSMC marker masses must be positive and finite.");
        }
        maximum_mass = std::max(maximum_mass, mass);
    }
    if (marker_count < 2 || dt == 0.0
        || p.bp_collision_strength == 0.0) {
        return stats;
    }

    const DSMCMarkerSummary before =
        dsmc_marker_summary(velocities, masses);
    double speed_bound = before.maximum_speed;
    stats.majorant_u_sigma = dsmc_pair_u_sigma_majorant(
        2.0 * speed_bound, maximum_mass, p);
    if (!(stats.majorant_u_sigma > 0.0)
        || !std::isfinite(stats.majorant_u_sigma)) {
        return stats;
    }

    // An exponential waiting time realizes the shell Poisson proposal process.
    // Uniform distinct indices sample each unordered pair with the same rate.
    DSMCSplitMix64 rng(seed);
    double collision_clock = 0.0;
    for (;;) {
        // There are N*(N-1)/2 unordered pairs. Multiplying by the kernel majorant
        // and dividing by physical shell volume gives the total proposal frequency.
        const double pair_count =
            0.5 * static_cast<double>(marker_count)
            * static_cast<double>(marker_count - 1);
        const double trial_rate =
            pair_count * stats.majorant_u_sigma / volume;
        if (!(trial_rate > 0.0) || !std::isfinite(trial_rate)) {
            break;
        }
        collision_clock += -std::log(rng.uniform_open()) / trial_rate;
        if (collision_clock >= dt) {
            break;
        }
        ++stats.trial_pairs;
        const std::size_t ia = static_cast<std::size_t>(
            rng.uniform_index(static_cast<std::uint64_t>(marker_count)));
        std::size_t ib = static_cast<std::size_t>(
            rng.uniform_index(static_cast<std::uint64_t>(marker_count - 1)));
        if (ib >= ia) {
            ++ib;
        }

        const double first_mass = masses[ia];
        const double second_mass = masses[ib];
        const double pair_mass = first_mass + second_mass;
        const BPFAVec3 v1 = dsmc_to_vec3(velocities[ia]);
        const BPFAVec3 v2 = dsmc_to_vec3(velocities[ib]);
        const BPFAVec3 relative = bp_fa_sub(v1, v2);
        const double u = bp_fa_norm(relative);
        const double pair_kernel = u * dsmc_pair_total_cross_section(
            u, first_mass, second_mass, p);
        if (!(pair_kernel > 0.0)) {
            continue;
        }
        if (pair_kernel > stats.majorant_u_sigma
            * (1.0 + 64.0 * std::numeric_limits<double>::epsilon())) {
            throw std::runtime_error(
                "Unequal-mass DSMC pair-kernel majorant was violated.");
        }
        if (rng.uniform_open() * stats.majorant_u_sigma >= pair_kernel) {
            continue;
        }

        // Hold the centre-of-mass momentum and relative speed fixed. Rotating only
        // the relative direction preserves pair momentum and kinetic energy.
        const BPFAVec3 momentum_before = bp_fa_add(
            bp_fa_scale(v1, first_mass),
            bp_fa_scale(v2, second_mass));
        const double kinetic_before =
            0.5 * first_mass * bp_fa_dot(v1, v1)
            + 0.5 * second_mass * bp_fa_dot(v2, v2);
        const BPFAVec3 centre =
            bp_fa_scale(momentum_before, 1.0 / pair_mass);
        const double mu = dsmc_sample_scatter_mu(u, p, rng);
        const double psi = 2.0 * PI * rng.uniform_open();
        BPFAVec3 outgoing_direction;
        if (bp_fa_kernel_name(p) == "yukawa_born") {
            outgoing_direction =
                bp_fa_scattered_relative_direction(relative, mu, psi);
        } else {
            const double sint =
                std::sqrt(std::max(0.0, 1.0 - mu * mu));
            outgoing_direction = {
                mu,
                sint * std::cos(psi),
                sint * std::sin(psi),
            };
        }
        const BPFAVec3 relative_after =
            bp_fa_scale(outgoing_direction, u);
        const BPFAVec3 out1 = bp_fa_add(
            centre,
            bp_fa_scale(relative_after, second_mass / pair_mass));
        const BPFAVec3 out2 = bp_fa_sub(
            centre,
            bp_fa_scale(relative_after, first_mass / pair_mass));
        velocities[ia] = dsmc_from_vec3(out1);
        velocities[ib] = dsmc_from_vec3(out2);
        ++stats.accepted_pairs;

        // An elastic unequal-mass event can increase one particle speed. Enlarge the
        // future proposal majorant immediately when its required speed bound grows.
        const double new_speed =
            std::max(bp_fa_norm(out1), bp_fa_norm(out2));
        if (new_speed > speed_bound) {
            speed_bound = new_speed;
            stats.majorant_u_sigma = std::max(
                stats.majorant_u_sigma,
                dsmc_pair_u_sigma_majorant(
                    2.0 * speed_bound, maximum_mass, p));
        }

        const BPFAVec3 momentum_after = bp_fa_add(
            bp_fa_scale(out1, first_mass),
            bp_fa_scale(out2, second_mass));
        const double kinetic_after =
            0.5 * first_mass * bp_fa_dot(out1, out1)
            + 0.5 * second_mass * bp_fa_dot(out2, out2);
        const BPFAVec3 momentum_error =
            bp_fa_sub(momentum_after, momentum_before);
        const double kinetic_error = kinetic_after - kinetic_before;
        stats.max_pair_momentum_error = std::max(
            stats.max_pair_momentum_error,
            bp_fa_norm(momentum_error));
        stats.max_pair_kinetic_energy_error = std::max(
            stats.max_pair_kinetic_energy_error,
            std::abs(kinetic_error));
    }

    // Sum the final whole-particle invariants independently of the pair residuals.
    // The returned shell budget records accumulated floating-point conservation error.
    const DSMCMarkerSummary after =
        dsmc_marker_summary(velocities, masses);
    stats.majorant_u_sigma = dsmc_pair_u_sigma_majorant(
        2.0 * after.maximum_speed, maximum_mass, p);
    stats.momentum_error_x =
        after.invariants[0] - before.invariants[0];
    stats.momentum_error_y =
        after.invariants[1] - before.invariants[1];
    stats.momentum_error_z =
        after.invariants[2] - before.invariants[2];
    stats.kinetic_energy_error =
        after.invariants[3] - before.invariants[3];
    stats.represented_event_density =
        static_cast<double>(stats.accepted_pairs) / volume;
    return stats;
}

/// Compatibility wrapper for Bird's original equal-weight NTC interface.
///
/// Unit marker masses and volume N/number_density map this call exactly onto
/// the unequal-mass whole-pair implementation above.
inline DSMCCollisionStats apply_dsmc_ntc_equal_weight(
    std::vector<DSMCVelocity>& velocities,
    double number_density,
    double dt,
    const Params& p,
    std::uint64_t seed) {
    const std::size_t marker_count = velocities.size();
    if (!(number_density > 0.0) || !std::isfinite(number_density)) {
        return {};
    }
    const std::vector<double> masses(marker_count, 1.0);
    const double volume =
        static_cast<double>(marker_count) / number_density;
    return apply_dsmc_ntc_unequal_mass(
        velocities, masses, volume, dt, p, seed);
}

}  // namespace ev
