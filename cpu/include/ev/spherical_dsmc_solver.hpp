#pragma once

// Spherical self-gravitating dark matter with persistent DSMC particles.
//
// Each record stores radius r, radial velocity v_r, specific angular momentum
// L, a persistent identity, and represented mass. Radius sorting supplies the
// enclosed mass for Newtonian gravity. Elastic pair encounters are sampled in
// radial shells whose widths are compared with the local mean free path.
// The phase-space grid supplies fresh initial data and output bins; the
// particles carry the evolving distribution between these operations.
//
// The equal-mass material model calibrates encounter rates with sigma/m.
// For unequal records this whole-object collision map uses their represented
// masses as both gravitational and inertial masses. Its physical interpretation
// therefore requires an explicit unequal-object cross-section law. Numerical
// sampling weights alone specify a different kinetic closure.
//
// Gravity uses kick-drift-sort-kick and an energy audit can replace difficult
// pericentre passages with a Levi-Civita orbit integration. Collision shells
// then use either a Poisson trial clock or the published integer-plus-remainder
// construction. Completed steps write mass and energy budgets, radial moments,
// and full particle restarts, including the carried collision timestep.
//
// The spherical force, shell selection, and published NTC/regularization
// construction follow van den Bosch & Tokayer (2026), arXiv:2610.28890,
// doi:10.1093/mnras/stag2015. Distinct numerical choices are described at their
// branches below. Quantities use the gravitational and velocity units in Params.

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <optional>

#include "ev/radial_orbit_regularization.hpp"
#include "ev/solver_setup.hpp"
#include "ev/spherical_dsmc_collision.hpp"

namespace ev {

struct SphericalDSMCParticle {
    double r = 0.0;
    double vr = 0.0;
    double L = 0.0;
    std::uint64_t id = 0;
    // Represented gravitational mass.  The whole-object collision path also
    // treats this as inertial partner mass.  That interpretation is exact for
    // equal-mass material SIDM and for an explicitly supplied physical
    // unequal-object law; an arbitrary unequal numerical sampling weight is
    // not, by itself, a microscopic SIDM particle mass.  Zero is accepted
    // for in-memory construction and means "use particle_mass".
    double mass = 0.0;
};

inline double spherical_dsmc_particle_mass(
    const SphericalDSMCParticle& particle,
    double legacy_particle_mass) {
    const double mass = particle.mass > 0.0
        ? particle.mass
        : legacy_particle_mass;
    if (!(mass > 0.0) || !std::isfinite(mass)) {
        throw std::runtime_error(
            "Spherical DSMC particle mass must be positive and finite.");
    }
    return mass;
}

struct SphericalDSMCSortWorkspace {
    std::vector<SphericalDSMCParticle> scratch;
    std::vector<std::size_t> adjacent_inversions;
    std::vector<std::size_t> long_range_inversions;
    std::vector<std::uint64_t> radix_histogram;
    std::vector<std::uint64_t> radix_offsets;
};

struct SphericalDSMCShell {
    std::size_t begin = 0;
    std::size_t end = 0;
    double inner_edge = 0.0;
    double outer_edge = 0.0;
    double density = 0.0;
    double number_density = 0.0;
    double minimum_mass = std::numeric_limits<double>::infinity();
    double maximum_mass = 0.0;
    double mean_free_path = std::numeric_limits<double>::infinity();
    std::uint64_t mean_free_path_pair_samples = 0;
    double orbital_velocity_dispersion = 0.0;
    double maximum_speed = 0.0;
    std::size_t maximum_speed_index = 0;
    double majorant_u_sigma = 0.0;
};

struct SphericalDSMCDiagnostics {
    long long lc_retries = 0;
    long long lc_accepted = 0;
    long long lc_unresolved = 0;
    long long lc_integrator_failed = 0;
    double max_ordinary_orbit_energy_error = 0.0;
    double max_final_orbit_energy_error = 0.0;
    std::uint64_t collision_trials = 0;
    std::uint64_t collision_accepted = 0;
    double collision_radial_momentum_error = 0.0;
    double collision_kinetic_error = 0.0;
    double max_pair_momentum_error = 0.0;
    double max_pair_kinetic_error = 0.0;
    std::size_t collision_shells = 0;
    std::size_t minimum_shell_particles = 0;
    double minimum_mean_free_path = std::numeric_limits<double>::infinity();
    double maximum_shell_width_over_mean_free_path = 0.0;
    std::uint64_t mean_free_path_pair_samples = 0;
    double minimum_orbital_velocity_dispersion =
        std::numeric_limits<double>::infinity();
    double maximum_orbital_velocity_dispersion = 0.0;
    std::size_t nonlocal_collision_shells = 0;
};

struct SphericalDSMCEnergy {
    double kinetic = 0.0;
    double potential = 0.0;
    double total = 0.0;
};

struct SphericalDSMCStateSums {
    SphericalDSMCEnergy energy;
    double radial_momentum = 0.0;
};

struct SphericalDSMCReductionWorkspace {
    std::vector<SphericalDSMCStateSums> partial;
};

struct SphericalDSMCShellWorkspace {
    std::vector<std::pair<std::size_t, std::size_t>> parents;
    std::vector<std::vector<SphericalDSMCShell>> children;
    // The paper-exact moving cursor can inspect the same candidate range
    // repeatedly while halving it.  Cache each marker speed once and retain
    // exact range maxima in fixed blocks so those inspections do not rescan
    // the particles or recompute hypot().
    std::vector<double> paper_speed;
    std::vector<double> paper_block_maximum_speed;
    std::vector<std::size_t> paper_block_maximum_speed_index;
};

struct SphericalDSMCCollisionWorkspace {
    struct ThreadVelocityCache {
        std::vector<DSMCVelocity> velocities;
        std::vector<unsigned char> initialized;
        std::vector<unsigned char> changed;
        std::vector<std::size_t> changed_indices;
    };

    std::vector<DSMCCollisionStats> stats;
    std::vector<std::size_t> shell_sizes;
    std::vector<ThreadVelocityCache> thread_cache;
    // Frank & Tokayer equation (8) assigns a persistent fractional-trial
    // accumulator to every particle.  Indexing by immutable particle id keeps
    // the accumulator attached to the particle as it migrates between shells.
    // This diagnostic state persists for the lifetime of a solver process.
    std::vector<double> paper_remainder_accumulator_by_id;
};

struct SphericalDSMCGravityWorkspace {
    std::vector<double> energy_before;
    std::vector<double> energy_scale_before;
    std::vector<double> paper_potential;
    std::vector<double> paper_outer_sum;
    std::vector<double> initial_enclosed_force_mass;
    std::vector<double> enclosed_force_mass;
    // The pre-step state is stored in the already sorted rank order.  Keeping
    // this 40-byte record contiguous avoids a full particle-ID-scrambled
    // write on every step; rank_by_id provides the exact lookup needed by the
    // energy audit and the rare LC retry.
    std::vector<SphericalDSMCParticle> initial_by_rank;
    std::vector<std::size_t> rank_by_id;
    std::vector<std::size_t> final_rank_by_id;
    std::vector<std::vector<std::size_t>> failed_by_thread;
    std::vector<std::vector<std::uint64_t>> failed_ids_by_thread;
    std::vector<std::size_t> failed;
    std::vector<std::uint64_t> failed_ids;
    std::vector<unsigned char> retry_converged;
    std::vector<unsigned char> retry_accepted;
    std::vector<double> retry_best_relative_error;
    std::vector<double> maximum_error_by_thread;
    std::string paper_failure_message;
    SphericalDSMCSortWorkspace sort;
};

// A shell stream is a deterministic function of the base seed, accepted
// step number, and shell index. Disjoint shells can run on different workers
// while retaining the same stream assignment.
inline std::uint64_t spherical_dsmc_seed(
    std::uint64_t base,
    std::uint64_t step,
    std::uint64_t stream) {
    std::uint64_t value = base
        ^ (step + UINT64_C(0x9e3779b97f4a7c15))
        ^ (stream * UINT64_C(0xbf58476d1ce4e5b9));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

inline bool spherical_dsmc_paper_exact_enabled() {
    const char* value = std::getenv("EV_SPHERICAL_DSMC_PAPER_EXACT");
    return value != nullptr && std::string(value) == "1";
}

inline bool spherical_dsmc_force_consistent_orbit_audit_enabled() {
    const char* value =
        std::getenv("EV_SPHERICAL_DSMC_FORCE_CONSISTENT_ORBIT_AUDIT");
    return value != nullptr && std::string(value) == "1";
}

inline bool spherical_dsmc_force_radix_sort_enabled() {
    const char* value = std::getenv("EV_SPHERICAL_DSMC_RADIX_SORT");
    return value != nullptr && std::string(value) == "1";
}

// The published algorithm fixes its shell sizes, angular law, and orbit
// tolerances together. Check that complete set before using its trial-count
// and orbit-energy conventions.
inline void spherical_dsmc_validate_paper_exact_configuration(
    const Params& p) {
    auto close = [](double a, double b) {
        return std::abs(a - b)
            <= 64.0 * std::numeric_limits<double>::epsilon()
                * std::max({1.0, std::abs(a), std::abs(b)});
    };
    if (p.bp_cross_section != "constant"
        || bp_fa_kernel_name(p) != "isotropic"
        || p.bp_dsmc_shell_particles != 4096
        || p.bp_dsmc_min_shell_particles != 32
        || !close(p.bp_dsmc_collision_epsilon, 1.0)
        || !close(p.bp_dsmc_gravity_eta, 0.01)
        || p.vp_orbit_regularization != "auto"
        || !close(p.vp_orbit_energy_tolerance, 0.01)) {
        throw std::runtime_error(
            "EV_SPHERICAL_DSMC_PAPER_EXACT=1 requires Frank & Tokayer's "
            "published constant/isotropic kernel, N_shell=4096, shell floor "
            "32, zeta=1, eta=0.01, automatic LC retry, and delta_crit=0.01.");
    }
}

inline double spherical_dsmc_tangential_speed(const SphericalDSMCParticle& particle) {
    if (particle.r == 0.0 && particle.L != 0.0) {
        throw std::runtime_error("A spherical DSMC particle at the origin requires L=0.");
    }
    return particle.L == 0.0 ? 0.0 : particle.L / particle.r;
}

inline DSMCVelocity spherical_dsmc_origin_velocity(
    double radial_speed, double cosine, double azimuth) {
    const double speed = std::abs(radial_speed);
    const double transverse = speed * std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
    return {speed * cosine, transverse * std::cos(azimuth), transverse * std::sin(azimuth)};
}

inline void spherical_dsmc_store_collision_velocity(
    SphericalDSMCParticle& particle, const DSMCVelocity& velocity) {
    // At r=0 the direction is isotropic and L=0. The outgoing radial
    // characteristic stores the full speed so that kinetic energy is retained.
    particle.vr = particle.r == 0.0
        ? std::hypot(velocity.x, std::hypot(velocity.y, velocity.z)) : velocity.x;
    particle.L = particle.r * std::hypot(velocity.y, velocity.z);
}

inline double spherical_dsmc_specific_kinetic_energy(
    const SphericalDSMCParticle& particle) {
    const double vt = spherical_dsmc_tangential_speed(particle);
    return 0.5 * (particle.vr * particle.vr + vt * vt);
}

inline double spherical_dsmc_specific_kepler_energy(
    const SphericalDSMCParticle& particle,
    double enclosed_mass,
    const Params& p) {
    const double radius = std::max(particle.r, 1.0e-300);
    const double angular_squared = particle.L * particle.L;
    const double radius_squared = radius * radius;
    const double angular_kinetic = particle.L == 0.0 ? 0.0
        : (angular_squared > 0.0 && std::isfinite(angular_squared)
            && radius_squared > 0.0 && std::isfinite(radius_squared)
            ? 0.5 * angular_squared / radius_squared
            : 0.5 * (particle.L / radius) * (particle.L / radius));
    return 0.5 * particle.vr * particle.vr
        + angular_kinetic
        - p.vp_gravity * enclosed_mass / radius;
}

/// Relative particle-energy defect used by the paper-exact orbit audit.
///
/// Frank & Tokayer divide by |E_i(t)|.  That expression is singular for a
/// parabolic orbit, which can be produced naturally by an elastic collision:
/// kinetic and potential energies remain finite while their sum passes
/// through zero.  Away from that singular limit this is their literal
/// equation.  Inside |E| < delta_crit (K+|Phi|), use the smallest continuous
/// local energy-scale floor.  The resulting absolute tolerance is
/// delta_crit^2 (K+|Phi|), so this guard remains substantially more
/// conservative than a blanket normalization by K+|Phi|.
inline double spherical_dsmc_paper_orbit_energy_defect(
    double energy_after,
    double energy_before,
    double energy_scale_before,
    double delta_crit) {
    const double scale = std::max(
        std::abs(energy_scale_before),
        1.0e-300);
    const double denominator = std::max(
        std::abs(energy_before),
        delta_crit * scale);
    return std::abs(energy_after - energy_before) / denominator;
}

inline void spherical_dsmc_insertion_sort_particle_range(
    std::vector<SphericalDSMCParticle>& particles,
    std::size_t begin,
    std::size_t end) {
    for (std::size_t i = begin + 1; i < end; ++i) {
        // In a Frank step almost every marker remains in radial order.  Avoid
        // copying and writing the full particle record in that common case;
        // the inverted cases still follow the literal stable insertion sort.
        if (!(particles[i].r < particles[i - 1].r)) {
            continue;
        }
        SphericalDSMCParticle key = particles[i];
        std::size_t j = i;
        while (j > begin && key.r < particles[j - 1].r) {
            particles[j] = particles[j - 1];
            --j;
        }
        particles[j] = key;
    }
}

inline void spherical_dsmc_insertion_sort_particles(
    std::vector<SphericalDSMCParticle>& particles) {
    if (particles.size() < 2) {
        return;
    }
    spherical_dsmc_insertion_sort_particle_range(
        particles, 0, particles.size());
}

inline void spherical_dsmc_sort_particles(
    std::vector<SphericalDSMCParticle>& particles,
    bool initial_sort = false) {
    if (particles.size() < 2) {
        return;
    }
    std::size_t adjacent_inversions = 0;
    if (!initial_sort) {
        for (std::size_t i = 1; i < particles.size(); ++i) {
            adjacent_inversions += particles[i].r < particles[i - 1].r ? 1U : 0U;
        }
    }
    if (initial_sort || adjacent_inversions > particles.size() / 64U) {
        std::stable_sort(
            particles.begin(),
            particles.end(),
            [](const SphericalDSMCParticle& a, const SphericalDSMCParticle& b) {
                return a.r < b.r;
            });
        return;
    }

    // Frank's particles normally move only a few ranks per global step, for
    // which insertion sort is close to O(N) and avoids general-sort overhead.
    spherical_dsmc_insertion_sort_particles(particles);
}

// Spherical gravity depends on radius rank. Stable ordering retains the
// previous order of equal radii, so sorting also fixes the self-shell force
// and the identities assigned to subsequent collision shells. The disorder
// scan selects a sorting method while preserving that ordering rule.
inline void spherical_dsmc_sort_particles_parallel(
    std::vector<SphericalDSMCParticle>& particles,
    const Params& p,
    SphericalDSMCSortWorkspace& workspace,
    bool initial_sort = false) {
    const std::size_t count = particles.size();
    if (count < 2) {
        return;
    }
    const int nthreads = effective_threads(p, static_cast<int>(count));
    if (nthreads == 1) {
        // Frank's production loop always applies insertion sort to the
        // radius ordering left by the preceding small global step.  Do not
        // route the one-thread paper-exact path through VENUS's general
        // adjacent-inversion heuristic: for a million particles it can
        // select std::stable_sort even though the existing order makes the
        // literal insertion pass much cheaper.  The threaded paper-exact
        // branch below already preserves this rule; serial execution must do
        // the same.
        if (spherical_dsmc_paper_exact_enabled() && !initial_sort) {
            spherical_dsmc_insertion_sort_particles(particles);
        } else {
            spherical_dsmc_sort_particles(particles, initial_sort);
        }
        return;
    }

    // Frank & Tokayer explicitly use insertion sort after every evolved
    // step.  Parallelize only that sorting operation: stable-insertion-sort
    // contiguous lanes, then stable-merge the lanes.  This is the OpenMP
    // decomposition of the same stable radius ordering; it is independent of
    // the general VENUS radix path and gives exactly the serial insertion-sort
    // result, including equal-radius order.
    if (spherical_dsmc_paper_exact_enabled() && !initial_sort) {
        workspace.adjacent_inversions.assign(
            static_cast<std::size_t>(nthreads),
            0);
        workspace.long_range_inversions.assign(
            static_cast<std::size_t>(nthreads),
            0);
        parallel_for_chunks(
            1,
            static_cast<int>(count),
            nthreads,
            [&](int lo, int hi, int tid) {
                std::size_t local = 0;
                std::size_t local_long_range = 0;
                for (int ii = lo; ii < hi; ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    local += particles[i].r < particles[i - 1].r ? 1U : 0U;
                    local_long_range +=
                        i >= 32 && particles[i].r < particles[i - 32].r
                        ? 1U : 0U;
                }
                workspace.adjacent_inversions[static_cast<std::size_t>(tid)] =
                    local;
                workspace.long_range_inversions[static_cast<std::size_t>(tid)] =
                    local_long_range;
            });
        const std::size_t adjacent_inversions = std::accumulate(
            workspace.adjacent_inversions.begin(),
            workspace.adjacent_inversions.end(),
            std::size_t{0});
        if (adjacent_inversions == 0) {
            return;
        }
        const std::size_t long_range_inversions = std::accumulate(
            workspace.long_range_inversions.begin(),
            workspace.long_range_inversions.end(),
            std::size_t{0});
        // Adjacent inversions alone cannot distinguish many one-rank swaps
        // from broad marker motion.  A 32-rank probe does: retain Frank's
        // literal pass when its local moves are cheaper than five full merge
        // levels, and use the parallel decomposition only for broad disorder.
        if (long_range_inversions <= count / 4U) {
            spherical_dsmc_insertion_sort_particles(particles);
            return;
        }

        const std::size_t block_count = static_cast<std::size_t>(
            std::min<int>(nthreads, static_cast<int>(count)));
        parallel_for_chunks(
            0,
            static_cast<int>(block_count),
            nthreads,
            [&](int lo, int hi, int) {
                for (int bb = lo; bb < hi; ++bb) {
                    const std::size_t block = static_cast<std::size_t>(bb);
                    const std::size_t begin = count * block / block_count;
                    const std::size_t end =
                        count * (block + 1) / block_count;
                    spherical_dsmc_insertion_sort_particle_range(
                        particles, begin, end);
                }
            });

        workspace.scratch.resize(count);
        std::vector<SphericalDSMCParticle>* source = &particles;
        std::vector<SphericalDSMCParticle>* destination = &workspace.scratch;
        for (std::size_t width = 1; width < block_count; width *= 2) {
            const std::size_t groups =
                (block_count + 2 * width - 1) / (2 * width);
            parallel_for_chunks(
                0,
                static_cast<int>(groups),
                std::min(nthreads, static_cast<int>(groups)),
                [&](int lo, int hi, int) {
                    for (int gg = lo; gg < hi; ++gg) {
                        const std::size_t group = static_cast<std::size_t>(gg);
                        const std::size_t first_block = 2 * width * group;
                        const std::size_t middle_block =
                            std::min(first_block + width, block_count);
                        const std::size_t final_block =
                            std::min(first_block + 2 * width, block_count);
                        const std::size_t begin =
                            count * first_block / block_count;
                        const std::size_t middle =
                            count * middle_block / block_count;
                        const std::size_t end =
                            count * final_block / block_count;
                        std::size_t left = begin;
                        std::size_t right = middle;
                        std::size_t out = begin;
                        while (left < middle && right < end) {
                            if ((*source)[right].r < (*source)[left].r) {
                                (*destination)[out++] = (*source)[right++];
                            } else {
                                (*destination)[out++] = (*source)[left++];
                            }
                        }
                        while (left < middle) {
                            (*destination)[out++] = (*source)[left++];
                        }
                        while (right < end) {
                            (*destination)[out++] = (*source)[right++];
                        }
                    }
                });
            std::swap(source, destination);
        }
        if (source != &particles) {
            parallel_for_chunks(
                0,
                static_cast<int>(count),
                nthreads,
                [&](int lo, int hi, int) {
                    std::copy(
                        source->begin() + lo,
                        source->begin() + hi,
                        particles.begin() + lo);
                });
        }
        return;
    }

    const bool force_radix = spherical_dsmc_force_radix_sort_enabled();
    std::size_t adjacent_inversions = count;
    if (!initial_sort) {
        workspace.adjacent_inversions.assign(
            static_cast<std::size_t>(nthreads),
            0);
        parallel_for_chunks(
            1,
            static_cast<int>(count),
            nthreads,
            [&](int lo, int hi, int tid) {
                std::size_t local = 0;
                for (int ii = lo; ii < hi; ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    local += particles[i].r < particles[i - 1].r ? 1U : 0U;
                }
                workspace.adjacent_inversions[static_cast<std::size_t>(tid)] =
                    local;
            });
        adjacent_inversions = std::accumulate(
            workspace.adjacent_inversions.begin(),
            workspace.adjacent_inversions.end(),
            std::size_t{0});
    }
    if (!initial_sort && adjacent_inversions == 0) {
        return;
    }
    if (force_radix || initial_sort) {
        workspace.scratch.resize(count);
        constexpr std::size_t radix = 256;
        constexpr int passes = 8;
        workspace.radix_histogram.assign(
            static_cast<std::size_t>(nthreads) * radix,
            0);
        workspace.radix_offsets.resize(
            static_cast<std::size_t>(nthreads) * radix);
        std::vector<SphericalDSMCParticle>* source = &particles;
        std::vector<SphericalDSMCParticle>* destination = &workspace.scratch;
        for (int pass = 0; pass < passes; ++pass) {
            std::fill(
                workspace.radix_histogram.begin(),
                workspace.radix_histogram.end(),
                0);
            const int shift = 8 * pass;
            parallel_for_chunks(
                0,
                static_cast<int>(count),
                nthreads,
                [&](int lo, int hi, int tid) {
                    std::uint64_t* local =
                        workspace.radix_histogram.data()
                        + static_cast<std::size_t>(tid) * radix;
                    for (int ii = lo; ii < hi; ++ii) {
                        std::uint64_t key = 0;
                        std::memcpy(
                            &key,
                            &(*source)[static_cast<std::size_t>(ii)].r,
                            sizeof(double));
                        ++local[(key >> shift) & UINT64_C(0xff)];
                    }
                });
            std::uint64_t bucket_start = 0;
            for (std::size_t digit = 0; digit < radix; ++digit) {
                std::uint64_t offset = bucket_start;
                for (int tid = 0; tid < nthreads; ++tid) {
                    const std::size_t slot =
                        static_cast<std::size_t>(tid) * radix + digit;
                    workspace.radix_offsets[slot] = offset;
                    offset += workspace.radix_histogram[slot];
                }
                bucket_start = offset;
            }
            parallel_for_chunks(
                0,
                static_cast<int>(count),
                nthreads,
                [&](int lo, int hi, int tid) {
                    std::uint64_t* local_offsets =
                        workspace.radix_offsets.data()
                        + static_cast<std::size_t>(tid) * radix;
                    for (int ii = lo; ii < hi; ++ii) {
                        const SphericalDSMCParticle& particle =
                            (*source)[static_cast<std::size_t>(ii)];
                        std::uint64_t key = 0;
                        std::memcpy(&key, &particle.r, sizeof(double));
                        const std::size_t digit =
                            static_cast<std::size_t>(
                                (key >> shift) & UINT64_C(0xff));
                        (*destination)[local_offsets[digit]++] = particle;
                    }
                });
            std::swap(source, destination);
        }
        return;
    }
    if (!initial_sort && adjacent_inversions <= count / 64U) {
        // The number of assignments in insertion sort is exactly the number
        // of inversions, so this remains the fastest path after a small step.
        spherical_dsmc_sort_particles(particles);
        return;
    }

    const std::size_t block_count = static_cast<std::size_t>(
        std::min<int>(nthreads, static_cast<int>(count)));
    parallel_for_chunks(
        0,
        static_cast<int>(block_count),
        nthreads,
        [&](int lo, int hi, int) {
            for (int bb = lo; bb < hi; ++bb) {
                const std::size_t block = static_cast<std::size_t>(bb);
                const std::size_t begin = count * block / block_count;
                const std::size_t end = count * (block + 1) / block_count;
                std::stable_sort(
                    particles.begin() + static_cast<std::ptrdiff_t>(begin),
                    particles.begin() + static_cast<std::ptrdiff_t>(end),
                    [](const SphericalDSMCParticle& a,
                       const SphericalDSMCParticle& b) {
                        return a.r < b.r;
                    });
            }
        });

    workspace.scratch.resize(count);
    std::vector<SphericalDSMCParticle>* source = &particles;
    std::vector<SphericalDSMCParticle>* destination = &workspace.scratch;
    for (std::size_t width = 1; width < block_count; width *= 2) {
        const std::size_t groups =
            (block_count + 2 * width - 1) / (2 * width);
        parallel_for_chunks(
            0,
            static_cast<int>(groups),
            std::min(nthreads, static_cast<int>(groups)),
            [&](int lo, int hi, int) {
                for (int gg = lo; gg < hi; ++gg) {
                    const std::size_t group = static_cast<std::size_t>(gg);
                    const std::size_t first_block = 2 * width * group;
                    const std::size_t middle_block =
                        std::min(first_block + width, block_count);
                    const std::size_t final_block =
                        std::min(first_block + 2 * width, block_count);
                    const std::size_t begin =
                        count * first_block / block_count;
                    const std::size_t middle =
                        count * middle_block / block_count;
                    const std::size_t end =
                        count * final_block / block_count;
                    std::size_t left = begin;
                    std::size_t right = middle;
                    std::size_t out = begin;
                    while (left < middle && right < end) {
                        if ((*source)[right].r < (*source)[left].r) {
                            (*destination)[out++] = (*source)[right++];
                        } else {
                            (*destination)[out++] = (*source)[left++];
                        }
                    }
                    while (left < middle) {
                        (*destination)[out++] = (*source)[left++];
                    }
                    while (right < end) {
                        (*destination)[out++] = (*source)[right++];
                    }
                }
            });
        std::swap(source, destination);
    }
    if (source != &particles) {
        parallel_for_chunks(
            0,
            static_cast<int>(count),
            nthreads,
            [&](int lo, int hi, int) {
                std::copy(
                    source->begin() + lo,
                    source->begin() + hi,
                    particles.begin() + lo);
            });
    }
}

// The grid distribution contributes mass 8*pi^2*L*f*dr*dw*dL. A single
// random offset selects equal-mass representatives systematically from its
// normalized cumulative mass. Within a selected cell, uniform L^2 sampling
// supplies the L*dL Jacobian; r and radial velocity are uniform in their bins.
inline std::vector<SphericalDSMCParticle> spherical_dsmc_sample_initial_particles(
    const SolverState& state,
    double& total_mass) {
    if (state.mpi.size != 1) {
        throw std::runtime_error("spherical_dsmc requires one Slurm/MPI task; use OpenMP threads.");
    }
    const Params& p = state.local_p;
    std::vector<double> r_edges = radial_control_volume_edges(state.r, p);
    r_edges.front() = 0.0;
    const std::vector<double> w_edges = control_volume_edges(state.w);
    const std::vector<double> L_edges = control_volume_edges(state.L);
    std::vector<double> cumulative;
    std::vector<std::size_t> active_index;
    cumulative.reserve(state.f.size());
    active_index.reserve(state.f.size());
    total_mass = 0.0;
    for (int i = 0; i < p.Nr; ++i) {
        const double dr = r_edges[static_cast<std::size_t>(i + 1)]
            - r_edges[static_cast<std::size_t>(i)];
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const double value = state.f[idx(i, j, k, p)];
                const double cell_mass = 8.0 * PI * PI
                    * std::max(state.L[static_cast<std::size_t>(k)], 0.0)
                    * std::max(value, 0.0)
                    * dr
                    * state.wq[static_cast<std::size_t>(j)]
                    * state.Lq[static_cast<std::size_t>(k)];
                if (!(cell_mass > 0.0) || !std::isfinite(cell_mass)) {
                    continue;
                }
                total_mass += cell_mass;
                cumulative.push_back(total_mass);
                active_index.push_back(idx(i, j, k, p));
            }
        }
    }
    if (!(total_mass > 0.0) || active_index.empty()) {
        throw std::runtime_error("Cannot sample spherical_dsmc markers from a zero-mass DF.");
    }
    const double sampling_mass = total_mass;

    const std::size_t count = static_cast<std::size_t>(p.bp_dsmc_particles);
    std::vector<SphericalDSMCParticle> particles(count);
    DSMCSplitMix64 rng(p.bp_dsmc_seed);
    const double systematic_offset = rng.uniform_open();
    for (std::size_t q = 0; q < count; ++q) {
        const double target = sampling_mass
            * (static_cast<double>(q) + systematic_offset)
            / static_cast<double>(count);
        const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), target);
        const std::size_t active = std::min<std::size_t>(
            static_cast<std::size_t>(it - cumulative.begin()),
            active_index.size() - 1);
        const std::size_t flat = active_index[active];
        const int i = static_cast<int>(flat / static_cast<std::size_t>(p.Nw * p.NL));
        const int rem = static_cast<int>(
            flat - static_cast<std::size_t>(i * p.Nw * p.NL));
        const int j = rem / p.NL;
        const int k = rem - j * p.NL;

        const double r0 = std::max(0.0, r_edges[static_cast<std::size_t>(i)]);
        const double r1 = r_edges[static_cast<std::size_t>(i + 1)];
        const double w0 = w_edges[static_cast<std::size_t>(j)];
        const double w1 = w_edges[static_cast<std::size_t>(j + 1)];
        const double l0 = std::max(0.0, L_edges[static_cast<std::size_t>(k)]);
        const double l1 = std::max(l0, L_edges[static_cast<std::size_t>(k + 1)]);
        SphericalDSMCParticle particle;
        particle.r = r0 + (r1 - r0) * rng.uniform_open();
        particle.r = std::max(particle.r, 1.0e-14 * std::max(1.0, r1));
        particle.vr = w0 + (w1 - w0) * rng.uniform_open();
        particle.L = std::sqrt(
            l0 * l0 + (l1 * l1 - l0 * l0) * rng.uniform_open());
        particle.id = static_cast<std::uint64_t>(q);
        particles[q] = particle;
    }
    spherical_dsmc_sort_particles(particles, true);
    if (p.bp_dsmc_total_mass > 0.0) {
        total_mass = p.bp_dsmc_total_mass;
    }
    const double particle_mass =
        total_mass / static_cast<double>(particles.size());
    for (SphericalDSMCParticle& particle : particles) {
        particle.mass = particle_mass;
    }
    return particles;
}

inline double spherical_dsmc_acceleration(
    const SphericalDSMCParticle& particle,
    std::size_t rank,
    double particle_mass,
    const Params& p) {
    const double radius = std::max(particle.r, 1.0e-300);
    const double enclosed =
        p.central_mass + (static_cast<double>(rank) + 0.5) * particle_mass;
    return particle.L * particle.L / (radius * radius * radius)
        - p.vp_gravity * enclosed / (radius * radius);
}

/// Mass interior to each sorted particle, including half of its own spherical
/// shell.  This is the unequal-mass form of (rank+1/2)*particle_mass.
// The force on a spherical shell uses all strictly interior mass and half
// of its own represented shell mass. Prefix sums extend the equal-mass rank
// formula to positive unequal masses without changing this force convention.
inline void spherical_dsmc_enclosed_force_masses(
    const std::vector<SphericalDSMCParticle>& particles,
    double legacy_particle_mass,
    const Params& p,
    std::vector<double>& enclosed) {
    enclosed.resize(particles.size());
    double interior = p.central_mass;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double mass =
            spherical_dsmc_particle_mass(particles[i], legacy_particle_mass);
        enclosed[i] = interior + 0.5 * mass;
        interior += mass;
    }
}

inline double spherical_dsmc_acceleration(
    const SphericalDSMCParticle& particle,
    double enclosed_force_mass,
    const Params& p) {
    const double radius = std::max(particle.r, 1.0e-300);
    return particle.L * particle.L / (radius * radius * radius)
        - p.vp_gravity * enclosed_force_mass / (radius * radius);
}

inline void spherical_dsmc_paper_specific_potential(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p,
    std::vector<double>& outer_sum,
    std::vector<double>& potential) {
    const std::size_t count = particles.size();
    outer_sum.resize(count + 1);
    potential.resize(count);
    outer_sum[count] = 0.0;
    for (std::size_t reverse = count; reverse > 0; --reverse) {
        const std::size_t i = reverse - 1;
        const double radius = std::max(particles[i].r, 1.0e-300);
        const double mass =
            spherical_dsmc_particle_mass(particles[i], particle_mass);
        outer_sum[i] = outer_sum[i + 1] + mass / radius;
    }
    double interior_and_self = p.central_mass;
    for (std::size_t i = 0; i < count; ++i) {
        const double radius = std::max(particles[i].r, 1.0e-300);
        interior_and_self +=
            spherical_dsmc_particle_mass(particles[i], particle_mass);
        // Frank & Tokayer equation (5), with their one-based rank
        // i_paper=i+1:
        //   Phi_i=-G m_p[(i+1)/r_i + sum_{j=i+1}^{N-1} 1/r_j].
        // The full self-shell term used in this diagnostic potential differs
        // from the half-shell force in equation (4), which is retained by
        // spherical_dsmc_acceleration above.
        potential[i] = -p.vp_gravity
            * (interior_and_self / radius + outer_sum[i + 1]);
    }
}

/// Particle potential used to judge the accuracy of an orbit step.
///
/// Equations (4) and (5) of Frank & Tokayer differ by half of the marker's
/// own shell: their force contains (i-1/2)m_p, while their printed particle
/// potential contains i m_p.  Consequently, the equation-(5) energy is not
/// an invariant of the equation-(4) orbit even in an otherwise frozen field.
/// This finite-N difference is normally negligible, but becomes singular for
/// the innermost marker as r approaches zero and cannot be removed by LC
/// refinement.  The opt-in audit below subtracts only that half-self term.
/// It leaves the published equation-(5) diagnostic, the physical force, the
/// total energy, and every collision operation unchanged.
// An orbit audit compares energy before and after transport using one
// potential convention. The optional force-consistent convention removes half
// of the particle self-potential; the published convention retains its full
// self term. Both choices leave the gravitational acceleration unchanged.
inline void spherical_dsmc_orbit_audit_specific_potential(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p,
    std::vector<double>& outer_sum,
    std::vector<double>& potential) {
    spherical_dsmc_paper_specific_potential(
        particles,
        particle_mass,
        p,
        outer_sum,
        potential);
    if (!spherical_dsmc_force_consistent_orbit_audit_enabled()) {
        return;
    }
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double half_self = 0.5 * p.vp_gravity
            * spherical_dsmc_particle_mass(particles[i], particle_mass);
        potential[i] += half_self
            / std::max(particles[i].r, 1.0e-300);
    }
}

inline std::vector<double> spherical_dsmc_specific_potential(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p) {
    std::vector<double> outer_sum;
    std::vector<double> potential;
    spherical_dsmc_paper_specific_potential(
        particles,
        particle_mass,
        p,
        outer_sum,
        potential);
    return potential;
}

// The total potential energy counts each pair once and assigns half of a
// spherical shell self-energy. Multiplication by each actual particle mass
// keeps kinetic and potential terms in the same total-energy normalization.
inline SphericalDSMCEnergy spherical_dsmc_total_energy(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p) {
    SphericalDSMCEnergy energy;
    double interior = p.central_mass;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double mass =
            spherical_dsmc_particle_mass(particles[i], particle_mass);
        energy.kinetic += mass
            * spherical_dsmc_specific_kinetic_energy(particles[i]);
        energy.potential -= p.vp_gravity * mass
            * (interior + 0.5 * mass)
            / std::max(particles[i].r, 1.0e-300);
        interior += mass;
    }
    energy.total = energy.kinetic + energy.potential;
    return energy;
}

inline SphericalDSMCStateSums spherical_dsmc_state_sums_parallel(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p,
    SphericalDSMCReductionWorkspace& workspace) {
    const int nthreads = effective_threads(p, static_cast<int>(particles.size()));
    workspace.partial.assign(
        static_cast<std::size_t>(nthreads),
        SphericalDSMCStateSums{});
    std::vector<double> enclosed;
    spherical_dsmc_enclosed_force_masses(
        particles, particle_mass, p, enclosed);
    parallel_for_chunks(
        0,
        static_cast<int>(particles.size()),
        nthreads,
        [&](int lo, int hi, int tid) {
            SphericalDSMCStateSums local;
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                const SphericalDSMCParticle& particle = particles[i];
                const double mass =
                    spherical_dsmc_particle_mass(particle, particle_mass);
                local.energy.kinetic += mass
                    * spherical_dsmc_specific_kinetic_energy(particle);
                local.energy.potential -= p.vp_gravity * mass
                    * enclosed[i]
                    / std::max(particle.r, 1.0e-300);
                local.radial_momentum += mass * particle.vr;
            }
            workspace.partial[static_cast<std::size_t>(tid)] = local;
        });
    SphericalDSMCStateSums result;
    for (const SphericalDSMCStateSums& local : workspace.partial) {
        result.energy.kinetic += local.energy.kinetic;
        result.energy.potential += local.energy.potential;
        result.radial_momentum += local.radial_momentum;
    }
    result.energy.total = result.energy.kinetic + result.energy.potential;
    return result;
}

inline double spherical_dsmc_radial_momentum(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass) {
    double momentum = 0.0;
    for (const SphericalDSMCParticle& particle : particles) {
        momentum +=
            spherical_dsmc_particle_mass(particle, particle_mass)
            * particle.vr;
    }
    return momentum;
}

inline double spherical_dsmc_total_particle_mass(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass) {
    double total = 0.0;
    for (const SphericalDSMCParticle& particle : particles) {
        total += spherical_dsmc_particle_mass(particle, particle_mass);
    }
    return total;
}

// The dynamical step limits radial motion through the local acceleration
// scale, sqrt(2*eta*r/|a|). Every particle participates in the minimum,
// including those near the origin and those with small represented mass.
inline double spherical_dsmc_gravity_timestep(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p) {
    const int nthreads = effective_threads(p, static_cast<int>(particles.size()));
    std::vector<double> partial(
        static_cast<std::size_t>(nthreads),
        std::numeric_limits<double>::infinity());
    std::vector<double> enclosed;
    spherical_dsmc_enclosed_force_masses(
        particles, particle_mass, p, enclosed);
    parallel_for_chunks(0, static_cast<int>(particles.size()), nthreads,
        [&](int lo, int hi, int tid) {
            double local = std::numeric_limits<double>::infinity();
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                const double acceleration = std::abs(
                    spherical_dsmc_acceleration(
                        particles[i], enclosed[i], p));
                if (acceleration > 1.0e-300) {
                    local = std::min(
                        local,
                        particles[i].r / acceleration);
                }
            }
            partial[static_cast<std::size_t>(tid)] = local;
        });
    const double minimum =
        *std::min_element(partial.begin(), partial.end());
    return std::isfinite(minimum)
        ? std::sqrt(2.0 * p.bp_dsmc_gravity_eta * minimum)
        : p.dt_max;
}

struct SphericalDSMCMeanFreePathEstimate {
    double mean_free_path = std::numeric_limits<double>::infinity();
    double orbital_velocity_dispersion = 0.0;
    std::uint64_t pair_samples = 0;
};

/// Velocity-dependent extension of Frank & Tokayer's shell mean free path.
///
/// Frank's equation lambda=1/[rho*(sigma/m)] assumes equal masses and a
/// velocity-independent cross section per unit mass.  For complete unequal
/// mass markers and Sigma_ij(u), the same collision-rate argument gives
///
///   lambda = <u> / [n_partner <u Sigma_ij(u)>],
///
/// where n_partner=(N-1)/V is the finite-shell density of other objects
/// available to one marker.  The usual n=N/V expression is its continuum
/// approximation and would overestimate the encounter rate in sparse shells.
///
/// The averages below are deterministic Monte-Carlo quadratures of the
/// current shell distribution, including the unresolved tangential azimuth.
/// Thus both the orbital velocity dispersion and the effective cross section
/// evolve with the distribution function.
// For a velocity-dependent cross section, estimate lambda=<|v-vmean|>/
// [n_partner*<u*sigma_ij(u)>]. The partner density uses N-1 available partners.
// A bounded deterministic pair sample resolves the pair-speed average and
// random tangential azimuths while keeping shell construction reproducible.
inline SphericalDSMCMeanFreePathEstimate
spherical_dsmc_velocity_averaged_mean_free_path(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    std::size_t begin,
    std::size_t end,
    double volume,
    const Params& p) {
    SphericalDSMCMeanFreePathEstimate result;
    const std::size_t count = end - begin;
    if (count < 2 || !(volume > 0.0)) {
        return result;
    }

    long double mass_sum = 0.0L;
    long double radial_momentum = 0.0L;
    long double speed_squared_moment = 0.0L;
    for (std::size_t i = begin; i < end; ++i) {
        const double mass =
            spherical_dsmc_particle_mass(particles[i], particle_mass);
        const double tangential_speed =
            spherical_dsmc_tangential_speed(particles[i]);
        mass_sum += mass;
        radial_momentum +=
            static_cast<long double>(mass)
                * (particles[i].r == 0.0 ? 0.0 : particles[i].vr);
        speed_squared_moment += static_cast<long double>(mass)
            * (particles[i].vr * particles[i].vr
               + tangential_speed * tangential_speed);
    }
    const long double mean_radial_velocity = radial_momentum / mass_sum;
    const long double centred_three_dimensional_variance = std::max(
        0.0L,
        speed_squared_moment / mass_sum
            - mean_radial_velocity * mean_radial_velocity);
    result.orbital_velocity_dispersion = std::sqrt(
        static_cast<double>(centred_three_dimensional_variance / 3.0L));

    const long double pair_count =
        0.5L * static_cast<long double>(count)
        * static_cast<long double>(count - 1);
    constexpr std::uint64_t requested_samples = UINT64_C(512);
    result.pair_samples = static_cast<std::uint64_t>(std::min(
        pair_count,
        static_cast<long double>(requested_samples)));
    if (result.pair_samples == 0) {
        return result;
    }

    const std::uint64_t seed =
        UINT64_C(0x6d66705f73706864)
        ^ particles[begin].id
        ^ (particles[end - 1].id + UINT64_C(0x9e3779b97f4a7c15))
        ^ static_cast<std::uint64_t>(begin)
        ^ (static_cast<std::uint64_t>(end) << 32);
    long double relative_speed_sum = 0.0L;
    long double collision_kernel_sum = 0.0L;
    auto sample_pair = [&](std::size_t first_local,
                           std::size_t second_local,
                           std::uint64_t sample) {
        const SphericalDSMCParticle& first =
            particles[begin + first_local];
        const SphericalDSMCParticle& second =
            particles[begin + second_local];
        const double first_mass =
            spherical_dsmc_particle_mass(first, particle_mass);
        const double second_mass =
            spherical_dsmc_particle_mass(second, particle_mass);
        const double relative_azimuth =
            2.0 * PI * dsmc_splitmix_uniform_open_at(
                seed, 3 * sample + 2);
        const double first_tangential =
            spherical_dsmc_tangential_speed(first);
        const double second_tangential =
            spherical_dsmc_tangential_speed(second);
        const double radial_difference = first.vr - second.vr;
        double relative_speed = std::sqrt(std::max(
            0.0,
            radial_difference * radial_difference
                + first_tangential * first_tangential
                + second_tangential * second_tangential
                - 2.0 * first_tangential * second_tangential
                    * std::cos(relative_azimuth)));
        if (first.r == 0.0 || second.r == 0.0) {
            DSMCVelocity first_velocity{first.vr, first_tangential, 0.0};
            DSMCVelocity second_velocity{second.vr,
                second_tangential * std::cos(relative_azimuth),
                second_tangential * std::sin(relative_azimuth)};
            const auto origin_draw = [&](std::uint64_t dimension) {
                return dsmc_splitmix_uniform_open_at(
                    seed ^ UINT64_C(0x6f726967696e5f76), 4 * sample + dimension);
            };
            if (first.r == 0.0) first_velocity = spherical_dsmc_origin_velocity(
                first.vr, 2.0 * origin_draw(0) - 1.0, 2.0 * PI * origin_draw(1));
            if (second.r == 0.0) second_velocity = spherical_dsmc_origin_velocity(
                second.vr, 2.0 * origin_draw(2) - 1.0, 2.0 * PI * origin_draw(3));
            relative_speed = bp_fa_norm(bp_fa_sub(
                dsmc_to_vec3(first_velocity), dsmc_to_vec3(second_velocity)));
        }
        relative_speed_sum += relative_speed;
        collision_kernel_sum += relative_speed
            * dsmc_pair_total_cross_section(
                relative_speed, first_mass, second_mass, p);
    };

    if (pair_count <= static_cast<long double>(requested_samples)) {
        std::uint64_t sample = 0;
        for (std::size_t first = 0; first < count; ++first) {
            for (std::size_t second = first + 1;
                 second < count;
                 ++second) {
                sample_pair(first, second, sample++);
            }
        }
    } else {
        for (std::uint64_t sample = 0;
             sample < result.pair_samples;
             ++sample) {
            const std::size_t first = std::min(
                count - 1,
                static_cast<std::size_t>(
                    dsmc_splitmix_uniform_open_at(seed, 3 * sample)
                    * static_cast<double>(count)));
            std::size_t second = std::min(
                count - 2,
                static_cast<std::size_t>(
                    dsmc_splitmix_uniform_open_at(seed, 3 * sample + 1)
                    * static_cast<double>(count - 1)));
            if (second >= first) {
                ++second;
            }
            sample_pair(first, second, sample);
        }
    }

    const double partner_number_density =
        static_cast<double>(count - 1) / volume;
    if (relative_speed_sum > 0.0L && collision_kernel_sum > 0.0L) {
        result.mean_free_path = static_cast<double>(
            relative_speed_sum
            / (static_cast<long double>(partner_number_density)
               * collision_kernel_sum));
    }
    return result;
}

// Shell geometry sets its physical volume, mass density, and finite-N
// partner density. The cross section and speed bound then determine both
// the mean-free-path comparison and a majorant for collision trials. The
// minimum population can end subdivision with a recorded nonlocal shell.
inline SphericalDSMCShell spherical_dsmc_describe_shell(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    std::size_t begin,
    std::size_t end,
    const Params& p) {
    const bool paper_exact = spherical_dsmc_paper_exact_enabled();
    // Frank & Tokayer equation (6) uses the particle extrema themselves.
    // The default VENUS path retains midpoint control-volume faces.
    const double inner_edge = paper_exact
        ? particles[begin].r
        : (begin == 0
               ? 0.0
               : 0.5 * (particles[begin - 1].r + particles[begin].r));
    const double outer_edge = paper_exact
        ? particles[end - 1].r
        : (end == particles.size()
               ? particles.back().r
               : 0.5 * (particles[end - 1].r + particles[end].r));
    const double volume = (4.0 * PI / 3.0)
        * std::max(
            outer_edge * outer_edge * outer_edge
                - inner_edge * inner_edge * inner_edge,
            1.0e-300);
    double shell_mass = 0.0;
    double minimum_mass = std::numeric_limits<double>::infinity();
    double maximum_mass = 0.0;
    double maximum_speed = 0.0;
    std::size_t maximum_speed_index = begin;
    for (std::size_t i = begin; i < end; ++i) {
        const double mass =
            spherical_dsmc_particle_mass(particles[i], particle_mass);
        shell_mass += mass;
        minimum_mass = std::min(minimum_mass, mass);
        maximum_mass = std::max(maximum_mass, mass);
        const double vt = spherical_dsmc_tangential_speed(particles[i]);
        const double speed = std::hypot(particles[i].vr, vt);
        if (speed > maximum_speed) {
            maximum_speed = speed;
            maximum_speed_index = i;
        }
    }
    const double density = shell_mass / volume;
    const double number_density =
        static_cast<double>(end - begin) / volume;
    const bool equal_marker_masses = minimum_mass == maximum_mass;
    const double partner_number_density =
        static_cast<double>(end - begin - 1) / volume;
    const SphericalDSMCMeanFreePathEstimate mean_free_path =
        paper_exact && equal_marker_masses
        ? SphericalDSMCMeanFreePathEstimate{
              partner_number_density > 0.0
                      && dsmc_pair_total_cross_section(
                             1.0, maximum_mass, maximum_mass, p) > 0.0
                  ? 1.0
                      / (partner_number_density
                         * dsmc_pair_total_cross_section(
                             1.0, maximum_mass, maximum_mass, p))
                  : std::numeric_limits<double>::infinity(),
              0.0,
              0}
        : spherical_dsmc_velocity_averaged_mean_free_path(
              particles, particle_mass, begin, end, volume, p);
    SphericalDSMCShell shell;
    shell.begin = begin;
    shell.end = end;
    shell.inner_edge = inner_edge;
    shell.outer_edge = outer_edge;
    shell.density = density;
    shell.number_density = number_density;
    shell.minimum_mass = minimum_mass;
    shell.maximum_mass = maximum_mass;
    shell.mean_free_path = mean_free_path.mean_free_path;
    shell.mean_free_path_pair_samples = mean_free_path.pair_samples;
    shell.orbital_velocity_dispersion =
        mean_free_path.orbital_velocity_dispersion;
    shell.maximum_speed = maximum_speed;
    shell.maximum_speed_index = maximum_speed_index;
    shell.majorant_u_sigma =
        dsmc_pair_u_sigma_majorant(
            (paper_exact ? 4.0 : 2.0) * maximum_speed,
            maximum_mass,
            p);
    return shell;
}

// Cache speeds and block maxima before moving the shell cursor. Reusing
// these exact extrema avoids repeated square roots during candidate halving;
// the strict comparison retains the earliest maximum when speeds tie.
inline void spherical_dsmc_prepare_paper_shell_speed_cache(
    const std::vector<SphericalDSMCParticle>& particles,
    const Params& p,
    SphericalDSMCShellWorkspace& workspace) {
    constexpr std::size_t block_size = 256;
    const std::size_t count = particles.size();
    const std::size_t block_count = (count + block_size - 1) / block_size;
    workspace.paper_speed.resize(count);
    workspace.paper_block_maximum_speed.resize(block_count);
    workspace.paper_block_maximum_speed_index.resize(block_count);
    const int nthreads = effective_threads(p, static_cast<int>(block_count));
    parallel_for_chunks(
        0,
        static_cast<int>(block_count),
        nthreads,
        [&](int lo, int hi, int) {
            for (int bb = lo; bb < hi; ++bb) {
                const std::size_t block = static_cast<std::size_t>(bb);
                const std::size_t begin = block * block_size;
                const std::size_t end = std::min(count, begin + block_size);
                double maximum_speed = 0.0;
                std::size_t maximum_speed_index = begin;
                for (std::size_t i = begin; i < end; ++i) {
                    const double speed = std::hypot(
                        particles[i].vr,
                        spherical_dsmc_tangential_speed(particles[i]));
                    workspace.paper_speed[i] = speed;
                    if (speed > maximum_speed) {
                        maximum_speed = speed;
                        maximum_speed_index = i;
                    }
                }
                workspace.paper_block_maximum_speed[block] = maximum_speed;
                workspace.paper_block_maximum_speed_index[block] =
                    maximum_speed_index;
            }
        });
}

inline std::pair<double, std::size_t>
spherical_dsmc_paper_cached_range_maximum_speed(
    std::size_t begin,
    std::size_t end,
    const SphericalDSMCShellWorkspace& workspace) {
    constexpr std::size_t block_size = 256;
    double maximum_speed = 0.0;
    std::size_t maximum_speed_index = begin;
    auto consider = [&](double speed, std::size_t index) {
        // Strict comparison reproduces the original scan's earliest-index
        // tie breaking exactly.
        if (speed > maximum_speed) {
            maximum_speed = speed;
            maximum_speed_index = index;
        }
    };

    std::size_t i = begin;
    while (i < end && i % block_size != 0) {
        consider(workspace.paper_speed[i], i);
        ++i;
    }
    while (i + block_size <= end) {
        const std::size_t block = i / block_size;
        consider(
            workspace.paper_block_maximum_speed[block],
            workspace.paper_block_maximum_speed_index[block]);
        i += block_size;
    }
    while (i < end) {
        consider(workspace.paper_speed[i], i);
        ++i;
    }
    return {maximum_speed, maximum_speed_index};
}

inline SphericalDSMCShell spherical_dsmc_describe_paper_shell_cached(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    std::size_t begin,
    std::size_t end,
    const Params& p,
    const SphericalDSMCShellWorkspace& workspace) {
    const double inner_edge = particles[begin].r;
    const double outer_edge = particles[end - 1].r;
    const double volume = (4.0 * PI / 3.0)
        * std::max(
            outer_edge * outer_edge * outer_edge
                - inner_edge * inner_edge * inner_edge,
            1.0e-300);
    double shell_mass = 0.0;
    double minimum_mass = std::numeric_limits<double>::infinity();
    double maximum_mass = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        const double mass =
            spherical_dsmc_particle_mass(particles[i], particle_mass);
        shell_mass += mass;
        minimum_mass = std::min(minimum_mass, mass);
        maximum_mass = std::max(maximum_mass, mass);
    }
    const double density = shell_mass / volume;
    const double number_density =
        static_cast<double>(end - begin) / volume;
    const auto maximum = spherical_dsmc_paper_cached_range_maximum_speed(
        begin, end, workspace);
    // The published equal-mass, constant-cross-section branch has an exact
    // algebraic MFP.  If an unequal-mass restart enters paper-exact mode, use
    // the same evolving pair average as the general path instead of silently
    // substituting the largest marker mass.
    const bool equal_marker_masses = minimum_mass == maximum_mass;
    const double partner_number_density =
        static_cast<double>(end - begin - 1) / volume;
    const double sigma = equal_marker_masses
        ? dsmc_pair_total_cross_section(
              1.0, maximum_mass, maximum_mass, p)
        : 0.0;
    const SphericalDSMCMeanFreePathEstimate mean_free_path =
        equal_marker_masses
        ? SphericalDSMCMeanFreePathEstimate{
              partner_number_density > 0.0 && sigma > 0.0
                  ? 1.0 / (partner_number_density * sigma)
                  : std::numeric_limits<double>::infinity(),
              0.0,
              0}
        : spherical_dsmc_velocity_averaged_mean_free_path(
              particles, particle_mass, begin, end, volume, p);
    SphericalDSMCShell shell;
    shell.begin = begin;
    shell.end = end;
    shell.inner_edge = inner_edge;
    shell.outer_edge = outer_edge;
    shell.density = density;
    shell.number_density = number_density;
    shell.minimum_mass = minimum_mass;
    shell.maximum_mass = maximum_mass;
    shell.mean_free_path = mean_free_path.mean_free_path;
    shell.mean_free_path_pair_samples = mean_free_path.pair_samples;
    shell.orbital_velocity_dispersion =
        mean_free_path.orbital_velocity_dispersion;
    shell.maximum_speed = maximum.first;
    shell.maximum_speed_index = maximum.second;
    shell.majorant_u_sigma = dsmc_pair_u_sigma_majorant(
        4.0 * shell.maximum_speed, maximum_mass, p);
    return shell;
}

// Bisect a radially ordered parent when its width exceeds its estimated
// mean free path and both children can retain the required population. The
// accepted leaves partition the original parent without particle duplication.
inline void spherical_dsmc_build_shell_recursive(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    std::size_t begin,
    std::size_t end,
    const Params& p,
    std::vector<SphericalDSMCShell>& shells) {
    if (end <= begin) {
        return;
    }
    const SphericalDSMCShell shell = spherical_dsmc_describe_shell(
        particles, particle_mass, begin, end, p);
    const std::size_t size = end - begin;
    const std::size_t minimum =
        static_cast<std::size_t>(p.bp_dsmc_min_shell_particles);
    if (shell.outer_edge - shell.inner_edge > shell.mean_free_path
        && size >= 2 * minimum) {
        const std::size_t middle = begin + size / 2;
        spherical_dsmc_build_shell_recursive(
            particles, particle_mass, begin, middle, p, shells);
        spherical_dsmc_build_shell_recursive(
            particles, particle_mass, middle, end, p, shells);
        return;
    }
    shells.push_back(shell);
}

inline std::vector<SphericalDSMCShell> spherical_dsmc_build_shells(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p,
    SphericalDSMCShellWorkspace& workspace) {
    const std::size_t target =
        static_cast<std::size_t>(p.bp_dsmc_shell_particles);
    const std::size_t minimum =
        static_cast<std::size_t>(p.bp_dsmc_min_shell_particles);
    const bool paper_exact = spherical_dsmc_paper_exact_enabled();
    if (paper_exact) {
        // Frank & Tokayer section 2.2 is a sequential moving cursor, not a
        // recursive partition of fixed 4096-particle parents.  At every new
        // cursor position, start again with the next N_shell=4096 remaining
        // particles.  If that candidate is wider than its local mean free
        // path, halve only the current candidate until it is local or reaches
        // the 32-particle floor.  Accept it, advance by its accepted size, and
        // reset the next candidate to N_shell=4096.
        workspace.parents.clear();
        workspace.children.clear();
        spherical_dsmc_prepare_paper_shell_speed_cache(
            particles, p, workspace);
        std::vector<SphericalDSMCShell> shells;
        shells.reserve((particles.size() + target - 1) / target);
        std::size_t begin = 0;
        while (begin < particles.size()) {
            std::size_t end = std::min(particles.size(), begin + target);
            SphericalDSMCShell shell =
                spherical_dsmc_describe_paper_shell_cached(
                    particles, particle_mass, begin, end, p, workspace);
            while (shell.outer_edge - shell.inner_edge > shell.mean_free_path
                   && end - begin >= 2 * minimum) {
                end = begin + (end - begin) / 2;
                shell = spherical_dsmc_describe_paper_shell_cached(
                    particles, particle_mass, begin, end, p, workspace);
            }
            shells.push_back(shell);
            begin = end;
        }
        return shells;
    }

    workspace.parents.clear();
    for (std::size_t begin = 0; begin < particles.size();) {
        std::size_t end = std::min(particles.size(), begin + target);
        if (particles.size() - end < minimum) {
            end = particles.size();
        }
        workspace.parents.emplace_back(begin, end);
        begin = end;
    }
    workspace.children.resize(workspace.parents.size());
    const int nthreads = effective_threads(
        p,
        static_cast<int>(workspace.parents.size()));
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(workspace.parents.size()),
        nthreads,
        1,
        [&](int lo, int hi, int) {
            for (int pp = lo; pp < hi; ++pp) {
                const std::size_t parent = static_cast<std::size_t>(pp);
                std::vector<SphericalDSMCShell>& children =
                    workspace.children[parent];
                children.clear();
                spherical_dsmc_build_shell_recursive(
                    particles,
                    particle_mass,
                    workspace.parents[parent].first,
                    workspace.parents[parent].second,
                    p,
                    children);
            }
        });

    std::size_t shell_count = 0;
    for (const std::vector<SphericalDSMCShell>& children : workspace.children) {
        shell_count += children.size();
    }
    std::vector<SphericalDSMCShell> shells;
    shells.reserve(shell_count);
    for (const std::vector<SphericalDSMCShell>& children : workspace.children) {
        shells.insert(shells.end(), children.begin(), children.end());
    }
    return shells;
}

inline std::vector<SphericalDSMCShell> spherical_dsmc_build_shells(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    const Params& p) {
    SphericalDSMCShellWorkspace workspace;
    return spherical_dsmc_build_shells(
        particles,
        particle_mass,
        p,
        workspace);
}

// Convert the shell pair-kernel bound into a per-particle encounter-rate
// bound using the finite-N partner density. The configured epsilon limits
// that rate times dt. The published branch separately divides out its
// four-times-speed trial majorant when selecting the physical timestep.
inline double spherical_dsmc_collision_timestep(
    const std::vector<SphericalDSMCShell>& shells,
    const Params& p) {
    double timestep = std::numeric_limits<double>::infinity();
    const bool paper_exact = spherical_dsmc_paper_exact_enabled();
    for (const SphericalDSMCShell& shell : shells) {
        // Frank & Tokayer equation (12) uses rho*sigma_m*v_max.  The
        // 4*v_max NTC majorant belongs only in equations (7) and (9), not in
        // the timestep limiter.
        const double number_density = shell.number_density > 0.0
            ? shell.number_density
            : shell.density;
        const std::size_t count = shell.end - shell.begin;
        const double partner_number_density = count > 1
            ? number_density * static_cast<double>(count - 1)
                / static_cast<double>(count)
            : number_density;
        const double rate = partner_number_density * shell.majorant_u_sigma
            / (paper_exact ? 4.0 : 1.0);
        if (rate > 0.0) {
            timestep = std::min(
                timestep,
                p.bp_dsmc_collision_epsilon / rate);
        }
    }
    return timestep;
}

inline double spherical_dsmc_collision_timestep_after_collisions(
    const std::vector<SphericalDSMCShell>& shells,
    const SphericalDSMCCollisionWorkspace& workspace,
    const Params& p) {
    if (workspace.stats.size() != shells.size()) {
        return spherical_dsmc_collision_timestep(shells, p);
    }
    double timestep = std::numeric_limits<double>::infinity();
    const bool paper_exact = spherical_dsmc_paper_exact_enabled();
    for (std::size_t i = 0; i < shells.size(); ++i) {
        const double number_density = shells[i].number_density > 0.0
            ? shells[i].number_density
            : shells[i].density;
        const std::size_t count = shells[i].end - shells[i].begin;
        const double partner_number_density = count > 1
            ? number_density * static_cast<double>(count - 1)
                / static_cast<double>(count)
            : number_density;
        const double rate =
            partner_number_density * workspace.stats[i].majorant_u_sigma
            / (paper_exact ? 4.0 : 1.0);
        if (rate > 0.0) {
            timestep = std::min(
                timestep,
                p.bp_dsmc_collision_epsilon / rate);
        }
    }
    return timestep;
}

// A whole expected trial contributes immediately. Its fractional remainder
// is distributed over the persistent particle identities; those contributions
// travel with particles when radial sorting changes their shell membership.
inline std::uint64_t spherical_dsmc_paper_trial_count(
    const std::vector<SphericalDSMCParticle>& particles,
    const SphericalDSMCShell& shell,
    double dt,
    std::vector<double>& remainder_accumulator_by_id) {
    const std::size_t marker_count = shell.end - shell.begin;
    const double number_density = shell.number_density > 0.0
        ? shell.number_density
        : shell.density;
    if (marker_count < 2 || !(number_density > 0.0) || !(dt > 0.0)
        || !(shell.majorant_u_sigma > 0.0)) {
        return 0;
    }
    if (remainder_accumulator_by_id.size() != particles.size()) {
        throw std::runtime_error(
            "Paper-exact DSMC remainder workspace does not match particle count.");
    }

    // Unequal-mass extension of Frank & Tokayer equation (7).  The shell
    // majorant is a physical pair kernel, so C(N,2) K_max/V is valid without
    // introducing a common marker mass.
    const double expected_trials =
        0.5 * static_cast<double>(marker_count - 1)
        * number_density * shell.majorant_u_sigma * dt;
    if (!(expected_trials >= 0.0) || !std::isfinite(expected_trials)
        || expected_trials
               > static_cast<double>(
                   std::numeric_limits<std::uint64_t>::max())) {
        throw std::runtime_error(
            "Paper-exact DSMC trial count is non-finite or overflows uint64.");
    }
    const double integral = std::floor(expected_trials);
    std::uint64_t trial_count = static_cast<std::uint64_t>(integral);
    const double fractional = expected_trials - integral;
    const double increment =
        fractional / static_cast<double>(marker_count);

    // Frank & Tokayer equation (8): the fractional count is distributed
    // equally among the particles currently in this shell, while each A_i
    // remains attached to its particle as shell membership evolves.
    double shell_accumulator = 0.0;
    for (std::size_t i = shell.begin; i < shell.end; ++i) {
        const std::uint64_t id = particles[i].id;
        if (id >= remainder_accumulator_by_id.size()) {
            throw std::runtime_error(
                "Paper-exact DSMC particle id is outside remainder workspace.");
        }
        double& value =
            remainder_accumulator_by_id[static_cast<std::size_t>(id)];
        value += increment;
        shell_accumulator += value;
    }
    if (!std::isfinite(shell_accumulator)) {
        throw std::runtime_error(
            "Paper-exact DSMC remainder accumulator became non-finite.");
    }
    if (shell_accumulator > 1.0) {
        if (trial_count == std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error(
                "Paper-exact DSMC remainder trial overflows uint64.");
        }
        ++trial_count;
        const double decrement =
            1.0 / static_cast<double>(marker_count);
        for (std::size_t i = shell.begin; i < shell.end; ++i) {
            remainder_accumulator_by_id[
                static_cast<std::size_t>(particles[i].id)] -= decrement;
        }
    }
    return trial_count;
}

// Choose two distinct partners uniformly, reconstruct their tangential
// directions for this trial, and accept with u*sigma/majorant. Scattering
// rotates their relative velocity in the centre-of-mass frame, preserving
// pair momentum and kinetic energy with the actual two inertial masses.
inline DSMCCollisionStats spherical_dsmc_apply_shell_collisions_paper_exact(
    std::vector<SphericalDSMCParticle>& particles,
    const SphericalDSMCShell& shell,
    double legacy_particle_mass,
    double dt,
    const Params& p,
    std::uint64_t collision_seed,
    std::vector<double>& remainder_accumulator_by_id) {
    DSMCCollisionStats stats;
    const std::size_t marker_count = shell.end - shell.begin;
    if (marker_count < 2 || !(shell.number_density > 0.0) || !(dt > 0.0)
        || p.bp_collision_strength == 0.0) {
        return stats;
    }

    // Equation (7) uses the fixed 4*max|v| majorant measured at the start of
    // this shell step.  The shell builder has already selected that bound.
    stats.majorant_u_sigma = shell.majorant_u_sigma;
    if (!(stats.majorant_u_sigma > 0.0)
        || !std::isfinite(stats.majorant_u_sigma)) {
        return stats;
    }
    const std::uint64_t trial_count = spherical_dsmc_paper_trial_count(
        particles,
        shell,
        dt,
        remainder_accumulator_by_id);
    stats.trial_pairs = trial_count;

    DSMCSplitMix64 rng(collision_seed);
    for (std::uint64_t trial = 0; trial < trial_count; ++trial) {
        const std::size_t ia = static_cast<std::size_t>(
            rng.uniform_index(static_cast<std::uint64_t>(marker_count)));
        std::size_t ib = static_cast<std::size_t>(
            rng.uniform_index(static_cast<std::uint64_t>(marker_count - 1)));
        if (ib >= ia) {
            ++ib;
        }

        SphericalDSMCParticle& particle_a = particles[shell.begin + ia];
        SphericalDSMCParticle& particle_b = particles[shell.begin + ib];
        const double mass_a =
            spherical_dsmc_particle_mass(particle_a, legacy_particle_mass);
        const double mass_b =
            spherical_dsmc_particle_mass(particle_b, legacy_particle_mass);
        const double pair_mass = mass_a + mass_b;
        const double tangential_a =
            spherical_dsmc_tangential_speed(particle_a);
        const double tangential_b =
            spherical_dsmc_tangential_speed(particle_b);
        // Frank & Tokayer draw fresh azimuths whenever a trial pair is
        // evaluated; tangential directions are not retained between pairs.
        const double angle_a = 2.0 * PI * rng.uniform_open();
        const double angle_b = 2.0 * PI * rng.uniform_open();
        BPFAVec3 v1 = {
            particle_a.vr,
            tangential_a * std::cos(angle_a),
            tangential_a * std::sin(angle_a),
        };
        BPFAVec3 v2 = {
            particle_b.vr,
            tangential_b * std::cos(angle_b),
            tangential_b * std::sin(angle_b),
        };
        if (particle_a.r == 0.0) {
            v1 = dsmc_to_vec3(spherical_dsmc_origin_velocity(
                particle_a.vr, 2.0 * rng.uniform_open() - 1.0, angle_a));
        }
        if (particle_b.r == 0.0) {
            v2 = dsmc_to_vec3(spherical_dsmc_origin_velocity(
                particle_b.vr, 2.0 * rng.uniform_open() - 1.0, angle_b));
        }
        const BPFAVec3 relative = bp_fa_sub(v1, v2);
        const double relative_speed = bp_fa_norm(relative);
        const double u_sigma = relative_speed
            * dsmc_pair_total_cross_section(
                relative_speed, mass_a, mass_b, p);
        if (!(u_sigma > 0.0)) {
            continue;
        }
        if (u_sigma > stats.majorant_u_sigma
            * (1.0 + 64.0 * std::numeric_limits<double>::epsilon())) {
            throw std::runtime_error(
                "Paper-exact spherical DSMC |u| sigma majorant was violated.");
        }
        if (rng.uniform_open() * stats.majorant_u_sigma >= u_sigma) {
            continue;
        }

        const BPFAVec3 pair_momentum_before = bp_fa_add(
            bp_fa_scale(v1, mass_a),
            bp_fa_scale(v2, mass_b));
        const double pair_energy_before =
            0.5 * mass_a * bp_fa_dot(v1, v1)
            + 0.5 * mass_b * bp_fa_dot(v2, v2);
        const BPFAVec3 centre =
            bp_fa_scale(pair_momentum_before, 1.0 / pair_mass);
        const double mu = dsmc_sample_scatter_mu(relative_speed, p, rng);
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
            bp_fa_scale(outgoing_direction, relative_speed);
        const BPFAVec3 out1 = bp_fa_add(
            centre, bp_fa_scale(relative_after, mass_b / pair_mass));
        const BPFAVec3 out2 = bp_fa_sub(
            centre, bp_fa_scale(relative_after, mass_a / pair_mass));
        spherical_dsmc_store_collision_velocity(particle_a, dsmc_from_vec3(out1));
        spherical_dsmc_store_collision_velocity(particle_b, dsmc_from_vec3(out2));
        ++stats.accepted_pairs;

        const BPFAVec3 pair_momentum_after = bp_fa_add(
            bp_fa_scale(out1, mass_a),
            bp_fa_scale(out2, mass_b));
        const double pair_energy_after =
            0.5 * mass_a * bp_fa_dot(out1, out1)
            + 0.5 * mass_b * bp_fa_dot(out2, out2);
        const double pair_momentum_error =
            bp_fa_norm(bp_fa_sub(
                pair_momentum_after,
                pair_momentum_before));
        const double pair_kinetic_error =
            pair_energy_after - pair_energy_before;
        stats.momentum_error_x +=
            pair_momentum_after.x - pair_momentum_before.x;
        stats.momentum_error_y +=
            pair_momentum_after.y - pair_momentum_before.y;
        stats.momentum_error_z +=
            pair_momentum_after.z - pair_momentum_before.z;
        stats.kinetic_energy_error += pair_kinetic_error;
        stats.max_pair_momentum_error = std::max(
            stats.max_pair_momentum_error,
            pair_momentum_error);
        stats.max_pair_kinetic_energy_error = std::max(
            stats.max_pair_kinetic_energy_error,
            std::abs(pair_kinetic_error));
    }

    // Frank's SCATTER2 computes DTCOLLMIN from the shell VMAX measured before
    // drawing any pairs and carries that value into SETDT.  It does not scan
    // every shell a second time after collisions.  Retaining the original
    // majorant is therefore both the literal algorithm and avoids a redundant
    // O(N) hypot/divide pass per global step.
    stats.majorant_u_sigma = shell.majorant_u_sigma;
    stats.represented_event_density =
        static_cast<double>(stats.accepted_pairs)
        * shell.number_density / static_cast<double>(marker_count);
    return stats;
}

// The general sampler advances an exponential trial clock at
// C(N,2)*majorant/V. A particle receives its tangential orientation on first
// use and retains its updated three-velocity during this shell step. Only
// changed records need conversion back to (v_r,L) at the end.
inline DSMCCollisionStats spherical_dsmc_apply_shell_collisions_lazy(
    std::vector<SphericalDSMCParticle>& particles,
    const SphericalDSMCShell& shell,
    double legacy_particle_mass,
    double dt,
    const Params& p,
    std::uint64_t orientation_seed,
    std::uint64_t collision_seed,
    SphericalDSMCCollisionWorkspace::ThreadVelocityCache& cache) {
    DSMCCollisionStats stats;
    const std::size_t marker_count = shell.end - shell.begin;
    if (marker_count < 2 || !(shell.number_density > 0.0) || !(dt > 0.0)
        || p.bp_collision_strength == 0.0) {
        return stats;
    }

    cache.velocities.resize(marker_count);
    cache.initialized.assign(marker_count, 0);
    cache.changed.assign(marker_count, 0);
    cache.changed_indices.clear();
    auto velocity = [&](std::size_t local) -> DSMCVelocity& {
        if (cache.initialized[local] == 0) {
            const SphericalDSMCParticle& particle =
                particles[shell.begin + local];
            const double vt = spherical_dsmc_tangential_speed(particle);
            const double angle = 2.0 * PI
                * dsmc_splitmix_uniform_open_at(
                    orientation_seed,
                    static_cast<std::uint64_t>(local));
            cache.velocities[local] = {
                particle.vr,
                vt * std::cos(angle),
                vt * std::sin(angle),
            };
            if (particle.r == 0.0) {
                cache.velocities[local] = spherical_dsmc_origin_velocity(
                    particle.vr,
                    2.0 * dsmc_splitmix_uniform_open_at(
                        orientation_seed ^ UINT64_C(0x6f726967696e5f76),
                        static_cast<std::uint64_t>(local)) - 1.0,
                    angle);
            }
            cache.initialized[local] = 1;
        }
        return cache.velocities[local];
    };
    auto mark_changed = [&](std::size_t local) {
        if (cache.changed[local] == 0) {
            cache.changed[local] = 1;
            cache.changed_indices.push_back(local);
        }
    };

    stats.majorant_u_sigma = shell.majorant_u_sigma;
    if (!(stats.majorant_u_sigma > 0.0)
        || !std::isfinite(stats.majorant_u_sigma)) {
        return stats;
    }
    DSMCSplitMix64 rng(collision_seed);
    double collision_clock = 0.0;
    double speed_bound = shell.maximum_speed;
    for (;;) {
        const double trial_rate = 0.5
            * static_cast<double>(marker_count - 1)
            * shell.number_density * stats.majorant_u_sigma;
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

        const BPFAVec3 v1 = dsmc_to_vec3(velocity(ia));
        const BPFAVec3 v2 = dsmc_to_vec3(velocity(ib));
        const double mass_a = spherical_dsmc_particle_mass(
            particles[shell.begin + ia], legacy_particle_mass);
        const double mass_b = spherical_dsmc_particle_mass(
            particles[shell.begin + ib], legacy_particle_mass);
        const double pair_mass = mass_a + mass_b;
        const BPFAVec3 relative = bp_fa_sub(v1, v2);
        const double u = bp_fa_norm(relative);
        const double u_sigma = u * dsmc_pair_total_cross_section(
            u, mass_a, mass_b, p);
        if (!(u_sigma > 0.0)) {
            continue;
        }
        if (u_sigma > stats.majorant_u_sigma
            * (1.0 + 64.0 * std::numeric_limits<double>::epsilon())) {
            throw std::runtime_error(
                "Lazy spherical DSMC |u| sigma majorant was violated.");
        }
        if (rng.uniform_open() * stats.majorant_u_sigma >= u_sigma) {
            continue;
        }

        const BPFAVec3 pair_momentum_before = bp_fa_add(
            bp_fa_scale(v1, mass_a),
            bp_fa_scale(v2, mass_b));
        const double pair_energy_before =
            0.5 * mass_a * bp_fa_dot(v1, v1)
            + 0.5 * mass_b * bp_fa_dot(v2, v2);
        const BPFAVec3 centre =
            bp_fa_scale(pair_momentum_before, 1.0 / pair_mass);
        const double mu = dsmc_sample_scatter_mu(u, p, rng);
        const double psi = 2.0 * PI * rng.uniform_open();
        BPFAVec3 outgoing_direction;
        if (bp_fa_kernel_name(p) == "yukawa_born") {
            outgoing_direction =
                bp_fa_scattered_relative_direction(relative, mu, psi);
        } else {
            const double sint = std::sqrt(std::max(0.0, 1.0 - mu * mu));
            outgoing_direction = {
                mu,
                sint * std::cos(psi),
                sint * std::sin(psi),
            };
        }
        const BPFAVec3 relative_after =
            bp_fa_scale(outgoing_direction, u);
        const BPFAVec3 out1 = bp_fa_add(
            centre, bp_fa_scale(relative_after, mass_b / pair_mass));
        const BPFAVec3 out2 = bp_fa_sub(
            centre, bp_fa_scale(relative_after, mass_a / pair_mass));
        cache.velocities[ia] = dsmc_from_vec3(out1);
        cache.velocities[ib] = dsmc_from_vec3(out2);
        mark_changed(ia);
        mark_changed(ib);
        ++stats.accepted_pairs;

        const double new_speed = std::max(bp_fa_norm(out1), bp_fa_norm(out2));
        // Elastic redistribution can increase one partner speed. Enlarge the
        // future trial majorant before drawing the next exponential waiting time;
        // this preserves an upper bound for subsequent acceptance probabilities.
        if (new_speed > speed_bound) {
            speed_bound = new_speed;
            stats.majorant_u_sigma = std::max(
                stats.majorant_u_sigma,
                dsmc_pair_u_sigma_majorant(
                    2.0 * speed_bound, shell.maximum_mass, p));
        }
        const BPFAVec3 pair_momentum_after = bp_fa_add(
            bp_fa_scale(out1, mass_a),
            bp_fa_scale(out2, mass_b));
        const double pair_energy_after =
            0.5 * mass_a * bp_fa_dot(out1, out1)
            + 0.5 * mass_b * bp_fa_dot(out2, out2);
        stats.max_pair_momentum_error = std::max(
            stats.max_pair_momentum_error,
            bp_fa_norm(bp_fa_sub(
                pair_momentum_after,
                pair_momentum_before)));
        stats.max_pair_kinetic_energy_error = std::max(
            stats.max_pair_kinetic_energy_error,
            std::abs(pair_energy_after - pair_energy_before));
    }

    bool original_maximum_changed = false;
    double maximum_changed_speed = 0.0;
    for (const std::size_t local : cache.changed_indices) {
        SphericalDSMCParticle& particle = particles[shell.begin + local];
        const double radial_before = particle.vr;
        const double kinetic_before =
            spherical_dsmc_specific_kinetic_energy(particle);
        const double mass =
            spherical_dsmc_particle_mass(particle, legacy_particle_mass);
        spherical_dsmc_store_collision_velocity(particle, cache.velocities[local]);
        const double kinetic_after =
            spherical_dsmc_specific_kinetic_energy(particle);
        stats.momentum_error_x += mass * (particle.vr - radial_before);
        stats.kinetic_energy_error +=
            mass * (kinetic_after - kinetic_before);
        maximum_changed_speed = std::max(
            maximum_changed_speed,
            std::sqrt(2.0 * kinetic_after));
        original_maximum_changed =
            original_maximum_changed
            || shell.begin + local == shell.maximum_speed_index;
    }

    double final_maximum_speed =
        std::max(shell.maximum_speed, maximum_changed_speed);
    if (original_maximum_changed) {
        final_maximum_speed = 0.0;
        for (std::size_t i = shell.begin; i < shell.end; ++i) {
            final_maximum_speed = std::max(
                final_maximum_speed,
                std::hypot(
                    particles[i].vr,
                    spherical_dsmc_tangential_speed(particles[i])));
        }
    }
    stats.majorant_u_sigma =
        dsmc_pair_u_sigma_majorant(
            2.0 * final_maximum_speed, shell.maximum_mass, p);
    stats.represented_event_density =
        static_cast<double>(stats.accepted_pairs)
        * shell.number_density / static_cast<double>(marker_count);
    return stats;
}

// Adapters for particle records with a common shell mass.
// Zero-mass records use the common shell mass from their input.
inline DSMCCollisionStats spherical_dsmc_apply_shell_collisions_paper_exact(
    std::vector<SphericalDSMCParticle>& particles,
    const SphericalDSMCShell& shell,
    double dt,
    const Params& p,
    std::uint64_t collision_seed,
    std::vector<double>& remainder_accumulator_by_id) {
    SphericalDSMCShell compatible_shell = shell;
    if (!(compatible_shell.number_density > 0.0)) {
        compatible_shell.number_density = compatible_shell.density;
    }
    const double legacy_mass = compatible_shell.number_density > 0.0
        ? compatible_shell.density / compatible_shell.number_density
        : 1.0;
    return spherical_dsmc_apply_shell_collisions_paper_exact(
        particles,
        compatible_shell,
        legacy_mass,
        dt,
        p,
        collision_seed,
        remainder_accumulator_by_id);
}

inline DSMCCollisionStats spherical_dsmc_apply_shell_collisions_lazy(
    std::vector<SphericalDSMCParticle>& particles,
    const SphericalDSMCShell& shell,
    double dt,
    const Params& p,
    std::uint64_t orientation_seed,
    std::uint64_t collision_seed,
    SphericalDSMCCollisionWorkspace::ThreadVelocityCache& cache) {
    SphericalDSMCShell compatible_shell = shell;
    if (!(compatible_shell.number_density > 0.0)) {
        compatible_shell.number_density = compatible_shell.density;
    }
    if (!(compatible_shell.maximum_mass > 0.0)) {
        compatible_shell.maximum_mass = 1.0;
    }
    const double legacy_mass = compatible_shell.number_density > 0.0
        ? compatible_shell.density / compatible_shell.number_density
        : 1.0;
    return spherical_dsmc_apply_shell_collisions_lazy(
        particles,
        compatible_shell,
        legacy_mass,
        dt,
        p,
        orientation_seed,
        collision_seed,
        cache);
}

// Collision shells contain disjoint records, so their pair updates run
// independently. Reduce statistics afterward in shell order. Radial momentum
// is a spherical diagnostic; each sampled three-dimensional pair carries
// the local vector momentum conservation law.
inline SphericalDSMCDiagnostics spherical_dsmc_apply_collisions(
    std::vector<SphericalDSMCParticle>& particles,
    const std::vector<SphericalDSMCShell>& shells,
    double particle_mass,
    double dt,
    int step,
    const Params& p,
    SphericalDSMCCollisionWorkspace& workspace) {
    SphericalDSMCDiagnostics total;
    total.collision_shells = shells.size();
    if (shells.empty() || !(dt > 0.0) || p.bp_collision_strength == 0.0) {
        return total;
    }
    workspace.stats.assign(shells.size(), DSMCCollisionStats{});
    workspace.shell_sizes.assign(shells.size(), 0);
    const bool paper_exact = spherical_dsmc_paper_exact_enabled();
    if (paper_exact
        && workspace.paper_remainder_accumulator_by_id.size()
               != particles.size()) {
        workspace.paper_remainder_accumulator_by_id.assign(
            particles.size(),
            0.0);
    }
    const int nthreads = effective_threads(
        p,
        static_cast<int>(shells.size()));
    workspace.thread_cache.resize(static_cast<std::size_t>(nthreads));
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(shells.size()),
        nthreads,
        1,
        [&](int lo, int hi, int tid) {
            SphericalDSMCCollisionWorkspace::ThreadVelocityCache& cache =
                workspace.thread_cache[static_cast<std::size_t>(tid)];
            for (int ss = lo; ss < hi; ++ss) {
                const std::size_t shell_index = static_cast<std::size_t>(ss);
                const SphericalDSMCShell& shell = shells[shell_index];
                const std::size_t count = shell.end - shell.begin;
                workspace.shell_sizes[shell_index] = count;
                if (paper_exact) {
                    workspace.stats[shell_index] =
                        spherical_dsmc_apply_shell_collisions_paper_exact(
                            particles,
                            shell,
                            particle_mass,
                            dt,
                            p,
                            spherical_dsmc_seed(
                                p.bp_dsmc_seed,
                                static_cast<std::uint64_t>(step),
                                shell_index),
                            workspace.paper_remainder_accumulator_by_id);
                } else {
                    workspace.stats[shell_index] =
                        spherical_dsmc_apply_shell_collisions_lazy(
                            particles,
                            shell,
                            particle_mass,
                            dt,
                            p,
                            spherical_dsmc_seed(
                                p.bp_dsmc_seed,
                                static_cast<std::uint64_t>(step),
                                UINT64_C(2) * shell_index),
                            spherical_dsmc_seed(
                                p.bp_dsmc_seed,
                                static_cast<std::uint64_t>(step),
                                UINT64_C(2) * shell_index + UINT64_C(1)),
                            cache);
                }
            }
        });
    for (std::size_t i = 0; i < shells.size(); ++i) {
        total.collision_trials += workspace.stats[i].trial_pairs;
        total.collision_accepted += workspace.stats[i].accepted_pairs;
        total.collision_radial_momentum_error +=
            workspace.stats[i].momentum_error_x;
        total.collision_kinetic_error +=
            workspace.stats[i].kinetic_energy_error;
        total.max_pair_momentum_error = std::max(
            total.max_pair_momentum_error,
            workspace.stats[i].max_pair_momentum_error);
        total.max_pair_kinetic_error = std::max(
            total.max_pair_kinetic_error,
            workspace.stats[i].max_pair_kinetic_energy_error);
        total.minimum_shell_particles =
            total.minimum_shell_particles == 0
            ? workspace.shell_sizes[i]
            : std::min(total.minimum_shell_particles, workspace.shell_sizes[i]);
        total.minimum_mean_free_path = std::min(
            total.minimum_mean_free_path,
            shells[i].mean_free_path);
        total.mean_free_path_pair_samples +=
            shells[i].mean_free_path_pair_samples;
        total.minimum_orbital_velocity_dispersion = std::min(
            total.minimum_orbital_velocity_dispersion,
            shells[i].orbital_velocity_dispersion);
        total.maximum_orbital_velocity_dispersion = std::max(
            total.maximum_orbital_velocity_dispersion,
            shells[i].orbital_velocity_dispersion);
        const double width =
            shells[i].outer_edge - shells[i].inner_edge;
        const double ratio = std::isfinite(shells[i].mean_free_path)
            && shells[i].mean_free_path > 0.0
            ? width / shells[i].mean_free_path
            : 0.0;
        total.maximum_shell_width_over_mean_free_path = std::max(
            total.maximum_shell_width_over_mean_free_path,
            ratio);
        total.nonlocal_collision_shells += ratio > 1.0 ? 1U : 0U;
    }
    return total;
}

inline SphericalDSMCDiagnostics spherical_dsmc_apply_collisions(
    std::vector<SphericalDSMCParticle>& particles,
    const std::vector<SphericalDSMCShell>& shells,
    double particle_mass,
    double dt,
    int step,
    const Params& p) {
    SphericalDSMCCollisionWorkspace workspace;
    return spherical_dsmc_apply_collisions(
        particles,
        shells,
        particle_mass,
        dt,
        step,
        p,
        workspace);
}

// Save the accepted ordered input and its energy by persistent identity.
// The first half kick uses the initial enclosed mass, drift changes radius
// rank, and the final half kick uses the newly sorted enclosed mass.
// Angular momentum remains constant during this central-force transport.
inline SphericalDSMCDiagnostics spherical_dsmc_gravity_step(
    std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    double dt,
    const Params& p,
    SphericalDSMCGravityWorkspace& workspace) {
    SphericalDSMCDiagnostics diagnostics;
    workspace.paper_failure_message.clear();
    const std::size_t count = particles.size();
    const bool paper_exact = spherical_dsmc_paper_exact_enabled();
    const double energy_tolerance = paper_exact
        ? 0.01
        : p.vp_orbit_energy_tolerance;
    const int nthreads = effective_threads(p, static_cast<int>(count));
    workspace.energy_before.resize(count);
    workspace.energy_scale_before.resize(count);
    workspace.initial_by_rank.resize(count);
    workspace.rank_by_id.resize(count);
    spherical_dsmc_enclosed_force_masses(
        particles,
        particle_mass,
        p,
        workspace.initial_enclosed_force_mass);
    spherical_dsmc_orbit_audit_specific_potential(
        particles,
        particle_mass,
        p,
        workspace.paper_outer_sum,
        workspace.paper_potential);
    parallel_for_chunks(
        0,
        static_cast<int>(count),
        nthreads,
        [&](int lo, int hi, int) {
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                const std::uint64_t id = particles[i].id;
                const double kinetic =
                    spherical_dsmc_specific_kinetic_energy(particles[i]);
                workspace.energy_before[i] =
                    kinetic + workspace.paper_potential[i];
                workspace.energy_scale_before[i] =
                    kinetic + std::abs(workspace.paper_potential[i]);
                workspace.initial_by_rank[i] = particles[i];
                workspace.rank_by_id[id] = i;
            }
        });

    parallel_for_chunks(
        0,
        static_cast<int>(count),
        nthreads,
        [&](int lo, int hi, int) {
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                particles[i].vr += 0.5 * dt
                    * spherical_dsmc_acceleration(
                        particles[i],
                        workspace.initial_enclosed_force_mass[i],
                        p);
                particles[i].r += dt * particles[i].vr;
                if (particles[i].r < 0.0) {
                    particles[i].r = -particles[i].r;
                    particles[i].vr = -particles[i].vr;
                }
                particles[i].r = std::max(particles[i].r, 1.0e-300);
            }
        });
    spherical_dsmc_sort_particles_parallel(
        particles,
        p,
        workspace.sort);
    spherical_dsmc_enclosed_force_masses(
        particles,
        particle_mass,
        p,
        workspace.enclosed_force_mass);
    parallel_for_chunks(
        0,
        static_cast<int>(count),
        nthreads,
        [&](int lo, int hi, int) {
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                particles[i].vr += 0.5 * dt
                    * spherical_dsmc_acceleration(
                        particles[i],
                        workspace.enclosed_force_mass[i],
                        p);
            }
        });

    // An invalid radial endpoint restores the complete saved input before
    // raising an error. This keeps the caller from receiving a partly updated
    // ensemble when an origin passage remains unresolved.
    auto validate_endpoint = [&]() {
        for (const SphericalDSMCParticle& particle : particles) {
            if (!(particle.r > 0.0) || !std::isfinite(particle.r)
                || !std::isfinite(particle.vr)
                || !(particle.L >= 0.0) || !std::isfinite(particle.L)) {
                const std::uint64_t id = particle.id;
                // This workspace already contains the full accepted input,
                // including its radius ordering. An unresolved passage through
                // the origin must leave that state available to the caller.
                particles = workspace.initial_by_rank;
                throw std::runtime_error(
                    "Spherical DSMC orbit endpoint is nonfinite or outside "
                    "its radial/angular domain; input state restored for "
                    "particle id=" + std::to_string(id));
            }
        }
    };
    if (p.vp_orbit_regularization == "off") {
        validate_endpoint();
        return diagnostics;
    }
    spherical_dsmc_orbit_audit_specific_potential(
        particles,
        particle_mass,
        p,
        workspace.paper_outer_sum,
        workspace.paper_potential);
    workspace.failed_by_thread.resize(static_cast<std::size_t>(nthreads));
    workspace.failed_ids_by_thread.resize(static_cast<std::size_t>(nthreads));
    workspace.maximum_error_by_thread.assign(
        static_cast<std::size_t>(nthreads),
        0.0);
    parallel_for_chunks(
        0,
        static_cast<int>(count),
        nthreads,
        [&](int lo, int hi, int tid) {
            std::vector<std::size_t>& local_failed =
                workspace.failed_by_thread[static_cast<std::size_t>(tid)];
            std::vector<std::uint64_t>& local_ids =
                workspace.failed_ids_by_thread[static_cast<std::size_t>(tid)];
            local_failed.clear();
            local_ids.clear();
            double maximum_error = 0.0;
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                const std::uint64_t id = particles[i].id;
                const double after =
                    spherical_dsmc_specific_kinetic_energy(particles[i])
                    + workspace.paper_potential[i];
                const std::size_t initial_rank = workspace.rank_by_id[id];
                const double before = workspace.energy_before[initial_rank];
                const double relative = paper_exact
                    ? spherical_dsmc_paper_orbit_energy_defect(
                          after,
                          before,
                          workspace.energy_scale_before[initial_rank],
                          energy_tolerance)
                    : std::abs(after - before)
                        / std::max(std::abs(before), 1.0e-30);
                maximum_error = std::max(maximum_error, relative);
                if (p.vp_orbit_regularization == "always"
                    || !std::isfinite(relative)
                    || relative > energy_tolerance) {
                    local_failed.push_back(i);
                    local_ids.push_back(id);
                }
            }
            workspace.maximum_error_by_thread[static_cast<std::size_t>(tid)] =
                maximum_error;
        });
    workspace.failed.clear();
    workspace.failed_ids.clear();
    std::size_t failed_count = 0;
    for (const std::vector<std::size_t>& local : workspace.failed_by_thread) {
        failed_count += local.size();
    }
    workspace.failed.reserve(failed_count);
    workspace.failed_ids.reserve(failed_count);
    for (int tid = 0; tid < nthreads; ++tid) {
        const std::vector<std::size_t>& local_failed =
            workspace.failed_by_thread[static_cast<std::size_t>(tid)];
        const std::vector<std::uint64_t>& local_ids =
            workspace.failed_ids_by_thread[static_cast<std::size_t>(tid)];
        workspace.failed.insert(
            workspace.failed.end(),
            local_failed.begin(),
            local_failed.end());
        workspace.failed_ids.insert(
            workspace.failed_ids.end(),
            local_ids.begin(),
            local_ids.end());
    }
    diagnostics.max_ordinary_orbit_energy_error = *std::max_element(
        workspace.maximum_error_by_thread.begin(),
        workspace.maximum_error_by_thread.end());
    diagnostics.lc_retries = static_cast<long long>(workspace.failed.size());
    if (workspace.failed.empty()) {
        validate_endpoint();
        return diagnostics;
    }

    // Retry selected identities from their saved pre-step states. Their
    // Levi-Civita characteristics hold the initial enclosed force mass fixed.
    // Rebuilding the full endpoint potential after sorting tests the resulting
    // orbits against the same ensemble energy criterion.
    const std::size_t retry_count = workspace.failed_ids.size();
    workspace.retry_converged.assign(retry_count, 0);
    workspace.retry_accepted.assign(retry_count, 0);
    workspace.retry_best_relative_error.assign(
        retry_count,
        std::numeric_limits<double>::infinity());
    workspace.final_rank_by_id.resize(count);
    parallel_for_chunks(
        0,
        static_cast<int>(count),
        nthreads,
        [&](int lo, int hi, int) {
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                workspace.final_rank_by_id[particles[i].id] = i;
            }
        });

    // Frank & Tokayer repeat the LC step, refining it as necessary, until
    // the full equation-(5) particle energy passes the epsilon_crit test.
    // Re-sort and rebuild Phi after every refinement because an LC retry can
    // change a particle's rank and therefore the mean-field potential.
    for (int refinement = 0;
         refinement <= p.vp_orbit_lc_max_refinements;
         ++refinement) {
        const int tau_subdivisions = 1 << std::min(refinement, 30);
        parallel_for_dynamic_chunks(
            0,
            static_cast<int>(retry_count),
            effective_threads(p, static_cast<int>(retry_count)),
            1,
            [&](int lo, int hi, int) {
                for (int ff = lo; ff < hi; ++ff) {
                    const std::size_t retry = static_cast<std::size_t>(ff);
                    if (workspace.retry_accepted[retry] != 0) {
                        continue;
                    }
                    const std::uint64_t id = workspace.failed_ids[retry];
                    workspace.retry_converged[retry] = 0;
                    const std::size_t current_rank =
                        workspace.final_rank_by_id[id];
                    SphericalDSMCParticle& particle = particles[current_rank];
                    const std::size_t retry_rank = workspace.rank_by_id[id];
                    const SphericalDSMCParticle initial =
                        workspace.initial_by_rank[retry_rank];
                    const double enclosed =
                        workspace.initial_enclosed_force_mass[retry_rank];
                    auto rhs = [&](const CharacteristicState& state) {
                        const double radius = std::max(state.r, 1.0e-300);
                        return CharacteristicDerivative{
                            state.w,
                            initial.L * initial.L / (radius * radius * radius)
                                - p.vp_gravity * enclosed / (radius * radius),
                        };
                    };
                    const RegularizedOrbitResult trial = paper_exact
                        ? advance_radial_characteristic_levi_civita_leapfrog(
                              {initial.r, initial.vr},
                              dt,
                              rhs,
                              tau_subdivisions,
                              // Appendix-C.2 reference implementation uses
                              // a 200-million-step safety ceiling.  A bound
                              // proportional only to N_tau can terminate a
                              // highly eccentric passage before accumulated
                              // physical time reaches dt.
                              std::max(
                                  p.vp_orbit_lc_max_steps,
                                  200000000))
                        : advance_radial_characteristic_levi_civita(
                              {initial.r, initial.vr},
                              dt,
                              rhs,
                              p.vp_orbit_lc_relative_tolerance
                                  * std::pow(0.1, refinement),
                              p.vp_orbit_lc_absolute_tolerance
                                  * std::pow(0.1, refinement),
                              p.vp_orbit_lc_max_steps);
                    if (trial.converged) {
                        particle.r = std::max(trial.state.r, 1.0e-300);
                        particle.vr = trial.state.w;
                        workspace.retry_converged[retry] = 1;
                    }
                }
            });
        spherical_dsmc_sort_particles_parallel(
            particles,
            p,
            workspace.sort);
        parallel_for_chunks(
            0,
            static_cast<int>(count),
            nthreads,
            [&](int lo, int hi, int) {
                for (int ii = lo; ii < hi; ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    workspace.final_rank_by_id[particles[i].id] = i;
                }
            });
        spherical_dsmc_orbit_audit_specific_potential(
            particles,
            particle_mass,
            p,
            workspace.paper_outer_sum,
            workspace.paper_potential);

        std::size_t unresolved = 0;
        for (std::size_t retry = 0; retry < retry_count; ++retry) {
            const std::uint64_t id = workspace.failed_ids[retry];
            const std::size_t rank = workspace.final_rank_by_id[id];
            const double after =
                spherical_dsmc_specific_kinetic_energy(particles[rank])
                + workspace.paper_potential[rank];
            const std::size_t initial_rank = workspace.rank_by_id[id];
            const double before = workspace.energy_before[initial_rank];
            const double relative = paper_exact
                ? spherical_dsmc_paper_orbit_energy_defect(
                      after,
                      before,
                      workspace.energy_scale_before[initial_rank],
                      energy_tolerance)
                : std::abs(after - before)
                    / std::max(std::abs(before), 1.0e-30);
            workspace.retry_best_relative_error[retry] = std::min(
                workspace.retry_best_relative_error[retry],
                relative);
            if (workspace.retry_converged[retry] != 0
                && std::isfinite(relative)
                && relative <= energy_tolerance) {
                workspace.retry_accepted[retry] = 1;
            } else {
                workspace.retry_accepted[retry] = 0;
                ++unresolved;
            }
        }
        if (unresolved == 0) {
            break;
        }
    }

    spherical_dsmc_orbit_audit_specific_potential(
        particles,
        particle_mass,
        p,
        workspace.paper_outer_sum,
        workspace.paper_potential);
    std::size_t diagnostic_retry = retry_count;
    double diagnostic_relative = -1.0;
    for (std::size_t retry = 0; retry < retry_count; ++retry) {
        const std::uint64_t id = workspace.failed_ids[retry];
        const std::size_t rank = workspace.final_rank_by_id[id];
        const double after =
            spherical_dsmc_specific_kinetic_energy(particles[rank])
            + workspace.paper_potential[rank];
        const std::size_t initial_rank = workspace.rank_by_id[id];
        const double before = workspace.energy_before[initial_rank];
        const double relative = paper_exact
            ? spherical_dsmc_paper_orbit_energy_defect(
                  after,
                  before,
                  workspace.energy_scale_before[initial_rank],
                  energy_tolerance)
            : std::abs(after - before)
                / std::max(std::abs(before), 1.0e-30);
        diagnostics.max_final_orbit_energy_error = std::max(
            diagnostics.max_final_orbit_energy_error,
            relative);
        diagnostics.lc_integrator_failed +=
            workspace.retry_converged[retry] == 0 ? 1 : 0;
        if (workspace.retry_accepted[retry] != 0) {
            ++diagnostics.lc_accepted;
        } else {
            ++diagnostics.lc_unresolved;
            if (!std::isfinite(relative)
                || diagnostic_retry == retry_count
                || relative > diagnostic_relative) {
                diagnostic_retry = retry;
                diagnostic_relative = relative;
            }
        }
    }
    // Retain the worst unresolved orbit in a detailed failure message. The
    // caller enforces the published-mode stop before applying collisions or
    // advancing the accepted global time.
    if (spherical_dsmc_paper_exact_enabled()
        && diagnostics.lc_unresolved != 0) {
        const std::uint64_t id = workspace.failed_ids[diagnostic_retry];
        const std::size_t final_rank = workspace.final_rank_by_id[id];
        const SphericalDSMCParticle& initial =
            workspace.initial_by_rank[workspace.rank_by_id[id]];
        const SphericalDSMCParticle& final = particles[final_rank];
        std::ostringstream message;
        message << std::setprecision(17)
            << "Paper-exact spherical DSMC LC refinement exhausted before "
            << "satisfying the equation-(5) particle-energy criterion: "
            << "unresolved=" << diagnostics.lc_unresolved
            << " id=" << id
            << " initial_rank=" << workspace.rank_by_id[id]
            << " final_rank=" << final_rank
            << " converged="
            << static_cast<int>(
                   workspace.retry_converged[diagnostic_retry])
            << " final_dE_over_E=" << diagnostic_relative
            << " best_dE_over_E="
            << workspace.retry_best_relative_error[diagnostic_retry]
            << " r0=" << initial.r
            << " vr0=" << initial.vr
            << " L=" << initial.L
            << " rf=" << final.r
            << " vrf=" << final.vr;
        workspace.paper_failure_message = message.str();
    }
    validate_endpoint();
    return diagnostics;
}

inline SphericalDSMCDiagnostics spherical_dsmc_gravity_step(
    std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    double dt,
    const Params& p) {
    SphericalDSMCGravityWorkspace workspace;
    return spherical_dsmc_gravity_step(
        particles,
        particle_mass,
        dt,
        p,
        workspace);
}

inline std::vector<double> spherical_dsmc_profile_edges(
    const SolverState& state) {
    std::vector<double> edges = radial_control_volume_edges(
        state.r,
        state.local_p);
    edges.front() = 0.0;
    if (!edges.empty()) {
        edges.back() = std::max(
            edges.back(),
            state.r.back());
    }
    return edges;
}

// Bin actual represented masses in spherical volumes. The radial pressure
// subtracts the radial bulk velocity, while the tangential column contains
// the sum of both tangential second moments. Profiles are observations of
// the particle state and leave the gravity and collision partitions intact.
inline void spherical_dsmc_write_profiles(
    std::ostream& output,
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    double time,
    int step,
    const SolverState& state) {
    const std::vector<double> edges = spherical_dsmc_profile_edges(state);
    const std::size_t bins = state.r.size();
    std::vector<double> mass(bins, 0.0);
    std::vector<double> sum_vr(bins, 0.0);
    std::vector<double> sum_vr2(bins, 0.0);
    std::vector<double> sum_vt2(bins, 0.0);
    for (const SphericalDSMCParticle& particle : particles) {
        auto it = std::upper_bound(edges.begin(), edges.end(), particle.r);
        if (it == edges.begin() || it == edges.end()) {
            continue;
        }
        const std::size_t bin =
            static_cast<std::size_t>(it - edges.begin() - 1);
        const double vt = spherical_dsmc_tangential_speed(particle);
        const double marker_mass =
            spherical_dsmc_particle_mass(particle, particle_mass);
        mass[bin] += marker_mass;
        sum_vr[bin] += marker_mass * particle.vr;
        sum_vr2[bin] += marker_mass * particle.vr * particle.vr;
        sum_vt2[bin] += marker_mass * vt * vt;
    }
    double enclosed = state.global_p.central_mass;
    for (std::size_t i = 0; i < bins; ++i) {
        const double volume = (4.0 * PI / 3.0)
            * (std::pow(edges[i + 1], 3) - std::pow(edges[i], 3));
        const double rho = volume > 0.0 ? mass[i] / volume : 0.0;
        const double mean_vr = mass[i] > 0.0 ? sum_vr[i] / mass[i] : 0.0;
        const double sigma_r2 = mass[i] > 0.0
            ? std::max(0.0, sum_vr2[i] / mass[i] - mean_vr * mean_vr)
            : 0.0;
        const double sigma_t2 = mass[i] > 0.0
            ? std::max(0.0, sum_vt2[i] / mass[i])
            : 0.0;
        enclosed += mass[i];
        output
            << std::setprecision(17)
            << time << ',' << step << ',' << state.r[i] << ','
            << rho << ',' << rho * sigma_r2 << ',' << rho * sigma_t2 << ','
            << rho * mean_vr << ',' << enclosed << ",0,0,1\n";
    }
}

inline double spherical_dsmc_central_density(
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass) {
    if (particles.empty()) {
        return 0.0;
    }
    // Frank & Tokayer define rho_0 from the sphere enclosing the innermost
    // 250 markers in their equal-mass sample.  Retain that radial sampling
    // convention but sum the actual marker masses for an unequal-mass state.
    constexpr std::size_t central_count = 250;
    const std::size_t count =
        std::min<std::size_t>(central_count, particles.size());
    const double radius = std::max(particles[count - 1].r, 1.0e-300);
    double central_mass = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        central_mass +=
            spherical_dsmc_particle_mass(particles[i], particle_mass);
    }
    return central_mass
        / ((4.0 * PI / 3.0) * radius * radius * radius);
}

// The carried next-collision step belongs to this cross-section law,
// normalization, shell policy, seed, and collision mode. Hash their literal
// values together so an exact restart checks the clock against its controls.
inline std::uint64_t spherical_dsmc_collision_control_fingerprint(
    const Params& p,
    bool paper_exact) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    auto mix_u64 = [&](std::uint64_t value) {
        for (unsigned int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (8U * byte)) & UINT64_C(255);
            hash *= UINT64_C(1099511628211);
        }
    };
    auto mix_string = [&](const std::string& value) {
        mix_u64(value.size());
        for (unsigned char byte : value) {
            hash ^= byte;
            hash *= UINT64_C(1099511628211);
        }
    };
    auto mix_double = [&](double value) {
        std::uint64_t bits;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        mix_u64(bits);
    };
    mix_string("SphDSMC collision clock controls 1");
    mix_string(p.bp_cross_section);
    mix_string(bp_fa_kernel_name(p));
    for (double value : {p.bp_collision_strength, p.bp_sigma0,
             p.bp_sigma_u0, p.bp_sigma_n, p.bp_yukawa_alpha,
             p.bp_yukawa_mchi, p.bp_yukawa_mphi, p.bp_subhalo_mass,
             p.bp_subhalo_radius, p.bp_coulomb_log, p.vp_gravity,
             p.bp_dsmc_collision_epsilon}) {
        mix_double(value);
    }
    mix_u64(static_cast<std::uint64_t>(p.bp_dsmc_shell_particles));
    mix_u64(static_cast<std::uint64_t>(p.bp_dsmc_min_shell_particles));
    mix_u64(p.bp_dsmc_seed);
    mix_u64(paper_exact ? 1U : 0U);
    return hash;
}

struct SphericalDSMCRestartHeader {
    char magic[16] = {};
    std::uint32_t version = 4;
    std::uint64_t particle_count = 0;
    double particle_mass = 0.0;
    double time = 0.0;
    std::int64_t step = 0;
    double initial_energy = 0.0;
    std::uint64_t collision_accepted_total = 0;
    std::int64_t lc_retries_total = 0;
    std::int64_t lc_unresolved_total = 0;
    // Versions 6/7 retain the next-step limit and its collision controls.
    // A particle-ensemble import has
    // no clock and reconstructs its first limit from the restored particles.
    std::optional<double> next_collision_dt;
    std::optional<std::uint64_t> collision_control_fingerprint;
};

// Write the complete accepted particle state to a temporary file before
// renaming it. Versions 4/5 include positive record masses; versions 6/7
// also carry the next collision limit and its control fingerprint. Odd
// versions retain the per-identity fractional trial accumulators.
inline void spherical_dsmc_write_restart(
    const std::filesystem::path& path,
    const std::vector<SphericalDSMCParticle>& particles,
    double particle_mass,
    double time,
    int step,
    double initial_energy,
    std::uint64_t collision_accepted_total,
    long long lc_retries_total,
    long long lc_unresolved_total,
    const std::vector<double>* paper_remainder_accumulator_by_id = nullptr,
    std::optional<double> next_collision_dt = std::nullopt,
    std::optional<std::uint64_t> collision_control_fingerprint = std::nullopt) {
    if (particles.size() < 2
        || !(particle_mass > 0.0)
        || !std::isfinite(particle_mass)
        || !(time >= 0.0)
        || !std::isfinite(time)
        || step < 0
        || !std::isfinite(initial_energy)
        || lc_retries_total < 0
        || lc_unresolved_total < 0
        || next_collision_dt.has_value() != collision_control_fingerprint.has_value()
        || (next_collision_dt.has_value()
            && (!(next_collision_dt.value() > 0.0)
                || std::isnan(next_collision_dt.value())))) {
        throw std::runtime_error(
            "Invalid spherical DSMC restart header values.");
    }
    std::vector<unsigned char> seen(particles.size(), 0);
    for (const SphericalDSMCParticle& particle : particles) {
        if (particle.id >= particles.size()
            || seen[static_cast<std::size_t>(particle.id)] != 0
            || !(particle.r > 0.0)
            || !std::isfinite(particle.r)
            || !std::isfinite(particle.vr)
            || !(particle.L >= 0.0)
            || !std::isfinite(particle.L)) {
            throw std::runtime_error(
                "Invalid particle record in spherical DSMC restart write.");
        }
        (void)spherical_dsmc_particle_mass(particle, particle_mass);
        seen[static_cast<std::size_t>(particle.id)] = 1;
    }
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not open spherical DSMC restart: " + temporary.string());
    }
    SphericalDSMCRestartHeader header;
    std::memcpy(header.magic, "VENUSSPHDSMC1", 13);
    // Versions 6/7 add the carried collision clock to the 4/5 mass layouts.
    // Omitting the clock initializes an independent particle ensemble.
    header.version = next_collision_dt.has_value()
        ? (paper_remainder_accumulator_by_id != nullptr ? 7U : 6U)
        : (paper_remainder_accumulator_by_id != nullptr ? 5U : 4U);
    if (paper_remainder_accumulator_by_id != nullptr
        && paper_remainder_accumulator_by_id->size() != particles.size()) {
        throw std::runtime_error(
            "Paper-exact spherical DSMC restart remainder state has the "
            "wrong particle count.");
    }
    header.particle_count = particles.size();
    header.particle_mass = particle_mass;
    header.time = time;
    header.step = step;
    header.initial_energy = initial_energy;
    header.collision_accepted_total = collision_accepted_total;
    header.lc_retries_total = lc_retries_total;
    header.lc_unresolved_total = lc_unresolved_total;
    output.write(header.magic, sizeof(header.magic));
    write_binary(output, header.version);
    write_binary(output, header.particle_count);
    write_binary(output, header.particle_mass);
    write_binary(output, header.time);
    write_binary(output, header.step);
    write_binary(output, header.initial_energy);
    write_binary(output, header.collision_accepted_total);
    write_binary(output, header.lc_retries_total);
    write_binary(output, header.lc_unresolved_total);
    if (next_collision_dt.has_value()) {
        write_binary(output, next_collision_dt.value());
        write_binary(output, collision_control_fingerprint.value());
    }
    for (const SphericalDSMCParticle& particle : particles) {
        write_binary(output, particle.r);
        write_binary(output, particle.vr);
        write_binary(output, particle.L);
        write_binary(output, particle.id);
        write_binary(
            output,
            spherical_dsmc_particle_mass(particle, particle_mass));
    }
    if (paper_remainder_accumulator_by_id != nullptr) {
        write_binary(
            output,
            static_cast<std::uint64_t>(
                paper_remainder_accumulator_by_id->size()));
        for (double value : *paper_remainder_accumulator_by_id) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Paper-exact spherical DSMC restart remainder state is "
                    "non-finite.");
            }
            write_binary(output, value);
        }
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed to write spherical DSMC restart: " + temporary.string());
    }
    output.close();
    std::filesystem::rename(temporary, path);
}

// Validate the schema and available payload before allocating records.
// Each particle must have a unique identity and finite physical state, and
// the optional fractional accumulators must cover all identities. Exact EOF
// closes the record sequence before the ensemble is sorted for gravity.
inline SphericalDSMCRestartHeader spherical_dsmc_read_restart(
    const std::filesystem::path& path,
    std::vector<SphericalDSMCParticle>& particles,
    std::vector<double>* paper_remainder_accumulator_by_id = nullptr) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open spherical DSMC restart: " + path.string());
    }
    SphericalDSMCRestartHeader header;
    input.read(header.magic, sizeof(header.magic));
    if (!input
        || !fixed_binary_identifier_equals(
            header.magic, "VENUSSPHDSMC1")) {
        throw std::runtime_error("Invalid spherical DSMC restart magic: " + path.string());
    }
    header.version = read_binary<std::uint32_t>(input);
    header.particle_count = read_binary<std::uint64_t>(input);
    header.particle_mass = read_binary<double>(input);
    header.time = read_binary<double>(input);
    header.step = read_binary<std::int64_t>(input);
    header.initial_energy = read_binary<double>(input);
    header.collision_accepted_total = read_binary<std::uint64_t>(input);
    header.lc_retries_total = read_binary<std::int64_t>(input);
    header.lc_unresolved_total = read_binary<std::int64_t>(input);
    if ((header.version != 2 && header.version != 3
         && header.version != 4 && header.version != 5
         && header.version != 6 && header.version != 7)
        || header.particle_count < 2
        || header.particle_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
        || !(header.particle_mass > 0.0)
        || !std::isfinite(header.particle_mass)
        || !(header.time >= 0.0)
        || !std::isfinite(header.time)
        || header.step < 0
        || header.step > static_cast<std::int64_t>(
            std::numeric_limits<int>::max())
        || !std::isfinite(header.initial_energy)
        || header.lc_retries_total < 0
        || header.lc_unresolved_total < 0) {
        throw std::runtime_error("Invalid spherical DSMC restart header: " + path.string());
    }
    if (header.version >= 6) {
        const double next_collision_dt = read_binary<double>(input);
        if (!(next_collision_dt > 0.0) || std::isnan(next_collision_dt)) {
            throw std::runtime_error(
                "Invalid spherical DSMC restart collision clock: " + path.string());
        }
        header.next_collision_dt = next_collision_dt;
        header.collision_control_fingerprint = read_binary<std::uint64_t>(input);
    }
    const std::streampos records_begin = input.tellg();
    input.seekg(0, std::ios::end);
    const std::streampos file_end = input.tellg();
    input.seekg(records_begin);
    const std::uint64_t record_bytes = header.version >= 4 ? 40U : 32U;
    if (!input || records_begin < std::streampos(0)
        || file_end < records_begin
        || header.particle_count
            > static_cast<std::uint64_t>(file_end - records_begin)
                / record_bytes) {
        throw std::runtime_error(
            "Spherical DSMC restart particle count exceeds the file payload: "
            + path.string());
    }
    particles.resize(static_cast<std::size_t>(header.particle_count));
    for (SphericalDSMCParticle& particle : particles) {
        particle.r = read_binary<double>(input);
        particle.vr = read_binary<double>(input);
        particle.L = read_binary<double>(input);
        particle.id = read_binary<std::uint64_t>(input);
        particle.mass = header.version >= 4
            ? read_binary<double>(input)
            : header.particle_mass;
    }
    std::vector<unsigned char> seen(particles.size(), 0);
    for (const SphericalDSMCParticle& particle : particles) {
        if (particle.id >= particles.size()
            || seen[static_cast<std::size_t>(particle.id)] != 0
            || !(particle.r > 0.0)
            || !std::isfinite(particle.r)
            || !std::isfinite(particle.vr)
            || !(particle.L >= 0.0)
            || !std::isfinite(particle.L)
            || !(particle.mass > 0.0)
            || !std::isfinite(particle.mass)) {
            throw std::runtime_error(
                "Invalid particle record in spherical DSMC restart: "
                + path.string());
        }
        seen[static_cast<std::size_t>(particle.id)] = 1;
    }
    if (header.version == 3 || header.version == 5 || header.version == 7) {
        const std::uint64_t remainder_count = read_binary<std::uint64_t>(input);
        if (remainder_count != header.particle_count) {
            throw std::runtime_error(
                "Invalid paper-exact remainder count in spherical DSMC "
                "restart: " + path.string());
        }
        if (paper_remainder_accumulator_by_id != nullptr) {
            paper_remainder_accumulator_by_id->resize(
                static_cast<std::size_t>(remainder_count));
        }
        for (std::uint64_t i = 0; i < remainder_count; ++i) {
            const double value = read_binary<double>(input);
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Non-finite paper-exact remainder in spherical DSMC "
                    "restart: " + path.string());
            }
            if (paper_remainder_accumulator_by_id != nullptr) {
                (*paper_remainder_accumulator_by_id)[
                    static_cast<std::size_t>(i)] = value;
            }
        }
    } else if (paper_remainder_accumulator_by_id != nullptr) {
        paper_remainder_accumulator_by_id->assign(particles.size(), 0.0);
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(
            "Unexpected trailing spherical DSMC restart payload: " + path.string());
    }
    spherical_dsmc_sort_particles(particles, true);
    return header;
}

// A fresh ensemble establishes its own energy reference and collision
// limit. A full restart restores time, counters, fractional trials, and the
// carried limit, then checks that its collision controls are unchanged.
// Shared-memory workers operate within this one spherical ensemble.
inline void run_spherical_dsmc_solver(SolverState& state) {
    if (state.mpi.size != 1) {
        throw std::runtime_error(
            "spherical_dsmc is OpenMP-only: launch one task with many threads, never MPI.");
    }
    std::filesystem::create_directories(state.outdir);
    std::filesystem::create_directories(state.restart_dir);
    const bool paper_exact = spherical_dsmc_paper_exact_enabled();
    const std::uint64_t collision_controls =
        spherical_dsmc_collision_control_fingerprint(state.global_p, paper_exact);
    if (paper_exact) {
        spherical_dsmc_validate_paper_exact_configuration(state.global_p);
        std::cout
            << "spherical_dsmc collision mode=paper_exact"
            << " (extrema shells, 4*vmax, integer+remainder,"
            << " fresh pair azimuths, persisted remainders)\n";
    }

    std::vector<SphericalDSMCParticle> particles;
    SphericalDSMCCollisionWorkspace collision_workspace;
    double particle_mass = 0.0;
    double initial_energy = 0.0;
    SphericalDSMCEnergy current_energy;
    std::uint64_t accepted_total = 0;
    long long lc_retries_total = 0;
    long long lc_unresolved_total = 0;
    std::optional<double> restored_collision_dt;
    std::uint32_t restart_input_version = 0;
    if (state.global_p.restart_from.empty()) {
        double total_mass = 0.0;
        particles = spherical_dsmc_sample_initial_particles(state, total_mass);
        particle_mass = total_mass / static_cast<double>(particles.size());
        current_energy = spherical_dsmc_total_energy(
            particles,
            particle_mass,
            state.global_p);
        initial_energy = current_energy.total;
        state.t = 0.0;
        state.step = 0;
    } else {
        const SphericalDSMCRestartHeader header = spherical_dsmc_read_restart(
            state.global_p.restart_from,
            particles,
            paper_exact
                ? &collision_workspace.paper_remainder_accumulator_by_id
                : nullptr);
        restart_input_version = header.version;
        particle_mass = header.particle_mass;
        state.t = header.time;
        state.step = static_cast<int>(header.step);
        initial_energy = header.initial_energy;
        accepted_total = header.collision_accepted_total;
        lc_retries_total = header.lc_retries_total;
        lc_unresolved_total = header.lc_unresolved_total;
        restored_collision_dt = header.next_collision_dt;
        if (restored_collision_dt.has_value()
            && header.collision_control_fingerprint != collision_controls) {
            throw std::runtime_error(
                "Spherical DSMC restart collision controls differ from the "
                "persisted collision clock. Exact continuation requires the "
                "same cross-section law, angular kernel, normalization, shell "
                "controls, timestep fraction, random seed, and collision mode.");
        }
        current_energy = spherical_dsmc_total_energy(
            particles,
            particle_mass,
            state.global_p);
    }
    // Metadata is written only after sampling or loading the persistent
    // markers so it records the actual mass/count and restart schema rather
    // than the input sentinel bp_dsmc_total_mass=0.
    write_metadata(
        state.outdir / "metadata.json",
        state.global_p,
        state.local_p,
        state.decomp,
        state.mpi,
        particles.size(),
        spherical_dsmc_total_particle_mass(particles, particle_mass),
        particle_mass,
        paper_exact ? 7U : 6U,
        restart_input_version,
        paper_exact,
        spherical_dsmc_force_consistent_orbit_audit_enabled(),
        spherical_dsmc_force_radix_sort_enabled());
    if (paper_exact
        && collision_workspace.paper_remainder_accumulator_by_id.size()
               != particles.size()) {
        collision_workspace.paper_remainder_accumulator_by_id.assign(
            particles.size(),
            0.0);
    }

    const bool append = !state.global_p.restart_from.empty()
        && std::filesystem::exists(state.outdir / "history.csv");
    std::ofstream history(
        state.outdir / "history.csv",
        append ? std::ios::app : std::ios::trunc);
    std::ofstream profiles(
        state.outdir / "profiles.csv",
        append ? std::ios::app : std::ios::trunc);
    std::ofstream operator_budget(
        state.outdir / "operator_mass_budget.csv",
        append ? std::ios::app : std::ios::trunc);
    if (!history || !profiles || !operator_budget) {
        throw std::runtime_error("Could not open spherical DSMC output streams.");
    }
    SphericalDSMCShellWorkspace timestep_shell_workspace;
    SphericalDSMCShellWorkspace collision_shell_workspace;
    SphericalDSMCGravityWorkspace gravity_workspace;
    SphericalDSMCReductionWorkspace reduction_workspace;
    double next_collision_dt;
    if (restored_collision_dt.has_value()) {
        next_collision_dt = restored_collision_dt.value();
    } else {
        const std::vector<SphericalDSMCShell> initial_shells =
            spherical_dsmc_build_shells(
                particles, particle_mass, state.global_p,
                timestep_shell_workspace);
        next_collision_dt = spherical_dsmc_collision_timestep(
            initial_shells, state.global_p);
        if (restart_input_version != 0) {
            std::cout << "spherical_dsmc restart version="
                      << restart_input_version
                      << " has no persisted collision clock; "
                         "reconstructing the first collision limit\n";
        }
    }

    if (!append) {
        history
            << "step,time,dt,total_mass,kinetic_energy,potential_energy,total_energy,"
            << "relative_energy_error,central_density,collision_trials_step,"
            << "collision_accepted_step,collision_accepted_total,collision_shells,"
            << "minimum_shell_particles,minimum_mean_free_path,"
            << "maximum_shell_width_over_mean_free_path,"
            << "mfp_pair_samples,min_orbital_sigma,max_orbital_sigma,"
            << "nonlocal_collision_shells,lc_retries_step,"
            << "lc_retries_total,lc_accepted_step,lc_unresolved_step,"
            << "lc_unresolved_total,lc_integrator_failed_step,"
            << "max_ordinary_orbit_energy_error,max_final_orbit_energy_error\n";
        profiles << "time,step,r,rho,p,pt,j,m,mu,lambda,gap\n";
        operator_budget
            << "step,time,dt,transport_mass_before,transport_mass_after,"
            << "transport_mass_delta,transport_energy_before,"
            << "transport_energy_after,transport_energy_delta,"
            << "collision_mass_before,collision_mass_after,collision_mass_delta,"
            << "collision_radial_momentum_before,collision_radial_momentum_after,"
            << "collision_radial_momentum_delta,collision_kinetic_before,"
            << "collision_kinetic_after,collision_kinetic_delta,"
            << "collision_relative_kinetic_delta,trial_pairs,accepted_pairs,"
            << "max_pair_momentum_error,max_pair_kinetic_energy_error\n";
    }

    auto write_snapshot = [&](double dt, const SphericalDSMCDiagnostics& diagnostics) {
        current_energy =
            spherical_dsmc_total_energy(particles, particle_mass, state.global_p);
        const double relative_energy_error = std::abs(
            (current_energy.total - initial_energy)
            / std::max(std::abs(initial_energy), 1.0e-300));
        history
            << std::setprecision(17)
            << state.step << ',' << state.t << ',' << dt << ','
            << spherical_dsmc_total_particle_mass(
                   particles, particle_mass) << ','
            << current_energy.kinetic << ',' << current_energy.potential << ','
            << current_energy.total << ','
            << relative_energy_error << ','
            << spherical_dsmc_central_density(particles, particle_mass) << ','
            << diagnostics.collision_trials << ','
            << diagnostics.collision_accepted << ','
            << accepted_total << ','
            << diagnostics.collision_shells << ','
            << diagnostics.minimum_shell_particles << ','
            << diagnostics.minimum_mean_free_path << ','
            << diagnostics.maximum_shell_width_over_mean_free_path << ','
            << diagnostics.mean_free_path_pair_samples << ','
            << (std::isfinite(
                    diagnostics.minimum_orbital_velocity_dispersion)
                    ? diagnostics.minimum_orbital_velocity_dispersion
                    : 0.0)
            << ',' << diagnostics.maximum_orbital_velocity_dispersion << ','
            << diagnostics.nonlocal_collision_shells << ','
            << diagnostics.lc_retries << ','
            << lc_retries_total << ','
            << diagnostics.lc_accepted << ','
            << diagnostics.lc_unresolved << ','
            << lc_unresolved_total << ','
            << diagnostics.lc_integrator_failed << ','
            << diagnostics.max_ordinary_orbit_energy_error << ','
            << diagnostics.max_final_orbit_energy_error << '\n';
        spherical_dsmc_write_profiles(
            profiles,
            particles,
            particle_mass,
            state.t,
            state.step,
            state);
        flush_output_checked(history, "spherical DSMC history.csv");
        flush_output_checked(profiles, "spherical DSMC profiles.csv");
    };

    // A new output directory records the incoming state before evolution.
    // Its dt=0 and default per-step collision diagnostics describe an observation
    // at the restored epoch; cumulative counters retain their incoming values.
    if (!append) {
        write_snapshot(0.0, {});
        spherical_dsmc_write_restart(
            state.restart_path,
            particles,
            particle_mass,
            state.t,
            state.step,
            initial_energy,
            accepted_total,
            lc_retries_total,
            lc_unresolved_total,
            paper_exact
                ? &collision_workspace.paper_remainder_accumulator_by_id
                : nullptr,
            next_collision_dt,
            collision_controls);
    }
    // Each accepted macrostep first transports the complete population, then
    // rebuilds collision shells at the transported positions and scatters for
    // the same dt. The next collision limit is carried forward from this batch.
    // Saving and restart frequencies observe accepted steps without clipping dt.
    while (state.t < state.global_p.t_end
           && state.step < state.global_p.max_steps) {
        const double gravity_dt = spherical_dsmc_gravity_timestep(
            particles,
            particle_mass,
            state.global_p);
        const double dt = std::min({
            state.global_p.dt_max,
            gravity_dt,
            next_collision_dt,
            state.global_p.t_end - state.t,
        });
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            throw std::runtime_error("spherical_dsmc selected a non-finite timestep.");
        }

        const double total_mass =
            spherical_dsmc_total_particle_mass(particles, particle_mass);
        const SphericalDSMCEnergy transport_before = current_energy;
        SphericalDSMCDiagnostics diagnostics = spherical_dsmc_gravity_step(
            particles,
            particle_mass,
            dt,
            state.global_p,
            gravity_workspace);
        if (paper_exact && diagnostics.lc_unresolved != 0) {
            throw std::runtime_error(
                "Paper-exact spherical DSMC left an unresolved LC orbit: "
                + gravity_workspace.paper_failure_message);
        }
        const SphericalDSMCStateSums transport_after =
            spherical_dsmc_state_sums_parallel(
                particles,
                particle_mass,
                state.global_p,
                reduction_workspace);
        const double radial_momentum_before =
            transport_after.radial_momentum;
        const double kinetic_before = transport_after.energy.kinetic;
        const std::vector<SphericalDSMCShell> collision_shells =
            spherical_dsmc_build_shells(
                particles,
                particle_mass,
                state.global_p,
                collision_shell_workspace);
        SphericalDSMCDiagnostics collision =
            spherical_dsmc_apply_collisions(
                particles,
                collision_shells,
                particle_mass,
                dt,
                state.step,
                state.global_p,
                collision_workspace);
        next_collision_dt =
            spherical_dsmc_collision_timestep_after_collisions(
                collision_shells,
                collision_workspace,
                state.global_p);
        const double kinetic_after =
            kinetic_before + collision.collision_kinetic_error;
        const double radial_momentum_after =
            radial_momentum_before
            + collision.collision_radial_momentum_error;
        current_energy = transport_after.energy;
        current_energy.kinetic = kinetic_after;
        current_energy.total =
            current_energy.kinetic + current_energy.potential;
        diagnostics.collision_trials = collision.collision_trials;
        diagnostics.collision_accepted = collision.collision_accepted;
        diagnostics.collision_kinetic_error = collision.collision_kinetic_error;
        diagnostics.max_pair_momentum_error = collision.max_pair_momentum_error;
        diagnostics.max_pair_kinetic_error = collision.max_pair_kinetic_error;
        diagnostics.collision_shells = collision.collision_shells;
        diagnostics.minimum_shell_particles = collision.minimum_shell_particles;
        diagnostics.minimum_mean_free_path = collision.minimum_mean_free_path;
        diagnostics.maximum_shell_width_over_mean_free_path =
            collision.maximum_shell_width_over_mean_free_path;
        diagnostics.mean_free_path_pair_samples =
            collision.mean_free_path_pair_samples;
        diagnostics.minimum_orbital_velocity_dispersion =
            collision.minimum_orbital_velocity_dispersion;
        diagnostics.maximum_orbital_velocity_dispersion =
            collision.maximum_orbital_velocity_dispersion;
        diagnostics.nonlocal_collision_shells =
            collision.nonlocal_collision_shells;
        accepted_total += diagnostics.collision_accepted;
        lc_retries_total += diagnostics.lc_retries;
        lc_unresolved_total += diagnostics.lc_unresolved;

        state.t += dt;
        ++state.step;
        state.last_dt = dt;
        operator_budget
            << std::setprecision(17)
            << state.step << ',' << state.t << ',' << dt << ','
            << total_mass << ',' << total_mass << ",0,"
            << transport_before.total << ','
            << transport_after.energy.total << ','
            << transport_after.energy.total - transport_before.total << ','
            << total_mass << ',' << total_mass << ",0,"
            << radial_momentum_before << ',' << radial_momentum_after << ','
            << radial_momentum_after - radial_momentum_before << ','
            << kinetic_before << ',' << kinetic_after << ','
            << kinetic_after - kinetic_before << ','
            << (kinetic_before != 0.0
                    ? (kinetic_after - kinetic_before) / kinetic_before
                    : 0.0)
            << ',' << diagnostics.collision_trials
            << ',' << diagnostics.collision_accepted
            << ',' << diagnostics.max_pair_momentum_error
            << ',' << diagnostics.max_pair_kinetic_error << '\n';

        // Advance the saved time and cumulative counters only after both operators
        // return. Flush mass/energy budgets with observations or checkpoints; a
        // final full restart retains the last completed state at loop exit.
        const bool save = state.step % std::max(1, state.global_p.save_every) == 0
            || state.t >= state.global_p.t_end;
        if (save) {
            write_snapshot(dt, diagnostics);
        }
        const bool restart =
            state.global_p.restart_every > 0
            && (state.step % state.global_p.restart_every == 0
                || state.t >= state.global_p.t_end);
        if (save || restart) {
            flush_output_checked(
                operator_budget,
                "spherical DSMC operator_mass_budget.csv");
        }
        if (restart) {
            spherical_dsmc_write_restart(
                state.restart_path,
                particles,
                particle_mass,
                state.t,
                state.step,
                initial_energy,
                accepted_total,
                lc_retries_total,
                lc_unresolved_total,
                paper_exact
                    ? &collision_workspace.paper_remainder_accumulator_by_id
                    : nullptr,
                next_collision_dt,
                collision_controls);
        }
    }
    spherical_dsmc_write_restart(
        state.restart_path,
        particles,
        particle_mass,
        state.t,
        state.step,
        initial_energy,
        accepted_total,
        lc_retries_total,
        lc_unresolved_total,
        paper_exact
            ? &collision_workspace.paper_remainder_accumulator_by_id
            : nullptr,
        next_collision_dt,
        collision_controls);
}

}  // namespace ev
