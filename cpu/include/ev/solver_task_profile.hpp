#pragma once

// Associate each CPU application name with its permitted physical equations
// and collision backend. Inputs are fully parsed Params; outputs are readable
// help or an exception describing an incompatible physical combination.
// The same numerical kernels serve all tasks. Validating this selection before
// state construction keeps an executable name consistent with its calculation.

#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

#include "ev/common.hpp"

namespace ev {

enum class SolverTaskProfile {
    einstein_vlasov_eulerian,
    vlasov_poisson_eulerian,
    boltzmann_poisson_fokker_planck,
    boltzmann_poisson_landau,
    boltzmann_poisson_finite_angle,
    boltzmann_poisson_exact,
    boltzmann_poisson_reversible,
    boltzmann_poisson_reversible_sampled,
    boltzmann_poisson_reversible_streamed,
    boltzmann_poisson_expanding_prompt_cusp,
    sidm_sphdsmc_constant,
    sidm_sphdsmc_velocity_dependent,
};

// Stable task names identify the physical/backend choice in help and
// errors; they do not select numerical constants inside a kernel.
inline const char* solver_task_profile_name(SolverTaskProfile profile) {
    switch (profile) {
    case SolverTaskProfile::einstein_vlasov_eulerian:
        return "einstein_vlasov_eulerian";
    case SolverTaskProfile::vlasov_poisson_eulerian:
        return "vlasov_poisson_eulerian";
    case SolverTaskProfile::boltzmann_poisson_fokker_planck:
        return "boltzmann_poisson_fokker_planck";
    case SolverTaskProfile::boltzmann_poisson_landau:
        return "boltzmann_poisson_landau";
    case SolverTaskProfile::boltzmann_poisson_finite_angle:
        return "boltzmann_poisson_finite_angle";
    case SolverTaskProfile::boltzmann_poisson_exact:
        return "boltzmann_poisson_exact";
    case SolverTaskProfile::boltzmann_poisson_reversible:
        return "boltzmann_poisson_reversible";
    case SolverTaskProfile::boltzmann_poisson_reversible_sampled:
        return "boltzmann_poisson_reversible_sampled";
    case SolverTaskProfile::boltzmann_poisson_reversible_streamed:
        return "boltzmann_poisson_reversible_streamed";
    case SolverTaskProfile::boltzmann_poisson_expanding_prompt_cusp:
        return "boltzmann_poisson_expanding_prompt_cusp";
    case SolverTaskProfile::sidm_sphdsmc_constant:
        return "sidm_sphdsmc_constant";
    case SolverTaskProfile::sidm_sphdsmc_velocity_dependent:
        return "sidm_sphdsmc_velocity_dependent";
    }
    return "unknown";
}

// Describe the model and collision law in the same terms as the input
// parameters, so the user can connect a task name to its physical equation.
inline const char* solver_task_profile_requirement(
    SolverTaskProfile profile) {
    switch (profile) {
    case SolverTaskProfile::einstein_vlasov_eulerian:
        return "collisionless physics_model=einstein_vlasov";
    case SolverTaskProfile::vlasov_poisson_eulerian:
        return "collisionless physics_model=vlasov_poisson (static or expanding)";
    case SolverTaskProfile::boltzmann_poisson_fokker_planck:
        return "Boltzmann-Poisson with bp_collision_operator=fokker_planck";
    case SolverTaskProfile::boltzmann_poisson_landau:
        return "Boltzmann-Poisson with bp_collision_operator=landau_fp";
    case SolverTaskProfile::boltzmann_poisson_finite_angle:
        return "Boltzmann-Poisson with an exact or reversible finite-angle operator";
    case SolverTaskProfile::boltzmann_poisson_exact:
        return "Boltzmann-Poisson with bp_collision_operator=boltzmann_exact";
    case SolverTaskProfile::boltzmann_poisson_reversible:
        return "Boltzmann-Poisson with bp_collision_operator=boltzmann_reversible";
    case SolverTaskProfile::boltzmann_poisson_reversible_sampled:
        return "Boltzmann-Poisson with bp_collision_operator=boltzmann_reversible_sampled";
    case SolverTaskProfile::boltzmann_poisson_reversible_streamed:
        return "Boltzmann-Poisson with bp_collision_operator=boltzmann_reversible_streamed";
    case SolverTaskProfile::boltzmann_poisson_expanding_prompt_cusp:
        return "expanding density-contrast exact BP with softened-gravity collisions";
    case SolverTaskProfile::sidm_sphdsmc_constant:
        return "SIDM spherical_dsmc with a constant isotropic cross section";
    case SolverTaskProfile::sidm_sphdsmc_velocity_dependent:
        return "SIDM spherical_dsmc with a supported velocity-dependent cross section";
    }
    return "a known CPU task profile";
}

inline void print_solver_task_profile_help(
    const char* executable,
    SolverTaskProfile profile) {
    std::cout
        << "Usage: " << executable << " --config FILE [task-compatible overrides]\n\n"
        << "Locked CPU task profile: " << solver_task_profile_name(profile)
        << "\nRequired physics: "
        << solver_task_profile_requirement(profile) << ".\n\n"
        << "Common controls include --threads N, --t-end T, --dt-max DT, "
           "--output-dir DIR, and --restart-dir DIR.  The INI file remains "
           "the complete parameter record.  Any incompatible model, "
           "collision operator, background, or cross section is rejected "
           "before state and output construction.\n"
        << "Use venus_modular_solver --help when a single command must select "
           "the physics model dynamically.\n";
}

// Direct and reversible angular collision operators all retain finite
// scattering angles; this family is distinct from the small-angle
// Fokker--Planck forms.
inline bool is_finite_angle_boltzmann_operator(const std::string& value) {
    return value == "boltzmann_exact"
        || value == "boltzmann_reversible"
        || value == "boltzmann_reversible_sampled"
        || value == "boltzmann_reversible_streamed";
}

inline bool is_velocity_dependent_cross_section(const std::string& value) {
    return value == "powerlaw"
        || value == "softened_gravity"
        || value == "yukawa_born";
}

[[noreturn]] inline void reject_solver_task_profile(
    SolverTaskProfile profile,
    const Params& p,
    const std::string& requirement) {
    std::ostringstream message;
    message
        << "CPU task profile '" << solver_task_profile_name(profile)
        << "' rejected the configuration: requires " << requirement
        << "; received physics_model=" << p.physics_model
        << ", vp_universe=" << p.vp_universe
        << ", vp_poisson_source=" << p.vp_poisson_source
        << ", bp_collision_operator=" << p.bp_collision_operator
        << ", bp_cross_section=" << p.bp_cross_section
        << ", bp_boltzmann_kernel=" << p.bp_boltzmann_kernel
        << ", bp_collision_strength=" << p.bp_collision_strength << '.';
    throw std::runtime_error(message.str());
}

// Compare the entire required physical tuple for each task. For the
// expanding subhalo problem this includes the background, density-contrast
// Poisson source and cross-section law, not just the model name.
inline void validate_solver_task_profile(
    SolverTaskProfile profile,
    const Params& p) {
    switch (profile) {
    case SolverTaskProfile::einstein_vlasov_eulerian:
        if (p.physics_model != "einstein_vlasov"
            || p.bp_collision_operator != "none") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=einstein_vlasov and "
                "bp_collision_operator=none");
        }
        return;

    case SolverTaskProfile::vlasov_poisson_eulerian:
        if (p.physics_model != "vlasov_poisson"
            || p.bp_collision_operator != "none") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=vlasov_poisson and "
                "bp_collision_operator=none");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_fokker_planck:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator != "fokker_planck") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=boltzmann_poisson and "
                "bp_collision_operator=fokker_planck");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_landau:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator != "landau_fp") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=boltzmann_poisson and "
                "bp_collision_operator=landau_fp");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_finite_angle:
        if (p.physics_model != "boltzmann_poisson"
            || !is_finite_angle_boltzmann_operator(
                p.bp_collision_operator)) {
            reject_solver_task_profile(
                profile, p,
                "physics_model=boltzmann_poisson and a finite-angle "
                "Boltzmann operator");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_exact:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator != "boltzmann_exact") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=boltzmann_poisson and "
                "bp_collision_operator=boltzmann_exact");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_reversible:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator != "boltzmann_reversible") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=boltzmann_poisson and "
                "bp_collision_operator=boltzmann_reversible");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_reversible_sampled:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator
                != "boltzmann_reversible_sampled") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=boltzmann_poisson and "
                "bp_collision_operator=boltzmann_reversible_sampled");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_reversible_streamed:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator
                != "boltzmann_reversible_streamed") {
            reject_solver_task_profile(
                profile, p,
                "physics_model=boltzmann_poisson and "
                "bp_collision_operator=boltzmann_reversible_streamed");
        }
        return;

    case SolverTaskProfile::boltzmann_poisson_expanding_prompt_cusp:
        if (p.physics_model != "boltzmann_poisson"
            || p.vp_universe != "expanding"
            || p.vp_poisson_source != "density_contrast"
            || p.bp_collision_operator != "boltzmann_exact"
            || p.bp_cross_section != "softened_gravity"
            || !(p.bp_collision_strength > 0.0)) {
            reject_solver_task_profile(
                profile, p,
                "expanding density-contrast Boltzmann-Poisson with "
                "bp_collision_operator=boltzmann_exact and "
                "bp_cross_section=softened_gravity with positive "
                "bp_collision_strength");
        }
        return;

    case SolverTaskProfile::sidm_sphdsmc_constant:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator != "spherical_dsmc"
            || p.bp_cross_section != "constant"
            || (p.bp_boltzmann_kernel != "auto"
                && p.bp_boltzmann_kernel != "isotropic")) {
            reject_solver_task_profile(
                profile, p,
                "Boltzmann-Poisson spherical_dsmc with a "
                "constant isotropic cross section");
        }
        return;

    case SolverTaskProfile::sidm_sphdsmc_velocity_dependent:
        if (p.physics_model != "boltzmann_poisson"
            || p.bp_collision_operator != "spherical_dsmc"
            || !is_velocity_dependent_cross_section(
                p.bp_cross_section)) {
            reject_solver_task_profile(
                profile, p,
                "Boltzmann-Poisson spherical_dsmc with a supported "
                "velocity-dependent cross section");
        }
        return;
    }
    reject_solver_task_profile(profile, p, "a known task profile");
}

}  // namespace ev
