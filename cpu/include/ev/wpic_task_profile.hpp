#pragma once

// Check the physical task selected by a weighted-particle executable.
// The shared drivers still perform the numerical evolution. These checks bind
// the parsed input to the advertised low/high Einstein--Vlasov scenario,
// collisionless/subhalo Newtonian calculation or material SIDM cross-section
// family before constructing state or opening simulation output.

#include <stdexcept>
#include <string>

namespace ev {

enum class EinsteinVlasovWPICTaskProfile {
    low_compactness,
    black_hole_formation,
};

// The compactness scenario chooses a different physical initial family.
// Require the scenario named by the executable rather than silently
// changing the requested low- or high-compactness experiment.
inline void validate_einstein_vlasov_wpic_task_profile(
    EinsteinVlasovWPICTaskProfile profile,
    const std::string& scenario) {
    const std::string expected =
        profile == EinsteinVlasovWPICTaskProfile::low_compactness
        ? "low" : "high";
    if (scenario != expected) {
        throw std::runtime_error(
            "Task-specific Einstein-Vlasov WPIC executable requires "
            "--scenario " + expected + "; received " + scenario + '.');
    }
}

enum class ExpandingWPICTaskProfile {
    collisionless_vlasov_poisson,
    collisional_subhalo,
};

// Zero encounter strength defines the collisionless task; strictly
// positive strength defines the collisional subhalo task. Background
// and force parameters remain the shared driver inputs.
inline void validate_expanding_wpic_task_profile(
    ExpandingWPICTaskProfile profile,
    double subhalo_collision_strength) {
    if (profile == ExpandingWPICTaskProfile::collisionless_vlasov_poisson
        && subhalo_collision_strength != 0.0) {
        throw std::runtime_error(
            "Collisionless expanding Vlasov-Poisson WPIC requires "
            "--subhalo-collision-strength 0.");
    }
    if (profile == ExpandingWPICTaskProfile::collisional_subhalo
        && !(subhalo_collision_strength > 0.0)) {
        throw std::runtime_error(
            "Collisional subhalo WPIC requires a strictly positive "
            "--subhalo-collision-strength.");
    }
}

enum class SIDMWPICTaskProfile {
    constant_cross_section,
    velocity_dependent_cross_section,
};

// Material SIDM tasks require positive scattering strength and the
// selected constant or velocity-dependent cross-section law.
inline void validate_sidm_wpic_task_profile(
    SIDMWPICTaskProfile profile,
    const std::string& cross_section,
    double collision_strength) {
    if (!(collision_strength > 0.0)) {
        throw std::runtime_error(
            "Task-specific SIDM WPIC executables require a strictly positive "
            "--collision-strength.");
    }
    if (profile == SIDMWPICTaskProfile::constant_cross_section
        && cross_section != "constant") {
        throw std::runtime_error(
            "Constant-cross-section SIDM WPIC requires "
            "--sidm-cross-section constant.");
    }
    if (profile == SIDMWPICTaskProfile::velocity_dependent_cross_section
        && cross_section != "powerlaw"
        && cross_section != "yukawa_born") {
        throw std::runtime_error(
            "Velocity-dependent SIDM WPIC requires "
            "--sidm-cross-section powerlaw|yukawa_born.");
    }
}

}  // namespace ev
