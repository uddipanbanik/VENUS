// Focused fail-closed tests for every task-specific CPU executable contract.

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ev/solver_task_profile.hpp"
#include "ev/wpic_task_profile.hpp"

namespace {

int failures = 0;

void expect_accept(const std::string& name, const std::function<void()>& run) {
    try {
        run();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << " unexpectedly rejected: "
                  << error.what() << '\n';
        ++failures;
    }
}

void expect_reject(const std::string& name, const std::function<void()>& run) {
    try {
        run();
        std::cerr << "FAIL " << name << " unexpectedly accepted\n";
        ++failures;
    } catch (const std::runtime_error&) {
        // Expected fail-closed result.
    }
}

ev::Params base_boltzmann() {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    return p;
}

}  // namespace

int main() {
    {
        std::ostringstream captured;
        std::streambuf* original = std::cout.rdbuf(captured.rdbuf());
        ev::print_solver_task_profile_help(
            "venus_cpu_einstein_vlasov_eulerian",
            ev::SolverTaskProfile::einstein_vlasov_eulerian);
        std::cout.rdbuf(original);
        const std::string help = captured.str();
        if (help.find("Locked CPU task profile: einstein_vlasov_eulerian")
                == std::string::npos
            || help.find("vlasov_poisson|boltzmann_poisson")
                != std::string::npos
            || help.find("--physics-model") != std::string::npos) {
            std::cerr << "FAIL task help advertised an unlocked physics model\n";
            ++failures;
        }
    }
    {
        ev::Params p;
        p.physics_model = "einstein_vlasov";
        p.bp_collision_operator = "none";
        expect_accept("EV accepts EV", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::einstein_vlasov_eulerian, p);
        });
        p.physics_model = "vlasov_poisson";
        expect_reject("EV rejects VP", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::einstein_vlasov_eulerian, p);
        });
    }
    {
        ev::Params p;
        p.physics_model = "vlasov_poisson";
        p.bp_collision_operator = "none";
        expect_accept("VP accepts collisionless VP", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::vlasov_poisson_eulerian, p);
        });
        p.bp_collision_operator = "fokker_planck";
        expect_reject("VP rejects collision operator", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::vlasov_poisson_eulerian, p);
        });
    }
    {
        ev::Params p = base_boltzmann();
        p.bp_collision_operator = "fokker_planck";
        expect_accept("FP accepts Fokker-Planck", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_fokker_planck, p);
        });
        p.bp_collision_operator = "landau_fp";
        expect_reject("FP rejects Landau backend", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_fokker_planck, p);
        });
        expect_accept("Landau accepts Landau backend", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_landau, p);
        });
        p.bp_collision_operator = "fokker_planck";
        expect_reject("Landau rejects Fokker-Planck backend", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_landau, p);
        });
        p.bp_collision_operator = "boltzmann_exact";
        expect_reject("FP rejects finite angle", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_fokker_planck, p);
        });
    }
    {
        ev::Params p = base_boltzmann();
        p.bp_collision_operator = "boltzmann_reversible_sampled";
        expect_accept("finite angle accepts sampled reversible", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_finite_angle, p);
        });
        p.bp_collision_operator = "spherical_dsmc";
        expect_reject("finite angle rejects SphDSMC", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_finite_angle, p);
        });
        p.bp_collision_operator = "boltzmann_exact";
        expect_accept("exact accepts exact", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_exact, p);
        });
        expect_reject("reversible rejects exact", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_reversible, p);
        });
        p.bp_collision_operator = "boltzmann_reversible";
        expect_accept("reversible accepts deterministic", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_reversible, p);
        });
        p.bp_collision_operator = "boltzmann_reversible_sampled";
        expect_accept("sampled accepts sampled", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_reversible_sampled,
                p);
        });
        expect_reject("streamed rejects sampled", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_reversible_streamed,
                p);
        });
        p.bp_collision_operator = "boltzmann_reversible_streamed";
        expect_accept("streamed accepts streamed", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_reversible_streamed,
                p);
        });
    }
    {
        ev::Params p = base_boltzmann();
        p.vp_universe = "expanding";
        p.vp_poisson_source = "density_contrast";
        p.bp_collision_operator = "boltzmann_exact";
        p.bp_cross_section = "softened_gravity";
        expect_accept("prompt cusp accepts registered tuple", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_expanding_prompt_cusp,
                p);
        });
        p.vp_universe = "static";
        expect_reject("prompt cusp rejects static universe", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_expanding_prompt_cusp,
                p);
        });
        p.vp_universe = "expanding";
        p.bp_collision_strength = 0.0;
        expect_reject("prompt cusp rejects zero collision strength", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::boltzmann_poisson_expanding_prompt_cusp,
                p);
        });
    }
    {
        ev::Params p = base_boltzmann();
        p.bp_collision_operator = "spherical_dsmc";
        p.bp_cross_section = "constant";
        p.bp_boltzmann_kernel = "isotropic";
        expect_accept("constant SphDSMC accepts constant", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::sidm_sphdsmc_constant, p);
        });
        p.bp_cross_section = "yukawa_born";
        expect_reject("constant SphDSMC rejects Yukawa", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::sidm_sphdsmc_constant, p);
        });
        p.bp_boltzmann_kernel = "auto";
        expect_accept("velocity SphDSMC accepts Yukawa", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::sidm_sphdsmc_velocity_dependent, p);
        });
        p.bp_cross_section = "constant";
        expect_reject("velocity SphDSMC rejects constant", [&] {
            ev::validate_solver_task_profile(
                ev::SolverTaskProfile::sidm_sphdsmc_velocity_dependent, p);
        });
    }

    expect_accept("low EV WPIC accepts low", [] {
        ev::validate_einstein_vlasov_wpic_task_profile(
            ev::EinsteinVlasovWPICTaskProfile::low_compactness, "low");
    });
    expect_reject("low EV WPIC rejects high", [] {
        ev::validate_einstein_vlasov_wpic_task_profile(
            ev::EinsteinVlasovWPICTaskProfile::low_compactness, "high");
    });
    expect_accept("BH EV WPIC accepts high", [] {
        ev::validate_einstein_vlasov_wpic_task_profile(
            ev::EinsteinVlasovWPICTaskProfile::black_hole_formation, "high");
    });
    expect_reject("BH EV WPIC rejects all", [] {
        ev::validate_einstein_vlasov_wpic_task_profile(
            ev::EinsteinVlasovWPICTaskProfile::black_hole_formation, "all");
    });
    expect_accept("expanding VP accepts zero strength", [] {
        ev::validate_expanding_wpic_task_profile(
            ev::ExpandingWPICTaskProfile::collisionless_vlasov_poisson, 0.0);
    });
    expect_reject("expanding VP rejects subhalo collision", [] {
        ev::validate_expanding_wpic_task_profile(
            ev::ExpandingWPICTaskProfile::collisionless_vlasov_poisson, 0.1);
    });
    expect_accept("subhalo accepts positive strength", [] {
        ev::validate_expanding_wpic_task_profile(
            ev::ExpandingWPICTaskProfile::collisional_subhalo, 0.1);
    });
    expect_reject("subhalo rejects collisionless", [] {
        ev::validate_expanding_wpic_task_profile(
            ev::ExpandingWPICTaskProfile::collisional_subhalo, 0.0);
    });
    expect_accept("constant SIDM WPIC accepts constant", [] {
        ev::validate_sidm_wpic_task_profile(
            ev::SIDMWPICTaskProfile::constant_cross_section,
            "constant", 1.0);
    });
    expect_reject("constant SIDM WPIC rejects power law", [] {
        ev::validate_sidm_wpic_task_profile(
            ev::SIDMWPICTaskProfile::constant_cross_section,
            "powerlaw", 1.0);
    });
    expect_accept("velocity SIDM WPIC accepts Yukawa", [] {
        ev::validate_sidm_wpic_task_profile(
            ev::SIDMWPICTaskProfile::velocity_dependent_cross_section,
            "yukawa_born", 1.0);
    });
    expect_reject("velocity SIDM WPIC rejects zero strength", [] {
        ev::validate_sidm_wpic_task_profile(
            ev::SIDMWPICTaskProfile::velocity_dependent_cross_section,
            "powerlaw", 0.0);
    });

    if (failures != 0) {
        return 1;
    }
    std::cout << "All CPU task-profile contracts passed.\n";
    return 0;
}
