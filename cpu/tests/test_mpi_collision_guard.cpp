// Focused regression tests for the MPI/operator compatibility guard.

#include "ev/common.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_multi_rank_rejection(const std::string& collision_operator) {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.bp_collision_operator = collision_operator;
    p.NL = 8;

    ev::MpiContext mpi;
    mpi.enabled = true;
    mpi.rank = 0;
    mpi.size = 2;

    bool rejected = false;
    try {
        ev::validate_mpi_execution_mode(p, mpi);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("OpenMP-only")
            != std::string::npos;
    }
    assert(rejected);
}

void test_every_collision_backend_rejects_multiple_ranks() {
    const std::vector<std::string> collision_operators = {
        "fokker_planck",
        "landau_fp",
        "boltzmann_exact",
        "boltzmann_reversible",
        "boltzmann_reversible_sampled",
        "boltzmann_reversible_streamed",
        "spherical_dsmc",
    };
    for (const std::string& collision_operator : collision_operators) {
        expect_multi_rank_rejection(collision_operator);
    }
}

void test_collisionless_transport_allows_valid_l_slab_decomposition() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.bp_collision_operator = "none";
    p.NL = 8;

    ev::MpiContext mpi;
    mpi.enabled = true;
    mpi.rank = 0;
    mpi.size = 2;

    ev::validate_mpi_execution_mode(p, mpi);
}

void test_collision_backend_allows_one_rank() {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.bp_collision_operator = "spherical_dsmc";
    p.NL = 8;

    ev::MpiContext mpi;
    mpi.enabled = false;
    mpi.rank = 0;
    mpi.size = 1;

    ev::validate_mpi_execution_mode(p, mpi);
}

void test_too_many_collisionless_ranks_still_rejected() {
    ev::Params p;
    p.physics_model = "einstein_vlasov";
    p.bp_collision_operator = "none";
    p.NL = 1;

    ev::MpiContext mpi;
    mpi.enabled = true;
    mpi.rank = 0;
    mpi.size = 2;

    bool rejected = false;
    try {
        ev::validate_mpi_execution_mode(p, mpi);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("mpi_size <= NL")
            != std::string::npos;
    }
    assert(rejected);
}

} // namespace

int main() {
    test_every_collision_backend_rejects_multiple_ranks();
    test_collisionless_transport_allows_valid_l_slab_decomposition();
    test_collision_backend_allows_one_rank();
    test_too_many_collisionless_ranks_still_rejected();
    std::cout << "MPI collision guard tests passed\n";
    return 0;
}
