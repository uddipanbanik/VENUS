
// Start the configured CPU Eulerian solver. The parameter file and command
// arguments select Einstein--Vlasov, Vlasov--Poisson or Boltzmann--Poisson
// physics, its initial or restart state and numerical controls.
// The shared implementation constructs the distribution and field, advances
// transport and any collisions in the prescribed order, and writes physical
// moments, invariant ledgers and restart records. This entry is common to
// serial, OpenMP and MPI builds. Developed by Uddipan Banik.

#include "ev/solver.hpp"

// Delegate the complete configured physics pipeline and its error handling.
int main(int argc, char** argv) {
    return ev::run_solver(argc, argv);
}
