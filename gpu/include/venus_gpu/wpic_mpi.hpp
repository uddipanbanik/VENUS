#pragma once

// Expose the collective one-rank-per-GPU interfaces in one include.
// The caller supplies an MPI communicator and binds its local CUDA device
// before constructing persistent state. Included adapters reduce global field
// moments or assign unique collision owners, then return local particles and
// global diagnostics. Collective calls must occur in the same rank order.

#include "venus_gpu/mpi_common.hpp"
#include "venus_gpu/mpi_material_sidm.hpp"
#include "venus_gpu/mpi_subhalo_collisions.hpp"
#include "venus_gpu/mpi_vlasov_poisson.hpp"
