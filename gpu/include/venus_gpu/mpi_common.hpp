#pragma once

// Assign a CUDA device to each MPI process before GPU storage is allocated.
// The input communicator identifies each node-local process group. The
// returned device index is local rank modulo the visible CUDA device count. Every rank must establish this binding
// before constructing particle state or a physics module, so their persistent
// allocations belong to the intended device.

#include <mpi.h>

namespace venus_gpu {

/// Bind this rank to its node-local CUDA device before constructing GPU state.
int bind_cuda_device_to_mpi_local_rank(MPI_Comm communicator);

}  // namespace venus_gpu
