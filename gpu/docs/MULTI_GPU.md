# Multi-GPU execution

MPI (Message Passing Interface) allows separate running copies of a program
to exchange data. Each copy is a process, also called a rank. VENUS assigns
one rank to each GPU. Decomposition divides particles between them. Device
binding selects the GPU used by a rank. A marker is a numerical particle
with a mass weight. Start with [the GPU guide](../README.md#6-choose-one-gpu-or-one-mpi-rank-per-gpu)
for build and submission commands.

## Notation

Rank index $`a`$ labels marker set $`P_a`$. $`|P_a|`$ Is its population and
$`N_p`$ the total marker count. Marker $`p`$ has mass $`m_p`$ and assigns
fraction $`a_{pi}`$ to radial cell $`i`$. The local and global cell masses are
$`M_i^{(a)}`$ and $`M_i`$. `Nr` is radial cell count, so `O(Nr)` refers to
communication proportional to the radial mesh size.

## Decomposition and device binding

The VENUS GPU module uses one MPI process per GPU. Rank $`a`$ owns a subset
$`P_a`$ of the marker population:

```math
N_p=\sum_a |P_a|.
```

Call device binding before allocating a `WpicState`:

```cpp
MPI_Init(&argc, &argv);
venus_gpu::bind_cuda_device_to_mpi_local_rank(MPI_COMM_WORLD);
venus_gpu::WpicState local_state(local_markers);
```

The helper constructs a node-local communicator and selects the GPU matching
the local MPI rank. A scheduler may expose one physical GPU as logical device
zero inside each task. Those logical devices still refer to distinct
accelerators.

## Global Vlasov–Poisson field

Every GPU deposits its local extensive moments,

```math
M_i^{(a)}=\sum_{p\in P_a}a_{pi}m_p,
```

then `MPI_Allreduce` forms $`M_i=\sum_a M_i^{(a)}`$, likewise for all other
moments and mass ledgers. Only `O(Nr)` doubles cross MPI. Every rank uploads
the global moments, solves identical spherical Poisson locally, and pushes its
own markers.

```text
local deposit -> allreduce -> old global field -> local half-kick/drift
local deposit -> allreduce -> new global field -> local endpoint half-kick
```

`MpiVlasovPoissonModule` implements both the fused KDK call and the split
transactional phases used by production collision deadlines.

## Correct global collision ownership

A local collision requires both partners from the same physical radial cell.
`MpiSubhaloCollisionModule` assigns complete cells to contiguous rank slabs and
migrates their markers with `MPI_Alltoallv`. Every collision cell is then
complete on exactly one GPU, which samples all its local pairs.

The current implementation is host-staged:

```text
device SoA -> host records -> MPI_Alltoallv -> device SoA
```

The host records provide the complete cell populations required by the
collision sampler. Frequent migrations can dominate runtime. Device packing,
CUDA-aware MPI, neighbor-only exchange and adaptive load balance are possible
extensions to this transfer stage.

## Rank-count-independent physics

The following four choices preserve the physical cell and random trial
identity under a change in rank count:

1. collision cells retain their global CPU cell number after slab ownership.
2. random draws use `(seed, epoch, global cell, substep, trial, dimension)`.
3. fixed-clock reference mode globally reduces the largest marker trial
   frequency, while adaptive mode lets each uniquely owned cell advance its
   own local clock inside the common operator batch.
4. captures stop the affected cell, compact the local population, and rebuild
   the exact proposal before that unfinished cell resumes.

Diagnostics are reduced globally with the appropriate sum, minimum, or
maximum. Rank-count comparisons should inspect the complete unequal-mass
marker population, including IDs, masses, internal states and event counts.

## Portable checkpointing

During output, each rank downloads its local population. Rank zero gathers the
records, sorts by unique marker ID, and writes the same version-6 checkpoint
format used by the one-GPU driver. On restart, every rank reads that file and
the initial round-robin distribution is replaced by collision-cell ownership
at the first collision. RNG counters, collision age, time-scale estimates,
time, and step are all preserved.

## API example

```cpp
venus_gpu::MpiVlasovPoissonModule vp(mesh, MPI_COMM_WORLD);
venus_gpu::MpiSubhaloCollisionModule collisions(mesh, MPI_COMM_WORLD);

vp.advance_kdk(local_state, time, dt, vp_options);
auto diagnostics = collisions.apply(
    local_state, collision_dt, collision_options, rng);
```

Destroy these objects before `MPI_Finalize` so their communicators and device
workspaces are released while MPI remains available.

## Scaling considerations

Collisionless Vlasov--Poisson exchanges only `O(Nr)` moment arrays. Its main
scaling limits are the two all-reductions per KDK step and load imbalance in
the local marker push. Use `venus_gpu_vlasov_poisson_mpi` when no collision
operator is required.

Subhalo collisions add host-staged marker migration. Fixed-clock mode also
adds a global maximum for every mesh-wide collision substep. Adaptive-cell
mode removes that per-substep collective but retains outer-batch and
diagnostic collectives. A short or collision-heavy calculation may
therefore run more slowly on several GPUs than on one. Measure the intended
marker count, radial distribution, collision cadence, and output cadence when
choosing a rank count. Report strong scaling at fixed total problem size and
weak scaling at fixed markers per GPU separately.

For a collisionless run, selecting the collisionless task prevents collision
scheduling entirely. Ranks still perform the global field reductions and
independently push their local marker partitions.
