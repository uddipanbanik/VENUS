# VENUS 1.0 GPU modules

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur

This directory contains the CUDA/C++ weighted particle-in-cell (WPIC)
solver for spherical Newtonian gravity. Weighted markers represent the
phase-space distribution. Their deposited radial moments determine the
Poisson field, which advances their orbits; local sampled pairs apply the
selected collision law. Marker state, moments, fields and collision workspaces
stay in device memory between output or checkpoint events.

The collisionless step is

```text
marker SoA -> CIC/TSC radial deposit -> spherical Poisson solve
           -> kick-drift-kick push -> endpoint deposit/Poisson/kick
```

Material SIDM uses constant, softened power-law, or Yukawa--Born cross
sections in radius-local shells. The separate subhalo operator treats
markers as unequal-mass Plummer objects and samples calibrated scattering or
capture, internal heating, and remnant compaction.

Material collision steps draw disjoint partner assignments with equal
inclusion probability for every physical pair. For a shell of `N` markers
and volume `V`, acceptance and its collision CFL use `m/V`, where `m=N-1`
for even `N` and `m=N` for odd `N`. The mean-free-path diagnostic uses the
physical partner density `(N-1)/V`. Each marker experiences at most one
collision per local step; quantitative runs require collision-CFL convergence.

Collisionless GPU WPIC uses the fixed `--maximum-timestep` KDK orbit step,
shortened only to land exactly on output or final time. Collision batches can
instead use mesh-adaptive local clocks: `--sidm-adaptive-shell-timestep`
advances each material shell independently, while
`--subhalo-adaptive-cell-timestep` advances each subhalo collision cell
independently. The latter groups up to
`--subhalo-adaptive-batch-substeps N` local steps in one kernel launch and
stops a cell immediately after capture so its proposal is rebuilt. Omitting
the flag retains the mesh-wide reference clock.

## Directory map

Selected full-particle and gravitational-field output is described in
[Particle and gravitational-field snapshots](docs/PHASE_SPACE_SNAPSHOTS.md).
The optional schedule uses existing profile times and retains the native
restart state and the actual endpoint potential together.

```text
gpu/
  include/venus_gpu/   public state, VP, collision, MPI, restart APIs
  src/                 CUDA kernels and host orchestration
  tools/               one-task runner mains, restart comparator, table builder
  benchmarks/          focused VP and material-SIDM performance programs
  examples/            restart-based example arguments
  scripts/             generic GPU profile/GIF front end
  docs/                GPU equations, algorithms, API, and MPI details
  ../output/gpu/       root-level run products only; not source
```

## One task, one executable

| Physical task | Single-GPU executable | One-rank-per-GPU executable |
|---|---|---|
| Collisionless Vlasov--Poisson | `venus_gpu_vlasov_poisson` | `venus_gpu_vlasov_poisson_mpi` |
| Subhalo Boltzmann--Poisson | `venus_gpu_subhalo_boltzmann_poisson` | `venus_gpu_subhalo_boltzmann_poisson_mpi` |
| Material-SIDM BP with explicit law | `venus_gpu_sidm_boltzmann_poisson` | `venus_gpu_sidm_boltzmann_poisson_mpi` |
| Constant material SIDM | `venus_gpu_sidm_constant` | `venus_gpu_sidm_constant_mpi` |
| Softened power-law material SIDM | `venus_gpu_sidm_power_law` | `venus_gpu_sidm_power_law_mpi` |
| Yukawa--Born material SIDM | `venus_gpu_sidm_yukawa_born` | `venus_gpu_sidm_yukawa_born_mpi` |

`run_wpic_subhalo_gpu` and `run_wpic_subhalo_gpu_mpi` are compatibility
dispatchers. They select collisionless or subhalo behavior from the
strength convention and are not the preferred interface for a new run.

Focused benchmarks are `benchmark_vlasov_poisson_gpu` and
`benchmark_material_sidm_gpu`; `benchmark_wpic` remains the aggregate
compatibility benchmark. They measure runtime for their stated mesh, marker
count and output cadence. Scientific use also requires the corresponding
resolution and timestep checks.

## Requirements and build

The complete build requires CMake 3.18 or newer, CUDA, a C++17 host compiler,
Python 3.8 or newer, and the sibling `../cpu/include` tree. MPI is optional.
Select the compute capability for the target GPU; for example, `80` for an
A100 or `90` for an H100/H200:

```bash
cd VENUS/gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 8
```

The CPU include tree supplies the authoritative unequal-Plummer calibration
table and CPU formula implementations used by comparison programs. Override
it only with a compatible tree:

```bash
cmake -S . -B build -DVENUS_CPU_REFERENCE_INCLUDE=/path/to/VENUS/cpu/include -DCMAKE_CUDA_ARCHITECTURES=80
```

For MPI:

```bash
cmake -S . -B build_mpi -DVENUS_GPU_ENABLE_MPI=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build_mpi -j 8
srun --ntasks=2 --gpus-per-task=1 ./build_mpi/venus_gpu_vlasov_poisson_mpi --help
```

Use one MPI rank per visible GPU. The MPI layer all-reduces radial VP moments.
For collisions, each complete physical shell or cell has one owner, which
samples its pairs once. Marker migration passes through host memory, allowing
ordinary MPI to exchange the records.

## Running

GPU runners import a versioned CPU/GPU WPIC restart containing the physical
marker population and persistent clocks. Single-GPU collisionless VP accepts static and
Einstein--de Sitter backgrounds. Its default is Einstein--de Sitter, with
alpha=2/3 and time offset 12. The MPI VP runner retains the expanding setup:

```bash
./build/venus_gpu_vlasov_poisson --restart restart_state.bin --output-directory output/vp --end-time 60
```

For the static 512-cell restart over 0.01--10000, select its background and mesh
explicitly. The corresponding restart sidecars must describe those controls:

```bash
./build/venus_gpu_vlasov_poisson --universe static \
  --restart static_restart.bin --output-directory output/static_vp --end-time 1 \
  --radial-cells 512 --radial-minimum 0.01 --radial-maximum 10000 \
  --maximum-timestep 0.0075
```

Static marker masses source total density. EdS marker masses source density
contrast, with no second homogeneous-density subtraction in the field solve.
Both VP backgrounds retain fixed-cap KDK with output/final-time shortening.
The collisionless task rejects collision options. Single-GPU subhalo tasks also accept static backgrounds; MPI VP/subhalo tasks retain EdS.


The [bound Gaussian shell example](../cpu/examples/modular/vp_bound_gaussian_shell_gpu.md) generates the example VP initial states
with `cpu/tools/generate_bound_gaussian_shell_restart.py`. It gives both complete
static and expanding commands, including their 512-cell mesh and output cadence.
The two files represent the same physical marker realization.

Subhalo scattering/capture requires an explicit positive strength:

```bash
./build/venus_gpu_subhalo_boltzmann_poisson --restart restart_state.bin --output-directory output/subhalo --end-time 60 --subhalo-collision-strength 0.1 --subhalo-adaptive-cell-timestep
```

Material SIDM evolves a static total-density halo. The constant-law
runner selects elastic material encounters and requires their parameters:

```bash
./build/venus_gpu_sidm_constant --restart restart_state.bin --output-directory output/sidm_constant --end-time 60 --sidm-collision-strength 1 --sidm-sigma0 1 --sidm-adaptive-shell-timestep
```

Use `venus_gpu_sidm_boltzmann_poisson` when selecting the cross-section law
explicitly, or the power-law/Yukawa--Born task names when the law should be
fixed by the executable. The material runners are locked to static
total-density halos. Quantitative core evolution requires shell-locality,
marker-number, timestep, random-seed, and rank-count convergence studies.
Every task rejects options belonging to another collision family. Run the
chosen executable with `--help` for cadence, mesh, output, locality, and
checkpoint controls.

## State, output, and restart

GPU state uses structure-of-arrays storage and persistent device workspaces.
Normal products include `metadata.json`, `history_gpu.csv`,
`profile_NNNNN.csv`, the applicable collision history, terminal status, and
`restart_state.bin`. Atomic checkpoint replacement protects the last
complete state. The restart carries time, step, ordered markers and their
weights, collision clocks, random counters, and subhalo internal state.
Keep the original command arguments with the run. A continuation uses the
same mesh, collision parameters, and numerical controls, together with the
saved checkpoint and a later end time.

The driver appends histories in an existing output directory and preserves
the earlier completed profile when a checkpoint falls between output times.
It checks existing profile and history times before writing. A diagnostic
later than the supplied checkpoint causes a clear error and leaves those
files intact. Use a new output directory when evolving an earlier checkpoint
along a separate continuation. `SIGUSR1` and `SIGTERM` request an atomic
checkpoint followed by exit status 99.

Profiles use moments deposited after the final KDK kick, so their radial
mean and variance describe the same completed state as the checkpoint.

`compare_wpic_restarts` compares compatible CPU/GPU checkpoints:

```bash
./build/compare_wpic_restarts cpu_restart.bin gpu_restart.bin 5e-14 5e-12
```

## Public C++ API

```cpp
#include "venus_gpu/wpic.hpp"

venus_gpu::WpicState state(host_markers);
venus_gpu::VlasovPoissonModule vp(radial_mesh);
venus_gpu::VlasovPoissonOptions options;
options.gravity = 1.0;
vp.advance_kdk(state, time, dt, options);
```

Material collisions use `CollisionModule` and `CollisionOptions`;
subhalo encounters use `SubhaloCollisionModule` and
`SubhaloCollisionOptions`. Transfer the population at initialization,
explicit output and checkpoints; successive particle kernels reuse its device
arrays.

## Plotting and GIFs

`scripts/render_venus_gpu_density_slope_gif.py` reads explicit GPU profile
directories and uses the shared CPU renderer. Begin with:

```bash
python3 scripts/render_venus_gpu_density_slope_gif.py --help
```

Its input and publication paths are explicit; generated frames live outside
the solver state and the GIF is published atomically.

## Documentation

- [`../notes/SOLVER_MODULES.md`](../notes/SOLVER_MODULES.md): exhaustive
  one-by-one CPU/GPU module and entry-point catalogue.
- [`../docs/venus_manual.pdf`](../docs/venus_manual.pdf): unified
  physics-oriented manual with a separate GPU implementation part.
- [`docs/WPIC_VLASOV_POISSON.md`](docs/WPIC_VLASOV_POISSON.md): equations
  and annotated CUDA VP algorithm.
- [`docs/WPIC_COLLISIONS.md`](docs/WPIC_COLLISIONS.md): material and
  subhalo collision equations and pairing.
- [`docs/CPU_VS_GPU.md`](docs/CPU_VS_GPU.md): implementation differences.
- [`docs/MULTI_GPU.md`](docs/MULTI_GPU.md): ownership, reductions, and
  scaling limits.

## Optional coupled subhalo clocks

`--coupled-receiver-clocks` selects the gravitational orbit bound and the
supported complete receiver field norm. It requires the single-GPU subhalo
task, adaptive cells, positive collision-timescale fraction and a fresh output
segment. `--collision-prestep-coordinate-due` adds the positive-age due
check before KDK. Both options default to off. The [clock guide](docs/COUPLED_RECEIVER_CLOCKS.md)
derives their equations, static/EdS scope, maximum-norm fallback, conservation
checks and restart chronology. The initial Gaussian restart for a background
is shared by its VP and BP calculations.
