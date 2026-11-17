# VENUS GPU guide

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur

The GPU modules implement weighted particle-in-cell (WPIC) relaxation in
spherical Newtonian gravity using NVIDIA CUDA. Marker weights supply the
self-consistent Poisson field. Local encounters add material scattering or
subhalo scattering and capture. This guide follows the calculation from
build and initial distribution through saved profiles and analysis.
Commands run from the **repository root**.

For a first run, the [examples index](../examples/README.md) links every
build script, input, local run script, and Slurm launcher. It also explains
restarts, analysis, and the technical terms used below.

The [cluster guide](../docs/CLUSTERS.md) supplies IAS Typhon, IAS Apollo and
Perimeter Symmetry recipes, module choices and portable scheduler settings.

## 1. Build for your GPU

Use CMake 3.18 or newer, CUDA, a compatible C++17 host compiler, and
Python 3.8 or newer. Keep `cpu/include`, which supplies the reference
physics and encounter tables. Install the shared analysis dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make gpu GPU_ARCH=80 BUILD_JOBS=8
```

`80` selects an A100. Select the CUDA compute capability of the actual device
when using another GPU. Compute capability is NVIDIA’s device-version number.
For example, `8.0` becomes the build setting `80`. The compiled programs appear in `build/gpu/`.
CMake also accepts an explicit source and build directory:

```bash
cmake -S gpu -B build/gpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DVENUS_GPU_BUILD_TESTS=ON
cmake --build build/gpu -j 8
ctest --test-dir build/gpu --output-on-failure --timeout 120
```

Run device tests inside a GPU allocation. Message Passing Interface (MPI) is optional and uses a separate
build, described in step 6.

## 2. Choose a physical module and background

Vlasov--Poisson (VP) modules describe collisionless relaxation.
Boltzmann--Poisson (BP) modules add the selected encounter operator.
All supplied GPU solvers use WPIC. The expanding modules use a prescribed
Einstein--de Sitter (EdS) background. Distributed runs use the Message Passing
Interface (MPI), with one process per GPU.

Material self-interacting dark matter (SIDM) modules use constant, power-law
or Yukawa--Born cross sections. Subhalo modules instead describe scattering
and capture between extended clumps with evolving masses and internal
velocity dispersions. These are distinct physical models with separate
rate normalizations and conservation budgets.

The [complete option reference](../docs/RUN_OPTIONS.md) lists controls for each
program. The [collision equations](docs/WPIC_COLLISIONS.md) specify the cross
sections, pair sampling and remnant updates.

| Physical category | Background | Executable in `build/gpu/` | MPI support |
|---|---|---|---|
| Non-relativistic collisionless | Static | `venus_gpu_vlasov_poisson_static` or `venus_gpu_vlasov_poisson --universe static` | Single GPU |
| Non-relativistic collisionless | Einstein--de Sitter expansion | `venus_gpu_vlasov_poisson --universe eds` | `_mpi` |
| Non-relativistic collisional, subhalo scattering/capture | Static or EdS | `venus_gpu_subhalo_boltzmann_poisson` | `_mpi` for EdS |
| Non-relativistic collisional, SIDM constant cross section | Static | `venus_gpu_sidm_constant` | `_mpi` |
| Non-relativistic collisional, SIDM softened power law | Static | `venus_gpu_sidm_power_law` | `_mpi` |
| Non-relativistic collisional, SIDM Yukawa--Born | Static | `venus_gpu_sidm_yukawa_born` | `_mpi` |
| Non-relativistic collisional, SIDM explicit-law selector | Static | `venus_gpu_sidm_boltzmann_poisson` | `_mpi` |
| Non-relativistic collisional, star clusters | Model-dependent | Dedicated gravitational stellar-relaxation GPU solver is future work | — |
| Relativistic collisionless | Isolated or cosmological | CPU Einstein--Vlasov (EV) modules are available; relativistic CUDA evolution is future work | — |
| Relativistic collisional, subhalos | Isolated or cosmological | Future work | — |
| Relativistic collisional, SIDM or star clusters | Isolated or cosmological | CPU elastic material library has component-tested scope; GPU/stellar encounter implementations are future work | — |

The [full physics selector](../notes/SOLVER_MODULES.md#physics-and-background-selector)
compares CPU and GPU availability. Relativistic transport requires the equations for spacetime geometry, trajectories through that
geometry and the matter that produces gravity to be solved together. Its CUDA
implementation is a separate development task. Adaptive-grid development
will be distributed when ready.

A static background sets `H=0` while the Poisson field evolves with the
particles. EdS has `H=(2/3)/(t+12)`. Static particle weights represent total
density. EdS VP/subhalo weights represent density above or below the prescribed
homogeneous background, called density contrast. Here `H` is the fractional expansion rate and `t` is simulation time.
The initial-state companion files record these conventions and the radial grid.

## 3. Generate and inspect the input

GPU runs read a compatible **WPIC binary state**, a program-defined byte
format containing marker masses,
positions, momenta, identities and persistent state. The CPU Eulerian INI and
`EVICRWL1` grid files describe a different representation. Generate a small
initial population with the supplied Python tool:

```bash
python3 cpu/tools/generate_bound_gaussian_shell_restart.py \
  --output-directory output/first_gaussian_ic --particles 4096
```

This creates `static_vp/restart_state.bin` and
`expanding_vp/restart_state.bin` below the selected directory, with companion files describing their format, grid and background, and a
manifest listing the generated files and parameters. The output directory must be fresh.
Both files describe the same physical marker realization: a Gaussian density
shell centered at radius 10, of width 1, with support between radii 7 and 13.
Their physical positions and velocities agree, while the expanding file stores
peculiar radial velocity `w=u-Hr`: the physical radial velocity `u` minus
the expansion velocity `Hr` at radius `r`. The generator documents the velocity
sampling and mass normalization. The field mesh has 512 logarithmic cells
between 0.01 and 10000. Repeat those mesh controls when importing either state.

For your own distribution, preserve the WPIC file format and its recorded
background conventions, normalize the represented mass, and verify the sampled density, mean velocity, and velocity dispersion.
The [initial-data tutorial](../cpu/examples/modular/vp_bound_gaussian_shell_gpu.md)
explains the larger research example and CPU/GPU comparison conditions.

## 4. Run, then add encounters

Begin with a short static collisionless relaxation:

```bash
build/gpu/venus_gpu_vlasov_poisson_static \
  --restart output/first_gaussian_ic/static_vp/restart_state.bin \
  --output-directory output/gpu/first_static_vp \
  --radial-cells 512 --radial-minimum 0.01 --radial-maximum 10000 \
  --maximum-timestep 0.0075 --end-time 0.03 --output-interval 0.015 \
  --history-every 1 --restart-every 1
```

The static executable fixes the background and uses the existing VP CUDA
kernel. For EdS, use `venus_gpu_vlasov_poisson --universe eds` with
`expanding_vp/restart_state.bin` and a fresh output directory. The general
VP executable defaults to EdS. Static and expanding VP use orbit steps bounded by `--maximum-timestep`, shortened to reach output or final
times. Each kick--drift--kick step updates velocity, advances position, then
completes the velocity update.

| Control | Meaning |
|---|---|
| `--restart` | Native input state, accompanied by its matching companion files |
| `--universe static` or `eds` | Cosmological background and source convention |
| `--radial-cells`, `--radial-minimum`, `--radial-maximum` | Field/collision mesh matching the state |
| `--maximum-timestep` | Orbit-step ceiling |
| `--end-time`, `--output-interval` | Physical endpoint and profile spacing |
| `--history-every`, `--restart-every` | Number of completed steps between history/checkpoint writes |
| `--output-directory` | Run destination, relative to the launch directory |

Subhalo encounters treat markers as unequal-mass Plummer objects, clumps
with a prescribed smooth central density profile. Select
`venus_gpu_subhalo_boltzmann_poisson`, the background-specific initial file,
the same mesh controls, and positive `--subhalo-collision-strength`. The
[complete example](../cpu/examples/modular/vp_bound_gaussian_shell_gpu.md#subhalo-encounters)
adds the object velocity scale, area per mass, collision interval and accuracy limits. The collision CFL (Courant--Friedrichs--Lewy) safety factor limits the timestep
to resolve encounter probabilities. The locality check compares a collision
cell width with the mean free path, the typical distance between encounters. `--subhalo-adaptive-cell-timestep` advances independent
collision-cell clocks on the fixed radial mesh.

Material SIDM uses the static population and a physical material scattering
law. Replace the executable in the static example by
`venus_gpu_sidm_constant` and add
`--sidm-collision-strength 1 --sidm-sigma0 1`. Use a fresh output directory.
`--sidm-adaptive-shell-timestep` enables independent shell collision clocks.
For a velocity-dependent law select `venus_gpu_sidm_power_law` or
`venus_gpu_sidm_yukawa_born` and supply that law's parameters, listed by
`--help` and derived in [the collision guide](docs/WPIC_COLLISIONS.md).
The power-law model requires a positive softening speed
`--sidm-sigma-u0` as well as its normalization `--sidm-sigma0` and exponent
`--sidm-sigma-power`. The softening speed sets the scale below which the
cross section remains finite. The supplied example makes all three explicit:

```bash
bash examples/run/gpu_sidm_power_law.sh --output output/gpu/first_power_law
```

Its parameter list is in [`examples/catalogue.json`](../examples/catalogue.json).
Use `--dry-run` with an example script to see the complete command. These
short examples use small collision strengths to check the interfaces. Choose
the physical cross section and duration for your application.

Each task enforces its own encounter family. Material scattering conserves
pair momentum and kinetic energy. Subhalo capture transfers orbital energy
into internal remnant energy and changes the population. Quantitative
collision studies need particle-count, timestep, random-seed and locality
checks for their selected model. The timestep must resolve encounter
probabilities, and the collision shell width must resolve the mean free path.

## 5. Locate and analyze the saved state

The example writes to `output/gpu/first_static_vp/`. Normal products are
`metadata.json`, `history_gpu.csv`, numbered `profile_*.csv`, terminal status,
collision histories where applicable, and `restart_state.bin` with its
companion files describing their grid and background. CSV (comma-separated
values) files are text tables with named columns in the documented run code units.
Profiles contain density, enclosed mass, mean velocities, velocity dispersions
and time. Histories
record integrated diagnostics. Profiles are deposited after the final orbit
kick, at the same completed state as the checkpoint.

```bash
python3 gpu/scripts/render_venus_gpu_density_slope_gif.py --help
python3 gpu/scripts/render_venus_gpu_density_slope_gif.py \
  --collisionless output/gpu/first_static_vp --title "Gaussian shell" \
  --output-directory output/gpu/first_static_vp/figures \
  --output output/gpu/first_static_vp/density_slope.gif --r-max 20
```

The shared renderer reads numbered profiles and writes density and slope
frames sampled by physical time. For full particle and gravitational field
output, select `--phase-space-snapshot-times` from the profile schedule and
follow [the snapshot guide](docs/PHASE_SPACE_SNAPSHOTS.md). Keep metadata,
initial state, all companion files, command arguments and the source revision with
each run so its normalization and physical model remain recoverable.

To resume, use the complete `restart_state.bin`, the same mesh/background/
collision controls, and a later `--end-time`. The driver validates existing
history times before appending. A separate branch uses a fresh destination.
`SIGUSR1` or `SIGTERM` requests a complete saved-state write followed by exit status 99. The final
checkpoint name becomes visible only after that write succeeds.
Density CSV files supply analysis data. The binary state preserves the full
particle population, clocks and random state required for continuation.

## 6. Choose one GPU or one MPI rank per GPU

```bash
make gpu-mpi GPU_ARCH=80 BUILD_JOBS=8
mpiexec -n 2 build/gpu-mpi/venus_gpu_vlasov_poisson_mpi --help
```

Run this command within an allocation containing two GPUs. The MPI library
must support your site's scheduler. The example Slurm templates default to
`mpiexec`. Set `VENUS_MPI_LAUNCHER=srun` only for a compatible direct Slurm
configuration. On Symmetry the tested HPC-X installation uses `mpiexec`.

Use the supported MPI task from the table and the complete input/mesh controls
from your single-GPU calculation. MPI VP/subhalo currently use EdS. MPI
material SIDM uses static halos. `--help` checks the interface and does not
start a physical evolution.

A rank is one MPI process. Within each rank, GPU numerical routines divide
particle-to-grid mass assignment, orbit updates, and local collision work
among GPU threads. Particle arrays, moments, fields and collision
workspaces remain in device memory between output/checkpoint events. MPI
combines the density and velocity sums from all ranks to reconstruct the
shared spherical field. Each
physical collision shell/cell has one owner, which samples its pairs once.
Here device memory means GPU memory and host memory means CPU memory.
Migration moves a particle record to the process responsible for its new cell.
Migration exchanges marker records through host memory, so an MPI installation that directly accesses GPU memory is
optional. Mesh arrays are replicated. A very small shell count or an uneven
particle distribution can limit scaling. File output, CPU coordination and operations involving all MPI processes
also contribute to runtime. See [the MPI guide](docs/MULTI_GPU.md)
for ownership and scaling details.

## 7. C++ library interface for developers

An application programming interface (API) is the set of functions a program
calls to use a library. The following example is for extending the C++ code.
The run scripts above provide the interface for a simulation user.

```cpp
#include "venus_gpu/wpic.hpp"

venus_gpu::WpicState state(host_markers);
venus_gpu::VlasovPoissonModule vp(radial_mesh);
venus_gpu::VlasovPoissonOptions options;
options.gravity = 1.0;
vp.advance_kdk(state, time, dt, options);
```

Material collisions use `CollisionModule` and `CollisionOptions`.
Subhalo encounters use `SubhaloCollisionModule` and
`SubhaloCollisionOptions`. Transfer the population at initialization,
explicit output and checkpoints. Successive particle kernels reuse its device
arrays.

## Plotting and GIFs

`gpu/scripts/render_venus_gpu_density_slope_gif.py` reads explicit GPU profile
directories and uses the shared CPU renderer. Begin with:

```bash
python3 gpu/scripts/render_venus_gpu_density_slope_gif.py --help
```

Its input and publication paths are explicit. Generated frames live outside
the solver state and the complete GIF replaces the destination only after it has been written successfully.

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
