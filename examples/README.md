# Build, run, and analyze an example

These examples introduce the interface in short, reproducible steps. Most run
for a small physical time so that you can check your installation quickly.
The low-compactness relativistic example follows the halo for long enough to
check its stated outcome. The expanding exact grid example checks initial-data
setup at zero time; expanding WPIC examples demonstrate actual evolution.
Resolution and duration for a scientific result belong to that physical problem.

Run the commands from the repository root, the folder containing the main
README. A **run** is one evolution of a chosen initial distribution. An
**input** specifies that distribution, its gravitational model, numerical
resolution, physical parameters, and output choices. An **output directory**
collects the saved results of that run.

## Find the right script

| What you want to do | Location |
|---|---|
| Compile a laptop or cluster version | [`scripts/build/`](../scripts/build/) |
| Read the physical and numerical inputs | [`inputs/`](inputs/), then [`catalogue.json`](catalogue.json) |
| Start a named example | [`run/`](run/) |
| Submit that example to a Slurm cluster | [`slurm/`](slurm/) |
| Submit build and GPU-test jobs | [`scripts/slurm/`](../scripts/slurm/) |
| Resume an example | `scripts/run_example.py NAME --resume --end-time T` |
| Read histories and draw profiles | [`scripts/analyze/run.py`](../scripts/analyze/run.py) |
| Refresh checksums after a reviewed source update | [`scripts/check/update_manifests.py`](../scripts/check/update_manifests.py) |
| Verify distributed file checksums | [`scripts/check/source_manifest.py`](../scripts/check/source_manifest.py) |
| Run numerical tests and all example runs | [`scripts/check/`](../scripts/check/) |
| Find specialist initial-data and analysis tools | [Script reference](../notes/SCRIPT_REFERENCE.md) |

The [cluster guide](../docs/CLUSTERS.md) supplies IAS Typhon, IAS Apollo and
Perimeter Symmetry recipes, module choices and portable scheduler settings.

## 1. Build

Install a C++17 compiler and Python 3.8 or newer, then install the Python
packages listed in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash scripts/build/cpu.sh
bash scripts/build/tools.sh
```

The `.venv` folder keeps the Python packages together for this project.
Activate it with `source .venv/bin/activate` in each new Terminal. An existing
scientific Python environment with the required packages can also be used.

A successful build creates the executables listed below. The second build
supplies small input generators and diagnostic tools.
Compiler choices and flags can be passed through `CXX`, `MPICXX`,
`BASE_CXXFLAGS`, and `OPENMP_FLAGS`. `BUILD_JOBS` sets simultaneous compiler
processes. On macOS, the build script first checks whether the default compiler links
a small program. If its SDK is incompatible, it tries the installed macOS 15
SDK. Explicit `CXX`, `SDKROOT`, and `BASE_CXXFLAGS` choices take precedence.

| Build script | What it creates |
|---|---|
| `scripts/build/cpu.sh` | Portable CPU solvers in `cpu/` |
| `scripts/build/cpu_openmp.sh` | CPU grid/SphDSMC executables ending in `_openmp` |
| `scripts/build/cpu_mpi.sh` | Collisionless CPU grid executables ending in `_mpi` |
| `scripts/build/gpu.sh` | CUDA programs in `build/gpu/` |
| `scripts/build/gpu_mpi.sh` | CUDA and distributed programs in `build/gpu-mpi/` |
| `scripts/build/tools.sh` | C++ input generators and diagnostic utilities |

For CUDA, set `GPU_ARCH` to the GPU compute capability, such as `80` for A100
or `90` for H200. Use the site's compatible compiler, CUDA, and MPI modules. On IAS,
`source scripts/build/ias_environment.sh` selects the supplied compiler setup
and preserves the site MPI paths while adding your Python installation.
MPI is a library that lets separate processes exchange data; GPU MPI uses one
process per GPU. The [CPU](../cpu/README.md) and [GPU](../gpu/README.md) guides
explain which work is divided between processes and which work is shared.

## 2. Run your first isolated halo

```bash
bash examples/run/vp_wpic_static.sh
```

The script chooses the collisionless Newtonian equations and creates a small
initial distribution of weighted particles. A weighted particle stands for a
specified amount of physical mass. The output goes to
`output/examples/vp_wpic_static_portable/`. The saved command is beside it,
in `vp_wpic_static_portable.command.json`.

For a different destination:

```bash
bash examples/run/vp_wpic_static.sh --output output/my_first_halo
python3 scripts/analyze/run.py output/my_first_halo --plot --r-max 10
```

This analysis writes a table summary and density/slope movie under
`output/my_first_halo/analysis/`. Density describes where mass lies; the
logarithmic slope describes how quickly it changes with radius. The histories
record how integrated physical quantities change with time. A successful
example prints `Saved run:` and the output path after checking that the saved
history reaches the requested endpoint. The short example shows how to use
the interface; the duration is chosen for installation checks.

Use `--dry-run` to print a command before executing it, and `--end-time T` to
choose a physical endpoint. Pick a fresh output directory for a fresh run.
For a grid example, its `.ini` file is a plain-text record of the parameters.
For a particle example, `catalogue.json` supplies the complete argument list.
The run script creates any required example data with the shipped generators.

## 3. Choose another physical problem

“Static” means zero cosmological expansion. The gravitational field still
responds to the matter. “Expanding” selects a prescribed cosmological background.
Relativistic isolated calculations evolve the spacetime metric, the field that
determines distances, times, and particle trajectories in general relativity.

Each row links a runnable script and a matching scheduler script. Add
`--variant openmp` for supported CPU grid tasks after building that version.
The MPI scheduler links use the matching `_mpi` executables. CPU particle
methods use a single process and its allocated threads.

### Non-relativistic collisionless / static

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `vp_grid_static` | [run](run/vp_grid_static.sh) | [Slurm](slurm/vp_grid_static.sbatch); [MPI](slurm/vp_grid_static_mpi.sbatch) | [`vp_grid_static.ini`](inputs/vp_grid_static.ini) |
| `vp_wpic_static` | [run](run/vp_wpic_static.sh) | [Slurm](slurm/vp_wpic_static.sbatch) | [parameters](catalogue.json) |
| `gpu_vp_static` | [run](run/gpu_vp_static.sh) | [Slurm](slurm/gpu_vp_static.sbatch) | [parameters](catalogue.json) |

### Non-relativistic collisionless / expanding

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `vp_grid_expanding` | [run](run/vp_grid_expanding.sh) | [Slurm](slurm/vp_grid_expanding.sbatch); [MPI](slurm/vp_grid_expanding_mpi.sbatch) | [`vp_grid_expanding.ini`](inputs/vp_grid_expanding.ini) |
| `vp_wpic_expanding` | [run](run/vp_wpic_expanding.sh) | [Slurm](slurm/vp_wpic_expanding.sbatch) | [parameters](catalogue.json) |
| `gpu_vp_expanding` | [run](run/gpu_vp_expanding.sh) | [Slurm](slurm/gpu_vp_expanding.sbatch); [MPI](slurm/gpu_vp_expanding_mpi.sbatch) | [parameters](catalogue.json) |

### Non-relativistic collisional / subhalos / static

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `subhalo_static` | [run](run/subhalo_static.sh) | [Slurm](slurm/subhalo_static.sbatch) | [parameters](catalogue.json) |
| `gpu_subhalo_static` | [run](run/gpu_subhalo_static.sh) | [Slurm](slurm/gpu_subhalo_static.sbatch) | [parameters](catalogue.json) |

### Non-relativistic collisional / subhalos / expanding

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `subhalo_expanding` | [run](run/subhalo_expanding.sh) | [Slurm](slurm/subhalo_expanding.sbatch) | [parameters](catalogue.json) |
| `gpu_subhalo_expanding` | [run](run/gpu_subhalo_expanding.sh) | [Slurm](slurm/gpu_subhalo_expanding.sbatch); [MPI](slurm/gpu_subhalo_expanding_mpi.sbatch) | [parameters](catalogue.json) |

### Non-relativistic collisional / SIDM / static

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `sidm_sphdsmc_constant` | [run](run/sidm_sphdsmc_constant.sh) | [Slurm](slurm/sidm_sphdsmc_constant.sbatch) | [`sidm_sphdsmc_constant.ini`](inputs/sidm_sphdsmc_constant.ini) |
| `sidm_sphdsmc_velocity_dependent` | [run](run/sidm_sphdsmc_velocity_dependent.sh) | [Slurm](slurm/sidm_sphdsmc_velocity_dependent.sbatch) | [`sidm_sphdsmc_velocity_dependent.ini`](inputs/sidm_sphdsmc_velocity_dependent.ini) |
| `sidm_wpic_constant` | [run](run/sidm_wpic_constant.sh) | [Slurm](slurm/sidm_wpic_constant.sbatch) | [parameters](catalogue.json) |
| `sidm_wpic_powerlaw` | [run](run/sidm_wpic_powerlaw.sh) | [Slurm](slurm/sidm_wpic_powerlaw.sbatch) | [parameters](catalogue.json) |
| `gpu_sidm_constant` | [run](run/gpu_sidm_constant.sh) | [Slurm](slurm/gpu_sidm_constant.sbatch); [MPI](slurm/gpu_sidm_constant_mpi.sbatch) | [parameters](catalogue.json) |
| `gpu_sidm_power_law` | [run](run/gpu_sidm_power_law.sh) | [Slurm](slurm/gpu_sidm_power_law.sbatch); [MPI](slurm/gpu_sidm_power_law_mpi.sbatch) | [parameters](catalogue.json) |
| `gpu_sidm_yukawa_born` | [run](run/gpu_sidm_yukawa_born.sh) | [Slurm](slurm/gpu_sidm_yukawa_born.sbatch); [MPI](slurm/gpu_sidm_yukawa_born_mpi.sbatch) | [parameters](catalogue.json) |
| `gpu_sidm_explicit_law` | [run](run/gpu_sidm_explicit_law.sh) | [Slurm](slurm/gpu_sidm_explicit_law.sbatch); [MPI](slurm/gpu_sidm_explicit_law_mpi.sbatch) | [parameters](catalogue.json) |

### Non-relativistic collisional / encounter operators / static

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `bp_fokker_planck` | [run](run/bp_fokker_planck.sh) | [Slurm](slurm/bp_fokker_planck.sbatch) | [`bp_fokker_planck.ini`](inputs/bp_fokker_planck.ini) |
| `bp_landau` | [run](run/bp_landau.sh) | [Slurm](slurm/bp_landau.sbatch) | [`bp_landau.ini`](inputs/bp_landau.ini) |
| `bp_exact` | [run](run/bp_exact.sh) | [Slurm](slurm/bp_exact.sbatch) | [`bp_exact.ini`](inputs/bp_exact.ini) |
| `bp_finite_angle` | [run](run/bp_finite_angle.sh) | [Slurm](slurm/bp_finite_angle.sbatch) | [`bp_finite_angle.ini`](inputs/bp_finite_angle.ini) |
| `bp_reversible` | [run](run/bp_reversible.sh) | [Slurm](slurm/bp_reversible.sbatch) | [`bp_reversible.ini`](inputs/bp_reversible.ini) |
| `bp_reversible_sampled` | [run](run/bp_reversible_sampled.sh) | [Slurm](slurm/bp_reversible_sampled.sbatch) | [`bp_reversible_sampled.ini`](inputs/bp_reversible_sampled.ini) |
| `bp_reversible_streamed` | [run](run/bp_reversible_streamed.sh) | [Slurm](slurm/bp_reversible_streamed.sbatch) | [`bp_reversible_streamed.ini`](inputs/bp_reversible_streamed.ini) |
| `bp_canonical_orbit` | [run](run/bp_canonical_orbit.sh) | [Slurm](slurm/bp_canonical_orbit.sbatch) | [parameters](catalogue.json) |

### Non-relativistic collisional / encounter operators / expanding

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `bp_expanding_prompt_cusp` | [run](run/bp_expanding_prompt_cusp.sh) | [Slurm](slurm/bp_expanding_prompt_cusp.sbatch) | [`bp_expanding_prompt_cusp.ini`](inputs/bp_expanding_prompt_cusp.ini) |

### Relativistic collisionless / isolated

| Example | Run script | Slurm script | Input |
|---|---|---|---|
| `ev_grid` | [run](run/ev_grid.sh) | [Slurm](slurm/ev_grid.sbatch); [MPI](slurm/ev_grid_mpi.sbatch) | [`ev_grid.ini`](inputs/ev_grid.ini) |
| `ev_wpic_low` | [run](run/ev_wpic_low.sh) | [Slurm](slurm/ev_wpic_low.sbatch) | [parameters](catalogue.json) |
| `ev_wpic_schwarzschild` | [run](run/ev_wpic_schwarzschild.sh) | [Slurm](slurm/ev_wpic_schwarzschild.sbatch) | [parameters](catalogue.json) |
| `ev_wpic_eddington_finkelstein` | [run](run/ev_wpic_eddington_finkelstein.sh) | [Slurm](slurm/ev_wpic_eddington_finkelstein.sbatch) | [parameters](catalogue.json) |

The encounter-operator examples introduce numerical collision methods. A star-cluster
application supplies its stellar encounter law and validates the chosen approximation.
Relativistic elastic material scattering is supplied as a component-tested C++ library;
its dedicated production examples will accompany the application runners.

## 4. Submit a cluster job

Slurm is a scheduler: it reserves the requested processors or GPUs, then
executes your script. First load the site's compiler/CUDA/MPI environment
and build the version you intend to use. Set the cluster's partition, account,
and QOS using its published instructions. A partition selects a group of nodes;
QOS selects the site's scheduling limits.

```bash
bash scripts/slurm/submit.sh vp_wpic_static --partition YOUR_CPU_PARTITION
bash scripts/slurm/submit.sh gpu_vp_static --partition YOUR_GPU_PARTITION
bash scripts/slurm/submit.sh gpu_vp_expanding_mpi --partition YOUR_GPU_PARTITION
```

To choose a fresh run directory, put simulation options after `--`:

```bash
bash scripts/slurm/submit.sh gpu_sidm_power_law_mpi \
  --partition YOUR_GPU_PARTITION -- --output output/my_cluster_run
```

The options before `--` go to Slurm; the options after it go to the example.
For an MPI build used for a single-GPU run, add
`--gpu-build build/gpu-mpi` after `--` as well.
The helper supplies `VENUS_ROOT` and submits the corresponding file from
`examples/slurm/`. It starts no simulation on the login node. The scheduler's
log is `slurm-JOBID.out` in the submission directory, and simulation output
uses the location shown by the run script. MPI templates first prepare any
initial data once, then start the MPI
program through `mpiexec`. The site's MPI installation must support its
scheduler. Set `VENUS_MPI_LAUNCHER=srun` only when your site supports direct
Slurm launches with the loaded MPI library. On Symmetry, the tested HPC-X
installation uses `mpiexec`. A rank means one MPI process.

To build through the scheduler, export `VENUS_ROOT` as the absolute repository
path and `VENUS_BUILD` as `cpu`, `cpu-openmp`, `cpu-mpi`, `gpu`, or `gpu-mpi`,
then submit `scripts/slurm/build.sbatch` with the appropriate site options.
`GPU_ARCH` and `BUILD_JOBS` select the device architecture and build workers.
The two-GPU test job is `scripts/slurm/check_gpu.sbatch`.

## 5. Continue a saved calculation

A **checkpoint**, also called a restart file, stores the complete evolving
state: particles or distribution, time, and any collision clocks and random
state. A density profile contains only selected averages. Resume with the
same physical and numerical settings and a later endpoint:

```bash
bash examples/run/vp_wpic_static.sh --output output/my_first_halo \
  --resume --end-time 0.02
```

The dispatcher selects the correct native restart option for that solver.
Grid continuations write new checkpoints into numbered `restarts_resume_*`
folders, preserving the source checkpoint.
GPU continuation uses the complete binary and all its background/mesh sidecars.
For a new physical branch, call the underlying executable with a fresh output
directory and the saved binary input. The printed command and each solver's
`--help` describe that interface.

## 6. Analyze and test

```bash
python3 scripts/analyze/run.py output/my_first_halo --plot --r-max 10
bash scripts/check/cpu.sh
python3 scripts/check/examples.py cpu --output output/check_all_cpu_examples
bash scripts/check/gpu.sh build/gpu
python3 scripts/check/examples.py gpu --output output/check_all_gpu_examples
```

CPU checks run as one process; request one Slurm task with the desired CPU cores.
Use a separate MPI example launch to exercise multiple processes.
GPU checks and examples run inside a GPU allocation. Choose a fresh destination
for the all-example checks. Each example gets a log, and `results.json` records
its process exit status: `0` means success. The adjacent command record also
stores the requested and achieved end times. The CPU checks include
conservation laws, equilibria,
coordinate transformations, collision rates, binary continuation, output readers,
and scheduler-script behavior. They establish the stated finite test conditions.
A new astrophysical application needs its own resolution and boundary studies.

The analysis script also recognizes individual relativistic particle runs.
It uses the appropriate saved density, compactness, observer, and time
coordinate. Specialist tools under `cpu/tools/` are listed in the
[script reference](../notes/SCRIPT_REFERENCE.md). Coordinate comparisons additionally use the authenticated
Schwarzschild-to-Eddington--Finkelstein map described in the CPU guide.

## Terms used in the guides

| Term | Meaning |
|---|---|
| Distribution function | The amount of matter at each position and velocity |
| Phase space | Positions and velocities considered together |
| Collisionless | Evolution through the collective gravitational field, with the explicit encounter operator switched off |
| Collisional | Evolution that also applies the selected scattering or capture law |
| Eulerian grid | Fixed cells in position and velocity that store the distribution |
| WPIC | Weighted particle-in-cell: particles carry mass weights and supply the gravitational field on a grid |
| SphDSMC | Spherical direct simulation Monte Carlo: persistent particles undergo statistically sampled local encounters |
| Mesh | The set of cells used to represent a field or distribution |
| Timestep | The physical time advanced in one numerical update |
| CFL condition | A restriction on the timestep used to resolve transport or encounter probabilities |
| Locality check | A comparison of a collision cell's width with the distance a particle typically travels between encounters |
| Sidecar | A small companion file recording parameters or the identity of a binary state |
| Kernel | A numerical operation that implements one part of the equations |
| API | The C++ functions a developer calls to use a library |
