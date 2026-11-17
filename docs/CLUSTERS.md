# Run VENUS on a cluster

A cluster supplies processors, a software environment, and a scheduler that
reserves those processors for your job. VENUS uses the same physical inputs
and output formats on a laptop and a cluster. The build selects CPU threads,
MPI processes, or NVIDIA GPUs. The job script requests the corresponding
resources.

Start with the [CPU guide](../cpu/README.md) or [GPU guide](../gpu/README.md)
to understand your first calculation. This page takes that calculation onto
IAS Typhon, IAS Apollo, and Perimeter Symmetry, then explains how to adapt it
to another cluster. All commands begin in the repository root.

## 1. Choose resources that match the solver

| Calculation | Slurm resources | Build |
|---|---|---|
| CPU grid, SphDSMC or WPIC in one process | One task, several CPUs per task | `cpu.sh`; optionally `cpu_openmp.sh` for grid/SphDSMC |
| Distributed collisionless CPU grid | Several tasks, CPUs per task for threads | `cpu_mpi.sh` |
| One NVIDIA GPU | One task and one GPU | `gpu.sh` |
| Distributed supported GPU task | One task per GPU, at least two GPUs | `gpu_mpi.sh` |

A Slurm **task** is one process. An MPI **rank** is one process participating
in an MPI calculation. CPU threads share that process's memory. Allocate as
many CPUs per task as the threads requested by your run. The example scripts
pass `SLURM_CPUS_PER_TASK` to the solver. Requesting additional tasks only
speeds up methods with an MPI implementation.

CPU MPI supports collisionless Einstein--Vlasov and Vlasov--Poisson grid
methods. CPU WPIC, SphDSMC and collisional grid evolution use one process
with shared-memory threads. GPU MPI supports expanding collisionless and
subhalo runs, and static material SIDM runs. The [module tables](../notes/SOLVER_MODULES.md)
identify these choices. Current GPU solvers use NVIDIA CUDA; CPU solvers
provide the path on systems with other GPU vendors.

## 2. Load software, then build

A module selects a compatible installed program and its libraries. Load the
compiler and MPI installation together. For GPU builds, also load CUDA and
CMake. Keep the same environment when compiling and running. VENUS uses
C++17, CMake 3.18 or newer for CUDA, and Python 3.8 or newer. The shipped
Python requirements should be installed in your project environment.

The site helpers below select the documented compiler and library combinations.
They keep the site compiler and MPI paths ahead of Python environments.
This matters when a Python installation also supplies its own MPI compiler.
Set `PYTHON` to the absolute path of your chosen scientific Python before
sourcing a helper. A project environment can be prepared once with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export PYTHON="$PWD/.venv/bin/python"
```

Use a Python installation with the required packages if your site already
provides one. Store the checkout and run outputs on a shared project or
scratch filesystem visible to the allocated nodes. Cluster scratch may have
a retention limit; copy the inputs, outputs and command records you need to
permanent storage.

### IAS Typhon: CPU threads and MPI

On the IAS login host:

```bash
source scripts/build/ias_environment.sh cpu
bash scripts/build/cpu.sh
bash scripts/build/cpu_mpi.sh
```

The helper loads `gcc-toolset/10` and `openmpi/gcc/4.1.0`. It selects your
`PYTHON`, or an available scientific Python installation. To use OpenMP grid
variants, also run `bash scripts/build/cpu_openmp.sh`.

Submit a particle example with one process and four CPU threads:

```bash
bash scripts/slurm/sites/submit.sh ias-typhon vp_wpic_static \
  --cpus-per-task=4 -- --output output/typhon_vp
```

Submit a collisionless grid example with two MPI processes:

```bash
bash scripts/slurm/sites/submit.sh ias-typhon vp_grid_static_mpi \
  --ntasks=2 --cpus-per-task=2 -- --output output/typhon_grid_mpi
```

The site helper selects partition `typhon` and QOS `short`. QOS is a scheduling
class that sets limits such as maximum walltime. Your account must have access
to it. The physical and numerical inputs are the same as the local examples.
Each run writes its own output folder and a neighboring command record.

### IAS Apollo: NVIDIA A100 GPUs

On the login host where the CUDA toolkit is installed:

```bash
source scripts/build/ias_environment.sh gpu
GPU_ARCH=80 bash scripts/build/gpu.sh
GPU_ARCH=80 bash scripts/build/gpu_mpi.sh
```

The helper adds `cudatoolkit/12.4` to the IAS compiler/MPI environment.
`GPU_ARCH=80` generates code for A100. The compiler `nvcc` belongs to the
CUDA toolkit; an allocated GPU node may supply the runtime and driver while
having a different toolkit installation. Build where the selected toolkit is
available, then run those binaries in the GPU allocation. Compilation on a
login host should follow your site's policy.

```bash
bash scripts/slurm/sites/submit.sh ias-apollo gpu_vp_static \
  -- --output output/apollo_vp
bash scripts/slurm/sites/submit.sh ias-apollo gpu_sidm_constant_mpi \
  -- --output output/apollo_sidm_mpi
```

These select partition `apollo` and QOS `gpu-short`. The first requests one
GPU. The second requests two tasks and one GPU per task. Allow the scheduler
to choose the GPUs rather than setting `CUDA_VISIBLE_DEVICES` yourself.

### Perimeter Symmetry: CPU and NVIDIA GPUs

The supplied environment selects GCC 11.2, NVIDIA HPC-X/CUDA 12.6,
CMake 3.27.4 and Python 3.11. HPC-X supplies the MPI library. Its compiler
wrapper defaults to NVIDIA `nvc++`; the helper sets `OMPI_CXX` to the selected
GCC compiler so CPU MPI builds use the same compiler and flags as the other
CPU targets. The helper uses
these module names:

```text
gcc/11.2.0  cmake/3.27.4  nvhpc-hpcx-cuda12/24.9
python/3.11  slurm/slurm/21.08.8
```

For CPU calculations:

```bash
source scripts/build/symmetry_environment.sh cpu
bash scripts/build/cpu.sh
bash scripts/build/cpu_mpi.sh
bash scripts/slurm/sites/submit.sh symmetry-cpu vp_wpic_static \
  -- --output output/symmetry_cpu
bash scripts/slurm/sites/submit.sh symmetry-cpu vp_grid_static_mpi \
  -- --output output/symmetry_cpu_mpi
```

For the RTX 2080 Ti nodes:

```bash
source scripts/build/symmetry_environment.sh gpu
GPU_ARCH=75 bash scripts/build/gpu.sh
GPU_ARCH=75 bash scripts/build/gpu_mpi.sh
bash scripts/slurm/sites/submit.sh symmetry-gpu gpu_vp_static \
  -- --output output/symmetry_gpu
bash scripts/slurm/sites/submit.sh symmetry-gpu gpu_sidm_power_law_mpi \
  -- --output output/symmetry_gpu_mpi
```

CPU examples select `debugq`; these GPU examples select `gpudebugq`.
Use `mpiexec` inside the allocation with this HPC-X configuration. The
supplied MPI templates default to `mpiexec`. Each template lets another
site's compatible installation select `VENUS_MPI_LAUNCHER=srun` explicitly.

Symmetry also provides H200 partitions. Build for `GPU_ARCH=90`, then use
`symmetry-h200` to select `h200debugq`. Keep an H200 build separate from an
RTX 2080 Ti build. Release checks cover RTX 2080 Ti and A100; the H200
resource recipe needs validation on the selected H200 allocation.

## 3. Submit, inspect, resume and analyze

The site helper is a small layer over the same indexed example scripts used
by the portable submission helper. Options before `--` go to Slurm; options
after it go to the example. Choose fresh output folders. If only the MPI GPU
build is installed, its directory also contains the single-GPU programs:

```bash
bash scripts/slurm/sites/submit.sh symmetry-gpu gpu_vp_static \
  -- --gpu-build build/gpu-mpi --output output/symmetry_one_gpu
```

Slurm prints a job number when it accepts a submission. `squeue -u "$USER"`
shows queued/running jobs. `sacct -j JOBID` shows completed jobs and exit
statuses. The scheduler log is `slurm-JOBID.out` in the submission directory.
A successful example prints `Saved run:` after verifying its saved endpoint.
Read that log if the job exits early.

To continue the example, retain the original parameters and choose a later
physical endpoint:

```bash
bash scripts/slurm/sites/submit.sh symmetry-gpu gpu_vp_static \
  -- --gpu-build build/gpu-mpi --output output/symmetry_one_gpu \
  --resume --end-time 0.06
python3 scripts/analyze/run.py output/symmetry_one_gpu --plot --r-max 20
```

Run analysis after the job has completed. This writes `analysis/summary.json`
and the density/slope movie below that run. The examples index explains the
complete restart files and available analysis tools.

The supplied short examples use periodic checkpoints. For production jobs,
choose the checkpoint interval with the requested walltime and your site's
preemption policy. A checkpoint is a complete binary state with its companion
files. Resume from a valid checkpoint and inspect the terminal diagnostics.
The [CPU parallel guide](../cpu/docs/cluster.md#wpic-launch-and-signals)
explains signal handling, administrative checkpoint exits and the numerical
stops that require diagnosis. Site-specific requeue policies belong in your
production scheduler wrapper.

## 4. Adapt the same scripts to another HPC system

| Choice | What to check |
|---|---|
| Compiler | A C++17 compiler; the tested builds use GCC on Linux and Clang on macOS |
| MPI | The same compiler/MPI installation for build and run; begin with `mpiexec` or the site's documented launcher |
| CUDA | An NVIDIA toolkit supported by the host compiler and GPU driver |
| GPU architecture | `75` for RTX 2080 Ti, `80` for A100, `90` for H200; select the actual device's capability |
| Python | An environment containing `requirements.txt`, selected through `PYTHON` |
| Scheduler | Partition, account, QOS, walltime, tasks, CPUs and GPUs matching your allocation |
| Filesystem | A writable shared run directory and enough space for profiles/checkpoints |

Load the site's modules and set `CXX`, `MPICXX`, `CUDACXX`, `PYTHON`,
`GPU_ARCH` and `BUILD_JOBS` as needed. The site helpers are optional; portable
build/run scripts use those choices directly. For example:

```bash
bash scripts/build/cpu_mpi.sh
bash scripts/slurm/submit.sh vp_grid_static_mpi \
  --partition YOUR_CPU_PARTITION --account YOUR_ACCOUNT \
  -- --output output/new_cluster_grid
```

A build for several supported NVIDIA generations can use an explicit list,
for example `GPU_ARCH='75;80;90'`. This increases compilation time and binary
size. Every selected architecture must be supported by the loaded CUDA
toolkit. Each device still runs the same numerical method.

CPU executables also need a rebuild when the compiler or MPI installation
changes. A fresh checkout provides that separation. To rebuild a particular
existing target explicitly, use a command such as
`make -B -C cpu venus_cpu_vlasov_poisson_eulerian_mpi` in the new environment.
The `-B` option forces compilation of that target.

A build directory remembers its compiler, MPI and CUDA choices in the CMake
cache. Use a new build directory after changing that toolchain. For example,
`make gpu-mpi BUILD_DIR=build/h200 GPU_ARCH=90` builds under
`build/h200/gpu-mpi`; pass that directory through `--gpu-build` when running.

The supplied scripts target Slurm. On a PBS/LSF site, retain the build,
example inputs, and `scripts/run_example.py` command; translate the resource
header and launcher to that scheduler's documented form. MPI rank ownership,
restart format and physical model remain those described in the guides.

## 5. Check the installation

Run CPU checks in a one-task CPU allocation. Run CUDA checks in a GPU
allocation, with two GPUs when testing an MPI build:

```bash
bash scripts/check/cpu.sh
bash scripts/check/gpu.sh build/gpu-mpi
```

The [validation record](../release/VALIDATION.md) states the environments and
finite numerical checks actually run. For scientific results, additionally
check resolution, timestep, boundary influence and the collision model's
conservation and locality diagnostics for your chosen initial distribution.
