# Cluster and parallel execution

Developed by Uddipan Banik and Frank van den Bosch

Start with [Run VENUS on a cluster](../../docs/CLUSTERS.md) for the step-by-step
IAS Typhon/Apollo and Perimeter Symmetry recipes. This page maps the VENUS CPU task executables onto OpenMP,
`std::thread`, Message Passing Interface (MPI), and Slurm resources. Physics and parameter definitions
are in the [unified manual](../../docs/venus_manual.pdf) and
[module catalogue](../../notes/SOLVER_MODULES.md).

## Supported parallel shapes

| Task family | Executable suffix | MPI ranks | Threads per rank |
|---|---|---:|---:|
| Eulerian Einstein--Vlasov | `_mpi` | one or more `L` slabs | one or more OpenMP threads |
| Eulerian collisionless Vlasov--Poisson | `_mpi` | one or more `L` slabs | one or more OpenMP threads |
| Eulerian EV/VP portable | none | one process | `std::thread` |
| Grid Boltzmann--Poisson collisions | `_openmp` or none | exactly one | OpenMP or `std::thread` |
| spherical direct simulation Monte Carlo (SphDSMC) | `_openmp` or none | exactly one | OpenMP or `std::thread` |
| CPU weighted particle-in-cell (WPIC) | none | exactly one | shared-memory threads where supported |

CPU collision calculations use one rank, which owns the coefficients,
angular momentum boundary fluxes, random sequence and persistent collision
shells. Threads within that rank divide the work. The launcher checks this
rank requirement before beginning the calculation.

## Build the intended task

From `VENUS/cpu`:

```bash
make organized-solvers
make cpu-task-solvers-openmp
make cpu-task-solvers-mpi
```

Examples of the resulting names:

- `venus_cpu_einstein_vlasov_eulerian_mpi`
- `venus_cpu_vlasov_poisson_eulerian_mpi`
- `venus_cpu_boltzmann_poisson_fokker_planck_openmp`
- `venus_cpu_boltzmann_poisson_exact_openmp`
- `venus_cpu_boltzmann_poisson_reversible_streamed_openmp`
- `venus_cpu_sidm_sphdsmc_constant_openmp`
- `venus_cpu_sidm_sphdsmc_velocity_dependent_openmp`
- `venus_cpu_subhalo_wpic`

The generic `venus_einstein_vlasov*`, `venus_modular_solver*`, and
original `run_*` programs are compatibility/specialist interfaces. Prefer a
task name so a mismatched physical model fails before output is created.

## MPI decomposition for Eulerian EV and VP

The distribution is stored as `f(r,w,L)`. The global `L` grid is divided
into contiguous slabs:

1. every rank keeps the complete `r` and `w` grids.
2. each rank owns only its local `L` values, quadrature weights, and
   `f(r,w,L_local)` slab.
3. each rank integrates local density, pressure/current, and other radial
   moments.
4. `MPI_Allreduce` forms the global radial arrays.
5. all ranks solve or receive the same radial metric/Poisson field. And
6. transport proceeds independently on local `L` slabs until the next
   moment reduction.

This layout keeps rank-local phase space storage contiguous and communicates
radial arrays rather than the full distribution. The solver rejects more
ranks than angular momentum cells. Distributed restarts record the rank
layout. Restart with the recorded rank count unless the selected reader
explicitly documents repartitioning.

The MPI path supports:

- Eulerian Einstein--Vlasov.
- static Eulerian Vlasov--Poisson.
- expanding Eulerian Vlasov--Poisson. And
- the collisionless VP equation represented by a BP configuration only when
  `bp_collision_operator = none`.

## Shared-memory decomposition

The loop wrapper in `include/ev/parallel.hpp` selects OpenMP when compiled
with `-fopenmp` and otherwise uses `std::thread`. Work is assigned so that
each output cell or shell has one writer. Reductions use thread-local partial
sums.

Threaded work includes:

- EV/VP moment integrals, timestep scans, advection, projections, field
  diagnostics, and adaptive velocity-support scans.
- VP finite volume and semi-Lagrangian transport, Hamiltonian tables, and
  boundary/escape accounting.
- BP bath/coefficient construction, conservative Fokker--Planck fluxes, RKL2
  stages, positivity repair, exact/reversible/sampled/streamed finite-angle
  loops, and collision diagnostics.
- SphDSMC gravity, orbit retries, adaptive shell construction, and independent
  no-time-counter collision shells. And
- WPIC orbit pushes and independent collision cells/shells.

Some deterministic mesh-deposition steps remain serial in individual WPIC
runners. Profile before assuming linear thread scaling.

## Choosing ranks and threads

For Eulerian EV/VP, begin with several `L` cells per rank and enough threads
to occupy the remaining cores. A reasonable first experiment is 4--8 ranks
per node and 4--16 threads per rank. Increase ranks only while communication
and the shrinking local `L` slab remain efficient.

For grid BP collisions, SphDSMC, and CPU WPIC:

- request `--ntasks=1`.
- set `--cpus-per-task` to the desired thread count.
- export `OMP_NUM_THREADS` to the same value for an OpenMP binary. And
- measure memory use before increasing marker number or thread-private BP
  collision buffers.

Avoid oversubscription:

```text
ntasks_per_node * cpus_per_task <= physical cores per node
```

## Portable Slurm templates

`scripts/slurm/` contains IAS and IIT Kanpur templates:

- `ias_hybrid_ev_vp.sbatch`
- `ias_openmp_bp.sbatch`
- `iitk_param_hybrid_ev_vp.sbatch`
- `iitk_param_openmp_bp.sbatch`

The templates accept these overrides:

- `VENUS_DIR`: CPU source directory.
- `VENUS_RUN_ROOT`: parent of generated output directories.
- `CONFIG`: INI parameter file.
- `OUTPUT_DIR` and `RESTART_DIR`: explicit run locations.
- `EXECUTABLE`: exact task-specific binary.
- `VENUS_BUILD=0`: use an existing executable.
- `VENUS_MAKE_JOBS`: build parallelism.
- `VENUS_SKIP_MODULES=1`: retain the loaded toolchain. And
- `VENUS_PURGE_MODULES=0`: do not purge preloaded modules.

Relative configuration/output paths are resolved from the Slurm submission
directory. A relative executable is resolved from `VENUS_DIR`. Set output
to a project or scratch filesystem rather than the source tree.

### Hybrid EV example

```bash
sbatch --nodes=1 --ntasks-per-node=4 --cpus-per-task=8 --export=ALL,EXECUTABLE=venus_cpu_einstein_vlasov_eulerian_mpi,CONFIG=examples/einstein_vlasov_low_compactness_virialized_halo.ini,VENUS_RUN_ROOT=/scratch/project/venus scripts/slurm/ias_hybrid_ev_vp.sbatch
```

For expanding VP, change the executable to
`venus_cpu_vlasov_poisson_eulerian_mpi` and the configuration to a VP INI.

### One-rank BP example

```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=32 --export=ALL,EXECUTABLE=venus_cpu_boltzmann_poisson_fokker_planck_openmp,CONFIG=examples/modular/bp_sidm_corecollapse_yukawa.ini,VENUS_RUN_ROOT=/scratch/project/venus scripts/slurm/ias_openmp_bp.sbatch
```

For exact/reversible/sampled/streamed finite-angle BP, choose the matching
task executable and configuration. For SphDSMC, choose
`venus_cpu_sidm_sphdsmc_constant_openmp` or the velocity-dependent task.

The wrappers set:

```text
OMP_NUM_THREADS = SLURM_CPUS_PER_TASK
OMP_PLACES       = cores
OMP_PROC_BIND    = spread
```

They print resolved inputs and resources before `srun`, which makes path or
allocation mistakes visible early.

## Site adaptation

Before submitting on a new cluster:

```bash
module avail
sinfo
scontrol show partition
```

Edit or override partition, account, compiler, MPI, and CUDA/module names.
The IAS templates provide placeholders for the GCC and Open MPI modules. The
IITK templates keep PARAM-specific names editable. Do not copy module names
between clusters without checking ABI compatibility.

## WPIC launch and signals

WPIC uses purpose-specific command-line options rather than the INI
environment used by Eulerian tasks. Launch one task and pass the allocated
thread count with `--threads` when that option is supported.

A restart-safe Slurm signal wrapper must:

1. launch the runner in the background.
2. trap `USR1` or `TERM`.
3. forward the signal to the child process.
4. wait for the child to finish its atomic checkpoint. And
5. requeue only the documented administrative checkpoint status.

The expanding WPIC runner distinguishes important terminal states:

- status 97: collision event ordering/cadence cannot be represented safely.
- status 98: the configured shell-width/mean-free-path locality limit is
  crossed. And
- status 99: an administrative walltime checkpoint was written.

Statuses 97 and 98 require diagnosis of the recorded temporal or spatial
limit before a continuation can be selected. A tighter collision cadence may also make a saved collision age
inadmissible. Resume from an earlier valid event boundary or begin a fresh
short run.

## Diagnostics for scaling and correctness

Record the executable, configuration, grid/marker counts, ranks, threads,
compiler flags, hardware, timestep/collision cadence, output frequency, and
separate compute/I/O timings. Compare runs with the same physical and
discretization parameters. Inspect:

- global mass and energy budgets.
- EV metric/compactness constraints.
- phase space boundary occupancy.
- BP positivity repair and collision substeps.
- collision acceptance/majorant diagnostics.
- SphDSMC or WPIC shell-width/mean-free-path ratios. And
- restart continuity at a common physical time.

A speed comparison is meaningful only when the two runs apply the same
physical operators and statistically equivalent collision work.

## IAS Typhon ingoing Eddington--Finkelstein (EF) production gate

The horizon-penetrating WPIC calculation has a sequence of prerequisites. Use
`scripts/slurm/submit_einstein_vlasov_wpic_parity.sh` first. Its exact
fine-A Schwarzschild donor must finish the solved-eikonal export and
`null_map_preflight.json` authentication before any auxiliary convergence
seed/donor starts. Only the strict two-resolution/two-realization parity
certificate can unlock
`submit_einstein_vlasov_wpic_angular_scan.sh`.

This gate is for the bound
`--high-initial-condition energy_cutoff_shell` family. Before either launcher
is used, local initialize-only and short dynamic tests must fix and validate:

- the static continuum `(C, ell0, y_c)` and its positive binding.
- one inward kick `(kappa,r_c)` held identical at every angular scale.
- one target initial Arnowitt--Deser--Misner (ADM) mass and an even marker count.
- a low-`L` EF endpoint with an apparent horizon and resolved post-horizon
  continuation.
- a high-`L` endpoint with positive realized binding, small escape, and
  sustained Jeans/radius/profile evidence for a halo.
- an exact-face map worldtube `R_B`, full evolution boundary `R_E`, donor
  duration, timestep, end time, cadence, and initial angular bracket. And
- successful TSC-support, vacuum-source, `b=1`, four-cell donor/crossing
  clearance, unique crossing, eikonal, analytic-extension, and artifact-hash
  preflight at the largest planned marker realization.

For `R_B<R_E`, the mapper solves the eikonal only on the authenticated grid
prefix through `R_B` and extends the vacuum part of the initial null slice
exactly with `m=M_null`, `a=1-2M_null/r`, `A=F=b=1`, and zero stress. The EF
run then evolves the entire `R_E` grid. The binary/manifest record both radii,
the analytic extension index, prefix/full-grid fingerprints, clearances, and
extension residuals.

Specify the continuum, perturbation, mass, end time, angular interval, grid,
and marker count for the physical family being evolved. Dense-donor storage
and memory scale with the measured accepted-step count, radial cells, saved
field state, and marker-crossing ledger. Measure a representative local or
short cluster run, include a safety margin, and size Typhon memory, walltime,
and scratch from that result.

Three mass fields are distinct in every mapped run:

- `fixed_family_adm_mass`: the Schwarzschild `tau=0` family normalization,
  compared to the explicitly chosen target.
- `null_slice_conserved_adm_mass`: the evolved donor/null-slice boundary
  anchor, including numerical donor drift. And
- `initial_deposited_adm_mass`: the EF constraint mass reconstructed from the
  mapped markers at shifted runtime time `v_hat=0`.

The analytic vacuum extension is anchored to the second mass. The latter two
measure numerical changes in the donor evolution and EF deposition. The
first mass retains the common physical normalization of the family. The
launcher authenticates both against the map and bounds donor drift and EF
deposition mismatch. The threshold classifier independently verifies the
SHA-protected postflight before accepting a point. Positive binding must be
recomputed after angular scaling, the fixed kick, and fixed-ADM normalization
at every scan value. A halo additionally requires the coordinate-invariant
family binding fraction
`(M0-fixed_family_adm_mass)/M0` to be positive and the late normalized GR
anisotropic balance residual
`dp_r/dr+(rho+p_r)*mu_r-(p_t,sum-2*p_r)/r` to remain small. Missing balance
data fail closed. The saved `2K/abs(W)` is a Newtonian report-only heuristic
and is not a relativistic halo gate. Require stationary `r50`, `r90`, maximum
compactness, and time-averaged profiles over multiple dynamical times, plus a
larger-domain and longer-time restart, before claiming sustained
virialization. A high-`L` escape is dispersal, not the halo side of the
requested threshold.

The parity minimum is exactly two jointly refined radial/time resolutions by
two independent deterministic quiet-PIC realizations. The convergence builder
must authenticate all four single-pair diagnostics before the strict fine
certificate can pass. Once it does, the scan may coarse-bracket a low-`L`
black hole and high-`L` halo and then bisect. Near-critical, nonmonotone, or
insufficient-duration points remain `undetermined`. Report a converged
dimensionless `ell/M_ADM,target` interval, not an unsupported decimal.
