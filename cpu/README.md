# VENUS CPU guide

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur

Build the programs, choose a physical problem and its initial distribution,
evolve that distribution, then read and plot the saved results. This guide
follows those steps, beginning with a small isolated Newtonian example.
Commands below run from the
**repository root**, the directory containing the top-level README.

For a first run, the [examples index](../examples/README.md) links every
build script, input, local run script, and Slurm launcher. It also explains
restarts, analysis, and the technical terms used below.

The [cluster guide](../docs/CLUSTERS.md) supplies IAS Typhon, IAS Apollo and
Perimeter Symmetry recipes, module choices and portable scheduler settings.

## 1. Build the solvers

A build translates the C++ source into executable programs. Install a C++17
compiler and Python 3.8 or newer. The analysis tools use
NumPy, Matplotlib, SciPy, Pillow, and ImageIO:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash scripts/build/cpu.sh
bash scripts/build/tools.sh
```

`scripts/build/cpu.sh` builds the portable task executables inside `cpu/`. Start with
these executables on a laptop. For a particular task, build just its target,
for example `make -C cpu venus_cpu_vlasov_poisson_wpic_static`.

```bash
bash scripts/build/cpu_openmp.sh  # shared-memory grid and SphDSMC variants
bash scripts/build/cpu_mpi.sh     # collisionless Eulerian EV/VP with MPI and OpenMP
bash scripts/build/tools.sh     # maintained C++ utilities
make -C cpu check  # finite C++ numerical tests
```

OpenMP lets one process use several CPU cores. MPI lets separate processes
exchange data, including across computers. Their optional builds need the
corresponding libraries and compatible compilers. Weighted particle-in-cell
(WPIC) runners use `--threads` to select shared-memory workers.
The [cluster guide](../docs/CLUSTERS.md) gives tested site recipes;
the [parallel implementation guide](docs/cluster.md) explains the decomposition.

## 2. Choose the equations, background, and representation

Begin with Newtonian gravity when particle speeds are small compared with
the speed of light and relativistic effects are weak. Choose relativistic
evolution to follow strong gravitational fields and black hole formation.
Within either setting, choose collisionless evolution to study orbits and
mixing in the collective field. Add a collision model when encounters between
subhalos, dark matter particles, or stars are part of the physical question.

VP means Vlasov--Poisson: the distribution evolves under the gravitational
field that it creates. BP means Boltzmann--Poisson: an encounter term also
redistributes the matter. EV means Einstein--Vlasov: matter and the spacetime
metric evolve together under general relativity. **Phase space** means
position and velocity considered together.
An Eulerian grid stores the distribution in fixed phase-space cells. WPIC uses
weighted particles, also called markers: each carries a specified amount of
mass and moves continuously through the gravitational field. Spherical direct
simulation Monte Carlo (SphDSMC) follows persistent particles and samples their
local encounters. SIDM means self-interacting dark matter. CLI means
command-line input, the options typed after an executable name.

A subhalo is a gravitationally bound clump inside a larger halo. Its scattering
and capture model describes encounters between these clumps. Material SIDM
models instead describe scattering between the constituent dark matter particles.
Schwarzschild (SC) and ingoing Eddington--Finkelstein (EF) are coordinate choices
for relativistic evolution; EF remains regular at a black hole horizon.

| Physical category | Background | Executable in `cpu/` | Input |
|---|---|---|---|
| Non-relativistic collisionless, grid | Static or expanding | `venus_cpu_vlasov_poisson_eulerian` | INI |
| Non-relativistic collisionless, WPIC | Static | `venus_cpu_vlasov_poisson_wpic_static` | CLI or native WPIC state |
| Non-relativistic collisionless, WPIC | Expanding | `venus_cpu_vlasov_poisson_wpic_expanding` | CLI or native WPIC state |
| Non-relativistic collisional, subhalos | Static or expanding | `venus_cpu_subhalo_wpic` | CLI, positive subhalo strength |
| Non-relativistic collisional, small-angle transport | Static or expanding | `venus_cpu_boltzmann_poisson_fokker_planck`, `venus_cpu_boltzmann_poisson_landau` | INI with the chosen collision law |
| Non-relativistic collisional, finite-angle grid | Static or expanding as allowed by the task | `venus_cpu_boltzmann_poisson_exact`, `_reversible`, `_reversible_sampled`, `_reversible_streamed` | INI with corresponding operator |
| Non-relativistic collisional, expanding prompt cusp | Expanding | `venus_cpu_boltzmann_poisson_expanding_prompt_cusp` | INI |
| Non-relativistic collisional, orbit-phase transport | Declared by task controls | `venus_cpu_boltzmann_poisson_canonical_orbit` | CLI |
| Non-relativistic collisional, SIDM WPIC | Static | `venus_cpu_sidm_wpic_constant`, `venus_cpu_sidm_wpic_velocity_dependent` | CLI with material collision controls |
| Non-relativistic collisional, SIDM SphDSMC | Static | `venus_cpu_sidm_sphdsmc_constant`, `venus_cpu_sidm_sphdsmc_velocity_dependent` | INI |
| Non-relativistic collisional, star clusters | Chosen physical model | Small-angle/encounter machinery above; stellar encounter law and application validation required | Model-dependent |
| Relativistic collisionless, grid | Isolated, evolving spacetime | `venus_cpu_einstein_vlasov_eulerian` | INI |
| Relativistic collisionless, low compactness WPIC | Isolated, evolving spacetime | `venus_cpu_einstein_vlasov_wpic_low_compactness` | CLI |
| Relativistic collisionless, compact WPIC | Isolated, evolving spacetime | `venus_cpu_einstein_vlasov_wpic_black_hole` | CLI, SC or ingoing EF |
| Relativistic collisional, SIDM | Isolated | `include/ev/relativistic_boltzmann/` coupled library | C++ API, component-tested scope |
| Relativistic collisional, subhalos or star clusters | Isolated or expanding | Dedicated encounter models and application runners are future work | — |

The Fokker--Planck and Landau methods describe the accumulated effect of many
small deflections. Finite-angle methods retain individual deflection angles.
“Exact” in a task name identifies direct numerical integration of that collision
operator; the distribution, domain, and integration still have numerical errors.
The reversible variants preserve the symmetry between a collision and its
reverse. Their sampled and streamed forms estimate or process the collision
integrals in batches. The canonical orbit-phase method represents matter along
bound gravitational orbits rather than ordinary position/velocity cells.

The [complete catalogue](../notes/SOLVER_MODULES.md) explains each algorithm
and its physical scope. The finite-angle compatibility selector
`venus_cpu_boltzmann_poisson_finite_angle` accepts several grid backends.
Choose the named operator for a new calculation. Adaptive-grid work remains
under development and will be distributed separately.

A static background sets cosmological expansion to zero. Both the Poisson
field and the relativistic metric still respond to the evolving matter.
Isolated Einstein--Vlasov WPIC is already supplied by the two compactness
runners above. Their compactness labels select initial-data families.

## 3. Supply an INI file and run a grid calculation

Start with an isolated shell in Newtonian gravity:

```bash
cpu/venus_cpu_vlasov_poisson_eulerian \
  --config examples/inputs/vp_grid_static.ini \
  --t-end 0.002 --dt-max 0.001 \
  --output-dir output/cpu/first_vp
```

This command reads the supplied initial distribution, reconstructs its gravity,
and advances it to time `0.002` in the run's units. Each step is at most
`0.001`. The coarse grid and short duration make this an installation example.
The output is in `output/cpu/first_vp/`.

An INI is a plain-text parameter file. Open
`examples/inputs/vp_grid_static.ini` in a text editor. Its section headings
group the physics, grid, initial distribution, boundaries, time controls and
output. This example constructs the distribution from an analytic formula;
it needs no separate data file. Its main initial-data controls are:

| Parameter | Meaning in this example |
|---|---|
| `R0 = 3`, `dR = 1` | Center and width of the initial Gaussian density shell |
| `w_in = -0.1` | Center of the radial-velocity distribution; a negative value describes infall |
| `beta = 2` | Controls the velocity spread in the chosen analytic distribution; increasing it narrows the spread |
| `amp = 0.0001` | Sets the distribution's normalization and therefore the represented matter density |
| `Nr = 12`, `Nw = 12`, `NL = 8` | Numbers of grid points in radius, radial velocity and specific angular momentum |
| `r_min`, `r_max`, `w_max`, `L_max` | Limits of the represented positions and velocities/angular momenta |

Specific angular momentum is angular momentum per unit mass. It distinguishes
radial from more tangential orbits. Here the radial grid extends from `0.1`
to `10`; reflecting radial boundaries return matter into the domain when it
reaches an edge. The outer boundary therefore matters if you extend the run.
The velocity boundary lets matter leave the represented velocity range.
Choose the domain and boundaries for the system you intend to simulate.

To keep a separate input and run together:

```bash
mkdir -p output/my_halo
cp examples/inputs/vp_grid_static.ini output/my_halo/input.ini
cpu/venus_cpu_vlasov_poisson_eulerian \
  --config output/my_halo/input.ini --output-dir output/my_halo/run
```

You can now edit `output/my_halo/input.ini` for a new run, using a fresh output
directory each time. Change the physical inputs to pose your problem, then
increase resolution and reduce the timestep to assess numerical accuracy.
For a relativistic grid example, use `examples/run/ev_grid.sh`; the script
selects Einstein--Vlasov and its corresponding INI. A binary initial-data file,
when selected instead of the analytic formula, stores the actual distribution.

Read the INI before extending it
to the physical duration and resolution needed for your problem. The INI
separates these choices:

| Input controls | What they set |
|---|---|
| `physics_model`, `vp_universe`, `vp_poisson_source` | Equations, static/expanding background, total density or density contrast |
| `initial_data`, `initial_data_file` | Analytic distribution or the file containing the initial distribution |
| `Nr`, `Nw`, `NL`, spatial and momentum bounds | Phase-space resolution and domain |
| `inner_boundary`, `outer_boundary`, `momentum_boundary` | Physical/numerical boundary conditions |
| `t_end`, `dt_max`, `cfl` | End time and timestep constraints |
| `bp_collision_operator`, cross-section and strength controls | Encounter approximation and interaction law for BP |
| `output_dir`, `save_every`, `phase_every`, `restart_every` | Destination and accepted-step output/checkpoint cadence |
| `threads` | Shared-memory workers |

Options following `--config` override the INI. Use hyphenated names such as
`--t-end` for command-line controls and underscore names such as `t_end` in INI files. **Relative paths written in an INI are resolved
from that INI's directory. Relative command-line paths use the directory
from which you launch the executable.**

For static Newtonian grid evolution, use
`cpu/examples/modular/vp_bound_infalling_gaussian_shell_train.ini` with
`venus_cpu_vlasov_poisson_eulerian`. For an expanding control, use
`cpu/examples/modular/vp_expanding_nfw_gamma2_collisionless.ini` with that
same executable. For material SIDM, begin with
`cpu/examples/modular/spherical_dsmc_frank_equilibrium.ini` and
`venus_cpu_sidm_sphdsmc_constant`. The [input guide](docs/examples.md)
explains the supplied binary data and each example.

A custom grid distribution uses the versioned `EVICRWL1` binary format,
which stores the coordinate arrays and distribution. Follow the
[custom-distribution tutorial](../README.md#use-your-own-initial-distribution)
to generate this file and connect it to `vp_custom_shell.ini`.

## 4. Run isolated or expanding particle calculations

WPIC uses hyphenated command-line options. It builds a fresh particle
population from the chosen sampler or imports a complete compatible native
state. A CSV density profile contains averages of the distribution; the
native binary retains its particles and restart state.

Start a small, isolated Vlasov--Poisson cusp calculation:

```bash
cpu/venus_cpu_vlasov_poisson_wpic_static \
  --gamma 1.5 --radial-cells 16 --radial-minimum 0.1 --radial-maximum 10 \
  --particles-per-radial-cell 128 --maximum-timestep 0.001 \
  --end-time 0.01 --output-interval 0.005 \
  --history-every 1 --restart-every 5 --threads 2 \
  --output-directory output/cpu/first_static_vp
```

`--gamma` sets the sampled cusp slope. The radial controls choose its field
mesh and domain, and markers per cell choose the sampling resolution.
`--maximum-timestep` caps each orbit step. The kick--drift--kick method
updates velocity, advances position, then completes the velocity update. Profile spacing
is measured in physical time, while history and checkpoint cadences count
accepted steps. This compact example checks operation of the solver.
Scientific evolution requires marker, mesh, timestep, and boundary studies.

For expansion, select `venus_cpu_vlasov_poisson_wpic_expanding` and add
`--vp-universe expanding --vp-alpha 0.6666666666666666 --vp-time-offset 12`.
For this background the scale factor is proportional to
`(t + time_offset)^alpha` and `H=alpha/(t+time_offset)`. The static executable
fixes `H=0`. Background controls and the mesh must match the imported state.
The [Gaussian-shell example](examples/modular/vp_bound_gaussian_shell_gpu.md)
generates explicit static and expanding WPIC initial states for CPU/GPU
comparisons.

For subhalo scattering and capture, select `venus_cpu_subhalo_wpic`, declare
`--vp-universe static` or `expanding`, and supply positive
`--subhalo-collision-strength`. The subhalo controls specify the object
velocity scale, size per unit mass, and encounter interval. The collision CFL
is a timestep restriction that keeps encounter probabilities resolved.
`--subhalo-adaptive-cell-timestep` gives collision cells independent clocks.
These clocks operate on the declared radial mesh.

For material SIDM select `venus_cpu_sidm_wpic_constant` or
`venus_cpu_sidm_wpic_velocity_dependent`. Supply a compatible material
initial state and positive `--collision-strength`, then choose
`--sidm-cross-section constant`, `powerlaw`, or `yukawa_born` as allowed by
the task. Material scattering and subhalo capture have distinct state and
parameter contracts. Run the selected executable with `--help` before
constructing a new collision model.

For isolated relativistic WPIC, begin with the low-compactness family:

```bash
cpu/venus_cpu_einstein_vlasov_wpic_low_compactness \
  --low-markers 2000 --low-radial-cells 64 \
  --low-t-end 45 --low-dt-max 0.06 --low-output-steps 20 \
  --threads 2 --output-dir output/cpu/first_ev_wpic
```

This task evolves the self-consistent relativistic metric and a Newtonian
comparison of the same weak-field family. Its static cosmological background
still permits dynamical collapse and relaxation. The high-compactness task
below follows collapse in Schwarzschild or horizon-penetrating EF coordinates.

## 5. Find, read, and analyze the output

Each command above names its destination explicitly. Grid output uses
`metadata.json`, `history.csv`, `profiles.csv`, optional phase-space files,
and complete restart files. `profiles.csv` groups radial rows by `time`.
Newtonian WPIC writes `profile_*.csv`, `history.csv`, `virial_budget.csv`,
`metadata.json`, `radial_mesh_contract.txt`, and `restart_state.bin`. Material SIDM WPIC
records its imported state and controls in `restart_import.txt`, alongside
its histories, profiles, and binary restart.
The relativistic particle examples use `low_compactness_history.csv` and
`low_compactness_profiles.csv`, or their `high_compactness_*` counterparts.
Their coordinate-specific binary restart preserves the evolving state.

CSV files have named columns and can be read with NumPy, pandas, or a
spreadsheet. Consult the header for the actual schema. Radius, density,
mean velocities and velocity dispersions, enclosed mass, and time use the
run's documented code units, the mass, length, and time scales chosen for
the equations. The gravitational coupling is set by the input; relativistic runs use
`G=c=1`. Keep the initial data, input controls, metadata, sidecars, and source
revision with every calculation.

```bash
python3 cpu/tools/plot_einstein_vlasov_outputs.py \
  --input-dir output/cpu/first_vp \
  --output-dir output/cpu/first_vp/figures --make-gifs
python3 cpu/tools/render_weighted_pic_subhalo_strength_scan_gif.py \
  --collisionless output/cpu/first_static_vp --gamma 1.5 \
  --output-directory output/cpu/first_static_vp/figures \
  --output output/cpu/first_static_vp/density_slope.gif --r-max 10
```

The grid plotter shows density, metric/compactness, mass histories, and saved
phase-space projections. The spherical renderer shows density and logarithmic
slope across saved times. Use the WPIC analysis and relativistic rendering
tools listed in the [script reference](../notes/SCRIPT_REFERENCE.md) for
particle-specific diagnostics. Compare the complete source and boundary
budget, timestep dependence, and resolution dependence before interpreting a
change in the density profile.

Resume Newtonian WPIC with `--restart RUN/restart_state.bin`, the same mesh
and background controls, and a later `--end-time`. Use a fresh output directory
for a separate branch. Grid evolution uses `--restart-from` with its native
restart. Relativistic WPIC uses `--resume` with its original run controls.
Binary restarts contain the full state; profiles supply analysis moments.

## 6. Choose MPI or shared-memory parallelism

| Calculation | Parallel work | Work shared by the calculation |
|---|---|---|
| Eulerian collisionless EV/VP with `_mpi` | MPI partitions the angular-momentum axis; OpenMP advances rank-local phase-space cells | Radial moments are globally reduced; each rank reconstructs the radial field; root coordinates shared output |
| Portable grid | Supported loops use C++ standard-library threads, selected by `--threads` | One process owns the distribution and outputs |
| `_openmp` grid | Supported radial/phase-space loops use CPU threads | One process owns the distribution and outputs |
| Grid BP and SphDSMC | Supported loops use shared-memory threads | One process owns encounter state and random sequence |
| CPU WPIC | Supported deposit/orbit/collision loops use `--threads` | One process owns particles, clocks, metric/field and restart |

For an MPI grid run, build `make cpu-mpi`, choose the `_mpi` executable, and
launch through the site scheduler with more than one rank only for
collisionless Eulerian EV/VP. Use at most the number of angular-momentum
cells. Set `OMP_NUM_THREADS` to the allocated CPUs per rank. Radial grids
remain replicated across MPI ranks, so this decomposition scales with the
angular-momentum resolution. CPU WPIC, SphDSMC and collisional BP use one MPI
rank with the allocated shared-memory threads.

## 7. Relativistic collapse and coordinate comparisons

Begin with the low-compactness example in step 4 to see how relativistic
particle output is organized. **Compactness** measures the strength of the
gravitational field: `2Gm/(rc^2)`, where `m` is enclosed gravitational mass,
`r` is areal radius, `G` is Newton's constant and `c` is the speed of light.
Areal radius means that a sphere has area `4 pi r^2`. The code uses units
with `G=c=1`.

At higher compactness, the choice of time coordinate matters. Schwarzschild
coordinates describe the exterior of a spherical collapsing system. Their
time coordinate becomes singular at the black hole horizon. Ingoing
Eddington--Finkelstein coordinates remain regular there and allow the
supported evolution to continue across it. These are two descriptions of
spacetime, each with its own numerical evolution and diagnostic checks.

First run a short Schwarzschild example:

```bash
bash examples/run/ev_wpic_schwarzschild.sh --output output/tutorial_collapse/sc
python3 scripts/analyze/run.py output/tutorial_collapse/sc --plot
```

Then run the supplied example that constructs compatible initial data for
Eddington--Finkelstein evolution:

```bash
bash examples/run/ev_wpic_eddington_finkelstein.sh \
  --output output/tutorial_collapse/ef
python3 scripts/analyze/run.py output/tutorial_collapse/ef --plot
```

The second script first follows a separate Schwarzschild donor calculation.
A donor supplies the particle trajectories needed to construct the new time
slice. The script writes that donor under `output/tutorial_collapse/ef_donor/`,
saves its physical initial distribution as `ef_common.phys`, constructs the
coordinate map in `ef_common.nullmap`, then evolves the imported EF state.
The two companion files sit beside the run directories. Keep them with the
results. These short examples demonstrate the input and coordinate interfaces;
black hole formation or a relaxed halo is a physical outcome to assess from
the chosen family and its resolved evolution.

The map is needed because equal values of the two coordinate times describe
different sets of physical events. A Schwarzschild time slice is spacelike;
an ingoing EF slice follows incoming light and is called null. The extractor
finds each particle's intersection with that null slice and transforms its
momentum there. The importer checks the saved file identities and physical
parameters before continuing. Compare the distributions at the same physical
events through this map.

For a scientific collapse study, select an initial family, total mass,
radial-velocity distribution and angular-momentum distribution, then vary the
parameter whose threshold you want to investigate. Use the high-compactness
runner's `--help` and the [module catalogue](../notes/SOLVER_MODULES.md) to
find those controls. Choose the donor duration, outer vacuum region and map
radius together so that every required particle reaches the selected slice.
Assess the map and evolution under radial, particle-count and timestep
refinement before interpreting a coordinate comparison.

The outputs distinguish total rest mass, total gravitational mass measured
from the distant exterior (ADM mass), binding energy, density, velocity
moments and maximum compactness. A bound distribution can continue to
oscillate, collapse, or partly escape. The halo classifier therefore combines
binding with sustained stationarity and a relativistic force-balance check.
A Newtonian virial ratio alone supplies a weak-field diagnostic. The
[full manual](../docs/venus_manual.pdf) derives the coordinate map, the
relativistic balance equation, the collapse stops, and the conditions for
continuing inside a trapped region.

The [figure reproduction guide](../analysis/README.md) identifies the larger
example calculations, their data and their resolved time ranges.

## Further references

See the [solver catalogue](../notes/SOLVER_MODULES.md), [script reference](../notes/SCRIPT_REFERENCE.md), and [full manual](../docs/venus_manual.pdf) for equations, specialist controls, and reproducible examples.
