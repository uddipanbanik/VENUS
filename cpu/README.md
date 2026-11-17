# VENUS CPU guide

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur

Start with the physical equations, choose their numerical representation,
then build, supply initial data, evolve the system, and analyze the saved
state. This guide follows that sequence. Commands below run from the
**repository root**, the directory containing the top-level README.

For a first run, the [examples index](../examples/README.md) links every
build script, input, local run script, and Slurm launcher. It also explains
restarts, analysis, and the technical terms used below.

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
The [cluster guide](docs/cluster.md) gives scheduler templates.

## 2. Choose the equations, background, and representation

VP means Vlasov--Poisson, BP means Boltzmann--Poisson, and EV means
Einstein--Vlasov. Collisionless evolution transports the distribution in its
self-consistent gravitational field. Collisional evolution adds the selected
encounter law. **Phase space** means position and velocity considered together.
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

Start with the supplied analytic Einstein--Vlasov shell:

```bash
cpu/venus_cpu_einstein_vlasov_eulerian \
  --config examples/inputs/ev_grid.ini \
  --t-end 0.002 --dt-max 0.001 \
  --output-dir output/cpu/first_ev
```

This short run demonstrates the interface. An INI is a plain-text parameter
file: open it in a text editor to read the initial density and velocity
choices, boundaries, numerical resolution, and output cadence. To make a
separate Newtonian example, copy `examples/inputs/vp_grid_static.ini`, edit
its parameters, then pass its filename to `--config`. The input binary, when
selected, supplies the actual distribution described by that INI.

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
  --input-dir output/cpu/first_ev \
  --output-dir output/cpu/first_ev/figures --make-gifs
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
| Portable grid or `_openmp` grid | Radial/phase-space loops use threads | One process owns the distribution and outputs |
| Grid BP and SphDSMC | Supported loops use shared-memory threads | One MPI rank owns encounter state and random sequence |
| CPU WPIC | Supported deposit/orbit/collision loops use `--threads` | One MPI rank owns particles, clocks, metric/field and restart |

For an MPI grid run, build `make cpu-mpi`, choose the `_mpi` executable, and
launch through the site scheduler with more than one rank only for
collisionless Eulerian EV/VP. Use at most the number of angular-momentum
cells. Set `OMP_NUM_THREADS` to the allocated CPUs per rank. Radial grids
remain replicated across MPI ranks, so this decomposition scales with the
angular-momentum resolution. CPU WPIC, SphDSMC and collisional BP use one MPI
rank with the allocated shared-memory threads.

## 7. Relativistic collapse and coordinate comparisons

The following controls are intended for detailed relativistic studies after a
first run. Compactness is `2Gm/(rc^2)`, where `m` is enclosed gravitational
mass, `r` is areal radius, `G` is Newton's constant, and `c` is the speed of light.
These runs use units with `G=c=1`. The ADM mass is the total gravitational mass
measured from the isolated system's distant exterior. A spacelike slice is a
surface of constant simulation time; the ingoing EF calculation instead uses
an incoming-light, or null, slice. Comparing coordinates requires matching
physical events on those different surfaces.

The high-compactness runner selects one of two independent weighted
Einstein--Vlasov kernels with `--coordinates`:

- `schwarzschild` uses
  `include/ev/weighted_einstein_vlasov_pic_schwarzschild.hpp` and retains the
  original polar--areal compactness-gap stop;
- `eddington_finkelstein` uses
  `include/ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp`, advances
  `log(p_r)` so the required future-directed branch `p_r > 0` is maintained at
  every Runge--Kutta stage, solves the advanced-null constraints, detects
  `a = 0`, and continues with optional trapped-region excision.

Both paths start from the same physical particle realization supplied through
`--physical-ic`. Schwarzschild data lie on a spacelike slice, while EF data
lie on an ingoing null slice. The null extractor follows the Schwarzschild
worldlines to their intersections with that null slice and transforms each
momentum at its intersection. The comparison tools then evaluate tensors
and particle distributions at the same physical events. The stored angular
momentum is each particle's magnitude in spherical symmetry.

High-compactness Einstein--Vlasov coordinate selection is explicit. A minimal
fresh solved-map sequence is:

```bash
./venus_cpu_einstein_vlasov_wpic_black_hole \
  --scenario high --coordinates schwarzschild --initialize-only \
  --physical-ic common_ic.bin \
  --high-r-max 20 --high-radial-cells 2048 \
  --output-dir output/ic

./venus_cpu_einstein_vlasov_wpic_black_hole \
  --scenario high --coordinates schwarzschild \
  --physical-ic common_ic.bin \
  --export-ef-null-initial-data ef_null.bin \
  --ef-null-donor-duration 24 \
  --high-r-max 20 --high-radial-cells 2048 \
  --high-specific-angular-momentum 1.0 \
  --high-target-initial-adm-mass 0.58 \
  --output-dir output/sc --expected-outcome any

./venus_cpu_einstein_vlasov_wpic_black_hole \
  --scenario high --coordinates eddington_finkelstein \
  --physical-ic common_ic.bin \
  --ef-null-initial-data ef_null.bin \
  --high-r-max 20 --high-radial-cells 2048 \
  --high-specific-angular-momentum 1.0 \
  --high-target-initial-adm-mass 0.58 \
  --high-inner-boundary origin_parity --high-radial-grid linear \
  --output-dir output/ef --expected-outcome any
```

The SC calculation saves its resolved history, the worldline intersections
and the resulting null slice. The EF importer checks their file identities
and physical controls before loading that slice. Each mapped summary distinguishes the
fixed-family Schwarzschild `tau=0` ADM mass, the numerically evolved
null-slice conserved ADM anchor, and the independently deposited EF initial
ADM mass. The latter two measure numerical changes during the coordinate extraction
and EF deposition. Compare them to the common initial target, 0.58 in this
example, under spatial and temporal refinement.
The normalized Schwarzschild `tau=0` family additionally records
`fixed_family_rest_weight = M0`, `fixed_family_binding_energy = M0-M_ADM`,
and `fixed_family_binding_fraction = (M0-M_ADM)/M0`. These values are shared
unchanged by the SC and mapped-EF summaries and survive restart through
`fixed_family_initial_bound_diagnostics.csv`. A halo classification requires
a positive, internally consistent global binding fraction, but this condition
is necessary rather than sufficient: a bound family may still breathe,
partially escape, or remain undetermined. The reported
`E_proxy=exp(mu)*sqrt(1+w^2+2*eta/r^2)` tail is evaluated only in the frozen
Schwarzschild `tau=0` metric and is not conserved in a dynamical spacetime; it
is never used as a hard per-particle halo cut. The `2K/abs(W)` column supplies a Newtonian virial diagnostic. Relativistic
stationary balance is assessed through the stress equation below.
The high-history file additionally records the normalized GR anisotropic
static-balance residual
`dp_r/dr + (rho+p_r)*mu_r - (p_t,sum-2*p_r)/r`. SC evaluates it directly;
pre-horizon EF outputs reconstruct the same polar-areal moments wherever
`a>0`. The threshold analyzer requires sustained small late residuals plus
stationary `r50`, `r90`, and maximum compactness. A missing residual makes a
halo label inconclusive, whereas `2K/abs(W)` remains report-only.
The mapped comparison requires every marker to cross the chosen null slice
once, an isolated vacuum exterior, and resolved eikonal component errors.
Choose the radial worldtube and resolution together with the duration of
the donor calculation. The coordinate-map tools and their controls are
derived in the manual.

The [figure reproduction guide](../analysis/README.md) identifies the example
calculations and their data. Input-format and custom-distribution examples
are under `examples/modular` and described in [docs/examples.md](docs/examples.md).
WPIC uses task-specific command-line options; Eulerian and SphDSMC tasks
use INI files with command-line overrides. Each executable's `--help`
interface lists its accepted controls.


## Further references

See the [solver catalogue](../notes/SOLVER_MODULES.md), [script reference](../notes/SCRIPT_REFERENCE.md), and [full manual](../docs/venus_manual.pdf) for equations, specialist controls, and reproducible examples.
