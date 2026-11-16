# VENUS 1.0 CPU modules

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur

This directory contains the C++17 CPU implementation of VENUS: Eulerian
Einstein--Vlasov and Vlasov--Poisson, grid Boltzmann--Poisson, weighted
particle-in-cell (WPIC), spherical direct-simulation Monte Carlo (SphDSMC),
and supporting Python reference, initial-condition, analysis, plotting, and
GIF tools.

## One task, one executable

Choose the executable for the physical equations and collision model you
want to solve. It calls the shared numerical routines and checks that the
supplied controls belong to that model before creating output.

| Physical task | Preferred executable |
|---|---|
| Eulerian Einstein--Vlasov | `venus_cpu_einstein_vlasov_eulerian` |
| Eulerian Vlasov--Poisson | `venus_cpu_vlasov_poisson_eulerian` |
| BP Fokker--Planck | `venus_cpu_boltzmann_poisson_fokker_planck` |
| BP Landau form | `venus_cpu_boltzmann_poisson_landau` |
| Broad finite-angle compatibility task | `venus_cpu_boltzmann_poisson_finite_angle` |
| Exact finite-angle BP | `venus_cpu_boltzmann_poisson_exact` |
| Reversible finite-angle BP | `venus_cpu_boltzmann_poisson_reversible` |
| Sampled reversible BP | `venus_cpu_boltzmann_poisson_reversible_sampled` |
| Streamed reversible BP | `venus_cpu_boltzmann_poisson_reversible_streamed` |
| Expanding prompt-cusp BP | `venus_cpu_boltzmann_poisson_expanding_prompt_cusp` |
| Canonical orbit-phase BP | `venus_cpu_boltzmann_poisson_canonical_orbit` |
| Constant-cross-section SphDSMC SIDM | `venus_cpu_sidm_sphdsmc_constant` |
| Velocity-dependent SphDSMC SIDM | `venus_cpu_sidm_sphdsmc_velocity_dependent` |
| Low-compactness EV WPIC | `venus_cpu_einstein_vlasov_wpic_low_compactness` |
| High-compactness EV WPIC collapse, Schwarzschild or ingoing EF | `venus_cpu_einstein_vlasov_wpic_black_hole` |
| Expanding VP WPIC | `venus_cpu_vlasov_poisson_wpic_expanding` |
| Subhalo scatter/capture WPIC | `venus_cpu_subhalo_wpic` |
| Constant-cross-section material-SIDM WPIC | `venus_cpu_sidm_wpic_constant` |
| Velocity-dependent material-SIDM WPIC | `venus_cpu_sidm_wpic_velocity_dependent` |

The unsuffixed grid and SphDSMC binaries use the portable
`std::thread` backend. Every one also has an `_openmp` build. Eulerian EV
and VP additionally have `_mpi` executables that combine MPI angular-momentum
slabs with OpenMP. Collisional BP and SphDSMC intentionally have no MPI
executable. WPIC task runners are shared-memory programs.

Newtonian collisionless WPIC runners use a fixed kick--drift--kick orbit step from
`--maximum-timestep`, apart from exact output/final-time landings. The
subhalo runner enables independent radial-cell collision clocks with
`--subhalo-adaptive-cell-timestep`. The constant-cross-section material-SIDM
runner enables independent fixed-shell clocks with
`--collision-adaptive-shell-timestep`; that CPU path requires equal marker
masses, `--collision-cfl <= 1`, and a constant cross section. Its outer
collision-batch ceiling is set by
`--collision-adaptive-maximum-interval`. Omitting either adaptive flag selects
the mesh-wide reference clock.

`venus_einstein_vlasov*`, `venus_modular_solver*`, and `run_*` provide generic
dispatchers or specialist interfaces. The task names above select the
corresponding physical model explicitly.

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

## Build

```bash
cd VENUS/cpu
make organized-solvers           # all portable task front ends
make cpu-task-solvers-openmp     # grid/SphDSMC OpenMP variants
make cpu-task-solvers-mpi        # Eulerian EV/VP MPI+OpenMP only
make cpu-task-solvers-all        # all supported grid/SphDSMC variants
make tools                       # compile maintained C++ tools
```

On macOS with Apple Clang, the portable task build is the safest starting
point because Apple Clang normally lacks an OpenMP runtime. On a cluster, use
one consistent compiler/MPI stack.

## Run examples

Eulerian low-compactness Einstein--Vlasov:

```bash
./venus_cpu_einstein_vlasov_eulerian --config examples/einstein_vlasov_low_compactness_virialized_halo.ini
```

Expanding Eulerian Vlasov--Poisson:

```bash
./venus_cpu_vlasov_poisson_eulerian --config examples/modular/vp_expanding_nfw_gamma2_collisionless.ini
```

Constant-cross-section SphDSMC:

```bash
./venus_cpu_sidm_sphdsmc_constant --config examples/modular/spherical_dsmc_frank_equilibrium.ini
```

Canonical orbit-phase Boltzmann--Poisson uses its task-specific command-line
interface:

```bash
./venus_cpu_boltzmann_poisson_canonical_orbit --help
```

Its orbit atlas is a mesh whose rows follow bound trajectories and whose
phase cells divide motion along each trajectory. Command-line controls select
this mesh, transport, encounters and their separate clocks. The manual lists
the full interface, including restart and convergence controls.

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

## Inputs, outputs, and restarts

Grid initial data may be analytic or use the documented `EVICRWL1` binary
format. WPIC and SphDSMC have distinct marker and restart state. Common grid
products include `metadata.json`, `history.csv`, `profiles.csv`,
phase-space snapshots, and restart files. Marker solvers add time-indexed
profiles, collision ledgers, mesh contracts, locality diagnostics, and
versioned binary restarts. Resume from the complete restart state, never by
reconstructing a state from CSV profiles.

## Parallel execution

Eulerian EV and collisionless VP may use MPI: each rank owns a contiguous
`L` slab, local moments are reduced to radial fields, and OpenMP or
`std::thread` handles rank-local work. Do not use more ranks than `L`
cells.

Grid collision operators, SphDSMC and WPIC use one MPI task with multiple
threads. That task owns the collision state and its random sequence. The
launcher therefore requires a single rank for these CPU calculations.
See [`docs/cluster.md`](docs/cluster.md) and the portable templates in
[`scripts/slurm/`](scripts/slurm/).

## Plotting and GIFs

Activate an environment with NumPy, Matplotlib, SciPy, Pillow, and ImageIO,
then inspect the matching tool:

```bash
python3 tools/plot_einstein_vlasov_outputs.py --help
python3 tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py --help
python3 tools/render_frank_venus_sphdsmc_gif.py --help
python3 tools/render_weighted_pic_subhalo_comparison_gif.py --help
python3 tools/render_weighted_pic_subhalo_strength_scan_gif.py --help
```

Plotters validate input schemas, use physical time rather than filename order,
and write derived products outside the solver state. The full tool map is
[`../notes/SCRIPT_REFERENCE.md`](../notes/SCRIPT_REFERENCE.md).

For paired high-compactness runs, render the standard three-panel SC/EF
overlay only from the authenticated null map:

```bash
python3 tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py \
  --schwarzschild RUN/sc --ef RUN/ef \
  --manifest RUN/common_ingress_null.bin.manifest.json \
  --eikonal-substeps "$VENUS_EIKONAL_SUBSTEPS" \
  --output RUN/sc_ef_black_hole_profiles.gif
```

The blue SC and red dashed EF curves are evaluated at the same mapped events
before the SC chart ends. The coordinate map determines which EF time
corresponds to each radius on the SC slice.
Later frames are EF-only when that run continues beyond the SC chart limit.
The panels show `log10(rho)` against `log10(r)`, logarithmic density slope
against `log10(r)`, and `2m/r` against linear `r`. Inside a trapped region
there is no static observer, so the EF energy-density curve is masked there;
compactness remains visible across the full saved EF slice. The GIF is
accompanied by PNG previews and exact frame/provenance sidecars.

## Documentation

- [`../notes/SOLVER_MODULES.md`](../notes/SOLVER_MODULES.md): exhaustive
  one-by-one module and executable catalogue.
- [`../docs/venus_manual.pdf`](../docs/venus_manual.pdf): complete
  physics, algorithms, parameters, I/O, ICs, compilation, execution,
  parallelism, plotting, and limitations.
- [`docs/examples.md`](docs/examples.md): supplied inputs and plotting paths.
- [`docs/cluster.md`](docs/cluster.md): OpenMP, MPI, Slurm, and cluster
  resource mapping.
