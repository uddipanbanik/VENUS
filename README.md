# VENUS 1.0

[Documentation and solver manual](docs/README.md) · [Build and run](#build-run-and-analyze)

VENUS is the Vlasov--Einstein NUmerical Solver. It follows how a spherical
distribution of matter moves under its own gravity and changes through
particle encounters. The distribution determines the gravitational field,
which in turn determines the trajectories. For collisional matter, scattering
also changes the velocities, and subhalo capture changes the masses and
internal energies.

The Newtonian solvers evolve the Vlasov--Poisson or Boltzmann--Poisson
equations. The relativistic solvers evolve Einstein--Vlasov transport in
Schwarzschild or ingoing Eddington--Finkelstein coordinates. A local
relativistic collision library couples elastic scattering to those
Einstein--Vlasov updates. VENUS represents the distribution either on a
phase-space grid or by particles with statistical weights. The particle
methods include weighted particle-in-cell (WPIC) and spherical direct
simulation Monte Carlo (SphDSMC).

Co-developers: Uddipan Banik, Frank van den Bosch, and Sachiet Kapur.

VENUS is distributed under the [BSD 3-Clause License](LICENSE).

For a first calculation, follow the [CPU quick start](#cpu-quick-start) or
[GPU quick start](#gpu-quick-start). The
[custom-distribution example](#use-your-own-initial-distribution) shows the
complete path from a user function to a simulation and its plots. The
[module catalogue](notes/SOLVER_MODULES.md) identifies the equations and
executable for each physical problem. Citation metadata are supplied in
[CITATION.cff](CITATION.cff).

## Build, run, and analyze

The steps below take you from compilation to saved profiles and plots.
Open a terminal in the directory containing this README, then run the
commands below from that directory. Choose a fresh output directory for a new calculation.
The [documentation index](docs/README.md) links the full manual and the
module, input-format and script references.

### CPU quick start

Requirements: a C++17 compiler, Python 3.8 or newer, and (for analysis) NumPy,
Matplotlib, SciPy, Pillow, and ImageIO. OpenMP and MPI are optional.

```bash
python3 -m pip install -r requirements.txt
make -C cpu organized-solvers  # all portable task-specific CPU front ends
make cpu-tools           # compile every maintained C++ tool main
make cpu-check-setup     # zero-time parser/setup examples
```

The setup check writes to `output/cpu/setup_serial` and
`output/cpu/setup_modular` in the release root. Override
`SETUP_OUTPUT_ROOT` to select another location.

The independent CPU tree builds task-specific front ends over shared kernels.
For example:

```bash
cpu/venus_cpu_einstein_vlasov_eulerian \
  --config cpu/examples/einstein_vlasov_low_compactness_virialized_halo.ini
cpu/venus_cpu_vlasov_poisson_eulerian \
  --config cpu/examples/modular/vp_expanding_nfw_gamma2_collisionless.ini
```

Use `venus_cpu_boltzmann_poisson_fokker_planck`,
`venus_cpu_boltzmann_poisson_landau`,
`venus_cpu_boltzmann_poisson_exact`,
`venus_cpu_boltzmann_poisson_reversible`,
`venus_cpu_boltzmann_poisson_reversible_sampled`, or
`venus_cpu_boltzmann_poisson_reversible_streamed` for the corresponding grid
collision backend. Use `venus_cpu_boltzmann_poisson_canonical_orbit` for
the orbit-atlas transport/collision solver. The broad `venus_modular_solver*` and
`venus_cpu_boltzmann_poisson_finite_angle*` names remain compatibility
dispatchers, not the preferred new-run interface.

From the release root, `make cpu-openmp` builds the OpenMP task fronts, while
`make cpu-mpi` builds the supported MPI+OpenMP fronts. Build generic
dispatchers explicitly with `make cpu-compat`. From inside `VENUS/cpu`, use
`make cpu-task-solvers-openmp` or `make cpu-task-solvers-mpi`.
Collisional grid Boltzmann--Poisson and SphDSMC jobs are single-MPI-rank
workloads; use OpenMP threads inside that rank. The manual explains the
collisionless MPI decomposition and the fail-closed launch checks.

### Choose a module

The CPU build above provides the following entry points. Run each executable
with `--help` for its controls. The [complete module catalogue](notes/SOLVER_MODULES.md)
lists every variant and its input, output and analysis tools.

| Calculation | Executable in `cpu/` | Input and run guide |
|---|---|---|
| Einstein–Vlasov on a phase-space grid | `venus_cpu_einstein_vlasov_eulerian` | Supplied `--config` example above |
| Vlasov–Poisson on a phase-space grid | `venus_cpu_vlasov_poisson_eulerian` | Supplied example above or the user-distribution example below |
| Boltzmann–Poisson on a phase-space grid | `venus_cpu_boltzmann_poisson_exact`, `venus_cpu_boltzmann_poisson_fokker_planck`, and the variants listed above | INI file selecting that collision operator, documented in the [module catalogue](notes/SOLVER_MODULES.md) |
| Vlasov–Poisson with weighted particles | `venus_cpu_vlasov_poisson_wpic_expanding` | Analytic initialization or a particle restart, with command-line controls in the [CPU guide](cpu/README.md) |
| Subhalo scattering and capture | `venus_cpu_subhalo_wpic` | Command-line initial-halo and encounter controls in the [CPU guide](cpu/README.md) |
| Material SIDM with weighted particles | `venus_cpu_sidm_wpic_constant`, `venus_cpu_sidm_wpic_velocity_dependent` | Command-line equilibrium, cross-section and restart controls in the [module catalogue](notes/SOLVER_MODULES.md) |
| Spherical DSMC | `venus_cpu_sidm_sphdsmc_constant` | `--config cpu/examples/modular/spherical_dsmc_frank_equilibrium.ini` from the release root |
| Schwarzschild or EF Einstein–Vlasov with weighted particles | `venus_cpu_einstein_vlasov_wpic_black_hole` | `--coordinates schwarzschild` or `--coordinates eddington_finkelstein`; [initial-data and coordinate guide](#relativistic-coordinate-comparison) |
| Canonical orbit-phase Boltzmann–Poisson | `venus_cpu_boltzmann_poisson_canonical_orbit` | Equilibrium table and orbit-grid controls described in the [module catalogue](notes/SOLVER_MODULES.md) |

Relativistic elastic collisions are supplied through the C++ library calls
`eb::advance_sc` and `eb::advance_ef`. They couple to their corresponding
Einstein–Vlasov updates. The [relativistic collision guide](docs/relativistic_boltzmann.md)
describes the state, controls, call sequence and restart ownership.

### GPU quick start

The complete GPU build uses the bundled CPU subhalo calibration table. It
requires CMake 3.18+, CUDA, Python 3.8 or newer, and a compatible C++17 host compiler.
The default CUDA architecture is `sm_80` (A100); use `GPU_ARCH=90` for
H100/H200 systems.

```bash
make gpu GPU_ARCH=80       # configure and build sm_80
```

Pass `GPU_ARCH` to the final dependent target, as above.  Calling
`gpu-configure` with one architecture and then invoking `make gpu` without the
same override would reconfigure the build with the Makefile default. 

For one MPI rank per GPU:

```bash
make gpu-mpi GPU_ARCH=80
# Run inside a two-GPU allocation:
srun --ntasks=2 --gpus-per-task=1 \
  build/gpu-mpi/venus_gpu_vlasov_poisson_mpi \
  --restart restart_state.bin --output-directory output/gpu/vp_mpi \
  --end-time 60
```

The GPU task runners import CPU-generated particle initial data in weighted-PIC
formats 1--6. The static CPU SIDM format 7 additionally preserves its frozen
potential and event timescales. To initialize a GPU calculation from that
particle state, export it at a completed collision interval:

```bash
make -C cpu inspect_weighted_pic_restart
cpu/inspect_weighted_pic_restart restart_state.bin \
  --export-particles gpu_initial_particles.bin
```

The export preserves the particles and initializes a fresh field and event
schedule in the receiving solver. The manual gives the conditions for this
conversion. A subhalo run is explicit:

```bash
build/gpu/venus_gpu_subhalo_boltzmann_poisson \
  --restart restart_state.bin \
  --output-directory output/gpu/example \
  --end-time 50.1 \
  --subhalo-collision-strength 0.1 \
  --subhalo-adaptive-cell-timestep
```

Use `venus_gpu_vlasov_poisson` for a collisionless restart and
`venus_gpu_sidm_constant`, `venus_gpu_sidm_power_law`, or
`venus_gpu_sidm_yukawa_born` for a static material-SIDM run. The corresponding
`_mpi` names select the one-rank-per-GPU build. Single-GPU VP and subhalo accept
`--universe static` or `--universe eds`, with EdS as the default. Its static
restart must carry total-density masses and the corresponding static background
and radial mesh settings. MPI builds restrict VP and subhalo tasks to EdS. VP uses fixed-cap KDK
with output/final-time shortening and rejects collision options. The GPU README
gives a static example. Quantitative material-SIDM
results require shell-locality, marker-number, timestep, random-seed, and
rank-count studies. Run the selected executable with `--help`; irrelevant
collision-family options are rejected. Initial slope and marker state come
from the restart.


The [bound Gaussian shell example](cpu/examples/modular/vp_bound_gaussian_shell_gpu.md) generates the example VP/BP initial states
with `cpu/tools/generate_bound_gaussian_shell_restart.py`. It gives both complete
static and expanding commands, including their 512-cell mesh and output cadence.
The two files represent the same physical marker realization.

### Inputs and outputs

Eulerian and SphDSMC CPU tasks use INI parameter files plus command-line
overrides, with the executable locking the equation/operator. Dedicated WPIC
CPU and GPU tasks use documented command-line options. The
CPU Python package and IC tools generate analytic and file-backed initial
conditions; GPU tasks import the common versioned weighted-PIC restart
format.

Typical grid-solver output includes `metadata.json`, `profiles.csv`,
`history.csv`, phase-space snapshots, operator ledgers, and restart files.
WPIC output uses `profile_NNNNN.csv`, `history.csv` or `history_gpu.csv`,
collision histories, `radial_mesh_contract.txt`, `metadata.json`, and
`restart_state.bin`. The exact columns, restart versions, atomic-write rules,
and conditions recorded at termination are tabulated in the manual.

WPIC continuation checks existing histories and every saved profile before
writing. Their times must be at or before the selected checkpoint. Use a
separate output directory to continue a saved checkpoint while preserving
the original later outputs.

### Use your own initial distribution

The grid solvers accept a nonnegative distribution $`f(r,w,L)`$ evaluated on a
user-selected grid. The supplied
[user_distribution.py](cpu/examples/user_distribution.py) defines a compact
inward-moving shell for a static Vlasov–Poisson example. Here $`r`$ is physical
radius, $`w`$ is radial velocity, and $`L`$ is specific angular momentum. Edit its
vectorized `distribution(r,w,L)` function to choose another population.
For Einstein–Vlasov input, $`w`$ is radial momentum per unit rest mass; for an
expanding Newtonian calculation, it is peculiar radial velocity. The chosen
solver and the distribution must use the same variables and units.

From the release root, generate the binary input, run the supplied
[configuration](cpu/examples/modular/vp_custom_shell.ini), and plot its output:

```bash
python3 cpu/tools/ic_generator_legacy.py \
  --function-file cpu/examples/user_distribution.py \
  --output output/cpu/custom_shell/initial.bin \
  --Nr 64 --Nw 64 --NL 32 --r-min 0.1 --r-max 20 \
  --radial-grid log --w-max 2 --L-max 4
make -C cpu venus_cpu_vlasov_poisson_eulerian
cpu/venus_cpu_vlasov_poisson_eulerian \
  --config cpu/examples/modular/vp_custom_shell.ini
python3 cpu/tools/plot_einstein_vlasov_outputs.py \
  --input-dir output/cpu/custom_shell/run \
  --output-dir output/cpu/custom_shell/plots --no-make-gifs
```

The generator writes `EVICRWL1`: its versioned header, the three coordinate
arrays, and row-major `f[Nr,Nw,NL]`, with angular momentum varying fastest.
The solver reads the grid from that binary and integrates its moments to
obtain the initial density and gravitational field. The INI file selects
static gravity, collisionless evolution, boundary conditions, integration
controls, and output cadence. Its file paths are resolved relative to the
INI file. Command-line options override individual settings.

The output directory contains `metadata.json`, `profiles.csv`, `history.csv`,
phase-space projections, and periodic restarts under `restarts/`. Inspect
the initial density, velocity support, mass budget, and boundaries before
extending a new physical calculation. Choose a fresh output directory for a
separate run. Weighted-particle solvers use their own initialization and
restart formats; their input routes are listed in the
[module catalogue](notes/SOLVER_MODULES.md) and derived in the manual.

### Plotting and GIFs

All maintained CPU/GPU plotters, analyzers, GIF makers, IC tools, and cluster
launchers are shipped. Start with:

```bash
python3 cpu/tools/plot_einstein_vlasov_outputs.py --help
python3 cpu/tools/render_weighted_pic_subhalo_strength_scan_gif.py --help
python3 cpu/tools/render_frank_venus_sphdsmc_gif.py --help
python3 gpu/scripts/render_venus_gpu_density_slope_gif.py --help
```

The exhaustive script catalogue is
[`notes/SCRIPT_REFERENCE.md`](notes/SCRIPT_REFERENCE.md). Portable scheduler
templates are separated from specialized launch recipes; inspect all site
paths and resource requests before reuse.

## Physical models and mathematical description

The CPU tree contains the full solver family. The GPU tree implements
spherical WPIC Vlasov--Poisson transport, material SIDM collisions, calibrated
subhalo scattering and capture, and restart input and output. It also
supports MPI with one process per GPU.
Einstein--Vlasov, grid finite-volume/semi-Lagrangian methods, canonical-orbit
Boltzmann--Poisson, and SphDSMC remain CPU-only.

The Einstein--Vlasov solver evolves neutral matter in Schwarzschild and
ingoing Eddington--Finkelstein (EF) coordinates. The EF metric

```math
ds^2=-ab^2\,dv^2+2b\,dv\,dr+r^2d\Omega^2
```

remains regular at the
apparent horizon $`a=0`$ for finite positive $`b`$. Matter can cross the
horizon while its trajectories and stress continue to evolve. The
`--ef-evolve-interior` option retains all interior markers, allowing the
interior density to be studied within the spatial and temporal resolution.
The outer boundary is chosen in the surrounding vacuum. If matter reaches
that boundary, the driver saves its last accepted state and reports that the
domain has become too small. The [interior-evolution guide](cpu/docs/ef_interior_evolution.md)
explains these controls and the conditions for interpreting the density.

Spherical symmetry permits a distribution of particle angular momenta with
zero net angular momentum. A neutral spherical black hole has a
Schwarzschild vacuum exterior. The Schwarzschild time slices approach the
horizon asymptotically. The EF formulation uses ingoing null slices and
continues through the horizon. A solved coordinate map identifies the same
physical events for comparison between the two formulations.

The high-compactness solver includes a finite-support, positively bound
static Einstein--Vlasov shell

```math
f=C(1-E/E_0)_+^k\,\delta(L-L_0).
```

Here $`E`$ is the conserved particle energy in the static metric, $`E_0`$
its cutoff, $`C`$ the normalization, $`k`$ the cutoff exponent, and $`L_0`$
the fixed specific angular momentum. The subscript $`+`$ keeps the positive
part, and $`\delta`$ is the Dirac distribution. Its continuum sources, radial constraints,
asymptotic lapse match, exact vacuum regions, scale covariance, invariant WPIC
weights, and quiet shifted sampling define the equilibrium. A fixed inward
kick then produces the collapse and bound-halo calculations.

## Governing equations

The distribution supplies the density or stress that determines gravity.
The resulting field advances the particle trajectories, and local encounters
redistribute their velocities or combine subhalos. The equations below use
the conventions of the [solver manual](docs/venus_manual.pdf).

### Newtonian transport and gravity

Let $`r`$ be physical radius, $`t`$ physical time, $`a_{\mathrm H}`$ the
cosmological scale factor, and $`H=\dot a_{\mathrm H}/a_{\mathrm H}`$ the
prescribed expansion rate. A dot denotes a derivative with respect to $`t`$.
The physical velocity is $`\boldsymbol v_{\mathrm{phys}}=H\boldsymbol r+\boldsymbol v`$,
where $`\boldsymbol v`$ is peculiar velocity and $`w`$ its radial component.
Specific angular momentum has magnitude $`L`$; the stored coordinate is
$`\eta=L^2/2`$, and tangential speed is $`L/r`$. The mass distribution
$`f(t,r,w,\eta)`$ is mass per physical position–velocity volume, so

```math
\mathrm dM=8\pi^2 f\,\mathrm dr\,\mathrm dw\,\mathrm d\eta,
\qquad
\rho(r,t)=\frac{2\pi}{r^2}\int f\,\mathrm dw\,\mathrm d\eta.
```

The potential $`\Phi`$ has units of speed squared. The kinetic equation and
its collisionless characteristics are

```math
\partial_t f+(Hr+w)\partial_r f
 +\left(\frac{2\eta}{r^3}-\Phi_r-Hw\right)\partial_w f
 =C[f],
\qquad
\frac{\mathrm dr}{\mathrm dt}=Hr+w,\qquad
\frac{\mathrm dw}{\mathrm dt}=\frac{2\eta}{r^3}-\Phi_r-Hw,
\qquad \frac{\mathrm d\eta}{\mathrm dt}=0.
```

This is the Boltzmann equation, with $`C[f]`$ the collision operator. Setting
$`C=0`$ gives the Vlasov equation, and $`H=0`$ gives a static background.
Here $`\Phi_r=\partial_r\Phi`$. With Newton's constant $`G`$ and any
specified central mass $`M_{\mathrm c}`$, Poisson's radial constraint gives

```math
\frac{1}{r^2}\partial_r(r^2\Phi_r)=4\pi G\rho_{\mathrm{src}},
\qquad
M_{\mathrm{src}}(r)=M_{\mathrm c}
 +4\pi\int_0^r\rho_{\mathrm{src}}(r')r'^2\,\mathrm dr',
\qquad \Phi_r=\frac{GM_{\mathrm{src}}(r)}{r^2}.
```

Thus the current distribution determines density, enclosed source mass, and
radial gravitational acceleration $`-\Phi_r`$, in that order. An expanding
total-density model uses $`\rho_{\mathrm{src}}=\rho-\bar\rho`$, where
$`\bar\rho`$ is the homogeneous background density. An already
background-subtracted distribution supplies its source directly. An isolated
static halo uses its full density. The field solve includes unresolved inner
mass. See the [Poisson solve and force gather](cpu/include/ev/vlasov_poisson.hpp#L361)
and [Newtonian characteristic update](cpu/include/ev/weighted_vlasov_pic.hpp#L1785).

### Elastic scattering and capture

At a fixed position, let $`f_N(\boldsymbol v)`$ be the number distribution,
$`u=|\boldsymbol v_1-\boldsymbol v_2|`$ the relative speed, and
$`\mathrm d\Sigma/\mathrm d\Omega`$ the differential cross section, with
dimensions of area per solid angle. The common Hubble velocity cancels from
$`u`$. For identical particles the Boltzmann gain and loss terms are

```math
C_N[f_N]\,(\boldsymbol v_1)=
\int \mathrm d^3\boldsymbol v_2\,\mathrm d\Omega\,
u\frac{\mathrm d\Sigma}{\mathrm d\Omega}
\left[f_N(\boldsymbol v'_1)f_N(\boldsymbol v'_2)
      -f_N(\boldsymbol v_1)f_N(\boldsymbol v_2)\right].
```

Primes denote outgoing velocities; the inverse elastic encounter supplies
the gain term. For physical particle mass $`m_\chi`$, the mass distribution
is $`f=m_\chi f_N`$ and its operator uses
$`(\mathrm d\Sigma/\mathrm d\Omega)/m_\chi`$.
An equal-mass SIDM marker of sampling mass $`m_p`$ uses pair area
$`\Sigma_{ij}/m_p=\Sigma/m_\chi`$. Each accepted pair removes its incoming
velocities and installs outgoing velocities. For masses $`m_1,m_2`$,
center-of-mass velocity $`\boldsymbol V=(m_1\boldsymbol v_1+m_2\boldsymbol v_2)/(m_1+m_2)`$,
and outgoing relative direction $`\widehat{\boldsymbol n}'`$, the elastic map is

```math
\boldsymbol v'_1=\boldsymbol V+\frac{m_2}{m_1+m_2}u\widehat{\boldsymbol n}',
\qquad
\boldsymbol v'_2=\boldsymbol V-\frac{m_1}{m_1+m_2}u\widehat{\boldsymbol n}'.
```

SIDM uses the prescribed material cross section. Subhalo scattering
and capture use the calibrated Plummer pair areas. The isotropic runtime
elastic map has $`\mathrm d\Sigma_{\mathrm{sc}}/\mathrm d\Omega=\Sigma_{\mathrm{sc}}/(4\pi)`$.
For whole-object state $`z=(\boldsymbol v,m,E_{\mathrm{int}})`$, with internal
energy $`E_{\mathrm{int}}`$ and
$`\mathrm dz=\mathrm d^3\boldsymbol v\,\mathrm dm\,\mathrm dE_{\mathrm{int}}`$,
write the number distribution as $`f_N(z)`$. Capture contributes

```math
\begin{aligned}
&C_{\mathrm{cap},N}(z)=\frac12\int\mathrm dz_1\,\mathrm dz_2\,
f_N(z_1)f_N(z_2)u\Sigma_{\mathrm{cap}}
\times\left[\delta(z-z_{\mathrm{rem}})
 -\delta(z-z_1)-\delta(z-z_2)\right],\\
&m_{\mathrm{rem}}=m_1+m_2,\qquad
\boldsymbol v_{\mathrm{rem}}=\boldsymbol V,\qquad E_{\mathrm{int,rem}}=E_{\mathrm{int},1}+E_{\mathrm{int},2}
 +\frac{m_1m_2u^2}{2(m_1+m_2)}.
\end{aligned}
```

The Dirac distributions act on the full state: two incoming objects are
removed and one remnant is added. The factor one-half counts unordered
pairs once. The elastic operator extends over partner mass and internal
energy with both quantities preserved in each elastic event; the total
operator is $`C_N=C_{\mathrm{sc},N}+C_{\mathrm{cap},N}`$.
Integrating $`m f_N`$ over mass and internal energy supplies the gravitational
mass distribution. See the [SIDM sampler](cpu/include/ev/weighted_vlasov_pic_collisions.hpp#L987)
and [subhalo pair laws](cpu/include/ev/weighted_vlasov_pic_subhalo_collisions.hpp#L325),
[scattering and remnant update](cpu/include/ev/weighted_vlasov_pic_subhalo_collisions.hpp#L1664).

### Einstein–Vlasov gravity

The relativistic modules use $`G=c=1`$, unit particle rest mass, and metric
signature $`(-,+,+,+)`$. Here $`f`$ is the particle-number distribution.
With spacetime indices $`\alpha,\beta,\gamma`$, four-momentum $`p^\alpha`$, metric
$`g_{\alpha\beta}`$, and particle proper time $`\tau_{\mathrm p}`$, repeated spacetime
indices are summed. The governing system is

```math
G_{\alpha\beta}=8\pi T_{\alpha\beta},\qquad
g_{\alpha\beta}p^\alpha p^\beta=-1,
\qquad
\frac{\mathrm dx^\alpha}{\mathrm d\tau_{\mathrm p}}=p^\alpha,\qquad
\frac{\mathrm dp^\alpha}{\mathrm d\tau_{\mathrm p}}=-\Gamma^\alpha_{\beta\gamma}p^\beta p^\gamma,
\qquad
\frac{\mathrm df}{\mathrm d\tau_{\mathrm p}}=0,\qquad
\frac{\mathrm d\eta}{\mathrm d\tau_{\mathrm p}}=0.
```

The Einstein tensor $`G_{\alpha\beta}`$ describes curvature; the particle
moments give the stress-energy tensor $`T_{\alpha\beta}`$. The coordinates
are $`x^\alpha`$, and $`\Gamma^\alpha_{\beta\gamma}`$ are the Christoffel
symbols computed from the metric. The geodesic characteristic equations evolve
position and momentum while carrying $`f`$ unchanged. Radial Einstein constraints reconstruct its metric
at each coupled particle stage.

In **Schwarzschild coordinates**, $`\tau`$ is coordinate time and $`r`$ is
areal radius. With $`\mathrm d\Omega^2`$ the unit-sphere metric, the functions
$`\mu,\lambda`$ determine the lapse and proper radial distance:

```math
\mathrm ds^2=-e^{2\mu}\mathrm d\tau^2+e^{2\lambda}\mathrm dr^2
 +r^2\mathrm d\Omega^2.
```

Here $`w`$ is local radial momentum per unit rest mass; its local radial
speed is $`w/\epsilon`$, with $`\epsilon=\sqrt{1+w^2+2\eta/r^2}`$.
The energy density $`\rho`$, radial pressure $`p_r`$, and radial momentum
density $`j`$ supply the radial constraints and the current relation:

```math
(\rho,p_r,j)=\frac{2\pi}{r^2}\int f
 \left(\epsilon,\frac{w^2}{\epsilon},w\right)
 \mathrm dw\,\mathrm d\eta,
\qquad
m_r=4\pi r^2\rho,\qquad e^{-2\lambda}=1-\frac{2m}{r},
\qquad
\mu_r=\frac{m/r^2+4\pi r p_r}{1-2m/r},\qquad
\lambda_\tau=-4\pi r e^{\lambda+\mu}j.
```

The enclosed gravitational mass is $`m`$; subscripts indicate partial
derivatives. The radial solve reconstructs $`m,\lambda,\mu`$, and the
$`\lambda_\tau`$ relation supplies the current contribution to the momentum
rate. These fields advance the collisionless characteristics:

```math
\frac{\mathrm dr}{\mathrm d\tau}=e^{\mu-\lambda}\frac{w}{\epsilon},
\qquad
\frac{\mathrm dw}{\mathrm d\tau}=-\lambda_\tau w
 -e^{\mu-\lambda}\mu_r\epsilon
 +e^{\mu-\lambda}\frac{2\eta}{r^3\epsilon}.
```

The regular-center solve starts from $`m(0)=0`$ and matches the lapse to
the outer vacuum. See the [Schwarzschild metric reconstruction](cpu/include/ev/weighted_einstein_vlasov_pic_schwarzschild.hpp#L716)
and [characteristics](cpu/include/ev/weighted_einstein_vlasov_pic_schwarzschild.hpp#L1022).

In **ingoing EF coordinates**, use the metric defined above, advanced time
$`v`$, and positive canonical radial momentum $`P`$ conjugate to $`r`$.
The dimensionless field $`a=1-2m/r`$ gives compactness, and positive
$`b`$ sets the advanced-time normalization relative to the outer vacuum.
Define $`\chi=1+2\eta/r^2`$. The covariant stress component $`T_{rr}`$ and
source combination $`S`$ determine the radial constraints:

```math
T_{rr}=\frac{2\pi}{r^2}\int P f\,\mathrm d\eta\,\mathrm dP,
\qquad
S=\frac{2\pi}{r^2}\int\frac{\chi}{P}f\,\mathrm d\eta\,\mathrm dP,
\qquad
\partial_r\ln b=4\pi rT_{rr},\qquad
\frac1b\partial_r(mb)=2\pi r^2(T_{rr}+S),
\qquad a=1-\frac{2m}{r}.
```

Isolated evolution fixes the outer gravitational mass and $`b=1`$ in the
surrounding vacuum. The reconstructed center supplies a constraint residual.
The Hamiltonian $`\mathcal H=-p_v=b(aP+\chi/P)/2`$, where $`p_v`$ is
conjugate to $`v`$, gives

```math
\frac{\mathrm dr}{\mathrm dv}=\frac b2\left(a-\frac\chi{P^2}\right),
\qquad
\frac{\mathrm dP}{\mathrm dv}=-\frac P2(ab_r+ba_r)
 -\frac{b_r\chi}{2P}+\frac{2b\eta}{Pr^3}.
```

The code evolves $`\ln P`$ to retain $`P>0`$ through its orbit stages.
See the [EF stress deposition and constraints](cpu/include/ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp#L655)
and [Hamiltonian rates](cpu/include/ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp#L1226).
The [interior-evolution guide](cpu/docs/ef_interior_evolution.md) explains
trapping, observer density, and the domain conditions for horizon crossing.

## Numerical methods and diagnostics

The equations above enter the numerical calculation in the following order.

1. **Construct the initial distribution or restore a checkpoint.** A grid
   solver stores values of the distribution on its phase-space mesh. A
   particle solver stores positions, momenta and positive weights. A
   collisional restart also restores the random state and elapsed collision
   times needed to continue the same calculation.
2. **Calculate the gravitational sources.** Integrate the distribution over
   momentum, or deposit the particle weights onto radial cells. Newtonian
   gravity uses mass density. Relativistic gravity also needs energy density,
   pressures and momentum density.
3. **Solve for gravity.** The Poisson solver integrates density to enclosed
   mass and obtains the radial force. The Einstein solvers integrate their
   radial constraint equations to obtain the metric used by the trajectories.
4. **Advance the collisionless motion.** Select a time interval that resolves
   the relevant motion and lands on any intervening field, collision or output
   deadline. Particle methods advance characteristics. Grid methods follow
   characteristics backward or transport the distribution between cells.
5. **Apply encounters when they are due.** Build collision groups from the
   current particle radii, select candidate pairs, and use random draws to
   accept encounters at the prescribed rate. Apply the outgoing velocity or
   momentum map and update any subhalo remnant properties. Recompute the
   gravitational sources after the particles change.
6. **Save the result and continue.** Write the requested profiles and
   diagnostics, preserve the complete state in checkpoints, and repeat from
   the current distribution.

The comments in each implementation explain its particular reconstruction,
quadrature, trajectory update and collision sampling. The manual develops
those steps from the governing equations. The analysis tools then use the
saved fields and particle data to calculate density slopes, energies and
virial ratios. For expanding backgrounds, they include the prescribed Hubble
terms in the physical energy and virial budgets. Checkpoints carry the
background settings so that a resumed run uses the same equations.

Controlled collapse experiments can reduce the initial physical velocities
with `--initial-velocity-factor F` (default 1). Positions and masses retain
their sampled values; fresh peculiar velocities become $`F(Hr+w)-Hr`$
and angular momenta are scaled by $`F`$. The initial-condition contract is
checked on resume, where saved velocities evolve directly. The separate
`virial_budget.csv` records global marker moments and the virial of the
actual applied force, including the inner carrier. This complements the
resolved-shell Newtonian ratio in the resolved-shell virial report.

The EF chapter explains the coordinate transformation, radial Einstein
constraints, horizon-crossing density observer, and restart controls. The manual covers all maintained CPU and GPU solver
families. The executed implementation checks and the scope of the example
results are summarized in [`release/VALIDATION.md`](release/VALIDATION.md).
The [figure reproduction guide](analysis/README.md) explains how to regenerate
the twelve example figures in [plots/](plots/README.md). The [movie guide](movies/README.md) identifies the
corresponding evolution movies and their rendering commands.

## Release layout

```text
VENUS/
  cpu/                 C++17 CPU solvers and analysis tools
  gpu/                 CUDA/C++ WPIC solvers, MPI, and benchmarks
  docs/                compiled manual and documentation index
  notes/               editable manual and solver/script catalogues
  scripts/             release-wide verification and utility scripts
  plots/               all twelve example figures as PDF and PNG
  analysis/            plotting inputs and commented analysis scripts
  movies/              simulation movies, inputs and renderers
  release/             source checksums and validation summary
  output/              empty destinations for local CPU/GPU runs
  Makefile             common build and documentation entry points
```

The CPU subtree is organized by role:

```text
cpu/
  src/apps/             one distinct main for every supported CPU task
  src/drivers/          shared WPIC execution engines
  include/ev/           shared equations, discretizations, collisions, I/O
  tools/                IC/restart utilities, analyzers, plots, and GIFs
  python/ev_solver/     Python reference solver and readers
  examples/
    initial_data/       supplied binary phase-space initial conditions
  scripts/slurm/        portable CPU Slurm templates
  docs/                 CPU examples and cluster/parallel reference
  ../output/cpu/        root-level CPU run products
```

The GPU subtree remains separate because its implementation is device-native:

```text
gpu/
  include/venus_gpu/    public WPIC, collision, restart, and MPI APIs
  src/                  CUDA kernels and host orchestration
  tools/                one distinct main per GPU task and restart utilities
  benchmarks/           focused VP and material-SIDM benchmarks
                        restart-based example arguments
  scripts/              GPU plotting/GIF front end
  docs/                 GPU equations, algorithms, and parallel reference
  ../output/gpu/        root-level GPU run products
```

The root commands orchestrate a subtree; direct subtree commands expose the
task builds:

```bash
make -C cpu organized-solvers       # portable CPU task executables
make -C cpu cpu-task-solvers-openmp
make -C cpu cpu-task-solvers-mpi    # Eulerian EV/VP only
make gpu GPU_ARCH=80                # all single-GPU task executables
make gpu-mpi GPU_ARCH=80            # all one-rank-per-GPU task executables
```

The numerical kernels are shared inside each implementation tree, but every
supported physical task has a separately named executable that locks
its equation and collision model. Generic dispatchers remain only for
compatibility. Generated profiles, restart dumps, binaries, caches, and
scheduler logs are not release source.

## Capability and scope summary

| Module | CPU status | GPU status |
|---|---|---|
| Grid Einstein--Vlasov | implemented grid task; convergence must be established for each calculation | absent |
| Weighted Einstein--Vlasov PIC | separate Schwarzschild/polar--areal and ingoing-EF modules; EF follows apparent-horizon formation and subsequent interior evolution | absent |
| Static/expanding spherical Vlasov--Poisson | grid and expanding-WPIC task fronts | static/expanding WPIC restart task |
| Grid Fokker--Planck/Landau Boltzmann--Poisson | implemented; problem-specific convergence required | absent |
| Exact/reversible/sampled/streamed finite-angle Boltzmann | separate operator-locked task fronts | absent |
| Canonical-orbit Boltzmann--Poisson | orbit-atlas task; method-specific convergence required | absent |
| Persistent-marker SphDSMC | separate constant and velocity-dependent task fronts | absent |
| Material SIDM, constant | WPIC and SphDSMC tasks with documented locality limits | static-halo restart task with documented locality limits |
| Material SIDM, power-law/Yukawa--Born | velocity-dependent WPIC/SphDSMC research scope | static-halo restart tasks; convergence required per run |
| Subhalo Plummer scatter/capture | expanding WPIC task | static/expanding restart task |
| Collision-cell/shell adaptive clocks | per-cell subhalo WPIC; per-shell constant-SIDM WPIC for equal-mass markers | per-cell subhalo and per-shell material SIDM, including constant, power-law, and Yukawa--Born laws |
| Shared-memory parallelism | OpenMP or `std::thread` | CUDA kernels |
| Distributed memory | MPI $`L`$ slabs for grid EV/VP only | one MPI rank per GPU; material rank-count invariance is not claimed |

The complete one-task/one-entry-point map is
[`notes/SOLVER_MODULES.md`](notes/SOLVER_MODULES.md); the full equations and
algorithmic limits are in
[`docs/venus_manual.pdf`](docs/venus_manual.pdf). Every quantitative
calculation still requires resolution, timestep, boundary, and statistical
convergence appropriate to its physical observable.

## Einstein--Vlasov coordinate modules

The compatibility header
[`cpu/include/ev/weighted_einstein_vlasov_pic.hpp`](cpu/include/ev/weighted_einstein_vlasov_pic.hpp)
exposes two distinct implementations:

- [`weighted_einstein_vlasov_pic_schwarzschild.hpp`](cpu/include/ev/weighted_einstein_vlasov_pic_schwarzschild.hpp)
  evolves the polar--areal variables and stops before
  $`2m/r=1`$, where that chart becomes singular;
- [`weighted_einstein_vlasov_pic_eddington_finkelstein.hpp`](cpu/include/ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp)
  evolves the canonical future-directed null-slice momentum $`P=p_r>0`$,
  solves the neutral advanced-null constraints, detects $`a=1-2m/r=0`$,
  and evolves matter through the apparent horizon.

The bound equilibrium generator is separate again:
[`weighted_einstein_vlasov_static_shell.hpp`](cpu/include/ev/weighted_einstein_vlasov_static_shell.hpp)
solves the normalized static continuum problem and emits invariant-weight
adjacent $`(+w,-w)`$ marker pairs. Those weights already include $`e^\lambda`$
and must not be canonicalized a second time.

Select the implementation with `--coordinates schwarzschild` or
`--coordinates eddington_finkelstein`. A raw physical-IC file is deliberately
**not** accepted as fresh EF data: a polar--areal $`\tilde t=\mathrm{constant}`$ slice is
spacelike, whereas an ingoing $`v=\mathrm{constant}`$ slice is null. VENUS therefore
first evolves a deterministic Schwarzschild donor history, solves the ingoing
eikonal equation, and intersects that null hypersurface with every marker
worldline. Here the eikonal equation is simply the equation requiring the
surface to be made of light rays. The resulting null-slice file records both
its provenance and SHA-256 content hashes; that checked file, rather than the
raw spacelike IC, initializes the EF run.

In polar--areal coordinates,

```math
ds^2=-e^{2\mu}\,d\tilde t^2+e^{2\lambda}\,dr^2+r^2d\Omega^2.
```


The future ingoing coordinate satisfies
$`g^{ab}\partial_a v\,\partial_b v=0`$, with the advanced root
$`\partial_r v=e^{\lambda-\mu}\partial_{\tilde t}v`$. Writing
$`A=\partial_{\tilde t}v>0`$ gives
$`dv=A\bigl[d\tilde t+e^{\lambda-\mu}dr\bigr]`$,
$`a=e^{-2\lambda}`$, and $`b=e^{\lambda+\mu}/A`$.
The inverse one-form is
$`d\tilde t=F\bigl[dv-dr/(ab)\bigr]`$, where $`F=1/A`$ and
$`\partial_r F=-\partial_v\bigl[F/(ab)\bigr]`$. In a static Schwarzschild exterior with
the clocks normalized at infinity, $`A=F=1`$ and
$`v=\tilde t+r_{\ast}`$, $`\tilde t=v-r_{\ast}`$, with
$`r_{\ast}=r+2M\log\left|r/(2M)-1\right|`$.

The forward equation is the full ingoing eikonal equation:

```math
-e^{-2\mu}(\partial_{\tilde t}v)^2+e^{-2\lambda}(\partial_r v)^2=0.
```


The displayed positive root chooses future-directed, inward-moving null
surfaces. In a dynamical spacetime, neither direction of the map is a static
tortoise-radius substitution: the forward or inverse eikonal equation must be
solved through the relevant spacetime history.

The same-event checker evolves the logarithms of the map Jacobians, not the
Jacobians directly. Along a forward null ray,
$`\frac{d\log A}{dr}=-\partial_{\tilde t}\bigl[e^{\lambda-\mu}\bigr]`$; along an inverse ray with
$`q=1/(ab)`$, $`\frac{d\log F}{dr}=\partial_v q`$. Thus $`A=\exp(\log A)>0`$ and
$`F=\exp(\log F)>0`$ by construction. This is the exact tangent equation. It
prevents a stiff explicit step from inventing a negative Jacobian; it does
not excuse insufficient resolution. The strict history check compares
otherwise identical $`n`$- and $`2n`$-substep traces using the raw changes in
coordinate time and in $`\log A`$ or $`\log F`$, each at the unchanged $`2\times10^{-5}`$
ceiling. It does not divide a log-Jacobian mismatch by $`|\log A|`$ or assume a
Richardson factor before fourth-order behavior has been demonstrated.
For a coordinate comparison, $`n`$ is one explicit positive integer set by
`VENUS_EIKONAL_SUBSTEPS`. The same value is authenticated in all four
resolution/realization diagnostics, the convergence evidence, and the final
certificate. Each diagnostic still computes both the $`n`$- and $`2n`$-substep
solutions and uses the refined $`2n`$ result. A missing or inconsistent value
stops the calculation. The production
Slurm scripts and Python tools are pinned to the staged `VENUS_SOURCE_ROOT`:
their submission-time SHA-256 identities are checked before and after every
dependent job and carried into the evidence. Inherited tool or Slurm-script
overrides are rejected.
The checker uses the production spatial reconstruction. On the Schwarzschild side it linearly joins
each face and center value of $`\lambda`$, uses the production slope-limited
cubic through the corresponding face/center values and radial derivatives of
$`\mu`$, and only then forms $`e^{\lambda-\mu}`$. On the EF side it linearly joins
face/center values of $`a`$ and $`b`$. The profile CSVs save these face data, and
version 2 of the authenticated `VENUSSCHIST1` donor history saves the SC face
data for negative $`\tilde t`$ as well. Parity certification requires the face and center values used by the
spatial interpolant, together with timestep refinement of the characteristic
integration. Any nonfinite Jacobian or inverse composition fails closed. The
separate composition gate uses
$`\frac{|\tau_{\mathrm{inverse}}-\tau|}{\max(1,|\tau|,|\tau_{\mathrm{inverse}}|)}`$ and
$`\left|\mathrm{expm1}(\log A+\log F)\right|`$, both at the default $`5\times10^{-4}`$ ceiling; this
normalization is not applied to the raw $`n`$-versus-$`2n`$ eikonal changes.

At a common event, the radial covector (the momentum component conjugate to
the radius) maps to

```math
P=p_r^{\mathrm{EF}}=e^\lambda(\epsilon+w)>0,
```

while $`\eta`$ and each marker's
invariant rest weight are unchanged. The EF pusher evolves $`u=\log P`$, so no
finite Runge--Kutta stage can change the required sign of $`P`$. Positive $`P`$
means future directed, not necessarily inward moving: the sign of
$`\frac{dr}{dv}=\frac b2\left[a-\frac{1+2\eta/r^2}{P^2}\right]`$ distinguishes infall, a turning point,
and outward motion. A distribution function is a scalar, meaning that its
value is unchanged at the same mapped event and momentum, with phase-space
measure relation $`dP/P=dw/\epsilon`$. A finite WPIC state represents
the empirical measure

```math
\widehat f[\psi]=\sum_p W_p\,\psi(r_p,\mathrm{momentum}_p,\eta_p),
```

not a
pointwise smooth function. The coordinate certificate therefore authenticates
unique IDs/source IDs, invariant weights and $`\eta`$, and mapped positions and
momenta, while separately checking the metric pullback (the explicit
Jacobian change of tensor components) and every covariant
stress-tensor component on the same events. A comparison is inconclusive if
the forward or inverse eikonal, event coverage, phase-space Jacobian, empirical
measure, tensor pullback, or resolution/PIC convergence evidence is absent.
At each sampled SC time, the candidate common domain is the one unbroken
radial interval attached to the common outer worldtube. Only a genuine stored-
history boundary or the $`a>10^{-3}`$ chart cutoff may end it. A failed forward
or inverse solve, failed composition, or failed metric identity remains an
unexplained hole inside the claimed interval and fails the certificate; it
cannot be used to shrink the domain until the remaining points look good. The
certifier chooses the longest contiguous block of radially adequate time
slices. Its strict defaults require at least 95% radial coverage on every
selected slice, 95% mapped-event coverage, 95% complete-time-slice coverage,
at least 64 mapped events, at least three complete times, and no holes or
interior re-entry. Field and tensor norms are rebuilt using only this declared
domain. In-domain maxima decide the gate, while separate all-attempted maxima
and exclusion counters preserve every failure outside it. The common domain
never extends past the SC chart; later post-horizon EF events have no
Schwarzschild-chart counterpart. The required
marker table proves the sampled distribution on the imported initial null
slice. Later mapped metric and every stress/moment component are checked, but
the current writer does not emit the optional later marker-trajectory table.
Accordingly a certificate may not claim later marker-by-marker or pointwise
distribution equality unless its
`later_trajectory_evidence_present` field is true.

The null mapper may use a vacuum worldtube $`R_{\mathrm B}`$ strictly inside the EF
evolution boundary $`R_{\mathrm E}`$. $`R_{\mathrm B}`$ must be an exact mesh face with four-cell
matter/crossing clearance. The eikonal is solved only through $`R_{\mathrm B}`$; on the
initial null slice, the verified vacuum interval $`R_{\mathrm B}<r\le R_{\mathrm E}`$ is continued
exactly with $`m=M_{\mathrm{null}}`$, $`a=1-2M_{\mathrm{null}}/r`$, $`A=F=b=1`$, zero stress, and the
Schwarzschild tortoise difference. This analytic continuation is initial-data
construction only—the EF solver subsequently evolves the complete $`R_{\mathrm E}`$
domain. Both radii, the extension index, map/full-grid fingerprints, vacuum
residuals, and artifact hashes are authenticated on import and resume.
Here “worldtube” means the fixed-radius surface $`r=R_{\mathrm B}`$ followed through
time; it is the boundary where the SC and EF clocks are normalized.

If the extracted surface has the worldtube label $`v=-T_{\mathrm{donor}}`$, runtime output
uses only the shifted origin $`\widehat v=v+T_{\mathrm{donor}}`$, so its imported initial
surface is $`\widehat v=0`$. This shift changes no tensor or momentum component.

Mapped-run summaries keep three masses deliberately separate. The
Schwarzschild $`\tau=0`$ normalization is `fixed_family_adm_mass` and defines the
chosen fixed-ADM angular-momentum family. The evolved donor value on the
extracted null slice is `null_slice_conserved_adm_mass`; it is not reset to the
target because its difference measures numerical donor conservation. Finally,
`initial_deposited_adm_mass` is independently reconstructed from the EF
markers on $`\widehat v=0`$. The exact vacuum extension is anchored to the null-slice
mass, not either of the other two. VENUS reports both mismatch channels and
the cluster preflight authenticates them, every map artifact, the vacuum
worldtube, unique crossings, and both component RK4 step-doubling eikonal
errors before dependent jobs can run.

A “realization” is one deterministic shift of the quiet marker sampler, not a
different physical shell. “Convergence” means joint refinement of radial
spacing, timestep, and marker count, with the coordinate errors decreasing
until they are no larger than the difference between realizations. The
minimum certificate therefore needs two resolutions and two realizations.
Its saved cadence must satisfy
$`\frac{\max(\mathtt{saved\_time\_gap})}{\min(\mathtt{radial\_spacing})}\le0.75`$ in both coordinates. For
the documented $`R_{\mathrm E}=24`$, $`N_r=2048`$, `dt_max=0.004` grid this permits
`output_steps <= 2`; strict parity saves every step. The threshold scout's
`output_steps=50` is a science cadence and cannot issue a new coordinate
certificate.

The same normalized Schwarzschild $`\tau=0`$ family also defines
`fixed_family_rest_weight` $`=M_0`$ and the coordinate-independent global binding
diagnostic `fixed_family_binding_fraction` $`=(M_0-M_{\mathrm{ADM}})/M_0`$. SC and mapped EF
summaries, including resumed runs, report the identical family value from an
immutable diagnostic sidecar; it is never recomputed from the later null
slice. A halo label requires a present, internally consistent positive binding
fraction. This is necessary global binding evidence, not proof of equilibrium.
The separately reported marker statistic
$`E_{\mathrm{proxy}}=e^\mu\sqrt{1+w^2+2\eta/r^2}`$ is evaluated in the instantaneously
frozen Schwarzschild $`\tau=0`$ metric. Its minimum, maximum, rest-weighted mean,
and rest-weight fraction with $`E_{\mathrm{proxy}}\ge1`$ are diagnostic only: a dynamical
spacetime has no corresponding conserved particle energy, so the proxy never
acts as a halo gate. Likewise, the reported $`2K/|W|`$ is explicitly a
Newtonian virial proxy rather than a relativistic virial identity.
At every saved output VENUS instead evaluates the GR anisotropic static-balance
residual

```math
R_J=\frac{dp_r}{dr}+(\rho+p_r)\frac{d\mu}{dr}-\frac{p_{t,\mathrm{sum}}-2p_r}{r}
```

and records a
shell-volume-weighted relative L1 norm. For EF data the required polar-areal
moments are pulled back only where $`a=1-2m/r>0`$, so this diagnostic is used for
halo evidence only on the regular pre-horizon domain. The conservative
threshold classifier requires sustained late-time Jeans balance together with
stationary `r50`, `r90`, and maximum compactness, small escape, positive global
binding energy, and no horizon. Missing balance data fail closed; an outlying
Newtonian $`2K/|W|`$ does not veto a relativistic halo. SC and EF call one
shared time-series classifier, record its complete thresholds/statistics and
`failed_checks` list under `high_compactness.halo_classifier`, and require the
maximum saved compactness to remain below $`0.99`$. Thus a Schwarzschild run
that only stalls near $`2m/r=1`$ is `undetermined`, never a halo; its fate must
be resolved by the mapped EF evolution.
The quantities `r50` and `r90` are the radii enclosing 50 and 90 percent of
the currently active invariant rest weight. “Fails closed” means that a
missing or failed check produces `undetermined` or stops the calculation; it
never counts as halo evidence. The manual states the exact late-time window,
percentile, fitted-slope, escape, compression, compactness, and Jeans-residual
inequalities used by this classifier.

A raw EF apparent horizon is likewise not a final black-hole label. It becomes
`black_hole_candidate` only if the run reports a consistent outer $`a=0`$
crossing, reaches saved compactness at least $`0.99`$, continues for
$`\frac{v_{\mathrm{final}}-v_{\mathrm{AH}}}{M_{\mathrm{family}}}\ge5`$, and has at least three saved rows strictly
after $`v_{\mathrm{AH}}`$. The driver audits the initial state and every accepted EF step,
not just saved output: every marker must keep $`P>0`$, every center/face must keep
$`b>0`$, all marker/field/source arrays must stay finite, and the EF constraint
must retain its fixed null-slice ADM anchor to roundoff. The extrema and flags
are checkpointed in `ef_post_horizon_audit.state`; a missing or stale sidecar
on resume disables promotion rather than forgetting pre-resume failures.

The candidate gate also requires positive and consistent target, fixed-family,
null-slice, and deposited ADM masses; saved ADM and rest-weight accounting;
explicit negative-$`a`$ excision and a positive integer buffer; an excision
row lying strictly after $`v_{\mathrm{AH}}`$ and satisfying $`0<r_{\mathrm{exc}}<r_{\mathrm{AH}}`$ whenever
excision is active; and actual positive rest weight excised
inside a trapped region. A candidate becomes `black_hole_certified` only when
a separate SHA-256-authenticated sensitivity manifest supplies at least one
intrinsic-candidate repeat with a different excision threshold at fixed buffer
and one with a different buffer at fixed threshold, all bound to the same IC,
map, angular momentum, and fixed ADM family. The analyzer also re-hashes each
run's `binary.sha256` and `source.sha256` records and requires the same
executable and exact source manifest for the primary and every repeat. Across
the primary and repeats,
$`\frac{\max(v_{\mathrm{AH}})-\min(v_{\mathrm{AH}})}{M_{\mathrm{family}}}\le1`$ and
$`\frac{\max(r_{\mathrm{AH}})-\min(r_{\mathrm{AH}})}{\mathrm{median}(r_{\mathrm{AH}})}\le0.05`$, using each run's first trapped
time and its positive summary apparent-horizon radius; users may tighten but
not loosen these limits. Missing sensitivity evidence leaves a candidate
uncertified, and an uncertified candidate can never close the final
angular-momentum bracket. This excision certificate is separate from the
SC--EF parity certificate: both are required for a final threshold
claim.

### Bound static shell and threshold family

For `--high-initial-condition energy_cutoff_shell`, define
$`y=\log E_0-\mu`$ and
$`W(r)^2=e^{2y}-1-\ell_0^2/r^2`$. Matter has $`|w|<W`$, and the delta-function
Jacobian is
$`\delta(\ell-\ell_0)=2\ell_0\,\delta(\ell^2-\ell_0^2)`$. The code evaluates the full
energy, radial-pressure, summed tangential-pressure, and rest-density
integrals and solves

```math
\frac{dm}{dr}=4\pi r^2\rho,
\qquad
\frac{dy}{dr}=-\frac{m/r^2+4\pi rp_r}{1-2m/r},
\qquad
\frac{dM_0}{dr}=4\pi r^2e^\lambda n_0.
```

At the outer support edge it fixes the lapse by
$`\log E_0=y(R_{\mathrm{out}})+\frac12\log(1-2M/R_{\mathrm{out}})`$, requires $`E_0<1`$ and $`M_0-M>0`$, and matches
the exact Schwarzschild vacuum. The manual writes every integral and boundary
condition explicitly.

The production sampler uses exactly $`N_{\mathrm{marker}}/2`$ shifted rank-one points and
emits adjacent $`(+w,-w)`$ markers with unique persistent IDs/source IDs. The
unperturbed $`s_L=1`$ state is an exact continuum equilibrium. One common kick
$`w\longmapsto w-\frac{\kappa r^2}{r^2+r_c^2}`$ (or the untapered $`w\longmapsto w-\kappa`$ when
$`r_c=0`$) is then fixed for the entire scan. Only
$`\eta\longmapsto s_L^2\eta`$ varies, followed by one common positive rest-weight factor
that restores the chosen initial ADM mass. Positive realized binding is
checked at every point. A sufficiently high-angular-momentum member of a
genuinely bound family is expected to settle into a halo; an escaping endpoint
is classified as dispersal, not as the requested halo side of a threshold.
For an EF threshold point, the physical SC donor evolves forward with the
incoming data; a separate momentum-reversed scratch copy supplies only the
negative-SC-time part needed by the null extractor. The outcome classifier
uses the EF scan output. A coarse angular-momentum scout searches for a
bracket but is not a coordinate-equivalence certificate and is not a final
threshold without an authenticated excision-sensitivity certificate,
refinement, and a second sampling realization.

The continuum, kick, target mass, $`R_{\mathrm B}`$, $`R_{\mathrm E}`$, donor duration, end time, and
initial angular bracket are deliberately not assigned production numbers in
the README. Local initialize-only and short dynamic preflights must establish
equilibrium quality, binding, TSC support clearance, a high-$`L`$ halo endpoint,
a low-$`L`$ intrinsic post-horizon candidate with threshold and buffer
sensitivity tests, and complete null-map coverage before longer cluster jobs are
submitted.

The manual gives the full Einstein tensor, Vlasov characteristics, source
normalization, radial constraints, discretization, Schwarzschild-to-EF and
EF-to-Schwarzschild transformations, dynamical eikonal equation, horizon and
excision rules, validation hierarchy, and angular-momentum threshold protocol.

## Timestep policy

Newtonian collisionless WPIC uses the fixed orbit step selected by
`--maximum-timestep`; only an exact output or final-time landing may shorten a
step. This preserves the time-symmetric kick--drift--kick energy behavior of a
constant-potential control. Collisional tasks retain their outer gravity and
operator-event clocks, but can advance disjoint collision regions on local
CFL clocks inside one collision batch. For CPU adaptive subhalo batches,
the outer deadline uses the dynamical and potential-change times, while local
collision CFL clocks resolve the encounters. The CUDA driver also retains the
measured collision time in its outer deadline:

- CPU subhalo WPIC: `--subhalo-adaptive-cell-timestep`;
- CPU constant-cross-section, equal-marker-mass material SIDM:
  `--collision-adaptive-shell-timestep`, with
  `--collision-adaptive-maximum-interval DT` as the outer batch cap;
- GPU subhalo WPIC: `--subhalo-adaptive-cell-timestep`, optionally with
  `--subhalo-adaptive-batch-substeps N`;
- GPU material SIDM: `--sidm-adaptive-shell-timestep` for constant,
  softened-power-law, or Yukawa--Born tasks.

With an adaptive clock, cell or shell $`c`$ advances by

```math
\min\!\left(\mathtt{batch\_remaining}[c],\frac{\mathtt{collision\_cfl}}{\mathtt{rate}[c]}\right);
```

Each region therefore advances on the encounter timescale of its own
particles, while all regions reach the end of the common collision batch. The fixed mesh-wide reference mode is
the default and remains available by omitting the adaptive flag. Collision
probabilities, locality requirements, conservation laws, and the outer
gravity/collision operator splitting are unchanged. The manual explains the
CPU and CUDA implementations separately, including proposal rebuilds after
subhalo capture and MPI ownership.

The optional single-GPU subhalo policy `--coupled-receiver-clocks` combines
an orbital bound with receiver-mass field variations. Its EdS observer
includes resolved shell mass and cloud mass inside the inner face; static
runs use the fallback maximum norm. The optional
`--collision-prestep-coordinate-due` applies a safe, positive carried age at
the current state when the next deadline is coordinate-close. The
[field-clock guide](gpu/docs/COUPLED_RECEIVER_CLOCKS.md) defines both
policies, their default-off interfaces and the checks on mass, field
variation and collision accuracy.





## Example calculations and custom inputs

The [analysis guide](analysis/README.md) maps each figure to its physical model,
saved numerical data and reproduction command. The [movie guide](movies/README.md)
provides the corresponding evolution clips. The Gaussian shell generator and
its [example](cpu/examples/modular/vp_bound_gaussian_shell_gpu.md) construct
the shared physical initial particles for static and expanding VP/BP.

The examples under `cpu/examples/modular` explain input formats, boundary
conditions and task interfaces. Use these together with the [initial-data
guide](cpu/examples/initial_data/README.md) when constructing a new physical
problem. The [manual](docs/venus_manual.pdf) derives the algorithms and lists
the controls that require resolution and conservation checks.


## Relativistic coordinate comparison

To compare the same high-compactness system in both coordinates, first
construct its particles on a spacelike slice. The null extractor follows
their worldlines to one ingoing null slice and writes the particles and
metric there. The EF run starts from that saved null slice. The upper-case shell variables below are symbolic
placeholders fixed by successful local preflight; they are not shipped
production defaults:

```bash
./venus_cpu_einstein_vlasov_wpic_black_hole --help
./venus_cpu_einstein_vlasov_wpic_black_hole \
  --scenario high --coordinates schwarzschild \
  --high-initial-condition energy_cutoff_shell \
  --physical-ic common_ic.bin \
  --high-energy-cutoff-amplitude "$STATIC_C" \
  --high-energy-cutoff-ell0 "$STATIC_ELL0" \
  --high-energy-cutoff-central-y "$STATIC_YC" \
  --high-markers "$MARKERS" \
  --high-sampling-realization "$REALIZATION" \
  --high-radial-infall "$FIXED_KICK" \
  --high-infall-core-radius "$FIXED_KICK_CORE" \
  --high-angular-momentum-scale "$PARITY_S_L" \
  --high-target-initial-adm-mass "$TARGET_ADM" \
  --export-ef-null-initial-data common_ingress_null.bin \
  --ef-null-donor-duration "$DONOR_DURATION" \
  --ef-map-worldtube-radius "$R_B" \
  --high-r-max "$R_E" --high-radial-cells "$N_R" \
  --output-dir output/sc --expected-outcome any
./venus_cpu_einstein_vlasov_wpic_black_hole \
  --scenario high --coordinates eddington_finkelstein \
  --high-initial-condition energy_cutoff_shell \
  --physical-ic common_ic.bin \
  --high-energy-cutoff-amplitude "$STATIC_C" \
  --high-energy-cutoff-ell0 "$STATIC_ELL0" \
  --high-energy-cutoff-central-y "$STATIC_YC" \
  --high-markers "$MARKERS" \
  --high-sampling-realization "$REALIZATION" \
  --high-radial-infall "$FIXED_KICK" \
  --high-infall-core-radius "$FIXED_KICK_CORE" \
  --high-angular-momentum-scale "$PARITY_S_L" \
  --high-target-initial-adm-mass "$TARGET_ADM" \
  --ef-null-initial-data common_ingress_null.bin \
  --high-r-max "$R_E" --high-radial-cells "$N_R" \
  --high-inner-boundary origin_parity --high-radial-grid linear \
  --output-dir output/ef --expected-outcome any
```

The `cpu/scripts/slurm/submit_einstein_vlasov_wpic_parity.sh` procedure
automates the requested two coordinate runs plus the minimum two-resolution
by two-independent-realization convergence matrix and produces a fail-closed
coordinate-equivalence certificate. It requires one explicit positive
`VENUS_EIKONAL_SUBSTEPS` value and binds that value, together with the
submitted script and tool hashes, across every diagnostic, the convergence
record, and the terminal certificate. The angular scan uses one bound static
shell, one fixed kick, and `--high-target-initial-adm-mass`, so changing
particle angular momentum does not silently change the initial gravitating
mass. Its analyzer separates an intrinsic `black_hole_candidate`, an
excision-sensitivity `black_hole_certified` endpoint, a time-resolved
virialized halo, dispersal, and an undetermined outcome. Only a converged
low-$`L`$ `black_hole_certified`/high-$`L`$ halo bracket is called the requested
threshold; a candidate plus a halo is explicitly inconclusive.

The map worldtube, evolution boundary, donor duration, marker count, and grid
spacing set the domain and resolution of this characteristic extraction.
Choose them from the largest-marker crossing and vacuum-boundary checks for
the specified initial distribution. Joint spatial and temporal refinement
then measures the accuracy of the strong-gravity evolution.

## Documentation

- [`docs/venus_manual.pdf`](docs/venus_manual.pdf): unified,
  physics-oriented CPU/GPU manual.
- [`notes/venus_manual.tex`](notes/venus_manual.tex): manual source.
- [`notes/SOLVER_MODULES.md`](notes/SOLVER_MODULES.md): exhaustive module,
  entry-point, status, I/O, parallelism, example, and limitation catalogue.
- [`notes/SCRIPT_REFERENCE.md`](notes/SCRIPT_REFERENCE.md): every maintained
  solver, IC, analysis, plot, GIF, and launch script.
Build the manual, then render every page for visual inspection with:

```bash
make manual
make -C notes render
```

## Validation and scientific scope

The [validation summary](release/VALIDATION.md) identifies the finite implementation
checks supporting each solver. Scientific calculations also require checks of
resolution, timestep, collision locality, particle sampling and domain boundaries.
The distribution supplies production and analysis sources; developer regression
suites are maintained separately.

### Relativistic material collisions

The [relativistic collision library](docs/relativistic_boltzmann.md) supplies local covariant elastic scattering in Schwarzschild (SC) and ingoing Eddington–Finkelstein (EF) coordinates. A packet carries represented rest mass $`W_i`$ and represents $`q_i=W_i/m_\chi`$ material particles of mass $`m_\chi`$. Its numerical weight tells the collision sampler how many physical particles share the stored momentum. For specific four-momentum $`p^\alpha`$ with $`p\cdot p=-1`$, the rest-mass distribution $`f`$ obeys

```math
\begin{aligned}
&p^\alpha\partial_\alpha f-\Gamma^i{}_{\alpha\beta}p^\alpha p^\beta\partial_{p^i}f=\mathcal C[f],\\
&\mathcal C[f_1]=\frac{1}{m_\chi}\int\mathrm d\Pi_2\,\mathcal F_{12}\int\mathrm d\Omega\,\frac{\mathrm d\Sigma}{\mathrm d\Omega}(f_3f_4-f_1f_2).
\end{aligned}
```

Here $`\partial_\alpha=\partial/\partial x^\alpha`$ for spacetime coordinates $`x^\alpha`$, $`i`$ labels three independent momentum components, and $`\Gamma^i{}_{\alpha\beta}`$ are the metric connection coefficients. The local invariant measure is $`\mathrm d\Pi=\mathrm d^3p/p^{\hat0}`$, and $`\mathcal F_{12}=\sqrt{(p_1\cdot p_2)^2-1}`$. Labels 1/2 and 3/4 denote incoming and outgoing momenta. The material differential cross section is measured in the pair centre-of-momentum frame. Physical momenta are $`k=m_\chi p`$ and the number distribution is $`f_N=f/m_\chi^4`$. The spherical coupled sampler uses constant isotropic area $`\mathrm d\Sigma/\mathrm d\Omega=\Sigma/(4\pi)`$. A separate local interface evaluates supplied reciprocal scalar amplitudes.

SC uses proper cell volume $`V`$, lapse $`e^\mu`$, and local specific energy $`\epsilon=\sqrt{1+w^2+2\eta/r^2}`$. EF uses coordinate shell volume $`V_0`$, metric coefficient $`b`$, and positive specific canonical momentum $`P=p_r`$. Here $`w`$ is SC local radial momentum, $`r`$ is areal radius, and $`\eta=L^2/2`$ for specific angular momentum $`L`$. Their packet hazards per SC coordinate time $`\tau`$ or advanced EF time $`v`$ are

```math
\begin{aligned}
&\Gamma_{ij}^{\rm SC}=\max(W_i,W_j)\frac{\Sigma}{m_\chi}\frac{e^\mu\mathcal F_{ij}}{V\epsilon_i\epsilon_j},\\
&\Gamma_{ij}^{\rm EF}=\max(W_i,W_j)\frac{\Sigma}{m_\chi}\frac{b\mathcal F_{ij}}{V_0P_iP_j}.
\end{aligned}
```

The EF rate counts particle flux through a null surface. `eb::advance_sc` and `eb::advance_ef` first stream the particles self-consistently, assign their current radii to cells, and use the returned `accepted_dt` for the frozen collision exposure. Each cell preserves its random state, residual optical depth, and number-weight bound. The sampler proposes two distinct packets uniformly, then accepts the pair with probability equal to its rate divided by the proposal-rate bound. An accepted event scatters equal portions of the two packets, each carrying the smaller weight. The larger packet retains its remaining weight at its incoming momentum. This realizes the physical pair rate while conserving the represented rest mass and four-momentum. The outgoing momenta are projected to SC or positive EF variables. After all cell batches, the routines reconstruct stress and the Einstein constraints. A zero area returns the direct EV update.

The [guide](docs/relativistic_boltzmann.md) gives the proposal/acceptance equations, exact draw order, conservative split, projection, field feedback, restart ownership, and source links. The [manual chapter](docs/venus_manual.pdf#nameddest=sec:relativistic-material-collisions) derives the slice measures and governing/discrete equations beside literal implementation excerpts. Finite component checks establish local conservation, slice-dependent rates, model-bound cell restart, the collisionless limit, and small source-feedback steps. Homogeneous finite radial cells require explicit opt-in and report separated exchanges. Compact-halo collapse requires equilibrium-preservation, continuous-chronology, spatial-locality, and convergence studies.
