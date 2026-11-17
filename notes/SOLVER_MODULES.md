# VENUS solver-module catalogue

This is the detailed method reference. For a first calculation, follow the
[CPU guide](../cpu/README.md) or [GPU guide](../gpu/README.md). VP means
Vlasov--Poisson (collective Newtonian gravity); BP means Boltzmann--Poisson
(with encounters); EV means Einstein--Vlasov (relativistic collisionless
gravity). WPIC uses particles with mass weights; an Eulerian method uses
fixed cells in position and velocity. A kernel is a numerical routine,
a backend is an implementation, and an input contract states the required
file format and physical conventions. “Convergence” means that improving
resolution and timestep accuracy leaves the result stable.

Additional terms are defined in the [plain-language glossary](../docs/TERMS.md).

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur

Begin with the physical problem in the selector below, then read that
module's equations, update sequence, inputs, outputs, and accuracy checks.
Each task-specific executable fixes the equation, numerical representation,
and collision family. It validates those choices before calling the shared
Einstein, Poisson, transport, or collision kernels. New runs use these named
tasks, while compatibility dispatchers continue to read older inputs.

## Scope language

- **Implemented in the stated scope** means that an end-to-end runner and
  restart/output contract exist for the named method and parameter domain. It
  is not a claim of convergence for an arbitrary new problem.
- **Completed example** means that the supplied parameter set has reached its
  stated endpoint. Read the limitation in the same row before making a
  scientific claim.
- **Research scope** means that the numerical machinery is available, but a
  quantitative result requires the convergence, boundary, restart, parallel,
  and target-physics checks stated for that module.
- **Formula/kernel scope** means that the local operator implements the stated
  analytic law and pair invariants; trajectory-level convergence remains a
  separate requirement.
- **Reference/compatibility** means that the path is retained for comparison
  or older inputs and is not the primary interface.
- **Absent** means that the backend is not implemented in this release.

## Physics and background selector

“Static background” means an isolated calculation with zero cosmological
expansion. Its self-consistent gravitational field evolves with the matter.
“Expanding” means the prescribed Newtonian background specified by the input.
The relativistic isolated calculations evolve their spacetime metric.

| Physical category | Background | CPU modules | GPU modules |
|---|---|---|---|
| Non-relativistic collisionless | Static | Eulerian VP; **static VP WPIC** | Static VP WPIC, single GPU |
| Non-relativistic collisionless | Expanding | Eulerian VP; expanding VP WPIC | EdS VP WPIC, single GPU or MPI |
| Non-relativistic collisional: subhalo scattering and capture | Static or expanding | Subhalo WPIC; grid BP with the chosen encounter approximation | Subhalo WPIC: static/EdS on one GPU, EdS with MPI |
| Non-relativistic collisional: self-interacting dark matter | Static | Constant or velocity-dependent SIDM WPIC and SphDSMC; grid BP | Constant, softened power-law and Yukawa--Born SIDM WPIC, single GPU or MPI |
| Non-relativistic collisional: material/encounter models | Expanding | Grid BP with explicitly selected cross section and background | Expanding subhalo model above; material SIDM GPU runners use static halos |
| Non-relativistic collisional: globular and nuclear star clusters | Model-dependent | Small-angle Fokker--Planck/Landau machinery and encounter kernels; stellar relaxation requires a suitable gravitational encounter model and validation | A dedicated stellar relaxation solver is future work |
| Relativistic collisionless | Isolated, evolving spacetime | Eulerian EV; low-compactness EV WPIC; high-compactness EV WPIC in Schwarzschild or ingoing EF coordinates | A relativistic CUDA solver is future work |
| Relativistic collisionless | Cosmological expansion | A cosmological EV runner is future work | Future work |
| Relativistic collisional: subhalos | Isolated or expanding | A relativistic subhalo scattering/capture runner is future work | Future work |
| Relativistic collisional: self-interacting dark matter | Isolated | Covariant local elastic-collision library coupled to SC/EF streaming; component-tested API | Future work |
| Relativistic collisional: star clusters | Isolated or expanding | A gravitational stellar-encounter model and complete application runner are future work | Future work |

The elastic material library describes microscopic scattering under its stated
interaction law. Stellar gravitational relaxation and subhalo capture require
their own encounter physics. Read the [relativistic collision guide](../docs/relativistic_boltzmann.md)
for the library's rate, locality, restart, and validation scope.

## Executable selector

| Physical task | Canonical executable | Backend | Status |
|---|---|---|---|
| Grid Einstein--Vlasov | `venus_cpu_einstein_vlasov_eulerian`, `_openmp`, or `_mpi` | CPU | Implemented grid task; method-specific convergence required |
| Grid Vlasov--Poisson | `venus_cpu_vlasov_poisson_eulerian`, `_openmp`, or `_mpi` | CPU | Implemented grid task; method-specific convergence required |
| Grid BP, Fokker--Planck | `venus_cpu_boltzmann_poisson_fokker_planck` or `_openmp` | CPU | Implemented small-angle task |
| Grid BP, Landau | `venus_cpu_boltzmann_poisson_landau` or `_openmp` | CPU | Implemented Landau-named small-angle task |
| Grid BP, exact finite angle | `venus_cpu_boltzmann_poisson_exact` or `_openmp` | CPU | Exact-quadrature task |
| Grid BP, reversible finite angle | `venus_cpu_boltzmann_poisson_reversible` or `_openmp` | CPU | Reversible-quadrature task |
| Grid BP, sampled reversible | `venus_cpu_boltzmann_poisson_reversible_sampled` or `_openmp` | CPU | Sampled-reversible task |
| Grid BP, streamed reversible | `venus_cpu_boltzmann_poisson_reversible_streamed` or `_openmp` | CPU | Streamed-reversible task |
| Grid BP, finite-angle compatibility selector | `venus_cpu_boltzmann_poisson_finite_angle` or `_openmp` | CPU | Compatibility selector |
| Expanding prompt-cusp BP task | `venus_cpu_boltzmann_poisson_expanding_prompt_cusp` or `_openmp` | CPU | Expanding exact finite-angle task; finite-box caveat |
| Canonical orbit-phase BP | `venus_cpu_boltzmann_poisson_canonical_orbit` or `_openmp` | CPU | Orbit-atlas task; method-specific convergence required |
| Constant-cross-section SphDSMC SIDM | `venus_cpu_sidm_sphdsmc_constant` or `_openmp` | CPU | Implemented for the documented finite-time scope |
| Velocity-dependent SphDSMC SIDM | `venus_cpu_sidm_sphdsmc_velocity_dependent` or `_openmp` | CPU | Formula/kernel scope; convergence required per run |
| Low-compactness Einstein--Vlasov WPIC | `venus_cpu_einstein_vlasov_wpic_low_compactness` | CPU | Low-compactness completed example |
| High-compactness Einstein--Vlasov WPIC | `venus_cpu_einstein_vlasov_wpic_black_hole --coordinates schwarzschild\|eddington_finkelstein` | CPU | Separate polar--areal and horizon-penetrating advanced-null kernels; quantitative convergence required |
| Static Vlasov--Poisson WPIC | `venus_cpu_vlasov_poisson_wpic_static` | CPU | Isolated collisionless task; H=0 |
| Expanding Vlasov--Poisson WPIC | `venus_cpu_vlasov_poisson_wpic_expanding` | CPU | Implemented in the documented marker scope |
| Subhalo collisional WPIC | `venus_cpu_subhalo_wpic` | CPU | Implemented operator/cadence path |
| Constant-cross-section SIDM WPIC | `venus_cpu_sidm_wpic_constant` | CPU | Implemented with the documented locality caveat |
| Velocity-dependent SIDM WPIC | `venus_cpu_sidm_wpic_velocity_dependent` | CPU | Formula/kernel scope; convergence required per run |
| GPU static Vlasov--Poisson WPIC | `venus_gpu_vlasov_poisson_static` | GPU | Single-GPU static front over the VP kernel |
| GPU Vlasov--Poisson WPIC | `venus_gpu_vlasov_poisson` or `_mpi` | GPU | Implemented restart-based task |
| GPU subhalo BP | `venus_gpu_subhalo_boltzmann_poisson` or `_mpi` | GPU | Implemented restart-based task |
| GPU material SIDM BP, generic law | `venus_gpu_sidm_boltzmann_poisson` or `_mpi` | GPU | Static-halo restart task |
| GPU constant material SIDM | `venus_gpu_sidm_constant` or `_mpi` | GPU | Static-halo restart task; locality/convergence required |
| GPU power-law material SIDM | `venus_gpu_sidm_power_law` or `_mpi` | GPU | Static-halo restart task; locality/convergence required |
| GPU Yukawa--Born material SIDM | `venus_gpu_sidm_yukawa_born` or `_mpi` | GPU | Static-halo restart task; angular/convergence checks required |

The legacy CPU `venus_modular_solver*`, weighted-PIC runner names, and GPU
`run_wpic_subhalo_gpu*` remain compatibility interfaces. A new run should
use the task name above so that the executable itself enforces the input
contract.

## Task-main source map

Each preferred executable has its own task source and distinct main function.
Shared physics remains in `cpu/include/ev/` plus `cpu/src/drivers/` on CPU,
and in `gpu/include/venus_gpu/` plus `gpu/src/` on GPU. The canonical-orbit
task keeps its task-specific orbit-atlas engine beside its main; the other
CPU fronts are thin selectors over the shared grid or WPIC engines.

| Task main | Source file |
|---|---|
| CPU Eulerian EV | `cpu/src/apps/einstein_vlasov/eulerian.cpp` |
| CPU Eulerian VP | `cpu/src/apps/vlasov_poisson/eulerian.cpp` |
| CPU BP Fokker--Planck | `cpu/src/apps/boltzmann_poisson/fokker_planck.cpp` |
| CPU BP Landau | `cpu/src/apps/boltzmann_poisson/landau.cpp` |
| CPU BP finite-angle compatibility | `cpu/src/apps/boltzmann_poisson/finite_angle.cpp` |
| CPU BP exact | `cpu/src/apps/boltzmann_poisson/exact.cpp` |
| CPU BP reversible | `cpu/src/apps/boltzmann_poisson/reversible.cpp` |
| CPU BP sampled reversible | `cpu/src/apps/boltzmann_poisson/reversible_sampled.cpp` |
| CPU BP streamed reversible | `cpu/src/apps/boltzmann_poisson/reversible_streamed.cpp` |
| CPU expanding prompt cusp | `cpu/src/apps/boltzmann_poisson/expanding_prompt_cusp.cpp` |
| CPU canonical orbit-phase BP | `cpu/src/apps/boltzmann_poisson/canonical_orbit.cpp` |
| CPU constant SphDSMC | `cpu/src/apps/sidm/sphdsmc_constant.cpp` |
| CPU velocity-dependent SphDSMC | `cpu/src/apps/sidm/sphdsmc_velocity_dependent.cpp` |
| CPU low-compactness EV WPIC | `cpu/src/apps/einstein_vlasov/wpic_low_compactness.cpp` |
| CPU high-compactness EV WPIC | `cpu/src/apps/einstein_vlasov/wpic_black_hole.cpp` |
| CPU static VP WPIC | `cpu/src/apps/vlasov_poisson/static_wpic.cpp` |
| CPU expanding VP WPIC | `cpu/src/apps/vlasov_poisson/expanding_wpic.cpp` |
| CPU subhalo WPIC | `cpu/src/apps/subhalo/expanding_wpic.cpp` |
| CPU constant SIDM WPIC | `cpu/src/apps/sidm/wpic_constant.cpp` |
| CPU velocity-dependent SIDM WPIC | `cpu/src/apps/sidm/wpic_velocity_dependent.cpp` |
| GPU static VP | `gpu/tools/venus_gpu_vlasov_poisson_static.cu` |
| GPU VP | `gpu/tools/venus_gpu_vlasov_poisson.cu` |
| GPU subhalo BP | `gpu/tools/venus_gpu_subhalo_boltzmann_poisson.cu` |
| GPU generic material BP | `gpu/tools/venus_gpu_sidm_boltzmann_poisson.cu` |
| GPU constant material SIDM | `gpu/tools/venus_gpu_sidm_constant.cu` |
| GPU power-law material SIDM | `gpu/tools/venus_gpu_sidm_power_law.cu` |
| GPU Yukawa--Born material SIDM | `gpu/tools/venus_gpu_sidm_yukawa_born.cu` |

The CPU WPIC mains select shared engines in `cpu/src/drivers/` through
fail-closed profile contracts. The GPU mains call the common engine
in `gpu/src/wpic_production_driver.cu`. This organization separates tasks
without copying numerical algorithms.

## Relativistic material collision library

The C++17 library under `cpu/include/ev/relativistic_boltzmann/` provides
local elastic material-particle collisions and SC/advanced-EF slice adapters.
`eb::advance_sc` and `eb::advance_ef` connect the maintained Einstein--Vlasov
streaming steps to a constant isotropic material collision process and outgoing
Einstein-source reconstruction. A packet carries represented rest mass `W`
and particle number `W/m_chi`. The local amplitude interface supports supplied
reciprocal scalar interactions with a certified differential envelope.

This library has finite component validation. Its model-bound restart stores
a local collision cell. Continuous collision chronology, full coupled restart
and compact-halo convergence require further validation. The current
black-hole applications use collisionless Einstein--Vlasov evolution. See the
[module guide](../docs/relativistic_boltzmann.md) for the slice measures,
spatial-locality domain, API and compilation requirements.

# CPU modules

## 1. Eulerian Einstein--Vlasov

**Physics scope and exclusions.** This grid module evolves
a spherical phase-space density on an `(r,w,L)` grid in a dynamical
polar-areal spacetime. It is intended for resolved distribution-function and
metric studies. It is separate from the weighted-marker low/high programs and
requires its own resolution, constraint, boundary, and parallel checks.

**Equations.** With
`ds^2=-alpha^2 dt^2+a^2 dr^2+r^2 dOmega^2`, the distribution obeys
`df/dt=0` along relativistic characteristics, while its moments source the
spherical Hamiltonian and slicing constraints. The detailed characteristic,
moment, metric, and compactness equations are in the manual's
“Einstein--Vlasov Equations” section.

**Algorithm.** Read or construct `f(r,w,L)` and integrate its density,
current, and radial/tangential pressures. Those moments supply the spherical
constraints for `a`, `alpha`, and `m`. The solver chooses a timestep from
occupied characteristics, traces each arrival point backward, and interpolates
the earlier distribution with the prescribed radial and momentum boundaries.
It then recomputes moments and metric and writes diagnostics and a restart.

**Source and task lock.** The front ends are
`venus_cpu_einstein_vlasov_eulerian`, `_openmp`, and `_mpi`; shared implementation lives in
`cpu/include/ev/{moments,metric,characteristic_integrators,advection,timestep,
solver_evolution}.hpp`. The front end requires
`physics_model=einstein_vlasov` and rejects VP/BP or collision settings.

**Parameters and invalid combinations.** The applicable controls are phase
grid, relativistic metric/compactness stops, characteristic integrator,
boundaries, timestep, IC, output, restart, threads, and supported MPI layout.
BP collision operators, material/subhalo cross sections, and SphDSMC marker
controls are invalid here.

**IC/restart and output.** Analytic or `binary_rwl` data initialize the grid.
The module writes `metadata.json`, `history.csv`, `profiles.csv`, optional
phase-space snapshots, and versioned serial or rank-manifest restarts.

**Parallelism.** Serial/std::thread, OpenMP, and the supported MPI `L`-slab
decomposition are available. MPI ranks all-reduce radial matter fields.

**Visualization, example, and limitation.** Use
`cpu/tools/plot_einstein_vlasov_outputs.py`; the example INIs are
`cpu/examples/einstein_vlasov_*`. These are controlled starting problems,
not long-time convergence results.

**Required checks.** Quantitative use needs resolution, constraint, boundary,
restart, and parallel-equivalence studies, with an
apparent/trapped-surface diagnostic for claims beyond a compactness proxy.

## 2. Eulerian Vlasov--Poisson

**Physics scope and exclusions.** This grid module evolves a
deterministic spherical distribution in static or expanding Newtonian
gravity. It includes nodal pullback, explicit finite volume, conservative
SLDG, and Hamiltonian-aligned variants. It is not WPIC and does not represent
marker noise.

**Equations.** In static variables,
`partial_t f + w partial_r f + (L^2/r^3-partial_r Phi) partial_w f=0`,
`partial_r Phi=G M(<r)/r^2`, and `partial_r M=4 pi r^2 rho`. Expanding runs retain physical radius and store radial velocity relative
to the Hubble flow. The expansion rate enters both radial motion and the
peculiar-velocity equation, with the background and Poisson source convention
specified in the manual.

**Algorithm.** Integrate density from the grid, accumulate enclosed mass,
and reconstruct the potential. The selected nodal, finite-volume, or SLDG
method transports the distribution with the specified boundary rule and,
where implemented, a midpoint field. Carrier and flux ledgers record matter
transferred between representations or across faces. The solver then writes
profiles, history, phase snapshots, and restart state.

**Source and task lock.** Use `venus_cpu_vlasov_poisson_eulerian`, `_openmp`,
or `_mpi`. Shared code
is in `vlasov_poisson.hpp`, `conservative_sldg_transport.hpp`,
`hamiltonian_sldg_transport.hpp`, `vp_expanding_interpolation.hpp`, and
`delayed_inner_orbit_carrier.hpp`. The executable requires
`physics_model=vlasov_poisson` and rejects an active BP collision operator.

**Parameters and invalid combinations.** Applicable controls include universe
(`static`/expanding), Poisson source, gravity, integrator, SLDG geometry,
boundaries, orbit regularization, grid, IC, cadence, and parallel layout.
Relativistic metric controls and every BP/SIDM/subhalo collision control are
invalid. Unsupported combinations such as expanding Hamiltonian transport
with prescribed inflow remain fail-closed.

**IC/restart and output.** Analytic and versioned binary phase-grid ICs are
supported. Output follows the grid schema described above and includes flux or
SLDG ledgers when those algorithms are selected.

**Parallelism.** Serial, OpenMP, and collisionless MPI `L` slabs are
supported. The selected integrator still needs its own parallel-equivalence
study; MPI compilation alone is insufficient.

**Visualization, example, and limitation.** Use
`cpu/tools/spherical_profile_rendering.py` for named-column profile tables or
the Python `ev_solver.plotting` helpers for full grid output.
`examples/modular/vp_*` provides input-interface examples.
Integrator-specific equilibrium, boundary-flux, and measured-order
convergence remain the user's responsibility.

**Required checks.** Establish integrator-specific equilibrium,
boundary-flux, restart, MPI, and measured-order convergence for the selected
calculation.

## 3. Grid Boltzmann--Poisson: Fokker--Planck/Landau

**Physics scope and exclusions.** This grid module adds the small-angle
Fokker--Planck/Landau collision approximation to Eulerian VP. It is not a
finite-angle DSMC sampler and must not be interchanged with a subhalo
capture law.

**Equations.** The left side is the VP transport operator and the right side is
`C_FP[f]=-div_v(A f)+1/2 partial_vi partial_vj(D_ij f)`, reduced to the
spherical `(w,L)` or `(w,eta)` grid. Cross-section moments determine the drift
and diffusion coefficients.

**Algorithm.** Complete the Vlasov/Poisson transport stage, then construct
the local collision moments that determine velocity drift and diffusion.
Projecting these coefficients to velocity-space faces gives conservative
Chang--Cooper/upwind fluxes. The solver integrates those fluxes explicitly or
with RKL2 super-time-stepping, checks positivity and conservation, and records
the operator budgets.

**Source and task lock.** Use
`venus_cpu_boltzmann_poisson_fokker_planck` (or `_openmp`) for the explicit
Fokker--Planck name and `venus_cpu_boltzmann_poisson_landau` (or `_openmp`)
for the Landau-named task. Implementation is dispatched by
`boltzmann_poisson.hpp`. Each binary locks its accepted operator token and
rejects finite-angle, canonical, and SphDSMC choices.

**Parameters and invalid combinations.** FP bath, coefficient refresh,
collision flux/integrator, RKL2, cross-section, strength, cadence, grid, and VP
controls apply. Finite-angle quadrature, pair-majorant, SphDSMC population, and
subhalo capture controls do not.

**IC/restart and output.** It consumes the Eulerian phase-grid IC/restart and
writes grid outputs plus collision coefficient/flux and invariant ledgers.

**Parallelism.** Use one MPI rank with OpenMP/std::thread workers; distributed
collision coefficient and `L`-ghost exchange are not implemented.

**Visualization, example, and limitation.** Use
`render_density_slope_sigma2_gif.py`, `analyze_sidm_benchmark.py`, and the
BP example INIs. A new physical result needs relaxation, resolution,
positivity, conservation, and boundary studies over its full time interval.

**Required checks.** Compare a resolution hierarchy against analytic
relaxation and finite-angle reference solutions. Distributed collision
exchange is not implemented.

## 4. Grid Boltzmann--Poisson: finite-angle family

**Physics scope and exclusions.** This grid task covers exact
quadrature, reversible, sampled reversible, and streamed reversible Eulerian
finite-angle collision operators. These are alternative discretizations of
`C_B[f]`; they are not SphDSMC or weighted-marker pair updates.

**Equations.** `partial_t f+{f,H}=C_B[f]`, with
`C_B at v = int d^3v_* int dOmega u (dSigma/dOmega)
[f(v')f(v_*')-f(v)f(v_*)]`. Constant, softened-gravity, power-law, and
Yukawa--Born kernels have distinct total and differential cross sections.

**Algorithm.** (1) Perform the selected Vlasov transport stage; (2) assemble
radial collision cells and the requested angular quadrature/proposal; (3)
evaluate loss and gain or reversible increments; (4) subcycle/stream within
the majorant and positivity limits; (5) remap outgoing states; (6) close mass,
momentum, energy, omission, and sampling-error ledgers.

**Source and task lock.** Use the backend-specific primary binary:
`venus_cpu_boltzmann_poisson_exact`,
`venus_cpu_boltzmann_poisson_reversible`,
`venus_cpu_boltzmann_poisson_reversible_sampled`, or
`venus_cpu_boltzmann_poisson_reversible_streamed`, with `_openmp` for the
shared-memory build. `venus_cpu_boltzmann_poisson_finite_angle` and its
`_openmp` form remain compatibility selectors. The implementation is in
`finite_angle_boltzmann.hpp`, `reversible_finite_angle_boltzmann.hpp`,
`sampled_reversible_boltzmann.hpp`, and
`streamed_reversible_boltzmann.hpp`. Every primary executable locks exactly
one operator and rejects FP, SphDSMC, and every other finite-angle backend.

**Parameters and invalid combinations.** Differential kernel, total
cross-section law, pair/scatter quadrature, proposal/majorant, sampling
replicates, streaming workspace, collision cadence, and VP/grid controls
apply according to the parameter-applicability table in the detailed manual.
FP-only flux/RKL controls do not select this operator. Yukawa scattering
requires the Yukawa differential kernel, not an isotropic alias.

**IC/restart and output.** Eulerian IC/restart input is used. Outputs add
operator mass/energy budgets, collision work/omission diagnostics, and
backend-specific ledgers.

**Parallelism.** One MPI rank only; shared-memory reductions are supported
where the selected backend permits them.

**Visualization, example, and limitation.** The prompt-cusp figure tool is
`cpu/tools/spherical_profile_rendering.py`. Broad end-to-end equilibrium,
quadrature/proposal, restart, and target-physics convergence remains
incomplete.

**Required checks.** Each finite-angle method requires separate equilibrium,
quadrature/proposal convergence, restart, and long-time target-physics studies.

## 5. Expanding prompt-cusp Boltzmann--Poisson task

**Physics scope and exclusions.** This is a fail-closed convenience task for
the expanding finite-angle BP setup that evolves a `gamma_i` halo toward a
prompt-cusp-like inner profile and NFW-like outer transition. It does not turn
one finite-box trajectory into proof of an asymptotic NFW tail.

**Equations and algorithm.** It uses the expanding VP transport plus the
finite-angle Boltzmann integral described in modules 2 and 4, with the
registered operator cadence and current minimum collision quadrature.

**Source and task lock.** Use
`venus_cpu_boltzmann_poisson_expanding_prompt_cusp`. It requires expanding
Boltzmann--Poisson, the finite-angle prompt-cusp operator family, and a
nonzero collision strength; static, collisionless, FP, and SphDSMC inputs are
rejected.

**Parameters and invalid combinations.** `gamma_i`, expansion convention,
finite box/grid, finite-angle kernel/quadrature, strength, cadence, and output
controls apply. The finite-angle quadrature requires at least
`4 x 2 x 4` angular points.

**IC/restart, output, and parallelism.** The task consumes the documented
phase-grid IC/restart, writes grid BP products, and runs one MPI rank with
OpenMP workers.

**Visualization and domain.** Use `cpu/tools/spherical_profile_rendering.py`
for the emitted profiles. A finite radial box defines the resolved physical
domain; an asymptotic outer slope requires domain refinement.

**Required checks.** Use a grid/box/quadrature hierarchy before interpreting
the prompt cusp or outer transition quantitatively.

## 6. Constant-cross-section material SIDM with SphDSMC

**Physics scope and exclusions.** This is the persistent-marker, paper-exact
spherical DSMC path for constant isotropic material SIDM. The supplied example
covers a finite interval and does not reach a gravothermal catastrophe.

**Equations.** Markers sample material particles under spherical Poisson
gravity. For a local pair, the collision exposure is proportional to
`u Sigma_ij dt/V_shell`, with constant `sigma/m=sigma0`; elastic center-of-mass
scattering conserves pair mass, momentum, and kinetic energy.

**Algorithm.** Sort persistent markers by radius to compute enclosed mass
and gravity. Use that field to choose the orbit timestep, advance the markers,
and retry stiff orbits when required. Adaptive local shells then define the
NTC partner population and isotropic outgoing directions. Invariant and
mean-free-path checks precede profiles, history, and restart output.

**Source and task lock.** Use `venus_cpu_sidm_sphdsmc_constant`. Shared code is
`spherical_dsmc_solver.hpp` and `spherical_dsmc_collision.hpp`. The lock
requires BP + `spherical_dsmc` + constant/isotropic scattering and rejects
power-law, Yukawa, softened-gravity, grid, and WPIC settings.

**Parameters and invalid combinations.** Particle count/mass, constant
`sigma0`, adaptive shell occupancy, collision CFL, locality threshold,
paper-exact mode, gravity/orbit tolerance, boundaries, cadence, and restart
apply. Paper-exact mode deliberately rejects every nonconstant/differential
kernel.

**IC/restart and output.** It can sample a phase-grid DF or resume the
versioned persistent-particle restart. It writes `metadata.json`,
`profiles.csv`, `history.csv`, `operator_mass_budget.csv`, and restart state.
Current versions 6/7 retain the next collision timestep and a fingerprint of
the collision law, shell controls, timestep fraction, seed and collision mode;
version 7 also retains the paper-exact fractional remainders. Exact continuation
rejects changed collision controls. Versions 2–5 import their physical ensemble
and available remainders, reconstruct the first collision limit, and record
that legacy reconstruction in metadata.

**Parallelism.** Exactly one MPI task; use OpenMP threads inside the task.

**Visualization, example, and limitation.** Use
`render_frank_venus_sphdsmc_gif.py` and example case 06, whose configured
endpoint is `t=20`. Treat it as a finite-time example.

**Required checks.** A full core-collapse calculation requires marker/shell
convergence, locality throughout the interpreted core, and thread-count
equivalence.

## 7. Velocity-dependent material SIDM with SphDSMC

**Physics scope and exclusions.** This research task exposes power-law,
softened-gravity, and Yukawa--Born SphDSMC kernels without mislabeling them as
the constant/isotropic collision calculation.

**Equations.** The pair exposure uses `u Sigma_ij(u)` and, for Yukawa--Born,
the outgoing angle is drawn from the normalized differential cross section.
See the detailed SphDSMC and cross-section sections for the power-law and Born
formulae.

**Algorithm.** It follows module 6, replacing the constant majorant and
isotropic angle with a velocity-dependent majorant and the selected angular
sampler.

**Source and task lock.** Use
`venus_cpu_sidm_sphdsmc_velocity_dependent`; it requires a nonconstant law and
rejects paper-exact constant mode, grid operators, WPIC, and subhalo capture.

**Parameters/invalid combinations.** Select exactly one supported law and its
required scale/exponent or effective Yukawa amplitude/shape parameters. Their
ratio sets the speed scale, and the amplitude must include the physical
area-per-mass unit conversion derived in the manual. Yukawa total cross section
with an isotropic kernel is invalid.

**IC, output, and parallelism.** They match module 6: persistent-marker
restart, SphDSMC outputs, one MPI task, shared-memory workers.

**Visualization and limitation.** The SphDSMC renderer applies. No supplied
long-time example establishes core evolution for the velocity-dependent laws.

**Required checks.** Establish Maxwellian, rate, angular-distribution,
mean-free-path, resolution, and long-time core-collapse convergence separately
for each law.

## 8. Low-compactness Einstein--Vlasov WPIC

**Physics scope and exclusions.** This weighted-marker task compares
relativistic Einstein--Vlasov and Newtonian VP evolution from the same
low-compactness sample of positions and momenta, using the weights appropriate
to each phase-space measure. The paired evolution measures the Newtonian limit.

**Equations and algorithm.** Marker weights are invariant along the
relativistic characteristics. Each step deposits relativistic moments, solves
the constraints, pushes markers with a self-consistent field, and writes a
paired EV/VP comparison. See the weighted EV PIC manual section.

**Source and task lock.** Use
`venus_cpu_einstein_vlasov_wpic_low_compactness`; implementation is shared with
`weighted_einstein_vlasov_pic.hpp`. The task fixes the low scenario and
rejects high-scenario and collisional options.

**Parameters/invalid combinations.** Low marker count, radial cells, total
mass, velocity factor, importance exponent, CFL, end time, output cadence, and
threads apply. High-shell/horizon parameters and all collision controls are
invalid.

**IC/restart and output.** The analytic weighted realization and paired
rotating EV/VP restart are supported. Outputs include low-compactness history
and profiles for both equations.

**Parallelism.** Shared-memory threads; distributed MPI is not supported.

**Visualization, example, and limitation.** Use the low branch of
`render_weighted_einstein_vlasov_pic_profile_gifs.py` and example case 01.
The supplied resolution is a virialized-halo example; precision EV/VP density
agreement requires a marker/radial/time convergence hierarchy.

**Required checks.** Establish marker/radial/time convergence and verify the
chosen Newtonian-limit agreement target.

## 9. High-compactness Einstein--Vlasov WPIC

**Physics scope and exclusions.** This neutral, spherically symmetric task
follows a finite-support static energy-cutoff shell, Plummer sphere, or
infalling trial shell in either the original
Schwarzschild/polar--areal chart or the ingoing
Eddington--Finkelstein/advanced-null chart. Particle angular momentum is an
isotropic magnitude: the net spacetime angular momentum is zero, and a formed
black hole has a Schwarzschild rather than Kerr exterior. Charged matter,
Maxwell fields, and axisymmetric rotation are absent.

**Bound static-shell module.** The production threshold family uses
`weighted_einstein_vlasov_static_shell.hpp` and
`--high-initial-condition energy_cutoff_shell`. It solves
`f=C(1-E/E0)_+ delta(ell-ell0)`, including the
`2 ell0 delta(ell^2-ell0^2)` Jacobian, all continuum sources,
`m'=4 pi r^2 rho`, the static lapse equation, rest mass, exact inner/outer
vacua, asymptotic lapse matching, positive binding, and scale covariance. Its
shifted rank-one sampler emits exactly the requested even marker count as
adjacent `(+w,-w)` invariant-weight pairs with unique persistent IDs and
source IDs. Those weights already contain `exp(lambda)` and must not be
canonicalized again. A zero kick at `s_L=1` is the equilibrium regression;
the threshold family applies one fixed inward kick and varies only `s_L`
before fixed-ADM normalization.

**Separate coordinate modules.** The compatibility include
`include/ev/weighted_einstein_vlasov_pic.hpp` exposes, but does not merge, two
kernels. `weighted_einstein_vlasov_pic_schwarzschild.hpp` stores the local
orthonormal radial momentum `w`, solves the polar--areal constraints, and must
stop while `1-2m/r > 0`. The independent
`weighted_einstein_vlasov_pic_eddington_finkelstein.hpp` stores the canonical
future-directed momentum `P=p_r>0` for
`ds^2=-a b^2 dv^2+2b dv dr+r^2 dOmega^2`, solves the neutral advanced-null
constraints, locates an outer apparent horizon at `a=1-2m/r=0`, and can
continue after causally excising a sufficiently trapped interior. Its pusher
advances `log(P)`, so every accepted Runge--Kutta stage remains on `P>0`.

**Source and task lock.** Use `venus_cpu_einstein_vlasov_wpic_black_hole` and
select exactly one kernel with `--coordinates schwarzschild` or
`--coordinates eddington_finkelstein`. The task locks the high scenario and
rejects low-comparison and collision options. EF-only excision controls are
`--ef-excision-a` and `--ef-excision-buffer-cells`.

**Parameters/invalid combinations.** High IC family, mass, shell/Plummer
geometry, inward momentum and dispersions, radial grid/boundary, resolution,
cadence, and threads apply. The static shell adds
`--high-energy-cutoff-amplitude`, `--high-energy-cutoff-ell0`,
`--high-energy-cutoff-central-y`, an even `--high-markers`, and
`--high-sampling-realization`; its fixed kick uses
`--high-radial-infall` and optional `--high-infall-core-radius`.
`--high-specific-angular-momentum` is rejected for this authenticated
delta-`ell0` family. `--high-angular-momentum-scale` rescales its realized
`eta=L^2/2` while preserving the base marker realization.
`--high-target-initial-adm-mass` then rescales all invariant marker rest
weights by one common positive factor, found by a bracketed polar-constraint
solve, so an angular scan does not also vary its initial gravitating mass. The
advanced-null chart has no implemented reflection through `r=0`, so an EF
run whose active characteristic reaches the origin fails closed rather than
inventing an outgoing branch.

**IC, coordinate map, and slice meaning.** `--initialize-only --physical-ic`
writes one immutable coordinate-neutral base realization. It may seed the SC
donor, but it is not by itself valid null initial data. A fresh EF run requires
`--ef-null-initial-data` created by an SC run with
`--export-ef-null-initial-data`. The exporter uses an independent East-style
`s=-t` donor copy, stores every accepted field, solves the ingoing eikonal and
integrating factor, and intersects every persistent marker worldline exactly
once. The physical incoming SC run itself is not reversed. Outside a horizon,
the same-event marker variables obey `P=e^lambda(w+epsilon)>0`; the dynamical
metric map is `a=e^-2lambda`, `b=e^(lambda+mu)/A` with `A=partial_t v>0`.
The manifest-hashed artifact preserves marker rest weights, IDs and `eta` and
records distinct coordinate-specific realized fingerprints. Raw equal marker
records on `t=constant` and `v=constant` are not the same hypersurface.
`--ef-map-worldtube-radius R_B` may place the exact-face vacuum normalization
inside the full `--high-r-max R_E` domain. For `R_B<R_E`, only the initial
null slice is extended analytically with `m=M_null`, `a=1-2M_null/r`,
`A=F=b=1`, zero stress, and the exact tortoise-time difference. Four-cell
matter/crossing clearance, vacuum sources, `b=1`, positive `a`, extension
residuals, extension index, prefix/full-grid fingerprints, and all hashes are
authenticated. The subsequent EF evolution uses the entire `R_E` mesh.

**Restart, output, and classification.** Each coordinate path has a versioned
restart and coordinate-specific history/profile schema. Schwarzschild output
records the compactness gap and its pre-horizon stop proxy. EF output records
`a`, `b`, apparent-horizon time/radius, trapped-region/excision bookkeeping,
fixed ADM mass, and post-horizon duration. `--expected-outcome any` makes a
scientifically legitimate halo endpoint distinct from a crashed scheduler
job.
Both production coordinate paths require a vacuum outer domain. A trial that
would carry positive rest weight beyond it is rejected before commit and the
last accepted state is saved. The Schwarzschild high-compactness outcome is
`sc_outer_boundary_evolution_limit`, with accepted time and rejected trial
duration under `schwarzschild_evolution_status`; it satisfies no completed
black-hole or halo classification. Library outflow bookkeeping remains
available to callers supplying a consistent exterior matter/lapse model.

**Parallel execution.** Both kernels use one process with shared-memory
threads; distributed MPI is unsupported. The Slurm parity helper executes a
two-resolution by two-independent-PIC-realization SC-donor/EF-consumer DAG,
builds authenticated convergence evidence, and issues a strict same-event
certificate. The angular scan is blocked until that certificate passes, holds
the initial ADM mass fixed, and reports target- and measured-ADM
normalizations before attempting an outcome bracket.

The scan is scientifically released only after local preflight identifies a
positively bound, sustained high-`L` halo endpoint and a low-`L` EF black-hole
endpoint from the same static shell and fixed kick. A high-`L` escape is
dispersal, not the halo side of the requested threshold. The continuum/kick,
target mass, map/evolution radii, donor duration, end time, and angular bracket
follow from the chosen static shell and its preflight checks.

**Visualization and limitations.** Use
`render_weighted_einstein_vlasov_coordinate_pair_gif.py` for the authenticated
three-panel SC/EF overlay and the high branch of
`render_weighted_einstein_vlasov_pic_profile_gifs.py` for a single-coordinate
sequence. The paired renderer shows same-event curves before the SC chart
limit and clearly labelled EF-only frames afterward; it plots
`log10(rho)` versus `log10(r)`, logarithmic density slope versus `log10(r)`,
and `2m/r` versus linear `r`. Coordinate equivalence means equality at mapped events: the
continuum scalar distribution with its momentum Jacobian, the finite-marker
empirical measure with unique IDs/invariant weights, all covariant stress
moments, and the metric/inverse metric after tensor pullback, together with
forward and inverse eikonal closure. Compactness milestones select useful slices but do
not replace the event map. Fixed outer ADM mass is valid only while all
dynamically relevant, non-excised matter remains within the radial domain.
East's charged-shell threshold is not a neutral prediction.

**Required checks.** Establish marker, radial, timestep, domain, excision, and
thread-count convergence; verify both radial constraints and the vacuum
continuation; reproduce a static transformed solution; compare pre-horizon
invariants between coordinates; and repeat the angular-momentum bracket at
increased resolution before interpreting it as a physical threshold.

## 10. Static and expanding Vlasov--Poisson WPIC

**Physics scope and exclusions.** These tasks advance weighted particles in static or prescribed expanding
Newtonian backgrounds with a self-consistent gravitational field and zero
explicit encounter strength. It is the explicit
collisionless counterpart of the subhalo runner.

**Equations.** Marker characteristics discretize the expanding VP equations;
the field uses spherical deposition, enclosed mass, and the documented
analytic scale-factor drift.

**Algorithm.** Generate or import markers, deposit their mass and moments,
and solve Poisson's equation. The resulting field supplies the KDK push with
expanding drift. After applying boundaries, write profiles, history, and
restart state, then repeat the collisionless update.

**Source and task lock.** Use `venus_cpu_vlasov_poisson_wpic_static` for an
isolated system. It fixes the expansion rate to zero and records the total
density as the Poisson source. Use `venus_cpu_vlasov_poisson_wpic_expanding`
for the prescribed expanding background. Both use the same numerical routines.
Shared code is `weighted_vlasov_pic.hpp`; the executable fixes collision
strength to zero and rejects every SIDM/subhalo collision option.

**Parameters/invalid combinations.** `gamma_i`, marker/radial resolution,
expansion/background, timestep, boundaries, deposition shape, output/restart,
and threads apply. Positive material or subhalo collision strength is invalid.

**IC/restart and output.** Analytic power-law sampling or the versioned WPIC
restart is used. Outputs are `profile_NNNNN.csv`, history, metadata,
`radial_mesh_contract.txt`, and `restart_state.bin`; no collision ledger is
required.

**Parallelism.** CPU shared-memory only for this runner.

**Visualization, example, and limitation.** Use
`render_weighted_pic_pure_gamma_gifs.py` or
`render_weighted_vlasov_pic_equilibrium_gif.py`; the supplied expanding
controls are starting points within their stated marker and mesh resolution.

**Required checks.** Establish marker-number, cloud-order, background,
boundary, and long-time equilibrium convergence for the chosen observable.

## 11. Subhalo collisional WPIC

**Physics scope and exclusions.** This operator/cadence path treats
each marker as a complete extended Plummer object. It models mutually
exclusive scattering or capture with unequal masses and evolving internal
energy. It is not microscopic material SIDM.

**Equations.** The encounter rate uses the calibrated velocity- and
mass-dependent Plummer scatter/capture cross sections. Elastic outcomes use
the center-of-mass map; capture replaces two objects by one while conserving
mass and momentum and transferring reduced-mass orbital energy into internal
energy/compaction.

**Algorithm.** Advance expanding WPIC to the earliest field or collision
deadline. Radial cells define local partners, and exact pair majorants set
upper trial rates. Deterministic counter-RNG trials in collision lanes select and accept pair
encounters. The physical encounter law determines scattering or capture.
Updating or merging the markers changes
the source, so the solver redeposits and refreshes the field when required.
Mass, pair-invariant, locality, and cadence checks precede a restart containing
the clocks and counters.

**Source and task lock.** Use `venus_cpu_subhalo_wpic`; implementation is in
`weighted_vlasov_pic_subhalo_collisions.hpp` and the weighted-PIC event code.
The lock requires positive subhalo strength and rejects material SIDM and grid
collision operators.

**Parameters/invalid combinations.** Subhalo strength, calibrated velocity
scale/table convention, Plummer area per mass, low-speed regularization,
collision interval/fraction/CFL/substeps, lanes, locality, seed, marker/grid,
cadence, and the optional per-cell adaptive collision clock apply.
`sidm_sigma0`, Yukawa parameters, and SphDSMC settings are invalid.

**IC/restart and output.** Analytic power-law markers or the complete WPIC
restart initialize masses, internal dispersions/energies, IDs, clocks, and RNG
state. Outputs add `collision_history.csv` and terminal locality/cadence
status records to the WPIC schema.

**Parallelism.** Shared-memory CPU execution assigns each adaptive collision
cell to one worker for its complete interval. Cell membership and local random
counters preserve the event sequence across thread counts. Diagnostics reduce
in cell order after all cells finish. A failed call leaves partial in-memory
state that the caller must discard; the driver propagates the failure before
updating event clocks or writing a new checkpoint.

**Visualization, example, and limitation.** Use
`render_weighted_pic_subhalo_{comparison,physical_time,collision_age,
strength_scan}_gif.py` and example case 04. The example is configured to
`t=50`; population-level conclusions require ensemble and resolution studies.

**Required checks.** Establish ensemble, resolution, collision-cell,
long-time population, and capture-compaction convergence.

## 12. Constant-cross-section material SIDM WPIC

**Physics scope and exclusions.** This task uses weighted markers to sample microscopic
material SIDM with constant `sigma/m`. The packaged deep-collapse example
has an explicitly documented late nonlocal-shell warning; post-warning
structure is not locality-resolved science.

**Equations.** Pair exposure in an adaptive shell is proportional to
`u (m_i+m_j) (sigma/m)/2`; isotropic elastic center-of-mass scattering
preserves represented mass, radial momentum, and kinetic energy.

**Algorithm.** Import or generate weighted markers, deposit their matter,
and advance KDK gravity to the scheduled potential or collision event.
Adaptive shells supply local NTC pairs, which scatter isotropically.
Reconstruction refreshes the field when required. The collision rate,
invariants, effective marker count, and `Delta r/lambda` are checked before
checkpointing.

**Source and task lock.** Use `venus_cpu_sidm_wpic_constant`; code is in
`weighted_vlasov_pic_collisions.hpp`, event scheduling, phase remap, and Frank
restart import. The lock requires constant material SIDM and rejects subhalo,
power-law, Yukawa, SphDSMC, and grid modes.

**Parameters/invalid combinations.** Constant `sigma0`, collision strength,
shell occupancy, collision CFL/substeps, locality policy, marker/grid/cloud,
event cadence, restart, threads, `--collision-adaptive-shell-timestep`, and its
outer maximum interval apply. The local-shell path requires constant cross
section, equal masses within each shell, and collision CFL no larger than one.
Velocity scale/exponent and Yukawa parameters are invalid.

**IC/restart and output.** Frank/particle or WPIC restart input is supported.
Outputs include WPIC profiles/history, collision history and timing, phase
projection where requested, metadata, and atomic restart.

**Parallelism.** One process with shared-memory workers.

**Visualization, example, and limitation.** Use
`render_frank_wpic_gif.py`, `render_frank_wpic_density_slope_gif.py`, or
`render_frank_wpic_uniform_time_gif.py`. Example case 05 targets deep core
contraction and permits nonlocal shells after its recorded warning.

**Required checks.** Quantitative post-warning science requires finer
marker/shell resolution and locality-converged reruns.

## 13. Velocity-dependent material SIDM WPIC

**Physics scope and exclusions.** This research task locks material SIDM to power-law or
Yukawa--Born velocity dependence. No supplied long-time example establishes
core evolution for either law.

**Equations.** Power-law scattering replaces the constant law by the
documented softened `sigma(u)/m`; Yukawa--Born uses its velocity-dependent
total cross section and forward-peaked differential angular CDF.

**Algorithm.** It follows module 12, recomputing the speed-dependent pair
majorant and drawing the law-specific outgoing angle at each collision event.
The CPU velocity-dependent path retains the mesh-wide reference collision
clock; its shell-local optimization is not enabled in this release.

**Source and task lock.** Use
`venus_cpu_sidm_wpic_velocity_dependent`; it requires `powerlaw` or
`yukawa_born` and rejects constant, subhalo, SphDSMC, and grid modes.

**Parameters/invalid combinations.** For power law, specify normalization,
speed scale, exponent, and regularization. For Yukawa, specify the effective
amplitude and shape parameters after converting the physical law to simulation
units, and select the Yukawa differential kernel. A
Yukawa total law with isotropic angular scattering is invalid.

**IC, output, parallelism, and visualization.** These match module 12; use the
Frank WPIC renderers and SIDM analyzers, with one shared-memory process.

**Limitation.** Local formula and pair-invariant agreement does not establish
long-time relaxation or converged core evolution.

**Required checks.** Establish constant-matched relaxation-time, Maxwellian,
core-evolution, locality, marker-number, and ensemble comparisons for each law.

## 14. Canonical orbit-phase Boltzmann--Poisson

**Scope and exclusions.** This CPU task evolves a distribution on a bound-orbit
atlas in canonical phase and couples transport to local finite-angle collision
events and a self-consistent Poisson field. It is not the Eulerian
`(r,w,L)` solver, SphDSMC, or the weighted-marker WPIC event library.

**Equations and algorithm.** The canonical method represents a distribution
on a bound-orbit atlas in phase, lifts finite-angle collision gain/loss into
orbit cells/open packets, compresses positive packets, and rebuilds the
Poisson field at events. One update (1) transports the atlas in canonical
phase; (2) selects local NTC collision pairs; (3) maps outgoing states back to
bound atlas cells or open packets; (4) closes mass, momentum, and energy
ledgers; (5) compresses positive packets; (6) reconstructs radial moments and
the field; and (7) advances event clocks and writes output/restart state.

**Source and task lock.** Use
`venus_cpu_boltzmann_poisson_canonical_orbit`; its main is
`cpu/src/apps/boltzmann_poisson/canonical_orbit.cpp`. Shared
implementation is in the `canonical_*` headers. The task uses its
orbit-atlas command-line schema and rejects substitution of a modular INI or a
WPIC/SphDSMC task.

**Parameters and invalid combinations.** Controls cover the mandatory source
Eulerian grid restart and stationary Eddington table, output and optional
canonical restart, atlas
`(eta,H,tau)` resolution, radial projection grid, transport/collision
times, collision coupling and marker sampling, event timestep and physical
clocks, locality/CFL limits, open-packet compression, reversible sampled or
streamed operators, output/restart cadence, and threads. Run `--help`
for the complete interface. Eulerian grid operators, WPIC collision options,
and MPI decomposition do not apply.

**IC/restart, output, parallelism, and visualization.** A versioned Eulerian
grid restart and a stationary Eddington table always construct the source
atlas. An optional canonical restart restores the evolved atlas and event
state; it replaces neither source input. Output includes `history.csv`, `profiles.csv`,
`operator_mass_budget.csv`, and atomic
`restart_state.bin`. The task uses one shared-memory process. Use
schema-compatible profile, density/slope, and invariant tools.

**Required checks.** Establish atlas/phase/radial resolution, event-timestep,
collision-marker and shell convergence, open-packet and boundary closure,
restart equivalence, and thread-count equivalence. The integrated
`weighted_pic_boltzmann_event.hpp` library remains an internal
algorithm/API rather than a separate public solver task.

## 15. Python Einstein--Vlasov reference package

`cpu/python/ev_solver/` is an importable, readable reference implementation
and output-analysis library. It separates configuration, grids, ICs, moments,
metric, numerics, state, setup, evolution, I/O, and plotting, but it is not an
alternative high-performance backend. Use a C++ task with the same initial distribution, grids, boundaries, and
timesteps to assess the reference calculation's field and diagnostic errors.

# GPU modules

The GPU code is a distinct CUDA implementation. It shares restart contracts
and selected CPU reference formulae; it does not call the CPU evolution loop.
Initial particle states are generated on the host and imported through the
shared restart format, allowing CPU and GPU comparisons to start from the
same sampled population.

## 16. GPU Vlasov--Poisson WPIC

**Physics scope and exclusions.** This is the restart-based GPU
collisionless marker path. It is not Einstein--Vlasov, grid VP, canonical
orbit, or SphDSMC.

**Equations.** It discretizes spherical VP with CIC/TSC deposition,
`dPhi/dr=G M(<r)/r^2`, and static/expanding KDK characteristics.

**Algorithm.** (1) Read and upload a WPIC restart; (2) deposit spherical
moments on the device; (3) scan enclosed mass and reconstruct force/potential;
(4) fixed-step KDK push, including analytic expanding drift, with shortening
only at output/final time; (5) apply boundaries;
(6) output/download only at requested profiles/checkpoints; (7) repeat without
collision scheduling.

**Source and task lock.** Use `venus_gpu_vlasov_poisson` or
`venus_gpu_vlasov_poisson_mpi`. Code is in `wpic_state.cu`,
`wpic_vlasov_poisson.cu`, and `wpic_mpi.cpp`. The task fixes collision strength
to zero and rejects material/subhalo collision options and unsupported restart
features such as delayed markers.

**Parameters/invalid combinations.** Restart, output, endpoint, fixed maximum
timestep, radial mesh, cadence, and supported boundary/background state apply.
The dynamical fraction is parsed for the shared interface but does not vary a
collisionless KDK step. Single-GPU VP accepts `--universe static` for total-density
masses and `--universe eds` for density-contrast masses. EdS remains the default,
and MPI VP retains EdS. Every collision-law parameter is invalid.

**IC/restart and output.** It consumes a versioned CPU/GPU WPIC restart and
writes `profile_NNNNN.csv`, `history_gpu.csv`, metadata, radial mesh contract,
terminal status, and portable restart.

**Parallelism.** Single GPU or one MPI rank per GPU. The MPI path
all-reduces radial moments and writes a gathered ID-sorted checkpoint.

**Visualization, example, and limitation.** Use the common WPIC renderers or
`gpu/scripts/render_venus_gpu_density_slope_gif.py` for its documented
run-directory organization. The supplied restart example illustrates CPU/GPU
handoff; long-time agreement still requires profile and convergence studies.

**Limitations.** `benchmark_vlasov_poisson_gpu` is the focused performance
executable. Delayed-inner and wider boundary/background modes are absent.

## 17. GPU subhalo Boltzmann--Poisson

**Physics scope and exclusions.** This CUDA task advances WPIC gravity
plus calibrated subhalo scatter/capture. It is the GPU counterpart of
CPU module 11, not material SIDM.

**Equations and algorithm.** The same calibrated pair areas, COM scatter,
capture energy, compaction, event clocks, locality, and counter-RNG semantics
as CPU module 11 are implemented in CUDA. Device prefix/majorant arrays remain
cached until capture changes their inputs; MPI owns physical radial cells and
migrates markers after capture. Reference mode uses one mesh-wide collision
step; `--subhalo-adaptive-cell-timestep` gives each cell its own CFL clock and
the GPU batches several local steps per launch, stopping early after capture.

**Source and task lock.** Use `venus_gpu_subhalo_boltzmann_poisson` or `_mpi`.
Implementation is in `wpic_subhalo_collisions.cu`, VP/state/restart code, and
the MPI layer. The task requires positive subhalo strength and rejects
material SIDM and collisionless-only invocation.

**Parameters/invalid combinations.** The subhalo controls from CPU module 11
plus `--subhalo-adaptive-cell-timestep`, its kernel batch size, and GPU
restart/output/device/MPI controls apply. Material `sigma/m` and Yukawa
parameters are invalid.

**IC/restart and output.** A CPU/GPU versioned WPIC checkpoint is mandatory.
Outputs add `collision_history_gpu.csv` and completion, walltime, locality, or
cadence status records.

**Parallelism.** One GPU or one rank per GPU. Fixed-clock MPI globally reduces
the collision-frequency maximum; adaptive MPI advances uniquely owned cells
locally inside the common outer batch. Migration and gathered restart define
the remaining supported MPI scope.

**Visualization, example, and limitation.** The GPU example continues the
`gamma_i=0.5` CPU restart. Use the subhalo comparison/strength renderers or
`gpu/scripts/render_venus_gpu_density_slope_gif.py`. Short restart agreement
does not imply indefinite bitwise identity of chaotic trajectories.

**Required checks.** Quantitative use requires ensembles, rank-count scaling,
and long-time statistical convergence across the selected parameter space.

## 18. GPU material SIDM Boltzmann--Poisson dispatcher

**Physics scope and exclusions.** This frontend advances static-halo GPU WPIC plus one
material SIDM law. The material scheduler is not rank-count invariant.

**Equations.** The collision exposure is the material Boltzmann rate
`u Sigma_ij(u)` with elastic COM scattering; the three laws are detailed in
modules 19--21.

**Algorithm.** (1) Read/upload restart; (2) VP KDK to the selected collision
deadline; (3) radius-bin markers into fixed-occupancy local shells; (4) form
the material collision majorant; (5) draw/accept pairs and outgoing angles;
(6) update device markers; (7) audit invariants and write collision history;
(8) checkpoint clocks/RNG state. Reference mode uses one mesh-wide collision
step. `--sidm-adaptive-shell-timestep` gives every fixed-occupancy shell an
independent clock and refreshes its velocity majorant after each local step.
The disjoint partner assignments give every unordered pair equal inclusion
probability `1/m`, with `m=N-1` for even shell population `N` and `m=N` for
odd `N`. Acceptance and its CFL bound use `m/V`; the physical mean-free-path
diagnostic retains `(N-1)/V`. At most one collision per marker per local step
makes collision-CFL convergence a required check.

**Source and task lock.** Use `venus_gpu_sidm_boltzmann_poisson` or `_mpi` and
select one explicit material law. Shared implementation is in
`wpic_collisions.cu`. Subhalo capture and collisionless mode are rejected.

**Parameters/invalid combinations.** Restart/output/mesh/cadence, material
law and its parameters, shell occupancy, collision CFL/interval, optional
adaptive shell clock, RNG, and device/MPI controls apply. Subhalo
calibration/compaction controls do not.

**IC/restart and output.** The GPU remains restart-based. It writes the common
GPU WPIC products plus a material collision ledger and terminal status.

**Parallelism.** Single-GPU execution is supported. `_mpi` names select the
one-rank-per-GPU build, but generic material scheduling is not rank-count
invariant; multi-GPU results require statistical and ownership checks.

**Visualization and limitation.** Use the generic density/slope/dispersion
GIF and SIDM analyzer. `benchmark_material_sidm_gpu` is the focused
performance program. No packaged long-time GPU material-SIDM example exists.

**Required checks.** Establish restart equivalence, a constant-law CPU/GPU
statistical trajectory comparison, and separate multi-GPU ownership/rank-count
studies.

## 19. GPU constant material SIDM

**Physics scope and exclusions.** `venus_gpu_sidm_constant` and `_mpi` lock module 18 to
constant `sigma/m` and isotropic scattering. Quantitative trajectories require
the locality and convergence checks below.

**Equation/algorithm.** `sigma(u)/m=sigma0`; pair exposure and elastic COM
scattering follow modules 12 and 18.

**Parameters and invalid combinations.** `sigma0`, strength, shell/CFL/event
controls, restart/output, and GPU layout apply. Power-law scales/exponents,
Yukawa masses/coupling, and subhalo options are rejected.

**I/O, plotting, and required checks.** GPU WPIC restart/output and the SIDM
analyzer/generic renderer apply. Establish matched constant-SIDM trajectories
and locality, marker-number, timestep, seed, and rank-count convergence.

## 20. GPU power-law material SIDM

**Physics scope and exclusions.** `venus_gpu_sidm_power_law` and `_mpi` lock module 18 to the
softened power-law material cross section. Quantitative trajectories require
the comparisons and convergence checks below.

**Equation/algorithm.** The documented `sigma(u)/m` power law replaces
`sigma0` in the local pair majorant; outgoing directions remain isotropic for
this law.

**Parameters and invalid combinations.** Normalization, reference speed,
power/softening, strength, shell/event, restart/output, and GPU layout apply.
Yukawa differential controls, constant-only assumptions, and subhalo options
are rejected.

**I/O, plotting, and required checks.** Module 18 contracts apply. Establish a
relaxation-time-matched constant/power-law trajectory ensemble together with
locality, marker-number, timestep, seed, and rank-count convergence.

## 21. GPU Yukawa--Born material SIDM

**Physics scope and exclusions.** `venus_gpu_sidm_yukawa_born` and `_mpi` lock module 18 to
the velocity-dependent, forward-peaked Yukawa--Born kernel. The total cross
section and inverse angular CDF are implemented; quantitative trajectories
require the checks below.

**Equation/algorithm.** The Born total cross section determines the rate and
the normalized differential cross section determines the outgoing angle; see
the GPU collision-equation section for the explicit expressions and stable
inverse CDF.

**Parameters and invalid combinations.** Coupling, dark-particle mass,
mediator mass, strength, shell/event, restart/output, and GPU layout apply. An
isotropic angular alias, power-law exponent, and subhalo controls are invalid.

**I/O, plotting, and required checks.** Module 18 contracts apply. Establish
forward-scattering resolution, rate, CPU/GPU statistical, locality, and
core-evolution convergence studies.

## 22. GPU compatibility dispatcher, MPI, restart, and benchmarks

`run_wpic_subhalo_gpu` and `_mpi` remain compatibility auto-mode names for old
launchers. They must report the resolved task and fail on ambiguous mixtures;
new commands should use modules 16--21. `wpic_restart.cpp` owns the portable
checkpoint; `wpic_mpi.cpp` owns one-rank-per-GPU synchronization. The focused
performance programs are `benchmark_vlasov_poisson_gpu` and
`benchmark_material_sidm_gpu`; `benchmark_wpic` remains the compatibility
aggregate. Benchmarks measure speed, not physical convergence.

## Backend exclusions

GPU Einstein--Vlasov, Eulerian VP/BP, canonical-orbit BP, SphDSMC, delayed
inner return, and material Plummer turning-point quadrature are absent from
VENUS. Select the corresponding CPU module rather than approximating them
by a similarly named GPU option.
