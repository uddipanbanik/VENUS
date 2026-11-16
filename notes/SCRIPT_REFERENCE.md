# VENUS script and tool reference

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur

Use this catalogue to find the program for a particular operation: creating
initial data, running a solver, reading a restart, calculating diagnostics, or
rendering saved output. Each entry explains its inputs and scientific purpose.
The source comments follow important calculations and branches in code order.
Command-line programs list their options through `--help`; specialized
no-argument diagnostics state their fixed inputs and outputs in source.

## CPU task mains and specialist runners

The preferred one-task/one-main sources are organized under `cpu/src/apps/`:

- `einstein_vlasov/eulerian.cpp`, `wpic_low_compactness.cpp`, and
  `wpic_black_hole.cpp`;
- `vlasov_poisson/eulerian.cpp` and `expanding_wpic.cpp`;
- `subhalo/expanding_wpic.cpp`;
- `sidm/sphdsmc_constant.cpp`, `sphdsmc_velocity_dependent.cpp`,
  `wpic_constant.cpp`, and `wpic_velocity_dependent.cpp`; and
- `boltzmann_poisson/fokker_planck.cpp`, `landau.cpp`,
  `finite_angle.cpp`, `exact.cpp`, `reversible.cpp`,
  `reversible_sampled.cpp`, `reversible_streamed.cpp`, and
  `expanding_prompt_cusp.cpp`, plus `canonical_orbit.cpp`.

`canonical_orbit.cpp` is the command-line orbit-atlas
Boltzmann--Poisson task. It always reads a versioned Eulerian grid restart and
a stationary Eddington table; an optional canonical restart restores the
evolved atlas and event state but replaces neither required source input. It
advances canonical transport/collision events and writes profiles, invariant
ledgers, and atomic restarts.

The WPIC task mains call shared engines in
`cpu/src/drivers/einstein_vlasov_wpic.cpp`, `expanding_wpic.cpp`, and
`sidm_wpic.cpp`. Grid and collision kernels remain under `cpu/include/ev/`.

- `run_weighted_vlasov_pic_expanding_powerlaw.cpp` is the compatibility driver
  for expanding VP and subhalo WPIC; new runs should select the locked
  `venus_cpu_vlasov_poisson_wpic_expanding` or `venus_cpu_subhalo_wpic` task.
- `run_weighted_vlasov_pic_frank_collisional.cpp` is the compatibility static
  Frank-core material-SIDM WPIC driver; new runs should select
  `venus_cpu_sidm_wpic_constant` or
  `venus_cpu_sidm_wpic_velocity_dependent`.

These are command-line applications, while `cpu/src/venus_modular_solver.cpp` is the INI-driven modular entry point and `cpu/src/venus_einstein_vlasov.cpp` is the compatible stand-alone EV entry point.

## Relativistic collision library

The header library in `cpu/include/ev/relativistic_boltzmann` couples SC/EF
transport to local elastic collisions. Its [library guide](../docs/relativistic_boltzmann.md)
defines the state, coordinate rates, finite-cell domain and compilation
requirements. Component validation covers local rates, conservation, cell
restart, collision-off controls and self-consistent source-feedback steps.

## CPU Python reference implementation

The importable `cpu/python/ev_solver/` package presents the Einstein--Vlasov
calculation in a compact form suitable for reading, small comparisons, and
analysis. Its modules follow the calculation from parameters to output:

- `__init__.py` defines the public package surface.
- `params.py` owns the Python parameter dataclass with range and consistency
  checks, and `config.py`
  maps INI/CLI input into it.
- `grids.py`, `moments.py`, and `metric.py` construct the phase-space grids,
  quadrature, matter moments, and spherical metric constraints.
- `initial_data.py` reads or constructs distributions, while
  `ic_generator.py` writes versioned `EVICRWL1` binary initial data.
- `numerics.py` contains characteristic pullback, timestep, boundary, and
  interpolation kernels; `physics.py` is their compatibility umbrella.
- `simulation_state.py`, `solver_setup.py`, `solver_evolution.py`, and
  `solver.py` separate mutable state, initialization, the timestep loop, and
  the public solve API.
- `io.py` writes Python-reference outputs/restarts; `output_reader.py` reads
  header-named output safely; and `plotting.py` supplies headless static/GIF
  rendering helpers.

Read the parameter and grid modules first, then follow moment integration,
metric reconstruction, characteristic transport, and output. The documentation
check verifies module syntax and documentation strings.

## CPU IC, restart, and conversion utilities

- `cpu/tools/generate_bound_gaussian_shell_restart.py` samples a bound radial Gaussian density shell and Gaussian velocities, checks its discrete spherical binding energies, and writes static/EdS WPIC-v6 initial states with identical physical particles and mesh sidecars. The student example is `cpu/examples/modular/vp_bound_gaussian_shell_gpu.md`.
- `generate_frank_core_cell_average_restart.py` constructs the gridded cell-average Frank equilibrium used by grid/canonical tests.
- `generate_frank_core_spherical_dsmc_restart.py` samples a persistent-marker SphDSMC initial population with masses and collision state.
- `generate_frank_nfwc40_fig2_restart.py` builds the concentrated NFW benchmark population used by the documented Frank comparison.
- `export_frank_eddington_table.py` evaluates and exports the Eddington-inverted distribution table used by the generators.
- `export_venus_restart_to_frank_ic.py` converts SphDSMC restart versions 2–5 into the Frank/NSphere comparison format, preserving current per-marker masses.
- `ic_generator_legacy.py` is the maintained general analytic grid-IC front end; it evaluates a user-supplied `distribution(r,w,L)` and writes `EVICRWL1`.
- `sidm_nfw_eddington.py` constructs isotropic equilibrium halo grids by Eddington inversion for NFW, cored, beta-core, Plummer, gravothermal-core, Frank-core, and King profiles.
- `refine_weighted_pic_restart.cpp INPUT OUTPUT SPLIT_FACTOR [--reinitialize-field]` splits collisionless or material-SIDM numerical samples at fixed phase-space position and preserves their mass-weighted moments. Internal energy and internal velocity dispersion must be zero; subhalo objects have a physical mass that enters their encounter law. A frozen-field input requires `--reinitialize-field`. The output is version-6 particle initialization with fresh field/event estimates at the saved physical time. Refinement requires `collision_event_age=0` so that the pending physical collision interval has completed.
- `inspect_weighted_pic_restart.cpp RESTART [--export-particles OUTPUT]` prints and validates WPIC checkpoint headers, clocks, population fields, and aggregate invariants. Inspection leaves the file unchanged. Explicit export retains the particles, internal state, IDs, random counters, time, and step in version 6, resets event ages and timescale estimates, and lets the receiving CPU/GPU task initialize its field and schedule. Export requires `collision_event_age=0` and a distinct output path.
- `inspect_canonical_restart.cpp` inspects the canonical atlas, packets, event clocks, and reproducibility counters in a canonical restart.

## CPU numerical diagnostics and profilers

- `diagnose_boltzmann_maxwellian_residual.cpp` evaluates the discrete collision residual of a Maxwellian equilibrium.
- `diagnose_implicit_reaction_shell.cpp` isolates the local implicit reaction-shell update and its positivity/conservation closure.
- `diagnose_sampled_boltzmann_shell_plan.cpp` prints shell planning, majorants, and expected proposal work before a sampled collision event.
- `diagnose_sampled_boltzmann_shells.cpp` applies and audits sampled finite-angle shell collisions.
- `diagnose_sampled_collision_shell.cpp` reduces the collision operator to a single shell for pair-level investigation.
- `diagnose_spherical_dsmc_collision_moments.cpp` checks SphDSMC momentum/energy moment closure for local pairs.
- `diagnose_spherical_dsmc_restart_shells.py` reads versions 2–5 and reports mass-weighted shell occupancy, velocity support, and locality inputs.
- `diagnose_vp_dynamical_timestep.cpp` decomposes the dynamical/potential timestep criteria for a VP state.
- `diagnose_weighted_vlasov_pic_subhalo_locality.cpp` computes subhalo collision rate, mean free path, and cell-width locality ratios.
- `profile_spherical_dsmc_step.cpp` times gravity, orbit, sorting, collision, and output phases of one representative SphDSMC step.

## CPU analysis and quantitative comparison

- `analyze_einstein_vlasov_wpic.py` validates coordinate-specific EV-WPIC
  output and reports conservative bound-static-family black-hole/halo
  brackets with both input-mass and measured initial-ADM normalizations. A
  halo requires positive realized binding, sustained anisotropic Jeans and
  radius/profile stationarity, small escape, and no horizon; high-angular-
  momentum dispersal is not relabeled as a halo. Its
  threshold command requires and authenticates a strict v2 dynamic-coordinate
  certificate, including the certificate sidecar/hash, coordinate-map
  manifest/artifacts, empirical distribution, inverse-map closure, and
  independently hashed two-resolution/two-realization convergence evidence. Its
  compactness-milestone parity command remains exploratory and cannot issue a
  coordinate certificate.
- `compare_wpic_dynamic_coordinates.py` performs the strict pre-horizon
  coordinate audit. It solves
  `v_r = exp(lambda-mu) v_tau` from a common outer-worldtube clock, evolves
  `A=v_tau`, solves the inverse EF eikonal map, checks forward/inverse
  composition, samples both fields at the same events `(v(tau,r),r)`, and
  compares the pulled-back metric, stress/matter moments, and manifest-hashed
  empirical marker distribution. It emits the
  `venus_wpic_dynamic_coordinate_certificate_v2` certificate and fails closed
  on cadence, overlap, horizon domain, artifact hashes, map closure, or mapped-
  slice provenance; equal coordinate-time labels are never an event map.
- `validate_wpic_null_map.py` is the cheaper fail-closed setup gate run before
  mapped EF jobs fan out.  It authenticates the complete SC-to-EF extraction
  artifact set and manifest, the exact-vacuum `R_B < R_E` extension and its
  clearances, the RK4 step-doubling eikonal estimate, forward/inverse-map
  residuals, and the distinct fixed-family, donor, null-slice, and initially
  deposited ADM-mass channels.  A passing preflight JSON and its SHA-256
  sidecar are setup evidence only; they do not replace the later same-event
  parity certificate.
- `build_wpic_dynamic_coordinate_convergence.py` authenticates a minimum
  two-resolution by two-realization matrix of clean single-resolution parity
  diagnostics, requires monotone refinement for metric/stress/moment/marker
  errors, and derives a safety-factor/Richardson uncertainty envelope for the
  designated fine-A coordinate pair.
- `analyze_frank_core_benchmark.py` compares core density/radius histories with the Frank benchmark contract.
- `analyze_frank_initial_equilibrium.py` measures initial profile, Jeans, and sampling residuals before evolution.
- `analyze_sidm_benchmark.py` computes density, slope, dispersion, core, and conservation summaries for SIDM runs.
- `analyze_vlasov_equilibrium_drift.py` measures collisionless equilibrium drift in profiles and invariants.
- `analyze_weighted_pic_ppc_convergence.py` compares particles-per-cell resolutions under a common radial/time selection.
- `analyze_weighted_pic_subhalo_strength_response.py` measures the physical response across collision-strength scans.
- `audit_frank_collision_density.py` compares the VENUS and Frank density conventions for version-2–5 equal-mass restarts and rejects unequal-mass input that Frank's number-count estimator cannot represent.
- `compare_weighted_pic_profiles.py` aligns two WPIC profile sequences by physical time and reports field-wise differences.
- `digitize_frank_core_figure3.py` converts the comparison curve into a machine-readable table with source metadata.
- `extract_frank_fig2_sphdmc_curve.cjs` reads PDF drawing operators with
  PDF.js and extracts the red SphDSMC reference curve without raster-colour
  ambiguity; install `pdfjs-dist` or set `PDFJS_DIST_MODULE`.
- `summarize_venus_collision_history.py` validates and summarizes material/subhalo collision-history CSVs, including event rates, invariant errors, locality, and coverage.

## CPU plotting and GIF construction

- `plot_einstein_vlasov_outputs.py` plots EV density, pressure/current, metric, compactness, history, and phase projections from modular output.
- `spherical_profile_rendering.py` is the common named-column spherical-profile reader and directly executable density/slope GIF renderer; it requires `time`, `step`, `r`, and `rho`.
- `plot_weighted_pic_ppc_overlay.py` overlays matched profiles from a marker-resolution sequence on a requested radial interval.
- `render_density_slope_sigma2_gif.py` combines density, logarithmic slope, and radial velocity-dispersion evolution into a timestamped GIF. Its `--radial-moment-convention auto` setting reads the solver metadata: persistent-marker SphDSMC writes the centered pressure, while the Newtonian grid writer stores the raw radial second moment. The renderer converts raw moments to centered pressure once, conserves mass and momentum when merging bins, and records the selected convention in a sidecar. Use `--radial-moment-convention central` or `raw` when importing profiles whose metadata are unavailable.
- `render_frank_venus_sphdsmc_gif.py` builds the benchmark SphDSMC comparison animation and its physical-time sidecars.
- `render_frank_wpic_density_slope_gif.py` animates Frank WPIC density and slope.
- `render_frank_wpic_gif.py` renders the three-panel Frank benchmark animation using its fixed mass, density, velocity, and time conversion. Its input units are `G=M_total=r_s=1` with `M_total/M_s=1.1252580933331495`; native `G=M_s=r_s=1` data require a separate conversion. It reconstructs logarithmic shell volumes from the stored arithmetic centers and keeps radial mean flow separate from centered dispersion.
- `render_frank_wpic_uniform_time_gif.py` selects uniformly spaced physical times before rendering, avoiding output-cadence bias.
- `render_weighted_einstein_vlasov_pic_profile_gifs.py` renders relativistic WPIC matter/metric/history sequences for the low- and high-compactness EV marker tasks.
- `render_weighted_einstein_vlasov_coordinate_pair_gif.py` renders the standard three-panel high-compactness SC/ingoing-EF overlay. It authenticates the null-map manifest and donor history, solves and checks the forward/inverse eikonal map at every displayed overlap sample, plots `log10(rho)`, logarithmic density slope, and `2m/r`, then switches to clearly labelled EF-only saved slices after the SC chart ends. It writes PNG previews plus exact frame-clock and SHA/provenance sidecars.
- `render_weighted_pic_pure_gamma_gifs.py` renders pure power-law/gamma run products under common axis and timing conventions.
- `render_weighted_pic_subhalo_collision_age_gif.py` compares profiles at
  common integrated collision ages. Its displayed profiles interpolate saved
  logarithmic densities at those ages, so this tool visualizes an age-based
  comparison. The physical-time movie tools and their saved-state clocks
  describe a different comparison.
- `render_weighted_pic_subhalo_comparison_gif.py` aligns and compares subhalo runs panel by panel.
- `render_weighted_pic_subhalo_physical_time_gif.py` aligns subhalo curves at requested physical times.
- `render_weighted_pic_subhalo_strength_scan_gif.py` displays collision-strength scans with a shared normalization and non-overlapping labels.
- `render_weighted_vlasov_pic_equilibrium_gif.py` visualizes collisionless WPIC equilibrium drift with schema-checked profile grids and portable Pillow/font handling.
- `normalize_gif_timing.py` rewrites frame delays to match physical-time metadata without recomputing the frames.
- `select_frank_wpic_profiles_uniform_time.py` selects the closest available profiles to a uniform physical-time grid and writes the selection record.
- `thin_frank_venus_render_inputs.py` copies/links a reproducible sparse render set and records which source profiles were retained.
- `monitor_frank_venus_central_density.py` incrementally reads a running job and refreshes the central-density diagnostic without changing solver data.

The GIF tools write or consume a physical frame-time record. Compare those
times with the solver outputs to distinguish the sampling interval from the
playback rate. Temporary images use the platform temporary directory or an
explicit work directory. Plotters check required columns and compatible input
sets before calculating slopes, moments, or animations.

## CPU Slurm templates

- `cpu/scripts/slurm/ias_einstein_vlasov_wpic.sbatch` builds VENUS from a source manifest
  source tree and runs one Schwarzschild or EF WPIC calculation on IAS Typhon,
  recording exact command, binary/source/IC/map-artifact hashes, status, and
  summary. Fresh EF production runs require a solved null-data import. Static
  parity/threshold runs also require a separate positive-binding attestation.
- `cpu/scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py`
  checks both fixed-family flags and the independent `B=M0-M_ADM`, `B/M0`
  algebra in a solver summary.
- `cpu/scripts/slurm/validate_einstein_vlasov_wpic_threshold_provenance.py`
  binds a strict parity certificate to the exact deployed source manifest and
  executable, the authenticated 2x2 static-shell signature, and the requested
  continuum/kick/ADM/null-map scan controls before the first `sbatch`.
- `cpu/scripts/slurm/ias_einstein_vlasov_wpic_parity_postprocess.sbatch` runs
  the strict same-event certifier only after authenticated convergence
  evidence exists.
- `cpu/scripts/slurm/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch` accepts
  one single-resolution audit only when its sole issue is missing convergence.
- `cpu/scripts/slurm/ias_einstein_vlasov_wpic_eikonal_reprocess.sbatch` is a
  diagnostic-only, non-overwriting reprocessor for already completed
  Schwarzschild/EF pairs. It runs no physics evolution. It repeats the solved
  event-map comparison for a default or explicitly selected increasing list
  of power-of-two eikonal substep counts, preserves each raw `n` versus `2n`
  result with source/input hashes and resource use, and fails closed on map,
  common-domain coverage, topology, composition, or provenance errors. Its
  summary and pass markers are substep diagnostics only: they cannot issue a
  parity certificate, replace the 2x2 resolution/realization convergence
  evidence, or authorize threshold jobs.
- `cpu/scripts/slurm/ias_einstein_vlasov_wpic_parity_convergence.sbatch`
  derives the authenticated convergence evidence from the four matrix audits.
  Production guardrails are 0.005 for metric components, 0.02 for every
  stress/matter moment, and 1e-10 for empirical distribution parity; these
  accompany rather than replace 2x2 convergence.
- `cpu/scripts/slurm/submit_einstein_vlasov_wpic_parity.sh` submits the
  build -> 2-resolution x 2-realization [base -> dense SC donor/null extraction
  -> mapped EF -> diagnostic] -> convergence evidence -> strict fine-A
  certificate `afterok` DAG. Production values for the bound static-shell
  continuum, fixed inward kick, target ADM mass, map worldtube `R_B`, evolution
  boundary `R_E`, donor duration, and end time are supplied only after local
  equilibrium/binding/crossing preflight. The required
  `PARITY_EXPECTED_EF_OUTCOME` makes a black-hole campaign fail before its
  diagnostics if the mapped EF job does not actually form a horizon. Dense
  parity diagnostics and terminal certification request 64 GiB because the
  certifier materializes both coordinate histories in memory.
- `cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long.sh` creates one
  new post-certificate long-run request without guessing any physical or
  numerical control. It pins and hashes the deployed science launcher and
  validators together with a separate immutable paired-long workflow bundle,
  records the certified fine-A physical and null initial data, and submits
  only a small gate job with an `afterok` dependency on the strict-certificate
  job. The helper refuses inherited workflow overrides and campaign-root
  reuse; it does not submit either evolution directly.
- `cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gate.sbatch`
  reauthenticates the strict certificate, its SHA-256 sidecar, the fine-A
  artifacts, all pinned tools, the fixed static-shell family controls, and the
  production eikonal count. It rebuilds nothing. After the certificate and
  provenance gates pass, it publishes immutable campaign controls and submits
  exactly one Schwarzschild run with outcome `any` and one mapped EF run with
  required outcome `black_hole`, both dependent on the gate. It records those
  jobs in one hashed receipt and invokes the pinned visualization submitter.
- `cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh` validates
  that the hashed paired-run receipt names exactly one gate and the intended
  Schwarzschild/EF jobs, that both jobs share the certified controls and
  physical initial data, and that the EF row names the authenticated null-map
  artifact. It refuses path overrides, reused requests, or reused output and
  queues the renderer only with `afterok` dependencies on both long runs.
- `cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch`
  rechecks every workflow, science-tool, receipt, control, request, and
  null-map hash before rendering. It produces the fixed 72-frame,
  12-frame-per-second SC/EF black-hole animation, including a limited
  same-event overlap followed by explicitly labelled EF-only continuation;
  verifies the GIF/PNG signatures, frame ledger, phase labels, provenance,
  and authenticated map inputs; and atomically publishes the completed
  visualization directory so a failed job cannot leave a partial result.
- `cpu/scripts/slurm/submit_einstein_vlasov_wpic_angular_scan.sh` submits a
  certificate-gated neutral scan over positive scales of the static shell's
  delta-angular-momentum magnitude. Every point uses the same continuum base,
  deterministic quiet realization, fixed inward kick, and target initial ADM
  mass; only `eta -> s_L^2 eta` and the common fixed-ADM weight normalization
  vary. Reconnaissance values are preflight-derived rather than assigned an
  unvalidated numerical interval.
  Every EF value receives its own SC donor and mapped null artifact; the helper
  deliberately excludes the unsupported radial \(L=0\) origin crossing. Its
  receipt records `L/M_ADM,target`; post-run threshold JSON/CSV records both
  target and measured-initial-ADM ratios explicitly. It stores a semantic
  certificate/source/binary/family/map attestation and forbids rebuilding the
  certified executable in the dependent scan DAG.
- `cpu/scripts/slurm/ias_hybrid_ev_vp.sbatch` launches supported EV/VP MPI+OpenMP runs on IAS.
- `cpu/scripts/slurm/ias_openmp_bp.sbatch` launches one-rank OpenMP BP runs and enforces the collision MPI restriction.
- `cpu/scripts/slurm/iitk_param_hybrid_ev_vp.sbatch` and `cpu/scripts/slurm/iitk_param_openmp_bp.sbatch` provide the corresponding PARAM Sanganak resource/module layouts.

## GPU task mains, utilities, and benchmarks

Each GPU task has a small source file under `gpu/tools/`:

- `venus_gpu_vlasov_poisson.cu` locks collisionless VP. The single-GPU task
  accepts static or Einstein--de Sitter evolution. MPI retains Einstein--de Sitter.
- `venus_gpu_subhalo_boltzmann_poisson.cu` locks expanding subhalo
  scattering/capture;
- `venus_gpu_sidm_boltzmann_poisson.cu` requires an explicit static-halo
  material-SIDM law;
- `venus_gpu_sidm_constant.cu` locks constant material SIDM;
- `venus_gpu_sidm_power_law.cu` locks the softened power-law law; and
- `venus_gpu_sidm_yukawa_born.cu` locks the Yukawa--Born law.

`run_wpic_subhalo_gpu_compat.cu` retains the older collisionless/subhalo
auto-selection interface. All task mains call the shared engine in
`gpu/src/wpic_production_driver.cu`.

- `generate_cuda_subhalo_table.py` reads the CPU Plummer calibration table
  and writes a CUDA header containing the same tabulated encounter data. This
  shared input keeps the CPU and GPU subhalo laws consistent.
- `compare_wpic_restarts.cpp` compares checkpoint metadata and ID-sorted
  marker records, exactly for discrete state and within tolerances for
  floating fields.
- `render_venus_gpu_density_slope_gif.py` reads explicit GPU run directories
  and renders density/slope animations with physical-time metadata through the
  shared CPU weighted-PIC renderer.

Focused performance sources are
`gpu/benchmarks/benchmark_vlasov_poisson_gpu.cu` and
`benchmark_material_sidm_gpu.cu`. `benchmark_wpic.cu` is the compatibility
aggregate, `benchmark_wpic_cpu.cpp` is the CPU comparison program, and
`benchmark_wpic_mpi.cu` measures one-rank-per-GPU execution.

## Release-wide utility

`scripts/check_shell_syntax.sh` runs non-executing Bash syntax checks over
the maintained shell and Slurm files.

## Diagnostics and validation tools

- `cpu/tools/render_weighted_pic_gamma_fits_gifs.py`: Sachiet Kapur's density/slope animation, initial/final comparisons, positive inner/outer cusp indices, fitted-value CSV, and recorded fit settings.
- `cpu/tools/plot_wpic_history.py`: maximum resolved density, mass, stored energy, timestep, and effective-marker histories, interpreted with the run's background and boundaries.
- `cpu/tools/plot_subhalo_virial_1.py`: resolved Newtonian orbital virial ratio with the arithmetic-center shell volumes used by WPIC.
- `cpu/tools/render_ef_interior_gif.py`: horizon-crossing observer density, slope, compactness, retained/excised weights, and interior preview frames.

The equations, conventions, commands, and limitations of these tools are in the unified manual's WPIC diagnostics and EF interior sections.

- `cpu/tools/plot_initial.py`: initial profiles and phase space from a zero-time grid-solver output; compactness panels follow the physics model.
- `cpu/tools/plot_color_rho_r.py`: time-colored density profiles from WPIC snapshots.
- `cpu/tools/plot_wpic_scaled_profiles.py`: Sachiet's radius-scaled profile comparison, with conservative shell masses and optional density normalization.
- `scripts/cluster_gpu_smoke.sh`: one-GPU rebuild and finite CUDA regression suite.
- `scripts/check_manual_log.py`: portable post-build check of resolved LaTeX references and unique labels.



## Complete particle snapshot analysis

`analysis/scripts/phase_space_snapshot_analysis.py` reads complete GPU version-6
particle snapshots and their contemporaneous field files. It checks the native
format and gravitational source, then accumulates mass-weighted phase-space
histograms using the physical radial and velocity bin widths.

`analysis/scripts/shifted_energy_snapshot_analysis.py` uses that reader to
compute instantaneous orbital energies and the spherical phase volume in
each energy bin. Dividing the marker mass by this volume gives the mean
distribution function as a function of energy above the central potential.
Both tools require NumPy and write portable NPZ arrays and JSON metadata.
The complete run, reduction and plotting commands are in
[the analysis guide](../analysis/README.md). The manual subsection
“Complete particle snapshots: phase-space and energy distributions” derives
the measures and shows the corresponding numerical operations.
