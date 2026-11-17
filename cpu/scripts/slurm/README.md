# Portable VENUS CPU Slurm templates

Developed by Uddipan Banik and Frank van den Bosch

These scripts request computing resources and start the selected VENUS
executable. Message Passing Interface (MPI) tasks divide angular momentum cells between ranks, while
shared-memory tasks divide work between threads in one rank. Edit partition, account, and module names after checking `sinfo` and
`module avail`. Full details are in
[`../../docs/cluster.md`](../../docs/cluster.md).

The coordinate recipes use areal radius `r`, enclosed mass `m`, Schwarzschild (SC) time
`tau` and advanced ingoing Eddington--Finkelstein (EF) time `v`, in `G=c=1` units. `R_B` is the map-worldtube
radius and `R_E` the full evolution boundary. `s_L` Is the multiplicative
specific angular momentum scale. In the vacuum-extension block, `M_null`
is conserved null-slice Arnowitt--Deser--Misner (ADM) mass, `a=1-2m/r` and `b` are EF metric functions,
`A=partial v/partial tau` and `F=1/A` are map derivatives, `T_ab` is
stress-energy, `r_*` the Schwarzschild tortoise radius, `s=-tau` the donor
lookback coordinate and `s_B` its boundary value at `R_B`.

| Template | Supported task shape |
|---|---|
| `ias_hybrid_ev_vp.sbatch` | IAS MPI+OpenMP Eulerian Einstein--Vlasov (EV) or collisionless Vlasov--Poisson (VP) |
| `iitk_param_hybrid_ev_vp.sbatch` | IITK PARAM MPI+OpenMP Eulerian EV or collisionless VP |
| `ias_openmp_bp.sbatch` | IAS one-rank grid Boltzmann--Poisson (BP) collision or spherical direct simulation Monte Carlo (SphDSMC) |
| `iitk_param_openmp_bp.sbatch` | IITK PARAM one-rank grid BP collision or SphDSMC |
| `ias_einstein_vlasov_wpic.sbatch` | IAS Typhon one-process/88-thread VENUS EV-WPIC Schwarzschild or EF evolution |
| `ias_einstein_vlasov_wpic_parity_diagnostic.sbatch` | One clean, deliberately single-resolution solved-eikonal audit |
| `ias_einstein_vlasov_wpic_eikonal_reprocess.sbatch` | Non-overwriting diagnostic-only eikonal substep study on completed parity output |
| `ias_einstein_vlasov_wpic_parity_convergence.sbatch` | Two-resolution/two-realization uncertainty-evidence builder |
| `ias_einstein_vlasov_wpic_parity_postprocess.sbatch` | Strict solved-eikonal same-event certificate postprocessing |
| `ias_einstein_vlasov_wpic_paired_long_gate.sbatch` | Certificate-dependent gate for one authenticated long SC/EF pair |

Use an exact task executable:

```bash
sbatch --nodes=1 --ntasks-per-node=4 --cpus-per-task=8 --export=ALL,EXECUTABLE=venus_cpu_vlasov_poisson_eulerian_mpi,CONFIG=examples/modular/vp_expanding_nfw_gamma2_collisionless.ini,VENUS_RUN_ROOT=/scratch/project/venus scripts/slurm/ias_hybrid_ev_vp.sbatch
```

```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=32 --export=ALL,EXECUTABLE=venus_cpu_boltzmann_poisson_fokker_planck_openmp,CONFIG=examples/modular/bp_sidm_corecollapse_yukawa.ini,VENUS_RUN_ROOT=/scratch/project/venus scripts/slurm/ias_openmp_bp.sbatch
```

For SphDSMC, substitute `venus_cpu_sidm_sphdsmc_constant_openmp` or
`venus_cpu_sidm_sphdsmc_velocity_dependent_openmp`. For finite-angle BP,
select the exact/reversible/sampled/streamed task front. BP and SphDSMC must
use `--ntasks=1`. Increase `--cpus-per-task` instead. Only Eulerian EV and
VP have CPU MPI task fronts.

Useful overrides:

- `VENUS_DIR=/path/to/VENUS/cpu`
- `VENUS_RUN_ROOT=/path/to/runs`
- `CONFIG=/path/to/run.ini`
- `EXECUTABLE=/path/to/task_binary`
- `OUTPUT_DIR=/path/to/output`
- `RESTART_DIR=/path/to/restarts`
- `VENUS_BUILD=0` to require a prebuilt binary
- `VENUS_MAKE_JOBS=N`
- `VENUS_SKIP_MODULES=1`
- `VENUS_PURGE_MODULES=0`

Relative configuration/output paths resolve from the Slurm submission
directory. A relative executable resolves from `VENUS_DIR`. By default the
templates create a new run directory outside the source tree. Point
`VENUS_RUN_ROOT` at project or scratch storage for long runs.

## VENUS Einstein--Vlasov weighted particle-in-cell (WPIC) on IAS Typhon

Export `VENUS_SOURCE_ROOT` as the absolute deployed repository path and
`VENUS_RUN_ROOT` as the absolute simulation-output root before using these
launchers. The source root must contain `cpu/Makefile`.

The WPIC launcher builds and runs
`venus_cpu_einstein_vlasov_wpic_black_hole`, records the resolved command and
source, binary, physical-IC, controls, and coordinate-map hashes, and validates
the machine-readable result before returning success. Concurrent campaign jobs
serialize a shared build with `flock`.

The checkout and run roots are explicit and overridable:

```text
VENUS_SOURCE_ROOT=/path/to/deployed/VENUS
VENUS_RUN_ROOT=/path/to/project-or-scratch/runs
```

`OUTPUT_DIR`, `PHYSICAL_IC`, `COORDINATES`, `EXPECTED_OUTCOME`,
`VENUS_BUILD`, `VENUS_MAKE_JOBS`, and `VENUS_RUN_ARGS_FILE` are also
overridable. An argument file contains one literal command-line argument per
nonblank line. Comment lines start with `#`. The launcher owns
`--coordinates`, `--expected-outcome`, `--output-dir`, `--physical-ic`, and
`--threads`, so those switches must not occur in the argument file.

### Physical initial data for a bound static shell

High-compactness parity and threshold production use only
`HIGH_INITIAL_CONDITION=energy_cutoff_shell`. The continuum is

```text
f(E,ell) = C (1 - E/E0)_+ delta(ell-ell0),
E = exp(mu(r)) sqrt(1 + w^2 + ell0^2/r^2).
```

The required continuum controls are `ENERGY_CUTOFF_AMPLITUDE=C`,
`ENERGY_CUTOFF_ELL0=ell0`, and
`ENERGY_CUTOFF_CENTRAL_Y=yc`. The static solver normalizes the lapse at infinity and solves the radial
Einstein constraints with the distribution above. This determines `E0` and
both matter-support edges. The selected branch must have finite support,
positive binding and compactness below unity. The quiet sampler emits
adjacent `(+w,-w)` pairs with already-invariant WPIC weights, so marker counts
must be even.

Choose one inward perturbation and keep it fixed throughout the family:

```text
w -> w - kappa T(r),
T(r) = 1                         if rc = 0,
     = r^2 / (r^2 + rc^2)        if rc > 0.
```

The corresponding controls are `RADIAL_INFALL=kappa` and
`INFALL_CORE_RADIUS=rc`. They remain identical at every angular momentum
point. Vary only

```text
ell_p(s_L) = s_L ell0,       eta_p(s_L) = (s_L ell0)^2 / 2,
```

using `ANGULAR_MOMENTUM_SCALE` for parity or the distinct values in
`ANGULAR_MOMENTUM_SCALES` for a scan. Marker radii, kicked radial momenta,
IDs, source IDs, and relative invariant weights remain fixed. The driver then
rescales all weights by one common positive factor to reach the explicitly
chosen `TARGET_INITIAL_ADM_MASS` and solves the nonlinear Schwarzschild
constraint again. For each realized member, calculate the rest mass `M0` and require positive
global binding `M0-M_ADM>0` after this normalization.

Parity and threshold helpers export `VENUS_REQUIRE_POSITIVE_BINDING=1` for
every initialize-only, donor, and evolution job. The launcher independently
reads `summary.json`, requires both fixed-family binding flags, and verifies
finite positive values satisfying `B=M0-M_ADM` and
`f_binding=B/M0`. It records `positive_binding_validation.json`. A failed or
missing audit fails the Slurm step even when the solver process returned zero.

Both production helpers require every `s_L` to be strictly positive.
`s_L=0` would create purely radial characteristics that can cross the center,
which the current EF origin treatment does not support. It must not be
inserted into a parity campaign, an EF threshold scan, or the
Schwarzschild-only path of the production scan helper.

### Choose the physical family and numerical resolution

The static continuum `(C,ell0,yc)`, fixed kick `(kappa,rc)`, parity or scan
`s_L` values, target ADM mass, map worldtube `R_B`, evolution boundary `R_E`,
reverse-donor duration, evolution end time, and threshold bracket are
chosen for the physical family and checked in a finite initialization and
short evolution before a long calculation.

The submission scripts still contain finite marker-count, radial-cell, and
timestep fallbacks so their resource shape is mechanically defined. Use these fallback values to understand the script interface. The required
resolution follows from equilibrium preservation and refinement checks for
the selected static shell. The same caution applies
to scheduler memory and walltime requests. Set
`PARITY_{COARSE,FINE}_{MARKERS,CELLS,DT_MAX}` or
`SCAN_MARKERS`, `SCAN_RADIAL_CELLS`, and `SCAN_DT_MAX` only after measuring a
short local/preflight run. Determine storage and memory from the actual dense
donor cadence. Do not submit a long job merely because a value appears in a
script.

Before scheduling long runs, verify locally that:

1. the unperturbed static solution has finite support, `E0<1`, positive
   compactness gap, positive binding, small constraint/Jeans residuals, and
   adequate vacuum buffer.
2. sampling refinement and at least two quiet realizations reproduce its
   mass, rest weight, moments, and short-time equilibrium behavior.
3. the chosen fixed kick remains positively bound and produces a plausible
   low-`L` collapse/high-`L` bound-halo pair rather than an unbound dispersal
   family.
4. marker/cell/timestep/domain refinement resolves the strong-field region.
   And
5. the proposed `R_B`, `R_E`, and donor duration satisfy every crossing,
   vacuum, and provenance gate below.

### Solved-null map with $`R_B<R_E`$

`VENUS_EXPORT_EF_NULL_INITIAL_DATA=/path/map.bin` selects a fresh evolving
Schwarzschild donor. `VENUS_EF_NULL_DONOR_DURATION` sets its preflighted
reverse-time span, and `VENUS_EF_MAP_WORLDTUBE_RADIUS=R_B` sets the numerical
map worldtube. `VENUS_EF_NULL_INITIAL_DATA=/path/map.bin` selects the dependent
EF consumer. `VENUS_HIGH_R_MAX=R_E` is the full evolution boundary.

Production uses `0<R_B<R_E`. The map worldtube must be an exact radial-mesh
face and `(R_B,R_E]` must contain at least four complete cells. All donor
matter and all marker crossings must retain the manifest's four-cell
clearance inside `R_B`. Beyond `R_B`, the imported initial null surface uses
the exact neutral vacuum extension

```text
m = M_null,    a = 1 - 2 M_null/r,    A = F = b = 1,
T_ab = 0,
r_*(r;M) = r + 2 M log(r/(2M) - 1),
s(r) = s_B + r_*(r;M_null) - r_*(R_B;M_null),
tau(r) = -s(r).
```

The extension applies in the exterior `r>2M_null`. It completes the initial
null surface through the vacuum region. The EF solver then advances the
particles and metric on the complete grid extending to `R_E`.

The map manifest and its nested artifacts must prove that every persistent
marker crosses the selected null surface exactly once, the full surface lies
inside the dense donor history, the numerical worldtube is vacuum, and the
eikonal map remains nondegenerate. The auxiliary donor uses East's
`s=-tau` construction and restores `w_physical=-w_reverse` at crossing. The
runtime additive shift is `v_hat=v+T_donor`, so the imported surface is
`v_hat=0`. Equal coordinate-time labels are never used as an event map.

A fresh EF production run without mapped null data is rejected. Resume keeps
the restart's coordinate tag and markers and must receive the same null
artifact again. The launcher reauthenticates the physical IC, controls,
realization, grid, angular scale, map/full-grid fingerprints, and all artifact
hashes.

`tools/validate_wpic_null_map.py` runs for every map exporter and consumer.
It fails closed on incomplete artifacts, nonunique crossings, insufficient
clearance, a nonvacuum worldtube or analytic extension, failed forward/inverse
closure, eikonal step-doubling error, inconsistent `R_B`/`R_E`, or
inconsistent fixed-family, null-slice, and deposited ADM-mass channels. Its `null_map_preflight.json` and SHA-256 sidecar record initialization
accuracy and input identity. The later coordinate comparison measures
whether the evolved SC and EF solutions agree at the same events.

### Four checks before an angular momentum scan

The scan checks the physical initial family, each angular momentum member,
the coordinate extraction and the numerical comparison in this order:

1. **Strict parity-provenance gate.** Before the first threshold `sbatch`,
   the scan helper authenticates a status-pass v2 same-event coordinate
   certificate, its SHA-256 sidecar or explicit digest, the null-map artifact
   graph, empirical marker comparison, inverse-map closure, and the
   two-resolution by two-realization convergence evidence. A second semantic
   audit requires the certificate's production summaries, 2x2 static-shell
   signature, source checksum manifest, and executable to match the deployed
   scan source/binary and requested
   `(C,ell0,yc,kappa,rc,M_ADM,R_B,R_E,T_donor)`. The scan retains
   `parity_provenance_attestation.json`. Only `s_L`, resolution, realization,
   cadence, and evolution duration may differ for their documented roles. The
   seed uses `VENUS_BUILD=0`, so a rebuild cannot replace the authenticated
   binary.
2. **Base initialize-only gate.** One `s_L=1` Schwarzschild job reuses the
   certified executable and freezes the coordinate-neutral physical IC. The driver must accept the
   static continuum, invariant paired sampler, kick, fixed-ADM solve, support
   buffers, and positive realized binding before any member is initialized.
3. **Per-$`s_L`$ initialize-only gate.** Every requested strictly positive
   scale gets its own Schwarzschild setup job after the base. It reapplies
   `eta=(s_L ell0)^2/2`, performs the common fixed-ADM normalization, checks
   actual compactness/support and positive binding, and records a summary
   before that member may start a long donor or evolution.
4. **Authenticated null-domain map gate.** For an EF scan,
   `NULL_DOMAIN_GATE_SCALE` names one preflight-selected member of
   `ANGULAR_MOMENTUM_SCALES`. Its donor is submitted first. Every other donor
   depends on its successful unique-crossing, vacuum-worldtube,
   exact-extension, eikonal, and provenance postflight. The gate scale is a
   coverage choice, not a claimed threshold.

The parity helper creates rather than consumes the strict certificate. Its
fine-A initialize-only job and authenticated donor are the corresponding
base/setup and production-map gates. The other three coordinate pairs do not
fan out until that exact fine-A map passes.

## Schwarzschild/EF coordinate comparison and refinement

Supply every physical and map control explicitly on an IAS login node:

```bash
cd "$VENUS_SOURCE_ROOT/cpu"
HIGH_INITIAL_CONDITION=energy_cutoff_shell \
ENERGY_CUTOFF_AMPLITUDE="$STATIC_C" \
ENERGY_CUTOFF_ELL0="$STATIC_ELL0" \
ENERGY_CUTOFF_CENTRAL_Y="$STATIC_YC" \
RADIAL_INFALL="$FIXED_KICK" \
INFALL_CORE_RADIUS="$FIXED_KICK_CORE" \
ANGULAR_MOMENTUM_SCALE="$PARITY_S_L" \
TARGET_INITIAL_ADM_MASS="$FIXED_ADM" \
PARITY_R_MAX="$R_E" \
EF_MAP_WORLDTUBE_RADIUS="$R_B" \
EF_NULL_DONOR_DURATION="$DONOR_DURATION" \
PARITY_T_END="$PARITY_END_TIME" \
PARITY_EXPECTED_EF_OUTCOME="$EXPECTED_EF_OUTCOME" \
PARITY_REALIZATION_A="$QUIET_REALIZATION_A" \
PARITY_REALIZATION_B="$QUIET_REALIZATION_B" \
PARITY_COARSE_MARKERS="$N_MARKER_COARSE" \
PARITY_COARSE_CELLS="$N_R_COARSE" \
PARITY_COARSE_DT_MAX="$DT_COARSE" \
PARITY_FINE_MARKERS="$N_MARKER_FINE" \
PARITY_FINE_CELLS="$N_R_FINE" \
PARITY_FINE_DT_MAX="$DT_FINE" \
VENUS_EIKONAL_SUBSTEPS="$EIKONAL_SUBSTEPS" \
CAMPAIGN_ROOT="$PARITY_CAMPAIGN_ROOT" \
  bash scripts/slurm/submit_einstein_vlasov_wpic_parity.sh
```

All placeholders above are preflight results, not recommendations. The helper
requires `PARITY_S_L>0`, distinct realization IDs, even marker counts, strict
fine-over-coarse marker and radial refinement, a smaller fine timestep, and
`R_B` on both meshes with at least four full vacuum-extension cells before
`R_E`. `PARITY_EXPECTED_EF_OUTCOME` is also mandatory. Use `black_hole` for
the requested horizon-formation comparison: every mapped EF job must then
form an apparent horizon before its dependent diagnostic can run. `halo` and
`any` are available for separately preflighted non-collapse comparisons. A
coordinate certificate alone is not evidence that a horizon was exercised.

`VENUS_EIKONAL_SUBSTEPS` has no production default. It must be one positive
integer chosen before submission. Each of the four coordinate diagnostics
integrates with `n` and `2n` substeps per radial cell, measures the raw change,
and retains the `2n` result. The same `n` is written to the controls manifest
and every 27-column receipt row, passed to all four diagnostics, and required
again by the convergence builder and terminal certificate. Missing, zero, or
mixed values fail before they can be certified.

The production helper pins its four Slurm scripts and all scientific Python
tools under the staged `VENUS_SOURCE_ROOT`. It records their submission-time
SHA-256 values in the controls and source manifests. Each job checks the
relevant script and tool before and after doing work, and the comparator and
builder identities are retained in the JSON evidence and certificate.
Inherited `SBATCH_SCRIPT`, `DIAGNOSTIC_SCRIPT`, `CONVERGENCE_SCRIPT`,
`POSTPROCESS_SCRIPT`, certifier, builder, analyzer, null-map validator, and
binding-validator overrides are rejected. These checks keep the source and analysis version fixed while jobs wait in
the queue. A changed file stops the affected job before its results are used.

Compare two resolutions and two independently shifted quiet particle
realizations. Resolution changes measure discretization error, while the
independent realizations measure sensitivity to particle sampling:

| Pair | Resolution | Quiet realization | Role |
|---|---|---|---|
| coarse-A | coarse | A | refinement and PIC evidence |
| coarse-B | coarse | B | refinement and PIC evidence |
| fine-A | fine | A | designated reference and production map gate |
| fine-B | fine | B | refinement and PIC evidence |

For each matrix entry the dependency chain is:

1. create an immutable coordinate-neutral, initialize-only physical IC.
2. evolve its Schwarzschild donor, solve the ingoing eikonal surface, cross
   every marker once, transform `w` to future-directed `P>0`, and authenticate
   the map.
3. evolve the mapped EF initial surface with identical physics and provenance.
4. solve the forward and inverse dynamical coordinate maps and compare the
   pulled-back metric, all stress components and matter moments, and the
   hashed empirical marker measure at identical events.

```text
fine-A base -> fine-A SC donor/map/postflight [production map gate]
            -> fine-A EF -> fine-A diagnostic
production map gate -> three auxiliary [base -> SC donor/map/postflight
                                      -> EF -> diagnostic]
four diagnostics -> convergence evidence -> strict fine-A certificate
```

The convergence builder requires common continuum, kick, `s_L`, ADM, domain,
and map controls. Distinct A/B IC fingerprints. Strict resolution refinement.
And decreasing metric, stress, moment, and empirical-distribution errors. It
also authenticates one resolution-independent static-shell signature across
all four pairs: continuum normalization/support, cutoff energy, invariant
weight convention, zero post-sampling canonicalization, and fixed kick.
Production relative (L_2) and (L_\infty) guardrails are `0.005` for every
metric component, `0.02` for every transformed stress and matter moment, and
`1e-10` for the empirical distribution. Assess these ceilings together with the changes under refinement and
particle resampling in the 2x2 matrix.
The builder derives its uncertainty envelope rather than accepting one
supplied by the user. The terminal strict comparison writes
`dynamic_coordinate_certificate.json` only on a status-pass result.
The dense fine SC/EF CSVs are parsed as Python object graphs. Both diagnostic
and terminal-certificate jobs therefore request 64 GiB rather than relying on
the substantially smaller on-disk CSV size.

### Diagnostic-only eikonal substep reprocessing

`ias_einstein_vlasov_wpic_eikonal_reprocess.sbatch` rereads one completed
Schwarzschild/EF pair and repeats only the same-event coordinate comparison.
It never starts either physics solver. For each requested value `n`, the
comparator integrates the map with `n` and `2n` substeps per radial cell and
records their raw difference in coordinate time and logarithmic Jacobian. It
does not divide that difference by the formal RK4 Richardson factor.

The required inputs are the completed pair, its null-map manifest and
crossings, a new `SWEEP_ROOT` below the campaign's
`diagnostic_reprocessing/` directory, and a frozen copy of the comparator in
`SWEEP_ROOT/source`. The default list is `2 4 8 16 32 64`. A smaller,
strictly increasing power-of-two subset can be selected without changing the
scientific tolerances, for example:

```bash
VENUS_EIKONAL_SUBSTEP_LIST="8 16 32" \
CAMPAIGN_ROOT="$PARITY_CAMPAIGN_ROOT" \
SWEEP_ROOT="$PARITY_CAMPAIGN_ROOT/diagnostic_reprocessing/$NEW_SWEEP_TAG" \
PAIR=fine_a \
FROZEN_CERTIFIER="$PARITY_CAMPAIGN_ROOT/diagnostic_reprocessing/$NEW_SWEEP_TAG/source/compare_wpic_dynamic_coordinates.py" \
  sbatch scripts/slurm/ias_einstein_vlasov_wpic_eikonal_reprocess.sbatch
```

Never reuse a sweep root: the job refuses to overwrite a pair directory. To
add `2 4 64` after an initial `8 16 32` study, create another sweep root. Each
substep result retains the exact command, JSON and sidecar, log, resource
usage, source checksum, and hashes of every input consumed by the comparator.
The job rechecks the inputs and frozen source after the last comparison.

A clean highest requested value says only that this diagnostic substep study
met its fixed raw-map, composition, and common-domain coverage gates. The
stronger `STRICT_SUBSTEP_GATE.PASS` marker is possible only when both `32`
and `64` are present and clean in the same root. Neither marker is a parity
certificate. This reprocessor cannot create convergence evidence, issue a
coordinate-parity certificate, or authorize angular momentum threshold jobs.
All four resolution/realization diagnostics must still pass, the convergence
builder must still run, and the terminal strict certificate must still be
authenticated before a threshold campaign is allowed.

The intermediate evidence and terminal certificate can be reproduced from the
authenticated matrix with:

```bash
python3 tools/build_wpic_dynamic_coordinate_convergence.py \
  --coarse-a "$PARITY_CAMPAIGN_ROOT/diagnostics/coarse_a.json" \
  --coarse-b "$PARITY_CAMPAIGN_ROOT/diagnostics/coarse_b.json" \
  --fine-a "$PARITY_CAMPAIGN_ROOT/diagnostics/fine_a.json" \
  --fine-b "$PARITY_CAMPAIGN_ROOT/diagnostics/fine_b.json" \
  --eikonal-substeps "$EIKONAL_SUBSTEPS" \
  --diagnostic-certifier-sha256 "$CERTIFIER_SHA256" \
  --convergence-builder-sha256 "$BUILDER_SHA256" \
  --metric-ceiling 0.005 \
  --stress-ceiling 0.02 \
  --moment-ceiling 0.02 \
  --distribution-ceiling 1e-10 \
  --output "$PARITY_CAMPAIGN_ROOT/dynamic_coordinate_convergence.json"

python3 tools/compare_wpic_dynamic_coordinates.py \
  --schwarzschild "$PARITY_CAMPAIGN_ROOT/pairs/fine_a/schwarzschild" \
  --ef "$PARITY_CAMPAIGN_ROOT/pairs/fine_a/eddington_finkelstein" \
  --manifest "$PARITY_CAMPAIGN_ROOT/pairs/fine_a/null_map/ef_null_initial_data.bin.manifest.json" \
  --marker-crossings "$PARITY_CAMPAIGN_ROOT/pairs/fine_a/null_map/ef_null_initial_data.bin.crossings.csv" \
  --convergence-evidence "$PARITY_CAMPAIGN_ROOT/dynamic_coordinate_convergence.json" \
  --eikonal-substeps "$EIKONAL_SUBSTEPS" \
  --empirical-distribution strict \
  --metric-tolerance 0.005 \
  --stress-tolerance 0.02 \
  --moment-tolerance 0.02 \
  --distribution-tolerance 1e-10 \
  --output "$PARITY_CAMPAIGN_ROOT/dynamic_coordinate_certificate.json"
```

The `analyze_einstein_vlasov_wpic.py parity` command compares compactness
milestones as an exploratory diagnostic. The solved coordinate map supplies
the same-event comparison needed to assess coordinate equivalence.

## One authenticated post-certificate long pair

Use the paired-long helper for one longer Schwarzschild/EF visualization at
the **same fine-A scale, resolution, timestep, and quiet realization** as a
strict parity campaign. This is not a threshold scan, and it does not relax
the threshold helper's requirement of at least two distinct angular momentum
values. No physical or numerical value has a default.

The first submission is a small gate job with
`afterok:$PARITY_CERTIFICATE_JOB_ID`. After the terminal certificate exists,
the gate authenticates its sidecar, numerical evidence, source, binary,
static family, fine-A physical IC, and fine-A null map. It then queues exactly
two children behind its own successful exit: Schwarzschild with `expected=any`
(the polar chart may stop at its gap proxy) and EF with
`expected=black_hole`. Both use `VENUS_BUILD=0` and new output directories.
Parity evidence is never resumed or overwritten.

```bash
cd "$VENUS_SOURCE_ROOT/cpu"
VENUS_RUN_ROOT="$RUN_ROOT" \
CAMPAIGN_ROOT="$NEW_PAIRED_LONG_ROOT" \
PARITY_CERTIFICATE_JOB_ID="$STRICT_CERTIFICATE_JOB_ID" \
PARITY_CERTIFICATE="$PARITY_ROOT/dynamic_coordinate_certificate.json" \
PARITY_FINE_PHYSICAL_IC="$PARITY_ROOT/pairs/fine_a/physical_ic.bin" \
PARITY_FINE_NULL_INITIAL_DATA="$PARITY_ROOT/pairs/fine_a/null_map/ef_null_initial_data.bin" \
HIGH_INITIAL_CONDITION=energy_cutoff_shell \
ENERGY_CUTOFF_AMPLITUDE="$STATIC_C" \
ENERGY_CUTOFF_ELL0="$STATIC_ELL0" \
ENERGY_CUTOFF_CENTRAL_Y="$STATIC_YC" \
RADIAL_INFALL="$FIXED_KICK" \
INFALL_CORE_RADIUS="$FIXED_KICK_CORE" \
ANGULAR_MOMENTUM_SCALE="$CERTIFIED_FINE_A_S_L" \
TARGET_INITIAL_ADM_MASS="$FIXED_ADM" \
SAMPLING_REALIZATION="$CERTIFIED_FINE_A_REALIZATION" \
LONG_MARKERS="$CERTIFIED_FINE_A_MARKERS" \
LONG_RADIAL_CELLS="$CERTIFIED_FINE_A_N_R" \
LONG_DT_MAX="$CERTIFIED_FINE_A_DT" \
LONG_R_MAX="$R_E" \
LONG_T_END="$LONG_END_TIME" \
LONG_OUTPUT_STEPS="$SAVED_OUTPUT_CADENCE" \
EF_MAP_WORLDTUBE_RADIUS="$R_B" \
EF_NULL_DONOR_DURATION="$DONOR_DURATION" \
  bash scripts/slurm/submit_einstein_vlasov_wpic_paired_long.sh
```

The campaign retains the request, strict-certificate audit, exact-reference
provenance attestation, copied certificate and physical IC, final controls,
job receipt, and SHA-256 sidecars. Reusing a campaign root or changing a
staged workflow identity fails before either child can run.

## Bound static-shell angular momentum scan

The threshold helper scans one positively bound static shell with one fixed
inward kick. It supplies neither a numerical `s_L` grid nor a presumed outcome:

```bash
cd "$VENUS_SOURCE_ROOT/cpu"
HIGH_INITIAL_CONDITION=energy_cutoff_shell \
ENERGY_CUTOFF_AMPLITUDE="$STATIC_C" \
ENERGY_CUTOFF_ELL0="$STATIC_ELL0" \
ENERGY_CUTOFF_CENTRAL_Y="$STATIC_YC" \
RADIAL_INFALL="$FIXED_KICK" \
INFALL_CORE_RADIUS="$FIXED_KICK_CORE" \
ANGULAR_MOMENTUM_SCALES="$PREFLIGHTED_POSITIVE_S_L_VALUES" \
TARGET_INITIAL_ADM_MASS="$FIXED_ADM" \
SAMPLING_REALIZATION="$QUIET_REALIZATION" \
SCAN_MARKERS="$N_MARKER_SCAN" \
SCAN_RADIAL_CELLS="$N_R_SCAN" \
SCAN_DT_MAX="$DT_SCAN" \
SCAN_R_MAX="$R_E" \
SCAN_T_END="$SCAN_END_TIME" \
EF_MAP_WORLDTUBE_RADIUS="$R_B" \
EF_NULL_DONOR_DURATION="$DONOR_DURATION" \
NULL_DOMAIN_GATE_SCALE="$PREFLIGHTED_MAP_GATE_S_L" \
PARITY_CERTIFICATE="$STRICT_PARITY_CERTIFICATE" \
CAMPAIGN_ROOT="$SCAN_CAMPAIGN_ROOT" \
  bash scripts/slurm/submit_einstein_vlasov_wpic_angular_scan.sh
```

The list must contain at least two distinct strictly positive values, and the
null-domain gate scale must be one of them. Every `s_L` receives its own
initialize-only bound/setup gate, SC donor, null extraction, and EF run.
Reusing one map across scales is forbidden because changing
`eta=(s_L ell0)^2/2` changes both the matter distribution and the geometry.

Angular momentum supplies a centrifugal barrier, so increasing it can favor
a halo while reducing it can favor collapse. Determine each outcome from
the measured stress balance, matter distribution and trapped surfaces. A halo endpoint must retain positive
realized binding, form no horizon, have small escape, and show sustained
Lagrangian-radius/profile/compactness stationarity plus resolved anisotropic
Einstein--Vlasov Jeans balance for multiple dynamical times. An expanding
high-`L` cloud is `dispersal`, not a halo. A low-`L` black hole endpoint must
contain explicit trapped-surface evidence. Failure to find a horizon in a
short run is `undetermined`.

Three masses remain separate throughout:

1. `fixed_family_adm_mass` is the normalized Schwarzschild `tau=0` family
   mass.
2. `null_slice_conserved_adm_mass` is the donor mass on the extracted null
   slice. And
3. `initial_deposited_adm_mass` is independently reconstructed from the
   imported EF markers at `v_hat=0`.

The donor drift and map/deposition error are recorded independently. Neither
changes the definition of the fixed family.

After all jobs and map postflights pass, run:

```bash
python3 tools/analyze_einstein_vlasov_wpic.py threshold \
  --campaign-root "$SCAN_CAMPAIGN_ROOT" \
  --parity-certificate "$SCAN_CAMPAIGN_ROOT/validated_parity_certificate.json" \
  --output "$SCAN_CAMPAIGN_ROOT/threshold.json" \
  --csv-output "$SCAN_CAMPAIGN_ROOT/threshold.csv"
```

Only a resolution-, timestep-, domain-, and realization-converged
lower-`L` black hole/higher-`L` halo bracket is a threshold result.
Dispersal or undetermined points do not close the bracket. Near a type-I
threshold the dwell time can grow as the bracket narrows, so end time and
late-time halo evidence must also be convergence tested. A final bracket must
be repeated with the second authenticated quiet realization and with enough
domain and duration to exclude boundary loss or a slowly evolving breathing
state.
