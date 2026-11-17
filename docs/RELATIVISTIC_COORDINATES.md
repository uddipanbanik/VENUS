# Einstein--Vlasov coordinates and collapse diagnostics

Einstein--Vlasov couples collisionless matter to relativistic gravity.
Schwarzschild (SC) and ingoing Eddington--Finkelstein (EF) are coordinate
choices for that evolution; EF remains regular at a black hole horizon.
Polar--areal coordinates use a radius whose sphere has area `4πr²`. A
null slice follows a lightlike surface; advanced time labels incoming light
surfaces. The metric specifies spacetime distances and times. Its constraint
equations relate the gravitational field to the matter on a slice.
Throughout the formulas below, `G=c=1`; `m` is enclosed gravitational mass
and `r` is areal radius, so `2m/r` is dimensionless compactness.

Additional terms are defined in the [plain-language glossary](TERMS.md).

Run commands from the repository root. Begin with the [CPU guide](../cpu/README.md)
and the [named module catalogue](../README.md#choose-a-module) for build, input
and output examples. This reference develops the coordinate map and the
conditions for interpreting collapse and halo diagnostics.

## Einstein--Vlasov coordinate modules

The compatibility header
[`cpu/include/ev/weighted_einstein_vlasov_pic.hpp`](../cpu/include/ev/weighted_einstein_vlasov_pic.hpp)
exposes two distinct implementations:

- [`weighted_einstein_vlasov_pic_schwarzschild.hpp`](../cpu/include/ev/weighted_einstein_vlasov_pic_schwarzschild.hpp)
  evolves the polar--areal variables and stops before
  $`2m/r=1`$, where that chart becomes singular;
- [`weighted_einstein_vlasov_pic_eddington_finkelstein.hpp`](../cpu/include/ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp)
  evolves the canonical future-directed null-slice momentum $`P=p_r>0`$,
  solves the neutral advanced-null constraints, detects $`a=1-2m/r=0`$,
  and evolves matter through the apparent horizon.

The bound equilibrium generator is separate again:
[`weighted_einstein_vlasov_static_shell.hpp`](../cpu/include/ev/weighted_einstein_vlasov_static_shell.hpp)
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


## Relativistic coordinate comparison

To compare the same high-compactness system in both coordinates, first
construct its particles on a spacelike slice. The null extractor follows
their worldlines to one ingoing null slice and writes the particles and
metric there. The EF run starts from that saved null slice. The upper-case shell variables below are symbolic
placeholders fixed by successful local preflight; they are not shipped
production defaults:

```bash
cpu/venus_cpu_einstein_vlasov_wpic_black_hole --help
cpu/venus_cpu_einstein_vlasov_wpic_black_hole \
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
cpu/venus_cpu_einstein_vlasov_wpic_black_hole \
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
