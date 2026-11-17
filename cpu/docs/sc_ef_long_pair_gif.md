# Schwarzschild–Eddington–Finkelstein long-run animation

The animation compares a Schwarzschild run with its corresponding ingoing Eddington--Finkelstein (EF) run.
It uses their saved profiles and the solved coordinate map to evaluate both
solutions at the same physical events.

## When the plot job starts

The strict parity certificate must pass before the long runs can be submitted.
The paired-long gate then starts exactly two evolutions from the same certified
fine-resolution initial state:

- Schwarzschild (SC) coordinates, with no required final outcome because this
  chart may stop as a horizon is approached. And
- ingoing Eddington–Finkelstein (EF) coordinates, which must report a black
  hole for this campaign.

After the gate writes and hashes `submission.tsv`, it calls
`submit_einstein_vlasov_wpic_paired_long_gif.sh`. The plotting job has the
Slurm dependency

```text
afterok:<SC job id>:<EF job id>
```

so it cannot start unless both long evolutions exit successfully. The
submitter and plotting job live in the separate, read-only
`VENUS_PAIRED_LONG_WORKFLOW_ROOT`. The evolution code, map code, and older plot
helpers remain in the frozen `VENUS_SOURCE_ROOT`. Every one of these files is
checked against its submission-time SHA-256 hash before and after rendering.

The plotting job gets both run paths and the certified null-map path from the
paired campaign receipt. It does not accept alternate SC, EF, map, or output
paths. The map manifest authenticates every map artifact by content hash, and
the job also requires those artifacts to remain in the manifest's original
directory.

## Notation

The coordinate map uses SC time $`\tau`$, advanced EF time $`v`$ and areal
radius $`r`$. The metric is
$`\mathrm ds^2=-e^{2\mu}\mathrm d\tau^2+e^{2\lambda}\mathrm dr^2+r^2\mathrm d\Omega^2`$
on SC and
$`\mathrm ds^2=-ab^2\mathrm dv^2+2b\mathrm dv\mathrm dr+r^2\mathrm d\Omega^2`$
on EF, where $`\mathrm d\Omega^2`$ is the unit-sphere line element.
Enclosed gravitational mass is $`m(r)`$ and $`a=1-2m/r`$, in units $`G=c=1`$.
Density $`\rho`$ is measured by the static observer in the exterior.
$`T_{vv}`$ Is the covariant advanced-time stress component and
$`\rho_{\mathrm{EF}}`$ its corresponding EF density. The integer $`n`$ is
the number of integration substeps. Logarithmic plot axes use the stated
simulation units. Density slope means $`\mathrm d\ln\rho/\mathrm d\ln r`$.

## What is compared

Before the SC chart ends, the red EF curve and blue SC curve are evaluated at
the same physical events. Equal coordinate-time numbers are not assumed to
mean equal events. The renderer instead solves the ingoing-light map

```math
\frac{\partial v}{\partial r}
  = \exp(\lambda-\mu)\frac{\partial v}{\partial \tau},
```

where $`\tau`$ is Schwarzschild time and $`v`$ is ingoing EF time. It repeats
the radial integration with $`n`$ and $`2n`$ substeps, uses the finer result,
and checks refinement and forward–inverse composition errors. After the SC
chart ends, the animation continues with EF-only frames so the horizon and
exterior matter remain visible.

The three panels overlay SC (solid blue) and EF (dashed red):

1. Density: horizontal axis $`\log_{10}r`$, vertical axis
   $`\log_{10}\rho`$.
2. Density slope: horizontal axis $`\log_{10}r`$, vertical axis
   $`\frac{\mathrm d\log\rho}{\mathrm d\log r}.`$

3. Compactness: horizontal axis $`r`$, vertical axis
   $`\frac{2m(r)}{r}.`$

The horizontal compactness level $`2m/r=1`$ marks the trapped-surface
condition. The EF density shown before trapping is the same static-observer
quantity used by the parity test,

```math
\rho_{\mathrm{EF}}=\frac{T_{vv}}{a b^2}.
```

The static observer exists in the exterior $`a>0`$. Density and its slope
are plotted in that region, while compactness is plotted throughout the
retained domain.

## Products and fail-closed behavior

All products are first written below a job-specific hidden directory. The job
checks file signatures, exactly 72 frame-ledger rows, both animation phases,
the input paths, current artifact hashes, and provenance. It then publishes
the whole directory with one rename to

```text
<paired campaign>/visualization/
```

The published directory contains:

- `sc_ef_black_hole_profiles.gif`.
- initial, middle, and final PNG previews.
- `sc_ef_black_hole_profiles_frames.csv`, which records the SC and mapped EF
  clocks for each frame.
- `sc_ef_black_hole_profiles_provenance.json`, which records definitions,
  paths, hashes, and map errors.
- `artifacts.sha256`. And
- `render_manifest.tsv` with its SHA-256 sidecar.

If either long job fails, Slurm never starts the plotting job. If any path,
hash, receipt, map check, frame check, or output check fails, the final
`visualization/` directory is not created. An existing visualization is never
overwritten.
