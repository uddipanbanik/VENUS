# VENUS example figures

The twelve example figure pairs are in [../plots](../plots/README.md). This
directory contains the analysis scripts and reduced simulation output used
to generate them. The PDF files provide vector graphics. The PNG files
provide previews.

The mathematical profiles below use physical radius $`r`$ in the stated
code units. Their numerical constants specify the supplied Gaussian shell.

## Reproduce the figures

Run these commands from this `analysis/` directory, using Python 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce_examples.py
python scripts/verify_reproduction.py
```

The entry point writes twelve PDF/PNG pairs to `reproduced/`. It authenticates the
reduced input files against `data/catalog.json` before drawing. NumPy,
Matplotlib, Pandas, and Pillow are the only Python dependencies. Matplotlib's
bundled fonts and MathText supply the figure typography. The plotting code
runs locally on a CPU, using the supplied sorting records and simulation output.

Supply other directories and select comparisons through the command line:

```sh
python scripts/reproduce_examples.py --data-dir data --output-dir reproduced
python scripts/verify_reproduction.py --figure 3 6 7 \
    --data-dir data --reference-dir ../plots --output-dir reproduced
```

All paths can be absolute or relative to the current directory. The default
paths are resolved relative to the script, so the scripts also work when
called from another directory. Each renderer is a separate function.
Importing the module makes these functions available for further analysis.

The pinned versions in `requirements.txt` reproduce all twelve reference PNGs
pixel for pixel in the recorded Python 3.12.14 environment. Regenerated PDFs
have identical content and a new creation timestamp. The verification script
compares every PNG channel and compares PDF bytes after removing that timestamp.

## Figures and input data

| Figure | File stem | Physical calculation and retained data |
| --- | --- | --- |
| 1 | `sorting_algorithms` | Insertion sort and stable radix sort on four fixed-ID records, with a decimal digit example. |
| 2 | `halo_virialization_summary` | Physical-Q curves for expanding Boltzmann--Poisson (BP) indices 0.3, 0.5, 0.8, expanding Vlasov--Poisson (VP) index 0.5 and both Gaussian VP cases through time 10000. The purple curve gives five converged static index0.5 self-bound selections. An open square gives the single negative-energy static Gaussian selection using its own gravity. |
| 3 | `halo_multigamma_density_slopes_shallow` | BP profiles for initial indices 0.3, 0.5, and 0.8, at four saved times per case; all three rows use CUDA. |
| 4 | `halo_multigamma_density_slopes_steep` | CUDA BP profiles for initial indices 1.2, 1.5, and 2, all through time10000. |
| 5 | `halo_g05_bp_vp_density_slopes` | Index-0.5 CUDA BP and VP calculations with common initial orbital coordinates and saved times through 10000. |
| 6 | `g05_bp_phase_distribution` | Radius and physical radial velocity in the index-0.5 CUDA BP calculation at times 0, 30, 300, 2000, 5000, and 10000. |
| 7 | `g05_bp_energy_distribution` | Phase-volume mean distribution and its logarithmic energy derivative for the same six times. |
| 8 | `cold_vp_static_expanding_virial` | Cold VP collapse in static and expanding backgrounds through time 3000: physical virial ratio, inertia, and static energy selections. |
| 9 | `gaussian_vp_full_radius_density_slopes` | Gaussian VP and BP shell evolution in static and expanding backgrounds, with density and logarithmic slopes through collapse and rebound. |
| 10 | `sidm_collapse_and_outer_dispersion` | Self-interacting dark matter (SIDM): a lowered NFW halo with weighted particle in cell (weighted particle-in-cell (WPIC)) and a nearly cored halo with spherical direct simulation Monte Carlo (SphDSMC). |
| 11 | `bh_coordinates` | Einstein–Vlasov collapse in Schwarzschild and ingoing Eddington–Finkelstein (EF) coordinates: compactness, lapse, horizon radius, and retained rest mass. |
| 12 | `bh_interior` | Ten EF density and compactness profiles through apparent-horizon formation, with horizon radii at the corresponding saved times. |

`data/catalog.json` records the figure order, selected times, physical run
parameters, units, data columns, and SHA256 hashes. Each reduced table also
records its native source filename and hash. The compressed CSV files contain
17-digit decimal values that preserve the plotted binary64 values exactly.
They retain the native radial bins used in the plots. Figures 3 and 5 use the same index-0.5 CUDA BP profiles.
The separate CPU profile groups retain the supplementary density comparisons. `data/catalog_figure2.json` selects the shallow-index profile inputs.
The two NPZ cache families supply Figures 6 and 7. Their compact manifests
record times, total masses, input identity and cache hashes. The source
analysis functions are documented in the solver manual.

The catalog provides the configuration of the plotted calculations. These
reduced products support figure regeneration and analysis of the displayed
quantities. Continuing a simulation uses the native restart formats described
in the code documentation.

## Quantities, units, and selections

**Sorting algorithms.** Figure 1 uses the four fixed-ID records in
`data/sorting_algorithms/records.json`. Gray underlines show the ordered
insertion prefix. Shaded frames show occupied decimal radix buckets, with
incoming order retained within each bucket. Each key retains its record
properties. The actual radius routine uses eight stable byte passes with
256 buckets on finite nonnegative binary64 keys.

After the latest drift, a permutation orders particles by increasing current
radius. For direct spherical-shell gravity, summing masses at smaller ranks
gives the enclosed mass from other shells. SphDSMC adds half of the current
shell's own mass when computing its force. WPIC uses radius ordering to group
nearby collision partners and computes its gravitational field from mass
deposited on the radial mesh.

**Halo profiles.** Radius, mass density, and time use code units with
gravitational constant `G=1`. The expanding calculations use physical radius,
peculiar velocities, expansion exponent `alpha=2/3`, and time offset `t0=12`.
Their initial resolved mass is 0.8. The initial density is a regularized,
tapered power law with central scale 0.01, taper radius 9, and taper width 1.
The BP encounter strength is 0.1. The VP calculation has zero encounter
strength. The 256-cell logarithmic field mesh spans radii 0.1 to 1000.

The logarithmic slope is the derivative of a quadratic fit to
`log10(density)` against `log10(radius)` over 21 adjacent native cells.
Thin curves retain positive native densities and finite derivatives.
Thick sections require effective marker population `neff >= 32` at their
centers. The entire derivative window requires positive density.
The effective population is the squared sum of deposited marker masses
divided by the sum of their squared deposited contributions. The outer ten
bins at either end have no centered derivative. Conventional power-law
guides use finite radial intervals and free density fits. The guide selection
and tolerances appear in `matching_regimes()`.

**Phase and energy distributions.** Figures 6 and 7 use the CUDA BP
calculation with initial density index 0.5 and the halo units above.
Each marker carries the mass of a represented subhalo and follows its
center of mass. The physical radial velocity is `u=w+H*r`, with `w` the
peculiar radial velocity and `H` the prescribed expansion rate. Specific
angular momentum `L` is stored through `eta=L^2/2`. The phase space mass
measure is `dM=8*pi^2*f*dr*du*deta`. Thus the Figure 6 color is the base-10
logarithm of the finite-bin angular momentum projection
`sum(m_p)/(8*pi^2*Delta_r*Delta_u)`. The radius axis is logarithmic and the
normalization uses the physical radial width. All six panels share a
velocity window containing every marker and a common plasma color scale.
The gray dashed line marks the first field face at radius 0.1.

Figure 7 uses instantaneous specific orbital energy
`E=u^2/2+eta/r^2+Phi(r,t)` and its shift `delta=E-Phi(0,t)`, with the saved
peculiar halo potential `Phi`. The internal energy stored in each subhalo
is separate. The prescribed background and halo potential evolve, so this
orbital energy varies with time. The central potential uses the saved
uniform central continuation, and the physical reflecting wall is at
radius 1000.

The mean phase space distribution is `M_bin/Delta_Omega`, where
`Omega(E)=(16*pi^2/3)*integral_0^1000 r^2*[2*(E-Phi)]_+^(3/2)*dr`,
`[x]_+=max(x,0)`, and `Delta_Omega` is the difference at the two energy
edges. Its logarithmic derivative uses a quadratic fit in nine adjacent
log-energy bins, spanning 0.20846 dex. Every bin in a derivative window
has positive distribution and effective population at least 8. Thick
segments require at least 32 throughout. Dashed segments touch an energy
bin whose phase-volume turning radius at its lower energy edge is below
the first field face or whose
phase volume inside that face exceeds one percent. The left panel applies
these cuts bin by bin. Every marker contributes to the cached energy mass.
The nine cached times provide a common plasma time-color assignment,
and the figures display the six times listed above.

The slope guides -4.5 and -2.5 correspond to density cusps with indices
1.5 and 1. For an isotropic, self-consistent cusp with density proportional
to `r^(-gamma)`, `0<gamma<2`, the finite central potential implies
`dln(f)/dln(E-Phi_c)=-(6-gamma)/(2*(2-gamma))`.
These asymptotic reference levels are labelled by their density slopes.
The measured energy-shell mean retains the anisotropy and time dependence
of the simulated distribution.

**Virial ratios.** Figure 2 uses
`Q=2*K_physical/abs(W_Phi+(H_dot+H^2)*I)`. Here `K_physical` uses
physical velocities, `W_Phi` is the halo gravitational force moment and
`I=integral(r^2 dM)` is the moment of inertia. The four expanding power-law
curves integrate resolved native velocity profiles with the complete central
source. The two Gaussian curves use emitted complete-particle kinetic and
applied-force moments. All six curves reach time 10000.

The purple curve gives the static index 0.5 control that retains the
expanding halo's initial physical velocities. At each complete snapshot,
iterative unbinding reconstructs the selected population's own potential,
zero at infinity, and removes nonnegative-energy particles until membership
is unchanged. The five converged selections at 300, 800, 2000, 5000, 10000
are joined by straight segments at their actual times. Earlier selections at 0, 10, 30, 100 remain unresolved after 32
iterations. The final selected mass is 0.13857136 and its own-field ratio
is 1.09722435. The complete static population has ratio 20.9518 in the
reflecting domain of radius 1000. These are instantaneous diagnostics.
The catalog in `data/virial_summary/` records the input and estimator
scope for every curve and selection. The cold controls appear separately.

Figure 8 uses physical velocities in both backgrounds and the force
moment `W_physical=W_Phi+(H_dot+H^2)*I`, where `W_Phi` is the force
moment from the sampled halo potential. The history ratio is
`2*K_physical/abs(W_physical)`, and `I=sum(m_p*r_p^2)` covers every marker.
Both curves reach time 3000. The static blue endpoint uses negative
instantaneous orbital energy in the complete potential, normalized to
zero at infinity, and its virial ratio recomputes the selected population's
own force. The static red endpoint gives the positive-energy inertia,
including inward wall-reflected matter. A gray curve uses a larger wall.
Vertical guides identify the start of the first-impact bracket at time
2590–2595. The expanding curve has zero wall impacts through time 3000.
Its global inertia continues growing while its physical virial ratio
approaches unity. The two initial distributions share the density and
physical velocity factor 0.25. Their background-dependent tangential
velocity scales give distinct initial physical kinetic energies.

**Gaussian profiles.** Figure 9 compares Vlasov–Poisson and Boltzmann–Poisson evolution of the same Gaussian shell in static and expanding backgrounds. Each row follows eight physical times through collapse, rebound and the subsequent spreading of outer matter. The BP sequences end at their latest displayed profiles, while VP reaches 10000. Colors identify time. Thick portions have `neff >= 32`, and thin curves and points show the remaining positive-density samples. Signed slopes use a quadratic fit over 21 positive native cells. Dashed references mark slopes -3, -1.5 and -1.

The initial density is proportional to $`\exp[-(r-10)^2/2]`$ on $`7<r<13`$. The 1048576 equal-mass markers share physical positions, velocities and angular momenta across all four cases. The zero-mean radial Gaussian velocity and transverse Gaussian velocity have scale 0.08481562, with speed truncation at three times that scale. The grid spans 0.01 to the reflecting radius 10000, with orbit cap 0.0075. BP adds subhalo encounters with strength 0.1, initial internal dispersion 0.1, Plummer area per unit mass 1 and low-speed regularization 0.05. `data/gaussian_vp/initial_distribution.json` gives the normalization.

Figure 2 uses eleven actual full-particle moment samples, separately identified from the density times. Its final complete-population ratios are 1.48084 (static) and 1.01950 (expanding). The static single selection with negative instantaneous orbital energy in the full zero-at-infinity self-potential contains 76.2684 percent of the mass. Its original velocities and reconstructed own gravity give 1.001174. Its membership is retained after the potential changes. The definition and small positive-energy remainder are in `data/gaussian_vp/static_vp/bound_component.json`. GPU outputs provide endpoint occupancy, while cumulative wall-impact counts remain unavailable. `data/gaussian_vp/catalog.json` records the force, background and mass conventions.

**SIDM profiles.** Each model uses `G=M_s=r_s=1`, density unit
`rho_s=M_s/(4 pi r_s^3)`, velocity unit `v_s=sqrt(G M_s/r_s)`, and time unit
`t_s=r_s/v_s`. Thus the renderer multiplies the saved density by `4 pi` for
the plotted density ratio. The `variance` column is the centered radial
velocity variance in units of `v_s^2`. For WPIC it is the native
`radial_variance`. For SphDSMC it is the native centered pressure divided by
density. `population` is the native WPIC effective population or the exact
SphDSMC bin count recovered from cumulative-mass increments. Thin curves
require population at least 8, and thick curves require at least 32. The dot
marks the last cored profile's maximum radial variance at radius above 10.
The subpanels below each density panel use the same logarithmic derivative
from a centered quadratic fit over 21 native cells and population thresholds. Every fitting
window contains positive densities. The horizontal dashed slope reference
-2.19 is the long-mean-free-path SIDM similarity envelope of
[Balberg, Shapiro & Inagaki (2002)](https://arxiv.org/abs/astro-ph/0110561),
Section 4 and Table 1. The measured finite envelopes retain radial variation.

The NFW initial condition is a self-consistent lowered distribution based on
the isotropic NFW fit of [Lane et al. (2022)](https://doi.org/10.1093/mnras/stab3755), with parent concentration 30.
The nearly cored initial density is proportional to
`x^(-0.1) (1+x^4)^(-0.975)`, where `x=r/r_s`, and its distribution is obtained
by Eddington inversion. The catalog gives their masses, finite construction
radii, particle counts, and scattering cross sections per mass. Each row's
times use its own model's `t_s`.

**Relativistic profiles.** The calculations use `G=c=1` and unit particle rest
mass. Radius is areal radius, and compactness is `2m/r`, where `m` is enclosed
gravitational mass. The Schwarzschild lapse is `exp(mu)`. The EF advanced
time satisfies `v=tau+28` at the clock-normalization radius 18, where `tau` is
Schwarzschild time. Rest-mass fractions divide the saved invariant marker
weight by its initial sum. All 65,536 particles are retained in these curves.

Figure 12 uses the density measured by the specified infalling observers,
saved as `rho_infall`. It displays bins above `1e-7` of each snapshot's peak
density. Its dashed vertical lines use the apparent-horizon radius from the
history row with exactly the same time and step as each density slice. The
radius is the outermost zero of the EF metric coefficient located by linear
interpolation of `r^2 a` between its bracketing faces. The ten retained slices
contain all 4096 native cell centers.

## Analyze complete snapshots from your own GPU calculation

The two native-analysis commands read the complete snapshot groups written
by the GPU particle runners. Each group contains `particles.bin`,
`field.json`, and `manifest.json`. Its version-6 native file stores every
marker and its accepted time. The field file stores the actual radial
source, potential, and background at that time. The reader verifies file
hashes, payload size, time and marker count, active positive masses, source
mass accounting, and the potential reconstructed from that source.

For an initial state at time zero, the following example selects four
complete snapshots. Run it from the repository root after building the GPU
runners into `build/gpu`, as described in the code guide. Use the background,
mesh, units, and initial particle data appropriate to the selected runner.

```sh
build/gpu/venus_gpu_vlasov_poisson \
  --restart my_initial_state.bin --output-directory output/my_halo \
  --end-time 300 --output-interval 10 \
  --phase-space-snapshot-times 0,30,100,300
python analysis/scripts/phase_space_snapshot_analysis.py \
  --snapshots output/my_halo/phase_space \
  --output-directory analysis/my_halo_phase
python analysis/scripts/shifted_energy_snapshot_analysis.py \
  --snapshots output/my_halo/phase_space \
  --output-directory analysis/my_halo_energy
```

Each analyzer also accepts one or several individual snapshot directories.
`--maximum-requested-time` selects their time range. The phase analyzer
uses 360 logarithmic radial bins, 240 physical radial-velocity bins, and
1024 linear orbital-energy bins by default. `--radial-bins`,
`--velocity-bins`, and `--energy-bins` set these counts. The shifted-energy
analyzer uses 192 logarithmic energy bins by default, set with `--bins`.
Both commands accept `--edges-from previous_snapshot.npz` to reuse exact
bin edges across comparisons.

The supported sources have positive active mass weights, positive marker
radii, a constant line-mass reconstruction, and potential zero at infinity.
They include the expanding runners' directly supplied density-contrast
weights and the static SIDM runners' total-mass weights. The energy-shell
mean additionally uses a finite central potential with zero central point
mass and a nonnegative attractive source. Logarithmic energy bins require
strictly positive `E-Phi_c`. Every selected marker contributes to the
histograms. The physical radial velocity is `u=w+H*r`, and the internal
energy of a represented subhalo is recorded separately from its orbital
energy.

The commands write one compressed NPZ per snapshot, an
`analysis_manifest.json`, and `analytic_checks.json`. The phase NPZ includes
`xedge=log10(r_edges)`, `uedge`, `mass_rw`, `count_rw`, `neff_rw`, and
`f_integrated_eta`. The energy NPZ includes `delta_edge`, `delta_center`,
`energy_mass`, `bin_phase_volume`, `mean_f_energy`, `neff_energy`, and the
inner-field selection flags. All numerical values retain the input code
units. For the time-300 example above, read and plot the mean distribution
as follows:

```python
import numpy as np
import matplotlib.pyplot as plt
p = np.load("analysis/my_halo_energy/snapshot_00030.npz")
keep = (p["mean_f_energy"] > 0) & (p["neff_energy"] >= 32)
plt.loglog(p["delta_center"][keep], p["mean_f_energy"][keep])
plt.xlabel(r"$E-\Phi_{\rm c}$")
plt.ylabel(r"$\langle f\rangle_E$")
plt.savefig("analysis/my_halo_energy/mean_distribution.png", dpi=200)
```

The snapshot manifests retain their input, executable, and run-metadata
identities. Optional `--expected-input-sha256`,
`--expected-binary-sha256`, and `--expected-metadata-sha256` require supplied
identities. `--run-metadata path/to/metadata.json` Also verifies that file's
bytes. Complete snapshot files are authenticated again after reduction.
Run either command with `--self-check-only --output-directory analysis/check`
to evaluate its analytic normalization and Jacobian checks separately.

The analysis commands' `--self-check-only` mode evaluates their native-format,
source-budget, weighted-projection and analytic phase-volume identities.
