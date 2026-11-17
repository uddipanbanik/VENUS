# self-interacting dark matter (SIDM) core collapse and cold collisionless collapse

These four movies follow the SIDM calculations in Figure 10 and the cold
collisionless calculations in Figure 8. Native density profiles, centered
radial velocity variance and particle moment histories supply the plotted
quantities.

| Case key | Movie | Physical time range |
|---|---|---:|
| `sidm_nfw` | [Initially NFW halo](sidm_nfw.mp4) | 0–193.98405 |
| `sidm_cored` | [Initially nearly cored halo](sidm_cored.mp4) | 0–68.41751 |
| `cold_static_vp` | [Cold static Vlasov–Poisson halo](cold_static_vp.mp4) | 0–3000 |
| `cold_expanding_vp` | [Cold expanding Vlasov–Poisson halo](cold_expanding_vp.mp4) | 0–3000 |

## Reproduce the movies

Run these commands from the repository root. Python needs NumPy, Matplotlib
and Pillow. FFmpeg provides H.264 encoding. Select it through `--ffmpeg`,
`VENUS_FFMPEG`, the executable search path or the optional `imageio_ffmpeg`
package.

```sh
python3 movies/scripts/render_sidm_cold_movies.py --case sidm_nfw
python3 movies/scripts/render_sidm_cold_movies.py --case sidm_cored
python3 movies/scripts/render_cold_vp_movies.py
python3 movies/scripts/render_cold_vp_movies.py --case cold_expanding_vp
```

The SIDM inputs are in `movies/data/sidm_cold`, and its generated outputs
go to `movies/generated/sidm_cold`. The cold VP inputs are in
`movies/data/cold_vp`, and its outputs go to `movies/generated/cold_vp`.
The paths are relative to the installed scripts. Explicit paths also work:

```sh
python3 movies/scripts/render_cold_vp_movies.py \
  --input-directory movies/data/cold_vp \
  --output-directory movies/generated/cold_vp \
  --ffmpeg /path/to/ffmpeg
```

The SIDM renderer also accepts both cold case keys and delegates them to
the cold VP renderer. Its `--cold-input-directory` option selects those
inputs. The separate cold renderer renders both cases when `--case` is
omitted. The `--preview-only` option exports the final still to the selected
output directory.

Each moving curve uses one native profile and prints its physical time.
The first frame shows the initial state, and the next frame advances in
time. A final hold aids viewing. Relative filenames, native SHA256 hashes
and physical times accompany every frame in the output ledgers.

## SIDM density and centered radial velocity variance

The NFW case starts from an isotropic, energy-truncated NFW distribution
with parent concentration 30 and four million weighted particles. The
nearly cored case starts from density proportional to
`x^(-0.1)*(1+x^4)^(-0.975)`, where `x=r/r_s`, and uses one million equal-mass
particles evolved with spherical direct simulation Monte Carlo.

For each model, `M_s` is its mass scale and `r_s` its radius scale. The
gravitational constant and scales satisfy `G=M_s=r_s=1`. The density scale
is `rho_s=M_s/(4*pi*r_s^3)`, the squared velocity scale is
`v_s^2=G*M_s/r_s`, and the time scale is `t_s=r_s/v_s`. The constant cross
sections per mass are `20.1062*r_s^2/M_s` and `32.3723*r_s^2/M_s` for the
NFW and nearly cored cases, respectively.

Density is shown on the left. The right panel shows the centered radial
variance, `sigma_r^2=<v_r^2>-<v_r>^2`, which separates random radial motion
from bulk radial flow. The weighted particle-in-cell (WPIC) input supplies `radial_variance` directly.
For the nearly cored case, the saved `p` column equals `rho*sigma_r^2`, so
the variance is `p/rho`. Its particle count per bin follows from successive
cumulative-mass increments divided by particle mass.

Heavy curves contain at least 32 effective markers for WPIC or particles
per bin for the nearly cored case. Thin curves contain eight through 31.
Smaller populations are omitted. Black dashed curves show the initial
state. Colors identify successive states, and the final nearly cored frame
marks the outer variance peak. The movies show core collapse and
heating together with the outward-moving outer peak.

The frames report the largest collision-shell width divided by mean free
path at that update. These calculations are finite-resolution
demonstrations. Their complete histories reach maximum ratios of about
38400 and 390 for the NFW and nearly cored cases. Quantitative collapse
times and late central structure require collision regions that resolve
the mean free path and checks at finer resolution and shorter timesteps.

## Cold static and expanding collapse

Both halos start with density index 0.5 and physical velocities reduced by
a factor of 0.25. Their velocity distributions are initialized separately
for their respective backgrounds. The logarithmic field grid has 256 cells,
first positive face 0.02 and an outer reflecting wall at radius 1000. Origin
parity allows central passage. The static background has `H=0`. The
expanding background has `H=2/[3*(t+12)]`, where `H` is the Hubble rate.
Both movies show all 601 native profiles from time 0 to 3000 in steps of 5.
At 24 frames per second, the delivered movies last about 27 seconds,
including a two-second final hold.

The upper panels show density and its signed logarithmic derivative. A
quadratic fit to log density over 21 native cells gives the derivative.
Its centered stencil omits the first and last ten centers. Thin curves
retain positive native densities and finite derivative estimates. Thick
sections require at least 32 effective markers at the displayed center.
Gray levels mark slopes −1, −1.5 and −3.

The lower panels follow all particles. Physical kinetic energy is
`K_physical=sum(m_p*|u_p|^2)/2`, where `u_p` is physical velocity. The
force moment is `W_physical=W_Phi+(H_dot+H^2)*I`, with
`W_Phi=-sum(m_p*r_p*dPhi/dr)` and `I=sum(m_p*r_p^2)`. Here `m_p` and
`r_p` are marker mass and radius, `Phi` is the halo potential, `I` is the
moment of inertia, and `H_dot` is the time derivative of `H`. The ratio is
`2*K_physical/|W_physical|`. The native particle histories provide these
moments. True initial-particle measurements supply their time-zero values.
The blue vertical line marks the current profile time.

At time 3000 the static full-population ratio is 1.14989. The bound subset
has negative orbital specific energy in the full potential, with zero at
infinity. Its blue virial point uses the selected population's own
gravity and gives 0.99672. The blue inertia point uses the same selection,
including its weakly bound outer particles. The red open point contains
all positive-energy particles, in both radial directions. They carry
99.57% of the full population's inertia. The first wall impact lies in
`2590<t<=2595`, and 1896 impacts are recorded by time 3000. The gray vertical
guide marks the lower edge of that observed interval. These wall effects
accompany the static full-population moments.

The expanding ratio reaches 1.00334 at time 3000, with zero recorded wall
impacts. Its moment of inertia is 44364.35 and continues to grow while the
instantaneous virial ratio approaches unity. Its panels describe the
complete population throughout the evolution.

## Native plotting inputs

`data/sidm_cold/native_sidm_manifest.json` gives SIDM physical settings,
frame times and native profile hashes. Its `reference_profiles.json`
identifies the profiles shown in the supplied figures. The cold inputs have their
own `data/cold_vp/catalog.json`: complete compressed native profiles,
physical moment histories, initial moments and final measurements. Each
compressed profile decompresses to its original CSV bytes. Frame ledgers
retain each physical time and input checksum, so the supplied movies can
be reproduced from these plotting inputs.
