# Halo density and virialization movies

Use $`r`$ for radius, $`\rho`$ for density and $`\gamma_{\rm i}`$ for the
initial positive cusp index. The virial diagnostic $`Q`$ uses physical kinetic
energy $`K_{\rm physical}`$, force virial
$`W_\Phi=-\int r\Phi_r\,\mathrm dM`$, inertia $`I=\int r^2\,\mathrm dM`$ and
expansion rate $`H=\dot a/a`$ for scale factor $`a(t)`$. A dot is a physical
time derivative and $`\Phi_r`$ the radial potential derivative.

These movies follow the Vlasov–Poisson (VP) and Boltzmann–Poisson (BP)
calculations in the supplied figures. VP evolves collisionless matter. BP includes
subhalo encounters with strength 0.1. The initial power-law density index
is defined by $`\rho\propto r^{-\gamma_{\rm i}}`$. Radius, density and time
use the simulation's code units.

| Movie | Physical calculation | Time interval |
|---|---|---:|
| [g03_bp.mp4](g03_bp.mp4) | Expanding BP, γᵢ=0.3 | 0–10000 |
| [g05_bp.mp4](g05_bp.mp4) | Expanding BP, γᵢ=0.5 | 0–10000 |
| [g05_vp.mp4](g05_vp.mp4) | Expanding VP, γᵢ=0.5 | 0–10000 |
| [g08_bp.mp4](g08_bp.mp4) | Expanding BP, γᵢ=0.8 | 0–10000 |
| [g12_bp.mp4](g12_bp.mp4) | Expanding BP, γᵢ=1.2 | 0–10000 |
| [g15_bp.mp4](g15_bp.mp4) | Expanding BP, γᵢ=1.5 | 0–10000 |
| [g20_bp.mp4](g20_bp.mp4) | Expanding BP, γᵢ=2 | 0–10000 |
| [gaussian_static_vp.mp4](gaussian_static_vp.mp4) | Static Gaussian VP | 0–10000 |
| [gaussian_expanding_vp.mp4](gaussian_expanding_vp.mp4) | Expanding Gaussian VP | 0–10000 |
| [gaussian_static_bp.mp4](gaussian_static_bp.mp4) | Static Gaussian BP | 0–5254 |
| [gaussian_expanding_bp.mp4](gaussian_expanding_bp.mp4) | Expanding Gaussian BP | 0–7326 |
| [static_g05_vp.mp4](static_g05_vp.mp4) | Static VP, γᵢ=0.5 | 0–10000 |
| [halo_virialization.mp4](halo_virialization.mp4) | Physical virial moments | 0–10000 |

The density movies place density on the left and its logarithmic slope on
the right. The slope is a quadratic fit to logarithmic density over 21
adjacent native radial cells. Effective marker count is the squared deposited
mass divided by the sum of squared deposited marker contributions. The
power-law movies display bins with at least 32 effective markers. The
Gaussian movies include the positive-density profile and emphasize regions
with at least 32 effective markers. Their colors progress from yellow
to dark blue with time, and the dotted profile shows the initial shell.
Reference lines identify the labelled power laws.

The static and expanding Gaussian calculations share physical initial
positions, velocities, angular momenta, masses and internal dispersions.
The expanding γᵢ=0.5 VP and BP calculations likewise share their initial
orbital state. BP internal energy belongs to the subhaloes and is distinct
from the orbital kinetic energy in the virial ratio. The static γᵢ=0.5
control retains the expanding calculation's physical initial velocities
and uses zero expansion with a reflecting outer boundary.

The virialization movie shows the physical ratio

Here $`K_{\rm physical}`$ is kinetic energy in physical velocities,
$`W_\Phi`$ is the gravitational force moment, and $`I=\int r^2\,\mathrm dM`$.
The background term is the prescribed cosmological acceleration's force
moment. The four expanding power-law curves use resolved native velocity
moments and their constant-line source, including the central mass in the
exterior force. Their radial cross term and inertia use shell centres.
Gaussian curves use emitted all-marker moments of the force applied to their
orbits. Each bound selection uses its own reconstructed gravity. The purple
curve joins the five iterated self-bound selections of the static γᵢ=0.5
control. The brown square marks the static Gaussian endpoint selected by
negative orbital energy in the full potential. These quantities describe
instantaneous states in finite radial domains.

```math
Q=\frac{2K_{\rm physical}}
 {|W_\Phi+(\dot H+H^2)I|}.
```

## Reproduce the movies

From the repository root:

```sh
python -m pip install -r movies/scripts/halo_requirements.txt
python movies/scripts/render_halo_movies.py --case g05_vp
python movies/scripts/render_gaussian_movies.py --case gaussian_static_vp
python movies/scripts/render_gaussian_movies.py --case gaussian_expanding_vp
python movies/scripts/render_gaussian_movies.py --case gaussian_static_bp
python movies/scripts/render_gaussian_movies.py --case gaussian_expanding_bp
python movies/scripts/render_completed_vp_movies.py --case static_g05_vp
python movies/scripts/render_virialization_movie.py
```

The renderers read the supplied plotting caches. Each displayed time belongs
to an actual saved state. Frame ledgers retain the complete physical clocks
and input identities. The Gaussian cache manifest selects inputs from
`movies/data/gaussian` and `movies/data/completed_vp`. The virial renderer
reads `analysis/data/virial_summary`. An installed FFmpeg or `imageio-ffmpeg`
supplies the MP4 encoder. Each renderer accepts `--preview-only` for stills
and `--output-directory` for another destination.

The [movie catalog](data/catalog.json) identifies all supplied movies. The
[halo catalog](data/halos/catalog.json),
[Gaussian cache manifest](data/gaussian/cache_manifest.json) and
[completed VP cache manifest](data/completed_vp/cache_manifest.json) identify
the physical inputs, native grids, density, effective counts and derivatives.
