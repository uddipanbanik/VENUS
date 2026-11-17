# Simulation movies

A profile shows a quantity, such as density, as a function of radius. A
physical clock is the simulation time associated with saved data. A cache
retains data needed to draw frames, and a frame ledger records their times
and source files. A virial comparison measures the balance between kinetic
motion and gravitational forces. Rendering turns the saved data into images.

Additional terms are defined in the [plain-language glossary](../docs/TERMS.md).

The movies below show the example simulations. Each title
gives the actual simulation time. The renderers use saved profiles and their
physical clocks. The supplied caches and frame ledgers retain the full
sampling information.

Evolving profiles progress from yellow at early times to dark blue at late times. Virial comparisons retain each run’s figure color, while their current-time markers follow this temporal palette. The purple static bound curve connects the five converged samples; the Gaussian bound endpoint is an open square.

The main files are MP4 movies. The supplied lossless plotting inputs and
commented scripts reproduce these movies without access to a computing
cluster or to the full particle checkpoints. For the static plots, see the
[figure reproduction guide](../analysis/README.md).

## Rendering

From the repository root, install the analysis dependencies and MP4 encoder:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install imageio-ffmpeg
python3 movies/scripts/render_relativistic_movies.py --help
python3 movies/scripts/render_relativistic_movies.py
```

The default destination is `movies/generated/relativity/`. Use
`--output-directory PATH` to choose another directory, `--case ef` or
`--case schwarzschild` to select one coordinate description, and `--gif`
for an additional animated GIF. A system encoder can be selected with
`--ffmpeg /path/to/ffmpeg`. `--preview-only` exports representative stills.
For the halo and Gaussian density/slope and virialization movies, see
[HALOS.md](HALOS.md). For the SIDM and cold collisionless collapse movies, see
[SIDM_COLD.md](SIDM_COLD.md). Those guides give the case keys, commands,
sampling masks and physical definitions.

## Movie catalog

| Movie | Figure(s) | Evolution shown |
|---|---|---|
| [Gamma 0.3 BP](g03_bp.mp4) | 3 | Expanding halo density and slope, t=0–10000 |
| [Gamma 0.5 BP](g05_bp.mp4) | 3, 5–7 | Expanding halo density and slope, t=0–10000 |
| [Gamma 0.5 VP](g05_vp.mp4) | 5 | Collisionless expanding halo, t=0–10000 |
| [Gamma 0.8 BP](g08_bp.mp4) | 3 | Expanding halo density and slope, t=0–10000 |
| [Gamma 1.2 BP](g12_bp.mp4) | 4 | Expanding halo density and slope, t=0–10000 |
| [Gamma 1.5 BP](g15_bp.mp4) | 4 | Expanding halo density and slope, t=0–10000 |
| [Gamma 2 BP](g20_bp.mp4) | 4 | Expanding halo density and slope, t=0–10000 |
| [Halo virialization](halo_virialization.mp4) | 2 | Eight physical virial datasets, including five converged power-law selections and one Gaussian selected endpoint |
| [Gaussian static VP](gaussian_static_vp.mp4) | 9 | Narrow-shell collisionless density and slope, t=0–10000 |
| [Gaussian expanding VP](gaussian_expanding_vp.mp4) | 9 | Narrow-shell collisionless density and slope, t=0–10000 |
| [Gaussian static BP](gaussian_static_bp.mp4) | 9 | Collisional narrow shell, t=0–5254 |
| [Gaussian expanding BP](gaussian_expanding_bp.mp4) | 9 | Collisional narrow shell, t=0–7326 |
| [Static gamma 0.5 VP](static_g05_vp.mp4) | 2 | Density and slope with the expanding halo's physical initial velocities, t=0–10000 |
| [Cold static collapse](cold_static_vp.mp4) | 8 | Collisionless collapse and virialization, t=0–3000 |
| [Cold expanding collapse](cold_expanding_vp.mp4) | 8 | Collisionless collapse with cosmological expansion, t=0–3000 |
| [Initially NFW SIDM](sidm_nfw.mp4) | 10 | Density and radial velocity variance, t/t_s=0–193.98405 |
| [Initially cored SIDM](sidm_cored.mp4) | 10 | Core collapse and the outer dispersion maximum, t/t_s=0–68.41751 |
| [Schwarzschild collapse](black_hole_schwarzschild.mp4) | 11 | Density, compactness and lapse, tau=0–160 |
| [EF collapse](black_hole_ef.mp4) | 11, 12 | Horizon formation and crossing, v=0–32.49976 |

Figures 6 and 7 show the phase-space and energy distributions of the same
gamma=0.5 BP calculation followed by its density movie. The cold VP movies include physical kinetic, force and inertia histories
through time 3000.

BP denotes Boltzmann–Poisson evolution with local subhalo encounters;
VP denotes collisionless Vlasov–Poisson evolution. Gamma is the initial
positive density index. The gamma=2 movie ends at the actual endpoint shown
in the supplied figures. The SIDM guides state the collision-locality limits of the
reported core-collapse demonstrations. The complete media and input
checksums are included in the repository source manifest.

## Black-hole collapse

The EF movie uses 4,096 radial cells and 228 saved profiles through advanced
time 32.4997608. All 65,536 particles continue to evolve after the apparent
horizon forms. A blue vertical dashed line identifies the instantaneous
outer apparent horizon in both panels. The density is the energy density
measured by the horizon-crossing observer defined in the manual;
compactness is `2m/r` in units `G=c=1`. The trapped rest-mass fraction is
computed from the contemporaneous particle history. The displayed interval
ends while the density maximum is resolved on the radial grid.

The Schwarzschild movie uses 2,048 radial cells and 281 saved profiles
through coordinate time 160. Its density is measured in the Schwarzschild
orthonormal frame. The lower panels show maximum compactness and minimum
lapse through the displayed time. The decreasing lapse explains why these
time slices approach the horizon asymptotically. Both calculations start
from the same physical shell, represented on their respective initial
slices.

The input manifests give native source checksums, actual saved steps,
physical time ranges and plotting-array checksums. The full equations,
coordinate conventions and numerical accuracy criteria are in the
[manual](../docs/venus_manual.pdf).
