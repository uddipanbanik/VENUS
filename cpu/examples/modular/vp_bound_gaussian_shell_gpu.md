# Bound Gaussian shell for single-GPU VP and BP

VP means Vlasov--Poisson: collisionless evolution in collective Newtonian
gravity. BP means Boltzmann--Poisson and also includes encounters. A marker
is a numerical particle carrying mass. A Gaussian shell has a bell-shaped
density profile in radius. A sidecar is a companion file recording the grid
and background conventions. Commands below start at the repository root.
For a short installation example, begin with the [GPU guide](../../../gpu/README.md).

Additional terms are defined in the [plain-language glossary](../../../docs/TERMS.md).

Generate the two background representations of one physical marker realization:

```bash
python3 cpu/tools/generate_bound_gaussian_shell_restart.py \
  --output-directory output/gaussian_ic --particles 1048576
```

The output directory must be new. The density is Gaussian in radius, centered at
10 with width 1 and truncated to 7 < r < 13. Sampling includes the spherical mass
Jacobian. Radial velocity has zero physical mean. All 1048576 markers carry equal
mass, with total mass 3.5968447592701334. Static and expanding files share physical
positions, velocities, angular momenta, masses and IDs. The expanding file stores
peculiar radial velocity w=u-Hr, with initial H=1/18. Their sidecars describe the
512-cell logarithmic mesh over 0.01--10000 and each background.

Run the static case with its own restart and matching mesh:

```bash
build/gpu/venus_gpu_vlasov_poisson \
  --restart output/gaussian_ic/static_vp/restart_state.bin \
  --output-directory output/gaussian_static_vp --end-time 10000 \
  --universe static --maximum-timestep 0.0075 --dynamical-fraction 0.04 \
  --radial-cells 512 --radial-minimum 0.01 --radial-maximum 10000 \
  --output-interval 2 --history-every 1000 --restart-every 1000 \
  --phase-space-snapshot-times 0,10,20,30,50,100,300,800,2000,5000,10000
```

Use the expanding representation for EdS evolution:

```bash
build/gpu/venus_gpu_vlasov_poisson \
  --restart output/gaussian_ic/expanding_vp/restart_state.bin \
  --output-directory output/gaussian_expanding_vp --end-time 10000 \
  --universe eds --maximum-timestep 0.0075 --dynamical-fraction 0.04 \
  --radial-cells 512 --radial-minimum 0.01 --radial-maximum 10000 \
  --output-interval 2 --history-every 1000 --restart-every 1000 \
  --phase-space-snapshot-times 0,10,20,30,50,100,300,800,2000,5000,10000
```

Each run uses a fresh output directory, fixed-cap KDK with output/final-time
shortening, total-density static sourcing or EdS density-contrast sourcing, origin
parity and a reflecting outer wall. The profiles show the collapsing shell and subsequent redistribution of
matter. Complete particle and field snapshots allow virial moments and
orbital binding to be calculated from the same state. The reflecting wall
defines a finite domain, so follow the outer particles when interpreting
late profiles. These two commands evolve collisionless VP.

For a short numerical check, use end-time 1, output-interval 1 and snapshot-times 1.
Repeat with maximum-timestep 0.00375 and a separate fresh output directory. CPU/GPU field and short-step comparisons assess the same physical initial
population. The plotting data are supplied as independent caches.

## Subhalo encounters

The same background-specific restart initializes BP. Use the subhalo task
with the VP mesh, output and orbit controls above, and append:

```sh
--subhalo-collision-strength 0.1 --subhalo-collision-interval 0.5 \
--subhalo-collision-timescale-fraction 0.05 --subhalo-collision-cfl 0.2 \
--subhalo-collision-maximum-substeps 1048576 \
--subhalo-adaptive-cell-timestep --subhalo-adaptive-batch-substeps 256 \
--subhalo-inner-collision-cells 1 --subhalo-collision-lanes 8 \
--subhalo-locality-maximum-ratio 1 --subhalo-velocity-scale 0.1 \
--subhalo-plummer-area-per-mass 1 --subhalo-low-speed-regularization 0.05 \
--coupled-receiver-clocks
```

The executable is `build/gpu/venus_gpu_subhalo_boltzmann_poisson` and each
calculation uses a fresh output directory. The [clock guide](../../../gpu/docs/COUPLED_RECEIVER_CLOCKS.md)
derives the optional field and cadence policies. The expanding BP continuation
also uses `--collision-prestep-coordinate-due` with its carried collision age
and random counter. Static BP retains the default due rule. Applying the
coordinate-close rule from the initial state defines a new numerical cadence
comparison of the same physical initial population. The saved profiles
and actual physical clocks are supplied with the example data.
