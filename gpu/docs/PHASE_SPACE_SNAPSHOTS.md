# Particle and gravitational-field snapshots

A snapshot saves the state at a selected time. Phase space combines position
and velocity; a weighted particle carries a specified amount of mass. A
restart is a saved state that can also continue a run. A callback is the
function invoked when the solver reaches an output time. A radial moment is
an integral over particle velocities, such as density or mean velocity.
Begin with the [GPU guide](../README.md) for the usual profile outputs.

Additional terms are defined in the [plain-language glossary](../../docs/TERMS.md).

The production driver can save all weighted particles and the gravitational
field evaluated from them at selected profile times. These files retain the
velocity distribution needed for phase-space and orbital-energy analysis.
Enable the schedule with

```sh
--output-interval 10 \
--phase-space-snapshot-times 0,10,30,100,300,800,2000,5000,10000
```

The default is an empty schedule. Every selected time must be a multiple of the
profile interval. The driver uses the existing output callback, so enabling the
snapshots preserves the orbit steps and collision deadlines. At that callback,
the orbit update and any due collision batch have completed. The driver has
then reconstructed the radial moments from the completed particle velocities.
It saves these particles together with that same field.

Each `phase_space/snapshot_NNNNN` directory contains three files:

* `particles.bin` is the native versioned restart. It retains particle mass,
  physical radius `r`, radial velocity `w`, `eta = L^2/2`, particle identity,
  internal energy, internal velocity dispersion, state, and all restart clocks.
  Here `L` is specific angular momentum. In an expanding run, `w` is peculiar
  radial velocity. In a static run it is physical radial velocity.
* `field.json` stores the actual endpoint potential and force, enclosed source
  masses and line density, deposited moments, physical mesh, inner mass budget,
  boundary conditions, gravitational constant, and expansion parameters. It
  states whether particle masses represent total density or density contrast.
  The potential has the finite-source zero-at-infinity normalization used by
  the Poisson solver.
* `manifest.json` gives the actual endpoint time, requested profile time,
  accepted step, particle count, SHA-256 checksums of both files, the input
  checkpoint checksum, executable checksum, initial run metadata checksum, and
  complete invocation. The requested and actual times make floating-point
  landing error explicit.

The driver first closes the particle, field and manifest files in a temporary
directory, then installs their complete group with one rename. This keeps the
particle state and the field used to interpret it together. A continuation preserves existing snapshots whose particle and
field checksums agree with the repeated endpoint. Existing snapshots later
than the supplied restart cause the same chronology rejection used for radial
profiles. A continuation in a fresh directory starts from its supplied time;
selected times preceding that state remain absent. Future entries in the
schedule remain available for subsequent continuations.

## Reconstructing the distribution function

For a spherical system the physical phase volume is

```math
  d\Gamma = 8\pi^2\,dr\,du\,d\eta,
  \qquad \eta=L^2/2,\qquad u=w+H(t)r .
```

Here `u` is physical radial velocity and `H` is the expansion rate saved with
the field. The two angular integrations give `4 pi` for position and `2 pi`
for the tangential-velocity angle. The remaining velocity measure is
`v_t d v_t = d eta/r^2`, which cancels the radial position factor `r^2`.
The transformation from `w` to `u` has unit Jacobian at fixed radius and time.
Consequently the mass distribution function in a bin is

```math
  f = \frac{\sum_p m_p}
  {8\pi^2\,\Delta r\,\Delta u\,\Delta\eta},
```

where `m_p` is the weight of particle `p` in that bin. In the expanding
density-contrast calculations this reconstructs the represented excess
distribution. The prescribed homogeneous background is a separate component.
A projection over `eta` gives mass per `dr du`; name that marginal quantity on
the colorbar. With a `log10 r` axis, retain the actual `Delta r` normalization,
or explicitly plot mass per `d log10 r du`. Save bin edges, represented mass
and effective particle counts with each analysis product.

## Distribution of orbital energy

For each particle calculate the instantaneous specific orbital energy

```math
  \epsilon = \frac{u^2}{2}+\frac{\eta}{r^2}+\Phi(r,t),
```

where `Phi` is the saved total potential in a static run and the saved peculiar
potential in an expanding run. The `L=0` centrifugal term has its regular zero
limit at the origin. Use the solver's constant-line-mass reconstruction between
mesh faces; this reconstructs the same potential as the force gather. The
saved inner mass and outer finite-source continuation specify the potential
outside the resolved cell centers. Keep the expansion acceleration
`a_ddot_over_a` separate from the halo potential.

The weighted histogram `dM/d epsilon` directly describes the distribution of
orbital energy. Negative and positive energies give the instantaneous binding
classification for this halo potential. Expansion and ongoing relaxation make
the energy time dependent, so escape also requires the subsequent particle
motion and boundary budget. Retain both signs and the full represented mass.

The energy histogram measures mass in each energy interval. To obtain the
phase-space density averaged over that interval, integrate `8 pi^2 dr du d eta` over each energy bin and divide the bin mass by
that phase volume. State the radial and angular-momentum bounds of this
quadrature. The full saved `r,w,eta` distribution retains the velocity
anisotropy needed to interpret this energy average.
