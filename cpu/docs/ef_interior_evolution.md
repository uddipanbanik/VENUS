# Retaining the black-hole interior

The EF driver defaults to `--ef-excision-horizon-fraction 0.1`. At each
constraint solve it limits the excision radius to at most 10% of the outer
apparent-horizon areal radius. It retains the existing negative-a threshold
(`--ef-excision-a -0.01`) and outward cell buffer
(`--ef-excision-buffer-cells 4`). The selected center, every center in its outward buffer, and the intervening
faces must satisfy the threshold, with finite positive b. The retained
boundary is checked again after the endpoint sources and metric are recomputed. If the cap lies in an untrapped
inner cavity, excision waits until a qualifying trapped boundary exists.

The apparent horizon is located by a=0. Excision places an inner numerical
boundary deeper in the trapped region. Particles cross a=0 and remain active
until reaching that boundary. Choosing fraction 1 permits a boundary up to the horizon radius;
the same trapped-buffer conditions apply to every candidate.
The fraction must satisfy 0<F<=1. A radius below the first available cell
cannot produce an excision surface; resolution must be sufficient. The
production pair uses 2048 cells over r=0..48, giving about seven cells within
the final cap r~0.17. This enables an interior diagnostic; it does not by
itself establish convergence of the deepest cells.

With excision enabled, the fixed ADM mass condition retains the gravitational
mass of the excised region. The metric extension inside the excision boundary
serves the numerical scheme; density profiles use retained, resolved cells.

## Evolution through the black-hole interior

Add `--ef-evolve-interior --expected-outcome any` to retain every interior marker.
Each field stage continues to detect the apparent horizon, deposits all
active sources, and applies a zero excision radius. Inward particles also
limit the timestep through their distance to the inner field face. A failed
trial is retried with a smaller timestep. At the numerical limit the driver
saves the last accepted state and records `interior_evolution_limit`.

The summary records the smallest active radius, the first cell center and
outer face, the rest weight inside those scales, and the raw inner-face
values of m, mb, and b. A regular empty central vacuum has m=0. A growing
residual there, or matter concentrated within one field cell, requires
radial and temporal refinement before interpreting a central density profile.
The [full interior description](../../notes/EF_INTERIOR_EVOLUTION.md) gives the
retry, restart, and diagnostic fields.

## Isolated outer boundary

The production solver fixes the outer ADM mass and uses vacuum normalization.
It therefore requires every positive-weight particle to remain within the
outer boundary. An escaping trial is rejected before committing the state.
The last accepted state is saved with `ef_outer_boundary_evolution_limit`.
Increasing the domain permits continued use of the isolated vacuum boundary.
A physical calculation with matter outside a finite outer face requires the
corresponding enclosed mass and exterior stress in its boundary data.

## Horizon-regular matter output

EF profiles append `rho_infall` and `excision_mask`. The density is measured
by the future-directed radial unit timelike observer

    u^v = 1/b,   u^r = (a-1)/2,   u^theta = u^phi = 0.

Its norm in ds^2=-a*b^2*dv^2+2*b*dv*dr+r^2*dOmega^2 is -1 at, outside,
and inside a=0. It is generally not a geodesic observer. With canonical
specific momentum p_r>0 and eta=L^2/2, each active marker contributes

    E = -p_mu*u^mu = (p_r + (1+2*eta/r^2)/p_r)/2
    rho_infall = sum(shape_weight * dm * E^2 / p_r) / shell_volume.

The positive expression avoids subtracting large covariant stress terms.
This expression remains finite through a=0 for finite positive p_r. Evaluate
each marker's energy at its own position, then spread its contribution among
the neighboring cells with the particle shape.
`excision_mask=1` identifies cell centers at or inside the excision boundary.
Particle shapes can contribute to those masked cells from neighboring
retained particles. The mask identifies the region removed from evolution.

A direct SC/EF density comparison requires the same physical observer and
the same events. The EF renderer displays the infalling-observer density
throughout the retained domain using the native advanced-time slices:

```sh
python3 cpu/tools/render_ef_interior_gif.py --run /path/to/ef_run \
  --output ef_interior.gif --v-min 30 --v-max 40 --frames 100
```

It needs numpy, pandas, matplotlib, and Pillow, uses saved slices, and shades
the excised region. History output appends `active_marker_count`,
`active_trapped_rest_weight` (evaluated using a<0 at each active marker), and
`active_min_radius` (-1 if none remain).

## Restart and validation

`ef_interior_controls.txt` binds fresh checkpoints to the excision policy.
Resume uses the same policy. Restart version 2 authenticates the excision
mode; version-1 files denote enabled excision. The policy
sidecar binds the radius cap, threshold, and buffer. Comparisons between
enabled and disabled excision start from the same solved null initial slice.
Appending to a CSV with an older header is also rejected.

Finite component validation covers horizon crossing, retained deep-interior
trajectories, causal buffered excision, untrapped-cavity rejection, observer
normalization, stress contraction and mass bookkeeping. Synthetic infall into
a prescribed central black hole and collapse from a regular shell represent
distinct initial conditions.

Use ordinary IEEE arithmetic with `-ffp-contract=off` for reproducible paired
current cancellation. Production calculations authenticate their physical
initial distribution and mapped null slice before evolution. Cross-platform
floating-point sampling can produce distinct particle realizations.

The radius cap is a convergence parameter. Interior comparisons specify the
same cap, threshold, and buffer, then vary the cap and radial resolution to
assess the resolved matter profile.
