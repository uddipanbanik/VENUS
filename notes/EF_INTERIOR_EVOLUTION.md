# EF evolution with interior particles retained

EF means ingoing Eddington--Finkelstein, a coordinate system that remains
regular at a black hole horizon. An apparent horizon marks zero outward
expansion of outgoing light on a chosen slice. A trapped region has both
future radial light families converging. Excision removes a central numerical
region behind a causally checked boundary. The metric describes spacetime
geometry; its constraints are equations that geometry and matter must satisfy.

Additional terms are defined in the [plain-language glossary](../docs/TERMS.md).

The ingoing Eddington–Finkelstein driver accepts `--ef-evolve-interior` to
evolve all particles inside an apparent horizon. Apply this option to the EF
command that imports the solved null initial slice. The physical initial data,
radial grid and accuracy controls specify the same calculation. The driver's
default mode enables excision inside a causally checked trapped region.

With `--ef-evolve-interior`, every metric stage sets its applied excision radius
to zero. Active interior particles deposit their stress and follow the EF
characteristic equations while the driver continues to locate the apparent
horizon. Profiles record a zero excision mask. The library controls are:

```cpp
ef::ConstraintOptions constraints;
constraints.enable_excision = false;
ef::EvolutionOptions evolution;
evolution.enable_excision = false;
```

The evolution routine passes this setting to the initial, midpoint and accepted
endpoint metric solves. Here `r` is the areal radius, `m` is the mass
inside that radius, and `a=1-2*m/r` and `b>0` are the EF metric coefficients.
The covariant radial momentum is `P=p_r>0`. The characteristic equations
remain finite at the apparent horizon `a=0` for finite positive `P` and `b`.
The interior density and particle trajectories can therefore be followed through
horizon formation over the interval resolved by the field grid and timestep.

An inward particle limits the timestep by its positive distance to the inner
field face, divided by its inward radial speed and multiplied by the Courant–Friedrichs–Lewy (CFL)
safety factor. This bound resolves the remaining coordinate domain inside the first
cell. Positive radii below the first cell center remain accessible. A trajectory
that crosses the inner face causes the trial to be retried with a shorter step.
The EF characteristic equations apply throughout this interval.

Recognized characteristic or constraint failures are retried with half the
timestep, up to 24 halvings and the representable-time limit. If these retries
are exhausted, the driver saves the complete last accepted state and records
`interior_evolution_limit`. The failed trial leaves the accepted particles and
metric intact. `--expected-outcome any` permits this incomplete outcome to be
reported with its measured horizon history.

The `ef_evolution_status` summary gives the final accepted advanced time,
requested-time completion, termination cause, last accepted and failed trial
timesteps, minimum active radius, first cell center and outer face, and active
weight below these resolution scales. It also records the raw inner-face
values of the mass `m`, product `m*b` and metric coefficient `b`. An empty central vacuum at a regular origin has
`m(0)=0`. Radial and timestep comparisons establish the accuracy of a growing
numerical residual there. Write the total gravitating mass as `M_ADM`,
the Arnowitt–Deser–Misner mass. An absolute residual `|m(0)|/M_ADM >= 1` records
`spatial_resolution_lost`; smaller residuals retain `requires_convergence`.
The saved peak width and radius, measured in cell units, show how many cells
resolve the collapsing core.

EF restart version 2 stores the evolution mode in the binary payload and its
integrity fingerprint. The interior-controls file records the same mode. A
resume requires consistent mode and controls. Version-1 restarts encode enabled
excision. An interior calculation requires all particles that will be followed
inside the horizon to be present in its imported null slice or checkpoint.

The isolated production driver requires its outer boundary to remain in vacuum.
It sets `EvolutionOptions::require_vacuum_outer_boundary=true`. A positive-weight
particle leaving that boundary rejects the complete trial before particles or
fields are committed. The driver saves the last accepted state and records
`vacuum_outer_boundary_limit` with outcome `ef_outer_boundary_evolution_limit`.
A larger domain supplies a longer isolated continuation.

This condition follows from the boundary values `m(r_max)=M_ADM` and
`b(r_max)=1`, which represent a vacuum exterior. An outgoing massive shell
changes the mass enclosed by a finite boundary. Following that shell outside
the domain requires an exterior field model. The reusable library option
`require_vacuum_outer_boundary` defaults to false, allowing callers to specify
such an exterior model explicitly.

Kernel tests follow a future-directed radial particle through the horizon and
check finite self-consistent sources with its weight retained. Driver tests
cover interior evolution, restart mode integrity, zero profile masks and
last-accepted-state output at an evolution limit. Their numerical limits are
interpreted through the same spatial and temporal resolution diagnostics.
