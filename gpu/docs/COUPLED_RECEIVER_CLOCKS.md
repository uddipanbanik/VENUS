# Coupled subhalo field and collision clocks

A collision clock tracks elapsed encounter evolution. A field clock estimates
how quickly gravity changes at selected locations, called receivers. Coupling
these clocks lets the timestep respond to both processes. A checkpoint is a
saved simulation state. KDK means kick--drift--kick: update velocity, advance
position, then update velocity again. VP means collisionless Vlasov--Poisson;
SIDM means self-interacting dark matter. This advanced option belongs to the
subhalo module introduced in the [GPU guide](../README.md).

Additional terms are defined in the [plain-language glossary](../../docs/TERMS.md).

The single-GPU subhalo solver offers an optional field-clock policy through
`--coupled-receiver-clocks`. The option requires positive subhalo strength,
`--subhalo-adaptive-cell-timestep`, a positive collision-timescale fraction and
a fresh output segment. The particle checkpoint carries the accumulated
collision age, sampler counter and timescale estimates into that segment.
The option defaults to off. Collisionless VP retains its fixed maximum KDK
step and output/final-time clipping. Material-SIDM and MPI tasks retain their
separate interfaces.

## Orbit and field times

The gravitational orbit bound is

```math
\Delta t_{\rm grav}=\min\!\left[\Delta t_{\max},
 f_{\rm dyn}\min_i\left(|\Phi_{r,i}|/r_i+H_*^2\right)^{-1/2}\right],
\qquad H_*=\frac{2}{3(t+12)}.
```

Here `f_dyn` is `--dynamical-fraction`. The minimum includes cells whose
deposited mass exceeds the existing `10^-12` fraction of deposited source
mass. Every particle still sources gravity and follows the characteristic
update. The angular drift is integrated analytically. The optional bound uses
the displayed `H_*` term for both supported backgrounds, so static runs also
receive this additional positive frequency envelope.

A receiver is a location weighted by the represented mass that samples the
change in potential or force. The complete receiver norm applies to expanding
EdS data with exponent
`2/3`, time offset `12`, zero central mass, TSC deposition, retained interior
mass and an already background-subtracted source. Other backgrounds use the
fallback maximum relative field-change clock. In particular, static subhalo
runs use that maximum-norm clock while retaining the optional orbit bound.

For potential or radial force values `A`, resolved shells contribute their
deposited masses at shell centers. A particle cloud partly inside the first
mesh face contributes its hidden mass at that particle's radius, where the
field uses the uniform interior continuation. Both contributions enter the
same quadratic sum. The relative field change over observation interval
`h` is

```math
\delta_A^2=
 \frac{\sum_k \omega_k(A_{1,k}-A_{0,k})^2}
 {\max\!\left(\sum_k\omega_k A_{0,k}^2,
                    \sum_k\omega_k A_{1,k}^2\right)}.
```

Resolved shell contributions use their deposited masses, while interior
contributions use the hidden fractions of the particle clouds. Here $`\epsilon_{\rm fp}`$ is binary64 machine precision. The effective
count is computed from resolved shell masses $`M_i`$ and their deposited counts,

```math
N_{\rm eff}=\frac{(\sum_i M_i)^2}
 {\sum_i M_i^2/N_{{\rm eff},i}},\qquad
\epsilon_{\rm samp}=\max(4096\epsilon_{\rm fp},0.25/\sqrt{N_{\rm eff}}).
```

The same sampling floor is retained when interior receivers are added.
A channel with `delta_A <= epsilon_samp` gives an infinite timescale. A
resolved channel gives
`h / sqrt(delta_A^2 - epsilon_samp^2)`. The smaller potential/force timescale
controls the field contribution. A zero quadratic denominator contributes
zero relative change. An unavailable complete receiver measure uses the
fallback maximum-norm clock.

The interior receiver sum must close the deposited interior mass within
`2048 * epsilon_fp` relative error, and total receiver mass must close the
resolved-plus-interior source ledger within `2e-11`. The cloud's resolved
fraction and hidden complement use the same separately rounded operations
as deposition. A failed closure stops the calculation and preserves bounded
field and particle-sample diagnostics.

## Collision chronology

The global collision deadline takes the minimum of the maximum interval and
the configured fractions of the controlling timescales. Within that elapsed
interval, each adaptive collision cell takes the local steps required by its
collision CFL. The driver first tries a KDK orbit endpoint, reconstructs its
field and recomputes the deadline. It accepts the endpoint when the carried
collision age fits that deadline. Otherwise it restores the incoming particles
and field, then shortens the orbit interval or applies the due collision batch
at the current state. Locality and collision-exposure limits apply throughout.

The additional option `--collision-prestep-coordinate-due` checks the newly
selected deadline before a KDK trial. Define the accumulated collision age
`a`, selected deadline `d`, and coordinate resolution
`q = 128 * epsilon_fp * max(1, abs(t))`. A batch is applied at the current
state when `a > 0`, `d > q`, `d - a <= q`, and the existing 64-epsilon
deadline-safety condition holds. The batch consumes the carried age through
the unchanged sampler. Unsafe overdue, unresolved-deadline and zero-age
cases stop for diagnosis. The default path keeps its existing
post-step decision.

`--coupled-field-trace` records receiver norms, selected clocks and cadence
decisions. It requires the receiver policy. Its bounded samples identify the
fields entering the time estimate; recording them leaves particle states and
random counters unchanged.

## Gaussian initial data

The [bound shell generator](../../cpu/tools/generate_bound_gaussian_shell_restart.py)
writes static and expanding representations of the same physical particles.
Their stored internal dispersion and energy are also the subhalo initial
values. Each background's restart can therefore initialize either VP or BP;
the task executable and positive collision strength select the evolution.
Use the generated mesh, background and initial-condition sidecars unchanged.
See the [Gaussian example](../../cpu/examples/modular/vp_bound_gaussian_shell_gpu.md)
for the common initial distribution and VP commands.

The implementation resides in `coupled_receiver_clocks.hpp`,
`receiver_sums_math.hpp`, `receiver_sums_host.hpp`, `receiver_sums_kernel.cu`,
the shared GPU driver and the CPU cadence helper. Spatial resolution,
collision-CFL sensitivity and the accuracy of the finite field-time policy
are separate scientific checks for a new distribution.
