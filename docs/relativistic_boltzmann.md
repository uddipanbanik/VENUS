# Relativistic collisions in a spherical gravitational field

## Conventions and symbols

Use $`G=c=1`$ and metric signature $`(-,+,+,+)`$. Radius $`r`$ is areal
radius, $`m(r)`$ enclosed gravitational mass, $`\tau`$ Schwarzschild (SC) time and $`v`$
advanced ingoing Eddington--Finkelstein (EF) time. SC metric functions $`\mu,\lambda`$ satisfy
$`\mathrm ds^2=-e^{2\mu}\mathrm d\tau^2+e^{2\lambda}\mathrm dr^2+r^2\mathrm d\Omega^2`$.
EF functions $`a=1-2m/r,b>0`$ satisfy
$`\mathrm ds^2=-ab^2\mathrm dv^2+2b\mathrm dv\mathrm dr+r^2\mathrm d\Omega^2`$.
Here $`\mathrm d\Omega^2`$ denotes the unit-sphere line element and
$`\mathrm d\Omega`$ in a scattering law is the scattering solid-angle element.
$`T_{rr}`$ is the covariant radial stress component. $`S`$ Is the EF source
combination defined below. The frame subscript $`U`$ denotes the prescribed
regular observer. Packet indices are $`i,j`$. Greek indices label spacetime
components and hatted components refer to a local orthonormal frame.

The proposal rate is $`R`$, acceptance probability $`\mathcal P_{ij}`$,
carried optical depth $`\mathcal T`$, and residual exposure $`h`$. The local
symbols $`A,V`$ in the proposal formula select the coordinate-dependent lapse
factor and cell volume. They are distinct from the coordinate-map derivatives
used in the coordinate reference. Symbols introduced for the material law,
packet weights and frame momenta are defined immediately below.

The collision library evolves elastic scattering together with the
Einstein–Vlasov trajectories and gravitational field. The trajectory update
follows motion through the curved spacetime. At an encounter, the code
converts the two particle momenta to a local orthonormal frame, applies the
relativistic scattering map there, and converts the outgoing momenta back to
the chosen coordinates. The outgoing particles then supply the stress used
to reconstruct gravity.

A material particle has physical rest mass $`m_\chi>0`$. One numerical
particle, called a packet in the library, carries represented rest mass
$`W_i`$. It therefore represents $`q_i=W_i/m_\chi`$ material particles.
The stored four-momentum $`p^\alpha`$ is momentum per unit material mass.
The local collision calculation uses physical four-momentum
$`k^\alpha=m_\chi p^\alpha`$. These two masses have different roles:
$`m_\chi`$ sets the physical scattering kinematics, while $`W_i`$ determines
how much of the distribution the numerical encounter changes.

The spherical sampler currently uses a constant material cross section
$`\Sigma`$ with isotropic centre-of-momentum scattering,
$`\mathrm d\Sigma/\mathrm d\Omega=\Sigma/(4\pi)`$. The separate
[amplitude interface](../cpu/include/ev/relativistic_boltzmann/invariant_amplitude.hpp)
evaluates a supplied scalar scattering amplitude and its differential cross
section. The [manual chapter](venus_manual.pdf#nameddest=sec:relativistic-material-collisions)
derives the invariant gain and loss terms, their coordinate form, and the
discrete updates described here.

## Slice measures and rates

The collision rate must count encounters per unit time in the coordinates
used by the trajectory solver. In Schwarzschild (SC) coordinates the time is
$`\tau`$, the lapse is $`e^\mu`$, and a radial collision cell has proper
volume $`V`$. The stored radial momentum $`w`$ is measured by the static
local observer. With areal radius $`r`$ and specific angular momentum $`L`$,
$`\eta=L^2/2`$ and
$`\epsilon=\sqrt{1+w^2+2\eta/r^2}`$ is local energy per material mass.

In ingoing Eddington–Finkelstein (EF) coordinates the time is $`v`$ and
the metric is
$`\mathrm ds^2=-ab^2\mathrm dv^2+2b\mathrm dv\mathrm dr+r^2\mathrm d\Omega^2`$.
The code stores positive specific canonical momentum $`P=p_r`$ and uses
coordinate shell volume $`V_0`$. This time slice is null, so its particle
measure counts directed crossings of the slice. The metric coefficient
$`b`$ enters when converting the invariant collision term to an evolution
rate per unit $`v`$.

Define the specific invariant pair flux
$`\mathcal F_{ij}=\sqrt{(p_i\cdot p_j)^2-1}`$, where the dot product uses
the metric. The numerical pair encounter rates are

Here $`U`$ is uniform on $`(0,1)`$, and $`\mathcal T`$ is carried optical depth. If the wait exceeds remaining exposure $`h`$, subtract $`Rh`$ and finish the cell. Otherwise advance its coordinate clock by the wait and process one proposal. Empty and one-packet cells advance without collision draws.

```math
\begin{aligned}
&\Gamma_{ij}^{\rm SC}=\max(W_i,W_j)\frac{\Sigma}{m_\chi}\frac{e^\mu\mathcal F_{ij}}{V\epsilon_i\epsilon_j},\\
&\Gamma_{ij}^{\rm EF}=\max(W_i,W_j)\frac{\Sigma}{m_\chi}\frac{b\mathcal F_{ij}}{V_0P_iP_j}.
\end{aligned}
```

The time components of specific momentum explain the different metric
factors: $`p^\tau=e^{-\mu}\epsilon`$ on SC and $`p^v=P/b`$ on EF.
Dividing the covariant kinetic equation by these components converts its
trajectory parameter to the corresponding coordinate time. The partner
momentum measure supplies the other inverse energy or inverse $`P`$
factor. This gives $`e^\mu/(\epsilon_i\epsilon_j)`$ or
$`b/(P_iP_j)`$ in the pair rate. The volumes count the represented material
on their respective slices.

For the local EF scattering map, the prescribed regular observer measures
$`\epsilon_U=(P+\chi/P)/2`$ and $`w_U=(P-\chi/P)/2`$, with
$`\chi=1+2\eta/r^2`$. Its physical energy and radial momentum are
$`m_\chi\epsilon_U`$ and $`m_\chi w_U`$. The
[frame conversion](../cpu/include/ev/relativistic_boltzmann/ef_frame.hpp)
therefore uses the same specific-momentum convention as the pair rate.

## Follow one collision update

The steps below describe one call in the order implemented by the code.
Each radial cell keeps its own collision time and random state. An encounter
changes the particles, and the next proposal uses those changed particles.

**1. Stream self-consistently.** Call [`eb::advance_sc` or `eb::advance_ef`](../cpu/include/ev/relativistic_boltzmann/coupled_evolution.hpp) with a coupled state, requested coordinate interval, maintained field grid, EV evolution controls, and `eb::SplitControls`. The state owns markers, global time, one persistent collision clock per radial cell, and the next unused packet ID. The EV step returns its actual `accepted_dt` and endpoint field. That accepted interval supplies the frozen collision exposure. A zero cross section returns the direct EV update with consistent coordinate clocks.

**2. Assign endpoint cells.** A binary search of increasing mesh faces assigns each streamed radius to its containing cell. The code refills cell packet lists and preserves each clock's random state, residual optical depth, carried number-weight majorant, and counters. Entering packets raise the majorant when needed. Source labels and fixed particle IDs remain distinct from cell membership. SC geometry uses eight-point radial quadrature for proper volume and volume-averaged lapse. EF geometry uses exact coordinate shell volume and the endpoint centre value of `b`.

**3. Advance the proposal clock.** In [`eb::advance_cell`](../cpu/include/ev/relativistic_boltzmann/spherical_collisions.hpp), let $`N`$ be the current packet count and $`q_{\max}`$ the carried upper number-weight bound. Set $`D_{\max}=1`$ on SC and refresh $`D_{\max}=\max_i(\epsilon_{Ui}/P_i)`$ on EF at each proposal loop. With $`(A,V)=(e^\mu,V)`$ on SC or $`(b,V_0)`$ on EF,

```math
R=\frac{A\Sigma}{V}q_{\max}N(N-1)D_{\max}^2,\qquad \mathcal T=-\ln U,\qquad \Delta t_{\rm trial}=\mathcal T/R.
```

**4. Choose and lift a pair.** Unbiased integer rejection draws two distinct indices. Two independent tangential azimuths lift their spherical momenta into physical local four-momenta. Compute the invariant flux and the slice factor $`B_{ij}=\mathcal F_{ij}/(\epsilon_i\epsilon_j)`$ on SC or $`B_{ij}=\mathcal F_{ij}/(P_iP_j)`$ on EF. An independent uniform draw accepts with

```math
\mathcal P_{ij}=\frac{\max(q_i,q_j)}{q_{\max}}\frac{B_{ij}}{2D_{\max}^2}.
```

The unordered selection probability $`2/[N(N-1)]`$ times $`R\mathcal P_{ij}`$ gives the packet hazard above. The code checks the envelope before changing momenta.

**5. Scatter and split.** Two direction draws select an isotropic centre-of-momentum angle. [`rb::scatter_isotropic`](../cpu/include/ev/relativistic_boltzmann/relativistic_collision.hpp) constructs the outgoing future-shell momenta and checks conditioning. Both scattered packets receive $`W_{\min}=\min(W_i,W_j)`$. The larger packet's remainder $`|W_i-W_j|`$ retains its incoming momentum and receives a new ID. Ancestry retains auxiliary marker properties. Represented number and local four-momentum are conserved.

**6. Project and continue.** Divide outgoing local momenta by $`m_\chi`$. Store SC radial momentum $`w`$ and $`\eta=r^2|p_\perp|^2/2`$, where $`p_\perp`$ is transverse specific momentum. EF stores $`P=\epsilon_U+w_U`$, evaluated as $`\chi/(\epsilon_U-w_U)`$ for inward motion to preserve positivity. Renew optical depth after every accepted or rejected proposal, then recompute the current rate. Packet/proposal caps and representable-clock guards bound the update.

**7. Reconstruct the outgoing field.** After all cell batches, the code deposits outgoing stress and solves the Einstein constraints. SC retains its positive polar-areal gap stop. EF retains every active particle and requires the isolated vacuum outer boundary. The EF source uses $`T_{rr}=V_0^{-1}\sum_iW_iP_i`$ and $`S=V_0^{-1}\sum_iW_i\chi_i/P_i`$. It records an independent regular-centre Arnowitt--Deser--Misner (ADM) reconstruction before reapplying the conserved outer ADM condition. The source-feedback residual is returned to the caller.

**8. Commit or retain the incoming state.** The entry points operate on a copy and replace the caller only after the complete update passes. A failed conditioning, capacity, boundary, or field check preserves the incoming markers, clocks, and random state. [`eb::serialize_cell`](../cpu/include/ev/relativistic_boltzmann/eb_restart.hpp) restores a model-bound local collision cell. A coupled restart additionally needs the global markers, field/grid controls, global time, next ID, and all persistent clocks.

## Initialize and call

Include `ev/relativistic_boltzmann/coupled_evolution.hpp` in the caller. For an existing SC marker population, the following initialization supplies the required state. `scgrid` and `scopt` are the maintained field grid and transport options described in the EV manual. `next_unused_id` exceeds all incoming marker IDs, and `material_mass` and `material_area` are the chosen physical model in code units.

```cpp
eb::SCState state;
state.markers = initial_markers;
state.collision_clocks.resize(scgrid.radius.size());
state.next_id = next_unused_id;
eb::SplitControls collisions;
collisions.model.material_mass = material_mass;
collisions.model.cross_section = material_area;
collisions.allow_radial_cell_approximation = true;
scopt.require_vacuum_outer_boundary = true;
const auto result = eb::advance_sc(state, requested_dt, scgrid, scopt, collisions);
```

This example begins at coordinate time zero. For another start time, set both `state.time` and every cell's `coordinate_time` consistently. Choose and record initial per-cell random states before the first call. The EF state uses the same persistent ownership with null-slice markers and `eb::advance_ef`. Its initial data must represent the chosen constant-`v` surface. The returned `result.streaming.accepted_dt` gives the actual exposure, and the state holds the committed endpoint.

## Compilation and physical domain

Compile a caller with C++17, the bundled `cpu/include` path, thread support,
and `-ffp-contract=off`. The library is header based, so an application can
include it directly alongside the maintained Einstein–Vlasov transport.

Component validation covers the horizon-regular EF frame, absolute null
rates, conservation, model-bound cell restart, exact collision-off controls,
and small self-consistent SC/EF source-feedback steps. The default spatial
policy uses co-located radial populations. `allow_radial_cell_approximation`
selects homogeneous finite radial cells and records separated exchanges.
The internal Poisson clock describes the frozen collision suboperator.
Continuous event-to-transport chronology, spatial convergence, and compact-halo
collapse require validation for the chosen physical calculation.
