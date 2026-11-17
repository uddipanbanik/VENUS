# GPU weighted particle-in-cell (WPIC) collision modules

## Notation

| Symbols | Convention in this guide |
|---|---|
| $`f,t,H,C[f],\{f,H\}`$ | Distribution, physical time, single-particle Hamiltonian, collision operator and Hamiltonian Poisson bracket. Here $`H`$ is a Hamiltonian; expansion rates are written explicitly in the transport guide. |
| $`r,w,L,\eta,v_t`$ | Radius, radial velocity, specific angular momentum, $`\eta=L^2/2`$, and tangential speed. |
| $`\mathbf v_i,\mathbf u,u,\mathbf V,\mathbf u',\hat{\mathbf n}`$ | Incoming velocity of marker $`i`$, relative velocity, its magnitude, pair center-of-mass velocity, outgoing relative velocity and its sampled unit direction. A prime labels an outgoing quantity. |
| $`v_x,v_y,v_z`$ | Local Cartesian velocity components with the $`x`$ axis chosen radial. |
| $`m_i,m_j,M_i,M_j,M`$ | Represented object masses; capital $`M`$ denotes the merged total in the capture section. Lower-case $`m`$ in $`\sigma/m`$ is the material mass defining area per mass. |
| $`\sigma(u),s(u),\Sigma_{ij},S,\sigma_0`$ | Material cross section, area per material mass, effective numerical pair area, dimensionless strength and law normalization. |
| $`u_*,u_0,n`$ | Regularized relative speed, low-speed softening scale and power-law exponent. The normalization $`\sigma_0`$ has the units required by that exponent. |
| $`\alpha,m_\chi,m_\phi,\theta,\mu,y,z,\Omega`$ | Yukawa coupling, dark matter and mediator mass parameters, scattering angle, $`\mu=\cos\theta`$, $`y=(m_\chi u/m_\phi)^2`$, uniform angular draw and scattering solid angle. The normalization follows the configured code-unit law. |
| $`s,N_s,V_s,m_s,B_s,\nu_s`$ | Shell index, population, volume, matching-round count, upper bound on $`u\Sigma_{ij}`$ and resulting proposal frequency. Matching count $`m_s`$ is distinct from marker mass. |
| $`P_{ij},\delta t_s,t_s,\Delta t_c,\Delta t_{\rm batch},C_{\rm coll}`$ | Assigned-pair acceptance probability, local shell step, elapsed shell time, common collision substep, outer batch duration and collision safety factor. |
| $`A,a,q,x,\sigma_i,\sigma_{ij}`$ | Plummer area per mass, Plummer scale radius, pair mass ratio, relative speed divided by pair internal dispersion, individual and pair internal velocity dispersion. Here $`\sigma_i`$ is a speed; $`\sigma(u)`$ above is an area. |
| $`A_{ij},F_{\rm scatt},F_{\rm cap},\Sigma_{\rm scatt},\Sigma_{\rm cap}`$ | Pair geometric area, dimensionless tabulated scattering/capture functions and physical channel areas. |
| $`B_{ij},C`$ | Subhalo upper bound on $`u\Sigma_{ij}`$ and its coefficient, which contains strength, area normalization and the tabulated bound. |
| $`\tau_{\rm coll},t_{\rm coll},t'`$ | Integrated collision age, selected physical collision timescale and time integration variable. |
| $`N,N_{\rm trial},z_{\rm round},p_{ij}`$ | Cell population, integer proposal count, uniform rounding draw on $`[0,1)`$ and subhalo acceptance probability. |
| $`c,t_c,\nu_{B,c},\delta t_c,\delta t_{\rm fixed}`$ | Collision-cell index, elapsed cell time, proposal frequency bound, adaptive cell step and common fixed step. In fixed-batch formulas, $`t`$ is elapsed batch time. |
| $`\mu_r,E_{{\rm int},i},E_{{\rm int},\rm rem},r_{\rm rem},\sigma_{\rm rem}`$ | Reduced pair mass, individual and remnant internal energies, remnant radius and remnant internal velocity dispersion. Subscript $`\rm rem`$ denotes the merged object. |
| $`n_{\rm partner},\lambda_{\rm mfp},\Delta r,\langle\cdot\rangle`$ | Partner number density, mean free path, collision-cell width and the pair average used for the locality estimate. |

Cell and marker indices are local to the formula in which they appear. The
subscripts `const`, `pow` and `Y` select the constant, power-law and Yukawa
material laws. The rate and angular normalizations below specify the physical
meaning of the configured cross section.

This chapter documents two physically distinct operators:

1. material self-interacting dark matter (SIDM) scattering with constant, softened power-law, or
   Yukawa–Born cross sections.
2. calibrated subhalo Plummer scattering and capture with unequal
   masses and evolving internal dispersions.

Both solve the collisional Vlasov equation

```math
\frac{\partial f}{\partial t}+\{f,H\}=C[f],
```

by sampling local particle pairs. Their elastic velocity update has the same
center-of-mass form. Their physical cross sections and methods for proposing
pairs are specified separately below.

## 1. Material SIDM physics

Let $`u=|\mathbf v_i-\mathbf v_j|`$. The configured material law is an area
per represented mass, $`s(u)=\sigma(u)/m`$. The same symmetric coarse-object
closure as the VENUS CPU module is

```math
\Sigma_{ij}(u)=\frac{m_i+m_j}{2}s(u).
```

It is symmetric and recovers the configured equal-mass collision frequency.
The strength S multiplies the collision law. The constant normalization is
sigma0. The softened power law additionally uses speed u0 and exponent n.
The Yukawa parameters alpha, m_chi and m_phi set its amplitude and velocity
dependence. The three enabled laws are

```math
s_{\rm const}=S\sigma_0,
```

```math
s_{\rm pow}=S\sigma_0
\left(u_*^2+u_0^2\right)^{-n/2},\qquad
u_*=\max(|u|,10^{-14}),
```

and the integrated Born-regime Yukawa law

```math
s_{\rm Y}(u)=
S\frac{4\pi\alpha^2m_\chi^2}
{m_\phi^4\left[1+m_\chi^2u_*^2/m_\phi^2\right]}.
```

The corresponding differential kernel is

```math
\frac{d\sigma}{d\Omega}=
S\frac{\alpha^2m_\chi^2}
{\left[m_\phi^2+\frac12m_\chi^2u^2(1-\mu)\right]^2},
\qquad \mu=\cos\theta.
```

The CUDA/host evaluator is structurally:

```cpp
const double x = mchi * mchi * safe_u * safe_u / (mphi * mphi);
return strength * 4.0 * pi * alpha * alpha * mchi * mchi
     / (pow(mphi, 4) * (1.0 + x));
```

Yukawa scattering uses the analytic inverse of its angular probability.
With $`y=(m_\chi u/m_\phi)^2`$ and a uniform probability $`z\in[0,1]`$
for a cosine above the selected value,

```math
\mu=1-\frac{2z}{1+y(1-z)}.
```

As $`y\to0`$, this gives the uniform cosine $`\mu=1-2z`$. For
$`y>1`$, the implementation divides numerator and denominator by $`y`$
to keep the evaluation bounded. CPU and GPU evaluators use these same constant, power-law,
Yukawa-total, angular, and pair-area formulas. Only their surrounding data
layout and execution schedule differ.

### Material GPU proposal

Active markers are device-sorted by `(radius,id)` into fixed-occupancy local
shells. For shell volume $`V_s`$ and population $`N_s\ge2`$, define the
number of partner-assignment rounds

```math
m_s=\begin{cases}N_s-1,&N_s\text{ even},\\N_s,&N_s\text{ odd}.\end{cases}
```

The rounds contain each unordered pair exactly once. Each round assigns
every marker at most one partner, and an odd-population round leaves one
marker out. A uniformly selected round includes each pair with probability
$`1/m_s`$. The 64-bit rejection draw in `collision_matching.hpp` supplies
this uniform round. Disjoint pair slots let CUDA threads update their two
markers safely. The kernel uses the shared host/device helper:

```cpp
const collision_matching::Pair proposed =
    collision_matching::pair_at(count, matching_round, pair);
const std::uint64_t first_local = proposed.first;
const std::uint64_t second_local = proposed.second;
const std::uint64_t first_index = order[shell.begin + first_local];
const std::uint64_t second_index = order[shell.begin + second_local];
```

An assigned pair has acceptance probability

```math
P_{ij}=\frac{m_s}{V_s}u\Sigma_{ij}(u)\,\delta t_s.
```

Its expected probability is therefore $`(1/m_s)P_{ij}
=u\Sigma_{ij}\delta t_s/V_s`$, the physical Boltzmann pair exposure.
The mean-free-path diagnostic retains the physical partner number density
$`(N_s-1)/V_s`$. Acceptance and the collision CFL instead use the proposal
exposure $`m_s/V_s`$, including $`N_s/V_s`$ for odd populations.

Let $`B_s`$ be the safe bound on $`u\Sigma_{ij}`$ constructed from the
maximum marker mass and speed. The conservative rate is

```math
\nu_s=\frac{m_s}{V_s}B_s,\qquad
B_s\ge\max_{ij\in s}(u\Sigma_{ij}).
```

The default reference schedule chooses one mesh-wide step from
$\max_s\nu_s$, visits every occupied shell, and enforces

```math
\nu_s\Delta t_c\le C_{\rm coll}\le1.
```

With `--sidm-adaptive-shell-timestep`, shell $s$ owns elapsed time $t_s$
and instead takes

```math
\delta t_s=\min\left(\Delta t_{\rm batch}-t_s,
\frac{C_{\rm coll}}{\nu_s}\right).
```

All active shells advance in one CUDA launch. The shell-description kernel
then recomputes maximum speed and the safe $u\Sigma$ majorant before the
next local step. Elastic scattering leaves radii and masses fixed, so sorted
membership, shell faces, density, and volume remain reusable. A zero-rate
shell advances directly to the batch endpoint. Reported `substeps` is the
largest local count and `shell_visits` is their sum. This path supports
constant, softened-power-law, and Yukawa--Born cross sections and unequal
marker masses. Local clocks avoid revisiting slow shells at every dense-core
step while preserving the Boltzmann acceptance law.

The pair marginals reproduce the instantaneous Boltzmann collision rate.
Each marker experiences at most one event per local step, so quantitative
evolution requires collision-CFL convergence. Material CPU/GPU comparisons
check formulas, invariants, timestep convergence, and statistical ensembles.

## 2. Unequal-mass elastic map

WPIC stores $`(r,w,\eta)`$, with tangential speed
$`v_t=\sqrt{2\eta}/r`$. A counter-RNG azimuth lifts a marker to a 3-vector.
For either collision model,

```math
\mathbf V=\frac{m_i\mathbf v_i+m_j\mathbf v_j}{m_i+m_j},
\qquad \mathbf u=\mathbf v_i-\mathbf v_j.
```

After drawing a direction $`\hat{\mathbf n}`$ and keeping
$`|\mathbf u'|=u`$,

```math
\mathbf v_i'=\mathbf V+\frac{m_j}{m_i+m_j}\mathbf u',\qquad
\mathbf v_j'=\mathbf V-\frac{m_i}{m_i+m_j}\mathbf u'.
```

Projection back is $`w'=v_x'`$ and
$`\eta'=r^2(v_y'^2+v_z'^2)/2`$. By construction this center-of-mass map
conserves pair momentum and kinetic energy, including for unequal masses.
Floating-point implementations retain those identities to roundoff.

## 3. Subhalo cross sections

The CUDA build generates its calibration header directly from the maintained
CPU table, keeping both evaluators tied to the same tabulated values. For object masses $`M_i,M_j`$, define

```math
A_{ij}=\frac{A}{4}\left(\sqrt{M_i}+\sqrt{M_j}\right)^2,
\qquad q=\frac{\min(M_i,M_j)}{\max(M_i,M_j)},
```

where $`A=\pi a^2/M`$ is `plummer_area_per_mass`. The evolving pair velocity
scale is

```math
\sigma_{ij}^2=
\frac{\sqrt{M_i}\sigma_i^2+\sqrt{M_j}\sigma_j^2}
{\sqrt{M_i}+\sqrt{M_j}},
\qquad x=\frac{u}{\sigma_{ij}}.
```

The mutually exclusive physical channels are

```math
\Sigma_{\rm scatt}=S A_{ij}F_{\rm scatt}(x,q),\qquad
\Sigma_{\rm cap}=S A_{ij}F_{\rm cap}(x,q).
```

`F_scatt` and `F_cap` use the same CPU table, log-speed/log-mass-ratio
interpolation, zero-aware linear closure, low-speed regularization, vanishing
high-speed capture, and $`x^{-4}`$ high-speed scatter continuation. A build
fails if the authoritative CPU table is unavailable.

The public formula hook mirrors the kernel:

```cpp
auto xs = subhalo_cross_sections(
    u, first.mass, second.mass,
    first.internal_velocity_dispersion,
    second.internal_velocity_dispersion, options);
// xs.scattering, xs.capture, xs.u_sigma_majorant
```

## 4. Exact current-CPU subhalo proposal

The subhalo CUDA module uses the CPU
`maximum_weight_importance` pair-majorant scheduler. Within each physical
collision cell it sorts a proposal index by `(sigma_internal,id)`. For
positions $`i<j`$, a safe pair weight is

```math
B_{ij}=C\,\sigma_j(\sqrt{M_i}+\sqrt{M_j})^2,
```

where $`C`$ contains the strength, area, and tabulated dimensionless bound.

### Collision time as the physical clock

Physical-time coverage must also be quoted in collision units. Define

```math
\tau_{\rm coll}(t)=\int^t \frac{dt'}{t_{\rm coll}(t')},
\qquad
\tau_{\rm coll}\simeq \frac{t}{t_{\rm coll}}
\quad\hbox{when }t_{\rm coll}\hbox{ is nearly constant}.
```

Both the proposal majorant and the physical cross sections are linear in the
configured strength $`S`$. Holding the instantaneous distribution fixed gives
$`t_{\rm coll}\propto S^{-1}`$, hence $`\tau_{\rm coll}\propto S t`$. A run
with ten times the strength therefore obtains the same nominal collision age
in one tenth the physical time. For an evolving nonlinear halo the preferred
quantity is the integral, because density, velocity dispersion, marker masses,
and velocity-dependent cross sections all change with time. The production
history's `collision_time` is the minimum sampled cell value. Any
`t/collision_time` derived from it must be identified as a minimum-cell
diagnostic. The halo-wide cumulative collision age requires a specified
population average and its integral through the evolving state.

Prefix sums of $`\sqrt M`$ and $`M`$ sample this exact distribution in
$`O(N\log N)`$, without materializing $`O(N^2)`$ pairs.

For cell volume $`V`$, the stochastic number of proposals is

```math
N_{\rm trial}=\left\lfloor
\frac{\Delta t_c}{V}\sum_{i\lt j}B_{ij}+z_{\rm round}
\right\rfloor.
```

A proposed pair is accepted with

```math
p_{ij}=\frac{u\,[\Sigma_{\rm scatt}+\Sigma_{\rm cap}]}
{B_{ij}}\le1.
```

One accepted collision per marker per substep is enforced exactly. A capture
changes mass and internal dispersion, so captured partners are compacted and
the proposal arrays are rebuilt before that cell advances again. The substep
obeys a per-marker collision-frequency CFL.

### Cached majorants and packed cell kernels

Elastic scattering changes orbital velocities while preserving marker mass
and internal dispersion. The pair bound ($`B_{ij}`$), its prefix sums and its
maximum trial frequency depend on those unchanged quantities, so the proposal
can be reused. The loop keeps the `DeviceCell`, prefix-root,
prefix-mass and pair-cumulative arrays resident. Capture changes their physical
inputs and marks them for rebuilding after compaction:

```cpp
bool proposal_dirty = true;
while (unfinished_cells != 0) {
    if (proposal_dirty) {
        build_pair_proposals_kernel(...);
        proposal_dirty = false;
    }
    collide_subhalo_cells_adaptive_kernel(..., cell_elapsed,
                                            cell_substeps,
                                            batch_substeps);
    if (stats.removed_markers != 0) {
        state.compact_removed_device(...);
        build_orders();
        proposal_dirty = true;
    }
}
```

The default reference mode uses one mesh-wide step,

```math
\delta t_{\rm fixed}=\min\left(\Delta t_{\rm batch}-t,
\frac{C_{\rm coll}}{\max_c\nu_{B,c}}\right),
```

and visits every populated cell. With `--subhalo-adaptive-cell-timestep`,
cell $c$ instead advances with

```math
\delta t_c=\min\left(\Delta t_{\rm batch}-t_c,
\frac{C_{\rm coll}}{\nu_{B,c}}\right).
```

One CUDA thread owns one cell and executes at most
`--subhalo-adaptive-batch-substeps` local steps per launch. Elastic events
reuse the exact mass/dispersion proposal. Capture ends that cell's loop.
Host orchestration compacts the remnant population, rebuilds proposals, and
resumes unfinished cells from saved clocks and local substep indices. The
cell kernels pack 256 cells per block while preserving serial operation order
inside each cell.

In multi-GPU adaptive mode, complete radial cells already have unique owners,
so each owned cell advances independently inside the synchronized outer
operator batch. Fixed reference mode still uses
`MPI_Allreduce(MPI_MAX)` for its common mesh-wide clock.

All random values use the CPU tuple

```text
(seed, epoch, global collision cell, substep, trial, dimension)
```

through the same counter mixer. CUDA block order and Message Passing Interface (MPI) rank count therefore
do not redefine a trial's random variates. Reproducibility then depends on
preserving the same global cell numbering, epoch, substep, trial index, and
capture-rebuild sequence.

## 5. Capture physics

Capture merges complete objects. With $`M=M_i+M_j`$ and reduced mass
$`\mu_r=M_iM_j/M`$,

```math
\mathbf v_{\rm rem}=\mathbf V,\qquad
r_{\rm rem}=\frac{M_i r_i+M_j r_j}{M},
```

```math
E_{{\rm int},\rm rem}=E_{{\rm int},i}+E_{{\rm int},j}
+\frac12\mu_r u^2,
\qquad
\sigma_{\rm rem}=\sqrt{\frac{2E_{{\rm int},\rm rem}}{3M}}.
```

Thus orbital relative kinetic energy becomes internal heat while total mass,
vector momentum, and orbital-plus-internal energy are conserved. The consumed
marker is removed by stable device compaction.

## 6. Mean free path and diagnostics

For the evolving pair kernel the locality diagnostic uses

```math
\lambda_{\rm mfp}=
\frac{\langle u\rangle}
{n_{\rm partner}\langle u\Sigma_{ij}(u)\rangle}.
```

The production runner records trial/accepted/scatter/capture counts, removed
markers, represented event masses, energy changes, invariant errors,
collision time, and $`\max(\Delta r/\lambda_{\rm mfp})`$. It warns above one
and stops above the configured hard limit, whose default is ten.

## 7. Interpreting CPU/GPU correspondence

The production subhalo backends share the cross-section table, counter-keyed
random variates, pair-majorant distribution, capture map, stable marker IDs,
and restart fields. Parallel atomics and reductions can nevertheless perturb
continuous fields in their last bits. In a long chaotic Monte-Carlo evolution,
such perturbations may eventually change which accept/reject boundary is
crossed. Compare conservation, resolved profiles, ensemble statistics, and
resolution convergence. These quantities establish whether the two backends
predict the same resolved physical evolution over the interval studied.
