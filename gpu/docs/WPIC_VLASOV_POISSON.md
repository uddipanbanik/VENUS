# GPU weighted Vlasov–Poisson module

## 1. Governing equations

For a spherical collisionless distribution in Newtonian gravity,

```math
\frac{\partial f}{\partial t}
+w\frac{\partial f}{\partial r}
+\left(\frac{L^2}{r^3}-\frac{\partial\Phi}{\partial r}\right)
 \frac{\partial f}{\partial w}=0,
\qquad \eta=\frac{L^2}{2}.
```

Here $`f`$ is the distribution function, $`r`$ is physical radius, $`w`$
is radial velocity in a static background, and $`L`$ is specific angular
momentum. The gravitational potential is $`\Phi`$, the mass density is
$`\rho`$, and $`G`$ is Newton's gravitational constant. In an expanding
background, $`w`$ denotes peculiar radial velocity, as defined in Section 5.

The spherical Poisson equation and radial force are

```math
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2\frac{\partial\Phi}{\partial r}\right)=4\pi G\rho,
\qquad
\Phi_r(r)=\frac{G M(\lt r)}{r^2}.
```

WPIC approximates phase-space integrals by sums over weighted markers. Their
mass weights source the radial density and Poisson field. A collisionless
characteristic keeps marker mass and $`\eta`$ fixed and evolves only
$`(r,w)`$.

## 2. Spherical deposition

The natural volume coordinate is

```math
q=\frac{r^3}{3},\qquad
\Delta V_i=4\pi(q_{i+1/2}-q_{i-1/2}).
```

For q-CIC between two centers,

```math
a_{pi}=\frac{q_{i+1}-q_p}{q_{i+1}-q_i},\qquad
a_{p,i+1}=1-a_{pi}.
```

Production uses the current CPU volume-corrected q-TSC weights. The kernel
deposits all extensive quantities with the same cloud weights:

```math
M_i=\sum_p a_{pi}m_p,\quad
P_i=\sum_p a_{pi}m_pw_p,
```

```math
R_{2,i}=\sum_p a_{pi}m_pw_p^2,\quad
T_{2,i}=\sum_p a_{pi}m_p\frac{2\eta_p}{r_p^2},
```

plus radial energy flux and squared cloud mass for effective marker number.
The CUDA structure is:

```cpp
const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
if (p < markers.size && markers.state[p] == active) {
    const double q = markers.r[p] * markers.r[p] * markers.r[p] / 3.0;
    // CPU-identical CIC or volume-corrected TSC cloud weights
    atomicAdd(&shell_mass[cell], cloud * markers.mass[p]);
    atomicAdd(&radial_momentum[cell],
              cloud * markers.mass[p] * markers.w[p]);
}
```

Atomics preserve every marker's contribution when several threads deposit
into the same shell. Their arrival order can change the last rounding bits,
which sets the scope of comparison with an x86 `long double` reduction.

### Origin-parity carrier

An active cloud below the first face is partitioned continuously between the
resolved mesh and the enclosed central mass. Let $`q_f=r_{\min}^3/3`$ be
the first face in volume coordinate and $`q_0`$ the first cell center in
that coordinate. The transition width is
$`\delta q=\min[2(q_0-q_f),q_f]`$, and
$`s=\min\{1,\max[0,(q-q_f+\delta q)/\delta q]\}`$. The deposited fraction is

```math
f_{\rm resolved}=s^2(3-2s),\qquad
M_{\rm mesh}+M_{\rm hidden}=m_p.
```

This is an algebraic mass identity for every marker. The halo production
drivers enable this path. The particles continue to move through the origin;
the first mesh face defines the radius below which the force uses a uniform
central-density approximation.

## 3. Conservative spherical Poisson solve

The GPU field solver reconstructs constant mass per radial length inside each
radial cell. This corresponds to the CPU constant-line-mass option. The CPU
also provides a positive quadratic line-mass reconstruction.

```math
\lambda_i=\frac{M_i}{\Delta r_i},\qquad
M(r)=M_{i-1/2}+\lambda_i(r-r_{i-1/2}).
```

A CUDA inclusive scan forms $`M_{i+1/2}`$; independent threads evaluate
center force:

```cpp
mass_center = mass_edge[i]
            + line_density[i] * (r_center[i] - r_edge[i]);
phi_r[i] = gravity * mass_center / (r_center[i] * r_center[i]);
```

Potential is integrated analytically. If
$`A_i=M_{i-1/2}-\lambda_i r_{i-1/2}`$,

```math
\int_{r_a}^{r_b}\frac{G M(r)}{r^2}\,dr
=G\left[A_i\left(\frac1{r_a}-\frac1{r_b}\right)
+\lambda_i\ln\left(\frac{r_b}{r_a}\right)\right].
```

A reverse scan starts from
$`\Phi(r_{\max})=-GM(r_{\max})/r_{\max}`$. The expanding halo production
drivers use masses that already represent density contrast and set
`subtract_homogeneous_background=false`.

When the supplied masses represent total density, the Poisson source uses
$`\rho-\bar\rho`$, where $`\bar\rho=3H^2/(8\pi G)`$ and $`H=\dot a/a`$.
The source mass at the first face therefore contains the central particle
mass, a prescribed point mass $`M_c`$, and the background subtraction
$`-4\pi\bar\rho r_{\min}^3/3`$. This subtraction changes the gravitational
source; the physical mass accounting retains the particle masses.

For $`0<r<r_{\min}`$, the force joins continuously to the first face:

```math
\Phi_r(r)=\frac{GM_c}{r^2}
 +\frac{G[M_{\mathrm{source}}(r_{\min})-M_c]}{r_{\min}^3}\,r.
```

The second term is the force of the uniformly represented central material.
An empty central cavity in a static calculation has zero force when
$`M_c=0`$. The implementation evaluates its coefficient from the first
face's enclosed source mass:

```cpp
const double core_coefficient = gravity * (source_mass_edge[0] - central_mass)
    / (edge[0] * edge[0] * edge[0]);
return gravity * central_mass / (r * r) + core_coefficient * r;
```

Resolved cells subtract the background using the same midpoint line-density
quadrature as the CPU solver, $`4\pi\bar\rho r_i^2\Delta r_i`$.
Uniform grid-represented density then gives zero force to roundoff. Sampling
the background with exact shell volumes instead produces a residual that
decreases as $`\Delta r^2`$ under radial refinement.

## 4. Kick–drift–kick characteristic map

In a static background the first half kick is

```math
w^{n+1/2}=w^n-\frac{\Delta t}{2}\Phi_r^n(r^n).
```

The drift is the exact free central-motion solution. With $`L=\sqrt{2\eta}`$,

```math
r_*^2=(r^n+\Delta t\,w^{n+1/2})^2
+\left(\Delta t\frac{L}{r^n}\right)^2,
```

```math
w_*=\frac{w^{n+1/2}(r^n+\Delta t\,w^{n+1/2})
+(L/r^n)(\Delta t L/r^n)}{r_*}.
```

After endpoint deposition and Poisson solve,

```math
w^{n+1}=w_*-\frac{\Delta t}{2}\Phi_r^{n+1}(r_*).
```

For $`L>0`$ and a drift that remains inside the radial walls, the CUDA code follows the
CPU formula. Here $`x=r/a`$, $`p=aw`$, and `drift` is the integral defined
in Section 5:

```cpp
double p = a0 * w - 0.5 * dt * a0 * phi_r;
const double radial_axis = x + drift * p;
const double tangential_axis = drift * L / x;
const double new_x = hypot(radial_axis, tangential_axis);
p = (p * radial_axis + (L / x) * tangential_axis) / new_x;
r = a1 * new_x;
w = p / a1;
```

At a finite reflecting wall, the code first solves for the time at which the
free trajectory reaches the wall. It reverses the physical radial velocity
there, preserves $`L`$, and advances the remaining part of the drift. For a
fixed physical wall at radius $`R`$, the peculiar velocity changes according
to $`w_{\mathrm{after}}=-w_{\mathrm{before}}-2H R`$, with $`H`$ evaluated at
the impact. CPU and GPU use the same host/device implementation:

```cpp
ev::advance_central_drift_with_walls(
    r, w, L, dt, start_cosmic_time, expansion_alpha,
    inner_radius, reflect_inner, outer_radius, reflect_outer);
```

This calculation includes trajectories that cross the inner wall and would
otherwise return outside it during the same step. In a static background,
each free reflection preserves $`w^2+L^2/r^2`$. With origin parity, the
physical inner boundary is $`r=0`$; a radial particle crosses the origin
with the corresponding change of sign in its radial velocity.

For exactly radial motion, $`L=0`$, the drift advances the signed coordinate
$`x+ p\int a^{-2}\,dt`$. Its absolute value gives the new comoving radius.
Crossing zero reverses the radial momentum. A forward step ending exactly at
the origin stores the outgoing velocity, and the next step continues the
same orbit. A particle initially at the origin is allowed when both $`L`$
and the central point mass vanish. The smooth central force is then zero.
The implementation avoids division by the radius in this case:

```cpp
const double signed_x = x + drift * p;
if (signed_x < 0.0) p = -p;
else if (signed_x == 0.0 && dt > 0.0) p = fabs(p);
r = a1 * fabs(signed_x);
w = p / a1;
```

With a nonzero central point mass, the force is singular at the origin and
the solver rejects a radial step that reaches or crosses it.

The collisionless production runner uses the fixed step supplied by
`--maximum-timestep`. Only a shorter exact landing on the next output or final
time is allowed. Fixed-step KDK is time symmetric and therefore gives bounded
energy error in a constant potential; a cell-dependent step would break that
symmetry. Collisional task runners may still apply a dynamical orbit ceiling
because collision-event splitting has already broken the collisionless
fixed-step map.

## 5. Expanding coordinates

For $`r=a(t)x`$, the physical radial velocity is $`\dot r=Hr+w`$, and the
canonical radial momentum is $`p=a w`$. The exact drift
factor is

```math
S(t_n,t_{n+1})=\int_{t_n}^{t_{n+1}}\frac{dt}{a(t)^2}.
```

For $`a=[(t+t_0)/t_0]^\alpha`$, the code evaluates the power-law integral
analytically and uses the logarithmic limit near $`1-2\alpha=0`$. The requested
runs use $`\alpha=2/3`$ and $`t_0=12`$, exactly as their CPU baseline.

## 6. Order of the GPU update

One `advance_kdk` call performs:

```text
deposit -> Poisson -> half kick + exact drift
        -> deposit -> Poisson -> endpoint half kick
```

Marker arrays, mesh, moments, scans, field, and rollback buffer remain on the
GPU. Host transfers occur for requested profiles, diagnostics, and restarts.
For collisional evolution, the driver can keep the incoming particle state
while testing an orbit endpoint. It accepts that endpoint or restores the
incoming state and shortens the step to a newly inferred collision deadline.

Before writing a profile, the production driver deposits the completed
particle state again. The final radial kick changes the radial mean and
variance while preserving density. This output deposition therefore gives
the profile and checkpoint the same velocity time level:

```cpp
vp.deposit_and_solve(state, time, vp_control);
field = vp.download_field();
```

Continue a checkpoint with the original mesh, collision parameters and
numerical controls. In an existing output directory, histories are appended
and the completed profile preceding an intermediate checkpoint is preserved.
The driver checks the times in existing profiles and histories before
writing. A time later than the checkpoint causes a clear rejection that
preserves those files. An earlier checkpoint can be evolved separately in a
new output directory. `SIGUSR1` and `SIGTERM` request a checkpoint and exit
status 99 after the accepted particle state has been saved.

## 7. Multiple GPUs

One MPI rank owns one GPU and a subset of markers. Each field solve is

```text
local GPU deposition
 -> MPI_Allreduce of O(Nr) extensive moments
 -> identical global Poisson solve on every GPU
 -> local marker push
```

Markers remain on their owning GPU during Vlasov–Poisson stepping; only the
radial moments are exchanged. KDK requires two such reductions. Scaling is therefore controlled by local marker work,
the two moment all-reductions, mesh size, and rank imbalance; measure both
strong and weak scaling for the intended configuration.

## 8. CPU/GPU numerical correspondence

The CPU and GPU paths use the same volume-coordinate clouds, conservative
spherical Poisson discretization, exact central-motion drift, analytic EdS
drift integral, boundary maps, and checkpoint variables. CUDA atomics and MPI
reductions need not accumulate floating-point terms in the CPU order, so
continuous fields should be compared with scale-aware tolerances. Marker IDs,
states, represented masses, counter keys, time, and step are the discrete
restart invariants.
