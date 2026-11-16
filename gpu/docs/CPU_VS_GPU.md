# How GPU code differs from CPU code

The CPU and GPU modules implement the same supported force and collision
formulas. The GPU arranges independent marker, shell and pair calculations
into parallel kernels. Their memory access, shared sums and exclusive pair
updates determine the execution order described here.

## Cores versus CUDA threads

A CPU runs a modest number of sophisticated, branch-tolerant cores. A GPU
runs thousands of lightweight threads most efficiently when neighboring
threads perform similar arithmetic. The CPU loop

```cpp
for (Marker& marker : markers) {
    push_one_marker(marker);
}
```

becomes a CUDA kernel:

```cpp
const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
if (p < size) {
    push_one_marker(r[p], w[p], eta[p]);
}
```

One launch supplies work for the whole population, spreading its fixed launch
cost across many marker updates.

## Structure of arrays

CPU code commonly stores complete marker records. `WpicState` transposes once
to adjacent device arrays:

```text
mass: m0 m1 m2 ...
r:    r0 r1 r2 ...
w:    w0 w1 w2 ...
eta:  e0 e1 e2 ...
```

A warp reading 32 radii performs coalesced memory transactions. All
current subhalo fields—internal energy, internal dispersion, state, IDs, and
restart counters—remain represented.

## Device memory must stay resident

Transfers between host and device add latency and move the full particle
record through the connection. The VENUS GPU module therefore retains markers,
mesh, moments, field, collision indexes and rollback storage on device.
Profiles, checkpoints and multi-GPU collision migration request the needed
host records explicitly.

## Reductions and numerical correspondence

Floating-point addition is not associative. CPU `long double` accumulation
and CUDA double atomics can add the same terms in different orders. Therefore

```math
(a+b)+c\ne a+(b+c)
```

in the last bits even though both implement the same discretization. CPU/GPU
comparisons should distinguish four increasingly macroscopic levels:

1. exact discrete state: IDs, states, counts, RNG counters, event ledgers;
2. roundoff-tolerant continuous state: `r`, `w`, field moments;
3. conservation and convergence;
4. statistical/profile agreement for long chaotic Monte-Carlo evolution.

Discrete checkpoint fields should agree whenever both backends use the same
counter-keyed event schedule. Continuous fields should be compared with a
roundoff-aware tolerance, while long chaotic integrations should be assessed
through conserved quantities, resolution convergence, and physical profiles.
The stated field tolerance must resolve physical errors while allowing the
measured rounding differences of the parallel sums.

## Race conditions

Deposition is many-to-one, so double-precision atomics protect shell moments.
Collision updates are two-to-two and need exclusive marker ownership.

The material SIDM module draws a round of disjoint partner assignments each
substep; every thread owns both endpoints. Each unordered pair has inclusion
probability `1/m`, where `m=N-1` for even shell population `N` and `m=N` for
odd `N`. Acceptance and its collision CFL use `m/V`, so the pair marginals
give the physical Boltzmann collision rate. The mean-free-path diagnostic
uses physical partner density `(N-1)/V`. The one-event-per-marker local step
requires collision-CFL convergence; CPU/GPU comparisons use conservation,
convergence, and statistical ensembles.

For adaptive material collisions, the host retains one elapsed-time clock per
shell, launches all active shells together, and refreshes velocity-dependent
shell majorants between launches. For adaptive subhalo collisions, one CUDA
thread advances one cell through several local steps and returns immediately
after capture. These device batches are the main implementation difference
from the CPU thread-parallel local-clock loops.

The production subhalo module uses the CPU pair-majorant proposal: a bound on
each pair's collision rate defines its proposal probability. Prefix sums draw
from that distribution with the same counter-based random numbers. A byte-valued
collision flag permits one accepted event per marker per substep. Capture
changes the proposal's masses and internal dispersions, so stable compaction
removes consumed partners and the affected proposal is rebuilt. This order
preserves the definition of each trial across the two implementations.

## Branch divergence

Threads in a warp share an instruction stream. If some reflect, scatter, or
capture while others do not, the warp serializes those branches. The KDK
boundary loop is short and bounded; scatter/capture branches operate in large
batches. The adaptive Plummer material-deflection quadrature remains CPU-only
because its deeply divergent control flow is poorly matched to the present GPU
batch design.

## Kernel launch, I/O, and synchronization costs

Large marker loops amortize launch overhead; very small runs may not. Output
forces a device synchronization, so production output is sparse. With MPI,
each KDK step performs two `O(Nr)` moment all-reductions. Subhalo collisions
currently host-stage marker migration into radial ownership slabs; this is
correct but can make two GPUs slower than one for short collision-heavy
intervals.

## Multiple GPUs

One MPI rank binds to one GPU. VP markers stay distributed while only field
moments are reduced. Before a subhalo collision, active markers migrate to the
rank owning their radial collision cell. Global cell numbers key the RNG, preserving a trial's identity when its cell
moves between ranks. A global maximum supplies the common fixed reference
clock, whereas adaptive mode advances the already disjoint owned cells
locally. Rank zero then gathers/sorts checkpoint markers by ID. These choices make the
physical event definition independent of the rank decomposition.

## Performance considerations

GPU speedup is workload-dependent. It generally increases with marker count
and with the amount of arithmetic performed between outputs, because larger
batches amortize launch and synchronization costs. Inner-shell atomic
contention, collision subcycling, capture compaction, host-staged MPI
migration, frequent profiles, and checkpoints can reduce it. Benchmark the
intended mesh, marker count, collision strength, output cadence, precision,
and accelerator so the measured runtime describes the intended calculation.
