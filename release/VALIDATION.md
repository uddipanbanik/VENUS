# VENUS validation

## Guide and module update, 16 November 2026

The current CPU candidate passes 29 C++ numerical test executables, 112 Python
tests, and the supplied driver, restart, coordinate and scheduler-workflow
checks. The complete CPU example catalogue passes on macOS and IAS Linux:
22 examples evolve to their requested physical endpoint, and the expanding
exact-grid example checks its initial-data setup at zero time. On IAS, the
three collisionless grid examples also pass with two MPI processes, and all
13 grid examples pass with two OpenMP threads.

All CPU production targets and 15 C++ utilities compile. The single-GPU and
MPI CUDA builds compile on IAS. On Symmetry, the CUDA 12.6/GCC 11.2 build
passes all ten registered numerical/interface tests, including collisionless
transport, material and subhalo encounters, CPU comparisons, endpoint moments
and two-process MPI. The additional host output-replacement check passes.
All eight GPU examples and all six two-GPU MPI examples reach their requested
physical endpoints on two NVIDIA GeForce RTX 2080 Ti devices. These include
the explicit static Vlasov--Poisson entry point, which uses the existing CUDA
evolution kernel. The power-law example supplies its required positive
softening speed and exponent explicitly.

The static CPU Vlasov--Poisson entry point is checked against the established
static driver, including conservation, binary continuation and rejection of
incompatible expansion or encounter parameters. Example checks verify the
actual saved endpoint. Restart examples preserve their source checkpoints.
The analysis wrappers have been exercised on grid, Newtonian particle,
material SIDM and both relativistic coordinate representations. The generated
density/slope movie was inspected for legibility and physical labeling.

All 107 shell and Slurm scripts pass syntax checks. The local JavaScript
syntax check requires Node, which is absent from this test environment.

These finite checks establish the interfaces and numerical conditions tested.
Each scientific application requires its own physical model, resolution,
timestep, boundary and collision-locality studies.

# Earlier VENUS 1.0 validation

The CPU implementation has passed its integrated C++ tests, driver checks,
111 Python reader tests and focused address/undefined-behavior sanitizer
checks. Three complete-snapshot analysis tests also check the particle/field
input contracts and the phase-space and energy measures. The CUDA implementation has passed ten A100 GPU tests and two
on-device random-number probes. These checks cover the supported numerical
paths, input/output contracts and conservation diagnostics at their tested
parameters.

The relativistic elastic-collision library has passed its coupled
Schwarzschild and Eddington–Finkelstein component tests. These exercise the
null-slice collision rate, local four-momentum conservation, persistent cell
clocks, cell restart, the collisionless limit and reconstruction of outgoing
Einstein sources. Compact collisional collapse requires equilibrium, locality
and convergence checks for its chosen initial data.

Individual source checksums are in `numerical_sources.json`.
`source_manifest.json` identifies the complete distribution, including
source, documentation, example inputs, and the example plots and movies.

The two Eddington–Finkelstein collapse calculations retain all 65,536
particles. The reported fine-grid interior profiles cover the spatially resolved interval
through advanced time 32.50. Later interior evolution requires finer radial
resolution. The Schwarzschild calculation reaches coordinate time 160
with the characteristic and metric checks described in the manual.

The figures in `plots` and movies in `movies` use the simulations
identified in their run catalogs. Their physical time ranges, sampling masks
and numerical scope are stated in the reproduction guides.

For a new scientific application, assess radial and velocity resolution,
particle count, timestep, collision locality and boundary influence for that
application. The manual defines the diagnostics used for these assessments.
