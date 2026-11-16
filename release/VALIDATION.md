# VENUS 1.0 validation

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
