# VENUS validation

## CPU checks

The CPU implementation passes 29 C++ numerical test executables, 121 Python
tests, and the supplied driver, restart, coordinate and scheduler-workflow
checks. All 23 CPU examples reach their stated endpoints on macOS and IAS
Linux. Of these, 22 perform finite evolution and the expanding exact-grid
example checks initial-data setup at zero time. The three collisionless grid
examples pass with two MPI processes; all 13 grid examples pass with two
OpenMP threads. All CPU production targets and 15 C++ utilities compile.

The static Vlasov--Poisson entry point is checked against the shared static
driver, including conservation, binary continuation and rejection of
incompatible expansion or encounter parameters. Driver checks cover the
WPIC backgrounds, collision-clock cadence, stale-output rejection, material
SIDM continuation, relativistic coordinate mapping and halo classification.
Focused address/undefined-behavior sanitizer checks and complete-snapshot
analysis checks cover the particle/field input contracts and the phase-space
and energy measures.

## GPU checks

Single-GPU and MPI CUDA targets compile with the IAS CUDA 12.4/GCC 10 and
Symmetry CUDA 12.6/GCC 11.2 environments. On two NVIDIA GeForce RTX 2080 Ti
devices, all ten registered numerical/interface tests pass. These cover
collisionless transport, material and subhalo encounters, CPU comparisons,
endpoint moments, binary restart, the command interface and two-process MPI.
The additional host output-replacement check passes.

All eight GPU examples and all six two-GPU MPI examples reach their requested
physical endpoints. These include static Vlasov--Poisson, expanding
Vlasov--Poisson, subhalo scattering/capture and the supported material SIDM
laws. The A100 validation includes ten on-device tests and two random-number
probes. The GPU architecture and toolkit used for an application remain part
of that application's reproducibility record.

## Launch, restart and analysis interfaces

Example scripts verify the actual saved endpoint. All 20 CPU production
help interfaces pass. The analysis wrappers read grid, Newtonian particle,
material SIDM and relativistic output in both supported coordinate systems.
The Newtonian grid, Schwarzschild and mapped Eddington--Finkelstein tutorial
runs and their analysis commands pass. Generated density/slope movies are
checked for legibility and physical labeling.

The direct README commands reach their stated endpoints: nine CPU starting
runs, one CPU continuation, the custom-distribution CPU example, and three
single-GPU examples for collisionless gravity, subhalo encounters and material
SIDM. The reversible Boltzmann--Poisson script and its analysis also pass.
All documented analysis commands for these runs complete. The GPU
commands include building each named target on an RTX 2080 Ti. The updated
manual has resolved references and visually checked opening pages; Markdown
links, section links and the main guide tables are checked.

The plotting smoke test starts all 49 command-line plotting/analysis tools
through `--help`. All twelve supplied figure pairs reproduce with identical
PNG pixels and PDF content after removing the creation timestamp. All seven
movie-rendering entry points export previews from supplied data. These checks
exercise imports, command interfaces and small rendering operations.

All 109 shell and Slurm scripts pass syntax checks. The shipped JavaScript
utility passes its syntax check with IAS Node 24.13.0. The cluster guide
states the compiler, MPI, CUDA and resource choices for IAS Typhon/Apollo
and Perimeter Symmetry. MPI templates use the site's selected launcher;
CPU threads, MPI processes and GPU resources follow the module's supported
parallel decomposition. Real Slurm submissions verify the Typhon CPU/WPIC
and collisionless CPU MPI site recipes, Symmetry CPU grid and CPU MPI recipes,
and Symmetry single-GPU, material GPU MPI and binary continuation recipes.
The Apollo single-GPU site recipe reaches its requested endpoint on the
allocated A100 hardware. The H200 resource recipe is supplied for that device;
on-device release checks use the GPU hardware identified above.

## Relativistic collision and collapse scope

The relativistic elastic-collision library passes coupled Schwarzschild and
Eddington--Finkelstein component checks. These exercise the null-slice
collision rate, local four-momentum conservation, persistent cell clocks,
cell restart, the collisionless limit and reconstruction of outgoing Einstein
sources. Compact collisional collapse requires equilibrium, locality and
convergence checks for its chosen initial data.

The two Eddington--Finkelstein example collapse calculations retain all
65,536 particles. Their fine-grid interior profiles cover the spatially
resolved interval through advanced time 32.50. Later interior evolution
requires finer radial resolution. The Schwarzschild example reaches
coordinate time 160 with the characteristic and metric checks described in
the manual. The figure and movie catalogs state each calculation's inputs,
physical time range, sampling and resolved domain.

## File identity and scientific accuracy

`numerical_sources.json` records individual numerical-source checksums.
`source_manifest.json` identifies the complete distribution, including the
source, documentation, initial-data interfaces, example figures and movies.
The distributed-file checker verifies those identities.

These finite checks establish the interfaces and numerical conditions tested.
For a scientific application, assess radial and velocity resolution, particle
count, timestep, collision locality and boundary influence for that physical
problem. The manual defines the corresponding conservation and source-budget
diagnostics.
