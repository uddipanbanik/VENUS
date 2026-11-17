# VENUS terms in plain language

Start with the [README](../README.md), then follow the [CPU guide](../cpu/README.md)
or [GPU guide](../gpu/README.md). Those pages explain the terms needed for a
first calculation. This reference explains the additional names used in the
method catalogues and advanced guides. Exact executable names, file names and
option names keep their spelling so that you can copy them into a command.

## The physical problem

| Term | Meaning |
|---|---|
| Self-gravitating | The matter produces the gravitational field in which it moves. |
| Newtonian / non-relativistic | Gravity is described by a potential; speeds and gravitational fields are sufficiently small for relativistic corrections to be negligible. |
| Relativistic | Motion and gravity are described using general relativity, including the geometry of spacetime. |
| Kinetic description | A description of how matter is distributed over both position and velocity. |
| Distribution function | The amount of matter per position-and-velocity volume, with the integration measure specified by the chosen coordinates. |
| Phase space | Position and velocity, or position and momentum, considered together. |
| Collisionless | Particles respond to the collective gravitational field; explicit pair encounters are omitted. |
| Collisional | An additional physical rule changes particle motion through encounters. |
| Relaxation | Redistribution of matter, energy or angular momentum toward a more settled state through orbital mixing or encounters. |
| Velocity dispersion / anisotropy | Dispersion is the spread about the mean velocity. Anisotropy describes differences between the radial and tangential spreads. |
| Halo / subhalo | A halo is an extended gravitationally bound distribution. A subhalo is a bound clump inside a larger halo. |
| SIDM / material SIDM | Self-interacting dark matter. “Material” distinguishes encounters between constituent dark matter particles from encounters between whole subhalos. |
| Cross section | An effective interaction area used to calculate encounter probabilities. Some controls specify area per unit mass. |
| Isotropic scattering | Scattering with equal probability per solid angle in the stated local frame. |
| Constant / power-law / Yukawa--Born law | Choices for the scattering cross section: independent of speed, a specified power of speed with low-speed regularization, or a screened-interaction model evaluated in the Born approximation. The selected module fixes the precise formula. |
| Cold initial state | An initial distribution with small random velocity dispersion. Cosmological cold dark matter also specifies a particle model with negligible primordial free streaming on the scales of interest. |
| Static background | Zero prescribed cosmological expansion. Matter and its gravitational field can still evolve. |
| Expanding background / EdS | A prescribed cosmological expansion. Einstein--de Sitter (EdS) is a matter-dominated model, with the time origin and normalization specified by the input. |
| Peculiar velocity | Physical velocity minus the prescribed expansion velocity. Radially, `w=u-Hr`, where `u` is physical velocity, `H` the expansion rate and `r` radius. |
| Density contrast | A density relative to a prescribed homogeneous background. Read the module convention to distinguish a density difference from a fractional contrast. |
| Cusp / core / NFW | A cusp has density increasing steeply toward the center; a core has a nearly flat central density. Navarro--Frenk--White (NFW) names a widely used cusped halo profile. |
| Prompt cusp | A central density cusp associated with early gravitational collapse; an example’s name identifies its selected initial-data family. |
| Equilibrium | A distribution whose macroscopic properties remain steady under the stated equations and boundaries. |
| Virial diagnostic | A comparison of kinetic motion, gravitational forces and boundary terms that helps assess whether a system has settled. |
| Code units | The chosen mass, length and time scales used to express numerical values. |

## The equations and numerical methods

| Term | Meaning |
|---|---|
| Solver / module / executable | A solver advances equations. A module implements a particular model or method. An executable is the compiled program you run. |
| VP / Vlasov--Poisson | Collisionless evolution in the Newtonian gravitational field created by the distribution. |
| BP / Boltzmann--Poisson | Newtonian evolution with an additional encounter term. |
| EV / Einstein--Vlasov | Collisionless matter coupled to the relativistic gravitational field. |
| Eulerian grid / cell / mesh | A fixed collection of position-and-velocity cells storing the distribution. A mesh is the arrangement of those cells. |
| WPIC / marker / packet | Weighted particle-in-cell. A numerical particle carries a specified amount of represented mass; “marker” and, in the collision library, “packet” refer to such numerical particles. |
| Deposition / TSC | Spreading a marker’s mass onto nearby grid cells to calculate gravity. Triangular-shaped-cloud (TSC) is one particular spreading rule. |
| SphDSMC / Monte Carlo | Spherical direct simulation Monte Carlo follows persistent numerical particles and samples local encounters using random numbers. Monte Carlo denotes this use of random sampling. |
| Fokker--Planck / Landau | Descriptions of the accumulated effect of many small deflections through drift and diffusion in velocity. The physical encounter law sets their coefficients. |
| Operator / kernel / backend | An operator is a mathematical rule acting on the distribution. A numerical kernel implements a calculation; in CUDA it is a function executed by GPU threads. A backend is the selected implementation. |
| Finite-angle / quadrature | Finite-angle methods retain individual deflection angles. Quadrature means numerical integration over specified sample points. “Exact” in a program name identifies direct integration of its collision operator; finite resolution still introduces errors. |
| Reversible / sampled / streamed | Reversible methods preserve collision/reverse-collision symmetry. Sampled methods estimate integrals using selected events. Streamed methods process the calculation in batches. |
| Canonical orbit phase / orbit atlas | A representation by bound orbit and position along that orbit. An orbit atlas tabulates the orbits used in that representation. |
| Eddington table | A tabulated equilibrium distribution calculated from a spherical density and potential under the specified velocity-isotropy assumptions. |
| Jacobian | A factor or matrix describing how coordinates, volumes or small displacements change under a transformation. |
| Stiff evolution | Evolution containing very different timescales, so an explicit update may require steps much shorter than the total duration. |
| Norm / residual | A norm measures the size of a quantity or difference. A residual measures how far a computed result departs from satisfying an equation or test. |
| Characteristic / streaming / advection | A characteristic is a trajectory along which the distribution is transported. Streaming or advection advances that transport. |
| Moment / invariant / budget | A moment is an integral of the distribution, such as mass or momentum. An invariant is a conserved quantity under stated conditions. A budget accounts for changes, including boundary flow and sources. |
| KDK / kick--drift--kick | A timestep that updates velocity, advances position, then updates velocity again. |
| CFL condition | The Courant--Friedrichs--Lewy timestep condition limits how far information or matter moves relative to a numerical cell in one step. Collision methods can impose additional rate-based timestep bounds. |
| Collision clock / optical depth | A collision clock tracks elapsed encounter evolution. An optical depth is an accumulated dimensionless encounter rate; it is used to schedule sampled events. |
| Receiver clock | An optional timestep estimate based on changes in the gravitational field sampled at mass-weighted locations. |
| Convergence | Stability of the result as spatial resolution, particle count and timestep accuracy are improved. |
| Locality | Encounters occur at the same physical location. Pairing particles within a finite cell introduces a spatial approximation that needs checking. |
| Closure / calibration | A closure supplies an assumed relation for unresolved quantities. Calibration fixes model parameters using an external reference. |

## Relativistic calculations

| Term | Meaning |
|---|---|
| Metric / constraint | The metric specifies spacetime distances and times. Einstein constraints are equations the gravitational field and matter must satisfy on a chosen time slice. |
| Compactness | `2GM/(rc²)` for enclosed mass `M` and radius `r`. It measures the strength of relativistic gravity. The code often uses `G=c=1`. |
| SC / EF | Schwarzschild and ingoing Eddington--Finkelstein coordinate systems. EF remains regular at a black hole horizon. |
| Areal radius / polar--areal | Areal radius `r` labels a sphere of area `4πr²`. Polar--areal coordinates combine this radius with a particular time slicing. |
| Apparent horizon / trapped region | On a chosen slice, the apparent horizon marks vanishing outward expansion of outgoing light. Inside a trapped region, both future-directed radial light families converge. These are diagnostics tied to the chosen slicing. |
| Excision | Removing a central numerical region behind a causally checked inner boundary and accounting for its retained gravitational mass. |
| Null slice / advanced time / eikonal map | A null slice follows a lightlike surface. Ingoing advanced time labels incoming light surfaces. An eikonal map solves the light-surface equation to compare coordinate descriptions at the same physical events. |
| Lapse / orthonormal frame | The lapse relates coordinate time to a local observer’s proper time for the stated slicing. An orthonormal frame is the local observer’s set of perpendicular, unit-normalized time and spatial directions. |
| Four-momentum / covariant | Four-momentum combines energy and spatial momentum. A covariant equation has a form that consistently transforms between coordinate systems. Covariant momentum components have lower indices and are obtained using the metric. |
| Stress--energy / ADM mass | Stress--energy describes matter’s energy, momentum and stresses, which source relativistic gravity. Arnowitt--Deser--Misner (ADM) mass is the total gravitational mass defined at spatial infinity for an asymptotically flat system. |
| Parity certificate / gate | A certificate records the stated agreement tests between coordinate descriptions. A gate is a script that checks those conditions before allowing a dependent calculation. |

## Building, clusters and files

| Term | Meaning |
|---|---|
| Repository root / checkout | The top-level folder of your local copy of VENUS, containing the main README. |
| Build / compiler / target | A compiler translates source code into a program. A build runs that translation and linking. A target names the particular program or build action requested. |
| CPU / GPU / CUDA | Central processing unit, graphics processing unit, and NVIDIA’s software system for GPU computation. Current VENUS GPU solvers require NVIDIA devices. |
| Host / device / architecture | Host refers to the CPU and its memory; device to the GPU and its memory. The CUDA architecture number selects the device instructions compiled into the program. |
| SDK / library / dependency | A software development kit supplies development files. A library supplies reusable routines. A dependency is software needed to build or run another program. |
| CMake / Make / C++17 | CMake prepares build instructions; Make executes build rules. C++17 names the version of the C++ language required by the source. |
| Environment / module | An environment selects programs, libraries and packages. A cluster’s `module` command selects installed software; this is distinct from a VENUS physics module. |
| Process / thread / shared memory | A process is a running copy of a program. Threads are workers sharing that process’s memory. |
| OpenMP / MPI / rank | Open Multi-Processing divides work between CPU threads. Message Passing Interface exchanges data between processes. A rank is one participating MPI process, also identified by its process number. |
| Reduction / decomposition / slab | A reduction combines contributions, such as summing mass from all workers. Decomposition divides the work. A slab is a contiguous range of cells assigned to one process. |
| Slurm / partition / allocation / QOS | Slurm schedules cluster jobs. A partition groups resources. An allocation reserves them for a job. Quality of service (QOS) sets scheduling limits and priorities. |
| Login node / compute node / walltime | A login node is used to prepare work. Compute nodes execute allocated jobs. Walltime is elapsed real time, including the job’s requested limit. |
| INI / CLI / flag | An INI file lists named settings in plain text. Command-line input (CLI) supplies options after a program name. A flag is an option supplied without a following value. |
| Checkpoint / restart / segment | A checkpoint saves simulation state. A restart reads it to continue evolution. A segment is one uninterrupted part of a resumed calculation. |
| Binary / CSV / JSON | Binary files store data in a program-defined byte format. Comma-separated values (CSV) store tables as text. JavaScript Object Notation (JSON) stores named settings and structured records as text. |
| Schema / contract / sidecar | A schema describes file structure. An input contract states required grid, background and representation conventions. A sidecar is a companion file carrying those descriptions. |
| Metadata / manifest / checksum | Metadata describes a run. A manifest lists files or parameters. A checksum is a content-derived identifier used to check that a file agrees with the recorded version. |
| Cache / ledger / provenance | A cache stores computed results for reuse. A ledger records events or accounted quantities. Provenance records the inputs, software version and commands that produced a result. |
| API / adapter / callback | An application programming interface exposes functions to another program. An adapter connects two interfaces. A callback is a function called at a specified event, such as saving output. |
| Rendering / projection / mask | Rendering turns saved data into figures or movies. A projection integrates or displays selected dimensions of the distribution. A mask marks which cells belong in a calculation or plot. |
