# VENUS method conventions

The [README](../README.md) gives the physical equations, representations and
named solvers. The following conventions distinguish models and numerical
choices used across the guides.

| Convention | Meaning in VENUS |
|---|---|
| Static background | Cosmological expansion is zero. The gravitational field remains self-consistent and responds to the relaxing distribution. |
| Prescribed expansion | The expansion history is an input. Its density and velocity conventions are part of the selected Newtonian module. |
| Source density | The density entering Poisson’s equation. Static halos use total density. Expanding modules use the stated density contrast or subtract the homogeneous background from total density. |
| Material self-interacting dark matter (SIDM) | Encounters between constituent dark matter particles. The configured cross section and material mass determine the rate. |
| Subhalo encounters | Scattering and capture between extended clumps. Remnant mass, internal energy and internal velocity dispersion evolve with capture. |
| Weighted particle-in-cell (WPIC) | Markers carry represented mass or invariant rest weight and deposit the appropriate sources on a radial field mesh. The weight convention is specified by the module. |
| Spherical direct simulation Monte Carlo (SphDSMC) | Persistent markers sample local encounters; the gravity calculation uses their spherical mass distribution. |
| Exact finite-angle operator | Direct quadrature of the retained discrete Boltzmann operator. Angular, velocity, spatial and temporal resolution remain finite. |
| Reversible operator | The discretized collision channels preserve collision/reverse-collision symmetry. Sampled and streamed variants change evaluation and storage. |
| Canonical orbit phase | A representation by bound orbit and phase along that orbit. The orbit atlas tabulates the reference orbits and their measures. |
| Collision clocks | Encounter exposure and random state persist across transport steps and restarts. Adaptive collision clocks operate on the fixed spatial mesh. |
| Locality | Finite-cell encounter sampling must resolve the physical spatial variation and mean free path required by the chosen model. |
| Coordinate comparison | Schwarzschild and ingoing Eddington--Finkelstein results are compared at mapped physical events using the solved null coordinate map. |
| Invariant rest weight / ADM mass | Marker rest weights account for represented matter. Arnowitt--Deser--Misner (ADM) mass includes the relativistic energy and binding contributions of the isolated system. |
| Excision | The inner boundary is placed in a causally checked trapped region, with the excised gravitational mass retained in the field budget. |
| Backend | The selected numerical implementation. Its equation, encounter law, representation and supported parallel layout determine applicability. |
| CUDA architecture | The build targets the device’s compute capability. The runtime GPU and toolkit must support those compiled instructions. |
| Message Passing Interface (MPI) decomposition | Collisionless grids divide angular momentum cells. Supported GPU tasks distribute markers and collision-cell ownership. The parallel guides identify replicated fields and collective operations. |
| Restart contract | The binary state and companion files preserve grid, background, representation, clocks and model parameters. Profiles alone contain reduced moments. |
| Validation scope | Interface and component checks establish their tested conditions. A physical application also requires convergence of its observables, conservation budgets, boundaries and encounter locality. |
