# VENUS documentation

For a first calculation, start with the [CPU guide](../cpu/README.md) or
[GPU guide](../gpu/README.md). The full manual develops the equations and
numerical methods in detail. Discretization means replacing continuous
quantities by values on cells or particles. A constraint is an equation the
computed gravitational field and matter must satisfy. A characteristic is
a particle trajectory; a diagnostic is a quantity used to inspect a result.

Additional terms are defined in the [plain-language glossary](../docs/TERMS.md).

Developed by Uddipan Banik, Frank van den Bosch, and Sachiet Kapur.

The reference is [`../docs/venus_manual.pdf`](../docs/venus_manual.pdf), generated from
[`venus_manual.tex`](venus_manual.tex). Each method begins with the physical problem and governing equations,
then follows their discretization and update order. The operating chapters
explain installation, input files, execution, restarts and analysis. Separate CPU and GPU chapters describe their
data layouts, algorithms, parallel decompositions and supported equations.

Weighted particle-in-cell and spherical direct simulation Monte Carlo chapters
explain Newtonian motion, spherical gravity, local collisions and diagnostics.
The relativistic chapters derive Einstein–Vlasov evolution in Schwarzschild and
ingoing Eddington–Finkelstein coordinates. They give the stress integrals,
particle characteristics, radial constraints, apparent horizon conditions and
coordinate transformations. The dynamical coordinate map follows the ingoing
eikonal equation with its stated branch and outer normalization.

The initial-condition chapter derives the static energy-cutoff shell,
its source integrals, radial equations, vacuum
boundaries, lapse normalization and particle sampling. It then describes the
inward perturbation, angular-momentum rescaling and ADM-mass normalization.
The EF chapter explains both interior evolution and the causally checked
excision mode; [EF interior evolution](EF_INTERIOR_EVOLUTION.md) gives a concise
operating guide and the associated resolution diagnostics.

Use these references for specific tasks:

- [`SOLVER_MODULES.md`](SOLVER_MODULES.md) maps each physical problem to its
  equations, algorithm, source files, executable, inputs and scientific scope.
- [`SCRIPT_REFERENCE.md`](SCRIPT_REFERENCE.md) describes initial-condition
  generators, analyzers, plotting and movie tools, and scheduler scripts.
- [`../docs/venus_manual.pdf`](../docs/venus_manual.pdf) provides the derivations, numerical
  implementation, code excerpts and complete operating reference.

Build and inspect the manual from this directory with:

```sh
make manual
make render
```

`make manual` compiles LaTeX to convergence and checks references. `make render`
renders every page for visual inspection. The module catalogue identifies the solver for each physical problem.
Its entry points and source excerpts connect the commands to the numerical
operations explained in the manual.

The diagnostics chapter defines signed density slopes and positive cusp
indices. It derives shell volumes, Newtonian virial ratios and the units of the
analysis tools, including Sachiet Kapur's density/slope animations, evolution
histories and virialization report.
