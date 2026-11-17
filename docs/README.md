# VENUS documentation

Start with the [step-by-step examples and script index](../examples/README.md),
then follow the [CPU guide](../cpu/README.md) or [GPU guide](../gpu/README.md).

Read the [solver manual](venus_manual.pdf) for the physical equations,
numerical updates, source explanations and operating instructions. It follows
the calculation from initial data through gravity, transport, collisions and
diagnostics.

Use the [solver-module catalogue](../notes/SOLVER_MODULES.md) to choose the
method for a physical problem. The [script and tool reference](../notes/SCRIPT_REFERENCE.md)
then identifies the programs for generating initial data, running a solver and
analyzing its output.

| Reference | Contents |
|---|---|
| [Solver manual](venus_manual.pdf) | Derivations, algorithms, code excerpts and complete operating reference |
| [CPU guide](../cpu/README.md) | CPU builds, task executables and examples |
| [GPU guide](../gpu/README.md) | CUDA builds, particle data, supported tasks and parallel execution |
| [Relativistic collision guide](relativistic_boltzmann.md) | Local rates, conservative scattering and SC/EF adapters |
| [CPU examples](../cpu/docs/examples.md) | Inputs and commands for the supplied physical setups |
| [Analysis and reproducible figures](../analysis/README.md) | Data analysis and figure reproduction |
| [Movies](../movies/README.md) | Density, slope and virial evolution with their saved physical clocks |

The [editable manual sources](../notes/README.md) remain under `notes/`.
From the repository root, build the checked PDF with:

```sh
make -C notes manual
```

This publishes `docs/venus_manual.pdf`. To render its pages for visual
inspection, use `make -C notes render`; the page images are written under
`notes/tmp/pdfs/`.
