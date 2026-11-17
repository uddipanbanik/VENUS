# Examples and initial conditions

Developed by Uddipan Banik and Frank van den Bosch

All paths are relative to `VENUS/cpu`. Eulerian and spherical direct simulation Monte Carlo (SphDSMC) tasks read an
INI file. Dedicated weighted particle-in-cell (WPIC) tasks read the options in a `run.args` file. Pass
them to the named executable exactly as documented by `--help`.

## Introductory INI examples

| Input | Task executable | Purpose |
|---|---|---|
| `examples/einstein_vlasov_low_compactness_virialized_halo.ini` | `venus_cpu_einstein_vlasov_eulerian` | weakly relativistic halo and metric/virial diagnostics |
| `examples/einstein_vlasov_high_compactness_horizon_formation.ini` | `venus_cpu_einstein_vlasov_eulerian` | compact shell and compactness-gap stopping condition |
| `examples/modular/vp_expanding_nfw_gamma2_collisionless.ini` | `venus_cpu_vlasov_poisson_eulerian` | collisionless expanding Einstein--de Sitter (EdS) control |
| `examples/modular/vp_bound_infalling_gaussian_shell_train.ini` | `venus_cpu_vlasov_poisson_eulerian` | bound infalling shell train in a static background |
| `examples/modular/vp_delayed_inner_carrier.ini` | `venus_cpu_vlasov_poisson_eulerian` | delayed-inner-carrier boundary example |
| `examples/modular/bp_expanding_nfw_gamma1p5_weak_collision.ini` | `venus_cpu_boltzmann_poisson_fokker_planck` | expanding weak-collision Boltzmann--Poisson (BP) with softened-gravity coefficients |
| `examples/modular/bp_sidm_corecollapse_yukawa.ini` | `venus_cpu_boltzmann_poisson_fokker_planck` | static Yukawa--Born transport-coefficient core evolution |
| `examples/modular/spherical_dsmc_frank_equilibrium.ini` | `venus_cpu_sidm_sphdsmc_constant` | compact constant-cross-section persistent-marker self-interacting dark matter (SIDM) example |

Build and launch, for example:

```bash
make organized-solvers
./venus_cpu_einstein_vlasov_eulerian --config examples/einstein_vlasov_low_compactness_virialized_halo.ini
./venus_cpu_vlasov_poisson_eulerian --config examples/modular/vp_expanding_nfw_gamma2_collisionless.ini
./venus_cpu_boltzmann_poisson_fokker_planck --config examples/modular/bp_sidm_corecollapse_yukawa.ini
./venus_cpu_sidm_sphdsmc_constant --config examples/modular/spherical_dsmc_frank_equilibrium.ini
```

Each task executable selects one system of equations and one collision operator. An INI that requests
`boltzmann_exact`, for example, belongs to
`venus_cpu_boltzmann_poisson_exact`. It is rejected by the Fokker--Planck
front. The broad `venus_cpu_boltzmann_poisson_finite_angle` and
`venus_modular_solver*` dispatchers are compatibility interfaces, not the
default for a new calculation.

## Example calculations

The [figure reproduction guide](../../analysis/README.md) maps the example
figures to their saved data and plotting commands. The [movie catalog](../../movies/README.md)
links each physical evolution to those figures. The Gaussian
[initial-data example](../examples/modular/vp_bound_gaussian_shell_gpu.md)
constructs the shared static and expanding particle populations.

The introductory INI files above are input-interface examples. Choose their
physical parameters, resolution and domain for the intended calculation.
Scheduler resource and checkpoint patterns are documented under `scripts/slurm/`.

## Initial-condition files

`examples/initial_data/` contains binary grid distributions used by the INI
examples, including:

- weakly relativistic and compact Einstein--Vlasov (EV) shell data.
- expanding zero-flux power-law/halo seeds.
- the bound infalling shell train.
- a static Yukawa--Born SIDM distribution. And
- a sampled Frank equilibrium used to construct the SphDSMC population.

Grid files use the versioned `EVICRWL1` binary format documented in the manual.
The reader verifies dimensions, grid monotonicity, finite values, and exact
payload length. WPIC and SphDSMC restarts are different formats and must not be
substituted for one another. A GPU run starts from a compatible WPIC restart.

## Outputs

Eulerian tasks normally write `metadata.json`, `history.csv`,
`profiles.csv`, optional phase space snapshots, and a serial restart or Message Passing Interface (MPI)
manifest plus rank payloads. SphDSMC adds operator mass budgets and persistent
marker state. WPIC writes `radial_mesh_contract.txt`, time-indexed
`profile_*.csv`, histories, the applicable collision ledger, metadata, and
`restart_state.bin`.

CSV files provide the sampled quantities needed for analysis. The binary
restart stores the complete evolving state and supplies the input for resuming.

## Plotting and GIFs

```bash
python3 tools/plot_einstein_vlasov_outputs.py --input-dir output/ev_run --output-dir output/ev_run/figures --make-gifs
python3 tools/spherical_profile_rendering.py --input-dir output/vp_run --output output/vp_run/density_slope.gif
python3 tools/render_frank_venus_sphdsmc_gif.py --help
python3 tools/render_frank_wpic_gif.py --help
python3 tools/render_weighted_pic_subhalo_comparison_gif.py --help
python3 tools/render_weighted_pic_subhalo_strength_scan_gif.py --help
```

Use a Python environment with NumPy, Matplotlib, SciPy, Pillow, and ImageIO.
Select a plotter matching the output schema and use physical time rather than
assuming that adjacent file numbers are equally spaced.

See [the unified manual](../../docs/venus_manual.pdf) and the
[one-by-one solver catalogue](../../notes/SOLVER_MODULES.md) for equations,
full parameter tables, algorithms, and interpretation limits.
