# Run options

Use the exact executable name from the [README module tables](../README.md#choose-a-module).
The sections below give every recognized long-option name in the shared
runtime parsers, followed by the task-specific executable help text.

An executable selects one equation and encounter family. Choose options
belonging to that family: collisionless Vlasov--Poisson (VP) uses zero encounter strength,
subhalo tasks use subhalo controls, and material tasks use their selected
self-interacting dark matter (SIDM) law. Static/expanding and coordinate restrictions are part of that task.
The shared parser index contains controls for several tasks. Their scope is
set out here and in the [module catalogue](../notes/SOLVER_MODULES.md).

`Flag` means write the option by itself. An integer, number, path or text
option takes its value as the next command-line argument. `true|false` options
take that explicit value. The parameter name identifies its stored control.
The executable help and [manual](venus_manual.pdf) provide physical definitions,
allowed values and numerical conditions. A numerical rate or cross section
uses the physical/code-unit convention of its selected module.

## Select the interface

| Task | Option section | Additional physical restrictions |
|---|---|---|
| CPU Eulerian EV/VP, grid Boltzmann--Poisson (BP) and spherical direct simulation Monte Carlo (SphDSMC) | [Grid and SphDSMC](#cpu-grid-and-sphdsmc) | Named program fixes equation/operator; its INI is the complete input record |
| CPU static/expanding VP and subhalo weighted particle-in-cell (WPIC) | [VP/subhalo WPIC](#cpu-vlasov--poisson-and-subhalo-wpic) | Static VP fixes zero expansion and collisions; subhalo task requires positive strength |
| CPU material SIDM WPIC | [Material SIDM](#cpu-material-sidm-wpic) | Constant task fixes `constant`; velocity-dependent task selects `powerlaw` or `yukawa_born` |
| CPU canonical orbit phase BP | [Canonical orbit phase](#cpu-canonical-orbit-phase-boltzmann--poisson) | Native source grid restart and orbit-atlas controls; `--evolution-time` is a duration |
| CPU low/compact Einstein--Vlasov (EV) WPIC | [EV WPIC](#cpu-einstein--vlasov-wpic) | Low task uses `--low-*`; compact task uses `--high-*` and coordinate controls |
| GPU VP/subhalo/material WPIC | [GPU WPIC](#gpu-wpic) | Task-specific help separates collisionless, subhalo and each material law |

For a new grid profile, choose an analytic construction in its INI or use
`--initial-data binary_rwl --initial-data-file PATH`. Analytic shell controls
include `--R0`, `--dR`, `--beta`, `--amp` and `--w-in`. Select one density
envelope in the INI: for example set `gaussian_density_shell=false` and
`top_hat_density_shell=true` to replace a Gaussian density shell with a top hat.

CPU/GPU WPIC binary inputs retain their particles and schema-specific
companion files. The Gaussian particle generator accepts `--particles` and
`--output-directory`. Changing a native particle distribution uses that
format's generation/import interface. Eulerian `EVICRWL1` files and WPIC
particle files are different representations.

## CPU grid and SphDSMC

Runtime parser: [`cpu/include/ev/config.hpp`](../cpu/include/ev/config.hpp).

| Option | Value syntax | Parameter |
|---|---|---|
| `--L-max` | Number | `L_max` |
| `--NL` | Integer | `NL` |
| `--Nr` | Integer | `Nr` |
| `--Nw` | Integer | `Nw` |
| `--R0` | Number | `R0` |
| `--R0-2` | Number | `R0_2` |
| `--active-floor-abs` | Number | `active_floor_abs` |
| `--active-floor-rel` | Number | `active_floor_rel` |
| `--adaptive-w` | Flag | `adaptive_w` |
| `--adaptive-w-every` | Integer | `adaptive_w_every` |
| `--adaptive-w-growth` | Number | `adaptive_w_growth` |
| `--adaptive-w-margin` | Number | `adaptive_w_margin` |
| `--adaptive-w-max` | Number | `adaptive_w_max` |
| `--adaptive-w-trigger` | Number | `adaptive_w_trigger` |
| `--allow-multi-rank-launch` | Flag | `allow_multi_rank_launch` |
| `--alpha` | Number | `vp_alpha` |
| `--amp` | Number | `amp` |
| `--amp-2` | Number | `amp_2` |
| `--beta` | Number | `beta` |
| `--bp-bath` | Text / path | `bp_bath` |
| `--bp-bath-speed-cutoff` | Number | `bp_bath_speed_cutoff` |
| `--bp-boltzmann-conserve-moments` | true or false | `bp_boltzmann_conserve_moments` |
| `--bp-boltzmann-kernel` | Text / path | `bp_boltzmann_kernel` |
| `--bp-boltzmann-mass-tol` | Number | `bp_boltzmann_mass_tol` |
| `--bp-boltzmann-pair-phi-points` | Integer | `bp_boltzmann_pair_phi_points` |
| `--bp-boltzmann-remap-max-skipped-fraction` | Number | `bp_boltzmann_remap_max_skipped_fraction` |
| `--bp-boltzmann-scatter-mu-points` | Integer | `bp_boltzmann_scatter_mu_points` |
| `--bp-boltzmann-scatter-phi-points` | Integer | `bp_boltzmann_scatter_phi_points` |
| `--bp-coeff-mu-points` | Integer | `bp_coeff_mu_points` |
| `--bp-coeff-refresh-chi` | Number | `bp_coeff_refresh_chi` |
| `--bp-coeff-refresh-every` | Integer | `bp_coeff_refresh_every` |
| `--bp-coeff-refresh-mode` | Text / path | `bp_coeff_refresh_mode` |
| `--bp-coeff-refresh-time-quantile` | Number | `bp_coeff_refresh_time_quantile` |
| `--bp-coeff-speed-points` | Integer | `bp_coeff_speed_points` |
| `--bp-coeff-table-points` | Integer | `bp_coeff_table_points` |
| `--bp-collision-apply-every` | Integer | `bp_collision_apply_every` |
| `--bp-collision-cfl` | Number | `bp_collision_cfl` |
| `--bp-collision-fail-on-nonlocal-cells` | true or false | `bp_collision_fail_on_nonlocal_cells` |
| `--bp-collision-flux` | Text / path | `bp_collision_flux` |
| `--bp-collision-integrator` | Text / path | `bp_collision_integrator` |
| `--bp-collision-l-min-rel` | Number | `bp_collision_l_min_rel` |
| `--bp-collision-locality-max-ratio` | Number | `bp_collision_locality_max_ratio` |
| `--bp-collision-max-substeps` | Integer | `bp_collision_max_substeps` |
| `--bp-collision-operator` | Text / path | `bp_collision_operator` |
| `--bp-collision-splitting` | Text / path | `bp_collision_splitting` |
| `--bp-collision-strength` | Number | `bp_collision_strength` |
| `--bp-coulomb-log` | Number | `bp_coulomb_log` |
| `--bp-cross-section` | Text / path | `bp_cross_section` |
| `--bp-density-floor-abs` | Number | `bp_density_floor_abs` |
| `--bp-density-floor-rel` | Number | `bp_density_floor_rel` |
| `--bp-differential-kernel` | Text / path | `bp_boltzmann_kernel` |
| `--bp-dsmc-collision-epsilon` | Number | `bp_dsmc_collision_epsilon` |
| `--bp-dsmc-gravity-eta` | Number | `bp_dsmc_gravity_eta` |
| `--bp-dsmc-min-shell-particles` | Integer | `bp_dsmc_min_shell_particles` |
| `--bp-dsmc-particles` | Integer | `bp_dsmc_particles` |
| `--bp-dsmc-seed` | Integer | `bp_dsmc_seed` |
| `--bp-dsmc-shell-particles` | Integer | `bp_dsmc_shell_particles` |
| `--bp-dsmc-total-mass` | Number | `bp_dsmc_total_mass` |
| `--bp-epsilon` | Number | `bp_subhalo_radius` |
| `--bp-fp-terms` | Text / path | `bp_fp_terms` |
| `--bp-n` | Number | `bp_sigma_n` |
| `--bp-partner-mass-tol` | Number | `bp_partner_mass_tol` |
| `--bp-reversible-active-shell-floor-fraction` | Number | `bp_reversible_active_shell_floor_fraction` |
| `--bp-reversible-candidate-radius` | Integer | `bp_reversible_candidate_radius` |
| `--bp-reversible-entropy-floor` | Number | `bp_reversible_entropy_floor` |
| `--bp-reversible-entropy-tolerance` | Number | `bp_reversible_entropy_tolerance` |
| `--bp-reversible-explicit-fast-path` | true or false | `bp_reversible_explicit_fast_path` |
| `--bp-reversible-max-velocity-cells` | Integer | `bp_reversible_max_velocity_cells` |
| `--bp-reversible-projection-tolerance` | Number | `bp_reversible_projection_tolerance` |
| `--bp-sampled-activity-boost-max` | Number | `bp_sampled_activity_boost_max` |
| `--bp-sampled-dense-cells` | Integer | `bp_sampled_dense_cells` |
| `--bp-sampled-energy-strata` | Integer | `bp_sampled_energy_strata` |
| `--bp-sampled-equilibrium-fraction` | Number | `bp_sampled_equilibrium_fraction` |
| `--bp-sampled-fail-on-unconverged` | true or false | `bp_sampled_fail_on_unconverged` |
| `--bp-sampled-fixed-geometric-sequence` | true or false | `bp_sampled_fixed_geometric_sequence` |
| `--bp-sampled-guided-fraction` | Number | `bp_sampled_guided_fraction` |
| `--bp-sampled-guided-pairs` | Integer | `bp_sampled_guided_pairs` |
| `--bp-sampled-guided-pilot-samples` | Integer | `bp_sampled_guided_pilot_samples` |
| `--bp-sampled-heat-floor-fraction` | Number | `bp_sampled_heat_floor_fraction` |
| `--bp-sampled-heat-tolerance` | Number | `bp_sampled_heat_tolerance` |
| `--bp-sampled-l-strata` | Integer | `bp_sampled_L_strata` |
| `--bp-sampled-max-samples-per-batch` | Integer | `bp_sampled_max_samples_per_batch` |
| `--bp-sampled-min-samples-per-batch` | Integer | `bp_sampled_min_samples_per_batch` |
| `--bp-sampled-ntc-pair-thinning` | true or false | `bp_sampled_ntc_pair_thinning` |
| `--bp-sampled-ntc-pilot-samples` | Integer | `bp_sampled_ntc_pilot_samples` |
| `--bp-sampled-ntc-quantile` | Number | `bp_sampled_ntc_quantile` |
| `--bp-sampled-operator-refresh-l1-fraction` | Number | `bp_sampled_operator_refresh_l1_fraction` |
| `--bp-sampled-outgoing-angle-samples` | Integer | `bp_sampled_outgoing_angle_samples` |
| `--bp-sampled-phase-density-fraction` | Number | `bp_sampled_phase_density_fraction` |
| `--bp-sampled-replicates-per-estimate` | Integer | `bp_sampled_replicates_per_estimate` |
| `--bp-sampled-reuse-operator-graph` | true or false | `bp_sampled_reuse_operator_graph` |
| `--bp-sampled-rhs-floor-fraction` | Number | `bp_sampled_rhs_floor_fraction` |
| `--bp-sampled-rhs-tolerance` | Number | `bp_sampled_rhs_tolerance` |
| `--bp-sampled-seed` | Integer | `bp_sampled_seed` |
| `--bp-sampled-stratified-fraction` | Number | `bp_sampled_stratified_fraction` |
| `--bp-sampled-stratum-uniform-fraction` | Number | `bp_sampled_stratum_uniform_fraction` |
| `--bp-sampled-tail-fraction` | Number | `bp_sampled_tail_fraction` |
| `--bp-sampled-tail-power` | Number | `bp_sampled_tail_power` |
| `--bp-sampled-uniform-fraction` | Number | `bp_sampled_uniform_fraction` |
| `--bp-sigma-model` | Text / path | `bp_cross_section` |
| `--bp-sigma-n` | Number | `bp_sigma_n` |
| `--bp-sigma-u0` | Number | `bp_sigma_u0` |
| `--bp-sigma0` | Number | `bp_sigma0` |
| `--bp-sts-max-stages` | Integer | `bp_sts_max_stages` |
| `--bp-subhalo-mass` | Number | `bp_subhalo_mass` |
| `--bp-subhalo-radius` | Number | `bp_subhalo_radius` |
| `--bp-temperature-floor` | Number | `bp_temperature_floor` |
| `--bp-u0` | Number | `bp_sigma_u0` |
| `--bp-yukawa-alpha` | Number | `bp_yukawa_alpha` |
| `--bp-yukawa-mchi` | Number | `bp_yukawa_mchi` |
| `--bp-yukawa-mphi` | Number | `bp_yukawa_mphi` |
| `--central-mass` | Number | `central_mass` |
| `--cfl` | Number | `cfl` |
| `--collision-operator` | Text / path | `bp_collision_operator` |
| `--config` | Text / path | Interface control |
| `--custom-ic-normalize-mass` | Flag | `custom_ic_normalize_mass` |
| `--custom-ic-target-mass` | Number | `custom_ic_target_mass` |
| `--custom-radial-coordinate` | Text / path | `custom_radial_coordinate` |
| `--dR` | Number | `dR` |
| `--dR-2` | Number | `dR_2` |
| `--double-gaussian-density-shell` | Flag | `shell_initial_df` |
| `--dt-max` | Number | `dt_max` |
| `--f-floor` | Number | `f_floor` |
| `--gap-stop` | Number | `gap_stop` |
| `--gaussian-density-shell` | Flag | `shell_initial_df` |
| `--help` | Flag | Interface control |
| `--initial-data` | Text / path | `initial_data` |
| `--initial-data-file` | Text / path | `initial_data_file` |
| `--initial-metric-iters` | Integer | `initial_metric_iters` |
| `--inner-boundary` | Text / path | `inner_boundary` |
| `--inner-refine-cells` | Integer | `inner_refine_cells` |
| `--inner-refine-radius` | Number | `inner_refine_radius` |
| `--input` | Text / path | Interface control |
| `--input-file` | Text / path | Interface control |
| `--integrator` | Text / path | `integrator` |
| `--integrator-order` | Integer | `time_integrator_order` |
| `--integrator-substeps` | Integer | `time_integrator_substeps` |
| `--kinetic-eta-grid` | Text / path | `kinetic_eta_grid` |
| `--kinetic-eta-inner-cells` | Integer | `kinetic_eta_inner_cells` |
| `--kinetic-eta-join` | Number | `kinetic_eta_join` |
| `--kinetic-eta-power` | Number | `kinetic_eta_power` |
| `--kinetic-grid` | Text / path | `kinetic_grid` |
| `--kinetic-ic-quadrature-points` | Integer | `kinetic_ic_quadrature_points` |
| `--max-steps` | Integer | `max_steps` |
| `--metric-floor` | Number | `metric_floor` |
| `--model` | Text / path | `physics_model` |
| `--momentum-boundary` | Text / path | `momentum_boundary` |
| `--outer-boundary` | Text / path | `outer_boundary` |
| `--output-dir` | Text / path | `output_dir` |
| `--phase-every` | Integer | `phase_every` |
| `--phase-r-stride` | Integer | `phase_r_stride` |
| `--phase-w-stride` | Integer | `phase_w_stride` |
| `--physics-model` | Text / path | `physics_model` |
| `--r-max` | Number | `r_max` |
| `--r-min` | Number | `r_min` |
| `--radial-coordinate` | Text / path | `custom_radial_coordinate` |
| `--radial-grid` | Text / path | `radial_grid` |
| `--restart-dir` | Text / path | `restart_dir` |
| `--restart-every` | Integer | `restart_every` |
| `--restart-from` | Text / path | `restart_from` |
| `--save-every` | Integer | `save_every` |
| `--shell-initial-df` | Flag | `shell_initial_df` |
| `--t-end` | Number | `t_end` |
| `--threads` | Integer | `threads` |
| `--time-integrator-order` | Integer | `time_integrator_order` |
| `--time-integrator-substeps` | Integer | `time_integrator_substeps` |
| `--top-hat-density-shell` | Flag | `shell_initial_df` |
| `--top-hat-shell` | Flag | `top_hat_shell` |
| `--tracer-initial-data-file` | Text / path | `tracer_initial_data_file` |
| `--vp-alpha` | Number | `vp_alpha` |
| `--vp-background` | Text / path | `vp_universe` |
| `--vp-dynamical-eta` | Number | `vp_dynamical_eta` |
| `--vp-g` | Number | `vp_gravity` |
| `--vp-gravity` | Number | `vp_gravity` |
| `--vp-hamiltonian-reference` | Text / path | `vp_hamiltonian_reference` |
| `--vp-hamiltonian-reference-energy-bins` | Integer | `vp_hamiltonian_reference_energy_bins` |
| `--vp-hamiltonian-reference-quadrature-points` | Integer | `vp_hamiltonian_reference_quadrature_points` |
| `--vp-inflow` | Flag | `vp_outer_inflow` |
| `--vp-inflow-coord-speed` | Number | `vp_inflow_coord_speed` |
| `--vp-inflow-count` | Integer | `vp_inflow_count` |
| `--vp-inflow-freefall-mass` | Number | `vp_inflow_freefall_mass` |
| `--vp-inflow-high-l-center` | Number | `vp_inflow_high_l_center` |
| `--vp-inflow-high-l-delay` | Number | `vp_inflow_high_l_delay` |
| `--vp-inflow-high-shell-mass` | Number | `vp_inflow_high_shell_mass` |
| `--vp-inflow-high-sigma-t` | Number | `vp_inflow_high_sigma_t` |
| `--vp-inflow-high-sigma-u` | Number | `vp_inflow_high_sigma_u` |
| `--vp-inflow-low-l-center` | Number | `vp_inflow_low_l_center` |
| `--vp-inflow-low-l-count` | Integer | `vp_inflow_low_l_count` |
| `--vp-inflow-low-sigma-t` | Number | `vp_inflow_low_sigma_t` |
| `--vp-inflow-period` | Number | `vp_inflow_period` |
| `--vp-inflow-shell-mass` | Number | `vp_inflow_shell_mass` |
| `--vp-inflow-sigma-u` | Number | `vp_inflow_sigma_u` |
| `--vp-inflow-start-time` | Number | `vp_inflow_start_time` |
| `--vp-inflow-time-cutoff` | Number | `vp_inflow_time_cutoff` |
| `--vp-inflow-time-sigma` | Number | `vp_inflow_time_sigma` |
| `--vp-inflow-vff-factor` | Number | `vp_inflow_vff_factor` |
| `--vp-inner-carrier-compress-min-packets` | Text / path | `vp_inner_carrier_compress_min_packets` |
| `--vp-inner-carrier-mass-tolerance` | Number | `vp_inner_carrier_mass_tolerance` |
| `--vp-inner-carrier-max-orbit-segments` | Integer | `vp_inner_carrier_max_orbit_segments` |
| `--vp-inner-carrier-max-packets` | Text / path | `vp_inner_carrier_max_packets` |
| `--vp-inner-carrier-max-refinement` | Integer | `vp_inner_carrier_max_refinement` |
| `--vp-inner-carrier-quadrature-points` | Integer | `vp_inner_carrier_quadrature_points` |
| `--vp-inner-carrier-return-time-bin-width` | Number | `vp_inner_carrier_return_time_bin_width` |
| `--vp-interpolation` | Text / path | `vp_interpolation` |
| `--vp-orbit-energy-tolerance` | Number | `vp_orbit_energy_tolerance` |
| `--vp-orbit-lc-absolute-tolerance` | Number | `vp_orbit_lc_absolute_tolerance` |
| `--vp-orbit-lc-max-refinements` | Integer | `vp_orbit_lc_max_refinements` |
| `--vp-orbit-lc-max-steps` | Integer | `vp_orbit_lc_max_steps` |
| `--vp-orbit-lc-relative-tolerance` | Number | `vp_orbit_lc_relative_tolerance` |
| `--vp-orbit-regularization` | Text / path | `vp_orbit_regularization` |
| `--vp-outer-inflow` | Flag | `vp_outer_inflow` |
| `--vp-poisson-reconstruction` | Text / path | `vp_poisson_reconstruction` |
| `--vp-poisson-source` | Text / path | `vp_poisson_source` |
| `--vp-potential-variation-eta` | Number | `vp_potential_variation_eta` |
| `--vp-sldg-eta-quadrature-points` | Integer | `vp_sldg_eta_quadrature_points` |
| `--vp-sldg-fallback-mass-tolerance` | Number | `vp_sldg_fallback_mass_tolerance` |
| `--vp-sldg-geometry-tolerance` | Number | `vp_sldg_geometry_tolerance` |
| `--vp-sldg-initial-transport-dt` | Number | `vp_sldg_initial_transport_dt` |
| `--vp-sldg-max-area-defect` | Number | `vp_sldg_max_area_defect` |
| `--vp-sldg-max-map-defect` | Number | `vp_sldg_max_map_defect` |
| `--vp-sldg-max-refinement` | Integer | `vp_sldg_max_refinement` |
| `--vp-sldg-max-work-per-subcell` | Integer | `vp_sldg_max_work_per_subcell` |
| `--vp-sldg-subcells` | Integer | `vp_sldg_subcells` |
| `--vp-sldg-target-fallback-fraction` | Number | `vp_sldg_target_fallback_fraction` |
| `--vp-sldg-timestep-growth` | Number | `vp_sldg_timestep_growth` |
| `--vp-sldg-timestep-shrink` | Number | `vp_sldg_timestep_shrink` |
| `--vp-t0` | Number | `vp_time_offset` |
| `--vp-time-offset` | Number | `vp_time_offset` |
| `--vp-timestep-mode` | Text / path | `vp_timestep_mode` |
| `--vp-timestep-omitted-mass-fraction` | Number | `vp_timestep_omitted_mass_fraction` |
| `--vp-transport-interpolation` | Text / path | `vp_interpolation` |
| `--vp-universe` | Text / path | `vp_universe` |
| `--w-center` | Number | `w_center` |
| `--w-in` | Number | `w_in` |
| `--w-in-2` | Number | `w_in_2` |
| `--w-max` | Number | `w_max` |
| `--w-stretch` | Number | `w_stretch` |
| `-h` | Flag | Interface control |
| `-i` | Text / path | Interface control |

## CPU Vlasov--Poisson and subhalo WPIC

Runtime parser: [`cpu/src/drivers/expanding_wpic.cpp`](../cpu/src/drivers/expanding_wpic.cpp).

| Option | Value syntax | Parameter |
|---|---|---|
| `--boundary-diagnostics` | Flag | `boundary_diagnostics` |
| `--collision-prestep-coordinate-due` | Flag | `collision_prestep_coordinate_due` |
| `--dynamical-fraction` | Number | `dynamical_fraction` |
| `--end-time` | Number | `end_time` |
| `--gamma` | Number | `gamma` |
| `--help` | Flag | `show_help` |
| `--history-every` | Integer | `history_every` |
| `--initial-condition-label` | Text / path | `initial_condition_label` |
| `--initial-velocity-factor` | Number | `initial_velocity_factor` |
| `--low-angular-source-ppc-multiplier` | Integer | `low_angular_source_ppc_multiplier` |
| `--low-angular-source-rmax` | Number | `low_angular_source_rmax` |
| `--low-angular-tail-fraction` | Number | `low_angular_tail_fraction` |
| `--low-angular-tail-probability` | Number | `low_angular_tail_probability` |
| `--maximum-timestep` | Number | `maximum_timestep` |
| `--output-directory` | Text / path | `output_directory` |
| `--output-interval` | Number | `output_interval` |
| `--particles-per-radial-cell` | Integer | `particles_per_radial_cell` |
| `--radial-cells` | Integer | `radial_cells` |
| `--radial-maximum` | Number | `radial_maximum` |
| `--radial-minimum` | Number | `radial_minimum` |
| `--restart` | Text / path | `restart` |
| `--restart-every` | Integer | `restart_every` |
| `--subhalo-adaptive-cell-timestep` | Flag | `subhalo_adaptive_cell_timestep` |
| `--subhalo-collision-cfl` | Number | `subhalo_collision_cfl` |
| `--subhalo-collision-interval` | Number | `subhalo_collision_interval` |
| `--subhalo-collision-lanes` | Integer | `subhalo_collision_lanes` |
| `--subhalo-collision-maximum-substeps` | Integer | `subhalo_collision_maximum_substeps` |
| `--subhalo-collision-seed` | Integer | `subhalo_collision_seed` |
| `--subhalo-collision-strength` | Number | `subhalo_collision_strength` |
| `--subhalo-collision-timescale-fraction` | Number | `subhalo_collision_timescale_fraction` |
| `--subhalo-compression-legacy-minimum-step` | Flag | `subhalo_compression_minimize_weight_variance` |
| `--subhalo-compression-maximum-mass-amplification` | Number | `subhalo_compression_maximum_mass_amplification` |
| `--subhalo-compression-minimum-markers` | Integer | `subhalo_compression_minimum_markers` |
| `--subhalo-compression-minimum-reduction-fraction` | Number | `subhalo_compression_minimum_reduction_fraction` |
| `--subhalo-compression-phase-subcells` | Integer | `subhalo_compression_phase_subcells` |
| `--subhalo-compression-preserve-source-cell` | Flag | `subhalo_compression_preserve_source_cell` |
| `--subhalo-compression-tolerance` | Number | `subhalo_compression_tolerance` |
| `--subhalo-compression-trigger-markers` | Integer | `subhalo_compression_trigger_markers` |
| `--subhalo-inner-collision-cells` | Integer | `subhalo_inner_collision_cells` |
| `--subhalo-locality-maximum-ratio` | Number | `subhalo_locality_maximum_ratio` |
| `--subhalo-low-speed-regularization` | Number | `subhalo_low_speed_regularization` |
| `--subhalo-pair-sampling` | Text / path | `subhalo_pair_sampling` |
| `--subhalo-plummer-area-per-mass` | Number | `subhalo_plummer_area_per_mass` |
| `--subhalo-reinjection-factor` | Integer | `subhalo_repopulation_refinement_factor` |
| `--subhalo-reinjection-trigger-markers` | Integer | `subhalo_repopulation_trigger_live_markers` |
| `--subhalo-repopulation-refinement-factor` | Integer | `subhalo_repopulation_refinement_factor` |
| `--subhalo-repopulation-trigger-live-markers` | Integer | `subhalo_repopulation_trigger_live_markers` |
| `--subhalo-velocity-scale` | Number | `subhalo_velocity_scale` |
| `--threads` | Integer | `threads` |
| `--vp-alpha` | Number | `vp_alpha` |
| `--vp-time-offset` | Number | `vp_time_offset` |
| `--vp-universe` | Text / path | `vp_universe` |
| `-h` | Flag | `show_help` |

## CPU material SIDM WPIC

Runtime parser: [`cpu/src/drivers/sidm_wpic.cpp`](../cpu/src/drivers/sidm_wpic.cpp).

| Option | Value syntax | Parameter |
|---|---|---|
| `--allow-nonlocal-collision-shell` | Flag | `fail_on_nonlocal_collision_shell` |
| `--angular-momentum-maximum` | Number | `angular_momentum_maximum` |
| `--collision-adaptive-maximum-interval` | Number | `collision_adaptive_maximum_interval` |
| `--collision-adaptive-shell-timestep` | Flag | `collision_adaptive_shell_timestep` |
| `--collision-cfl` | Number | `collision_cfl` |
| `--collision-event-eta` | Number | `collision_event_eta` |
| `--collision-maximum-substeps` | Integer | `collision_maximum_substeps` |
| `--collision-minimum-shell-markers` | Integer | `collision_minimum_shell_markers` |
| `--collision-strength` | Number | `collision_strength` |
| `--collision-target-shell-markers` | Integer | `collision_target_shell_markers` |
| `--dynamical-eta` | Number | `dynamical_eta` |
| `--end-time` | Number | `end_time` |
| `--eta-cells` | Integer | `eta_cells` |
| `--eta-grid-power` | Number | `eta_grid_power` |
| `--frank-restart` | Text / path | `frank_restart` |
| `--help` | Flag | Interface control |
| `--marker-limit` | Integer | `marker_limit` |
| `--maximum-carrier-mass-fraction` | Number | `maximum_carrier_mass_fraction` |
| `--maximum-timestep` | Number | `maximum_timestep` |
| `--output-directory` | Text / path | `output_directory` |
| `--output-every` | Integer | `output_every` |
| `--phase-space-projection-at-output-only` | Flag | `phase_space_projection_at_events` |
| `--pic-restart` | Text / path | `pic_restart` |
| `--potential-event-eta` | Number | `potential_event_eta` |
| `--radial-cells` | Integer | `radial_cells` |
| `--radial-maximum` | Number | `radial_maximum` |
| `--radial-minimum` | Number | `radial_minimum` |
| `--radial-velocity-cells` | Integer | `radial_velocity_cells` |
| `--radial-velocity-maximum` | Number | `radial_velocity_maximum` |
| `--refresh-restart-field` | Flag | `refresh_restart_field` |
| `--remap-candidate-radius` | Integer | `remap_candidate_radius` |
| `--restart-every` | Integer | `restart_every` |
| `--shape` | Text / path | `shape` |
| `--sidm-cross-section` | Text / path | `sidm_cross_section` |
| `--sidm-power` | Number | `sidm_power` |
| `--sidm-sigma0` | Number | `sidm_sigma0` |
| `--sidm-u0` | Number | `sidm_u0` |
| `--skip-final-phase-space-projection` | Flag | `phase_space_projection_at_final_output` |
| `--threads` | Integer | `threads` |
| `--yukawa-alpha` | Number | `yukawa_alpha` |
| `--yukawa-mchi` | Number | `yukawa_mchi` |
| `--yukawa-mphi` | Number | `yukawa_mphi` |
| `-h` | Flag | Interface control |

## CPU canonical orbit phase Boltzmann--Poisson

Runtime parser: [`cpu/src/apps/boltzmann_poisson/canonical_orbit.cpp`](../cpu/src/apps/boltzmann_poisson/canonical_orbit.cpp).

| Option | Value syntax | Parameter |
|---|---|---|
| `--L-max` | Number | `L_max` |
| `--absorb-detached-carriers-on-field-rebuild` | Text / path | `absorb_detached_carriers_on_field_rebuild` |
| `--atlas-eta-points` | Integer | `atlas_eta_points` |
| `--atlas-initialization` | Text / path | `atlas_initialization` |
| `--boundary-segments` | Integer | `boundary_segments` |
| `--canonical-restart` | Text / path | `canonical_restart` |
| `--collision-balanced-convergence-levels` | Integer | `collision_balanced_convergence_levels` |
| `--collision-batches` | Integer | `collision_batches` |
| `--collision-cfl` | Number | `collision_cfl` |
| `--collision-cfl-safety` | Number | `collision_cfl_safety` |
| `--collision-convergence-levels` | Integer | `collision_convergence_levels` |
| `--collision-coupling` | Text / path | `collision_coupling` |
| `--collision-fraction` | Number | `collision_fraction` |
| `--collision-locality-max-ratio` | Number | `collision_locality_max_ratio` |
| `--collision-marker-convergence-levels` | Integer | `collision_marker_convergence_levels` |
| `--collision-markers` | Integer | `collision_markers` |
| `--collision-markers-per-occupied-cell` | Integer | `collision_markers_per_occupied_cell` |
| `--collision-minimum-shell-markers` | Integer | `collision_minimum_shell_markers` |
| `--collision-strength` | Number | `collision_strength` |
| `--collision-target-shell-markers` | Integer | `collision_target_shell_markers` |
| `--collision-time` | Number | `collision_time` |
| `--core-importance-fraction` | Number | `core_importance_fraction` |
| `--core-importance-radius` | Number | `core_importance_radius` |
| `--direct-fh-diagnostic-only` | Text / path | `direct_fh_diagnostic_only` |
| `--eddington-table` | Text / path | `eddington_table` |
| `--energy-power` | Number | `energy_power` |
| `--energy-rows` | Integer | `energy_rows` |
| `--eta-grid` | Text / path | `eta_grid` |
| `--eta-inner-cells` | Integer | `eta_inner_cells` |
| `--eta-join` | Number | `eta_join` |
| `--eta-power` | Number | `eta_power` |
| `--event-dt` | Number | `event_dt` |
| `--event-dt-convergence-levels` | Integer | `event_dt_convergence_levels` |
| `--event-dt-convergence-time` | Number | `event_dt_convergence_time` |
| `--evolution-events` | Integer | `evolution_events` |
| `--evolution-time` | Number | `evolution_time` |
| `--help` | Flag | Interface control |
| `--mapping-energy-points` | Integer | `mapping_energy_points` |
| `--mapping-phase-points` | Integer | `mapping_phase_points` |
| `--maximum-collision-batches` | Integer | `maximum_collision_batches` |
| `--maximum-collision-escaped-increment-fraction` | Number | `maximum_collision_escaped_increment_fraction` |
| `--maximum-collision-excluded-open-mass-fraction` | Number | `maximum_collision_excluded_open_mass_fraction` |
| `--maximum-collision-moment-repair-mass-fraction` | Number | `maximum_collision_moment_repair_mass_fraction` |
| `--maximum-detached-mass-fraction` | Number | `maximum_detached_mass_fraction` |
| `--maximum-lift-phase-cell-error` | Number | `maximum_lift_phase_cell_error` |
| `--maximum-open-mass-fraction` | Number | `maximum_open_mass_fraction` |
| `--maximum-unrepresented-mass-fraction` | Number | `maximum_unrepresented_mass_fraction` |
| `--minimum-energy-guard-fraction` | Number | `minimum_energy_guard_fraction` |
| `--minimum-shell-trials` | Number | `minimum_shell_trials` |
| `--nl` | Integer | `nl` |
| `--nr` | Integer | `nr` |
| `--nw` | Integer | `nw` |
| `--open-compression-subcells` | Integer | `open_compression_subcells` |
| `--open-compression-tolerance` | Number | `open_compression_tolerance` |
| `--open-compression-trigger` | Integer | `open_compression_trigger` |
| `--output-directory` | Text / path | `output_directory` |
| `--output-every` | Integer | `output_every` |
| `--phase-cells` | Integer | `phase_cells` |
| `--potential-fraction` | Number | `potential_fraction` |
| `--r-max` | Number | `r_max` |
| `--r-min` | Number | `r_min` |
| `--reconstruction-cycles` | Integer | `reconstruction_cycles` |
| `--restart` | Text / path | `restart` |
| `--restart-every` | Integer | `restart_every` |
| `--reversible-operator` | Text / path | `reversible_operator` |
| `--sampled-dense-cells` | Integer | `sampled_dense_cells` |
| `--sampled-fail-on-unconverged` | Text / path | `sampled_fail_on_unconverged` |
| `--sampled-guided-fraction` | Number | `sampled_guided_fraction` |
| `--sampled-heat-tolerance` | Number | `sampled_heat_tolerance` |
| `--sampled-max-samples` | Integer | `sampled_max_samples` |
| `--sampled-min-samples` | Integer | `sampled_min_samples` |
| `--sampled-rhs-tolerance` | Number | `sampled_rhs_tolerance` |
| `--source-row-fraction` | Number | `source_row_fraction` |
| `--startup-field-iterations` | Integer | `startup_field_iterations` |
| `--startup-field-tolerance` | Number | `startup_field_tolerance` |
| `--threads` | Integer | `threads` |
| `--transport-time` | Number | `transport_time` |
| `--w-max` | Number | `w_max` |
| `-h` | Flag | Interface control |

## CPU Einstein--Vlasov WPIC

Runtime parser: [`cpu/src/drivers/einstein_vlasov_wpic.cpp`](../cpu/src/drivers/einstein_vlasov_wpic.cpp).

| Option | Value syntax | Parameter |
|---|---|---|
| `--coordinates` | Text / path | `coordinates` |
| `--ef-evolve-interior` | Flag | `ef_enable_excision` |
| `--ef-excision-a` | Number | `ef_excision_a` |
| `--ef-excision-buffer-cells` | Integer | `ef_excision_buffer_cells` |
| `--ef-excision-horizon-fraction` | Number | `ef_excision_horizon_fraction` |
| `--ef-map-worldtube-radius` | Number | `ef_map_worldtube_radius` |
| `--ef-null-donor-duration` | Number | `ef_null_donor_duration` |
| `--ef-null-initial-data` | Text / path | `ef_null_initial_data_path` |
| `--expected-outcome` | Text / path | `expected_outcome` |
| `--export-ef-null-initial-data` | Text / path | `export_ef_null_initial_data_path` |
| `--help` | Flag | Interface control |
| `--high-angular-momentum-scale` | Number | `high_angular_momentum_scale` |
| `--high-dt-max` | Number | `high_dt_max` |
| `--high-energy-cutoff-amplitude` | Number | `high_energy_cutoff_amplitude` |
| `--high-energy-cutoff-central-y` | Number | `high_energy_cutoff_central_y` |
| `--high-energy-cutoff-ell0` | Number | `high_energy_cutoff_ell0` |
| `--high-energy-cutoff-power` | Number | `high_energy_cutoff_power` |
| `--high-gap-stop` | Number | `high_gap_stop` |
| `--high-infall-core-radius` | Number | `high_infall_core_radius` |
| `--high-initial-condition` | Text / path | `high_initial_condition` |
| `--high-inner-boundary` | Text / path | `high_inner_boundary` |
| `--high-markers` | Integer | `high_markers` |
| `--high-output-steps` | Integer | `high_output_steps` |
| `--high-r-max` | Number | `high_r_max` |
| `--high-r-min` | Number | `high_r_min` |
| `--high-radial-cells` | Integer | `high_radial_cells` |
| `--high-radial-grid` | Text / path | `high_radial_grid` |
| `--high-radial-importance-exponent` | Number | `high_radial_importance_exponent` |
| `--high-radial-infall` | Number | `high_radial_infall` |
| `--high-sampling-realization` | Integer | `high_sampling_realization` |
| `--high-scale-radius` | Number | `high_scale_radius` |
| `--high-shell-center` | Number | `high_shell_center` |
| `--high-shell-radial-importance-factor` | Number | `high_shell_radial_importance_factor` |
| `--high-shell-radial-momentum` | Number | `high_shell_radial_momentum` |
| `--high-shell-radial-momentum-dispersion` | Number | `high_shell_radial_momentum_dispersion` |
| `--high-shell-tangential-velocity-dispersion` | Number | `high_shell_tangential_velocity_dispersion` |
| `--high-shell-width` | Number | `high_shell_width` |
| `--high-specific-angular-momentum` | Number | `high_specific_angular_momentum` |
| `--high-t-end` | Number | `high_t_end` |
| `--high-target-initial-adm-mass` | Number | `high_target_initial_adm_mass` |
| `--high-total-mass` | Number | `high_total_mass` |
| `--high-velocity-factor` | Number | `high_velocity_factor` |
| `--initialize-only` | Flag | `initialize_only` |
| `--low-cfl` | Number | `low_cfl` |
| `--low-dt-max` | Number | `low_dt_max` |
| `--low-markers` | Integer | `low_markers` |
| `--low-output-steps` | Integer | `low_output_steps` |
| `--low-radial-cells` | Integer | `low_radial_cells` |
| `--low-radial-importance-exponent` | Number | `low_radial_importance_exponent` |
| `--low-t-end` | Number | `low_t_end` |
| `--low-total-mass` | Number | `low_total_mass` |
| `--low-velocity-factor` | Number | `low_velocity_factor` |
| `--output-dir` | Text / path | `output_dir` |
| `--physical-ic` | Text / path | `physical_ic_path` |
| `--quick` | Flag | `quick` |
| `--resume` | Flag | `resume` |
| `--scenario` | Text / path | `scenario` |
| `--spacetime-coordinates` | Text / path | `coordinates` |
| `--threads` | Integer | `threads` |
| `--timestep-omitted-mass-fraction` | Number | `timestep_omitted_mass_fraction` |
| `-h` | Flag | Interface control |

## GPU WPIC

Runtime parser: [`gpu/src/wpic_production_driver.cu`](../gpu/src/wpic_production_driver.cu).

| Option | Value syntax | Parameter |
|---|---|---|
| `--collision-prestep-coordinate-due` | Flag | `collision_prestep_coordinate_due` |
| `--coupled-field-trace` | Flag | `coupled_field_trace` |
| `--coupled-receiver-clocks` | Flag | `coupled_receiver_clocks` |
| `--cross-section-model` | Text / path | `material_cross_section` |
| `--dynamical-fraction` | Number | `dynamical_fraction` |
| `--end-time` | Number | `end_time` |
| `--help` | Flag | `show_help` |
| `--history-every` | Integer | `history_every` |
| `--maximum-timestep` | Number | `maximum_timestep` |
| `--output-directory` | Text / path | `output` |
| `--output-interval` | Number | `output_interval` |
| `--phase-space-snapshot-times` | Text / path | `phase_space_snapshot_times` |
| `--radial-cells` | Integer | `radial_cells` |
| `--radial-maximum` | Number | `radial_maximum` |
| `--radial-minimum` | Number | `radial_minimum` |
| `--restart` | Text / path | `restart` |
| `--restart-every` | Integer | `restart_every` |
| `--sidm-adaptive-shell-timestep` | Flag | `material_adaptive_shell_timestep` |
| `--sidm-collision-cfl` | Number | `collision_cfl` |
| `--sidm-collision-interval` | Number | `maximum_collision_interval` |
| `--sidm-collision-maximum-substeps` | Integer | `collision_maximum_substeps` |
| `--sidm-collision-strength` | Number | `material_collision_strength` |
| `--sidm-fail-on-nonlocal-shell` | Flag | `material_fail_on_nonlocal_shell` |
| `--sidm-minimum-shell-markers` | Integer | `material_minimum_shell_markers` |
| `--sidm-sigma-power` | Number | `material_sigma_power` |
| `--sidm-sigma-u0` | Number | `material_sigma_u0` |
| `--sidm-sigma0` | Number | `material_sigma0` |
| `--sidm-target-shell-markers` | Integer | `material_target_shell_markers` |
| `--sidm-yukawa-alpha` | Number | `material_yukawa_alpha` |
| `--sidm-yukawa-mchi` | Number | `material_yukawa_mchi` |
| `--sidm-yukawa-mphi` | Number | `material_yukawa_mphi` |
| `--subhalo-adaptive-batch-substeps` | Integer | `subhalo_adaptive_batch_substeps` |
| `--subhalo-adaptive-cell-timestep` | Flag | `subhalo_adaptive_cell_timestep` |
| `--subhalo-collision-adaptive-batch-substeps` | Integer | `subhalo_adaptive_batch_substeps` |
| `--subhalo-collision-adaptive-cell-timestep` | Integer | `subhalo_adaptive_cell_timestep` |
| `--subhalo-collision-cfl` | Number | `collision_cfl` |
| `--subhalo-collision-interval` | Number | `maximum_collision_interval` |
| `--subhalo-collision-lanes` | Integer | `collision_lanes` |
| `--subhalo-collision-maximum-substeps` | Integer | `collision_maximum_substeps` |
| `--subhalo-collision-strength` | Number | `collision_strength` |
| `--subhalo-collision-timescale-fraction` | Number | `collision_timescale_fraction` |
| `--subhalo-inner-collision-cells` | Integer | `inner_collision_cells` |
| `--subhalo-locality-maximum-ratio` | Number | `locality_maximum_ratio` |
| `--subhalo-low-speed-regularization` | Number | `low_speed_regularization` |
| `--subhalo-plummer-area-per-mass` | Number | `plummer_area_per_mass` |
| `--subhalo-velocity-scale` | Number | `velocity_scale` |
| `--universe` | Text / path | `universe` |
| `-h` | Flag | `show_help` |

## Named executable help

Each block lists the interface of its exact program. The parser tables above
include advanced options and aliases in addition to these summaries.

<details>
<summary><code>venus_cpu_boltzmann_poisson_canonical_orbit</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_canonical_orbit --restart PATH [options]

Required initial state:
  --restart PATH                 source cell-average Eulerian restart
  --eddington-table PATH         optional stationary f(H) table
  --canonical-restart PATH       resume a canonical-atlas state

Atlas and physical evolution:
  --energy-rows N --phase-cells N --atlas-eta-points N
  --evolution-time T | --evolution-events N
  --event-dt DT --potential-fraction X --collision-fraction X
  --collision-strength S --collision-coupling NAME
  --reversible-operator {exact|reversible|sampled|streamed}

Grid, output, and execution:
  --nr N --nw N --nl N --r-min R --r-max R --w-max W --L-max L
  --output-directory DIR --output-every N --restart-every N
  --threads N

Advanced atlas, compression, sampling, locality, and convergence
controls are described in the VENUS manual.
  --help, -h                     show this summary
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_exact</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_exact --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_exact
Required physics: Boltzmann-Poisson with bp_collision_operator=boltzmann_exact.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_expanding_prompt_cusp</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_expanding_prompt_cusp --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_expanding_prompt_cusp
Required physics: expanding density-contrast exact BP with softened-gravity collisions.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_finite_angle</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_finite_angle --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_finite_angle
Required physics: Boltzmann-Poisson with an exact or reversible finite-angle operator.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_fokker_planck</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_fokker_planck --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_fokker_planck
Required physics: Boltzmann-Poisson with bp_collision_operator=fokker_planck.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_landau</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_landau --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_landau
Required physics: Boltzmann-Poisson with bp_collision_operator=landau_fp.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_reversible</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_reversible --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_reversible
Required physics: Boltzmann-Poisson with bp_collision_operator=boltzmann_reversible.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_reversible_sampled</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_reversible_sampled --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_reversible_sampled
Required physics: Boltzmann-Poisson with bp_collision_operator=boltzmann_reversible_sampled.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_boltzmann_poisson_reversible_streamed</code></summary>

```text
Usage: cpu/venus_cpu_boltzmann_poisson_reversible_streamed --config FILE [task-compatible overrides]

Locked CPU task profile: boltzmann_poisson_reversible_streamed
Required physics: Boltzmann-Poisson with bp_collision_operator=boltzmann_reversible_streamed.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_einstein_vlasov_eulerian</code></summary>

```text
Usage: cpu/venus_cpu_einstein_vlasov_eulerian --config FILE [task-compatible overrides]

Locked CPU task profile: einstein_vlasov_eulerian
Required physics: collisionless physics_model=einstein_vlasov.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_einstein_vlasov_wpic_black_hole</code></summary>

```text
Usage: cpu/venus_cpu_einstein_vlasov_wpic_black_hole [--output-dir PATH] [--threads N] [--physical-ic PATH] [--initialize-only] [--export-ef-null-initial-data PATH] [--ef-null-initial-data PATH] [--ef-null-donor-duration T] [--ef-map-worldtube-radius R_B] [--coordinates schwarzschild|eddington_finkelstein] [--expected-outcome black_hole|halo|any] [--high-markers N] [--high-radial-cells N] [--high-t-end T] [--high-dt-max DT] [--high-gap-stop G] [--high-r-min R] [--high-r-max R] [--high-total-mass M] [--high-initial-condition shell|plummer|energy_cutoff_shell] [--high-shell-center R] [--high-shell-width SIGMA] [--high-shell-radial-importance-factor F] [--high-shell-radial-momentum W] [--high-shell-radial-momentum-dispersion SIGMA_W] [--high-shell-tangential-velocity-dispersion SIGMA_VT] [--high-energy-cutoff-amplitude C] [--high-energy-cutoff-ell0 ELL0] [--high-energy-cutoff-central-y Y0] [--high-energy-cutoff-power K] [--high-specific-angular-momentum L] [--high-angular-momentum-scale S] [--high-sampling-realization UINT64] [--high-target-initial-adm-mass M] [--timestep-omitted-mass-fraction F] [--ef-excision-a A] [--ef-excision-buffer-cells N] [--ef-excision-horizon-fraction F] [--ef-evolve-interior] [--high-inner-boundary reflecting|origin_parity] [--high-radial-grid log|linear] [--high-output-steps N] [--resume] [--quick]

Locked task profile: high-compactness collapse / black-hole-formation Einstein-Vlasov WPIC.  --scenario is unnecessary and, if supplied, must be high.
```

</details>

<details>
<summary><code>venus_cpu_einstein_vlasov_wpic_low_compactness</code></summary>

```text
Usage: cpu/venus_cpu_einstein_vlasov_wpic_low_compactness [--output-dir PATH] [--threads N] [--physical-ic PATH] [--initialize-only] [--export-ef-null-initial-data PATH] [--ef-null-initial-data PATH] [--ef-null-donor-duration T] [--ef-map-worldtube-radius R_B] [--low-markers N] [--low-radial-cells N] [--low-t-end T] [--low-dt-max DT] [--low-cfl C] [--low-total-mass M] [--low-velocity-factor F] [--low-radial-importance-exponent P] [--timestep-omitted-mass-fraction F] [--low-output-steps N] [--resume] [--quick]

Locked task profile: low-compactness virialized-halo Einstein-Vlasov WPIC.  --scenario is unnecessary and, if supplied, must be low.
```

</details>

<details>
<summary><code>venus_cpu_sidm_sphdsmc_constant</code></summary>

```text
Usage: cpu/venus_cpu_sidm_sphdsmc_constant --config FILE [task-compatible overrides]

Locked CPU task profile: sidm_sphdsmc_constant
Required physics: SIDM spherical_dsmc with a constant isotropic cross section.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_sidm_sphdsmc_velocity_dependent</code></summary>

```text
Usage: cpu/venus_cpu_sidm_sphdsmc_velocity_dependent --config FILE [task-compatible overrides]

Locked CPU task profile: sidm_sphdsmc_velocity_dependent
Required physics: SIDM spherical_dsmc with a supported velocity-dependent cross section.

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_sidm_wpic_constant</code></summary>

```text
Usage: cpu/venus_cpu_sidm_wpic_constant (--frank-restart PATH | --pic-restart PATH) [options]
Locked task profile: constant-cross-section SIDM WPIC; --sidm-cross-section may only be constant and collision strength must be positive.
  --output-directory DIR --end-time X --marker-limit N
  --refresh-restart-field (rebuild field after changing restart controls)
  --maximum-timestep X --dynamical-eta X
  --potential-event-eta X --collision-event-eta X
  --radial-cells N --radial-velocity-cells N --eta-cells N
  --radial-minimum X --radial-maximum X
  --radial-velocity-maximum X --angular-momentum-maximum X
  --eta-grid-power X --remap-candidate-radius N
  --maximum-carrier-mass-fraction X
  --phase-space-projection-at-output-only
  --skip-final-phase-space-projection
  --output-every N --restart-every N --threads N --shape cic|tsc
  --sidm-cross-section constant
  --collision-strength X --sidm-sigma0 X --sidm-u0 X --sidm-power X
  --yukawa-alpha X --yukawa-mchi X --yukawa-mphi X
  --collision-target-shell-markers N
  --collision-minimum-shell-markers N --collision-cfl X
  --collision-maximum-substeps N
  --collision-adaptive-shell-timestep
  --collision-adaptive-maximum-interval X
  --allow-nonlocal-collision-shell
  --help, -h
```

</details>

<details>
<summary><code>venus_cpu_sidm_wpic_velocity_dependent</code></summary>

```text
Usage: cpu/venus_cpu_sidm_wpic_velocity_dependent (--frank-restart PATH | --pic-restart PATH) [options]
Locked task profile: velocity-dependent SIDM WPIC; --sidm-cross-section must be powerlaw or yukawa_born and collision strength must be positive.
  --output-directory DIR --end-time X --marker-limit N
  --refresh-restart-field (rebuild field after changing restart controls)
  --maximum-timestep X --dynamical-eta X
  --potential-event-eta X --collision-event-eta X
  --radial-cells N --radial-velocity-cells N --eta-cells N
  --radial-minimum X --radial-maximum X
  --radial-velocity-maximum X --angular-momentum-maximum X
  --eta-grid-power X --remap-candidate-radius N
  --maximum-carrier-mass-fraction X
  --phase-space-projection-at-output-only
  --skip-final-phase-space-projection
  --output-every N --restart-every N --threads N --shape cic|tsc
  --sidm-cross-section powerlaw|yukawa_born
  --collision-strength X --sidm-sigma0 X --sidm-u0 X --sidm-power X
  --yukawa-alpha X --yukawa-mchi X --yukawa-mphi X
  --collision-target-shell-markers N
  --collision-minimum-shell-markers N --collision-cfl X
  --collision-maximum-substeps N
  --collision-adaptive-shell-timestep
  --collision-adaptive-maximum-interval X
  --allow-nonlocal-collision-shell
  --help, -h
```

</details>

<details>
<summary><code>venus_cpu_subhalo_wpic</code></summary>

```text
Usage: cpu/venus_cpu_subhalo_wpic [options]

Locked task profile: Newtonian collisional subhalo WPIC.
--subhalo-collision-strength must be strictly positive; zero is rejected.

Radial mesh controls for fresh resolution studies:
  --radial-cells N       logarithmic radial cells (default: 256)
  --radial-minimum R     first resolved radius (default: 0.1)
  --radial-maximum R     outer resolved radius (default: 1000)
  --initial-condition-label NAME  describe an explicit native input [powerlaw]
  --boundary-diagnostics  count accepted reflecting impacts and radius maxima [off]

A finer mesh reduces the physical collision-cell width Delta r; it does not relax
the Delta-r/lambda_mfp locality test.  When continuing a restart, repeat the
same radial mesh values used by the fresh run.

Background controls (fresh runs and matching restarts):
  --vp-universe expanding|static     default: expanding
  --vp-alpha X                       0 <= X <= 1; default: 2/3
  --vp-time-offset X                 positive; default: 12
  Static mode sets alpha=0; the supplied density is a cusp IC.
  --initial-velocity-factor F       positive physical IC velocity factor; default: 1
  Fresh data use u_new=F*u and eta_new=F^2*eta. Repeat F on resume.

Collision-clock controls:
  --subhalo-collision-interval X       global operator-batch cap
  --subhalo-collision-timescale-fraction X
  --subhalo-collision-cfl X
  --subhalo-collision-maximum-substeps N
  --subhalo-adaptive-cell-timestep     independent cell CFL clocks
  --collision-prestep-coordinate-due   apply safe coordinate-close deadlines before KDK

Orbit, sampling, output and restart controls:
  --gamma X                         cusp slope (default: 1.5)
  --particles-per-radial-cell N      fresh marker sampling (default: 2048)
  --end-time T                      final physical time (default: 2)
  --maximum-timestep DT             fixed KDK cap (default: 4)
  --output-interval DT              profile spacing (default: 1)
  --history-every N                 history step cadence (default: 100)
  --restart-every N                 checkpoint step cadence (default: 1000)
  --threads N                       shared-memory workers (default: 1)
  --output-directory PATH           run products
  --restart PATH                    complete native particle state

Use --help or -h to print this summary.  Established orbit, marker, output, restart, and task-compatible collision controls remain available.
```

</details>

<details>
<summary><code>venus_cpu_vlasov_poisson_eulerian</code></summary>

```text
Usage: cpu/venus_cpu_vlasov_poisson_eulerian --config FILE [task-compatible overrides]

Locked CPU task profile: vlasov_poisson_eulerian
Required physics: collisionless physics_model=vlasov_poisson (static or expanding).

Common controls include --threads N, --t-end T, --dt-max DT, --output-dir DIR, and --restart-dir DIR.  The INI file remains the complete parameter record.  Any incompatible model, collision operator, background, or cross section is rejected before state and output construction.
Use venus_modular_solver --help when a single command must select the physics model dynamically.
```

</details>

<details>
<summary><code>venus_cpu_vlasov_poisson_wpic_expanding</code></summary>

```text
Usage: cpu/venus_cpu_vlasov_poisson_wpic_expanding [options]

Locked task profile: collisionless Vlasov-Poisson WPIC.
--subhalo-collision-strength is pinned to 0; every nonzero value is rejected.

Radial mesh controls for fresh resolution studies:
  --radial-cells N       logarithmic radial cells (default: 256)
  --radial-minimum R     first resolved radius (default: 0.1)
  --radial-maximum R     outer resolved radius (default: 1000)
  --initial-condition-label NAME  describe an explicit native input [powerlaw]
  --boundary-diagnostics  count accepted reflecting impacts and radius maxima [off]

A finer mesh reduces the physical collision-cell width Delta r; it does not relax
the Delta-r/lambda_mfp locality test.  When continuing a restart, repeat the
same radial mesh values used by the fresh run.

Background controls (fresh runs and matching restarts):
  --vp-universe expanding|static     default: expanding
  --vp-alpha X                       0 <= X <= 1; default: 2/3
  --vp-time-offset X                 positive; default: 12
  Static mode sets alpha=0; the supplied density is a cusp IC.
  --initial-velocity-factor F       positive physical IC velocity factor; default: 1
  Fresh data use u_new=F*u and eta_new=F^2*eta. Repeat F on resume.

Collision-clock controls:
  --subhalo-collision-interval X       global operator-batch cap
  --subhalo-collision-timescale-fraction X
  --subhalo-collision-cfl X
  --subhalo-collision-maximum-substeps N
  --subhalo-adaptive-cell-timestep     independent cell CFL clocks
  --collision-prestep-coordinate-due   apply safe coordinate-close deadlines before KDK

Orbit, sampling, output and restart controls:
  --gamma X                         cusp slope (default: 1.5)
  --particles-per-radial-cell N      fresh marker sampling (default: 2048)
  --end-time T                      final physical time (default: 2)
  --maximum-timestep DT             fixed KDK cap (default: 4)
  --output-interval DT              profile spacing (default: 1)
  --history-every N                 history step cadence (default: 100)
  --restart-every N                 checkpoint step cadence (default: 1000)
  --threads N                       shared-memory workers (default: 1)
  --output-directory PATH           run products
  --restart PATH                    complete native particle state

Use --help or -h to print this summary.  Established orbit, marker, output, restart, and task-compatible collision controls remain available.
```

</details>

<details>
<summary><code>venus_cpu_vlasov_poisson_wpic_static</code></summary>

```text
Usage: cpu/venus_cpu_vlasov_poisson_wpic_static [options]

Locked task profile: collisionless static Vlasov-Poisson WPIC.
--vp-universe static and zero collision strength are required.

Radial mesh controls for fresh resolution studies:
  --radial-cells N       logarithmic radial cells (default: 256)
  --radial-minimum R     first resolved radius (default: 0.1)
  --radial-maximum R     outer resolved radius (default: 1000)
  --initial-condition-label NAME  describe an explicit native input [powerlaw]
  --boundary-diagnostics  count accepted reflecting impacts and radius maxima [off]

A finer mesh reduces the physical collision-cell width Delta r; it does not relax
the Delta-r/lambda_mfp locality test.  When continuing a restart, repeat the
same radial mesh values used by the fresh run.

Background controls (fresh runs and matching restarts):
  --vp-universe static              fixed static background
  --vp-alpha X                       0 <= X <= 1; default: 2/3
  --vp-time-offset X                 positive; default: 12
  Static mode sets alpha=0; the supplied density is a cusp IC.
  --initial-velocity-factor F       positive physical IC velocity factor; default: 1
  Fresh data use u_new=F*u and eta_new=F^2*eta. Repeat F on resume.

Collision-clock controls:
  --subhalo-collision-interval X       global operator-batch cap
  --subhalo-collision-timescale-fraction X
  --subhalo-collision-cfl X
  --subhalo-collision-maximum-substeps N
  --subhalo-adaptive-cell-timestep     independent cell CFL clocks
  --collision-prestep-coordinate-due   apply safe coordinate-close deadlines before KDK

Orbit, sampling, output and restart controls:
  --gamma X                         cusp slope (default: 1.5)
  --particles-per-radial-cell N      fresh marker sampling (default: 2048)
  --end-time T                      final physical time (default: 2)
  --maximum-timestep DT             fixed KDK cap (default: 4)
  --output-interval DT              profile spacing (default: 1)
  --history-every N                 history step cadence (default: 100)
  --restart-every N                 checkpoint step cadence (default: 1000)
  --threads N                       shared-memory workers (default: 1)
  --output-directory PATH           run products
  --restart PATH                    complete native particle state

Use --help or -h to print this summary.  Established orbit, marker, output, restart, and task-compatible collision controls remain available.
```

</details>

<details>
<summary><code>venus_gpu_sidm_boltzmann_poisson</code></summary>

```text
VENUS GPU static-WPIC sidm_boltzmann_poisson production driver

Usage:
  build/gpu/venus_gpu_sidm_boltzmann_poisson --restart FILE --output-directory DIR --end-time T [options]

Required:
  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.
  --output-directory DIR                Run directory for profiles, histories, and restarts.
  --end-time T                          Final simulation time; T > restart time and T > 0.

Evolution and output:
  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.
  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].
  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].
  --output-interval DT                  Profile-time spacing; DT > 0 [10].
  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].
  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].
  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].

Logarithmic radial mesh:
  --radial-cells N                      Number of cells; N >= 2 [256].
  --radial-minimum R                    Inner face; R > 0 [0.1].
  --radial-maximum R                    Outer face; R > radial minimum [1000].

Material-SIDM collisions:
  --sidm-collision-strength S           Positive collision-rate multiplier.
  --cross-section-model MODEL           Required: constant, power_law, or yukawa_born.
  --sidm-sigma0 X                       Constant/power-law normalization; X > 0.
  --sidm-sigma-power P                  Power-law exponent; P > 0 [4].
  --sidm-sigma-u0 U                     Power-law softening speed; U > 0.
  --sidm-yukawa-alpha A                 Effective kernel amplitude; A > 0 [0.01].
  --sidm-yukawa-mchi M                  Positive shape parameter [1].
  --sidm-yukawa-mphi M                  Shape parameter; mphi/mchi sets speed [1].
  --sidm-collision-interval DT          Global Boltzmann operator-batch cap; DT > 0 [0.5].
  --sidm-collision-cfl C                Per-substep probability bound; 0 < C <= 1 [0.2].
  --sidm-collision-maximum-substeps N   Per-event safety limit; N >= 1 [16384].
  --sidm-adaptive-shell-timestep        Independent shell-local CFL clocks.
  --sidm-target-shell-markers N         Target radius-local shell occupancy [4096].
  --sidm-minimum-shell-markers N        Minimum collision shell occupancy [32].
  --sidm-fail-on-nonlocal-shell         Stop if a shell exceeds its mean free path.

Optional receiver field clock:
  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.
  --coupled-field-trace                 Bounded field/sample caches and observational timing.

Information:
  --help, -h                            Print this interface and exit.

Fixed spatial physics: G=1, TSC, origin parity, and a reflecting outer wall.
Material SIDM is locked to a static halo; an EdS request is rejected. Marker masses source the total density.
This collisional executable requires a positive, explicit collision strength.
```

</details>

<details>
<summary><code>venus_gpu_sidm_constant</code></summary>

```text
VENUS GPU static-WPIC sidm_constant production driver

Usage:
  build/gpu/venus_gpu_sidm_constant --restart FILE --output-directory DIR --end-time T [options]

Required:
  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.
  --output-directory DIR                Run directory for profiles, histories, and restarts.
  --end-time T                          Final simulation time; T > restart time and T > 0.

Evolution and output:
  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.
  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].
  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].
  --output-interval DT                  Profile-time spacing; DT > 0 [10].
  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].
  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].
  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].

Logarithmic radial mesh:
  --radial-cells N                      Number of cells; N >= 2 [256].
  --radial-minimum R                    Inner face; R > 0 [0.1].
  --radial-maximum R                    Outer face; R > radial minimum [1000].

Material-SIDM collisions:
  --sidm-collision-strength S           Positive collision-rate multiplier.
  Cross-section model is pinned to constant.
  --sidm-sigma0 X                       Constant/power-law normalization; X > 0.
  --sidm-collision-interval DT          Global Boltzmann operator-batch cap; DT > 0 [0.5].
  --sidm-collision-cfl C                Per-substep probability bound; 0 < C <= 1 [0.2].
  --sidm-collision-maximum-substeps N   Per-event safety limit; N >= 1 [16384].
  --sidm-adaptive-shell-timestep        Independent shell-local CFL clocks.
  --sidm-target-shell-markers N         Target radius-local shell occupancy [4096].
  --sidm-minimum-shell-markers N        Minimum collision shell occupancy [32].
  --sidm-fail-on-nonlocal-shell         Stop if a shell exceeds its mean free path.

Optional receiver field clock:
  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.
  --coupled-field-trace                 Bounded field/sample caches and observational timing.

Information:
  --help, -h                            Print this interface and exit.

Fixed spatial physics: G=1, TSC, origin parity, and a reflecting outer wall.
Material SIDM is locked to a static halo; an EdS request is rejected. Marker masses source the total density.
This collisional executable requires a positive, explicit collision strength.
```

</details>

<details>
<summary><code>venus_gpu_sidm_power_law</code></summary>

```text
VENUS GPU static-WPIC sidm_power_law production driver

Usage:
  build/gpu/venus_gpu_sidm_power_law --restart FILE --output-directory DIR --end-time T [options]

Required:
  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.
  --output-directory DIR                Run directory for profiles, histories, and restarts.
  --end-time T                          Final simulation time; T > restart time and T > 0.

Evolution and output:
  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.
  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].
  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].
  --output-interval DT                  Profile-time spacing; DT > 0 [10].
  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].
  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].
  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].

Logarithmic radial mesh:
  --radial-cells N                      Number of cells; N >= 2 [256].
  --radial-minimum R                    Inner face; R > 0 [0.1].
  --radial-maximum R                    Outer face; R > radial minimum [1000].

Material-SIDM collisions:
  --sidm-collision-strength S           Positive collision-rate multiplier.
  Cross-section model is pinned to power_law.
  --sidm-sigma0 X                       Constant/power-law normalization; X > 0.
  --sidm-sigma-power P                  Power-law exponent; P > 0 [4].
  --sidm-sigma-u0 U                     Power-law softening speed; U > 0.
  --sidm-collision-interval DT          Global Boltzmann operator-batch cap; DT > 0 [0.5].
  --sidm-collision-cfl C                Per-substep probability bound; 0 < C <= 1 [0.2].
  --sidm-collision-maximum-substeps N   Per-event safety limit; N >= 1 [16384].
  --sidm-adaptive-shell-timestep        Independent shell-local CFL clocks.
  --sidm-target-shell-markers N         Target radius-local shell occupancy [4096].
  --sidm-minimum-shell-markers N        Minimum collision shell occupancy [32].
  --sidm-fail-on-nonlocal-shell         Stop if a shell exceeds its mean free path.

Optional receiver field clock:
  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.
  --coupled-field-trace                 Bounded field/sample caches and observational timing.

Information:
  --help, -h                            Print this interface and exit.

Fixed spatial physics: G=1, TSC, origin parity, and a reflecting outer wall.
Material SIDM is locked to a static halo; an EdS request is rejected. Marker masses source the total density.
This collisional executable requires a positive, explicit collision strength.
```

</details>

<details>
<summary><code>venus_gpu_sidm_yukawa_born</code></summary>

```text
VENUS GPU static-WPIC sidm_yukawa_born production driver

Usage:
  build/gpu/venus_gpu_sidm_yukawa_born --restart FILE --output-directory DIR --end-time T [options]

Required:
  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.
  --output-directory DIR                Run directory for profiles, histories, and restarts.
  --end-time T                          Final simulation time; T > restart time and T > 0.

Evolution and output:
  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.
  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].
  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].
  --output-interval DT                  Profile-time spacing; DT > 0 [10].
  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].
  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].
  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].

Logarithmic radial mesh:
  --radial-cells N                      Number of cells; N >= 2 [256].
  --radial-minimum R                    Inner face; R > 0 [0.1].
  --radial-maximum R                    Outer face; R > radial minimum [1000].

Material-SIDM collisions:
  --sidm-collision-strength S           Positive collision-rate multiplier.
  Cross-section model is pinned to yukawa_born.
  --sidm-yukawa-alpha A                 Effective kernel amplitude; A > 0 [0.01].
  --sidm-yukawa-mchi M                  Positive shape parameter [1].
  --sidm-yukawa-mphi M                  Shape parameter; mphi/mchi sets speed [1].
  --sidm-collision-interval DT          Global Boltzmann operator-batch cap; DT > 0 [0.5].
  --sidm-collision-cfl C                Per-substep probability bound; 0 < C <= 1 [0.2].
  --sidm-collision-maximum-substeps N   Per-event safety limit; N >= 1 [16384].
  --sidm-adaptive-shell-timestep        Independent shell-local CFL clocks.
  --sidm-target-shell-markers N         Target radius-local shell occupancy [4096].
  --sidm-minimum-shell-markers N        Minimum collision shell occupancy [32].
  --sidm-fail-on-nonlocal-shell         Stop if a shell exceeds its mean free path.

Optional receiver field clock:
  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.
  --coupled-field-trace                 Bounded field/sample caches and observational timing.

Information:
  --help, -h                            Print this interface and exit.

Fixed spatial physics: G=1, TSC, origin parity, and a reflecting outer wall.
Material SIDM is locked to a static halo; an EdS request is rejected. Marker masses source the total density.
This collisional executable requires a positive, explicit collision strength.
```

</details>

<details>
<summary><code>venus_gpu_subhalo_boltzmann_poisson</code></summary>

```text
VENUS GPU expanding-WPIC subhalo_boltzmann_poisson production driver

Usage:
  build/gpu/venus_gpu_subhalo_boltzmann_poisson --restart FILE --output-directory DIR --end-time T [options]

Required:
  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.
                                         Positive subhalo strength also requires consistent internal dispersions.
  --output-directory DIR                Run directory for profiles, histories, and restarts.
  --end-time T                          Final simulation time; T > restart time and T > 0.

Evolution and output:
  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.
  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].
  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].
  --output-interval DT                  Profile-time spacing; DT > 0 [10].
  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].
  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].
  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].

Logarithmic radial mesh:
  --radial-cells N                      Number of cells; N >= 2 [256].
  --radial-minimum R                    Inner face; R > 0 [0.1].
  --radial-maximum R                    Outer face; R > radial minimum [1000].

Subhalo collisions:
  --subhalo-collision-strength S        Dimensionless rate multiplier; S >= 0 [0].
  --subhalo-collision-interval DT       Maximum event interval; DT > 0 [0.5].
  --subhalo-collision-timescale-fraction F
                                         Adaptive fraction; 0 <= F <= 1 [0.05].
  --subhalo-collision-cfl C             Collision probability controller; 0 < C <= 1 [0.2].
  --subhalo-collision-maximum-substeps N
                                         Per-event safety limit; N >= 1 [16384].
  --subhalo-adaptive-cell-timestep       Independent cell-local CFL clocks.
  --collision-prestep-coordinate-due     Apply safe coordinate-close deadlines before KDK.
  --subhalo-adaptive-batch-substeps N    Local substeps per CUDA launch; N >= 1 [256].
  --subhalo-inner-collision-cells N     Virtual cells inside the mesh inner face; 1 <= N <= 256 [1].
  --subhalo-collision-lanes N           Deterministic pairing strata; 1 <= N <= 256 [8].
  --subhalo-locality-maximum-ratio X    Stop threshold for max(Delta r/lambda); X >= 0 [10].
  --subhalo-velocity-scale V            Compatibility value; restart dispersions are used; V > 0 [0.1].
  --subhalo-plummer-area-per-mass A     Plummer area per mass; A > 0 [1].
  --subhalo-low-speed-regularization E  Table-floor regularization; 0.05 <= E < 1 [0.05].

Optional receiver field clock:
  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.
  --coupled-field-trace                 Bounded field/sample caches and observational timing.

Information:
  --help, -h                            Print this interface and exit.

Fixed spatial physics: G=1, TSC, origin parity, and a reflecting outer wall.
VP/subhalo accepts EdS expansion (alpha=2/3, time offset=12) or a static halo. EdS marker masses represent density contrast. Static marker masses represent total density.
This collisional executable requires a positive, explicit collision strength.
```

</details>

<details>
<summary><code>venus_gpu_vlasov_poisson</code></summary>

```text
VENUS GPU expanding-WPIC vlasov_poisson production driver

Usage:
  build/gpu/venus_gpu_vlasov_poisson --restart FILE --output-directory DIR --end-time T [options]

Required:
  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.
  --output-directory DIR                Run directory for profiles, histories, and restarts.
  --end-time T                          Final simulation time; T > restart time and T > 0.

Evolution and output:
  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.
  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].
  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].
  --output-interval DT                  Profile-time spacing; DT > 0 [10].
  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].
  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].
  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].

Logarithmic radial mesh:
  --radial-cells N                      Number of cells; N >= 2 [256].
  --radial-minimum R                    Inner face; R > 0 [0.1].
  --radial-maximum R                    Outer face; R > radial minimum [1000].

Optional receiver field clock:
  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.
  --coupled-field-trace                 Bounded field/sample caches and observational timing.

Information:
  --help, -h                            Print this interface and exit.

Fixed spatial physics: G=1, TSC, origin parity, and a reflecting outer wall.
VP/subhalo accepts EdS expansion (alpha=2/3, time offset=12) or a static halo. EdS marker masses represent density contrast. Static marker masses represent total density.
This executable rejects every collision option and never constructs a collision module.
```

</details>

<details>
<summary><code>venus_gpu_vlasov_poisson_static</code></summary>

```text
Static Vlasov-Poisson WPIC: H=0; --universe static is fixed.
VENUS GPU expanding-WPIC vlasov_poisson production driver

Usage:
  build/gpu/venus_gpu_vlasov_poisson_static --restart FILE --output-directory DIR --end-time T [options]

Required:
  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.
  --output-directory DIR                Run directory for profiles, histories, and restarts.
  --end-time T                          Final simulation time; T > restart time and T > 0.

Evolution and output:
  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.
  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].
  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].
  --output-interval DT                  Profile-time spacing; DT > 0 [10].
  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].
  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].
  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].

Logarithmic radial mesh:
  --radial-cells N                      Number of cells; N >= 2 [256].
  --radial-minimum R                    Inner face; R > 0 [0.1].
  --radial-maximum R                    Outer face; R > radial minimum [1000].

Optional receiver field clock:
  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.
  --coupled-field-trace                 Bounded field/sample caches and observational timing.

Information:
  --help, -h                            Print this interface and exit.

Fixed spatial physics: G=1, TSC, origin parity, and a reflecting outer wall.
VP/subhalo accepts EdS expansion (alpha=2/3, time offset=12) or a static halo. EdS marker masses represent density contrast. Static marker masses represent total density.
This executable rejects every collision option and never constructs a collision module.
```

</details>
