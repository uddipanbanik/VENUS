
// Tabulate physical subhalo collision cells for a freshly sampled t=0 state.
// Inputs are the same physical, background and mesh arguments as the shared
// weighted-particle driver. Output records physical cell width and volume,
// represented mass, velocity dispersion, mean free path and collision time.
// Monte Carlo work-sharing lanes are recombined before interpreting these
// quantities. A lane is a sampling partition, not a smaller physical volume.

#include "../src/drivers/expanding_wpic.cpp"

int main(int argc, char** argv) {
    try {
        // Parse exactly the production runner's physics, mesh, and sampling
        // controls, but restrict this diagnostic to a newly sampled t=0 state.
        const Options run = parse_options(argc, argv);
        if (run.show_help) {
            print_usage(argv[0]);
            return 0;
        }
        if (!run.restart.empty()) {
            throw std::runtime_error(
                "The locality table currently audits a fresh t=0 sample; "
                "omit --restart.");
        }
        if (!(run.subhalo_collision_strength > 0.0)) {
            throw std::runtime_error(
                "A positive --subhalo-collision-strength is required.");
        }

        // Construct the same unequal-mass marker realization used by the
        // expanding WPIC runner and attach the subhalos' initial internal heat.
        ev::Params parameters = make_parameters(run);
        ev::WeightedPICFieldGrid grid =
            ev::make_weighted_pic_field_grid(parameters, parameters.Nr);
        PowerLawInitialCondition initial(run.gamma, grid.mesh);
        std::vector<ev::WeightedPICMarker> markers =
            initialize_markers(run, parameters, grid, initial);

        // Assign the same unresolved internal velocity dispersion and energy
        // as the initial subhalos in the physical driver. Their internal heat
        // sets the encounter law and differs from orbital velocity dispersion.
        const double specific_internal_energy =
            1.5 * run.subhalo_velocity_scale
            * run.subhalo_velocity_scale;
        for (ev::WeightedPICMarker& marker : markers) {
            marker.internal_energy =
                marker.mass * specific_internal_energy;
            marker.internal_velocity_dispersion =
                run.subhalo_velocity_scale;
        }

        // Deposit the evolving marker distribution to obtain the gravitational
        // dynamical clock against which the local collision clock is compared.
        const ev::WeightedPICDepositedMoments deposit =
            ev::deposit_weighted_vlasov_pic_parallel(
                markers,
                grid.mesh,
                ev::WeightedPICParticleShape::tsc,
                run.threads,
                true);
        // The current deposited density supplies the gravitational clock.
        // At the initial state a measured field-difference clock is unavailable
        // and is represented by positive infinity.
        const double dynamical_time =
            minimum_dynamical_timescale(deposit, grid.mesh, parameters);
        const double potential_change_time =
            std::numeric_limits<double>::infinity();

        // Apply an effectively zero collision interval.  This performs the
        // production cell grouping and pair-rate calculation without advancing
        // or scattering markers, exposing the instantaneous mean free paths.
        const ev::WeightedPICSubhaloCollisionOptions options =
            make_collision_options(run);
        ev::WeightedPICCollisionSamplerState sampler{
            run.subhalo_collision_seed, 0};
        ev::WeightedPICSubhaloCollisionWorkspace workspace;
        const auto total =
            ev::apply_weighted_pic_subhalo_collision_interval(
                markers,
                grid.mesh,
                std::numeric_limits<double>::min(),
                options,
                sampler,
                workspace,
                run.threads);

        const std::size_t inner =
            static_cast<std::size_t>(options.inner_collision_cells);
        const std::size_t lanes =
            static_cast<std::size_t>(options.collision_lanes);

        // Recombine Monte-Carlo lanes into physical collision cells before
        // reporting density, partner density, velocity dispersion, and
        // Delta r/lambda_mfp.  Lanes change work sharing, not physical volume.
        std::cout
            << "physical_cell,region,resolved_shell,left_radius,right_radius,"
               "cell_width,cell_volume,marker_count,represented_mass,density,"
               "partner_number_density,pair_samples,orbital_sigma,"
               "mean_free_path,cell_width_over_mean_free_path,"
               "collision_time,mean_relative_speed,mean_u_sigma,"
               "minimum_global_dynamical_time,"
               "initial_potential_change_time\n";
        std::cout << std::setprecision(17);
        for (std::size_t physical = 0;
             physical < workspace.physical_cell_diagnostics.size();
             ++physical) {
            const bool is_inner = physical < inner;
            const std::size_t resolved =
                is_inner ? 0 : physical - inner;
            double left = 0.0;
            double right = 0.0;
            // Hidden inner cells partition the volume inside the first field face.
            // Their radii scale with the cube root of the volume fraction; resolved
            // physical cells instead use the actual field-mesh faces.
            if (is_inner) {
                left = grid.mesh.edge.front()
                    * std::cbrt(
                        static_cast<double>(physical)
                        / static_cast<double>(inner));
                right = grid.mesh.edge.front()
                    * std::cbrt(
                        static_cast<double>(physical + 1)
                        / static_cast<double>(inner));
            } else {
                left = grid.mesh.edge[resolved];
                right = grid.mesh.edge[resolved + 1];
            }
            const double width = right - left;
            // Multiply the lane volume by the lane count to recover the physical
            // collision-cell volume. Mass and partner density must use this same
            // physical volume when reporting a mean free path.
            const double volume =
                ev::weighted_pic_subhalo_collision_detail::
                    collision_cell_volume(physical * lanes, grid.mesh, options)
                * static_cast<double>(lanes);
            const auto& members = workspace.physical_cell_members[physical];
            long double represented_mass = 0.0L;
            for (const std::size_t marker_index : members) {
                represented_mass += markers[marker_index].mass;
            }
            const double density =
                static_cast<double>(represented_mass) / volume;
            const double partner_number_density =
                members.size() > 1
                    ? static_cast<double>(members.size() - 1) / volume
                    : 0.0;
            const auto& diagnostics =
                workspace.physical_cell_diagnostics[physical];
            const double mean_free_path =
                diagnostics.minimum_mean_free_path;
            const double collision_time =
                diagnostics.minimum_collision_timescale;
            // The printed speed and u*Sigma are inferred from the reported
            // physical path/time and partner-density definitions. They summarize
            // this same sampled collision diagnostic, rather than an independent
            // analytic closure for the evolving distribution.
            const double mean_relative_speed =
                std::isfinite(mean_free_path)
                    && std::isfinite(collision_time)
                    && collision_time > 0.0
                ? mean_free_path / collision_time
                : 0.0;
            const double mean_u_sigma =
                partner_number_density > 0.0
                    && std::isfinite(collision_time)
                    && collision_time > 0.0
                ? 1.0 / (partner_number_density * collision_time)
                : 0.0;
            std::cout
                << physical << ','
                << (is_inner ? "hidden_inner" : "resolved_mesh") << ',';
            if (is_inner) {
                std::cout << -1;
            } else {
                std::cout << resolved;
            }
            std::cout
                << ',' << left
                << ',' << right
                << ',' << width
                << ',' << volume
                << ',' << members.size()
                << ',' << static_cast<double>(represented_mass)
                << ',' << density
                << ',' << partner_number_density
                << ',' << diagnostics.mean_free_path_pair_samples
                << ',' << diagnostics.minimum_orbital_velocity_dispersion
                << ',' << mean_free_path
                << ','
                << diagnostics.maximum_cell_width_over_mean_free_path
                << ',' << collision_time
                << ',' << mean_relative_speed
                << ',' << mean_u_sigma
                << ',' << dynamical_time
                << ',' << potential_change_time
                << '\n';
        }

        // Print the global worst-resolved cell and fastest physical clocks to
        // stderr so launch scripts can enforce one conservative locality gate.
        std::cerr
            << std::setprecision(17)
            << "maximum_cell_width_over_mean_free_path="
            << total.maximum_cell_width_over_mean_free_path
            << " minimum_mean_free_path="
            << total.minimum_mean_free_path
            << " minimum_collision_time="
            << total.minimum_collision_timescale
            << " minimum_global_dynamical_time="
            << dynamical_time << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
