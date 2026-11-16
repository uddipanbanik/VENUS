#pragma once

// Read spherical phase distributions and construct analytic initial data.
// An EVICRWL1 file contains a fixed identifier, version, three grid sizes, the
// r/w/L coordinate arrays, then f with L varying fastest. The reader checks finite
// positive radial nodes and a finite nonnegative distribution; an MPI rank reads
// only its angular slab. Analytic data combine a radial envelope with a shifted
// relativistic energy profile, optionally normalized to a prescribed density.
// Parameter checks keep the chosen field, transport, collision and boundary
// approximations mutually compatible before the solver allocates its full state.

#include "ev/grids.hpp"
#include "ev/phase_space_cells.hpp"
#include "ev/parallel.hpp"
#include "ev/mpi.hpp"

namespace ev {

/// Reads one plain-old-data value from a binary IC file and reports short reads clearly.
template <typename T>
T read_ic_pod(std::ifstream& in, const std::filesystem::path& path) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected end of initial-data file: " + path.string());
    }
    return value;
}

/// Reads a fixed number of stored binary double values from a binary IC file.
void read_ic_doubles(
    std::ifstream& in,
    std::vector<double>& values,
    const std::filesystem::path& path,
    const std::string& label) {
    in.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Unexpected end of " + label + " array in " + path.string());
    }
}

/// Checks that a grid is strictly increasing at every adjacent pair.
bool strictly_increasing(const std::vector<double>& values) {
    if (values.empty() || !std::isfinite(values.front())) {
        return false;
    }
    for (int i = 1; i < static_cast<int>(values.size()); ++i) {
        if (!std::isfinite(values[i]) || !(values[i] > values[i - 1])) {
            return false;
        }
    }
    return true;
}

/// Reads and validates a complete EVICRWL1 file, including the full f(r,w,L) array.
[[maybe_unused]] InitialDataRWL read_initial_data_rwl(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open binary initial-data file: " + path.string());
    }

    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVICRWL1")) {
        throw std::runtime_error(
            "Initial-data file has wrong magic header. Expected EVICRWL1: " + path.string());
    }
    const std::int32_t version = read_ic_pod<std::int32_t>(in, path);
    if (version != 1) {
        throw std::runtime_error("Unsupported initial-data file version in " + path.string());
    }
    const std::int32_t nr = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nw = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nl = read_ic_pod<std::int32_t>(in, path);
    if (nr < 3 || nw < 3 || nl < 2) {
        throw std::runtime_error("Initial-data file needs Nr >= 3, Nw >= 3, and NL >= 2.");
    }

    InitialDataRWL ic;
    ic.Nr = nr;
    ic.Nw = nw;
    ic.NL = nl;
    ic.r.assign(static_cast<std::size_t>(nr), 0.0);
    ic.w.assign(static_cast<std::size_t>(nw), 0.0);
    ic.L.assign(static_cast<std::size_t>(nl), 0.0);
    ic.f.assign(
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nw) * static_cast<std::size_t>(nl),
        0.0);
    read_ic_doubles(in, ic.r, path, "r");
    read_ic_doubles(in, ic.w, path, "w");
    read_ic_doubles(in, ic.L, path, "L");
    read_ic_doubles(in, ic.f, path, "f");

    if (!strictly_increasing(ic.r) || ic.r.front() <= 0.0) {
        throw std::runtime_error("Initial-data r grid must be positive and strictly increasing.");
    }
    if (!strictly_increasing(ic.w)) {
        throw std::runtime_error("Initial-data w grid must be strictly increasing.");
    }
    if (!strictly_increasing(ic.L) || ic.L.front() < 0.0) {
        throw std::runtime_error("Initial-data L grid must be nonnegative and strictly increasing.");
    }
    for (double value : ic.f) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("Initial-data f values must be finite and nonnegative.");
        }
    }
    return ic;
}

/// Reads and validates EVICRWL1 header and coordinate grids without loading all f values.
InitialDataRWL read_initial_data_rwl_metadata(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open binary initial-data file: " + path.string());
    }

    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVICRWL1")) {
        throw std::runtime_error(
            "Initial-data file has wrong magic header. Expected EVICRWL1: " + path.string());
    }
    const std::int32_t version = read_ic_pod<std::int32_t>(in, path);
    if (version != 1) {
        throw std::runtime_error("Unsupported initial-data file version in " + path.string());
    }
    const std::int32_t nr = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nw = read_ic_pod<std::int32_t>(in, path);
    const std::int32_t nl = read_ic_pod<std::int32_t>(in, path);
    if (nr < 3 || nw < 3 || nl < 2) {
        throw std::runtime_error("Initial-data file needs Nr >= 3, Nw >= 3, and NL >= 2.");
    }

    InitialDataRWL ic;
    ic.Nr = nr;
    ic.Nw = nw;
    ic.NL = nl;
    ic.r.assign(static_cast<std::size_t>(nr), 0.0);
    ic.w.assign(static_cast<std::size_t>(nw), 0.0);
    ic.L.assign(static_cast<std::size_t>(nl), 0.0);
    read_ic_doubles(in, ic.r, path, "r");
    read_ic_doubles(in, ic.w, path, "w");
    read_ic_doubles(in, ic.L, path, "L");

    if (!strictly_increasing(ic.r) || ic.r.front() <= 0.0) {
        throw std::runtime_error("Initial-data r grid must be positive and strictly increasing.");
    }
    if (!strictly_increasing(ic.w)) {
        throw std::runtime_error("Initial-data w grid must be strictly increasing.");
    }
    if (!strictly_increasing(ic.L) || ic.L.front() < 0.0) {
        throw std::runtime_error("Initial-data L grid must be nonnegative and strictly increasing.");
    }
    return ic;
}

/// Computes the byte offset at which the f(r,w,L) array begins in an EVICRWL1 file.
std::streamoff initial_data_f_offset(const InitialDataRWL& ic) {
    return static_cast<std::streamoff>(16 + 4 * sizeof(std::int32_t))
        + static_cast<std::streamoff>(ic.Nr + ic.Nw + ic.NL) * static_cast<std::streamoff>(sizeof(double));
}

/// Reads only the local MPI L-slab of the EVICRWL1 distribution array.
std::vector<double> read_initial_data_rwl_local_values(
    const std::filesystem::path& path,
    const InitialDataRWL& ic,
    const LDecomposition& decomp) {
    std::vector<double> f(
        static_cast<std::size_t>(ic.Nr) * ic.Nw * decomp.count,
        0.0);
    if (decomp.count == 0) {
        return f;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open binary initial-data file: " + path.string());
    }
    // Each (r,w) row has a contiguous global L array. Seek directly to the
    // rank-owned angular slab so every rank reads precisely its local distribution.
    const std::streamoff f0 = initial_data_f_offset(ic);
    for (int i = 0; i < ic.Nr; ++i) {
        for (int j = 0; j < ic.Nw; ++j) {
            const std::uint64_t global_base =
                (static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(ic.Nw)
                 + static_cast<std::uint64_t>(j))
                    * static_cast<std::uint64_t>(ic.NL)
                + static_cast<std::uint64_t>(decomp.offset);
            const std::streamoff offset =
                f0 + static_cast<std::streamoff>(global_base * static_cast<std::uint64_t>(sizeof(double)));
            in.seekg(offset);
            if (!in) {
                throw std::runtime_error("Failed to seek local initial-data slab in " + path.string());
            }
            double* dest = f.data()
                + (static_cast<std::size_t>(i) * ic.Nw + static_cast<std::size_t>(j))
                    * static_cast<std::size_t>(decomp.count);
            in.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(decomp.count * sizeof(double)));
            if (!in) {
                throw std::runtime_error("Unexpected end of local initial-data slab in " + path.string());
            }
            for (int k = 0; k < decomp.count; ++k) {
                const double value = dest[k];
                if (!std::isfinite(value) || value < 0.0) {
                    throw std::runtime_error("Initial-data f values must be finite and nonnegative.");
                }
            }
        }
    }
    return f;
}

/// Copies validated binary initial-data values into the solver storage layout.
std::vector<double> custom_initial_distribution_rwl(
    const InitialDataRWL& ic,
    const Params& p) {
    std::vector<double> f(static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    std::copy(ic.f.begin(), ic.f.end(), f.begin());
    return f;
}

// Parameter validation catches the most common unphysical or internally
// inconsistent runs before a large phase-space array is allocated.
/// Validates numerical, physical, and output parameters before memory allocation or evolution.
void validate_params(const Params& p) {
    const bool binary_ic = using_binary_initial_data(p);
    if (!(p.physics_model == "einstein_vlasov"
          || p.physics_model == "vlasov_poisson"
          || p.physics_model == "boltzmann_poisson")) {
        throw std::runtime_error(
            "physics_model must be 'einstein_vlasov', 'vlasov_poisson', or 'boltzmann_poisson'.");
    }
    if (!(p.vp_universe == "static" || p.vp_universe == "expanding")) {
        throw std::runtime_error("vp_universe must be 'static' or 'expanding'.");
    }
    if (!(p.vp_poisson_source == "density_contrast" || p.vp_poisson_source == "total_density")) {
        throw std::runtime_error("vp_poisson_source must be 'density_contrast' or 'total_density'.");
    }
    if (!(p.vp_poisson_reconstruction == "constant_line_mass"
          || p.vp_poisson_reconstruction == "p2_line_mass")) {
        throw std::runtime_error(
            "vp_poisson_reconstruction must be 'constant_line_mass' or "
            "'p2_line_mass'.");
    }
    // The quadratic radial source represents conservative static total mass. Its
    // admission conditions keep that reconstruction with its defined physical measure.
    if (p.vp_poisson_reconstruction == "p2_line_mass"
        && (p.vp_universe != "static"
            || p.vp_poisson_source != "total_density"
            || !using_cell_average_eta_grid(p)
            || !using_conservative_phase_space_cells(p))) {
        throw std::runtime_error(
            "p2_line_mass Poisson reconstruction requires a static, "
            "total-density, conservative cell-average phase-space model.");
    }
    if (!(p.vp_interpolation == "raw"
          || p.vp_interpolation == "hamiltonian"
          || p.vp_interpolation == "hamiltonian_conservative"
          || p.vp_interpolation == "conservative_sldg"
          || p.vp_interpolation == "hamiltonian_sldg"
          || p.vp_interpolation == "energy_shell"
          || p.vp_interpolation == "expanding_physical_velocity")) {
        throw std::runtime_error(
            "vp_interpolation must be 'raw', 'hamiltonian', 'hamiltonian_conservative', "
            "'conservative_sldg', 'hamiltonian_sldg', 'energy_shell', or "
            "'expanding_physical_velocity'.");
    }
    if (!(p.kinetic_grid == "legacy_nodal" || p.kinetic_grid == "cell_average_eta")) {
        throw std::runtime_error(
            "kinetic_grid must be 'legacy_nodal' or 'cell_average_eta'.");
    }
    if ((p.vp_interpolation == "hamiltonian"
         || p.vp_interpolation == "hamiltonian_conservative"
         || p.vp_interpolation == "hamiltonian_sldg"
         || p.vp_interpolation == "energy_shell")
        && p.vp_universe != "static") {
        throw std::runtime_error(
            "Hamiltonian VP transport requires vp_universe='static'; "
            "the frozen static-frame Hamiltonian is not an invariant in the expanding-background VP system.");
    }
    if (p.vp_interpolation == "expanding_physical_velocity" && p.vp_universe != "expanding") {
        throw std::runtime_error(
            "vp_interpolation='expanding_physical_velocity' requires vp_universe='expanding'.");
    }
    if (p.vp_interpolation == "hamiltonian_conservative" && p.outer_boundary == "vp_outer_inflow") {
        throw std::runtime_error(
            "vp_interpolation='hamiltonian_conservative' does not support prescribed outer inflow; "
            "use a closed reflecting box or an outflow boundary.");
    }
    // Cell-integrated Hamiltonian transport assumes a static potential over an
    // orbit step, conservative eta cells and a defined treatment of the positive
    // inner face. Its dynamical clock and orbit retry are checked as one method.
    if (using_any_sldg_transport(p)) {
        if (p.vp_universe != "static") {
            throw std::runtime_error(
                "Cell-integrated SLDG transport currently requires vp_universe=static.");
        }
        if (!using_cell_average_eta_grid(p)) {
            throw std::runtime_error(
                "Cell-integrated SLDG transport requires kinetic_grid=cell_average_eta.");
        }
        if (p.inner_boundary != "reflecting"
            && p.inner_boundary != "delayed_return") {
            throw std::runtime_error(
                "Cell-integrated SLDG transport uses a finite positive radial "
                "domain and requires inner_boundary=reflecting or delayed_return at r_min.");
        }
        if (p.vp_timestep_mode != "frank_dynamical") {
            throw std::runtime_error(
                "Cell-integrated SLDG transport requires vp_timestep_mode=frank_dynamical; "
                "the conservative remap must not be driven by a velocity-cell crossing CFL.");
        }
        if (!using_kick_drift_kick_integrator(p)) {
            throw std::runtime_error(
                "Cell-integrated SLDG transport requires "
                "integrator=semi_lagrangian_kick_drift_kick.");
        }
        if (p.vp_orbit_regularization == "off") {
            throw std::runtime_error(
                "Cell-integrated SLDG transport requires Levi-Civita orbit "
                "regularization (vp_orbit_regularization=auto or always).");
        }
        if (p.vp_inner_carrier_quadrature_points < 1
            || p.vp_inner_carrier_quadrature_points > 4
            || p.vp_inner_carrier_max_refinement < 0
            || p.vp_inner_carrier_max_orbit_segments < 1
            || p.vp_inner_carrier_max_packets < 1
            || !(p.vp_inner_carrier_mass_tolerance > 0.0)
            || p.vp_inner_carrier_return_time_bin_width < 0.0
            || !std::isfinite(p.vp_inner_carrier_return_time_bin_width)
            || p.vp_inner_carrier_compress_min_packets < 1) {
            throw std::runtime_error(
                "Delayed-return carrier controls require quadrature points 1..4, "
                "nonnegative refinement, positive orbit/packet limits, and a "
                "positive mass tolerance and compression threshold.  The "
                "return-time bin width may be zero to disable compression.");
        }
        if (p.inner_boundary == "delayed_return"
            && p.central_mass != 0.0) {
            throw std::runtime_error(
                "inner_boundary=delayed_return currently requires central_mass=0.");
        }
        if (p.vp_sldg_subcells < 1
            || p.vp_sldg_eta_quadrature_points < 1
            || p.vp_sldg_eta_quadrature_points > 12
            || !(p.vp_sldg_geometry_tolerance > 0.0)
            || !(p.vp_sldg_max_area_defect > 0.0)
            || !(p.vp_sldg_max_map_defect > 0.0)
            || p.vp_sldg_max_refinement < 0
            || p.vp_sldg_max_work_per_subcell < 1
            || p.vp_sldg_fallback_mass_tolerance < 0.0
            || p.vp_sldg_fallback_mass_tolerance >= 1.0
            || p.vp_sldg_initial_transport_dt < 0.0
            || !(p.vp_sldg_target_fallback_fraction > 0.0)
            || !(p.vp_sldg_target_fallback_fraction < 1.0)
            || !(p.vp_sldg_timestep_growth > 1.0)
            || !(p.vp_sldg_timestep_shrink > 0.0)
            || !(p.vp_sldg_timestep_shrink < 1.0)) {
            throw std::runtime_error(
                "Conservative SLDG requires positive subcell/geometry/map/work controls, "
                "one through twelve eta quadrature points, "
                "and fallback-cadence controls with a nonnegative initial dt, "
                "0 < target < 1, growth > 1, and 0 < shrink < 1.");
        }
    }
    if (!(p.vp_timestep_mode == "active_cell_cfl"
          || p.vp_timestep_mode == "frank_dynamical")) {
        throw std::runtime_error(
            "vp_timestep_mode must be 'active_cell_cfl' or 'frank_dynamical'.");
    }
    if (!(p.vp_hamiltonian_reference == "orbit_average"
          || p.vp_hamiltonian_reference == "none")
        || p.vp_hamiltonian_reference_energy_bins < 8
        || p.vp_hamiltonian_reference_quadrature_points < 2
        || p.vp_hamiltonian_reference_quadrature_points > 12) {
        throw std::runtime_error(
            "Hamiltonian reference controls require orbit_average or none, "
            "at least eight energy bins, and 2..12 quadrature points.");
    }
    if (!(p.vp_dynamical_eta > 0.0)
        || !(p.vp_potential_variation_eta > 0.0)
        || !(p.vp_potential_variation_eta < 1.0)
        || !std::isfinite(p.vp_potential_variation_eta)
        || p.vp_timestep_omitted_mass_fraction < 0.0
        || p.vp_timestep_omitted_mass_fraction >= 1.0
        || !std::isfinite(p.vp_timestep_omitted_mass_fraction)) {
        throw std::runtime_error(
            "VP timestep factors require vp_dynamical_eta > 0 and "
            "0 < vp_potential_variation_eta < 1, with a finite "
            "0 <= vp_timestep_omitted_mass_fraction < 1.");
    }
    // A characteristic orbit step and an explicit finite-volume flux step obey
    // different stability restrictions. The latter retains its velocity-cell CFL.
    if (p.vp_timestep_mode == "frank_dynamical"
        && p.integrator == "conservative_finite_volume") {
        throw std::runtime_error(
            "vp_timestep_mode='frank_dynamical' is only valid for semi-Lagrangian "
            "characteristic transport; conservative finite-volume transport still requires its CFL.");
    }
    if (p.vp_gravity <= 0.0) {
        throw std::runtime_error("vp_gravity must be positive.");
    }
    if (p.vp_time_offset <= 0.0) {
        throw std::runtime_error("vp_time_offset must be positive.");
    }
    if (p.vp_alpha < 0.0) {
        throw std::runtime_error("vp_alpha must be non-negative.");
    }
    if (!(p.bp_collision_operator == "none"
          || p.bp_collision_operator == "fokker_planck"
          || p.bp_collision_operator == "landau_fp"
          || p.bp_collision_operator == "boltzmann_exact"
          || p.bp_collision_operator == "boltzmann_reversible"
          || p.bp_collision_operator == "boltzmann_reversible_sampled"
          || p.bp_collision_operator == "boltzmann_reversible_streamed"
          || p.bp_collision_operator == "spherical_dsmc")) {
        throw std::runtime_error(
            "bp_collision_operator must be 'none', 'fokker_planck', 'landau_fp', "
            "'boltzmann_exact', 'boltzmann_reversible', "
            "'boltzmann_reversible_sampled', "
            "'boltzmann_reversible_streamed', or 'spherical_dsmc'.");
    }
    if (!(p.bp_cross_section == "powerlaw"
          || p.bp_cross_section == "constant"
          || p.bp_cross_section == "softened_gravity"
          || p.bp_cross_section == "yukawa_born")) {
        throw std::runtime_error(
            "bp_cross_section must be 'powerlaw', 'constant', 'softened_gravity', or 'yukawa_born'.");
    }
    if (!(p.bp_boltzmann_kernel == "auto"
          || p.bp_boltzmann_kernel == "isotropic"
          || p.bp_boltzmann_kernel == "yukawa_born")) {
        throw std::runtime_error(
            "bp_boltzmann_kernel must be 'auto', 'isotropic', or 'yukawa_born'.");
    }
    if (!(p.bp_bath == "instantaneous_df"
          || p.bp_bath == "local_maxwellian"
          || p.bp_bath == "global_maxwellian")) {
        throw std::runtime_error(
            "bp_bath must be 'instantaneous_df', 'local_maxwellian', or 'global_maxwellian'.");
    }
    if (!(p.bp_fp_terms == "full"
          || p.bp_fp_terms == "vr_only"
          || p.bp_fp_terms == "L_only"
          || p.bp_fp_terms == "no_mixed"
          || p.bp_fp_terms == "no_azimuth"
          || p.bp_fp_terms == "no_drift"
          || p.bp_fp_terms == "drift_only"
          || p.bp_fp_terms == "Drr_only"
          || p.bp_fp_terms == "DLL_only"
          || p.bp_fp_terms == "Daa_only")) {
        throw std::runtime_error(
            "bp_fp_terms must be 'full', 'vr_only', 'L_only', 'no_mixed', 'no_azimuth', "
            "'no_drift', 'drift_only', 'Drr_only', 'DLL_only', or 'Daa_only'.");
    }
    if (!(p.bp_collision_flux == "centered" || p.bp_collision_flux == "chang_cooper")) {
        throw std::runtime_error("bp_collision_flux must be 'centered' or 'chang_cooper'.");
    }
    if (!(p.bp_collision_integrator == "explicit" || p.bp_collision_integrator == "sts_rkl2")) {
        throw std::runtime_error("bp_collision_integrator must be 'explicit' or 'sts_rkl2'.");
    }
    if (!(p.bp_collision_splitting == "after_vlasov" || p.bp_collision_splitting == "strang")) {
        throw std::runtime_error("bp_collision_splitting must be 'after_vlasov' or 'strang'.");
    }
    if (!(p.bp_coeff_refresh_mode == "steps" || p.bp_coeff_refresh_mode == "collision_time")) {
        throw std::runtime_error("bp_coeff_refresh_mode must be 'steps' or 'collision_time'.");
    }
    if (p.bp_coeff_refresh_chi <= 0.0) {
        throw std::runtime_error("bp_coeff_refresh_chi must be positive.");
    }
    // The persistent whole-particle DSMC backend supplies static enclosed-mass
    // gravity and local shell collisions. Validate its population and physical mass
    // controls before selecting this distinct representation.
    if (using_bp_spherical_dsmc(p)) {
        if (p.vp_universe != "static"
            || p.vp_poisson_source != "total_density") {
            throw std::runtime_error(
                "spherical_dsmc implements static enclosed-total-mass "
                "gravity: use vp_universe=static and "
                "vp_poisson_source=total_density.");
        }
        if (p.bp_dsmc_particles < 2
            || p.bp_dsmc_shell_particles < 2
            || p.bp_dsmc_min_shell_particles < 2
            || p.bp_dsmc_particles < p.bp_dsmc_min_shell_particles
            || p.bp_dsmc_min_shell_particles > p.bp_dsmc_shell_particles
            || !std::isfinite(p.bp_dsmc_total_mass)
            || p.bp_dsmc_total_mass < 0.0
            || !(p.bp_dsmc_collision_epsilon > 0.0)
            || !(p.bp_dsmc_gravity_eta > 0.0)) {
            throw std::runtime_error(
                "spherical_dsmc requires at least two particles, "
                "2 <= bp_dsmc_min_shell_particles <= bp_dsmc_shell_particles, "
                "a finite nonnegative total mass, and positive "
                "collision-epsilon/gravity-eta.");
        }
        if (p.tracer_initial_data) {
            throw std::runtime_error(
                "spherical_dsmc does not yet support a passive tracer ensemble.");
        }
    }
    if (p.bp_coeff_refresh_time_quantile < 0.0 || p.bp_coeff_refresh_time_quantile > 1.0) {
        throw std::runtime_error("bp_coeff_refresh_time_quantile must lie in [0,1].");
    }
    if (p.bp_collision_l_min_rel < 0.0 || p.bp_collision_l_min_rel >= 1.0) {
        throw std::runtime_error("bp_collision_l_min_rel must lie in [0,1).");
    }
    if (using_boltzmann_poisson(p) && !using_vlasov_poisson(p)) {
        throw std::runtime_error("Boltzmann-Poisson collisions require the Vlasov-Poisson family.");
    }
    if (p.physics_model == "boltzmann_poisson" && p.bp_collision_operator == "none") {
        throw std::runtime_error(
            "physics_model=boltzmann_poisson requires a collision operator; "
            "use spherical_dsmc for the persistent-marker SIDM backend.");
    }
    if (p.bp_collision_operator == "boltzmann_exact"
        || p.bp_collision_operator == "boltzmann_reversible"
        || p.bp_collision_operator == "boltzmann_reversible_sampled"
        || p.bp_collision_operator == "boltzmann_reversible_streamed"
        || p.bp_collision_operator == "spherical_dsmc") {
        if (p.bp_cross_section == "yukawa_born"
            && !(p.bp_boltzmann_kernel == "auto" || p.bp_boltzmann_kernel == "yukawa_born")) {
            throw std::runtime_error(
                "bp_cross_section=yukawa_born requires bp_boltzmann_kernel=auto or yukawa_born "
                "for finite-angle Boltzmann and spherical_dsmc collisions.");
        }
        if ((p.bp_cross_section == "constant" || p.bp_cross_section == "powerlaw"
             || p.bp_cross_section == "softened_gravity")
            && p.bp_boltzmann_kernel == "yukawa_born") {
            throw std::runtime_error(
                "bp_boltzmann_kernel=yukawa_born requires bp_cross_section=yukawa_born.");
        }
    }
    // Reversible discrete reactions use positive canonical cell averages, a
    // conservative transport map and Strang operator ordering. These requirements
    // make their collision invariants refer to the same phase-space measure.
    if (p.bp_collision_operator == "boltzmann_reversible"
        || p.bp_collision_operator == "boltzmann_reversible_sampled"
        || p.bp_collision_operator == "boltzmann_reversible_streamed") {
        if (!using_cell_average_eta_grid(p)
            || !using_any_sldg_transport(p)) {
            throw std::runtime_error(
                "reversible Boltzmann backends require kinetic_grid=cell_average_eta and "
                "a cell-integrated SLDG transport.");
        }
        if (p.bp_collision_splitting != "strang") {
            throw std::runtime_error(
                "reversible Boltzmann backends require bp_collision_splitting=strang.");
        }
        if (p.bp_reversible_candidate_radius < 1
            || !(p.bp_reversible_projection_tolerance > 0.0)
            || !(p.bp_reversible_entropy_floor > 0.0)
            || p.bp_reversible_entropy_tolerance < 0.0
            || !std::isfinite(p.bp_reversible_entropy_tolerance)
            || !(p.bp_reversible_active_shell_floor_fraction > 0.0)
            || !(p.bp_reversible_active_shell_floor_fraction <= 1.0)
            || !std::isfinite(
                p.bp_reversible_active_shell_floor_fraction)) {
            throw std::runtime_error(
                "reversible Boltzmann backends require positive projection, entropy, "
                "stencil, and active-shell controls.");
        }
        if (p.bp_boltzmann_mass_tol != 0.0) {
            throw std::runtime_error(
                "reversible Boltzmann backends require bp_boltzmann_mass_tol=0; "
                "their sampled backend controls work with an unbiased full-support proposal.");
        }
    }
    if (p.bp_collision_operator == "boltzmann_reversible"
        && p.bp_reversible_max_velocity_cells < 4) {
        throw std::runtime_error(
            "boltzmann_reversible requires bp_reversible_max_velocity_cells >= 4.");
    }
    if (p.bp_collision_operator == "boltzmann_reversible_sampled"
        || p.bp_collision_operator == "boltzmann_reversible_streamed") {
        // The sampled reaction proposal combines targeted components with a strictly
        // positive uniform contribution. That contribution supplies full pair support,
        // while independent batch tolerances control the sampled operator and heat flux.
        const double mixture_sum =
            p.bp_sampled_tail_fraction
            + p.bp_sampled_stratified_fraction
            + p.bp_sampled_equilibrium_fraction
            + p.bp_sampled_phase_density_fraction
            + p.bp_sampled_uniform_fraction;
        if (p.bp_sampled_dense_cells < 0
            || p.bp_sampled_min_samples_per_batch < 1
            || p.bp_sampled_max_samples_per_batch
                < p.bp_sampled_min_samples_per_batch
            || p.bp_sampled_replicates_per_estimate < 1
            || p.bp_sampled_outgoing_angle_samples < 1
            || p.bp_sampled_guided_pilot_samples < 0
            || p.bp_sampled_guided_pairs < 0
            || !(p.bp_sampled_guided_fraction >= 0.0)
            || !(p.bp_sampled_guided_fraction < 1.0)
            || !std::isfinite(p.bp_sampled_guided_fraction)
            || !(p.bp_sampled_operator_refresh_l1_fraction >= 0.0)
            || !std::isfinite(
                p.bp_sampled_operator_refresh_l1_fraction)
            || p.bp_sampled_ntc_pilot_samples < 1
            || !(p.bp_sampled_ntc_quantile > 0.0)
            || !(p.bp_sampled_ntc_quantile <= 1.0)
            || !std::isfinite(p.bp_sampled_ntc_quantile)
            || p.bp_sampled_energy_strata < 1
            || p.bp_sampled_L_strata < 1
            || p.bp_sampled_tail_fraction < 0.0
            || p.bp_sampled_stratified_fraction < 0.0
            || p.bp_sampled_equilibrium_fraction < 0.0
            || !std::isfinite(p.bp_sampled_equilibrium_fraction)
            || p.bp_sampled_phase_density_fraction < 0.0
            || !std::isfinite(p.bp_sampled_phase_density_fraction)
            || !(p.bp_sampled_stratum_uniform_fraction >= 0.0)
            || !(p.bp_sampled_stratum_uniform_fraction <= 1.0)
            || !std::isfinite(p.bp_sampled_stratum_uniform_fraction)
            || !(p.bp_sampled_uniform_fraction > 0.0)
            || !(mixture_sum < 1.0)
            || !std::isfinite(mixture_sum)
            || p.bp_sampled_tail_power < 0.0
            || !std::isfinite(p.bp_sampled_tail_power)
            || !(p.bp_sampled_rhs_tolerance > 0.0)
            || !(p.bp_sampled_rhs_floor_fraction > 0.0)
            || !(p.bp_sampled_rhs_floor_fraction <= 1.0)
            || !std::isfinite(p.bp_sampled_rhs_floor_fraction)
            || !(p.bp_sampled_heat_tolerance > 0.0)
            || !(p.bp_sampled_heat_floor_fraction > 0.0)
            || !(p.bp_sampled_heat_floor_fraction <= 1.0)
            || !std::isfinite(p.bp_sampled_heat_floor_fraction)
            || !(p.bp_sampled_activity_boost_max >= 1.0)
            || !std::isfinite(p.bp_sampled_activity_boost_max)) {
            throw std::runtime_error(
                "boltzmann_reversible_sampled requires positive sample bounds, "
                "replicate, outgoing-angle, and NTC pilot counts, an NTC quantile "
                "in (0,1], positive tolerances, a heat-floor fraction in "
                "(0,1], an RHS-floor fraction in (0,1], nonnegative mixture weights "
                "summing below one, a strictly positive uniform full-support fraction, "
                "a finite nonnegative operator-graph refresh L1 fraction, and "
                "activity_boost_max >= 1.");
        }
    }
    if (using_boltzmann_poisson(p)) {
        if (p.bp_collision_strength < 0.0) {
            throw std::runtime_error("bp_collision_strength must be non-negative.");
        }
        if (p.bp_sigma0 < 0.0 || p.bp_sigma_u0 <= 0.0 || p.bp_sigma_n < 0.0) {
            throw std::runtime_error("Require bp_sigma0 >= 0, bp_sigma_u0 > 0, and bp_sigma_n >= 0.");
        }
        if (p.bp_subhalo_mass <= 0.0 || p.bp_subhalo_radius <= 0.0 || p.bp_coulomb_log < 0.0) {
            throw std::runtime_error(
                "Require bp_subhalo_mass > 0, bp_subhalo_radius > 0, and bp_coulomb_log >= 0.");
        }
        if (p.bp_yukawa_alpha < 0.0 || p.bp_yukawa_mchi <= 0.0 || p.bp_yukawa_mphi <= 0.0) {
            throw std::runtime_error(
                "Require bp_yukawa_alpha >= 0, bp_yukawa_mchi > 0, and bp_yukawa_mphi > 0.");
        }
        if (p.bp_temperature_floor <= 0.0
            || p.bp_density_floor_abs < 0.0
            || p.bp_density_floor_rel < 0.0
            || p.bp_partner_mass_tol < 0.0
            || p.bp_partner_mass_tol >= 1.0) {
            throw std::runtime_error(
                "Require bp_temperature_floor > 0, non-negative BP density floors, and 0 <= bp_partner_mass_tol < 1.");
        }
        if (p.bp_bath_speed_cutoff <= 0.0
            || p.bp_coeff_table_points < 4
            || p.bp_coeff_speed_points < 2
            || p.bp_coeff_mu_points < 2) {
            throw std::runtime_error(
                "Require positive bp_bath_speed_cutoff, at least four BP table points, and at least two quadrature points.");
        }
        if (p.bp_collision_cfl <= 0.0 || p.bp_collision_cfl > 1.0
            || p.bp_collision_max_substeps < 1
            || !(p.bp_collision_locality_max_ratio > 0.0)
            || !std::isfinite(p.bp_collision_locality_max_ratio)) {
            throw std::runtime_error(
                "Require 0 < bp_collision_cfl <= 1, "
                "bp_collision_max_substeps >= 1, and a positive finite "
                "bp_collision_locality_max_ratio.");
        }
        if (p.bp_boltzmann_pair_phi_points < 1
            || p.bp_boltzmann_scatter_mu_points < 1
            || p.bp_boltzmann_scatter_phi_points < 1
            || p.bp_boltzmann_mass_tol < 0.0
            || p.bp_boltzmann_mass_tol >= 1.0
            || p.bp_boltzmann_remap_max_skipped_fraction < 0.0
            || p.bp_boltzmann_remap_max_skipped_fraction >= 1.0) {
            throw std::runtime_error(
                "Require positive finite-angle Boltzmann quadrature counts, "
                "0 <= bp_boltzmann_mass_tol < 1, and "
                "0 <= bp_boltzmann_remap_max_skipped_fraction < 1.");
        }
        // Azimuthal quadrature must resolve an isotropic kernel before any physical
        // anisotropy is inferred from the calculated collision operator.
        if (p.bp_boltzmann_pair_phi_points < 4
            || p.bp_boltzmann_scatter_mu_points < 2
            || p.bp_boltzmann_scatter_phi_points < 4) {
            throw std::runtime_error(
                "Finite-angle Boltzmann quadrature requires pair_phi >= 4, "
                "scatter_mu >= 2, and scatter_phi >= 4. Smaller azimuth rules "
                "alias an isotropic collision kernel.");
        }
    }
    if (!(p.initial_data == "analytic" || binary_ic)) {
        throw std::runtime_error("initial_data must be 'analytic' or 'binary_rwl'.");
    }
    if (binary_ic && p.initial_data_file.empty()) {
        throw std::runtime_error("initial_data_file is required when initial_data=binary_rwl.");
    }
    if (binary_ic && p.adaptive_w) {
        throw std::runtime_error(
            "adaptive_w=true is disabled for binary_rwl initial data in this release: "
            "restart files store the current distribution but not a full replacement binary-IC w grid. "
            "Use adaptive_w=false for binary IC/restart workflows.");
    }
    // EVICRWL1 values are nodal data. A cell-average solver requires an explicit
    // measure-preserving conversion or its own compatible cell-average restart.
    if (binary_ic && using_cell_average_eta_grid(p)) {
        throw std::runtime_error(
            "kinetic_grid=cell_average_eta currently requires analytic initial data or a restart "
            "written by the same cell-average solver. Convert nodal EVICRWL1 data explicitly "
            "before using the fully kinetic path.");
    }
    if (p.tracer_initial_data) {
        if (!binary_ic) {
            throw std::runtime_error("tracer initial data currently requires initial_data=binary_rwl.");
        }
        if (p.tracer_initial_data_file.empty()) {
            throw std::runtime_error("tracer_initial_data_file is required when tracer_initial_data=true.");
        }
        if (p.adaptive_w) {
            throw std::runtime_error("tracer initial data currently requires adaptive_w=false.");
        }
    }
    if (p.custom_ic_normalize_mass && p.custom_ic_target_mass <= 0.0) {
        throw std::runtime_error("custom_ic_target_mass must be positive when normalization is enabled.");
    }
    if (p.Nr < 3 || p.Nw < 3 || p.NL < 2) {
        throw std::runtime_error("Need Nr >= 3, Nw >= 3, and NL >= 2.");
    }
    if (p.r_min <= 0.0 || p.r_max <= p.r_min) {
        throw std::runtime_error("Require 0 < r_min < r_max.");
    }
    if (!(p.custom_radial_coordinate == "linear" || p.custom_radial_coordinate == "log")) {
        throw std::runtime_error("custom_radial_coordinate must be 'linear' or 'log'.");
    }
    radial_grid_code(p.radial_grid);
    if (using_cell_average_eta_grid(p)
        && !(p.radial_grid == "log" || p.radial_grid == "linear")) {
        throw std::runtime_error(
            "kinetic_grid=cell_average_eta currently supports radial_grid=log or linear.");
    }
    if (using_cell_average_eta_grid(p)
        && !(p.kinetic_eta_grid == "uniform"
             || p.kinetic_eta_grid == "power"
             || p.kinetic_eta_grid == "composite_power")) {
        throw std::runtime_error(
            "kinetic_grid=cell_average_eta requires kinetic_eta_grid="
            "uniform, power, or composite_power.");
    }
    if (using_cell_average_eta_grid(p)
        && (!(p.kinetic_eta_power >= 1.0)
            || !std::isfinite(p.kinetic_eta_power))) {
        throw std::runtime_error(
            "kinetic_grid=cell_average_eta requires finite kinetic_eta_power >= 1.");
    }
    if (using_cell_average_eta_grid(p)
        && p.kinetic_eta_grid == "composite_power"
        && (p.kinetic_eta_inner_cells < 1
            || p.kinetic_eta_inner_cells >= p.NL
            || !(p.kinetic_eta_join > 0.0)
            || !(p.kinetic_eta_join < 0.5 * p.L_max * p.L_max)
            || !std::isfinite(p.kinetic_eta_join))) {
        throw std::runtime_error(
            "composite_power eta grid requires 1 <= inner cells < NL and "
            "0 < eta join < L_max^2/2.");
    }
    if (using_cell_average_eta_grid(p)
        && (p.kinetic_ic_quadrature_points < 1
            || p.kinetic_ic_quadrature_points > 12)) {
        throw std::runtime_error(
            "kinetic_grid=cell_average_eta requires "
            "1 <= kinetic_ic_quadrature_points <= 12.");
    }
    if (using_cell_average_eta_grid(p)
        && !using_any_sldg_transport(p)) {
        throw std::runtime_error(
            "kinetic_grid=cell_average_eta is an opt-in conservative representation "
            "and currently requires a cell-integrated SLDG transport.");
    }
    if (p.radial_grid == "piecewise-linear") {
        if (!(p.r_min < p.inner_refine_radius && p.inner_refine_radius < p.r_max)) {
            throw std::runtime_error(
                "Require r_min < inner_refine_radius < r_max for piecewise-linear grid.");
        }
        if (p.inner_refine_cells < 1 || p.inner_refine_cells > p.Nr - 2) {
            throw std::runtime_error(
                "Require 1 <= inner_refine_cells <= Nr - 2 for piecewise-linear grid.");
        }
    }
    if (p.w_max <= 0.0 || p.L_max <= 0.0) {
        throw std::runtime_error("w_max and L_max must be positive.");
    }
    if (p.w_stretch < 0.0) {
        throw std::runtime_error("w_stretch must be non-negative.");
    }
    if (using_cell_average_eta_grid(p) && std::abs(p.w_center) > 1.0e-12) {
        throw std::runtime_error(
            "kinetic_grid=cell_average_eta requires w_center=0 because its canonical "
            "radial-velocity cells span the symmetric interval [-w_max,w_max].");
    }
    if (using_cell_average_eta_grid(p) && p.adaptive_w) {
        throw std::runtime_error(
            "adaptive_w is not yet compatible with the cell-average eta geometry.");
    }
    if (!(p.w_center > -p.w_max && p.w_center < p.w_max)) {
        throw std::runtime_error("Require -w_max < w_center < w_max.");
    }
    if (std::abs(p.w_center) > 1.0e-12) {
        std::cerr
            << "Warning: the inner reflecting wall mirrors w about 0, but "
            << "--w-center is nonzero. This can under-resolve the reflected "
            << "outgoing branch. Prefer --w-center 0 for reflecting-center runs.\n";
    }
    if (p.adaptive_w) {
        if (p.adaptive_w_every < 1) {
            throw std::runtime_error("adaptive_w_every must be positive.");
        }
        if (!(p.adaptive_w_trigger > 0.0 && p.adaptive_w_trigger < 1.0)) {
            throw std::runtime_error("adaptive_w_trigger must lie between 0 and 1.");
        }
        if (p.adaptive_w_margin <= 1.0 || p.adaptive_w_growth <= 1.0) {
            throw std::runtime_error("adaptive_w_margin and adaptive_w_growth must be > 1.");
        }
        if (p.adaptive_w_max > 0.0 && p.adaptive_w_max <= p.w_max) {
            throw std::runtime_error("adaptive_w_max must exceed the initial w_max when set.");
        }
    }
    // A prescribed central mass gives an exterior-domain problem. Keep the inner
    // face safely outside its Schwarzschild radius; regular matter uses its own
    // central boundary condition.
    if (p.central_mass > 0.0 && p.r_min <= 2.05 * p.central_mass) {
        throw std::runtime_error(
            "For a nonzero central mass use r_min safely outside 2M, or leave "
            "central_mass=0 for the reflecting regular-center run.");
    }
    if (binary_ic && p.radial_grid != "custom") {
        throw std::runtime_error("Binary initial data must set radial_grid=custom after loading.");
    }
    if (!binary_ic && p.radial_grid == "custom") {
        throw std::runtime_error("radial_grid=custom is only available with initial_data=binary_rwl.");
    }
    if (!binary_ic && (p.shell_initial_df || p.double_gaussian_density_shell) && p.R0 <= 0.0) {
        throw std::runtime_error("Require R0 > 0 for shell initial data.");
    }
    if (!binary_ic && (p.shell_initial_df || p.double_gaussian_density_shell) && p.dR <= 0.0) {
        throw std::runtime_error("Require dR > 0 for shell initial data.");
    }
    if (!binary_ic && (p.shell_initial_df || p.double_gaussian_density_shell)
        && (p.R0 < p.r_min || p.R0 > p.r_max)) {
        throw std::runtime_error("Shell center R0 must lie inside the radial grid.");
    }
    const int shell_mode_count = (p.top_hat_shell ? 1 : 0)
        + (p.top_hat_density_shell ? 1 : 0)
        + (p.gaussian_density_shell ? 1 : 0)
        + (p.double_gaussian_density_shell ? 1 : 0);
    if (!binary_ic && shell_mode_count > 1) {
        throw std::runtime_error(
            "Choose only one shell radial mode: --top-hat-shell, "
            "--top-hat-density-shell, --gaussian-density-shell, "
            "or --double-gaussian-density-shell.");
    }
    if (!binary_ic && p.double_gaussian_density_shell) {
        if (p.R0_2 <= 0.0 || p.dR_2 <= 0.0) {
            throw std::runtime_error("Require R0_2 > 0 and dR_2 > 0 for double shell data.");
        }
        if (p.R0_2 < p.r_min || p.R0_2 > p.r_max) {
            throw std::runtime_error("Second shell center R0_2 must lie inside the radial grid.");
        }
        if (p.amp_2 < 0.0) {
            throw std::runtime_error("Require amp_2 >= 0 for double shell data.");
        }
        if (p.w_in_2 <= -p.w_max || p.w_in_2 >= p.w_max) {
            throw std::runtime_error("The second shell velocity center w_in_2 must lie inside the w grid.");
        }
    }
    if (!binary_ic && (p.top_hat_shell || p.top_hat_density_shell)
        && (p.R0 - 0.5 * p.dR < p.r_min || p.R0 + 0.5 * p.dR > p.r_max)) {
        std::cerr
            << "Warning: top-hat shell support extends outside the radial grid; "
            << "the initial shell will be truncated.\n";
    }
    if (!binary_ic && p.beta <= 0.0) {
        throw std::runtime_error("Require beta > 0 for the boundary-seed initial DF.");
    }
    if (!binary_ic && p.amp < 0.0) {
        throw std::runtime_error("Require amp >= 0 for the initial DF.");
    }
    if (!binary_ic && (p.w_in <= -p.w_max || p.w_in >= p.w_max)) {
        throw std::runtime_error("The initial DF velocity center w_in must lie inside the w grid.");
    }
    if (p.t_end < 0.0 || p.dt_max <= 0.0 || p.cfl <= 0.0) {
        throw std::runtime_error("Require t_end >= 0, dt_max > 0, and cfl > 0.");
    }
    // Prescribed inflow is a physical boundary source. Validate its timing,
    // positive masses, velocity widths and angular populations as a complete source.
    if (using_vp_outer_inflow(p)) {
        if (!using_vlasov_poisson(p)) {
            throw std::runtime_error("vp_outer_inflow is only implemented for physics_model=vlasov_poisson.");
        }
        if (p.vp_inflow_period <= 0.0 || p.vp_inflow_time_sigma <= 0.0
            || p.vp_inflow_time_cutoff <= 0.0) {
            throw std::runtime_error(
                "Require positive vp_inflow_period, vp_inflow_time_sigma, and vp_inflow_time_cutoff.");
        }
        if (p.vp_inflow_count <= 0) {
            throw std::runtime_error("Require vp_inflow_count > 0 when vp_outer_inflow is enabled.");
        }
        if (p.vp_inflow_low_l_count < 0 || p.vp_inflow_low_l_count > p.vp_inflow_count) {
            throw std::runtime_error("Require 0 <= vp_inflow_low_l_count <= vp_inflow_count.");
        }
        if (p.vp_inflow_high_l_delay < 0.0) {
            throw std::runtime_error("Require vp_inflow_high_l_delay >= 0.");
        }
        if (p.vp_inflow_shell_mass <= 0.0) {
            throw std::runtime_error("Require vp_inflow_shell_mass > 0.");
        }
        if (p.vp_inflow_high_shell_mass == 0.0) {
            throw std::runtime_error("vp_inflow_high_shell_mass must be positive or negative to reuse shell mass.");
        }
        if (p.vp_inflow_sigma_u <= 0.0 || p.vp_inflow_low_sigma_t <= 0.0
            || p.vp_inflow_high_sigma_t <= 0.0) {
            throw std::runtime_error(
                "Require positive vp_inflow_sigma_u, vp_inflow_low_sigma_t, and vp_inflow_high_sigma_t.");
        }
        if (p.vp_inflow_high_sigma_u == 0.0) {
            throw std::runtime_error("vp_inflow_high_sigma_u must be positive or negative to reuse sigma_u.");
        }
        if (p.vp_inflow_low_l_center < 0.0 || p.vp_inflow_high_l_center < 0.0) {
            throw std::runtime_error("Require non-negative vp_inflow L centers.");
        }
        if (p.vp_inflow_freefall_mass < 0.0 || p.vp_inflow_vff_factor < 0.0
            || p.vp_inflow_coord_speed < 0.0) {
            throw std::runtime_error(
                "Require non-negative vp_inflow_freefall_mass, vp_inflow_vff_factor, and vp_inflow_coord_speed.");
        }
        if (p.vp_inflow_coord_speed <= 0.0
            && !(p.vp_inflow_freefall_mass > 0.0 && p.vp_inflow_vff_factor > 0.0)) {
            throw std::runtime_error(
                "vp_outer_inflow needs either vp_inflow_coord_speed > 0 or positive freefall mass and vff factor.");
        }
    }
    if (!using_characteristic_integrator(p)
        && p.integrator != "conservative_finite_volume") {
        throw std::runtime_error(
            "integrator must be 'semi_lagrangian_kick_drift_kick', 'semi_lagrangian_rk', "
            "'runge_kutta', 'rk', 'semi_lagrangian_adams_bashforth', 'adams_bashforth', "
            "or 'conservative_finite_volume'.");
    }
    if (p.integrator == "conservative_finite_volume" && !using_vlasov_poisson(p)) {
        throw std::runtime_error(
            "integrator=conservative_finite_volume is implemented for Vlasov-Poisson/Boltzmann-Poisson runs.");
    }
    // The explicit upwind flux update retains CFL<=1. Characteristic integrator
    // orders and orbit-error tolerances are validated separately below.
    if (p.integrator == "conservative_finite_volume" && p.cfl > 1.0) {
        throw std::runtime_error("conservative_finite_volume requires 0 < cfl <= 1 for explicit upwind stability.");
    }
    if (p.vp_orbit_regularization != "off"
        && p.vp_orbit_regularization != "auto"
        && p.vp_orbit_regularization != "always") {
        throw std::runtime_error("vp_orbit_regularization must be off, auto, or always.");
    }
    if (!(p.vp_orbit_energy_tolerance > 0.0)
        || !(p.vp_orbit_lc_relative_tolerance > 0.0)
        || !(p.vp_orbit_lc_absolute_tolerance > 0.0)
        || p.vp_orbit_lc_max_steps < 8
        || p.vp_orbit_lc_max_refinements < 0) {
        throw std::runtime_error(
            "VP orbit regularization requires positive tolerances, max_steps >= 8, "
            "and max_refinements >= 0.");
    }
    if (using_rk_characteristic_integrator(p)) {
        if (p.time_integrator_order < 2 || p.time_integrator_order > 32) {
            throw std::runtime_error(
                "time_integrator_order must be 2..4 for explicit RK or 5..32 for extrapolated midpoint.");
        }
        if (p.time_integrator_substeps < 1) {
            throw std::runtime_error("time_integrator_substeps must be positive.");
        }
    }
    if (using_adams_bashforth_characteristic_integrator(p)) {
        if (p.time_integrator_order < 2 || p.time_integrator_order > 12) {
            throw std::runtime_error("time_integrator_order must be between 2 and 12 for Adams-Bashforth.");
        }
        if (p.time_integrator_substeps < 1) {
            throw std::runtime_error("time_integrator_substeps must be positive.");
        }
    }
    if (p.inner_boundary != "reflecting"
        && p.inner_boundary != "delayed_return"
        && p.inner_boundary != "origin_parity") {
        throw std::runtime_error(
            "Only inner_boundary=reflecting, delayed_return, or origin_parity is implemented.");
    }
    if (p.inner_boundary == "delayed_return"
        && !using_any_sldg_transport(p)) {
        throw std::runtime_error(
            "inner_boundary=delayed_return requires conservative_sldg or "
            "hamiltonian_sldg transport.");
    }
    if (p.outer_boundary != "outflow_vacuum"
        && p.outer_boundary != "vp_outer_inflow"
        && p.outer_boundary != "reflecting") {
        throw std::runtime_error("Only outer_boundary=outflow_vacuum, reflecting, or VP outer inflow is implemented.");
    }
    if (p.momentum_boundary != "outflow_vacuum") {
        throw std::runtime_error("Only momentum_boundary=outflow_vacuum is implemented.");
    }
    if (p.threads < 1) {
        throw std::runtime_error("Require threads >= 1.");
    }
    if (p.max_steps < 0) {
        throw std::runtime_error("Require max_steps >= 0.");
    }
    if (p.metric_floor <= 0.0 || p.gap_stop <= 0.0) {
        throw std::runtime_error("Require metric_floor > 0 and gap_stop > 0.");
    }
    if (p.active_floor_rel < 0.0 || p.active_floor_abs < 0.0 || p.f_floor < 0.0) {
        throw std::runtime_error("Active-cell floors must be non-negative.");
    }
}

// Initial radial shell envelope.  Density-weighted shell modes later multiply
// this by a moment-integral correction so rho(r,t=0) follows the requested
// envelope instead of inheriting a spurious spherical-measure power law.
/// Evaluates the requested radial shell envelope at one radius for analytic initial data.
double shell_profile(double r, const Params& p) {
    if (p.top_hat_shell || p.top_hat_density_shell) {
        return (r >= p.R0 - 0.5 * p.dR && r <= p.R0 + 0.5 * p.dR) ? 1.0 : 0.0;
    }
    const double x = (r - p.R0) / p.dR;
    return std::exp(-0.5 * x * x);
}

/// Interpolates a radial profile at an arbitrary radius using the selected radial coordinate.
double interpolate_radial_grid_value(
    const std::vector<double>& values,
    double rp,
    const std::vector<double>& r,
    const std::vector<double>& x_r,
    const Params& p) {
    const double rc = std::clamp(rp, r.front(), r.back());
    const double x = radial_coordinate(rc, p);
    auto it = std::upper_bound(x_r.begin(), x_r.end(), x);
    int i = static_cast<int>(std::distance(x_r.begin(), it)) - 1;
    i = std::clamp(i, 0, static_cast<int>(r.size()) - 2);
    const double x0 = x_r[i];
    const double x1 = x_r[i + 1];
    const double q = (x - x0) / (x1 - x0);
    return (1.0 - q) * values[i] + q * values[i + 1];
}

/// Constructs the built-in analytic initial distribution on the local (r,w,L) grid.
std::vector<double> initial_distribution(
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const std::vector<double>& mu_seed,
    const std::vector<double>& x_r,
    const Params& p) {
    std::vector<double> f(static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    // Construct each shell with its own bulk radial velocity and radial width,
    // then add their nonnegative phase distributions on the shared grid.
    if (p.double_gaussian_density_shell) {
        auto add_density_gaussian_shell = [&](double rb, double width, double amp, double w_center_shell) {
            const double eVb = std::exp(interpolate_radial_grid_value(mu_seed, rb, r, x_r, p));
            const double b = std::max(p.beta * eVb, 1.0e-300);
            const double norm = amp * std::sqrt(b / (2.0 * PI)) * b / (rb * rb);
            std::vector<double> base(static_cast<std::size_t>(p.Nw) * p.NL, 0.0);
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    const double ell_over_rb = L[k] / rb;
                    const double eps_shift = std::sqrt(
                        1.0 + (w[j] - w_center_shell) * (w[j] - w_center_shell)
                        + ell_over_rb * ell_over_rb);
                    base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)]
                        = norm * std::exp(-b * (eps_shift - 1.0));
                }
            }

            // VP rho is a mass density; Einstein--Vlasov rho includes
            // particle energy. Normalize the requested shell with that
            // same moment that subsequently sources the selected field.
            auto moment_integral = [&](double radius) {
                double out = 0.0;
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const double eps = std::sqrt(
                            1.0 + w[j] * w[j] + (L[k] / radius) * (L[k] / radius));
                        out += wq[j] * Lq[k] * L[k]
                            * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)]
                            * (using_vlasov_poisson(p) ? 1.0 : eps);
                    }
                }
                return out;
            };

            // Normalize the shell with the same density moment that sources gravity.
            // The r^2 factor compensates the spherical velocity Jacobian in that moment.
            const double I_rb = moment_integral(rb);
            for (int i = 0; i < p.Nr; ++i) {
                const double x = (r[i] - rb) / width;
                double shell = std::exp(-0.5 * x * x);
                if (shell <= 0.0) {
                    continue;
                }
                const double I_r = std::max(moment_integral(r[i]), 1.0e-300);
                shell *= (r[i] * r[i] / (rb * rb)) * (I_rb / I_r);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        f[idx(i, j, k, p)] += shell
                            * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)];
                    }
                }
            }
        };

        add_density_gaussian_shell(p.R0, p.dR, p.amp, p.w_in);
        add_density_gaussian_shell(p.R0_2, p.dR_2, p.amp_2, p.w_in_2);
        return f;
    }

    if (p.shell_initial_df) {
        const double rb = p.R0;
        const double eVb = std::exp(interpolate_radial_grid_value(mu_seed, rb, r, x_r, p));
        const double b = std::max(p.beta * eVb, 1.0e-300);
        const double norm = p.amp * std::sqrt(b / (2.0 * PI)) * b / (rb * rb);
        std::vector<double> base(static_cast<std::size_t>(p.Nw) * p.NL, 0.0);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const double ell_over_rb = L[k] / rb;
                const double eps_shift = std::sqrt(
                    1.0 + (w[j] - p.w_in) * (w[j] - p.w_in) + ell_over_rb * ell_over_rb);
                base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)]
                    = norm * std::exp(-b * (eps_shift - 1.0));
            }
        }

        // Use the density moment of the selected gravitational model.
        auto moment_integral = [&](double radius) {
            double out = 0.0;
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    const double eps = std::sqrt(1.0 + w[j] * w[j] + (L[k] / radius) * (L[k] / radius));
                    out += wq[j] * Lq[k] * L[k]
                        * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)]
                        * (using_vlasov_poisson(p) ? 1.0 : eps);
                }
            }
            return out;
        };
        // A requested density envelope includes the local moment correction. A
        // plain distribution envelope multiplies f directly and keeps its natural density.
        const bool density_weighted_shell = p.top_hat_density_shell || p.gaussian_density_shell;
        const double I_rb = density_weighted_shell ? moment_integral(rb) : 1.0;

        for (int i = 0; i < p.Nr; ++i) {
            double shell = shell_profile(r[i], p);
            if (shell > 0.0 && density_weighted_shell) {
                const double I_r = std::max(moment_integral(r[i]), 1.0e-300);
                shell *= (r[i] * r[i] / (rb * rb)) * (I_rb / I_r);
            }
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    f[idx(i, j, k, p)] = shell
                        * base[static_cast<std::size_t>(j) * p.NL + static_cast<std::size_t>(k)];
                }
            }
        }
        return f;
    }

    for (int i = 0; i < p.Nr; ++i) {
        const double eV = std::exp(mu_seed[i]);
        const double b = std::max(p.beta * eV, 1.0e-300);
        const double norm = p.amp * std::sqrt(b / (2.0 * PI)) * b / (r[i] * r[i]);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const double ell_over_r = L[k] / r[i];
                const double eps_shift = std::sqrt(
                    1.0 + (w[j] - p.w_in) * (w[j] - p.w_in) + ell_over_r * ell_over_r);
                f[idx(i, j, k, p)] = norm * std::exp(-b * (eps_shift - 1.0));
            }
        }
    }
    return f;
}

/// Integrates the analytic initial DF over every canonical (r,w,eta) cell.
///
/// Only one radial Gauss plane is materialized at a time, limiting temporary
/// storage to O(Nr (q Nw) (q NL)) rather than q^3 times the full state.
inline std::vector<double> initial_distribution_cell_average_eta(
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& mu_seed,
    const std::vector<double>& x_r,
    const Params& p) {
    validate_cell_average_eta_geometry(
        geometry, p, "analytic cell-average initial data");
    if (mu_seed.size() != static_cast<std::size_t>(p.Nr)
        || x_r.size() != static_cast<std::size_t>(p.Nr)) {
        throw std::runtime_error(
            "Analytic cell-average initial data received an incompatible radial seed.");
    }
    const auto& rule =
        phase_space_gauss_legendre_rule(p.kinetic_ic_quadrature_points);
    const std::vector<double>& node = rule.first;
    const std::vector<double>& weight = rule.second;
    const int nq = static_cast<int>(node.size());

    // Evaluate the analytic distribution at positive Gauss nodes inside each
    // velocity/eta cell. Dividing the eta quadrature weight by L converts it into
    // the compatible L*dL measure expected by the analytic moment calculation.
    Params sample_p = p;
    sample_p.Nw = p.Nw * nq;
    sample_p.NL = p.NL * nq;
    std::vector<double> sample_w(static_cast<std::size_t>(sample_p.Nw), 0.0);
    std::vector<double> sample_wq(static_cast<std::size_t>(sample_p.Nw), 0.0);
    for (int j = 0; j < p.Nw; ++j) {
        const double center =
            geometry.radial_velocity.center[static_cast<std::size_t>(j)];
        const double half_width =
            0.5 * geometry.radial_velocity.width[static_cast<std::size_t>(j)];
        for (int g = 0; g < nq; ++g) {
            const std::size_t q = static_cast<std::size_t>(j * nq + g);
            sample_w[q] = center + half_width * node[static_cast<std::size_t>(g)];
            sample_wq[q] = half_width * weight[static_cast<std::size_t>(g)];
        }
    }

    std::vector<double> sample_L(static_cast<std::size_t>(sample_p.NL), 0.0);
    std::vector<double> sample_Lq(static_cast<std::size_t>(sample_p.NL), 0.0);
    for (int k = 0; k < p.NL; ++k) {
        const double center = geometry.eta.center[static_cast<std::size_t>(k)];
        const double half_width =
            0.5 * geometry.eta.width[static_cast<std::size_t>(k)];
        for (int g = 0; g < nq; ++g) {
            const std::size_t q = static_cast<std::size_t>(k * nq + g);
            const double eta =
                center + half_width * node[static_cast<std::size_t>(g)];
            sample_L[q] = std::sqrt(2.0 * eta);
            sample_Lq[q] =
                half_width * weight[static_cast<std::size_t>(g)] / sample_L[q];
        }
    }

    std::vector<double> averaged(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    // Accumulate a true coordinate cell average in radial, velocity and eta
    // directions. Processing one radial Gauss plane at a time bounds temporary storage.
    for (int gr = 0; gr < nq; ++gr) {
        std::vector<double> sample_r(static_cast<std::size_t>(p.Nr), 0.0);
        std::vector<double> sample_mu(static_cast<std::size_t>(p.Nr), 0.0);
        for (int i = 0; i < p.Nr; ++i) {
            const double center =
                geometry.radial.center[static_cast<std::size_t>(i)];
            const double half_width =
                0.5 * geometry.radial.width[static_cast<std::size_t>(i)];
            sample_r[static_cast<std::size_t>(i)] =
                center + half_width * node[static_cast<std::size_t>(gr)];
            sample_mu[static_cast<std::size_t>(i)] =
                interpolate_radial_grid_value(
                    mu_seed,
                    sample_r[static_cast<std::size_t>(i)],
                    geometry.radial.center,
                    x_r,
                    p);
        }
        const std::vector<double> sample_x_r =
            radial_coordinate_grid(sample_r, sample_p);
        const std::vector<double> sampled = initial_distribution(
            sample_r,
            sample_w,
            sample_L,
            sample_wq,
            sample_Lq,
            sample_mu,
            sample_x_r,
            sample_p);
        const double radial_average_weight =
            0.5 * weight[static_cast<std::size_t>(gr)];
        for (int i = 0; i < p.Nr; ++i) {
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    double velocity_average = 0.0;
                    for (int gw = 0; gw < nq; ++gw) {
                        for (int ge = 0; ge < nq; ++ge) {
                            const double velocity_weight =
                                0.25
                                * weight[static_cast<std::size_t>(gw)]
                                * weight[static_cast<std::size_t>(ge)];
                            velocity_average += velocity_weight * sampled[idx(
                                i,
                                j * nq + gw,
                                k * nq + ge,
                                sample_p)];
                        }
                    }
                    averaged[idx(i, j, k, p)] +=
                        radial_average_weight * velocity_average;
                }
            }
        }
    }
    validate_finite_nonnegative_distribution(
        averaged,
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL,
        "analytic cell-average initial data");
    return averaged;
}
}  // namespace ev
