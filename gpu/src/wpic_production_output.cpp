// Write GPU endpoint profiles and selected gravitational-field snapshots.
// Inputs are completed shell moments, the radial mesh, physical time, and
// background/boundary options. Outputs are CSV profiles and JSON field files.
// The writer converts extensive moments using their shell mass and volume,
// records the actual field reconstruction, and publishes complete files by
// atomic rename. It leaves particle motion and collision clocks unchanged.

#include "venus_gpu/production_output.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace venus_gpu {
namespace driver {

// Parse finite nonnegative physical times, then order them and reject
// duplicates so each requested particle/field snapshot has one callback.
std::vector<double> parse_phase_space_snapshot_times(const std::string& text) {
    std::vector<double> times;
    if (text.empty()) return times;
    std::stringstream input(text);
    std::string token;
    while (std::getline(input, token, ',')) {
        std::size_t used = 0;
        double value = 0.0;
        try {
            value = std::stod(token, &used);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid phase-space snapshot time.");
        }
        if (!std::isfinite(value) || value < 0.0
            || token.find_first_not_of(" \t\r", used) != std::string::npos) {
            throw std::runtime_error("Invalid phase-space snapshot time.");
        }
        times.push_back(value);
    }
    if (text.back() == ',') {
        throw std::runtime_error("Empty phase-space snapshot time.");
    }
    std::sort(times.begin(), times.end());
    if (std::adjacent_find(times.begin(), times.end()) != times.end()) {
        throw std::runtime_error("Repeated phase-space snapshot time.");
    }
    return times;
}

// Restrict snapshots to existing profile times. This observation schedule
// therefore adds output without introducing a new orbit-step boundary.
void validate_phase_space_snapshot_times(
    const std::vector<double>& times, double output_interval) {
    if (!(output_interval > 0.0) || !std::isfinite(output_interval)) {
        throw std::runtime_error("Invalid phase-space profile spacing.");
    }
    std::int64_t previous = -1;
    for (const double time : times) {
        const double index = time / output_interval;
        const double rounded = std::round(index);
        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(time));
        if (!std::isfinite(time) || time < 0.0 || !std::isfinite(index)
            || rounded >= static_cast<double>(
                std::numeric_limits<std::int64_t>::max())
            || std::abs(time - rounded * output_interval) > tolerance
            || static_cast<std::int64_t>(rounded) <= previous) {
            throw std::runtime_error(
                "Phase-space snapshot times must be distinct profile times.");
        }
        previous = static_cast<std::int64_t>(rounded);
    }
}

// Match the accepted callback time to its scheduled index using the same
// endpoint tolerance as profile publication.
std::int64_t phase_space_snapshot_index(
    const std::vector<double>& times, double time, double output_interval) {
    for (const double selected : times) {
        // This is the production profile callback's existing time tolerance.
        // Store the actual endpoint time as well as the requested grid index.
        if (std::abs(time - selected)
            <= 1.0e-11 * std::max(1.0, std::abs(selected))) {
            return static_cast<std::int64_t>(std::round(selected / output_interval));
        }
    }
    return -1;
}

// Save the actual endpoint field and its reconstruction inputs. Validate
// dimensions first so each shell moment shares one radial mesh and epoch.
void write_phase_space_field_json(
    const std::string& path,
    double time,
    const FieldHost& field,
    const RadialMesh& mesh,
    const VlasovPoissonOptions& options,
    bool density_contrast_weights) {
    const std::size_t cells = mesh.cells();
    const std::vector<const std::vector<double>*> cell_arrays = {
        &mesh.radius_center, &mesh.shell_volume,
        &field.shell_mass, &field.shell_radial_momentum,
        &field.shell_radial_second_moment,
        &field.shell_tangential_second_moment,
        &field.shell_radial_energy_flux, &field.shell_mass_square,
        &field.effective_marker_number, &field.source_line_density,
        &field.phi, &field.phi_r,
    };
    for (const auto* values : cell_arrays) {
        if (values->size() != cells) {
            throw std::runtime_error("Inconsistent phase-space field lengths.");
        }
    }
    if (field.source_mass_edge.size() != cells + 1
        || mesh.edge.size() != cells + 1 || !std::isfinite(time)) {
        throw std::runtime_error("Invalid phase-space field mesh or time.");
    }
    // Record a, H, and a_ddot/a for the selected background. These convert
    // peculiar radial velocity to physical velocity u=w+H*r in later analysis.
    const double tau = time + options.expansion.time_offset;
    const double scale = options.expansion.enabled
        ? std::pow(tau / options.expansion.time_offset, options.expansion.alpha)
        : 1.0;
    const double hubble = options.expansion.enabled
        ? options.expansion.alpha / tau : 0.0;
    const double acceleration = options.expansion.enabled
        ? options.expansion.alpha * (options.expansion.alpha - 1.0) / (tau * tau)
        : 0.0;
    if (!(scale > 0.0) || !std::isfinite(scale)
        || !std::isfinite(hubble) || !std::isfinite(acceleration)) {
        throw std::runtime_error("Invalid phase-space expansion state.");
    }
    const fs::path temporary = path + ".tmp";
    std::ofstream output(temporary);
    output << std::setprecision(17)
        << "{\n  \"schema\": \"VENUS_phase_space_field_1\",\n"
        << "  \"time\": " << time
        << ",\n  \"radial_coordinate\": \"physical_radius\",\n"
        << "  \"particle_radial_velocity\": \"w; u=w+H*r\",\n"
        << "  \"particle_angular_coordinate\": \"eta=L^2/2\",\n"
        << "  \"phase_volume\": \"8*pi^2*dr*du*deta\",\n"
        << "  \"marker_weight\": \""
        << (density_contrast_weights ? "density_contrast_mass" : "total_mass")
        << "\",\n  \"potential\": \""
        << (density_contrast_weights ? "peculiar" : "total")
        << "\",\n  \"potential_normalization\": \"zero_at_infinity\",\n"
        << "  \"poisson_reconstruction\": \"constant_line_mass\",\n"
        << "  \"gravity\": " << options.gravity
        << ",\n  \"central_mass\": " << options.central_mass
        << ",\n  \"scale_factor\": " << scale
        << ",\n  \"H\": " << hubble
        << ",\n  \"a_ddot_over_a\": " << acceleration
        << ",\n  \"expansion_enabled\": "
        << (options.expansion.enabled ? "true" : "false")
        << ",\n  \"expansion_alpha\": " << options.expansion.alpha
        << ",\n  \"expansion_time_offset\": " << options.expansion.time_offset
        << ",\n  \"subtract_homogeneous_background\": "
        << (options.subtract_homogeneous_background ? "true" : "false")
        << ",\n  \"inner_boundary\": \""
        << (options.inner_boundary == InnerBoundary::origin_parity
            ? "origin_parity" : "reflecting_wall")
        << "\",\n  \"outer_boundary\": \""
        << (options.outer_boundary == OuterBoundary::reflecting_wall
            ? "reflecting_wall" : "outflow")
        << "\",\n  \"carry_active_mass_inside_inner_face\": "
        << (options.carry_active_mass_inside_inner_face ? "true" : "false");
    // Every serialized array must be finite. Source face masses and line
    // densities preserve the force/potential reconstruction used by the solver.
    const auto array = [&](const char* name, const std::vector<double>& values) {
        output << ",\n  \"" << name << "\": [";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i])) {
                throw std::runtime_error("Non-finite phase-space field value.");
            }
            if (i) output << ',';
            output << values[i];
        }
        output << ']';
    };
    array("radius_edge", mesh.edge);
    array("radius_center", mesh.radius_center);
    array("shell_volume", mesh.shell_volume);
    array("shell_mass", field.shell_mass);
    array("shell_radial_momentum", field.shell_radial_momentum);
    array("shell_radial_second_moment", field.shell_radial_second_moment);
    array("shell_tangential_second_moment", field.shell_tangential_second_moment);
    array("shell_radial_energy_flux", field.shell_radial_energy_flux);
    array("shell_mass_square", field.shell_mass_square);
    array("effective_marker_number", field.effective_marker_number);
    array("source_mass_edge", field.source_mass_edge);
    array("source_line_density", field.source_line_density);
    array("phi", field.phi);
    array("phi_r", field.phi_r);
    output << ",\n  \"resolved_active_mass\": "
        << field.diagnostics.resolved_active_mass
        << ",\n  \"deposited_mesh_mass\": " << field.diagnostics.deposited_mesh_mass
        << ",\n  \"hidden_inner_mass\": " << field.diagnostics.hidden_inner_mass
        << ",\n  \"escaped_mass\": " << field.diagnostics.escaped_mass
        << ",\n  \"mass_roundoff\": " << field.diagnostics.mass_roundoff
        << "\n}\n";
    output.flush();
    if (!output) throw std::runtime_error("Failed writing phase-space field.");
    output.close();
    fs::rename(temporary, path);
}

// Convert extensive moments to density and mass-weighted velocity moments
// on the same completed field state. Empty shells receive explicit zeros.
void write_profile_csv(
    const std::string& directory,
    std::int64_t index,
    double time,
    const FieldHost& field,
    const RadialMesh& mesh) {
    const std::size_t cells = mesh.shell_volume.size();
    if (field.shell_mass.size() != cells
        || field.shell_radial_momentum.size() != cells
        || field.shell_radial_second_moment.size() != cells
        || field.shell_tangential_second_moment.size() != cells
        || field.effective_marker_number.size() != cells
        || mesh.radius_center.size() != cells) {
        throw std::runtime_error(
            "GPU profile field/mesh arrays have inconsistent lengths.");
    }
    std::ostringstream name;
    name << "profile_" << std::setw(5) << std::setfill('0') << index << ".csv";
    const fs::path target = fs::path(directory) / name.str();
    const fs::path temporary = target.string() + ".tmp";
    std::ofstream output(temporary);
    output << std::setprecision(17)
           << "time,radius,density,radial_mean,radial_variance,"
              "tangential_second,neff\n";
    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (!(mesh.shell_volume[cell] > 0.0)) {
            throw std::runtime_error(
                "GPU profile mesh contains a non-positive shell volume.");
        }
        // The central second moment removes the bulk radial flow. Clamp a
        // small negative subtraction residual to zero for a valid variance.
        const double shell_mass = field.shell_mass[cell];
        const double density = shell_mass / mesh.shell_volume[cell];
        const double mean = shell_mass > 0.0
            ? field.shell_radial_momentum[cell] / shell_mass : 0.0;
        const double radial_second = shell_mass > 0.0
            ? field.shell_radial_second_moment[cell] / shell_mass : 0.0;
        const double tangential = shell_mass > 0.0
            ? field.shell_tangential_second_moment[cell] / shell_mass : 0.0;
        output << time << ',' << mesh.radius_center[cell] << ',' << density
               << ',' << mean << ','
               << std::max(0.0, radial_second - mean * mean) << ','
               << tangential << ',' << field.effective_marker_number[cell]
               << '\n';
    }
    // Publish one complete profile only after its stream has been flushed
    // and closed, keeping asynchronous plotters on a single file generation.
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed writing GPU profile.");
    }
    output.close();
    fs::rename(temporary, target);
}

}  // namespace driver
}  // namespace venus_gpu
