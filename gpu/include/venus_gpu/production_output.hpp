#pragma once

// Publish host-side diagnostics from completed GPU particle endpoints.
// Inputs are extensive shell moments, mesh geometry, time, and field options.
// The writer returns CSV profiles and selected field snapshots with explicit
// normalization and reconstruction conventions. Profile publication uses a
// flushed temporary sibling and atomic rename, so readers receive one complete
// profile. Snapshot schedules reuse existing profile callbacks and step times.

#include "venus_gpu/vlasov_poisson.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace venus_gpu {
namespace driver {

/// Atomically write profile_NNNNN.csv through a temporary sibling file.
/// Density is shell mass/volume. Velocity means and variances divide their
/// extensive moments by that same shell mass, with explicit empty-cell values.
void write_profile_csv(
    const std::string& directory,
    std::int64_t index,
    double time,
    const FieldHost& field,
    const RadialMesh& mesh);

/// Parse a comma-separated physical-time schedule; an empty string disables it.
std::vector<double> parse_phase_space_snapshot_times(const std::string& text);

/// Each snapshot uses an existing profile callback, preserving orbit step times.
void validate_phase_space_snapshot_times(
    const std::vector<double>& times, double output_interval);

/// Return the profile-grid index for a selected callback, or -1 otherwise.
std::int64_t phase_space_snapshot_index(
    const std::vector<double>& times, double time, double output_interval);

/// Save the actual field used at a completed particle endpoint, including its
/// finite-volume reconstruction, mesh, expansion, and potential convention.
/// The saved source masses and line densities allow the finite-volume
/// potential and force used at this endpoint to be reconstructed.
void write_phase_space_field_json(
    const std::string& path,
    double time,
    const FieldHost& field,
    const RadialMesh& mesh,
    const VlasovPoissonOptions& options,
    bool density_contrast_weights);

}  // namespace driver
}  // namespace venus_gpu
