#pragma once

// Binary restart of the canonical Boltzmann--Poisson event representation.
// The state includes orbit rows, phase masses, Eulerian overlap integrals, open
// carriers, the held Poisson field, event ages and random counters. Keeping these
// quantities together preserves the next transport/collision ordering on resume.
// The writer stores a version and byte-order sentinel, writes a temporary file,
// flushes it and renames it into place. The reader checks the encoded counts,
// enumerated states and event metadata while restoring the defined field order.
// Scalar/vector storage uses the host binary representation identified by the
// format. Format changes require a distinct version and explicit reading branch.

#include "ev/canonical_boltzmann_poisson_event.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <type_traits>

namespace ev {
namespace canonical_restart_detail {

// Primitive binary I/O.  Every variable-length payload carries a checked
// element count before storage is allocated, limiting damage from a corrupt
// or incompatible checkpoint.
template <class T>
inline void write_scalar(std::ofstream& output, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "restart scalar");
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error(
            "Failed while writing canonical Boltzmann-Poisson restart.");
    }
}

template <class T>
inline T read_scalar(std::ifstream& input) {
    static_assert(std::is_trivially_copyable<T>::value, "restart scalar");
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error(
            "Unexpected end of canonical Boltzmann-Poisson restart.");
    }
    return value;
}

template <class T>
inline void write_vector(std::ofstream& output, const std::vector<T>& value) {
    static_assert(std::is_trivially_copyable<T>::value, "restart vector");
    write_scalar(output, static_cast<std::uint64_t>(value.size()));
    if (!value.empty()) {
        output.write(
            reinterpret_cast<const char*>(value.data()),
            static_cast<std::streamsize>(sizeof(T) * value.size()));
        if (!output) {
            throw std::runtime_error(
                "Failed while writing canonical restart vector.");
        }
    }
}

template <class T>
inline std::vector<T> read_vector(
    std::ifstream& input,
    std::uint64_t maximum_size = UINT64_C(1000000000)) {
    static_assert(std::is_trivially_copyable<T>::value, "restart vector");
    const std::uint64_t size = read_scalar<std::uint64_t>(input);
    if (size > maximum_size
        || size > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "Canonical restart vector exceeds its size bound.");
    }
    std::vector<T> value(static_cast<std::size_t>(size));
    if (!value.empty()) {
        input.read(
            reinterpret_cast<char*>(value.data()),
            static_cast<std::streamsize>(sizeof(T) * value.size()));
        if (!input) {
            throw std::runtime_error(
                "Unexpected end of canonical restart vector.");
        }
    }
    return value;
}

// Store the complete Poisson source and potential at the event time.  A
// restart therefore resumes with the same force field rather than solving a
// slightly different field from a reconstructed distribution.
// The held potential and its complete radial source reconstruction belong to
// the dynamical state. Re-solving from a diagnostic f could introduce a field event.
inline void write_field(
    std::ofstream& output,
    const VlasovPoissonField& field) {
    write_vector(output, field.source_mass);
    write_vector(output, field.phi);
    write_vector(output, field.phi_r);
    write_vector(output, field.radial_edge);
    write_vector(output, field.source_mass_edge);
    write_vector(output, field.source_line_density);
    write_vector(output, field.source_line_density_linear);
    write_vector(output, field.source_line_density_quadratic);
    write_vector(output, field.phi_edge);
    write_scalar(output, field.background.tau);
    write_scalar(output, field.background.a);
    write_scalar(output, field.background.H);
    write_scalar(output, field.background.rho_bar);
    write_scalar(output, field.matter_mass);
    write_scalar(output, field.outer_source_mass);
    write_scalar(output, field.hidden_carrier_mass);
    write_scalar(output, field.fixed_central_mass);
    write_scalar(output, field.min_gap);
}

inline VlasovPoissonField read_field(
    std::ifstream& input,
    std::uint32_t restart_version) {
    VlasovPoissonField field;
    field.source_mass = read_vector<double>(input);
    field.phi = read_vector<double>(input);
    field.phi_r = read_vector<double>(input);
    field.radial_edge = read_vector<double>(input);
    field.source_mass_edge = read_vector<double>(input);
    field.source_line_density = read_vector<double>(input);
    if (restart_version >= 3) {
        field.source_line_density_linear = read_vector<double>(input);
        field.source_line_density_quadratic = read_vector<double>(input);
    }
    field.phi_edge = read_vector<double>(input);
    field.background.tau = read_scalar<double>(input);
    field.background.a = read_scalar<double>(input);
    field.background.H = read_scalar<double>(input);
    field.background.rho_bar = read_scalar<double>(input);
    field.matter_mass = read_scalar<double>(input);
    field.outer_source_mass = read_scalar<double>(input);
    field.hidden_carrier_mass = read_scalar<double>(input);
    field.fixed_central_mass = read_scalar<double>(input);
    field.min_gap = read_scalar<double>(input);
    return field;
}

// Preserve the orbit map for one canonical row, including visible travel
// time, hidden-return time, turning points, and sampled phase locations.
// An orbit row needs its turning radii and visible/hidden travel times. These
// data map phase back to (r,w) and determine when inner carriers return.
inline void write_phase_geometry(
    std::ofstream& output,
    const HamiltonianOrbitPhaseGeometry& geometry) {
    write_scalar(output, geometry.eta);
    write_scalar(output, geometry.energy);
    write_scalar(output, geometry.pericenter);
    write_scalar(output, geometry.apocenter);
    write_scalar(output, static_cast<std::uint8_t>(
        geometry.inner_is_turning_point));
    write_scalar(output, geometry.visible_outgoing_time);
    write_scalar(output, geometry.hidden_return_time);
    write_scalar(output, geometry.radial_period);
    write_vector(output, geometry.alpha_edge);
    write_vector(output, geometry.outgoing_time_edge);
    write_scalar(output, geometry.time_quadrature_scale);
    write_scalar(
        output, static_cast<std::uint64_t>(geometry.phase_center.size()));
    for (const HamiltonianOrbitPhaseGeometry::Location& location :
         geometry.phase_center) {
        write_scalar(output, location.state.r);
        write_scalar(output, location.state.w);
        write_scalar(output, static_cast<std::uint8_t>(location.visible));
        write_scalar(output, location.hidden_remaining_time);
    }
}

inline HamiltonianOrbitPhaseGeometry read_phase_geometry(
    std::ifstream& input) {
    HamiltonianOrbitPhaseGeometry geometry;
    geometry.eta = read_scalar<double>(input);
    geometry.energy = read_scalar<double>(input);
    geometry.pericenter = read_scalar<double>(input);
    geometry.apocenter = read_scalar<double>(input);
    geometry.inner_is_turning_point =
        read_scalar<std::uint8_t>(input) != 0;
    geometry.visible_outgoing_time = read_scalar<double>(input);
    geometry.hidden_return_time = read_scalar<double>(input);
    geometry.radial_period = read_scalar<double>(input);
    geometry.alpha_edge = read_vector<double>(input);
    geometry.outgoing_time_edge = read_vector<double>(input);
    geometry.time_quadrature_scale = read_scalar<double>(input);
    const std::uint64_t locations = read_scalar<std::uint64_t>(input);
    if (locations > UINT64_C(1000000000)) {
        throw std::runtime_error(
            "Canonical restart phase map exceeds its size bound.");
    }
    geometry.phase_center.resize(static_cast<std::size_t>(locations));
    for (HamiltonianOrbitPhaseGeometry::Location& location :
         geometry.phase_center) {
        location.state.r = read_scalar<double>(input);
        location.state.w = read_scalar<double>(input);
        location.visible = read_scalar<std::uint8_t>(input) != 0;
        location.hidden_remaining_time = read_scalar<double>(input);
    }
    return geometry;
}

// Serialize the canonical orbit atlas in physical dependency order: orbit
// rows, eta-plane ownership, Eulerian overlaps, and conservation ledgers.
// Serialize phase mass and its first two energy moments before the overlap
// geometry. The corresponding column budgets partition visible, hidden and escaped
// phase volume on the same frozen Hamiltonian.
inline void write_atlas(
    std::ofstream& output,
    const CanonicalOrbitAtlas& atlas) {
    write_scalar(output, atlas.generation);
    write_scalar(output, atlas.field_signature);
    write_scalar(output, atlas.geometry_signature);
    write_scalar(output, static_cast<std::uint64_t>(atlas.row.size()));
    for (const CanonicalOrbitRow& row : atlas.row) {
        write_scalar(output, static_cast<std::uint8_t>(row.kind));
        write_scalar(output, row.eta_cell);
        write_scalar(output, row.eta_plane);
        write_scalar(output, row.eta);
        write_scalar(output, row.eta_weight);
        write_scalar(output, row.action_left);
        write_scalar(output, row.action_right);
        write_scalar(output, row.energy_left);
        write_scalar(output, row.energy_right);
        write_scalar(output, row.energy);
        write_scalar(output, row.period);
        write_scalar(output, row.omega);
        write_scalar(output, row.phase_offset);
        write_scalar(output, row.hidden_phase_begin);
        write_scalar(output, row.phase_volume);
        write_scalar(output, static_cast<std::uint8_t>(
            row.left_edge_is_inner_separatrix));
        write_scalar(output, static_cast<std::uint8_t>(
            row.right_edge_is_inner_separatrix));
        write_scalar(output, row.first_cell);
        write_vector(output, row.phase_mass);
        write_vector(output, row.phase_energy_mass);
        write_vector(output, row.phase_energy2_mass);
        write_phase_geometry(output, row.phase_geometry);
    }
    write_vector(output, atlas.eta_row_offset);
    write_vector(output, atlas.eta_plane_row_offset);
    write_vector(output, atlas.eta_plane_cell);
    write_vector(output, atlas.eta_plane_coordinate);
    write_scalar(output, static_cast<std::uint64_t>(atlas.overlap.size()));
    for (const CanonicalOverlap& overlap : atlas.overlap) {
        write_scalar(output, overlap.eulerian_cell);
        write_scalar(output, overlap.canonical_cell);
        write_scalar(output, overlap.volume);
        write_scalar(output, overlap.energy_volume);
        write_scalar(output, overlap.energy2_volume);
        for (double moment : overlap.eulerian_p2_moment) {
            write_scalar(output, moment);
        }
    }
    write_scalar(output, static_cast<std::uint64_t>(atlas.column.size()));
    for (const CanonicalColumnLedger& column : atlas.column) {
        write_scalar(output, column.column_volume);
        write_scalar(output, column.visible_volume);
        write_scalar(output, column.hidden_volume);
        write_scalar(output, column.escaped_volume);
        write_scalar(output, column.residual);
        write_scalar(output, column.geometric_scale);
        write_scalar(output, column.maximum_patch_scale_error);
        write_scalar(output, column.accumulated_patch_excess_error);
        write_scalar(output, column.canonical_quadrature_volume);
        for (double moment : column.energy_volume_moment) {
            write_scalar(output, moment);
        }
        for (double moment : column.hidden_energy_volume_moment) {
            write_scalar(output, moment);
        }
        for (double moment : column.escaped_energy_volume_moment) {
            write_scalar(output, moment);
        }
        write_scalar(output, static_cast<std::uint8_t>(
            column.partition_energy_moments_resolved));
    }
}

inline CanonicalOrbitAtlas read_atlas(std::ifstream& input) {
    CanonicalOrbitAtlas atlas;
    atlas.generation = read_scalar<std::uint64_t>(input);
    atlas.field_signature = read_scalar<std::uint64_t>(input);
    atlas.geometry_signature = read_scalar<std::uint64_t>(input);
    const std::uint64_t rows = read_scalar<std::uint64_t>(input);
    if (rows > UINT64_C(100000000)) {
        throw std::runtime_error(
            "Canonical restart atlas exceeds its row bound.");
    }
    atlas.row.resize(static_cast<std::size_t>(rows));
    for (CanonicalOrbitRow& row : atlas.row) {
        const std::uint8_t kind = read_scalar<std::uint8_t>(input);
        if (kind > static_cast<std::uint8_t>(
                CanonicalOrbitRowKind::velocity_outflow)) {
            throw std::runtime_error(
                "Canonical restart contains an invalid row kind.");
        }
        row.kind = static_cast<CanonicalOrbitRowKind>(kind);
        row.eta_cell = read_scalar<int>(input);
        row.eta_plane = read_scalar<int>(input);
        row.eta = read_scalar<double>(input);
        row.eta_weight = read_scalar<double>(input);
        row.action_left = read_scalar<double>(input);
        row.action_right = read_scalar<double>(input);
        row.energy_left = read_scalar<double>(input);
        row.energy_right = read_scalar<double>(input);
        row.energy = read_scalar<double>(input);
        row.period = read_scalar<double>(input);
        row.omega = read_scalar<double>(input);
        row.phase_offset = read_scalar<double>(input);
        row.hidden_phase_begin = read_scalar<double>(input);
        row.phase_volume = read_scalar<double>(input);
        row.left_edge_is_inner_separatrix =
            read_scalar<std::uint8_t>(input) != 0;
        row.right_edge_is_inner_separatrix =
            read_scalar<std::uint8_t>(input) != 0;
        row.first_cell = read_scalar<std::uint32_t>(input);
        row.phase_mass = read_vector<double>(input);
        row.phase_energy_mass = read_vector<double>(input);
        row.phase_energy2_mass = read_vector<double>(input);
        row.phase_geometry = read_phase_geometry(input);
        // Rebuild the derived phase-clock representation from the restored row data.
        // The stored phase offset remains the physical location along that orbit.
        canonical_orbit_synchronize_row_phase_clock(row);
    }
    atlas.eta_row_offset = read_vector<std::uint32_t>(input);
    atlas.eta_plane_row_offset = read_vector<std::uint32_t>(input);
    atlas.eta_plane_cell = read_vector<int>(input);
    atlas.eta_plane_coordinate = read_vector<double>(input);
    const std::uint64_t overlaps = read_scalar<std::uint64_t>(input);
    if (overlaps > UINT64_C(1000000000)) {
        throw std::runtime_error(
            "Canonical restart exceeds its overlap bound.");
    }
    atlas.overlap.resize(static_cast<std::size_t>(overlaps));
    for (CanonicalOverlap& overlap : atlas.overlap) {
        overlap.eulerian_cell = read_scalar<std::uint32_t>(input);
        overlap.canonical_cell = read_scalar<std::uint32_t>(input);
        overlap.volume = read_scalar<double>(input);
        overlap.energy_volume = read_scalar<double>(input);
        overlap.energy2_volume = read_scalar<double>(input);
        for (double& moment : overlap.eulerian_p2_moment) {
            moment = read_scalar<double>(input);
        }
    }
    const std::uint64_t columns = read_scalar<std::uint64_t>(input);
    if (columns > UINT64_C(1000000000)) {
        throw std::runtime_error(
            "Canonical restart exceeds its column bound.");
    }
    atlas.column.resize(static_cast<std::size_t>(columns));
    for (CanonicalColumnLedger& column : atlas.column) {
        column.column_volume = read_scalar<double>(input);
        column.visible_volume = read_scalar<double>(input);
        column.hidden_volume = read_scalar<double>(input);
        column.escaped_volume = read_scalar<double>(input);
        column.residual = read_scalar<double>(input);
        column.geometric_scale = read_scalar<double>(input);
        column.maximum_patch_scale_error = read_scalar<double>(input);
        column.accumulated_patch_excess_error = read_scalar<double>(input);
        column.canonical_quadrature_volume = read_scalar<double>(input);
        for (double& moment : column.energy_volume_moment) {
            moment = read_scalar<double>(input);
        }
        for (double& moment : column.hidden_energy_volume_moment) {
            moment = read_scalar<double>(input);
        }
        for (double& moment : column.escaped_energy_volume_moment) {
            moment = read_scalar<double>(input);
        }
        column.partition_energy_moments_resolved =
            read_scalar<std::uint8_t>(input) != 0;
    }
    return atlas;
}

// Open and detached collision carriers are genuine pieces of represented
// phase-space mass, so their mass, orbit state, ownership, and hidden-return
// clock must survive a restart exactly.
// A carrier stores its complete positive mass, trajectory state, source
// ownership and remaining inner travel time so that it can resume its motion.
inline void write_marker(
    std::ofstream& output,
    const WeightedPICMarker& marker) {
    write_scalar(output, marker.mass);
    write_scalar(output, marker.r);
    write_scalar(output, marker.w);
    write_scalar(output, marker.eta);
    write_scalar(output, marker.id);
    write_scalar(output, marker.source_cell);
    write_scalar(output, static_cast<std::uint8_t>(marker.state));
    write_scalar(output, marker.inner_boundary_speed);
    write_scalar(output, marker.remaining_inner_time);
}

inline WeightedPICMarker read_marker(std::ifstream& input) {
    WeightedPICMarker marker;
    marker.mass = read_scalar<double>(input);
    marker.r = read_scalar<double>(input);
    marker.w = read_scalar<double>(input);
    marker.eta = read_scalar<double>(input);
    marker.id = read_scalar<std::uint64_t>(input);
    marker.source_cell = read_scalar<std::uint64_t>(input);
    const std::uint8_t state = read_scalar<std::uint8_t>(input);
    if (state > static_cast<std::uint8_t>(WeightedPICMarkerState::escaped)) {
        throw std::runtime_error(
            "Canonical restart contains an invalid open-marker state.");
    }
    marker.state = static_cast<WeightedPICMarkerState>(state);
    marker.inner_boundary_speed = read_scalar<double>(input);
    marker.remaining_inner_time = read_scalar<double>(input);
    return marker;
}

}  // namespace canonical_restart_detail

// Write one complete event state to a temporary file, flush it, and rename it
// atomically.  An interruption can leave a disposable temporary file but
// cannot replace the last valid physical state with a partial checkpoint.
inline void write_canonical_boltzmann_poisson_restart(
    const std::filesystem::path& path,
    const CanonicalBoltzmannPoissonEventState& state) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Could not open canonical restart for writing: "
            + temporary.string());
    }
    const char magic[16] = {
        'V', 'E', 'N', 'U', 'S', 'C', 'A', 'N', 'B', 'P', '1', '\0'};
    output.write(magic, sizeof(magic));
    canonical_restart_detail::write_scalar(output, std::uint32_t{4});
    canonical_restart_detail::write_scalar(output, UINT32_C(0x01020304));
    canonical_restart_detail::write_scalar(output, state.time);
    canonical_restart_detail::write_scalar(output, state.step);
    // Potential and collision ages measure time since different operators.
    // Save their sampled deadlines together with both deterministic random counters.
    canonical_restart_detail::write_scalar(output, state.potential_age);
    canonical_restart_detail::write_scalar(output, state.collision_age);
    canonical_restart_detail::write_scalar(output, state.potential_timescale);
    canonical_restart_detail::write_scalar(
        output, state.potential_deadline_age);
    canonical_restart_detail::write_scalar(output, state.collision_timescale);
    canonical_restart_detail::write_scalar(
        output, state.collision_majorant_timescale);
    canonical_restart_detail::write_scalar(
        output, state.initial_physical_mass);
    canonical_restart_detail::write_scalar(
        output, state.hidden_physical_mass);
    canonical_restart_detail::write_scalar(
        output, state.startup_field_iterations);
    canonical_restart_detail::write_scalar(
        output, state.startup_field_mismatch);
    canonical_restart_detail::write_scalar(
        output, state.collision.marker_sampler.seed);
    canonical_restart_detail::write_scalar(
        output, state.collision.marker_sampler.next_counter);
    canonical_restart_detail::write_scalar(
        output, state.collision.collision_sampler.seed);
    canonical_restart_detail::write_scalar(
        output, state.collision.collision_sampler.next_counter);
    canonical_restart_detail::write_scalar(
        output, state.collision.next_open_marker_id);
    canonical_restart_detail::write_scalar(
        output,
        static_cast<std::uint64_t>(state.collision.open_marker.size()));
    for (const WeightedPICMarker& marker : state.collision.open_marker) {
        canonical_restart_detail::write_marker(output, marker);
    }
    canonical_restart_detail::write_scalar(
        output,
        static_cast<std::uint64_t>(state.collision.detached_marker.size()));
    for (const WeightedPICMarker& marker : state.collision.detached_marker) {
        canonical_restart_detail::write_marker(output, marker);
    }
    canonical_restart_detail::write_scalar(
        output,
        static_cast<std::uint64_t>(
            state.collision.signed_increment_positive.size()));
    for (const WeightedPICMarker& marker :
         state.collision.signed_increment_positive) {
        canonical_restart_detail::write_marker(output, marker);
    }
    canonical_restart_detail::write_scalar(
        output,
        static_cast<std::uint64_t>(
            state.collision.signed_increment_negative.size()));
    for (const WeightedPICMarker& marker :
         state.collision.signed_increment_negative) {
        canonical_restart_detail::write_marker(output, marker);
    }
    canonical_restart_detail::write_field(output, state.field);
    canonical_restart_detail::write_field(output, state.poisson_reference);
    canonical_restart_detail::write_vector(output, state.eulerian_f);
    canonical_restart_detail::write_atlas(output, state.atlas);
    // Publish the checkpoint after all serialized fields have reached the stream.
    // A failed write leaves the previous destination in place.
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "Failed to flush canonical restart: " + temporary.string());
    }
    output.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        throw std::runtime_error(
            "Could not install canonical restart: " + path.string());
    }
}

// Read each supported format version in its defined field order, restoring
// event clocks and random counters before orbit and collision work resumes.
// Bounds, enum ranges, finite physical metadata, magic, and byte order are
// checked before the state is returned to the solver.
inline CanonicalBoltzmannPoissonEventState
read_canonical_boltzmann_poisson_restart(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Could not open canonical restart: " + path.string());
    }
    char magic[16] = {};
    input.read(magic, sizeof(magic));
    const char expected[16] = {
        'V', 'E', 'N', 'U', 'S', 'C', 'A', 'N', 'B', 'P', '1', '\0'};
    if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error(
            "Invalid canonical restart magic: " + path.string());
    }
    const std::uint32_t version =
        canonical_restart_detail::read_scalar<std::uint32_t>(input);
    const std::uint32_t endian =
        canonical_restart_detail::read_scalar<std::uint32_t>(input);
    if ((version != 1 && version != 2 && version != 3 && version != 4)
        || endian != UINT32_C(0x01020304)) {
        throw std::runtime_error(
            "Unsupported canonical restart version or byte order.");
    }
    CanonicalBoltzmannPoissonEventState state;
    state.time = canonical_restart_detail::read_scalar<double>(input);
    state.step = canonical_restart_detail::read_scalar<std::int64_t>(input);
    state.potential_age =
        canonical_restart_detail::read_scalar<double>(input);
    state.collision_age =
        canonical_restart_detail::read_scalar<double>(input);
    state.potential_timescale =
        canonical_restart_detail::read_scalar<double>(input);
    state.potential_deadline_age =
        canonical_restart_detail::read_scalar<double>(input);
    state.collision_timescale =
        canonical_restart_detail::read_scalar<double>(input);
    state.collision_majorant_timescale =
        canonical_restart_detail::read_scalar<double>(input);
    state.initial_physical_mass =
        canonical_restart_detail::read_scalar<double>(input);
    state.hidden_physical_mass =
        canonical_restart_detail::read_scalar<double>(input);
    state.startup_field_iterations =
        canonical_restart_detail::read_scalar<int>(input);
    state.startup_field_mismatch =
        canonical_restart_detail::read_scalar<double>(input);
    state.collision.marker_sampler.seed =
        canonical_restart_detail::read_scalar<std::uint64_t>(input);
    state.collision.marker_sampler.next_counter =
        canonical_restart_detail::read_scalar<std::uint64_t>(input);
    state.collision.collision_sampler.seed =
        canonical_restart_detail::read_scalar<std::uint64_t>(input);
    state.collision.collision_sampler.next_counter =
        canonical_restart_detail::read_scalar<std::uint64_t>(input);
    state.collision.next_open_marker_id =
        canonical_restart_detail::read_scalar<std::uint64_t>(input);
    const std::uint64_t open_count =
        canonical_restart_detail::read_scalar<std::uint64_t>(input);
    if (open_count > UINT64_C(1000000000)) {
        throw std::runtime_error(
            "Canonical restart open-marker count exceeds its bound.");
    }
    state.collision.open_marker.resize(static_cast<std::size_t>(open_count));
    for (WeightedPICMarker& marker : state.collision.open_marker) {
        marker = canonical_restart_detail::read_marker(input);
    }
    // Version 2 adds detached collision carriers, version 3 adds polynomial source
    // coefficients, and version 4 adds signed collision increments. Each branch consumes
    // exactly its version-defined payload.
    if (version >= 2) {
        const std::uint64_t detached_count =
            canonical_restart_detail::read_scalar<std::uint64_t>(input);
        if (detached_count > UINT64_C(1000000000)) {
            throw std::runtime_error(
                "Canonical restart detached-marker count exceeds its bound.");
        }
        state.collision.detached_marker.resize(
            static_cast<std::size_t>(detached_count));
        for (WeightedPICMarker& marker : state.collision.detached_marker) {
            marker = canonical_restart_detail::read_marker(input);
        }
    }
    if (version >= 4) {
        auto read_markers = [&](std::vector<WeightedPICMarker>& marker,
                                const char* label) {
            const std::uint64_t count =
                canonical_restart_detail::read_scalar<std::uint64_t>(input);
            if (count > UINT64_C(1000000000)) {
                throw std::runtime_error(
                    std::string("Canonical restart ") + label
                    + " count exceeds its bound.");
            }
            marker.resize(static_cast<std::size_t>(count));
            for (WeightedPICMarker& value : marker) {
                value = canonical_restart_detail::read_marker(input);
            }
        };
        read_markers(
            state.collision.signed_increment_positive,
            "signed-positive marker");
        read_markers(
            state.collision.signed_increment_negative,
            "signed-negative marker");
    }
    state.field = canonical_restart_detail::read_field(input, version);
    state.poisson_reference = canonical_restart_detail::read_field(
        input, version);
    state.eulerian_f = canonical_restart_detail::read_vector<double>(input);
    state.atlas = canonical_restart_detail::read_atlas(input);
    // Validate finite forward event time/ages and positive initial represented
    // mass before returning the reconstructed state to the evolution controller.
    if (!(state.time >= 0.0) || !std::isfinite(state.time) || state.step < 0
        || !(state.potential_age >= 0.0)
        || !std::isfinite(state.potential_age)
        || !(state.collision_age >= 0.0)
        || !std::isfinite(state.collision_age)
        || !(state.initial_physical_mass > 0.0)
        || !std::isfinite(state.initial_physical_mass)) {
        throw std::runtime_error(
            "Canonical restart contains invalid event metadata.");
    }
    return state;
}

}  // namespace ev
