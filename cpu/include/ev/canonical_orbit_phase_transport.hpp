#pragma once

// Represent a spherical distribution by finite energy rows and periodic orbit
// phase cells, and transfer it conservatively to/from the (r,w,eta) collision
// grid. The atlas is this table of orbit rows and phase cells. A carrier is
// an explicit numerical mass element; a lift transfers a particle's mass and
// moments into target atlas cells. Inputs are cell geometry, a radial
// potential, atlas resolution and
// optional stationary f(H,eta).  Outputs include orbit geometry, sparse overlap
// volumes, phase-cell mass/energy moments and explicit hidden/escape ledgers.
// Here H=w^2/2+Phi(r)+eta/r^2, eta=L^2/2, and normalized phase advances by dt/T.
//
// A frozen potential preserves H and eta, so transport translates phase offsets.
// Collision products withdraw mass from its owned phase interval and change
// energy/angular momentum through conservative deposits.  A changing field
// requires a new orbit geometry and a separately checked distribution transfer.
// Sparse column/row indices connect these operations to the same represented
// phase volumes.  Reduced atlas masses acquire the factor 8*pi^2 in physical
// spherical phase-space integrals; eta quadrature weights are already included.

#include "ev/conservative_sldg_transport.hpp"
#include "ev/canonical_open_carrier_compression.hpp"
#include "ev/hamiltonian_orbit_phase.hpp"
#include "ev/weighted_vlasov_pic_event_scheduler.hpp"
#include "ev/weighted_vlasov_pic_collisions.hpp"
#include "ev/weighted_vlasov_pic_phase_remap.hpp"

#include <chrono>
#include <functional>
#include <map>

namespace ev {

// Row kind distinguishes closed radial cycles, cycles with hidden inner time,
// and trajectories that leave the represented radial or velocity domain.
enum class CanonicalOrbitRowKind {
    bound_periodic,
    delayed_inner_periodic,
    outer_outflow,
    velocity_outflow,
};

// One row spans an energy/action interval on an actual eta quadrature plane.
// phase_mass and its H/H^2 moments live in stored phase cells; phase_offset
// relates those cells to the current geometric orbit phase.
struct CanonicalOrbitRow {
    CanonicalOrbitRowKind kind = CanonicalOrbitRowKind::bound_periodic;
    int eta_cell = -1;
    int eta_plane = -1;
    double eta = 0.0;
    double eta_weight = 0.0;
    double action_left = 0.0;
    double action_right = 0.0;
    double energy_left = 0.0;
    double energy_right = 0.0;
    double energy = 0.0;
    double period = 0.0;
    double omega = 0.0;
    double phase_offset = 0.0;
    double hidden_phase_begin = 1.0;
    double phase_volume = 0.0;
    bool left_edge_is_inner_separatrix = false;
    bool right_edge_is_inner_separatrix = false;
    std::uint32_t first_cell = 0;
    std::vector<double> phase_mass;
    // First and second H moments travel with the phase cell.  They make a
    // collision lift energy-conservative even before the full P2(H,tau)
    // reconstruction is evaluated for projection.
    std::vector<double> phase_energy_mass;
    std::vector<double> phase_energy2_mass;
    HamiltonianOrbitPhaseGeometry phase_geometry;
};

/// Keeps the row's phase clock consistent with the representative orbit used
/// by its phase map.  The energy-integrated period controls phase-space volume;
/// the representative orbit's own period normalizes its trajectory.
inline void canonical_orbit_synchronize_row_phase_clock(
    CanonicalOrbitRow& row) {
    const double orbit_period = row.phase_geometry.radial_period;
    if (!(orbit_period > 0.0) || !std::isfinite(orbit_period)) {
        throw std::runtime_error(
            "Canonical orbit row has an invalid representative period.");
    }
    row.period = orbit_period;
    row.omega = 2.0 * PI / orbit_period;
    row.hidden_phase_begin = 1.0;
    if (row.kind != CanonicalOrbitRowKind::delayed_inner_periodic) {
        return;
    }
    const double visible_fraction =
        2.0 * row.phase_geometry.visible_outgoing_time / orbit_period;
    const double tolerance = 4096.0
        * std::numeric_limits<double>::epsilon();
    if (visible_fraction < -tolerance
        || visible_fraction > 1.0 + tolerance
        || !std::isfinite(visible_fraction)) {
        throw std::runtime_error(
            "Canonical delayed-return phase interval exceeds its orbit period.");
    }
    row.hidden_phase_begin = std::clamp(visible_fraction, 0.0, 1.0);
}

// A sparse entry integrates one canonical column inside one Eulerian cell,
// retaining the geometric moments needed by quadratic reconstruction.
struct CanonicalOverlap {
    std::uint32_t eulerian_cell = 0;
    std::uint32_t canonical_cell = 0;
    double volume = 0.0;
    // Geometric integrals of H and H^2 over this overlap.  They let a
    // canonical P2(H) density project without flattening an energy row.
    double energy_volume = 0.0;
    double energy2_volume = 0.0;
    // Integrals of x, y, x^2-1/12, xy, y^2-1/12 over the overlap,
    // where x and y are normalized to the Eulerian parent cell.
    std::array<double, 5> eulerian_p2_moment{};
};

// Partition a complete canonical phase volume into visible, hidden and escaped
// portions.  H^0 through H^4 moments support a quadratic density times H^0..H^2.
struct CanonicalColumnLedger {
    double column_volume = 0.0;
    double visible_volume = 0.0;
    double hidden_volume = 0.0;
    double escaped_volume = 0.0;
    double residual = 0.0;
    double geometric_scale = 1.0;
    double maximum_patch_scale_error = 0.0;
    double accumulated_patch_excess_error = 0.0;
    double canonical_quadrature_volume = 0.0;
    // Integral of H^q over the complete canonical column, q=0,...,4.
    std::array<double, 5> energy_volume_moment{};
    // The delayed central carrier and true outflow need their own H moments.
    // A P2(H) density can then be integrated over each destination without
    // replacing the energy distribution by its row-center value.
    std::array<double, 5> hidden_energy_volume_moment{};
    std::array<double, 5> escaped_energy_volume_moment{};
    bool partition_energy_moments_resolved = false;
};

struct CanonicalHiddenCellMoments {
    double mass = 0.0;
    double energy_mass = 0.0;
    double energy2_mass = 0.0;
};

// Signatures identify the field/mesh defining these sparse maps.  Eta-cell and
// actual-eta-plane offsets locate contiguous energy rows for collision lifts.
struct CanonicalOrbitAtlas {
    std::uint64_t generation = 0;
    std::uint64_t field_signature = 0;
    std::uint64_t geometry_signature = 0;
    std::vector<CanonicalOrbitRow> row;
    // Rows are contiguous within each eta cell.  These offsets make collision
    // lifts logarithmic in the number of energy rows by restricting each
    // accepted pair's search to its own invariant plane.
    std::vector<std::uint32_t> eta_row_offset;
    // A cell may contain several Gauss points in eta.  Rows are also
    // contiguous within each of these invariant planes.  Collision lifts and
    // action-preserving field rebuilds use this finer index.
    std::vector<std::uint32_t> eta_plane_row_offset;
    std::vector<int> eta_plane_cell;
    std::vector<double> eta_plane_coordinate;
    std::vector<CanonicalOverlap> overlap;
    std::vector<CanonicalColumnLedger> column;
};

// Reduced mass closes across projected visible cells and boundary partitions.
struct CanonicalProjectionLedger {
    double source_mass = 0.0;
    double visible_mass = 0.0;
    double hidden_mass = 0.0;
    double escaped_mass = 0.0;
    double destination_mass = 0.0;
    double residual = 0.0;
};

struct CanonicalStationaryHiddenCompletion {
    double visible_mass = 0.0;
    double hidden_mass_added = 0.0;
    std::size_t completed_rows = 0;
};

struct CanonicalStationaryPhaseInitialization {
    double mass_before = 0.0;
    double mass_after = 0.0;
    std::size_t initialized_rows = 0;
};

struct CanonicalPreservedTopologyTransfer {
    double mass_before = 0.0;
    double mass_after = 0.0;
    double residual = 0.0;
    double maximum_normalized_action_edge_shift = 0.0;
    std::size_t transferred_rows = 0;
    std::size_t changed_row_kinds = 0;
};

// Resolution and integration controls determine geometric map accuracy.
// Optional stationary functions set initial masses or place energy edges.
struct CanonicalAtlasBuildOptions {
    int phase_cells = 64;
    int maximum_patch_depth = 6;
    int mapped_boundary_segments = 16;
    unsigned int threads = 1;
    double mapped_volume_tolerance = 2.0e-3;
    double mapped_volume_excess_budget_fraction = 0.25;
    double overlap_closure_tolerance = 2.0e-12;
    double maximum_energy = std::numeric_limits<double>::infinity();
    // The exactly circular (or zero-amplitude radial) orbit at the minimum of
    // Phi_eff has coincident turning points.  Start the first finite-action
    // row this fraction of the represented energy span above that endpoint.
    // Convergence of the represented mass requires reducing this numerical
    // endpoint guard, especially when sampling a cusp.
    double minimum_energy_guard_fraction = 1.0e-6;
    double energy_action_fraction_power = 1.0;
    bool direct_tensor_quadrature = false;
    int mapping_energy_points = 8;
    int mapping_phase_points = 8;
    int energy_moment_panels = 16;
    int eta_quadrature_points = 1;
    // Optional normalized radial-action edges for each eta plane.  Potential
    // rebuilds use these to retain the current canonical grid while allowing
    // its energy edges to move with Phi(r,t).
    std::vector<std::vector<double>> action_fraction_edge_by_eta;
    // Absolute radial-action edges used by an adiabatic field rebuild.  The
    // physical outer boundary remains fixed, so only internal edges are
    // inverted under the new potential; the last entry records the old extent.
    std::vector<std::vector<double>> action_edge_by_eta;
    // Optional positive weight used only to place energy edges.  A stationary
    // f(H,eta) can be supplied here so equal-weight rows follow phase-space
    // mass and give more resolution to populated energy ranges.
    std::function<double(double, double)> energy_partition_weight;
    // Optional analytic or tabulated stationary f(H,eta).  When supplied, the
    // atlas is initialized by integrating f times the radial period over each
    // energy row, including the delayed central phase from the outset.
    std::function<double(double, double)> stationary_distribution;
};

// Export normalized action edges per actual eta plane, combining rows split
// by the inner-boundary separatrix so rebuilding can preserve their topology.
inline std::vector<std::vector<double>>
canonical_orbit_action_fraction_edges(
    const CanonicalOrbitAtlas& atlas,
    int eta_cells) {
    if (eta_cells < 1
        || atlas.eta_row_offset.size()
            != static_cast<std::size_t>(eta_cells + 1)) {
        throw std::runtime_error(
            "Canonical atlas has no valid eta-row index for action edges.");
    }
    const bool has_plane_index = !atlas.eta_plane_row_offset.empty();
    if (has_plane_index
        && (atlas.eta_plane_row_offset.size()
                != atlas.eta_plane_coordinate.size() + 1
            || atlas.eta_plane_cell.size()
                != atlas.eta_plane_coordinate.size())) {
        throw std::runtime_error(
            "Canonical atlas has an inconsistent eta-plane index.");
    }
    const std::size_t plane_count = has_plane_index
        ? atlas.eta_plane_coordinate.size()
        : static_cast<std::size_t>(eta_cells);
    std::vector<std::vector<double>> result(plane_count);
    for (std::size_t plane = 0; plane < plane_count; ++plane) {
        const std::size_t begin = has_plane_index
            ? atlas.eta_plane_row_offset[plane]
            : atlas.eta_row_offset[plane];
        const std::size_t end = has_plane_index
            ? atlas.eta_plane_row_offset[plane + 1]
            : atlas.eta_row_offset[plane + 1];
        if (begin >= end) {
            continue;
        }
        const double total_action = atlas.row[end - 1].action_right;
        if (!(total_action > 0.0) || !std::isfinite(total_action)) {
            throw std::runtime_error(
                "Canonical eta plane has invalid radial-action extent.");
        }
        auto& edge = result[plane];
        edge.reserve(end - begin + 1);
        edge.push_back(0.0);
        for (std::size_t row = begin; row < end; ++row) {
            if (atlas.row[row].right_edge_is_inner_separatrix
                && row + 1 < end) {
                continue;
            }
            edge.push_back(std::clamp(
                atlas.row[row].action_right / total_action, 0.0, 1.0));
        }
        edge.back() = 1.0;
        for (std::size_t q = 1; q < edge.size(); ++q) {
            if (!(edge[q] > edge[q - 1])) {
                throw std::runtime_error(
                    "Canonical radial-action edges are not strictly increasing.");
            }
        }
    }
    return result;
}

// Recover absolute radial action from the normalized per-plane edges.
inline std::vector<std::vector<double>> canonical_orbit_action_edges(
    const CanonicalOrbitAtlas& atlas,
    int eta_cells) {
    std::vector<std::vector<double>> result =
        canonical_orbit_action_fraction_edges(atlas, eta_cells);
    const bool has_plane_index = !atlas.eta_plane_row_offset.empty();
    for (std::size_t plane = 0; plane < result.size(); ++plane) {
        auto& edge = result[plane];
        if (edge.empty()) {
            continue;
        }
        const std::size_t end = has_plane_index
            ? atlas.eta_plane_row_offset[plane + 1]
            : atlas.eta_row_offset[plane + 1];
        const double total_action = atlas.row[end - 1].action_right;
        for (double& value : edge) {
            value *= total_action;
        }
        edge.front() = 0.0;
        edge.back() = total_action;
    }
    return result;
}

struct CanonicalOrbitPhaseState {
    CanonicalOrbitAtlas atlas;
    WeightedPICEventClockState clock;
    DelayedInnerCarrier migration_carrier;
    double visible_mass = 0.0;
    double hidden_mass = 0.0;
    double escaped_mass = 0.0;
};

// Assign exact upper-face coordinates to the final adjacent cell and report
// outside coordinates explicitly to the boundary/escape caller.
inline int canonical_orbit_containing_cell(
    double coordinate,
    const std::vector<double>& edge) {
    if (edge.size() < 2 || coordinate < edge.front()
        || coordinate > edge.back()) {
        return -1;
    }
    if (coordinate == edge.back()) {
        return static_cast<int>(edge.size()) - 2;
    }
    const auto upper = std::upper_bound(edge.begin(), edge.end(), coordinate);
    return std::clamp(
        static_cast<int>(upper - edge.begin()) - 1,
        0,
        static_cast<int>(edge.size()) - 2);
}

// Convert physical overlap moments to the Eulerian degree-two polynomial
// basis whose nonconstant terms each have zero cell average,
// including the supplied eta width in the reduced phase-space volume.
inline std::array<double, 5> canonical_overlap_p2_moments(
    const SLDGExtendedPolygonMoments& physical,
    double center_r,
    double center_w,
    double width_r,
    double width_w,
    double eta_width) {
    if (!(physical.area > 0.0L) || !(width_r > 0.0) || !(width_w > 0.0)) {
        return {};
    }
    const long double area = physical.area * eta_width;
    const long double first_r = area * physical.centroid_r;
    const long double first_w = area * physical.centroid_w;
    const long double normalized_first_r =
        (first_r - center_r * area) / width_r;
    const long double normalized_first_w =
        (first_w - center_w * area) / width_w;
    const long double normalized_second_rr = (
        eta_width * physical.second_rr - 2.0L * center_r * first_r
        + center_r * center_r * area) / (width_r * width_r);
    const long double normalized_second_rw = (
        eta_width * physical.second_rw
        - center_r * first_w - center_w * first_r
        + center_r * center_w * area) / (width_r * width_w);
    const long double normalized_second_ww = (
        eta_width * physical.second_ww - 2.0L * center_w * first_w
        + center_w * center_w * area) / (width_w * width_w);
    return {
        static_cast<double>(normalized_first_r),
        static_cast<double>(normalized_first_w),
        static_cast<double>(normalized_second_rr - area / 12.0L),
        static_cast<double>(normalized_second_rw),
        static_cast<double>(normalized_second_ww - area / 12.0L),
    };
}

/// Clips one visible mapped canonical patch against the Eulerian collision
/// grid.  The sum of returned overlap volumes must equal the patch area.
inline CanonicalColumnLedger canonical_orbit_append_visible_patch_overlaps(
    const SLDGPolygon& mapped_patch,
    std::uint32_t canonical_cell,
    int eta_cell,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    std::vector<CanonicalOverlap>& overlap,
    double relative_tolerance = 2.0e-12) {
    if (eta_cell < 0 || eta_cell >= p.NL || !(relative_tolerance >= 0.0)
        || !std::isfinite(relative_tolerance)) {
        throw std::runtime_error("Invalid canonical overlap request.");
    }
    const SLDGExtendedClipContext context(mapped_patch);
    if (!context.valid()) {
        throw std::runtime_error("Canonical mapped patch is degenerate.");
    }
    double minimum_r = mapped_patch.front().r;
    double maximum_r = minimum_r;
    double minimum_w = mapped_patch.front().w;
    double maximum_w = minimum_w;
    for (const SLDGPoint& point : mapped_patch) {
        minimum_r = std::min(minimum_r, point.r);
        maximum_r = std::max(maximum_r, point.r);
        minimum_w = std::min(minimum_w, point.w);
        maximum_w = std::max(maximum_w, point.w);
    }
    const int i_begin = canonical_orbit_containing_cell(
        std::max(minimum_r, geometry.radial.edge.front()),
        geometry.radial.edge);
    const int i_end = canonical_orbit_containing_cell(
        std::min(maximum_r, geometry.radial.edge.back()),
        geometry.radial.edge);
    const int j_begin = canonical_orbit_containing_cell(
        std::max(minimum_w, geometry.radial_velocity.edge.front()),
        geometry.radial_velocity.edge);
    const int j_end = canonical_orbit_containing_cell(
        std::min(maximum_w, geometry.radial_velocity.edge.back()),
        geometry.radial_velocity.edge);
    if (i_begin < 0 || i_end < i_begin || j_begin < 0 || j_end < j_begin) {
        throw std::runtime_error(
            "Canonical visible patch lies outside the Eulerian collision grid.");
    }

    CanonicalColumnLedger ledger;
    const double eta_width =
        geometry.eta.width[static_cast<std::size_t>(eta_cell)];
    ledger.column_volume =
        static_cast<double>(context.physical_area()) * eta_width;
    const std::size_t first_overlap = overlap.size();
    long double visible = 0.0L;
    for (int i = i_begin; i <= i_end; ++i) {
        for (int j = j_begin; j <= j_end; ++j) {
            const SLDGExtendedPolygonMoments moments = context.rectangle_moments(
                geometry.radial.edge[static_cast<std::size_t>(i)],
                geometry.radial.edge[static_cast<std::size_t>(i + 1)],
                geometry.radial_velocity.edge[static_cast<std::size_t>(j)],
                geometry.radial_velocity.edge[static_cast<std::size_t>(j + 1)]);
            if (!(moments.area > 0.0L)) {
                continue;
            }
            CanonicalOverlap entry;
            entry.eulerian_cell = static_cast<std::uint32_t>(
                idx(i, j, eta_cell, p));
            entry.canonical_cell = canonical_cell;
            entry.volume = static_cast<double>(moments.area) * eta_width;
            entry.eulerian_p2_moment = canonical_overlap_p2_moments(
                moments,
                geometry.radial.center[static_cast<std::size_t>(i)],
                geometry.radial_velocity.center[static_cast<std::size_t>(j)],
                geometry.radial.width[static_cast<std::size_t>(i)],
                geometry.radial_velocity.width[static_cast<std::size_t>(j)],
                eta_width);
            overlap.push_back(entry);
            visible += moments.area * eta_width;
        }
    }
    ledger.visible_volume = static_cast<double>(visible);
    ledger.residual = ledger.column_volume - ledger.visible_volume;
    const double scale = std::max(1.0, ledger.column_volume);
    if (std::abs(ledger.residual) > relative_tolerance * scale) {
        overlap.resize(first_overlap);
        std::ostringstream message;
        message << "Canonical visible-column overlap failed closure: volume="
                << std::setprecision(17) << ledger.column_volume
                << ", visible=" << ledger.visible_volume
                << ", residual=" << ledger.residual
                << ", bounds_r=[" << minimum_r << ',' << maximum_r << ']'
                << ", bounds_w=[" << minimum_w << ',' << maximum_w << ']'
                << ", domain_r=[" << geometry.radial.edge.front() << ','
                << geometry.radial.edge.back() << ']'
                << ", domain_w=[" << geometry.radial_velocity.edge.front() << ','
                << geometry.radial_velocity.edge.back() << ']'
                << ", tolerance=" << relative_tolerance * scale << '.';
        throw std::runtime_error(message.str());
    }
    return ledger;
}

// Integrate an Eulerian cell average plus optional P2 modes over this overlap.
// A positive reconstruction admits only finite nonnegative overlap mass.
inline double canonical_overlap_integral(
    const CanonicalOverlap& overlap,
    const std::vector<double>& f,
    const std::vector<std::array<double, 5>>* mode = nullptr) {
    if (overlap.eulerian_cell >= f.size()
        || (mode != nullptr && overlap.eulerian_cell >= mode->size())) {
        throw std::runtime_error("Canonical overlap references an invalid Eulerian cell.");
    }
    long double integral = static_cast<long double>(
        f[overlap.eulerian_cell]) * overlap.volume;
    if (mode != nullptr) {
        for (std::size_t q = 0; q < overlap.eulerian_p2_moment.size(); ++q) {
            integral += static_cast<long double>(
                (*mode)[overlap.eulerian_cell][q])
                * overlap.eulerian_p2_moment[q];
        }
    }
    const double scale = std::max(
        1.0, std::abs(f[overlap.eulerian_cell] * overlap.volume));
    if (integral < 0.0L
        && std::abs(integral)
            <= 512.0L * std::numeric_limits<double>::epsilon() * scale) {
        integral = 0.0L;
    }
    if (!(integral >= 0.0L) || !std::isfinite(static_cast<double>(integral))) {
        throw std::runtime_error(
            "A positive Eulerian reconstruction produced a negative canonical overlap mass.");
    }
    return static_cast<double>(integral);
}

// Frozen-field evolution changes only the periodic phase offset.  The stored
// masses and energy moments remain attached to their original phase intervals.
inline void canonical_orbit_advance_frozen(
    CanonicalOrbitAtlas& atlas,
    double dt) {
    if (!std::isfinite(dt)) {
        throw std::runtime_error("Canonical frozen-potential interval must be finite.");
    }
    for (CanonicalOrbitRow& row : atlas.row) {
        if (!(row.period > 0.0) || !std::isfinite(row.period)) {
            throw std::runtime_error("Canonical orbit row has an invalid period.");
        }
        row.phase_offset = hamiltonian_orbit_wrap_phase(
            row.phase_offset + dt / row.period);
    }
}

struct CanonicalPhaseCellMoments {
    double mass = 0.0;
    double energy_mass = 0.0;
    double energy2_mass = 0.0;
};

// Fit periodic neighboring phase-cell averages with a mean-preserving
// quadratic.  Optional common slope/curvature scaling keeps its full-cell
// minimum nonnegative, including an interior quadratic vertex.
inline std::array<double, 3> canonical_phase_p2_coefficients(
    const std::vector<double>& cell_average,
    std::size_t cell,
    bool preserve_positivity) {
    if (cell_average.empty() || cell >= cell_average.size()) {
        throw std::runtime_error("Invalid canonical phase P2 request.");
    }
    const std::size_t count = cell_average.size();
    const double left = cell_average[(cell + count - 1) % count];
    const double center = cell_average[cell];
    const double right = cell_average[(cell + 1) % count];
    if (!std::isfinite(left) || !std::isfinite(center)
        || !std::isfinite(right)
        || (preserve_positivity
            && (!(left >= 0.0) || !(center >= 0.0) || !(right >= 0.0)))) {
        throw std::runtime_error(
            "Canonical phase P2 reconstruction received invalid cell averages.");
    }
    double slope = 0.5 * (right - left);
    double curvature = 0.5 * (right - 2.0 * center + left);
    if (preserve_positivity && center > 0.0) {
        const auto value = [&](double coordinate) {
            return center + slope * coordinate
                + curvature * (coordinate * coordinate - 1.0 / 12.0);
        };
        double minimum = std::min(value(-0.5), value(0.5));
        if (curvature > 0.0) {
            const double vertex = -0.5 * slope / curvature;
            if (vertex > -0.5 && vertex < 0.5) {
                minimum = std::min(minimum, value(vertex));
            }
        }
        if (minimum < 0.0) {
            const double theta = std::clamp(
                center / (center - minimum), 0.0, 1.0)
                * (1.0 - 64.0 * std::numeric_limits<double>::epsilon());
            slope *= theta;
            curvature *= theta;
        }
    } else if (preserve_positivity) {
        slope = 0.0;
        curvature = 0.0;
    }
    return {center, slope, curvature};
}

// Analytically integrate the phase polynomial over part of one stored cell.
inline double canonical_phase_p2_integral(
    const std::array<double, 3>& coefficient,
    double left,
    double right) {
    if (!(right >= left) || left < -0.5 || right > 0.5) {
        throw std::runtime_error("Invalid canonical phase P2 integral.");
    }
    const double width = right - left;
    return coefficient[0] * width
        + 0.5 * coefficient[1] * (right * right - left * left)
        + coefficient[2] * (
            (right * right * right - left * left * left) / 3.0
            - width / 12.0);
}

// Integrate a stored row quantity over the shifted geometric phase cell,
// splitting at each stored-cell edge and the periodic seam.
inline double canonical_orbit_geometric_cell_quantity(
    const CanonicalOrbitRow& row,
    const std::vector<double>& quantity,
    std::size_t phase_cell) {
    if (row.phase_mass.empty() || quantity.size() != row.phase_mass.size()
        || phase_cell >= row.phase_mass.size()
        || !(row.phase_volume > 0.0)) {
        throw std::runtime_error("Invalid canonical phase-quantity row.");
    }
    const double begin = static_cast<double>(phase_cell)
        / static_cast<double>(row.phase_mass.size());
    const double end = static_cast<double>(phase_cell + 1)
        / static_cast<double>(row.phase_mass.size());
    double source_phase = hamiltonian_orbit_wrap_phase(
        begin - row.phase_offset);
    double remaining = end - begin;
    long double integral = 0.0L;
    const double cells = static_cast<double>(quantity.size());
    while (remaining > 0.0) {
        int source_cell = static_cast<int>(std::floor(source_phase * cells));
        source_cell = std::clamp(
            source_cell, 0, static_cast<int>(quantity.size()) - 1);
        const double source_right =
            static_cast<double>(source_cell + 1) / cells;
        double width = std::min(remaining, source_right - source_phase);
        if (!(width > 0.0)) {
            source_phase = 0.0;
            continue;
        }
        const std::array<double, 3> coefficient =
            canonical_phase_p2_coefficients(
                quantity, static_cast<std::size_t>(source_cell), false);
        const double local_left = source_phase * cells
            - static_cast<double>(source_cell) - 0.5;
        const double local_right = local_left + width * cells;
        integral += canonical_phase_p2_integral(
            coefficient,
            std::clamp(local_left, -0.5, 0.5),
            std::clamp(local_right, -0.5, 0.5));
        remaining -= width;
        source_phase += width;
        if (source_phase >= 1.0
            - 8.0 * std::numeric_limits<double>::epsilon()) {
            source_phase = 0.0;
        }
    }
    return static_cast<double>(integral);
}

// Transport positive phase mass with its limited polynomial.  Each piece
// carries its source cell's mean H and H^2, closing all three row moments.
inline CanonicalPhaseCellMoments canonical_orbit_geometric_cell_moments(
    const CanonicalOrbitRow& row,
    std::size_t phase_cell) {
    if (row.phase_mass.empty()
        || row.phase_energy_mass.size() != row.phase_mass.size()
        || row.phase_energy2_mass.size() != row.phase_mass.size()
        || phase_cell >= row.phase_mass.size()) {
        throw std::runtime_error(
            "Invalid canonical phase-moment row.");
    }
    const double cells = static_cast<double>(row.phase_mass.size());
    const double begin = static_cast<double>(phase_cell) / cells;
    const double end = static_cast<double>(phase_cell + 1) / cells;
    double source_phase = hamiltonian_orbit_wrap_phase(
        begin - row.phase_offset);
    double remaining = end - begin;
    std::array<long double, 3> result{};
    while (remaining > 0.0) {
        int source_cell = static_cast<int>(std::floor(source_phase * cells));
        source_cell = std::clamp(
            source_cell, 0, static_cast<int>(row.phase_mass.size()) - 1);
        const double source_right =
            static_cast<double>(source_cell + 1) / cells;
        const double width = std::min(remaining, source_right - source_phase);
        if (!(width > 0.0)) {
            source_phase = 0.0;
            continue;
        }
        const std::size_t source = static_cast<std::size_t>(source_cell);
        const std::array<double, 3> mass_coefficient =
            canonical_phase_p2_coefficients(row.phase_mass, source, true);
        const double local_left = source_phase * cells
            - static_cast<double>(source_cell) - 0.5;
        const double local_right = local_left + width * cells;
        double mass = canonical_phase_p2_integral(
            mass_coefficient,
            std::clamp(local_left, -0.5, 0.5),
            std::clamp(local_right, -0.5, 0.5));
        const double mass_scale = std::max(1.0, row.phase_mass[source]);
        if (mass < 0.0
            && std::abs(mass) <= 4096.0
                * std::numeric_limits<double>::epsilon() * mass_scale) {
            mass = 0.0;
        }
        if (!(mass >= 0.0) || !std::isfinite(mass)) {
            throw std::runtime_error(
                "Canonical positivity-only phase P2 remap produced negative mass.");
        }
        result[0] += mass;
        if (row.phase_mass[source] > 0.0) {
            result[1] += mass * row.phase_energy_mass[source]
                / row.phase_mass[source];
            result[2] += mass * row.phase_energy2_mass[source]
                / row.phase_mass[source];
        }
        remaining -= width;
        source_phase += width;
        if (source_phase >= 1.0
            - 8.0 * std::numeric_limits<double>::epsilon()) {
            source_phase = 0.0;
        }
    }
    return {
        static_cast<double>(result[0]),
        static_cast<double>(result[1]),
        static_cast<double>(result[2]),
    };
}

inline double canonical_orbit_geometric_cell_mass(
    const CanonicalOrbitRow& row,
    std::size_t phase_cell) {
    return canonical_orbit_geometric_cell_moments(row, phase_cell).mass;
}

struct CanonicalEnergyP2Reconstruction {
    double center = 0.0;
    double scale = 1.0;
    std::array<double, 3> coefficient{};
    double positivity_theta = 1.0;
};

// Reconstruct a quadratic density in normalized H from mass, H and H^2.
// Geometric moments form the 3x3 Gram system; singular rows use their mean.
// Positivity scales toward that mean and can reduce higher-moment detail.
inline CanonicalEnergyP2Reconstruction canonical_energy_p2_reconstruction(
    const CanonicalOrbitRow& row,
    double phase_mass,
    double phase_energy_mass,
    double phase_energy2_mass,
    const CanonicalColumnLedger& column) {
    if (!(phase_mass >= 0.0) || !std::isfinite(phase_mass)
        || !std::isfinite(phase_energy_mass)
        || !std::isfinite(phase_energy2_mass)
        || !(row.energy_right > row.energy_left)
        || !(column.energy_volume_moment[0] > 0.0)) {
        throw std::runtime_error("Invalid canonical P2 energy reconstruction.");
    }
    CanonicalEnergyP2Reconstruction result;
    result.center = 0.5 * (row.energy_left + row.energy_right);
    result.scale = 0.5 * (row.energy_right - row.energy_left);
    std::array<long double, 5> normalized_moment{};
    constexpr long double binomial[5][5] = {
        {1.0L, 0.0L, 0.0L, 0.0L, 0.0L},
        {1.0L, 1.0L, 0.0L, 0.0L, 0.0L},
        {1.0L, 2.0L, 1.0L, 0.0L, 0.0L},
        {1.0L, 3.0L, 3.0L, 1.0L, 0.0L},
        {1.0L, 4.0L, 6.0L, 4.0L, 1.0L},
    };
    for (int power = 0; power <= 4; ++power) {
        long double value = 0.0L;
        for (int h_power = 0; h_power <= power; ++h_power) {
            value += binomial[power][h_power]
                * std::pow(-static_cast<long double>(result.center),
                           power - h_power)
                * column.energy_volume_moment[static_cast<std::size_t>(h_power)];
        }
        normalized_moment[static_cast<std::size_t>(power)] = value
            / std::pow(static_cast<long double>(result.scale), power);
    }
    std::array<std::array<long double, 4>, 3> system{};
    const long double mass = phase_mass;
    const long double energy_mass = phase_energy_mass;
    const long double energy2_mass = phase_energy2_mass;
    const std::array<long double, 3> right_hand_side{
        mass,
        (energy_mass - result.center * mass) / result.scale,
        (energy2_mass - 2.0L * result.center * energy_mass
         + result.center * result.center * mass)
            / (result.scale * result.scale),
    };
    for (int equation = 0; equation < 3; ++equation) {
        for (int mode = 0; mode < 3; ++mode) {
            system[equation][mode] = normalized_moment[
                static_cast<std::size_t>(equation + mode)];
        }
        system[equation][3] = right_hand_side[
            static_cast<std::size_t>(equation)];
    }
    for (int pivot = 0; pivot < 3; ++pivot) {
        int best = pivot;
        for (int row_index = pivot + 1; row_index < 3; ++row_index) {
            if (std::abs(system[row_index][pivot])
                > std::abs(system[best][pivot])) {
                best = row_index;
            }
        }
        if (!(std::abs(system[best][pivot])
              > 1024.0L * std::numeric_limits<long double>::epsilon()
                    * normalized_moment[0])) {
            result.coefficient[0] = static_cast<double>(
                mass / normalized_moment[0]);
            return result;
        }
        if (best != pivot) {
            std::swap(system[best], system[pivot]);
        }
        const long double divisor = system[pivot][pivot];
        for (int column_index = pivot; column_index < 4; ++column_index) {
            system[pivot][column_index] /= divisor;
        }
        for (int row_index = 0; row_index < 3; ++row_index) {
            if (row_index == pivot) {
                continue;
            }
            const long double factor = system[row_index][pivot];
            for (int column_index = pivot; column_index < 4; ++column_index) {
                system[row_index][column_index] -=
                    factor * system[pivot][column_index];
            }
        }
    }
    for (int mode = 0; mode < 3; ++mode) {
        result.coefficient[static_cast<std::size_t>(mode)] =
            static_cast<double>(system[mode][3]);
    }
    const auto value = [&](double x) {
        return result.coefficient[0]
            + x * (result.coefficient[1] + x * result.coefficient[2]);
    };
    double minimum = std::min(value(-1.0), value(1.0));
    if (result.coefficient[2] > 0.0) {
        const double vertex = -0.5 * result.coefficient[1]
            / result.coefficient[2];
        if (vertex > -1.0 && vertex < 1.0) {
            minimum = std::min(minimum, value(vertex));
        }
    }
    if (minimum < 0.0) {
        const double mean = static_cast<double>(
            mass / normalized_moment[0]);
        result.positivity_theta = std::clamp(
            mean / (mean - minimum), 0.0, 1.0);
        result.positivity_theta *=
            1.0 - 64.0 * std::numeric_limits<double>::epsilon();
        result.coefficient[0] = mean + result.positivity_theta
            * (result.coefficient[0] - mean);
        result.coefficient[1] *= result.positivity_theta;
        result.coefficient[2] *= result.positivity_theta;
    }
    return result;
}

// Integrate f(H)*H^q over a hidden or escaped geometric partition.  The
// degree-two density requires moments through H^(q+2), with q=0,1,2.
inline double canonical_energy_p2_partition_moment(
    const CanonicalEnergyP2Reconstruction& mode,
    const std::array<double, 5>& energy_volume_moment,
    int moment_power) {
    if (moment_power < 0 || moment_power > 2) {
        throw std::runtime_error(
            "Canonical P2 partition moment power is out of range.");
    }
    for (double value : energy_volume_moment) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Invalid canonical P2 partition moments.");
        }
    }
    const double volume = energy_volume_moment[0];
    if (!(volume >= 0.0)) {
        throw std::runtime_error("Invalid canonical P2 partition moments.");
    }
    if (volume == 0.0) {
        return 0.0;
    }
    const std::size_t power = static_cast<std::size_t>(moment_power);
    const double base = energy_volume_moment[power];
    const double x_volume = (
        energy_volume_moment[power + 1] - mode.center * base)
        / mode.scale;
    const double x2_volume = (
        energy_volume_moment[power + 2]
        - 2.0 * mode.center * energy_volume_moment[power + 1]
        + mode.center * mode.center * base)
        / (mode.scale * mode.scale);
    double result = mode.coefficient[0] * base
        + mode.coefficient[1] * x_volume
        + mode.coefficient[2] * x2_volume;
    const double scale = std::max(
        1.0,
        std::abs(mode.coefficient[0] * base)
            + std::abs(mode.coefficient[1] * x_volume)
            + std::abs(mode.coefficient[2] * x2_volume));
    if (moment_power == 0 && result < 0.0
        && std::abs(result)
            <= 4096.0 * std::numeric_limits<double>::epsilon() * scale) {
        result = 0.0;
    }
    if ((moment_power == 0 && !(result >= 0.0)) || !std::isfinite(result)) {
        throw std::runtime_error(
            "Canonical P2 reconstruction produced negative partition mass.");
    }
    return result;
}

// First determine each column's hidden/escaped mass from its H distribution.
// Positive visible overlap contributions are then normalized to that column
// budget and divided by Eulerian cell volume to obtain f.
inline CanonicalProjectionLedger canonical_orbit_project_to_eulerian(
    const CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    std::vector<double>& f,
    std::vector<CanonicalHiddenCellMoments>* hidden_moments_by_cell = nullptr) {
    const std::size_t cell_count = static_cast<std::size_t>(p.Nr)
        * p.Nw * p.NL;
    f.assign(cell_count, 0.0);
    std::vector<long double> mass(cell_count, 0.0L);
    std::vector<double> canonical_mass(atlas.column.size(), 0.0);
    std::vector<double> target_visible_mass(atlas.column.size(), 0.0);
    std::vector<double> raw_visible_mass(atlas.column.size(), 0.0);
    std::vector<CanonicalEnergyP2Reconstruction> reconstruction(
        atlas.column.size());
    std::vector<double> overlap_contribution(atlas.overlap.size(), 0.0);
    if (hidden_moments_by_cell != nullptr) {
        hidden_moments_by_cell->assign(
            atlas.column.size(), CanonicalHiddenCellMoments{});
    }
    CanonicalProjectionLedger ledger;
    for (const CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t canonical_cell = row.first_cell + q;
            if (canonical_cell >= atlas.column.size()) {
                throw std::runtime_error("Canonical row references a missing column ledger.");
            }
            const CanonicalColumnLedger& column = atlas.column[canonical_cell];
            const CanonicalPhaseCellMoments phase_moments =
                canonical_orbit_geometric_cell_moments(row, q);
            const double cell_mass = phase_moments.mass;
            const double cell_energy_mass = phase_moments.energy_mass;
            const double cell_energy2_mass = phase_moments.energy2_mass;
            canonical_mass[canonical_cell] = cell_mass;
            ledger.source_mass += cell_mass;
            if (!(column.column_volume > 0.0)) {
                if (cell_mass != 0.0) {
                    throw std::runtime_error("A populated canonical cell has zero volume.");
                }
                continue;
            }
            reconstruction[canonical_cell] = canonical_energy_p2_reconstruction(
                row,
                cell_mass,
                cell_energy_mass,
                cell_energy2_mass,
                column);
            const CanonicalEnergyP2Reconstruction& energy_mode =
                reconstruction[canonical_cell];
            CanonicalHiddenCellMoments hidden_moments;
            CanonicalHiddenCellMoments escaped_moments;
            if (column.partition_energy_moments_resolved) {
                hidden_moments.mass = canonical_energy_p2_partition_moment(
                    energy_mode, column.hidden_energy_volume_moment, 0);
                hidden_moments.energy_mass =
                    canonical_energy_p2_partition_moment(
                        energy_mode, column.hidden_energy_volume_moment, 1);
                hidden_moments.energy2_mass =
                    canonical_energy_p2_partition_moment(
                        energy_mode, column.hidden_energy_volume_moment, 2);
                escaped_moments.mass = canonical_energy_p2_partition_moment(
                    energy_mode, column.escaped_energy_volume_moment, 0);
                escaped_moments.energy_mass =
                    canonical_energy_p2_partition_moment(
                        energy_mode, column.escaped_energy_volume_moment, 1);
                escaped_moments.energy2_mass =
                    canonical_energy_p2_partition_moment(
                        energy_mode, column.escaped_energy_volume_moment, 2);
            } else {
                const double hidden_fraction = column.hidden_volume
                    / column.column_volume;
                const double escaped_fraction = column.escaped_volume
                    / column.column_volume;
                hidden_moments = {
                    cell_mass * hidden_fraction,
                    cell_energy_mass * hidden_fraction,
                    cell_energy2_mass * hidden_fraction,
                };
                escaped_moments = {
                    cell_mass * escaped_fraction,
                    cell_energy_mass * escaped_fraction,
                    cell_energy2_mass * escaped_fraction,
                };
            }
            const double hidden_mass = hidden_moments.mass;
            const double escaped_mass = escaped_moments.mass;
            double visible_target = cell_mass - hidden_mass - escaped_mass;
            const double partition_scale = std::max(1.0, cell_mass);
            if (visible_target < 0.0
                && std::abs(visible_target)
                    <= 4096.0 * std::numeric_limits<double>::epsilon()
                        * partition_scale) {
                visible_target = 0.0;
            }
            if (!(visible_target >= 0.0) || !std::isfinite(visible_target)) {
                throw std::runtime_error(
                    "Canonical P2 hidden/outflow partition exceeds cell mass.");
            }
            target_visible_mass[canonical_cell] = visible_target;
            ledger.hidden_mass += hidden_mass;
            if (hidden_moments_by_cell != nullptr) {
                (*hidden_moments_by_cell)[canonical_cell] = hidden_moments;
            }
            ledger.escaped_mass += escaped_mass;
        }
    }
    for (std::size_t entry_index = 0;
         entry_index < atlas.overlap.size();
         ++entry_index) {
        const CanonicalOverlap& entry = atlas.overlap[entry_index];
        if (entry.canonical_cell >= atlas.column.size()
            || entry.eulerian_cell >= mass.size()) {
            throw std::runtime_error("Canonical overlap index is out of range.");
        }
        const CanonicalColumnLedger& column = atlas.column[entry.canonical_cell];
        if (!(column.column_volume > 0.0)) {
            throw std::runtime_error("Canonical overlap ownership is inconsistent.");
        }
        const CanonicalEnergyP2Reconstruction& mode =
            reconstruction[entry.canonical_cell];
        const double raw_mean_x = (
            entry.energy_volume - mode.center * entry.volume)
            / (mode.scale * entry.volume);
        const double mean_x = std::clamp(raw_mean_x, -1.0, 1.0);
        const double raw_mean_x2 = (
            entry.energy2_volume
            - 2.0 * mode.center * entry.energy_volume
            + mode.center * mode.center * entry.volume)
            / (mode.scale * mode.scale * entry.volume);
        const double mean_x2 = std::clamp(
            raw_mean_x2, mean_x * mean_x, 1.0);
        const double x_volume = mean_x * entry.volume;
        const double x2_volume = mean_x2 * entry.volume;
        double contribution = mode.coefficient[0] * entry.volume
            + mode.coefficient[1] * x_volume
            + mode.coefficient[2] * x2_volume;
        const double contribution_scale = std::max(
            1.0,
            canonical_mass[entry.canonical_cell]
                * entry.volume / column.column_volume);
        if (contribution < 0.0
            && std::abs(contribution)
                <= 4096.0 * std::numeric_limits<double>::epsilon()
                    * contribution_scale) {
            contribution = 0.0;
        }
        if (!(contribution >= 0.0) || !std::isfinite(contribution)) {
            std::ostringstream message;
            message
                << "Canonical P2 projection produced a negative overlap mass: "
                << std::setprecision(17)
                << "contribution=" << contribution
                << ", canonical_cell=" << entry.canonical_cell
                << ", volume=" << entry.volume
                << ", mean_x=" << x_volume / entry.volume
                << ", mean_x2=" << x2_volume / entry.volume
                << ", coefficient=[" << mode.coefficient[0] << ','
                << mode.coefficient[1] << ',' << mode.coefficient[2] << ']'
                << ", theta=" << mode.positivity_theta << '.';
            throw std::runtime_error(message.str());
        }
        overlap_contribution[entry_index] = contribution;
        raw_visible_mass[entry.canonical_cell] += contribution;
    }
    for (std::size_t entry_index = 0;
         entry_index < atlas.overlap.size();
         ++entry_index) {
        const CanonicalOverlap& entry = atlas.overlap[entry_index];
        const double raw = raw_visible_mass[entry.canonical_cell];
        const double target = target_visible_mass[entry.canonical_cell];
        if (!(raw > 0.0)) {
            if (target != 0.0) {
                throw std::runtime_error(
                    "Canonical P2 projection has zero visible normalization.");
            }
            continue;
        }
        const double contribution = overlap_contribution[entry_index]
            * target / raw;
        mass[entry.eulerian_cell] += contribution;
        ledger.visible_mass += contribution;
    }
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = idx(i, j, k, p);
                const double volume = canonical_cell_measure(
                    geometry, i, j, k);
                f[cell] = static_cast<double>(mass[cell] / volume);
                ledger.destination_mass += static_cast<double>(mass[cell]);
            }
        }
    }
    ledger.residual = ledger.source_mass
        - ledger.visible_mass - ledger.hidden_mass - ledger.escaped_mass;
    return ledger;
}

// Integrate visible Eulerian overlaps into canonical mass and energy moments,
// add supplied hidden moments and reset the stored phase origin to geometry.
inline CanonicalProjectionLedger canonical_orbit_project_from_eulerian(
    CanonicalOrbitAtlas& atlas,
    const std::vector<double>& f,
    const std::vector<std::array<double, 5>>* mode = nullptr,
    const std::vector<CanonicalHiddenCellMoments>* hidden_moments = nullptr) {
    std::vector<long double> cell_mass(atlas.column.size(), 0.0L);
    std::vector<long double> cell_energy_mass(atlas.column.size(), 0.0L);
    std::vector<long double> cell_energy2_mass(atlas.column.size(), 0.0L);
    for (const CanonicalOverlap& entry : atlas.overlap) {
        const double contribution = canonical_overlap_integral(entry, f, mode);
        cell_mass[entry.canonical_cell] += contribution;
        if (!(entry.volume > 0.0)) {
            continue;
        }
        cell_energy_mass[entry.canonical_cell] += contribution
            * entry.energy_volume / entry.volume;
        cell_energy2_mass[entry.canonical_cell] += contribution
            * entry.energy2_volume / entry.volume;
    }
    CanonicalProjectionLedger ledger;
    for (CanonicalOrbitRow& row : atlas.row) {
        row.phase_offset = 0.0;
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t canonical_cell = row.first_cell + q;
            if (hidden_moments != nullptr) {
                if (hidden_moments->size() != atlas.column.size()) {
                    throw std::runtime_error("Canonical hidden-mass array has the wrong size.");
                }
                const CanonicalHiddenCellMoments& hidden =
                    (*hidden_moments)[canonical_cell];
                cell_mass[canonical_cell] += hidden.mass;
                cell_energy_mass[canonical_cell] +=
                    hidden.energy_mass;
                cell_energy2_mass[canonical_cell] +=
                    hidden.energy2_mass;
                ledger.hidden_mass += hidden.mass;
            }
            row.phase_mass[q] = static_cast<double>(cell_mass[canonical_cell]);
            row.phase_energy_mass[q] = static_cast<double>(
                cell_energy_mass[canonical_cell]);
            row.phase_energy2_mass[q] = static_cast<double>(
                cell_energy2_mass[canonical_cell]);
            ledger.source_mass += row.phase_mass[q];
        }
    }
    ledger.destination_mass = ledger.source_mass;
    ledger.visible_mass = ledger.source_mass - ledger.hidden_mass;
    ledger.residual = 0.0;
    return ledger;
}

// Refine a one-dimensional integral using the difference between whole and
// split Simpson estimates, sharing endpoints and halving the error budget.
template <class Integrand>
inline double canonical_adaptive_simpson_integral(
    double left,
    double right,
    Integrand&& integrand,
    double relative_tolerance = 1.0e-9,
    int maximum_depth = 18) {
    if (!(right > left) || !(relative_tolerance > 0.0)
        || !std::isfinite(relative_tolerance) || maximum_depth < 1) {
        throw std::runtime_error("Invalid canonical adaptive integral request.");
    }
    const double middle = 0.5 * (left + right);
    const double f_left = integrand(left);
    const double f_middle = integrand(middle);
    const double f_right = integrand(right);
    const double whole = (right - left)
        * (f_left + 4.0 * f_middle + f_right) / 6.0;
    const double absolute_tolerance = std::max(
        relative_tolerance * std::abs(whole),
        256.0 * std::numeric_limits<double>::epsilon()
            * (right - left)
            * std::max({std::abs(f_left), std::abs(f_middle),
                        std::abs(f_right)}));
    std::function<double(double, double, double, double, double,
                         double, double, int)> refine;
    refine = [&](double a,
                 double b,
                 double fa,
                 double fm,
                 double fb,
                 double estimate,
                 double tolerance,
                 int depth) {
        const double m = 0.5 * (a + b);
        const double lm = 0.5 * (a + m);
        const double rm = 0.5 * (m + b);
        const double flm = integrand(lm);
        const double frm = integrand(rm);
        const double left_estimate = (m - a)
            * (fa + 4.0 * flm + fm) / 6.0;
        const double right_estimate = (b - m)
            * (fm + 4.0 * frm + fb) / 6.0;
        const double improved = left_estimate + right_estimate;
        const double correction = improved - estimate;
        if (depth == 0 || std::abs(correction) <= 15.0 * tolerance) {
            return improved + correction / 15.0;
        }
        return refine(
                   a, m, fa, flm, fm, left_estimate,
                   0.5 * tolerance, depth - 1)
            + refine(
                   m, b, fm, frm, fb, right_estimate,
                   0.5 * tolerance, depth - 1);
    };
    return refine(
        left,
        right,
        f_left,
        f_middle,
        f_right,
        whole,
        absolute_tolerance,
        maximum_depth);
}

/// Completes the unresolved part of a stationary f(H,L) initialization.
/// At fixed H and L the phase density is constant, so the density measured on
/// the visible portion of one orbit row also fixes its delayed r<r_min part.
/// This routine is only for loading a stationary initial condition; evolved
/// states retain their explicitly transported hidden phase masses.
inline CanonicalStationaryHiddenCompletion
canonical_orbit_complete_stationary_hidden_phase(
    CanonicalOrbitAtlas& atlas) {
    CanonicalStationaryHiddenCompletion result;
    for (CanonicalOrbitRow& row : atlas.row) {
        long double visible_mass = 0.0L;
        long double visible_volume = 0.0L;
        long double hidden_volume = 0.0L;
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            if (cell >= atlas.column.size()) {
                throw std::runtime_error(
                    "Stationary hidden completion found a missing atlas column.");
            }
            visible_mass += row.phase_mass[q];
            visible_volume += atlas.column[cell].visible_volume;
            hidden_volume += atlas.column[cell].hidden_volume;
        }
        result.visible_mass += static_cast<double>(visible_mass);
        if (!(hidden_volume > 0.0L)) {
            continue;
        }
        if (!(visible_volume > 0.0L)) {
            if (visible_mass != 0.0L) {
                throw std::runtime_error(
                    "A populated stationary orbit row has no visible volume.");
            }
            continue;
        }
        const long double phase_density = visible_mass / visible_volume;
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            const double addition = static_cast<double>(
                phase_density * atlas.column[cell].hidden_volume);
            row.phase_mass[q] += addition;
            row.phase_energy_mass[q] += row.energy * addition;
            row.phase_energy2_mass[q] += row.energy * row.energy * addition;
            result.hidden_mass_added += addition;
        }
        ++result.completed_rows;
    }
    return result;
}

/// Initializes an analytic stationary f(H,eta) state on the orbit atlas.
/// This is called once after loading the initial condition and completing its
/// delayed central phase.  Every orbit row keeps its mass and receives the
/// phase-independent density required by a stationary distribution.  Evolved
/// collision and transport states retain their evolved phase structure.
inline CanonicalStationaryPhaseInitialization
canonical_orbit_initialize_stationary_phase(CanonicalOrbitAtlas& atlas) {
    CanonicalStationaryPhaseInitialization diagnostics;
    for (CanonicalOrbitRow& row : atlas.row) {
        if (row.phase_mass.empty()) {
            continue;
        }
        long double row_mass = 0.0L;
        long double row_energy_mass = 0.0L;
        long double row_energy2_mass = 0.0L;
        for (double mass : row.phase_mass) {
            if (!(mass >= 0.0) || !std::isfinite(mass)) {
                throw std::runtime_error(
                    "Stationary orbit initialization found invalid phase mass.");
            }
            row_mass += mass;
        }
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            if (!std::isfinite(row.phase_energy_mass[q])
                || !std::isfinite(row.phase_energy2_mass[q])) {
                throw std::runtime_error(
                    "Stationary orbit initialization found invalid energy moments.");
            }
            row_energy_mass += row.phase_energy_mass[q];
            row_energy2_mass += row.phase_energy2_mass[q];
        }
        const double equal_mass = static_cast<double>(
            row_mass / static_cast<long double>(row.phase_mass.size()));
        const double equal_energy_mass = static_cast<double>(
            row_energy_mass
            / static_cast<long double>(row.phase_mass.size()));
        const double equal_energy2_mass = static_cast<double>(
            row_energy2_mass
            / static_cast<long double>(row.phase_mass.size()));
        diagnostics.mass_before += static_cast<double>(row_mass);
        std::fill(row.phase_mass.begin(), row.phase_mass.end(), equal_mass);
        std::fill(
            row.phase_energy_mass.begin(),
            row.phase_energy_mass.end(),
            equal_energy_mass);
        std::fill(
            row.phase_energy2_mass.begin(),
            row.phase_energy2_mass.end(),
            equal_energy2_mass);
        diagnostics.mass_after += equal_mass
            * static_cast<double>(row.phase_mass.size());
        ++diagnostics.initialized_rows;
    }
    return diagnostics;
}

/// Transfers a canonical state between two atlases with the same eta, radial-
/// action, and phase-cell topology.  Mass and phase structure are copied
/// exactly.  The normalized within-row energy moments are carried to the new
/// energy interval by an affine H transformation.  Potential refreshes call
/// this after rebuilding orbit geometry; no Eulerian inverse remap is needed.
inline CanonicalPreservedTopologyTransfer
canonical_orbit_transfer_preserved_topology(
    const CanonicalOrbitAtlas& source,
    CanonicalOrbitAtlas& destination) {
    if (source.row.size() != destination.row.size()
        || source.column.size() != destination.column.size()
        || source.eta_row_offset != destination.eta_row_offset
        || source.eta_plane_row_offset != destination.eta_plane_row_offset
        || source.eta_plane_cell != destination.eta_plane_cell
        || source.eta_plane_coordinate != destination.eta_plane_coordinate) {
        throw std::runtime_error(
            "Canonical preserved-topology transfer found different atlas topology.");
    }
    CanonicalPreservedTopologyTransfer result;
    for (std::size_t row_index = 0; row_index < source.row.size(); ++row_index) {
        const CanonicalOrbitRow& old_row = source.row[row_index];
        CanonicalOrbitRow& new_row = destination.row[row_index];
        if (old_row.eta_cell != new_row.eta_cell
            || old_row.eta_plane != new_row.eta_plane
            || !phase_space_nearly_equal(old_row.eta, new_row.eta)
            || old_row.phase_mass.size() != new_row.phase_mass.size()
            || old_row.phase_energy_mass.size() != old_row.phase_mass.size()
            || old_row.phase_energy2_mass.size() != old_row.phase_mass.size()
            || new_row.phase_energy_mass.size() != new_row.phase_mass.size()
            || new_row.phase_energy2_mass.size() != new_row.phase_mass.size()) {
            throw std::runtime_error(
                "Canonical preserved-topology row contract changed.");
        }
        const double old_scale = 0.5
            * (old_row.energy_right - old_row.energy_left);
        const double old_center = 0.5
            * (old_row.energy_right + old_row.energy_left);
        const double new_scale = 0.5
            * (new_row.energy_right - new_row.energy_left);
        const double new_center = 0.5
            * (new_row.energy_right + new_row.energy_left);
        if (!(old_scale > 0.0) || !(new_scale > 0.0)) {
            throw std::runtime_error(
                "Canonical preserved-topology row has zero energy width.");
        }
        const double old_action_scale = std::max(
            old_row.action_right, std::numeric_limits<double>::min());
        result.maximum_normalized_action_edge_shift = std::max({
            result.maximum_normalized_action_edge_shift,
            std::abs(new_row.action_left - old_row.action_left)
                / old_action_scale,
            std::abs(new_row.action_right - old_row.action_right)
                / old_action_scale,
        });
        new_row.phase_offset = old_row.phase_offset;
        result.changed_row_kinds += old_row.kind != new_row.kind ? 1 : 0;
        for (std::size_t q = 0; q < old_row.phase_mass.size(); ++q) {
            const long double mass = old_row.phase_mass[q];
            const long double energy_mass = old_row.phase_energy_mass[q];
            const long double energy2_mass = old_row.phase_energy2_mass[q];
            if (!(mass >= 0.0L) || !std::isfinite(static_cast<double>(mass))
                || !std::isfinite(static_cast<double>(energy_mass))
                || !std::isfinite(static_cast<double>(energy2_mass))) {
                throw std::runtime_error(
                    "Canonical preserved-topology transfer found invalid moments.");
            }
            const long double normalized_energy_mass =
                (energy_mass - old_center * mass) / old_scale;
            const long double normalized_energy2_mass = (
                energy2_mass - 2.0L * old_center * energy_mass
                + old_center * old_center * mass)
                / (old_scale * old_scale);
            new_row.phase_mass[q] = static_cast<double>(mass);
            new_row.phase_energy_mass[q] = static_cast<double>(
                new_center * mass
                + new_scale * normalized_energy_mass);
            new_row.phase_energy2_mass[q] = static_cast<double>(
                new_center * new_center * mass
                + 2.0L * new_center * new_scale
                    * normalized_energy_mass
                + new_scale * new_scale * normalized_energy2_mass);
            result.mass_before += static_cast<double>(mass);
            result.mass_after += new_row.phase_mass[q];
        }
        ++result.transferred_rows;
    }
    result.residual = result.mass_before - result.mass_after;
    destination.generation = source.generation + 1;
    return result;
}

/// Rebuilds the atlas from an Eulerian state produced by an atlas projection.
/// Each Eulerian cell is a finite-volume mixing bin, so its complete cell mass
/// is shared among every canonical column that overlaps it.  Normalizing the
/// transpose by the summed overlap in that Eulerian cell makes this event
/// remap conservative.  Cells with no canonical support enter escaped_mass.
inline CanonicalProjectionLedger
canonical_orbit_project_from_eulerian_conservative(
    CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const std::vector<double>& f,
    const std::vector<std::array<double, 5>>* mode = nullptr,
    const std::vector<CanonicalHiddenCellMoments>* hidden_moments = nullptr) {
    const std::size_t eulerian_cells = static_cast<std::size_t>(p.Nr)
        * p.Nw * p.NL;
    validate_finite_nonnegative_distribution(
        f, eulerian_cells, "conservative canonical event remap");
    if (mode != nullptr && mode->size() != eulerian_cells) {
        throw std::runtime_error(
            "Canonical event-remap mode array has the wrong size.");
    }
    std::vector<long double> raw_overlap_mass(atlas.overlap.size(), 0.0L);
    std::vector<long double> raw_mass_by_eulerian(eulerian_cells, 0.0L);
    for (std::size_t q = 0; q < atlas.overlap.size(); ++q) {
        const CanonicalOverlap& entry = atlas.overlap[q];
        const long double contribution = canonical_overlap_integral(
            entry, f, mode);
        const long double tolerance = 512.0L
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(f[entry.eulerian_cell]))
            * entry.volume;
        if (contribution < -tolerance) {
            throw std::runtime_error(
                "Canonical event remap received a negative overlap integral.");
        }
        raw_overlap_mass[q] = std::max(0.0L, contribution);
        raw_mass_by_eulerian[entry.eulerian_cell] += raw_overlap_mass[q];
    }

    std::vector<long double> target_mass(eulerian_cells, 0.0L);
    CanonicalProjectionLedger ledger;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = idx(i, j, k, p);
                target_mass[cell] = static_cast<long double>(f[cell])
                    * canonical_cell_measure(geometry, i, j, k);
                ledger.source_mass += static_cast<double>(target_mass[cell]);
                if (target_mass[cell] > 0.0L
                    && !(raw_mass_by_eulerian[cell] > 0.0L)) {
                    ledger.escaped_mass += static_cast<double>(target_mass[cell]);
                }
            }
        }
    }

    std::vector<long double> canonical_mass(atlas.column.size(), 0.0L);
    std::vector<long double> canonical_energy_mass(
        atlas.column.size(), 0.0L);
    std::vector<long double> canonical_energy2_mass(
        atlas.column.size(), 0.0L);
    for (std::size_t q = 0; q < atlas.overlap.size(); ++q) {
        const CanonicalOverlap& entry = atlas.overlap[q];
        const long double raw_total =
            raw_mass_by_eulerian[entry.eulerian_cell];
        if (!(raw_total > 0.0L)) {
            continue;
        }
        const long double assigned_mass =
            target_mass[entry.eulerian_cell]
            * raw_overlap_mass[q] / raw_total;
        canonical_mass[entry.canonical_cell] += assigned_mass;
        if (!(entry.volume > 0.0)) {
            throw std::runtime_error(
                "Canonical event remap found a zero-volume overlap.");
        }
        canonical_energy_mass[entry.canonical_cell] += assigned_mass
            * entry.energy_volume / entry.volume;
        canonical_energy2_mass[entry.canonical_cell] += assigned_mass
            * entry.energy2_volume / entry.volume;
    }

    if (hidden_moments != nullptr
        && hidden_moments->size() != atlas.column.size()) {
        throw std::runtime_error(
            "Canonical hidden-mass array has the wrong size.");
    }
    for (CanonicalOrbitRow& row : atlas.row) {
        row.phase_offset = 0.0;
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t canonical_cell = row.first_cell + q;
            if (hidden_moments != nullptr) {
                const CanonicalHiddenCellMoments& hidden =
                    (*hidden_moments)[canonical_cell];
                canonical_mass[canonical_cell] += hidden.mass;
                canonical_energy_mass[canonical_cell] +=
                    hidden.energy_mass;
                canonical_energy2_mass[canonical_cell] +=
                    hidden.energy2_mass;
                ledger.hidden_mass += hidden.mass;
                ledger.source_mass += hidden.mass;
            }
            row.phase_mass[q] = static_cast<double>(
                canonical_mass[canonical_cell]);
            row.phase_energy_mass[q] = static_cast<double>(
                canonical_energy_mass[canonical_cell]);
            row.phase_energy2_mass[q] = static_cast<double>(
                canonical_energy2_mass[canonical_cell]);
            ledger.destination_mass += row.phase_mass[q];
        }
    }
    ledger.visible_mass = ledger.destination_mass - ledger.hidden_mass;
    ledger.residual = ledger.source_mass
        - ledger.destination_mass - ledger.escaped_mass;
    return ledger;
}

template <class EffectivePotential>
inline double canonical_effective_potential_minimum_radius(
    double left,
    double right,
    EffectivePotential&& effective_potential);

// Construct one finite energy row at fixed actual eta.  Integrating T(H)dH
// supplies its phase volume and radial-action width; a representative orbit
// supplies its phase clock.  Later overlap integration resolves the full row.
template <class EffectivePotential>
inline std::size_t canonical_orbit_append_bound_row(
    CanonicalOrbitAtlas& atlas,
    int eta_cell,
    double energy_left,
    double energy_right,
    const std::vector<double>& turning_scan_radius,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    EffectivePotential&& effective_potential,
    const CanonicalAtlasBuildOptions& options = {},
    const std::function<double(double)>& hidden_return_time = {},
    std::map<double, HamiltonianOrbitPhaseGeometry>* shared_orbit_cache = nullptr,
    std::map<std::pair<double, double>,
             HamiltonianOrbitPhaseGeometry::Location>* shared_location_cache = nullptr,
    std::map<double, double>* shared_period_cache = nullptr,
    double eta_override = std::numeric_limits<double>::quiet_NaN(),
    double eta_weight_override = std::numeric_limits<double>::quiet_NaN(),
    int eta_plane = -1) {
    if (eta_cell < 0 || eta_cell >= p.NL
        || !(energy_right > energy_left)
        || options.phase_cells < 4
        || options.maximum_patch_depth < 0
        || options.mapped_boundary_segments < 1
        || !(options.mapped_volume_tolerance > 0.0)
        || !(options.mapped_volume_excess_budget_fraction >= 0.0)
        || !std::isfinite(options.mapped_volume_excess_budget_fraction)
        || !(options.overlap_closure_tolerance >= 0.0)
        || options.energy_moment_panels < 1
        || options.energy_moment_panels > 64) {
        throw std::runtime_error("Invalid canonical bound-row request.");
    }
    if (options.direct_tensor_quadrature
        && (options.mapping_energy_points < 1
            || options.mapping_energy_points > 12
            || options.mapping_phase_points < 1
            || options.mapping_phase_points > 12)) {
        throw std::runtime_error(
            "Canonical tensor quadrature supports one through twelve points per axis.");
    }
    const double eta = std::isfinite(eta_override)
        ? eta_override
        : geometry.eta.center[static_cast<std::size_t>(eta_cell)];
    const double eta_width = std::isfinite(eta_weight_override)
        ? eta_weight_override
        : geometry.eta.width[static_cast<std::size_t>(eta_cell)];
    if (!(eta >= geometry.eta.edge[static_cast<std::size_t>(eta_cell)])
        || !(eta <= geometry.eta.edge[static_cast<std::size_t>(eta_cell + 1)])
        || !(eta_width > 0.0)) {
        std::ostringstream message;
        message.precision(17);
        message
            << "Canonical eta quadrature point lies outside its parent cell; "
            << "cell=" << eta_cell
            << ", eta=" << eta
            << ", eta_edges=["
            << geometry.eta.edge[static_cast<std::size_t>(eta_cell)] << ','
            << geometry.eta.edge[static_cast<std::size_t>(eta_cell + 1)] << ']'
            << ", eta_weight=" << eta_width
            << ", eta_plane=" << eta_plane;
        throw std::runtime_error(message.str());
    }
    std::map<double, HamiltonianOrbitPhaseGeometry> local_orbit_cache;
    auto& orbit_cache = shared_orbit_cache != nullptr
        ? *shared_orbit_cache
        : local_orbit_cache;
    auto orbit = [&](double energy) -> const HamiltonianOrbitPhaseGeometry& {
        auto found = orbit_cache.find(energy);
        if (found != orbit_cache.end()) {
            if (!phase_space_nearly_equal(found->second.eta, eta)) {
                throw std::runtime_error(
                    "A shared canonical orbit cache was reused across eta rows.");
            }
            return found->second;
        }
        const double hidden = hidden_return_time
            ? hidden_return_time(energy)
            : std::numeric_limits<double>::quiet_NaN();
        auto inserted = orbit_cache.emplace(
            energy,
            build_hamiltonian_orbit_phase_geometry(
                eta,
                energy,
                turning_scan_radius,
                options.phase_cells,
                effective_potential,
                hidden));
        return inserted.first->second;
    };
    std::map<double, double> local_period_cache;
    auto& period_cache = shared_period_cache != nullptr
        ? *shared_period_cache
        : local_period_cache;
    auto radial_period = [&](double energy) {
        const auto full = orbit_cache.find(energy);
        if (full != orbit_cache.end()) {
            return full->second.radial_period;
        }
        const auto found = period_cache.find(energy);
        if (found != period_cache.end()) {
            return found->second;
        }
        const double hidden = hidden_return_time
            ? hidden_return_time(energy)
            : std::numeric_limits<double>::quiet_NaN();
        const double value = hamiltonian_orbit_radial_period(
            energy,
            turning_scan_radius,
            effective_potential,
            hidden,
            64);
        period_cache.emplace(energy, value);
        return value;
    };
    // For canonical radial motion dJ/dH=T/(2*pi).  Integrating the actual
    // period across the row distinguishes its action/volume from the period
    // of the one representative orbit used to translate phase.
    const auto integrated_period = [&](double left, double right) {
        return canonical_adaptive_simpson_integral(
            left, right, radial_period, 1.0e-8, 18);
    };
    const double period_integral = integrated_period(
        energy_left, energy_right);
    if (!(period_integral > 0.0) || !std::isfinite(period_integral)) {
        throw std::runtime_error("Canonical orbit row has zero action width.");
    }
    // Geometric H moments and optional stationary f(H,eta) moments use the
    // same period-weighted energy measure, including the hidden inner phase.
    const auto& energy_rule = phase_space_gauss_legendre_rule(12);
    const int energy_panels = options.energy_moment_panels;
    std::array<long double, 5> geometric_energy_integral{};
    long double stationary_mass_integral = 0.0L;
    long double stationary_energy_integral = 0.0L;
    long double stationary_energy2_integral = 0.0L;
    for (int panel = 0; panel < energy_panels; ++panel) {
      for (std::size_t q = 0; q < energy_rule.first.size(); ++q) {
        const double panel_coordinate = (
            static_cast<double>(panel) + 0.5
            + 0.5 * energy_rule.first[q])
            / static_cast<double>(energy_panels);
        const double energy = std::fma(
            panel_coordinate,
            energy_right - energy_left,
            energy_left);
        const double weighted_period = energy_rule.second[q]
            * radial_period(energy)
            / static_cast<double>(energy_panels);
        long double energy_power = 1.0L;
        for (std::size_t power = 0;
             power < geometric_energy_integral.size();
             ++power) {
            geometric_energy_integral[power] +=
                weighted_period * energy_power;
            energy_power *= energy;
        }
        if (options.stationary_distribution) {
            const double density = options.stationary_distribution(eta, energy);
            if (!(density >= 0.0) || !std::isfinite(density)) {
                throw std::runtime_error(
                    "Canonical stationary distribution must be finite and nonnegative.");
            }
            stationary_mass_integral += weighted_period * density;
            stationary_energy_integral += weighted_period * density * energy;
            stationary_energy2_integral +=
                weighted_period * density * energy * energy;
        }
      }
    }
    const double half_energy_width = 0.5 * (energy_right - energy_left);
    for (long double& moment : geometric_energy_integral) {
        moment *= half_energy_width;
    }
    geometric_energy_integral[0] = period_integral;
    stationary_mass_integral *= half_energy_width;
    stationary_energy_integral *= half_energy_width;
    stationary_energy2_integral *= half_energy_width;

    CanonicalOrbitRow row;
    row.kind = CanonicalOrbitRowKind::bound_periodic;
    row.eta_cell = eta_cell;
    row.eta_plane = eta_plane;
    row.eta = eta;
    row.eta_weight = eta_width;
    row.energy_left = energy_left;
    row.energy_right = energy_right;
    row.energy = static_cast<double>(
        geometric_energy_integral[1] / period_integral);
    row.phase_geometry = orbit(row.energy);
    if (orbit(row.energy).hidden_return_time > 0.0) {
        row.kind = CanonicalOrbitRowKind::delayed_inner_periodic;
    }
    canonical_orbit_synchronize_row_phase_clock(row);
    row.action_left = 0.0;
    for (auto previous = atlas.row.rbegin(); previous != atlas.row.rend(); ++previous) {
        if ((eta_plane >= 0 && previous->eta_plane == eta_plane)
            || (eta_plane < 0 && previous->eta == eta)) {
            row.action_left = previous->action_right;
            break;
        }
    }
    row.action_right = row.action_left + period_integral / (2.0 * PI);
    row.phase_volume = eta_width * period_integral;
    row.first_cell = static_cast<std::uint32_t>(atlas.column.size());
    row.phase_mass.assign(static_cast<std::size_t>(options.phase_cells), 0.0);
    row.phase_energy_mass.assign(
        static_cast<std::size_t>(options.phase_cells), 0.0);
    row.phase_energy2_mass.assign(
        static_cast<std::size_t>(options.phase_cells), 0.0);
    if (options.stationary_distribution) {
        const long double phase_normalization = eta_width
            / static_cast<long double>(options.phase_cells);
        const double cell_mass = static_cast<double>(
            phase_normalization * stationary_mass_integral);
        const double cell_energy_mass = static_cast<double>(
            phase_normalization * stationary_energy_integral);
        const double cell_energy2_mass = static_cast<double>(
            phase_normalization * stationary_energy2_integral);
        std::fill(row.phase_mass.begin(), row.phase_mass.end(), cell_mass);
        std::fill(
            row.phase_energy_mass.begin(),
            row.phase_energy_mass.end(),
            cell_energy_mass);
        std::fill(
            row.phase_energy2_mass.begin(),
            row.phase_energy2_mass.end(),
            cell_energy2_mass);
    }
    atlas.column.resize(
        atlas.column.size() + static_cast<std::size_t>(options.phase_cells));
    const std::size_t row_index = atlas.row.size();
    atlas.row.push_back(row);

    std::map<std::pair<double, double>, HamiltonianOrbitPhaseGeometry::Location>
        local_location_cache;
    auto& location_cache = shared_location_cache != nullptr
        ? *shared_location_cache
        : local_location_cache;
    auto location = [&](double energy, double phase)
        -> const HamiltonianOrbitPhaseGeometry::Location& {
        const std::pair<double, double> key{energy, phase};
        auto found = location_cache.find(key);
        if (found != location_cache.end()) {
            return found->second;
        }
        auto inserted = location_cache.emplace(
            key,
            hamiltonian_orbit_location_at_phase_cached(
                orbit(energy), phase, effective_potential));
        return inserted.first->second;
    };

    if (options.direct_tensor_quadrature) {
        const auto& energy_quadrature = phase_space_gauss_legendre_rule(
            options.mapping_energy_points);
        const auto& phase_quadrature = phase_space_gauss_legendre_rule(
            options.mapping_phase_points);
        std::vector<double> mapping_energy_cut{energy_left, energy_right};
        for (double radius : geometry.radial.edge) {
            const double crossing_energy = effective_potential(radius);
            if (crossing_energy > energy_left
                && crossing_energy < energy_right) {
                mapping_energy_cut.push_back(crossing_energy);
            }
        }
        std::sort(mapping_energy_cut.begin(), mapping_energy_cut.end());
        mapping_energy_cut.erase(
            std::unique(
                mapping_energy_cut.begin(),
                mapping_energy_cut.end(),
                [](double first, double second) {
                    return phase_space_nearly_equal(first, second);
                }),
            mapping_energy_cut.end());

        struct TensorEnergyNode {
            double energy = 0.0;
            double energy_weight = 0.0;
            std::vector<double> phase_cut;
        };
        struct VelocityBoundary {
            double speed = 0.0;
            bool outgoing = false;
            bool incoming = false;
        };
        std::map<double, VelocityBoundary> velocity_boundary_by_speed;
        for (double edge : geometry.radial_velocity.edge) {
            if (edge == 0.0) {
                continue;
            }
            const double speed = std::abs(edge);
            VelocityBoundary& boundary = velocity_boundary_by_speed[speed];
            boundary.speed = speed;
            boundary.outgoing = boundary.outgoing || edge > 0.0;
            boundary.incoming = boundary.incoming || edge < 0.0;
        }
        // At each energy, split phase at radial/velocity cell crossings and
        // the hidden boundary before tensor quadrature.  This resolves the
        // discontinuous destination-cell indicator inside an energy row.
        std::vector<VelocityBoundary> velocity_boundary;
        velocity_boundary.reserve(velocity_boundary_by_speed.size());
        for (const auto& item : velocity_boundary_by_speed) {
            velocity_boundary.push_back(item.second);
        }
        std::vector<TensorEnergyNode> tensor_energy_node;
        tensor_energy_node.reserve(
            (mapping_energy_cut.size() - 1)
            * energy_quadrature.first.size());
        for (std::size_t energy_segment = 0;
             energy_segment + 1 < mapping_energy_cut.size();
             ++energy_segment) {
          const double energy_center = 0.5
              * (mapping_energy_cut[energy_segment]
                 + mapping_energy_cut[energy_segment + 1]);
          const double energy_half_width = 0.5
              * (mapping_energy_cut[energy_segment + 1]
                 - mapping_energy_cut[energy_segment]);
          for (std::size_t qh = 0;
               qh < energy_quadrature.first.size();
               ++qh) {
            TensorEnergyNode node;
            node.energy = energy_center
                + energy_half_width * energy_quadrature.first[qh];
            const HamiltonianOrbitPhaseGeometry& energy_orbit =
                orbit(node.energy);
            node.energy_weight = eta_width * energy_half_width
                * energy_quadrature.second[qh]
                * energy_orbit.radial_period;
            node.phase_cut = {0.0, 1.0};
            const auto add_phase_cut = [&](double phase) {
                if (phase > 0.0 && phase < 1.0
                    && std::isfinite(phase)) {
                    node.phase_cut.push_back(phase);
                }
            };
            std::vector<double> radius_cut{
                energy_orbit.pericenter,
                energy_orbit.apocenter,
            };
            for (double radius : geometry.radial.edge) {
                if (radius > energy_orbit.pericenter
                    && radius < energy_orbit.apocenter) {
                    radius_cut.push_back(radius);
                }
            }
            const double velocity_peak_radius =
                canonical_effective_potential_minimum_radius(
                    energy_orbit.pericenter,
                    energy_orbit.apocenter,
                    effective_potential);
            if (velocity_peak_radius > energy_orbit.pericenter
                && velocity_peak_radius < energy_orbit.apocenter) {
                radius_cut.push_back(velocity_peak_radius);
            }
            std::sort(radius_cut.begin(), radius_cut.end());
            radius_cut.erase(
                std::unique(
                    radius_cut.begin(),
                    radius_cut.end(),
                    [](double first, double second) {
                        return phase_space_nearly_equal(first, second);
                    }),
                radius_cut.end());
            for (double radius : radius_cut) {
                if (!(radius > energy_orbit.pericenter)
                    || !(radius < energy_orbit.apocenter)) {
                    continue;
                }
                const double radial_fraction = std::clamp(
                    (radius - energy_orbit.pericenter)
                        / (energy_orbit.apocenter
                           - energy_orbit.pericenter),
                    0.0,
                    1.0);
                const double alpha = std::asin(
                    std::sqrt(radial_fraction));
                const double outgoing_time =
                    hamiltonian_orbit_outgoing_time_at_alpha(
                        energy_orbit, alpha, effective_potential);
                add_phase_cut(
                    outgoing_time / energy_orbit.radial_period);
                add_phase_cut(
                    (2.0 * energy_orbit.visible_outgoing_time
                     - outgoing_time)
                    / energy_orbit.radial_period);
            }
            for (std::size_t interval = 0;
                 interval + 1 < radius_cut.size();
                 ++interval) {
                const double left = radius_cut[interval];
                const double right = radius_cut[interval + 1];
                const double left_speed = std::sqrt(std::max(
                    0.0,
                    2.0 * (node.energy - effective_potential(left))));
                const double right_speed = std::sqrt(std::max(
                    0.0,
                    2.0 * (node.energy - effective_potential(right))));
                const double low_speed = std::min(left_speed, right_speed);
                const double high_speed = std::max(left_speed, right_speed);
                const double speed_tolerance = 256.0
                    * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, high_speed);
                for (const VelocityBoundary& boundary : velocity_boundary) {
                    if (!(boundary.speed > low_speed + speed_tolerance)
                        || !(boundary.speed
                             < high_speed - speed_tolerance)) {
                        continue;
                    }
                    const double crossing_energy = node.energy
                        - 0.5 * boundary.speed * boundary.speed;
                    const double crossing_radius =
                        hamiltonian_orbit_bisect_turning_point(
                            left,
                            right,
                            crossing_energy,
                            effective_potential);
                    const double radial_fraction = std::clamp(
                        (crossing_radius - energy_orbit.pericenter)
                            / (energy_orbit.apocenter
                               - energy_orbit.pericenter),
                        0.0,
                        1.0);
                    const double alpha = std::asin(
                        std::sqrt(radial_fraction));
                    const double outgoing_time =
                        hamiltonian_orbit_outgoing_time_at_alpha(
                            energy_orbit, alpha, effective_potential);
                    if (boundary.outgoing) {
                        add_phase_cut(
                            outgoing_time / energy_orbit.radial_period);
                    }
                    if (boundary.incoming) {
                        add_phase_cut(
                            (2.0 * energy_orbit.visible_outgoing_time
                             - outgoing_time)
                            / energy_orbit.radial_period);
                    }
                }
            }
            add_phase_cut(
                2.0 * energy_orbit.visible_outgoing_time
                / energy_orbit.radial_period);
            std::sort(node.phase_cut.begin(), node.phase_cut.end());
            node.phase_cut.erase(
                std::unique(
                    node.phase_cut.begin(),
                    node.phase_cut.end(),
                    [](double first, double second) {
                        return phase_space_nearly_equal(first, second);
                    }),
                node.phase_cut.end());
            tensor_energy_node.push_back(std::move(node));
          }
        }
        for (int phase_cell = 0;
             phase_cell < options.phase_cells;
             ++phase_cell) {
            const double phase_left = static_cast<double>(phase_cell)
                / static_cast<double>(options.phase_cells);
            const double phase_right = static_cast<double>(phase_cell + 1)
                / static_cast<double>(options.phase_cells);
            const std::uint32_t canonical_cell = row.first_cell
                + static_cast<std::uint32_t>(phase_cell);
            CanonicalColumnLedger& column = atlas.column[canonical_cell];
            column.column_volume = row.phase_volume
                / static_cast<double>(options.phase_cells);
            for (std::size_t power = 0;
                 power < column.energy_volume_moment.size();
                 ++power) {
                column.energy_volume_moment[power] = static_cast<double>(
                    eta_width * geometric_energy_integral[power]
                    / static_cast<long double>(options.phase_cells));
            }
            std::map<std::uint32_t, CanonicalOverlap> accumulated;
            long double visible_volume = 0.0L;
            std::array<long double, 5> hidden_energy_volume_moment{};
            for (const TensorEnergyNode& node : tensor_energy_node) {
                const double energy = node.energy;
                const double energy_weight = node.energy_weight;
                std::vector<double> phase_cut{phase_left, phase_right};
                const auto first_cut = std::upper_bound(
                    node.phase_cut.begin(),
                    node.phase_cut.end(),
                    phase_left);
                const auto last_cut = std::lower_bound(
                    node.phase_cut.begin(),
                    node.phase_cut.end(),
                    phase_right);
                phase_cut.insert(phase_cut.end(), first_cut, last_cut);
                std::sort(phase_cut.begin(), phase_cut.end());
                for (std::size_t segment = 0;
                     segment + 1 < phase_cut.size();
                     ++segment) {
                  const double segment_center = 0.5
                      * (phase_cut[segment] + phase_cut[segment + 1]);
                  const double segment_half_width = 0.5
                      * (phase_cut[segment + 1] - phase_cut[segment]);
                  for (std::size_t qs = 0;
                       qs < phase_quadrature.first.size();
                       ++qs) {
                    const double phase = segment_center
                        + segment_half_width * phase_quadrature.first[qs];
                    const double weight = energy_weight * segment_half_width
                        * phase_quadrature.second[qs];
                    const auto& point = location(energy, phase);
                    if (!point.visible) {
                        long double energy_power = 1.0L;
                        for (long double& moment :
                             hidden_energy_volume_moment) {
                            moment += weight * energy_power;
                            energy_power *= energy;
                        }
                        continue;
                    }
                    const int i = canonical_orbit_containing_cell(
                        point.state.r, geometry.radial.edge);
                    const int j = canonical_orbit_containing_cell(
                        point.state.w, geometry.radial_velocity.edge);
                    if (i < 0 || j < 0) {
                        throw std::runtime_error(
                            "Canonical tensor quadrature left the Eulerian grid.");
                    }
                    const std::uint32_t eulerian_cell =
                        static_cast<std::uint32_t>(idx(i, j, eta_cell, p));
                    CanonicalOverlap& entry = accumulated[eulerian_cell];
                    entry.eulerian_cell = eulerian_cell;
                    entry.canonical_cell = canonical_cell;
                    entry.volume += weight;
                    entry.energy_volume += weight * energy;
                    entry.energy2_volume += weight * energy * energy;
                    const double x = (
                        point.state.r
                        - geometry.radial.center[static_cast<std::size_t>(i)])
                        / geometry.radial.width[static_cast<std::size_t>(i)];
                    const double y = (
                        point.state.w
                        - geometry.radial_velocity.center[static_cast<std::size_t>(j)])
                        / geometry.radial_velocity.width[static_cast<std::size_t>(j)];
                    const std::array<double, 5> basis{
                        x,
                        y,
                        x * x - 1.0 / 12.0,
                        x * y,
                        y * y - 1.0 / 12.0,
                    };
                    for (std::size_t mode = 0; mode < basis.size(); ++mode) {
                        entry.eulerian_p2_moment[mode] += weight * basis[mode];
                    }
                    visible_volume += weight;
                  }
                }
            }
            // Close the numerical tensor weights to the independently
            // period-integrated canonical column volume, scaling all stored
            // geometric moments by the same factor.
            const double sampled_volume = static_cast<double>(
                visible_volume + hidden_energy_volume_moment[0]);
            if (!(sampled_volume > 0.0)) {
                throw std::runtime_error(
                    "Canonical tensor quadrature has zero column volume.");
            }
            column.geometric_scale = column.column_volume / sampled_volume;
            column.maximum_patch_scale_error =
                std::abs(column.geometric_scale - 1.0);
            for (auto& item : accumulated) {
                CanonicalOverlap entry = item.second;
                entry.volume *= column.geometric_scale;
                entry.energy_volume *= column.geometric_scale;
                entry.energy2_volume *= column.geometric_scale;
                for (double& moment : entry.eulerian_p2_moment) {
                    moment *= column.geometric_scale;
                }
                atlas.overlap.push_back(std::move(entry));
            }
            column.visible_volume = static_cast<double>(visible_volume)
                * column.geometric_scale;
            for (std::size_t power = 0;
                 power < column.hidden_energy_volume_moment.size();
                 ++power) {
                column.hidden_energy_volume_moment[power] =
                    static_cast<double>(
                        hidden_energy_volume_moment[power])
                    * column.geometric_scale;
            }
            column.hidden_volume = column.hidden_energy_volume_moment[0];
            column.escaped_energy_volume_moment.fill(0.0);
            column.partition_energy_moments_resolved = true;
            column.residual = column.column_volume
                - column.visible_volume - column.hidden_volume;
        }
        return row_index;
    }

    for (int phase_cell = 0; phase_cell < options.phase_cells; ++phase_cell) {
        const std::uint32_t canonical_cell = row.first_cell
            + static_cast<std::uint32_t>(phase_cell);
        const double phase_left = static_cast<double>(phase_cell)
            / static_cast<double>(options.phase_cells);
        const double phase_right = static_cast<double>(phase_cell + 1)
            / static_cast<double>(options.phase_cells);
        CanonicalColumnLedger& column = atlas.column[canonical_cell];
        const std::size_t first_column_overlap = atlas.overlap.size();
        column.column_volume = row.phase_volume
            / static_cast<double>(options.phase_cells);
        for (std::size_t power = 0;
             power < column.energy_volume_moment.size();
             ++power) {
            column.energy_volume_moment[power] = static_cast<double>(
                eta_width * geometric_energy_integral[power]
                / static_cast<long double>(options.phase_cells));
        }
        auto append_visible_location = [&](SLDGPolygon& patch,
                                           double energy,
                                           double phase) {
            const auto& value = location(energy, phase);
            if (!value.visible) {
                std::ostringstream message;
                message
                    << "A canonical visible boundary point entered the hidden carrier: H="
                    << std::setprecision(17) << energy
                    << ", phase=" << phase
                    << ", hidden_begin="
                    << 2.0 * orbit(energy).visible_outgoing_time
                        / orbit(energy).radial_period << '.';
                throw std::runtime_error(message.str());
            }
            patch.push_back({value.state.r, value.state.w});
        };
        // Polygon overlaps estimate H at their Eulerian-coordinate centroid
        // and clamp it to this row.  The tensor route above instead directly
        // integrates H and H^2 at the energy quadrature nodes.
        auto assign_overlap_energy_moments = [&](std::size_t first_overlap) {
            for (std::size_t entry_index = first_overlap;
                 entry_index < atlas.overlap.size();
                 ++entry_index) {
                CanonicalOverlap& entry = atlas.overlap[entry_index];
                const std::size_t phase_cell_index = entry.eulerian_cell;
                const int k = static_cast<int>(phase_cell_index % p.NL);
                const std::size_t radial_velocity_index =
                    phase_cell_index / static_cast<std::size_t>(p.NL);
                const int j = static_cast<int>(
                    radial_velocity_index % static_cast<std::size_t>(p.Nw));
                const int i = static_cast<int>(
                    radial_velocity_index / static_cast<std::size_t>(p.Nw));
                if (k != eta_cell || !(entry.volume > 0.0)) {
                    throw std::runtime_error(
                        "Canonical overlap has inconsistent energy ownership.");
                }
                const double mean_x = entry.eulerian_p2_moment[0]
                    / entry.volume;
                const double mean_y = entry.eulerian_p2_moment[1]
                    / entry.volume;
                const double radius = std::fma(
                    mean_x,
                    geometry.radial.width[static_cast<std::size_t>(i)],
                    geometry.radial.center[static_cast<std::size_t>(i)]);
                const double velocity = std::fma(
                    mean_y,
                    geometry.radial_velocity.width[static_cast<std::size_t>(j)],
                    geometry.radial_velocity.center[static_cast<std::size_t>(j)]);
                const double energy = std::clamp(
                    effective_potential(radius) + 0.5 * velocity * velocity,
                    energy_left,
                    energy_right);
                entry.energy_volume = energy * entry.volume;
                entry.energy2_volume = energy * energy * entry.volume;
            }
        };
        // Compare the mapped boundary area with canonical T(H)dH*ds volume.
        // Admitted local geometric corrections scale every overlap moment
        // together; the column separately records their size and excess.
        auto deposit_visible_polygon = [&](SLDGPolygon patch,
                                           double exact_volume,
                                           double allowed_volume_error) {
            if (!(exact_volume > 0.0)) {
                return;
            }
            if (sldg_polygon_signed_area(patch) < 0.0) {
                std::reverse(patch.begin(), patch.end());
            }
            const SLDGExtendedClipContext context(patch);
            const double mapped_volume = context.valid()
                ? static_cast<double>(context.physical_area()) * eta_width
                : 0.0;
            const double volume_error = std::abs(mapped_volume - exact_volume)
                / std::max(exact_volume, std::numeric_limits<double>::min());
            const double absolute_error = std::abs(
                mapped_volume - exact_volume);
            const double patch_excess_error = std::max(
                0.0,
                absolute_error
                    - options.mapped_volume_tolerance * exact_volume);
            const double remaining_column_excess_budget = std::max(
                0.0,
                options.mapped_volume_excess_budget_fraction
                    * options.mapped_volume_tolerance * column.column_volume
                    - column.accumulated_patch_excess_error);
            if (!context.valid()
                || (volume_error > options.mapped_volume_tolerance
                    && patch_excess_error
                        > remaining_column_excess_budget)) {
                std::ostringstream message;
                message
                    << "Canonical visible boundary did not converge: mapped_volume="
                    << std::setprecision(17) << mapped_volume
                    << ", canonical_volume=" << exact_volume
                    << ", relative_error=" << volume_error
                    << ", absolute_error=" << absolute_error
                    << ", patch_excess_error=" << patch_excess_error
                    << ", remaining_column_excess_budget="
                    << remaining_column_excess_budget
                    << ", tolerance=" << allowed_volume_error << '.';
                throw std::runtime_error(message.str());
            }
            const std::size_t first_overlap = atlas.overlap.size();
            const CanonicalColumnLedger leaf =
                canonical_orbit_append_visible_patch_overlaps(
                    patch,
                    canonical_cell,
                    eta_cell,
                    geometry,
                    p,
                    atlas.overlap,
                    options.overlap_closure_tolerance);
            assign_overlap_energy_moments(first_overlap);
            const double scale = exact_volume / leaf.visible_volume;
            column.maximum_patch_scale_error = std::max(
                column.maximum_patch_scale_error,
                std::abs(scale - 1.0));
            column.accumulated_patch_excess_error += patch_excess_error;
            for (std::size_t entry = first_overlap;
                 entry < atlas.overlap.size();
                 ++entry) {
                atlas.overlap[entry].volume *= scale;
                atlas.overlap[entry].energy_volume *= scale;
                atlas.overlap[entry].energy2_volume *= scale;
                for (double& moment : atlas.overlap[entry].eulerian_p2_moment) {
                    moment *= scale;
                }
            }
            column.visible_volume += exact_volume;
        };
        // This fallback preserves the prescribed canonical volume for a
        // nondegenerate boundary polygon and records its geometric scaling.
        auto deposit_visible_polygon_scaled = [&](SLDGPolygon patch,
                                                  double exact_volume) {
            if (!(exact_volume > 0.0)) {
                return;
            }
            if (sldg_polygon_signed_area(patch) < 0.0) {
                std::reverse(patch.begin(), patch.end());
            }
            const SLDGExtendedClipContext context(patch);
            if (!context.valid()) {
                throw std::runtime_error(
                    "A conservatively scaled canonical polygon is degenerate.");
            }
            const double mapped_volume =
                static_cast<double>(context.physical_area()) * eta_width;
            const std::size_t first_overlap = atlas.overlap.size();
            const CanonicalColumnLedger leaf =
                canonical_orbit_append_visible_patch_overlaps(
                    patch,
                    canonical_cell,
                    eta_cell,
                    geometry,
                    p,
                    atlas.overlap,
                    options.overlap_closure_tolerance);
            assign_overlap_energy_moments(first_overlap);
            const double scale = exact_volume / leaf.visible_volume;
            column.maximum_patch_scale_error = std::max(
                column.maximum_patch_scale_error,
                std::abs(scale - 1.0));
            column.accumulated_patch_excess_error += std::max(
                0.0,
                std::abs(mapped_volume - exact_volume)
                    - options.mapped_volume_tolerance * exact_volume);
            for (std::size_t entry = first_overlap;
                 entry < atlas.overlap.size();
                 ++entry) {
                atlas.overlap[entry].volume *= scale;
                atlas.overlap[entry].energy_volume *= scale;
                atlas.overlap[entry].energy2_volume *= scale;
                for (double& moment : atlas.overlap[entry].eulerian_p2_moment) {
                    moment *= scale;
                }
            }
            column.visible_volume += exact_volume;
        };
        // Resolve a difficult visible patch in canonical energy/phase space.
        // Each child retains its allocated canonical volume even when the
        // mapped boundary becomes singular near a turning point.
        auto deposit_visible_canonical_tessellation = [&]
            (double h_left,
             double h_right,
             const std::function<double(double)>& phase_lower,
             const std::function<double(double)>& phase_upper,
             double exact_volume) {
            if (!(h_right > h_left) || !(exact_volume > 0.0)) {
                return;
            }
            struct MappedCanonicalPoint {
                SLDGPoint point;
                int radial_cell = -1;
                int velocity_cell = -1;
                bool visible = false;
            };
            auto mapped_point = [&](double energy, double unit_phase) {
                MappedCanonicalPoint mapped;
                const double lower = phase_lower(energy);
                const double upper = phase_upper(energy);
                if (!(upper > lower)) {
                    return mapped;
                }
                const double phase = std::fma(
                    unit_phase, upper - lower, lower);
                const auto& point = location(energy, phase);
                if (!point.visible) {
                    return mapped;
                }
                mapped.point = {point.state.r, point.state.w};
                mapped.radial_cell = canonical_orbit_containing_cell(
                    point.state.r, geometry.radial.edge);
                mapped.velocity_cell = canonical_orbit_containing_cell(
                    point.state.w, geometry.radial_velocity.edge);
                mapped.visible = mapped.radial_cell >= 0
                    && mapped.velocity_cell >= 0;
                return mapped;
            };
            auto energy_weight = [&](double energy) {
                return orbit(energy).radial_period
                    * std::max(0.0, phase_upper(energy) - phase_lower(energy));
            };
            const int maximum_tessellation_depth = std::clamp(
                options.maximum_patch_depth, 5, 6);
            std::function<void(double, double, double, double, double, int)>
                deposit_tile;
            deposit_tile = [&](double energy_a,
                               double energy_b,
                               double unit_a,
                               double unit_b,
                               double tile_volume,
                               int depth) {
                const double energy_middle = 0.5 * (energy_a + energy_b);
                const double unit_middle = 0.5 * (unit_a + unit_b);
                const std::array<MappedCanonicalPoint, 9> mapped{
                    mapped_point(energy_a, unit_a),
                    mapped_point(energy_middle, unit_a),
                    mapped_point(energy_b, unit_a),
                    mapped_point(energy_a, unit_middle),
                    mapped_point(energy_middle, unit_middle),
                    mapped_point(energy_b, unit_middle),
                    mapped_point(energy_a, unit_b),
                    mapped_point(energy_middle, unit_b),
                    mapped_point(energy_b, unit_b),
                };
                int minimum_i = std::numeric_limits<int>::max();
                int maximum_i = -1;
                int minimum_j = std::numeric_limits<int>::max();
                int maximum_j = -1;
                bool valid = true;
                for (const MappedCanonicalPoint& point : mapped) {
                    valid = valid && point.visible;
                    minimum_i = std::min(minimum_i, point.radial_cell);
                    maximum_i = std::max(maximum_i, point.radial_cell);
                    minimum_j = std::min(minimum_j, point.velocity_cell);
                    maximum_j = std::max(maximum_j, point.velocity_cell);
                }
                const bool cell_resolved = valid
                    && maximum_i - minimum_i <= 1
                    && maximum_j - minimum_j <= 1;

                SLDGPolygon patch;
                if (valid) {
                    patch = {
                        mapped[0].point,
                        mapped[2].point,
                        mapped[8].point,
                        mapped[6].point,
                    };
                    if (sldg_polygon_signed_area(patch) < 0.0) {
                        std::reverse(patch.begin(), patch.end());
                    }
                }
                const SLDGExtendedClipContext context(patch);
                const bool maximum_depth = depth >= maximum_tessellation_depth;
                if (valid && context.valid()
                    && (cell_resolved || maximum_depth)) {
                    const std::size_t first_overlap = atlas.overlap.size();
                    try {
                        const CanonicalColumnLedger leaf =
                            canonical_orbit_append_visible_patch_overlaps(
                                patch,
                                canonical_cell,
                                eta_cell,
                                geometry,
                                p,
                                atlas.overlap,
                                options.overlap_closure_tolerance);
                        assign_overlap_energy_moments(first_overlap);
                        const double scale = tile_volume / leaf.visible_volume;
                        column.maximum_patch_scale_error = std::max(
                            column.maximum_patch_scale_error,
                            std::abs(scale - 1.0));
                        for (std::size_t entry = first_overlap;
                             entry < atlas.overlap.size();
                             ++entry) {
                            atlas.overlap[entry].volume *= scale;
                            atlas.overlap[entry].energy_volume *= scale;
                            atlas.overlap[entry].energy2_volume *= scale;
                            for (double& moment :
                                 atlas.overlap[entry].eulerian_p2_moment) {
                                moment *= scale;
                            }
                        }
                        column.visible_volume += tile_volume;
                        column.canonical_quadrature_volume += tile_volume;
                        return;
                    } catch (const std::exception&) {
                        atlas.overlap.resize(first_overlap);
                        if (!maximum_depth) {
                            // Continue below and split the singular image.
                        } else {
                            // At terminal depth, use bounded tensor Simpson
                            // weights over the nine resolved visible points.
                            // Retain this volume in the quadrature ledger.
                            std::map<std::uint32_t, CanonicalOverlap> sampled;
                            const std::array<double, 3> simpson{1.0, 4.0, 1.0};
                            long double sampled_weight = 0.0L;
                            for (int qh = 0; qh < 3; ++qh) {
                                const double energy = qh == 0
                                    ? energy_a
                                    : (qh == 1 ? energy_middle : energy_b);
                                const double h_weight = simpson[qh]
                                    * energy_weight(energy);
                                for (int qs = 0; qs < 3; ++qs) {
                                    const MappedCanonicalPoint& point =
                                        mapped[static_cast<std::size_t>(3 * qs + qh)];
                                    const long double weight = h_weight * simpson[qs];
                                    sampled_weight += weight;
                                    const std::uint32_t cell =
                                        static_cast<std::uint32_t>(idx(
                                            point.radial_cell,
                                            point.velocity_cell,
                                            eta_cell,
                                            p));
                                    CanonicalOverlap& entry = sampled[cell];
                                    entry.eulerian_cell = cell;
                                    entry.canonical_cell = canonical_cell;
                                    entry.volume += static_cast<double>(weight);
                                    entry.energy_volume +=
                                        static_cast<double>(weight) * energy;
                                    entry.energy2_volume +=
                                        static_cast<double>(weight)
                                        * energy * energy;
                                    const double x = (
                                        point.point.r
                                        - geometry.radial.center[
                                            static_cast<std::size_t>(point.radial_cell)])
                                        / geometry.radial.width[
                                            static_cast<std::size_t>(point.radial_cell)];
                                    const double y = (
                                        point.point.w
                                        - geometry.radial_velocity.center[
                                            static_cast<std::size_t>(point.velocity_cell)])
                                        / geometry.radial_velocity.width[
                                            static_cast<std::size_t>(point.velocity_cell)];
                                    const std::array<double, 5> basis{
                                        x,
                                        y,
                                        x * x - 1.0 / 12.0,
                                        x * y,
                                        y * y - 1.0 / 12.0,
                                    };
                                    for (std::size_t mode = 0;
                                         mode < basis.size();
                                         ++mode) {
                                        entry.eulerian_p2_moment[mode] +=
                                            static_cast<double>(weight) * basis[mode];
                                    }
                                }
                            }
                            if (!(sampled_weight > 0.0L)) {
                                throw std::runtime_error(
                                    "Canonical terminal quadrature has zero weight.");
                            }
                            const double scale = tile_volume
                                / static_cast<double>(sampled_weight);
                            for (auto& item : sampled) {
                                CanonicalOverlap entry = item.second;
                                entry.volume *= scale;
                                entry.energy_volume *= scale;
                                entry.energy2_volume *= scale;
                                for (double& moment : entry.eulerian_p2_moment) {
                                    moment *= scale;
                                }
                                atlas.overlap.push_back(std::move(entry));
                            }
                            column.visible_volume += tile_volume;
                            column.canonical_quadrature_volume += tile_volume;
                            return;
                        }
                    }
                }
                const int radial_span = maximum_i - minimum_i;
                const int velocity_span = maximum_j - minimum_j;
                const bool split_energy = !valid || depth % 2 == 0
                    ? radial_span >= velocity_span
                    : radial_span > velocity_span;
                if (split_energy) {
                    const double left_weight = 0.5
                        * (energy_middle - energy_a)
                        * (energy_weight(energy_a)
                           + energy_weight(energy_middle));
                    const double right_weight = 0.5
                        * (energy_b - energy_middle)
                        * (energy_weight(energy_middle)
                           + energy_weight(energy_b));
                    const double weight_sum = left_weight + right_weight;
                    if (!(weight_sum > 0.0)) {
                        throw std::runtime_error(
                            "Canonical tessellation has zero energy weight.");
                    }
                    const double left_volume = tile_volume
                        * left_weight / weight_sum;
                    deposit_tile(
                        energy_a,
                        energy_middle,
                        unit_a,
                        unit_b,
                        left_volume,
                        depth + 1);
                    deposit_tile(
                        energy_middle,
                        energy_b,
                        unit_a,
                        unit_b,
                        tile_volume - left_volume,
                        depth + 1);
                } else {
                    deposit_tile(
                        energy_a,
                        energy_b,
                        unit_a,
                        unit_middle,
                        0.5 * tile_volume,
                        depth + 1);
                    deposit_tile(
                        energy_a,
                        energy_b,
                        unit_middle,
                        unit_b,
                        0.5 * tile_volume,
                        depth + 1);
                }
            };
            deposit_tile(h_left, h_right, 0.0, 1.0, exact_volume, 0);
        };
        // Build ordinary mapped boundaries first.  Hidden/visible strips
        // require energy-dependent phase limits because inner return times
        // vary across a finite energy row.
        const int maximum_polygon_depth = options.maximum_patch_depth;
        std::function<void(double, double, double, double, int)> append_patch;
        append_patch = [&](double h0, double h1, double s0, double s1, int depth) {
            const auto& lower_left = location(h0, s0);
            const auto& upper_left = location(h1, s0);
            const auto& lower_right = location(h0, s1);
            const auto& upper_right = location(h1, s1);
            int visible_corners = static_cast<int>(lower_left.visible)
                + static_cast<int>(upper_left.visible)
                + static_cast<int>(lower_right.visible)
                + static_cast<int>(upper_right.visible);
            const auto visible_phase_end = [&](double energy) {
                const auto& value = orbit(energy);
                return 2.0 * value.visible_outgoing_time
                    / value.radial_period;
            };
            bool delayed_row = false;
            double minimum_visible_phase = 1.0;
            double maximum_visible_phase = 0.0;
            const int visibility_samples = std::max(
                4, options.mapped_boundary_segments);
            for (int q = 0; q <= visibility_samples; ++q) {
                const double energy = h0 + (h1 - h0)
                    * static_cast<double>(q)
                    / static_cast<double>(visibility_samples);
                delayed_row = delayed_row
                    || orbit(energy).hidden_return_time > 0.0;
                const double phase_end = visible_phase_end(energy);
                minimum_visible_phase = std::min(
                    minimum_visible_phase, phase_end);
                maximum_visible_phase = std::max(
                    maximum_visible_phase, phase_end);
            }
            // Probe the whole energy strip before treating its corners as
            // fully visible/hidden; the hidden boundary can curve between them.
            if (delayed_row) {
                if (s0 >= maximum_visible_phase) {
                    visible_corners = 0;
                } else if (s1 < std::nextafter(
                               minimum_visible_phase, 0.0)) {
                    visible_corners = 4;
                } else {
                    visible_corners = 2;
                }
            }
            const double exact_volume = eta_width * (s1 - s0)
                * integrated_period(h0, h1);
            if (visible_corners == 0) {
                column.hidden_volume += exact_volume;
                return;
            }
            if (visible_corners != 4) {
                // Integrate T(H) times the visible phase width separately,
                // then assign its complement to the persistent inner phase.
                const auto visible_width = [&](double energy) {
                    return std::max(
                        0.0,
                        std::min(s1, visible_phase_end(energy)) - s0);
                };
                const int energy_segments = std::max(
                    4, options.mapped_boundary_segments);
                const auto& gauss = phase_space_gauss_legendre_rule(12);
                std::function<void(double, double, int)> append_mixed_strip;
                append_mixed_strip = [&](double a, double b, int strip_depth) {
                    const double middle = 0.5 * (a + b);
                    const double wa = visible_width(a);
                    const double wm = visible_width(middle);
                    const double wb = visible_width(b);
                    const double strip_volume = eta_width * (s1 - s0)
                        * integrated_period(a, b);
                    if (!(wa > 0.0) && !(wm > 0.0) && !(wb > 0.0)) {
                        column.hidden_volume += strip_volume;
                        return;
                    }
                    if (!(wa > 0.0) && !(wb > 0.0) && wm > 0.0) {
                        if (strip_depth >= 24) {
                            throw std::runtime_error(
                                "Delayed-return phase boundary is not resolved in energy.");
                        }
                        append_mixed_strip(a, middle, strip_depth + 1);
                        append_mixed_strip(middle, b, strip_depth + 1);
                        return;
                    }
                    double active_left = a;
                    double active_right = b;
                    auto activity = [&](double energy) {
                        return visible_phase_end(energy) - s0;
                    };
                    if (!(wa > 0.0)) {
                        double left = a;
                        double right = wm > 0.0 ? middle : b;
                        for (int iteration = 0; iteration < 64; ++iteration) {
                            const double trial = 0.5 * (left + right);
                            if (activity(trial) > 0.0) {
                                right = trial;
                            } else {
                                left = trial;
                            }
                        }
                        active_left = right;
                    }
                    if (!(wb > 0.0)) {
                        double left = wm > 0.0 ? middle : a;
                        double right = b;
                        for (int iteration = 0; iteration < 64; ++iteration) {
                            const double trial = 0.5 * (left + right);
                            if (activity(trial) > 0.0) {
                                left = trial;
                            } else {
                                right = trial;
                            }
                        }
                        active_right = left;
                    }
                    if (!(active_right > active_left)) {
                        column.hidden_volume += strip_volume;
                        return;
                    }
                    long double visible_volume = 0.0L;
                    const double half_width = 0.5 * (active_right - active_left);
                    const double center = 0.5 * (active_right + active_left);
                    for (std::size_t q = 0; q < gauss.first.size(); ++q) {
                        const double energy = center + half_width * gauss.first[q];
                        visible_volume += static_cast<long double>(
                            half_width * gauss.second[q] * eta_width
                            * orbit(energy).radial_period * visible_width(energy));
                    }
                    const double represented_visible = static_cast<double>(visible_volume);
                    if (!(represented_visible > 0.0)) {
                        column.hidden_volume += strip_volume;
                        return;
                    }
                    const auto clipped_top = [&](double energy) {
                        const double top = std::min(s1, visible_phase_end(energy));
                        return std::max(s0, std::nextafter(top, s0));
                    };
                    SLDGPolygon mixed_patch;
                    const int boundary_points = std::max(
                        8, options.mapped_boundary_segments);
                    mixed_patch.reserve(
                        static_cast<std::size_t>(4 * boundary_points));
                    auto append_mixed_point = [&](double energy, double phase) {
                        const auto& point = location(energy, phase);
                        if (!point.visible) {
                            throw std::runtime_error(
                                "Delayed-return polygon entered the hidden carrier.");
                        }
                        mixed_patch.push_back({point.state.r, point.state.w});
                    };
                    for (int q = 0; q < boundary_points; ++q) {
                        const double fraction = static_cast<double>(q)
                            / static_cast<double>(boundary_points);
                        const double energy = std::fma(
                            fraction,
                            active_right - active_left,
                            active_left);
                        append_mixed_point(energy, s0);
                    }
                    for (int q = 0; q < boundary_points; ++q) {
                        const double fraction = static_cast<double>(q)
                            / static_cast<double>(boundary_points);
                        const double top = clipped_top(active_right);
                        append_mixed_point(
                            active_right,
                            std::fma(fraction, top - s0, s0));
                    }
                    for (int q = 0; q < boundary_points; ++q) {
                        const double fraction = static_cast<double>(q)
                            / static_cast<double>(boundary_points);
                        const double energy = std::fma(
                            -fraction,
                            active_right - active_left,
                            active_right);
                        append_mixed_point(energy, clipped_top(energy));
                    }
                    for (int q = 0; q < boundary_points; ++q) {
                        const double fraction = static_cast<double>(q)
                            / static_cast<double>(boundary_points);
                        const double top = clipped_top(active_left);
                        append_mixed_point(
                            active_left,
                            std::fma(-fraction, top - s0, top));
                    }
                    try {
                        deposit_visible_polygon_scaled(
                            std::move(mixed_patch), represented_visible);
                    } catch (const std::exception&) {
                        if (strip_depth < 8) {
                            append_mixed_strip(a, middle, strip_depth + 1);
                            append_mixed_strip(middle, b, strip_depth + 1);
                            return;
                        }
                        deposit_visible_canonical_tessellation(
                            active_left,
                            active_right,
                            [=](double) { return s0; },
                            clipped_top,
                            represented_visible);
                    }
                    column.hidden_volume += std::max(
                        0.0, strip_volume - represented_visible);
                };
                for (int segment = 0; segment < energy_segments; ++segment) {
                    const double a = h0 + (h1 - h0)
                        * static_cast<double>(segment)
                        / static_cast<double>(energy_segments);
                    const double b = h0 + (h1 - h0)
                        * static_cast<double>(segment + 1)
                        / static_cast<double>(energy_segments);
                    append_mixed_strip(a, b, 0);
                }
                return;
            }
            auto build_visible_patch = [&](int boundary_segments) {
                SLDGPolygon built;
                built.reserve(static_cast<std::size_t>(4 * boundary_segments));
                for (int q = 0; q < boundary_segments; ++q) {
                    append_visible_location(
                        built,
                        h0 + (h1 - h0) * static_cast<double>(q)
                            / static_cast<double>(boundary_segments),
                        s0);
                }
                for (int q = 0; q < boundary_segments; ++q) {
                    append_visible_location(
                        built,
                        h1,
                        s0 + (s1 - s0) * static_cast<double>(q)
                            / static_cast<double>(boundary_segments));
                }
                for (int q = 0; q < boundary_segments; ++q) {
                    append_visible_location(
                        built,
                        h1 - (h1 - h0) * static_cast<double>(q)
                            / static_cast<double>(boundary_segments),
                        s1);
                }
                for (int q = 0; q < boundary_segments; ++q) {
                    append_visible_location(
                        built,
                        h0,
                        s1 - (s1 - s0) * static_cast<double>(q)
                            / static_cast<double>(boundary_segments));
                }
                return built;
            };
            const int boundary_segments = options.mapped_boundary_segments;
            SLDGPolygon patch = build_visible_patch(boundary_segments);
            const double signed_area = sldg_polygon_signed_area(patch);
            if (signed_area < 0.0) {
                std::reverse(patch.begin(), patch.end());
            }
            const SLDGExtendedClipContext context(patch);
            const double mapped_volume = context.valid()
                ? static_cast<double>(context.physical_area()) * eta_width
                : 0.0;
            const double absolute_error = std::abs(
                mapped_volume - exact_volume);
            const double volume_error = absolute_error
                / std::max(exact_volume, std::numeric_limits<double>::min());
            const double patch_excess_error = std::max(
                0.0,
                absolute_error
                    - options.mapped_volume_tolerance * exact_volume);
            const double remaining_column_excess_budget = std::max(
                0.0,
                options.mapped_volume_excess_budget_fraction
                    * options.mapped_volume_tolerance * column.column_volume
                    - column.accumulated_patch_excess_error);
            const bool patch_fits_column_error_budget =
                volume_error <= options.mapped_volume_tolerance
                || patch_excess_error <= remaining_column_excess_budget;
            auto deposit_refined_boundary = [&]() {
                for (int factor : {2, 4, 8}) {
                    SLDGPolygon refined = build_visible_patch(
                        factor * boundary_segments);
                    try {
                        deposit_visible_polygon_scaled(
                            std::move(refined), exact_volume);
                        return true;
                    } catch (const std::exception&) {
                    }
                }
                return false;
            };
            auto split_folded_patch = [&]() {
                if (depth >= maximum_polygon_depth) {
                    return false;
                }
                const double hm = 0.5 * (h0 + h1);
                const double sm = 0.5 * (s0 + s1);
                append_patch(h0, hm, s0, sm, depth + 1);
                append_patch(hm, h1, s0, sm, depth + 1);
                append_patch(h0, hm, sm, s1, depth + 1);
                append_patch(hm, h1, sm, s1, depth + 1);
                return true;
            };
            if ((!context.valid()
                 || !patch_fits_column_error_budget)
                && depth < maximum_polygon_depth) {
                const double sm = 0.5 * (s0 + s1);
                const double phase_epsilon = 64.0
                    * std::numeric_limits<double>::epsilon();
                const bool touches_turning_phase =
                    std::abs(s0) <= phase_epsilon
                    || std::abs(s1 - 0.5) <= phase_epsilon
                    || std::abs(s0 - 0.5) <= phase_epsilon
                    || std::abs(s1 - 1.0) <= phase_epsilon;
                if (touches_turning_phase) {
                    const double relative_energy_width =
                        (h1 - h0) / (energy_right - energy_left);
                    const double relative_phase_width =
                        (s1 - s0) * options.phase_cells;
                    if (relative_energy_width > relative_phase_width) {
                        const double hm = 0.5 * (h0 + h1);
                        append_patch(h0, hm, s0, s1, depth + 1);
                        append_patch(hm, h1, s0, s1, depth + 1);
                    } else {
                        append_patch(h0, h1, s0, sm, depth + 1);
                        append_patch(h0, h1, sm, s1, depth + 1);
                    }
                } else {
                    const double hm = 0.5 * (h0 + h1);
                    append_patch(h0, hm, s0, sm, depth + 1);
                    append_patch(hm, h1, s0, sm, depth + 1);
                    append_patch(h0, hm, sm, s1, depth + 1);
                    append_patch(hm, h1, sm, s1, depth + 1);
                }
                return;
            }
            if (!context.valid() || !patch_fits_column_error_budget) {
                if (context.valid()) {
                    try {
                        deposit_visible_polygon_scaled(patch, exact_volume);
                    } catch (const std::exception&) {
                        if (!deposit_refined_boundary()) {
                            if (!split_folded_patch()) {
                                deposit_visible_canonical_tessellation(
                                    h0,
                                    h1,
                                    [=](double) { return s0; },
                                    [=](double) { return s1; },
                                    exact_volume);
                            }
                        }
                    }
                } else {
                    if (!deposit_refined_boundary()) {
                        if (!split_folded_patch()) {
                            deposit_visible_canonical_tessellation(
                                h0,
                                h1,
                                [=](double) { return s0; },
                                [=](double) { return s1; },
                                exact_volume);
                        }
                    }
                }
                return;
            }
            const double leaf_relative_tolerance = std::max(
                options.mapped_volume_tolerance,
                (options.mapped_volume_tolerance * exact_volume
                 + remaining_column_excess_budget)
                    / std::max(exact_volume,
                               std::numeric_limits<double>::min()));
            try {
                deposit_visible_polygon(
                    patch,
                    exact_volume,
                    leaf_relative_tolerance);
            } catch (const std::exception&) {
                try {
                    deposit_visible_polygon_scaled(patch, exact_volume);
                } catch (const std::exception&) {
                    if (!deposit_refined_boundary()) {
                        if (!split_folded_patch()) {
                            deposit_visible_canonical_tessellation(
                                h0,
                                h1,
                                [=](double) { return s0; },
                                [=](double) { return s1; },
                                exact_volume);
                        }
                    }
                }
            }
        };
        append_patch(
            energy_left, energy_right, phase_left, phase_right, 0);
        // Close each column after its visible/hidden partition.  Excessive
        // geometric normalization is an error; admitted normalization scales
        // the entire sparse column and its moments consistently.
        const double accounted_volume = column.visible_volume
            + column.hidden_volume + column.escaped_volume;
        if (!(accounted_volume > 0.0)) {
            throw std::runtime_error("Canonical column has no represented volume.");
        }
        column.geometric_scale = column.column_volume / accounted_volume;
        if (std::abs(column.geometric_scale - 1.0)
            > 64.0 * options.mapped_volume_tolerance) {
            std::ostringstream message;
            message
                << "Canonical column quadrature normalization is too large: "
                << "eta_cell=" << eta_cell
                << ", H=[" << std::setprecision(17)
                << energy_left << ',' << energy_right << ']'
                << ", phase_cell=" << phase_cell
                << ", phase=[" << phase_left << ',' << phase_right << ']'
                << ", column_volume=" << column.column_volume
                << ", visible_volume=" << column.visible_volume
                << ", hidden_volume=" << column.hidden_volume
                << ", escaped_volume=" << column.escaped_volume
                << ", geometric_scale=" << column.geometric_scale
                << ", accumulated_patch_excess_error="
                << column.accumulated_patch_excess_error << '.';
            throw std::runtime_error(message.str());
        }
        for (std::size_t entry = first_column_overlap;
             entry < atlas.overlap.size();
             ++entry) {
            atlas.overlap[entry].volume *= column.geometric_scale;
            atlas.overlap[entry].energy_volume *= column.geometric_scale;
            atlas.overlap[entry].energy2_volume *= column.geometric_scale;
            for (double& moment : atlas.overlap[entry].eulerian_p2_moment) {
                moment *= column.geometric_scale;
            }
        }
        column.visible_volume *= column.geometric_scale;
        column.hidden_volume *= column.geometric_scale;
        column.escaped_volume *= column.geometric_scale;
        column.canonical_quadrature_volume *= column.geometric_scale;
        column.residual = column.column_volume
            - column.visible_volume - column.hidden_volume
            - column.escaped_volume;
        const double scale = std::max(1.0, column.column_volume);
        if (std::abs(column.residual)
            > 32.0 * options.overlap_closure_tolerance * scale) {
            throw std::runtime_error(
                "Canonical column visible/hidden/escaped ledger failed closure.");
        }
    }
    return row_index;
}

struct CanonicalAtlasConstructionDiagnostics {
    int eta_rows_requested = 0;
    int eta_rows_built = 0;
    int energy_rows_built = 0;
    int eta_rows_without_closed_orbits = 0;
    double minimum_energy_span = std::numeric_limits<double>::infinity();
    double maximum_energy_span = 0.0;
    double maximum_patch_scale_error = 0.0;
    double maximum_column_residual = 0.0;
    double total_canonical_quadrature_volume = 0.0;
    double maximum_column_quadrature_fraction = 0.0;
};

struct CanonicalAtlasConstructionResult {
    CanonicalOrbitAtlas atlas;
    CanonicalAtlasConstructionDiagnostics diagnostics;
};

// Weighted collision sampling has explicit population, mass-representation,
// phase-lift and locality controls.  Continuous products retain their actual
// collision coordinates until a later requested atlas absorption.
struct CanonicalNTCCollisionOptions {
    std::size_t marker_budget = 262144;
    int markers_per_occupied_cell = 32;
    unsigned int threads = 1;
    double maximum_roundoff_negative_mass_fraction = 1.0e-13;
    double maximum_unrepresented_mass_fraction = 1.0e-10;
    double hidden_central_mass = 0.0;
    double maximum_source_row_geometry_fraction = 0.5;
    double maximum_lift_phase_interpolation_cell_error = 0.1;
    int virtual_collision_batches = 1;
    // Raise the number of independent, fractionally weighted NTC replicas in
    // shells whose physical trial rate is small.  Zero disables adaptation.
    double minimum_expected_trials_per_shell = 0.0;
    int maximum_virtual_collision_batches = 4096;
    // Reserve this fraction of the nonmandatory marker budget for orbit rows
    // that currently contribute mass inside core_importance_radius.
    double core_importance_radius = 0.0;
    double core_importance_fraction = 0.0;
    bool sample_eulerian_cell_centers = true;
    // Keep collided packets as continuous phase-space carriers.  This avoids
    // a velocity-grid remap after every collision event; the canonical atlas
    // continues to represent only mass that has not yet collided.
    bool retain_collision_products_as_detached_carriers = true;
    // Withdraw incoming packets from their owned canonical cells immediately
    // and retain the positive outgoing trajectories until the next collision
    // evaluation. The signed-carrier restart representation includes a
    // readable negative vector.
    bool retain_collision_increment_as_signed_carriers = false;
    CanonicalOpenCarrierCompressionOptions open_compression;
    WeightedPICCollisionOptions collision;
    WeightedPICPhaseRemapOptions remap;

    CanonicalNTCCollisionOptions() {
        collision.fail_on_nonlocal_shell = true;
        collision.at_most_one_collision_per_marker = true;
        collision.radial_cell_shells = false;
        collision.input_ordered_by_radius = true;
        collision.maximum_substeps = 1;
        remap.maximum_carrier_mass_fraction = 1.0e-8;
    }
};

// Cache one source cell's orbit only for its exact mean H, eta, hidden mass
// and phase resolution; changed physical fields invalidate the outer cache.
struct CanonicalNTCSourceOrbitCacheEntry {
    bool valid = false;
    double energy = 0.0;
    double eta = 0.0;
    double hidden_central_mass = 0.0;
    int phase_cells = 0;
    HamiltonianOrbitPhaseGeometry geometry;
};

struct CanonicalNTCSourceOrbitCache {
    std::vector<CanonicalNTCSourceOrbitCacheEntry> cell;
    std::vector<double> radial_potential;

    void clear() {
        cell.clear();
        radial_potential.clear();
    }
};

struct CanonicalNTCCollisionState {
    WeightedPICSamplerState marker_sampler;
    WeightedPICCollisionSamplerState collision_sampler;
    WeightedPICCollisionWorkspace workspace;
    // Exact source-orbit tables are reusable while the potential and atlas
    // topology remain fixed.  Cell mean energies are checked on every use,
    // and field rebuilds clear the cache explicitly.
    CanonicalNTCSourceOrbitCache source_orbit_cache;
    // A post-collision marker can be physically outside the closed-orbit
    // atlas.  Retain its full mass and continuous state explicitly.
    std::vector<WeightedPICMarker> open_marker;
    // Representable post-collision packets remain continuous between field
    // rebuilds.  They participate in subsequent local NTC collisions and are
    // advanced by the same frozen-field characteristic integrator.
    std::vector<WeightedPICMarker> detached_marker;
    // Deferred collision packets.  New events append outgoing trajectories to
    // signed_increment_positive after withdrawing their incoming ownership
    // from the atlas.  signed_increment_negative stores only packets read from
    // older two-sign restarts; it enters moments with a minus sign until it can
    // be materialized safely.
    std::vector<WeightedPICMarker> signed_increment_positive;
    std::vector<WeightedPICMarker> signed_increment_negative;
    // Canonical samples use the low half of the id range.  Open collision
    // packets use the high half so every retained packet has an independent
    // orientation stream, including across virtual batches and restarts.
    std::uint64_t next_open_marker_id = UINT64_C(0x8000000000000000);
};

struct CanonicalNTCCollisionDiagnostics {
    CanonicalProjectionLedger atlas_to_eulerian;
    CanonicalProjectionLedger eulerian_to_atlas;
    WeightedPICInitializationDiagnostics initialization;
    WeightedPICPhaseRemapDiagnostics remap_before;
    WeightedPICPhaseRemapDiagnostics remap_after;
    WeightedPICCollisionDiagnostics collision;
    CanonicalOpenCarrierCompressionDiagnostics open_compression;
    double physical_mass_before = 0.0;
    double physical_mass_after = 0.0;
    double physical_mass_residual = 0.0;
    double physical_radial_momentum_residual = 0.0;
    double physical_kinetic_energy_residual = 0.0;
    double positivity_roundoff_mass = 0.0;
    double positivity_limited_pair_mass = 0.0;
    double positivity_limited_pair_fraction = 0.0;
    double maximum_relative_density_noise = 0.0;
    double maximum_relative_energy_flux_noise = 0.0;
    double maximum_relative_collision_radial_energy_batch_error = 0.0;
    double maximum_relative_collision_heat_flux_batch_error = 0.0;
    std::size_t collision_batch_a_events = 0;
    std::size_t collision_batch_b_events = 0;
    std::size_t canonical_canonical_events = 0;
    std::size_t canonical_open_events = 0;
    std::size_t open_open_events = 0;
    std::size_t active_open_collision_sources = 0;
    std::size_t active_detached_collision_sources = 0;
    double physical_open_mass = 0.0;
    double physical_detached_mass = 0.0;
    double physical_signed_increment_mass = 0.0;
    double signed_increment_materialized_mass_residual = 0.0;
    double signed_increment_materialized_energy_residual = 0.0;
    double signed_increment_materialized_energy2_residual = 0.0;
    std::size_t signed_increment_positive_markers = 0;
    std::size_t signed_increment_negative_markers = 0;
    double physical_high_energy_open_mass = 0.0;
    double unrepresented_eta_mass = 0.0;
    double unrepresented_missing_eta_row_mass = 0.0;
    double unrepresented_energy_mass = 0.0;
    double unrepresented_orbit_geometry_mass = 0.0;
    double source_lift_failure_mass = 0.0;
    double atlas_energy_quantization_error = 0.0;
    double marker_initialization_seconds = 0.0;
    double marker_layout_seconds = 0.0;
    double marker_generation_seconds = 0.0;
    double marker_radial_sort_seconds = 0.0;
    std::size_t maximum_marker_ownership_terms = 0;
    double collision_pair_seconds = 0.0;
    double outgoing_lift_seconds = 0.0;
    std::size_t source_row_geometry_reuses = 0;
    std::size_t source_exact_orbit_uses = 0;
    std::size_t source_exact_orbit_builds = 0;
    std::size_t source_exact_orbit_cache_hits = 0;
    std::size_t source_unique_exact_orbit_cells = 0;
    double effective_marker_number = 0.0;
    std::size_t outgoing_cached_phase_lifts = 0;
    std::size_t outgoing_exact_phase_lifts = 0;
    double maximum_estimated_lift_phase_cell_error = 0.0;
    double physical_collision_timescale =
        std::numeric_limits<double>::infinity();
    double collision_majorant_timescale =
        std::numeric_limits<double>::infinity();
    double maximum_source_energy_row_fraction = 0.0;
    double maximum_source_eta_cell_fraction = 0.0;
    int virtual_collision_batches = 0;
    int minimum_shell_virtual_batches = 0;
    int maximum_shell_virtual_batches = 0;
    std::size_t core_importance_rows = 0;
    std::size_t core_importance_markers = 0;
};

// A lift term assigns positive mass and its actual H to one stored phase cell.
struct CanonicalNTCLiftTerm {
    std::uint32_t canonical_cell = 0;
    double weight = 0.0;
    double energy = 0.0;
};

// Unsupported products remain explicit packets with a classified reason,
// allowing domain limitations to enter the representation budget.
enum class CanonicalNTCOpenReason : std::uint8_t {
    none = 0,
    eta_outside_atlas = 1,
    eta_row_missing = 2,
    energy_above_atlas = 3,
    energy_outside_row = 4,
    orbit_geometry_failure = 5,
};

struct CanonicalNTCMarkerLift {
    std::array<CanonicalNTCLiftTerm, 4> term{};
    int terms = 0;
    double energy = 0.0;
    double represented_energy = 0.0;
    bool open = false;
    CanonicalNTCOpenReason open_reason = CanonicalNTCOpenReason::none;
};

inline bool canonical_ntc_eta_nearly_equal(double first, double second) {
    const double scale = std::max({
        std::abs(first),
        std::abs(second),
        std::numeric_limits<double>::min(),
    });
    return std::abs(first - second)
        <= 1024.0 * std::numeric_limits<double>::epsilon() * scale;
}

// Ownership records physical mass withdrawn from its original atlas cell;
// it follows the marker through sorting and accepted collision proposals.
struct CanonicalNTCOwnershipTerm {
    std::uint32_t canonical_cell = 0;
    double physical_mass = 0.0;
};

// Most markers have one or two source terms.  Inline storage covers those
// cases while overflow retains an arbitrary conservative ownership partition.
struct CanonicalNTCOwnership {
    std::array<CanonicalNTCOwnershipTerm, 2> inline_term{};
    std::uint8_t inline_terms = 0;
    std::vector<CanonicalNTCOwnershipTerm> overflow;

    bool empty() const {
        return inline_terms == 0 && overflow.empty();
    }

    std::size_t size() const {
        return static_cast<std::size_t>(inline_terms) + overflow.size();
    }

    CanonicalNTCOwnershipTerm& back() {
        return overflow.empty()
            ? inline_term[static_cast<std::size_t>(inline_terms - 1)]
            : overflow.back();
    }

    void push_back(CanonicalNTCOwnershipTerm term) {
        if (inline_terms < inline_term.size()) {
            inline_term[static_cast<std::size_t>(inline_terms++)] = term;
        } else {
            overflow.push_back(term);
        }
    }

    template <class Function>
    void for_each(Function&& function) const {
        for (std::size_t q = 0; q < inline_terms; ++q) {
            function(inline_term[q]);
        }
        for (const CanonicalNTCOwnershipTerm& term : overflow) {
            function(term);
        }
    }
};

struct CanonicalNTCMarkerInitialization {
    WeightedPICInitializationResult sample;
    std::vector<CanonicalNTCOwnership> ownership;
    std::size_t source_row_geometry_reuses = 0;
    std::size_t source_exact_orbit_uses = 0;
    std::size_t source_exact_orbit_builds = 0;
    std::size_t source_exact_orbit_cache_hits = 0;
    std::size_t source_unique_exact_orbit_cells = 0;
    double effective_marker_number = 0.0;
    double maximum_source_energy_row_fraction = 0.0;
    double maximum_source_eta_cell_fraction = 0.0;
    std::size_t core_importance_rows = 0;
    std::size_t core_importance_markers = 0;
    double layout_seconds = 0.0;
    double generation_seconds = 0.0;
    double radial_sort_seconds = 0.0;
    std::size_t maximum_ownership_terms = 0;
};

// Locate the effective-potential minimum by bounded golden-section search and
// compare both physical endpoints, supplying turning/action integration scans.
template <class EffectivePotential>
inline double canonical_effective_potential_minimum_radius(
    double left,
    double right,
    EffectivePotential&& effective_potential) {
    if (!(left > 0.0) || !(right > left)) {
        throw std::runtime_error(
            "Invalid interval for the effective-potential minimum.");
    }
    const double domain_left = left;
    const double domain_right = right;
    constexpr double inverse_golden =
        0.618033988749894848204586834365638118;
    double x0 = right - inverse_golden * (right - left);
    double x1 = left + inverse_golden * (right - left);
    double f0 = effective_potential(x0);
    double f1 = effective_potential(x1);
    for (int iteration = 0; iteration < 96; ++iteration) {
        if (f0 < f1) {
            right = x1;
            x1 = x0;
            f1 = f0;
            x0 = right - inverse_golden * (right - left);
            f0 = effective_potential(x0);
        } else {
            left = x0;
            x0 = x1;
            f0 = f1;
            x1 = left + inverse_golden * (right - left);
            f1 = effective_potential(x1);
        }
    }
    double radius = f0 < f1 ? x0 : x1;
    double value = std::min(f0, f1);
    const double left_value = effective_potential(domain_left);
    if (left_value < value) {
        radius = domain_left;
        value = left_value;
    }
    const double right_value = effective_potential(domain_right);
    if (right_value < value) {
        radius = domain_right;
    }
    return radius;
}

// Sum reduced atlas mass; callers apply 8*pi^2 when joining physical packets.
inline double canonical_orbit_atlas_mass(const CanonicalOrbitAtlas& atlas) {
    long double mass = 0.0L;
    for (const CanonicalOrbitRow& row : atlas.row) {
        for (double value : row.phase_mass) {
            mass += value;
        }
    }
    return static_cast<double>(mass);
}

/// Builds weighted collision markers directly from the visible part of the
/// canonical atlas.  Every marker belongs to one visible orbit-phase stratum
/// and owns a disjoint part of that stratum's mass.  The ownership intervals
/// partition the visible atlas mass exactly, so an accepted packet can be
/// withdrawn once without any positivity repair.
template <class PotentialAtRadius>
inline CanonicalNTCMarkerInitialization initialize_canonical_ntc_markers(
    const CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const CanonicalNTCCollisionOptions& options,
    WeightedPICSamplerState& sampler,
    PotentialAtRadius&& potential_at_radius,
    CanonicalNTCSourceOrbitCache* persistent_source_cache = nullptr) {
    const auto initialization_begin = std::chrono::steady_clock::now();
    struct VisibleStratum {
        std::uint32_t canonical_cell = 0;
        std::size_t row = 0;
        double stored_phase_begin = 0.0;
        double stored_phase_end = 0.0;
        long double physical_mass = 0.0L;
    };
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(PI) * static_cast<long double>(PI);
    std::vector<VisibleStratum> stratum;
    stratum.reserve(atlas.column.size() + atlas.row.size());
    std::vector<std::size_t> row_stratum_begin(atlas.row.size(), 0);
    std::vector<std::size_t> row_stratum_end(atlas.row.size(), 0);
    std::vector<long double> row_visible_mass(atlas.row.size(), 0.0L);
    long double total_visible_mass = 0.0L;
    std::size_t occupied_cells = 0;
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        const CanonicalOrbitRow& row = atlas.row[row_index];
        row_stratum_begin[row_index] = stratum.size();
        const int phase_cells = static_cast<int>(row.phase_mass.size());
        if (phase_cells == 0) {
            row_stratum_end[row_index] = stratum.size();
            continue;
        }
        long double row_mass = 0.0L;
        const double visible_phase_end = row.kind
                == CanonicalOrbitRowKind::delayed_inner_periodic
            ? row.hidden_phase_begin
            : 1.0;
        for (int phase_cell = 0; phase_cell < phase_cells; ++phase_cell) {
            const double cell_mass = row.phase_mass[
                static_cast<std::size_t>(phase_cell)];
            if (!(cell_mass > 0.0)) {
                continue;
            }
            bool cell_visible = false;
            const double stored_begin = static_cast<double>(phase_cell)
                / static_cast<double>(phase_cells);
            const double stored_end = static_cast<double>(phase_cell + 1)
                / static_cast<double>(phase_cells);
            for (int image = -1; image <= 1; ++image) {
                const double physical_begin = stored_begin + row.phase_offset
                    + static_cast<double>(image);
                const double physical_end = stored_end + row.phase_offset
                    + static_cast<double>(image);
                const double overlap_begin = std::max(0.0, physical_begin);
                const double overlap_end = std::min(
                    visible_phase_end, physical_end);
                if (!(overlap_end > overlap_begin)) {
                    continue;
                }
                const double segment_begin = overlap_begin - row.phase_offset
                    - static_cast<double>(image);
                const double segment_end = overlap_end - row.phase_offset
                    - static_cast<double>(image);
                const double fraction = (segment_end - segment_begin)
                    * static_cast<double>(phase_cells);
                const long double mass = phase_factor * cell_mass * fraction;
                if (!(mass > 0.0L)) {
                    continue;
                }
                total_visible_mass += mass;
                row_mass += mass;
                stratum.push_back({
                    row.first_cell + static_cast<std::uint32_t>(phase_cell),
                    row_index,
                    segment_begin,
                    segment_end,
                    mass,
                });
                cell_visible = true;
            }
            if (cell_visible) {
                ++occupied_cells;
            }
        }
        row_visible_mass[row_index] = row_mass;
        row_stratum_end[row_index] = stratum.size();
    }
    if (!(total_visible_mass > 0.0L) || stratum.empty()) {
        throw std::runtime_error(
            "Canonical collision sampler found no visible atlas mass.");
    }
    // Cover every nonempty visible stratum before allocating extra samples.
    // Finite marker budgets therefore change statistical weights while keeping
    // support on every represented visible source interval.
    const std::size_t nominal = occupied_cells
        > std::numeric_limits<std::size_t>::max()
                / static_cast<std::size_t>(options.markers_per_occupied_cell)
        ? std::numeric_limits<std::size_t>::max()
        : occupied_cells
            * static_cast<std::size_t>(options.markers_per_occupied_cell);
    const std::size_t marker_count = std::max<std::size_t>(
        2, std::min(nominal, options.marker_budget));
    const std::size_t minimum_marker_count = stratum.size();
    if (marker_count < minimum_marker_count) {
        throw std::runtime_error(
            "Canonical collision marker budget must cover every visible "
            "orbit-phase cell; increase marker_budget.");
    }
    if (sampler.next_counter
        > std::numeric_limits<std::uint64_t>::max() - marker_count - 1) {
        throw std::runtime_error("Canonical collision marker counter overflow.");
    }
    const std::uint64_t epoch = sampler.next_counter;
    // Keep every marker inside one visible orbit-phase stratum.  One marker
    // per stratum guarantees support; the remaining budget follows physical
    // mass and optional core importance.  Weighted NTC samples a pair with a
    // rate proportional to max(m_i,m_j) and transfers min(m_i,m_j), so its
    // expected collision measure remains exactly m_i m_j.
    std::vector<std::size_t> stratum_marker_count(stratum.size(), 1);
    std::vector<long double> stratum_core_mass(stratum.size(), 0.0L);
    std::vector<long double> row_core_mass(atlas.row.size(), 0.0L);
    long double total_core_mass = 0.0L;
    if (options.core_importance_fraction > 0.0
        && options.core_importance_radius > geometry.radial.edge.front()) {
        constexpr int phase_subsamples = 4;
        for (std::size_t row_index = 0;
             row_index < atlas.row.size();
             ++row_index) {
            const CanonicalOrbitRow& row = atlas.row[row_index];
            if (!(row_visible_mass[row_index] > 0.0L)) {
                continue;
            }
            auto effective_potential = [&](double radius) {
                return potential_at_radius(radius) + row.eta / (radius * radius);
            };
            long double core_mass = 0.0L;
            for (std::size_t stratum_index = row_stratum_begin[row_index];
                 stratum_index < row_stratum_end[row_index];
                 ++stratum_index) {
                const VisibleStratum& visible = stratum[stratum_index];
                int core_samples = 0;
                for (int sample = 0; sample < phase_subsamples; ++sample) {
                    const double fraction =
                        (static_cast<double>(sample) + 0.5)
                        / phase_subsamples;
                    const double stored_phase = std::fma(
                        fraction,
                        visible.stored_phase_end
                            - visible.stored_phase_begin,
                        visible.stored_phase_begin);
                    const double physical_phase = hamiltonian_orbit_wrap_phase(
                        stored_phase + row.phase_offset);
                    const auto location = hamiltonian_orbit_location_at_phase_table(
                        row.phase_geometry, physical_phase, effective_potential);
                    core_samples += location.visible
                        && location.state.r <= options.core_importance_radius;
                }
                const long double visible_core_mass = visible.physical_mass
                    * static_cast<long double>(core_samples)
                    / phase_subsamples;
                stratum_core_mass[stratum_index] = visible_core_mass;
                core_mass += visible_core_mass;
            }
            row_core_mass[row_index] = core_mass;
            total_core_mass += core_mass;
        }
    }
    // Allocate extra markers by physical mass, optionally emphasizing core
    // orbits.  Largest fractional remainders close the exact integer budget.
    std::size_t remaining_markers = marker_count - minimum_marker_count;
    std::vector<std::pair<long double, std::size_t>> allocation_remainder;
    allocation_remainder.reserve(stratum.size());
    std::size_t allocated_extra = 0;
    for (std::size_t stratum_index = 0;
         stratum_index < stratum.size();
         ++stratum_index) {
        long double allocation_probability =
            stratum[stratum_index].physical_mass / total_visible_mass;
        if (total_core_mass > 0.0L) {
            allocation_probability =
                (1.0 - options.core_importance_fraction)
                    * allocation_probability
                + options.core_importance_fraction
                    * stratum_core_mass[stratum_index] / total_core_mass;
        }
        const long double ideal = static_cast<long double>(remaining_markers)
            * allocation_probability;
        const std::size_t whole = static_cast<std::size_t>(std::floor(ideal));
        stratum_marker_count[stratum_index] += whole;
        allocated_extra += whole;
        allocation_remainder.emplace_back(ideal - whole, stratum_index);
    }
    if (allocated_extra > remaining_markers) {
        throw std::runtime_error(
            "Canonical stratum marker allocation exceeded its budget.");
    }
    std::stable_sort(
        allocation_remainder.begin(),
        allocation_remainder.end(),
        [](const auto& first, const auto& second) {
            return first.first != second.first
                ? first.first > second.first
                : first.second < second.second;
        });
    for (std::size_t q = 0; q < remaining_markers - allocated_extra; ++q) {
        ++stratum_marker_count[allocation_remainder[q].second];
    }
    std::vector<std::size_t> marker_stratum;
    std::vector<std::size_t> marker_in_stratum;
    marker_stratum.reserve(marker_count);
    marker_in_stratum.reserve(marker_count);
    for (std::size_t stratum_index = 0;
         stratum_index < stratum.size();
         ++stratum_index) {
        const std::size_t count = stratum_marker_count[stratum_index];
        for (std::size_t q = 0; q < count; ++q) {
            marker_stratum.push_back(stratum_index);
            marker_in_stratum.push_back(q);
        }
    }
    if (marker_stratum.size() != marker_count) {
        throw std::runtime_error(
            "Canonical stratum marker allocation failed closure.");
    }
    CanonicalNTCMarkerInitialization result;
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        if (row_core_mass[row_index] > 0.0L) {
            ++result.core_importance_rows;
        }
    }
    for (std::size_t stratum_index = 0;
         stratum_index < stratum.size();
         ++stratum_index) {
        if (stratum_core_mass[stratum_index] > 0.0L) {
            result.core_importance_markers +=
                stratum_marker_count[stratum_index];
        }
    }
    result.sample.markers.reserve(marker_count);
    result.ownership.resize(marker_count);
    // Each sampled packet uses its owned phase cell's mean H and actual eta.
    // These are the moments withdrawn when a collision consumes that packet.
    std::vector<double> cell_mean_energy(atlas.column.size(), 0.0);
    std::vector<double> cell_eta(atlas.column.size(), 0.0);
    for (const CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            cell_mean_energy[cell] = row.phase_mass[q] > 0.0
                ? row.phase_energy_mass[q] / row.phase_mass[q]
                : row.energy;
            cell_eta[cell] = row.eta;
        }
    }
    std::vector<double> row_minimum_radius(
        atlas.row.size(), geometry.radial.edge.front());
    std::map<double, double> minimum_radius_by_eta;
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        const double eta = atlas.row[row_index].eta;
        auto found = minimum_radius_by_eta.find(eta);
        if (found == minimum_radius_by_eta.end()) {
            auto effective_potential = [&](double radius) {
                return potential_at_radius(radius) + eta / (radius * radius);
            };
            found = minimum_radius_by_eta.emplace(
                eta,
                canonical_effective_potential_minimum_radius(
                    geometry.radial.edge.front(),
                    geometry.radial.edge.back(),
                    effective_potential)).first;
        }
        row_minimum_radius[row_index] = found->second;
    }
    if (marker_count
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Canonical marker initialization exceeds the parallel index range.");
    }
    result.sample.markers.resize(marker_count);
    std::vector<unsigned char> reused_row_geometry(marker_count, 0);
    std::vector<unsigned char> built_exact_geometry(marker_count, 0);
    const int marker_threads = static_cast<int>(std::min<std::size_t>(
        options.threads, marker_count));
    std::vector<std::size_t> exact_geometry_builds_by_thread(
        static_cast<std::size_t>(marker_threads), 0);
    std::vector<std::size_t> exact_geometry_hits_by_thread(
        static_cast<std::size_t>(marker_threads), 0);
    std::vector<double> source_energy_fraction(marker_count, 0.0);
    std::vector<double> source_eta_fraction(marker_count, 0.0);
    CanonicalNTCSourceOrbitCache local_source_cache;
    CanonicalNTCSourceOrbitCache& source_cache = persistent_source_cache
        ? *persistent_source_cache : local_source_cache;
    std::vector<double> current_radial_potential(
        geometry.radial.edge.size(), 0.0);
    for (std::size_t q = 0; q < geometry.radial.edge.size(); ++q) {
        current_radial_potential[q] = potential_at_radius(
            geometry.radial.edge[q]);
    }
    if (source_cache.cell.size() != atlas.column.size()
        || source_cache.radial_potential != current_radial_potential) {
        source_cache.cell.clear();
        source_cache.cell.resize(atlas.column.size());
        source_cache.radial_potential = std::move(current_radial_potential);
    }
    // A canonical cell's visible segments and markers are contiguous.  Move
    // each parallel boundary to the next cell so only one task can update a
    // given cache entry; this avoids locks in the hot source-sampling loop.
    std::vector<int> marker_task_begin(
        static_cast<std::size_t>(marker_threads) + 1, 0);
    marker_task_begin.front() = 0;
    marker_task_begin.back() = static_cast<int>(marker_count);
    for (int task = 1; task < marker_threads; ++task) {
        std::size_t boundary = std::max<std::size_t>(
            static_cast<std::size_t>(marker_task_begin[
                static_cast<std::size_t>(task - 1)]),
            marker_count * static_cast<std::size_t>(task)
                / static_cast<std::size_t>(marker_threads));
        while (boundary < marker_count
               && stratum[marker_stratum[boundary]].canonical_cell
                    == stratum[marker_stratum[boundary - 1]].canonical_cell) {
            ++boundary;
        }
        marker_task_begin[static_cast<std::size_t>(task)] =
            static_cast<int>(boundary);
    }
    result.layout_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initialization_begin).count();
    const auto generation_begin = std::chrono::steady_clock::now();
    parallel_for_chunks(
        0,
        marker_threads,
        marker_threads,
        [&](int begin_task, int end_task, int thread_index) {
    for (int task = begin_task; task < end_task; ++task) {
      const int begin_marker = marker_task_begin[static_cast<std::size_t>(task)];
      const int end_marker = marker_task_begin[static_cast<std::size_t>(task + 1)];
      for (int marker_number = begin_marker;
           marker_number < end_marker;
           ++marker_number) {
        const std::size_t marker_index =
            static_cast<std::size_t>(marker_number);
        const std::size_t selected_stratum = marker_stratum[marker_index];
        const VisibleStratum& selected = stratum[selected_stratum];
        const std::size_t local_marker = marker_in_stratum[marker_index];
        const std::size_t markers_in_stratum =
            stratum_marker_count[selected_stratum];
        const long double marker_mass = selected.physical_mass
            / static_cast<long double>(markers_in_stratum);
        // Keep the quadrature point fixed in stored phase.  The row's phase
        // offset transports it exactly in a frozen potential, while fresh
        // marker ids below still randomize tangential collision orientations.
        const double source_quiet = weighted_pic_quiet_coordinate(
            local_marker,
            2,
            sampler.seed,
            0,
            static_cast<std::uint64_t>(selected_stratum),
            0);
        const double selected_local = (
            static_cast<double>(local_marker) + source_quiet)
            / static_cast<double>(markers_in_stratum);
        const double selected_stored_phase = std::fma(
            selected_local,
            selected.stored_phase_end - selected.stored_phase_begin,
            selected.stored_phase_begin);
        const double marker_physical_mass = static_cast<double>(marker_mass);
        result.ownership[marker_index].push_back({
            selected.canonical_cell,
            marker_physical_mass,
        });
        const CanonicalOrbitRow& row = atlas.row[selected.row];
        const double marker_energy = cell_mean_energy[selected.canonical_cell];
        const double marker_eta = cell_eta[selected.canonical_cell];
        const double physical_phase = hamiltonian_orbit_wrap_phase(
            selected_stored_phase + row.phase_offset);
        auto effective_potential = [&](double radius) {
            return potential_at_radius(radius)
                + marker_eta / (radius * radius);
        };
        const double energy_row_fraction = std::abs(marker_energy - row.energy)
            / (row.energy_right - row.energy_left);
        const double eta_cell_fraction = std::abs(marker_eta - row.eta)
            / geometry.eta.width[static_cast<std::size_t>(row.eta_cell)];
        source_energy_fraction[marker_index] = energy_row_fraction;
        source_eta_fraction[marker_index] = eta_cell_fraction;
        auto row_effective_potential = [&](double radius) {
            return potential_at_radius(radius) + row.eta / (radius * radius);
        };
        HamiltonianOrbitPhaseGeometry::Location location =
            hamiltonian_orbit_location_at_phase_table(
                row.phase_geometry,
                physical_phase,
                row_effective_potential);
        const double marker_speed2 = location.visible
            ? 2.0 * (marker_energy - effective_potential(location.state.r))
            : -std::numeric_limits<double>::infinity();
        const double speed_tolerance = 2048.0
            * std::numeric_limits<double>::epsilon()
            * std::max({
                1.0,
                std::abs(marker_energy),
                std::abs(effective_potential(location.state.r)),
            });
        // Reuse representative row radii only inside the allowed H/eta
        // offset and accessible speed range.  Recover w from the packet's H
        // so its sampled kinetic energy follows the owned mean energy.
        bool use_row_orbit =
            energy_row_fraction <= options.maximum_source_row_geometry_fraction
            && eta_cell_fraction <= options.maximum_source_row_geometry_fraction
            && location.visible
            && marker_speed2 >= -speed_tolerance;
        if (use_row_orbit) {
            const bool outgoing = physical_phase
                    * row.phase_geometry.radial_period
                <= row.phase_geometry.visible_outgoing_time;
            const double speed = std::sqrt(std::max(0.0, marker_speed2));
            location.state.w = outgoing ? speed : -speed;
            reused_row_geometry[marker_index] = 1;
        }
        // Larger offsets or inaccessible representative points require a
        // complete orbit at the packet's actual H/eta and inner-return model.
        if (!use_row_orbit) {
            CanonicalNTCSourceOrbitCacheEntry& cache_entry =
                source_cache.cell[selected.canonical_cell];
            const int phase_cells = static_cast<int>(row.phase_mass.size());
            const double boundary_effective = effective_potential(
                geometry.radial.edge.front());
            const bool uses_hidden_return = using_delayed_inner_return(p)
                && marker_energy > boundary_effective;
            const double hidden_mass_key = uses_hidden_return
                ? options.hidden_central_mass : 0.0;
            const bool cache_match = cache_entry.valid
                && cache_entry.energy == marker_energy
                && cache_entry.eta == marker_eta
                && cache_entry.hidden_central_mass
                    == hidden_mass_key
                && cache_entry.phase_cells == phase_cells;
            if (!cache_match) {
                std::vector<double> turning_scan;
                turning_scan.reserve(
                    geometry.radial.edge.size()
                    + geometry.radial.center.size() + 2);
                turning_scan.insert(
                    turning_scan.end(),
                    geometry.radial.edge.begin(),
                    geometry.radial.edge.end());
                turning_scan.insert(
                    turning_scan.end(),
                    geometry.radial.center.begin(),
                    geometry.radial.center.end());
                turning_scan.push_back(row.phase_geometry.pericenter);
                turning_scan.push_back(row.phase_geometry.apocenter);
                turning_scan.push_back(row_minimum_radius[selected.row]);
                std::sort(turning_scan.begin(), turning_scan.end());
                turning_scan.erase(
                    std::unique(
                        turning_scan.begin(),
                        turning_scan.end(),
                        [](double first, double second) {
                            return phase_space_nearly_equal(first, second);
                        }),
                    turning_scan.end());
                double hidden_return_time =
                    std::numeric_limits<double>::quiet_NaN();
                if (uses_hidden_return) {
                    hidden_return_time =
                        delayed_inner_uniform_cell_round_trip_time(
                            std::sqrt(
                                2.0 * (marker_energy - boundary_effective)),
                            std::sqrt(2.0 * marker_eta),
                            options.hidden_central_mass,
                            p);
                }
                try {
                    cache_entry.geometry = build_hamiltonian_orbit_phase_geometry(
                        marker_eta,
                        marker_energy,
                        turning_scan,
                        phase_cells,
                        effective_potential,
                        hidden_return_time);
                } catch (const std::exception& error) {
                    std::ostringstream message;
                    message << "Canonical collision source orbit failed: marker="
                            << marker_index << ", row=" << selected.row
                            << ", eta=" << std::setprecision(17) << marker_eta
                            << ", H=" << marker_energy
                            << ", row_H=" << row.energy
                            << ", H_edges=[" << row.energy_left << ','
                            << row.energy_right << "]"
                            << ", phase=" << physical_phase
                            << ", row_location_r=" << location.state.r
                            << ", marker_speed2=" << marker_speed2
                            << ": " << error.what();
                    throw std::runtime_error(message.str());
                }
                cache_entry.valid = true;
                cache_entry.energy = marker_energy;
                cache_entry.eta = marker_eta;
                cache_entry.hidden_central_mass = hidden_mass_key;
                cache_entry.phase_cells = phase_cells;
                ++exact_geometry_builds_by_thread[
                    static_cast<std::size_t>(thread_index)];
            } else {
                ++exact_geometry_hits_by_thread[
                    static_cast<std::size_t>(thread_index)];
            }
            const HamiltonianOrbitPhaseGeometry& sampled_orbit =
                cache_entry.geometry;
            double sampled_phase = physical_phase;
            const double row_visible_duration =
                2.0 * row.phase_geometry.visible_outgoing_time;
            const double sampled_visible_duration =
                2.0 * sampled_orbit.visible_outgoing_time;
            const double row_physical_time = physical_phase
                * row.phase_geometry.radial_period;
            if (row_visible_duration > 0.0
                && sampled_visible_duration > 0.0) {
                // The delayed central carrier changes the hidden fraction of
                // an orbit across one finite H row.  Preserve position within
                // the visible out-and-back interval when switching from the
                // row geometry to the marker's exact (H,eta) geometry.
                const double visible_fraction = std::clamp(
                    row_physical_time / row_visible_duration, 0.0, 1.0);
                sampled_phase = visible_fraction
                    * sampled_visible_duration / sampled_orbit.radial_period;
                const double last_visible_phase = std::nextafter(
                    sampled_visible_duration / sampled_orbit.radial_period,
                    0.0);
                sampled_phase = std::clamp(
                    sampled_phase, 0.0, last_visible_phase);
            }
            location = hamiltonian_orbit_location_at_phase_table(
                sampled_orbit,
                sampled_phase,
                effective_potential);
            built_exact_geometry[marker_index] = 1;
        }
        if (!location.visible) {
            throw std::runtime_error(
                "Canonical visible marker sampler selected a hidden phase.");
        }
        WeightedPICMarker marker;
        marker.mass = marker_physical_mass;
        marker.r = location.state.r;
        marker.w = location.state.w;
        marker.eta = marker_eta;
        marker.id = epoch + marker_index;
        const int radial_cell = canonical_orbit_containing_cell(
            marker.r, geometry.radial.edge);
        const int velocity_cell = canonical_orbit_containing_cell(
            marker.w, geometry.radial_velocity.edge);
        const int eta_cell = canonical_orbit_containing_cell(
            marker.eta, geometry.eta.edge);
        if (radial_cell < 0 || velocity_cell < 0 || eta_cell < 0) {
            throw std::runtime_error(
                "Canonical collision marker lies outside the collision mesh.");
        }
        marker.source_cell = idx(
            radial_cell, velocity_cell, eta_cell, p);
        result.sample.markers[marker_index] = marker;
      }
    }
        });
    result.generation_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - generation_begin).count();
    result.source_row_geometry_reuses = static_cast<std::size_t>(
        std::count(
            reused_row_geometry.begin(), reused_row_geometry.end(), 1));
    result.source_exact_orbit_uses = static_cast<std::size_t>(
        std::count(
            built_exact_geometry.begin(), built_exact_geometry.end(), 1));
    result.source_exact_orbit_builds = std::accumulate(
        exact_geometry_builds_by_thread.begin(),
        exact_geometry_builds_by_thread.end(),
        std::size_t{0});
    result.source_exact_orbit_cache_hits = std::accumulate(
        exact_geometry_hits_by_thread.begin(),
        exact_geometry_hits_by_thread.end(),
        std::size_t{0});
    std::vector<unsigned char> exact_orbit_cell(atlas.column.size(), 0);
    long double marker_mass2 = 0.0L;
    for (std::size_t marker_index = 0;
         marker_index < marker_count;
         ++marker_index) {
        const double mass = result.sample.markers[marker_index].mass;
        marker_mass2 += static_cast<long double>(mass) * mass;
        if (built_exact_geometry[marker_index] != 0) {
            exact_orbit_cell[stratum[marker_stratum[marker_index]].canonical_cell]
                = 1;
        }
    }
    result.source_unique_exact_orbit_cells = static_cast<std::size_t>(
        std::count(exact_orbit_cell.begin(), exact_orbit_cell.end(), 1));
    result.effective_marker_number = marker_mass2 > 0.0L
        ? static_cast<double>(total_visible_mass * total_visible_mass
            / marker_mass2)
        : 0.0;
    result.maximum_source_energy_row_fraction = *std::max_element(
        source_energy_fraction.begin(), source_energy_fraction.end());
    result.maximum_source_eta_cell_fraction = *std::max_element(
        source_eta_fraction.begin(), source_eta_fraction.end());
    // Before collisions, prove that all samples together own at most the
    // available mass of each stored source cell, including partial visibility.
    std::vector<long double> owned_physical_mass(atlas.column.size(), 0.0L);
    for (const CanonicalNTCOwnership& ownership : result.ownership) {
        ownership.for_each([&](const CanonicalNTCOwnershipTerm& term) {
            if (term.canonical_cell >= owned_physical_mass.size()) {
                throw std::runtime_error(
                    "Canonical marker ownership references an invalid cell.");
            }
            owned_physical_mass[term.canonical_cell] += term.physical_mass;
        });
    }
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        const CanonicalOrbitRow& row = atlas.row[row_index];
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            const long double available = phase_factor * row.phase_mass[q];
            const long double owned = owned_physical_mass[cell];
            const long double scale = std::max(
                std::abs(available), std::abs(owned));
            const long double tolerance = 4096.0L
                * std::numeric_limits<double>::epsilon() * scale;
            if (owned > available + tolerance) {
                std::ostringstream message;
                message.precision(17);
                message
                    << "Canonical marker ownership exceeds its atlas cell; "
                    << "cell=" << cell
                    << ", row=" << row_index
                    << ", phase_cell=" << q
                    << ", owned=" << static_cast<double>(owned)
                    << ", available=" << static_cast<double>(available)
                    << ", phase_offset=" << row.phase_offset
                    << ", hidden_phase_begin=" << row.hidden_phase_begin
                    << ", row_kind=" << static_cast<int>(row.kind);
                throw std::runtime_error(message.str());
            }
        }
    }
    std::vector<std::size_t> order(marker_count, 0);
    std::iota(order.begin(), order.end(), 0);
    const auto sort_begin = std::chrono::steady_clock::now();
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](std::size_t first, std::size_t second) {
            return result.sample.markers[first].r
                < result.sample.markers[second].r;
        });
    // Apply one stable radial permutation to both packets and ownership.
    // Spatial shell sampling can then withdraw the original atlas mass.
    std::vector<WeightedPICMarker> sorted_marker(marker_count);
    std::vector<CanonicalNTCOwnership> sorted_ownership(
        marker_count);
    for (std::size_t q = 0; q < marker_count; ++q) {
        sorted_marker[q] = result.sample.markers[order[q]];
        sorted_ownership[q] = std::move(result.ownership[order[q]]);
    }
    result.sample.markers = std::move(sorted_marker);
    result.ownership = std::move(sorted_ownership);
    result.radial_sort_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sort_begin).count();
    for (const auto& ownership : result.ownership) {
        result.maximum_ownership_terms = std::max(
            result.maximum_ownership_terms, ownership.size());
    }
    sampler.next_counter += marker_count + 1;
    result.sample.diagnostics.occupied_source_cells = occupied_cells;
    result.sample.diagnostics.marker_count = marker_count;
    result.sample.diagnostics.source_mass =
        static_cast<double>(total_visible_mass);
    result.sample.diagnostics.represented_mass =
        weighted_pic_total_marker_mass(result.sample.markers);
    return result;
}

struct CanonicalNTCPhaseInterpolation {
    bool accepted = false;
    double phase = 0.0;
    double estimated_cell_error = std::numeric_limits<double>::infinity();
};

// Estimate physical phase from nearby energy-row maps in the smooth orbit
// interior.  Quadratic/linear disagreement in phase-cell units controls reuse;
// turning-point or unsupported interpolation falls back to the exact orbit.
template <class EffectivePotential>
inline CanonicalNTCPhaseInterpolation
canonical_ntc_interpolate_physical_phase(
    const CanonicalOrbitAtlas& atlas,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t destination_row,
    double energy,
    double radius,
    double radial_velocity,
    double maximum_cell_error,
    EffectivePotential&& effective_potential) {
    struct Sample {
        double energy = 0.0;
        double phase = 0.0;
        std::size_t row = 0;
    };
    std::array<Sample, 5> sample{};
    std::size_t sample_count = 0;
    const std::size_t scan_begin = destination_row > row_begin + 2
        ? destination_row - 2 : row_begin;
    const std::size_t scan_end = std::min(row_end, destination_row + 3);
    for (std::size_t q = scan_begin; q < scan_end; ++q) {
        const CanonicalOrbitRow& row = atlas.row[q];
        const double span = row.phase_geometry.apocenter
            - row.phase_geometry.pericenter;
        if (!(span > 0.0)) {
            continue;
        }
        const double radial_fraction =
            (radius - row.phase_geometry.pericenter) / span;
        // The phase derivative diverges at a turning point.  Exact geometry is
        // retained there; cached interpolation is used in the smooth interior.
        if (!(radial_fraction > 1.0e-5)
            || !(radial_fraction < 1.0 - 1.0e-5)) {
            continue;
        }
        const double phase = hamiltonian_orbit_phase_of_state(
            row.phase_geometry,
            CharacteristicState{radius, radial_velocity},
            effective_potential);
        if (std::isfinite(phase)) {
            sample[sample_count++] = {row.energy, phase, q};
        }
    }
    if (sample_count < 3) {
        return {};
    }
    std::sort(
        sample.begin(), sample.begin() + static_cast<std::ptrdiff_t>(sample_count),
        [&](const Sample& first, const Sample& second) {
            return std::abs(first.energy - energy)
                < std::abs(second.energy - energy);
        });
    std::sort(
        sample.begin(), sample.begin() + 3,
        [](const Sample& first, const Sample& second) {
            return first.energy < second.energy;
        });
    if (!(sample.front().energy <= energy)
        || !(energy <= sample.back().energy)) {
        return {};
    }
    const double x0 = sample[0].energy;
    const double x1 = sample[1].energy;
    const double x2 = sample[2].energy;
    if (!(x0 < x1) || !(x1 < x2)) {
        return {};
    }
    const double y0 = sample[0].phase;
    const double y1 = sample[1].phase;
    const double y2 = sample[2].phase;
    const double quadratic =
        y0 * (energy - x1) * (energy - x2) / ((x0 - x1) * (x0 - x2))
        + y1 * (energy - x0) * (energy - x2) / ((x1 - x0) * (x1 - x2))
        + y2 * (energy - x0) * (energy - x1) / ((x2 - x0) * (x2 - x1));
    const bool lower_pair = energy <= x1;
    const double left_x = lower_pair ? x0 : x1;
    const double right_x = lower_pair ? x1 : x2;
    const double left_y = lower_pair ? y0 : y1;
    const double right_y = lower_pair ? y1 : y2;
    const double linear = std::fma(
        (energy - left_x) / (right_x - left_x),
        right_y - left_y,
        left_y);
    const int phase_cells = static_cast<int>(
        atlas.row[destination_row].phase_mass.size());
    const double cell_error = std::abs(quadratic - linear) * phase_cells;
    if (!std::isfinite(quadratic) || quadratic < 0.0 || quadratic >= 1.0
        || !(cell_error <= maximum_cell_error)) {
        return {};
    }
    return {true, quadratic, cell_error};
}

/// Locates one collision marker in the current orbit atlas.  The marker's
/// exact eta and H select the finite-volume row.  Its radial phase is computed
/// from checked neighboring cached orbit maps, with an exact-orbit fallback
/// near turning points.  The row phase offset converts physical phase back to
/// the stored, co-moving phase coordinate.
template <class PotentialAtRadius>
inline CanonicalNTCMarkerLift canonical_ntc_lift_marker(
    const CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    double radius,
    const BPFAVec3& velocity,
    double hidden_central_mass,
    const std::vector<double>& base_turning_scan,
    double maximum_phase_cell_error,
    std::size_t* cached_phase_lifts,
    std::size_t* exact_phase_lifts,
    double* maximum_estimated_phase_cell_error,
    PotentialAtRadius&& potential_at_radius) {
    CanonicalNTCMarkerLift lift;
    const double radius2 = radius * radius;
    const double eta = 0.5 * radius2
        * (velocity.y * velocity.y + velocity.z * velocity.z);
    const double potential = potential_at_radius(radius);
    lift.energy = potential + 0.5 * bp_fa_dot(velocity, velocity);
    if (!(eta >= 0.0) || !std::isfinite(eta)
        || !std::isfinite(lift.energy)) {
        throw std::runtime_error(
            "Canonical collision lift received a non-finite marker state.");
    }
    const int containing_eta_cell = canonical_orbit_containing_cell(
        eta, geometry.eta.edge);
    const bool has_eta_plane_index = !atlas.eta_plane_coordinate.empty();
    const bool valid_eta_plane_index = has_eta_plane_index
        && atlas.eta_plane_row_offset.size()
            == atlas.eta_plane_coordinate.size() + 1
        && atlas.eta_plane_cell.size() == atlas.eta_plane_coordinate.size()
        && std::is_sorted(
            atlas.eta_plane_coordinate.begin(),
            atlas.eta_plane_coordinate.end());
    if (containing_eta_cell < 0
        || geometry.eta.center.size() != static_cast<std::size_t>(p.NL)
        || atlas.eta_row_offset.size()
            != static_cast<std::size_t>(p.NL + 1)
        || (has_eta_plane_index && !valid_eta_plane_index)) {
        lift.open = true;
        lift.open_reason = CanonicalNTCOpenReason::eta_outside_atlas;
        return lift;
    }
    std::array<int, 2> eta_plane{};
    std::array<double, 2> eta_weight{};
    int eta_planes = 1;
    const auto& eta_coordinate = has_eta_plane_index
        ? atlas.eta_plane_coordinate : geometry.eta.center;
    const double eta_support_scale = std::max({
        1.0,
        std::abs(eta),
        std::abs(eta_coordinate.front()),
        std::abs(eta_coordinate.back()),
    });
    const double eta_support_tolerance = 1024.0
        * std::numeric_limits<double>::epsilon() * eta_support_scale;
    if (eta < eta_coordinate.front() - eta_support_tolerance
        || eta > eta_coordinate.back() + eta_support_tolerance) {
        // Positive eta interpolation requires support inside the actual plane
        // convex hull.  An outside point remains an explicit carrier with
        // its physical angular momentum and represented mass.
        lift.open = true;
        lift.open_reason = CanonicalNTCOpenReason::eta_outside_atlas;
        return lift;
    }
    const auto eta_lower = std::lower_bound(
        eta_coordinate.begin(), eta_coordinate.end(), eta);
    int snapped_eta_plane = -1;
    auto consider_eta_plane = [&](std::ptrdiff_t candidate) {
        if (candidate < 0
            || candidate >= static_cast<std::ptrdiff_t>(eta_coordinate.size())) {
            return;
        }
        const double coordinate = eta_coordinate[
            static_cast<std::size_t>(candidate)];
        if (canonical_ntc_eta_nearly_equal(eta, coordinate)) {
            snapped_eta_plane = static_cast<int>(candidate);
        }
    };
    consider_eta_plane(eta_lower - eta_coordinate.begin());
    consider_eta_plane(eta_lower - eta_coordinate.begin() - 1);
    if (snapped_eta_plane >= 0) {
        eta_plane[0] = snapped_eta_plane;
        eta_weight[0] = 1.0;
    } else if (eta_lower == eta_coordinate.begin()) {
        eta_plane[0] = 0;
        eta_weight[0] = 1.0;
    } else if (eta_lower == eta_coordinate.end()) {
        eta_plane[0] = static_cast<int>(eta_coordinate.size()) - 1;
        eta_weight[0] = 1.0;
    } else {
        eta_planes = 2;
        eta_plane[1] = static_cast<int>(
            eta_lower - eta_coordinate.begin());
        eta_plane[0] = eta_plane[1] - 1;
        const double left_eta = eta_coordinate[
            static_cast<std::size_t>(eta_plane[0])];
        const double right_eta = eta_coordinate[
            static_cast<std::size_t>(eta_plane[1])];
        eta_weight[1] = (eta - left_eta) / (right_eta - left_eta);
        eta_weight[0] = 1.0 - eta_weight[1];
    }

    // Positive interpolation between actual eta planes preserves eta.  At
    // fixed r/w, H depends linearly on eta, so plane energies preserve the
    // original H under the same weights.
    for (int plane = 0; plane < eta_planes; ++plane) {
        if (!(eta_weight[plane] > 0.0)) {
            continue;
        }
        const int plane_index = eta_plane[plane];
        const int eta_cell = has_eta_plane_index
            ? atlas.eta_plane_cell[static_cast<std::size_t>(plane_index)]
            : plane_index;
        const double plane_eta = eta_coordinate[
            static_cast<std::size_t>(plane_index)];
        const double plane_energy = potential
            + 0.5 * velocity.x * velocity.x + plane_eta / radius2;
        const std::size_t row_begin = has_eta_plane_index
            ? atlas.eta_plane_row_offset[static_cast<std::size_t>(plane_index)]
            : atlas.eta_row_offset[static_cast<std::size_t>(eta_cell)];
        const std::size_t row_end = has_eta_plane_index
            ? atlas.eta_plane_row_offset[static_cast<std::size_t>(plane_index + 1)]
            : atlas.eta_row_offset[static_cast<std::size_t>(eta_cell + 1)];
        if (row_begin >= row_end || row_end > atlas.row.size()) {
            lift.open = true;
            lift.open_reason = CanonicalNTCOpenReason::eta_row_missing;
            return lift;
        }
        const auto begin = atlas.row.begin()
            + static_cast<std::ptrdiff_t>(row_begin);
        const auto end = atlas.row.begin()
            + static_cast<std::ptrdiff_t>(row_end);
        const auto found = std::lower_bound(
            begin,
            end,
            plane_energy,
            [](const CanonicalOrbitRow& row, double energy) {
                return row.energy_right < energy;
            });
        if (found == end) {
            lift.open = true;
            lift.open_reason = CanonicalNTCOpenReason::energy_above_atlas;
            return lift;
        }
        const double energy_scale = std::max({
            1.0,
            std::abs(plane_energy),
            std::abs(found->energy_left),
            std::abs(found->energy_right),
        });
        const double energy_tolerance = 512.0
            * std::numeric_limits<double>::epsilon() * energy_scale;
        if (plane_energy < found->energy_left - energy_tolerance
            || plane_energy > found->energy_right + energy_tolerance) {
            lift.open = true;
            lift.open_reason = CanonicalNTCOpenReason::energy_outside_row;
            return lift;
        }
        const CanonicalOrbitRow& row = *found;
        if (row.eta_cell != eta_cell
            || (has_eta_plane_index && row.eta_plane != plane_index)
            || row.phase_mass.empty()) {
            throw std::runtime_error(
                "Canonical collision lift found an inconsistent eta-row index.");
        }
        auto effective_potential = [&](double value) {
            return potential_at_radius(value) + plane_eta / (value * value);
        };
        const std::size_t destination_row = static_cast<std::size_t>(
            found - atlas.row.begin());
        const CanonicalNTCPhaseInterpolation interpolated =
            canonical_ntc_interpolate_physical_phase(
                atlas,
                row_begin,
                row_end,
                destination_row,
                plane_energy,
                radius,
                velocity.x,
                maximum_phase_cell_error,
                effective_potential);
        double physical_phase = interpolated.phase;
        if (interpolated.accepted) {
            if (cached_phase_lifts != nullptr) {
                ++*cached_phase_lifts;
            }
            if (maximum_estimated_phase_cell_error != nullptr) {
                *maximum_estimated_phase_cell_error = std::max(
                    *maximum_estimated_phase_cell_error,
                    interpolated.estimated_cell_error);
            }
        } else {
            // The exact-orbit fallback augments the turning scan with this
            // product radius and potential minimum.  Build that extra scan
            // only when the checked cached interpolation requires it.
            std::vector<double> turning_scan = base_turning_scan;
            const auto insertion = std::lower_bound(
                turning_scan.begin(), turning_scan.end(), radius);
            if (insertion == turning_scan.end()
                || !phase_space_nearly_equal(*insertion, radius)) {
                turning_scan.insert(insertion, radius);
            }
            const double minimum_radius =
                canonical_effective_potential_minimum_radius(
                    row.phase_geometry.pericenter,
                    row.phase_geometry.apocenter,
                    effective_potential);
            const auto minimum_insertion = std::lower_bound(
                turning_scan.begin(), turning_scan.end(), minimum_radius);
            if (minimum_insertion == turning_scan.end()
                || !phase_space_nearly_equal(
                    *minimum_insertion, minimum_radius)) {
                turning_scan.insert(minimum_insertion, minimum_radius);
            }
            double hidden_return_time =
                std::numeric_limits<double>::quiet_NaN();
            if (using_delayed_inner_return(p)) {
                const double boundary_effective = effective_potential(
                    geometry.radial.edge.front());
                if (plane_energy > boundary_effective) {
                    const double boundary_speed = std::sqrt(
                        2.0 * (plane_energy - boundary_effective));
                    hidden_return_time =
                        delayed_inner_uniform_cell_round_trip_time(
                            boundary_speed,
                            std::sqrt(2.0 * plane_eta),
                            hidden_central_mass,
                            p);
                }
            }
            try {
                const HamiltonianOrbitExactPhase exact =
                    hamiltonian_orbit_exact_phase_of_state(
                    plane_energy,
                    CharacteristicState{radius, velocity.x},
                    turning_scan,
                    effective_potential,
                    hidden_return_time,
                    48);
                physical_phase = exact.phase;
            } catch (const std::runtime_error&) {
                lift.open = true;
                lift.open_reason = CanonicalNTCOpenReason::orbit_geometry_failure;
                return lift;
            }
            if (exact_phase_lifts != nullptr) {
                ++*exact_phase_lifts;
            }
        }
        const double stored_phase = hamiltonian_orbit_wrap_phase(
            physical_phase - row.phase_offset);
        const int phase_cells = static_cast<int>(row.phase_mass.size());
        // Deposit to the two periodic phase-cell centers in stored phase.
        // Combined with two eta planes this gives at most four positive terms.
        const double centered_coordinate = stored_phase * phase_cells - 0.5;
        const int raw_left = static_cast<int>(std::floor(centered_coordinate));
        const double fraction = centered_coordinate
            - std::floor(centered_coordinate);
        auto wrapped_cell = [phase_cells](int value) {
            value %= phase_cells;
            return value < 0 ? value + phase_cells : value;
        };
        const int left = wrapped_cell(raw_left);
        const int right = wrapped_cell(raw_left + 1);
        auto append = [&](int phase_cell, double phase_weight) {
            if (!(phase_weight > 0.0)) {
                return;
            }
            if (lift.terms >= static_cast<int>(lift.term.size())) {
                throw std::runtime_error(
                    "Canonical collision lift exceeded its CIC stencil.");
            }
            lift.term[static_cast<std::size_t>(lift.terms++)] = {
                row.first_cell + static_cast<std::uint32_t>(phase_cell),
                eta_weight[plane] * phase_weight,
                plane_energy,
            };
        };
        append(left, 1.0 - fraction);
        append(right, fraction);
        lift.represented_energy += eta_weight[plane] * plane_energy;
    }
    if (lift.terms == 0) {
        throw std::runtime_error("Canonical collision lift produced no CIC terms.");
    }
    long double weight_sum = 0.0L;
    for (int term = 0; term < lift.terms; ++term) {
        weight_sum += static_cast<long double>(
            lift.term[static_cast<std::size_t>(term)].weight);
    }
    const long double weight_tolerance = 2048.0L
        * std::numeric_limits<double>::epsilon();
    if (std::abs(weight_sum - 1.0L) > weight_tolerance) {
        throw std::runtime_error(
            "Canonical collision CIC weights do not form a partition of unity.");
    }
    // Normalize roundoff in the partition, then independently check the
    // weighted H against the original continuous collision product.
    long double represented_energy = 0.0L;
    long double represented_energy_condition = 0.0L;
    for (int term = 0; term < lift.terms; ++term) {
        CanonicalNTCLiftTerm& entry =
            lift.term[static_cast<std::size_t>(term)];
        entry.weight = static_cast<double>(
            static_cast<long double>(entry.weight) / weight_sum);
        const long double contribution =
            static_cast<long double>(entry.weight) * entry.energy;
        represented_energy += contribution;
        represented_energy_condition += std::abs(contribution);
    }
    const long double represented_energy_scale = std::max({
        1.0L,
        std::abs(static_cast<long double>(lift.energy)),
        std::abs(represented_energy),
        represented_energy_condition,
    });
    if (std::abs(represented_energy - lift.energy)
        > 4096.0L * std::numeric_limits<double>::epsilon()
            * represented_energy_scale) {
        std::ostringstream message;
        message.precision(17);
        message
            << "Canonical collision eta interpolation did not preserve H; "
            << "r=" << radius
            << ", eta=" << eta
            << ", H=" << lift.energy
            << ", represented_H=" << static_cast<double>(represented_energy)
            << ", difference="
            << static_cast<double>(represented_energy - lift.energy)
            << ", condition_scale="
            << static_cast<double>(represented_energy_condition)
            << ", eta_planes=" << eta_planes
            << ", eta0=" << eta_coordinate[
                static_cast<std::size_t>(eta_plane[0])]
            << ", weight0=" << eta_weight[0];
        if (eta_planes == 2) {
            message
                << ", eta1=" << eta_coordinate[
                    static_cast<std::size_t>(eta_plane[1])]
                << ", weight1=" << eta_weight[1];
        }
        throw std::runtime_error(message.str());
    }
    lift.represented_energy = static_cast<double>(represented_energy);
    return lift;
}

struct CanonicalDetachedAbsorptionDiagnostics {
    double physical_mass_before = 0.0;
    double physical_mass_after = 0.0;
    double physical_mass_residual = 0.0;
    double absorbed_mass = 0.0;
    double retained_hidden_mass = 0.0;
    double retained_escaped_mass = 0.0;
    double promoted_open_mass = 0.0;
    double energy_moment_residual = 0.0;
    std::size_t absorbed_markers = 0;
    std::size_t retained_hidden_markers = 0;
    std::size_t retained_escaped_markers = 0;
    std::size_t promoted_open_markers = 0;
    std::size_t cached_phase_lifts = 0;
    std::size_t exact_phase_lifts = 0;
    double maximum_estimated_phase_cell_error = 0.0;
};

struct CanonicalSignedIncrementMaterializationDiagnostics {
    double represented_mass_before = 0.0;
    double represented_mass_after = 0.0;
    double mass_residual = 0.0;
    double hamiltonian_residual = 0.0;
    double hamiltonian2_residual = 0.0;
    double positive_mass_materialized = 0.0;
    double negative_mass_materialized = 0.0;
    double positive_mass_retained = 0.0;
    double negative_mass_retained = 0.0;
    double positive_open_mass_promoted = 0.0;
    std::size_t positive_markers_materialized = 0;
    std::size_t negative_markers_materialized = 0;
    std::size_t positive_markers_retained = 0;
    std::size_t negative_markers_retained = 0;
    std::size_t positive_open_markers_promoted = 0;
    std::size_t cached_phase_lifts = 0;
    std::size_t exact_phase_lifts = 0;
    double positivity_fallback_mass = 0.0;
    std::size_t positivity_fallback_markers = 0;
    std::array<double, 6> retained_open_mass_by_reason{};
    std::array<std::size_t, 6> retained_open_markers_by_reason{};
    std::array<double, 6> promoted_open_mass_by_reason{};
    std::array<std::size_t, 6> promoted_open_markers_by_reason{};
    double maximum_estimated_phase_cell_error = 0.0;
    CanonicalOpenCarrierCompressionDiagnostics positive_compression;
    CanonicalOpenCarrierCompressionDiagnostics negative_compression;
    double elapsed_seconds = 0.0;
};

/// Deposits the transported signed collision perturbation onto the canonical
/// finite-volume atlas.  Incoming and outgoing packets are accumulated in one
/// transaction, so cancellation occurs before positivity is checked.  Hidden
/// and escaped packets remain explicit until they return or leave the domain.
template <class PotentialAtRadius>
inline CanonicalSignedIncrementMaterializationDiagnostics
canonical_materialize_signed_collision_increment(
    CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    double hidden_central_mass,
    double maximum_phase_cell_error,
    unsigned int threads,
    const CanonicalOpenCarrierCompressionOptions& compression_options,
    CanonicalNTCCollisionState& state,
    PotentialAtRadius&& potential_at_radius) {
    if (!(maximum_phase_cell_error > 0.0)
        || !std::isfinite(maximum_phase_cell_error)
        || threads == 0) {
        throw std::runtime_error(
            "Invalid canonical signed-increment materialization request.");
    }
    CanonicalSignedIncrementMaterializationDiagnostics diagnostics;
    if (state.signed_increment_positive.empty()
        && state.signed_increment_negative.empty()) {
        return diagnostics;
    }
    // Stage both signs and the atlas moments.  A failed lift or positivity
    // check must leave the caller's distribution and packet lists unchanged.
    std::vector<WeightedPICMarker> staged_positive =
        state.signed_increment_positive;
    std::vector<WeightedPICMarker> staged_negative =
        state.signed_increment_negative;
    const auto materialization_begin = std::chrono::steady_clock::now();
    CanonicalOpenCarrierCompressionOptions positive_compression =
        compression_options;
    positive_compression.threads = threads;
    positive_compression.preserve_source_cell = false;
    diagnostics.positive_compression = compress_canonical_open_carriers(
        staged_positive,
        geometry,
        positive_compression,
        potential_at_radius);
    CanonicalOpenCarrierCompressionOptions negative_compression =
        compression_options;
    negative_compression.threads = threads;
    negative_compression.preserve_source_cell = true;
    diagnostics.negative_compression = compress_canonical_open_carriers(
        staged_negative,
        geometry,
        negative_compression,
        potential_at_radius);
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(PI) * static_cast<long double>(PI);
    std::vector<std::uint32_t> cell_to_row(
        atlas.column.size(), std::numeric_limits<std::uint32_t>::max());
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        const CanonicalOrbitRow& row = atlas.row[row_index];
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            cell_to_row[row.first_cell + q] =
                static_cast<std::uint32_t>(row_index);
        }
    }
    std::vector<long double> mass(atlas.column.size(), 0.0L);
    std::vector<long double> energy(atlas.column.size(), 0.0L);
    std::vector<long double> energy2(atlas.column.size(), 0.0L);
    long double atlas_mass_before = 0.0L;
    long double atlas_energy_before = 0.0L;
    long double atlas_energy2_before = 0.0L;
    for (const CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            mass[cell] = row.phase_mass[q];
            energy[cell] = row.phase_energy_mass[q];
            energy2[cell] = row.phase_energy2_mass[q];
            atlas_mass_before += mass[cell];
            atlas_energy_before += energy[cell];
            atlas_energy2_before += energy2[cell];
        }
    }
    std::vector<double> turning_scan;
    turning_scan.reserve(
        geometry.radial.edge.size() + geometry.radial.center.size());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.edge.begin(), geometry.radial.edge.end());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.center.begin(), geometry.radial.center.end());
    std::sort(turning_scan.begin(), turning_scan.end());
    turning_scan.erase(
        std::unique(
            turning_scan.begin(), turning_scan.end(),
            [](double first, double second) {
                return phase_space_nearly_equal(first, second);
            }),
        turning_scan.end());

    struct LiftedSignedMarker {
        CanonicalNTCMarkerLift lift;
        std::size_t cached = 0;
        std::size_t exact = 0;
        double maximum_error = 0.0;
        bool active = false;
    };
    std::vector<WeightedPICMarker> promoted_open;
    long double promoted_open_h = 0.0L;
    long double promoted_open_h2 = 0.0L;
    // Lift active packets in parallel, then combine their signed moments in
    // source order.  Positive unsupported products become explicit open mass;
    // negative unsupported holes remain signed until an admissible transfer.
    auto materialize_sign = [&](std::vector<WeightedPICMarker>& source,
                                int sign,
                                double& materialized_mass,
                                double& retained_mass,
                                std::size_t& materialized_markers,
                                std::size_t& retained_markers,
                                long double& signed_marker_h,
                                long double& signed_marker_h2) {
        std::vector<LiftedSignedMarker> lifted(source.size());
        if (source.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "Canonical signed increment exceeds the parallel index range.");
        }
        const int lift_threads = static_cast<int>(std::min<std::size_t>(
            threads, std::max<std::size_t>(1, source.size())));
        parallel_for_chunks(
            0,
            static_cast<int>(source.size()),
            lift_threads,
            [&](int begin, int end, int) {
                for (int number = begin; number < end; ++number) {
                    const std::size_t index = static_cast<std::size_t>(number);
                    const WeightedPICMarker& marker = source[index];
                    validate_weighted_pic_marker(
                        marker, "Canonical signed collision increment");
                    if (marker.state != WeightedPICMarkerState::active) {
                        continue;
                    }
                    const double vt = std::sqrt(2.0 * marker.eta)
                        / std::max(marker.r, 1.0e-300);
                    LiftedSignedMarker& output = lifted[index];
                    output.active = true;
                    output.lift = canonical_ntc_lift_marker(
                        atlas,
                        geometry,
                        p,
                        marker.r,
                        BPFAVec3{marker.w, vt, 0.0},
                        hidden_central_mass,
                        turning_scan,
                        maximum_phase_cell_error,
                        &output.cached,
                        &output.exact,
                        &output.maximum_error,
                        potential_at_radius);
                }
            });
        std::vector<WeightedPICMarker> retained;
        retained.reserve(source.size());
        for (std::size_t index = 0; index < source.size(); ++index) {
            const WeightedPICMarker& marker = source[index];
            const long double marker_h = potential_at_radius(marker.r)
                + 0.5L * marker.w * marker.w
                + marker.eta / std::max(
                    static_cast<long double>(marker.r) * marker.r,
                    1.0e-300L);
            signed_marker_h += sign * marker.mass * marker_h;
            signed_marker_h2 += sign * marker.mass * marker_h * marker_h;
            const LiftedSignedMarker& output = lifted[index];
            if (!output.active) {
                retained.push_back(marker);
                retained_mass += marker.mass;
                ++retained_markers;
                continue;
            }
            if (output.lift.open) {
                const std::size_t reason = static_cast<std::size_t>(
                    output.lift.open_reason);
                if (reason == 0
                    || reason
                        >= diagnostics.retained_open_mass_by_reason.size()) {
                    throw std::runtime_error(
                        "Canonical signed collision lift retained an "
                        "unclassified active marker.");
                }
                if (sign > 0) {
                    promoted_open.push_back(marker);
                    diagnostics.positive_open_mass_promoted += marker.mass;
                    ++diagnostics.positive_open_markers_promoted;
                    diagnostics.promoted_open_mass_by_reason[reason] +=
                        marker.mass;
                    ++diagnostics.promoted_open_markers_by_reason[reason];
                    promoted_open_h += marker.mass * marker_h;
                    promoted_open_h2 += marker.mass * marker_h * marker_h;
                } else {
                    retained.push_back(marker);
                    retained_mass += marker.mass;
                    ++retained_markers;
                    diagnostics.retained_open_mass_by_reason[reason] +=
                        marker.mass;
                    ++diagnostics.retained_open_markers_by_reason[reason];
                }
                continue;
            }
            diagnostics.cached_phase_lifts += output.cached;
            diagnostics.exact_phase_lifts += output.exact;
            diagnostics.maximum_estimated_phase_cell_error = std::max(
                diagnostics.maximum_estimated_phase_cell_error,
                output.maximum_error);
            // A negative hole must leave positive mass and admissible H/H^2
            // moments in every destination row.  Otherwise withdraw from its
            // recorded source cell with that cell's mean moments and report it.
            bool use_positivity_fallback = false;
            if (sign < 0) {
                for (int q = 0; q < output.lift.terms; ++q) {
                    const CanonicalNTCLiftTerm& term =
                        output.lift.term[static_cast<std::size_t>(q)];
                    const long double removed = marker.mass * term.weight
                        / phase_factor;
                    const long double candidate =
                        mass[term.canonical_cell] - removed;
                    const long double tolerance = 16384.0L
                        * std::numeric_limits<double>::epsilon()
                        * std::max(
                            1.0L,
                            std::abs(mass[term.canonical_cell]));
                    if (candidate < -tolerance) {
                        use_positivity_fallback = true;
                        break;
                    }
                    if (candidate <= tolerance) {
                        use_positivity_fallback = true;
                        break;
                    }
                    if (candidate > tolerance) {
                        const long double candidate_energy =
                            energy[term.canonical_cell]
                            - removed * term.energy;
                        const long double candidate_energy2 =
                            energy2[term.canonical_cell]
                            - removed * term.energy * term.energy;
                        const std::uint32_t row_index =
                            cell_to_row[term.canonical_cell];
                        if (row_index
                            == std::numeric_limits<std::uint32_t>::max()) {
                            throw std::runtime_error(
                                "Canonical signed collision cell has no row.");
                        }
                        const CanonicalOrbitRow& row = atlas.row[row_index];
                        const long double mean_energy =
                            candidate_energy / candidate;
                        const long double mean_energy2 =
                            candidate_energy2 / candidate;
                        const long double support_scale = std::max({
                            1.0L,
                            std::abs(mean_energy),
                            std::abs(static_cast<long double>(row.energy_left)),
                            std::abs(static_cast<long double>(row.energy_right)),
                        });
                        const long double support_tolerance = 32768.0L
                            * std::numeric_limits<double>::epsilon()
                            * support_scale;
                        if (mean_energy < row.energy_left - support_tolerance
                            || mean_energy
                                > row.energy_right + support_tolerance) {
                            use_positivity_fallback = true;
                            break;
                        }
                        const long double variance = candidate_energy2
                            * candidate
                            - candidate_energy * candidate_energy;
                        const long double variance_tolerance = 32768.0L
                            * std::numeric_limits<double>::epsilon()
                            * std::max(
                                1.0L,
                                std::abs(candidate_energy2 * candidate));
                        if (variance < -variance_tolerance) {
                            use_positivity_fallback = true;
                            break;
                        }
                        const long double secant_upper =
                            (static_cast<long double>(row.energy_left)
                             + static_cast<long double>(row.energy_right))
                                * mean_energy
                            - static_cast<long double>(row.energy_left)
                                * row.energy_right;
                        const long double second_scale = std::max({
                            1.0L,
                            std::abs(mean_energy2),
                            std::abs(secant_upper),
                        });
                        const long double second_tolerance = 32768.0L
                            * std::numeric_limits<double>::epsilon()
                            * second_scale;
                        if (mean_energy2
                            > secant_upper + second_tolerance) {
                            use_positivity_fallback = true;
                            break;
                        }
                    }
                }
            }
            if (use_positivity_fallback) {
                if (marker.source_cell >= mass.size()) {
                    throw std::runtime_error(
                        "Canonical signed collision hole lost its source cell.");
                }
                const long double reduced = -marker.mass / phase_factor;
                const long double candidate =
                    mass[marker.source_cell] + reduced;
                const long double tolerance = 16384.0L
                    * std::numeric_limits<double>::epsilon()
                    * std::max(1.0L, std::abs(mass[marker.source_cell]));
                if (candidate < -tolerance) {
                    throw std::runtime_error(
                        "Canonical signed collision source cell violates "
                        "positivity; reduce the collision interval.");
                }
                if (!(mass[marker.source_cell] > 0.0L)) {
                    throw std::runtime_error(
                        "Canonical signed collision source cell is empty.");
                }
                const long double source_mean_h =
                    energy[marker.source_cell] / mass[marker.source_cell];
                const long double source_mean_h2 =
                    energy2[marker.source_cell] / mass[marker.source_cell];
                mass[marker.source_cell] += reduced;
                energy[marker.source_cell] += reduced * source_mean_h;
                energy2[marker.source_cell] +=
                    reduced * source_mean_h2;
                diagnostics.positivity_fallback_mass += marker.mass;
                ++diagnostics.positivity_fallback_markers;
            } else {
                for (int q = 0; q < output.lift.terms; ++q) {
                    const CanonicalNTCLiftTerm& term =
                        output.lift.term[static_cast<std::size_t>(q)];
                    const long double reduced = sign * marker.mass
                        * term.weight / phase_factor;
                    mass[term.canonical_cell] += reduced;
                    energy[term.canonical_cell] += reduced * term.energy;
                    energy2[term.canonical_cell] +=
                        reduced * term.energy * term.energy;
                }
            }
            materialized_mass += marker.mass;
            ++materialized_markers;
        }
        source = std::move(retained);
    };

    long double signed_marker_h_before = 0.0L;
    long double signed_marker_h2_before = 0.0L;
    const long double signed_mass_before =
        weighted_pic_total_marker_mass(state.signed_increment_positive)
        - weighted_pic_total_marker_mass(state.signed_increment_negative);
    materialize_sign(
        staged_positive,
        +1,
        diagnostics.positive_mass_materialized,
        diagnostics.positive_mass_retained,
        diagnostics.positive_markers_materialized,
        diagnostics.positive_markers_retained,
        signed_marker_h_before,
        signed_marker_h2_before);
    materialize_sign(
        staged_negative,
        -1,
        diagnostics.negative_mass_materialized,
        diagnostics.negative_mass_retained,
        diagnostics.negative_markers_materialized,
        diagnostics.negative_markers_retained,
        signed_marker_h_before,
        signed_marker_h2_before);

    long double atlas_mass_after = 0.0L;
    long double atlas_energy_after = 0.0L;
    long double atlas_energy2_after = 0.0L;
    for (std::size_t cell = 0; cell < mass.size(); ++cell) {
        const long double scale = std::max(1.0L, std::abs(mass[cell]));
        const long double tolerance = 16384.0L
            * std::numeric_limits<double>::epsilon() * scale;
        if (mass[cell] < -tolerance) {
            throw std::runtime_error(
                "Canonical signed collision increment violates positivity; "
                "reduce the collision interval or increase phase resolution.");
        }
        if (mass[cell] < 0.0L) {
            mass[cell] = 0.0L;
            energy[cell] = 0.0L;
            energy2[cell] = 0.0L;
        }
        if (mass[cell] > 0.0L) {
            const std::uint32_t row_index = cell_to_row[cell];
            if (row_index == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "Canonical signed collision cell has no row.");
            }
            const CanonicalOrbitRow& row = atlas.row[row_index];
            const long double mean_energy = energy[cell] / mass[cell];
            const long double mean_energy2 = energy2[cell] / mass[cell];
            const long double support_scale = std::max({
                1.0L,
                std::abs(mean_energy),
                std::abs(static_cast<long double>(row.energy_left)),
                std::abs(static_cast<long double>(row.energy_right)),
            });
            const long double support_tolerance = 32768.0L
                * std::numeric_limits<double>::epsilon() * support_scale;
            if (mean_energy < row.energy_left - support_tolerance
                || mean_energy > row.energy_right + support_tolerance) {
                throw std::runtime_error(
                    "Canonical signed collision increment moved the mean "
                    "Hamiltonian outside its row.");
            }
            const long double variance = energy2[cell] * mass[cell]
                - energy[cell] * energy[cell];
            const long double variance_tolerance = 32768.0L
                * std::numeric_limits<double>::epsilon()
                * std::max(1.0L, std::abs(energy2[cell] * mass[cell]));
            if (variance < -variance_tolerance) {
                throw std::runtime_error(
                    "Canonical signed collision increment produced a negative "
                    "Hamiltonian variance.");
            }
            const long double secant_upper =
                (static_cast<long double>(row.energy_left)
                 + static_cast<long double>(row.energy_right)) * mean_energy
                - static_cast<long double>(row.energy_left) * row.energy_right;
            const long double second_scale = std::max({
                1.0L, std::abs(mean_energy2), std::abs(secant_upper)});
            const long double second_tolerance = 32768.0L
                * std::numeric_limits<double>::epsilon() * second_scale;
            if (mean_energy2 > secant_upper + second_tolerance) {
                throw std::runtime_error(
                    "Canonical signed collision increment moved the second "
                    "Hamiltonian moment outside its row.");
            }
        }
        atlas_mass_after += mass[cell];
        atlas_energy_after += energy[cell];
        atlas_energy2_after += energy2[cell];
    }
    // Join atlas, retained signed packets and promoted open products in
    // common physical mass/H/H^2 ledgers.  The mass gate is enforced here;
    // Hamiltonian residuals remain explicit diagnostics for the caller.
    long double retained_h = 0.0L;
    long double retained_h2 = 0.0L;
    auto retained_moments = [&](const std::vector<WeightedPICMarker>& source,
                                int sign) {
        for (const WeightedPICMarker& marker : source) {
            const long double h = potential_at_radius(marker.r)
                + 0.5L * marker.w * marker.w
                + marker.eta / std::max(
                    static_cast<long double>(marker.r) * marker.r,
                    1.0e-300L);
            retained_h += sign * marker.mass * h;
            retained_h2 += sign * marker.mass * h * h;
        }
    };
    retained_moments(staged_positive, +1);
    retained_moments(staged_negative, -1);
    const long double signed_mass_after =
        weighted_pic_total_marker_mass(staged_positive)
        - weighted_pic_total_marker_mass(staged_negative);
    diagnostics.represented_mass_before = static_cast<double>(
        phase_factor * atlas_mass_before + signed_mass_before);
    diagnostics.represented_mass_after = static_cast<double>(
        phase_factor * atlas_mass_after + signed_mass_after
        + diagnostics.positive_open_mass_promoted);
    diagnostics.mass_residual = diagnostics.represented_mass_after
        - diagnostics.represented_mass_before;
    diagnostics.hamiltonian_residual = static_cast<double>(
        phase_factor * (atlas_energy_after - atlas_energy_before)
        + retained_h + promoted_open_h - signed_marker_h_before);
    diagnostics.hamiltonian2_residual = static_cast<double>(
        phase_factor * (atlas_energy2_after - atlas_energy2_before)
        + retained_h2 + promoted_open_h2 - signed_marker_h2_before);
    const double mass_scale = std::max(
        1.0, std::abs(diagnostics.represented_mass_before));
    if (std::abs(diagnostics.mass_residual)
        > 32768.0 * std::numeric_limits<double>::epsilon() * mass_scale) {
        throw std::runtime_error(
            "Canonical signed collision materialization changed total mass.");
    }
    for (CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            row.phase_mass[q] = static_cast<double>(mass[cell]);
            row.phase_energy_mass[q] = static_cast<double>(energy[cell]);
            row.phase_energy2_mass[q] = static_cast<double>(energy2[cell]);
        }
    }
    state.signed_increment_positive = std::move(staged_positive);
    state.signed_increment_negative = std::move(staged_negative);
    state.open_marker.insert(
        state.open_marker.end(),
        std::make_move_iterator(promoted_open.begin()),
        std::make_move_iterator(promoted_open.end()));
    diagnostics.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - materialization_begin).count();
    return diagnostics;
}

/// Reabsorbs active post-collision carriers into the canonical atlas at a
/// reconstruction boundary.  Collisions can therefore retain their exact
/// local phase-space points between field updates without allowing a growing
/// particle component to replace the finite-volume distribution.  Mass and
/// the first two Hamiltonian moments are deposited with the same partition of
/// unity used by direct collision lifts.  Hidden and escaped carriers remain
/// explicit because they do not have an active point inside the atlas.
template <class PotentialAtRadius>
inline CanonicalDetachedAbsorptionDiagnostics
canonical_absorb_active_detached_markers(
    CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    double hidden_central_mass,
    double maximum_phase_cell_error,
    unsigned int threads,
    CanonicalNTCCollisionState& state,
    PotentialAtRadius&& potential_at_radius) {
    if (!(maximum_phase_cell_error > 0.0)
        || !std::isfinite(maximum_phase_cell_error)
        || threads == 0) {
        throw std::runtime_error(
            "Invalid canonical detached-carrier absorption request.");
    }
    CanonicalDetachedAbsorptionDiagnostics diagnostics;
    if (state.detached_marker.empty()) {
        return diagnostics;
    }
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(PI) * static_cast<long double>(PI);
    auto represented_mass = [&]() {
        long double result = phase_factor * canonical_orbit_atlas_mass(atlas);
        for (const WeightedPICMarker& marker : state.open_marker) {
            result += marker.mass;
        }
        for (const WeightedPICMarker& marker : state.detached_marker) {
            result += marker.mass;
        }
        return result;
    };
    diagnostics.physical_mass_before = static_cast<double>(
        represented_mass());

    std::vector<double> turning_scan;
    turning_scan.reserve(
        geometry.radial.edge.size() + geometry.radial.center.size());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.edge.begin(),
        geometry.radial.edge.end());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.center.begin(),
        geometry.radial.center.end());
    std::sort(turning_scan.begin(), turning_scan.end());
    turning_scan.erase(
        std::unique(
            turning_scan.begin(),
            turning_scan.end(),
            [](double first, double second) {
                return phase_space_nearly_equal(first, second);
            }),
        turning_scan.end());

    struct MarkerLift {
        CanonicalNTCMarkerLift lift;
        std::size_t cached = 0;
        std::size_t exact = 0;
        double maximum_error = 0.0;
        bool active = false;
    };
    std::vector<MarkerLift> lift(state.detached_marker.size());
    if (lift.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Canonical detached-carrier absorption exceeds the parallel index range.");
    }
    const int lift_threads = static_cast<int>(std::min<std::size_t>(
        threads, std::max<std::size_t>(1, lift.size())));
    parallel_for_chunks(
        0,
        static_cast<int>(lift.size()),
        lift_threads,
        [&](int begin, int end, int) {
            for (int marker_number = begin;
                 marker_number < end;
                 ++marker_number) {
                const std::size_t marker_index =
                    static_cast<std::size_t>(marker_number);
                const WeightedPICMarker& marker =
                    state.detached_marker[marker_index];
                validate_weighted_pic_marker(
                    marker, "Canonical detached absorption source");
                if (marker.state != WeightedPICMarkerState::active
                    || !(marker.mass > 0.0)) {
                    continue;
                }
                if (!(marker.r > 0.0)) {
                    throw std::runtime_error(
                        "An active detached carrier has non-positive radius.");
                }
                const double tangential_speed2 =
                    2.0 * marker.eta / (marker.r * marker.r);
                if (!(tangential_speed2 >= 0.0)
                    || !std::isfinite(tangential_speed2)) {
                    throw std::runtime_error(
                        "An active detached carrier has invalid angular momentum.");
                }
                MarkerLift& output = lift[marker_index];
                output.active = true;
                output.lift = canonical_ntc_lift_marker(
                    atlas,
                    geometry,
                    p,
                    marker.r,
                    BPFAVec3{
                        marker.w,
                        std::sqrt(tangential_speed2),
                        0.0},
                    hidden_central_mass,
                    turning_scan,
                    maximum_phase_cell_error,
                    &output.cached,
                    &output.exact,
                    &output.maximum_error,
                    potential_at_radius);
            }
        });

    std::vector<WeightedPICMarker> retained_detached;
    retained_detached.reserve(state.detached_marker.size());
    std::vector<WeightedPICMarker> promoted_open;
    promoted_open.reserve(state.detached_marker.size());
    std::vector<std::uint32_t> cell_to_row(
        atlas.column.size(), std::numeric_limits<std::uint32_t>::max());
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        const CanonicalOrbitRow& row = atlas.row[row_index];
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            cell_to_row[row.first_cell + q] =
                static_cast<std::uint32_t>(row_index);
        }
    }
    // Absorb active admissible products through their positive lift terms.
    // Hidden/escaped products keep their boundary state, and unsupported
    // active products move to the explicit open population.
    long double absorbed_energy_mass = 0.0L;
    long double deposited_energy_mass = 0.0L;
    for (std::size_t marker_index = 0;
         marker_index < state.detached_marker.size();
         ++marker_index) {
        WeightedPICMarker marker = state.detached_marker[marker_index];
        if (!(marker.mass > 0.0)) {
            continue;
        }
        if (marker.state == WeightedPICMarkerState::delayed_inner) {
            retained_detached.push_back(marker);
            diagnostics.retained_hidden_mass += marker.mass;
            ++diagnostics.retained_hidden_markers;
            continue;
        }
        if (marker.state == WeightedPICMarkerState::escaped) {
            retained_detached.push_back(marker);
            diagnostics.retained_escaped_mass += marker.mass;
            ++diagnostics.retained_escaped_markers;
            continue;
        }
        const MarkerLift& output = lift[marker_index];
        if (!output.active) {
            throw std::runtime_error(
                "An active detached carrier was not processed for absorption.");
        }
        diagnostics.cached_phase_lifts += output.cached;
        diagnostics.exact_phase_lifts += output.exact;
        diagnostics.maximum_estimated_phase_cell_error = std::max(
            diagnostics.maximum_estimated_phase_cell_error,
            output.maximum_error);
        if (output.lift.open) {
            promoted_open.push_back(marker);
            diagnostics.promoted_open_mass += marker.mass;
            ++diagnostics.promoted_open_markers;
            continue;
        }
        const long double reduced_mass = marker.mass / phase_factor;
        long double deposited_weight = 0.0L;
        for (int q = 0; q < output.lift.terms; ++q) {
            const CanonicalNTCLiftTerm& term =
                output.lift.term[static_cast<std::size_t>(q)];
            if (term.canonical_cell >= atlas.column.size()) {
                throw std::runtime_error(
                    "Canonical detached absorption selected a missing cell.");
            }
            const long double term_mass = reduced_mass * term.weight;
            const std::uint32_t row_index = cell_to_row[term.canonical_cell];
            if (row_index == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "Canonical detached absorption could not locate its row.");
            }
            CanonicalOrbitRow& row = atlas.row[row_index];
            const std::size_t phase_cell = term.canonical_cell
                - row.first_cell;
            row.phase_mass[phase_cell] += static_cast<double>(term_mass);
            row.phase_energy_mass[phase_cell] += static_cast<double>(
                term_mass * term.energy);
            row.phase_energy2_mass[phase_cell] += static_cast<double>(
                term_mass * term.energy * term.energy);
            deposited_weight += term.weight;
            deposited_energy_mass += marker.mass * term.weight * term.energy;
        }
        const long double weight_tolerance = 4096.0L
            * std::numeric_limits<double>::epsilon();
        if (std::abs(deposited_weight - 1.0L) > weight_tolerance) {
            throw std::runtime_error(
                "Canonical detached absorption weights failed closure.");
        }
        absorbed_energy_mass += marker.mass * output.lift.energy;
        diagnostics.absorbed_mass += marker.mass;
        ++diagnostics.absorbed_markers;
    }
    state.detached_marker = std::move(retained_detached);
    state.open_marker.insert(
        state.open_marker.end(),
        promoted_open.begin(),
        promoted_open.end());
    diagnostics.energy_moment_residual = static_cast<double>(
        deposited_energy_mass - absorbed_energy_mass);
    diagnostics.physical_mass_after = static_cast<double>(represented_mass());
    diagnostics.physical_mass_residual =
        diagnostics.physical_mass_after - diagnostics.physical_mass_before;
    const double mass_scale = std::max(
        1.0, diagnostics.physical_mass_before);
    if (std::abs(diagnostics.physical_mass_residual)
        > 2048.0 * std::numeric_limits<double>::epsilon() * mass_scale) {
        throw std::runtime_error(
            "Canonical detached absorption changed represented mass.");
    }
    const double energy_scale = std::max(
        1.0, std::abs(static_cast<double>(absorbed_energy_mass)));
    if (std::abs(diagnostics.energy_moment_residual)
        > 4096.0 * std::numeric_limits<double>::epsilon() * energy_scale) {
        throw std::runtime_error(
            "Canonical detached absorption changed its Hamiltonian moment.");
    }
    return diagnostics;
}

namespace canonical_ntc_detail {

// Preserve the representation owning an incoming packet so accepted events
// withdraw from the correct atlas, detached or open mass ledger.
enum class SourceKind : std::uint8_t {
    canonical,
    detached,
    open,
};

struct SourceRef {
    SourceKind kind = SourceKind::canonical;
    std::size_t index = 0;
};

// A proposal carries incoming/outgoing velocities and a fractional physical
// packet mass.  Final depletion acceptance is evaluated in deterministic order.
struct WeightedEvent {
    std::size_t first_source = 0;
    std::size_t second_source = 0;
    double packet_mass = 0.0;
    int proposal_replicas = 1;
    double depletion_uniform = 0.5;
    BPFAVec3 first_before{};
    BPFAVec3 second_before{};
    BPFAVec3 first_after{};
    BPFAVec3 second_after{};
};

struct PairSampler {
    std::vector<std::size_t> local_by_weight;
    std::vector<long double> cumulative_pair_weight;
    long double pair_weight_sum = 0.0L;
};

struct ShellBatchResult {
    std::uint64_t trial_pairs = 0;
    std::uint64_t accepted_pairs = 0;
    std::uint64_t skipped_already_collided_trials = 0;
    double represented_event_mass = 0.0;
    double radial_momentum_error = 0.0;
    double kinetic_energy_error = 0.0;
    double max_pair_momentum_error = 0.0;
    double max_pair_kinetic_energy_error = 0.0;
    std::vector<WeightedEvent> event;
};

// Use exact spherical shell volume for density.  A reference pair RMS speed
// estimates mean free path; the maximum speed supplies a separate rigorous
// proposal-rate majorant over every pair in the shell.
inline WeightedPICCollisionShell describe_weighted_shell(
    const std::vector<WeightedPICMarker>& marker,
    const WeightedPICRadialMesh& mesh,
    std::size_t begin,
    std::size_t end,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    double inner_edge = std::numeric_limits<double>::quiet_NaN(),
    double outer_edge = std::numeric_limits<double>::quiet_NaN()) {
    if (!(begin < end) || end > marker.size()) {
        throw std::runtime_error("Invalid canonical weighted collision shell.");
    }
    WeightedPICCollisionShell shell;
    shell.begin = begin;
    shell.end = end;
    shell.inner_edge = std::isfinite(inner_edge)
        ? inner_edge
        : (begin == 0
            ? mesh.edge.front()
            : weighted_pic_collision_detail::radius_between(
                marker[begin - 1].r, marker[begin].r));
    shell.outer_edge = std::isfinite(outer_edge)
        ? outer_edge
        : (end == marker.size()
            ? mesh.edge.back()
            : weighted_pic_collision_detail::radius_between(
                marker[end - 1].r, marker[end].r));
    if (!(shell.outer_edge > shell.inner_edge)) {
        throw std::runtime_error(
            "Canonical weighted collision shell has non-positive volume.");
    }
    shell.volume = (4.0 * PI / 3.0)
        * (std::pow(shell.outer_edge, 3) - std::pow(shell.inner_edge, 3));
    long double mass = 0.0L;
    long double mass_speed2 = 0.0L;
    for (std::size_t q = begin; q < end; ++q) {
        const double vt = std::sqrt(2.0 * marker[q].eta)
            / std::max(marker[q].r, 1.0e-300);
        const double speed = std::hypot(marker[q].w, vt);
        mass += marker[q].mass;
        mass_speed2 += static_cast<long double>(marker[q].mass)
            * speed * speed;
        shell.maximum_speed = std::max(shell.maximum_speed, speed);
    }
    shell.density = static_cast<double>(mass / shell.volume);
    const double rms = mass > 0.0L
        ? std::sqrt(static_cast<double>(mass_speed2 / mass))
        : 0.0;
    const double sigma =
        weighted_pic_collision_detail::total_cross_section_per_mass(
            std::max(std::sqrt(2.0) * rms, 1.0e-14), p, options);
    if (shell.density > 0.0 && sigma > 0.0) {
        shell.mean_free_path = 1.0 / (shell.density * sigma);
    }
    shell.majorant_u_sigma =
        weighted_pic_collision_detail::u_sigma_majorant(
            2.0 * shell.maximum_speed, p, options);
    return shell;
}

// Split wide occupancy shells when both children can retain the required
// minimum population.  A later option-controlled gate rejects residual
// width/mean-free-path violations.
inline void append_weighted_shell_recursive(
    const std::vector<WeightedPICMarker>& marker,
    const WeightedPICRadialMesh& mesh,
    std::size_t begin,
    std::size_t end,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    std::vector<WeightedPICCollisionShell>& shell) {
    const WeightedPICCollisionShell candidate = describe_weighted_shell(
        marker, mesh, begin, end, p, options);
    const std::size_t count = end - begin;
    const std::size_t middle = begin + count / 2;
    const bool separable = middle > begin && middle < end
        && marker[middle - 1].r < marker[middle].r;
    if (candidate.outer_edge - candidate.inner_edge > candidate.mean_free_path
        && count >= 2 * options.minimum_shell_markers
        && separable) {
        append_weighted_shell_recursive(
            marker, mesh, begin, middle, p, options, shell);
        append_weighted_shell_recursive(
            marker, mesh, middle, end, p, options, shell);
        return;
    }
    shell.push_back(candidate);
}

// Choose actual radial mesh cells or variable-occupancy shells.  Equal-radius
// packets stay together so every constructed shell has a meaningful volume.
inline void build_weighted_shells(
    const std::vector<WeightedPICMarker>& marker,
    const WeightedPICRadialMesh& mesh,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    WeightedPICCollisionWorkspace& workspace) {
    weighted_pic_collision_detail::validate_options(options, p);
    workspace.active_by_radius.resize(marker.size());
    std::iota(
        workspace.active_by_radius.begin(),
        workspace.active_by_radius.end(),
        std::size_t{0});
    workspace.shells.clear();
    if (marker.empty()) {
        return;
    }
    if (options.radial_cell_shells) {
        for (std::size_t begin = 0; begin < marker.size();) {
            const int cell = canonical_orbit_containing_cell(
                marker[begin].r, mesh.edge);
            if (cell < 0) {
                throw std::runtime_error(
                    "Canonical collision source lies outside the radial mesh.");
            }
            std::size_t end = begin + 1;
            while (end < marker.size()
                   && canonical_orbit_containing_cell(marker[end].r, mesh.edge)
                        == cell) {
                ++end;
            }
            workspace.shells.push_back(describe_weighted_shell(
                marker,
                mesh,
                begin,
                end,
                p,
                options,
                mesh.edge[static_cast<std::size_t>(cell)],
                mesh.edge[static_cast<std::size_t>(cell + 1)]));
            begin = end;
        }
        return;
    }
    for (std::size_t begin = 0; begin < marker.size();) {
        std::size_t end = std::min(
            marker.size(), begin + options.target_shell_markers);
        while (end < marker.size()
               && marker[end - 1].r == marker[end].r) {
            ++end;
        }
        if (marker.size() - end < options.minimum_shell_markers) {
            end = marker.size();
        }
        append_weighted_shell_recursive(
            marker, mesh, begin, end, p, options, workspace.shells);
        begin = end;
    }
}

// Sort by weight: a high-rank packet has rank lower-weight partners.
// Weighting that rank by its mass proposes each unordered pair in proportion
// to max(m_i,m_j), without storing the quadratic pair list.
inline PairSampler make_pair_sampler(
    const std::vector<WeightedPICMarker>& marker,
    const WeightedPICCollisionShell& shell,
    const std::vector<unsigned char>* included_source = nullptr) {
    PairSampler sampler;
    sampler.local_by_weight.reserve(shell.end - shell.begin);
    for (std::size_t source = shell.begin; source < shell.end; ++source) {
        if (included_source == nullptr
            || (*included_source)[source] != 0) {
            sampler.local_by_weight.push_back(source - shell.begin);
        }
    }
    std::stable_sort(
        sampler.local_by_weight.begin(),
        sampler.local_by_weight.end(),
        [&](std::size_t first, std::size_t second) {
            const WeightedPICMarker& a = marker[shell.begin + first];
            const WeightedPICMarker& b = marker[shell.begin + second];
            return a.mass != b.mass ? a.mass < b.mass : a.id < b.id;
        });
    const std::size_t count = sampler.local_by_weight.size();
    if (count < 2) {
        return sampler;
    }
    sampler.cumulative_pair_weight.reserve(count - 1);
    for (std::size_t rank = 1; rank < count; ++rank) {
        const double maximum_weight = marker[
            shell.begin + sampler.local_by_weight[rank]].mass;
        sampler.pair_weight_sum += static_cast<long double>(rank)
            * maximum_weight;
        sampler.cumulative_pair_weight.push_back(sampler.pair_weight_sum);
    }
    return sampler;
}

// Sum pair majorant rate times dt; random rounding converts its expectation
// into an integer trial count while preserving the fractional part in mean.
inline long double expected_weighted_shell_trials(
    const WeightedPICCollisionShell& shell,
    const PairSampler& pair_sampler,
    double dt) {
    if (!(dt > 0.0) || !(shell.volume > 0.0)
        || !(shell.majorant_u_sigma > 0.0)
        || !(pair_sampler.pair_weight_sum > 0.0L)) {
        return 0.0L;
    }
    return pair_sampler.pair_weight_sum / shell.volume
        * shell.majorant_u_sigma * dt;
}

// Sample weighted elastic pair proposals from the shell's frozen incoming
// population.  Virtual replicas transfer min(m_i,m_j)/replicas per event;
// a later depletion test handles mass already consumed by earlier proposals.
inline ShellBatchResult sample_weighted_shell_batch(
    const std::vector<WeightedPICMarker>& marker,
    const WeightedPICCollisionShell& shell,
    const PairSampler& pair_sampler,
    double dt,
    int virtual_batches,
    int virtual_batch_index,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    std::uint64_t orientation_seed,
    std::uint64_t event_seed) {
    ShellBatchResult result;
    const std::size_t shell_count = shell.end - shell.begin;
    if (pair_sampler.local_by_weight.size() < 2
        || !(dt > 0.0) || !(shell.majorant_u_sigma > 0.0)
        || !(pair_sampler.pair_weight_sum > 0.0L)) {
        return result;
    }
    std::vector<BPFAVec3> velocity(shell_count);
    std::vector<unsigned char> initialized(shell_count, 0);
    auto source_velocity = [&](std::size_t local) -> const BPFAVec3& {
        if (initialized[local] == 0) {
            const WeightedPICMarker& source = marker[shell.begin + local];
            const double vt = std::sqrt(2.0 * source.eta)
                / std::max(source.r, 1.0e-300);
            const double angle = 2.0 * PI * dsmc_splitmix_uniform_open_at(
                orientation_seed, source.id);
            velocity[local] = {
                source.w, vt * std::cos(angle), vt * std::sin(angle)};
            initialized[local] = 1;
        }
        return velocity[local];
    };
    // Interleave replica indices in one quiet sequence so changing shell
    // work preserves a defined deterministic draw ordering.
    auto quiet_unit = [&](std::uint64_t draw, int base, int dimension) {
        const std::uint64_t interleaved_draw = draw
            * static_cast<std::uint64_t>(virtual_batches)
            + static_cast<std::uint64_t>(virtual_batch_index);
        return weighted_pic_quiet_coordinate(
            static_cast<std::size_t>(interleaved_draw),
            base,
            event_seed,
            0,
            static_cast<std::uint64_t>(shell.begin),
            static_cast<std::uint64_t>(dimension));
    };
    const long double expected_trials = expected_weighted_shell_trials(
        shell, pair_sampler, dt);
    if (!(expected_trials > 0.0L) || !std::isfinite(expected_trials)) {
        return result;
    }
    if (expected_trials
        > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::runtime_error(
            "Canonical weighted NTC trial count overflowed.");
    }
    std::uint64_t trial_count = static_cast<std::uint64_t>(
        std::floor(expected_trials));
    const long double fractional_trial = expected_trials - trial_count;
    if (quiet_unit(0, 13, 0) < fractional_trial) {
        ++trial_count;
    }
    for (std::uint64_t trial = 0; trial < trial_count; ++trial) {
        ++result.trial_pairs;
        const long double target = quiet_unit(trial, 2, 1)
            * pair_sampler.pair_weight_sum;
        const auto upper = std::lower_bound(
            pair_sampler.cumulative_pair_weight.begin(),
            pair_sampler.cumulative_pair_weight.end(),
            target);
        const std::size_t high_rank = static_cast<std::size_t>(
            upper - pair_sampler.cumulative_pair_weight.begin()) + 1;
        const std::size_t low_rank = std::min(
            high_rank - 1,
            static_cast<std::size_t>(
                quiet_unit(trial, 3, 2) * high_rank));
        const std::size_t first_local =
            pair_sampler.local_by_weight[low_rank];
        const std::size_t second_local =
            pair_sampler.local_by_weight[high_rank];
        const BPFAVec3 first_before = source_velocity(first_local);
        const BPFAVec3 second_before = source_velocity(second_local);
        const BPFAVec3 relative = bp_fa_sub(first_before, second_before);
        const double relative_speed = bp_fa_norm(relative);
        const double u_sigma = relative_speed
            * weighted_pic_collision_detail::total_cross_section_per_mass(
                relative_speed, p, options);
        if (!(u_sigma > 0.0)) {
            continue;
        }
        if (u_sigma > shell.majorant_u_sigma
            * (1.0 + 64.0 * std::numeric_limits<double>::epsilon())) {
            throw std::runtime_error(
                "Canonical weighted NTC majorant was violated.");
        }
        if (quiet_unit(trial, 5, 3) * shell.majorant_u_sigma >= u_sigma) {
            continue;
        }
        const BPFAVec3 momentum_before = bp_fa_add(
            first_before, second_before);
        const double energy_before = 0.5 * (
            bp_fa_dot(first_before, first_before)
            + bp_fa_dot(second_before, second_before));
        const BPFAVec3 centre = bp_fa_scale(momentum_before, 0.5);
        const double mu =
            weighted_pic_collision_detail::sample_scatter_mu_from_unit(
                relative_speed, p, options, quiet_unit(trial, 7, 4));
        const double psi = 2.0 * PI * quiet_unit(trial, 11, 5);
        BPFAVec3 direction;
        if (options.physics == WeightedPICCollisionPhysics::venus_cross_section
            && bp_fa_kernel_name(p) != "yukawa_born") {
            const double sine = std::sqrt(std::max(0.0, 1.0 - mu * mu));
            direction = {mu, sine * std::cos(psi), sine * std::sin(psi)};
        } else {
            direction = bp_fa_scattered_relative_direction(relative, mu, psi);
        }
        const BPFAVec3 relative_after = bp_fa_scale(
            direction, relative_speed);
        const BPFAVec3 first_after = bp_fa_add(
            centre, bp_fa_scale(relative_after, 0.5));
        const BPFAVec3 second_after = bp_fa_sub(
            centre, bp_fa_scale(relative_after, 0.5));
        // max-weight proposal rate times min-weight packet mass reproduces
        // m_i*m_j/V.  Replica division preserves that expectation when the
        // same physical interval receives several fractional realizations.
        const int proposal_replicas = virtual_batches;
        const double packet_divisor = static_cast<double>(proposal_replicas);
        const double packet_mass = std::min(
            marker[shell.begin + first_local].mass,
            marker[shell.begin + second_local].mass)
            / packet_divisor;
        result.event.push_back({
            shell.begin + first_local,
            shell.begin + second_local,
            packet_mass,
            proposal_replicas,
            quiet_unit(trial, 17, 6),
            first_before,
            second_before,
            first_after,
            second_after,
        });
        ++result.accepted_pairs;
        result.represented_event_mass += packet_mass;
        const BPFAVec3 momentum_after = bp_fa_add(first_after, second_after);
        const double energy_after = 0.5 * (
            bp_fa_dot(first_after, first_after)
            + bp_fa_dot(second_after, second_after));
        const double momentum_error = bp_fa_norm(
            bp_fa_sub(momentum_after, momentum_before));
        const double energy_error = std::abs(energy_after - energy_before);
        result.max_pair_momentum_error = std::max(
            result.max_pair_momentum_error, momentum_error);
        result.max_pair_kinetic_energy_error = std::max(
            result.max_pair_kinetic_energy_error, energy_error);
        result.radial_momentum_error += packet_mass * (
            first_after.x + second_after.x
            - first_before.x - second_before.x);
        result.kinetic_energy_error += packet_mass * (
            energy_after - energy_before);
    }
    return result;
}

}  // namespace canonical_ntc_detail

/// Applies local NTC collisions to a quiet marker sample of the instantaneous
/// distribution.  Canonical and open carriers share one weighted pair law;
/// only accepted pre/post collision packets are lifted into canonical cells.
template <class PotentialAtRadius>
inline CanonicalNTCCollisionDiagnostics apply_canonical_ntc_collision(
    CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    const CanonicalNTCCollisionOptions& options,
    CanonicalNTCCollisionState& state,
    PotentialAtRadius&& potential_at_radius) {
    if (!(dt >= 0.0) || !std::isfinite(dt)
        || options.marker_budget < 2
        || options.markers_per_occupied_cell < 10
        || options.threads == 0
        || !(options.maximum_roundoff_negative_mass_fraction >= 0.0)
        || !std::isfinite(options.maximum_roundoff_negative_mass_fraction)
        || !(options.maximum_unrepresented_mass_fraction >= 0.0)
        || !std::isfinite(options.maximum_unrepresented_mass_fraction)
        || !(options.hidden_central_mass >= 0.0)
        || !std::isfinite(options.hidden_central_mass)
        || !(options.maximum_source_row_geometry_fraction > 0.0)
        || !std::isfinite(options.maximum_source_row_geometry_fraction)
        || !(options.maximum_lift_phase_interpolation_cell_error > 0.0)
        || !std::isfinite(
            options.maximum_lift_phase_interpolation_cell_error)
        || options.virtual_collision_batches < 1
        || options.virtual_collision_batches > 4096
        || !(options.minimum_expected_trials_per_shell >= 0.0)
        || !std::isfinite(options.minimum_expected_trials_per_shell)
        || options.maximum_virtual_collision_batches
            < options.virtual_collision_batches
        || options.maximum_virtual_collision_batches > 4096
        || !(options.core_importance_radius >= 0.0)
        || !std::isfinite(options.core_importance_radius)
        || !(options.core_importance_fraction >= 0.0)
        || !(options.core_importance_fraction <= 1.0)
        || !std::isfinite(options.core_importance_fraction)
        || (options.retain_collision_products_as_detached_carriers
            && options.retain_collision_increment_as_signed_carriers)) {
        throw std::runtime_error("Invalid canonical NTC collision request.");
    }
    const auto has_active_signed_increment = [](const auto& marker) {
        return std::any_of(
            marker.begin(), marker.end(), [](const WeightedPICMarker& value) {
                return value.state == WeightedPICMarkerState::active
                    && value.mass > 0.0;
            });
    };
    if (options.retain_collision_increment_as_signed_carriers
        && (has_active_signed_increment(state.signed_increment_positive)
            || has_active_signed_increment(
                state.signed_increment_negative))) {
        throw std::runtime_error(
            "Canonical signed collision increment must be materialized before "
            "the next collision evaluation.");
    }
    CanonicalNTCCollisionDiagnostics diagnostics;
    if (dt == 0.0 || p.bp_collision_strength == 0.0) {
        return diagnostics;
    }
    diagnostics.open_compression = compress_canonical_open_carriers(
        state.open_marker,
        geometry,
        options.open_compression,
        potential_at_radius);
    Params pic_p = p;
    pic_p.physics_model = "vlasov_poisson";
    const auto initialization_begin = std::chrono::steady_clock::now();
    CanonicalNTCMarkerInitialization canonical_initialization =
        initialize_canonical_ntc_markers(
            atlas,
            geometry,
            pic_p,
            options,
            state.marker_sampler,
            potential_at_radius,
            &state.source_orbit_cache);
    diagnostics.marker_initialization_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initialization_begin).count();
    diagnostics.marker_layout_seconds = canonical_initialization.layout_seconds;
    diagnostics.marker_generation_seconds =
        canonical_initialization.generation_seconds;
    diagnostics.marker_radial_sort_seconds =
        canonical_initialization.radial_sort_seconds;
    diagnostics.maximum_marker_ownership_terms =
        canonical_initialization.maximum_ownership_terms;
    diagnostics.source_row_geometry_reuses =
        canonical_initialization.source_row_geometry_reuses;
    diagnostics.source_exact_orbit_uses =
        canonical_initialization.source_exact_orbit_uses;
    diagnostics.source_exact_orbit_builds =
        canonical_initialization.source_exact_orbit_builds;
    diagnostics.source_exact_orbit_cache_hits =
        canonical_initialization.source_exact_orbit_cache_hits;
    diagnostics.source_unique_exact_orbit_cells =
        canonical_initialization.source_unique_exact_orbit_cells;
    diagnostics.effective_marker_number =
        canonical_initialization.effective_marker_number;
    diagnostics.maximum_source_energy_row_fraction =
        canonical_initialization.maximum_source_energy_row_fraction;
    diagnostics.maximum_source_eta_cell_fraction =
        canonical_initialization.maximum_source_eta_cell_fraction;
    diagnostics.core_importance_rows =
        canonical_initialization.core_importance_rows;
    diagnostics.core_importance_markers =
        canonical_initialization.core_importance_markers;
    WeightedPICInitializationResult& initialized =
        canonical_initialization.sample;
    diagnostics.initialization = initialized.diagnostics;
    const WeightedPICRadialMesh radial_mesh =
        make_weighted_pic_radial_mesh(geometry);
    std::vector<WeightedPICMarker> collision_source;
    std::vector<canonical_ntc_detail::SourceRef> collision_source_ref;
    // Combine visible atlas samples with active explicit products before
    // describing local shells.  Hidden and escaped mass stays outside this
    // collision population while remaining in the gravitational/mass budget.
    const bool has_active_external_source = std::any_of(
            state.detached_marker.begin(),
            state.detached_marker.end(),
            [](const WeightedPICMarker& marker) {
                return marker.state == WeightedPICMarkerState::active
                    && marker.mass > 0.0;
            })
        || std::any_of(
            state.open_marker.begin(),
            state.open_marker.end(),
            [](const WeightedPICMarker& marker) {
                return marker.state == WeightedPICMarkerState::active
                    && marker.mass > 0.0;
            });
    collision_source.reserve(
        initialized.markers.size() + state.detached_marker.size()
        + state.open_marker.size());
    collision_source_ref.reserve(
        initialized.markers.size() + state.detached_marker.size()
        + state.open_marker.size());
    const std::size_t canonical_source_count = initialized.markers.size();
    for (std::size_t q = 0; q < canonical_source_count; ++q) {
        collision_source_ref.push_back({
            canonical_ntc_detail::SourceKind::canonical, q});
    }
    collision_source.insert(
        collision_source.end(),
        initialized.markers.begin(),
        initialized.markers.end());
    for (std::size_t q = 0; q < state.detached_marker.size(); ++q) {
        const WeightedPICMarker& marker = state.detached_marker[q];
        validate_weighted_pic_marker(
            marker, "Canonical detached collision source");
        if (marker.state != WeightedPICMarkerState::active
            || !(marker.mass > 0.0)) {
            continue;
        }
        if (marker.r < radial_mesh.edge.front()
            || marker.r > radial_mesh.edge.back()) {
            throw std::runtime_error(
                "An active detached carrier lies outside the collision mesh.");
        }
        collision_source.push_back(marker);
        collision_source_ref.push_back({
            canonical_ntc_detail::SourceKind::detached, q});
        ++diagnostics.active_detached_collision_sources;
    }
    for (std::size_t q = 0; q < state.open_marker.size(); ++q) {
        const WeightedPICMarker& marker = state.open_marker[q];
        validate_weighted_pic_marker(marker, "Canonical open collision source");
        if (marker.state != WeightedPICMarkerState::active
            || !(marker.mass > 0.0)) {
            continue;
        }
        if (marker.r < radial_mesh.edge.front()
            || marker.r > radial_mesh.edge.back()) {
            throw std::runtime_error(
                "An active canonical open carrier lies outside the collision mesh.");
        }
        collision_source.push_back(marker);
        collision_source_ref.push_back({
            canonical_ntc_detail::SourceKind::open, q});
        ++diagnostics.active_open_collision_sources;
    }
    if (has_active_external_source) {
        std::vector<std::size_t> source_order(collision_source.size(), 0);
        std::iota(source_order.begin(), source_order.end(), std::size_t{0});
        std::stable_sort(
            source_order.begin(),
            source_order.end(),
            [&](std::size_t first, std::size_t second) {
                const WeightedPICMarker& a = collision_source[first];
                const WeightedPICMarker& b = collision_source[second];
                if (a.r != b.r) {
                    return a.r < b.r;
                }
                if (a.id != b.id) {
                    return a.id < b.id;
                }
                return first < second;
            });
        std::vector<WeightedPICMarker> sorted_source(collision_source.size());
        std::vector<canonical_ntc_detail::SourceRef> sorted_ref(
            collision_source_ref.size());
        for (std::size_t q = 0; q < source_order.size(); ++q) {
            sorted_source[q] = collision_source[source_order[q]];
            sorted_ref[q] = collision_source_ref[source_order[q]];
        }
        collision_source = std::move(sorted_source);
        collision_source_ref = std::move(sorted_ref);
    }
    const WeightedPICNoiseDiagnostics noise_before = diagnose_weighted_pic_noise(
        collision_source, radial_mesh);
    const WeightedPICDepositedMoments deposited_before =
        deposit_weighted_vlasov_pic(collision_source, radial_mesh);
    canonical_ntc_detail::build_weighted_shells(
        collision_source,
        radial_mesh,
        p,
        options.collision,
        state.workspace);
    // A sampled physical rate and the all-pair majorant define distinct
    // timescales.  Record both, then enforce locality and the proposal CFL.
    diagnostics.physical_collision_timescale =
        weighted_pic_sampled_collision_timescale(
            collision_source,
            p,
            options.collision,
            state.workspace,
            state.collision_sampler.seed
                ^ state.collision_sampler.next_counter);
    diagnostics.collision_majorant_timescale =
        weighted_pic_collision_majorant_timescale(state.workspace.shells);
    diagnostics.collision.active_markers = collision_source.size();
    if (!collision_source.empty()) {
        double minimum_weight = collision_source.front().mass;
        double maximum_weight = collision_source.front().mass;
        for (const WeightedPICMarker& marker : collision_source) {
            minimum_weight = std::min(minimum_weight, marker.mass);
            maximum_weight = std::max(maximum_weight, marker.mass);
        }
        diagnostics.collision.maximum_relative_weight_spread =
            maximum_weight > 0.0
            ? (maximum_weight - minimum_weight) / maximum_weight
            : 0.0;
    }
    for (const WeightedPICCollisionShell& shell : state.workspace.shells) {
        const std::size_t population = shell.end - shell.begin;
        diagnostics.collision.minimum_shell_population =
            diagnostics.collision.minimum_shell_population == 0
            ? population
            : std::min(
                diagnostics.collision.minimum_shell_population, population);
        diagnostics.collision.minimum_mean_free_path = std::min(
            diagnostics.collision.minimum_mean_free_path,
            shell.mean_free_path);
        const double locality = std::isfinite(shell.mean_free_path)
            && shell.mean_free_path > 0.0
            ? (shell.outer_edge - shell.inner_edge) / shell.mean_free_path
            : 0.0;
        diagnostics.collision.maximum_shell_width_over_mean_free_path =
            std::max(
                diagnostics.collision.maximum_shell_width_over_mean_free_path,
                locality);
        if (locality > 1.0) {
            ++diagnostics.collision.nonlocal_shells;
        }
    }
    if (options.collision.fail_on_nonlocal_shell
        && diagnostics.collision.nonlocal_shells != 0) {
        throw std::runtime_error(
            "Canonical weighted collision shell exceeds its mean free path.");
    }
    const double collision_limit = weighted_pic_collision_timestep(
        state.workspace.shells, options.collision);
    if (std::isfinite(collision_limit)
        && dt > collision_limit
            * (1.0 + 32.0 * std::numeric_limits<double>::epsilon())) {
        std::ostringstream message;
        message.precision(17);
        message
            << "Canonical weighted NTC interval exceeds its collision CFL; "
            << "dt=" << dt << ", limit=" << collision_limit
            << ". Reduce the scheduled collision interval.";
        throw std::runtime_error(message.str());
    }
    std::vector<canonical_ntc_detail::PairSampler> all_pair_sampler;
    all_pair_sampler.reserve(state.workspace.shells.size());
    for (const WeightedPICCollisionShell& shell : state.workspace.shells) {
        all_pair_sampler.push_back(
            canonical_ntc_detail::make_pair_sampler(collision_source, shell));
    }
    // Sparse physical trial counts may receive more virtual replicas to
    // reduce sampling noise; each proposal transfers a smaller packet mass.
    std::vector<int> shell_virtual_batches(
        state.workspace.shells.size(), options.virtual_collision_batches);
    int maximum_shell_batches = options.virtual_collision_batches;
    std::uint64_t shell_visits = 0;
    for (std::size_t shell = 0; shell < state.workspace.shells.size(); ++shell) {
        const long double expected_trials =
            canonical_ntc_detail::expected_weighted_shell_trials(
                state.workspace.shells[shell],
                all_pair_sampler[shell],
                dt);
        if (options.minimum_expected_trials_per_shell > 0.0
            && expected_trials > 0.0L) {
            const long double requested = std::ceil(
                options.minimum_expected_trials_per_shell / expected_trials);
            const int adaptive_batches = requested
                    >= options.maximum_virtual_collision_batches
                ? options.maximum_virtual_collision_batches
                : static_cast<int>(requested);
            shell_virtual_batches[shell] = std::max(
                shell_virtual_batches[shell], adaptive_batches);
        }
        maximum_shell_batches = std::max(
            maximum_shell_batches, shell_virtual_batches[shell]);
        shell_visits += static_cast<std::uint64_t>(
            shell_virtual_batches[shell]);
    }
    std::vector<canonical_ntc_detail::WeightedEvent> collision_event;
    const auto pair_begin = std::chrono::steady_clock::now();
    diagnostics.virtual_collision_batches = maximum_shell_batches;
    if (!shell_virtual_batches.empty()) {
        diagnostics.minimum_shell_virtual_batches = *std::min_element(
            shell_virtual_batches.begin(), shell_virtual_batches.end());
        diagnostics.maximum_shell_virtual_batches = maximum_shell_batches;
    }
    const std::size_t radial_cells = geometry.radial.width.size();
    std::array<std::vector<long double>, 2> radial_energy_transfer{
        std::vector<long double>(radial_cells, 0.0L),
        std::vector<long double>(radial_cells, 0.0L)};
    std::array<std::vector<long double>, 2> radial_heat_flux_moment{
        std::vector<long double>(radial_cells, 0.0L),
        std::vector<long double>(radial_cells, 0.0L)};
    std::vector<std::vector<canonical_ntc_detail::ShellBatchResult>>
        sampled_batch(static_cast<std::size_t>(maximum_shell_batches));
    for (auto& batch : sampled_batch) {
        batch.resize(state.workspace.shells.size());
    }
    const int threads = static_cast<int>(std::min<std::size_t>(
        options.threads,
        std::max<std::size_t>(1, state.workspace.shells.size())));
    const std::uint64_t counter_base = state.collision_sampler.next_counter;
    if (!state.workspace.shells.empty()) {
        parallel_for_chunks(
            0,
            static_cast<int>(state.workspace.shells.size()),
            threads,
            [&](int begin, int end, int) {
                for (int shell_number = begin;
                     shell_number < end;
                     ++shell_number) {
                    const std::size_t shell =
                        static_cast<std::size_t>(shell_number);
                    for (int batch = 0;
                         batch < shell_virtual_batches[shell];
                         ++batch) {
                        sampled_batch[static_cast<std::size_t>(batch)][shell] =
                            canonical_ntc_detail::sample_weighted_shell_batch(
                                collision_source,
                                state.workspace.shells[shell],
                                all_pair_sampler[shell],
                                dt,
                                shell_virtual_batches[shell],
                                batch,
                                p,
                                options.collision,
                                weighted_pic_collision_detail::collision_seed(
                                    state.collision_sampler.seed,
                                    counter_base
                                        + static_cast<std::uint64_t>(batch),
                                    UINT64_C(2) * shell),
                                weighted_pic_collision_detail::collision_seed(
                                    state.collision_sampler.seed,
                                    counter_base,
                                    UINT64_C(2) * shell + UINT64_C(1)));
                    }
                }
            });
    }
    state.collision_sampler.next_counter += static_cast<std::uint64_t>(
        maximum_shell_batches);
    diagnostics.collision.substeps += static_cast<std::uint64_t>(
        maximum_shell_batches);
    diagnostics.collision.shell_visits += shell_visits;
    // Parallel proposals share the frozen source.  Consume them afterward
    // in deterministic batch/shell order with remaining-mass acceptance,
    // preventing the same physical mass from being withdrawn twice.
    std::vector<long double> remaining_source_mass(collision_source.size(), 0.0L);
    for (std::size_t q = 0; q < collision_source.size(); ++q) {
        remaining_source_mass[q] = collision_source[q].mass;
    }
    bool has_independent_replicates = false;
    for (int batch = 0; batch < maximum_shell_batches; ++batch) {
        const auto& shell_result =
            sampled_batch[static_cast<std::size_t>(batch)];
        std::vector<canonical_ntc_detail::WeightedEvent> batch_event;
        for (std::size_t collision_shell = 0;
             collision_shell < shell_result.size();
             ++collision_shell) {
            if (batch >= shell_virtual_batches[collision_shell]) {
                continue;
            }
            const canonical_ntc_detail::ShellBatchResult& result =
                shell_result[collision_shell];
            diagnostics.collision.trial_pairs += result.trial_pairs;
            diagnostics.collision.skipped_already_collided_trials +=
                result.skipped_already_collided_trials;
            diagnostics.collision.max_pair_momentum_error = std::max(
                diagnostics.collision.max_pair_momentum_error,
                result.max_pair_momentum_error);
            diagnostics.collision.max_pair_kinetic_energy_error = std::max(
                diagnostics.collision.max_pair_kinetic_energy_error,
                result.max_pair_kinetic_energy_error);
            const int replicate = batch & 1;
            const int shell_batches = shell_virtual_batches[collision_shell];
            const int replicate_count = replicate == 0
                ? (shell_batches + 1) / 2 : shell_batches / 2;
            has_independent_replicates = has_independent_replicates
                || (replicate_count > 0 && shell_batches >= 2);
            const long double replicate_scale = replicate_count > 0
                ? static_cast<long double>(shell_batches) / replicate_count
                : 0.0L;
            for (canonical_ntc_detail::WeightedEvent event : result.event) {
                long double& first_remaining =
                    remaining_source_mass[event.first_source];
                long double& second_remaining =
                    remaining_source_mass[event.second_source];
                const long double packet_mass = std::min({
                    static_cast<long double>(event.packet_mass),
                    first_remaining,
                    second_remaining,
                });
                if (!(packet_mass > 0.0L)) {
                    ++diagnostics.collision.skipped_already_collided_trials;
                    continue;
                }
                const long double maximum_initial_mass = std::max(
                    static_cast<long double>(
                        collision_source[event.first_source].mass),
                    static_cast<long double>(
                        collision_source[event.second_source].mass));
                // Correct the initial max-weight proposal to the current
                // product of remaining source masses and the actual packet
                // transfer, including the virtual-replica normalization.
                const long double depletion_probability =
                    first_remaining * second_remaining
                    / (static_cast<long double>(event.proposal_replicas)
                        * maximum_initial_mass * packet_mass);
                const long double probability_tolerance = 4096.0L
                    * std::numeric_limits<double>::epsilon();
                if (depletion_probability > 1.0L + probability_tolerance) {
                    throw std::runtime_error(
                        "Canonical depleted NTC acceptance exceeded unity.");
                }
                if (event.depletion_uniform
                    >= std::min(1.0L, depletion_probability)) {
                    ++diagnostics.collision.skipped_already_collided_trials;
                    continue;
                }
                event.packet_mass = static_cast<double>(packet_mass);
                first_remaining -= packet_mass;
                second_remaining -= packet_mass;
                ++diagnostics.collision.accepted_pairs;
                diagnostics.collision.represented_event_mass +=
                    event.packet_mass;
                diagnostics.collision.radial_momentum_error +=
                    event.packet_mass * (
                        event.first_after.x + event.second_after.x
                        - event.first_before.x - event.second_before.x);
                diagnostics.collision.kinetic_energy_error +=
                    0.5 * event.packet_mass * (
                        bp_fa_dot(event.first_after, event.first_after)
                        + bp_fa_dot(event.second_after, event.second_after)
                        - bp_fa_dot(event.first_before, event.first_before)
                        - bp_fa_dot(event.second_before, event.second_before));
                if (replicate == 0) {
                    ++diagnostics.collision_batch_a_events;
                } else {
                    ++diagnostics.collision_batch_b_events;
                }
                const WeightedPICMarker& first =
                    collision_source[event.first_source];
                const int radial_cell = canonical_orbit_containing_cell(
                    first.r, geometry.radial.edge);
                if (radial_cell < 0) {
                    throw std::runtime_error(
                        "Canonical collision event lies outside the radial mesh.");
                }
                if (shell_batches >= 2) {
                    const long double marker_mass = event.packet_mass
                        * replicate_scale;
                    const long double radial_transfer = 0.5L * marker_mass * (
                        event.first_after.x * event.first_after.x
                        + event.second_after.x * event.second_after.x
                        - event.first_before.x * event.first_before.x
                        - event.second_before.x * event.second_before.x);
                    const long double heat_flux = 0.5L * marker_mass * (
                        bp_fa_dot(event.first_after, event.first_after)
                            * event.first_after.x
                        + bp_fa_dot(event.second_after, event.second_after)
                            * event.second_after.x
                        - bp_fa_dot(event.first_before, event.first_before)
                            * event.first_before.x
                        - bp_fa_dot(event.second_before, event.second_before)
                            * event.second_before.x);
                    radial_energy_transfer[static_cast<std::size_t>(replicate)]
                        [static_cast<std::size_t>(radial_cell)] += radial_transfer;
                    radial_heat_flux_moment[static_cast<std::size_t>(replicate)]
                        [static_cast<std::size_t>(radial_cell)] += heat_flux;
                }
                batch_event.push_back(event);
            }
        }
        collision_event.insert(
            collision_event.end(), batch_event.begin(), batch_event.end());
    }
    // Even/odd replica transfers provide radial kinetic-energy and heat-flux
    // disagreement diagnostics.  A one-percent global signal floor stabilizes
    // relative errors in shells with weak local transfer.
    if (has_independent_replicates) {
        std::vector<double> radial_signal(radial_cells, 0.0);
        std::vector<double> radial_error(radial_cells, 0.0);
        std::vector<double> heat_signal(radial_cells, 0.0);
        std::vector<double> heat_error(radial_cells, 0.0);
        double maximum_radial_signal = 0.0;
        double maximum_heat_signal = 0.0;
        for (std::size_t shell = 0; shell < radial_cells; ++shell) {
            const long double radial_a = radial_energy_transfer[0][shell];
            const long double radial_b = radial_energy_transfer[1][shell];
            const long double heat_a = radial_heat_flux_moment[0][shell];
            const long double heat_b = radial_heat_flux_moment[1][shell];
            radial_signal[shell] = static_cast<double>(
                0.5L * (radial_a + radial_b));
            radial_error[shell] = static_cast<double>(
                0.5L * std::abs(radial_a - radial_b));
            heat_signal[shell] = static_cast<double>(
                0.5L * (heat_a + heat_b));
            heat_error[shell] = static_cast<double>(
                0.5L * std::abs(heat_a - heat_b));
            maximum_radial_signal = std::max(
                maximum_radial_signal, std::abs(radial_signal[shell]));
            maximum_heat_signal = std::max(
                maximum_heat_signal, std::abs(heat_signal[shell]));
        }
        for (std::size_t shell = 0; shell < radial_cells; ++shell) {
            const double radial_scale = std::max(
                std::abs(radial_signal[shell]),
                0.01 * maximum_radial_signal);
            const double heat_scale = std::max(
                std::abs(heat_signal[shell]),
                0.01 * maximum_heat_signal);
            if (radial_scale > 0.0) {
                diagnostics.maximum_relative_collision_radial_energy_batch_error =
                    std::max(
                        diagnostics.maximum_relative_collision_radial_energy_batch_error,
                        radial_error[shell] / radial_scale);
            }
            if (heat_scale > 0.0) {
                diagnostics.maximum_relative_collision_heat_flux_batch_error =
                    std::max(
                        diagnostics.maximum_relative_collision_heat_flux_batch_error,
                        heat_error[shell] / heat_scale);
            }
        }
    }
    diagnostics.collision_pair_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - pair_begin).count();
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(PI) * static_cast<long double>(PI);
    std::vector<double> turning_scan;
    turning_scan.reserve(
        geometry.radial.edge.size() + geometry.radial.center.size());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.edge.begin(),
        geometry.radial.edge.end());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.center.begin(),
        geometry.radial.center.end());
    std::sort(turning_scan.begin(), turning_scan.end());
    turning_scan.erase(
        std::unique(
            turning_scan.begin(),
            turning_scan.end(),
            [](double first, double second) {
                return phase_space_nearly_equal(first, second);
            }),
        turning_scan.end());

    struct LiftedPair {
        CanonicalNTCMarkerLift first_after;
        CanonicalNTCMarkerLift second_after;
        double packet_mass = 0.0;
        std::size_t first_source = 0;
        std::size_t second_source = 0;
        canonical_ntc_detail::SourceRef first_ref;
        canonical_ntc_detail::SourceRef second_ref;
        std::size_t cached_phase_lifts = 0;
        std::size_t exact_phase_lifts = 0;
        double maximum_estimated_phase_cell_error = 0.0;
    };
    std::vector<LiftedPair> lifted(collision_event.size());
    const auto lift_begin = std::chrono::steady_clock::now();
    if (lifted.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Canonical collision lift exceeds the parallel index range.");
    }
    const int lift_threads = static_cast<int>(std::min<std::size_t>(
        options.threads, std::max<std::size_t>(1, lifted.size())));
    const bool retain_continuous_collision_increment =
        options.retain_collision_products_as_detached_carriers
        || options.retain_collision_increment_as_signed_carriers;
    if (!lifted.empty() && !retain_continuous_collision_increment) {
        parallel_for_chunks(
            0,
            static_cast<int>(lifted.size()),
            lift_threads,
            [&](int begin, int end, int) {
                for (int event_number = begin;
                     event_number < end;
                     ++event_number) {
                    const std::size_t event_index =
                        static_cast<std::size_t>(event_number);
                    const canonical_ntc_detail::WeightedEvent& event =
                        collision_event[event_index];
                    const WeightedPICMarker& first =
                        collision_source[event.first_source];
                    const WeightedPICMarker& second =
                        collision_source[event.second_source];
                    LiftedPair& pair = lifted[event_index];
                    pair.packet_mass = event.packet_mass;
                    pair.first_source = event.first_source;
                    pair.second_source = event.second_source;
                    pair.first_ref = collision_source_ref[event.first_source];
                    pair.second_ref = collision_source_ref[event.second_source];
                    pair.first_after = canonical_ntc_lift_marker(
                        atlas,
                        geometry,
                        p,
                        first.r,
                        event.first_after,
                        options.hidden_central_mass,
                        turning_scan,
                        options.maximum_lift_phase_interpolation_cell_error,
                        &pair.cached_phase_lifts,
                        &pair.exact_phase_lifts,
                        &pair.maximum_estimated_phase_cell_error,
                        potential_at_radius);
                    pair.second_after = canonical_ntc_lift_marker(
                        atlas,
                        geometry,
                        p,
                        second.r,
                        event.second_after,
                        options.hidden_central_mass,
                        turning_scan,
                        options.maximum_lift_phase_interpolation_cell_error,
                        &pair.cached_phase_lifts,
                        &pair.exact_phase_lifts,
                        &pair.maximum_estimated_phase_cell_error,
                        potential_at_radius);
                }
            });
    } else if (retain_continuous_collision_increment) {
        for (std::size_t event_index = 0;
             event_index < collision_event.size();
             ++event_index) {
            const canonical_ntc_detail::WeightedEvent& event =
                collision_event[event_index];
            LiftedPair& pair = lifted[event_index];
            pair.packet_mass = event.packet_mass;
            pair.first_source = event.first_source;
            pair.second_source = event.second_source;
            pair.first_ref = collision_source_ref[event.first_source];
            pair.second_ref = collision_source_ref[event.second_source];
        }
    }
    for (const LiftedPair& pair : lifted) {
        diagnostics.outgoing_cached_phase_lifts += pair.cached_phase_lifts;
        diagnostics.outgoing_exact_phase_lifts += pair.exact_phase_lifts;
        diagnostics.maximum_estimated_lift_phase_cell_error = std::max(
            diagnostics.maximum_estimated_lift_phase_cell_error,
            pair.maximum_estimated_phase_cell_error);
    }
    diagnostics.outgoing_lift_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - lift_begin).count();
    const double atlas_mass_before = canonical_orbit_atlas_mass(atlas);
    long double open_mass_before = 0.0L;
    for (const WeightedPICMarker& marker : state.open_marker) {
        open_mass_before += marker.mass;
    }
    long double detached_mass_before = 0.0L;
    for (const WeightedPICMarker& marker : state.detached_marker) {
        detached_mass_before += marker.mass;
    }
    long double signed_positive_mass_before = 0.0L;
    for (const WeightedPICMarker& marker : state.signed_increment_positive) {
        signed_positive_mass_before += marker.mass;
    }
    long double signed_negative_mass_before = 0.0L;
    for (const WeightedPICMarker& marker : state.signed_increment_negative) {
        signed_negative_mass_before += marker.mass;
    }
    diagnostics.physical_mass_before = static_cast<double>(
        phase_factor * atlas_mass_before + open_mass_before
        + detached_mass_before + signed_positive_mass_before
        - signed_negative_mass_before);
    std::vector<long double> atlas_cell_mass(atlas.column.size(), 0.0L);
    std::vector<long double> atlas_cell_energy_mass(
        atlas.column.size(), 0.0L);
    std::vector<long double> atlas_cell_energy2_mass(
        atlas.column.size(), 0.0L);
    for (const CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            atlas_cell_mass[row.first_cell + q] = row.phase_mass[q];
            atlas_cell_energy_mass[row.first_cell + q] =
                row.phase_energy_mass[q];
            atlas_cell_energy2_mass[row.first_cell + q] =
                row.phase_energy2_mass[q];
        }
    }
    std::vector<long double> delta(atlas.column.size(), 0.0L);
    std::vector<long double> absolute_mass_transfer(
        atlas.column.size(), 0.0L);
    std::vector<long double> delta_energy(atlas.column.size(), 0.0L);
    std::vector<long double> delta_energy2(atlas.column.size(), 0.0L);
    long double represented_pair_mass = 0.0L;
    auto add_delta = [&](const CanonicalNTCMarkerLift& lift,
                         long double packet_mass) {
        for (int q = 0; q < lift.terms; ++q) {
            const std::uint32_t cell =
                lift.term[static_cast<std::size_t>(q)].canonical_cell;
            const long double mass = packet_mass
                * lift.term[static_cast<std::size_t>(q)].weight;
            const long double energy =
                lift.term[static_cast<std::size_t>(q)].energy;
            delta[cell] += mass;
            absolute_mass_transfer[cell] += std::abs(mass);
            delta_energy[cell] += mass * energy;
            delta_energy2[cell] += mass * energy * energy;
        }
    };
    std::vector<std::uint32_t> cell_to_row(
        atlas.column.size(), std::numeric_limits<std::uint32_t>::max());
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        const CanonicalOrbitRow& row = atlas.row[row_index];
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            cell_to_row[row.first_cell + q] =
                static_cast<std::uint32_t>(row_index);
        }
    }
    // Withdraw the sampled packet's fraction from its original owned cells
    // and carry those cells' mean H/H^2.  Particle coordinates alone cannot
    // identify this finite-volume source mass after phase transport.
    auto withdraw_ownership = [&](std::size_t marker_index,
                                  double packet_mass) {
        long double withdrawn = 0.0L;
        long double represented_energy_mass = 0.0L;
        const double source_marker_mass =
            initialized.markers[marker_index].mass;
        const double ownership_fraction = packet_mass / source_marker_mass;
        canonical_initialization.ownership[marker_index].for_each(
            [&](const CanonicalNTCOwnershipTerm& term) {
            if (term.canonical_cell >= delta.size()
                || cell_to_row[term.canonical_cell]
                    == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "Canonical collision ownership references a missing atlas cell.");
            }
            const long double owned_physical_mass = term.physical_mass
                * ownership_fraction;
            const long double reduced = owned_physical_mass / phase_factor;
            delta[term.canonical_cell] -= reduced;
            absolute_mass_transfer[term.canonical_cell] += std::abs(reduced);
            if (!(atlas_cell_mass[term.canonical_cell] > 0.0L)) {
                throw std::runtime_error(
                    "Canonical collision ownership withdrew an empty cell.");
            }
            const long double mean_energy =
                atlas_cell_energy_mass[term.canonical_cell]
                / atlas_cell_mass[term.canonical_cell];
            const long double mean_energy2 =
                atlas_cell_energy2_mass[term.canonical_cell]
                / atlas_cell_mass[term.canonical_cell];
            delta_energy[term.canonical_cell] -= reduced * mean_energy;
            delta_energy2[term.canonical_cell] -= reduced * mean_energy2;
            withdrawn += owned_physical_mass;
            represented_energy_mass += owned_physical_mass * mean_energy;
        });
        const double tolerance = 2048.0
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, packet_mass);
        if (std::abs(static_cast<double>(withdrawn) - packet_mass)
            > tolerance) {
            throw std::runtime_error(
                "Canonical marker ownership does not sum to marker mass.");
        }
        return packet_mass > 0.0
            ? static_cast<double>(represented_energy_mass / packet_mass)
            : 0.0;
    };
    std::vector<WeightedPICMarker> staged_open_marker = state.open_marker;
    std::vector<WeightedPICMarker> staged_detached_marker =
        state.detached_marker;
    std::vector<long double> canonical_withdrawal(
        initialized.markers.size(), 0.0L);
    std::vector<long double> open_withdrawal(
        state.open_marker.size(), 0.0L);
    std::vector<long double> detached_withdrawal(
        state.detached_marker.size(), 0.0L);
    std::vector<WeightedPICMarker> new_open_marker;
    new_open_marker.reserve(2 * lifted.size());
    std::vector<WeightedPICMarker> new_detached_marker;
    new_detached_marker.reserve(2 * lifted.size());
    std::vector<WeightedPICMarker> new_signed_positive;
    new_signed_positive.reserve(2 * lifted.size());
    std::uint64_t staged_next_open_marker_id = state.next_open_marker_id;
    auto source_energy = [&](std::size_t source_index,
                             const BPFAVec3& velocity) {
        const WeightedPICMarker& source = collision_source[source_index];
        return potential_at_radius(source.r)
            + 0.5 * bp_fa_dot(velocity, velocity);
    };
    // Track cumulative withdrawal separately for atlas and explicit sources;
    // reject any over-consumption before publishing their staged distributions.
    auto withdraw_source = [&](const canonical_ntc_detail::SourceRef& source,
                               std::size_t source_index,
                               const BPFAVec3& velocity,
                               double packet_mass) {
        if (source.kind == canonical_ntc_detail::SourceKind::canonical) {
            if (source.index >= canonical_withdrawal.size()) {
                throw std::runtime_error(
                    "Canonical collision source index is out of range.");
            }
            const double available = initialized.markers[source.index].mass;
            canonical_withdrawal[source.index] += packet_mass;
            const long double tolerance = 4096.0L
                * std::numeric_limits<double>::epsilon()
                * std::max(
                    static_cast<long double>(available),
                    static_cast<long double>(packet_mass));
            if (canonical_withdrawal[source.index]
                > static_cast<long double>(available) + tolerance) {
                throw std::runtime_error(
                    "Canonical weighted NTC over-withdrew an atlas source.");
            }
            return withdraw_ownership(source.index, packet_mass);
        }
        if (source.kind == canonical_ntc_detail::SourceKind::detached) {
            if (source.index >= staged_detached_marker.size()) {
                throw std::runtime_error(
                    "Canonical detached collision source index is out of range.");
            }
            const double available = state.detached_marker[source.index].mass;
            detached_withdrawal[source.index] += packet_mass;
            const long double tolerance = 4096.0L
                * std::numeric_limits<double>::epsilon()
                * std::max(
                    static_cast<long double>(available),
                    static_cast<long double>(packet_mass));
            if (detached_withdrawal[source.index]
                > static_cast<long double>(available) + tolerance) {
                throw std::runtime_error(
                    "Canonical weighted NTC over-withdrew a detached source.");
            }
            long double remaining = static_cast<long double>(available)
                - detached_withdrawal[source.index];
            if (remaining < 0.0L && remaining >= -tolerance) {
                remaining = 0.0L;
            }
            staged_detached_marker[source.index].mass =
                static_cast<double>(remaining);
            return source_energy(source_index, velocity);
        }
        if (source.index >= staged_open_marker.size()) {
            throw std::runtime_error(
                "Canonical open collision source index is out of range.");
        }
        const double available = state.open_marker[source.index].mass;
        open_withdrawal[source.index] += packet_mass;
        const long double tolerance = 4096.0L
            * std::numeric_limits<double>::epsilon()
            * std::max(
                static_cast<long double>(available),
                static_cast<long double>(packet_mass));
        if (open_withdrawal[source.index]
            > static_cast<long double>(available) + tolerance) {
            throw std::runtime_error(
                "Canonical weighted NTC over-withdrew an open source.");
        }
        long double remaining = static_cast<long double>(available)
            - open_withdrawal[source.index];
        if (remaining < 0.0L && remaining >= -tolerance) {
            remaining = 0.0L;
        }
        staged_open_marker[source.index].mass = static_cast<double>(remaining);
        return source_energy(source_index, velocity);
    };
    // New products retain physical radius and full outgoing velocity, receive
    // unique high-range IDs, and carry a classified representation limitation.
    auto retain_open_marker = [&](const WeightedPICMarker& source,
                                  const BPFAVec3& velocity,
                                  double packet_mass,
                                  CanonicalNTCOpenReason reason) {
        if (staged_next_open_marker_id == 0) {
            throw std::runtime_error(
                "Canonical open marker id counter overflowed.");
        }
        WeightedPICMarker marker = source;
        marker.mass = packet_mass;
        marker.w = velocity.x;
        marker.eta = 0.5 * marker.r * marker.r
            * (velocity.y * velocity.y + velocity.z * velocity.z);
        marker.source_cell = 0;
        marker.state = WeightedPICMarkerState::active;
        marker.id = staged_next_open_marker_id++;
        marker.inner_boundary_speed = 0.0;
        marker.remaining_inner_time = 0.0;
        new_open_marker.push_back(marker);
        diagnostics.physical_open_mass += packet_mass;
        switch (reason) {
        case CanonicalNTCOpenReason::energy_above_atlas:
            diagnostics.physical_high_energy_open_mass += packet_mass;
            break;
        case CanonicalNTCOpenReason::eta_outside_atlas:
            diagnostics.unrepresented_eta_mass += packet_mass;
            break;
        case CanonicalNTCOpenReason::eta_row_missing:
            diagnostics.unrepresented_missing_eta_row_mass += packet_mass;
            break;
        case CanonicalNTCOpenReason::energy_outside_row:
            diagnostics.unrepresented_energy_mass += packet_mass;
            break;
        case CanonicalNTCOpenReason::orbit_geometry_failure:
            diagnostics.unrepresented_orbit_geometry_mass += packet_mass;
            break;
        case CanonicalNTCOpenReason::none:
            throw std::runtime_error(
                "Canonical open marker has no classified lift reason.");
        }
    };
    auto retain_detached_marker = [&](const WeightedPICMarker& source,
                                      const BPFAVec3& velocity,
                                      double packet_mass) {
        if (staged_next_open_marker_id == 0) {
            throw std::runtime_error(
                "Canonical detached marker id counter overflowed.");
        }
        WeightedPICMarker marker = source;
        marker.mass = packet_mass;
        marker.w = velocity.x;
        marker.eta = 0.5 * marker.r * marker.r
            * (velocity.y * velocity.y + velocity.z * velocity.z);
        marker.source_cell = 0;
        marker.state = WeightedPICMarkerState::active;
        marker.id = staged_next_open_marker_id++;
        marker.inner_boundary_speed = 0.0;
        marker.remaining_inner_time = 0.0;
        new_detached_marker.push_back(marker);
        diagnostics.physical_detached_mass += packet_mass;
    };
    auto retain_signed_marker = [&](std::vector<WeightedPICMarker>& target,
                                    const WeightedPICMarker& source,
                                    const BPFAVec3& velocity,
                                    double packet_mass,
                                    std::uint32_t source_cell = 0) {
        if (staged_next_open_marker_id == 0) {
            throw std::runtime_error(
                "Canonical signed marker id counter overflowed.");
        }
        WeightedPICMarker marker = source;
        marker.mass = packet_mass;
        marker.w = velocity.x;
        marker.eta = 0.5 * marker.r * marker.r
            * (velocity.y * velocity.y + velocity.z * velocity.z);
        marker.source_cell = source_cell;
        marker.state = WeightedPICMarkerState::active;
        marker.id = staged_next_open_marker_id++;
        marker.inner_boundary_speed = 0.0;
        marker.remaining_inner_time = 0.0;
        target.push_back(marker);
    };
    for (std::size_t event_index = 0; event_index < lifted.size(); ++event_index) {
        const LiftedPair& pair = lifted[event_index];
        const double packet_mass = pair.packet_mass;
        const canonical_ntc_detail::WeightedEvent& event =
            collision_event[event_index];
        double represented_first_before = 0.0;
        double represented_second_before = 0.0;
        // The incoming packet belongs to known canonical source cells.  Those
        // cells co-move under frozen-field transport, including through the
        // delayed central phase, so withdraw the owned mass immediately.
        // Only the post-collision trajectories need continuous carriers.
        represented_first_before = withdraw_source(
            pair.first_ref,
            pair.first_source,
            event.first_before,
            packet_mass);
        represented_second_before = withdraw_source(
            pair.second_ref,
            pair.second_source,
            event.second_before,
            packet_mass);
        const long double reduced_mass = packet_mass / phase_factor;
        represented_pair_mass += packet_mass;

        if (options.retain_collision_increment_as_signed_carriers) {
            retain_signed_marker(
                new_signed_positive,
                collision_source[pair.first_source],
                event.first_after,
                packet_mass);
            retain_signed_marker(
                new_signed_positive,
                collision_source[pair.second_source],
                event.second_after,
                packet_mass);
        } else if (options.retain_collision_products_as_detached_carriers) {
            retain_detached_marker(
                collision_source[pair.first_source],
                event.first_after,
                packet_mass);
            retain_detached_marker(
                collision_source[pair.second_source],
                event.second_after,
                packet_mass);
        } else {
            if (pair.first_after.open) {
                retain_open_marker(
                    collision_source[pair.first_source],
                    event.first_after,
                    packet_mass,
                    pair.first_after.open_reason);
            } else {
                add_delta(pair.first_after, reduced_mass);
            }
            if (pair.second_after.open) {
                retain_open_marker(
                    collision_source[pair.second_source],
                    event.second_after,
                    packet_mass,
                    pair.second_after.open_reason);
            } else {
                add_delta(pair.second_after, reduced_mass);
            }
        }
        diagnostics.physical_radial_momentum_residual += packet_mass * (
            event.first_after.x + event.second_after.x
            - event.first_before.x - event.second_before.x);
        diagnostics.physical_kinetic_energy_residual += 0.5 * packet_mass * (
            bp_fa_dot(event.first_after, event.first_after)
            + bp_fa_dot(event.second_after, event.second_after)
            - bp_fa_dot(event.first_before, event.first_before)
            - bp_fa_dot(event.second_before, event.second_before));
        const double represented_before = represented_first_before
            + represented_second_before;
        const double represented_after =
            retain_continuous_collision_increment
            ? source_energy(pair.first_source, event.first_after)
                + source_energy(pair.second_source, event.second_after)
            : (pair.first_after.open
                    ? pair.first_after.energy
                    : pair.first_after.represented_energy)
                + (pair.second_after.open
                    ? pair.second_after.energy
                    : pair.second_after.represented_energy);
        diagnostics.atlas_energy_quantization_error +=
            packet_mass * (represented_after - represented_before);
        const bool first_canonical = pair.first_ref.kind
            == canonical_ntc_detail::SourceKind::canonical;
        const bool second_canonical = pair.second_ref.kind
            == canonical_ntc_detail::SourceKind::canonical;
        const bool first_open = pair.first_ref.kind
            == canonical_ntc_detail::SourceKind::open;
        const bool second_open = pair.second_ref.kind
            == canonical_ntc_detail::SourceKind::open;
        if (first_canonical && second_canonical) {
            ++diagnostics.canonical_canonical_events;
        } else if (first_open && second_open) {
            ++diagnostics.open_open_events;
        } else {
            ++diagnostics.canonical_open_events;
        }
    }
    // Apply all paired increments together, then test nonnegative mass and
    // admissible row moments: H in its energy interval, nonnegative variance
    // and the H^2 secant upper bound for that finite interval.
    for (std::size_t cell = 0; cell < atlas_cell_mass.size(); ++cell) {
        const long double initial_cell_mass = atlas_cell_mass[cell];
        atlas_cell_mass[cell] += delta[cell];
        atlas_cell_energy_mass[cell] += delta_energy[cell];
        atlas_cell_energy2_mass[cell] += delta_energy2[cell];
        const long double cancellation_scale =
            std::abs(initial_cell_mass) + absolute_mass_transfer[cell];
        const long double cancellation_tolerance = std::max(
            static_cast<long double>(
                options.maximum_roundoff_negative_mass_fraction),
            4096.0L * std::numeric_limits<double>::epsilon())
            * cancellation_scale;
        if (atlas_cell_mass[cell] < -cancellation_tolerance) {
            std::ostringstream message;
            message.precision(17);
            message
                << "Canonical direct collision lift produced negative cell mass; "
                << "cell=" << cell
                << ", initial=" << static_cast<double>(initial_cell_mass)
                << ", delta=" << static_cast<double>(delta[cell])
                << ", final=" << static_cast<double>(atlas_cell_mass[cell])
                << ", absolute_transfer="
                << static_cast<double>(absolute_mass_transfer[cell])
                << ", cancellation_tolerance="
                << static_cast<double>(cancellation_tolerance);
            throw std::runtime_error(message.str());
        }
        if (absolute_mass_transfer[cell] > 0.0L
            && std::abs(atlas_cell_mass[cell]) <= cancellation_tolerance) {
            diagnostics.positivity_roundoff_mass += static_cast<double>(
                phase_factor * std::abs(atlas_cell_mass[cell]));
            atlas_cell_mass[cell] = 0.0L;
            atlas_cell_energy_mass[cell] = 0.0L;
            atlas_cell_energy2_mass[cell] = 0.0L;
        }
        if (atlas_cell_mass[cell] > 0.0L) {
            const std::size_t row_index = cell_to_row[cell];
            if (row_index >= atlas.row.size()) {
                throw std::runtime_error(
                    "Canonical collision lift lost the destination row index.");
            }
            const CanonicalOrbitRow& row = atlas.row[row_index];
            const long double mean_energy =
                atlas_cell_energy_mass[cell] / atlas_cell_mass[cell];
            const long double mean_energy2 =
                atlas_cell_energy2_mass[cell] / atlas_cell_mass[cell];
            const long double energy_scale = std::max({
                1.0L,
                std::abs(mean_energy),
                std::abs(static_cast<long double>(row.energy_left)),
                std::abs(static_cast<long double>(row.energy_right)),
            });
            // These moments are stored as doubles between events.  Use the
            // same roundoff allowance as signed-increment materialization so
            // a state accepted there remains admissible here.
            const long double energy_tolerance = 32768.0L
                * std::numeric_limits<double>::epsilon() * energy_scale;
            if (mean_energy
                    < static_cast<long double>(row.energy_left) - energy_tolerance
                || mean_energy
                    > static_cast<long double>(row.energy_right) + energy_tolerance) {
                std::ostringstream message;
                message.precision(17);
                message
                    << "Canonical collision lift produced an H mean outside its row; "
                    << "cell=" << cell
                    << ", row=" << row_index
                    << ", mean=" << static_cast<double>(mean_energy)
                    << ", H_edges=[" << row.energy_left << ','
                    << row.energy_right << ']';
                throw std::runtime_error(message.str());
            }
            const long double variance_numerator =
                atlas_cell_energy2_mass[cell]
                    * atlas_cell_mass[cell]
                - atlas_cell_energy_mass[cell]
                    * atlas_cell_energy_mass[cell];
            const long double variance_tolerance = 32768.0L
                * std::numeric_limits<double>::epsilon()
                * std::max(
                    1.0L,
                    std::abs(atlas_cell_energy2_mass[cell]
                        * atlas_cell_mass[cell]));
            if (variance_numerator < -variance_tolerance) {
                throw std::runtime_error(
                    "Canonical collision lift produced a negative H variance.");
            }
            const long double secant_upper =
                (static_cast<long double>(row.energy_left)
                 + static_cast<long double>(row.energy_right)) * mean_energy
                - static_cast<long double>(row.energy_left)
                    * static_cast<long double>(row.energy_right);
            const long double second_moment_scale = std::max({
                1.0L, std::abs(mean_energy2), std::abs(secant_upper)});
            const long double second_moment_tolerance = 32768.0L
                * std::numeric_limits<double>::epsilon()
                * second_moment_scale;
            if (mean_energy2 > secant_upper + second_moment_tolerance) {
                std::ostringstream message;
                message.precision(17);
                message
                    << "Canonical collision lift produced an H2 moment outside its row; "
                    << "cell=" << cell
                    << ", row=" << row_index
                    << ", mean_H2=" << static_cast<double>(mean_energy2)
                    << ", admissible_upper="
                    << static_cast<double>(secant_upper)
                    << ", mass="
                    << static_cast<double>(atlas_cell_mass[cell])
                    << ", H_mass="
                    << static_cast<double>(atlas_cell_energy_mass[cell])
                    << ", H2_mass="
                    << static_cast<double>(atlas_cell_energy2_mass[cell])
                    << ", variance_numerator="
                    << static_cast<double>(variance_numerator)
                    << ", H_edges=[" << row.energy_left << ','
                    << row.energy_right << ']';
                throw std::runtime_error(message.str());
            }
        }
    }
    long double atlas_mass_after_accumulator = 0.0L;
    for (long double mass : atlas_cell_mass) {
        atlas_mass_after_accumulator += mass;
    }
    const double atlas_mass_after = static_cast<double>(
        atlas_mass_after_accumulator);
    staged_open_marker.erase(
        std::remove_if(
            staged_open_marker.begin(),
            staged_open_marker.end(),
            [](const WeightedPICMarker& marker) {
                return !(marker.mass > 0.0);
            }),
        staged_open_marker.end());
    staged_open_marker.insert(
        staged_open_marker.end(),
        new_open_marker.begin(),
        new_open_marker.end());
    staged_detached_marker.erase(
        std::remove_if(
            staged_detached_marker.begin(),
            staged_detached_marker.end(),
            [](const WeightedPICMarker& marker) {
                return !(marker.mass > 0.0);
            }),
        staged_detached_marker.end());
    staged_detached_marker.insert(
        staged_detached_marker.end(),
        new_detached_marker.begin(),
        new_detached_marker.end());
    long double open_mass_after = 0.0L;
    for (const WeightedPICMarker& marker : staged_open_marker) {
        validate_weighted_pic_marker(
            marker, "Canonical staged open collision output");
        open_mass_after += marker.mass;
    }
    long double detached_mass_after = 0.0L;
    for (const WeightedPICMarker& marker : staged_detached_marker) {
        validate_weighted_pic_marker(
            marker, "Canonical staged detached collision output");
        detached_mass_after += marker.mass;
    }
    long double signed_positive_mass_after = signed_positive_mass_before;
    for (const WeightedPICMarker& marker : new_signed_positive) {
        validate_weighted_pic_marker(
            marker, "Canonical signed positive collision output");
        signed_positive_mass_after += marker.mass;
    }
    const long double signed_negative_mass_after =
        signed_negative_mass_before;
    diagnostics.physical_signed_increment_mass = static_cast<double>(
        signed_positive_mass_after - signed_negative_mass_after);
    diagnostics.signed_increment_positive_markers =
        new_signed_positive.size();
    diagnostics.signed_increment_negative_markers = 0;
    diagnostics.physical_mass_after = static_cast<double>(
        phase_factor * atlas_mass_after + open_mass_after
        + detached_mass_after + signed_positive_mass_after
        - signed_negative_mass_after);
    diagnostics.physical_mass_residual =
        diagnostics.physical_mass_after - diagnostics.physical_mass_before;
    const double mass_scale = std::max(1.0, diagnostics.physical_mass_before);
    if (std::abs(diagnostics.physical_mass_residual)
        > 1024.0 * std::numeric_limits<double>::epsilon() * mass_scale) {
        throw std::runtime_error(
            "Canonical direct collision lift changed total mass.");
    }
    // Ordinary high-energy open products have their own physical ledger.
    // Other eta/row/geometry failures consume the declared atlas error budget.
    const double unrepresented_mass =
        diagnostics.unrepresented_eta_mass
        + diagnostics.unrepresented_missing_eta_row_mass
        + diagnostics.unrepresented_energy_mass
        + diagnostics.unrepresented_orbit_geometry_mass;
    if (unrepresented_mass
        > options.maximum_unrepresented_mass_fraction * mass_scale) {
        std::ostringstream message;
        message
            << "Canonical collision products exceed the validated atlas "
            << "representation budget; total=" << unrepresented_mass
            << ", eta=" << diagnostics.unrepresented_eta_mass
            << ", missing_eta_row="
            << diagnostics.unrepresented_missing_eta_row_mass
            << ", energy=" << diagnostics.unrepresented_energy_mass
            << ", orbit_geometry="
            << diagnostics.unrepresented_orbit_geometry_mass
            << ", limit="
            << options.maximum_unrepresented_mass_fraction * mass_scale;
        throw std::runtime_error(message.str());
    }
    // Commit atlas and open-carrier updates together only after every ledger
    // and positivity check above has passed.
    for (CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            row.phase_mass[q] = static_cast<double>(
                atlas_cell_mass[row.first_cell + q]);
            row.phase_energy_mass[q] = static_cast<double>(
                atlas_cell_energy_mass[row.first_cell + q]);
            row.phase_energy2_mass[q] = static_cast<double>(
                atlas_cell_energy2_mass[row.first_cell + q]);
        }
    }
    state.open_marker = std::move(staged_open_marker);
    state.detached_marker = std::move(staged_detached_marker);
    state.signed_increment_positive.insert(
        state.signed_increment_positive.end(),
        new_signed_positive.begin(),
        new_signed_positive.end());
    state.next_open_marker_id = staged_next_open_marker_id;
    diagnostics.positivity_limited_pair_fraction = represented_pair_mass > 0.0L
        ? diagnostics.positivity_limited_pair_mass
            / static_cast<double>(represented_pair_mass)
        : 0.0;
    diagnostics.eulerian_to_atlas.source_mass = atlas_mass_before;
    diagnostics.eulerian_to_atlas.destination_mass = atlas_mass_after;
    diagnostics.eulerian_to_atlas.escaped_mass = static_cast<double>(
        (open_mass_after + detached_mass_after
            - open_mass_before - detached_mass_before
            + signed_positive_mass_after - signed_positive_mass_before
            - signed_negative_mass_after + signed_negative_mass_before)
        / phase_factor);
    diagnostics.eulerian_to_atlas.residual = atlas_mass_before
        - atlas_mass_after - diagnostics.eulerian_to_atlas.escaped_mass;

    const double maximum_density = *std::max_element(
        deposited_before.density.begin(), deposited_before.density.end());
    double maximum_flux = 0.0;
    for (double value : deposited_before.radial_energy_flux_density) {
        maximum_flux = std::max(maximum_flux, std::abs(value));
    }
    for (std::size_t i = 0; i < noise_before.density_uncertainty.size(); ++i) {
        const double density_scale = std::max(
            std::abs(deposited_before.density[i]),
            0.01 * maximum_density);
        if (density_scale > 0.0) {
            diagnostics.maximum_relative_density_noise = std::max(
                diagnostics.maximum_relative_density_noise,
                noise_before.density_uncertainty[i] / density_scale);
        }
        const double flux_scale = std::max(
            std::abs(deposited_before.radial_energy_flux_density[i]),
            0.01 * maximum_flux);
        if (flux_scale > 0.0) {
            diagnostics.maximum_relative_energy_flux_noise = std::max(
                diagnostics.maximum_relative_energy_flux_noise,
                noise_before.radial_energy_flux_uncertainty[i] / flux_scale);
        }
    }
    return diagnostics;
}

/// Builds equal-action bound-orbit rows for every eta cell.  The potential
/// callable returns Phi(r); the centrifugal contribution is added here.
/// Orbit and phase maps are cached for an entire eta plane and reused by all
/// neighboring energy rows.
template <class PotentialAtRadius>
inline CanonicalAtlasConstructionResult build_canonical_bound_orbit_atlas(
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    int energy_rows,
    PotentialAtRadius&& potential_at_radius,
    double hidden_central_mass = 0.0,
    const CanonicalAtlasBuildOptions& options = {}) {
    const int eta_quadrature_points = options.eta_quadrature_points;
    const std::size_t eta_plane_count = static_cast<std::size_t>(p.NL)
        * static_cast<std::size_t>(std::max(0, eta_quadrature_points));
    if (energy_rows < 1 || p.NL < 1 || options.threads == 0
        || eta_quadrature_points < 1 || eta_quadrature_points > 12
        || !(options.energy_action_fraction_power > 0.0)
        || !std::isfinite(options.energy_action_fraction_power)
        || !(options.minimum_energy_guard_fraction > 0.0)
        || !(options.minimum_energy_guard_fraction < 1.0e-2)
        || !std::isfinite(options.minimum_energy_guard_fraction)
        || !(hidden_central_mass >= 0.0)
        || !std::isfinite(hidden_central_mass)
        || (!options.action_fraction_edge_by_eta.empty()
            && options.action_fraction_edge_by_eta.size()
                != eta_plane_count)
        || (!options.action_edge_by_eta.empty()
            && options.action_edge_by_eta.size()
                != eta_plane_count)
        || (!options.action_fraction_edge_by_eta.empty()
            && !options.action_edge_by_eta.empty())) {
        throw std::runtime_error("Invalid canonical atlas construction request.");
    }
    validate_cell_average_eta_geometry(
        geometry, p, "canonical orbit atlas");
    CanonicalAtlasConstructionResult result;
    result.diagnostics.eta_rows_requested = static_cast<int>(eta_plane_count);
    std::vector<double> turning_scan_radius;
    turning_scan_radius.reserve(
        geometry.radial.edge.size() + geometry.radial.center.size());
    turning_scan_radius.insert(
        turning_scan_radius.end(),
        geometry.radial.edge.begin(),
        geometry.radial.edge.end());
    turning_scan_radius.insert(
        turning_scan_radius.end(),
        geometry.radial.center.begin(),
        geometry.radial.center.end());
    std::sort(turning_scan_radius.begin(), turning_scan_radius.end());
    turning_scan_radius.erase(
        std::unique(
            turning_scan_radius.begin(),
            turning_scan_radius.end(),
            [](double first, double second) {
                return phase_space_nearly_equal(first, second);
            }),
        turning_scan_radius.end());

    // Radau endpoint rules include the extreme eta faces for positive lift
    // support; interior cells use Gauss-Legendre nodes and actual eta weights.
    const auto eta_rule_for_cell = [&](int eta_cell)
        -> const std::pair<std::vector<double>, std::vector<double>>& {
        if (p.NL > 1 && eta_cell == 0) {
            return phase_space_gauss_radau_left_rule(eta_quadrature_points);
        }
        if (p.NL > 1 && eta_cell == p.NL - 1) {
            return phase_space_gauss_radau_right_rule(eta_quadrature_points);
        }
        return phase_space_gauss_legendre_rule(eta_quadrature_points);
    };
    std::vector<CanonicalAtlasConstructionResult> eta_result(eta_plane_count);
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(eta_plane_count),
        std::max(
            1,
            std::min(
                static_cast<int>(options.threads),
                static_cast<int>(eta_plane_count))),
        1,
        [&](int eta_begin, int eta_end, int) {
    for (int eta_plane = eta_begin; eta_plane < eta_end; ++eta_plane) {
        CanonicalAtlasConstructionResult& result =
            eta_result[static_cast<std::size_t>(eta_plane)];
        const int eta_cell = eta_plane / eta_quadrature_points;
        const int eta_node = eta_plane % eta_quadrature_points;
        const auto& eta_rule = eta_rule_for_cell(eta_cell);
        const double eta_half_width = 0.5
            * geometry.eta.width[static_cast<std::size_t>(eta_cell)];
        const double eta = std::clamp(
            std::fma(
                eta_half_width,
                eta_rule.first[static_cast<std::size_t>(eta_node)],
                geometry.eta.center[static_cast<std::size_t>(eta_cell)]),
            geometry.eta.edge[static_cast<std::size_t>(eta_cell)],
            geometry.eta.edge[static_cast<std::size_t>(eta_cell + 1)]);
        const double eta_weight = eta_half_width
            * eta_rule.second[static_cast<std::size_t>(eta_node)];
        auto effective_potential = [&](double radius) {
            return potential_at_radius(radius) + eta / (radius * radius);
        };
        const double minimum_radius =
            canonical_effective_potential_minimum_radius(
                geometry.radial.edge.front(),
                geometry.radial.edge.back(),
                effective_potential);
        const double minimum_effective = effective_potential(minimum_radius);
        std::vector<double> eta_turning_scan = turning_scan_radius;
        eta_turning_scan.push_back(minimum_radius);
        std::sort(eta_turning_scan.begin(), eta_turning_scan.end());
        eta_turning_scan.erase(
            std::unique(
                eta_turning_scan.begin(),
                eta_turning_scan.end(),
                [](double first, double second) {
                    return phase_space_nearly_equal(first, second);
                }),
            eta_turning_scan.end());
        const double outer_effective = effective_potential(
            geometry.radial.edge.back());
        // The closed-row domain is bounded by the outer radial face, the
        // represented radial velocity ceiling and optional maximum H.  Small
        // endpoint guards regularize circular/turning degeneracies and require
        // convergence checks on the omitted represented mass.
        const double speed_ceiling = minimum_effective
            + 0.5 * p.w_max * p.w_max;
        double upper_energy = std::min({
            outer_effective,
            speed_ceiling,
            options.maximum_energy});
        const double energy_scale = std::max({
            1.0,
            std::abs(minimum_effective),
            std::abs(upper_energy)});
        const double unguarded_span = upper_energy - minimum_effective;
        upper_energy -= std::max(
            1.0e-8 * std::max(0.0, unguarded_span),
            512.0 * std::numeric_limits<double>::epsilon() * energy_scale);
        const double raw_span = upper_energy - minimum_effective;
        if (!(raw_span > 1.0e-5 * energy_scale)
            || !std::isfinite(raw_span)) {
            ++result.diagnostics.eta_rows_without_closed_orbits;
            continue;
        }
        const double lower_energy = minimum_effective + std::max(
            options.minimum_energy_guard_fraction * raw_span,
            1024.0 * std::numeric_limits<double>::epsilon() * energy_scale);
        if (!(upper_energy > lower_energy)) {
            ++result.diagnostics.eta_rows_without_closed_orbits;
            continue;
        }
        const double span = upper_energy - lower_energy;
        result.diagnostics.minimum_energy_span = std::min(
            result.diagnostics.minimum_energy_span, span);
        result.diagnostics.maximum_energy_span = std::max(
            result.diagnostics.maximum_energy_span, span);

        std::map<double, HamiltonianOrbitPhaseGeometry> orbit_cache;
        std::map<double, double> period_cache;
        std::map<std::pair<double, double>,
                 HamiltonianOrbitPhaseGeometry::Location> location_cache;
        const std::function<double(double)> hidden_return =
            using_delayed_inner_return(p)
            ? std::function<double(double)>([&](double energy) {
                const double boundary_effective = effective_potential(
                    geometry.radial.edge.front());
                if (!(energy > boundary_effective)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double boundary_speed = std::sqrt(
                    2.0 * (energy - boundary_effective));
                return delayed_inner_uniform_cell_round_trip_time(
                    boundary_speed,
                    std::sqrt(2.0 * eta),
                    hidden_central_mass,
                    p);
            })
            : std::function<double(double)>{};
        auto cached_period = [&](double energy) {
            const auto full = orbit_cache.find(energy);
            if (full != orbit_cache.end()) {
                return full->second.radial_period;
            }
            const auto found = period_cache.find(energy);
            if (found != period_cache.end()) {
                return found->second;
            }
            const double hidden = hidden_return
                ? hidden_return(energy)
                : std::numeric_limits<double>::quiet_NaN();
            const double value = hamiltonian_orbit_radial_period(
                energy,
                eta_turning_scan,
                effective_potential,
                hidden,
                64);
            period_cache.emplace(energy, value);
            return value;
        };

        // A new atlas can use weighted action fractions, or invert absolute
        // old action edges for a field update.  Each choice retains an explicit
        // energy-domain bound set by the current potential and physical faces.
        const auto& preserved_action_fraction =
            options.action_fraction_edge_by_eta.empty()
            ? std::vector<double>{}
            : options.action_fraction_edge_by_eta[
                static_cast<std::size_t>(eta_plane)];
        const auto& preserved_action = options.action_edge_by_eta.empty()
            ? std::vector<double>{}
            : options.action_edge_by_eta[static_cast<std::size_t>(eta_plane)];
        if (!preserved_action_fraction.empty()) {
            if (preserved_action_fraction.size() < 2
                || std::abs(preserved_action_fraction.front())
                    > 64.0 * std::numeric_limits<double>::epsilon()
                || std::abs(preserved_action_fraction.back() - 1.0)
                    > 64.0 * std::numeric_limits<double>::epsilon()) {
                throw std::runtime_error(
                    "Preserved canonical action fractions need endpoints 0 and 1.");
            }
            for (std::size_t q = 1;
                 q < preserved_action_fraction.size();
                 ++q) {
                if (!(preserved_action_fraction[q]
                      > preserved_action_fraction[q - 1])) {
                    throw std::runtime_error(
                        "Preserved canonical action fractions are not increasing.");
                }
            }
        }
        if (!preserved_action.empty()) {
            if (preserved_action.size() < 2
                || std::abs(preserved_action.front())
                    > 64.0 * std::numeric_limits<double>::epsilon()) {
                throw std::runtime_error(
                    "Preserved canonical actions need a zero left endpoint.");
            }
            for (std::size_t q = 1; q < preserved_action.size(); ++q) {
                if (!(preserved_action[q] > preserved_action[q - 1])
                    || !std::isfinite(preserved_action[q])) {
                    throw std::runtime_error(
                        "Preserved canonical actions are not increasing.");
                }
            }
        }

        // These samples place energy-row edges only.  Sixteen samples per row
        // resolve the cumulative action without paying for hundreds of full
        // phase maps when a deliberately coarse calculation uses few energy
        // rows.
        const int action_samples = std::max(128, 16 * energy_rows);
        std::vector<double> sampled_energy(
            static_cast<std::size_t>(action_samples + 1), 0.0);
        std::vector<double> cumulative_period(
            static_cast<std::size_t>(action_samples + 1), 0.0);
        for (int q = 0; q <= action_samples; ++q) {
            const double fraction = static_cast<double>(q)
                / static_cast<double>(action_samples);
            const double sine = std::sin(0.5 * PI * fraction);
            sampled_energy[static_cast<std::size_t>(q)] = lower_energy
                + span * sine * sine;
        }
        for (int q = 0; q < action_samples; ++q) {
            const double h0 = sampled_energy[static_cast<std::size_t>(q)];
            const double h1 = sampled_energy[static_cast<std::size_t>(q + 1)];
            const double weight0 =
                preserved_action_fraction.empty() && preserved_action.empty()
                && options.energy_partition_weight
                ? options.energy_partition_weight(eta, h0)
                : 1.0;
            const double weight1 =
                preserved_action_fraction.empty() && preserved_action.empty()
                && options.energy_partition_weight
                ? options.energy_partition_weight(eta, h1)
                : 1.0;
            if (!(weight0 >= 0.0) || !std::isfinite(weight0)
                || !(weight1 >= 0.0) || !std::isfinite(weight1)) {
                throw std::runtime_error(
                    "Canonical energy-partition weight must be finite and nonnegative.");
            }
            cumulative_period[static_cast<std::size_t>(q + 1)] =
                cumulative_period[static_cast<std::size_t>(q)]
                + 0.5 * (h1 - h0)
                    * (cached_period(h0) * weight0
                       + cached_period(h1) * weight1);
        }
        const double total_period_integral =
            preserved_action_fraction.empty() && preserved_action.empty()
            ? cumulative_period.back()
            : canonical_adaptive_simpson_integral(
                lower_energy, upper_energy, cached_period, 2.0e-9, 18);
        if (!(total_period_integral > 0.0)
            || !std::isfinite(total_period_integral)) {
            ++result.diagnostics.eta_rows_without_closed_orbits;
            continue;
        }
        const int requested_energy_rows = !preserved_action.empty()
            ? static_cast<int>(preserved_action.size()) - 1
            : (preserved_action_fraction.empty()
                ? energy_rows
                : static_cast<int>(preserved_action_fraction.size()) - 1);
        std::vector<double> energy_edge(
            static_cast<std::size_t>(requested_energy_rows + 1), lower_energy);
        energy_edge.back() = upper_energy;
        for (int row = 1; row < requested_energy_rows; ++row) {
            const double target_period_integral = !preserved_action.empty()
                ? 2.0 * PI * preserved_action[static_cast<std::size_t>(row)]
                : (!preserved_action_fraction.empty()
                    ? total_period_integral
                        * preserved_action_fraction[static_cast<std::size_t>(row)]
                    : total_period_integral * std::pow(
                    static_cast<double>(row)
                        / static_cast<double>(requested_energy_rows),
                    options.energy_action_fraction_power));
            if (!(target_period_integral > 0.0)
                || !(target_period_integral < total_period_integral)) {
                std::ostringstream message;
                message.precision(17);
                message
                    << "Preserved radial action lies outside the rebuilt atlas; "
                    << "eta_plane=" << eta_plane
                    << ", eta=" << eta
                    << ", row_edge=" << row
                    << ", requested_rows=" << requested_energy_rows
                    << ", target_action="
                    << preserved_action[static_cast<std::size_t>(row)]
                    << ", target_period_integral=" << target_period_integral
                    << ", available_period_integral=" << total_period_integral
                    << ", lower_energy=" << lower_energy
                    << ", upper_energy=" << upper_energy
                    << ", minimum_effective=" << minimum_effective
                    << ", outer_effective=" << outer_effective
                    << ", speed_ceiling=" << speed_ceiling << '.';
                throw std::runtime_error(message.str());
            }
            const double target_fraction = target_period_integral
                / total_period_integral;
            const double sampled_target = cumulative_period.back()
                * target_fraction;
            const auto upper = std::lower_bound(
                cumulative_period.begin(), cumulative_period.end(),
                sampled_target);
            const int interval = std::clamp(
                static_cast<int>(upper - cumulative_period.begin()) - 1,
                0,
                action_samples - 1);
            const double c0 = cumulative_period[static_cast<std::size_t>(interval)];
            const double c1 = cumulative_period[static_cast<std::size_t>(interval + 1)];
            const double fraction = (sampled_target - c0) / (c1 - c0);
            double edge =
                sampled_energy[static_cast<std::size_t>(interval)]
                + fraction
                    * (sampled_energy[static_cast<std::size_t>(interval + 1)]
                       - sampled_energy[static_cast<std::size_t>(interval)]);
            if (!preserved_action_fraction.empty() || !preserved_action.empty()) {
                const double target = target_period_integral;
                double bracket_left = lower_energy;
                double bracket_right = upper_energy;
                for (int iteration = 0; iteration < 20; ++iteration) {
                    const double represented = canonical_adaptive_simpson_integral(
                        lower_energy, edge, cached_period, 5.0e-9, 16);
                    const double residual = represented - target;
                    if (std::abs(residual)
                        <= 2.0e-8 * total_period_integral) {
                        break;
                    }
                    if (residual < 0.0) {
                        bracket_left = edge;
                    } else {
                        bracket_right = edge;
                    }
                    double next = edge - residual / cached_period(edge);
                    if (!(next > bracket_left && next < bracket_right)
                        || !std::isfinite(next)) {
                        next = 0.5 * (bracket_left + bracket_right);
                    }
                    edge = next;
                }
            }
            energy_edge[static_cast<std::size_t>(row)] = edge;
        }
        // Split exactly at the inner-impact separatrix, where the orbit
        // changes from fully visible to including a hidden return interval.
        if (using_delayed_inner_return(p)) {
            const double inner_separatrix = effective_potential(
                geometry.radial.edge.front());
            const double edge_tolerance = 1024.0
                * std::numeric_limits<double>::epsilon() * energy_scale;
            if (inner_separatrix > lower_energy + edge_tolerance
                && inner_separatrix < upper_energy - edge_tolerance) {
                const auto location = std::lower_bound(
                    energy_edge.begin(), energy_edge.end(), inner_separatrix);
                const bool already_present = location != energy_edge.end()
                    && std::abs(*location - inner_separatrix) <= edge_tolerance;
                const bool previous_present = location != energy_edge.begin()
                    && std::abs(*(location - 1) - inner_separatrix)
                        <= edge_tolerance;
                if (!already_present && !previous_present) {
                    energy_edge.insert(location, inner_separatrix);
                }
            }
        }
        const int constructed_energy_rows =
            static_cast<int>(energy_edge.size()) - 1;
        for (int energy_row = 0;
             energy_row < constructed_energy_rows;
             ++energy_row) {
            canonical_orbit_append_bound_row(
                result.atlas,
                eta_cell,
                energy_edge[static_cast<std::size_t>(energy_row)],
                energy_edge[static_cast<std::size_t>(energy_row + 1)],
                eta_turning_scan,
                geometry,
                p,
                effective_potential,
                options,
                hidden_return,
                &orbit_cache,
                &location_cache,
                &period_cache,
                eta,
                eta_weight,
                eta_plane);
            if (using_delayed_inner_return(p)) {
                CanonicalOrbitRow& appended = result.atlas.row.back();
                const double inner_separatrix = effective_potential(
                    geometry.radial.edge.front());
                const double edge_tolerance = 2048.0
                    * std::numeric_limits<double>::epsilon()
                    * std::max({
                        1.0,
                        std::abs(inner_separatrix),
                        std::abs(appended.energy_left),
                        std::abs(appended.energy_right),
                    });
                appended.left_edge_is_inner_separatrix =
                    std::abs(appended.energy_left - inner_separatrix)
                    <= edge_tolerance;
                appended.right_edge_is_inner_separatrix =
                    std::abs(appended.energy_right - inner_separatrix)
                    <= edge_tolerance;
            }
            ++result.diagnostics.energy_rows_built;
        }
        ++result.diagnostics.eta_rows_built;
    }
        });
    // Merge independently built eta planes in fixed plane order, shifting
    // sparse column indices together and constructing both lookup tables.
    result.atlas.eta_row_offset.assign(
        static_cast<std::size_t>(p.NL + 1), 0);
    result.atlas.eta_plane_row_offset.assign(eta_plane_count + 1, 0);
    result.atlas.eta_plane_cell.resize(eta_plane_count, -1);
    result.atlas.eta_plane_coordinate.resize(eta_plane_count, 0.0);
    for (int eta_cell = 0; eta_cell < p.NL; ++eta_cell) {
        result.atlas.eta_row_offset[static_cast<std::size_t>(eta_cell)] =
            static_cast<std::uint32_t>(result.atlas.row.size());
        for (int eta_node = 0;
             eta_node < eta_quadrature_points;
             ++eta_node) {
            const auto& eta_rule = eta_rule_for_cell(eta_cell);
            const std::size_t eta_plane = static_cast<std::size_t>(eta_cell)
                * static_cast<std::size_t>(eta_quadrature_points)
                + static_cast<std::size_t>(eta_node);
            CanonicalAtlasConstructionResult& plane = eta_result[eta_plane];
            result.atlas.eta_plane_row_offset[eta_plane] =
                static_cast<std::uint32_t>(result.atlas.row.size());
            result.atlas.eta_plane_cell[eta_plane] = eta_cell;
            const double eta_half_width = 0.5
                * geometry.eta.width[static_cast<std::size_t>(eta_cell)];
            result.atlas.eta_plane_coordinate[eta_plane] = std::clamp(
                std::fma(
                    eta_half_width,
                    eta_rule.first[static_cast<std::size_t>(eta_node)],
                    geometry.eta.center[static_cast<std::size_t>(eta_cell)]),
                geometry.eta.edge[static_cast<std::size_t>(eta_cell)],
                geometry.eta.edge[static_cast<std::size_t>(eta_cell + 1)]);
            const std::uint32_t column_offset = static_cast<std::uint32_t>(
                result.atlas.column.size());
            for (CanonicalOrbitRow& row : plane.atlas.row) {
                row.first_cell += column_offset;
                result.atlas.row.push_back(std::move(row));
            }
            for (CanonicalOverlap& overlap : plane.atlas.overlap) {
                overlap.canonical_cell += column_offset;
                result.atlas.overlap.push_back(std::move(overlap));
            }
            result.atlas.column.insert(
                result.atlas.column.end(),
                std::make_move_iterator(plane.atlas.column.begin()),
                std::make_move_iterator(plane.atlas.column.end()));
            result.diagnostics.eta_rows_built +=
                plane.diagnostics.eta_rows_built;
            result.diagnostics.energy_rows_built +=
                plane.diagnostics.energy_rows_built;
            result.diagnostics.eta_rows_without_closed_orbits +=
                plane.diagnostics.eta_rows_without_closed_orbits;
            result.diagnostics.minimum_energy_span = std::min(
                result.diagnostics.minimum_energy_span,
                plane.diagnostics.minimum_energy_span);
            result.diagnostics.maximum_energy_span = std::max(
                result.diagnostics.maximum_energy_span,
                plane.diagnostics.maximum_energy_span);
        }
    }
    result.atlas.eta_row_offset[static_cast<std::size_t>(p.NL)] =
        static_cast<std::uint32_t>(result.atlas.row.size());
    result.atlas.eta_plane_row_offset[eta_plane_count] =
        static_cast<std::uint32_t>(result.atlas.row.size());
    for (const CanonicalColumnLedger& column : result.atlas.column) {
        result.diagnostics.maximum_patch_scale_error = std::max(
            result.diagnostics.maximum_patch_scale_error,
            column.maximum_patch_scale_error);
        result.diagnostics.maximum_column_residual = std::max(
            result.diagnostics.maximum_column_residual,
            std::abs(column.residual));
        result.diagnostics.total_canonical_quadrature_volume +=
            column.canonical_quadrature_volume;
        if (column.column_volume > 0.0) {
            result.diagnostics.maximum_column_quadrature_fraction = std::max(
                result.diagnostics.maximum_column_quadrature_fraction,
                column.canonical_quadrature_volume / column.column_volume);
        }
    }
    result.atlas.generation = 1;
    return result;
}

}  // namespace ev
