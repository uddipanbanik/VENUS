// Tests positive canonical open-carrier reduction and preservation of mass,
// momentum, energy, and return-time moments.

#include "ev/canonical_open_carrier_compression.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>

namespace {

void require_close(double before, double after, double scale) {
    const double tolerance =
        5.0e-10 * std::max(scale, 1.0e-300);
    if (!(std::abs(after - before) <= tolerance)) {
        std::cerr << "compression closure failed: before=" << before
                  << " after=" << after
                  << " difference=" << after - before
                  << " tolerance=" << tolerance << '\n';
        assert(false);
    }
}

}  // namespace

int main() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "power";
    p.kinetic_eta_power = 2.0;
    p.radial_grid = "log";
    p.r_min = 0.01;
    p.r_max = 4.0;
    p.w_max = 4.0;
    p.L_max = 8.0;
    p.Nr = 16;
    p.Nw = 20;
    p.NL = 12;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const ev::WeightedPICRadialMesh mesh =
        ev::make_weighted_pic_radial_mesh(geometry);

    std::vector<ev::WeightedPICMarker> marker;
    marker.reserve(6002);
    for (std::size_t q = 0; q < 6000; ++q) {
        const int radial_cell = 3 + static_cast<int>((q / 300) % 4);
        const int velocity_cell = 7 + static_cast<int>((q / 75) % 5);
        const int eta_cell = 2 + static_cast<int>((q / 25) % 4);
        const double fr = (static_cast<double>((17 * q) % 997) + 0.5) / 997.0;
        const double fw = (static_cast<double>((29 * q) % 991) + 0.5) / 991.0;
        const double fe = (static_cast<double>((43 * q) % 983) + 0.5) / 983.0;
        const double radius = q < 600
            ? geometry.radial.edge.front()
                * 0.95
                * std::cbrt(
                    (static_cast<double>(q) + 0.5) / 600.0)
            : std::fma(
                  fr,
                  geometry.radial.width[static_cast<std::size_t>(radial_cell)],
                  geometry.radial.edge[static_cast<std::size_t>(radial_cell)]);
        const double radial_velocity = std::fma(
            fw,
            geometry.radial_velocity.width[
                static_cast<std::size_t>(velocity_cell)],
            geometry.radial_velocity.edge[
                static_cast<std::size_t>(velocity_cell)]);
        const double eta = std::fma(
            fe,
            geometry.eta.width[static_cast<std::size_t>(eta_cell)],
            geometry.eta.edge[static_cast<std::size_t>(eta_cell)]);
        marker.push_back({
            1.0e-9 * (1.0 + static_cast<double>(q % 13)),
            radius,
            radial_velocity,
            eta,
            UINT64_C(0x8000000000000000) + q,
            q % 2,
            ev::WeightedPICMarkerState::active,
            0.0,
            0.0,
            1.0e-11 * (1.0 + static_cast<double>(q % 17)),
        });
        marker.back().internal_velocity_dispersion = std::sqrt(
            2.0 * marker.back().internal_energy
            / (3.0 * marker.back().mass));
    }
    marker.push_back({
        3.0e-8,
        geometry.radial.edge.front(),
        0.1,
        geometry.eta.center.front(),
        UINT64_C(0x9000000000000000),
        0,
        ev::WeightedPICMarkerState::delayed_inner,
        0.1,
        0.2,
    });
    marker.push_back({
        5.0e-8,
        geometry.radial.edge.back(),
        0.2,
        geometry.eta.center.front(),
        UINT64_C(0x9000000000000001),
        0,
        ev::WeightedPICMarkerState::escaped,
        0.0,
        0.0,
    });

    const auto before = ev::deposit_weighted_vlasov_pic_parallel(
        marker, mesh, ev::WeightedPICParticleShape::tsc, 4, true);
    const std::vector<ev::WeightedPICMarker> original = marker;
    ev::CanonicalOpenCarrierCompressionOptions options;
    options.trigger_markers = 1;
    options.phase_subcells = 1;
    options.threads = 4;
    options.particle_shape = ev::WeightedPICParticleShape::tsc;
    options.relative_moment_tolerance = 2.0e-10;
    auto potential = [](double radius) {
        return 0.5 * radius * radius;
    };
    auto energy_moments = [&](const auto& values) {
        std::array<long double, 2> result{};
        for (const auto& value : values) {
            const long double h = potential(value.r)
                + 0.5L * value.w * value.w
                + value.eta / (static_cast<long double>(value.r) * value.r);
            result[0] += value.mass * h;
            result[1] += value.mass * h * h;
        }
        return result;
    };
    const auto energy_before = energy_moments(original);
    auto internal_energy = [](const auto& values) {
        long double result = 0.0L;
        for (const auto& value : values) {
            result += value.internal_energy;
        }
        return result;
    };
    auto internal_state_moments = [](const auto& values) {
        std::array<long double, 3> result{};
        for (const auto& value : values) {
            result[0] += value.mass
                * value.internal_velocity_dispersion;
            result[1] += value.internal_energy;
            result[2] += value.mass * value.w
                * value.internal_velocity_dispersion;
        }
        return result;
    };
    const long double internal_before = internal_energy(original);
    const auto internal_state_before = internal_state_moments(original);
    const auto diagnostics = ev::compress_canonical_open_carriers(
        marker, geometry, options, potential);
    const auto after = ev::deposit_weighted_vlasov_pic_parallel(
        marker, mesh, ev::WeightedPICParticleShape::tsc, 4, true);
    const auto energy_after = energy_moments(marker);
    const long double internal_after = internal_energy(marker);
    const auto internal_state_after = internal_state_moments(marker);

    assert(diagnostics.applied);
    assert(diagnostics.active_before == 6000);
    assert(diagnostics.active_after < diagnostics.active_before);
    assert(marker.size() == diagnostics.active_after + 2);
    assert(diagnostics.maximum_relative_moment_residual
        < options.relative_moment_tolerance);
    assert(std::count_if(
        marker.begin(),
        marker.end(),
        [](const ev::WeightedPICMarker& value) {
            return value.state != ev::WeightedPICMarkerState::active;
        }) == 2);
    for (const ev::WeightedPICMarker& value : marker) {
        assert(value.mass > 0.0);
        ev::validate_weighted_pic_marker(
            value, "Open-carrier compression regression output");
    }

    require_close(
        before.active_marker_mass,
        after.active_marker_mass,
        before.active_marker_mass);
    require_close(
        before.hidden_inner_mass,
        after.hidden_inner_mass,
        before.active_marker_mass);
    assert(before.hidden_inner_mass > 0.0);
    require_close(
        static_cast<double>(energy_before[0]),
        static_cast<double>(energy_after[0]),
        static_cast<double>(std::abs(energy_before[0])));
    require_close(
        static_cast<double>(energy_before[1]),
        static_cast<double>(energy_after[1]),
        static_cast<double>(std::abs(energy_before[1])));
    require_close(
        static_cast<double>(internal_before),
        static_cast<double>(internal_after),
        static_cast<double>(std::abs(internal_before)));
    for (std::size_t q = 0; q < internal_state_before.size(); ++q) {
        require_close(
            static_cast<double>(internal_state_before[q]),
            static_cast<double>(internal_state_after[q]),
            static_cast<double>(std::abs(internal_state_before[q])));
    }
    assert(diagnostics.effective_sample_size_before > 0.0);
    assert(diagnostics.effective_sample_size_after > 0.0);
    require_close(
        diagnostics.active_mass_before,
        diagnostics.active_mass_after,
        diagnostics.active_mass_before);
    assert(diagnostics.minimum_positive_mass_after > 0.0);
    assert(diagnostics.maximum_mass_after
        >= diagnostics.minimum_positive_mass_after);
    assert(diagnostics.maximum_mass_after
        <= diagnostics.maximum_mass_before
            * (1.0 + 32.0 * std::numeric_limits<double>::epsilon()));
    for (std::size_t i = 0; i < before.shell_mass.size(); ++i) {
        require_close(
            before.shell_mass[i], after.shell_mass[i], before.active_marker_mass);
        require_close(
            before.shell_radial_momentum[i],
            after.shell_radial_momentum[i],
            std::abs(before.shell_radial_momentum[i])
                + before.active_marker_mass);
        require_close(
            before.shell_radial_second_moment[i],
            after.shell_radial_second_moment[i],
            before.shell_radial_second_moment[i]
                + before.active_marker_mass);
        require_close(
            before.shell_tangential_second_moment[i],
            after.shell_tangential_second_moment[i],
            before.shell_tangential_second_moment[i]
                + before.active_marker_mass);
        require_close(
            before.shell_radial_energy_flux[i],
            after.shell_radial_energy_flux[i],
            std::abs(before.shell_radial_energy_flux[i])
                + before.active_marker_mass);
    }

    std::vector<ev::WeightedPICMarker> repeated = original;
    ev::CanonicalOpenCarrierCompressionOptions single_thread = options;
    single_thread.threads = 1;
    const auto repeated_diagnostics = ev::compress_canonical_open_carriers(
        repeated, geometry, single_thread, potential);
    assert(repeated_diagnostics.active_after == diagnostics.active_after);
    assert(repeated.size() == marker.size());
    for (std::size_t q = 0; q < marker.size(); ++q) {
        assert(repeated[q].id == marker[q].id);
        assert(repeated[q].mass == marker[q].mass);
        assert(repeated[q].r == marker[q].r);
        assert(repeated[q].w == marker[q].w);
        assert(repeated[q].eta == marker[q].eta);
        assert(repeated[q].state == marker[q].state);
    }

    // Choosing the lower-variance null-space endpoint must not change any
    // constrained moment.  On this deliberately unequal-weight population it
    // should also retain at least as much mass-weight effective resolution as
    // the minimum-step endpoint.
    std::vector<ev::WeightedPICMarker> legacy = original;
    ev::CanonicalOpenCarrierCompressionOptions legacy_options = options;
    legacy_options.minimize_retained_weight_variance = false;
    const auto legacy_diagnostics = ev::compress_canonical_open_carriers(
        legacy, geometry, legacy_options, potential);
    assert(legacy_diagnostics.applied);
    assert(diagnostics.effective_sample_size_after
        >= legacy_diagnostics.effective_sample_size_after
            * (1.0 - 1.0e-13));
    const auto legacy_internal_state = internal_state_moments(legacy);
    for (std::size_t q = 0; q < internal_state_before.size(); ++q) {
        require_close(
            static_cast<double>(internal_state_before[q]),
            static_cast<double>(legacy_internal_state[q]),
            static_cast<double>(std::abs(internal_state_before[q])));
    }

    std::vector<ev::WeightedPICMarker> floored = original;
    ev::CanonicalOpenCarrierCompressionOptions floor_options = options;
    floor_options.minimum_total_markers = 5900;
    floor_options.preserve_source_cell = true;
    const auto floor_diagnostics = ev::compress_canonical_open_carriers(
        floored, geometry, floor_options, potential);
    const auto floor_deposit = ev::deposit_weighted_vlasov_pic_parallel(
        floored, mesh, ev::WeightedPICParticleShape::tsc, 4, true);
    assert(floor_diagnostics.applied);
    assert(floored.size() >= floor_options.minimum_total_markers);
    assert(floored.size() < original.size());
    assert(floor_diagnostics.groups_retained_for_marker_floor > 0);
    require_close(
        before.active_marker_mass,
        floor_deposit.active_marker_mass,
        before.active_marker_mass);
    const auto floor_energy = energy_moments(floored);
    require_close(
        static_cast<double>(energy_before[0]),
        static_cast<double>(floor_energy[0]),
        static_cast<double>(std::abs(energy_before[0])));
    require_close(
        static_cast<double>(energy_before[1]),
        static_cast<double>(floor_energy[1]),
        static_cast<double>(std::abs(energy_before[1])));
    std::array<long double, 3> source_mass_before{};
    std::array<long double, 3> source_mass_after{};
    for (const auto& value : original) {
        if (value.state == ev::WeightedPICMarkerState::active) {
            source_mass_before.at(
                static_cast<std::size_t>(value.source_cell)) += value.mass;
        }
    }
    for (const auto& value : floored) {
        if (value.state == ev::WeightedPICMarkerState::active) {
            source_mass_after.at(
                static_cast<std::size_t>(value.source_cell)) += value.mass;
        }
    }
    for (std::size_t q = 0; q < source_mass_before.size(); ++q) {
        require_close(
            static_cast<double>(source_mass_before[q]),
            static_cast<double>(source_mass_after[q]),
            static_cast<double>(source_mass_before[q]));
    }

    std::cout << "Canonical open compression: active="
              << diagnostics.active_before << " -> "
              << diagnostics.active_after
              << " groups=" << diagnostics.groups
              << " moment_residual="
              << diagnostics.maximum_relative_moment_residual << '\n';
}
