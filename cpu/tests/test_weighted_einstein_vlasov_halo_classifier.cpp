#include "ev/weighted_einstein_vlasov_halo_classifier.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool failed(
    const ev::EinsteinVlasovHaloClassification& result,
    const std::string& name) {
    return std::find(
        result.failed_checks.begin(), result.failed_checks.end(), name)
        != result.failed_checks.end();
}

std::vector<ev::EinsteinVlasovHaloHistorySample> balanced_halo_fixture() {
    std::vector<ev::EinsteinVlasovHaloHistorySample> result;
    for (int i = 0; i <= 20; ++i) {
        const double time = 5.0 * i;
        ev::EinsteinVlasovHaloHistorySample row;
        row.time = time;
        row.r50 = 4.0 - 0.01 * std::min(i, 10);
        row.r90 = 5.0 - 0.012 * std::min(i, 10);
        row.maximum_compactness = 0.35 + 0.001 * std::min(i, 10);
        row.anisotropic_jeans_relative_l1 = 0.32 - 0.01 * std::min(i, 12);
        if (i == 6) {
            // A resolved contraction exceeding five percent is required; a
            // merely unchanged initial equilibrium is not called virialized.
            row.r50 = 3.70;
        }
        if (i >= 14) {
            const double phase = static_cast<double>(i - 14);
            row.r50 = 3.86 + 0.015 * std::sin(phase);
            row.r90 = 4.84 + 0.018 * std::cos(phase);
            row.maximum_compactness = 0.36
                + 0.003 * std::sin(0.7 * phase);
            row.anisotropic_jeans_relative_l1 = 0.19
                + 0.015 * std::cos(phase);
        }
        result.push_back(row);
    }
    return result;
}

ev::EinsteinVlasovHaloClassifierInput balanced_input() {
    ev::EinsteinVlasovHaloClassifierInput result;
    result.fixed_family_binding_diagnostics_available = true;
    result.fixed_family_binding_fraction = 0.04;
    result.horizon_formed = false;
    result.configured_end_time = 100.0;
    result.initial_rest_weight = 1.0;
    result.escaped_rest_weight = 1.0e-3;
    return result;
}

}  // namespace

int main() {
    const auto balanced = balanced_halo_fixture();
    const auto input = balanced_input();
    const auto halo = ev::classify_einstein_vlasov_halo(balanced, input);
    assert(halo.virialized_halo);
    assert(halo.failed_checks.empty());
    assert(halo.history_samples == 21);
    assert(halo.late_samples == 7);
    assert(halo.late_jeans_samples == 7);
    assert(halo.minimum_late_jeans_samples == 6);
    assert(halo.completed_end_time);
    assert(halo.positive_binding_energy);
    assert(halo.no_horizon);
    assert(halo.escape_small);
    assert(halo.compression_occurred);
    assert(halo.not_near_horizon_collapse);
    assert(halo.maximum_history_compactness < 0.99);
    assert(halo.late_anisotropic_jeans_relative_l1_p90 < 0.25);

    // A positive
    // binding, contraction, and a small final radius all hold, but the trace
    // is near trapped and badly violates relativistic Jeans balance.
    auto near_trapped = balanced;
    for (std::size_t i = 14; i < near_trapped.size(); ++i) {
        near_trapped[i].maximum_compactness = 0.996
            + 2.0e-4 * std::sin(static_cast<double>(i));
        near_trapped[i].anisotropic_jeans_relative_l1 = 0.80
            + 0.01 * std::cos(static_cast<double>(i));
    }
    const auto false_halo =
        ev::classify_einstein_vlasov_halo(near_trapped, input);
    assert(!false_halo.virialized_halo);
    assert(false_halo.maximum_history_compactness > 0.995);
    assert(failed(false_halo, "not_near_horizon_collapse"));
    assert(failed(false_halo, "late_anisotropic_jeans_balance"));

    auto missing_jeans = balanced;
    for (std::size_t i = 14; i < missing_jeans.size(); ++i) {
        missing_jeans[i].anisotropic_jeans_relative_l1 = -1.0;
    }
    const auto no_jeans =
        ev::classify_einstein_vlasov_halo(missing_jeans, input);
    assert(!no_jeans.virialized_halo);
    assert(failed(no_jeans, "late_anisotropic_jeans_samples"));
    assert(failed(no_jeans, "late_anisotropic_jeans_balance"));
    assert(failed(no_jeans, "late_anisotropic_jeans_nonsecular"));

    const std::vector<ev::EinsteinVlasovHaloHistorySample> missing_history;
    const auto missing =
        ev::classify_einstein_vlasov_halo(missing_history, input);
    assert(!missing.virialized_halo);
    assert(failed(missing, "valid_history"));

    const std::vector<ev::EinsteinVlasovHaloHistorySample> one_sample{
        balanced.front()};
    const auto one =
        ev::classify_einstein_vlasov_halo(one_sample, input);
    assert(!one.virialized_halo);
    assert(failed(one, "valid_history"));

    std::vector<ev::EinsteinVlasovHaloHistorySample> short_history(
        balanced.end() - 4, balanced.end());
    const auto insufficient =
        ev::classify_einstein_vlasov_halo(short_history, input);
    assert(!insufficient.virialized_halo);
    assert(failed(insufficient, "late_series_available"));

    auto incomplete_input = input;
    incomplete_input.configured_end_time = 120.0;
    const auto incomplete =
        ev::classify_einstein_vlasov_halo(balanced, incomplete_input);
    assert(!incomplete.virialized_halo);
    assert(failed(incomplete, "completed_end_time"));

    auto horizon_input = input;
    horizon_input.horizon_formed = true;
    const auto horizon =
        ev::classify_einstein_vlasov_halo(balanced, horizon_input);
    assert(!horizon.virialized_halo);
    assert(failed(horizon, "no_horizon"));

    auto escaping_input = input;
    escaping_input.escaped_rest_weight = 0.051;
    const auto escaping =
        ev::classify_einstein_vlasov_halo(balanced, escaping_input);
    assert(!escaping.virialized_halo);
    assert(failed(escaping, "escape_small"));

    auto unbound_input = input;
    unbound_input.fixed_family_binding_fraction = -1.0e-3;
    const auto unbound =
        ev::classify_einstein_vlasov_halo(balanced, unbound_input);
    assert(!unbound.virialized_halo);
    assert(failed(unbound, "fixed_family_positive_binding_energy"));

    std::cout << "Einstein--Vlasov shared halo classifier tests passed.\n";
    return 0;
}
