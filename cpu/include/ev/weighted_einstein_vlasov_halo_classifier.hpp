
// Classify the saved finite evolution using explicit halo-settlement criteria.
// Inputs are physical radius/compactness/Jeans-residual histories, a configured
// end time, binding evidence and escaped rest mass. Output retains every check
// and each failed condition. The same dimensionless rules can be applied within
// an SC-time history or within an EF-time history; they do not identify events
// on those different slices or establish a continuum stability theorem.

#ifndef EV_WEIGHTED_EINSTEIN_VLASOV_HALO_CLASSIFIER_HPP
#define EV_WEIGHTED_EINSTEIN_VLASOV_HALO_CLASSIFIER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ev {

/// One coordinate-neutral saved diagnostic sample used to decide whether a
/// high-compactness Einstein--Vlasov evolution has actually settled.  The
/// time coordinate may be Schwarzschild t or ingoing EF v; the classifier
/// only uses dimensionless trends over the final fraction of that coordinate
/// interval and applies the same rules to both modules.
struct EinsteinVlasovHaloHistorySample {
    double time = 0.0;
    double r50 = 0.0;
    double r90 = 0.0;
    double maximum_compactness = 0.0;
    // A negative value is the explicit sentinel for an unavailable residual.
    double anisotropic_jeans_relative_l1 = -1.0;
};

/// Conservative production thresholds.  These defaults intentionally match
/// cpu/tools/analyze_einstein_vlasov_wpic.py.  The additional Cmax<0.99
/// gate prevents a horizonless flag caused solely by finite output cadence or
/// a polar-areal stopping tolerance from certifying a near-trapped state as a
/// halo.
// These thresholds define the finite diagnostic classification. Completion,
// binding, actual compression, small escape and sufficient late samples are
// all required in addition to bounded late-time trends.
struct EinsteinVlasovHaloClassifierThresholds {
    double end_time_relative_tolerance = 1.0e-6;
    double late_window_fraction = 0.30;
    std::size_t minimum_late_samples = 5;
    double compression_fraction = 0.95;
    double maximum_late_r50_over_initial = 1.50;
    double maximum_r50_relative_spread = 0.75;
    double maximum_dimensionless_r50_trend = 0.25;
    double maximum_dimensionless_r90_trend = 0.25;
    double maximum_compactness_absolute_spread = 0.05;
    double maximum_compactness_absolute_trend = 0.05;
    double maximum_anisotropic_jeans_relative_l1 = 0.25;
    double maximum_anisotropic_jeans_late_increase = 0.05;
    double minimum_late_jeans_fraction = 0.75;
    double maximum_escaped_rest_fraction = 0.05;
    double near_horizon_compactness = 0.99;
};

struct EinsteinVlasovHaloClassifierInput {
    bool fixed_family_binding_diagnostics_available = false;
    double fixed_family_binding_fraction = -1.0;
    bool horizon_formed = false;
    double configured_end_time = -1.0;
    double initial_rest_weight = -1.0;
    double escaped_rest_weight = -1.0;
};

/// The complete fail-closed audit.  Unavailable real-valued diagnostics retain
/// the sentinel -1, and failed_checks contains stable machine-readable names.
struct EinsteinVlasovHaloClassification {
    EinsteinVlasovHaloClassifierThresholds thresholds;
    bool virialized_halo = false;
    bool history_valid = false;
    bool completed_end_time = false;
    bool positive_binding_energy = false;
    bool no_horizon = false;
    bool escape_small = false;
    bool compression_occurred = false;
    bool late_series_available = false;
    bool not_near_horizon_collapse = false;
    std::size_t history_samples = 0;
    std::size_t late_samples = 0;
    std::size_t late_jeans_samples = 0;
    std::size_t minimum_late_jeans_samples = 0;
    double final_time = -1.0;
    double minimum_r50 = -1.0;
    double maximum_history_compactness = -1.0;
    double escaped_rest_fraction = -1.0;
    double late_r50_median = -1.0;
    double late_r50_relative_spread = -1.0;
    double late_r50_dimensionless_trend = -1.0;
    double late_r90_dimensionless_trend = -1.0;
    double late_compactness_absolute_spread = -1.0;
    double late_compactness_absolute_trend = -1.0;
    double late_anisotropic_jeans_relative_l1_median = -1.0;
    double late_anisotropic_jeans_relative_l1_p90 = -1.0;
    double late_anisotropic_jeans_absolute_trend = -1.0;
    std::vector<std::string> failed_checks;
};

namespace weighted_einstein_vlasov_halo_classifier_detail {

inline bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

inline double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction
        * static_cast<double>(values.size() - 1);
    const std::size_t left = static_cast<std::size_t>(std::floor(position));
    const std::size_t right = static_cast<std::size_t>(std::ceil(position));
    if (left == right) {
        return values[left];
    }
    const double weight = position - static_cast<double>(left);
    return (1.0 - weight) * values[left] + weight * values[right];
}

// Fit a straight trend against actual saved times, centering both
// variables before accumulation. A zero time baseline has no resolved
// trend and returns an unavailable nonfinite result.
inline double linear_slope(
    const std::vector<double>& times,
    const std::vector<double>& values) {
    const double center_time = std::accumulate(
        times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    const double center_value = std::accumulate(
        values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
    long double numerator = 0.0L;
    long double denominator = 0.0L;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const long double dt = times[i] - center_time;
        numerator += dt * (values[i] - center_value);
        denominator += dt * dt;
    }
    if (!(denominator > 0.0L)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(numerator / denominator);
}

inline void validate_thresholds(
    const EinsteinVlasovHaloClassifierThresholds& value) {
    const bool valid =
        finite_positive(value.end_time_relative_tolerance)
        && value.end_time_relative_tolerance < 1.0
        && finite_positive(value.late_window_fraction)
        && value.late_window_fraction <= 1.0
        && value.minimum_late_samples >= 2
        && finite_positive(value.compression_fraction)
        && value.compression_fraction < 1.0
        && finite_positive(value.maximum_late_r50_over_initial)
        && finite_positive(value.maximum_r50_relative_spread)
        && finite_positive(value.maximum_dimensionless_r50_trend)
        && finite_positive(value.maximum_dimensionless_r90_trend)
        && finite_positive(value.maximum_compactness_absolute_spread)
        && finite_positive(value.maximum_compactness_absolute_trend)
        && finite_positive(value.maximum_anisotropic_jeans_relative_l1)
        && finite_positive(value.maximum_anisotropic_jeans_late_increase)
        && finite_positive(value.minimum_late_jeans_fraction)
        && value.minimum_late_jeans_fraction <= 1.0
        && finite_positive(value.maximum_escaped_rest_fraction)
        && finite_positive(value.near_horizon_compactness)
        && value.near_horizon_compactness < 1.0;
    if (!valid) {
        throw std::invalid_argument(
            "Invalid Einstein--Vlasov halo-classifier thresholds.");
    }
}

inline void fail(
    EinsteinVlasovHaloClassification& result,
    const char* check,
    bool passed) {
    if (!passed) {
        result.failed_checks.emplace_back(check);
    }
}

}  // namespace weighted_einstein_vlasov_halo_classifier_detail

inline EinsteinVlasovHaloClassification classify_einstein_vlasov_halo(
    const std::vector<EinsteinVlasovHaloHistorySample>& history,
    const EinsteinVlasovHaloClassifierInput& input,
    const EinsteinVlasovHaloClassifierThresholds& thresholds = {}) {
    namespace detail = weighted_einstein_vlasov_halo_classifier_detail;
    detail::validate_thresholds(thresholds);

    EinsteinVlasovHaloClassification result;
    result.thresholds = thresholds;
    result.history_samples = history.size();
    // Positive binding is supplied by the separately defined fixed physical
    // family diagnostic. Radius settlement alone supplies no binding-energy
    // measurement.
    result.positive_binding_energy =
        input.fixed_family_binding_diagnostics_available
        && std::isfinite(input.fixed_family_binding_fraction)
        && input.fixed_family_binding_fraction > 0.0;
    result.no_horizon = !input.horizon_formed;
    result.escape_small = detail::finite_positive(input.initial_rest_weight)
        && std::isfinite(input.escaped_rest_weight)
        && input.escaped_rest_weight >= 0.0;
    if (result.escape_small) {
        result.escaped_rest_fraction =
            input.escaped_rest_weight / input.initial_rest_weight;
        result.escape_small = result.escaped_rest_fraction
            <= thresholds.maximum_escaped_rest_fraction;
    }

    // Require finite physical samples and strictly increasing saved time.
    // The negative Jeans sentinel remains unavailable data and is counted
    // separately when the late balance test is formed.
    result.history_valid = history.size() >= 2;
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& row = history[i];
        result.history_valid = result.history_valid
            && std::isfinite(row.time)
            && detail::finite_positive(row.r50)
            && detail::finite_positive(row.r90)
            && std::isfinite(row.maximum_compactness)
            && row.maximum_compactness >= 0.0
            && std::isfinite(row.anisotropic_jeans_relative_l1)
            && (i == 0 || row.time > history[i - 1].time);
    }

    detail::fail(result, "valid_history", result.history_valid);
    detail::fail(
        result, "fixed_family_positive_binding_energy",
        result.positive_binding_energy);
    detail::fail(result, "no_horizon", result.no_horizon);
    detail::fail(result, "escape_small", result.escape_small);
    if (!result.history_valid) {
        return result;
    }

    const double initial_r50 = history.front().r50;
    const double initial_r90 = history.front().r90;
    result.final_time = history.back().time;
    result.completed_end_time = std::isfinite(input.configured_end_time)
        && result.final_time >= (1.0 - thresholds.end_time_relative_tolerance)
            * input.configured_end_time;
    result.minimum_r50 = history.front().r50;
    result.maximum_history_compactness = history.front().maximum_compactness;
    for (const auto& row : history) {
        result.minimum_r50 = std::min(result.minimum_r50, row.r50);
        result.maximum_history_compactness = std::max(
            result.maximum_history_compactness,
            row.maximum_compactness);
    }
    // A compressed-and-settled halo must first have contracted from its
    // initial half-mass radius. A horizonless flag alone also needs the
    // separate near-horizon compactness screen below.
    result.compression_occurred = result.minimum_r50
        < thresholds.compression_fraction * initial_r50;
    result.not_near_horizon_collapse = result.maximum_history_compactness
        < thresholds.near_horizon_compactness;

    detail::fail(result, "completed_end_time", result.completed_end_time);
    detail::fail(result, "compression_occurred", result.compression_occurred);
    detail::fail(
        result, "not_near_horizon_collapse",
        result.not_near_horizon_collapse);

    const double duration = result.final_time - history.front().time;
    // Choose the final fraction of the actual observed interval. At least
    // the configured number of samples must lie there before fitting
    // a late radius, metric or Jeans trend.
    const double cutoff = result.final_time
        - thresholds.late_window_fraction * std::max(0.0, duration);
    std::vector<EinsteinVlasovHaloHistorySample> late;
    for (const auto& row : history) {
        if (row.time >= cutoff) {
            late.push_back(row);
        }
    }
    result.late_samples = late.size();
    result.late_series_available =
        late.size() >= thresholds.minimum_late_samples;
    detail::fail(
        result, "late_series_available", result.late_series_available);
    if (!result.late_series_available) {
        return result;
    }

    std::vector<double> late_times;
    std::vector<double> late_r50;
    std::vector<double> late_r90;
    std::vector<double> late_compactness;
    std::vector<double> jeans_times;
    std::vector<double> late_jeans;
    late_times.reserve(late.size());
    late_r50.reserve(late.size());
    late_r90.reserve(late.size());
    late_compactness.reserve(late.size());
    for (const auto& row : late) {
        late_times.push_back(row.time);
        late_r50.push_back(row.r50);
        late_r90.push_back(row.r90);
        late_compactness.push_back(row.maximum_compactness);
        if (row.anisotropic_jeans_relative_l1 >= 0.0) {
            jeans_times.push_back(row.time);
            late_jeans.push_back(row.anisotropic_jeans_relative_l1);
        }
    }
    result.late_jeans_samples = late_jeans.size();
    result.minimum_late_jeans_samples = std::max<std::size_t>(
        3,
        static_cast<std::size_t>(std::ceil(
            thresholds.minimum_late_jeans_fraction
            * static_cast<double>(late.size()))));

    // Percentile spreads reduce sensitivity to one saved outlier. Radius
    // trends are made dimensionless with the late baseline and the initial
    // radius, while compactness is already dimensionless.
    result.late_r50_median = detail::percentile(late_r50, 0.5);
    result.late_r50_relative_spread = (
        detail::percentile(late_r50, 0.9)
        - detail::percentile(late_r50, 0.1))
        / std::max(result.late_r50_median, 1.0e-300);
    const double baseline = late_times.back() - late_times.front();
    result.late_r50_dimensionless_trend = detail::linear_slope(
        late_times, late_r50) * baseline / initial_r50;
    result.late_r90_dimensionless_trend = detail::linear_slope(
        late_times, late_r90) * baseline / initial_r90;
    result.late_compactness_absolute_spread =
        detail::percentile(late_compactness, 0.9)
        - detail::percentile(late_compactness, 0.1);
    result.late_compactness_absolute_trend = detail::linear_slope(
        late_times, late_compactness) * baseline;

    const bool late_radius_bounded = result.late_r50_median
        <= thresholds.maximum_late_r50_over_initial * initial_r50;
    const bool late_radius_nonsecular =
        std::abs(result.late_r50_dimensionless_trend)
        <= thresholds.maximum_dimensionless_r50_trend;
    const bool late_outer_profile_nonsecular =
        std::abs(result.late_r90_dimensionless_trend)
        <= thresholds.maximum_dimensionless_r90_trend;
    const bool late_radius_spread_bounded =
        result.late_r50_relative_spread
        <= thresholds.maximum_r50_relative_spread;
    const bool late_metric_spread_bounded =
        result.late_compactness_absolute_spread
        <= thresholds.maximum_compactness_absolute_spread;
    const bool late_metric_nonsecular =
        std::abs(result.late_compactness_absolute_trend)
        <= thresholds.maximum_compactness_absolute_trend;

    // A sufficient fraction of late samples must carry an actual
    // nonnegative Jeans residual. Bound both its median and upper percentile
    // and check whether the measured imbalance increases secularly.
    const bool late_jeans_available = late_jeans.size()
        >= result.minimum_late_jeans_samples;
    bool late_jeans_balance = false;
    bool late_jeans_nonsecular = false;
    if (late_jeans_available) {
        result.late_anisotropic_jeans_relative_l1_median =
            detail::percentile(late_jeans, 0.5);
        result.late_anisotropic_jeans_relative_l1_p90 =
            detail::percentile(late_jeans, 0.9);
        const double jeans_baseline = jeans_times.back() - jeans_times.front();
        result.late_anisotropic_jeans_absolute_trend =
            detail::linear_slope(jeans_times, late_jeans) * jeans_baseline;
        late_jeans_balance =
            result.late_anisotropic_jeans_relative_l1_median
                <= thresholds.maximum_anisotropic_jeans_relative_l1
            && result.late_anisotropic_jeans_relative_l1_p90
                <= thresholds.maximum_anisotropic_jeans_relative_l1;
        late_jeans_nonsecular =
            result.late_anisotropic_jeans_absolute_trend
            <= thresholds.maximum_anisotropic_jeans_late_increase;
    }

    detail::fail(result, "late_radius_bounded", late_radius_bounded);
    detail::fail(result, "late_radius_nonsecular", late_radius_nonsecular);
    detail::fail(
        result, "late_outer_profile_nonsecular",
        late_outer_profile_nonsecular);
    detail::fail(
        result, "late_radius_spread_bounded",
        late_radius_spread_bounded);
    detail::fail(
        result, "late_metric_spread_bounded",
        late_metric_spread_bounded);
    detail::fail(result, "late_metric_nonsecular", late_metric_nonsecular);
    detail::fail(
        result, "late_anisotropic_jeans_samples", late_jeans_available);
    detail::fail(
        result, "late_anisotropic_jeans_balance", late_jeans_balance);
    detail::fail(
        result, "late_anisotropic_jeans_nonsecular",
        late_jeans_nonsecular);

    // This Boolean means that every declared finite diagnostic criterion
    // passed. The full metrics and failure list remain the interpretable
    // result for scientific assessment.
    result.virialized_halo = result.failed_checks.empty();
    return result;
}

}  // namespace ev

#endif  // EV_WEIGHTED_EINSTEIN_VLASOV_HALO_CLASSIFIER_HPP
