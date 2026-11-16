#pragma once

// Reconstruct the spherical Schwarzschild metric from radial matter moments.
// In G=c=1 units the constraints give m_prime=4pi*r^2*rho,
// exp(-2lambda)=1-2m/r and mu_prime=(m/r^2+4pi*r*p)/(1-2m/r).
// The outer lapse is normalized to the exterior Schwarzschild solution.
// Outputs retain the raw compactness gap together with metric coefficients
// and derivatives used by the characteristic equations.

#include "ev/moments.hpp"

namespace ev {

// Solve the spherical Einstein constraints after taking Vlasov moments of f.
// The outer boundary condition is asymptotically Schwarzschild-like:
// mu(r_max)=0.5 log(1-2M/r_max).
/// Solves the spherical Einstein constraint equations for m, lambda, mu, and derivatives.
Metric solve_metric(
    const Moments& moments,
    const std::vector<double>& r,
    const Params& p) {
    Metric metric;
    // The finite first radial node omits an inner volume. With no prescribed
    // central mass, approximate that regular inner matter mass by constant
    // central density. A specified central mass supplies the inner boundary
    // integration constant instead.
    const double inner_matter_mass = (p.central_mass > 0.0)
        ? 0.0
        : (4.0 * PI / 3.0) * moments.rho.front() * r.front() * r.front() * r.front();
    metric.m.assign(p.Nr, p.central_mass + inner_matter_mass);
    metric.lambda.assign(p.Nr, 0.0);
    metric.mu.assign(p.Nr, 0.0);
    metric.lambda_r.assign(p.Nr, 0.0);
    metric.mu_r.assign(p.Nr, 0.0);
    metric.lambda_t.assign(p.Nr, 0.0);
    metric.gap.assign(p.Nr, 1.0);

    // Integrate the enclosed energy density outward with the radial
    // trapezoidal rule, retaining the prescribed inner mass contribution.
    for (int i = 1; i < p.Nr; ++i) {
        const double y0 = r[i - 1] * r[i - 1] * moments.rho[i - 1];
        const double y1 = r[i] * r[i] * moments.rho[i];
        metric.m[i] = metric.m[i - 1] + 4.0 * PI * 0.5 * (y0 + y1) * (r[i] - r[i - 1]);
    }

    metric.min_gap = 1.0e300;
    for (int i = 0; i < p.Nr; ++i) {
        // Store the physical gap before applying the evaluation floor. The
        // floor keeps logarithms and divisions defined; min_gap still records
        // the actual approach to the polar-areal coordinate limit.
        const double gap_raw = 1.0 - 2.0 * metric.m[i] / r[i];
        metric.gap[i] = gap_raw;
        metric.min_gap = std::min(metric.min_gap, gap_raw);
        const double gap = std::max(gap_raw, p.metric_floor);
        metric.lambda[i] = -0.5 * std::log(gap);
        metric.lambda_r[i] = (4.0 * PI * r[i] * moments.rho[i] - metric.m[i] / (r[i] * r[i])) / gap;
        metric.mu_r[i] = (metric.m[i] / (r[i] * r[i]) + 4.0 * PI * r[i] * moments.p[i]) / gap;
    }

    // Fix the additive time-normalization constant in mu at the outer
    // boundary, then integrate its known radial derivative inward.
    const double outer_gap = std::max(1.0 - 2.0 * metric.m.back() / r.back(), p.metric_floor);
    metric.mu.back() = 0.5 * std::log(outer_gap);
    for (int i = p.Nr - 2; i >= 0; --i) {
        const double dr = r[i + 1] - r[i];
        metric.mu[i] = metric.mu[i + 1] - 0.5 * (metric.mu_r[i + 1] + metric.mu_r[i]) * dr;
    }

    for (int i = 0; i < p.Nr; ++i) {
        // The radial energy current determines the metric time derivative.
        // This term enters momentum transport in a time-dependent spacetime.
        metric.lambda_t[i] = -4.0 * PI * r[i] * std::exp(metric.lambda[i] + metric.mu[i]) * moments.j[i];
    }
    return metric;
}

}  // namespace ev
