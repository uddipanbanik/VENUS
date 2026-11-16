#pragma once

// Neutral, spherically symmetric, static Einstein--Vlasov shells for WPIC.
//
// The continuum ansatz is
//
//   f(E,ell) = C (1 - E/E0)_+^k delta(ell-ell0),       ell0 > 0,
//   E        = exp(mu(r)) sqrt(1+w^2+ell0^2/r^2).
//
// Here 1 <= k <= 8 controls the support-edge smoothness (default k=1).
// ell is the angular-momentum magnitude.  The Schwarzschild WPIC module
// uses eta=ell^2/2 and the invariant measure
//
//   4 pi^2 exp(lambda) f dr dw d(ell^2).
//
// Consequently delta(ell-ell0) contributes the Jacobian 2 ell0.  Equivalently
// C2=2*ell0*C is the coefficient of delta(ell^2-ell0^2).  Making this factor
// explicit prevents the common factor-of-2*ell0 normalization error.
//
// With y=log(E0)-mu and W^2=exp(2y)-1-ell0^2/r^2, the static field equations
// are
//
//   m' = 4 pi r^2 rho,
//   y' = -(m/r^2 + 4 pi r p_r)/(1-2m/r).
//
// The inner angular-momentum barrier is an exact vacuum.  At the outer edge R
// of the matter, asymptotic lapse normalization gives
//
//   log(E0) = y(R) + 1/2 log(1-2M/R).
//
// Thus E0 is not imposed with an arbitrary lapse offset: it is obtained by
// matching to the normalized Schwarzschild exterior.  A bracketed shooting
// helper is provided when a particular E0 is desired.
//
// Marker generation uses tensor Gauss--Legendre quadrature and emits adjacent
// (+w,-w) pairs with identical r, eta, and invariant rest weight.  Their
// deposited current therefore cancels pairwise.  The returned weights are
// already invariant Schwarzschild WPIC weights and enter the particle module
// directly, with their canonical volume factor already included.
// A realization-dependent shifted-stratified rule is also available for
// independent quiet-PIC convergence trials.  It applies deterministic
// Cranley--Patterson shifts to equal-measure radial and positive-w lattices;
// changing `realization` changes phase-space locations and weights while
// retaining exact adjacent (+w,-w) cancellation.

#include "ev/weighted_vlasov_pic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ev::static_einstein_vlasov {

struct Parameters {
    // C in C(1-E/E0)_+^k delta(ell-ell0), not the coefficient of
    // delta(ell^2-ell0^2).  cutoff_power=k controls the differentiability
    // of the compact-support edge; k=1 gives a linear cutoff in energy.
    // The defaults select the weakly relativistic shell with
    // C2=2*ell0*C=1, ell0=1, and y(0)=0.13.  It is a useful equilibrium
    // reference point for the equilibrium equations and discretization.
    double amplitude = 0.50;
    double angular_momentum = 1.0;
    double cutoff_power = 1.0;
    // y(0)=log(E0)-mu(0).  Since ell0>0, the center is vacuum and y is
    // constant up to the inner matter edge.
    double central_y = 0.13;

    double maximum_radius = 64.0;
    double initial_radial_step = 1.0e-3;
    double maximum_radial_step = 2.0e-2;
    double minimum_radial_step = 1.0e-10;
    double relative_tolerance = 2.0e-10;
    double absolute_tolerance = 2.0e-12;
    double minimum_compactness_gap = 1.0e-8;
    std::size_t maximum_accepted_steps = 2000000;
};

struct Sources {
    double rho = 0.0;
    double radial_pressure = 0.0;
    // Sum of the two equal orthonormal tangential pressures.  The pressure in
    // either angular direction is tangential_pressure_sum/2.
    double tangential_pressure_sum = 0.0;
    double rest_number_density = 0.0;
    double maximum_radial_momentum = 0.0;
};

struct ProfileNode {
    double radius = 0.0;
    double mass = 0.0;
    double y = 0.0;
    double mu = 0.0;
    double lambda = 0.0;
    double rest_weight = 0.0;
    Sources sources;
    double dm_dr = 0.0;
    double dy_dr = 0.0;
    double drest_weight_dr = 0.0;
};

struct Diagnostics {
    double maximum_compactness = 0.0;
    double minimum_compactness_gap = 1.0;
    // Maximum step-integrated defect of the two radial Einstein constraints,
    // normalized by the total shell mass and total lapse-potential change.
    // A global normalization is essential: a local relative residual is
    // undefined at both support edges, where the exact step change vanishes.
    // Each defect uses Simpson integration of endpoint and Hermite-midpoint
    // right-hand sides.
    double constraint_residual_linf = 0.0;
    // Relative L2 residual of
    //   p_r' + (rho+p_r) mu' - (p_t_sum-2p_r)/r = 0.
    double jeans_residual_relative_l2 = 0.0;
};

struct Solution {
    Parameters parameters;
    double cutoff_energy = 0.0;
    double inner_matter_radius = 0.0;
    double outer_matter_radius = 0.0;
    double adm_mass = 0.0;
    double rest_weight = 0.0;
    double binding_energy = 0.0;
    double binding_fraction = 0.0;
    double central_lapse = 0.0;
    Diagnostics diagnostics;
    std::vector<ProfileNode> profile;
};

enum class MarkerSamplingRule {
    tensor_gauss_legendre,
    shifted_stratified,
};

struct MarkerSamplingOptions {
    std::size_t radial_bins = 256;
    int radial_quadrature_order = 2;
    int positive_momentum_quadrature_order = 8;
    MarkerSamplingRule rule = MarkerSamplingRule::tensor_gauss_legendre;
    // Dedicated production count for shifted_stratified sampling.  When
    // nonzero, exactly 2*shifted_pair_count markers are emitted and the
    // tensor Gauss--Legendre bin/order fields are ignored.  This makes clear
    // that those orders are quadrature controls only for the tensor rule;
    // they do not change a fixed-size rank-one shifted lattice.
    std::size_t shifted_pair_count = 0;
    // Selects an independent deterministic shifted-stratified realization.
    // It is deliberately ignored by the exact tensor Gauss--Legendre rule.
    std::uint64_t realization = 0;
    std::uint64_t first_marker_id = 1;
};

struct MarkerSet {
    std::vector<WeightedPICMarker> markers;
    double represented_rest_weight = 0.0;
    double rest_weight_relative_error = 0.0;
    double maximum_energy_over_cutoff = 0.0;
};

namespace detail {

inline std::uint64_t splitmix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30))
        * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27))
        * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

inline double open_unit_from_hash(std::uint64_t value) {
    // The upper 52 bits are exactly representable in a double.  Adding one
    // half before scaling keeps the shift strictly inside (0,1).
    return (static_cast<double>(splitmix64(value) >> 12) + 0.5)
        * 0x1.0p-52;
}

inline double fractional_part(double value) {
    return value - std::floor(value);
}

inline double shifted_lattice_coordinate(
    std::size_t index,
    std::size_t count,
    double shift) {
    if (count == 0 || index >= count) {
        throw std::runtime_error("Invalid static-shell shifted lattice.");
    }
    const double base = (
        static_cast<double>(index) + 0.5)
        / static_cast<double>(count);
    double coordinate = fractional_part(base + shift);
    // Avoid a rounded endpoint, where the k=1 distribution is exactly zero.
    coordinate = std::max(
        std::nextafter(0.0, 1.0),
        std::min(std::nextafter(1.0, 0.0), coordinate));
    return coordinate;
}

inline std::size_t rank_one_lattice_generator(std::size_t count) {
    if (count < 2) {
        throw std::runtime_error("Static-shell rank-one lattice is too small.");
    }
    constexpr long double inverse_golden_ratio =
        0.61803398874989484820458683436563811772L;
    std::size_t generator = static_cast<std::size_t>(
        inverse_golden_ratio * static_cast<long double>(count));
    generator = std::max<std::size_t>(1, std::min(count - 1, generator));
    while (std::gcd(generator, count) != 1) {
        ++generator;
        if (generator == count) {
            generator = 1;
        }
    }
    return generator;
}

inline std::size_t add_modulo_without_overflow(
    std::size_t value,
    std::size_t increment,
    std::size_t modulus) {
    return value >= modulus - increment
        ? value - (modulus - increment)
        : value + increment;
}

inline void validate_parameters(const Parameters& p) {
    if (!(p.amplitude > 0.0) || !(p.angular_momentum > 0.0)
        || !(p.cutoff_power >= 1.0) || !(p.cutoff_power <= 8.0)
        || !(p.central_y > 0.0) || !(p.maximum_radius > 0.0)
        || !(p.initial_radial_step > 0.0)
        || !(p.maximum_radial_step >= p.initial_radial_step)
        || !(p.minimum_radial_step > 0.0)
        || !(p.minimum_radial_step < p.maximum_radial_step)
        || !(p.relative_tolerance > 0.0)
        || !(p.absolute_tolerance > 0.0)
        || !(p.minimum_compactness_gap > 0.0)
        || !(p.minimum_compactness_gap < 1.0)
        || p.maximum_accepted_steps == 0
        || !std::isfinite(p.amplitude)
        || !std::isfinite(p.angular_momentum)
        || !std::isfinite(p.cutoff_power)
        || !std::isfinite(p.central_y)
        || !std::isfinite(p.maximum_radius)) {
        throw std::runtime_error(
            "Invalid neutral static Einstein-Vlasov shell parameters.");
    }
}

inline std::pair<std::vector<double>, std::vector<double>>
gauss_legendre_rule(int order) {
    if (order == 2) {
        return {{-0.57735026918962576451, 0.57735026918962576451},
                {1.0, 1.0}};
    }
    if (order == 4) {
        return {{
                    -0.86113631159405257522,
                    -0.33998104358485626480,
                    0.33998104358485626480,
                    0.86113631159405257522,
                },
                {
                    0.34785484513745385737,
                    0.65214515486254614263,
                    0.65214515486254614263,
                    0.34785484513745385737,
                }};
    }
    if (order == 8) {
        return {{
                    -0.96028985649753623168,
                    -0.79666647741362673959,
                    -0.52553240991632898582,
                    -0.18343464249564980494,
                    0.18343464249564980494,
                    0.52553240991632898582,
                    0.79666647741362673959,
                    0.96028985649753623168,
                },
                {
                    0.10122853629037625915,
                    0.22238103445337447054,
                    0.31370664587788728734,
                    0.36268378337836198297,
                    0.36268378337836198297,
                    0.31370664587788728734,
                    0.22238103445337447054,
                    0.10122853629037625915,
                }};
    }
    if (order == 16) {
        return {{
                    -0.98940093499164993260,
                    -0.94457502307323257608,
                    -0.86563120238783174388,
                    -0.75540440835500303390,
                    -0.61787624440264374845,
                    -0.45801677765722738634,
                    -0.28160355077925891323,
                    -0.09501250983763744019,
                    0.09501250983763744019,
                    0.28160355077925891323,
                    0.45801677765722738634,
                    0.61787624440264374845,
                    0.75540440835500303390,
                    0.86563120238783174388,
                    0.94457502307323257608,
                    0.98940093499164993260,
                },
                {
                    0.02715245941175409485,
                    0.06225352393864789286,
                    0.09515851168249278481,
                    0.12462897125553387205,
                    0.14959598881657673208,
                    0.16915651939500253819,
                    0.18260341504492358887,
                    0.18945061045506849629,
                    0.18945061045506849629,
                    0.18260341504492358887,
                    0.16915651939500253819,
                    0.14959598881657673208,
                    0.12462897125553387205,
                    0.09515851168249278481,
                    0.06225352393864789286,
                    0.02715245941175409485,
                }};
    }
    throw std::runtime_error(
        "Static-shell Gauss-Legendre order must be 2, 4, 8, or 16.");
}

struct MomentumIntegrals {
    long double energy = 0.0L;
    long double radial_pressure = 0.0L;
    long double tangential_pressure_sum = 0.0L;
    long double rest_number = 0.0L;
    long double maximum_radial_momentum = 0.0L;
};

inline long double support_squared(
    long double radius,
    long double y,
    const Parameters& p) {
    const long double ell = p.angular_momentum;
    return std::expm1(2.0L * y) - ell * ell / (radius * radius);
}

// At a fixed radius, the energy cutoff limits |w| to W(r). Integrate the
// same nonnegative distribution for energy, radial pressure, both tangential
// pressures and rest number, retaining their distinct momentum factors.
inline MomentumIntegrals momentum_integrals(
    double radius,
    double y,
    const Parameters& p) {
    MomentumIntegrals result;
    if (!(radius > 0.0)) {
        return result;
    }
    const long double r = radius;
    const long double yy = y;
    const long double w_squared = support_squared(r, yy, p);
    if (!(w_squared > 0.0L)) {
        return result;
    }
    const long double w_max = std::sqrt(w_squared);
    const long double z = std::exp(yy);
    const long double ell_over_r = p.angular_momentum / r;
    const long double ell_over_r_squared = ell_over_r * ell_over_r;
    const auto rule = gauss_legendre_rule(16);
    for (std::size_t q = 0; q < rule.first.size(); ++q) {
        const long double x = rule.first[q];
        const long double weight = rule.second[q] * w_max;
        const long double w = w_max * x;
        const long double epsilon = std::sqrt(
            1.0L + w * w + ell_over_r_squared);
        // ((z-epsilon)/z)^k, with the base evaluated without cancellation
        // near the cutoff.  k>1 makes the density edge progressively
        // smoother while retaining exact compact support.
        const long double phi_base =
            (w_squared - w * w) / (z * (z + epsilon));
        const long double phi = std::pow(phi_base, p.cutoff_power);
        result.energy += weight * phi * epsilon;
        result.radial_pressure += weight * phi * w * w / epsilon;
        result.tangential_pressure_sum +=
            weight * phi * ell_over_r_squared / epsilon;
        result.rest_number += weight * phi;
    }
    result.maximum_radial_momentum = w_max;
    return result;
}

// Integrating delta(ell-ell0) in the local measure 2*pi*ell*dell*dw/r^2
// supplies the factor 2*pi*C*ell0/r^2. The four moments share this normalization.
inline Sources sources(double radius, double y, const Parameters& p) {
    Sources result;
    if (!(radius > 0.0)) {
        return result;
    }
    const MomentumIntegrals integral = momentum_integrals(radius, y, p);
    const long double factor = 2.0L * PI * p.amplitude
        * p.angular_momentum / (radius * radius);
    result.rho = static_cast<double>(factor * integral.energy);
    result.radial_pressure = static_cast<double>(
        factor * integral.radial_pressure);
    result.tangential_pressure_sum = static_cast<double>(
        factor * integral.tangential_pressure_sum);
    result.rest_number_density = static_cast<double>(
        factor * integral.rest_number);
    result.maximum_radial_momentum = static_cast<double>(
        integral.maximum_radial_momentum);
    return result;
}

struct State {
    double mass = 0.0;
    double y = 0.0;
    double rest_weight = 0.0;
};

struct Derivative {
    double mass = 0.0;
    double y = 0.0;
    double rest_weight = 0.0;
};

inline State add_scaled(
    const State& value,
    const Derivative& derivative,
    double scale) {
    return {
        value.mass + scale * derivative.mass,
        value.y + scale * derivative.y,
        value.rest_weight + scale * derivative.rest_weight,
    };
}

// The radial Einstein equations determine enclosed mass and y=log(E0)-mu.
// Rest weight additionally uses the proper radial volume exp(lambda)*4*pi*r^2 dr.
inline Derivative rhs(
    double radius,
    const State& state,
    const Parameters& p) {
    if (!(radius > 0.0) || !(state.mass >= 0.0)
        || !(state.rest_weight >= 0.0) || !std::isfinite(state.y)) {
        throw std::runtime_error(
            "Static Einstein-Vlasov shell ODE reached an invalid state.");
    }
    const double gap = 1.0 - 2.0 * state.mass / radius;
    if (!(gap > p.minimum_compactness_gap) || !std::isfinite(gap)) {
        throw std::runtime_error(
            "Static Einstein-Vlasov shell left the polar-areal exterior.");
    }
    const Sources matter = sources(radius, state.y, p);
    const double exp_lambda = 1.0 / std::sqrt(gap);
    return {
        4.0 * PI * radius * radius * matter.rho,
        -(state.mass / (radius * radius)
            + 4.0 * PI * radius * matter.radial_pressure) / gap,
        4.0 * PI * radius * radius * exp_lambda
            * matter.rest_number_density,
    };
}

inline State rk4_step(
    double radius,
    const State& state,
    double step,
    const Parameters& p) {
    const Derivative k1 = rhs(radius, state, p);
    const Derivative k2 = rhs(
        radius + 0.5 * step,
        add_scaled(state, k1, 0.5 * step), p);
    const Derivative k3 = rhs(
        radius + 0.5 * step,
        add_scaled(state, k2, 0.5 * step), p);
    const Derivative k4 = rhs(
        radius + step,
        add_scaled(state, k3, step), p);
    return {
        state.mass + step * (k1.mass + 2.0 * k2.mass
            + 2.0 * k3.mass + k4.mass) / 6.0,
        state.y + step * (k1.y + 2.0 * k2.y
            + 2.0 * k3.y + k4.y) / 6.0,
        state.rest_weight + step * (k1.rest_weight
            + 2.0 * k2.rest_weight + 2.0 * k3.rest_weight
            + k4.rest_weight) / 6.0,
    };
}

struct RawNode {
    double radius = 0.0;
    State state;
    Derivative derivative;
};

// Interpolate the solution and its radial derivatives together. This
// continuous profile supplies matter sampling and locates the finite support edge.
inline State hermite_state(
    const RawNode& left,
    const RawNode& right,
    double radius) {
    const double width = right.radius - left.radius;
    if (!(width > 0.0)) {
        throw std::runtime_error("Invalid static-shell Hermite interval.");
    }
    const double x = std::clamp(
        (radius - left.radius) / width, 0.0, 1.0);
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double h00 = 2.0 * x3 - 3.0 * x2 + 1.0;
    const double h10 = x3 - 2.0 * x2 + x;
    const double h01 = -2.0 * x3 + 3.0 * x2;
    const double h11 = x3 - x2;
    auto component = [&](double y0, double d0, double y1, double d1) {
        return h00 * y0 + h10 * width * d0
            + h01 * y1 + h11 * width * d1;
    };
    return {
        component(
            left.state.mass, left.derivative.mass,
            right.state.mass, right.derivative.mass),
        component(
            left.state.y, left.derivative.y,
            right.state.y, right.derivative.y),
        component(
            left.state.rest_weight, left.derivative.rest_weight,
            right.state.rest_weight, right.derivative.rest_weight),
    };
}

inline double state_error(
    const State& coarse,
    const State& fine,
    const Parameters& p) {
    auto component = [&](double first, double second) {
        const double scale = p.absolute_tolerance
            + p.relative_tolerance
                * std::max(std::abs(first), std::abs(second));
        // The n-versus-2n RK4 difference is 15 times the leading local error.
        return std::abs(first - second) / (15.0 * scale);
    };
    return std::max({
        component(coarse.mass, fine.mass),
        component(coarse.y, fine.y),
        component(coarse.rest_weight, fine.rest_weight),
    });
}

inline RawNode make_raw_node(
    double radius,
    const State& state,
    const Parameters& p) {
    return {radius, state, rhs(radius, state, p)};
}

inline State interpolate_profile_state(
    const Solution& solution,
    double radius) {
    if (solution.profile.size() < 2
        || radius < solution.inner_matter_radius
        || radius > solution.outer_matter_radius) {
        throw std::runtime_error(
            "Static-shell profile interpolation radius is out of range.");
    }
    auto upper = std::upper_bound(
        solution.profile.begin(), solution.profile.end(), radius,
        [](double value, const ProfileNode& node) {
            return value < node.radius;
        });
    if (upper == solution.profile.begin()) {
        upper = solution.profile.begin() + 1;
    }
    if (upper == solution.profile.end()) {
        upper = solution.profile.end() - 1;
    }
    const ProfileNode& right = *upper;
    const ProfileNode& left = *(upper - 1);
    RawNode raw_left{
        left.radius,
        {left.mass, left.y, left.rest_weight},
        {left.dm_dr, left.dy_dr, left.drest_weight_dr},
    };
    RawNode raw_right{
        right.radius,
        {right.mass, right.y, right.rest_weight},
        {right.dm_dr, right.dy_dr, right.drest_weight_dr},
    };
    return hermite_state(raw_left, raw_right, radius);
}

// Test static radial force balance by differentiating p(r,y(r)) with the
// chain rule. The anisotropy term uses the sum of both tangential pressures.
inline double jeans_residual(
    const std::vector<RawNode>& raw,
    const Parameters& p) {
    double maximum_rho = 0.0;
    for (const RawNode& node : raw) {
        maximum_rho = std::max(
            maximum_rho, sources(node.radius, node.state.y, p).rho);
    }
    long double residual_norm = 0.0L;
    long double reference_norm = 0.0L;
    for (std::size_t i = 1; i + 1 < raw.size(); ++i) {
        const RawNode& node = raw[i];
        const Sources matter = sources(node.radius, node.state.y, p);
        if (!(matter.rho > 1.0e-7 * maximum_rho)) {
            continue;
        }
        const double radial_delta = std::min(
            1.0e-5 * std::max(1.0, node.radius),
            0.2 * std::min(
                node.radius - raw.front().radius,
                raw.back().radius - node.radius));
        const double y_delta = 1.0e-6
            * std::max(1.0, std::abs(node.state.y));
        if (!(radial_delta > 0.0) || !(y_delta > 0.0)) {
            continue;
        }
        const double dp_dr_at_fixed_y = (
            sources(
                node.radius + radial_delta, node.state.y, p)
                    .radial_pressure
            - sources(
                node.radius - radial_delta, node.state.y, p)
                    .radial_pressure) / (2.0 * radial_delta);
        const double dp_dy_at_fixed_r = (
            sources(
                node.radius, node.state.y + y_delta, p)
                    .radial_pressure
            - sources(
                node.radius, node.state.y - y_delta, p)
                    .radial_pressure) / (2.0 * y_delta);
        const double pressure_derivative = dp_dr_at_fixed_y
            + dp_dy_at_fixed_r * node.derivative.y;
        const double mu_derivative = -node.derivative.y;
        const double gravity =
            (matter.rho + matter.radial_pressure) * mu_derivative;
        const double anisotropy =
            (matter.tangential_pressure_sum
                - 2.0 * matter.radial_pressure) / node.radius;
        const double residual = pressure_derivative + gravity - anisotropy;
        const double reference = std::abs(pressure_derivative)
            + std::abs(gravity) + std::abs(anisotropy);
        const double weight = 0.5
            * (raw[i + 1].radius - raw[i - 1].radius);
        residual_norm += weight * residual * residual;
        reference_norm += weight * reference * reference;
    }
    return reference_norm > 0.0L
        ? std::sqrt(static_cast<double>(residual_norm / reference_norm))
        : 0.0;
}

}  // namespace detail

inline Sources evaluate_sources(
    double radius,
    double y,
    const Parameters& parameters) {
    detail::validate_parameters(parameters);
    return detail::sources(radius, y, parameters);
}

// Coefficient C2 in f=C2(1-E/E0)_+ delta(ell^2-ell0^2).
inline double delta_angular_momentum_squared_amplitude(
    const Parameters& parameters) {
    detail::validate_parameters(parameters);
    return 2.0 * parameters.angular_momentum * parameters.amplitude;
}

// Start at the exact inner angular-momentum barrier with an empty interior.
// Integrate the same distribution and metric outward until the occupied support
// closes, then match the exterior lapse to determine the physical cutoff energy.
inline Solution solve(const Parameters& parameters) {
    detail::validate_parameters(parameters);
    const double ell = parameters.angular_momentum;
    const double inner_radius = ell / std::sqrt(
        std::expm1(2.0 * parameters.central_y));
    if (!(inner_radius > 0.0)
        || !(inner_radius < parameters.maximum_radius)) {
        throw std::runtime_error(
            "Static-shell angular barrier lies outside the radial domain.");
    }

    std::vector<detail::RawNode> raw;
    raw.reserve(4096);
    detail::State state{0.0, parameters.central_y, 0.0};
    double radius = inner_radius;
    raw.push_back(detail::make_raw_node(radius, state, parameters));
    double step = std::min(
        parameters.initial_radial_step,
        parameters.maximum_radius - radius);
    bool matter_seen = false;
    bool outer_edge_found = false;

    for (std::size_t accepted = 0;
         accepted < parameters.maximum_accepted_steps;
         /* incremented only after acceptance */) {
        if (!(radius < parameters.maximum_radius)) {
            break;
        }
        const double remaining = parameters.maximum_radius - radius;
        if (!(remaining > 0.0)) {
            break;
        }
        // The final trial step may legitimately be shorter than the usual
        // adaptive lower bound.  std::clamp would otherwise receive an
        // inverted [minimum,remaining] interval.
        step = remaining <= parameters.minimum_radial_step
            ? remaining
            : std::clamp(
                step,
                parameters.minimum_radial_step,
                std::min(parameters.maximum_radial_step, remaining));
        // Step doubling compares fourth-order integrations over one radial interval.
        // Reject an unresolved radial trial before adding any node to the equilibrium.
        const detail::State coarse = detail::rk4_step(
            radius, state, step, parameters);
        const detail::State half = detail::rk4_step(
            radius, state, 0.5 * step, parameters);
        const detail::State fine = detail::rk4_step(
            radius + 0.5 * step, half, 0.5 * step, parameters);
        const double error = detail::state_error(coarse, fine, parameters);
        if (!(error <= 1.0) || !std::isfinite(error)) {
            step *= std::max(
                0.1, 0.85 * std::pow(std::max(error, 1.0), -0.2));
            if (!(step >= parameters.minimum_radial_step)) {
                throw std::runtime_error(
                    "Static-shell ODE exhausted its minimum radial step.");
            }
            continue;
        }

        const double next_radius = radius + step;
        const detail::RawNode next = detail::make_raw_node(
            next_radius, fine, parameters);
        const long double next_support = detail::support_squared(
            next_radius, fine.y, parameters);
        if (next_support > 64.0L
                * std::numeric_limits<long double>::epsilon()) {
            matter_seen = true;
        }

        // The first outward support closure after occupied matter defines the outer
        // shell edge. Bisect its zero using the continuous radial solution.
        if (matter_seen && !(next_support > 0.0L)) {
            const detail::RawNode& left = raw.back();
            double low = left.radius;
            double high = next.radius;
            for (int iteration = 0; iteration < 80; ++iteration) {
                const double middle = 0.5 * (low + high);
                const detail::State middle_state =
                    detail::hermite_state(left, next, middle);
                if (detail::support_squared(
                        middle, middle_state.y, parameters) > 0.0L) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            const double edge_radius = 0.5 * (low + high);
            const detail::State edge_state =
                detail::hermite_state(left, next, edge_radius);
            raw.push_back(detail::make_raw_node(
                edge_radius, edge_state, parameters));
            radius = edge_radius;
            state = edge_state;
            outer_edge_found = true;
            break;
        }

        raw.push_back(next);
        radius = next_radius;
        state = fine;
        ++accepted;
        const double growth = error > 0.0
            ? 0.9 * std::pow(error, -0.2) : 2.0;
        step *= std::clamp(growth, 0.2, 2.0);
    }
    if (!outer_edge_found || raw.size() < 3) {
        throw std::runtime_error(
            "Static shell did not acquire a finite outer matter edge; "
            "increase maximum_radius or choose a bound shooting branch.");
    }

    const detail::RawNode& outer = raw.back();
    const double outer_gap =
        1.0 - 2.0 * outer.state.mass / outer.radius;
    if (!(outer_gap > parameters.minimum_compactness_gap)) {
        throw std::runtime_error(
            "Static shell has no subcritical Schwarzschild exterior match.");
    }
    // Vacuum matching fixes mu(R)=log(1-2M/R)/2 and hence log(E0)=y(R)+mu(R).
    // The bound branch requires 0<E0<1 with this asymptotic time normalization.
    const double y_infinity = outer.state.y + 0.5 * std::log(outer_gap);
    const double cutoff_energy = std::exp(y_infinity);
    if (!(cutoff_energy > 0.0) || !(cutoff_energy < 1.0)
        || !std::isfinite(cutoff_energy)) {
        throw std::runtime_error(
            "Static shell is not on a finite bound E0<1 branch.");
    }

    Solution solution;
    solution.parameters = parameters;
    solution.cutoff_energy = cutoff_energy;
    solution.inner_matter_radius = inner_radius;
    solution.outer_matter_radius = outer.radius;
    solution.adm_mass = outer.state.mass;
    solution.rest_weight = outer.state.rest_weight;
    solution.binding_energy = solution.rest_weight - solution.adm_mass;
    solution.binding_fraction = solution.rest_weight > 0.0
        ? solution.binding_energy / solution.rest_weight : -1.0;
    solution.central_lapse = std::exp(y_infinity - parameters.central_y);
    solution.profile.reserve(raw.size());

    for (const detail::RawNode& node : raw) {
        const double gap = 1.0 - 2.0 * node.state.mass / node.radius;
        if (!(gap > parameters.minimum_compactness_gap)) {
            throw std::runtime_error(
                "Static-shell profile contains a trapped radial node.");
        }
        ProfileNode profile;
        profile.radius = node.radius;
        profile.mass = node.state.mass;
        profile.y = node.state.y;
        profile.mu = y_infinity - node.state.y;
        profile.lambda = -0.5 * std::log(gap);
        profile.rest_weight = node.state.rest_weight;
        profile.sources = detail::sources(
            node.radius, node.state.y, parameters);
        profile.dm_dr = node.derivative.mass;
        profile.dy_dr = node.derivative.y;
        profile.drest_weight_dr = node.derivative.rest_weight;
        solution.profile.push_back(profile);
        const double compactness =
            2.0 * node.state.mass / node.radius;
        solution.diagnostics.maximum_compactness = std::max(
            solution.diagnostics.maximum_compactness, compactness);
    }
    solution.diagnostics.minimum_compactness_gap =
        1.0 - solution.diagnostics.maximum_compactness;

    // Compare actual radial state increments with Simpson-integrated constraint
    // sources. Global mass/lapse scales keep the residual meaningful at support edges.
    double constraint_residual = 0.0;
    const double mass_constraint_scale = std::max(
        solution.adm_mass, parameters.absolute_tolerance);
    const double lapse_constraint_scale = std::max(
        std::abs(raw.back().state.y - raw.front().state.y),
        parameters.absolute_tolerance);
    for (std::size_t i = 0; i + 1 < raw.size(); ++i) {
        const detail::RawNode& left = raw[i];
        const detail::RawNode& right = raw[i + 1];
        const double width = right.radius - left.radius;
        const double middle_radius = 0.5 * (left.radius + right.radius);
        const detail::State middle_state = detail::hermite_state(
            left, right, middle_radius);
        const detail::Derivative middle = detail::rhs(
            middle_radius, middle_state, parameters);
        auto residual = [&] (
            double change,
            double d0,
            double dm,
            double d1,
            double global_scale) {
            const double expected = width * (d0 + 4.0 * dm + d1) / 6.0;
            return std::abs(change - expected) / global_scale;
        };
        constraint_residual = std::max({
            constraint_residual,
            residual(
                right.state.mass - left.state.mass,
                left.derivative.mass, middle.mass,
                right.derivative.mass,
                mass_constraint_scale),
            residual(
                right.state.y - left.state.y,
                left.derivative.y, middle.y,
                right.derivative.y,
                lapse_constraint_scale),
        });
    }
    solution.diagnostics.constraint_residual_linf = constraint_residual;
    solution.diagnostics.jeans_residual_relative_l2 =
        detail::jeans_residual(raw, parameters);
    return solution;
}

// Evaluates the fully normalized equilibrium at any areal radius.  The
// exact regular inner vacuum and Schwarzschild outer vacuum are included so
// callers do not have to extrapolate the tabulated matter interval.
inline ProfileNode evaluate_profile(
    const Solution& solution,
    double radius) {
    if (!(radius >= 0.0) || !std::isfinite(radius)
        || solution.profile.size() < 2
        || !(solution.cutoff_energy > 0.0)
        || !(solution.adm_mass > 0.0)) {
        throw std::runtime_error(
            "Invalid static-shell profile evaluation request.");
    }

    ProfileNode result;
    result.radius = radius;
    const double log_cutoff = std::log(solution.cutoff_energy);
    if (radius <= solution.inner_matter_radius) {
        result.y = solution.parameters.central_y;
        result.mu = log_cutoff - result.y;
        return result;
    }
    if (radius >= solution.outer_matter_radius) {
        const double gap = 1.0 - 2.0 * solution.adm_mass / radius;
        if (!(gap > 0.0)) {
            throw std::runtime_error(
                "Static-shell exterior evaluation entered a trapped radius.");
        }
        result.mass = solution.adm_mass;
        result.rest_weight = solution.rest_weight;
        result.mu = 0.5 * std::log(gap);
        result.lambda = -result.mu;
        result.y = log_cutoff - result.mu;
        result.dy_dr = -solution.adm_mass
            / (radius * radius * gap);
        return result;
    }

    const detail::State state =
        detail::interpolate_profile_state(solution, radius);
    const detail::Derivative derivative =
        detail::rhs(radius, state, solution.parameters);
    const double gap = 1.0 - 2.0 * state.mass / radius;
    if (!(gap > solution.parameters.minimum_compactness_gap)) {
        throw std::runtime_error(
            "Static-shell matter interpolation reached a trapped point.");
    }
    result.mass = state.mass;
    result.y = state.y;
    result.mu = log_cutoff - state.y;
    result.lambda = -0.5 * std::log(gap);
    result.rest_weight = state.rest_weight;
    result.sources = detail::sources(radius, state.y, solution.parameters);
    result.dm_dr = derivative.mass;
    result.dy_dr = derivative.y;
    result.drest_weight_dr = derivative.rest_weight;
    return result;
}

// A specified cutoff energy selects an equilibrium through central y.
// A sign-changing bracket confines bisection to the requested solution branch.
inline Solution shoot_to_cutoff_energy(
    Parameters parameters,
    double target_cutoff_energy,
    double lower_central_y,
    double upper_central_y,
    double relative_tolerance = 2.0e-10,
    int maximum_iterations = 80) {
    if (!(target_cutoff_energy > 0.0)
        || !(target_cutoff_energy < 1.0)
        || !(lower_central_y > 0.0)
        || !(upper_central_y > lower_central_y)
        || !(relative_tolerance > 0.0)
        || maximum_iterations < 1) {
        throw std::runtime_error("Invalid static-shell shooting request.");
    }
    parameters.central_y = lower_central_y;
    Solution lower = solve(parameters);
    parameters.central_y = upper_central_y;
    Solution upper = solve(parameters);
    double f_lower = lower.cutoff_energy - target_cutoff_energy;
    double f_upper = upper.cutoff_energy - target_cutoff_energy;
    if (f_lower == 0.0) {
        return lower;
    }
    if (f_upper == 0.0) {
        return upper;
    }
    if (std::signbit(f_lower) == std::signbit(f_upper)) {
        throw std::runtime_error(
            "Static-shell cutoff shooting bracket does not change sign.");
    }
    Solution best = std::abs(f_lower) < std::abs(f_upper) ? lower : upper;
    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        const double middle_y =
            0.5 * (lower_central_y + upper_central_y);
        parameters.central_y = middle_y;
        Solution middle = solve(parameters);
        const double f_middle =
            middle.cutoff_energy - target_cutoff_energy;
        if (std::abs(f_middle)
                <= relative_tolerance * target_cutoff_energy) {
            return middle;
        }
        if (std::abs(f_middle)
            < std::abs(best.cutoff_energy - target_cutoff_energy)) {
            best = middle;
        }
        if (std::signbit(f_middle) == std::signbit(f_lower)) {
            lower_central_y = middle_y;
            lower = std::move(middle);
            f_lower = f_middle;
        } else {
            upper_central_y = middle_y;
            upper = std::move(middle);
        }
    }
    if (std::abs(best.cutoff_energy - target_cutoff_energy)
        > 10.0 * relative_tolerance * target_cutoff_energy) {
        throw std::runtime_error(
            "Static-shell cutoff shooting failed to converge.");
    }
    return best;
}

// Geometrized homology scales radius, mass and angular momentum together.
// Density and pressure scale as length^-2, leaving compactness and the cutoff
// energy unchanged while setting the desired ADM mass.
inline Solution rescale_to_adm_mass(
    const Solution& source,
    double target_adm_mass) {
    if (!(source.adm_mass > 0.0) || !(target_adm_mass > 0.0)
        || source.profile.empty() || !std::isfinite(target_adm_mass)) {
        throw std::runtime_error("Invalid static-shell ADM rescaling.");
    }
    const double scale = target_adm_mass / source.adm_mass;
    Solution result = source;
    result.parameters.angular_momentum *= scale;
    result.parameters.amplitude /= scale;
    result.parameters.maximum_radius *= scale;
    result.parameters.initial_radial_step *= scale;
    result.parameters.maximum_radial_step *= scale;
    result.parameters.minimum_radial_step *= scale;
    result.inner_matter_radius *= scale;
    result.outer_matter_radius *= scale;
    result.adm_mass *= scale;
    result.rest_weight *= scale;
    result.binding_energy *= scale;
    for (ProfileNode& node : result.profile) {
        node.radius *= scale;
        node.mass *= scale;
        node.rest_weight *= scale;
        node.sources.rho /= scale * scale;
        node.sources.radial_pressure /= scale * scale;
        node.sources.tangential_pressure_sum /= scale * scale;
        node.sources.rest_number_density /= scale * scale;
        node.dy_dr /= scale;
        // dm/dr and dM0/dr are invariant when both numerator and radial
        // coordinate are scaled by the same factor.
    }
    return result;
}

inline MarkerSet sample_markers(
    const Solution& solution,
    const MarkerSamplingOptions& options = {}) {
    if (solution.profile.size() < 2 || !(solution.cutoff_energy > 0.0)
        || !(solution.cutoff_energy < 1.0)) {
        throw std::runtime_error("Invalid static-shell marker sampling request.");
    }
    if (!(options.rule == MarkerSamplingRule::tensor_gauss_legendre
          || options.rule == MarkerSamplingRule::shifted_stratified)) {
        throw std::runtime_error("Unknown static-shell marker sampling rule.");
    }
    const bool shifted =
        options.rule == MarkerSamplingRule::shifted_stratified;
    const bool dedicated_shifted_count =
        shifted && options.shifted_pair_count != 0;
    if ((!dedicated_shifted_count && options.radial_bins == 0)
        || (dedicated_shifted_count && options.shifted_pair_count < 2)) {
        throw std::runtime_error("Invalid static-shell marker sampling count.");
    }
    std::pair<std::vector<double>, std::vector<double>> radial_rule;
    std::pair<std::vector<double>, std::vector<double>> momentum_rule;
    if (!dedicated_shifted_count) {
        radial_rule = detail::gauss_legendre_rule(
            options.radial_quadrature_order);
        momentum_rule = detail::gauss_legendre_rule(
            options.positive_momentum_quadrature_order);
    }
    MarkerSet result;
    const std::size_t radial_order = radial_rule.first.size();
    const std::size_t momentum_order = momentum_rule.first.size();
    const std::size_t size_max = std::numeric_limits<std::size_t>::max();
    if (!dedicated_shifted_count
        && (options.radial_bins > size_max / radial_order
            || options.radial_bins * radial_order
                > size_max / momentum_order)) {
        throw std::runtime_error("Static-shell marker count overflows size_t.");
    }
    const std::size_t pair_count = dedicated_shifted_count
        ? options.shifted_pair_count
        : options.radial_bins * radial_order * momentum_order;
    if (pair_count > (
            size_max / 2)) {
        throw std::runtime_error("Static-shell marker count overflows size_t.");
    }
    const std::size_t marker_count_upper_bound = 2 * pair_count;
    if (options.first_marker_id == 0
        || marker_count_upper_bound
            > std::numeric_limits<std::uint64_t>::max()
                - options.first_marker_id) {
        throw std::runtime_error("Static-shell marker IDs overflow uint64_t.");
    }
    result.markers.reserve(marker_count_upper_bound);
    const double full_radial_width =
        solution.outer_matter_radius - solution.inner_matter_radius;
    std::uint64_t identifier = options.first_marker_id;
    long double represented = 0.0L;
    double maximum_energy_ratio = 0.0;

    // Sample the positive-w half of the support and emit its opposite partner.
    // Each positive invariant weight includes exp(lambda), the delta-angular Jacobian
    // and the radial/momentum quadrature measure; the pair current cancels exactly.
    auto append_pair = [&] (
        double radius,
        double radial_measure,
        double positive_momentum_coordinate,
        double momentum_unit_measure) {
        const detail::State state =
            detail::interpolate_profile_state(solution, radius);
        const long double w_squared = detail::support_squared(
            radius, state.y, solution.parameters);
        if (!(w_squared > 0.0L)) {
            throw std::runtime_error(
                "Static-shell interior quadrature node left the E<E0 support.");
        }
        const double w_max = std::sqrt(static_cast<double>(w_squared));
        const double gap = 1.0 - 2.0 * state.mass / radius;
        if (!(gap > solution.parameters.minimum_compactness_gap)) {
            throw std::runtime_error(
                "Static-shell marker sampling reached a trapped point.");
        }
        const double exp_lambda = 1.0 / std::sqrt(gap);
        const double z = std::exp(state.y);
        const double w = w_max * positive_momentum_coordinate;
        const double momentum_measure =
            w_max * momentum_unit_measure;
        const double epsilon = std::sqrt(
            1.0 + w * w
            + solution.parameters.angular_momentum
                * solution.parameters.angular_momentum
                / (radius * radius));
        const double phi_base =
            (static_cast<double>(w_squared) - w * w)
            / (z * (z + epsilon));
        const double phi = std::pow(
            phi_base, solution.parameters.cutoff_power);
        const double invariant_weight = 8.0 * PI * PI
            * solution.parameters.amplitude
            * solution.parameters.angular_momentum
            * exp_lambda * phi
            * radial_measure * momentum_measure;
        if (!(invariant_weight > 0.0)
            || !std::isfinite(invariant_weight)) {
            throw std::runtime_error(
                "Static-shell quadrature generated an invalid marker weight.");
        }
        const double energy_ratio = epsilon / z;
        if (!(energy_ratio < 1.0) || !(energy_ratio > 0.0)) {
            throw std::runtime_error(
                "Static-shell marker violates the strict E<E0 support.");
        }
        for (int sign : {1, -1}) {
            WeightedPICMarker marker;
            marker.mass = invariant_weight;
            marker.r = radius;
            marker.w = sign * w;
            marker.eta = 0.5
                * solution.parameters.angular_momentum
                * solution.parameters.angular_momentum;
            marker.id = identifier++;
            // source_cell is the persistent source-marker identity used by
            // the SC->EF crossing manifest, not a deposition-bin label.
            // Keeping it one-to-one with id lets the same empirical measure
            // be authenticated across both coordinate representations.
            marker.source_cell = marker.id;
            marker.state = WeightedPICMarkerState::active;
            result.markers.push_back(marker);
            represented += invariant_weight;
        }
        maximum_energy_ratio = std::max(
            maximum_energy_ratio, energy_ratio);
    };

    // The shifted rank-one lattice changes the deterministic phase locations.
    // Its positive weights include the transformed dr dw measure, and the same
    // paired signs preserve zero radial current for each sampled radius.
    if (shifted) {
        // A rank-one lattice supplies one point in every stratum of each
        // transformed coordinate.  Cranley--Patterson shifts select the
        // realization.  Under r=Rin+u_r(Rout-Rin), w=u_w W(r), each point
        // represents du_r du_w=1/N, hence dr dw=(Rout-Rin)W(r)/N.
        const double radial_shift = detail::open_unit_from_hash(
            options.realization ^ UINT64_C(0x243f6a8885a308d3));
        const double momentum_shift = detail::open_unit_from_hash(
            options.realization ^ UINT64_C(0x13198a2e03707344));
        const std::size_t generator =
            detail::rank_one_lattice_generator(pair_count);
        std::size_t momentum_index = 0;
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const double radial_coordinate =
                detail::shifted_lattice_coordinate(
                    pair, pair_count, radial_shift);
            const double momentum_coordinate =
                detail::shifted_lattice_coordinate(
                    momentum_index, pair_count, momentum_shift);
            const double radius = solution.inner_matter_radius
                + full_radial_width * radial_coordinate;
            append_pair(
                radius,
                full_radial_width / static_cast<double>(pair_count),
                momentum_coordinate,
                1.0);
            momentum_index = detail::add_modulo_without_overflow(
                momentum_index, generator, pair_count);
        }
    } else {
        const double radial_width =
            full_radial_width / static_cast<double>(options.radial_bins);
        for (std::size_t bin = 0; bin < options.radial_bins; ++bin) {
            const double left = solution.inner_matter_radius
                + static_cast<double>(bin) * radial_width;
            const double right = left + radial_width;
            for (std::size_t qr = 0; qr < radial_order; ++qr) {
                const double radius = 0.5 * (left + right)
                    + 0.5 * radial_width * radial_rule.first[qr];
                const double radial_measure =
                    0.5 * radial_width * radial_rule.second[qr];
                for (std::size_t qw = 0; qw < momentum_order; ++qw) {
                    append_pair(
                        radius,
                        radial_measure,
                        0.5 * (1.0 + momentum_rule.first[qw]),
                        0.5 * momentum_rule.second[qw]);
                }
            }
        }
    }
    result.represented_rest_weight = static_cast<double>(represented);
    result.rest_weight_relative_error = std::abs(
        result.represented_rest_weight - solution.rest_weight)
        / std::max(solution.rest_weight, 1.0e-300);
    result.maximum_energy_over_cutoff = maximum_energy_ratio;
    return result;
}

}  // namespace ev::static_einstein_vlasov
