#pragma once

// Integrate the characteristic ODE for radius r and radial momentum w.
// The caller supplies the physical right-hand side and a projection that
// applies its boundary or regular-coordinate rule. The signed interval also
// supports backward tracing to a semi-Lagrangian interpolation footpoint.
// KDK/RK, modified-midpoint extrapolation and Adams--Bashforth choices share
// these state operations; distribution interpolation and field solves remain
// in the calling transport module.

#include "ev/radial_orbit_regularization.hpp"

namespace ev {

inline CharacteristicState operator+(
    const CharacteristicState& y,
    const CharacteristicDerivative& k) {
    return {y.r + k.dr, y.w + k.dw};
}

inline CharacteristicState scaled_add(
    const CharacteristicState& y,
    double scale,
    const CharacteristicDerivative& k) {
    return {y.r + scale * k.dr, y.w + scale * k.dw};
}

/// Stores an explicit Runge-Kutta tableau in row-major a_{ij} form.
struct RKTableau {
    int order = 1;
    int stages = 1;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
};

/// Precomputed data for the selected characteristic ODE integrator.
struct CharacteristicIntegratorPlan {
    std::string method = "kick_drift_kick";
    int requested_order = 2;
    int effective_order = 2;
    int substeps = 1;
    RKTableau rk;
    std::vector<double> adams_bashforth_weights;
    std::vector<int> extrapolation_substeps;
    std::vector<double> extrapolation_weights;
};

/// Returns an explicit RK tableau for orders 1--4; higher-order requests use extrapolation.
inline RKTableau make_rk_tableau(int requested_order) {
    const int order = std::clamp(requested_order, 1, 4);
    if (order == 1) {
        return {1, 1, {0.0}, {1.0}, {0.0}};
    }
    if (order == 2) {
        return {
            2,
            2,
            {
                0.0, 0.0,
                0.5, 0.0,
            },
            {0.0, 1.0},
            {0.0, 0.5},
        };
    }
    if (order == 3) {
        return {
            3,
            3,
            {
                0.0, 0.0, 0.0,
                0.5, 0.0, 0.0,
                -1.0, 2.0, 0.0,
            },
            {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0},
            {0.0, 0.5, 1.0},
        };
    }
    return {
        4,
        4,
        {
            0.0, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        },
        {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0},
        {0.0, 0.5, 0.5, 1.0},
    };
}

/// Multiplies a polynomial by a linear factor a*x+b.
inline std::vector<double> multiply_polynomial_by_linear(
    const std::vector<double>& poly,
    double a,
    double b) {
    std::vector<double> out(poly.size() + 1, 0.0);
    for (std::size_t i = 0; i < poly.size(); ++i) {
        out[i] += b * poly[i];
        out[i + 1] += a * poly[i];
    }
    return out;
}

/// Returns Adams-Bashforth weights b_0,b_1,... for any requested order.
// Interpolate past derivatives at nodes 0,-1,... using Lagrange
// polynomials, then integrate each basis polynomial over the next unit
// interval. The resulting weights multiply the derivative history.
inline std::vector<double> make_adams_bashforth_weights(int requested_order) {
    const int order = std::max(1, requested_order);
    std::vector<double> weights(static_cast<std::size_t>(order), 0.0);
    for (int j = 0; j < order; ++j) {
        const double xj = -static_cast<double>(j);
        std::vector<double> poly{1.0};
        for (int m = 0; m < order; ++m) {
            if (m == j) {
                continue;
            }
            const double xm = -static_cast<double>(m);
            const double denom = xj - xm;
            poly = multiply_polynomial_by_linear(poly, 1.0 / denom, -xm / denom);
        }
        double integral = 0.0;
        for (std::size_t d = 0; d < poly.size(); ++d) {
            integral += poly[d] / static_cast<double>(d + 1);
        }
        weights[static_cast<std::size_t>(j)] = integral;
    }
    return weights;
}

/// Lagrange weights for evaluating a polynomial through (x_j,y_j) at x=0.
// Combine estimates at finite squared substep sizes to evaluate their
// interpolation polynomial at zero size. This cancels successive even
// error powers of the modified-midpoint sequence.
inline std::vector<double> make_zero_extrapolation_weights(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<double> weights(static_cast<std::size_t>(n), 1.0);
    for (int j = 0; j < n; ++j) {
        double w = 1.0;
        for (int m = 0; m < n; ++m) {
            if (m == j) {
                continue;
            }
            w *= -x[static_cast<std::size_t>(m)]
                / (x[static_cast<std::size_t>(j)] - x[static_cast<std::size_t>(m)]);
        }
        weights[static_cast<std::size_t>(j)] = w;
    }
    return weights;
}

/// Precomputes the selected characteristic integrator coefficients.
inline CharacteristicIntegratorPlan make_characteristic_integrator_plan(const Params& p) {
    CharacteristicIntegratorPlan plan;
    plan.requested_order = p.time_integrator_order;
    plan.substeps = std::max(1, p.time_integrator_substeps);
    if (using_kick_drift_kick_integrator(p)) {
        plan.method = "kick_drift_kick";
        plan.effective_order = 2;
        return plan;
    }
    // The multistep method needs enough internal substeps to build its
    // requested derivative history. Startup uses an explicit RK method.
    if (using_adams_bashforth_characteristic_integrator(p)) {
        plan.method = "adams_bashforth";
        plan.effective_order = std::max(1, p.time_integrator_order);
        plan.adams_bashforth_weights = make_adams_bashforth_weights(plan.effective_order);
        plan.substeps = std::max(plan.substeps, plan.effective_order);
        return plan;
    }
    if (using_rk_characteristic_integrator(p) && p.time_integrator_order <= 4) {
        plan.method = "runge_kutta";
        plan.effective_order = std::max(1, p.time_integrator_order);
        plan.rk = make_rk_tableau(plan.effective_order);
        return plan;
    }
    if (using_rk_characteristic_integrator(p)) {
        plan.method = "extrapolated_midpoint";
        // Modified midpoint has an error expansion in inverse even powers of
        // its substep count. The selected levels and zero-extrapolation weights
        // construct the requested even effective order.
        const int levels = std::max(1, (p.time_integrator_order + 1) / 2);
        plan.effective_order = 2 * levels;
        plan.extrapolation_substeps.resize(static_cast<std::size_t>(levels));
        std::vector<double> x(static_cast<std::size_t>(levels), 0.0);
        const int multiplier = std::max(1, p.time_integrator_substeps);
        for (int j = 0; j < levels; ++j) {
            const int nsub = 2 * (j + 1) * multiplier;
            plan.extrapolation_substeps[static_cast<std::size_t>(j)] = nsub;
            x[static_cast<std::size_t>(j)] = 1.0 / static_cast<double>(nsub * nsub);
        }
        plan.extrapolation_weights = make_zero_extrapolation_weights(x);
        return plan;
    }
    throw std::runtime_error("Unknown characteristic integrator: " + p.integrator);
}

template <class RHS, class Project>
// Form every explicit RK stage from the same incoming state, apply the
// caller projection, then evaluate the physical derivative there. The
// weighted stage sum supplies the returned characteristic state.
CharacteristicState characteristic_rk_step(
    CharacteristicState y,
    double h,
    const RKTableau& tableau,
    RHS&& rhs,
    Project&& project) {
    project(y);
    std::vector<CharacteristicDerivative> k(static_cast<std::size_t>(tableau.stages));
    for (int s = 0; s < tableau.stages; ++s) {
        CharacteristicState stage = y;
        for (int q = 0; q < s; ++q) {
            const double a = tableau.a[static_cast<std::size_t>(s * tableau.stages + q)];
            stage = scaled_add(stage, h * a, k[static_cast<std::size_t>(q)]);
        }
        project(stage);
        k[static_cast<std::size_t>(s)] = rhs(stage);
    }
    CharacteristicState out = y;
    for (int s = 0; s < tableau.stages; ++s) {
        out = scaled_add(out, h * tableau.b[static_cast<std::size_t>(s)], k[static_cast<std::size_t>(s)]);
    }
    project(out);
    return out;
}

template <class RHS, class Project>
// Divide the signed caller interval into equal RK substeps. Each
// substep uses the same selected tableau and physical RHS interface.
CharacteristicState characteristic_rk_advance(
    CharacteristicState y,
    double h,
    const CharacteristicIntegratorPlan& plan,
    RHS&& rhs,
    Project&& project) {
    const int nsub = std::max(1, plan.substeps);
    const double hs = h / static_cast<double>(nsub);
    for (int n = 0; n < nsub; ++n) {
        y = characteristic_rk_step(y, hs, plan.rk, rhs, project);
    }
    return y;
}

template <class RHS, class Project>
// This derivative history belongs to the current characteristic advance.
// RK starts it, then the newest derivative is inserted at the front after
// each internal substep; older entries beyond the order are discarded.
CharacteristicState characteristic_adams_bashforth_advance(
    CharacteristicState y,
    double h,
    const CharacteristicIntegratorPlan& plan,
    RHS&& rhs,
    Project&& project) {
    const std::vector<double>& weights = plan.adams_bashforth_weights;
    const int ab_order = static_cast<int>(weights.size());
    const int nsub = std::max(std::max(1, plan.substeps), ab_order);
    const double hs = h / static_cast<double>(nsub);
    const RKTableau starter = make_rk_tableau(std::min(ab_order, 4));
    std::vector<CharacteristicDerivative> history;
    history.reserve(static_cast<std::size_t>(ab_order));

    project(y);
    history.push_back(rhs(y));
    for (int n = 0; n < nsub; ++n) {
        if (static_cast<int>(history.size()) < ab_order) {
            y = characteristic_rk_step(y, hs, starter, rhs, project);
        } else {
            CharacteristicState out = y;
            for (int m = 0; m < ab_order; ++m) {
                out = scaled_add(out, hs * weights[static_cast<std::size_t>(m)], history[static_cast<std::size_t>(m)]);
            }
            project(out);
            y = out;
        }
        history.insert(history.begin(), rhs(y));
        if (static_cast<int>(history.size()) > ab_order) {
            history.pop_back();
        }
    }
    return y;
}

template <class RHS, class Project>
// Two staggered states implement z_next=z_prev+2h*f(z_curr).
// The final average combines the staggered states with the endpoint
// derivative to obtain the modified-midpoint estimate.
CharacteristicState characteristic_modified_midpoint_step(
    CharacteristicState y,
    double h,
    int substeps,
    RHS&& rhs,
    Project&& project) {
    const int nsub = std::max(2, substeps);
    const double hs = h / static_cast<double>(nsub);
    project(y);
    CharacteristicState z_prev = y;
    CharacteristicState z_curr = scaled_add(y, hs, rhs(y));
    project(z_curr);
    for (int n = 1; n < nsub; ++n) {
        CharacteristicState z_next = scaled_add(z_prev, 2.0 * hs, rhs(z_curr));
        project(z_next);
        z_prev = z_curr;
        z_curr = z_next;
    }
    const CharacteristicDerivative f_end = rhs(z_curr);
    CharacteristicState out{
        0.5 * (z_curr.r + z_prev.r + hs * f_end.dr),
        0.5 * (z_curr.w + z_prev.w + hs * f_end.dw),
    };
    project(out);
    return out;
}

template <class RHS, class Project>
// Reintegrate the same incoming characteristic with each prescribed
// substep count, then combine the estimates at zero squared substep
// size. The final projection enforces the caller coordinate domain.
CharacteristicState characteristic_extrapolated_midpoint_advance(
    CharacteristicState y,
    double h,
    const CharacteristicIntegratorPlan& plan,
    RHS&& rhs,
    Project&& project) {
    const int levels = static_cast<int>(plan.extrapolation_substeps.size());
    std::vector<CharacteristicState> estimates(static_cast<std::size_t>(levels));
    for (int j = 0; j < levels; ++j) {
        estimates[static_cast<std::size_t>(j)] = characteristic_modified_midpoint_step(
            y, h, plan.extrapolation_substeps[static_cast<std::size_t>(j)], rhs, project);
    }
    CharacteristicState out{0.0, 0.0};
    for (int j = 0; j < levels; ++j) {
        const double weight = plan.extrapolation_weights[static_cast<std::size_t>(j)];
        out.r += weight * estimates[static_cast<std::size_t>(j)].r;
        out.w += weight * estimates[static_cast<std::size_t>(j)].w;
    }
    project(out);
    return out;
}

}  // namespace ev
