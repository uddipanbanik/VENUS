#pragma once

// Measure discrete Newtonian invariants of the canonical cell-average
// kinetic distribution. Inputs are nonnegative f and its (r,w,eta) cell
// geometry, with eta=L^2/2. The phase-space measure is
// 8pi^2 dr dw deta. Outputs are represented mass, radial momentum,
// kinetic energy, Boltzmann H and the distribution extrema. Gravitational
// potential energy is measured by the separate field calculation.

#include "ev/phase_space_cells.hpp"

namespace ev {

/// Canonical discrete invariants and Boltzmann H for one distribution.
struct KineticInvariantLedger {
    double mass = 0.0;
    double radial_momentum = 0.0;
    double kinetic_energy = 0.0;
    double boltzmann_H = 0.0;
    double minimum_f = std::numeric_limits<double>::infinity();
    double maximum_f = 0.0;
};

/// Difference between two invariant ledgers.
struct KineticInvariantDelta {
    double mass = 0.0;
    double radial_momentum = 0.0;
    double kinetic_energy = 0.0;
    double boltzmann_H = 0.0;
};

/// Evaluates invariants with the exact canonical measure 8 pi^2 dr dw deta.
inline KineticInvariantLedger evaluate_kinetic_invariant_ledger(
    const std::vector<double>& f,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    double entropy_floor = 1.0e-300) {
    const std::size_t expected =
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    validate_cell_average_eta_geometry(
        geometry, p, "kinetic invariant ledger");
    validate_finite_nonnegative_distribution(
        f, expected, "kinetic invariant ledger");
    KineticInvariantLedger ledger;
    const double phase_factor = 8.0 * PI * PI;
    for (int i = 0; i < p.Nr; ++i) {
        const double ri = geometry.radial.center[static_cast<std::size_t>(i)];
        for (int j = 0; j < p.Nw; ++j) {
            const double wr = geometry.radial_velocity.center[static_cast<std::size_t>(j)];
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t q = idx(i, j, k, p);
                const double fq = f[q];
                const double eta = geometry.eta.center[static_cast<std::size_t>(k)];
                // The specific kinetic energy is w^2/2+L^2/(2r^2)=w^2/2+eta/r^2.
                // Multiplying by the cell measure and f forms its phase-space integral.
                const double energy = 0.5 * wr * wr + eta / (ri * ri);
                const double measure =
                    phase_factor * canonical_cell_measure(geometry, i, j, k);
                ledger.mass += measure * fq;
                ledger.radial_momentum += measure * wr * fq;
                ledger.kinetic_energy += measure * energy * fq;
                // The analytic limit of f(log f-1) is zero at f=0. Only positive
                // values enter the logarithm; the entropy floor resolves extremely
                // small positive floating-point values.
                if (fq > 0.0) {
                    ledger.boltzmann_H +=
                        measure * fq * (std::log(std::max(fq, entropy_floor)) - 1.0);
                }
                ledger.minimum_f = std::min(ledger.minimum_f, fq);
                ledger.maximum_f = std::max(ledger.maximum_f, fq);
            }
        }
    }
    if (!std::isfinite(ledger.minimum_f)) {
        ledger.minimum_f = 0.0;
    }
    return ledger;
}

/// Forms an after-minus-before ledger delta.
// After-minus-before differences identify the budgets changed by a
// particular operator. Whether a given budget should vanish follows
// from that operator and its physical boundary conditions.
inline KineticInvariantDelta kinetic_invariant_delta(
    const KineticInvariantLedger& before,
    const KineticInvariantLedger& after) {
    KineticInvariantDelta delta;
    delta.mass = after.mass - before.mass;
    delta.radial_momentum = after.radial_momentum - before.radial_momentum;
    delta.kinetic_energy = after.kinetic_energy - before.kinetic_energy;
    delta.boltzmann_H = after.boltzmann_H - before.boltzmann_H;
    return delta;
}

/// Scale-aware relative error used by the component acceptance tests.
inline double kinetic_relative_error(double before, double after) {
    return std::abs(after - before) / std::max({1.0, std::abs(before), std::abs(after)});
}

}  // namespace ev
