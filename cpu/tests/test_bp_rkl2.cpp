// RKL2 accuracy for the actual frozen spherical FP flux operator.
// A positive three-cell Neumann diffusion mode has a known semidiscrete
// exponential decay, so temporal error is isolated from spatial truncation.
#include "ev/boltzmann_poisson.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct FrozenDiffusion {
    ev::Params p;
    std::vector<double> w{-1.0, 0.0, 1.0};
    std::vector<double> L{0.5, 1.5};
    std::vector<double> Lq{1.0, 1.0};
    std::vector<ev::BPCoeff> coefficients;
    std::vector<double> initial;
    static constexpr double diffusion = 0.4;
    static constexpr double decay_rate = 2.0 * diffusion;

    FrozenDiffusion() {
        p.Nr = 1;
        p.Nw = 3;
        p.NL = 2;
        p.threads = 1;
        p.bp_fp_terms = "vr_only";
        p.bp_collision_flux = "chang_cooper";
        p.bp_collision_l_min_rel = 0.0;
        coefficients.resize(6);
        initial.resize(6);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t q = ev::idx(0, j, k, p);
                coefficients[q].D_rr = 2.0 * diffusion;
                initial[q] = 1.0 + 0.2 * w[j];
            }
        }
        // The nodal endpoint control volumes have half width. The
        // conservative zero-flux stencil therefore gives L*w=-2*D*w.
        const auto rhs = ev::bp_collision_rhs(
            initial, coefficients, w, L, Lq, p);
        for (int j = 0; j < p.Nw; ++j)
            for (int k = 0; k < p.NL; ++k)
                require(std::abs(rhs[ev::idx(0, j, k, p)]
                    + 0.2 * decay_rate * w[j]) < 2.0e-15,
                    "The frozen FP fixture is not the stated decay mode.");
    }

    std::vector<double> evolve(double time, int steps, int stages) const {
        auto f = initial;
        for (int step = 0; step < steps; ++step)
            ev::apply_bp_rkl2_cycle(f, coefficients, w, L, Lq,
                                  time / steps, stages, p);
        return f;
    }

    double error(const std::vector<double>& f, double time) const {
        double result = 0.0;
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const double exact = 1.0 + 0.2 * w[j]
                    * std::exp(-decay_rate * time);
                result = std::max(result,
                    std::abs(f[ev::idx(0, j, k, p)] - exact));
            }
        }
        return result;
    }
};

}  // namespace

int main() {
    const FrozenDiffusion test;
    const auto rhs = ev::bp_collision_rhs(test.initial, test.coefficients,
        test.w, test.L, test.Lq, test.p);
    const auto rhs2 = ev::bp_collision_rhs(rhs, test.coefficients,
        test.w, test.L, test.Lq, test.p);
    for (int stages : {2, 3, 8}) {
        // Local consistency with f+h*L(f)+h²*L²(f)/2, checked on the
        // actual production flux and stage routine, without positivity repair.
        const double h = 1.0e-3;
        const auto one = test.evolve(h, 1, stages);
        double local_error = 0.0;
        for (std::size_t q = 0; q < one.size(); ++q) {
            const double second_order = test.initial[q] + h * rhs[q]
                + 0.5 * h * h * rhs2[q];
            local_error = std::max(local_error,
                std::abs(one[q] - second_order));
        }
        require(local_error < 2.0e-11,
            "RKL2 has an incorrect first- or second-order time coefficient.");

        const double time = 1.0;
        const auto coarse = test.evolve(time, 8, stages);
        const auto medium = test.evolve(time, 16, stages);
        const auto fine = test.evolve(time, 32, stages);
        const double e8 = test.error(coarse, time);
        const double e16 = test.error(medium, time);
        const double e32 = test.error(fine, time);
        require(e8/e16 > 3.7 && e8/e16 < 4.4
            && e16/e32 > 3.7 && e16/e32 < 4.4,
            "Frozen FP RKL2 evolution does not converge at second order.");
        long double initial_mass = 0.0L, final_mass = 0.0L;
        for (int j = 0; j < test.p.Nw; ++j) {
            for (int k = 0; k < test.p.NL; ++k) {
                const std::size_t q = ev::idx(0, j, k, test.p);
                const double w_width = j == 1 ? 1.0 : 0.5;
                const double volume_weight =
                    w_width * test.L[k] * test.Lq[k];
                initial_mass += volume_weight * test.initial[q];
                final_mass += volume_weight * fine[q];
                require(fine[q] > 0.0, "Positive diffusion state became negative.");
            }
        }
        require(std::abs(final_mass - initial_mass) < 2.0e-13L,
            "RKL2 lost the conservative FP shell mass.");
        std::cout << "stages=" << stages << " local_error=" << local_error
                  << " global_errors=" << e8 << ',' << e16 << ',' << e32
                  << " ratios=" << e8/e16 << ',' << e16/e32 << '\n';
    }
    std::cout << "Frozen FP RKL2 consistency, order, positivity and mass passed\n";
}
