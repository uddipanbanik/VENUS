// The angular sampler in the production initializer draws from
// p(L) proportional to L exp[-(L-center)^2/(2 width^2)], 0 <= L <= maximum.
#include "../src/drivers/expanding_wpic.cpp"

int main() {
    struct Quantile {
        double center, width, maximum, probability, expected;
    };
    // Independent positive-integrand quadrature, followed by root finding,
    // provides these references. The first three resolve a narrow tail
    // whose unshifted erf primitive is numerically constant in double precision.
    const Quantile references[] = {
        {30, 2, 8, 0.1, 7.597790344256035},
        {30, 2, 8, 0.5, 7.878132138292165},
        {30, 2, 8, 0.9, 7.981431122371615},
        {1, 0.08, 8, 0.5, 1.006386416111819},
        {0, 1, 3, 0.5, 1.1679892428540184},
        {1, 1, 1e-4, 0.5, 7.071136842240926e-5},
        {25, 2, 8, 0.5, 7.8444052724985065},
        {0.4, 0.3, 8, 0.99, 1.2195454100447107},
    };
    for (const auto& reference : references) {
        const double measured = sample_angular_momentum(
            reference.probability, reference.center, reference.width,
            reference.maximum);
        std::cout << std::setprecision(17) << "angular quantile center="
                  << reference.center << " width=" << reference.width
                  << " maximum=" << reference.maximum
                  << " probability=" << reference.probability
                  << " measured=" << measured
                  << " expected=" << reference.expected << '\n';
        if (!(std::abs(measured - reference.expected)
              <= 2e-11 * reference.maximum)) {
            std::cerr << "Truncated angular quantile differs from quadrature.\n";
            return 1;
        }
        double previous = 0.0;
        for (int i = 1; i < 200; ++i) {
            const double value = sample_angular_momentum(
                i / 200.0, reference.center, reference.width,
                reference.maximum);
            if (!(std::isfinite(value) && value > previous
                  && value <= reference.maximum)) {
                std::cerr << "Angular quantiles must increase within their support.\n";
                return 1;
            }
            previous = value;
        }
    }
    // At zero center the distribution is a truncated Rayleigh law, whose
    // inverse CDF is analytic and probes probabilities close to both edges.
    for (double probability : {1e-12, 1e-8, 0.001, 0.5, 0.999, 1.0 - 1e-12}) {
        const double expected = std::sqrt(-2.0 * std::log1p(
            probability * std::expm1(-4.5)));
        const double measured = sample_angular_momentum(probability, 0, 1, 3);
        if (std::abs(measured - expected) > 3e-12) {
            std::cerr << "Rayleigh quantile regression failed.\n";
            return 1;
        }
    }
    std::cout << "Expanding WPIC initial-distribution tests passed.\n";
}
