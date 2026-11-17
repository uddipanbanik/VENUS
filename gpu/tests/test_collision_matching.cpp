#include "venus_gpu/collision_matching.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// Exact pair-space tests use the same host/device mapping as the CUDA kernel.
// They establish support, uniqueness, race freedom, and physical rate weights
// without requiring a CUDA device or relying on a Maxwellian's random ordering.
int main() {
    using namespace venus_gpu::collision_matching;
    assert(round_count(0) == 0 && round_count(1) == 0);
    for (std::uint64_t count = 2; count <= 129; ++count) {
        const std::uint64_t rounds = round_count(count);
        std::vector<unsigned int> visits(count * count, 0);
        std::vector<unsigned int> absent(count, 0);
        long double matching_rate = 0.0L;
        for (std::uint64_t round = 0; round < rounds; ++round) {
            std::vector<unsigned int> used(count, 0);
            for (std::uint64_t slot = 0; slot < count / 2; ++slot) {
                const Pair pair = pair_at(count, round, slot);
                assert(pair.first < count && pair.second < count);
                assert(pair.first != pair.second);
                assert(++used[pair.first] == 1 && ++used[pair.second] == 1);
                const auto lo = std::min(pair.first, pair.second);
                const auto hi = std::max(pair.first, pair.second);
                ++visits[lo * count + hi];
                // Adversarial identity-ordered alternating radial velocities.
                // With m=rounds, inclusion 1/m and acceptance m*K cancel.
                const int first_velocity = pair.first % 2 ? -1 : 1;
                const int second_velocity = pair.second % 2 ? -1 : 1;
                matching_rate += std::abs(first_velocity - second_velocity);
            }
            for (std::uint64_t marker = 0; marker < count; ++marker) {
                absent[marker] += used[marker] == 0;
            }
        }
        long double direct_rate = 0.0L;
        for (std::uint64_t first = 0; first < count; ++first) {
            assert(absent[first] == (count % 2 ? 1U : 0U));
            for (std::uint64_t second = first + 1; second < count; ++second) {
                assert(visits[first * count + second] == 1);
                const int v1 = first % 2 ? -1 : 1;
                const int v2 = second % 2 ? -1 : 1;
                direct_rate += std::abs(v1 - v2);
            }
        }
        assert(matching_rate == direct_rate);
    }
    // Exercise non-power-of-two bounds and rejection-based device RNG mapping.
    for (const std::uint64_t count : {3, 4, 6, 9, 32, 33}) {
        const std::uint64_t rounds = round_count(count);
        std::vector<std::uint64_t> counts(rounds, 0);
        constexpr std::uint64_t samples = 200000;
        for (std::uint64_t key = 0; key < samples; ++key) {
            const auto round = random_round(count, key);
            assert(round < rounds);
            ++counts[round];
        }
        const double expected = static_cast<double>(samples) / rounds;
        const double sigma = std::sqrt(expected * (1.0 - 1.0 / rounds));
        for (const auto measured : counts) {
            assert(std::abs(static_cast<double>(measured) - expected) < 7 * sigma);
        }
    }
    std::cout << "collision matching: complete exact pair support for N=2..129, "
                 "disjoint updates, odd/even physical rate weights, "
                 "and bounded random-round frequencies passed.\n";
}
