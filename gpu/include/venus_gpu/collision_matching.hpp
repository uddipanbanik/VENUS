#pragma once

// Construct disjoint candidate pairs from a shell containing N particles.
// The inputs are a population, a random round, and a pair slot; pair_at()
// returns particle indices within that population. An even population has
// N-1 round-robin matchings. An odd population uses a ghost partner, giving
// N rounds with one real particle idle in each round. Every physical pair
// therefore has inclusion probability 1/m, with m=N-1 or N respectively.
// Multiplying the pair acceptance by m compensates for this sampling. The
// physical partner density used for a mean free path remains (N-1)/V, where
// V is shell volume. Each round requires linear work and disjoint writes.

#include <cstdint>

#ifdef __CUDACC__
#define VENUS_MATCH_HD __host__ __device__
#else
#define VENUS_MATCH_HD
#endif

namespace venus_gpu {
namespace collision_matching {

struct Pair {
    std::uint64_t first;
    std::uint64_t second;
};

// Fewer than two particles have no physical pair. The returned round count
// also supplies the acceptance multiplier used by the collision kernels.
VENUS_MATCH_HD inline std::uint64_t round_count(std::uint64_t count) {
    return count < 2 ? 0 : count - ((count & UINT64_C(1)) == 0 ? 1 : 0);
}

// Preconditions: count>=2, round<round_count(count), pair<count/2. Pair slots
// cover disjoint real vertices. The odd-population ghost pair is omitted.
VENUS_MATCH_HD inline Pair pair_at(
    std::uint64_t count, std::uint64_t round, std::uint64_t pair) {
    const bool even = (count & UINT64_C(1)) == 0;
    if (even && pair == 0) {
        return {count - 1, round};
    }
    const std::uint64_t modulus = round_count(count);
    const std::uint64_t offset = even ? pair : pair + 1;
    // Modular addition/subtraction without overflowing an unsigned sum.
    const std::uint64_t first = round >= modulus - offset
        ? round - (modulus - offset) : round + offset;
    const std::uint64_t second = round >= offset
        ? round - offset : modulus - (offset - round);
    return {first, second};
}

// Mix a counter key into a reproducible 64-bit random word before selecting
// a round. Integer arithmetic is identical on host and device.
VENUS_MATCH_HD inline std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

// Rejection removes the modulo bias of a full 64-bit random word. Actual
// shell populations are small compared with 2^64, so a retry is extremely
// rare. The key identifies one shell/operator round; retries consume only
// this local counter and never another shell's stream.
VENUS_MATCH_HD inline std::uint64_t random_round(
    std::uint64_t count, std::uint64_t key) {
    const std::uint64_t bound = round_count(count);
    if (bound <= 1) {
        return 0;
    }
    const std::uint64_t threshold = (UINT64_C(0) - bound) % bound;
    for (std::uint64_t attempt = 0;; ++attempt) {
        const std::uint64_t word = mix64(
            key + attempt * UINT64_C(0x9e3779b97f4a7c15));
        if (word >= threshold) {
            return word % bound;
        }
    }
}

}  // namespace collision_matching
}  // namespace venus_gpu

#undef VENUS_MATCH_HD
