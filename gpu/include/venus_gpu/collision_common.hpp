#pragma once

// Carry the random-number state shared by the GPU Boltzmann operators.
// The sampler takes a seed and the next unused counter, then returns the
// advanced counter after a batch. A checkpoint stores both values so a
// continuation retains the sequence of encounter draws. CUDA launch order
// and the physical time represented by a batch are separate quantities.

#include <cstdint>

namespace venus_gpu {

/// Portable counter-RNG state shared by material and subhalo operators.
struct CollisionSamplerState {
    std::uint64_t seed = UINT64_C(0x243f6a8885a308d3);
    // The counter labels sampler work already consumed; restoring a
    // checkpoint restores this value together with particle and clock state.
    std::uint64_t next_counter = 0;
};

}  // namespace venus_gpu
