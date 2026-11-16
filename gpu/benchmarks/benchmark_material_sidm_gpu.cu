// Measure single-GPU throughput for elastic material dark matter collisions.
// The included harness accepts particle/cell counts, constructs the common
// deterministic population, and reports collision time and encounter counts.
// The compile-time selection below uses the shared workload with its VP stage
// omitted, so this executable measures the material operator on its own.

#define VENUS_BENCHMARK_MATERIAL_ONLY 1
#include "benchmark_wpic.cu"
