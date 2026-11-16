// Measure single-GPU throughput for spherical Vlasov-Poisson evolution.
// The included harness takes particle/cell/step counts, creates a reproducible
// population, and reports time per KDK step and particle-step throughput.
// This compile-time selection runs the common orbit workload on its own.

#define VENUS_BENCHMARK_VP_ONLY 1
#include "benchmark_wpic.cu"
