#!/usr/bin/env bash
# Build and run finite CUDA checks inside an existing reviewed cluster allocation.
# Input is the staged release root. GPU_ARCH selects the device architecture and GPU_MPI enables the distributed layer.
# Build products stay below build/gpu-smoke. The temporary output-test directory is removed on exit.

# Finite CUDA check for a small allocation on symmetry/H200.
# Set GPU_MPI=ON and request two GPUs to include the distributed test.
# All tests use small fixtures. Build products stay in the staged source.
set -euo pipefail
root=${1:?usage: cluster_gpu_smoke.sh STAGED_RELEASE_ROOT}
cd "$root"
# Limit host numerical-library threading so the small fixtures use a predictable CPU budget.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
# Configure Release compilation for the requested device architecture before building any target.
cmake -S gpu -B build/gpu-smoke -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH:-90}" \
  -DVENUS_GPU_ENABLE_MPI="${GPU_MPI:-OFF}" -DVENUS_GPU_BUILD_BENCHMARKS=OFF
cmake --build build/gpu-smoke -j "${SLURM_CPUS_PER_TASK:-4}"
# The output test unlinks an open file. A node-local working directory
# supports that operation directly on systems with NFS source storage.
# Use node-local scratch for the output fixture, whose open-file replacement semantics require it.
work=$(mktemp -d "${TMPDIR:-/tmp}/venus-output.XXXXXX")
trap 'rm -rf "$work"' EXIT
(cd "$work" && "$root/build/gpu-smoke/test_production_output")
# Run the other registered checks with an explicit per-test timeout and visible failure output.
ctest --test-dir build/gpu-smoke --output-on-failure --timeout 120 \
  --exclude-regex '^production_output$'
printf '%s\n' 'VENUS_GPU_SMOKE_PASS'
