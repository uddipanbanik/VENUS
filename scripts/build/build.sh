#!/usr/bin/env bash
# Build a selected backend. Compiler variables and GPU_ARCH may be set by the caller.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
mode=${1:-cpu}
jobs=${BUILD_JOBS:-2}
case "$mode" in cpu|cpu-openmp|cpu-mpi|tools)
  source "$root/scripts/build/macos_toolchain.sh"
  venus_select_macos_sdk
  ;;
esac
case "$mode" in gpu|gpu-mpi)
  command -v cmake >/dev/null || { echo 'Load CMake 3.18 or newer before building CUDA targets.' >&2; exit 1; }
  if ! command -v "${CUDACXX:-nvcc}" >/dev/null; then
    echo 'CUDA compiler unavailable. Load the site CUDA module and build on a host with its toolkit installed; GPU allocations may provide only the runtime.' >&2
    exit 1
  fi
  ;;
esac
case "$mode" in
  cpu) make -j "$jobs" cpu ;;
  cpu-openmp) make -j "$jobs" cpu-openmp ;;
  cpu-mpi) make -j "$jobs" cpu-mpi ;;
  gpu) make gpu GPU_ARCH="${GPU_ARCH:-80}" BUILD_JOBS="$jobs" ;;
  gpu-mpi) make gpu-mpi GPU_ARCH="${GPU_ARCH:-80}" BUILD_JOBS="$jobs" ;;
  tools) make -j "$jobs" cpu-tools ;;
  *) echo "usage: scripts/build/build.sh cpu|cpu-openmp|cpu-mpi|gpu|gpu-mpi|tools" >&2; exit 2 ;;
esac
