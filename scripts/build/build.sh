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
case "$mode" in
  cpu) make -j "$jobs" cpu ;;
  cpu-openmp) make -j "$jobs" cpu-openmp ;;
  cpu-mpi) make -j "$jobs" cpu-mpi ;;
  gpu) make gpu GPU_ARCH="${GPU_ARCH:-80}" BUILD_JOBS="$jobs" ;;
  gpu-mpi) make gpu-mpi GPU_ARCH="${GPU_ARCH:-80}" BUILD_JOBS="$jobs" ;;
  tools) make -j "$jobs" cpu-tools ;;
  *) echo "usage: scripts/build/build.sh cpu|cpu-openmp|cpu-mpi|gpu|gpu-mpi|tools" >&2; exit 2 ;;
esac
