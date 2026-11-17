#!/usr/bin/env bash
# Source on Perimeter Symmetry: source scripts/build/symmetry_environment.sh cpu|gpu
# Keep compiler/MPI paths ahead of scientific-Python environments.
venus_symmetry_environment() {
  local mode=${1:-gpu} python_path
  case "$mode" in cpu|gpu) ;; *) echo 'Choose cpu or gpu.' >&2; return 2;; esac
  type module >/dev/null 2>&1 || { echo 'Start a Symmetry login shell so the module command is available.' >&2; return 1; }
  module load gcc/11.2.0 cmake/3.27.4 nvhpc-hpcx-cuda12/24.9 python/3.11 slurm/slurm/21.08.8 || return 1
  export CXX=$(command -v g++)
  export MPICXX=$(command -v mpicxx)
  # HPC-X defaults to nvc++; the CPU build flags and CMake host build use GCC.
  export OMPI_CXX="$CXX"
  if [[ $mode == gpu ]]; then
    export CUDACXX=$(command -v nvcc)
    export CUDAHOSTCXX="$CXX"
    export GPU_ARCH=${GPU_ARCH:-75}
  fi
  python_path=${PYTHON:-$(command -v python3)}
  export PYTHON=$(command -v "$python_path")
  [[ -x "$CXX" && -x "$MPICXX" && -x "$PYTHON" ]] || { echo 'The selected compiler, MPI or Python is unavailable.' >&2; return 1; }
  export PATH="$(dirname "$MPICXX"):$(dirname "$CXX"):$(dirname "$PYTHON"):$PATH"
  export VENUS_MPI_LAUNCHER=${VENUS_MPI_LAUNCHER:-mpiexec}
}
venus_symmetry_environment "$@"
