#!/usr/bin/env bash
# Source on IAS: source scripts/build/ias_environment.sh cpu|gpu
# Absolute compiler paths preserve the site MPI stack while selecting Python.
venus_ias_environment() {
  local mode=${1:-gpu} python_path
  case "$mode" in cpu|gpu) ;; *) echo 'Choose cpu or gpu.' >&2; return 2;; esac
  type module >/dev/null 2>&1 || { echo 'Start an IAS login shell so the module command is available.' >&2; return 1; }
  module load gcc-toolset/10 openmpi/gcc/4.1.0 || return 1
  export CXX=$(command -v g++)
  export MPICXX=$(command -v mpicxx)
  if [[ $mode == gpu ]]; then
    module load cudatoolkit/12.4 || return 1
    export CUDACXX=$(command -v nvcc)
    export CUDAHOSTCXX="$CXX"
    export GPU_ARCH=${GPU_ARCH:-80}
  fi
  if [[ -n ${PYTHON:-} ]]; then
    python_path=$PYTHON
  elif [[ -x "$HOME/anaconda3/bin/python" ]]; then
    python_path="$HOME/anaconda3/bin/python"
  else
    python_path=python3
  fi
  export PYTHON=$(command -v "$python_path")
  [[ -x "$CXX" && -x "$MPICXX" && -x "$PYTHON" ]] || { echo 'The selected compiler, MPI or Python is unavailable.' >&2; return 1; }
  export PATH="$(dirname "$MPICXX"):$(dirname "$CXX"):$(dirname "$PYTHON"):$PATH"
  export VENUS_MPI_LAUNCHER=${VENUS_MPI_LAUNCHER:-mpiexec}
}
venus_ias_environment "$@"
