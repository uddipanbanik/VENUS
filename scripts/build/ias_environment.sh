#!/usr/bin/env bash
# Source this file on IAS before building or submitting the supplied examples.
# Absolute compiler paths preserve the site MPI stack while selecting Python.
module load gcc-toolset/10 cudatoolkit/12.4 openmpi/gcc/4.1.0 || return 1
export CXX=$(command -v g++)
export MPICXX=$(command -v mpicxx)
export CUDACXX=$(command -v nvcc)
if [[ -z ${PYTHON:-} ]]; then
  if command -v python3 >/dev/null && python3 -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then
    export PYTHON=$(command -v python3)
  elif [[ -x "$HOME/anaconda3/bin/python" ]]; then
    export PYTHON="$HOME/anaconda3/bin/python"
  else
    echo 'Select Python 3.8 or newer through PYTHON before continuing.' >&2
    return 1
  fi
fi
export PATH="$(dirname "$MPICXX"):$(dirname "$CXX"):$(dirname "$(command -v "$PYTHON")"):$PATH"
export GPU_ARCH=${GPU_ARCH:-80}
