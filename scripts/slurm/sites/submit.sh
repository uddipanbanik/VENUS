#!/usr/bin/env bash
# Site defaults plus the common, physics-specific Slurm templates.
# usage: bash scripts/slurm/sites/submit.sh SITE EXAMPLE [sbatch options] [-- example options]
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
site=${1:?Choose ias-typhon, ias-apollo, symmetry-cpu, symmetry-gpu or symmetry-h200}
example=${2:?Choose an example from examples/README.md}
shift 2
case "$site" in
  ias-typhon) source "$root/scripts/build/ias_environment.sh" cpu; options=(--partition typhon --qos short); backend=cpu ;;
  ias-apollo) source "$root/scripts/build/ias_environment.sh" gpu; options=(--partition apollo --qos gpu-short); backend=gpu ;;
  symmetry-cpu) source "$root/scripts/build/symmetry_environment.sh" cpu; options=(--partition debugq); backend=cpu ;;
  symmetry-gpu) source "$root/scripts/build/symmetry_environment.sh" gpu; options=(--partition gpudebugq); backend=gpu ;;
  symmetry-h200) export GPU_ARCH=${GPU_ARCH:-90}; source "$root/scripts/build/symmetry_environment.sh" gpu; options=(--partition h200debugq); backend=gpu ;;
  *) echo "Unknown site: $site" >&2; exit 2 ;;
esac
case "$example" in gpu_*) actual=gpu ;; *) actual=cpu ;; esac
[[ $backend == "$actual" ]] || { echo "Example $example needs a $actual site choice." >&2; exit 2; }
exec bash "$root/scripts/slurm/submit.sh" "$example" "${options[@]}" "$@"
