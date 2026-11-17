#!/usr/bin/env bash
# Submit one indexed example with explicit site scheduler options.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
example=${1:?usage: scripts/slurm/submit.sh EXAMPLE [sbatch options] [-- example options]}
shift
case "$example" in *[!a-zA-Z0-9_]*|'') echo 'Invalid example name' >&2; exit 2;; esac
script="$root/examples/slurm/$example.sbatch"
[[ -f "$script" ]] || { echo "Unknown example: $example (see examples/README.md)" >&2; exit 2; }
scheduler_options=()
example_options=()
while (($#)); do
  if [[ $1 == -- ]]; then
    shift
    example_options=("$@")
    break
  fi
  scheduler_options+=("$1")
  shift
done
export VENUS_ROOT="$root"
# Expand only nonempty arrays for compatibility with Bash 3.2 and strict mode.
set -- "$script"
if ((${#scheduler_options[@]})); then set -- "${scheduler_options[@]}" "$@"; fi
if ((${#example_options[@]})); then set -- "$@" "${example_options[@]}"; fi
exec sbatch --export=ALL "$@"
