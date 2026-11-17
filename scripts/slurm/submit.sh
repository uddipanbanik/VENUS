#!/usr/bin/env bash
# Submit one indexed example with explicit site scheduler options.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
example=${1:?usage: scripts/slurm/submit.sh EXAMPLE [sbatch options]}
shift
case "$example" in *[!a-zA-Z0-9_]*|'') echo 'Invalid example name' >&2; exit 2;; esac
script="$root/examples/slurm/$example.sbatch"
[[ -f "$script" ]] || { echo "Unknown example: $example (see examples/README.md)" >&2; exit 2; }
export VENUS_ROOT="$root"
exec sbatch --export=ALL "$@" "$script"
