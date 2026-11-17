#!/usr/bin/env bash
# Parse every shipped shell/Slurm script and each Node .cjs utility.
# This reads source grammar through bash -n/node --check. It starts no launcher or simulation.

# Syntax-check every shipped shell or Slurm launcher without executing it.
#
# This covers the maintained portable launch templates. A successful parse
# confirms shell grammar only; site paths, scheduler partitions, resource
# requests, and scientific parameters still require review before submission.

set -euo pipefail

# Resolve the release from this script, allowing the checker to run from any working directory.
release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
checked=0

# Read NUL-delimited paths so spaces in a filename remain part of that filename.
while IFS= read -r -d '' script_path; do
  bash -n "$script_path"
  checked=$((checked + 1))
done < <(
  find "$release_root" -type f \
    \( -name '*.sh' -o -name '*.sbatch' -o -name '*.slurm' \) \
    ! -path "$release_root/build/*" \
    ! -path "$release_root/cpu/build/*" \
    ! -path "$release_root/gpu/build/*" -print0
)

# Check the JavaScript extractor when Node is installed and report that scope explicitly otherwise.
if command -v node >/dev/null 2>&1; then
  while IFS= read -r -d '' script_path; do
    node --check "$script_path"
    checked=$((checked + 1))
  done < <(find "$release_root" -type f -name '*.cjs' -print0)
else
  printf '%s\n' 'SKIP: node is unavailable; .cjs syntax was not checked.' >&2
fi

printf 'PASS: parsed %d shell/Slurm/Node scripts.\n' "$checked"
