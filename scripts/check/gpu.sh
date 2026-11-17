#!/usr/bin/env bash
# Run inside an allocation with one GPU (two when the MPI build is selected).
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
build=${1:-build/gpu}
build=$(cd "$build" && pwd)
python=${PYTHON:-python3}
ctest --test-dir "$build" --output-on-failure --timeout 120 --exclude-regex '^production_output$'
# This file replacement test uses node-local storage for portable NFS behavior.
work=$(mktemp -d "${TMPDIR:-/tmp}/venus-output.XXXXXX")
trap 'rm -rf "$work"' EXIT
(cd "$work" && "$build/test_production_output")
"$python" gpu/tests/test_production_driver_cli.py \
 "$build/venus_gpu_vlasov_poisson" "$build/venus_gpu_subhalo_boltzmann_poisson" \
 "$build/venus_gpu_sidm_boltzmann_poisson" "$build/venus_gpu_sidm_constant" \
 "$build/venus_gpu_vlasov_poisson_static"
