#!/usr/bin/env bash
# Run the shipped finite numerical, reader, restart and workflow checks.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
python=${PYTHON:-python3}
source "$root/scripts/build/macos_toolchain.sh"
venus_select_macos_sdk
make -C cpu -j "${BUILD_JOBS:-2}" check
"$python" -m pytest cpu/tests -q
"$python" cpu/tests/test_wpic_background_integration.py --binary cpu/venus_cpu_vlasov_poisson_wpic_expanding
"$python" cpu/tests/test_static_sidm_restart_integration.py --binary cpu/venus_cpu_sidm_wpic_constant
bash cpu/tests/test_expanding_wpic_cadence_integration.sh "$root/cpu/venus_cpu_subhalo_wpic"
bash cpu/tests/test_einstein_vlasov_wpic_coordinate_driver.sh "$root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole" "$python"
bash cpu/tests/test_einstein_vlasov_halo_outcome_driver.sh "$root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole" "$python"
bash cpu/tests/test_einstein_vlasov_wpic_static_workflows.sh
bash cpu/tests/test_einstein_vlasov_wpic_paired_long_gif.sh
bash scripts/check_shell_syntax.sh
