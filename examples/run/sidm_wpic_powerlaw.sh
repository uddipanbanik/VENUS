#!/usr/bin/env bash
# Non-relativistic collisional / SIDM / static; input and finite parameters: examples/catalogue.json.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${PYTHON:-python3}" "$root/scripts/run_example.py" sidm_wpic_powerlaw "$@"
