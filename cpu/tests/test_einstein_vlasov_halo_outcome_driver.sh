#!/usr/bin/env bash
set -euo pipefail

binary_arg=${1:?usage: test_einstein_vlasov_halo_outcome_driver.sh BINARY PYTHON}
python=${2:-python3}
binary=$(cd "$(dirname "$binary_arg")" && pwd)/$(basename "$binary_arg")
scratch=$(mktemp -d "${TMPDIR:-/tmp}/venus-halo-outcome.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# Produce a real, positively bound static-shell checkpoint.  The two synthetic
# history files below are classifier regression fixtures, not claims about the
# short physical evolution: resuming at the already reached endpoint exercises
# the production summary and --expected-outcome exit path without another step.
common=(
  --coordinates schwarzschild
  --physical-ic "$scratch/common.phys"
  --threads 2
  --high-initial-condition energy_cutoff_shell
  --high-markers 128
  --high-radial-cells 256
  --high-r-min 0.01
  --high-r-max 24
  --high-inner-boundary origin_parity
  --high-radial-grid linear
  --high-energy-cutoff-amplitude 0.5
  --high-energy-cutoff-ell0 1
  --high-energy-cutoff-central-y 0.13
  --high-radial-infall 0.01
  --high-angular-momentum-scale 1
  --high-sampling-realization 0
  --high-target-initial-adm-mass 0.85
  --high-gap-stop 0.001
  --high-t-end 0.1
  --high-dt-max 0.005
  --high-output-steps 1
)

"$binary" --expected-outcome any --output-dir "$scratch/base" \
  "${common[@]}" >/dev/null
cp -R "$scratch/base" "$scratch/balanced"
cp -R "$scratch/base" "$scratch/near_trapped"

"$python" - "$scratch" <<'PY'
import csv
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source = root / "base" / "high_compactness_history.csv"
with source.open(newline="") as stream:
    reader = csv.DictReader(stream)
    fieldnames = reader.fieldnames
    original = list(reader)
assert fieldnames is not None and len(original) >= 21
final_step = int(original[-1]["step"])
final_time = float(original[-1]["time"])
assert final_step >= 20 and math.isclose(final_time, 0.1, abs_tol=2.0e-15)

def make_balanced():
    rows = []
    for i in range(21):
        row = dict(original[0])
        phase = float(i - 14)
        row["step"] = str(round(final_step * i / 20.0))
        row["time"] = repr(final_time * i / 20.0)
        row["dt"] = "0" if i == 0 else repr(final_time / 20.0)
        row["r10"] = "3.0"
        row["r50"] = repr(4.0 - 0.01 * min(i, 10))
        row["r90"] = repr(5.0 - 0.012 * min(i, 10))
        row["virial_ratio"] = "1.0"
        row["max_compactness"] = repr(0.35 + 0.001 * min(i, 10))
        row["min_gap"] = repr(1.0 - float(row["max_compactness"]))
        row["anisotropic_jeans_relative_l1"] = repr(
            0.32 - 0.01 * min(i, 12)
        )
        if i == 6:
            row["r50"] = "3.70"
        if i >= 14:
            row["r50"] = repr(3.86 + 0.015 * math.sin(phase))
            row["r90"] = repr(4.84 + 0.018 * math.cos(phase))
            row["max_compactness"] = repr(
                0.36 + 0.003 * math.sin(0.7 * phase)
            )
            row["min_gap"] = repr(1.0 - float(row["max_compactness"]))
            row["anisotropic_jeans_relative_l1"] = repr(
                0.19 + 0.015 * math.cos(phase)
            )
        rows.append(row)
    rows[-1]["step"] = str(final_step)
    rows[-1]["time"] = repr(final_time)
    return rows

balanced = make_balanced()
near_trapped = [dict(row) for row in balanced]
for i in range(14, len(near_trapped)):
    compactness = 0.996 + 2.0e-4 * math.sin(float(i))
    near_trapped[i]["max_compactness"] = repr(compactness)
    near_trapped[i]["min_gap"] = repr(1.0 - compactness)
    near_trapped[i]["anisotropic_jeans_relative_l1"] = repr(
        0.80 + 0.01 * math.cos(float(i))
    )

for name, rows in (("balanced", balanced), ("near_trapped", near_trapped)):
    path = root / name / "high_compactness_history.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
PY

"$binary" --resume --expected-outcome halo \
  --output-dir "$scratch/balanced" "${common[@]}" >/dev/null

set +e
"$binary" --resume --expected-outcome halo \
  --output-dir "$scratch/near_trapped" "${common[@]}" >/dev/null
near_trapped_status=$?
set -e
test "$near_trapped_status" -eq 2

"$python" - "$scratch" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
balanced = json.loads(
    (root / "balanced" / "summary.json").read_text()
)["high_compactness"]
near = json.loads(
    (root / "near_trapped" / "summary.json").read_text()
)["high_compactness"]

assert balanced["fixed_family_positive_binding_energy"] is True
assert balanced["virialized_halo"] is True
assert balanced["outcome"] == "virialized_halo"
audit = balanced["halo_classifier"]
assert audit["schema"] == "venus_wpic_halo_classifier_v1"
assert audit["virialized_halo"] is True
assert audit["failed_checks"] == []
assert audit["history_samples"] == 21
assert audit["late_samples"] == 7
assert audit["late_anisotropic_jeans_samples"] == 7
assert audit["thresholds"] == {
    "end_time_relative_tolerance": 1.0e-6,
    "late_window_fraction": 0.30,
    "minimum_late_samples": 5,
    "compression_fraction": 0.95,
    "maximum_late_r50_over_initial": 1.50,
    "maximum_r50_relative_spread": 0.75,
    "maximum_dimensionless_r50_trend": 0.25,
    "maximum_dimensionless_r90_trend": 0.25,
    "maximum_compactness_absolute_spread": 0.05,
    "maximum_compactness_absolute_trend": 0.05,
    "maximum_anisotropic_jeans_relative_l1": 0.25,
    "maximum_anisotropic_jeans_late_increase": 0.05,
    "minimum_late_jeans_fraction": 0.75,
    "maximum_escaped_rest_fraction": 0.05,
    "near_horizon_compactness": 0.99,
}

# This fixture satisfies every former three-term shortcut condition.  It must
# nevertheless be undetermined, and --expected-outcome halo returned status 2.
assert near["fixed_family_positive_binding_energy"] is True
assert near["minimum_r50"] < 0.95 * near["initial_r50"]
assert near["final_r50"] < 1.5 * near["initial_r50"]
assert near["horizon_formed"] is False
assert near["virialized_halo"] is False
assert near["outcome"] == "undetermined"
near_audit = near["halo_classifier"]
assert near_audit["maximum_history_compactness"] > 0.995
assert "not_near_horizon_collapse" in near_audit["failed_checks"]
assert "late_anisotropic_jeans_balance" in near_audit["failed_checks"]
PY

echo "Einstein--Vlasov halo outcome driver regression passed."
