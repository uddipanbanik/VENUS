#!/bin/bash
# Re-evaluate coordinate-map traces from completed Schwarzschild/EF outputs at increasing eikonal substeps.
# The stored particle and metric evolution remains fixed. Each comparison uses n and 2n traces and records their raw change.
# Output bundles preserve inputs, resource use, coverage, observed refinement order and the distinction between a clean diagnostic and independent convergence evidence.

# Reprocess completed SC/EF parity products at explicitly selected eikonal
# substep counts.  This job never invokes a VENUS evolution executable.
#
# Each --eikonal-substeps=n comparison advances both n and 2n traces and
# reports their raw difference.  A result therefore remains diagnostic until
# the independent resolution/realization convergence builder is run later.

#SBATCH --job-name=venus-eikonal-reprocess
#SBATCH --partition=typhon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 027

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?set CAMPAIGN_ROOT}"
SWEEP_ROOT="${SWEEP_ROOT:?set SWEEP_ROOT}"
PAIR="${PAIR:?set PAIR}"
FROZEN_CERTIFIER="${FROZEN_CERTIFIER:?set FROZEN_CERTIFIER}"
SUBSTEP_TEXT="${VENUS_EIKONAL_SUBSTEP_LIST:-2 4 8 16 32 64}"
TIME_BINARY="${VENUS_EIKONAL_TIME_BINARY:-/usr/bin/time}"
TEST_MODE="${VENUS_EIKONAL_REPROCESS_TEST_MODE:-0}"
EXPECTED_CERTIFIER_SHA256="ce0e31132a1ef481c5ef95308ea5e28ef614cd6805c575eda1badb6d9465f3f6"

case "$PAIR" in
  fine_a|coarse_a|fine_b|coarse_b) ;;
  *) echo "Unsupported parity pair: $PAIR" >&2; exit 1 ;;
esac
case "$TEST_MODE" in 0|1) ;; *) echo "Invalid test-mode flag." >&2; exit 1 ;; esac
if [ "$TEST_MODE" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "The local test harness cannot run inside a Slurm allocation." >&2
  exit 1
fi
if [ "$TEST_MODE" != 1 ] && [ "$TIME_BINARY" != /usr/bin/time ]; then
  echo "A non-system time binary is allowed only by the local test harness." >&2
  exit 1
fi

for directory in "$CAMPAIGN_ROOT" "$SWEEP_ROOT"; do
  if [ ! -d "$directory" ]; then
    echo "Required directory is missing: $directory" >&2
    exit 1
  fi
done
for artifact in "$FROZEN_CERTIFIER" "$TIME_BINARY"; do
  if [ ! -s "$artifact" ]; then
    echo "Required executable/input is missing or empty: $artifact" >&2
    exit 1
  fi
done
if [ ! -x "$TIME_BINARY" ]; then
  echo "Time wrapper is not executable: $TIME_BINARY" >&2
  exit 1
fi
certifier_sha="$(sha256sum "$FROZEN_CERTIFIER" | awk '{print $1}')"
if [ "$TEST_MODE" != 1 ] \
    && [ "$certifier_sha" != "$EXPECTED_CERTIFIER_SHA256" ]; then
  echo "Frozen comparator SHA-256 is not the reviewed parity source." >&2
  echo "Expected: $EXPECTED_CERTIFIER_SHA256" >&2
  echo "Actual:   $certifier_sha" >&2
  exit 1
fi

campaign_real="$(cd "$CAMPAIGN_ROOT" && pwd -P)"
sweep_real="$(cd "$SWEEP_ROOT" && pwd -P)"
certifier_real="$(cd "$(dirname "$FROZEN_CERTIFIER")" && pwd -P)/$(basename "$FROZEN_CERTIFIER")"
if [ "$TEST_MODE" = 1 ]; then
  case "$campaign_real" in
    /tmp/*|/private/tmp/*|/private/var/folders/*) ;;
    *)
      echo "The test harness is restricted to an operating-system temporary directory." >&2
      exit 1
      ;;
  esac
fi
case "$sweep_real" in
  "$campaign_real"/diagnostic_reprocessing/*) ;;
  *)
    echo "SWEEP_ROOT must be a new child of CAMPAIGN_ROOT/diagnostic_reprocessing." >&2
    exit 1
    ;;
esac
case "$certifier_real" in
  "$sweep_real"/source/*) ;;
  *) echo "FROZEN_CERTIFIER must be inside SWEEP_ROOT/source." >&2; exit 1 ;;
esac

# Powers of two in increasing order provide adjacent refinement ratios with an unambiguous observed-order comparison.
read -r -a substep_tokens <<< "$SUBSTEP_TEXT"
substeps=()
previous=0
for value in "${substep_tokens[@]}"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "VENUS_EIKONAL_SUBSTEP_LIST contains a non-positive integer." >&2
    exit 1
  fi
  if [ $((value & (value - 1))) -ne 0 ]; then
    echo "Eikonal substep values must be powers of two: $value" >&2
    exit 1
  fi
  if [ "$value" -le "$previous" ]; then
    echo "Eikonal substep values must be unique and strictly increasing." >&2
    exit 1
  fi
  substeps+=("$value")
  previous="$value"
done
if [ "${#substeps[@]}" -eq 0 ]; then
  echo "VENUS_EIKONAL_SUBSTEP_LIST is empty." >&2
  exit 1
fi

PAIR_ROOT="$CAMPAIGN_ROOT/pairs/$PAIR"
SCHWARZSCHILD_OUTPUT="$PAIR_ROOT/schwarzschild"
EF_OUTPUT="$PAIR_ROOT/eddington_finkelstein"
NULL_BASE="$PAIR_ROOT/null_map/ef_null_initial_data.bin"
COORDINATE_MAP_MANIFEST="$NULL_BASE.manifest.json"
MARKER_CROSSINGS="$NULL_BASE.crossings.csv"
# Each completed pair receives a fresh output directory so repeated diagnostic levels cannot overwrite earlier results.
OUTDIR="$SWEEP_ROOT/$PAIR"

for directory in "$PAIR_ROOT" "$SCHWARZSCHILD_OUTPUT" "$EF_OUTPUT"; do
  if [ ! -d "$directory" ]; then
    echo "Completed parity directory is missing: $directory" >&2
    exit 1
  fi
done
for artifact in "$COORDINATE_MAP_MANIFEST" "$MARKER_CROSSINGS"; do
  if [ ! -s "$artifact" ]; then
    echo "Completed parity artifact is missing or empty: $artifact" >&2
    exit 1
  fi
done
if [ -e "$OUTDIR" ]; then
  echo "Refusing to overwrite a prior pair sweep: $OUTDIR" >&2
  exit 1
fi
mkdir "$OUTDIR"

if ! (
  cd "$SWEEP_ROOT/source"
  sha256sum -c source.sha256
) > "$OUTDIR/source_precheck.log" 2>&1; then
  echo "Frozen reprocessor source failed its SHA-256 preflight." >&2
  exit 1
fi
if ! (
  cd "$CAMPAIGN_ROOT"
  sha256sum -c campaign_controls.manifest.sha256
) > "$OUTDIR/campaign_controls_precheck.log" 2>&1; then
  echo "Campaign controls failed their SHA-256 preflight." >&2
  exit 1
fi

printf '%s\n' "${substeps[*]}" > "$OUTDIR/requested_substeps.txt"

# Enumerate every file directly consumed by the comparator, all authenticated
# manifest artifacts, and the completed-run provenance files.  Importing the
# frozen comparator ensures that supported manifest spellings are
# resolved exactly as they are during comparison.
python3 - \
  "$FROZEN_CERTIFIER" "$CAMPAIGN_ROOT" "$PAIR_ROOT" \
  "$SCHWARZSCHILD_OUTPUT" "$EF_OUTPUT" \
  "$COORDINATE_MAP_MANIFEST" "$MARKER_CROSSINGS" \
  "$OUTDIR/input_artifacts.list" <<'PY_INPUTS'
from __future__ import print_function

import importlib.util
import json
import os
import sys


certifier, campaign, pair_root, sc_dir, ef_dir, manifest_path, crossings, output = (
    sys.argv[1:]
)
campaign = os.path.realpath(campaign)
spec = importlib.util.spec_from_file_location("frozen_venus_certifier", certifier)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import the frozen coordinate comparator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open(manifest_path, "r") as stream:
    manifest = json.load(stream)
# Resolve artifact spellings through the same frozen comparator that will perform the scientific comparison.
records = module.manifest_artifacts(manifest, manifest_path)
unverified = sorted(
    name for name, record in records.items()
    if record.get("verified") is not True
)
if unverified:
    raise SystemExit(
        "manifest artifacts fail SHA-256 verification: {}".format(
            ", ".join(unverified)
        )
    )

paths = [
    os.path.join(campaign, "campaign_controls.manifest.tsv"),
    os.path.join(campaign, "campaign_controls.manifest.sha256"),
    os.path.join(campaign, "workflow_source.sha256"),
    os.path.join(campaign, "submission.tsv"),
    os.path.join(pair_root, "physical_ic.bin"),
    os.path.join(sc_dir, "summary.json"),
    os.path.join(sc_dir, "high_compactness_profiles.csv"),
    os.path.join(sc_dir, "source.sha256"),
    os.path.join(sc_dir, "binary.sha256"),
    os.path.join(sc_dir, "resolved_command.sh"),
    os.path.join(ef_dir, "summary.json"),
    os.path.join(ef_dir, "high_compactness_profiles.csv"),
    os.path.join(ef_dir, "initial_coordinate_map_residuals.csv"),
    os.path.join(ef_dir, "source.sha256"),
    os.path.join(ef_dir, "binary.sha256"),
    os.path.join(ef_dir, "resolved_command.sh"),
    manifest_path,
    crossings,
]
# Include every authenticated map artifact with the directly consumed field/summary files in the input hash inventory.
paths.extend(record["path"] for record in records.values())
resolved = []
for raw_path in paths:
    path = os.path.realpath(raw_path)
    try:
        in_campaign = os.path.commonpath((campaign, path)) == campaign
    except ValueError:
        in_campaign = False
    if not in_campaign:
        raise SystemExit("input escapes the campaign root: {}".format(path))
    if "\n" in path or "\r" in path:
        raise SystemExit("newline in input path is unsupported")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise SystemExit("required input is missing or empty: {}".format(path))
    resolved.append(path)
with open(output, "w") as stream:
    for path in sorted(set(resolved)):
        stream.write(path + "\n")
PY_INPUTS

: > "$OUTDIR/input_artifacts.sha256"
while IFS= read -r artifact; do
  sha256sum -- "$artifact" >> "$OUTDIR/input_artifacts.sha256"
done < "$OUTDIR/input_artifacts.list"
(
  cd "$OUTDIR"
  sha256sum input_artifacts.sha256 > input_artifacts.sha256.sha256
)
if ! (
  cd "$OUTDIR"
  sha256sum -c input_artifacts.sha256.sha256
) > "$OUTDIR/input_manifest_sha256_check.log" 2>&1; then
  echo "The input-artifact hash manifest failed self-verification." >&2
  exit 1
fi
if ! sha256sum -c "$OUTDIR/input_artifacts.sha256" \
    > "$OUTDIR/input_precheck.log" 2>&1; then
  echo "A reprocessing input changed during the input preflight." >&2
  exit 1
fi

input_set_sha="$(sha256sum "$OUTDIR/input_artifacts.sha256" | awk '{print $1}')"
operational_failure=0
# Run each requested eikonal refinement against the same completed evolution and strict common-domain criteria.
for n in "${substeps[@]}"; do
  printf -v tag 'n%03d' "$n"
  run_dir="$OUTDIR/$tag"
  mkdir "$run_dir"
  diagnostic="$run_dir/diagnostic.json"

  command=(
    python3 "$FROZEN_CERTIFIER"
    --schwarzschild "$SCHWARZSCHILD_OUTPUT"
    --ef "$EF_OUTPUT"
    --manifest "$COORDINATE_MAP_MANIFEST"
    --marker-crossings "$MARKER_CROSSINGS"
    --empirical-distribution strict
    --r-min 0.0
    --r-max inf
    --time-samples 32
    --radial-samples 64
    --minimum-events 64
    --minimum-time-slices 3
    --minimum-event-coverage 0.95
    --minimum-time-coverage 0.95
    --minimum-radial-coverage 0.95
    --minimum-marker-samples 64
    --minimum-a 1.0e-3
    --minimum-stress-signal 1.0e-12
    --absolute-floor 1.0e-12
    --max-cadence-to-dr 0.75
    --eikonal-substeps "$n"
    --eikonal-convergence-tolerance 2.0e-5
    --composition-tolerance 5.0e-4
    --metric-tolerance 0.005
    --stress-tolerance 0.02
    --moment-tolerance 0.02
    --distribution-tolerance 1.0e-10
    --crossing-residual-tolerance 1.0e-8
    --outer-vacuum-tolerance 1.0e-12
    --output "$diagnostic"
    --sha256-output "$diagnostic.sha256"
  )
  {
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "$run_dir/resolved_command.sh"

  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  "$TIME_BINARY" -v -o "$run_dir/resource_usage.txt" \
    "${command[@]}" > "$run_dir/diagnostic.log" 2>&1
  certifier_status=$?
  set -e
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' "$certifier_status" > "$run_dir/exit_status.txt"

  # No convergence evidence is supplied, so even a clean single-resolution
  # comparison must be scientifically inconclusive with process status 2.
  # Single-resolution traces retain scientific inconclusive status until independent discretization and sampling convergence is supplied.
  if [ "$certifier_status" -ne 2 ]; then
    operational_failure=1
  fi
  if [ ! -s "$diagnostic" ] || [ ! -s "$diagnostic.sha256" ]; then
    operational_failure=1
  elif ! (
    cd "$run_dir"
    sha256sum -c diagnostic.json.sha256
  ) > "$run_dir/diagnostic_sha256_check.log" 2>&1; then
    operational_failure=1
  fi

  {
    printf 'pair\t%s\n' "$PAIR"
    printf 'eikonal_substeps_n\t%s\n' "$n"
    printf 'accepted_trace_substeps_2n\t%s\n' "$((2 * n))"
    printf 'start_utc\t%s\n' "$start_utc"
    printf 'end_utc\t%s\n' "$end_utc"
    printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID:-local}"
    printf 'node\t%s\n' "${SLURMD_NODENAME:-$(hostname)}"
    printf 'allocated_cpus\t%s\n' "${SLURM_CPUS_PER_TASK:-4}"
    printf 'requested_memory\t64G\n'
    printf 'certifier_sha256\t%s\n' "$certifier_sha"
    printf 'input_artifact_manifest_sha256\t%s\n' "$input_set_sha"
    printf 'comparator_exit_status\t%s\n' "$certifier_status"
  } > "$run_dir/metadata.tsv"

  bundle=(
    resolved_command.sh diagnostic.log resource_usage.txt exit_status.txt
    metadata.tsv
  )
  [ ! -s "$diagnostic" ] || bundle+=(diagnostic.json)
  [ ! -s "$diagnostic.sha256" ] || bundle+=(diagnostic.json.sha256)
  [ ! -s "$run_dir/diagnostic_sha256_check.log" ] || \
    bundle+=(diagnostic_sha256_check.log)
  (
    cd "$run_dir"
    sha256sum "${bundle[@]}" > bundle.sha256
  )
done

if ! sha256sum -c "$OUTDIR/input_artifacts.sha256" \
    > "$OUTDIR/input_postcheck.log" 2>&1; then
  operational_failure=1
fi
if ! (
  cd "$SWEEP_ROOT/source"
  sha256sum -c source.sha256
) > "$OUTDIR/source_postcheck.log" 2>&1; then
  operational_failure=1
fi

set +e
python3 - "$OUTDIR" "${substeps[@]}" <<'PY_SUMMARY'
from __future__ import print_function

import hashlib
import json
import math
import os
import sys


MISSING_CONVERGENCE = (
    "single-resolution agreement has no independent quantified "
    "discretization/PIC convergence evidence"
)
RAW_TOLERANCE = 2.0e-5
COMPOSITION_TOLERANCE = 5.0e-4
COVERAGE_TOLERANCE = 0.95
DOMAIN_DEFINITION = (
    "maximal_contiguous_exterior_pre_schwarzschild_terminal"
)
RAW_DEFINITION = (
    "max(abs(delta coordinate time),abs(delta log Jacobian)); "
    "no Richardson division without demonstrated asymptotic order"
)


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def require_bound(reasons, mapping, name, ceiling):
    value = mapping.get(name)
    if not finite_number(value) or value > ceiling:
        reasons.append("{} is absent/non-finite or exceeds {}".format(
            name, ceiling
        ))


# Evaluate schema, empirical-distribution evidence, map integrity, raw trace error and connected common-domain coverage together.
def inspect_level(path, n):
    reasons = []
    try:
        with open(path, "r") as stream:
            value = json.load(stream)
    except (IOError, OSError, ValueError) as error:
        return {
            "n": n,
            "clean": False,
            "integrity_error": str(error),
            "reasons": ["diagnostic JSON is unreadable"],
        }
    if value.get("schema") != "venus_wpic_dynamic_coordinate_certificate_v2":
        reasons.append("wrong certificate schema")
    if value.get("status") != "inconclusive":
        reasons.append("single-resolution status is not inconclusive")
    if value.get("issues") != [MISSING_CONVERGENCE]:
        reasons.append("issues are not exactly missing convergence evidence")
    if value.get("failed_fields") not in ([], None):
        reasons.append("field/stress/moment comparison failed")

    distribution = value.get("distribution", {})
    if distribution.get("mode") != "strict":
        reasons.append("empirical distribution mode is not strict")
    if not distribution.get("evidence"):
        reasons.append("strict empirical distribution evidence is absent")
    if distribution.get("failed_fields") not in ([], None):
        reasons.append("empirical distribution comparison failed")

    manifest = value.get("provenance", {}).get(
        "coordinate_map_manifest", {}
    )
    artifacts = manifest.get("artifacts", {})
    if not artifacts or any(
            record.get("verified") is not True
            for record in artifacts.values()):
        reasons.append("coordinate-map artifacts are not all authenticated")

    mapping = value.get("mapping", {})
    if mapping.get("step_refinement_error_definition") != RAW_DEFINITION:
        reasons.append("step-refinement statistic is not the strict raw change")
    require_bound(
        reasons, mapping, "forward_maximum_step_refinement_error",
        RAW_TOLERANCE,
    )
    require_bound(
        reasons, mapping, "inverse_maximum_step_refinement_error",
        RAW_TOLERANCE,
    )
    require_bound(
        reasons, mapping, "maximum_composition_time_error",
        COMPOSITION_TOLERANCE,
    )
    require_bound(
        reasons, mapping, "maximum_composition_jacobian_error",
        COMPOSITION_TOLERANCE,
    )

    domain = mapping.get("common_domain", {})
    if domain.get("definition") != DOMAIN_DEFINITION:
        reasons.append("wrong or absent common-domain definition")
    for name in (
            "unexplained_holes", "time_topology_gap_slices",
            "disconnected_non_horizon_support_samples"):
        if domain.get(name) != 0:
            reasons.append("{} is not zero".format(name))
    for name in (
            "event_coverage", "time_slice_coverage",
            "minimum_per_slice_radial_coverage"):
        coverage = domain.get(name)
        if not finite_number(coverage) or coverage < COVERAGE_TOLERANCE:
            reasons.append("{} is below {}".format(
                name, COVERAGE_TOLERANCE
            ))
    if not isinstance(domain.get("compared_events"), int) \
            or domain.get("compared_events") < 64:
        reasons.append("fewer than 64 common-domain events were compared")
    if not isinstance(domain.get("complete_time_slices"), int) \
            or domain.get("complete_time_slices") < 3:
        reasons.append("fewer than three complete common-domain time slices")

    forward = mapping.get("forward_maximum_step_refinement_error")
    inverse = mapping.get("inverse_maximum_step_refinement_error")
    raw_values = [
        item for item in (forward, inverse) if finite_number(item)
    ]
    # Use the larger forward/inverse raw trace change for the refinement-order sequence.
    combined_raw = max(raw_values) if raw_values else None
    return {
        "n": n,
        "accepted_trace_substeps": 2 * n,
        "clean": not reasons,
        "reasons": reasons,
        "forward_raw_error": forward,
        "inverse_raw_error": inverse,
        "combined_raw_error": combined_raw,
        "forward_all_attempted_raw_error": mapping.get(
            "forward_all_attempted_maximum_step_refinement_error"
        ),
        "inverse_all_attempted_raw_error": mapping.get(
            "inverse_all_attempted_maximum_step_refinement_error"
        ),
        "maximum_composition_time_error": mapping.get(
            "maximum_composition_time_error"
        ),
        "maximum_composition_jacobian_error": mapping.get(
            "maximum_composition_jacobian_error"
        ),
        "common_domain": domain,
        "forward_worst_event": mapping.get(
            "forward_worst_step_refinement_event"
        ),
        "inverse_worst_event": mapping.get(
            "inverse_worst_step_refinement_event"
        ),
    }


def observed_order(coarse, fine):
    if not finite_number(coarse) or not finite_number(fine):
        return None
    if coarse <= 0.0 or fine <= 0.0:
        return None
    return math.log(coarse / fine, 2.0)


outdir = os.path.abspath(sys.argv[1])
levels = [int(item) for item in sys.argv[2:]]
audits = []
integrity_failure = False
for n in levels:
    audit = inspect_level(
        os.path.join(outdir, "n{:03d}".format(n), "diagnostic.json"), n
    )
    integrity_failure = integrity_failure or bool(audit.get("integrity_error"))
    audits.append(audit)

by_level = dict((audit["n"], audit) for audit in audits)
highest = max(levels)
highest_clean = by_level[highest]["clean"]
strict_two_level_evaluable = 32 in by_level and 64 in by_level
strict_two_level_pass = (
    strict_two_level_evaluable
    and by_level[32]["clean"] and by_level[64]["clean"]
)

orders = []
for index in range(1, len(levels)):
    lower = levels[index - 1]
    upper = levels[index]
    if upper != 2 * lower:
        continue
    orders.append({
        "lower_n": lower,
        "upper_n": upper,
        "observed_order": observed_order(
            by_level[lower].get("combined_raw_error"),
            by_level[upper].get("combined_raw_error"),
        ),
    })

# A persistent high-resolution error plateau suggests a saved-time interpolation limit, while explicit coverage gaps are classified separately.
high_orders = orders[-2:]
plateau = (
    not highest_clean
    and len(high_orders) == 2
    and all(
        finite_number(item["observed_order"])
        and item["observed_order"] <= 0.5
        for item in high_orders
    )
)
highest_reasons = by_level[highest]["reasons"]
coverage_blocker = any(
    token in reason
    for reason in highest_reasons
    for token in (
        "common-domain", "unexplained_holes", "time_topology_gap_slices",
        "disconnected_non_horizon_support_samples", "coverage",
        "common-domain events", "common-domain time slices",
    )
)
# Prioritize missing/corrupt inputs, then numerical accuracy and coverage when summarizing the sweep.
if integrity_failure:
    classification = "operational_or_integrity_failure"
elif highest_clean:
    classification = "highest_requested_level_clean"
elif coverage_blocker:
    classification = "coverage_or_common_domain_blocker"
elif plateau:
    classification = "saved_time_interpolation_blocker"
else:
    classification = "substep_refinement_or_other_strict_gate_blocker"

result = {
    "schema": "venus_wpic_eikonal_reprocessing_sweep_v1",
    "pair": os.path.basename(outdir),
    "requested_substeps": levels,
    "raw_step_refinement_tolerance": RAW_TOLERANCE,
    "composition_tolerance": COMPOSITION_TOLERANCE,
    "minimum_coverages": {
        "events": COVERAGE_TOLERANCE,
        "time_slices": COVERAGE_TOLERANCE,
        "per_slice_radial": COVERAGE_TOLERANCE,
    },
    "highest_requested_level": highest,
    "highest_requested_level_clean": highest_clean,
    "strict_two_level_gate_levels": [32, 64],
    "strict_two_level_gate_evaluable_in_this_root": (
        strict_two_level_evaluable
    ),
    "strict_two_level_gate_pass": strict_two_level_pass,
    "classification": classification,
    "observed_orders": orders,
    "levels": audits,
}
rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
summary_path = os.path.join(outdir, "sweep_summary.json")
temporary_path = summary_path + ".tmp"
with open(temporary_path, "w") as stream:
    stream.write(rendered)
os.replace(temporary_path, summary_path)
digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
with open(summary_path + ".sha256", "w") as stream:
    stream.write("{}  {}\n".format(digest, os.path.basename(summary_path)))

if integrity_failure:
    raise SystemExit(1)
raise SystemExit(0 if highest_clean else 2)
PY_SUMMARY
summary_status=$?
set -e

if [ ! -s "$OUTDIR/sweep_summary.json" ] \
    || [ ! -s "$OUTDIR/sweep_summary.json.sha256" ]; then
  operational_failure=1
elif ! (
  cd "$OUTDIR"
  sha256sum -c sweep_summary.json.sha256
) > "$OUTDIR/sweep_summary_sha256_check.log" 2>&1; then
  operational_failure=1
fi

if [ "$summary_status" -eq 0 ] && [ "$operational_failure" -eq 0 ]; then
  (
    cd "$OUTDIR"
    sha256sum sweep_summary.json > HIGHEST_REQUESTED_LEVEL.PASS
  )
  if python3 - "$OUTDIR/sweep_summary.json" <<'PY_STRICT'
from __future__ import print_function
import json
import sys
with open(sys.argv[1], "r") as stream:
    value = json.load(stream)
raise SystemExit(0 if value.get("strict_two_level_gate_pass") is True else 1)
PY_STRICT
  then
    (
      cd "$OUTDIR"
      sha256sum sweep_summary.json > STRICT_SUBSTEP_GATE.PASS
    )
  fi
fi

# Bind the level reports and closing input/source checks into the final pair inventory.
pair_bundle=(
  requested_substeps.txt source_precheck.log campaign_controls_precheck.log
  input_artifacts.list input_artifacts.sha256
  input_artifacts.sha256.sha256 input_manifest_sha256_check.log
  input_precheck.log input_postcheck.log
  source_postcheck.log sweep_summary.json sweep_summary.json.sha256
  sweep_summary_sha256_check.log
)
for n in "${substeps[@]}"; do
  printf -v tag 'n%03d' "$n"
  pair_bundle+=("$tag/bundle.sha256")
done
[ ! -s "$OUTDIR/HIGHEST_REQUESTED_LEVEL.PASS" ] || \
  pair_bundle+=(HIGHEST_REQUESTED_LEVEL.PASS)
[ ! -s "$OUTDIR/STRICT_SUBSTEP_GATE.PASS" ] || \
  pair_bundle+=(STRICT_SUBSTEP_GATE.PASS)
(
  cd "$OUTDIR"
  sha256sum "${pair_bundle[@]}" > pair_bundle.sha256
)

if [ "$operational_failure" -ne 0 ] || [ "$summary_status" -eq 1 ]; then
  echo "Eikonal reprocessing encountered an operational/integrity failure." >&2
  exit 1
fi
if [ "$summary_status" -ne 0 ]; then
  echo "Eikonal reprocessing completed but the highest requested level is blocked." >&2
  exit 2
fi
echo "Authenticated eikonal reprocessing complete: $OUTDIR"
