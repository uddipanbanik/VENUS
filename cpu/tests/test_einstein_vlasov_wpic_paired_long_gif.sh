#!/usr/bin/env bash
# Shell/static contract test for the certificate-gated paired-long GIF job.

set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
slurm_dir="$source_root/cpu/scripts/slurm"
submitter_source="$slurm_dir/submit_einstein_vlasov_wpic_paired_long_gif.sh"
wrapper_source="$slurm_dir/ias_einstein_vlasov_wpic_paired_long_gif.sbatch"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/venus-paired-gif.XXXXXX")

cleanup() {
  case "$scratch" in
    "${TMPDIR:-/tmp}"/venus-paired-gif.*) rm -rf "$scratch" ;;
    *) echo "Refusing unsafe test cleanup path: $scratch" >&2 ;;
  esac
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

bash -n "$submitter_source"
bash -n "$wrapper_source"

workflow="$scratch/workflow"
scientific="$scratch/scientific"
campaign="$scratch/campaign"
mkdir -p "$workflow/cpu/scripts/slurm" "$workflow/cpu/tools" \
  "$scientific/cpu/tools" "$campaign/slurm" "$campaign/schwarzschild" \
  "$campaign/eddington_finkelstein" "$scratch/certified-null" "$scratch/bin"
cp "$submitter_source" "$workflow/cpu/scripts/slurm/"
cp "$wrapper_source" "$workflow/cpu/scripts/slurm/"
chmod +x "$workflow/cpu/scripts/slurm/"*

for dependency in compare_wpic_dynamic_coordinates.py \
    render_weighted_einstein_vlasov_pic_profile_gifs.py \
    render_weighted_pic_pure_gamma_gifs.py; do
  printf '# immutable test dependency\n' > "$scientific/cpu/tools/$dependency"
done

# A lightweight renderer fixture writes the six products expected from the
# real renderer.  The production wrapper still performs its own signatures,
# frame-ledger, provenance, path-containment, hash, and atomic-publication
# checks with the real Python interpreter.
renderer="$workflow/cpu/tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py"
printf '%s\n' \
  '#!/usr/bin/env python3' \
  'import argparse, csv, json' \
  'from pathlib import Path' \
  'p = argparse.ArgumentParser()' \
  'p.add_argument("--schwarzschild", type=Path, required=True)' \
  'p.add_argument("--ef", type=Path, required=True)' \
  'p.add_argument("--manifest", type=Path, required=True)' \
  'p.add_argument("--output", type=Path, required=True)' \
  'p.add_argument("--frames", type=int, required=True)' \
  'p.add_argument("--fps")' \
  'p.add_argument("--maximum-same-event-frames")' \
  'p.add_argument("--mapped-radial-samples")' \
  'p.add_argument("--eikonal-substeps")' \
  'a = p.parse_args()' \
  'a.output.write_bytes(b"GIF89a;")' \
  'stem = a.output.with_suffix("")' \
  'for name in ("initial", "middle", "final"):' \
  '    stem.with_name(stem.name + "_" + name + ".png").write_bytes(b"\x89PNG\r\n\x1a\n")' \
  'ledger = stem.with_name(stem.name + "_frames.csv")' \
  'fields = ["frame", "phase", "sc_tau", "ef_v_min", "ef_v_max", "mapped_radius_min", "mapped_radius_max", "mapped_radial_fraction", "maximum_forward_refinement_error", "maximum_inverse_refinement_error", "maximum_composition_error"]' \
  'with ledger.open("w", newline="") as f:' \
  '    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()' \
  '    for i in range(a.frames):' \
  '        phase = "same_event_overlap" if i < a.frames // 2 else "ef_only_after_sc_chart_limit"' \
  '        w.writerow({"frame": i, "phase": phase})' \
  'artifact = a.manifest.parent / "ef_null_initial_data.bin"' \
  'import hashlib' \
  'artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()' \
  'provenance = {"frames": a.frames, "inputs": {' \
  '    "schwarzschild_profiles": str((a.schwarzschild / "high_compactness_profiles.csv").resolve()),' \
  '    "eddington_finkelstein_profiles": str((a.ef / "high_compactness_profiles.csv").resolve()),' \
  '    "manifest": str(a.manifest.resolve()),' \
  '    "authenticated_artifacts": {"null_initial_data": {"path": str(artifact.resolve()), "verified": True, "expected_sha256": artifact_sha, "actual_sha256": artifact_sha}}},' \
  '    "gif": str(a.output), "frame_ledger": str(ledger)}' \
  'stem.with_name(stem.name + "_provenance.json").write_text(json.dumps(provenance) + "\n")' \
  > "$renderer"
chmod +x "$renderer"

printf '{"coordinates":"schwarzschild"}\n' \
  > "$campaign/schwarzschild/summary.json"
printf 'time,r,rho,m\n0,1,1,0.1\n' \
  > "$campaign/schwarzschild/high_compactness_profiles.csv"
printf '{"coordinates":"eddington_finkelstein"}\n' \
  > "$campaign/eddington_finkelstein/summary.json"
printf 'advanced_time,r,T_vv,m\n0,1,1,0.1\n' \
  > "$campaign/eddington_finkelstein/high_compactness_profiles.csv"
printf 'null-data\n' > "$scratch/certified-null/ef_null_initial_data.bin"
printf '{}\n' > "$scratch/certified-null/ef_null_initial_data.bin.manifest.json"
printf 'control\tvalue\n' > "$campaign/campaign_controls.manifest.tsv"
fixture_controls_sha=$(sha256sum "$campaign/campaign_controls.manifest.tsv" | awk '{print $1}')
fixture_certificate_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
printf '%s\n' \
  $'role\tjob_id\tdependency\tcoordinates\texpected_outcome\toutput_dir\tphysical_ic\tnull_initial_data\tcertificate_sha256\tcontrols_sha256' \
  $'certificate_gate\t77\tafterok:66\t-\t-\t'"$campaign"$'\t'"$campaign/physical_ic.bin"$'\t'"$scratch/certified-null/ef_null_initial_data.bin"$'\t'"$fixture_certificate_sha"$'\t'"$fixture_controls_sha" \
  $'long_evolution\t101\tafterok:77\tschwarzschild\tany\t'"$campaign/schwarzschild"$'\t'"$campaign/physical_ic.bin"$'\t-\t'"$fixture_certificate_sha"$'\t'"$fixture_controls_sha" \
  $'long_evolution\t102\tafterok:77\teddington_finkelstein\tblack_hole\t'"$campaign/eddington_finkelstein"$'\t'"$campaign/physical_ic.bin"$'\t'"$scratch/certified-null/ef_null_initial_data.bin"$'\t'"$fixture_certificate_sha"$'\t'"$fixture_controls_sha" \
  > "$campaign/submission.tsv"

for artifact in "$campaign/submission.tsv" \
    "$campaign/campaign_controls.manifest.tsv"; do
  (
    cd "$(dirname "$artifact")"
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256"
  )
done

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'printf "%q " "$@" >> "$SBATCH_LOG"' \
  'printf "\\n" >> "$SBATCH_LOG"' \
  'printf "9001\\n"' \
  > "$scratch/bin/sbatch"
chmod +x "$scratch/bin/sbatch"
export PATH="$scratch/bin:$PATH"
export SBATCH_LOG="$scratch/sbatch.log"
: > "$SBATCH_LOG"

# A workflow-local module with a production dependency's name would shadow the
# frozen source on Python's search path.  Reject it before reaching Slurm.
shadow="$workflow/cpu/tools/compare_wpic_dynamic_coordinates.py"
printf '# forbidden shadow\n' > "$shadow"
if env \
    VENUS_SOURCE_ROOT="$scientific" \
    VENUS_PAIRED_LONG_WORKFLOW_ROOT="$workflow" \
    CAMPAIGN_ROOT="$campaign" \
    VENUS_PAIRED_LONG_SC_JOB_ID=101 \
    VENUS_PAIRED_LONG_EF_JOB_ID=102 \
    VENUS_EIKONAL_SUBSTEPS=8 \
    bash "$workflow/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh" \
    > "$scratch/shadow.out" 2>&1; then
  fail "visualization submitter accepted a workflow-local dependency shadow"
fi
grep -Eq 'shadows a frozen scientific dependency' "$scratch/shadow.out" \
  || fail "dependency-shadow rejection was not explicit"
[[ ! -s "$SBATCH_LOG" ]] || fail "dependency shadow reached sbatch"
rm "$shadow"

env \
  VENUS_SOURCE_ROOT="$scientific" \
  VENUS_PAIRED_LONG_WORKFLOW_ROOT="$workflow" \
  CAMPAIGN_ROOT="$campaign" \
  VENUS_PAIRED_LONG_SC_JOB_ID=101 \
  VENUS_PAIRED_LONG_EF_JOB_ID=102 \
  VENUS_EIKONAL_SUBSTEPS=8 \
  bash "$workflow/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh" \
  > "$scratch/submit.out"

[[ "$(wc -l < "$SBATCH_LOG" | tr -d ' ')" = 1 ]] \
  || fail "visualization submitter did not queue exactly one job"
grep -Eq -- '--dependency=afterok:101:102' "$SBATCH_LOG" \
  || fail "visualization job lacks afterok on both long jobs"
grep -Eq 'ias_einstein_vlasov_wpic_paired_long_gif.sbatch' "$SBATCH_LOG" \
  || fail "visualization submitter did not use its immutable wrapper"
[[ -s "$campaign/visualization_request.manifest.tsv" \
   && -s "$campaign/visualization_request.manifest.tsv.sha256" \
   && -s "$campaign/visualization_submission.tsv" \
   && -s "$campaign/visualization_submission.tsv.sha256" ]] \
  || fail "visualization request/receipt sidecars are incomplete"
grep -Eq $'dependency\tafterok:101:102' \
  "$campaign/visualization_request.manifest.tsv" \
  || fail "immutable request does not bind both prerequisite jobs"

# Reuse must fail before another scheduler mutation.
if env VENUS_SOURCE_ROOT="$scientific" \
    VENUS_PAIRED_LONG_WORKFLOW_ROOT="$workflow" CAMPAIGN_ROOT="$campaign" \
    VENUS_PAIRED_LONG_SC_JOB_ID=101 VENUS_PAIRED_LONG_EF_JOB_ID=102 \
    VENUS_EIKONAL_SUBSTEPS=8 \
    bash "$workflow/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh" \
    > "$scratch/reuse.out" 2>&1; then
  fail "visualization submitter reused an existing request"
fi
[[ "$(wc -l < "$SBATCH_LOG" | tr -d ' ')" = 1 ]] \
  || fail "reuse rejection reached sbatch"

sha() { sha256sum "$1" | awk '{print $1}'; }
env \
  SLURM_JOB_ID=9001 \
  VENUS_SOURCE_ROOT="$scientific" \
  VENUS_PAIRED_LONG_WORKFLOW_ROOT="$workflow" \
  CAMPAIGN_ROOT="$campaign" \
  VENUS_PAIRED_LONG_SC_JOB_ID=101 \
  VENUS_PAIRED_LONG_EF_JOB_ID=102 \
  VENUS_EIKONAL_SUBSTEPS=8 \
  VENUS_PAIRED_LONG_GIF_SUBMITTER_SHA256="$(sha "$workflow/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh")" \
  VENUS_PAIRED_LONG_GIF_WRAPPER_SHA256="$(sha "$workflow/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch")" \
  VENUS_PAIRED_LONG_GIF_RENDERER_SHA256="$(sha "$renderer")" \
  VENUS_PARITY_CERTIFIER_SHA256="$(sha "$scientific/cpu/tools/compare_wpic_dynamic_coordinates.py")" \
  VENUS_PROFILE_RENDERER_SHA256="$(sha "$scientific/cpu/tools/render_weighted_einstein_vlasov_pic_profile_gifs.py")" \
  VENUS_ATOMIC_GIF_HELPER_SHA256="$(sha "$scientific/cpu/tools/render_weighted_pic_pure_gamma_gifs.py")" \
  VENUS_PAIRED_LONG_SUBMISSION_SHA256="$(sha "$campaign/submission.tsv")" \
  VENUS_PAIRED_LONG_CONTROLS_SHA256="$(sha "$campaign/campaign_controls.manifest.tsv")" \
  VENUS_PAIRED_LONG_NULL_MANIFEST_SHA256="$(sha "$scratch/certified-null/ef_null_initial_data.bin.manifest.json")" \
  VENUS_PAIRED_LONG_GIF_REQUEST_SHA256="$(sha "$campaign/visualization_request.manifest.tsv")" \
  bash "$workflow/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch" \
  > "$scratch/render.out"

final="$campaign/visualization"
[[ -d "$final" && ! -e "$campaign/.visualization.tmp.9001" ]] \
  || fail "visualization directory was not atomically published"
for output in sc_ef_black_hole_profiles.gif \
    sc_ef_black_hole_profiles_initial.png \
    sc_ef_black_hole_profiles_middle.png \
    sc_ef_black_hole_profiles_final.png \
    sc_ef_black_hole_profiles_frames.csv \
    sc_ef_black_hole_profiles_provenance.json \
    artifacts.sha256 render_manifest.tsv render_manifest.tsv.sha256; do
  [[ -s "$final/$output" ]] || fail "published product is missing: $output"
done
(
  cd "$final"
  sha256sum -c artifacts.sha256 >/dev/null
)
python3 - "$final/sc_ef_black_hole_profiles_provenance.json" "$final" <<'PY'
import json
import pathlib
import sys
document = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
assert pathlib.Path(document["gif"]) == root / "sc_ef_black_hole_profiles.gif"
assert pathlib.Path(document["frame_ledger"]) == root / "sc_ef_black_hole_profiles_frames.csv"
PY

# A changed staged dependency must fail before touching the published result.
before="$(sha "$final/sc_ef_black_hole_profiles.gif")"
printf '# mutation\n' >> "$scientific/cpu/tools/compare_wpic_dynamic_coordinates.py"
if env \
    SLURM_JOB_ID=9002 VENUS_SOURCE_ROOT="$scientific" \
    VENUS_PAIRED_LONG_WORKFLOW_ROOT="$workflow" CAMPAIGN_ROOT="$campaign" \
    VENUS_PAIRED_LONG_SC_JOB_ID=101 VENUS_PAIRED_LONG_EF_JOB_ID=102 \
    VENUS_EIKONAL_SUBSTEPS=8 \
    VENUS_PAIRED_LONG_GIF_SUBMITTER_SHA256="$(sha "$workflow/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh")" \
    VENUS_PAIRED_LONG_GIF_WRAPPER_SHA256="$(sha "$workflow/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch")" \
    VENUS_PAIRED_LONG_GIF_RENDERER_SHA256="$(sha "$renderer")" \
    VENUS_PARITY_CERTIFIER_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
    VENUS_PROFILE_RENDERER_SHA256="$(sha "$scientific/cpu/tools/render_weighted_einstein_vlasov_pic_profile_gifs.py")" \
    VENUS_ATOMIC_GIF_HELPER_SHA256="$(sha "$scientific/cpu/tools/render_weighted_pic_pure_gamma_gifs.py")" \
    VENUS_PAIRED_LONG_SUBMISSION_SHA256="$(sha "$campaign/submission.tsv")" \
    VENUS_PAIRED_LONG_CONTROLS_SHA256="$(sha "$campaign/campaign_controls.manifest.tsv")" \
    VENUS_PAIRED_LONG_NULL_MANIFEST_SHA256="$(sha "$scratch/certified-null/ef_null_initial_data.bin.manifest.json")" \
    VENUS_PAIRED_LONG_GIF_REQUEST_SHA256="$(sha "$campaign/visualization_request.manifest.tsv")" \
    bash "$workflow/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch" \
    > "$scratch/mutation.out" 2>&1; then
  fail "visualization wrapper accepted a changed source dependency"
fi
grep -Eq 'changed after submission' "$scratch/mutation.out" \
  || fail "dependency-mutation rejection was not explicit"
[[ "$(sha "$final/sc_ef_black_hole_profiles.gif")" = "$before" ]] \
  || fail "failed rerun changed the published GIF"

echo "PASS: paired-long SC/EF GIF workflow"
