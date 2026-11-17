#!/usr/bin/env bash
# End-to-end regression for the expanding WPIC transactional collision clock.
#
# The final 1e-7 interval emulates an output/deadline cleanup step: its
# particle-deposition residual must not manufacture t_phi -> 0.  The second
# half checks that a restart preserves collision age, clock estimates, marker
# state, and collision RNG closely enough to reproduce the final checkpoint.

set -euo pipefail
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: $0 PATH_TO_EXPANDING_WPIC_DRIVER" >&2
    exit 2
fi
driver=$1
work=$(mktemp -d "${TMPDIR:-/tmp}/venus-wpic-cadence.XXXXXX")
trap 'rm -rf "$work"' EXIT

common=(
    --gamma 2.5
    --maximum-timestep 0.005
    --dynamical-fraction 0.04
    --output-interval 0.005
    --history-every 1
    --restart-every 1
    --particles-per-radial-cell 128
    --radial-cells 16
    --radial-minimum 0.1
    --radial-maximum 10
    --subhalo-collision-strength 0.3
    --subhalo-collision-interval 0.005
    --subhalo-collision-timescale-fraction 0.1
    --subhalo-collision-cfl 0.2
    --subhalo-collision-maximum-substeps 1024
    --subhalo-inner-collision-cells 2
    --threads 2
)

"$driver" "${common[@]}" \
    --end-time 0.0100001 \
    --output-directory "$work/full" \
    >"$work/full.log" 2>"$work/full.err"

"$driver" "${common[@]}" \
    --end-time 0.005 \
    --output-directory "$work/segment" \
    >"$work/segment.log" 2>"$work/segment.err"

"$driver" "${common[@]}" \
    --end-time 0.0100001 \
    --restart "$work/segment/restart_state.bin" \
    --output-directory "$work/resume" \
    >"$work/resume.log" 2>"$work/resume.err"

cmp "$work/full/restart_state.bin" "$work/resume/restart_state.bin"

# Optional accepted-step boundary measurements preserve the exact trajectory.
"$driver" "${common[@]}" --boundary-diagnostics \
    --end-time 0.0100001 --output-directory "$work/measured" \
    >"$work/measured.log" 2>"$work/measured.err"
cmp "$work/full/restart_state.bin" "$work/measured/restart_state.bin"

# A descriptive native-input label records the actual starting particles.
"$driver" "${common[@]}" --initial-condition-label gaussian_shell \
    --end-time 0.0100001 --restart "$work/segment/restart_state.bin" \
    --output-directory "$work/labeled" >"$work/labeled.log" 2>"$work/labeled.err"
cmp "$work/full/restart_state.bin" "$work/labeled/restart_state.bin"
python3 - "$work" <<'PY'
from pathlib import Path
import csv,hashlib,json,math,struct,sys
root=Path(sys.argv[1])
for name in ['profile_00000.csv','profile_00001.csv','profile_00002.csv','profile_00003.csv']:
    if (root/'full'/name).exists():
        assert (root/'full'/name).read_bytes()==(root/'measured'/name).read_bytes()
def physical_rows(path):
    rows=list(csv.DictReader(path.open()))
    return [{k:v for k,v in row.items() if k!='wall_seconds'} for row in rows]
assert physical_rows(root/'full/history.csv')==physical_rows(root/'measured/history.csv')
assert not (root/'full/boundary_history.csv').exists()
boundary=physical_rows(root/'measured/boundary_history.csv')
assert float(boundary[-1]['time'])==0.0100001
assert all(int(row['outer_wall_impacts'])==0 for row in boundary)
raw=(root/'segment/restart_state.bin').read_bytes()
n=struct.unpack_from('<Q',raw,24)[0]
mass=math.fsum(struct.unpack_from('<d',raw,112+81*i)[0] for i in range(n))
metadata=json.loads((root/'labeled/metadata.json').read_text())
assert metadata['gamma_i'] is None and metadata['initial_condition_label']=='gaussian_shell'
assert metadata['native_input_sha256']==hashlib.sha256(raw).hexdigest()
assert abs(metadata['marker_mass_at_start']-mass)<1e-14
assert 'initial_origin_core_mass' not in metadata
print('Boundary diagnostics and native labels preserve particles/profiles/physical history.')
PY

python3 - \
    "$work/full/collision_history.csv" \
    "$work/resume/collision_history.csv" <<'PY'
import csv
import math
import pathlib
import sys

for name in sys.argv[1:]:
    path = pathlib.Path(name)
    rows = list(csv.DictReader(path.open(newline="")))
    assert rows, f"empty collision history: {path}"
    ratios = [
        float(row["max_interval_over_selection_timescale"])
        for row in rows
    ]
    assert all(math.isfinite(value) and value >= 0.0 for value in ratios)
    assert max(ratios) <= 0.1 * (1.0 + 2.0e-12), (
        path,
        max(ratios),
    )
PY

# A valid continuation in the same directory preserves earlier diagnostics.
# A stale checkpoint must be rejected before metadata, profiles or histories
# are changed, including the alternative collision-history formats.
python3 - "$driver" "$work" "${common[@]}" <<'PY'
import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

driver, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
common = sys.argv[3:]
def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}
def continue_to(directory, checkpoint):
    return subprocess.run([str(driver), *common, '--end-time', '0.0100001',
                           '--restart', str(checkpoint), '--output-directory', str(directory)],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

same = root/'same_directory'
shutil.copytree(root/'segment', same)
prior_profiles = {p.name:p.read_bytes() for p in same.glob('profile_*.csv')}
prior_history = (same/'history.csv').read_bytes()
completed = continue_to(same, same/'restart_state.bin')
assert completed.returncode == 0, completed.stderr
assert (same/'restart_state.bin').read_bytes() == (root/'full/restart_state.bin').read_bytes()
assert (same/'history.csv').read_bytes().startswith(prior_history)
for name, data in prior_profiles.items():
    assert (same/name).read_bytes() == data, name

histories = ['history.csv', 'virial_budget.csv', 'collision_history.csv',
             'collision_history_adaptive.csv', 'collision_history_population_control.csv',
             'boundary_history.csv']
for index, name in enumerate([*histories, 'profile_00001.csv']):
    output = root/f'stale_{index}'
    shutil.copytree(root/'segment', output)
    path = output/name
    if not path.exists():
        shutil.copy2(output/'collision_history.csv', path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    assert rows, name
    # Changing the final row also checks profiles beyond their first row.
    rows[-1]['time'] = '0.006'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    before = hashes(output)
    rejected = continue_to(output, root/'segment/restart_state.bin')
    assert rejected.returncode != 0, f'stale output accepted: {name}'
    assert 'later than restart time' in rejected.stderr, (name, rejected.stderr)
    assert hashes(output) == before, f'rejected restart modified output: {name}'
print('Same-directory continuation and seven stale-output rejection cases passed.')
PY

echo "expanding WPIC transactional-cadence and output-continuation integration test passed"
