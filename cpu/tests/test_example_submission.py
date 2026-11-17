"""Check scheduler/simulation option boundaries with an isolated scheduler stub."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('scheduler,simulation', [
    ([], []),
    (['--partition', 'cpu'], []),
    ([], ['--output', 'a run directory']),
    (['--partition', 'cpu'], ['--output', 'a run directory', '--end-time', '0.04']),
])
def test_example_submission_options(tmp_path, scheduler, simulation):
    receipt = tmp_path / 'submission.json'
    stub = tmp_path / 'sbatch'
    stub.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'from pathlib import Path\n'
                    'Path(os.environ["VENUS_SUBMISSION_RECEIPT"]).write_text('
                    'json.dumps({"args":sys.argv[1:],"root":os.environ["VENUS_ROOT"]}))\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               VENUS_SUBMISSION_RECEIPT=str(receipt))
    args = ['bash', str(ROOT / 'scripts/slurm/submit.sh'), 'vp_wpic_static'] + scheduler
    if simulation:
        args += ['--'] + simulation
    subprocess.run(args, env=env, cwd=tmp_path, check=True)
    result = json.loads(receipt.read_text())
    assert result['args'] == ['--export=ALL'] + scheduler + [
        str(ROOT / 'examples/slurm/vp_wpic_static.sbatch')] + simulation
    assert result['root'] == str(ROOT)


def test_example_submission_rejects_path(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/slurm/submit.sh'), '../vp_wpic_static'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Invalid example name' in result.stderr
