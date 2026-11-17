#!/usr/bin/env python3
"""Focused local tests for the diagnostic-only eikonal reprocessor."""

from __future__ import print_function

import ast
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT / "scripts" / "slurm"
    / "ias_einstein_vlasov_wpic_eikonal_reprocess.sbatch"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


FAKE_CERTIFIER = r'''#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def manifest_artifacts(manifest, manifest_path):
    root = os.path.dirname(os.path.abspath(manifest_path))
    result = {}
    for name, record in manifest.get("artifacts", {}).items():
        raw_path = record["path"]
        path = raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path)
        path = os.path.abspath(path)
        actual = file_sha256(path)
        result[name] = {
            "path": path,
            "expected_sha256": record["sha256"],
            "actual_sha256": actual,
            "verified": actual == record["sha256"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eikonal-substeps", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256-output", required=True)
    args, _unknown = parser.parse_known_args()
    error = 1.0e-6
    common_domain = {
        "definition": "maximal_contiguous_exterior_pre_schwarzschild_terminal",
        "candidate_events": 128,
        "compared_events": 128,
        "event_coverage": 1.0,
        "time_slices": 4,
        "complete_time_slices": 4,
        "time_slice_coverage": 1.0,
        "minimum_per_slice_radial_coverage": 1.0,
        "unexplained_holes": 0,
        "time_topology_gap_slices": 0,
        "disconnected_non_horizon_support_samples": 0,
        "first_schwarzschild_time": 0.0,
        "last_schwarzschild_time": 1.0,
        "terminal_reason": "end_of_outer_clock_overlap",
    }
    mapping = {
        "common_domain": common_domain,
        "forward_maximum_step_refinement_error": error,
        "inverse_maximum_step_refinement_error": error,
        "forward_all_attempted_maximum_step_refinement_error": error,
        "inverse_all_attempted_maximum_step_refinement_error": error,
        "maximum_composition_time_error": error,
        "maximum_composition_jacobian_error": error,
        "step_refinement_error_definition": (
            "max(abs(delta coordinate time),abs(delta log Jacobian)); "
            "no Richardson division without demonstrated asymptotic order"
        ),
        "forward_worst_step_refinement_event": {
            "tau": 0.5, "r": 2.0, "forward_step_refinement_error": error,
        },
        "inverse_worst_step_refinement_event": {
            "v": 0.75, "r": 2.0, "inverse_step_refinement_error": error,
        },
    }
    missing = (
        "single-resolution agreement has no independent quantified "
        "discretization/PIC convergence evidence"
    )
    result = {
        "schema": "venus_wpic_dynamic_coordinate_certificate_v2",
        "status": "inconclusive",
        "issues": [missing],
        "failed_fields": [],
        "distribution": {
            "mode": "strict", "evidence": [{"kind": "synthetic"}],
            "failed_fields": [],
        },
        "provenance": {
            "coordinate_map_manifest": {
                "artifacts": {"crossings": {"verified": True}}
            }
        },
        "mapping": mapping,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with open(args.output, "w") as stream:
        stream.write(rendered)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    with open(args.sha256_output, "w") as stream:
        stream.write("{}  {}\n".format(digest, os.path.basename(args.output)))
    return 2


if __name__ == "__main__":
    sys.exit(main())
'''


FAKE_TIME = r'''#!/bin/bash
set -u
if [ "$1" != -v ] || [ "$2" != -o ]; then
  echo "unexpected time-wrapper arguments" >&2
  exit 1
fi
resource_output="$3"
shift 3
printf 'Command being timed: synthetic comparator\nMaximum resident set size: 1024\n' \
  > "$resource_output"
set +e
"$@"
status=$?
set -e
exit "$status"
'''


class EikonalReprocessRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.campaign = self.root / "campaign"
        self.sweep = (
            self.campaign / "diagnostic_reprocessing" / "synthetic_subset"
        )
        self.source = self.sweep / "source"
        self.source.mkdir(parents=True)

        self.staged_runner = self.source / RUNNER.name
        shutil.copyfile(str(RUNNER), str(self.staged_runner))
        self.certifier = self.source / "compare_wpic_dynamic_coordinates.py"
        write(self.certifier, FAKE_CERTIFIER)
        self.time_wrapper = self.root / "fake_gnu_time.sh"
        write(self.time_wrapper, FAKE_TIME)
        self.time_wrapper.chmod(
            self.time_wrapper.stat().st_mode | stat.S_IXUSR
        )
        write(
            self.source / "source.sha256",
            "{}  {}\n{}  {}\n".format(
                sha256(self.staged_runner), self.staged_runner.name,
                sha256(self.certifier), self.certifier.name,
            ),
        )

        controls = self.campaign / "campaign_controls.manifest.tsv"
        write(controls, "schema\tvenus_v3_static_parity_campaign\n")
        write(
            self.campaign / "campaign_controls.manifest.sha256",
            "{}  {}\n".format(sha256(controls), controls.name),
        )
        write(self.campaign / "workflow_source.sha256", "synthetic source\n")
        write(self.campaign / "submission.tsv", "role\tjob_id\nsynthetic\t1\n")

        pair = self.campaign / "pairs" / "fine_a"
        sc = pair / "schwarzschild"
        ef = pair / "eddington_finkelstein"
        null_map = pair / "null_map"
        for directory in (sc, ef, null_map):
            directory.mkdir(parents=True)
        write(pair / "physical_ic.bin", "synthetic physical IC\n")
        for directory, coordinates in (
                (sc, "schwarzschild"),
                (ef, "eddington_finkelstein")):
            write(
                directory / "summary.json",
                json.dumps({"coordinates": coordinates}) + "\n",
            )
            write(
                directory / "high_compactness_profiles.csv",
                "step,time,r,a\n0,0,1,1\n",
            )
            write(directory / "source.sha256", "synthetic source hash\n")
            write(directory / "binary.sha256", "synthetic binary hash\n")
            write(directory / "resolved_command.sh", "synthetic command\n")
        write(
            ef / "initial_coordinate_map_residuals.csv",
            "r,a\n1,1\n",
        )

        null_data = null_map / "ef_null_initial_data.bin"
        crossings = null_map / "ef_null_initial_data.bin.crossings.csv"
        write(null_data, "synthetic null data\n")
        write(crossings, "id,r\n1,1\n")
        manifest = {
            "artifacts": {
                "null_data": {
                    "path": null_data.name,
                    "sha256": sha256(null_data),
                },
                "crossings": {
                    "path": crossings.name,
                    "sha256": sha256(crossings),
                },
            }
        }
        write(
            null_map / "ef_null_initial_data.bin.manifest.json",
            json.dumps(manifest, indent=2) + "\n",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def environment(self):
        value = os.environ.copy()
        # This subprocess emulates the local test harness. Its environment
        # stays local even when the unittest parent is a real batch job;
        # production's allocation guard is retained in the unchanged runner.
        value.pop("SLURM_JOB_ID", None)
        value.pop("SLURM_JOBID", None)
        value.update({
            "CAMPAIGN_ROOT": str(self.campaign),
            "SWEEP_ROOT": str(self.sweep),
            "PAIR": "fine_a",
            "FROZEN_CERTIFIER": str(self.certifier),
            "VENUS_EIKONAL_SUBSTEP_LIST": "8 16 32",
            "VENUS_EIKONAL_TIME_BINARY": str(self.time_wrapper),
            "VENUS_EIKONAL_REPROCESS_TEST_MODE": "1",
            "SLURM_CPUS_PER_TASK": "4",
        })
        return value

    def test_bash_and_embedded_python_are_compatible(self):
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        source = RUNNER.read_text()
        blocks = re.findall(
            r"<<'PY_[A-Z]+'\n(.*?)\nPY_[A-Z]+'?\n", source, re.DOTALL
        )
        self.assertGreaterEqual(len(blocks), 3)
        for block in blocks:
            try:
                ast.parse(block, feature_version=(3, 6))
            except TypeError:
                try:
                    ast.parse(block, feature_version=6)
                except TypeError:
                    ast.parse(block)

    def test_subset_dry_run_is_authenticated_and_non_overwriting(self):
        run = subprocess.run(
            ["bash", str(self.staged_runner)],
            cwd=str(self.root),
            env=self.environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

        pair_output = self.sweep / "fine_a"
        self.assertEqual(
            sorted(path.name for path in pair_output.glob("n[0-9][0-9][0-9]")),
            ["n008", "n016", "n032"],
        )
        summary = json.loads((pair_output / "sweep_summary.json").read_text())
        self.assertEqual(summary["requested_substeps"], [8, 16, 32])
        self.assertTrue(summary["highest_requested_level_clean"])
        self.assertFalse(summary["strict_two_level_gate_evaluable_in_this_root"])
        self.assertTrue((pair_output / "HIGHEST_REQUESTED_LEVEL.PASS").is_file())
        self.assertFalse((pair_output / "STRICT_SUBSTEP_GATE.PASS").exists())
        for tag in ("n008", "n016", "n032"):
            level = pair_output / tag
            self.assertTrue((level / "resource_usage.txt").is_file())
            self.assertTrue((level / "resolved_command.sh").is_file())
            check = subprocess.run(
                ["sha256sum", "-c", "bundle.sha256"],
                cwd=str(level),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

        repeated = subprocess.run(
            ["bash", str(self.staged_runner)],
            cwd=str(self.root),
            env=self.environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("Refusing to overwrite", repeated.stderr)

    def test_default_full_sweep_emits_strict_two_level_marker(self):
        environment = self.environment()
        del environment["VENUS_EIKONAL_SUBSTEP_LIST"]
        run = subprocess.run(
            ["bash", str(self.staged_runner)],
            cwd=str(self.root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        pair_output = self.sweep / "fine_a"
        summary = json.loads((pair_output / "sweep_summary.json").read_text())
        self.assertEqual(summary["requested_substeps"], [2, 4, 8, 16, 32, 64])
        self.assertTrue(summary["strict_two_level_gate_evaluable_in_this_root"])
        self.assertTrue(summary["strict_two_level_gate_pass"])
        marker = pair_output / "STRICT_SUBSTEP_GATE.PASS"
        self.assertTrue(marker.is_file())
        check = subprocess.run(
            ["sha256sum", "-c", marker.name],
            cwd=str(pair_output),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)


if __name__ == "__main__":
    unittest.main()
