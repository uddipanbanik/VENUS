#!/usr/bin/env python3
"""Host-only fail-closed CLI checks for the GPU task frontends.

Every rejected probe fails during parsing, before CUDA device discovery.  The
test can therefore run on a login node after a CUDA-capable build even when no
GPU is visible.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require_help(binary: str, present: tuple[str, ...], absent: tuple[str, ...]) -> None:
    result = run(binary, "--help")
    if result.returncode != 0:
        raise AssertionError(f"{binary} --help failed: {result.stderr}")
    for token in present:
        if token not in result.stdout:
            raise AssertionError(f"{binary} help omitted {token!r}")
    for token in absent:
        if token in result.stdout:
            raise AssertionError(f"{binary} help advertised forbidden {token!r}")


def require_rejection(binary: str, message: str, *arguments: str) -> None:
    result = run(binary, *arguments)
    combined = result.stdout + result.stderr
    if result.returncode == 0:
        raise AssertionError(f"{binary} unexpectedly accepted {arguments!r}")
    if message not in combined:
        raise AssertionError(
            f"{binary} rejected for the wrong reason; expected {message!r}:\n{combined}"
        )


def main() -> int:
    if len(sys.argv) not in (5, 6):
        raise SystemExit("usage: test_production_driver_cli.py VP SUBHALO SIDM SIDM_CONSTANT [STATIC_VP]")
    vp, subhalo, sidm, sidm_constant = sys.argv[1:5]

    require_help(
        vp,
        ("vlasov_poisson", "--universe MODE", "density contrast"),
        ("--subhalo-", "--sidm-"),
    )
    require_help(
        subhalo,
        ("subhalo_boltzmann_poisson", "--subhalo-collision-strength"),
        ("--sidm-", "--cross-section-model"),
    )
    require_help(
        sidm,
        ("static-WPIC", "--sidm-collision-strength", "--cross-section-model"),
        ("--subhalo-", "density contrast"),
    )
    require_help(
        sidm_constant,
        ("pinned to constant", "--sidm-sigma0"),
        ("--cross-section-model", "--sidm-sigma-power", "--sidm-yukawa-"),
    )

    require_rejection(
        vp,
        "rejects all collision options",
        "--subhalo-collision-strength",
        "1",
    )
    require_rejection(
        vp,
        "Invalid --universe",
        "--universe",
        "invalid",
    )
    require_rejection(
        subhalo,
        "requires an explicit positive",
        "--subhalo-collision-strength",
        "0",
    )
    require_rejection(
        sidm,
        "requires --cross-section-model",
        "--sidm-collision-strength",
        "1",
        "--sidm-sigma0",
        "1",
    )
    require_rejection(
        sidm_constant,
        "Pinned SIDM executable requires model constant",
        "--sidm-collision-strength",
        "1",
        "--cross-section-model",
        "yukawa_born",
        "--sidm-yukawa-alpha",
        "0.01",
    )
    require_rejection(
        sidm_constant,
        "locked to --universe static",
        "--universe",
        "eds",
        "--sidm-collision-strength",
        "1",
        "--sidm-sigma0",
        "1",
    )
    require_rejection(
        sidm_constant,
        "rejects parameters from a different cross-section model",
        "--sidm-collision-strength",
        "1",
        "--sidm-sigma0",
        "1",
        "--sidm-yukawa-alpha",
        "0.02",
    )

    # Output metadata is checked before CUDA discovery.  A generic frontend
    # must therefore refuse to append constant-cross-section output to a run
    # whose existing metadata names the Yukawa--Born operator.
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "metadata.json").write_text(
            """{
  "physics_task": "sidm_boltzmann_poisson",
  "universe": "static",
  "poisson_source": "total_density",
  "collision_model": "material_sidm_yukawa_born"
}\n""",
            encoding="utf-8",
        )
        require_rejection(
            sidm,
            "different universe, Poisson source, or collision model",
            "--restart",
            "unused.restart",
            "--output-directory",
            directory,
            "--end-time",
            "1",
            "--sidm-collision-strength",
            "1",
            "--cross-section-model",
            "constant",
            "--sidm-sigma0",
            "1",
        )
    if len(sys.argv) == 6:
        static = sys.argv[5]
        require_help(static, ("Static Vlasov-Poisson", "H=0"), ())
        require_rejection(static, "requires --universe static", "--universe", "eds")
        require_rejection(static, "requires --universe static", "--universe", "static", "--universe", "eds")
        require_rejection(static, "rejects all collision options", "--sidm-collision-strength", "1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
