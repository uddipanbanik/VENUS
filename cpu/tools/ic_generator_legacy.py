#!/usr/bin/env python3
# Standalone EVICRWL1 nodal-distribution generator for direct tools/ use.
# Input is an optional distribution(r,w,L) callable plus grid controls. Output stores little-endian grids and f.
# The solver later integrates f with its spherical momentum measure.

"""Generate binary f(r,w,L) initial data for the user-friendly EV solver.

The solver reads the EVICRWL1 format written here:

    magic[16] = "EVICRWL1\\0..."
    int32 version = 1
    int32 Nr, Nw, NL
    float64 r[Nr], w[Nw], L[NL]
    float64 f[Nr,Nw,NL]  # row-major: ((i * Nw) + j) * NL + k

Users can provide a Python file with a vectorized function

    def distribution(r, w, L):
        ...

where r, w, L are broadcast-ready NumPy arrays.  The function should return
nonnegative f values with shape broadcastable to (Nr, Nw, NL).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import struct
from pathlib import Path
from typing import Callable

import numpy as np


# Section: grid and user-function helpers.
# This standalone generator supplies the python.ev_solver.ic_generator interface for
# users who call scripts from tools/ directly.

def stretched_w_grid(w_max: float, stretch: float, center: float, n: int) -> np.ndarray:
    if stretch <= 0.0:
        return np.linspace(-w_max, w_max, n)
    # A sinh map concentrates radial-momentum nodes near the chosen center.
    u = np.linspace(-1.0, 1.0, n)
    denom = math.sinh(stretch)
    left = center - (center + w_max) * np.sinh(stretch * (-u)) / denom
    right = center + (w_max - center) * np.sinh(stretch * u) / denom
    return np.where(u < 0.0, left, right)


def load_distribution(path: Path) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    # Import the explicit user file and require the vectorized distribution interface.
    spec = importlib.util.spec_from_file_location("user_initial_distribution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    distribution = getattr(module, "distribution", None)
    if distribution is None or not callable(distribution):
        raise RuntimeError(f"{path} must define callable distribution(r, w, L)")
    return distribution


def default_distribution(r: np.ndarray, w: np.ndarray, L: np.ndarray) -> np.ndarray:
    R0 = 5.0
    dR = 1.0
    beta = 4.0
    amp = 1.0e-2
    w_in = -0.6
    shell = np.exp(-0.5 * ((r - R0) / dR) ** 2)
    norm = amp * np.sqrt(beta / (2.0 * np.pi)) * beta / (R0 * R0)
    # Use the shell-center relativistic energy to define the default shifted momentum seed.
    eps_shift = np.sqrt(1.0 + (w - w_in) ** 2 + (L / R0) ** 2)
    maxwellian_seed = norm * np.exp(-beta * (eps_shift - 1.0))
    return shell * maxwellian_seed


# Section: EVICRWL1 writer.

def write_ev_ic(path: Path, r: np.ndarray, w: np.ndarray, L: np.ndarray, f: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert the distribution to the fixed on-disk float64 representation before validating it.
    f = np.asarray(f, dtype="<f8")
    expected = (r.size, w.size, L.size)
    if f.shape != expected:
        try:
            f = np.broadcast_to(f, expected).copy()
        except ValueError as exc:
            raise ValueError(f"distribution shape {f.shape} is not broadcastable to {expected}") from exc
    if np.any(~np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("distribution must be finite and nonnegative everywhere")
    with path.open("wb") as handle:
        magic = b"EVICRWL1"
        handle.write(magic + b"\0" * (16 - len(magic)))
        # Write dimensions before r,w,L and the C-ordered f payload so the reader can reconstruct their shapes.
        handle.write(struct.pack("<iiii", 1, r.size, w.size, L.size))
        handle.write(np.asarray(r, dtype="<f8").tobytes(order="C"))
        handle.write(np.asarray(w, dtype="<f8").tobytes(order="C"))
        handle.write(np.asarray(L, dtype="<f8").tobytes(order="C"))
        handle.write(np.asarray(f, dtype="<f8").tobytes(order="C"))


# Section: command-line interface and initial-data generation.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("initial_data/example_shell_rwl.bin"))
    parser.add_argument("--function-file", type=Path, help="Python file defining distribution(r, w, L)")
    parser.add_argument("--Nr", type=int, default=96)
    parser.add_argument("--Nw", type=int, default=96)
    parser.add_argument("--NL", type=int, default=32)
    parser.add_argument("--r-min", type=float, default=1.0e-3)
    parser.add_argument("--r-max", type=float, default=20.0)
    parser.add_argument("--radial-grid", choices=("log", "linear"), default="linear")
    parser.add_argument("--w-max", type=float, default=4.0)
    parser.add_argument("--w-grid", choices=("stretched", "linear"), default="stretched")
    parser.add_argument("--w-stretch", type=float, default=2.0)
    parser.add_argument("--w-center", type=float, default=0.0)
    parser.add_argument("--L-max", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.Nr < 3 or args.Nw < 3 or args.NL < 2:
        raise ValueError("Require Nr >= 3, Nw >= 3, and NL >= 2")
    if args.r_min <= 0.0 or args.r_max <= args.r_min:
        raise ValueError("Require 0 < r_min < r_max")
    if args.w_max <= 0.0 or args.L_max <= 0.0:
        raise ValueError("Require w_max > 0 and L_max > 0")
    if not (-args.w_max < args.w_center < args.w_max):
        raise ValueError("Require -w_max < w_center < w_max")

    if args.radial_grid == "log":
        r = np.geomspace(args.r_min, args.r_max, args.Nr)
    else:
        r = np.linspace(args.r_min, args.r_max, args.Nr)
    if args.w_grid == "linear":
        w = np.linspace(-args.w_max, args.w_max, args.Nw)
    else:
        w = stretched_w_grid(args.w_max, args.w_stretch, args.w_center, args.Nw)
    L = np.linspace(0.0, args.L_max, args.NL)

    distribution = load_distribution(args.function_file) if args.function_file else default_distribution
    # Give each coordinate a separate broadcast axis before evaluating the full (r,w,L) array.
    rr = r[:, None, None]
    ww = w[None, :, None]
    LL = L[None, None, :]
    f = distribution(rr, ww, LL)
    write_ev_ic(args.output, r, w, L, f)
    print(f"Wrote {args.output.resolve()} with shape ({r.size}, {w.size}, {L.size})")


if __name__ == "__main__":
    main()
