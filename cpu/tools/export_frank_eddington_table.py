#!/usr/bin/env python3
# Export the cored model's isotropic Eddington distribution as a two-column CSV.
# Inputs control the radial model, binding-energy nodes and inversion quadrature.
# Output relative energy and f(E) retain the shared constructor's code-unit normalization.

"""Export the Frank Core Eddington table used by the VENUS IC generator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generate_frank_core_cell_average_restart import frank_eddington_table


def main() -> None:
    """Tabulate the isotropic equilibrium distribution versus binding energy."""

    # Resolution controls cover the spherical mass model, binding-energy
    # samples, and quadrature used by the Eddington inversion.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r-min", type=float, default=1.0e-3)
    parser.add_argument("--model-r-max", type=float, default=1.0e6)
    parser.add_argument("--n-model", type=int, default=65536)
    parser.add_argument("--n-energy", type=int, default=8192)
    parser.add_argument("--n-eddington-quad", type=int, default=192)
    args = parser.parse_args()

    # The shared constructor solves the Eddington inversion for the Frank core;
    # the interpolator is the physical distribution f(E) consumed by IC tools.
    _, _, _, interpolator, diagnostics = frank_eddington_table(
        args.r_min,
        args.model_r_max,
        args.n_model,
        args.n_energy,
        args.n_eddington_quad,
    )

    # Export the native energy nodes and distribution values without further
    # smoothing, preserving the equilibrium represented by the inversion.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        # Write named columns for the same tabulated energy nodes used by the initial-data constructor.
        writer.writerow(["relative_energy", "distribution"])
        writer.writerows(zip(interpolator.x, interpolator(interpolator.x)))
    print(
        f"wrote {interpolator.x.size} Frank Eddington nodes to {args.output}; "
        f"psi_max={diagnostics['eddington_psi_max']:.17g}"
    )


if __name__ == "__main__":
    main()
