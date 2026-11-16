# Serialize reference-solver profiles, phase projections, restarts and final arrays.
# CSV columns use solver code units. EVLOGRST1 stores little-endian metadata and nodal f.
# Restart compatibility is checked before interpreting the large distribution array.

"""Output and restart helpers for the Python Einstein--Vlasov reference solver.

This module owns all file-format details for the Python solver layer:
CSV profile blocks, phase-space projections, strided phase snapshots, binary
restart files, metadata, and the compressed final-state archive.  Keeping these
helpers here lets the timestep code focus on numerical evolution rather than
path and serialization details.
"""

from __future__ import annotations

import csv
import json
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .grids import radial_grid_code, radial_grid_name, using_binary_initial_data
from .params import EVParams


# Section: CSV profile and phase-space writers.
# These routines write human-readable diagnostics consumed by the plotting
# scripts.  They deliberately avoid hidden unit conversions: columns correspond
# directly to solver arrays and profile names.

def write_profile_rows(
    writer: csv.writer,
    t: float,
    step: int,
    r: np.ndarray,
    rho: np.ndarray,
    p: np.ndarray,
    j: np.ndarray,
    metric: dict[str, np.ndarray | float],
) -> None:
    """Write one saved radial profile block to an already-open CSV writer."""
    m = metric["m"]
    mu = metric["mu"]
    lam = metric["lambda"]
    gap = metric["gap"]
    assert isinstance(m, np.ndarray)
    assert isinstance(mu, np.ndarray)
    assert isinstance(lam, np.ndarray)
    assert isinstance(gap, np.ndarray)
    # One row joins matter and metric values at one radius and one accepted time.
    for row in zip(r, rho, p, j, m, mu, lam, gap):
        writer.writerow([f"{t:.17e}", step, *[f"{float(x):.17e}" for x in row]])


def write_phase_space(path: Path, r: np.ndarray, w: np.ndarray, g: np.ndarray) -> None:
    """Write a complete fixed-time g(r,w)=int dL L f projection."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["r", "w", "g"])
        for i, ri in enumerate(r):
            for j, wj in enumerate(w):
                writer.writerow([f"{ri:.17e}", f"{wj:.17e}", f"{g[i, j]:.17e}"])


def write_phase_snapshot_rows(
    writer: csv.writer,
    t: float,
    step: int,
    r: np.ndarray,
    w: np.ndarray,
    g: np.ndarray,
    par: EVParams,
) -> None:
    """Append one strided phase-space snapshot to an open snapshot CSV writer."""
    # Save explicit grid indices as well as coordinates so readers can detect incomplete blocks.
    for i in range(0, len(r), par.phase_r_stride):
        for j_idx in range(0, len(w), par.phase_w_stride):
            writer.writerow(
                [
                    f"{t:.17e}",
                    step,
                    i,
                    j_idx,
                    f"{r[i]:.17e}",
                    f"{w[j_idx]:.17e}",
                    f"{g[i, j_idx]:.17e}",
                ]
            )


def phase_space_projection(f: np.ndarray, L: np.ndarray, L_weights: np.ndarray) -> np.ndarray:
    """Integrate L f over L to form the two-dimensional g(r,w) diagnostic."""
    return np.einsum("ijk,k,k->ij", f, L, L_weights, optimize=True)


# Section: Binary restart format.
# The EVLOGRST1 restart stores enough grid metadata to reject incompatible
# restarts before the large f array is reshaped.  Version 3 includes the
# piecewise-linear grid controls used to reconstruct the saved mesh.

def _close_enough(a: float, b: float, rel: float = 1.0e-12) -> bool:
    """Compare restart metadata values with a relative tolerance."""
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def _validate_restart_value(name: str, got: float, expected: float) -> None:
    """Raise a restart-grid mismatch error with the offending field name."""
    if not _close_enough(got, expected):
        raise ValueError(f"Restart grid mismatch for {name}")


def write_restart_state(path: Path, par: EVParams, t: float, step: int, f: np.ndarray) -> None:
    """Write the full Python restart state in the EVLOGRST1 binary format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a complete temporary restart, then replace the destination after the stream closes.
    tmp = path.with_name(path.name + ".tmp")
    f64 = np.ascontiguousarray(f, dtype="<f8")
    with tmp.open("wb") as handle:
        handle.write(b"EVLOGRST1\0\0\0\0\0\0\0")
        handle.write(struct.pack("<i", 3))
        handle.write(struct.pack("<iii", par.Nr, par.Nw, par.NL))
        handle.write(struct.pack("<i", radial_grid_code(par.radial_grid)))
        handle.write(
            struct.pack(
                "<dddidddddd",
                par.r_min,
                par.r_max,
                par.inner_refine_radius,
                int(par.inner_refine_cells),
                par.w_max,
                par.L_max,
                par.w_stretch,
                par.w_center,
                par.central_mass,
                t,
            )
        )
        # The accepted step/time and array size describe precisely the f payload written next.
        handle.write(struct.pack("<q", int(step)))
        handle.write(struct.pack("<Q", int(f64.size)))
        handle.write(f64.tobytes(order="C"))
    tmp.replace(path)


def read_restart_state(path: Path, par: EVParams) -> tuple[float, int, np.ndarray]:
    """Read and validate a Python restart state against requested parameters."""
    with path.open("rb") as handle:
        magic = handle.read(16)
        if magic.rstrip(b"\0") != b"EVLOGRST1":
            raise ValueError("Restart file has wrong magic header")
        version = struct.unpack("<i", handle.read(4))[0]
        if version not in (1, 2, 3):
            raise ValueError("Unsupported restart file version")
        nr, nw, nl = struct.unpack("<iii", handle.read(12))
        if (nr, nw, nl) != (par.Nr, par.Nw, par.NL):
            raise ValueError("Restart file dimensions do not match requested grid")
        saved_radial_grid = radial_grid_name(struct.unpack("<i", handle.read(4))[0]) if version >= 2 else "log"
        if saved_radial_grid != par.radial_grid:
            raise ValueError("Restart radial grid does not match requested radial grid")
        # Version 3 carries the two-segment mesh controls. Earlier versions use the older fixed header.
        if version >= 3:
            (
                r_min,
                r_max,
                inner_refine_radius,
                inner_refine_cells,
                w_max,
                L_max,
                w_stretch,
                w_center,
                central_mass,
                t,
            ) = struct.unpack("<dddidddddd", handle.read(76))
        else:
            (
                r_min,
                r_max,
                w_max,
                L_max,
                w_stretch,
                w_center,
                central_mass,
                t,
            ) = struct.unpack("<dddddddd", handle.read(64))
        _validate_restart_value("r_min", r_min, par.r_min)
        _validate_restart_value("r_max", r_max, par.r_max)
        if version >= 3:
            _validate_restart_value("inner_refine_radius", inner_refine_radius, par.inner_refine_radius)
            if int(inner_refine_cells) != int(par.inner_refine_cells):
                raise ValueError("Restart grid mismatch for inner_refine_cells")
        _validate_restart_value("w_max", w_max, par.w_max)
        _validate_restart_value("L_max", L_max, par.L_max)
        _validate_restart_value("w_stretch", w_stretch, par.w_stretch)
        _validate_restart_value("w_center", w_center, par.w_center)
        _validate_restart_value("central_mass", central_mass, par.central_mass)
        step = struct.unpack("<q", handle.read(8))[0]
        n_values = struct.unpack("<Q", handle.read(8))[0]
        # Require the saved element count and byte length to agree with the requested three-dimensional grid.
        expected_values = par.Nr * par.Nw * par.NL
        if n_values != expected_values:
            raise ValueError("Restart state array has wrong size")
        raw = handle.read(n_values * 8)
        if len(raw) != n_values * 8:
            raise ValueError("Unexpected end of restart state array")
    f = np.frombuffer(raw, dtype="<f8").copy().reshape((par.Nr, par.Nw, par.NL))
    return float(t), int(step), f


# Section: Complete run-output writer.
# The Python solver returns in-memory arrays; this routine translates that
# result dictionary into the same directory layout expected by the readers and
# plotting helpers.

def write_outputs(result: dict[str, object], outdir: Path) -> None:
    """Write metadata, profiles, history, phase projections, restart, and final npz."""
    outdir.mkdir(parents=True, exist_ok=True)
    par = result["params"]
    assert isinstance(par, EVParams)
    r = np.asarray(result["r"])
    w = np.asarray(result["w"])
    final_time = float(result["final_time"])
    final_step = int(result["final_step"])

    with (outdir / "metadata.json").open("w") as handle:
        metadata = asdict(par)
        metadata["radial_grid"] = par.radial_grid
        metadata["initial_df"] = (
            "binary_f_of_r_w_L"
            if using_binary_initial_data(par)
            else (
                "shell_boundary_seed_F0_with_R0_for_zeta_b"
                if par.shell_initial_df
                else "boundary_seed_F0_with_r_for_zeta"
            )
        )
        metadata["shell_radial_mode"] = (
            "density_tophat"
            if par.top_hat_density_shell
            else (
                "density_gaussian"
                if par.gaussian_density_shell
                else ("f_tophat" if par.top_hat_shell else "f_gaussian")
            )
        )
        metadata["time_integrator"] = "semi_lagrangian_kick_drift_kick_metric_frozen"
        metadata["time_integrator_order"] = par.time_integrator_order
        metadata["time_integrator_substeps"] = par.time_integrator_substeps
        metadata[
            "inner_boundary_condition"
        ] = "mirror_r_about_r_min_and_w_about_zero_with_boundary_slice_symmetrization"
        metadata["phase_projection"] = "g(r,w)=int_dL_L_f"
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Write each saved epoch as a radial block with an explicit time and step in every row.
    with (outdir / "profiles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "step", "r", "rho", "p", "j", "m", "mu", "lambda", "gap"])
        for snap in result["snapshots"]:  # type: ignore[union-attr]
            assert isinstance(snap, dict)
            metric = {
                "m": np.asarray(snap["m"]),
                "mu": np.asarray(snap["mu"]),
                "lambda": np.asarray(snap["lambda"]),
                "gap": np.asarray(snap["gap"]),
            }
            write_profile_rows(
                writer,
                float(snap["time"]),
                int(snap["step"]),
                r,
                np.asarray(snap["rho"]),
                np.asarray(snap["p"]),
                np.asarray(snap["j"]),
                metric,
            )

    with (outdir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "time",
                "dt",
                "matter_mass",
                "max_rho",
                "min_gap",
                "last_outer_escape_est",
                "last_w_low_escape_est",
                "last_w_high_escape_est",
                "cum_outer_escape_est",
                "cum_w_low_escape_est",
                "cum_w_high_escape_est",
                "last_outer_cells",
                "last_w_low_cells",
                "last_w_high_cells",
            ],
        )
        writer.writeheader()
        for row in result["history"]:  # type: ignore[union-attr]
            writer.writerow(row)

    write_phase_space(outdir / "phase_space_initial.csv", r, w, np.asarray(result["f_initial_projection"]))
    write_phase_space(outdir / "phase_space_final.csv", r, w, np.asarray(result["f_final_projection"]))
    write_restart_state(outdir / "restart_state.bin", par, final_time, final_step, np.asarray(result["f"]))

    # Preserve complete final arrays for Python analysis alongside the portable binary restart.
    np.savez_compressed(
        outdir / "final_state.npz",
        r=r,
        w=w,
        L=np.asarray(result["L"]),
        f=np.asarray(result["f"]),
        time=final_time,
        step=final_step,
        f_initial_projection=np.asarray(result["f_initial_projection"]),
        f_final_projection=np.asarray(result["f_final_projection"]),
    )
