#!/usr/bin/env python3
# Compare initial/final cell-average distribution restarts and radial moment profiles.
# The supported log-r, power-law-eta grid fixes the canonical spherical measure 8*pi^2*dr*dw*deta.
# Output JSON distinguishes visible distribution drift, hidden-carrier mass and observable profile drift.

"""Measure phase-space and profile drift in a collisionless equilibrium run."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


# ---- Decode the native finite-volume/semi-Lagrangian restart ----------------
# The binary header fixes the phase-space geometry; the payload contains the
# reduced distribution f(r,w,eta) and, in newer versions, delayed carriers and
# angular modal moments needed to account for all represented matter.
def read_value(handle, code: str):
    size = struct.calcsize(code)
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("Unexpected end of restart header")
    return struct.unpack("<" + code, payload)[0]


def read_restart(path: Path) -> tuple[dict[str, float | int], np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).split(b"\0", 1)[0]
        if magic != b"EVLOGRST1":
            raise ValueError(f"{path} is not an EVLOGRST1 restart")
        version = read_value(handle, "i")
        if version not in (6, 7, 8):
            raise ValueError(f"Expected restart version 6, 7, or 8, found {version}")
        nr = read_value(handle, "i")
        nw = read_value(handle, "i")
        nl = read_value(handle, "i")
        radial_grid_code = read_value(handle, "i")
        kinetic_grid_code = read_value(handle, "i")
        eta_grid_code = read_value(handle, "i")
        eta_power = read_value(handle, "d")
        r_min = read_value(handle, "d")
        r_max = read_value(handle, "d")
        inner_refine_radius = read_value(handle, "d")
        inner_refine_cells = read_value(handle, "i")
        w_max = read_value(handle, "d")
        l_max = read_value(handle, "d")
        w_stretch = read_value(handle, "d")
        w_center = read_value(handle, "d")
        central_mass = read_value(handle, "d")
        time = read_value(handle, "d")
        step = read_value(handle, "q")
        has_tracer = read_value(handle, "i")
        has_carrier = read_value(handle, "i") if version >= 7 else 0
        has_modes = read_value(handle, "i") if version >= 8 else 0
        has_tracer_modes = read_value(handle, "i") if version >= 8 else 0
        count = read_value(handle, "Q")
        expected = nr * nw * nl
        if count != expected:
            raise ValueError(f"Restart has {count} values; expected {expected}")
        values = np.fromfile(handle, dtype="<f8", count=count)
        if values.size != expected:
            raise ValueError("Restart distribution is truncated")
        if has_tracer:
            tracer_count = read_value(handle, "Q")
            handle.seek(8 * tracer_count, 1)
        carrier_packets: list[tuple[float, int, float, float]] = []
        # Read delayed carrier packets separately from the resolved cell-average distribution.
        if has_carrier:
            packet_count = read_value(handle, "Q")
            for _ in range(packet_count):
                carrier_packets.append((
                    read_value(handle, "d"),
                    read_value(handle, "i"),
                    read_value(handle, "d"),
                    read_value(handle, "d"),
                ))
            if has_tracer:
                tracer_packet_count = read_value(handle, "Q")
                handle.seek((8 + 4 + 8 + 8) * tracer_packet_count, 1)
        if has_modes:
            mode_cells = read_value(handle, "Q")
            if mode_cells != expected:
                raise ValueError(
                    f"Restart has P2 modes for {mode_cells} cells; expected {expected}"
                )
            handle.seek(5 * 8 * mode_cells, 1)
        if has_tracer_modes:
            if not has_tracer:
                raise ValueError("Restart contains tracer P2 modes without a tracer")
            tracer_mode_cells = read_value(handle, "Q")
            if tracer_mode_cells != expected:
                raise ValueError(
                    "Restart tracer P2-mode array has the wrong cell count"
                )
            handle.seek(5 * 8 * tracer_mode_cells, 1)
        # A complete known-format payload must end after all declared optional arrays.
        if handle.read(1):
            raise ValueError("Restart has an unexpected trailing payload")
    header = {
        "version": version,
        "Nr": nr,
        "Nw": nw,
        "NL": nl,
        "radial_grid_code": radial_grid_code,
        "kinetic_grid_code": kinetic_grid_code,
        "eta_grid_code": eta_grid_code,
        "eta_power": eta_power,
        "r_min": r_min,
        "r_max": r_max,
        "inner_refine_radius": inner_refine_radius,
        "inner_refine_cells": inner_refine_cells,
        "w_max": w_max,
        "L_max": l_max,
        "w_stretch": w_stretch,
        "w_center": w_center,
        "central_mass": central_mass,
        "time": time,
        "step": step,
        "has_tracer": has_tracer,
        "has_carrier": has_carrier,
        "has_modes": has_modes,
        "has_tracer_modes": has_tracer_modes,
        "carrier_packet_count": len(carrier_packets),
    }
    if carrier_packets:
        fraction = np.linspace(0.0, 1.0, nl + 1)
        eta_edge = 0.5 * l_max**2 * fraction**eta_power
        # Restore each carrier's angular-cell width and spherical phase factor when computing its represented mass.
        header["hidden_carrier_mass"] = float(8.0 * np.pi**2 * sum(
            (eta_edge[index + 1] - eta_edge[index]) * reduced_mass
            for reduced_mass, index, _, _ in carrier_packets
        ))
    else:
        header["hidden_carrier_mass"] = 0.0
    return header, values.reshape(nr, nw, nl)


# ---- Reconstruct the canonical phase-space measure --------------------------
# Matching grids are required because the drift norms compare the same physical
# phase-space cells.  The measure dr dw d(eta), multiplied by 8 pi^2, converts
# the reduced distribution into mass.
def require_matching_grids(first: dict, second: dict) -> None:
    keys = (
        "Nr",
        "Nw",
        "NL",
        "radial_grid_code",
        "kinetic_grid_code",
        "eta_grid_code",
        "eta_power",
        "r_min",
        "r_max",
        "w_max",
        "L_max",
        "w_stretch",
        "w_center",
    )
    mismatched = [key for key in keys if first[key] != second[key]]
    if mismatched:
        raise ValueError(f"Restart grids differ in: {', '.join(mismatched)}")
    if first["kinetic_grid_code"] != 1:
        raise ValueError("This diagnostic requires the cell_average_eta grid")
    if first["radial_grid_code"] != 0:
        raise ValueError("This diagnostic currently requires a logarithmic radial grid")
    if first["eta_grid_code"] != 1:
        raise ValueError("This diagnostic currently requires a power-law eta grid")
    if first["w_stretch"] < 0.0 or first["w_center"] != 0.0:
        raise ValueError("This diagnostic requires a symmetric centered w grid")


def canonical_measure(header: dict) -> tuple[np.ndarray, np.ndarray]:
    nr = int(header["Nr"])
    nw = int(header["Nw"])
    nl = int(header["NL"])
    r_edge = np.geomspace(header["r_min"], header["r_max"], nr + 1)
    w_coordinate = np.linspace(-1.0, 1.0, nw + 1)
    w_stretch = float(header["w_stretch"])
    if w_stretch == 0.0:
        w_edge = header["w_max"] * w_coordinate
    else:
        w_edge = (
            header["w_max"]
            * np.sinh(w_stretch * w_coordinate)
            / np.sinh(w_stretch)
        )
        w_edge[0] = -header["w_max"]
        w_edge[-1] = header["w_max"]
    fraction = np.linspace(0.0, 1.0, nl + 1)
    eta_edge = 0.5 * header["L_max"] ** 2 * fraction ** header["eta_power"]
    # Use actual radial, stretched-momentum and eta widths for every phase-space cell.
    measure = (
        np.diff(r_edge)[:, None, None]
        * np.diff(w_edge)[None, :, None]
        * np.diff(eta_edge)[None, None, :]
    )
    return r_edge, measure


# ---- Measure distribution, entropy, and total-mass drift --------------------
def phase_space_metrics(
    initial_header: dict,
    initial: np.ndarray,
    final_header: dict,
    final: np.ndarray,
) -> dict[str, float]:
    require_matching_grids(initial_header, final_header)
    _, measure = canonical_measure(initial_header)
    delta = final - initial
    visible_mass0 = 8.0 * np.pi**2 * np.sum(measure * initial)
    visible_mass1 = 8.0 * np.pi**2 * np.sum(measure * final)
    hidden_mass0 = float(initial_header["hidden_carrier_mass"])
    hidden_mass1 = float(final_header["hidden_carrier_mass"])
    # Include hidden carriers in mass conservation even though distribution L1/L2 compare the visible cell averages.
    mass0 = visible_mass0 + hidden_mass0
    mass1 = visible_mass1 + hidden_mass1
    l1 = np.sum(measure * np.abs(delta)) / np.sum(measure * np.abs(initial))
    l2 = np.sqrt(np.sum(measure * delta**2) / np.sum(measure * initial**2))

    def boltzmann_h(distribution: np.ndarray) -> float:
        positive = distribution > 0.0
        values = np.zeros_like(distribution)
        # Evaluate f*(log f-1) only on positive f so empty cells contribute the finite limiting value zero.
        values[positive] = distribution[positive] * (
            np.log(distribution[positive]) - 1.0
        )
        return float(8.0 * np.pi**2 * np.sum(measure * values))

    h0 = boltzmann_h(initial)
    h1 = boltzmann_h(final)
    return {
        "initial_time": float(initial_header["time"]),
        "final_time": float(final_header["time"]),
        "final_step": int(final_header["step"]),
        "distribution_weighted_l1": float(l1),
        "distribution_weighted_l2": float(l2),
        "relative_mass_change": float((mass1 - mass0) / mass0),
        "initial_visible_mass": float(visible_mass0),
        "final_visible_mass": float(visible_mass1),
        "initial_hidden_carrier_mass": hidden_mass0,
        "final_hidden_carrier_mass": hidden_mass1,
        "initial_carrier_packet_count": int(initial_header["carrier_packet_count"]),
        "final_carrier_packet_count": int(final_header["carrier_packet_count"]),
        "relative_boltzmann_h_change": float((h1 - h0) / abs(h0)),
        "relative_maximum_f_change": float((final.max() - initial.max()) / initial.max()),
    }


# ---- Measure observable density and velocity-dispersion drift ---------------
# Profile errors are weighted by the mass in each spherical shell so an empty
# outer cell cannot dominate an equilibrium assessment.
def profile_metrics(paths: list[Path], r_edge: np.ndarray) -> dict[str, float]:
    segments = [
        np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        for path in paths
    ]
    table = np.concatenate(segments)
    initial_time = float(np.min(table["time"]))
    final_time = float(np.max(table["time"]))
    initial = table[np.isclose(table["time"], initial_time, rtol=0.0, atol=1.0e-14)]
    final = table[np.isclose(table["time"], final_time, rtol=0.0, atol=1.0e-14)]
    order0 = np.argsort(initial["r"])
    order1 = np.argsort(final["r"])
    initial = initial[order0]
    final = final[order1]
    if initial.size != final.size or initial.size != r_edge.size - 1:
        raise ValueError("Profile snapshots do not match the restart radial grid")
    if not np.allclose(initial["r"], final["r"], rtol=1.0e-13, atol=0.0):
        raise ValueError("Initial and final profile radii differ")
    # Profile drift uses spherical radial-volume weights on the same restart mesh.
    shell_weight = 4.0 * np.pi * initial["r"] ** 2 * np.diff(r_edge)
    rho0 = initial["rho"]
    rho1 = final["rho"]
    sigma0 = np.divide(initial["p"], rho0, out=np.zeros_like(rho0), where=rho0 > 0.0)
    sigma1 = np.divide(final["p"], rho1, out=np.zeros_like(rho1), where=rho1 > 0.0)

    def weighted_l1(value0: np.ndarray, value1: np.ndarray, mask: np.ndarray) -> float:
        weights = shell_weight[mask] * rho0[mask]
        return float(
            np.sum(weights * np.abs(value1[mask] - value0[mask]))
            / np.sum(weights * np.maximum(np.abs(value0[mask]), 1.0e-300))
        )

    # Separate the populated comparison region and inner core before reporting weighted observable errors.
    all_cells = rho0 > rho0.max() * 1.0e-14
    core = all_cells & (initial["r"] < 1.0)
    density_l1 = float(
        np.sum(shell_weight * np.abs(rho1 - rho0))
        / np.sum(shell_weight * np.abs(rho0))
    )
    core_density_l1 = float(
        np.sum(shell_weight[core] * np.abs(rho1[core] - rho0[core]))
        / np.sum(shell_weight[core] * np.abs(rho0[core]))
    )
    peak = int(np.argmax(rho0))
    return {
        "profile_initial_time": initial_time,
        "profile_final_time": final_time,
        "density_weighted_l1": density_l1,
        "core_density_weighted_l1_r_lt_1": core_density_l1,
        "sigma_r2_weighted_l1": weighted_l1(sigma0, sigma1, all_cells),
        "core_sigma_r2_weighted_l1_r_lt_1": weighted_l1(sigma0, sigma1, core),
        "relative_peak_cell_density_change": float((rho1[peak] - rho0[peak]) / rho0[peak]),
        "maximum_density_initial": float(rho0.max()),
        "maximum_density_final": float(rho1.max()),
    }


# ---- Command-line assembly of the equilibrium-audit report ------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-restart", type=Path, required=True)
    parser.add_argument("--final-restart", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    initial_header, initial = read_restart(args.initial_restart)
    final_header, final = read_restart(args.final_restart)
    r_edge, _ = canonical_measure(initial_header)
    report = {
        **phase_space_metrics(initial_header, initial, final_header, final),
        **profile_metrics(args.profiles, r_edge),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
