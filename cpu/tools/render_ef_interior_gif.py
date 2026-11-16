#!/usr/bin/env python3
# Render saved EF density, compactness and active/excised rest-weight diagnostics.
# Input CSVs contain native advanced-time slices and horizon-regular observer density in relativistic code units.
# Output GIF, representative PNGs and a frame ledger preserve the selected actual steps and times.

"""Native EF interior animation using the horizon-regular rho_infall column.

The observer has u^v=1/b, u^r=(a-1)/2 and unit timelike norm throughout
the retained region. The density uses native EF events, including a<=0.
Excised cells are masked. A geodesic observer requires its own trajectory.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "venus-ef-interior-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
from PIL import Image
from render_weighted_einstein_vlasov_pic_profile_gifs import boundary_aware_log_profile


def load_complete_profile_selection(path, frame_count, v_min=None, v_max=None):
    """Select native slices whose complete radial block has reached disk."""
    metadata = []
    for chunk in pd.read_csv(path, usecols=["step", "advanced_time", "metric_sample_kind", "metric_cell"], chunksize=100000):
        # An outer-face record identifies the declared cell count of a complete saved radial block.
        faces = chunk[chunk.metric_sample_kind == "outer_face"]
        if not faces.empty:
            metadata.append(faces[["step", "advanced_time", "metric_cell"]])
    if not metadata:
        raise RuntimeError("No complete EF profile blocks with an outer face")
    times = pd.concat(metadata, ignore_index=True)
    if times.duplicated(["step"]).any() or not np.isfinite(times.to_numpy()).all():
        raise RuntimeError("Invalid or duplicate EF profile step metadata")
    cells = times.metric_cell.to_numpy()
    if not np.all(cells == cells.astype(int)) or not np.all(cells == cells[0]) or cells[0] < 1:
        raise RuntimeError("Inconsistent EF radial profile extent")
    lower = float(times.advanced_time.min()) if v_min is None else v_min
    upper = float(times.advanced_time.max()) if v_max is None else v_max
    times = times[(times.advanced_time >= lower) & (times.advanced_time <= upper)]
    if len(times) < 2:
        raise RuntimeError("Fewer than two complete saved profiles in requested time range")
    targets = np.linspace(times.advanced_time.min(), times.advanced_time.max(), frame_count)
    indices = sorted(set(int(np.argmin(abs(times.advanced_time.to_numpy() - time))) for time in targets))
    chosen = times.iloc[indices]
    steps = set(chosen.step.astype(int))
    columns = ["step", "advanced_time", "r", "a", "compactness", "rho_infall", "excision_mask", "metric_sample_kind", "metric_cell"]
    pieces = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100000):
        part = chunk[chunk.step.isin(steps) & (chunk.metric_sample_kind == "cell_center")]
        if not part.empty:
            pieces.append(part)
    if not pieces:
        raise RuntimeError("Selected EF profiles have no radial cells")
    selected = pd.concat(pieces, ignore_index=True)
    # Require all cell indices, a common native time and finite ordered geometry for every selected slice.
    for _, expected in chosen.iterrows():
        group = selected[selected.step == expected.step].sort_values("metric_cell")
        if (len(group) != int(expected.metric_cell)
                or not np.array_equal(group.metric_cell.to_numpy(), np.arange(int(expected.metric_cell)))
                or not (group.advanced_time == expected.advanced_time).all()
                or not np.isfinite(group[["r", "a", "compactness", "rho_infall", "excision_mask"]].to_numpy()).all()
                or not (group.r > 0).all() or not (np.diff(group.r.to_numpy()) > 0).all()):
            raise RuntimeError("Incomplete or inconsistent EF profile radial block")
    return selected, lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--v-min", type=float)
    parser.add_argument("--v-max", type=float)
    args = parser.parse_args()
    if args.frames < 2 or not 1 <= args.fps <= 100:
        parser.error("At least two frames and 1<=fps<=100 are required")
    history = pd.read_csv(args.run / "high_compactness_history.csv")
    selected, lower, upper = load_complete_profile_selection(
        args.run / "high_compactness_profiles.csv", args.frames, args.v_min, args.v_max)
    curves = []
    for step, group in selected.groupby("step", sort=True):
        group = group.sort_values("r")
        r = group.r.to_numpy()
        rho = group.rho_infall.to_numpy().copy()
        # Mask excised cells before differentiating. The observer density remains meaningful in active trapped cells.
        rho[group.excision_mask.to_numpy() != 0] = np.nan
        rho[rho <= 0] = np.nan
        rr, lrho, slope = boundary_aware_log_profile(r, rho, 1e-9, 7, False)
        # Join each plotted field to exactly one history row at the same step and advanced time.
        matching_history = history[(history.step == step)
            & (history.advanced_time == group.advanced_time.iloc[0])]
        if len(matching_history) != 1:
            raise RuntimeError("EF profile lacks one history row at the same accepted step and time")
        h = matching_history.iloc[0]
        curves.append((step, group, rho, rr, lrho, slope, h))
    finite = [c[4][np.isfinite(c[4])] for c in curves if np.any(np.isfinite(c[4]))]
    if not finite:
        raise RuntimeError("No active matter in requested time range")
    density_values = np.concatenate(finite)
    positive_r = np.concatenate([c[3] for c in curves if len(c[3])])
    rlo = max(float(selected.r.min()), float(positive_r.min()) * 0.7)
    # Include both the sampled matter extent and recorded horizon radius when setting the common radial domain.
    rhi = max(float(positive_r.max()) * 1.2, float(history.apparent_horizon_radius.max()) * 1.4)
    compact = selected.loc[(selected.excision_mask == 0) & (selected.r >= rlo) & (selected.r <= rhi), "compactness"]
    cmax = max(1.2, float(compact.max()) * 1.05)
    ymin, ymax = float(density_values.min()) - 0.3, float(density_values.max()) + 0.3
    rendered = []
    ledger = []
    total = history.total_rest_weight.to_numpy()
    trapped_times = history.loc[history.apparent_horizon_radius > 0, "advanced_time"]
    horizon_time = float(trapped_times.iloc[0]) if len(trapped_times) else None
    for step, group, rho, rr, lrho, slope, h in curves:
        fig, ax = plt.subplots(2, 2, figsize=(12.8, 8), dpi=100)
        axes = ax.ravel()
        for axis in axes[:3]:
            axis.set_xscale("log")
            axis.set_xlim(rlo, rhi)
            axis.xaxis.set_major_locator(FixedLocator([
                x for x in (.01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50)
                if rlo <= x <= rhi]))
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel("Areal radius r")
            axis.minorticks_on()
            axis.tick_params(which="both", direction="in", top=True, right=True)
            axis.grid(False)
            if h.excision_radius > 0:
                axis.axvspan(rlo, h.excision_radius, color="0.7", alpha=0.5)
                axis.axvline(h.excision_radius, color="0.3", ls=":", label="excision")
            if h.apparent_horizon_radius > 0:
                axis.axvline(h.apparent_horizon_radius, color="#7a3e9d", ls="--", label="apparent horizon")
        axes[0].plot(rr, lrho, color="#007f86", lw=2)
        axes[0].set(ylabel=r"$\log_{10}\rho_{\rm infall}$", ylim=(ymin, ymax))
        axes[0].set_title("Matter in a horizon-regular observer frame")
        axes[1].plot(rr, slope, color="#007f86", lw=2)
        axes[1].set(ylabel=r"$d\ln\rho_{\rm infall}/d\ln r$", ylim=(-15, 15))
        axes[1].set_title("Local density slope")
        valid = group.excision_mask.to_numpy() == 0
        axes[2].plot(group.r[valid], group.compactness[valid], color="#007f86", lw=2)
        axes[2].axhline(1, color="#7a3e9d", ls="--", label="2m/r = 1")
        axes[2].set(ylabel="Compactness 2m/r", ylim=(0, cmax))
        axes[2].set_yscale("symlog", linthresh=1)
        axes[2].yaxis.set_major_locator(FixedLocator([
            x for x in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
            if x <= cmax]))
        axes[2].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axes[2].set_title("Metric in the retained region")
        if h.apparent_horizon_radius > 0:
            axes[2].legend(fontsize=9)
        # Normalize active and excised budgets by the same total rest weight and mark the displayed native epoch.
        axes[3].plot(history.advanced_time, history.active_rest_weight / total, label="active", color="#007f86")
        axes[3].plot(history.advanced_time, history.excised_rest_weight / total, label="excised", color="0.4")
        if "active_trapped_rest_weight" in history:
            axes[3].plot(history.advanced_time, history.active_trapped_rest_weight / total, label="active and trapped", color="#7a3e9d")
        axes[3].axvline(h.advanced_time, color="black", lw=1, label="shown slice")
        if horizon_time is not None:
            axes[3].axvline(horizon_time, color="#7a3e9d", ls=":", lw=1, label="first saved horizon")
        axes[3].set(xlim=(lower, upper), ylim=(-0.03, 1.03), xlabel="EF advanced time v", ylabel="Fraction of total rest weight")
        axes[3].set_title("Active and excised matter")
        axes[3].legend(fontsize=9, loc="lower left")
        axes[3].grid(False)
        if "active_marker_count" in h:
            axes[3].text(.98, .75,
                f"Active particles: {int(h.active_marker_count):,}\n"
                f"Active rest weight: {h.active_rest_weight / h.total_rest_weight:.2%}\n"
                f"Excision radius: {h.excision_radius:.3f}",
                transform=axes[3].transAxes, ha="right", va="center", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        if not len(rr):
            for axis in axes[:2]:
                axis.text(.5, .55, "No active matter in retained region", transform=axis.transAxes, ha="center")
        fig.suptitle(f"VENUS · EF interior · v = {h.advanced_time:.5f}", fontsize=17)
        fig.text(.5, .025, r"Observer: $u^v=1/b$, $u^r=(a-1)/2$ (unit timelike observer). Gray = excised region.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .06, 1, .95))
        fig.canvas.draw()
        rendered.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
        # The frame ledger counts retained positive-density cells with a negative metric a separately from the excised population.
        inside = (group.a.to_numpy() < 0) & np.isfinite(rho)
        ledger.append({"step": int(step), "ef_v": float(h.advanced_time),
            "active_density_cells": int(np.isfinite(rho).sum()),
            "trapped_density_cells": int(inside.sum()),
            "active_trapped_rest_weight": float(h.get("active_trapped_rest_weight", 0)),
            "excision_radius": float(h.excision_radius),
            "apparent_horizon_radius": float(h.apparent_horizon_radius)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(args.output, save_all=True, append_images=rendered[1:], duration=round(1000 / args.fps), loop=0)
    rendered[0].save(args.output.with_suffix(".initial.png"))
    rendered[-1].save(args.output.with_suffix(".final.png"))
    interior_index = max(range(len(ledger)), key=lambda i: ledger[i]["active_trapped_rest_weight"])
    rendered[interior_index].save(args.output.with_suffix(".interior.png"))
    args.output.with_suffix(".frames.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"Rendered {len(rendered)} native saved EF slices to {args.output}")


if __name__ == "__main__":
    main()
