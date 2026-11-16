#!/usr/bin/env python3
# Plot physical virial ratios from complete-population and selected-population tables.
# Inputs identify their own kinetic/force conventions and actual time samples through a catalog.
# Output is a PDF/PNG summary and scalar diagnostics with the same population definitions.

"""Render physical virial ratios for complete and bound halo populations.

Only cached native/profile moments are consumed. The source contract records
the selected force and kinetic-energy conventions and each curve's time range.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "venus-virial-summary-mpl")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np


def sha(path):
    """Return the complete digest identifying one reduced input table."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trapezoidal_integral(values, coordinates):
    """Use the NumPy trapezoidal API available in the installed release."""
    # Use the available NumPy name for the same trapezoidal quadrature.
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(values, coordinates)


def render(input_directory, output_directory):
    """Draw physical virial balance from authenticated reduced moment tables."""
    base = Path(input_directory).resolve()
    out = Path(output_directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((base / "catalog.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 18,
            "axes.labelsize": 23,
            "axes.titlesize": 20,
            "axes.linewidth": 1.8,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "lines.linewidth": 2.8,
            "pdf.fonttype": 42,
            "savefig.dpi": 220,
        }
    )
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    axes = [ax]
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.70)
    cases = [
        ("g03_bp", "#087cff", "-", r"BP, expanding: $\gamma_i=0.3$"),
        ("g05_bp", "#f02c33", "-", r"BP, expanding: $\gamma_i=0.5$"),
        ("g08_bp", "#00a651", "-", r"BP, expanding: $\gamma_i=0.8$"),
        ("g05_vp", "black", (0, (6, 3)), r"VP, expanding: $\gamma_i=0.5$"),
        ("gaussian_static_vp", "#b57300", (0, (4, 2, 1, 2)), "VP, Gaussian: static"),
        ("gaussian_expanding_vp", "#00a8b8", (0, (2, 2)), "VP, Gaussian: expanding"),
        ("static_g05_bound", "#8b2bc0", "-", r"VP, static: bound $\gamma_i=0.5$"),
        ("gaussian_static_bound", "#b57300", "None", "VP, Gaussian: bound static"),
    ]
    diagnostics = {}
    handles = []
    for case, color, style, label in cases:
        binding = contract["sources"][case]
        path = base / binding["file"].split("/")[-1]
        assert sha(path) == binding["sha256"], str(path)
        data = np.atleast_1d(
            np.genfromtxt(path, names=True, delimiter=",", dtype=None, encoding="utf-8")
        )
        if case == "static_g05_bound":
            # The static power-law curve admits only iterative selections with unchanged final membership.
            unresolved = data["status"] != "STABLE_MEMBERSHIP"
            assert np.array_equal(data["time"][unresolved], [0, 10, 30, 100])
            data = data[~unresolved]
            assert np.array_equal(data["time"], [300, 800, 2000, 5000, 10000])
            q = np.asarray(data["Q_stable"], dtype=float)
        elif case.startswith("gaussian_"):
            # Gaussian measurements use all-marker physical kinetic energy and the applied background-plus-halo force.
            total_force = data["W_force"] + (data["H_dot"] + data["H"]**2)*data["I"]
            q = 2*data["K_physical"] / np.abs(total_force)
            assert np.max(np.abs(q-data["Q_physical"])) < 1e-13
        else:
            # Resolved-profile measurements keep their own center quadrature and hidden-inner-mass force convention.
            total_force = (
                data["W_resolved_constant_line_with_inner_mass"]
                + data["background_acceleration"] * data["I_resolved_center"]
            )
            q = 2 * data["K_physical_resolved_center"] / np.abs(total_force)
            assert np.max(abs(q - data["Q_physical_profile_total_acceleration"])) < 1e-13
        t = np.asarray(data["time"], dtype=float)
        assert np.all(np.diff(t) > 0) and t[-1] == 10000
        if case not in ["static_g05_bound","gaussian_static_bound"]:
            assert t[0] == 0
        assert np.all(np.isfinite(q)) and np.all(q > 0)
        if case == "static_g05_bound":
            (line,) = axes[0].plot(
                np.log10(1 + t), q, color=color, ls=style, lw=1.4, label=label,
                zorder=5, clip_on=False,
            )
        # This selection supplies one own-field endpoint and therefore receives an isolated open square.
        elif case == "gaussian_static_bound":
            assert np.array_equal(t,[10000])
            (line,) = axes[0].plot(np.log10(1+t),q,color=color,ls="None",label=label,
                                    marker="s",markersize=11,markerfacecolor="none",
                                    markeredgewidth=2.2,zorder=8,clip_on=False)
        else:
            (line,) = axes[0].plot(np.log10(1 + t), q, color=color, ls=style, label=label,
                                    marker='o' if case.startswith('gaussian_') else None,
                                    markersize=6.5 if case.startswith('gaussian_') else 0)
        handles.append(line)
        if case not in ["static_g05_bound","gaussian_static_bound"]:
            late = t >= 7000
        diagnostics[case] = {
            "first_time": float(t[0]),
            "last_time": float(t[-1]),
            "points": len(t),
            "Q_first_plotted": float(q[0]),
            "Q_final": float(q[-1]),
            "Q_min": float(q.min()),
            "Q_max": float(q.max()),
        }
        if case not in ["static_g05_bound","gaussian_static_bound"]:
            # A time average needs at least two distinct actual epochs. A lone late point remains an endpoint measurement.
            lt = t[late]
            lq = q[late]
            diagnostics[case].update(
                late_Q_min=float(lq.min()),
                late_Q_max=float(lq.max()),
                late_sample_count=len(lt),
                late_Q_time_mean=(float(trapezoidal_integral(lq, lt) / (lt[-1] - lt[0]))
                                  if len(lt)>=2 and lt[-1]>lt[0] else None),
                late_Q_mean_status=('TIME_AVERAGE' if len(lt)>=2 and lt[-1]>lt[0] else 'ENDPOINT_ONLY'),
            )
    for ax in axes:
        ax.axhline(1, color=".45", lw=1.4, ls=":", zorder=0)
        ax.set_ylabel(r"$Q$")
        ax.grid(False)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=9, width=1.7, pad=6)
        ax.tick_params(which="minor", length=5, width=1.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    axes[0].set(
        title="Physical virial balance",
        xlabel=r"$\log_{10}(1+t)$",
        xlim=(0, 4.055),
        ylim=(0, 2.05),
    )
    axes[0].xaxis.set_major_locator(MultipleLocator(1))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.5))
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.505, 0.995),
        ncol=3,
        fontsize=14,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.1,
        labelspacing=0.65,
    )
    stem = out / "halo_virialization_summary"
    for extension in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + extension), bbox_inches=None)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    visible_text = list(legend.get_texts())
    for ax in axes:
        visible_text.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            visible_text.extend(
                label
                for value, label in zip(
                    axis.get_majorticklocs(), axis.get_majorticklabels()
                )
                if limits[0] <= value <= limits[1]
            )
    # Test the exported-size text bounds after plotting and legend placement are complete.
    for text in visible_text:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if (
                box.x0 < -1
                or box.y0 < -1
                or box.x1 > bounds.x1 + 1
                or box.y1 > bounds.y1 + 1
            ):
                outside.append(text.get_text())
    assert not outside, outside
    plt.close(fig)
    receipt = {
        "renderer_sha256": sha(__file__),
        "contract_sha256": sha(base / "catalog.json"),
        "artifacts": {
            p.name: sha(p) for p in (stem.with_suffix(".pdf"), stem.with_suffix(".png"))
        },
        "diagnostics": diagnostics,
        "text_bounds_check": "PASS",
    }
    (out / "plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    """Resolve portable input/output paths and render the summary figure."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        "--input-directory",
        dest="data_dir",
        type=Path,
        default=root / "data/virial_summary",
    )
    parser.add_argument(
        "--output-dir",
        "--output-directory",
        dest="output_dir",
        type=Path,
        default=root / "reproduced",
    )
    args = parser.parse_args()
    render(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
