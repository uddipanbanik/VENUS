#!/usr/bin/env python3
# Render cold VP density/slopes and all-particle physical virial/inertia histories.
# Compressed source profiles and independent particle moment tables are authenticated through catalog.json.
# The endpoint partition uses its stated instantaneous energy selection and reconstructed own gravity.

"""Render native cold Vlasov–Poisson profiles and physical virial histories.

Each movie uses all 601 saved states from time 0 through 3000. Profiles supply
density and its centered 21-bin logarithmic derivative. Independent native
particle diagnostics supply physical kinetic energy, force and inertia.
The static bound partition is displayed at its measured final state.
"""

from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "venus-movie-matplotlib")
)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

MOVIE_DIRECTORY = Path(__file__).resolve().parents[1]
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 18,
        "axes.labelsize": 24,
        "axes.titlesize": 22,
        "axes.linewidth": 1.8,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 16,
        "savefig.dpi": 120,
    }
)


def sha(path):
    """Hash a complete input file for the frame ledger."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def axes_style(ax):
    """Apply the example tick directions and grid-free axes."""
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=8.5, width=1.7, pad=5)
    ax.tick_params(which="minor", length=4.5, width=1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))


def native_slope(a, n=21):
    """Differentiate the existing local quadratic in log radius and density."""
    x = np.log10(a["radius"])
    y = np.log10(np.maximum(a["density"], np.finfo(float).tiny))
    result = np.full(len(x), np.nan)
    half = n // 2
    for j in range(half, len(x) - half):
        # Fit the centered 21-cell logarithmic window only when every source density is positive.
        select = slice(j - half, j + half + 1)
        if np.any(a["density"][select] <= 0):
            continue
        result[j] = np.polynomial.polynomial.polyfit(x[select] - x[j], y[select], 2)[1]
    return result


def checked_inputs(directory, case):
    """Authenticate compressed native tables and exact particle diagnostics."""
    catalog = json.loads((directory / "catalog.json").read_text())
    entry = catalog["cases"][case]
    for name, wanted in entry["files"].items():
        assert sha(directory / case / name) == wanted, name
    frames = []
    for record in entry["frames"]:
        path = directory / record["file"]
        assert sha(path) == record["sha256"]
        # Authenticate both the compressed artifact and the original CSV bytes before parsing the native profile.
        raw = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record["native_sha256"]
        a = np.atleast_1d(np.genfromtxt(io.BytesIO(raw), names=True, delimiter=","))
        assert len(a) == 256 and np.all(a["time"] == record["actual_time"])
        frames.append(dict(record, data=a, slope=native_slope(a)))
    clock = np.array([z["actual_time"] for z in frames])
    assert len(frames) == 601 and clock[0] == 0 and clock[-1] == 3000
    assert np.all(np.abs(clock - 5 * np.arange(601)) < 1e-8) and np.all(
        np.diff(clock) > 0
    )
    read = lambda name: np.atleast_1d(
        np.genfromtxt(directory / case / name, names=True, delimiter=",")
    )
    budget = read("virial_budget.csv")
    boundary = read("boundary_history.csv")
    initial = json.loads((directory / case / "initial_moments.json").read_text())
    endpoint = json.loads((directory / case / "endpoint_moments.json").read_text())
    assert (
        initial["time"] == 0
        and endpoint["time"] == budget["time"][-1] == boundary["time"][-1] == 3000
    )
    assert np.all(np.diff(budget["time"]) > 0) and budget["time"][0] > 0
    # The physical acceleration includes both the halo force and the
    # prescribed cosmological background. Outer-wall impulses are separate.
    force = budget["W_force"] + (budget["H_dot"] + budget["H"] ** 2) * budget["I"]
    force0 = initial["W_force"] + (initial["H_dot"] + initial["H"] ** 2) * initial["I"]
    # Prepend the independently measured initial virial ratio to the subsequent actual moment history.
    q = np.r_[
        2 * initial["K_physical"] / abs(force0),
        2 * budget["K_physical"] / np.abs(force),
    ]
    history = dict(
        time=np.r_[0, budget["time"]], q=q, inertia=np.r_[initial["I"], budget["I"]]
    )
    q_endpoint = (
        2
        * endpoint["K_physical"]
        / abs(
            endpoint["W_force"]
            + (endpoint["H_dot"] + endpoint["H"] ** 2) * endpoint["I"]
        )
    )
    assert abs(q[-1] - q_endpoint) < 1e-13
    assert abs(history["inertia"][-1] - endpoint["I"]) < 1e-12 * endpoint["I"]
    # Only the static endpoint has the supplied bound/positive-energy partition; its scope stays at that time.
    partition = (
        json.loads((directory / case / "endpoint_partition.json").read_text())
        if case == "cold_static_vp"
        else None
    )
    return entry, frames, history, boundary, partition


def make_cold(directory, case):
    """Build the native profile and physical-moment panels."""
    entry, frames, history, boundary, partition = checked_inputs(directory, case)
    expanding = case == "cold_expanding_vp"
    fig, axs = plt.subplots(2, 2, figsize=(13.7, 9.0), dpi=120)
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.19, top=0.82, wspace=0.24, hspace=0.62
    )
    model = "expanding" if expanding else "static"
    fig.text(
        0.085,
        0.965,
        rf"Cold {model} Vlasov–Poisson collapse: $\gamma_{{\rm i}}=0.5,\ f_v=0.25$",
        fontsize=21,
        va="center",
    )
    time_label = fig.text(0.985, 0.905, "", ha="right", va="center", fontsize=21)
    wall_label = fig.text(0.085, 0.905, "", ha="left", va="center", fontsize=15)
    fig.text(
        0.085,
        0.08,
        r"$W_{\rm physical}=-\sum_p m_p r_p\partial_r\Phi+(\dot H+H^2)I$,  $I=\sum_p m_p r_p^2$.",
        fontsize=14,
    )
    sampling_label = fig.text(0.085, 0.025, "", fontsize=13.5)
    density, slope, qax, iax = axs.flat
    profile_lines = []
    for col, ax in enumerate([density, slope]):
        axes_style(ax)
        ax.set(xlim=(-1.7, 3.0), xlabel=r"$\log_{10}r$")
        ax.set_title("Density" if col == 0 else "Signed logarithmic slope", pad=12)
        ax.set_ylabel(
            r"$\log_{10}\rho$" if col == 0 else r"$\mathrm{d}\log\rho/\mathrm{d}\log r$"
        )
        ax.set_ylim((-12, 3.0) if col == 0 else (-5.0, 1.0))
        a = frames[0]["data"]
        y = (
            np.log10(np.maximum(a["density"], 1e-300))
            if col == 0
            else frames[0]["slope"]
        )
        valid = a["density"] > 0 if col == 0 else np.isfinite(y)
        # Use heavy initial segments for cells with at least 32 effective markers and thin segments for the remaining samples.
        strong = (a["density"] > 0) & (a["neff"] >= 32)
        for mask, lw in [(valid, 0.9), (strong, 2.2)]:
            ax.plot(
                np.log10(a["radius"]),
                np.where(mask, y, np.nan),
                color="k",
                ls="--",
                lw=lw,
            )
        profile_lines.append(
            [ax.plot([], [], lw=lw, color="#087cff")[0] for lw in [0.9, 3.0]]
        )
    handles = [
        Line2D([], [], color="k", lw=2.2, ls="--", label="Initial state"),
        Line2D([], [], color="#087cff", lw=3.0, label="Current state"),
    ]
    density_legend = density.legend(
        handles=handles, loc="lower left", frameon=False, handlelength=3.0
    )
    for reference, style in [(-1, (0, (4, 4))), (-1.5, (0, (1, 3))), (-3, (0, (7, 3)))]:
        slope.axhline(reference, color=".5", lw=1.2, ls=style)
        slope.text(
            2.9,
            reference + 0.05,
            rf"${reference:g}$",
            color=".4",
            ha="right",
            va="bottom",
            fontsize=14,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
        )
    traces, cursors = [], []
    for col, ax in enumerate([qax, iax]):
        axes_style(ax)
        ax.set(xlim=(0, 3.53), xlabel=r"$\log_{10}(1+t)$")
        ax.set_ylabel(
            r"$2K_{\rm physical}/|W_{\rm physical}|$" if col == 0 else r"$\log_{10}I$"
        )
        ax.set_title(
            "Virial ratio of all particles" if col == 0 else "Inertia of all particles",
            pad=12,
        )
        ax.set_ylim((0, 1.9) if col == 0 else (0, 5.35))
        traces.append(
            ax.plot(
                [],
                [],
                color="k",
                lw=2.6,
                marker="o",
                markevery=[-1],
                ms=4,
                label="Full population",
            )[0]
        )
        cursors.append(ax.axvline(0, color="#087cff", lw=1.2, ls=":"))
        if not expanding:
            ax.axvline(np.log10(2591), color=".6", lw=1.2, ls=(0, (2, 3)), zorder=0)
    qax.axhline(1, color=".4", ls=":", lw=1.5)
    endpoint_lines = []
    if partition:
        endpoint_lines = [
            qax.plot(
                [],
                [],
                "o",
                ms=9,
                mfc="#087cff",
                mec="white",
                mew=1,
                label="Bound endpoint",
            )[0],
            iax.plot(
                [],
                [],
                "o",
                ms=9,
                mfc="#087cff",
                mec="white",
                mew=1,
                label="Bound endpoint",
            )[0],
            iax.plot(
                [],
                [],
                "o",
                ms=9,
                mfc="white",
                mec="#f02c33",
                mew=2,
                label="Positive orbital energy",
            )[0],
        ]
    qhandles = [traces[0]]
    if partition:
        qhandles += [
            Line2D(
                [],
                [],
                color=".6",
                lw=1.2,
                ls=(0, (2, 3)),
                label="First wall-impact interval",
            ),
            endpoint_lines[0],
        ]
    qlegend = qax.legend(
        handles=qhandles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        fontsize=13,
    )
    ilegend = iax.legend(loc="upper left", frameon=False, fontsize=13)

    def render(index):
        """Draw one native profile and the histories up to its time."""
        z = frames[index]
        a, t = z["data"], z["actual_time"]
        color = plt.cm.plasma_r(0.13 + 0.84 * index / (len(frames) - 1))
        for col in range(2):
            values = (
                np.log10(np.maximum(a["density"], 1e-300)) if col == 0 else z["slope"]
            )
            valid = a["density"] > 0 if col == 0 else np.isfinite(values)
            strong = (a["density"] > 0) & (a["neff"] >= 32)
            for line, mask in zip(profile_lines[col], [valid, strong]):
                line.set_data(np.log10(a["radius"]), np.where(mask, values, np.nan))
                line.set_color(color)
                line.set_visible(index > 0)
        density_legend.get_lines()[1].set_color(color)
        density_legend.get_lines()[1].set_visible(index > 0)
        density_legend.get_texts()[1].set_visible(index > 0)
        time_label.set_text(rf"$t={t:.0f}$")
        # Show histories only through the current profile time, even when their saved cadence is denser.
        past = history["time"] <= t + 1e-10
        for col, values in enumerate([history["q"], np.log10(history["inertia"])]):
            traces[col].set_data(np.log10(1 + history["time"][past]), values[past])
            cursors[col].set_xdata([np.log10(1 + t)] * 2)
        # Read cumulative wall impacts from the latest actual boundary row at or before this frame.
        saved_boundary = np.flatnonzero(boundary["time"] <= t + 1e-10)[-1]
        impacts = int(boundary["outer_wall_impacts"][saved_boundary])
        wall_label.set_text(rf"Outer wall $R=1000$; impacts: {impacts}")
        final = index == len(frames) - 1
        for legend, offset in [(qlegend, 2), (ilegend, 1)]:
            for line, text in zip(
                legend.get_lines()[offset:], legend.get_texts()[offset:]
            ):
                line.set_visible(final)
                text.set_visible(final)
        # Reveal the selected populations only at their measured final endpoint.
        if partition and final:
            bound = partition["instantaneously_bound_in_full_potential"]
            positive = partition["instantaneously_unbound_in_full_potential"]
            x = np.log10(1 + t)
            endpoint_lines[0].set_data([x], [bound["Q_self_force"]])
            endpoint_lines[1].set_data([x], [np.log10(bound["I"])])
            endpoint_lines[2].set_data([x], [np.log10(positive["I"])])
        else:
            for line in endpoint_lines:
                line.set_data([], [])
        sampling_label.set_text(
            "Thick: at least 32 effective markers; thin: native estimates."
            + (
                "  Blue: bound self-gravity; red: positive energy."
                if partition and final
                else (
                    r"  Gray vertical guide: first wall impact in $2590<t\leq2595$."
                    if not expanding
                    else r"  Expanding background: $H=2/[3(t+12)]$."
                )
            )
        )
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    return fig, frames, render


def encode(directory, output, case, ffmpeg, fps, preview_only=False):
    from PIL import Image

    fig, frames, render = make_cold(directory, case)
    if preview_only:
        Image.fromarray(render(len(frames) - 1)).save(output / (case + ".png"))
        plt.close(fig)
        return
    first = render(0)
    height, width = first.shape[:2]
    assert width % 2 == height % 2 == 0
    movie = output / (case + ".mp4")
    temporary = output / ("." + case + ".tmp.mp4")
    command = [
        str(ffmpeg),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    ledger = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        # Encode every source state once, followed by the explicitly repeated final viewing hold.
        sequence = list(range(len(frames))) + [len(frames) - 1] * (2 * fps)
        for number, index in enumerate(sequence):
            z = frames[index]
            process.stdin.write((first if number == 0 else render(index)).tobytes())
            ledger.append(
                dict(
                    movie_frame=number,
                    movie_seconds=number / fps,
                    native_profile=z["file"],
                    native_sha256=z["native_sha256"],
                    compressed_sha256=z["sha256"],
                    actual_time=z["actual_time"],
                    native_cells=z["native_cells"],
                    native_index=z["index"],
                )
            )
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Movie encoding failed.")
    plt.close(fig)
    check = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(temporary),
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    # Decode the complete temporary MP4 and verify its frame count before publishing the output filename.
    decoded = [int(x) for x in re.findall(r"^frame=(\d+)$", check.stdout, flags=re.M)]
    assert check.returncode == 0 and not check.stderr and decoded[-1] == len(ledger)
    assert (
        ledger[0]["actual_time"] == 0
        and ledger[1]["actual_time"] > 0
        and ledger[-1]["actual_time"] == 3000
    )
    # Move the completed, decoded movie into place after both clock and encoding checks pass.
    temporary.replace(movie)
    metadata = dict(
        case=case,
        fps=fps,
        width=width,
        height=height,
        native_time_range=[0, 3000],
        clock="Every native saved state at five-unit intervals; the opening state advances on the next frame; final state holds for two seconds.",
        native_frames=len(frames),
        encoded_frames=len(ledger),
        complete_decode_PASS=True,
        physical_virial_definition="2*K_physical/abs(W_Phi+(H_dot+H^2)*I)",
        source_catalog_sha256=sha(directory / "catalog.json"),
        renderer_sha256=sha(__file__),
        movie_sha256=sha(movie),
        encoder=dict(codec="H.264", pixel_format="yuv420p", quality_crf=18),
        frames=ledger,
    )
    (output / (case + "_frames.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in metadata.items() if k != "frames"}, indent=2),
        flush=True,
    )


def main():
    """Resolve portable input paths and render the selected cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["cold_static_vp", "cold_expanding_vp"])
    parser.add_argument(
        "--input-directory", type=Path, default=MOVIE_DIRECTORY / "data/cold_vp"
    )
    parser.add_argument(
        "--output-directory", type=Path, default=MOVIE_DIRECTORY / "generated/cold_vp"
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--ffmpeg",
        help="Encoder path; defaults to VENUS_FFMPEG, PATH or imageio_ffmpeg.",
    )
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    assert args.fps > 0
    directory, output = (
        args.input_directory.expanduser().resolve(),
        args.output_directory.expanduser().resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    encoder = args.ffmpeg or os.environ.get("VENUS_FFMPEG") or shutil.which("ffmpeg")
    if encoder is None and not args.preview_only:
        import imageio_ffmpeg

        encoder = imageio_ffmpeg.get_ffmpeg_exe()
    for case in [args.case] if args.case else ["cold_static_vp", "cold_expanding_vp"]:
        encode(
            directory,
            output,
            case,
            Path(encoder) if encoder else None,
            args.fps,
            args.preview_only,
        )


if __name__ == "__main__":
    main()
