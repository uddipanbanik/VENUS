#!/usr/bin/env python3
# Render SIDM density/radial-variance profiles and dispatch the cold VP renderer.
# SIDM inputs are native CSV profiles plus saved-time/source manifests, in their halo scale units.
# The output ledger retains each physical profile epoch and its population/locality diagnostics.

"""Render SIDM profiles and delegate cold VP evolution to its native renderer.

SIDM uses density and centered radial variance with the example figures' 8/32 population
masks. Cold VP uses its complete saved profiles and physical virial histories.
"""

from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, tempfile

# Use a writable platform cache while honoring an explicit user override.
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
INPUT_DIRECTORY = MOVIE_DIRECTORY / "data/sidm_cold"
COLD_INPUT_DIRECTORY = MOVIE_DIRECTORY / "data/cold_vp"
OUTPUT_DIRECTORY = MOVIE_DIRECTORY / "generated/sidm_cold"
FFMPEG = None  # Set from --ffmpeg, VENUS_FFMPEG, PATH or imageio_ffmpeg below.
FPS = 12
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
        "legend.fontsize": 17,
        "savefig.dpi": 120,
    }
)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: np.atleast_1d(np.genfromtxt(p, delimiter=",", names=True))


def axes_style(ax):
    """Apply the example tick directions and grid-free axes."""
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=8.5, width=1.7, pad=5)
    ax.tick_params(which="minor", length=4.5, width=1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))


def rgb(fig):
    """Copy the current canvas as an RGB image for the encoder."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def sidm_data(case):
    """Read and authenticate native density and radial-variance profiles."""
    manifest = json.loads((INPUT_DIRECTORY / "native_sidm_manifest.json").read_text())
    frames = [dict(r) for r in manifest["frames"] if r["case"] == case]
    for z in frames:
        p = INPUT_DIRECTORY / z["local"]
        assert sha(p) == z["sha256"]
        a = read(p)
        # The NFW weighted-particle profiles provide radial variance and effective count directly.
        if case == "sidm_nfw":
            z["r"] = a["radius"]
            z["rho"] = a["density"]
            z["variance"] = a["radial_variance"]
            z["population"] = a["neff"]
            z["locality_ratio"] = float(z["history"]["max_shell_width_over_mfp"])
        else:
            z["r"] = a["r"]
            z["rho"] = a["rho"]
            # For the cored spherical-DSMC table, pressure divided by density gives the radial velocity variance.
            z["variance"] = np.divide(
                a["p"], a["rho"], out=np.zeros(len(a)), where=a["rho"] > 0
            )
            # Convert enclosed-mass increments to particle counts using the fixed equal-particle mass.
            pop = np.diff(np.r_[0, a["m"]]) / (1.1252580933331495 / 1e6)
            assert np.max(np.abs(pop - np.round(pop))) < 1e-6
            z["population"] = np.round(pop)
            z["locality_ratio"] = float(
                z["history"]["maximum_shell_width_over_mean_free_path"]
            )
        assert np.all(a["time"] == z["actual_time"])
        z["native_cells"] = len(a)
    # Check the hashes of the native profiles displayed in the example figures.
    paper = json.loads((INPUT_DIRECTORY / "reference_profiles.json").read_text())
    for p in paper["profiles"]:
        if p["case"] != case:
            continue
        z = min(frames, key=lambda z: abs(z["actual_time"] - p["time"]))
        assert z["actual_time"] == p["time"] and z["sha256"] == p["sha256"]
    return frames


def make_sidm(case):
    """Build the two SIDM panels with the example population masks."""
    frames = sidm_data(case)
    fig, axs = plt.subplots(1, 2, figsize=(13.7, 6.5), dpi=120)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.24, top=0.78, wspace=0.24)
    model = (
        "SIDM core collapse: initially NFW halo"
        if case == "sidm_nfw"
        else "SIDM core collapse: initially cored halo"
    )
    fig.text(0.085, 0.955, model, ha="left", va="center", fontsize=20)
    time_label = fig.text(0.985, 0.875, "", ha="right", va="center", fontsize=21)
    detail = (
        r"Cross section per mass: $20.1\,r_{\rm s}^2/M_{\rm s}$; $c=30$"
        if case == "sidm_nfw"
        else r"Cross section per mass: $32.4\,r_{\rm s}^2/M_{\rm s}$; $\gamma_{\rm i}=0.1$"
    )
    fig.text(0.085, 0.875, detail, ha="left", va="center", fontsize=16)
    count = "effective markers" if case == "sidm_nfw" else "particles per bin"
    fig.text(
        0.085,
        0.09,
        f"Thin segments: 8–31 {count}; heavy segments: at least 32.",
        fontsize=14,
    )
    locality = fig.text(0.085, 0.04, "", fontsize=14)
    lines = []
    for col, ax in enumerate(axs):
        axes_style(ax)
        ax.set_xlim(-4, 3.03)
        ax.set_xlabel(r"$\log_{10}(r/r_{\rm s})$")
        ax.set_title("Density" if col == 0 else "Radial velocity dispersion", pad=13)
        ax.set_ylabel(
            r"$\log_{10}(\rho/\rho_{\rm s})$"
            if col == 0
            else r"$\log_{10}(\sigma_r^2/v_{\rm s}^2)$"
        )
        if col == 0:
            ax.set_ylim((-12.4, 8.4) if case == "sidm_nfw" else (-11.4, 6.1))
            ax.set_yticks([-10, -5, 0, 5])
        else:
            ax.set_ylim(-4, 0.55)
            ax.set_yticks([-4, -3, -2, -1, 0])
        l = []
        for lw, ls, color in [
            (1.25, "--", "k"),
            (3.0, "--", "k"),
            (1.25, "-", "#087cff"),
            (3.0, "-", "#087cff"),
        ]:
            l.append(ax.plot([], [], lw=lw, ls=ls, color=color)[0])
        lines.append(l)
    handles = [
        Line2D([], [], color="k", lw=3, ls="--", label="Initial state"),
        Line2D([], [], color="#087cff", lw=3, label="Current state"),
    ]
    legend = axs[0].legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        handlelength=3.0,
        labelspacing=0.3,
    )
    peak = axs[1].plot(
        [], [], "o", ms=6, mec="white", mew=0.8, color=plt.cm.plasma_r(0.97)
    )[0]

    def segment(z, col, threshold):
        """Select a population threshold and convert to logarithmic axes."""
        # The density conversion matches rho/rho_s in the reference normalization. Variance already uses v_s^2.
        y = 4 * np.pi * z["rho"] if col == 0 else z["variance"]
        good = (
            (z["r"] > 0)
            & (z["rho"] > 0)
            & (y > 0)
            & np.isfinite(y)
            & (z["population"] >= threshold)
        )
        return np.log10(z["r"]), np.where(good, np.log10(np.maximum(y, 1e-300)), np.nan)

    for col in range(2):
        # Draw the initial curves with the same 8/32 population thresholds as each evolving curve.
        for j, threshold in enumerate([8, 32]):
            lines[col][j].set_data(*segment(frames[0], col, threshold))

    def render(i):
        """Draw one saved SIDM state and its collision-locality measure."""
        z = frames[i]
        color = plt.cm.plasma_r(0.13 + 0.84 * i / max(1, len(frames) - 1))
        for col in range(2):
            for j, threshold in enumerate([8, 32]):
                lines[col][2 + j].set_data(*segment(z, col, threshold))
                lines[col][2 + j].set_color(color)
                lines[col][2 + j].set_visible(i > 0)
        legend.get_lines()[1].set_color(color)
        legend.get_lines()[1].set_visible(i > 0)
        legend.get_texts()[1].set_visible(i > 0)
        time_label.set_text(rf'$t/t_{{\rm s}}={z["actual_time"]:.5f}$')
        locality.set_text(
            "Maximum collision-shell width / mean free path: "
            + (f'{z["locality_ratio"]:.3g}' if i else "0")
        )
        # Mark the measured resolved outer dispersion maximum only in the final cored profile.
        if case == "sidm_cored" and i == len(frames) - 1:
            m = (z["r"] > 10) & (z["population"] >= 32) & (z["variance"] > 0)
            j = np.argmax(np.where(m, z["variance"], -np.inf))
            peak.set_data([np.log10(z["r"][j])], [np.log10(z["variance"][j])])
        else:
            peak.set_data([], [])
        return rgb(fig)

    return fig, frames, render


def encode(case, preview_only=False):
    """Render the final preview or encode every selected native state as MP4.

    The output frame ledger contains relative input paths, SHA256 hashes and
    physical times. A final hold repeats the last saved state.
    """
    # Cold VP diagnostics have their own physical-force and boundary-budget renderer.
    if case in ["cold_static_vp", "cold_expanding_vp"]:
        import importlib.util

        path = Path(__file__).with_name("render_cold_vp_movies.py")
        specification = importlib.util.spec_from_file_location(
            "venus_cold_movies", path
        )
        renderer = importlib.util.module_from_spec(specification)
        # Restore SIDM plotting settings after the independent cold renderer returns.
        with plt.rc_context():
            specification.loader.exec_module(renderer)
            renderer.encode(
                COLD_INPUT_DIRECTORY, OUTPUT_DIRECTORY, case, FFMPEG, 24, preview_only
            )
        return
    from PIL import Image

    fig, frames, render = make_sidm(case)
    if preview_only:
        final_image = render(len(frames) - 1)
        Image.fromarray(final_image).save(OUTPUT_DIRECTORY / f"{case}.png")
        plt.close(fig)
        print(f'{case}: final preview at native time {frames[-1]["actual_time"]:.12g}')
        return

    first_image = render(0)
    height, width = first_image.shape[:2]
    assert width % 2 == height % 2 == 0
    movie = OUTPUT_DIRECTORY / f"{case}.mp4"
    temporary = OUTPUT_DIRECTORY / f".{case}.tmp.mp4"
    argv = [
        str(FFMPEG),
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
        str(FPS),
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
    with subprocess.Popen(argv, stdin=subprocess.PIPE) as process:
        # Visit all native SIDM states, then repeat the final state for the declared viewing hold.
        sequence = list(range(len(frames))) + [len(frames) - 1] * (2 * FPS)
        for i in sequence:
            z = frames[i]
            process.stdin.write(render(i).tobytes())
            ledger.append(
                dict(
                    movie_frame=len(ledger),
                    movie_seconds=len(ledger) / FPS,
                    native_profile=z["local"],
                    native_sha256=z["sha256"],
                    actual_time=z["actual_time"],
                    native_cells=z["native_cells"],
                    step=z.get("step"),
                    locality_ratio=z.get("locality_ratio"),
                )
            )
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg could not encode the movie.")
    plt.close(fig)
    check = subprocess.run(
        [str(FFMPEG), "-v", "error", "-i", str(temporary), "-f", "null", "-"],
        capture_output=True,
    )
    if check.returncode or check.stderr:
        raise RuntimeError("Movie playback validation failed: " + check.stderr.decode())
    # Publish the movie after the full temporary stream decodes successfully.
    temporary.replace(movie)
    metadata = dict(
        case=case,
        fps=FPS,
        width=width,
        height=height,
        native_time_range=[frames[0]["actual_time"], frames[-1]["actual_time"]],
        clock="Native saved states, sampled more densely during rapid changes; the final state has a 2-second viewing hold.",
        encoder=dict(codec="H.264", pixel_format="yuv420p", quality_crf=18),
        frames=ledger,
    )
    (OUTPUT_DIRECTORY / f"{case}_frames.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"{case}: {len(ledger)} frames; native times "
        f'{frames[0]["actual_time"]:.12g} to {frames[-1]["actual_time"]:.12g}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--case",
        choices=["sidm_nfw", "sidm_cored", "cold_static_vp", "cold_expanding_vp"],
        help="Render one case; omit this option to render all four.",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=INPUT_DIRECTORY,
        help="Directory containing the public native inputs and manifests.",
    )
    parser.add_argument(
        "--cold-input-directory",
        type=Path,
        default=COLD_INPUT_DIRECTORY,
        help="Directory containing the cold VP profile catalog and inputs.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory for generated MP4s and frame ledgers.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Render the final PNG for each selected case.",
    )
    parser.add_argument(
        "--ffmpeg",
        help="FFmpeg executable; otherwise use VENUS_FFMPEG, PATH or imageio_ffmpeg.",
    )
    args = parser.parse_args()
    INPUT_DIRECTORY = args.input_directory.expanduser().resolve()
    COLD_INPUT_DIRECTORY = args.cold_input_directory.expanduser().resolve()
    OUTPUT_DIRECTORY = args.output_directory.expanduser().resolve()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if not args.preview_only:
        encoder = (
            args.ffmpeg or os.environ.get("VENUS_FFMPEG") or shutil.which("ffmpeg")
        )
        if encoder is None:
            try:
                import imageio_ffmpeg

                encoder = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                parser.error("Provide --ffmpeg or install ffmpeg/imageio_ffmpeg.")
        FFMPEG = Path(encoder).expanduser().resolve()
        if not FFMPEG.is_file():
            parser.error("The selected FFmpeg executable does not exist.")
    for case in (
        [args.case]
        if args.case
        else ["sidm_nfw", "sidm_cored", "cold_static_vp", "cold_expanding_vp"]
    ):
        encode(case, args.preview_only)
