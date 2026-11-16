#!/usr/bin/env python3
# Render expanding power-law halo density and logarithmic-slope movies.
# The manifest binds NPZ profiles, source clocks and finite-radius reference guides.
# Outputs are encoded movies, optional GIF/stills and a source-linked frame ledger.

"""Render saved halo density and signed-slope evolution from authenticated caches.

Example:
    python render_halo_movies.py --case g05_bp --input-directory inputs         --output-directory movies --gif

Inputs are movie_cache_manifest.json, reference_guides.json, and the referenced
NPZ cache plus JSON native-frame ledger. The NPZ stores each native radius grid,
density, effective count, signed 21-cell quadratic derivative, and display mask.
The renderer changes visual presentation only. Every distinct saved state is
shown in time order; the final hold repeats the same labeled state.
RGB frames stream directly into ffmpeg, so memory stays bounded. Supplying
--ffmpeg selects an encoder explicitly; PATH and imageio-ffmpeg are fallbacks.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time, shutil, tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'venus-movie-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import AutoMinorLocator
ROOT = Path(__file__).resolve().parent
FFMPEG = None
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def guides(ax, a, regimes):
    """Draw the retained example power-law references on their finite radial spans."""
    for name, z in regimes.items():
        # Draw a finite-range guide only when its catalog entry identifies it as a fitted or comparison reference.
        if not (z['guide_accepted'] or z.get('comparison_reference', False)):
            continue
        lo = z['log_r_min']
        hi = z['log_r_max']
        center = 0.5 * (lo + hi)
        target = z['reference_slope']
        offset = 1.1 if name != 'shoulder' else 0.75
        # Place the reference slope through the fitted radial band with a display offset that separates it from data.
        intercept = z['fit_intercept'] + (z['fitted_slope'] - target) * center + offset
        xx = np.linspace(lo, hi, 60)
        yy = target * xx + intercept
        ax.plot(xx, yy, color='.35', ls=(0, (5, 3)), lw=1.8, zorder=1)
        label = z.get('label', f'$r^{{{target:g}}}$')
        if z.get('comparison_reference', False):
            label += '\nreference'
        label_center = hi - 0.03 if z.get('comparison_reference', False) else center
        ax.text(z.get('label_x', label_center), z.get('label_y', target * center + intercept + 0.25), label, color='.25', ha='center', va='bottom', fontsize=19, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1, 'alpha': 0.85})

def main():
    """Authenticate cached curves and stream their chronological movie frames."""
    global ROOT, FFMPEG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, help='Case key in movie_cache_manifest.json.')
    p.add_argument('--input-directory', type=Path, default=ROOT.parent / 'data' / 'halos', help='Directory containing the manifest, cache/, and reference_guides.json.')
    p.add_argument('--output-directory', type=Path, default=ROOT.parent / 'generated' / 'halos', help='Destination for movies, frame ledgers or explicitly requested stills.')
    p.add_argument('--ffmpeg', help='Encoder executable. Otherwise use ffmpeg on PATH or imageio-ffmpeg.')
    p.add_argument('--gif', action='store_true', help='Also encode GIF from the completed MP4.')
    p.add_argument('--preview-only', action='store_true', help='Render three source frames without encoding a movie.')
    args = p.parse_args()
    INPUT = args.input_directory.resolve()
    ROOT = (args.output_directory or INPUT).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    FFMPEG = args.ffmpeg or shutil.which('ffmpeg')
    if not FFMPEG and (not args.preview_only):
        try:
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            p.error('Supply --ffmpeg or install imageio-ffmpeg.')
    manifest = json.loads((INPUT / 'movie_cache_manifest.json').read_text())
    case = next((r for r in manifest['cases'] if r['case'] == args.case))
    # Authenticate the reduced numerical arrays and their per-state source ledger together.
    cache = INPUT / case['cache']
    assert sha(cache) == case['cache_sha256']
    a = np.load(cache)
    source = json.loads((INPUT / case['frame_ledger']).read_text())
    assert sha(INPUT / case['frame_ledger']) == case['frame_ledger_sha256']
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 19, 'axes.labelsize': 27, 'axes.titlesize': 24, 'axes.linewidth': 1.8, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 21, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.2), dpi=120)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.81, wspace=0.23)
    mode = 'Vlasov–Poisson' if case['model'] == 'VP' else 'Boltzmann–Poisson'
    fig.text(0.5, 0.955, f"{mode} halo: $\\gamma_{{\\rm i}}={case['gamma_i']:g}$ (expanding background)", ha='center', va='center', fontsize=27)
    fig.text(0.08, 0.905, 'Collisionless' if case['model'] == 'VP' else 'Subhalo encounter strength $0.1$', ha='left', va='center', fontsize=22)
    label = fig.text(0.97, 0.905, '', ha='right', va='center', fontsize=27, animated=True)
    curves = []
    x0 = np.log10(a['radius'][0])
    # The cache owns the sampling decision. Use that same mask for initial density and slope curves.
    m = a['mask'][0]
    for col, ax in enumerate(axs):
        ax.set_xlim(-1, 2.05)
        ax.set_xlabel('$\\log_{10}r$')
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(which='major', length=9, width=1.7, pad=6)
        ax.tick_params(which='minor', length=5, width=1.2)
        ax.grid(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        y = np.log10(np.maximum(a['density'][0], np.finfo(float).tiny)) if col == 0 else a['slope'][0]
        ax.plot(x0, np.where(m, y, np.nan), color='.58', ls=':', lw=2.8, label='Initial profile', zorder=2)
        line, = ax.plot([], [], color=plt.get_cmap('plasma_r')(.05), lw=3.4, label='Current profile', animated=True, zorder=5)
        curves.append(line)
    axs[0].set_ylim(-9.2, 0.2)
    axs[0].set_yticks([-8, -6, -4, -2, 0])
    axs[0].set_ylabel('$\\log_{10}\\rho$')
    axs[0].set_title('Density', pad=12)
    density_legend=axs[0].legend(frameon=False, loc='lower left', fontsize=21, handlelength=2.8)
    density_legend.set_animated(True)
    axs[1].set_ylim(-4.65, 0.3)
    axs[1].set_yticks([-4, -3, -2, -1, 0])
    axs[1].set_ylabel('$\\mathrm{d}\\log\\rho/\\mathrm{d}\\log r$')
    axs[1].set_title('Logarithmic slope', pad=12)
    for ref, ls in [(-1, (0, (4, 4))), (-1.5, (0, (1, 3))), (-3, (0, (7, 3)))]:
        axs[1].axhline(ref, color='.5', lw=1.3, ls=ls, zorder=0)
        axs[1].text(1.97, ref + 0.06, f'${ref:g}$', color='.35', ha='right', va='bottom', fontsize=19, bbox=dict(facecolor='white', edgecolor='none', pad=0.5, alpha=0.9))
    regimes = json.loads((INPUT / 'reference_guides.json').read_text())[args.case]
    end = np.zeros(a['radius'].shape[1], dtype=[('radius', float), ('density', float), ('neff', float)])
    end['radius'] = a['radius'][-1]
    end['density'] = a['density'][-1]
    end['neff'] = a['neff'][-1]
    if args.case == 'g20_bp':
        regimes['inner'].update(label_x=0.08, label_y=-0.85)
    guides(axs[0], end, regimes)
    fig.canvas.draw()
    # Store the fixed axes once. Subsequent frames replace the moving curves and clock label.
    background = fig.canvas.copy_from_bbox(fig.bbox)
    width, height = fig.canvas.get_width_height()
    assert width % 2 == height % 2 == 0
    fps = 8 if args.case == 'g20_bp' else 24
    n = len(a['time'])
    repeats = np.ones(n, dtype=int)
    # Keep the final actual state visible by repeating its identical image at the same labeled time.
    repeats[-1] = round(2.4 * fps)
    preview = {0: 'initial', n // 2: 'middle', n - 1: 'final'}
    mp4 = ROOT / (args.case + '.mp4')
    ledger = []
    encoded = 0
    started = time.time()
    command = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1', '-movflags', '+faststart', str(mp4)]
    writer = None if args.preview_only else subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i, t in enumerate(a['time']):
            if args.preview_only and i not in preview:
                continue
            fig.canvas.restore_region(background)
            color=plt.get_cmap('plasma_r')(.05+.90*(float(t)-float(a['time'][0]))/(float(a['time'][-1])-float(a['time'][0])))
            for line in curves:line.set_color(color)
            density_legend.get_lines()[1].set_color(color)
            # Select this epoch's source mask and values without recomputing the physical profile.
            mask = a['mask'][i]
            x = np.log10(a['radius'][i])
            curves[0].set_data(x, np.where(mask, np.log10(np.maximum(a['density'][i], np.finfo(float).tiny)), np.nan))
            curves[1].set_data(x, np.where(mask, a['slope'][i], np.nan))
            label.set_text(f'$t={t:.8g}$')
            for ax, line in zip(axs, curves):
                ax.draw_artist(line)
            axs[0].draw_artist(density_legend)
            fig.draw_artist(label)
            if args.preview_only and i in preview:
                Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(ROOT / (args.case + '_' + preview[i] + '.png'))
            if writer:
                raw = bytes(fig.canvas.buffer_rgba())
                for _ in range(int(repeats[i])):
                    writer.stdin.write(raw)
            # Record the source file/hash/time alongside the first encoded frame and its repeat count.
            ledger.append(dict(native_index=i, actual_time=float(t), source_profile=source[i]['source'], source_sha256=source[i]['sha256'], encoded_first_frame=encoded, encoded_repeat_count=int(repeats[i]), display_duration_seconds=float(repeats[i] / fps)))
            encoded += int(repeats[i])
    finally:
        if writer:
            writer.stdin.close()
            assert writer.wait() == 0
        plt.close(fig)
    if args.preview_only:
        print(args.case, 'previews complete')
        return
    led = ROOT / (args.case + '_frames.json')
    led.write_text(json.dumps(ledger, indent=2) + '\n')
    artifacts = {mp4.name: dict(sha256=sha(mp4), bytes=mp4.stat().st_size)}
    # Build the optional GIF from the completed MP4 using a shared generated palette.
    if args.gif:
        palette = ROOT / (args.case + '_palette.png')
        gif = ROOT / (args.case + '.gif')
        subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(mp4), '-vf', 'palettegen=max_colors=128:stats_mode=diff', '-frames:v', '1', '-threads', '1', str(palette)], check=True)
        subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(mp4), '-i', str(palette), '-lavfi', 'paletteuse=dither=sierra2_4a', '-loop', '0', '-threads', '1', str(gif)], check=True)
        artifacts[gif.name] = dict(sha256=sha(gif), bytes=gif.stat().st_size)
    record = dict(case=case['case'], gamma_i=case['gamma_i'], model=case['model'], physical_time=[float(a['time'][0]), float(a['time'][-1])], native_distinct_frames=n, encoded_frames=encoded, fps=fps, mp4_duration_seconds=encoded / fps, pixel_size=[width, height], actual_nonuniform_time_steps=np.unique(np.round(np.diff(a['time']), 8)).tolist(), frame_ledger=led.name, frame_ledger_sha256=sha(led), cache_sha256=sha(cache), reference_guides_sha256=sha(INPUT / 'reference_guides.json'), renderer_sha256=sha(Path(__file__)), initial_reference='Gray dotted native t0 profile with same sampling mask.', current_curve='Native density and signed derivative; plasma_r(.05+.90*f), with f the elapsed fraction of actual saved time. Early yellow and late dark blue.', reference_guides='Gray example finite density guides and signed slope references -1,-1.5,-3.', axes=dict(log10r=[-1, 2.05], log10rho=[-9.2, 0.2], signed_slope=[-4.65, 0.3]), temporal_sampling='Every distinct native saved state, in time order. The final hold repeats the same labeled state; no interpolation.', history_scope='Density/slope movies; global virial histories have separate force/boundary budgets in the example analysis.', artifacts=artifacts, render_elapsed_seconds=time.time() - started)
    (ROOT / (args.case + '_movie.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
if __name__ == '__main__':
    main()
