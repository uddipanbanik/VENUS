#!/usr/bin/env python3
# Render stored relativistic collapse profiles and contemporaneous metric histories.
# Inputs are checksum-bound NPZ arrays with each coordinate system's saved time and step.
# Use G=c=1. Output MP4/GIF frames keep their actual source times and apparent-horizon diagnostics.

"""Render black-hole collapse movies from supplied native-profile plotting arrays.

The EF case evolves all 65,536 particles through the apparent horizon. Its
endpoint v=32.49976078148166 is the resolved interval shown in the example figures.
The Schwarzschild case reaches coordinate time160. Each displayed time is an
actual saved state; the final hold repeats that state. All quantities use
G=c=1. See ../README.md and the manual for the density observers and metric.

Run from any directory; --input-directory and --output-directory override the
repository defaults. Install imageio-ffmpeg or supply --ffmpeg to encode MP4.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'venus-movie-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, GifImagePlugin

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#087cff'
FPS = 10


def sha(path):
    """Identify a supplied input or exported movie by SHA256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(case, record, directory):
    """Load lossless plotting arrays and their actual contemporaneous history."""
    path = directory / record['file']
    if sha(path) != record['sha256']:
        raise ValueError(f'Plotting input checksum differs: {path}')
    with np.load(path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    # Keep EF advanced time and Schwarzschild coordinate time in separate data/plot branches.
    ef = case.startswith('ef')
    assert np.all(np.diff(a['time']) > 0)
    assert np.all(np.diff(a['step']) > 0)
    assert a['radius'].shape == a['density'].shape == a['compactness'].shape
    assert np.all(np.diff(a['radius'], axis=1) > 0)
    assert np.isfinite(a['radius']).all() and np.isfinite(a['density']).all()
    assert np.isfinite(a['compactness']).all() and np.all(a['density'] >= 0)
    data = dict(case=case, ef=ef, time=a['time'], step=a['step'],
                radius=a['radius'], density=a['density'], compactness=a['compactness'])
    # Separate diagnostics joined directly to profile frames from the denser independent history columns.
    data['frame_history'] = {k[6:]:v for k,v in a.items() if k.startswith('frame_')}
    data['history'] = {k[8:]:v for k,v in a.items() if k.startswith('history_')}
    return data


def style(ax):
    """Apply the example figures' inward ticks and grid-free axes."""
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=8, width=1.5, pad=5)
    ax.tick_params(which='minor', length=4, width=1.1)
    ax.grid(False)


def scene(data):
    """Plot contemporaneous density, compactness and coordinate-time histories."""
    ef = data['ef']
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
        'font.size': 18, 'axes.labelsize': 22, 'axes.titlesize': 21,
        'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 16,
        'axes.linewidth': 1.6, 'lines.linewidth': 3.0})
    fig, axes = plt.subplots(1 if ef else 2, 2,
                             figsize=(14.4, 6.5 if ef else 10.1), dpi=100)
    fig.subplots_adjust(left=.085, right=.975, bottom=.20 if ef else .14,
                        top=.84 if ef else .885, wspace=.30, hspace=.38)
    axes = np.asarray(axes).reshape(-1)
    top = fig.suptitle('', y=.965, fontsize=23)
    footer = fig.text(.52, .025, '', ha='center', va='bottom', fontsize=16)
    for ax in axes:
        style(ax)
    radius = data['radius'][0]
    rho0 = data['density'][0]
    # This renderer's relative-density display mask follows each snapshot peak; the cached density remains unchanged.
    good = rho0 > rho0.max() * 1e-7
    axes[0].plot(np.log10(radius[good]), np.log10(rho0[good]),
                 color='.5', lw=1.7, ls=(0, (5, 3)), label='Initial shell')
    density_line, = axes[0].plot([], [], color=BLUE, label='Evolving shell')
    axes[0].set(xlim=(-.35, .7), xlabel=r'$\log_{10} r$',
                ylabel=r'$\log_{10}\rho_u$' if ef else r'$\log_{10}\rho$',
                title='Energy density during collapse')
    visible = ((data['radius'] > 10**(-.35)) & (data['radius'] < 10**.7))
    ymax = .3 if ef else max(.3, np.log10(data['density'][visible].max()) + .25)
    axes[0].set_ylim(-7.5, ymax)
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, loc='lower left', handlelength=2.8)
    axes[1].plot(radius, data['compactness'][0], color='.5', lw=1.7, ls=(0, (5, 3)))
    compact_line, = axes[1].plot([], [], color=BLUE)
    axes[1].axhline(1, color='.3', lw=1.8, ls=(0, (5, 3)), label=r'$2m/r=1$')
    axes[1].set(xlim=(0, 4.7), ylim=(0, 1.4 if ef else 1.025),
                xlabel=r'Areal radius $r$', ylabel=r'$2m/r$',
                title='Compactness during collapse')
    axes[1].legend(frameon=False, loc='upper right', handlelength=2.8)
    ah_density = axes[0].axvline(0, color=BLUE, ls=(0, (5, 3)), lw=2,
                                visible=False, label='Apparent horizon')
    ah_compact = axes[1].axvline(0, color=BLUE, ls=(0, (5, 3)), lw=2, visible=False)
    horizon_legend_visible = [False]
    history_lines = None
    if not ef:
        qline, = axes[2].plot([], [], color=BLUE)
        lapse_line, = axes[3].plot([], [], color='#e42c28')
        axes[2].axhline(1, color='.3', ls=(0, (5, 3)), lw=1.6)
        axes[2].set(xlim=(0, 160), ylim=(.45, 1.025),
                    xlabel=r'Schwarzschild time $\tau$', ylabel=r'$\max_r(2m/r)$',
                    title='Maximum compactness')
        axes[3].set(xlim=(0, 160), ylim=(-5.5, 0),
                    xlabel=r'Schwarzschild time $\tau$', ylabel=r'$\log_{10}(\min_r e^\mu)$',
                    title='Decrease of the lapse')
        history_lines = qline, lapse_line

    def render(i):
        t = data['time'][i]
        color=plt.get_cmap('plasma_r')(.05+.90*(float(t)-float(data['time'][0]))/(float(data['time'][-1])-float(data['time'][0])))
        for artist in [density_line,compact_line,ah_density,ah_compact,*list(history_lines or [])]:artist.set_color(color)
        r, rho = data['radius'][i], data['density'][i]
        # Apply the same relative display mask to the current profile before taking its logarithm.
        mask = rho > rho.max() * 1e-7
        density_line.set_data(np.log10(r[mask]), np.log10(rho[mask]))
        compact_line.set_data(r, data['compactness'][i])
        if ef:
            row = SimpleNamespace(**{k: v[i] for k, v in data['frame_history'].items()})
            # Use the saved apparent-horizon radius to position guides. Its positivity controls their visibility.
            horizon = float(row.apparent_horizon_radius)
            for artist, x in [(ah_density, np.log10(horizon) if horizon > 0 else 0),
                              (ah_compact, horizon)]:
                artist.set_visible(horizon > 0)
                artist.set_xdata([x, x])
            if horizon > 0 and not horizon_legend_visible[0]:
                axes[0].legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, loc='lower left', handlelength=2.8)
                horizon_legend_visible[0] = True
            elif horizon <= 0 and horizon_legend_visible[0]:
                handles, labels = axes[0].get_legend_handles_labels()
                axes[0].legend(handles[:2], labels[:2], frameon=True, facecolor='white', edgecolor='none', framealpha=1,
                               loc='lower left', handlelength=2.8)
                horizon_legend_visible[0] = False
            top.set_text(rf'Black-hole formation in EF coordinates: $v={t:.4f}$')
            fraction = row.active_trapped_rest_weight / row.total_rest_weight
            footer.set_text(f'Trapped rest mass: {100*fraction:.2f}%')
        else:
            h = data['history']
            # Reveal Schwarzschild history only through the step associated with the current saved profile.
            cut = h['step'] <= data['step'][i]
            history_lines[0].set_data(h['time'][cut], h['max_compactness'][cut])
            history_lines[1].set_data(h['time'][cut], np.log10(h['min_lapse'][cut]))
            top.set_text(rf'Black-hole formation in Schwarzschild coordinates: $\tau={t:.2f}$')
            footer.set_text('')
        for axis in axes:
            legend=axis.get_legend()
            if legend is not None:
                for handle,text in zip(legend.get_lines(),legend.get_texts()):
                    if text.get_text() in ['Evolving shell','Apparent horizon']:handle.set_color(color)
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    return fig, render


def encode(data, output, ffmpeg, gif=False, preview_only=False):
    """Stream actual saved states to the encoder with bounded memory use."""
    fig, render = scene(data)
    count = len(data['time'])
    selected = {0, count // 2, count - 1}
    if data['ef']:
        # Include the first saved trapped-state frame among the exported scientific previews.
        selected.add(int(np.flatnonzero(data['frame_history']['apparent_horizon_radius'] > 0)[0]))
    stem = 'black_hole_ef' if data['ef'] else 'black_hole_schwarzschild'
    if preview_only:
        for index in sorted(selected):
            render(index).save(output / f'{stem}_frame{index:04d}.png')
        plt.close(fig)
        return
    height = 650 if data['ef'] else 1010
    palette = None
    if gif:
        canvas = Image.new('RGB', (1440, height * 3), 'white')
        for j, i in enumerate([0, count // 2, count - 1]):
            canvas.paste(render(i), (0, j * height))
        palette = canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    movie = output / (stem + '.mp4')
    command = [ffmpeg, '-y', '-v', 'error', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'1440x{height}',
               '-framerate', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-threads', '1', '-movflags', '+faststart', str(movie)]
    durations = []
    gif_stream = (output / (stem + '.gif')).open('wb') if gif else None
    with (output / (stem + '_encoding.log')).open('wb') as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i in range(count):
                image = render(i)
                # The last frame has a longer viewing duration while preserving the same actual physical epoch.
                duration = 2400 if i == count - 1 else 100
                durations.append(duration)
                if gif_stream:
                    frame = image.quantize(palette=palette, dither=Image.Dither.NONE)
                    if i == 0:
                        header, _ = GifImagePlugin.getheader(frame, info={'loop':0, 'optimize':False})
                        for block in header:
                            gif_stream.write(block)
                    for block in GifImagePlugin.getdata(frame, duration=duration,
                                                       disposal=2, include_color_table=False):
                        gif_stream.write(block)
                # Holds affect viewing duration; physical time remains in the title.
                rgb = image.tobytes()
                for _ in range(duration * FPS // 1000):
                    encoder.stdin.write(rgb)
            if gif_stream:
                gif_stream.write(b';')
        finally:
            encoder.stdin.close()
            if gif_stream:
                gif_stream.close()
            plt.close(fig)
        if encoder.wait() != 0:
            raise RuntimeError(f'Encoder failed; see {log.name}')
    # Store the actual times and step numbers together with viewing durations and the movie digest.
    (output / (stem + '_frames.json')).write_text(json.dumps({
        'actual_time':data['time'].tolist(), 'native_step':data['step'].tolist(),
        'duration_ms':durations, 'movie_sha256':sha(movie),
        'temporal_interpolation':'none'}, indent=2) + '\n')
    print(f'{movie}: {count} native states, {sum(durations)/1000:g} seconds')


def main():
    """Parse portable paths and render either or both coordinate descriptions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all','ef','schwarzschild'], default='all')
    parser.add_argument('--input-directory', type=Path, default=ROOT/'data/relativity')
    parser.add_argument('--output-directory', type=Path, default=ROOT/'generated/relativity')
    parser.add_argument('--ffmpeg', help='Encoder executable; defaults to PATH or imageio-ffmpeg.')
    parser.add_argument('--gif', action='store_true', help='Also produce an animated GIF.')
    parser.add_argument('--preview-only', action='store_true', help='Export selected still frames only.')
    args = parser.parse_args()
    ffmpeg = args.ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg and not args.preview_only:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            parser.error('Install imageio-ffmpeg or supply --ffmpeg.')
    manifest = json.loads((args.input_directory/'manifest.json').read_text())
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for case, record in manifest['cases'].items():
        if args.case != 'all' and case.startswith('ef') != (args.case == 'ef'):
            continue
        encode(prepare(case, record, args.input_directory), args.output_directory,
               ffmpeg, args.gif, args.preview_only)


if __name__ == '__main__':
    main()
