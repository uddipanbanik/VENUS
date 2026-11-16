#!/usr/bin/env python3
# Reveal saved physical virial measurements on a shared actual-time clock.
# Inputs are eight catalog-bound moment tables. Q uses each table's stated force and population definition.
# Outputs are an MP4 and a ledger recording exactly which samples each frame reveals.

"""Animate the eight physical virial datasets used by the example summary.

The movie reveals actual saved samples through each displayed input clock.
Three BP and one VP expanding-halo curves use resolved profile moments.
Gaussian curves use emitted all-marker physical moments. The purple line
joins the five converged static self-bound samples, using each subset's
own gravitational field. The source catalog records every definition.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'venus-eight-virial-movie'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
from PIL import Image

STYLES = [
    ('g03_bp', '#087cff', '-', r'BP, expanding: $\gamma_i=0.3$'),
    ('g05_bp', '#f02c33', '-', r'BP, expanding: $\gamma_i=0.5$'),
    ('g08_bp', '#00a651', '-', r'BP, expanding: $\gamma_i=0.8$'),
    ('g05_vp', 'black', (0, (6, 3)), r'VP, expanding: $\gamma_i=0.5$'),
    ('gaussian_static_vp', '#b57300', (0, (4, 2, 1, 2)), 'VP, Gaussian: static'),
    ('gaussian_expanding_vp', '#00a8b8', (0, (2, 2)), 'VP, Gaussian: expanding'),
    ('static_g05_bound', '#8b2bc0', '-', r'VP, static: bound $\gamma_i=0.5$'),
    ('gaussian_static_bound','#b57300','None','VP, Gaussian: bound static'),
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_curves(base):
    """Recompute the accepted physical definitions from exact cached moments."""
    catalog = json.loads((base/'catalog.json').read_text())
    require(set(catalog['sources']) == {s[0] for s in STYLES}, 'Eight-case source inventory differs')
    curves = {}
    for name, color, style, label in STYLES:
        source = catalog['sources'][name]
        path = base/Path(source['file']).name
        require(sha(path) == source['sha256'], 'Source table SHA differs')
        data = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=',', dtype=None, encoding='utf-8'))
        if name == 'static_g05_bound':
            # Use only converged iterative own-potential memberships for the static power-law bound population.
            stable = data['status'] == 'STABLE_MEMBERSHIP'
            require(np.array_equal(data['time'][~stable], [0,10,30,100]), 'Unresolved static times differ')
            data = data[stable]
            require(np.array_equal(data['time'], [300,800,2000,5000,10000]), 'Stable static times differ')
            q = np.asarray(data['Q_stable'], dtype=float)
        elif name.startswith('gaussian_'):
            # For all-marker Gaussian data, add the physical background acceleration times inertia to W_Phi.
            force = data['W_force'] + (data['H_dot'] + data['H']**2)*data['I']
            q = 2*data['K_physical']/np.abs(force)
            require(np.max(np.abs(q-data['Q_physical'])) < 1e-13, 'Gaussian physical Q differs')
        else:
            # The power-law tables use resolved profile quadratures with their recorded background contribution.
            force = data['W_resolved_constant_line_with_inner_mass'] + data['background_acceleration']*data['I_resolved_center']
            q = 2*data['K_physical_resolved_center']/np.abs(force)
            require(np.max(np.abs(q-data['Q_physical_profile_total_acceleration'])) < 1e-13, 'Profile physical Q differs')
        clock = np.asarray(data['time'], dtype=float)
        require(np.all(np.diff(clock)>0) and clock[-1] == 10000, 'Saved curve chronology differs')
        require(np.all(np.isfinite(q)) and np.all(q>0), 'Invalid physical Q')
        curves[name] = {'time':clock, 'Q':q, 'source_sha256':source['sha256']}
    # The reveal clock is the sorted union of actual measured epochs from all eight datasets.
    clock = np.unique(np.concatenate([v['time'] for v in curves.values()]))
    require(clock[0] == 0 and clock[-1] == 10000, 'Reveal clock endpoints differ')
    return catalog, curves, clock


def main():
    """Render one panel with unchanged scientific samples and a native clock."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory', type=Path, default=here.parents[1]/'analysis/data/virial_summary')
    parser.add_argument('--output-directory', type=Path, default=here.parent/'generated/halos')
    parser.add_argument('--ffmpeg', help='Encoder executable, with PATH as fallback.')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    base = args.input_directory.resolve()
    out = args.output_directory.resolve()
    out.mkdir(parents=True, exist_ok=True)
    catalog, data, clock = load_curves(base)
    encoder = args.ffmpeg or shutil.which('ffmpeg')
    if encoder is None and not args.preview_only:
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            parser.error('Supply --ffmpeg or install imageio-ffmpeg.')
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':19, 'axes.labelsize':27, 'axes.titlesize':25, 'axes.linewidth':1.8, 'xtick.labelsize':20, 'ytick.labelsize':20, 'legend.fontsize':19, 'lines.linewidth':3.1})
    fig, ax = plt.subplots(figsize=(16,7.2), dpi=120)
    fig.subplots_adjust(left=.075, right=.97, bottom=.245, top=.685)
    handles = []
    current_markers=[]
    for name, color, style, label in STYLES:
        options = {'zorder':6} if name == 'static_g05_bound' else ({'marker':'o','markersize':6.5} if name.startswith('gaussian_') else {'marker':'o', 'markersize':4, 'markevery':[-1]})
        # The Gaussian bound population supplies one own-gravity endpoint, drawn as an open square.
        if name=='gaussian_static_bound':options={'marker':'s','markersize':11,'markerfacecolor':'none','markeredgewidth':2.2,'zorder':8}
        line, = ax.plot([], [], color=color, ls=style, label=label, animated=True, **options)
        handles.append(line)
        current_markers.append(ax.plot([],[],marker='o',ms=7,ls='None',color=plt.get_cmap('plasma_r')(.05),animated=True,zorder=10)[0])
    ax.axhline(1, color='.45', lw=1.5, ls=':', zorder=0)
    ax.set(title='Physical virial balance', xlabel=r'$\log_{10}(1+t)$', ylabel=r'$Q$', xlim=(0,4.055), ylim=(0,2.05))
    ax.grid(False)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=9, width=1.7, pad=6)
    ax.tick_params(which='minor', length=5, width=1.2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.505,.995), ncol=3, frameon=False, fontsize=19, handlelength=2.8, columnspacing=1.1, labelspacing=.65)
    label = fig.text(.97,.765,'',ha='right',va='center',fontsize=27,animated=True)
    fig.text(.5,.07,r'$Q=2K_{\rm physical}/|W_\Phi+(\dot H+H^2)I|$',ha='center',fontsize=21)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*legend.get_texts(), ax.title, ax.xaxis.label, ax.yaxis.label]:
        box = text.get_window_extent(renderer)
        require(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, 'Text extends beyond canvas')
    background = fig.canvas.copy_from_bbox(fig.bbox)
    width, height = fig.canvas.get_width_height()
    fps = 24
    repeats = np.ones(len(clock), dtype=int)
    repeats[-1] = round(2.4*fps)
    movie = out/'halo_virialization.mp4'
    writer = None
    if not args.preview_only:
        command = [encoder,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(movie)]
        writer = subprocess.Popen(command,stdin=subprocess.PIPE)
    ledger = []
    encoded = 0
    previews = {0:'initial', int(np.searchsorted(clock,5000)):'middle', len(clock)-1:'final'}
    started = time.monotonic()
    try:
        for index,current in enumerate(clock):
            if args.preview_only and index not in previews:
                continue
            fig.canvas.restore_region(background)
            scope = {}
            time_color=plt.get_cmap('plasma_r')(.05+.90*(float(current)-float(clock[0]))/(float(clock[-1])-float(clock[0])))
            for (name, unused_color, unused_style, unused_label), line in zip(STYLES,handles):
                source = data[name]
                # Reveal samples through the displayed clock. Each curve keeps its original measured epochs.
                count = int(np.searchsorted(source['time'],current,side='right'))
                line.set_data(np.log10(1+source['time'][:count]),source['Q'][:count])
                ax.draw_artist(line)
                marker=current_markers[len(scope)]
                marker.set_data([np.log10(1+source['time'][count-1])],[source['Q'][count-1]])if count else marker.set_data([],[])
                marker.set_color(time_color)
                # The two bound selections retain their distinct sampling symbols/line style and receive no moving cursor.
                if name not in ['static_g05_bound','gaussian_static_bound']:ax.draw_artist(marker)
                scope[name] = {'points':count, 'last_actual_time':float(source['time'][count-1]) if count else None}
            label.set_text(f'$t={current:.10g}$')
            fig.draw_artist(label)
            if index in previews:
                Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(out/f'halo_virialization_{previews[index]}.png')
            if writer:
                raw = bytes(fig.canvas.buffer_rgba())
                for unused in range(int(repeats[index])):
                    writer.stdin.write(raw)
            # Save the last actual epoch and number of visible values for each population in every encoded frame.
            ledger.append({'native_reveal_index':index,'actual_reveal_time':float(current),'curves':scope,'encoded_first_frame':encoded,'encoded_repeat_count':int(repeats[index])})
            encoded += int(repeats[index])
    finally:
        if writer:
            writer.stdin.close()
            require(writer.wait(timeout=60) == 0, 'Encoder failed')
        plt.close(fig)
    if args.preview_only:
        print(json.dumps({'status':'PREVIEWS_RENDERED','native_reveal_frames':len(clock)}))
        return
    path = out/'halo_virialization_frames.json'
    path.write_text(json.dumps(ledger,indent=2)+'\n')
    record = {'case':'halo_virialization','renderer_sha256':sha(__file__),'source_catalog_sha256':sha(base/'catalog.json'),'sources':{name:{'rows':len(values['time']),'sha256':values['source_sha256'],'first_time':float(values['time'][0]),'last_time':float(values['time'][-1]),'Q_final':float(values['Q'][-1])} for name,values in data.items()},'native_reveal_frames':len(clock),'encoded_frames':encoded,'fps':fps,'duration_seconds':encoded/fps,'pixel_size':[width,height],'frame_ledger_sha256':sha(path),'artifacts':{movie.name:{'sha256':sha(movie),'bytes':movie.stat().st_size}},'time_scope':'Sorted union of actual accepted source clocks. Every curve reveals only samples at or before the displayed clock. Initial state appears once and final state is held 2.4 seconds.','bound_scope':'The purple line joins five stable own-potential selections. A brown square shows the single Gaussian static E_full<0 endpoint using its own gravity. Each selected population has its stated membership; symbols have no connecting lines.','force_scope':'Four power-law curves use resolved profile quadratures. Gaussian curves use emitted all-marker applied-force moments. Each static bound point uses its own reconstructed gravitational field.','current_sample_encoding':'Power-law curves mark the latest revealed sample. Gaussian circles show all revealed native snapshots. The open brown square appears only at time10000.','render_seconds':time.monotonic()-started}
    (out/'halo_virialization_movie.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
