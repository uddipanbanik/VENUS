#!/usr/bin/env python3
# Render g05 BP radial, phase-space and shifted-energy figures from reduced native data.
# Inputs preserve actual state clocks, physical velocity u and phase-volume normalization.
# Output figures and numerical receipts expose sampling and central-potential sensitivity separately.

"""Render the completed gamma=.5 BP density, phase, and energy figures.

Inputs are authenticated reduced native outputs. Defaults are relative to this
script. NumPy, Matplotlib, and pandas are required. The numerical simulation is
never invoked. Existing figure files in the selected output directory are
replaced only for these three figure stems.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'venus-g05-example-mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import reproduce_figures as paper_figures

TIMES = (0, 30, 300, 2000, 5000, 10000)


def sha(path):
    """Authenticate an immutable reduced input before plotting it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style(ax):
    """Apply inward major/minor ticks and grid-free axes."""
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=8, width=1.6, pad=5)
    ax.tick_params(which='minor', length=4, width=1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(False)


def save(fig, output, stem):
    """Export matching PDF and PNG content with a compact outer margin."""
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'{stem}.{extension}', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def load_caches(data_dir, family):
    """Check every cache hash and select the six common requested times."""
    manifest = json.loads((data_dir/family/'analysis_manifest.json').read_text())
    selected = []
    for record in manifest['snapshots']:
        path = data_dir/family/record['cache_file']
        if sha(path) != record['cache_sha256']:
            raise ValueError(f'Cache checksum mismatch: {path}')
        if record['requested_time'] in TIMES:
            selected.append((record, dict(np.load(path))))
    if tuple(r['requested_time'] for r, _ in selected) != TIMES:
        raise ValueError('The six common snapshot times are required.')
    return manifest, selected


def energy_curve(ax, a, color, key):
    """Distinguish finite-bin population and central-field sensitivity."""
    x = np.log10(a['delta_center'])
    value = a[key]
    # Positive values with N_eff>=8 define visible energy bins. Dashed pieces mark sensitivity to the inner potential continuation.
    present = (value > 0) & (a['neff_energy'] >= 8)
    central = a['central_continuation_bin'] | (a['inner_phase_volume_fraction'] > .01)
    connected_values = []
    for keep, width, dash in [
            (present & ~central, 1.15, '-'),
            (present & central, 1.7, (0, (5, 3))),
            (present & ~central & (a['neff_energy'] >= 32), 2.9, '-')]:
        y = np.where(keep, np.log10(np.maximum(value, np.finfo(float).tiny)), np.nan)
        ax.plot(x, y, color=color, lw=width, ls=dash)
        connected = keep & (np.r_[False, keep[:-1]] | np.r_[keep[1:], False])
        connected_values.extend(np.log10(value[connected]).tolist())
    return connected_values


def energy_slope(a, window=9):
    """Differentiate a local quadratic over supported log-energy bins.

    All bins in the window must have positive distribution and Neff >= 8.
    A thick segment requires Neff >= 32 throughout the window. A window
    touching the central potential continuation is displayed with dashes.
    """
    x = np.log10(a['delta_center'])
    value = a['mean_f_energy']
    present = (value > 0) & (a['neff_energy'] >= 8)
    central = a['central_continuation_bin'] | (a['inner_phase_volume_fraction'] > .01)
    derivative = np.full(len(x), np.nan)
    strong = np.zeros(len(x), dtype=bool)
    sensitive = np.zeros(len(x), dtype=bool)
    half = window // 2
    for j in range(half, len(x)-half):
        selected = slice(j-half, j+half+1)
        if np.all(present[selected]):
            # The coefficient linear in centered log energy gives the derivative at the central bin.
            coefficients = np.polynomial.polynomial.polyfit(
                x[selected]-x[j], np.log10(value[selected]), 2)
            derivative[j] = coefficients[1]
            # A heavy derivative requires sufficient effective population throughout the complete fitting window.
            strong[j] = np.all(a['neff_energy'][selected] >= 32)
            sensitive[j] = np.any(central[selected])
    return derivative, strong, sensitive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent/'data')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent.parent/'figures')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((args.data_dir/'catalog_figure2.json').read_text())
    paper_figures.verify_inputs(args.data_dir, catalog)
    paper_figures.render_halo_plate(args.data_dir, args.output_dir, catalog,
        [('g03', .3), ('g05', .5), ('g08', .8)], 'halo_multigamma_density_slopes_shallow')
    pm, phase = load_caches(args.data_dir, 'phase_through10000')
    em, energy = load_caches(args.data_dir, 'shifted_through10000')
    # Phase and energy reductions must describe outputs of the same evolution executable.
    if pm['binary_sha256'] != em['binary_sha256']:
        raise ValueError('Phase and energy caches require the same executable.')
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
        'font.size': 18, 'axes.labelsize': 23, 'axes.titlesize': 22,
        'axes.linewidth': 1.7, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
        'legend.fontsize': 16, 'lines.linewidth': 2.9, 'savefig.dpi': 190,
        'pdf.fonttype': 42})
    # Assign each snapshot its consistent plasma time color.
    all_times = [r['requested_time'] for r in em['snapshots']]
    color_by_time = {all_times[0]: 'black'}
    color_by_time.update({t: plt.cm.plasma_r(q) for t, q in
                         zip(all_times[1:], np.linspace(.12, .995, len(all_times)-1))})
    colors = [color_by_time[t] for t in TIMES]
    receipt = {'requested_times': TIMES, 'phase': [], 'energy': [],
        'time_colors': {str(t): matplotlib.colors.to_hex(c) for t, c in zip(TIMES, colors)},
        'phase_manifest_sha256': sha(args.data_dir/'phase_through10000/analysis_manifest.json'),
        'energy_manifest_sha256': sha(args.data_dir/'shifted_through10000/analysis_manifest.json')}

    # True radial widths enter the cached density despite the logarithmic axis.
    fig = plt.figure(figsize=(16.2, 8.4), layout='constrained')
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .045])
    vmax = 1.; vmin = -7.
    cmap = plt.colormaps['plasma'].copy(); cmap.set_bad('white')
    for i, (record, a) in enumerate(phase):
        ax = fig.add_subplot(grid[i//3, i%3])
        value = a['f_integrated_eta']
        # Convert positive projected distribution values to log10 while leaving empty bins without color.
        plotted = np.where(value > 0, np.log10(np.maximum(value, np.finfo(float).tiny)), np.nan)
        mesh = ax.pcolormesh(a['xedge'], a['uedge'], plotted.T, cmap=cmap,
            vmin=vmin, vmax=vmax, rasterized=True, shading='flat')
        ax.axhline(0, color='white', lw=.8, alpha=.65)
        ax.axvline(math.log10(energy[i][0]['first_field_face']), color='.65', lw=1.3, ls=(0, (5, 3)))
        ax.set_xlabel(r'$\log_{10}r$'); ax.set_ylabel(r'$u$')
        ax.set_title(rf'$t={record["requested_time"]:g}$', pad=9); style(ax)
        # Integrate with physical dr rather than d(log r), restoring the spherical 8*pi^2 phase measure.
        reconstructed_mass = np.sum(value * (8*np.pi**2*np.diff(10**a['xedge'])[:, None]*np.diff(a['uedge'])[None, :]))
        receipt['phase'].append({'time': record['requested_time'],
            'total_mass': record['total_mass'], 'integrated_projected_mass': float(reconstructed_mass),
            'mass_relative_error': float(abs(reconstructed_mass/record['total_mass']-1)),
            'velocity_range': a['uedge'][[0, -1]].tolist()})
    cax = fig.add_subplot(grid[:, 3]); bar = fig.colorbar(mesh, cax=cax, extend='min')
    bar.set_label(r'$\log_{10}[\int f\,\mathrm{d}\eta]$', fontsize=23, labelpad=12)
    bar.ax.tick_params(labelsize=17, width=1.5, length=7)
    save(fig, args.output_dir, 'g05_bp_phase_distribution')

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), layout='constrained')
    values = []
    for (record, a), color in zip(energy, colors):
        values.extend(energy_curve(axes[0], a, color, 'mean_f_energy'))
        derivative, strong, sensitive = energy_slope(a)
        finite = np.isfinite(derivative)
        for keep, width, dash in [
                (finite & ~sensitive, 1.15, '-'),
                (finite & sensitive, 1.7, (0, (5, 3))),
                (finite & strong & ~sensitive, 2.9, '-')]:
            axes[1].plot(np.log10(a['delta_center']),
                np.where(keep, derivative, np.nan), color=color, lw=width, ls=dash)
    for ax in axes:
        ax.set_xlabel(r'$\log_{10}(E-\Phi_{\rm c})$')
        ax.set_xlim(np.log10(energy[0][1]['delta_edge'][[0, -1]]))
        style(ax)
    axes[0].set_ylabel(r'$\log_{10}\langle f\rangle_E$')
    axes[0].set_ylim(math.floor(min(values))-.25, math.ceil(max(values))+.25)
    axes[1].set_ylabel(r'$\mathrm{d}\ln\langle f\rangle_E/\mathrm{d}\ln(E-\Phi_{\rm c})$')
    axes[1].set_ylim(-8, 1)
    for level, label in [(-4.5, r'$\rho$ slope $-1.5$'), (-2.5, r'$\rho$ slope $-1$')]:
        axes[1].axhline(level, color='.45', lw=1.5, ls=(0, (5, 3)), zorder=0)
        axes[1].text(.03, (level+8)/9+.018, label, transform=axes[1].transAxes,
                     ha='left', va='bottom', color='.3', fontsize=16)
    axes[0].legend(handles=[Line2D([], [], color=c, lw=2.9, label=rf'$t={t:g}$')
        for t, c in zip(TIMES, colors)], frameon=False, ncol=2, fontsize=15,
        handlelength=2.2, loc='lower left')
    for record, a in energy:
        # Cross-check energy-spectrum mass and phase-volume times mean-f against each snapshot's total mass.
        receipt['energy'].append({'time': record['requested_time'],
            'total_mass': record['total_mass'],
            'spectrum_mass': float(np.sum(a['dM_ddelta']*np.diff(a['delta_edge']))),
            'phase_mean_mass': float(np.sum(a['mean_f_energy']*a['bin_phase_volume']))})
    save(fig, args.output_dir, 'g05_bp_energy_distribution')
    receipt['renderer_sha256'] = sha(__file__)
    receipt['figures'] = {p.name: sha(p) for p in args.output_dir.glob('*')
        if p.stem in ['halo_multigamma_density_slopes_shallow', 'g05_bp_phase_distribution', 'g05_bp_energy_distribution']}
    (args.output_dir/'render_metadata.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
