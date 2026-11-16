#!/usr/bin/env python3
# Render Gaussian VP/BP density profiles and their signed logarithmic slopes.
# Inputs are catalog-bound compressed native CSV tables in code units.
# Output PDF/PNG panels preserve all positive radial support and distinguish N_eff>=32 samples.

"""Render the full positive radial support of Gaussian VP and BP profiles.

All curves use saved physical times. Density is the spherical shell-volume
average. Signed density slopes use a centered quadratic fit over 21 cells.
The radial axis spans all positive support. The displayed ordinates focus on
log10 density from -15 to 1 and signed logarithmic slopes from -10 to 5.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'venus-gaussian-vp-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np


def sha(path):
    """Return the complete digest of one portable cached input."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(base, record, compressed=False):
    """Authenticate one table before converting its numerical columns."""
    path = base/record['file']
    assert sha(path) == record['sha256']
    # Decompress only after checking the portable file digest. Parse columns by their saved names.
    text = gzip.decompress(path.read_bytes()).decode() if compressed else path.read_text()
    return np.atleast_1d(np.genfromtxt(io.StringIO(text),names=True,delimiter=','))


def slope(a, n=21):
    """Differentiate a local quadratic fit in log10(radius), using 21 native bins.

    The ten bins at each boundary have no centered estimate. Every density
    value in a fitting window must be positive.
    """
    x = np.log10(a['radius'])
    y = np.log10(np.maximum(a['density'], np.finfo(float).tiny))
    dy = np.full(len(x), np.nan)
    ys = np.full(len(x), np.nan)
    half = n // 2
    for j in range(half, len(x) - half):
        # The 21-cell window is centered on j. Every source density in it must be positive for log fitting.
        sel = slice(j - half, j + half + 1)
        if np.any(a['density'][sel] <= 0):
            continue
        # Center log radius on this cell. The linear polynomial coefficient is its logarithmic slope.
        p = np.polynomial.polynomial.polyfit(x[sel] - x[j], y[sel], 2)
        dy[j] = p[1]
        ys[j] = p[0]
    return (dy, ys)


def style_axis(ax):
    """Apply the example figures' inward ticks and grid-free axes."""
    ax.grid(False)
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.tick_params(which='major',length=8,width=1.5,pad=5)
    ax.tick_params(which='minor',length=4,width=1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))


def save(fig, out, stem, receipt):
    """Check visible text bounds and save both requested scientific formats."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend([ax.title,ax.xaxis.label,ax.yaxis.label])
        for axis, limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            artists.extend(label for value,label in zip(axis.get_majorticklocs(),axis.get_majorticklabels())
                           if limits[0] <= value <= limits[1])
    for legend in fig.legends:
        artists.extend(legend.get_texts())
    outside = []
    for text in artists:
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.x1+1 or bounds.y1 > fig.bbox.y1+1:
                outside.append(text.get_text())
    assert not outside, outside
    for extension in ['pdf','png']:
        path = out/(stem+'.'+extension)
        fig.savefig(path)
        receipt['artifacts'][path.name] = sha(path)
    plt.close(fig)


def padded_limits(values, fraction, minimum_padding):
    """Enclose every finite plotted value with a visible margin."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    assert len(values)
    lower = float(values.min())
    upper = float(values.max())
    padding = max(minimum_padding, fraction * (upper - lower))
    return [lower - padding, upper + padding]


def render(base, out):
    """Plot actual native times with complete radial extent and focused ordinates."""
    out.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((base / 'catalog.json').read_text())
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.size': 18,
        'axes.labelsize': 23,
        'axes.titlesize': 20,
        'axes.linewidth': 1.7,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 17,
        'lines.linewidth': 2.5,
        'pdf.fonttype': 42,
        'savefig.dpi': 200,
    })
    requested = catalog.get('profile_times', [0, 10, 30, 100, 300, 800, 2000, 5000, 10000])
    cached = {}
    all_x = []
    all_density = []
    all_slope = []
    rows = []
    # Read each case at its own listed actual epochs while enforcing the common source format.
    for name, record in catalog['cases'].items():
        requested = record.get('profile_times', requested)
        assert [x['nominal_time'] for x in record['profiles']] == requested
        for binding in record['profiles']:
            a = load(base, binding, True)
            assert len(a) == binding['rows'] == catalog.get('radial_cells', 709)
            assert np.all(np.isfinite(a['radius'])) and np.all(a['radius'] > 0)
            assert np.all(np.isfinite(a['density'])) and np.all(a['density'] >= 0)
            assert binding['actual_time'] == binding['nominal_time']
            x = np.log10(a['radius'])
            positive = a['density'] > 0
            y = np.full(len(a), np.nan)
            y[positive] = np.log10(a['density'][positive])
            derivative, _ = slope(a)
            finite_slope = positive & np.isfinite(derivative)
            # Effective count controls the heavy-curve encoding independently of density positivity or slope support.
            reliable = positive & (a['neff'] >= 32)
            cached[name, binding['nominal_time']] = (
                a, x, y, derivative, positive, reliable, finite_slope
            )
            # Accumulate positive support across all cases so the shared radial limits include every plotted cell.
            all_x.extend(x[positive])
            all_density.extend(y[positive])
            all_slope.extend(derivative[finite_slope])
            rows.append({
                'case': name,
                'actual_time': binding['actual_time'],
                'file': binding['file'],
                'sha256': binding['sha256'],
                'native_rows': len(a),
                'positive_density_bins': int(positive.sum()),
                'neff_ge_32_positive_bins': int(reliable.sum()),
                'finite_centered_slope_bins': int(finite_slope.sum()),
                'finite_neff_ge_32_slope_bins': int((finite_slope & reliable).sum()),
                'positive_log_radius_range': [float(x[positive].min()), float(x[positive].max())],
                'positive_log_density_range': [float(y[positive].min()), float(y[positive].max())],
                'finite_slope_range': [
                    float(derivative[finite_slope].min()),
                    float(derivative[finite_slope].max()),
                ],
            })
    limits = {
        'log_radius': catalog.get('log_radius_limits', padded_limits(all_x, .015, .1)),
        'log_density': [-15.0, 1.0],
        'signed_slope': [-10.0, 5.0],
    }
    ncase = len(catalog['cases'])
    fig, axes = plt.subplots(ncase, 2, figsize=(12, 9.2 if ncase == 2 else 14.3), sharex=True, sharey='col', squeeze=False)
    fig.subplots_adjust(left=.105, right=.97, bottom=.10 if ncase==2 else .07, top=.875 if ncase==2 else .90, wspace=.32, hspace=.90 if ncase==2 else .65)
    for row, (name, record) in enumerate(catalog['cases'].items()):
        handles = []
        requested = record.get('profile_times', catalog.get('profile_times', requested))
        # Order each case's actual times from early yellow to late dark blue.
        colors = plt.get_cmap('plasma_r')(np.linspace(.05, .95, len(requested)))
        for ordinal, time in enumerate(requested):
            color = colors[ordinal]
            line_style = '--' if ordinal == 0 else '-'
            # Use the same saved density, derivative and sampling masks in both columns of this row.
            a, x, y, derivative, positive, reliable, finite_slope = cached[name, time]
            # The point marks also expose isolated occupied cells.
            axes[row, 0].plot(
                x, np.where(positive, y, np.nan), color=color, ls=line_style,
                lw=.85, alpha=.60, marker='.', markersize=2, zorder=2,
            )
            line, = axes[row, 0].plot(
                x, np.where(reliable, y, np.nan), color=color,
                label=rf'${time:g}$', ls=line_style, lw=2.5, zorder=3,
            )
            axes[row, 1].plot(
                x, np.where(finite_slope, derivative, np.nan), color=color, ls=line_style,
                lw=.85, alpha=.60, marker='.', markersize=2, zorder=2,
            )
            # Heavy slope segments require both a complete positive fitting window and the effective-count threshold.
            axes[row, 1].plot(
                x, np.where(finite_slope & reliable, derivative, np.nan),
                color=color, ls=line_style, lw=2.5, zorder=3,
            )
            handles.append(line)
        top = axes[row,0].get_position().y1
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, top+(.055 if ncase==2 else .040)),
                   ncol=len(handles), frameon=False, handlelength=1.5, columnspacing=.8, handletextpad=.35,
                   fontsize=16)
        titles = {
            'static_vp': 'Vlasov–Poisson (static background)',
            'expanding_vp': 'Vlasov–Poisson (expanding background)',
            'static_bp': 'Boltzmann–Poisson (static background)',
            'expanding_bp': 'Boltzmann–Poisson (expanding background)',
        }
        fig.text(.535, top+(.073 if ncase==2 else .052), titles[name],ha='center',va='bottom',fontsize=20)
        axes[row, 0].set(ylabel=r'$\log_{10}\rho$', ylim=limits['log_density'])
        axes[row, 1].set(ylabel=r'$d\log\rho/d\log r$', ylim=limits['signed_slope'])
        # Slope references are horizontal comparisons. They do not alter the source derivative values.
        for value,xfrac in [(-3.,.97),(-1.5,.90),(-1.,.97)]:
            axes[row,1].axhline(value,color='.58',ls='--',lw=1.2,zorder=.5)
            axes[row,1].annotate(rf'${value:g}$',xy=(xfrac,value),
                xycoords=axes[row,1].get_yaxis_transform(),xytext=(0,-1),
                textcoords='offset points',ha='right',va='bottom',fontsize=15,
                color='.30',zorder=7)
    for ax in axes.flat:
        style_axis(ax)
        ax.set_xlim(limits['log_radius'])
        ax.xaxis.set_major_locator(MultipleLocator(2))
    for ax in axes[-1]:
        ax.set_xlabel(r'$\log_{10}r$')
    receipt = {
        'renderer_sha256': sha(__file__),
        'catalog_sha256': sha(base / 'catalog.json'),
        'artifacts': {},
        'ordered_time_colormap': 'plasma_r, early yellow to late dark blue',
        'slope_guides': [-3.0,-1.5,-1.0],
        'profile_times_by_case': {k:v.get('profile_times',catalog.get('profile_times')) for k,v in catalog['cases'].items()},
        'profiles': rows,
        'limits': limits,
        'scope': 'All positive native density bins and every finite centered21-cell slope are supplied to the artists over the complete radial support of the saved profiles. Ordinates display log10 density from -15 to 1 and slopes from -10 to 5.',
        'ordinate_clipping': True,
        'ordinate_display': 'Density outside [10^-15,10^1] and slopes outside [-10,5] lie outside the visible ordinates. All source values and masks are retained.',
        'abscissa_clipping': False,
        'positive_bin_markers': 'Small points accompany the thin curves, including isolated occupied cells.',
        'sampling_encoding': 'Thin positive density curves. Thick curves use Neff>=32. Slopes require21 positive adjacent native densities.',
        'unit_convention': 'Code units, unchanged from the input catalog.',
        'text_bounds': 'PASS',
    }
    for values, key in [(all_x, 'log_radius')]:
        lo, hi = limits[key]
        assert np.all((np.asarray(values) > lo) & (np.asarray(values) < hi))
    receipt['outside_visible_ordinates'] = {
        'positive_density_samples': int(np.count_nonzero((np.asarray(all_density) < limits['log_density'][0]) | (np.asarray(all_density) > limits['log_density'][1]))),
        'finite_slope_samples': int(np.count_nonzero((np.asarray(all_slope) < limits['signed_slope'][0]) | (np.asarray(all_slope) > limits['signed_slope'][1]))),
    }
    # Save the figure after radial-support and visible-text checks, then record its numerical input bindings.
    save(fig, out, 'gaussian_vp_full_radius_density_slopes', receipt)
    (out / 'plot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'artifacts': receipt['artifacts'], 'limits': limits}, indent=2))


def main():
    """Read the portable authenticated tables and render a new output pair."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data/gaussian_vp')
    parser.add_argument('--output-dir', type=Path, default=root / 'reproduced')
    args = parser.parse_args()
    render(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()

