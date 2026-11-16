# Regenerate halo, SIDM and relativistic example panels from compressed native tables.
# Input catalog.json records file identities, units, physical controls and saved epochs.
# Outputs are selected PDF/PNG figures. Numerical evolution remains separate from this rendering code.

"""Render VENUS example figures from reduced native output.

Each renderer loads only the fields used by its figure. Density slopes are
computed from the retained native radial bins. All lengths, times, velocity
moments, sampling cuts and source hashes are described in data/catalog.json.
Run with --help for paths and figure selection. Importing this module draws
nothing; figures are produced only through main().
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'venus-example-matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

def read_table(data_dir, filename):
    """Read round-trip decimal CSV values into a named NumPy array."""
    import csv
    path = data_dir / Path(filename).relative_to('data')
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = list(rows[0])
    # Parse each retained decimal through Python float before constructing named float64 columns.
    return np.array([tuple((float(row[k]) for k in names)) for row in rows], dtype=[(name, 'f8') for name in names])

def read_frame(data_dir, filename):
    """Load all-float reduced columns with Python's exact float parser."""
    array = read_table(data_dir, filename)
    return pd.DataFrame({key: array[key] for key in array.dtype.names})

def save(fig, output_dir, stem, **kwargs):
    """Write both shareable formats, keeping the example panel geometry."""
    for extension in ('pdf', 'png'):
        fig.savefig(output_dir / (stem + '.' + extension), **kwargs)
    plt.close(fig)

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
        sel = slice(j - half, j + half + 1)
        if np.any(a['density'][sel] <= 0):
            continue
        # The centered quadratic coefficient p[1] estimates d(log rho)/d(log r) at the current cell.
        p = np.polynomial.polynomial.polyfit(x[sel] - x[j], y[sel], 2)
        dy[j] = p[1]
        ys[j] = p[0]
    return (dy, ys)

def matching_regimes(a):
    """Choose finite radial spans for the conventional power-law guides.

    Fits have a free slope and use bins with effective population at least 32.
    A guide requires a slope within 0.22 (0.25 for the outer shoulder), density
    residual at most 0.07 dex, and derivative RMS at most 0.4 from the reference.
    The gamma_i=0.8 intermediate guide is also shown as an explicit reference.
    """
    x = np.log10(a['radius'])
    dy, _ = slope(a)
    rho = a['density']
    out = {}
    for name, target, width, limits, tol in [('inner', -1.5, 0.5, (-0.78, 0.15), 0.22), ('intermediate', -1.0, 0.45, (-0.05, 0.9), 0.22), ('shoulder', -3.0, 0.6, (0.5, 2.05), 0.25)]:
        candidates = []
        for lo in np.arange(limits[0], limits[1] - width + 0.0001, 1 / 64):
            hi = lo + width
            # Fit only the finite radial band's positive, sufficiently populated cells with available centered slopes.
            m = (x >= lo) & (x <= hi) & (rho > 0) & (a['neff'] >= 32) & np.isfinite(dy)
            if m.sum() < 0.8 * width * 64:
                continue
            # Fit the slope freely before comparing it with a reference. The reference exponent is not imposed on density data.
            c = np.polyfit(x[m], np.log10(rho[m]), 1)
            residual = np.log10(rho[m]) - np.polyval(c, x[m])
            rms = float(np.sqrt(np.mean(residual ** 2)))
            slope_rms = float(np.sqrt(np.mean((dy[m] - target) ** 2)))
            candidates.append((slope_rms, {'reference_slope': target, 'log_r_min': float(lo), 'log_r_max': float(hi), 'radius_min': float(10 ** lo), 'radius_max': float(10 ** hi), 'width_dex': width, 'bins': int(m.sum()), 'fitted_slope': float(c[0]), 'rms_log10_density': rms, 'rms_derivative_from_reference': slope_rms, 'minimum_neff': float(np.min(a['neff'][m])), 'fit_intercept': float(c[1]), 'guide_accepted': bool(abs(c[0] - target) <= tol and rms <= 0.07 and (slope_rms <= 0.4))}))
        # Choose the available band with the smallest derivative deviation and retain its separate acceptance flag.
        out[name] = min(candidates, key=lambda z: z[0])[1] if candidates else {'guide_accepted': False, 'reason': 'insufficient resolved bins in finite window'}
    return out

def halo_style(ax):
    """Apply the density-figure inward ticks and minor subdivisions."""
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=9, width=1.7, pad=6)
    ax.tick_params(which='minor', length=5, width=1.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(False)

def draw(axs, a, color, lw, ls, label):
    """Show native estimates thin, with Neff >= 32 sections thick.

    Density requires a positive native value. Slopes use finite
    centered estimates. The thicker sections retain the displayed-center
    effective-population threshold used for the quantitative comparisons.
    """
    x = np.log10(a['radius'])
    dy, _ = slope(a)
    positive = a['density'] > 0
    # Positive native values remain visible thin. Effective count at the displayed cell controls heavy segments.
    mask = positive & (a['neff'] >= 32)
    density = np.log10(np.maximum(a['density'], 1e-300))
    axs[0].plot(x, np.where(positive, density, np.nan), color=color, lw=0.9, ls=ls, zorder=1.5)
    axs[1].plot(x, np.where(np.isfinite(dy), dy, np.nan), color=color, lw=0.9, ls=ls, zorder=1.5)
    axs[0].plot(x, np.where(mask, density, np.nan), color=color, lw=lw, ls=ls, label=label, zorder=4 if lw > 3 else 2)
    axs[1].plot(x, np.where(mask, dy, np.nan), color=color, lw=lw, ls=ls, zorder=4 if lw > 3 else 2)

def guides(ax, a, regimes):
    """Offset each finite-range reference above its fitted final density."""
    for name, z in regimes.items():
        if not (z['guide_accepted'] or z.get('comparison_reference', False)):
            continue
        lo = z['log_r_min']
        hi = z['log_r_max']
        center = 0.5 * (lo + hi)
        target = z['reference_slope']
        offset = 1.1 if name != 'shoulder' else 0.75
        intercept = z['fit_intercept'] + (z['fitted_slope'] - target) * center + offset
        xx = np.linspace(lo, hi, 60)
        yy = target * xx + intercept
        ax.plot(xx, yy, color='.35', ls=(0, (5, 3)), lw=1.8, zorder=1)
        label = z.get('label', f'$r^{{{target:g}}}$')
        if z.get('comparison_reference', False):
            label += '\nreference'
        label_center = hi - 0.03 if z.get('comparison_reference', False) else center
        ax.text(z.get('label_x', label_center), z.get('label_y', target * center + intercept + 0.25), label, color='.25', ha='center', va='bottom', fontsize=19, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1, 'alpha': 0.85})

def halo_profiles(data_dir, catalog, case):
    """Return the exact saved snapshots selected for this halo case."""
    return [(item['time'], item['file'], read_table(data_dir, item['file'])) for item in catalog['halo_profiles'][case]]

def render_halo_plate(data_dir, output_dir, catalog, selected, stem):
    """Density on the left; a 21-bin logarithmic derivative on the right."""
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 18, 'axes.labelsize': 24, 'axes.titlesize': 22, 'axes.linewidth': 1.8, 'xtick.labelsize': 19, 'ytick.labelsize': 19, 'legend.fontsize': 15, 'lines.linewidth': 2.6, 'savefig.dpi': 220, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(len(selected), 2, figsize=(13.4, 3.55 * len(selected)), sharex=True, sharey='col', layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.065, hspace=0.06)
    for row, (tag, gamma) in enumerate(selected):
        ps = halo_profiles(data_dir, catalog, tag + '_bp')
        end = ps[-1][0]
        times = [0, 100, 800, end] if end > 5000 else [0, 50, 200, end]
        shown = []
        for j, target in enumerate(times):
            # Choose an actual saved epoch nearest the requested representative time and label its actual value.
            t, _, a = min(ps, key=lambda z: abs(z[0] - target))
            if t in shown:
                continue
            shown.append(t)
            color = 'k' if t == 0 else plt.cm.plasma_r(0.13 + 0.84 * (j - 1) / max(1, len(times) - 2))
            draw(axs[row], a, color, 3.4 if t == end else 2.8 if t == 0 else 2.25, ':' if t == 0 else '-', f'$t={t:g}$')
        a = ps[-1][2]
        regs = matching_regimes(a)
        if tag == 'g08':
            regs['intermediate']['comparison_reference'] = True
        if tag == 'g20':
            regs['inner'].update(label_x=0.08, label_y=-0.85)
        guides(axs[row, 0], a, regs)
        for col in range(2):
            ax = axs[row, col]
            halo_style(ax)
            ax.set_xlim(-1, 2.05)
            ax.text(0.975, 0.94, f'$\\gamma_{{\\rm i}}={gamma:g},\\ A_{{\\rm coll}}=0.1$', transform=ax.transAxes, ha='right', va='top', fontsize=20, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2, 'alpha': 0.9})
            if row == len(selected) - 1:
                ax.set_xlabel('$\\log_{10}r$')
        axs[row, 0].set_ylim(-9.2, 0.2)
        axs[row, 0].set_yticks([-8, -6, -4, -2, 0])
        axs[row, 0].set_ylabel('$\\log_{10}\\rho$')
        axs[row, 0].legend(loc='lower left', ncol=2, frameon=True, facecolor='white', framealpha=1, edgecolor='0.7', handlelength=2.8, columnspacing=0.8, labelspacing=0.25, borderpad=0.35, fontsize=16)
        ax = axs[row, 1]
        ax.set_ylim(-4.65, 0.3)
        ax.set_yticks([-4, -3, -2, -1, 0])
        ax.set_ylabel('$\\mathrm{d}\\log\\rho/\\mathrm{d}\\log r$')
        for ref, ls in [(-1, (0, (4, 4))), (-1.5, (0, (1, 3))), (-3, (0, (7, 3)))]:
            ax.axhline(ref, color='.5', lw=1.3, ls=ls, zorder=0)
            ax.text(1.98, ref + 0.06, f'${ref:g}$', color='.35', ha='right', va='bottom', fontsize=16, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.4, 'alpha': 0.9})
    axs[0, 0].set_title('Density evolution', pad=12)
    axs[0, 1].set_title('Measured logarithmic slope', pad=12)
    save(fig, output_dir, stem, bbox_inches='tight', pad_inches=0.04)

def virial_style(ax):
    """Apply the virial-figure tick lengths and minor subdivisions."""
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=9, width=1.7, pad=6)
    ax.tick_params(which='minor', length=5, width=1.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(False)

def render_virial(data_dir, output_dir, catalog):
    """Show one physical virial estimator for each halo calculation."""
    from render_virial_summary import render
    render(Path(data_dir) / 'virial_summary', output_dir)


def render_bp_vp(data_dir, output_dir, catalog):
    """Show BP and VP snapshots with identical density and slope estimators."""
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 18, 'axes.labelsize': 24, 'axes.titlesize': 22, 'axes.linewidth': 1.8, 'xtick.labelsize': 19, 'ytick.labelsize': 19, 'legend.fontsize': 15, 'lines.linewidth': 2.6, 'savefig.dpi': 220, 'pdf.fonttype': 42})
    TIMES = [0, 100, 800, 10000]
    endpoint = {model: halo_profiles(data_dir, catalog, 'g05_' + model)[-1][2] for model in ['bp', 'vp']}
    fig, axs = plt.subplots(2, 2, figsize=(13.4, 8.15), sharex=True, sharey='col', layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06, wspace=0.065, hspace=0.09)
    for row, model in enumerate(['bp', 'vp']):
        shown = []
        for j, t in enumerate(TIMES):
            _, p, a = min(halo_profiles(data_dir, catalog, 'g05_' + model),
                          key=lambda item: abs(item[0] - t))
            # The BP/VP comparison requires its listed common epochs to be present within the clock tolerance.
            assert abs(float(a['time'][0]) - t) < 1e-06
            color = 'k' if t == 0 else plt.cm.plasma_r([0.13, 0.55, 0.97][j - 1])
            draw(axs[row], a, color, 3.4 if t == 10000 else 2.5, ':' if t == 0 else '-', f'$t={t:g}$')
            shown.append(p)
        regs = matching_regimes(endpoint[model])
        guides(axs[row, 0], endpoint[model], regs)
        title = 'Boltzmann–Poisson: $A_{\\rm coll}=0.1$' if model == 'bp' else 'Vlasov–Poisson: $A_{\\rm coll}=0$'
        for col in range(2):
            ax = axs[row, col]
            halo_style(ax)
            ax.set_xlim(-1, 2.05)
            if col == 0:
                ax.text(0.975, 0.95, title, transform=ax.transAxes, ha='right', va='top', fontsize=19, bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.95))
            if row == 1:
                ax.set_xlabel('$\\log_{10}r$')
        axs[row, 0].set_ylim(-9.2, 0.2)
        axs[row, 0].set_yticks([-8, -6, -4, -2, 0])
        axs[row, 0].set_ylabel('$\\log_{10}\\rho$')
        axs[row, 0].legend(loc='lower left', ncol=2, frameon=True, facecolor='white', framealpha=1, edgecolor='0.7', handlelength=2.8, columnspacing=1.2, labelspacing=0.3, borderpad=0.4, fontsize=17)
        ax = axs[row, 1]
        ax.set_ylim(-4.65, 0.3)
        ax.set_yticks([-4, -3, -2, -1, 0])
        ax.set_ylabel('$\\mathrm{d}\\log\\rho/\\mathrm{d}\\log r$')
        for ref, ls in [(-1, (0, (4, 4))), (-1.5, (0, (1, 3))), (-3, (0, (7, 3)))]:
            ax.axhline(ref, color='.5', lw=1.3, ls=ls, zorder=0)
            ax.text(1.97, ref + 0.06, f'${ref:g}$', color='.35', ha='right', va='bottom', fontsize=16, bbox=dict(facecolor='white', edgecolor='none', pad=0.4, alpha=0.9))
    axs[0, 0].set_title(r'Density evolution, $\gamma_{\rm i}=0.5$', pad=12)
    axs[0, 1].set_title('Measured logarithmic slope', pad=12)
    save(fig, output_dir, 'halo_g05_bp_vp_density_slopes', bbox_inches='tight', pad_inches=0.04)

def render_cold(data_dir, output_dir, catalog):
    """Render both cold backgrounds using the physical virial force."""
    from render_cold_vp_virial import render
    return render(data_dir, output_dir, catalog)


def sidm_style(ax, minor=True):
    """Use the same inward ticks in all SIDM panels."""
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=8.5, width=1.7, pad=5)
    ax.tick_params(which='minor', length=4.5, width=1.1)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

def render_sidm(data_dir, output_dir, catalog):
    """Density, centered radial variance and native-bin slopes for SIDM."""
    families = []
    for family in ['nfw', 'cored']:
        curves = []
        for item in catalog['sidm_profiles'][family]:
            a = read_table(data_dir, item['file'])
            # Apply the same native 21-cell derivative to both SIDM density families.
            derivative, _ = slope(a)
            curves.append(dict(r=a['radius'], rho=a['density'], variance=a['variance'], population=a['population'], slope=derivative, time=item['time']))
        families.append(curves)
    nfw, cored = families
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 18, 'axes.labelsize': 25, 'axes.titlesize': 23, 'axes.linewidth': 1.8, 'xtick.labelsize': 19, 'ytick.labelsize': 19, 'legend.fontsize': 17, 'savefig.dpi': 250, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(13.7, 10.5), layout='constrained')
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, .52, 1.0, .52])
    density_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[2, 0])]
    slope_axes = [fig.add_subplot(grid[1, 0], sharex=density_axes[0]), fig.add_subplot(grid[3, 0], sharex=density_axes[1])]
    variance_axes = [fig.add_subplot(grid[:2, 1]), fig.add_subplot(grid[2:, 1])]
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.035, wspace=0.065, hspace=0.04)
    for row, curves in enumerate([nfw, cored]):
        dax, sax, vax = density_axes[row], slope_axes[row], variance_axes[row]
        for j, z in enumerate(curves):
            color = 'k' if j == 0 else plt.cm.plasma_r(0.13 + 0.84 * (j - 1) / (len(curves) - 2))
            ls = '--' if j == 0 else '-'
            label = '$t=0$' if j == 0 else f"$t={z['time']:.3f}$" if z['time'] > 193 else f"$t={z['time']:.2f}$"
            # Existing density/variance units, data and population cuts are exact.
            x = np.log10(z['r'])
            # Convert code density to rho/rho_s; radial variance already uses the velocity-scale normalization.
            for ax, y in [(dax, 4*np.pi*z['rho']), (vax, z['variance'])]:
                valid = (z['r'] > 0) & (z['rho'] > 0) & (y > 0) & np.isfinite(y)
                # The two population thresholds encode sampling support for density and velocity dispersion separately.
                thin = valid & (z['population'] >= 8)
                thick = valid & (z['population'] >= 32)
                yy = np.full(len(y), np.nan)
                yy[thin] = np.log10(y[thin])
                ax.plot(x, yy, color=color, lw=1.25, ls=ls, zorder=2)
                yy[~thick] = np.nan
                ax.plot(x, yy, color=color, lw=3.0, ls=ls, label=label if ax is dax else None, zorder=3)
            finite = np.isfinite(z['slope']) & (z['rho'] > 0)
            sax.plot(x, np.where(finite & (z['population'] >= 8), z['slope'], np.nan), color=color, lw=1.25, ls=ls, zorder=2)
            sax.plot(x, np.where(finite & (z['population'] >= 32), z['slope'], np.nan), color=color, lw=3.0, ls=ls, zorder=3)
        dax.legend(loc='lower left', frameon=False, ncol=2, handlelength=2.8, columnspacing=1.1, labelspacing=0.3, borderpad=0.15)
        dax.text(0.97, 0.98, 'NFWc30: weighted PIC' if row == 0 else 'Nearly cored: spherical DSMC', transform=dax.transAxes, ha='right', va='top', fontsize=18)
        for ax, panel in [(dax, 'a' if row == 0 else 'd'), (vax, 'b' if row == 0 else 'e'), (sax, 'c' if row == 0 else 'f')]:
            sidm_style(ax)
            ax.set_xlim(-4, 3.03)
            ax.text(.965, .78 if ax is dax else .94, '('+panel+')', transform=ax.transAxes, ha='right', va='top', fontsize=20)
        dax.tick_params(labelbottom=False)
        sax.set_xlabel('$\\log_{10}(r/r_{\\rm s})$')
        vax.set_xlabel('$\\log_{10}(r/r_{\\rm s})$')
        dax.set_ylabel('$\\log_{10}(\\rho/\\rho_{\\rm s})$')
        vax.set_ylabel('$\\log_{10}(\\sigma_r^2/v_{\\rm s}^2)$')
        sax.set_ylabel('$\\mathrm{d}\\log\\rho/\\mathrm{d}\\log r$', fontsize=21)
        sax.set_ylim(-5.5, .5)
        sax.set_yticks([-4, -2, 0])
        # This horizontal line is the published long-mean-free-path self-similar reference, not a fitted data exponent.
        sax.axhline(-2.19, color='.42', lw=1.4, ls='--', zorder=.5)
        sax.text(.76, -2.19+.12, '$-2.19$', transform=sax.get_yaxis_transform(), color='.3', fontsize=17, ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='none', pad=.2, alpha=.9))
    density_axes[0].set_ylim(-12.4, 8.4)
    density_axes[0].set_yticks([-10, -5, 0, 5])
    density_axes[1].set_ylim(-11.4, 6.1)
    density_axes[1].set_yticks([-10, -5, 0, 5])
    for ax in variance_axes:
        ax.set_ylim(-4.0, .55)
        ax.set_yticks([-4, -3, -2, -1, 0])
    density_axes[0].set_title('Density profile', pad=12)
    variance_axes[0].set_title('Radial velocity dispersion', pad=12)
    peak = cored[-1]
    i = int(np.argmax(np.where(peak['r'] > 10, peak['variance'], 0)))
    variance_axes[1].plot(np.log10(peak['r'][i]), np.log10(peak['variance'][i]), 'o', ms=5.5, color=plt.cm.plasma_r(.97), markeredgecolor='white', markeredgewidth=.8, zorder=6)
    save(fig, output_dir, 'sidm_collapse_and_outer_dispersion', bbox_inches='tight', pad_inches=.045)

def gr_style(ax):
    """Use common tick marks for the two relativistic figures."""
    ax.tick_params(which='both', direction='in', top=True, right=True, width=1.5, length=7)
    ax.tick_params(which='minor', length=3.8, width=1.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def render_bh_coordinates(data_dir, output_dir, catalog):
    """Show horizon approach, horizon crossing, and retained invariant weight."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 16, 'axes.labelsize': 19, 'axes.titlesize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'axes.linewidth': 1.6, 'lines.linewidth': 3.2, 'legend.fontsize': 14, 'savefig.bbox': 'tight'})
    paths = catalog['relativistic']
    sc = read_frame(data_dir, paths['schwarzschild_history'])
    ef = read_frame(data_dir, paths['ef_history'])
    # Normalize retained and trapped rest weights by the same initial total represented rest mass.
    initial_weight = float(ef.total_rest_weight.iloc[0])
    vh = float(ef.loc[ef.apparent_horizon_radius > 0, 'advanced_time'].iloc[0])
    last = float(ef.advanced_time.iloc[-1])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(sc.time, sc.max_compactness, color='#0072ff')
    ax.axhline(1, color='black', ls='--', lw=2.0)
    ax.set(xlabel='Schwarzschild time $\\tau$', ylabel='$\\max_r\\,2m/r$', title='Horizon approach in Schwarzschild coordinates', xlim=(0, 160), ylim=(0.45, 1.025))
    ax.text(0.05, 0.86, '(a)', transform=ax.transAxes)
    ax = axes[1, 0]
    ax.plot(sc.time, np.log10(sc.min_lapse), color='#d62728')
    ax.set(xlabel='Schwarzschild time $\\tau$', ylabel='$\\log_{10}(\\min_r e^{\\mu})$', title='Slowing of Schwarzschild time', xlim=(0, 160))
    ax.text(0.05, 0.82, '(c)', transform=ax.transAxes)
    ax = axes[0, 1]
    good = ef.advanced_time >= 28
    ah = good & (ef.apparent_horizon_radius > 0)
    ax.plot(ef.loc[ah, 'advanced_time'], ef.loc[ah, 'apparent_horizon_radius'], color='black', label='Apparent horizon')
    ax.plot(ef.loc[good, 'advanced_time'], ef.loc[good, 'active_min_radius'], color='#0072ff', label='Innermost particle')
    ax.axvline(vh, ymax=0.72, color='.6', lw=1.8, ls=':')
    ax.set(xlabel='Advanced time $v$', ylabel='Areal radius', title='Matter enters the trapped region', xlim=(28, last), ylim=(0, 2.5))
    ax.legend(frameon=False, loc='upper right')
    ax.text(0.05, 0.92, '(b)', transform=ax.transAxes)
    ax = axes[1, 1]
    for column, label, color in [('active_rest_weight', 'Evolved matter', '#0072ff'), ('active_trapped_rest_weight', 'Trapped matter', '#18a74b')]:
        ax.plot(ef.loc[good, 'advanced_time'], ef.loc[good, column] / initial_weight, color=color, label=label)
    ax.axvline(vh, color='.6', lw=1.8, ls=':')
    ax.set(xlabel='Advanced time $v$', ylabel='Fraction of initial rest mass', title='Matter retained through horizon formation', xlim=(28, last), ylim=(-0.035, 1.08))
    ax.legend(frameon=False, loc='center left', handlelength=2.8)
    ax.text(0.05, 0.88, '(d)', transform=ax.transAxes)
    ax.text(0.05, 0.1, '65,536 particles retained', transform=ax.transAxes, fontsize=14)
    for ax in axes.flat:
        gr_style(ax)
    for extension in ('pdf', 'png'):
        fig.savefig(output_dir / f'bh_coordinates.{extension}', dpi=190)
    plt.close(fig)

def render_bh_interior(data_dir, output_dir, catalog):
    """Plot ten resolved EF slices and the horizon at each saved time."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 16, 'axes.labelsize': 19, 'axes.titlesize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'axes.linewidth': 1.6, 'lines.linewidth': 3.2, 'legend.fontsize': 14, 'savefig.bbox': 'tight'})
    paths = catalog['relativistic']
    ef = read_frame(data_dir, paths['ef_history'])
    profiles = read_frame(data_dir, paths['ef_profiles'])
    times = np.sort(profiles.advanced_time.unique())
    wanted = [0, 28, 31.5, 31.7, 31.85, 32.0, 32.2, 32.3, 32.4, 32.5]
    # Choose actual stored EF profiles near the desired times, then verify complete radial blocks.
    selected = sorted(set((float(times[np.argmin(abs(times - time))]) for time in wanted)))
    assert len(selected) == 10
    cache = profiles[profiles.advanced_time.isin(selected)]
    for time in selected:
        block = cache[cache.advanced_time == time]
        assert len(block) == 4096
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.29, wspace=0.25)
    colors = plt.cm.plasma(np.linspace(0.05, 0.82, len(selected)))
    for time, color in zip(selected, colors):
        block = cache[cache.advanced_time == time].sort_values('r')
        # Keep the resolved density above the stated snapshot-relative floor.
        matter = (block.r > 0) & (block.rho_infall > block.rho_infall.max() * 1e-07)
        axes[0].plot(np.log10(block.loc[matter, 'r']), np.log10(block.loc[matter, 'rho_infall']), color=color, label=f'{time:.2f}')
        # Horizon lines use the identical native time and integer step as
        # each density slice; no interpolation between saved times is used.
        # Join each horizon radius to the density slice's actual time and integer step before drawing it.
        history_row = ef.iloc[int(np.argmin(abs(ef.advanced_time - time)))]
        assert abs(float(history_row.advanced_time) - time) <= 8 * np.finfo(float).eps * max(1, abs(time))
        assert int(history_row.step) == int(block.step.iloc[0])
        horizon_radius = float(history_row.apparent_horizon_radius)
        if horizon_radius > 0:
            axes[0].axvline(np.log10(horizon_radius), color=color, ls=(0, (5, 3)), lw=1.8)
        axes[1].plot(block.r, block.compactness, color=color)
    axes[0].set(xlabel='$\\log_{10} r$', ylabel='$\\log_{10}\\rho_u$', title='Density during inward collapse', xlim=(-0.35, 0.7), ylim=(-7.5, 0.3))
    axes[1].axhline(1, color='black', ls='--', lw=2.0)
    axes[1].set(xlabel='Areal radius $r$', ylabel='$2m/r$', title='Growth of the trapped region', xlim=(0, 4.7), ylim=(0, 1.4))
    axes[1].text(0.97, 0.76, '$2m/r=1$', ha='right', transform=axes[1].transAxes, fontsize=14, bbox=dict(facecolor='white', edgecolor='none', pad=1))
    for label, ax in zip(('(a)', '(b)'), axes):
        ax.text(0.035, 0.92, label, transform=ax.transAxes)
        gr_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, 0.005), ncol=5, frameon=False, fontsize=14, title='Advanced time $v$', title_fontsize=15, handlelength=2.6, columnspacing=1.3, labelspacing=0.65)
    for extension in ('pdf', 'png'):
        fig.savefig(output_dir / f'bh_interior.{extension}', dpi=190)
    plt.close(fig)

def verify_inputs(data_dir, catalog):
    """Authenticate every bundled reduced input before plotting."""
    for item in catalog['data_files']:
        path = data_dir / Path(item['file']).relative_to('data')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Changed figure input: ' + str(path))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data-dir', type=Path, default=root / 'data',
                        help='directory containing catalog.json and reduced CSV files')
    parser.add_argument('--output-dir', type=Path, default=root / 'reproduced',
                        help='directory for the regenerated PDF and PNG pairs')
    parser.add_argument('--figure', type=int, nargs='+', choices=range(1, 9), default=list(range(1, 9)),
                        help='renderer selectors 1..8 correspond to example Figures 2,3,4,5,8,10,11,12')
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((data_dir / 'catalog.json').read_text())
    verify_inputs(data_dir, catalog)
    # Render each selected figure in its own Matplotlib context so style settings remain local.
    for number in args.figure:
        with plt.rc_context():
            if number == 1:
                render_virial(data_dir, output_dir, catalog)
            elif number == 2:
                render_halo_plate(data_dir, output_dir, catalog, [('g03', 0.3), ('g05', 0.5), ('g08', 0.8)], 'halo_multigamma_density_slopes_shallow')
            elif number == 3:
                render_halo_plate(data_dir, output_dir, catalog, [('g12', 1.2), ('g15', 1.5), ('g20', 2.0)], 'halo_multigamma_density_slopes_steep')
            elif number == 4:
                render_bp_vp(data_dir, output_dir, catalog)
            elif number == 5:
                render_cold(data_dir, output_dir, catalog)
            elif number == 6:
                render_sidm(data_dir, output_dir, catalog)
            elif number == 7:
                render_bh_coordinates(data_dir, output_dir, catalog)
            elif number == 8:
                render_bh_interior(data_dir, output_dir, catalog)
        print('Wrote figure', {1: 2, 2: 3, 3: 4, 4: 5, 5: 8, 6: 10, 7: 11, 8: 12}[number], 'to', output_dir)
if __name__ == '__main__':
    main()
