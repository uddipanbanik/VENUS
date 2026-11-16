#!/usr/bin/env python3
# Plot cold VP physical virial balance and inertia for static/expanding backgrounds.
# Inputs are catalog-bound force/kinetic/inertia histories, endpoint particle moments and boundary diagnostics.
# Output PDF/PNG compares full populations and the separately measured static endpoint selections.

"""Render the static and expanding cold Vlasov--Poisson virial diagnostics.

The input catalog supplies reduced physical moment histories and exact t=0
and endpoint particle moments. The physical force includes the prescribed
background acceleration. All paths are relative to the selected data directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'venus-cold-figure-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def render(data_dir, out, catalog):
    """Calculate the physical virial ratio and plot both inertia histories."""
    # Authenticate every reduced table before deriving a physical diagnostic.
    for item in catalog['data_files']:
        path = data_dir / Path(item['file']).relative_to('data')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Changed plotting input: '+str(path))
    config = catalog['cold_vp']
    moments = json.loads((data_dir/Path(config['moments']).relative_to('data')).read_text())
    def table(relative):
        path = data_dir/Path(config['tables'][relative]).relative_to('data')
        return np.atleast_1d(np.genfromtxt(path, names=True, delimiter=','))
    def moment(relative):
        return moments[relative]
    initial_static = moment('static_initial/checkpoint_moments/result.json')
    initial_expanding = moment('expanding_initial/checkpoint_moments/result.json')
    static_partition = moment('static_final/checkpoint_bound_static/result.json')
    final_expanding = moment('expanding_final/checkpoint_moments/result.json')
    control = table('outer_control_extended_virial.csv')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix',
        'font.size':18, 'axes.labelsize':22, 'axes.titlesize':20,
        'axes.linewidth':1.8, 'xtick.labelsize':17, 'ytick.labelsize':17,
        'legend.fontsize':14, 'lines.linewidth':2.6, 'savefig.dpi':220,
        'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.25), sharex=True,
                             sharey='col', layout='constrained')
    diagnostics = {}
    for row, (case, initial, title) in enumerate([
        ('cold_static_vp', initial_static, 'Static background'),
        ('cold_expanding_vp', initial_expanding, 'Expanding background')]):
        budget = table(case + '/virial_budget.csv')
        history = table(case + '/history.csv')
        boundary = table(case + '/boundary_history.csv')
        assert budget['time'][-1] == history['time'][-1] == boundary['time'][-1] == 3000
        assert np.all(np.diff(budget['time']) > 0)
        # This is the physical virial identity between outer-wall impulses.
        total_force = budget['W_force'] + (budget['H_dot'] + budget['H']**2) * budget['I']
        # The denominator includes the halo force and the prescribed physical background acceleration.
        q = 2 * budget['K_physical'] / np.abs(total_force)
        initial_force = initial['W_force'] + (initial['H_dot'] + initial['H']**2) * initial['I']
        q0 = 2 * initial['K_physical'] / abs(initial_force)
        # Append the independently measured initial moment to the actual positive-time history.
        clock = np.r_[0, budget['time']]
        x = np.log10(1 + clock)
        axes[row, 0].plot(x, np.r_[q0, q], 'k-', label='Full population')
        axes[row, 1].plot(x, np.log10(np.r_[initial['I'], budget['I']]), 'k-', label='Full population')
        axes[row, 0].axhline(1, color='.4', ls=':', lw=1.5)
        axes[row, 0].set_title(title)
        axes[row, 1].set_title('Moment of inertia')
        axes[row, 0].set(ylim=(0, 1.9), ylabel=r'$2K_{\rm physical}/|W_{\rm physical}|$')
        axes[row, 1].set(ylim=(0, 5.35), ylabel=r'$\log_{10} I$')
        # Record virial/inertia derivatives and conservation/boundary information from the same final epoch.
        diagnostics[case] = dict(time=3000, steps=int(history['step'][-1]),
            Q_initial=float(q0), Q_final=float(q[-1]), I_final=float(budget['I'][-1]),
            I_dot_final=float(2 * (budget['C'][-1] + budget['H'][-1]*budget['I'][-1])),
            I_ddot_half_final=float(2*budget['K_physical'][-1]+total_force[-1]),
            mass_relative_drift_max=float(np.max(np.abs(history['mobile_mass']/initial['mass']-1))),
            wall_impacts=int(boundary['outer_wall_impacts'][-1]),
            solver_wall_seconds=float(history['wall_seconds'][-1]))
        for ax in axes[row]:
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=9, width=1.7, pad=6)
            ax.tick_params(which='minor', length=5, width=1.2)
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.grid(False)
            ax.set_xlim(0, 3.53)
            if row == 1:
                ax.set_xlabel(r'$\log_{10}(1+t)$')

    # This is a single E_full<0 selection. Its displayed virial ratio uses the selected population's own field.
    bound = static_partition['instantaneously_bound_in_full_potential']
    positive = static_partition['instantaneously_unbound_in_full_potential']
    all_static = static_partition['all']
    endpoint = np.log10(3001)
    axes[0, 0].plot(endpoint, bound['Q_self_force'], 'o', ms=9, mfc='#087cff',
                   mec='white', mew=1, label=r'Bound at $t=3000$', zorder=5)
    axes[0, 1].plot(np.log10(1+control['time']), np.log10(control['I']), color='.4',
                   lw=2.1, ls=(0,(5,3)), label=r'Full population: $R=2051$')
    axes[0, 1].plot(endpoint, np.log10(bound['I']), 'o', ms=9, color='#087cff',
                   label='Bound endpoint', zorder=5)
    axes[0, 1].plot(endpoint, np.log10(positive['I']), 'o', ms=9, mfc='white',
                   mec='#f02c33', mew=2, label='Positive orbital energy', zorder=6)
    # Quantify how much of the full static inertia belongs to the positive-energy endpoint population.
    fraction = 100*positive['I']/all_static['I']
    axes[0, 1].annotate(rf'{fraction:.2f}% of $I$',
        xy=(endpoint, np.log10(positive['I'])), xytext=(2.0,3.9),
        fontsize=17, color='#f02c33',
        arrowprops={'arrowstyle':'->', 'color':'#f02c33', 'lw':1.4})
    for ax in axes[0]:
        ax.axvline(np.log10(2591), color='.6', lw=1.2, ls=(0,(2,3)), zorder=0)
    # Opaque white, borderless legend keeps the wall-impact guide clear of text.
    axes[0, 0].legend(loc='upper right', frameon=True, facecolor='white',
                      edgecolor='none', framealpha=1, handlelength=2.7)
    axes[0, 1].legend(loc='upper left', frameon=False, handlelength=2.7, fontsize=13.5)
    axes[1, 0].text(.96,.14,r'$t=3000:\ 1.00334$', transform=axes[1,0].transAxes,
                    ha='right', va='bottom', fontsize=17)
    axes[1, 1].text(.05,.94,'Full population', transform=axes[1,1].transAxes,
                    ha='left', va='top', fontsize=16)
    # Check that the history endpoint agrees with the independent full-particle moment measurement.
    final_force = final_expanding['W_force'] + (final_expanding['H_dot'] + final_expanding['H']**2)*final_expanding['I']
    assert abs(diagnostics['cold_expanding_vp']['Q_final']-2*final_expanding['K_physical']/abs(final_force)) < 1e-13
    stem = out/'cold_vp_static_expanding_virial'
    for extension in ['pdf','png']:
        fig.savefig(stem.with_suffix('.'+extension), bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    return diagnostics


def main():
    """Read portable command-line paths and write the PDF/PNG pair."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root/'data')
    parser.add_argument('--output-dir', type=Path, default=root/'figures')
    args = parser.parse_args()
    catalog = json.loads((args.data_dir/'catalog.json').read_text())
    render(args.data_dir.resolve(), args.output_dir.resolve(), catalog)


if __name__ == '__main__':
    main()
