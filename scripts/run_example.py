#!/usr/bin/env python3
"""Run a catalogued finite VENUS example with explicit inputs and provenance.

Use --list to select a task, --dry-run to inspect its command, and --variant
openmp/mpi for supported parallel builds. Outputs live in a fresh directory.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = json.loads((ROOT / 'examples/catalogue.json').read_text())


def execute(command):
    print(shlex.join([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True)


def initial_state(destination):
    if not destination.exists():
        execute([sys.executable, ROOT / 'cpu/tools/generate_bound_gaussian_shell_restart.py',
                 '--output-directory', destination, '--particles', '4096'])
    for background in ['static_vp', 'expanding_vp']:
        if not (destination / background / 'restart_state.bin').is_file():
            raise RuntimeError('Initial state is incomplete: ' + str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', nargs='?', choices=sorted(CATALOGUE))
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--variant', choices=['portable', 'openmp', 'mpi'], default='portable')
    parser.add_argument('--launcher', choices=['mpiexec', 'srun'], default='mpiexec')
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--initial-directory', type=Path, default=ROOT / 'output/example_initial')
    parser.add_argument('--gpu-build', type=Path)
    parser.add_argument('--resume', action='store_true', help='Import the saved state in --output with identical controls')
    parser.add_argument('--end-time', type=float, help='Override physical endpoint; set a later time for --resume')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.list:
        for name, item in CATALOGUE.items():
            print(f"{name:36s} {item['backend']:3s} {item['category']}")
        return
    if args.example is None:
        parser.error('Choose an example or --list.')
    if args.threads < 1 or args.ranks < 1:
        parser.error('Threads and ranks must be positive.')
    item = CATALOGUE[args.example]
    if args.variant != 'portable' and (item['backend'] == 'cpu' and not item.get('grid')):
        parser.error('CPU particle and orbit-phase runners use one shared-memory process.')
    if args.variant == 'mpi' and item['backend'] == 'cpu' and args.example not in ['ev_grid','vp_grid_static','vp_grid_expanding']:
        parser.error('CPU MPI supports collisionless Eulerian EV/VP.')
    if args.variant == 'mpi' and item['backend'] == 'gpu' and args.example in ['gpu_vp_static','gpu_subhalo_static']:
        parser.error('GPU MPI VP/subhalo use the expanding background.')
    if args.variant == 'openmp' and item['backend'] == 'gpu':
        parser.error('Use portable or mpi for GPU tasks.')
    output = (args.output or ROOT / 'output/examples' / (args.example + '_' + args.variant)).resolve()
    initial = args.initial_directory.resolve()
    suffix = '' if args.variant == 'portable' else '_' + args.variant
    gpu_build = args.gpu_build or ROOT / 'build' / ('gpu-mpi' if args.variant == 'mpi' else 'gpu')
    executable = (ROOT / 'cpu' if item['backend'] == 'cpu' else gpu_build) / (item['executable'] + suffix)
    values = dict(output=str(output), initial=str(initial), grid_initial=str(initial.parent / 'example_grid_initial'), compact_initial=str(initial.parent / 'example_compact_initial'), canonical_initial=str(initial.parent / 'example_canonical_initial'), threads=str(args.threads),
                  sphdsmc_restart=str(output.parent / 'canonical_seed/restarts/restart.bin'))
    options = [arg.format(**values) for arg in item['args']]
    # The EF example first constructs the required null slice from SC worldlines.
    donor = None
    if args.example == 'ev_wpic_eddington_finkelstein':
        physical = output.parent / 'ef_common.phys'
        null = output.parent / 'ef_common.nullmap'
        common = ['--expected-outcome','any','--physical-ic',str(physical),'--threads',str(args.threads),
                  '--high-initial-condition','shell','--high-markers','256','--high-radial-cells','64',
                  '--high-r-min','0.01','--high-r-max','6','--high-inner-boundary','origin_parity',
                  '--high-radial-grid','linear','--high-total-mass','0.08','--high-shell-center','2',
                  '--high-shell-width','0.20','--high-shell-radial-momentum','-0.30',
                  '--high-shell-radial-momentum-dispersion','0.01',
                  '--high-shell-tangential-velocity-dispersion','0','--high-specific-angular-momentum','0.08',
                  '--high-target-initial-adm-mass','0.07','--high-t-end','0.08','--high-dt-max','0.04','--high-output-steps','1']
        donor = [executable,'--coordinates','schwarzschild','--export-ef-null-initial-data',null,
                 '--ef-null-donor-duration','6','--ef-map-worldtube-radius','5.25',
                 '--output-dir',output.parent / 'ef_donor'] + common
        options = ['--coordinates','eddington_finkelstein','--ef-null-initial-data',str(null),
                   '--output-dir',str(output)] + common
    if args.end_time is not None:
        endpoint = '--t-end' if item.get('grid') else '--end-time'
        if args.example == 'ev_wpic_low': endpoint = '--low-t-end'
        elif args.example.startswith('ev_wpic_'): endpoint = '--high-t-end'
        elif args.example == 'bp_canonical_orbit':
            endpoint = '--evolution-time'
            if '--evolution-events' in options:
                index = options.index('--evolution-events')
                del options[index:index + 2]
        options += [endpoint, str(args.end_time)]
    if args.resume:
        if not output.is_dir(): parser.error('--resume requires an existing --output directory.')
        if args.end_time is None: parser.error('--resume requires a later --end-time.')
        saved_history = [output / name for name in ['history.csv','history_gpu.csv',
                         'low_compactness_history.csv','high_compactness_history.csv']]
        saved_history = [path for path in saved_history if path.exists()]
        if not saved_history: parser.error('No saved history found for continuation.')
        with saved_history[0].open() as stream:
            saved_rows = list(csv.DictReader(stream))
        if not saved_rows: parser.error('The saved history is empty.')
        saved_time = float(saved_rows[-1].get('time',saved_rows[-1].get('advanced_time','nan')))
        if not args.end_time > saved_time:
            parser.error('--end-time must be later than the saved history time.')
        if args.example == 'bp_canonical_orbit':
            # The canonical runner takes a duration; the common example
            # interface takes an absolute endpoint like the other solvers.
            index = options.index('--evolution-time')
            options[index + 1] = str(args.end_time - saved_time)
        if item.get('grid'):
            candidates = [path for path in output.rglob('restart_state.bin') if path.is_file()]
            if not candidates:
                candidates = [path for path in output.rglob('*.manifest') if path.is_file()]
            if not candidates: parser.error('No native grid restart found in the run directory.')
            source = max(candidates, key=lambda path: path.stat().st_mtime)
            segment = 1
            destination = output / f'restarts_resume_{segment:04d}'
            while destination.exists():
                segment += 1
                destination = output / f'restarts_resume_{segment:04d}'
            options += ['--restart-from',str(source),'--restart-dir',str(destination)]
        elif args.example.startswith('ev_wpic_'): options += ['--resume']
        elif args.example == 'bp_canonical_orbit': options += ['--canonical-restart',str(output / 'restart_state.bin')]
        else:
            options += ['--pic-restart' if args.example.startswith('sidm_wpic_') else '--restart',str(output / 'restart_state.bin')]
    command = [executable] + options
    if args.variant == 'mpi':
        command = [args.launcher, '-n', str(args.ranks)] + command
    if args.dry_run:
        if item.get('initial'): print('Generate/validate initial state:',initial)
        if donor: print(shlex.join([str(x) for x in donor]))
        print(shlex.join([str(x) for x in command])); return
    if output.exists() and not args.resume:
        parser.error('Output already exists; select a fresh --output or use --resume.')
    if not executable.is_file():
        parser.error(f'Missing {executable}; run the matching script in scripts/build/ first.')
    if item.get('initial'): initial_state(initial)
    if item.get('canonical_seed'):
        canonical = initial.parent / 'example_canonical_initial'
        canonical.mkdir(parents=True, exist_ok=True)
        if not (canonical / 'restart.bin').exists():
            execute([sys.executable, ROOT / 'cpu/tools/generate_frank_core_cell_average_restart.py',
                     '--output', canonical / 'restart.bin', '--report', canonical / 'report.json',
                     '--nr','48','--nw','32','--nl','24','--r-min','0.001','--r-max','1000',
                     '--w-max','3','--L-max','100','--eta-power','10',
                     '--n-model','12000','--n-energy','512','--n-eddington-quad','64'])
        if not (canonical / 'frank_core_eddington_8192.csv').exists():
            execute([sys.executable, ROOT / 'cpu/tools/export_frank_eddington_table.py',
                     '--output', canonical / 'frank_core_eddington_8192.csv', '--r-min','0.001',
                     '--n-model','12000','--n-energy','512','--n-eddington-quad','64'])
    if item.get('compact_seed'):
        compact = initial.parent / 'example_compact_initial/restart.bin'
        if not compact.exists():
            generator = ROOT / 'cpu/build/toolcheck/generate_collision_example'
            if not generator.exists():
                raise RuntimeError('Build the small input generator: scripts/build/tools.sh')
            execute([generator, compact])
    if item.get('grid_seed'):
        grid_initial = initial.parent / 'example_grid_initial'
        grid_initial.mkdir(parents=True, exist_ok=True)
        if not (grid_initial / 'restart.bin').exists():
            execute([sys.executable, ROOT / 'cpu/tools/generate_frank_core_cell_average_restart.py',
                     '--output', grid_initial / 'restart.bin', '--report', grid_initial / 'report.json',
                     '--nr','8','--nw','16','--nl','16','--r-min','0.1','--r-max','10',
                     '--w-max','3','--L-max','8','--eta-power','6',
                     '--n-model','2000','--n-energy','128','--n-eddington-quad','32'])
        if not (grid_initial / 'frank_core_eddington_8192.csv').exists():
            execute([sys.executable, ROOT / 'cpu/tools/export_frank_eddington_table.py',
                     '--output', grid_initial / 'frank_core_eddington_8192.csv', '--r-min','0.1',
                     '--n-model','2000','--n-energy','128','--n-eddington-quad','32'])
    if item.get('sphdsmc'):
        seed=output.parent/'canonical_seed'
        if not seed.exists():
            execute([sys.executable,ROOT/'scripts/run_example.py','sidm_sphdsmc_constant','--output',seed])
        candidates=[p for p in seed.rglob('*') if p.is_file() and 'restart' in p.name and p.suffix=='.bin']
        if not candidates: raise RuntimeError('SphDSMC seed did not produce a binary restart.')
        options[options.index('--restart')+1]=str(sorted(candidates)[-1])
        command=[executable]+options
    if donor and not args.resume: execute(donor)
    # Provenance lives beside the solver output, leaving the solver ownership
    # check responsible for creation of its own fresh directory.
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt=output.with_name(output.name+'.command.json')
    receipt.write_text(json.dumps(dict(example=args.example,command=[str(x) for x in command],
                                     category=item['category'],variant=args.variant),indent=2)+'\n')
    execute(command)
    expected = args.end_time if args.end_time is not None else item['end_time']
    histories = [output / name for name in ['history.csv','history_gpu.csv',
                 'low_compactness_history.csv','high_compactness_history.csv']]
    histories = [path for path in histories if path.exists()]
    if not histories:
        raise RuntimeError('The example produced no history table.')
    with histories[0].open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError('The example history contains no completed state.')
    achieved = float(rows[-1].get('time', rows[-1].get('advanced_time', 'nan')))
    if not abs(achieved - expected) <= 1e-10 * max(1.0, abs(expected)):
        raise RuntimeError(f'Example stopped at {achieved}; requested endpoint is {expected}. Read its log before continuing.')
    record = json.loads(receipt.read_text())
    record.update(expected_end_time=expected, achieved_end_time=achieved)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print('Saved run:',output)


if __name__ == '__main__':
    main()
