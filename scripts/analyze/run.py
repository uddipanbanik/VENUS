#!/usr/bin/env python3
"""Summarize a VENUS run and optionally draw its saved radial profiles.

The JSON report records available tables, first/final times, and the recorded
diagnostic columns. Physical interpretation requires the complete model-specific budget.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,help='Analysis directory; default RUN/analysis')
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--gamma',type=float,default=None,help='Reference cusp slope for particle profile labels')
    parser.add_argument('--r-max',type=float,default=20)
    args=parser.parse_args(); run=args.run.resolve()
    if not run.is_dir():parser.error('Run directory does not exist.')
    output=(args.output or run/'analysis').resolve();output.mkdir(parents=True,exist_ok=True)
    report={'run':str(run),'tables':{}}
    for p in sorted(run.glob('*history*.csv'))+sorted(run.glob('*budget*.csv')):
        with p.open() as stream:rows=list(csv.DictReader(stream))
        if not rows:continue
        report['tables'][p.name]={'rows':len(rows),'columns':list(rows[0]),'first':rows[0],'last':rows[-1]}
    report['numbered_profiles']=len(list(run.glob('profile_*.csv')))
    report['restarts']=[str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and ('restart' in p.name or p.suffix=='.restart')]
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,item in report['tables'].items():
        first=item['first'];last=item['last'];print(name,':',item['rows'],'saved rows; time',first.get('time',first.get('advanced_time',first.get('t','?'))),'to',last.get('time',last.get('advanced_time',last.get('t','?'))))
    if args.plot:
        if report['numbered_profiles']:
            command=[sys.executable,str(ROOT/'gpu/scripts/render_venus_gpu_density_slope_gif.py'),
                     '--collisionless',str(run),'--output-directory',str(output),'--output',str(output/'density_slope.gif'),
                     '--r-max',str(args.r_max),'--series-label','Evolved distribution']
            command += ['--gamma',str(args.gamma)] if args.gamma is not None else ['--title','Particle distribution']
        elif (run/'profiles.csv').exists():
            command=[sys.executable,str(ROOT/'cpu/tools/plot_einstein_vlasov_outputs.py'),
                     '--input-dir',str(run),'--output-dir',str(output),'--make-gifs']
        elif (run/'low_compactness_profiles.csv').exists() or (run/'high_compactness_profiles.csv').exists():
            sys.path.insert(0,str(ROOT/'cpu/tools'))
            from render_weighted_einstein_vlasov_pic_profile_gifs import load_csv, select_times, render_case
            low=run/'low_compactness_profiles.csv'
            high=run/'high_compactness_profiles.csv'
            if low.exists():
                data=load_csv(low,{'time','r','ev_rho','ev_compactness','vp_rho','vp_compactness'})
                render_case(data,select_times(data,24),
                    [('ev_rho','ev_compactness','b','-','Einstein--Vlasov'),
                     ('vp_rho','vp_compactness','r','--','Vlasov--Poisson')],
                    output/'profiles.gif',10,1e-7,11,11,None,origin_parity=False)
                command=None
            else:
                with high.open() as stream:header=next(csv.reader(stream))
                if 'advanced_time' in header:
                    command=[sys.executable,str(ROOT/'cpu/tools/render_ef_interior_gif.py'),
                             '--run',str(run),'--output',str(output/'profiles.gif'),'--frames','24']
                else:
                    data=load_csv(high,{'time','r','rho','compactness'})
                    render_case(data,select_times(data,24),
                        [('rho','compactness','b','-','Einstein--Vlasov')],
                        output/'profiles.gif',10,1e-7,11,11,None,origin_parity=True)
                    command=None
        else:
            parser.error('No recognized radial profile table was found; summary.json has been saved.')
        if command is not None:subprocess.run(command,check=True)
    print('Saved analysis:',output)


if __name__=='__main__':main()
