#!/usr/bin/env python3
"""Run all examples for one backend in fresh directories and save their statuses."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('backend',choices=['cpu','gpu']);p.add_argument('--output',type=Path,required=True)
p.add_argument('--gpu-build',type=Path);args=p.parse_args();output=args.output.resolve()
output.mkdir(parents=True,exist_ok=False);results={}
for name,item in json.loads((ROOT/'examples/catalogue.json').read_text()).items():
 if item['backend']!=args.backend:continue
 command=[sys.executable,str(ROOT/'scripts/run_example.py'),name,'--output',str(output/name)]
 if args.gpu_build:command+=['--gpu-build',str(args.gpu_build.resolve())]
 with (output/(name+'.log')).open('w') as stream:
  result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
 results[name]=result.returncode;print(name,result.returncode,flush=True)
 (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
sys.exit(0 if all(code==0 for code in results.values()) else 1)
