#!/usr/bin/env python3
"""Verify every file recorded in the distributed VENUS source manifest."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
manifest=ROOT/'release/source_manifest.json'
expected=(ROOT/'release/source_manifest.sha256').read_text().split()[0]
failures=[]
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:
    failures.append('release/source_manifest.json checksum')
data=json.loads(manifest.read_text())
for entry in data['files']:
    path=ROOT/entry['path']
    if not path.is_file():
        failures.append(entry['path']+' missing');continue
    content=path.read_bytes()
    if len(content)!=entry['bytes'] or hashlib.sha256(content).hexdigest()!=entry['sha256']:
        failures.append(entry['path']+' differs')
if failures:
    print('\n'.join(failures),file=sys.stderr);sys.exit(1)
print(f"Verified {len(data['files'])} distributed files.")
