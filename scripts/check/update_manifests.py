#!/usr/bin/env python3
"""Regenerate release checksums from Git's distributed, nonignored file set.

Run after reviewing the candidate files. This records file identity; numerical
validation remains documented separately in release/VALIDATION.md.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    raw = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    names = sorted(set(name.decode() for name in raw.split(b'\0') if name))
    names = [name for name in names if (ROOT / name).is_file()]
    numerical_path = ROOT / 'release/numerical_sources.json'
    numerical = json.loads(numerical_path.read_text())
    prefixes = ('cpu/include/', 'cpu/src/', 'cpu/tools/',
                'gpu/include/', 'gpu/src/', 'gpu/tools/')
    files = {name: digest(ROOT / name) for name in names
             if name.startswith(prefixes) and Path(name).suffix in ('.hpp', '.h', '.cpp', '.cu', '.cuh')}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    numerical.update(scope='Numerical, production-driver and C++/CUDA utility sources in the VENUS distribution',
                     files=files, manifest_identity_sha256=identity)
    write_json(numerical_path, numerical)
    source_path = ROOT / 'release/source_manifest.json'
    source = json.loads(source_path.read_text())
    excluded = {'release/source_manifest.json', 'release/source_manifest.sha256'}
    entries = [{'path': name, 'bytes': (ROOT / name).stat().st_size, 'sha256': digest(ROOT / name)}
               for name in names if name not in excluded]
    source.update(numerical_file_count=len(files), numerical_manifest_identity_sha256=identity,
                  file_count=len(entries), files=entries)
    write_json(source_path, source)
    (ROOT / 'release/source_manifest.sha256').write_text(digest(source_path) + '  source_manifest.json\n')
    print(f'Recorded {len(files)} numerical/utility sources and {len(entries)} distributed files.')


if __name__ == '__main__':
    main()
