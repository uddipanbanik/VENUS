#!/usr/bin/env python3
# Regenerate the twelve example figures from portable reduced-data catalogs.
# Inputs are catalog.json and its named data files. Output is a directory of PDF/PNG figure pairs.
# Separate renderer processes keep each figure's plotting configuration local.

"""Reproduce all twelve VENUS example figure pairs from the bundled reduced data.

The illustration and simulation renderers run in separate processes,
keeping their
Matplotlib settings independent. Every input is authenticated before drawing. This entry point requires the packages in ../requirements.txt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def main():
    """Resolve portable paths and render all twelve figures into one directory."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data',
                        help='directory containing catalog.json and reduced data')
    parser.add_argument('--output-dir', type=Path, default=root / 'reproduced',
                        help='directory for twelve PDF/PNG pairs and render metadata')
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    scripts = root / 'scripts'
    # Use the catalog to locate and authenticate the sorting illustration input before rendering.
    catalog = json.loads((data_dir / 'catalog.json').read_text())
    sorting = catalog['sorting_algorithms']['input']
    sorting_input = data_dir / Path(sorting['file']).relative_to('data')
    if hashlib.sha256(sorting_input.read_bytes()).hexdigest() != sorting['sha256']:
        raise RuntimeError('Changed sorting illustration input: ' + str(sorting_input))
    # Run the illustration and specialized Gaussian renderer before the remaining simulation figures.
    subprocess.run([sys.executable, str(scripts / 'render_sorting_algorithms.py'),
                    '--input', str(sorting_input), '--output-dir', str(output_dir)],
                   check=True)
    subprocess.run([sys.executable, str(scripts / 'render_gaussian_vp_full_radius.py'),
                    '--data-dir', str(data_dir / 'gaussian_vp'),
                    '--output-dir', str(output_dir)], check=True)
    # Select the figure numbers handled by the general renderer. The final call handles the g05/virial/SIDM family.
    subprocess.run([sys.executable, str(scripts / 'reproduce_figures.py'),
                    '--data-dir', str(data_dir), '--output-dir', str(output_dir),
                    '--figure', '1', '3', '4', '5', '6', '7', '8'], check=True)
    subprocess.run([sys.executable, str(scripts / 'render_g05_figures.py'),
                    '--data-dir', str(data_dir), '--output-dir', str(output_dir)],
                   check=True)


if __name__ == '__main__':
    main()
