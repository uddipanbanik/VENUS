# Compare regenerated figure pixels and PDF content against catalog-bound reference files.
# Input directories contain the same twelve figure stems. Exit status reports whether all requested pairs agree.
# PDF comparison normalizes only Matplotlib's creation timestamp.

"""Compare regenerated example figures with the supplied reference images.

PNG comparison uses every decoded color channel. PDF comparison removes
only the creation timestamp before comparing the complete file bytes.
The reference software versions are recorded in data/catalog.json.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

from PIL import Image, ImageChops


def digest(path):
    """Return a file's SHA256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_without_date(path):
    """Remove Matplotlib's creation timestamp while preserving page content."""
    return re.sub(rb'/CreationDate \([^)]*\)', b'/CreationDate (DATE)',
                  path.read_bytes())


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--reference-dir', type=Path, default=root.parent / 'plots')
    parser.add_argument('--output-dir', type=Path, default=root / 'reproduced')
    parser.add_argument('--figure', type=int, nargs='+', choices=range(1, 13),
                        default=list(range(1, 13)))
    args = parser.parse_args()
    catalog = json.loads((args.data_dir / 'catalog.json').read_text())
    passed = True
    for item in catalog['figures']:
        if item['number'] not in args.figure:
            continue
        stem = item['stem']
        # First establish that the supplied reference files themselves match the catalog.
        for extension in ('png', 'pdf'):
            reference = args.reference_dir / (stem + '.' + extension)
            if digest(reference) != item['files'][extension]['sha256']:
                raise RuntimeError('Changed reference figure: ' + str(reference))
        with Image.open(args.reference_dir / (stem + '.png')) as source:
            reference = source.convert('RGBA')
        with Image.open(args.output_dir / (stem + '.png')) as source:
            reproduced = source.convert('RGBA')
        # Pixel equality requires the same image dimensions and zero differences in every channel.
        same_png = reference.size == reproduced.size
        if same_png:
            difference = ImageChops.difference(reference, reproduced)
            # Test every channel separately; an all-zero alpha difference
            # would otherwise hide changes in the color channels.
            same_png = all(channel.getbbox() is None
                           for channel in difference.split())
        # Compare the entire remaining PDF byte stream after the single timestamp substitution.
        same_pdf = (pdf_without_date(args.reference_dir / (stem + '.pdf')) ==
                    pdf_without_date(args.output_dir / (stem + '.pdf')))
        print('Figure {}: PNG pixels {}; PDF except timestamp {}'.format(
            item['number'], 'PASS' if same_png else 'DIFFER',
            'PASS' if same_pdf else 'DIFFER'))
        # Combine all requested figure outcomes into one process exit status.
        passed &= same_png and same_pdf
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
