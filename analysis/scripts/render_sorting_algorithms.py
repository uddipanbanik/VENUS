#!/usr/bin/env python3
# Draw a record-preserving insertion sort and stable decimal radix example.
# Input JSON assigns fixed IDs, keys and colors. Output PDF/PNG and traces show the actual list operations.
# The decimal illustration explains stability; production radii use binary64 byte keys.

"""Draw insertion and stable least-significant-digit radix sorting.

The common decimal keys are a teaching example. The radius implementation
uses eight stable base-256 byte passes on finite nonnegative binary64 keys.
Each key retains its record properties and fixed ID during sorting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'venus-sorting-algorithms-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def sha(path):
    """Identify exact source, input or artifact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def insertion_trace(records):
    """Apply strict insertion comparisons while moving complete records."""
    ordered = list(records)
    trace = []
    for index in range(1,len(ordered)):
        held = ordered[index]
        position = index
        shifted = []
        # Shift only strictly larger preceding keys, so equal keys retain their incoming relative order.
        while position > 0 and held['key'] < ordered[position-1]['key']:
            shifted.append(ordered[position-1]['id'])
            ordered[position] = ordered[position-1]
            position -= 1
        ordered[position] = held
        trace.append({'held_id':held['id'],'held_key':held['key'],
                      'shifted_ids':shifted,'sorted_prefix':index+1,
                      'records':list(ordered)})
    return trace


def radix_trace(records):
    """Collect stable decimal buckets from units to tens."""
    ordered = list(records)
    trace = []
    for power in [0,1]:
        buckets = {}
        for record in ordered:
            digit = (record['key']//(10**power)) % 10
            # Append records to the current digit's temporary list in their existing order.
            buckets.setdefault(digit,[]).append(record)
        # Read digit lists in ascending digit order. This preserves the order established by the preceding pass.
        ordered = [record for digit in sorted(buckets) for record in buckets[digit]]
        trace.append({'power':power,'buckets':buckets,'records':list(ordered)})
    return trace


def record_row(ax,y,records,prefix=0):
    """Draw fixed-ID boxes, with an optional underline for the ordered prefix."""
    width = .224
    for index,record in enumerate(records):
        x = .018+index*.248
        ax.add_patch(Rectangle((x,y-.045),width,.09,facecolor='white',
                               edgecolor=record['color'],linewidth=1.1,clip_on=False))
        ax.text(x+width/2,y,f"ID{record['id']}:{record['key']}",
                ha='center',va='center',fontsize=10.5,color=record['color'],clip_on=False)
    if prefix:
        ax.plot([.018,.018+(prefix-1)*.248+width],[y-.057,y-.057],
                color='.30',lw=2.5,solid_capstyle='butt')


def radix_record_row(ax, y, records, digit=None):
    """Draw an ordered record list and emphasize the digit under examination."""
    width = .224
    for index, record in enumerate(records):
        x = .018 + index * .248
        ax.add_patch(Rectangle((x, y - .045), width, .09, facecolor='white',
                               edgecolor=record['color'], linewidth=1.1,
                               clip_on=False))
        ax.text(x + width / 2, y, digit_label(record, digit),
                ha='center', va='center', fontsize=10.5,
                color=record['color'], clip_on=False)


def digit_label(record, digit):
    """Keep the fixed ID visible while bolding either the units or tens digit."""
    key = str(record['key'])
    if digit == 'units':
        key = '$' + key[0] + r'\mathbf{' + key[1] + '}$'
    elif digit == 'tens':
        key = '$' + r'\mathbf{' + key[0] + '}' + key[1] + '$'
    return f"ID{record['id']}:" + key


def bucket_row(ax, y, pass_record, digit):
    """Append records in incoming order to separate, labelled temporary lists."""
    groups = sorted(pass_record['buckets'].items())
    record_pitch = .218
    record_width = .206
    gap = .055 if len(groups) == 3 else .090
    cursor = .018
    for value, records in groups:
        width = (len(records) - 1) * record_pitch + record_width
        ax.add_patch(Rectangle((cursor, y - .045), width, .140,
                               facecolor='.95', edgecolor='.35', linewidth=1.2))
        ax.plot([cursor, cursor + width], [y + .033, y + .033],
                color='.55', linewidth=.7)
        ax.text(cursor + width / 2, y + .065, f'digit {value}',
                ha='center', va='center', fontsize=10.5)
        for index, record in enumerate(records):
            ax.text(cursor + record_width / 2 + index * record_pitch,
                    y - .009, digit_label(record, digit),
                    ha='center', va='center', fontsize=10.5,
                    color=record['color'], clip_on=False)
        cursor += width + gap


def render(input_path,out):
    """Draw and validate the two algorithms at the final journal width."""
    data = json.loads(input_path.read_text())
    records = data['records']
    assert [(r['id'],r['key']) for r in records] == [(1,21),(2,13),(3,12),(4,22)]
    insertion = insertion_trace(records)
    radix = radix_trace(records)
    assert [[r['key'] for r in step['records']] for step in insertion] == [[13,21,12,22],[12,13,21,22],[12,13,21,22]]
    assert [step['shifted_ids'] for step in insertion] == [[1],[1,2],[]]
    assert [[r['key'] for r in step['records']] for step in radix] == [[21,12,22,13],[12,13,21,22]]
    assert [r['id'] for r in radix[0]['buckets'][2]] == [3,4]
    # Verify every intermediate list contains the same complete records attached to the same fixed IDs.
    original = {r['id']:r for r in records}
    for step in insertion+radix:
        assert {r['id']:r for r in step['records']} == original
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix',
                         'font.size':11,'pdf.fonttype':42,'savefig.dpi':250})
    fig = plt.figure(figsize=(7.1,4.05))
    left = fig.add_axes([.025,.105,.455,.745])
    right = fig.add_axes([.525,.105,.455,.745])
    for ax in [left,right]:
        ax.set(xlim=(0,1),ylim=(0,1))
        ax.axis('off')
    fig.text(.5,.965,r'ID: fixed record identifier'+r'$\qquad$'+r'$K$: sort key',
             ha='center',va='center',fontsize=11.5)
    left.text(.5,1.035,'(a) Insertion sort',ha='center',va='center',fontsize=14)
    right.text(.5,1.035,'(b) Stable radix sort',ha='center',va='center',fontsize=14)
    left.text(.5,.945,r'Shift while $K_{\rm left}>K_{\rm hold}$',ha='center',va='center',fontsize=11)
    right.text(.5,.945,'Read buckets in digit order after each pass.',
               ha='center',va='center',fontsize=10.5)
    for ax in [left,right]:
        ax.text(0,.866,'Input',ha='left',va='center',fontsize=11.5)
        if ax is left:
            record_row(ax,.785,records,prefix=1)
        else:
            radix_record_row(ax,.785,records,digit='units')

    # The inequalities name every comparison that causes a rightward shift.
    left.text(.5,.659,r'Hold ID2: $13<21$, shift ID1',ha='center',va='center',fontsize=11)
    record_row(left,.575,insertion[0]['records'],prefix=2)
    left.text(.5,.444,r'Hold ID3: $12<21$, then $12<13$',ha='center',va='center',fontsize=11)
    left.text(.5,.391,'Shift ID1, then ID2',ha='center',va='center',fontsize=10.5)
    record_row(left,.306,insertion[1]['records'],prefix=3)
    left.text(.5,.171,r'ID4: $22\geq21$, keep its position',ha='center',va='center',fontsize=11)
    record_row(left,.084,insertion[2]['records'],prefix=4)

    # A pass appends whole-ID records to digit lists, then reads the lists in order.
    right.text(.5,.663,r'1. Order by units digit ($K\ \mathrm{mod}\ 10$)',
               ha='center',va='center',fontsize=11.5)
    bucket_row(right,.525,radix[0],digit='units')
    radix_record_row(right,.405,radix[0]['records'])
    right.text(.5,.320,'Clear buckets. Refill from this row.',
               ha='center',va='center',fontsize=10.5)
    right.text(.5,.261,r'2. Order by tens digit ($\lfloor K/10\rfloor$)',
               ha='center',va='center',fontsize=11.5)
    bucket_row(right,.128,radix[1],digit='tens')
    radix_record_row(right,.018,radix[1]['records'])
    fig.text(.5,.037,'Each key retains its record properties. Color follows the fixed ID.',
             ha='center',va='center',fontsize=11)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)+list(left.texts)+list(right.texts)
    outside = []
    # Check the final canvas bounds after all record boxes and labels have been drawn.
    for text in texts:
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.x1 or bounds.y1 > fig.bbox.y1:
            outside.append(text.get_text())
    for ax in [left,right]:
        for patch in ax.patches:
            bounds = patch.get_window_extent(renderer)
            if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.x1 or bounds.y1 > fig.bbox.y1:
                outside.append('record or bucket boundary')
    assert not outside,outside
    out.mkdir(parents=True,exist_ok=True)
    artifacts = {}
    for extension in ['pdf','png']:
        path = out/('sorting_algorithms.'+extension)
        fig.savefig(path)
        artifacts[path.name] = sha(path)
    plt.close(fig)
    receipt = {'source_sha256':sha(__file__),'input_sha256':sha(input_path),
               'figure_inches':[7.1,4.05],'text_bounds':'PASS','artifacts':artifacts,
               'insertion_trace':insertion,'radix_trace':radix,
               'complete_record_identity_preserved':True,
               'stable_units_bucket2_IDs':[3,4],
               'bucket_definition':'A temporary list holding records with the same current digit. Records append in incoming order.',
               'pass_transition':'Clear the buckets, scan the preceding output row, and refill them using the next digit.',
               'final_ID_order':[3,2,1,4],
               'implementation_scope':'Decimal teaching example. Actual finite nonnegative binary64 radius keys use eight stable base-256 byte passes.'}
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'artifacts':artifacts,'text_bounds':'PASS'},indent=2))


def main():
    """Resolve portable input and output arguments."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=root/'data/sorting_algorithms/records.json')
    parser.add_argument('--output-dir',type=Path,default=root/'reproduced')
    args = parser.parse_args()
    render(args.input,args.output_dir)


if __name__ == '__main__':
    main()
