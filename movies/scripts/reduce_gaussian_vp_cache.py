#!/usr/bin/env python3
# Recompute Gaussian profile derivatives and sampling flags from a saved NPZ cache.
# Input arrays contain actual times and per-cell radius, density and effective count.
# An optional output NPZ preserves those arrays with the checked slopes and masks.

"""Recompute centered slopes and sampling masks from a portable native VP cache.

The input retains exact saved times, radius, density and effective count. Each
fit uses 21 positive native cells and a quadratic in log10 radius. All valid
saved columns are solved together. No states or times are interpolated.
"""
import argparse
from pathlib import Path
import numpy as np


def slopes(radius, density):
    """Apply the maintained local quadratic derivative to complete saved columns."""
    x = np.log10(radius)
    result = np.full(density.shape, np.nan)
    for j in range(10, len(x)-10):
        # Use ten neighboring cells on each side of the center: a complete 21-cell derivative window.
        window = density[:, j-10:j+11]
        # A logarithmic fit requires positive density at every node of the window.
        valid = np.all(window > 0, axis=1)
        if np.any(valid):
            # Center log radius on the evaluation cell. The quadratic linear coefficient is d(log rho)/d(log r).
            result[valid,j] = np.polynomial.polynomial.polyfit(
                x[j-10:j+11]-x[j], np.log10(window[valid]).T, 2)[1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_cache', type=Path)
    parser.add_argument('--output-cache', type=Path)
    args = parser.parse_args()
    with np.load(args.input_cache, allow_pickle=False) as source:
        arrays = {key:source[key] for key in source.files}
    clock = arrays['time']
    assert clock[0] == 0 and clock[-1] == 10000 and np.all(np.diff(clock)>0)
    radius = arrays['radius']
    # The batched polynomial fit uses the common radius grid shared by all saved epochs.
    assert np.array_equal(radius, np.broadcast_to(radius[0], radius.shape))
    derivative = slopes(radius[0], arrays['density'])
    assert np.array_equal(np.isnan(derivative), np.isnan(arrays['slope']))
    assert np.array_equal(derivative[np.isfinite(derivative)],
                          arrays['slope'][np.isfinite(derivative)])
    assert np.array_equal(arrays['density_positive'], arrays['density']>0)
    assert np.array_equal(arrays['slope_positive_window'], np.isfinite(derivative))
    # Effective count controls the heavy-curve sampling flag independently of slope availability.
    assert np.array_equal(arrays['mask'], arrays['neff']>=32)
    # Write only when explicitly requested, after all source slope/mask comparisons have passed.
    if args.output_cache:
        arrays.update(slope=derivative, density_positive=arrays['density']>0,
                      slope_positive_window=np.isfinite(derivative), mask=arrays['neff']>=32)
        np.savez_compressed(args.output_cache, **arrays)
    print('PASS:', len(clock), 'actual saved states, all derivative values and masks')


if __name__ == '__main__':
    main()
