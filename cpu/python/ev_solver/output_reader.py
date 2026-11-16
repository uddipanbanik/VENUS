# Read named CSV profiles and phase-space projections for analysis and plotting.
# Each returned phase frame carries its own coordinate arrays, including after momentum remapping.

"""CSV readers shared by plotting scripts and notebooks.

Sections read named-column tables, profile blocks, phase files, and appended
snapshots.  Live files are handled conservatively: an incomplete final
snapshot is ignored rather than returned as physical data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Section: generic named-column reader.

# Loads a named-column CSV and preserves one-row files as arrays.
def load_named_csv(path: Path) -> np.ndarray:
    """Load a named CSV while preserving one-row files as one-dimensional arrays."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


# Chooses evenly spaced representative times from an available time array.
def select_times(times: np.ndarray, n_snapshots: int) -> np.ndarray:
    unique = np.unique(times)
    if unique.size <= n_snapshots:
        return unique
    # Choose actual stored epochs by index. No intermediate physical profiles are generated.
    indices = np.unique(np.linspace(0, unique.size - 1, n_snapshots, dtype=int))
    return unique[indices]


# Section: phase-space projection readers.

# Loads a full phase-space projection CSV into r, w, and g arrays.
def load_phase(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_named_csv(path)
    r = np.unique(data["r"])
    w = np.unique(data["w"])
    g = data["g"].reshape(r.size, w.size)
    return r, w, g


# Loads time-dependent phase snapshots while allowing each frame to have its own w grid.
def load_phase_snapshots(path: Path) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load complete phase snapshots, allowing r/w grids to vary with time.

    Adaptive momentum remaps keep the same number of saved w indices but change
    their coordinates.  A global ``np.unique(w)`` would therefore create a fake
    oversized grid and reshape data incorrectly, so each time slice carries its
    own r and w coordinate arrays.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, invalid_raise=False)
    raw = np.atleast_2d(raw)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    if raw.size == 0:
        return np.array([]), [], [], []

    # The saved index sets are fixed while adaptive momentum remaps change
    # their coordinates. Infer the complete extent from the file, so a live
    # final block ending at a radial-row boundary remains incomplete.
    all_r_indices = np.unique(raw[:, 2].astype(int))
    all_w_indices = np.unique(raw[:, 3].astype(int))
    # A complete frame contains the Cartesian product of the saved r and w index sets.
    expected_cells = all_r_indices.size * all_w_indices.size

    times: list[float] = []
    r_frames: list[np.ndarray] = []
    w_frames: list[np.ndarray] = []
    snapshots: list[np.ndarray] = []
    for t in np.unique(raw[:, 0]):
        rows = raw[raw[:, 0] == t]
        r_indices = np.unique(rows[:, 2].astype(int))
        w_indices = np.unique(rows[:, 3].astype(int))
        if (rows.shape[0] != expected_cells
                or not np.array_equal(r_indices, all_r_indices)
                or not np.array_equal(w_indices, all_w_indices)):
            # Skip incomplete snapshot rows encountered during file writes.
            continue
        # Sort by radial index, then momentum index, before reshaping the table into g(r,w).
        order = np.lexsort((rows[:, 3], rows[:, 2]))
        rows = rows[order]
        pairs = rows[:, 2:4]
        if np.unique(pairs, axis=0).shape[0] != expected_cells:
            # Duplicate index pairs cannot replace missing phase cells.
            continue
        r_by_index = {int(i): float(rows[rows[:, 2].astype(int) == int(i)][0, 4]) for i in r_indices}
        w_by_index = {int(j): float(rows[rows[:, 3].astype(int) == int(j)][0, 5]) for j in w_indices}
        r = np.array([r_by_index[int(i)] for i in r_indices], dtype=float)
        w = np.array([w_by_index[int(j)] for j in w_indices], dtype=float)
        snapshots.append(rows[:, 6].reshape(r.size, w.size))
        times.append(float(t))
        r_frames.append(r)
        w_frames.append(w)
    return np.asarray(times, dtype=float), r_frames, w_frames, snapshots
