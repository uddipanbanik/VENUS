# Store the grids, distribution, accepted clock and output resources of one reference run.
# SimulationState is shared by setup, evolution and serialization, so each sees the same accepted state.

"""Mutable state container for Python Einstein--Vlasov reference runs.

The record owns grids, the evolving distribution, metric/output state, clocks,
and streams.  Keeping these values explicit makes restart-relevant state and
the setup and evolution routines can share one explicit simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .params import EVParams


# Section: escape-diagnostic defaults.

# Creates the default zero-valued escape-statistics dictionary.
def default_escape_stats() -> dict[str, float | int]:
    return {
        "outer_mass": 0.0,
        "w_low_mass": 0.0,
        "w_high_mass": 0.0,
        "total_mass": 0.0,
        "outer_cells": 0,
        "w_low_cells": 0,
        "w_high_cells": 0,
    }


# Section: simulation state dataclass.

# Holds mutable Python solver state, output buffers, and open snapshot resources.
@dataclass
class SimulationState:
    par: EVParams
    r: np.ndarray
    w: np.ndarray
    L: np.ndarray
    f: np.ndarray
    # These integration weights belong to the stored grids and accompany every moment or escape diagnostic.
    r_weights: np.ndarray
    w_weights: np.ndarray
    L_weights: np.ndarray
    radial_step: float
    min_dw: float
    outdir: Path
    # Time and step identify the accepted distribution. last_dt describes its most recent advancement.
    t: float = 0.0
    step: int = 0
    last_dt: float = 0.0
    last_escape: dict[str, float | int] = field(default_factory=default_escape_stats)
    # Accumulate separate estimated losses at the outer radial and momentum boundaries.
    cumulative_escape: dict[str, float] = field(
        default_factory=lambda: {"outer_mass": 0.0, "w_low_mass": 0.0, "w_high_mass": 0.0}
    )
    f_initial_projection: np.ndarray | None = None
    snapshots: list[dict[str, np.ndarray | float | int]] = field(default_factory=list)
    history: list[dict[str, float | int]] = field(default_factory=list)
    phase_snapshot_handle: TextIO | None = None
    phase_snapshot_writer: Any = None

    # Closes the optional phase-snapshot CSV handle if it was opened.
    def close_phase_snapshots(self) -> None:
        if self.phase_snapshot_handle is not None:
            self.phase_snapshot_handle.close()
            self.phase_snapshot_handle = None
            self.phase_snapshot_writer = None

    # Converts the mutable simulation state into the result dictionary consumed by write_outputs.
    def to_result(self, f_final_projection: np.ndarray) -> dict[str, object]:
        assert self.f_initial_projection is not None
        # Expose the final arrays together with saved diagnostics and their parameter record.
        return {
            "r": self.r,
            "w": self.w,
            "L": self.L,
            "f": self.f,
            "f_initial_projection": self.f_initial_projection,
            "f_final_projection": f_final_projection,
            "snapshots": self.snapshots,
            "history": self.history,
            "params": self.par,
            "final_time": self.t,
            "final_step": self.step,
        }
