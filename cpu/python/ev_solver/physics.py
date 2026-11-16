# Expose moment integration and metric reconstruction through one import module.
# The imported functions consume spherical distributions/profiles in G=c=1 code units.

"""Compatibility umbrella for the Python physics kernels."""

from __future__ import annotations

# Section: re-exported physics helpers.
# Re-export the moment and metric functions through ev_solver.physics.
# Their implementations are defined in the modules imported below.

# Keep the actual implementations in metric.py and moments.py so each physical operation has one definition.
from .metric import cumulative_trapezoid, solve_metric
from .moments import compute_moments

__all__ = ["compute_moments", "cumulative_trapezoid", "solve_metric"]
