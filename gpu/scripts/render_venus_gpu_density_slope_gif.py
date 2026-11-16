#!/usr/bin/env python3
# Render named-column GPU profiles with the shared CPU/GPU density-slope renderer.
# Input columns carry actual time, radius, density and effective marker count in code units.
# The shared implementation owns time selection, derivatives and output-file finalization.

"""Render GPU weighted-PIC density/slope profiles with the shared renderer.

GPU and CPU weighted-PIC runs write the same named-column ``profile_*.csv``
schema: ``time``, ``radius``, ``density``, and effective marker count ``neff``.
This front end delegates to the maintained hardware-independent renderer in
``cpu/tools`` so both backends use identical density floors, local-polynomial
``d log(rho) / d log(r)`` estimates, colors, and fixed movie limits.

Run directories and the output path are explicit command-line arguments.
Exact mode requires common saved times, while ``--freeze-after-last`` holds
each collisional curve at its latest complete state on the collisionless
timeline.  The shared writer streams full frames to a temporary GIF and
atomically replaces the requested output after finalization, with frame-time
sidecars beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Locate the bundled shared renderer relative to this file so the entry point is independent of the working directory.
CPU_TOOLS = Path(__file__).resolve().parents[2] / "cpu" / "tools"
if str(CPU_TOOLS) not in sys.path:
    sys.path.insert(0, str(CPU_TOOLS))

from render_weighted_pic_subhalo_strength_scan_gif import (  # noqa: E402
    parse_arguments as parse_shared_arguments,
    render as render_shared_profiles,
)


def main() -> None:
    """Parse explicit profile paths and render the shared atomic GIF output."""
    # Pass the parsed scientific and display options through unchanged to the shared implementation.
    output = render_shared_profiles(parse_shared_arguments())
    print(output)


if __name__ == "__main__":
    main()
