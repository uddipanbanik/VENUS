#!/usr/bin/env python3
# Normalize an existing GIF's timing records at a requested viewing FPS.
# The byte-level edit preserves compressed image data, palette and source frame order.
# Output atomically replaces the input after writing and syncing a complete temporary GIF.

"""Normalize GIF frame timing, disposal, and looping without re-encoding pixels.

The script edits the raw GIF89a control records of an existing animation.  It
requires a NETSCAPE loop extension and one graphics-control extension per
displayed frame, sets infinite looping and full-frame disposal, and distributes
fractional-centisecond delays cumulatively so the total duration tracks the
requested FPS without systematic per-frame rounding drift.

The original path is replaced atomically only after the complete byte stream
has been written and synchronized to a sibling temporary file.  Image data,
palette entries, and frame order are left unchanged.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


GRAPHICS_CONTROL = b"\x21\xf9\x04"
NETSCAPE_LOOP = b"\x21\xff\x0bNETSCAPE2.0\x03\x01"


def main() -> None:
    """Patch validated GIF control records and atomically replace the input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gif", type=Path)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()

    # A GIF stores loop behavior in the NETSCAPE application extension.
    # A two-byte zero loop count means repeat indefinitely.
    payload = bytearray(args.gif.read_bytes())
    loop_offset = payload.find(NETSCAPE_LOOP)
    if loop_offset < 0:
        raise RuntimeError("GIF has no NETSCAPE2.0 loop extension")
    loop_count_offset = loop_offset + len(NETSCAPE_LOOP)
    payload[loop_count_offset : loop_count_offset + 2] = b"\x00\x00"

    # Every displayed frame has a graphics-control extension (GCE).  Its
    # packed byte contains the disposal mode and its next two bytes contain
    # the frame delay in centiseconds.
    positions: list[int] = []
    cursor = 0
    while True:
        cursor = payload.find(GRAPHICS_CONTROL, cursor)
        if cursor < 0:
            break
        if cursor + 8 > len(payload) or payload[cursor + 7] != 0:
            raise RuntimeError(f"Malformed graphics-control extension at {cursor}")
        positions.append(cursor)
        cursor += 8
    if not positions:
        raise RuntimeError("GIF has no graphics-control extensions")

    # Distribute fractional-centisecond delays cumulatively.  For example,
    # 12 fps alternates integer delays so the total movie duration remains
    # close to N/12 seconds instead of accumulating per-frame rounding error.
    ideal_centiseconds = 100.0 / args.fps
    for index, position in enumerate(positions):
        before = math.floor(index * ideal_centiseconds + 1.0e-12)
        after = math.floor((index + 1) * ideal_centiseconds + 1.0e-12)
        delay = max(1, after - before)
        # Set disposal mode 2 while preserving unrelated packed control bits.
        payload[position + 3] = (payload[position + 3] & 0xE3) | (2 << 2)
        # Encode the integer centisecond delay in the GIF format's little-endian order.
        payload[position + 4 : position + 6] = delay.to_bytes(2, "little")

    # Publish atomically: a monitor sees either the old valid GIF or the fully
    # rewritten one, never a partially modified animation.
    temporary = args.gif.with_name(args.gif.name + ".timing.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(args.gif))
    print(f"normalized {len(positions)} frames at {args.fps:g} fps")


if __name__ == "__main__":
    main()
