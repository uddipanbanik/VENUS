#!/usr/bin/env python3
# Convert CPU unequal-Plummer lookup arrays into a flat generated CUDA header.
# Inputs are one CPU header and the destination filename. Numerical entries remain their original decimal strings.
# Array cardinality checks preserve the two interpolation axes and table ordering.

"""Flatten the authoritative VENUS CPU Plummer table for CUDA upload.

The CPU header deliberately uses nested std::array objects.  CUDA kernels use
flat device arrays instead, but the numerical entries must have one source of
truth.  CMake runs this script against the bundled VENUS CPU include tree and
places the generated header in the build directory; no values are rounded or
hand-copied into the GPU source.

The two positional inputs are the CPU table header and generated-header path;
the only output is that deterministic header.  No solver or particle state is
read.  Expected axis/table cardinalities are checked before writing, and the
source SHA-256 is embedded so a CUDA binary can be traced to its exact physics
table.  CMake serializes this build dependency before compiling the subhalo
kernel, so the script itself requires no parallel coordination.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re


NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def between(text: str, start: str, end: str) -> str:
    # Use named declaration boundaries to isolate one authoritative array at a time.
    begin = text.index(start) + len(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def values(block: str) -> list[str]:
    # Keep decimal tokens as strings, so conversion introduces no floating-point rounding.
    return NUMBER.findall(block)


def format_array(name: str, numbers: list[str]) -> str:
    rows = []
    # Group the existing tokens into readable rows without reordering their flattened indices.
    for offset in range(0, len(numbers), 4):
        rows.append("    " + ", ".join(numbers[offset : offset + 4]))
    return (
        f"static constexpr double {name}[{len(numbers)}] = {{\n"
        + ",\n".join(rows)
        + "\n};\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")

    # Read the mass-ratio axis, speed axis and rate tables in their existing nested-array order.
    mass_ratio = values(between(text, "mass_ratio = {", "};"))
    speed = values(between(text, "speed = {", "};"))
    capture = values(between(text, "mass_ratio_count> capture = {{", "}};"))
    scatter = values(between(text, "mass_ratio_count> scatter = {{", "}};"))
    high = values(
        between(text, "scatter_high_speed_coefficient = {", "};")
    )
    capture_bound = values(
        between(text, "capture_u_shape_bound =", ";")
    )
    scatter_bound = values(
        between(text, "scatter_u_shape_bound =", ";")
    )

    # A shape mismatch indicates an incompatible declaration or parse and stops header generation.
    expected = {
        "mass_ratio": (mass_ratio, 11),
        "speed": (speed, 72),
        "capture": (capture, 11 * 72),
        "scatter": (scatter, 11 * 72),
        "scatter_high_speed_coefficient": (high, 11),
        "capture_u_shape_bound": (capture_bound, 1),
        "scatter_u_shape_bound": (scatter_bound, 1),
    }
    for name, (entries, count) in expected.items():
        if len(entries) != count:
            raise SystemExit(
                f"could not parse {name}: expected {count}, found {len(entries)}"
            )

    # Embed the exact input-header digest beside the arrays to identify their numerical source.
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    body = [
        "#pragma once\n",
        "// Generated from the authoritative VENUS CPU unequal-Plummer table.\n",
        f'static constexpr const char source_sha256[] = "{digest}";\n',
        "namespace venus_gpu { namespace generated_subhalo_table {\n",
        "static constexpr int mass_ratio_count = 11;\n",
        "static constexpr int speed_count = 72;\n",
        format_array("mass_ratio", mass_ratio),
        format_array("speed", speed),
        format_array("capture", capture),
        format_array("scatter", scatter),
        format_array("scatter_high_speed_coefficient", high),
        f"static constexpr double capture_u_shape_bound = {capture_bound[0]};\n",
        f"static constexpr double scatter_u_shape_bound = {scatter_bound[0]};\n",
        "}}  // namespace venus_gpu::generated_subhalo_table\n",
    ]
    # Create the build output directory and write the deterministic generated header.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(body), encoding="utf-8")


if __name__ == "__main__":
    main()
