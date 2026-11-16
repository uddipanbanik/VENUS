#!/usr/bin/env python3
# Extract the published colored vector curves of the cored SIDM benchmark into a numerical CSV.
# Input is the specified vector PDF/page. Measured panel borders map page coordinates to log radius and each observable's axis values.
# Output retains method, physical epoch, normalized epoch and every qualifying vector point.

"""Extract the vector curves from Figure 3 of the Frank SIDM benchmark paper."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


COLOR_NAMES = {
    (0.0, 0.0, 0.0): "black",
    (0.199951, 0.599609, 0.800781): "nbody",
    (1.0, 0.599609, 0.199951): "sphdsmc",
    (0.599609, 0.199951, 1.0): "gtf",
    (1.0, 0.0, 0.0): "beta_zero",
}

# Figure 3 axes in PDF page coordinates. These values are the common borders
# of the four columns and three rows, measured from the vector axis strokes.
X_EDGES = np.array([82.987720, 199.136464, 315.053836, 431.202580, 547.119952])
Y_EDGES = {
    "density": (604.351170, 720.268542),
    "sigma": (488.202426, 604.351170),
    "beta": (372.285150405, 488.202426),
}
X_LIMITS = (-2.5, 2.8)
Y_LIMITS = {
    "density": (-11.0, 5.0),
    "sigma": (0.0, 0.51),
    "beta": (-1.0, 1.0),
}
TIMES_GYR = (0.0, 2.63, 5.01, 5.67)
T_CORE_MYR = 22.68


@dataclass
class Stroke:
    color: str
    points: np.ndarray


def matrix(values: list[object]) -> np.ndarray:
    a, b, c, d, e, f = (float(value) for value in values)
    return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])


def transform(points: list[tuple[float, float]], ctm: np.ndarray) -> np.ndarray:
    xy1 = np.column_stack((np.asarray(points, dtype=float), np.ones(len(points))))
    return (ctm @ xy1.T).T[:, :2]


def parse_colored_strokes(pdf: Path, page_number: int) -> list[Stroke]:
    """Read one PDF page after reporting an actionable optional-dependency error."""

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Figure-3 vector extraction requires pypdf; install the release "
            "dependencies with 'python3 -m pip install -r requirements.txt'."
        ) from exc
    operations = PdfReader(str(pdf)).pages[page_number - 1].get_contents().operations
    ctm = np.eye(3)
    color: tuple[float, ...] | None = None
    path: list[tuple[float, float]] = []
    stack: list[tuple[np.ndarray, tuple[float, ...] | None]] = []
    strokes: list[Stroke] = []

    for operands, raw_operator in operations:
        operator = raw_operator.decode() if isinstance(raw_operator, bytes) else raw_operator
        if operator == "q":
            stack.append((ctm.copy(), color))
        elif operator == "Q":
            ctm, color = stack.pop()
        elif operator == "cm":
            # Compose PDF drawing transformations so path coordinates refer to the common page axes.
            ctm = ctm @ matrix(operands)
        elif operator == "SC":
            color = tuple(round(float(value), 6) for value in operands)
        elif operator == "m":
            path = [(float(operands[0]), float(operands[1]))]
        elif operator == "l":
            path.append((float(operands[0]), float(operands[1])))
        elif operator in ("S", "s"):
            # Recognized stroke colors identify the published numerical methods. Filled objects and unrelated strokes are excluded.
            color_name = COLOR_NAMES.get(color)
            if color_name is not None and len(path) >= 2:
                strokes.append(Stroke(color_name, transform(path, ctm)))
            path = []
        elif operator in ("n", "f", "F", "f*", "B", "B*", "b", "b*"):
            path = []
    return strokes


def classify_stroke(stroke: Stroke) -> tuple[str, int] | None:
    x_mid = float(np.median(stroke.points[:, 0]))
    # The median page position assigns a curve to one time column and one observable row.
    panel = int(np.searchsorted(X_EDGES, x_mid, side="right") - 1)
    if not 0 <= panel < 4:
        return None
    y_mid = float(np.median(stroke.points[:, 1]))
    for quantity, (bottom, top) in Y_EDGES.items():
        if bottom - 1.0 <= y_mid <= top + 1.0:
            return quantity, panel
    return None


def map_to_data(points: np.ndarray, quantity: str, panel: int) -> tuple[np.ndarray, np.ndarray]:
    x0, x1 = X_EDGES[panel : panel + 2]
    y0, y1 = Y_EDGES[quantity]
    # Affine axis calibration converts page distances to the published log-radius and ordinate values.
    x = X_LIMITS[0] + (points[:, 0] - x0) * (X_LIMITS[1] - X_LIMITS[0]) / (x1 - x0)
    limits = Y_LIMITS[quantity]
    y = limits[0] + (points[:, 1] - y0) * (limits[1] - limits[0]) / (y1 - y0)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--page", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for stroke in parse_colored_strokes(args.pdf, args.page):
        if stroke.color == "black":
            continue
        classification = classify_stroke(stroke)
        if classification is None:
            continue
        quantity, panel = classification
        # The colored legend keys contain only a handful of points and are not
        # monotonic across a substantial fraction of an axis.
        x, y = map_to_data(stroke.points, quantity, panel)
        if len(x) < 20 or np.ptp(x) < 1.0:
            continue
        in_axis = (
            (x >= X_LIMITS[0] - 0.03)
            & (x <= X_LIMITS[1] + 0.03)
            & (y >= Y_LIMITS[quantity][0] - 0.03)
            & (y <= Y_LIMITS[quantity][1] + 0.03)
        )
        x = x[in_axis]
        y = y[in_axis]
        # Keep every in-axis point and order equal-radius records stably before writing the reusable table.
        order = np.argsort(x, kind="stable")
        for x_value, y_value in zip(x[order], y[order]):
            rows.append(
                {
                    "source": stroke.color,
                    "quantity": quantity,
                    "time_gyr": TIMES_GYR[panel],
                    "time_tcore": 1000.0 * TIMES_GYR[panel] / T_CORE_MYR,
                    "log10_r_over_rs": x_value,
                    "value": y_value,
                }
            )

    if not rows:
        raise RuntimeError(
            "No qualifying Frank Figure-3 vector curves were found; verify "
            "the paper edition, page number, and vector (not raster) PDF."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = sorted(
        {
            (row["source"], row["quantity"], row["time_gyr"])
            for row in rows
        }
    )
    print(f"Wrote {len(rows)} vector points in {len(groups)} curve groups to {args.output}")
    for group in groups:
        count = sum(
            (row["source"], row["quantity"], row["time_gyr"]) == group
            for row in rows
        )
        print(f"{group[0]:8s} {group[1]:7s} t={group[2]:4.2f} Gyr: {count:4d} points")


if __name__ == "__main__":
    main()
