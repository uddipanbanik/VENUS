#!/usr/bin/env python3
# Compare Schwarzschild and ingoing EF fields at common spacetime events using the solved null-coordinate map.
# Inputs include face-aware saved metrics, rest-weight realization identifiers and authenticated donor/map artifacts in G=c=1 units.
# The GIF and frame ledger distinguish mapped overlap from later native EF slices and retain the static-observer density's domain of validity.

"""Render one Venus-style SC/EF black-hole profile animation.

The first part of the animation overlays Schwarzschild/polar--areal (SC) and
ingoing Eddington--Finkelstein (EF) fields at the *same spacetime events*.
Equal values of ``time`` and ``advanced_time`` are never paired.  Instead the
renderer imports the production parity mapper, solves the ingoing eikonal map

    v_r = exp(lambda-mu) v_tau,

with the authenticated outer-worldtube clock, checks its n-versus-2n and
forward/inverse composition errors, and samples EF independently at
``v(tau,r)``.  Once the SC chart ends, later saved EF slices may continue as
EF-only frames so horizon formation and exterior evolution remain visible.

The three panels are, from left to right, ``log10(rho)`` versus ``log10(r)``,
``d log(rho)/d log(r)`` versus ``log10(r)``, and compactness ``2m/r`` versus
linear ``r``.  Before the horizon, EF ``rho`` is the static-observer energy
density ``T_vv/(a b^2)``, exactly the moment used by the parity comparator.
That observer ceases to exist where ``a<=0``; EF density and slope are therefore
masked there rather than silently switching to a different density.  The
compactness panel remains defined and shows the full saved EF radial slice.

The GIF is streamed and atomically published.  Initial/middle/final PNGs, a
CSV frame ledger, and a JSON provenance sidecar record the exact coordinate
times, mapped radial domain, inputs, hashes, and plotting definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "venus-matplotlib-cache")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOOLS_DIRECTORY = Path(__file__).resolve().parent
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))

import compare_wpic_dynamic_coordinates as parity
from render_weighted_einstein_vlasov_pic_profile_gifs import (
    boundary_aware_log_profile,
    finite_density_profile,
    padded_limits,
    style_axis,
)
from render_weighted_pic_pure_gamma_gifs import AtomicGif, figure_to_rgb


@dataclass
class ProfileCurve:
    """One radial density/compactness curve in one coordinate chart."""

    radius: np.ndarray
    density: np.ndarray
    compactness: np.ndarray
    origin_parity: bool


@dataclass
class FrameData:
    """One animation frame and the coordinate ledger behind it."""

    phase: str
    sc_tau: float | None
    ef_v_min: float
    ef_v_max: float
    sc: ProfileCurve | None
    ef: ProfileCurve
    mapped_radius_min: float | None
    mapped_radius_max: float | None
    mapped_radial_fraction: float | None
    maximum_forward_refinement_error: float | None
    maximum_inverse_refinement_error: float | None
    maximum_composition_error: float | None


@dataclass
class PairContext:
    """Authenticated profiles and null-map constants used by the renderer."""

    sc: parity.RectilinearProfile
    ef: parity.RectilinearProfile
    manifest: dict
    artifacts: dict
    manifest_path: Path
    boundary_radius: float
    evolution_outer_radius: float
    v_offset: float
    donor_history_audit: dict
    summary_audit: dict


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one input or derived product."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_manifest_value(manifest: dict, *paths: str) -> float:
    """Read one finite numeric manifest value through explicit key aliases."""
    value = parity.nested_value(manifest, *paths)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            "Null-map manifest lacks a finite numeric value for " + paths[0]
        )
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(
            "Null-map manifest has a non-finite value for " + paths[0]
        )
    return result


def load_summary(run_directory: Path, expected_coordinates: str) -> tuple[Path, dict]:
    """Load one run summary and require its declared coordinate chart."""
    path = run_directory / "summary.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {path}: {error}") from error
    if value.get("coordinates") != expected_coordinates:
        raise RuntimeError(
            f"{path} does not declare coordinates={expected_coordinates}"
        )
    high = value.get("high_compactness")
    if not isinstance(high, dict):
        raise RuntimeError(f"{path} has no high_compactness object")
    return path, high


def require_close_values(label: str, values: tuple[float, ...]) -> float:
    """Require a manifest value and both summary values to agree."""
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{label} is not finite")
    if any(not parity.close(values[0], value, scale=1.0) for value in values[1:]):
        raise RuntimeError(f"{label} disagrees between SC, EF, and manifest")
    return values[0]


def load_pair_context(
    schwarzschild_directory: Path,
    ef_directory: Path,
    manifest_path: Path,
) -> PairContext:
    """Load a hash-authenticated SC/EF pair and production null-map history."""
    sc_summary_path, sc_summary = load_summary(
        schwarzschild_directory, "schwarzschild"
    )
    ef_summary_path, ef_summary = load_summary(
        ef_directory, "eddington_finkelstein"
    )
    if sc_summary.get("eikonal_mapped_initial_data") is not True \
            or ef_summary.get("eikonal_mapped_initial_data") is not True:
        raise RuntimeError(
            "Both summaries must authenticate eikonal-mapped initial data"
        )
    # The two coordinate descriptions must originate from the same coordinate-neutral physical particle realization.
    sc_base = parity.normalized_fingerprint(sc_summary.get("base_ic_fingerprint"))
    ef_base = parity.normalized_fingerprint(ef_summary.get("base_ic_fingerprint"))
    if sc_base is None or sc_base != ef_base:
        raise RuntimeError("SC and EF base initial-condition fingerprints differ")

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {manifest_path}: {error}") from error
    try:
        # Authenticate each map input before reading its worldtube clock and geometry.
        artifacts = parity.manifest_artifacts(manifest, str(manifest_path))
    except parity.ProfileError as error:
        raise RuntimeError(f"Null-map artifact authentication failed: {error}") from error
    failed_artifacts = [
        name for name, record in artifacts.items()
        if record.get("verified") is not True
    ]
    if failed_artifacts:
        raise RuntimeError(
            "Null-map artifact SHA-256 mismatch: " + ", ".join(failed_artifacts)
        )

    manifest_boundary = numeric_manifest_value(
        manifest,
        "map_worldtube_radius",
        "eikonal_boundary_radius",
        "outer_boundary_radius",
        "outer_radius",
        "mapping.outer_radius",
        "eikonal.outer_radius",
        "boundary.radius",
    )
    manifest_offset = numeric_manifest_value(
        manifest,
        "eikonal_v_offset",
        "ef_v_offset",
        "v_offset",
        "outer_reverse_time",
        "mapping.v_offset",
        "mapping.outer_reverse_time",
        "eikonal.v_offset",
        "extraction.outer_reverse_time",
        "boundary.v_offset",
    )
    boundary_radius = require_close_values(
        "map worldtube radius",
        (
            float(sc_summary.get("eikonal_boundary_radius", math.nan)),
            float(ef_summary.get("eikonal_boundary_radius", math.nan)),
            manifest_boundary,
        ),
    )
    v_offset = require_close_values(
        "EF clock offset",
        (
            float(sc_summary.get("eikonal_v_offset", math.nan)),
            float(ef_summary.get("eikonal_v_offset", math.nan)),
            manifest_offset,
        ),
    )
    evolution_outer_radius = float(
        manifest.get("evolution_outer_radius", boundary_radius)
    )
    if not math.isfinite(evolution_outer_radius) \
            or evolution_outer_radius < boundary_radius:
        raise RuntimeError("Null-map evolution outer radius is invalid")

    sc = parity.RectilinearProfile(
        str(schwarzschild_directory / "high_compactness_profiles.csv"),
        "time",
        {
            "m", "a", "lambda", "mu", "lambda_t", "rho",
            "p_radial", "p_tangential_sum", "j",
        },
    )
    ef = parity.RectilinearProfile(
        str(ef_directory / "high_compactness_profiles.csv"),
        "advanced_time",
        {
            "a", "b", "m", "T_rr", "S", "T_vv", "T_vr",
            "T_theta_theta",
        },
    )
    if not sc.has_production_metric_reconstruction \
            or not ef.has_production_metric_reconstruction:
        raise RuntimeError(
            "Profiles lack the center/face metric data used by production mapping"
        )
    # Extend saved metric sampling to the declared exterior domain with the same analytic vacuum rules used by the comparator.
    sc.configure_vacuum_outer_continuation(
        evolution_outer_radius, "schwarzschild"
    )
    ef.configure_vacuum_outer_continuation(
        evolution_outer_radius, "eddington_finkelstein"
    )
    _donor_name, donor_record = parity.artifact_by_alias(
        artifacts,
        ("donor_history", "history", "spacetime_history"),
    )
    if donor_record is None or donor_record.get("verified") is not True:
        raise RuntimeError(
            "Null-map manifest has no verified dense donor history for tau<0"
        )
    try:
        # Earlier eikonal traces need the saved time-reversed donor history, including the actual face reconstruction.
        donor_history_audit = parity.prepend_time_reversed_donor_history(
            sc, donor_record["path"]
        )
    except (OSError, parity.ProfileError) as error:
        raise RuntimeError(f"Dense donor history failed validation: {error}") from error
    if donor_history_audit.get("format_version") != 2 \
            or donor_history_audit.get("production_metric_reconstruction") is not True:
        raise RuntimeError("Dense donor history is not production face-data version 2")

    return PairContext(
        sc=sc,
        ef=ef,
        manifest=manifest,
        artifacts=artifacts,
        manifest_path=manifest_path.resolve(),
        boundary_radius=boundary_radius,
        evolution_outer_radius=evolution_outer_radius,
        v_offset=v_offset,
        donor_history_audit=donor_history_audit,
        summary_audit={
            "schwarzschild": str(sc_summary_path.resolve()),
            "eddington_finkelstein": str(ef_summary_path.resolve()),
            "base_ic_fingerprint": sc_base,
        },
    )


def cell_radii(profile: parity.RectilinearProfile) -> np.ndarray:
    """Return cell centers while excluding the exact outer-face audit row."""
    radii = np.asarray(profile.radii, dtype=float)
    if profile.metric_reconstruction is not None:
        first = profile.metric_reconstruction[0]
        # Native CSV slices store one array per reconstruction field; donor
        # history slices retain the older list with one record per cell.
        count = len(first["face_left_r"]) if isinstance(first, dict) else len(first)
        return radii[:count]
    return radii


def evenly_spaced_values(values: list[float], requested: int) -> list[float]:
    """Select endpoint-preserving, index-uniform values without duplicates."""
    if requested <= 0 or not values:
        return []
    if requested >= len(values):
        return values
    if requested == 1:
        return [values[-1]]
    indices = parity.evenly_spaced_indices(len(values), requested)
    return [values[index] for index in indices]


def reconstruct_ef_density(a: float, b: float, t_vv: float, minimum_a: float) -> float:
    """Return the SC-frame EF energy density outside the trapped chart region."""
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(t_vv)):
        return math.nan
    if a <= minimum_a or b <= 0.0:
        return math.nan
    # The EF coordinate stress becomes static-observer energy density through this observer normalization, valid only in the declared a-positive domain.
    density = t_vv / (a * b * b)
    return density if math.isfinite(density) and density > 0.0 else math.nan


def direct_curve(
    profile: parity.RectilinearProfile,
    coordinate_time: float,
    coordinates: str,
    minimum_a: float,
) -> ProfileCurve:
    """Read one native saved-time curve, reconstructing EF density if needed."""
    radius = cell_radii(profile)
    density: list[float] = []
    compactness: list[float] = []
    for value in radius:
        mass = profile.sample("m", coordinate_time, float(value))
        compactness.append(2.0 * mass / float(value))
        if coordinates == "schwarzschild":
            sample = profile.sample("rho", coordinate_time, float(value))
            density.append(sample if sample > 0.0 else math.nan)
        elif coordinates == "eddington_finkelstein":
            density.append(reconstruct_ef_density(
                profile.sample("a", coordinate_time, float(value)),
                profile.sample("b", coordinate_time, float(value)),
                profile.sample("T_vv", coordinate_time, float(value)),
                minimum_a,
            ))
        else:
            raise ValueError(f"Unknown coordinate chart {coordinates}")
    finite_density = np.isfinite(density)
    reaches_origin = bool(
        finite_density.size
        and finite_density[0]
        and np.isclose(radius[0], cell_radii(profile)[0], rtol=0.0, atol=1.0e-14)
    )
    return ProfileCurve(
        radius=radius,
        density=np.asarray(density, dtype=float),
        compactness=np.asarray(compactness, dtype=float),
        origin_parity=reaches_origin,
    )


def mapped_overlap_frame(
    sc: parity.RectilinearProfile,
    ef: parity.RectilinearProfile,
    tau: float,
    boundary_radius: float,
    v_offset: float,
    eikonal_substeps: int,
    eikonal_tolerance: float,
    composition_tolerance: float,
    minimum_a: float,
    r_min: float,
    r_max: float,
    mapped_radial_samples: int,
    density_floor_fraction: float,
) -> FrameData:
    """Map one SC slice into EF and retain its contiguous common exterior.

    ``a<=minimum_a`` and genuine profile boundaries can end the common
    exterior.  A step-refinement or composition failure is a numerical hole,
    not a chart boundary, and aborts rendering instead of being hidden.
    """
    all_sc_radii = cell_radii(sc)
    eligible_full = np.asarray([
        radius for radius in all_sc_radii
        if r_min <= radius <= r_max and radius <= boundary_radius
    ], dtype=float)
    if eligible_full.size < 3:
        raise RuntimeError("Fewer than three SC cells lie in the mapped display domain")
    requested = min(mapped_radial_samples, eligible_full.size)
    # Put most costly exact eikonal samples where the matter actually lives,
    # while retaining a few full-domain nodes for the compactness curve.
    sc_density_on_grid = np.asarray([
        sc.sample("rho", tau, float(radius)) for radius in eligible_full
    ])
    positive = np.isfinite(sc_density_on_grid) & (sc_density_on_grid > 0.0)
    if np.any(positive):
        support = np.flatnonzero(
            positive
            & (sc_density_on_grid
               > np.max(sc_density_on_grid[positive]) * density_floor_fraction)
        )
    else:
        support = np.empty(0, dtype=int)
    # Concentrate costly coordinate traces within matter support while retaining outer nodes for compactness and domain coverage.
    support_count = min(
        support.size,
        max(5, int(round(0.75 * requested))),
    )
    domain_count = max(2, requested - support_count)
    selected = set(parity.evenly_spaced_indices(eligible_full.size, domain_count))
    if support_count:
        selected.update(
            int(support[index])
            for index in parity.evenly_spaced_indices(support.size, support_count)
        )
    radial_indices = sorted(selected)
    eligible = eligible_full[radial_indices]

    records: list[tuple[float, float, float, float, float, float, float]] = []
    chart_boundary_seen = False
    maximum_forward = 0.0
    maximum_inverse = 0.0
    maximum_composition = 0.0
    # Trace inward from the known outer clock so the accepted comparison remains a connected exterior interval.
    for radius in eligible[::-1]:
        radius_value = float(radius)
        try:
            coarse_v, _coarse_a, coarse_log_a = parity.trace_to_outer_boundary(
                sc, tau, radius_value, eikonal_substeps, v_offset, boundary_radius
            )
            mapped_v, _map_a, map_log_a = parity.trace_to_outer_boundary(
                sc, tau, radius_value, 2 * eikonal_substeps, v_offset,
                boundary_radius,
            )
        except parity.OutsideProfile:
            chart_boundary_seen = True
            continue
        except parity.ProfileError as error:
            raise RuntimeError(
                f"Forward eikonal failed at tau={tau:g}, r={radius_value:g}: {error}"
            ) from error
        # Compare n and 2n forward traces in both coordinate time and log Jacobian before accepting the refined map.
        forward_error = max(
            parity.rk4_step_doubling_change(coarse_v, mapped_v),
            parity.rk4_step_doubling_change(coarse_log_a, map_log_a),
        )
        maximum_forward = max(maximum_forward, forward_error)
        if forward_error > eikonal_tolerance:
            raise RuntimeError(
                "Forward eikonal n-versus-2n error exceeds tolerance at "
                f"tau={tau:g}, r={radius_value:g}: "
                f"{forward_error:.6g}>{eikonal_tolerance:.6g}"
            )
        try:
            coarse_tau, _coarse_f, coarse_log_f = (
                parity.inverse_trace_to_outer_boundary(
                    ef, mapped_v, radius_value, eikonal_substeps, v_offset,
                    boundary_radius,
                )
            )
            inverse_tau, _map_f, map_log_f = parity.inverse_trace_to_outer_boundary(
                ef, mapped_v, radius_value, 2 * eikonal_substeps, v_offset,
                boundary_radius,
            )
        except parity.OutsideProfile:
            chart_boundary_seen = True
            continue
        except parity.ProfileError as error:
            if "a<=0 or b<=0" in str(error):
                chart_boundary_seen = True
                continue
            raise RuntimeError(
                f"Inverse eikonal failed at tau={tau:g}, r={radius_value:g}: {error}"
            ) from error
        # The inverse trace supplies an independent numerical refinement check in the opposite coordinate chart.
        inverse_error = max(
            parity.rk4_step_doubling_change(coarse_tau, inverse_tau),
            parity.rk4_step_doubling_change(coarse_log_f, map_log_f),
        )
        maximum_inverse = max(maximum_inverse, inverse_error)
        if inverse_error > eikonal_tolerance:
            raise RuntimeError(
                "Inverse eikonal n-versus-2n error exceeds tolerance at "
                f"tau={tau:g}, r={radius_value:g}: "
                f"{inverse_error:.6g}>{eikonal_tolerance:.6g}"
            )
        # Compose forward and inverse times/Jacobians to test that both descriptions refer to the same event.
        composition_error = max(
            abs(inverse_tau - tau) / max(1.0, abs(tau), abs(inverse_tau)),
            parity.jacobian_composition_error(map_log_a, map_log_f),
        )
        maximum_composition = max(maximum_composition, composition_error)
        if composition_error > composition_tolerance:
            raise RuntimeError(
                "Forward/inverse map composition exceeds tolerance at "
                f"tau={tau:g}, r={radius_value:g}: "
                f"{composition_error:.6g}>{composition_tolerance:.6g}"
            )
        try:
            sc_a = sc.sample("a", tau, radius_value)
            ef_a = ef.sample("a", mapped_v, radius_value)
            ef_b = ef.sample("b", mapped_v, radius_value)
        except parity.OutsideProfile:
            chart_boundary_seen = True
            continue
        if sc_a <= minimum_a or ef_a <= minimum_a or ef_b <= 0.0:
            chart_boundary_seen = True
            continue
        # Once an inner chart boundary is reached, later re-entry would indicate disconnected coverage and aborts the mapped frame.
        if chart_boundary_seen:
            raise RuntimeError(
                "Mapped support re-entered outside an inner chart boundary at "
                f"tau={tau:g}, r={radius_value:g}"
            )
        sc_density = sc.sample("rho", tau, radius_value)
        ef_t_vv = ef.sample("T_vv", mapped_v, radius_value)
        ef_density = reconstruct_ef_density(
            ef_a,
            ef_b,
            ef_t_vv,
            minimum_a,
        )
        if not math.isfinite(sc_density) or not math.isfinite(ef_t_vv):
            raise RuntimeError(
                f"Mapped density source is not finite at tau={tau:g}, "
                f"r={radius_value:g}"
            )
        # Exact outer-vacuum cells legitimately carry rho=T_vv=0.  Keep their
        # compactness and mapping evidence, but mask them before logarithms.
        sc_density = sc_density if sc_density > 0.0 else math.nan
        sc_mass = sc.sample("m", tau, radius_value)
        ef_mass = ef.sample("m", mapped_v, radius_value)
        records.append((
            radius_value,
            mapped_v,
            sc_density,
            ef_density,
            2.0 * sc_mass / radius_value,
            2.0 * ef_mass / radius_value,
            ef_a,
        ))

    if len(records) < 3:
        raise RuntimeError(f"No resolved mapped exterior remains at tau={tau:g}")
    records.reverse()
    values = np.asarray(records, dtype=float)
    radius = values[:, 0]
    reaches_origin = bool(np.isclose(
        radius[0], all_sc_radii[0], rtol=0.0,
        atol=32.0 * np.finfo(float).eps * max(1.0, abs(all_sc_radii[0])),
    ))
    return FrameData(
        phase="same_event_overlap",
        sc_tau=float(tau),
        ef_v_min=float(np.min(values[:, 1])),
        ef_v_max=float(np.max(values[:, 1])),
        sc=ProfileCurve(
            radius=radius,
            density=values[:, 2],
            compactness=values[:, 4],
            origin_parity=reaches_origin,
        ),
        ef=ProfileCurve(
            radius=radius,
            density=values[:, 3],
            compactness=values[:, 5],
            origin_parity=reaches_origin,
        ),
        mapped_radius_min=float(radius[0]),
        mapped_radius_max=float(radius[-1]),
        mapped_radial_fraction=float(radius.size / eligible.size),
        maximum_forward_refinement_error=maximum_forward,
        maximum_inverse_refinement_error=maximum_inverse,
        maximum_composition_error=maximum_composition,
    )


def build_frames(context: PairContext, arguments: argparse.Namespace) -> list[FrameData]:
    """Build mapped overlap frames followed by optional native EF-only frames."""
    # Leave a floating-point margin at the terminal EF clock while selecting outer-worldtube overlaps.
    terminal_clock_margin = 8.0e-13 * max(
        1.0, abs(context.ef.times[-1]), abs(context.v_offset)
    )
    sc_native_times = [
        float(value) for value in context.sc.times
        if value >= 0.0
        and context.ef.times[0] <= value + context.v_offset
        and value + context.v_offset
        < context.ef.times[-1] - terminal_clock_margin
    ]
    if len(sc_native_times) < 2:
        raise RuntimeError("SC and EF outputs have fewer than two outer-clock overlaps")

    last_outer_v = sc_native_times[-1] + context.v_offset
    clock_tolerance = 1.0e-12 * max(
        1.0, abs(last_outer_v), abs(context.ef.times[-1])
    )
    ef_later_times = [
        float(value) for value in context.ef.times
        if value > last_outer_v + clock_tolerance
    ]
    if ef_later_times:
        post_count = max(2, int(round(arguments.frames * arguments.ef_only_fraction)))
        post_count = min(post_count, arguments.frames - 2, len(ef_later_times))
    else:
        post_count = 0
    overlap_count = min(
        arguments.frames - post_count,
        arguments.maximum_same_event_frames,
        len(sc_native_times),
    )
    if ef_later_times:
        post_count = min(arguments.frames - overlap_count, len(ef_later_times))
    overlap_times = evenly_spaced_values(sc_native_times, overlap_count)
    frames = [
        mapped_overlap_frame(
            context.sc,
            context.ef,
            tau,
            context.boundary_radius,
            context.v_offset,
            arguments.eikonal_substeps,
            arguments.eikonal_tolerance,
            arguments.composition_tolerance,
            arguments.minimum_a,
            arguments.r_min,
            arguments.r_max,
            arguments.mapped_radial_samples,
            arguments.density_floor_fraction,
        )
        for tau in overlap_times
    ]
    selected_ef_later_times = (
        evenly_spaced_values(ef_later_times, post_count) if post_count else []
    )
    # Later frames use complete native EF slices after the SC comparison ends, with that distinct phase recorded explicitly.
    for advanced_time in selected_ef_later_times:
        curve = direct_curve(
            context.ef,
            advanced_time,
            "eddington_finkelstein",
            arguments.minimum_a,
        )
        frames.append(FrameData(
            phase="ef_only_after_sc_chart_limit",
            sc_tau=None,
            ef_v_min=advanced_time,
            ef_v_max=advanced_time,
            sc=None,
            ef=curve,
            mapped_radius_min=None,
            mapped_radius_max=None,
            mapped_radial_fraction=None,
            maximum_forward_refinement_error=None,
            maximum_inverse_refinement_error=None,
            maximum_composition_error=None,
        ))
    if len(frames) < 2:
        raise RuntimeError("At least two complete frames are required")
    return frames


def derived_curve(
    curve: ProfileCurve,
    density_floor_fraction: float,
    slope_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw log-density and a boundary-aware logarithmic slope.

    The exact same-event map is intentionally sampled on a modest number of
    radii because tracing every point independently is quadratic in radial
    resolution.  Fitting those sparse density points can visibly flatten a
    narrow shell, so the density panel connects their unmodified values.
    Slope remains the documented local-polynomial diagnostic.
    """
    # Density artists use the unmodified mapped samples. The separate local fit supplies the logarithmic slope.
    density_radius, density = finite_density_profile(
        curve.radius,
        curve.density,
        density_floor_fraction,
    )
    log_density = np.log10(density)
    slope_radius, _unused_density, slope = boundary_aware_log_profile(
        curve.radius,
        curve.density,
        density_floor_fraction,
        slope_window,
        curve.origin_parity,
    )
    return (
        np.log10(density_radius),
        log_density,
        np.log10(slope_radius),
        slope,
    )


def collect_plot_limits(
    frames: list[FrameData],
    initial: ProfileCurve,
    arguments: argparse.Namespace,
) -> dict[str, tuple[float, float]]:
    """Choose one set of limits from every displayed curve and frame."""
    density_x: list[np.ndarray] = []
    density_y: list[np.ndarray] = []
    slope_y: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    compactness: list[np.ndarray] = []
    for curve in [initial] + [
        value for frame in frames for value in (frame.sc, frame.ef)
        if value is not None
    ]:
        log_rho_r, log_rho, _slope_r, slope = derived_curve(
            curve,
            arguments.density_floor_fraction,
            arguments.slope_window,
        )
        density_x.append(log_rho_r)
        density_y.append(log_rho)
        slope_y.append(slope)
        finite_radius = curve.radius[np.isfinite(curve.radius) & (curve.radius > 0.0)]
        radii.append(finite_radius)
        compactness.append(curve.compactness[np.isfinite(curve.compactness)])
    joined_log_radius = np.concatenate(density_x)
    if arguments.minimum_log_radius is not None:
        joined_log_radius = joined_log_radius[
            joined_log_radius >= arguments.minimum_log_radius
        ]
    joined_radius = np.concatenate(radii)
    if arguments.r_max < math.inf:
        joined_radius = joined_radius[joined_radius <= arguments.r_max]
    cmax = float(np.max(np.concatenate(compactness)))
    return {
        "log_radius": (
            float(np.min(joined_log_radius)),
            float(np.max(joined_log_radius)),
        ),
        "radius": (float(np.min(joined_radius)), float(np.max(joined_radius))),
        "density": padded_limits(density_y, 0.002, 0.998, 1.0),
        "slope": (arguments.slope_min, arguments.slope_max),
        "compactness": (0.0, max(1.05, 1.06 * cmax)),
    }


def write_frame_ledger(output: Path, frames: list[FrameData]) -> Path:
    """Write exact frame clocks and mapping diagnostics as a CSV sidecar."""
    path = output.with_name(output.stem + "_frames.csv")
    temporary = path.with_name(path.name + ".tmp")
    fields = [
        "frame", "phase", "sc_tau", "ef_v_min", "ef_v_max",
        "mapped_radius_min", "mapped_radius_max", "mapped_radial_fraction",
        "maximum_forward_refinement_error",
        "maximum_inverse_refinement_error",
        "maximum_composition_error",
    ]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, frame in enumerate(frames):
            writer.writerow({
                "frame": index,
                "phase": frame.phase,
                "sc_tau": "" if frame.sc_tau is None else f"{frame.sc_tau:.17g}",
                "ef_v_min": f"{frame.ef_v_min:.17g}",
                "ef_v_max": f"{frame.ef_v_max:.17g}",
                "mapped_radius_min": (
                    "" if frame.mapped_radius_min is None
                    else f"{frame.mapped_radius_min:.17g}"
                ),
                "mapped_radius_max": (
                    "" if frame.mapped_radius_max is None
                    else f"{frame.mapped_radius_max:.17g}"
                ),
                "mapped_radial_fraction": (
                    "" if frame.mapped_radial_fraction is None
                    else f"{frame.mapped_radial_fraction:.17g}"
                ),
                "maximum_forward_refinement_error": (
                    "" if frame.maximum_forward_refinement_error is None
                    else f"{frame.maximum_forward_refinement_error:.17g}"
                ),
                "maximum_inverse_refinement_error": (
                    "" if frame.maximum_inverse_refinement_error is None
                    else f"{frame.maximum_inverse_refinement_error:.17g}"
                ),
                "maximum_composition_error": (
                    "" if frame.maximum_composition_error is None
                    else f"{frame.maximum_composition_error:.17g}"
                ),
            })
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def render(arguments: argparse.Namespace) -> Path:
    """Validate inputs, build same-event frames, and atomically write the GIF."""
    context = load_pair_context(
        arguments.schwarzschild,
        arguments.ef,
        arguments.manifest,
    )
    frames = build_frames(context, arguments)
    first_native_sc_time = min(value for value in context.sc.times if value >= 0.0)
    initial = direct_curve(
        context.sc, first_native_sc_time, "schwarzschild", arguments.minimum_a
    )
    limits = collect_plot_limits(frames, initial, arguments)

    figure, axes = plt.subplots(1, 3, figsize=(16.8, 5.45))
    figure.subplots_adjust(
        left=0.065, right=0.985, bottom=0.19, top=0.79, wspace=0.28
    )
    ylabels = (
        r"$\log_{10}\rho$",
        r"$\mathrm{d}\log\rho/\mathrm{d}\log r$",
        r"$2m/r$",
    )
    for index, (axis, ylabel) in enumerate(zip(axes, ylabels)):
        style_axis(axis)
        axis.set_xlabel(
            r"$\log_{10} r$" if index < 2 else r"$r$",
            fontsize=21,
        )
        axis.set_ylabel(ylabel, fontsize=21)
    axes[0].set_xlim(*limits["log_radius"])
    axes[1].set_xlim(*limits["log_radius"])
    axes[2].set_xlim(*limits["radius"])
    axes[0].set_ylim(*limits["density"])
    axes[1].set_ylim(*limits["slope"])
    axes[2].set_ylim(*limits["compactness"])
    axes[2].axhline(1.0, color="0.45", linestyle=":", linewidth=2.0, zorder=1)
    axes[2].text(
        0.98,
        1.0,
        r"$2m/r=1$",
        transform=axes[2].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=14,
        color="0.35",
    )

    initial_density_x, initial_density_y, initial_slope_x, initial_slope = (
        derived_curve(
            initial,
            arguments.density_floor_fraction,
            arguments.slope_window,
        )
    )
    initial_lines = [
        axes[0].plot(
            initial_density_x, initial_density_y, color="k", linestyle=":",
            linewidth=2.4, label="Initial SC", zorder=2,
        )[0],
        axes[1].plot(
            initial_slope_x, initial_slope, color="k", linestyle=":",
            linewidth=2.4, zorder=2,
        )[0],
        axes[2].plot(
            initial.radius, initial.compactness, color="k", linestyle=":",
            linewidth=2.4, zorder=2,
        )[0],
    ]
    sc_lines = [
        axes[0].plot([], [], color="b", linestyle="-", linewidth=2.8,
                     label="Schwarzschild", zorder=3)[0],
        axes[1].plot([], [], color="b", linestyle="-", linewidth=2.8,
                     zorder=3)[0],
        axes[2].plot([], [], color="b", linestyle="-", linewidth=2.8,
                     zorder=3)[0],
    ]
    ef_lines = [
        axes[0].plot([], [], color="r", linestyle="--", linewidth=2.8,
                     label="Eddington--Finkelstein", zorder=4)[0],
        axes[1].plot([], [], color="r", linestyle="--", linewidth=2.8,
                     zorder=4)[0],
        axes[2].plot([], [], color="r", linestyle="--", linewidth=2.8,
                     zorder=4)[0],
    ]
    figure.legend(
        handles=[initial_lines[0], sc_lines[0], ef_lines[0]],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        fontsize=15,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.8,
    )
    title = figure.suptitle("", fontsize=21, y=0.975)

    output = arguments.output.resolve()
    writer = AtomicGif(output, arguments.fps)
    preview_indices = {
        0: output.with_name(output.stem + "_initial.png"),
        len(frames) // 2: output.with_name(output.stem + "_middle.png"),
        len(frames) - 1: output.with_name(output.stem + "_final.png"),
    }
    # Remove the SC artists in EF-only frames so a completed coordinate comparison cannot appear to continue evolving.
    for index, frame in enumerate(frames):
        if frame.sc is None:
            for line in sc_lines:
                line.set_data([], [])
        else:
            sc_density_x, sc_density_y, sc_slope_x, sc_slope = derived_curve(
                frame.sc,
                arguments.density_floor_fraction,
                arguments.slope_window,
            )
            sc_lines[0].set_data(sc_density_x, sc_density_y)
            sc_lines[1].set_data(sc_slope_x, sc_slope)
            sc_lines[2].set_data(frame.sc.radius, frame.sc.compactness)
        ef_density_x, ef_density_y, ef_slope_x, ef_slope = derived_curve(
            frame.ef,
            arguments.density_floor_fraction,
            arguments.slope_window,
        )
        ef_lines[0].set_data(ef_density_x, ef_density_y)
        ef_lines[1].set_data(ef_slope_x, ef_slope)
        ef_lines[2].set_data(frame.ef.radius, frame.ef.compactness)
        if frame.phase == "same_event_overlap":
            title.set_text(
                rf"Same events before the SC chart limit: $\tau={frame.sc_tau:.3f}$, "
                rf"$v\in[{frame.ef_v_min:.3f},{frame.ef_v_max:.3f}]$"
            )
        else:
            title.set_text(
                rf"EF-only after the SC chart limit: $v={frame.ef_v_min:.3f}$"
            )
        rgb = figure_to_rgb(figure)
        writer.append(rgb)
        if index in preview_indices:
            plt.imsave(preview_indices[index], rgb)
    writer.close()
    plt.close(figure)

    # Publish measured coordinate times, mapping coverage/errors and exact input hashes beside the completed movie.
    ledger = write_frame_ledger(output, frames)
    provenance_path = output.with_name(output.stem + "_provenance.json")
    provenance = {
        "schema_version": 1,
        "gif": str(output),
        "gif_sha256": sha256_file(output),
        "frame_ledger": str(ledger),
        "frame_ledger_sha256": sha256_file(ledger),
        "frames": len(frames),
        "same_event_frames": sum(
            frame.phase == "same_event_overlap" for frame in frames
        ),
        "ef_only_frames": sum(
            frame.phase == "ef_only_after_sc_chart_limit" for frame in frames
        ),
        "inputs": {
            "schwarzschild_profiles": str(
                (arguments.schwarzschild / "high_compactness_profiles.csv").resolve()
            ),
            "schwarzschild_profiles_sha256": sha256_file(
                arguments.schwarzschild / "high_compactness_profiles.csv"
            ),
            "eddington_finkelstein_profiles": str(
                (arguments.ef / "high_compactness_profiles.csv").resolve()
            ),
            "eddington_finkelstein_profiles_sha256": sha256_file(
                arguments.ef / "high_compactness_profiles.csv"
            ),
            "manifest": str(context.manifest_path),
            "manifest_sha256": sha256_file(context.manifest_path),
            "summaries": context.summary_audit,
            "authenticated_artifacts": context.artifacts,
            "donor_history": context.donor_history_audit,
        },
        "mapping": {
            "same_event_rule": "SC(tau,r) versus EF(v(tau,r),r)",
            "integration": "per-cell RK4 n-versus-2n; accept 2n",
            "eikonal_substeps_n": arguments.eikonal_substeps,
            "mapped_radial_samples_per_frame": arguments.mapped_radial_samples,
            "maximum_same_event_frames": arguments.maximum_same_event_frames,
            "eikonal_tolerance": arguments.eikonal_tolerance,
            "composition_tolerance": arguments.composition_tolerance,
            "minimum_a": arguments.minimum_a,
            "map_worldtube_radius": context.boundary_radius,
            "evolution_outer_radius": context.evolution_outer_radius,
            "v_offset": context.v_offset,
        },
        "panels": {
            "density": {
                "x": "log10(r)",
                "y": "log10(rho)",
                "sc_definition": "polar-areal static-observer energy density",
                "ef_definition": "T_vv/(a*b^2), only where a>minimum_a",
            },
            "density_slope": {
                "x": "log10(r)",
                "y": "d log(rho)/d log(r)",
                "window_points": arguments.slope_window,
            },
            "compactness": {"x": "r", "y": "2m/r"},
            "density_floor_fraction": arguments.density_floor_fraction,
            "density_smoothing": "none; exact mapped values are connected",
            "fixed_limits": limits,
        },
    }
    temporary = provenance_path.with_name(provenance_path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(provenance_path)
    return output


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse pair inputs, strict map checks, and non-invasive display choices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schwarzschild", type=Path, required=True)
    parser.add_argument("--ef", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--maximum-same-event-frames",
        type=int,
        default=12,
        help=(
            "cap costly exact mapped overlap frames; remaining frames follow "
            "native EF evolution after the SC chart ends"
        ),
    )
    parser.add_argument(
        "--mapped-radial-samples",
        type=int,
        default=16,
        help="exact mapped radii per overlap frame (all receive strict checks)",
    )
    parser.add_argument(
        "--ef-only-fraction",
        type=float,
        default=0.35,
        help="fraction of frames reserved for EF evolution after SC ends",
    )
    parser.add_argument("--eikonal-substeps", type=int, required=True)
    parser.add_argument("--eikonal-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--composition-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--minimum-a", type=float, default=1.0e-3)
    parser.add_argument("--density-floor-fraction", type=float, default=1.0e-7)
    parser.add_argument("--slope-window", type=int, default=7)
    parser.add_argument("--slope-min", type=float, default=-6.0)
    parser.add_argument("--slope-max", type=float, default=2.0)
    parser.add_argument("--minimum-log-radius", type=float)
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=float("inf"))
    arguments = parser.parse_args(argv)
    if arguments.frames < 2:
        parser.error("--frames must be at least two")
    if arguments.fps <= 0.0:
        parser.error("--fps must be positive")
    if arguments.maximum_same_event_frames < 2:
        parser.error("--maximum-same-event-frames must be at least two")
    if arguments.mapped_radial_samples < 5:
        parser.error("--mapped-radial-samples must be at least five")
    if not 0.0 <= arguments.ef_only_fraction < 1.0:
        parser.error("--ef-only-fraction must lie in [0,1)")
    if arguments.eikonal_substeps < 1:
        parser.error("--eikonal-substeps must be positive")
    for name in (
        "eikonal_tolerance", "composition_tolerance", "minimum_a",
        "density_floor_fraction",
    ):
        if getattr(arguments, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("slope_window",):
        if getattr(arguments, name) < 3:
            parser.error(f"--{name.replace('_', '-')} must be at least three")
    if arguments.slope_min >= arguments.slope_max:
        parser.error("--slope-min must be below --slope-max")
    if arguments.r_min < 0.0 or arguments.r_min >= arguments.r_max:
        parser.error("require 0 <= --r-min < --r-max")
    return arguments


if __name__ == "__main__":
    print(render(parse_arguments()))
