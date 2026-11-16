#!/usr/bin/env python3
# Authenticate a solved Schwarzschild-to-EF initial-null-slice mapping before dependent runs use it.
# Inputs are a map manifest, its named artifact files and an optional evolution summary.
# Output JSON and checksum report complete crossing coverage, face-aware donor history, vacuum matching and ADM/ODE tolerances.

"""Fail-closed setup gate for a solved SC-to-EF null-map artifact.

This check is intentionally cheaper than the full same-event parity
certificate.  It authenticates the complete extraction artifact set and
rejects a donor before dependent production jobs fan out when coverage,
vacuum normalization, fixed-ADM provenance, or the RK4 step-doubling
eikonal estimator misses the certifier's tolerances.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import struct
import sys


INCONCLUSIVE = 2
REQUIRED_ARTIFACTS = {
    "null_initial_data", "crossings", "slice", "accepted_steps",
    "donor_history",
}
EIKONAL_METHOD = "per_cell_RK4_step_doubling_n_vs_2n_accept_2n"
EIKONAL_NORMALIZATION = (
    "abs(Y_2n-Y_n)/(15*max(1,abs(Y_in),abs(Y_2n)));"
    "Y={s_sigma,ln_A}"
)
DONOR_HISTORY_MAGIC = b"VENUSSCHIST1\0\0\0\0"
DONOR_HISTORY_FORMAT_VERSION = 2
SAVED_METRIC_INTERPOLATION = (
    "SC center-face lambda plus monotone-Hermite mu; EF center-face a,b"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_sha256(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        return None
    return value


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def close(left, right, relative=2.0e-12):
    return abs(left - right) <= relative * max(1.0, abs(left), abs(right))


def sidecar(path, digest):
    sidecar_path = pathlib.Path(str(path) + ".sha256")
    sidecar_path.write_text("{}  {}\n".format(digest, pathlib.Path(path).name))
    return str(sidecar_path.resolve())


def audit_donor_history(path, expected_snapshots, expected_cells):
    """Authenticate the face-aware dense-history binary layout.

    Version 1 contains cell-centre fields for diagnostic reconstruction.
    Production SC metric sampling requires the four complete face arrays and
    the cell-centre mu derivative stored in every version-2 snapshot.  The exact file length proves that payload is present for
    every advertised snapshot; the full comparator later validates its
    numerical geometry and uses it to reconstruct the metric.
    """
    if isinstance(expected_snapshots, bool) \
            or not isinstance(expected_snapshots, int) \
            or expected_snapshots < 2:
        raise ValueError("manifest has no valid donor snapshot count")
    if isinstance(expected_cells, bool) \
            or not isinstance(expected_cells, int) or expected_cells < 1:
        raise ValueError("manifest has no valid donor radial-cell count")
    payload = pathlib.Path(path).read_bytes()
    if len(payload) < 36 or payload[:16] != DONOR_HISTORY_MAGIC:
        raise ValueError("donor history has invalid VENUSSCHIST1 magic")
    version, snapshots, cells = struct.unpack_from("<IQQ", payload, 16)
    if version != DONOR_HISTORY_FORMAT_VERSION:
        raise ValueError(
            "donor history format version {} is legacy; version {} with "
            "face reconstruction is required".format(
                version, DONOR_HISTORY_FORMAT_VERSION
            )
        )
    if snapshots != expected_snapshots or cells != expected_cells:
        raise ValueError(
            "donor history header shape ({},{}) disagrees with manifest "
            "({},{})".format(
                snapshots, cells, expected_snapshots, expected_cells
            )
        )
    # Per v2 snapshot: five scalar diagnostics, seven centre field arrays,
    # four (cells+1) face arrays, and one centre mu_r array.
    # Count every declared center/face array and scalar to verify that the donor-history binary contains its complete versioned payload.
    doubles_per_snapshot = 5 + 7 * cells + 4 * (cells + 1) + cells
    expected_bytes = 36 + 8 * snapshots * doubles_per_snapshot
    if len(payload) != expected_bytes:
        raise ValueError(
            "donor history v2 payload has {} bytes; {} required for every "
            "face-aware snapshot".format(len(payload), expected_bytes)
        )
    return {
        "path": str(pathlib.Path(path).resolve()),
        "format_version": version,
        "snapshots": snapshots,
        "radial_cells": cells,
        "face_reconstruction": True,
        "bytes": len(payload),
    }


def validate(args):
    manifest_path = pathlib.Path(args.manifest).resolve()
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("manifest root is not an object")
    issues = []
    actual_manifest_sha = sha256_file(manifest_path)

    # These exact schema and convention fields bind the map direction, momentum reversal, integration method and field interpolation.
    exact = {
        "schema": "venus_v3_sc_to_ef_null_map",
        "schema_version": 2,
        "complete": True,
        "all_markers_crossed_once": True,
        "outer_worldtube_verified_vacuum": True,
        "mapping_scope": "initial_null_hypersurface",
        "donor_direction": "East_time_reversed_outgoing_s=-tau",
        "time_reversal_momentum": "w_reverse=-w_schwarzschild",
        "eikonal_integration_method": EIKONAL_METHOD,
        "eikonal_richardson_error_normalization": EIKONAL_NORMALIZATION,
        "donor_history_format_version": DONOR_HISTORY_FORMAT_VERSION,
        "saved_metric_interpolation": SAVED_METRIC_INTERPOLATION,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            issues.append("manifest {} does not equal {!r}".format(key, expected))

    # One unique crossing per marker and one donor snapshot per accepted step are separate completeness conditions.
    marker_count = manifest.get("marker_count")
    crossing_count = manifest.get("crossing_count")
    unique_count = manifest.get("unique_crossing_count")
    if isinstance(marker_count, bool) or not isinstance(marker_count, int) \
            or marker_count <= 0 or crossing_count != marker_count \
            or unique_count != marker_count:
        issues.append("marker/crossing counts do not prove one crossing per marker")
    accepted = manifest.get("accepted_step_count")
    snapshots = manifest.get("history_snapshot_count")
    if isinstance(accepted, bool) or not isinstance(accepted, int) \
            or accepted <= 0 or snapshots != accepted + 1:
        issues.append("dense donor history is inconsistent with accepted-step ledger")

    # Step doubling estimates reverse-time and log-Jacobian errors separately; the reported combined residual must be their maximum.
    reverse_error = number(
        manifest.get("maximum_eikonal_reverse_time_richardson_error")
    )
    log_a_error = number(
        manifest.get("maximum_eikonal_log_A_richardson_error")
    )
    legacy_error = number(manifest.get("maximum_eikonal_ode_residual"))
    for key, value in (
            ("maximum_eikonal_reverse_time_richardson_error", reverse_error),
            ("maximum_eikonal_log_A_richardson_error", log_a_error),
            ("maximum_eikonal_ode_residual", legacy_error)):
        if value is None or value < 0.0 \
                or value > args.eikonal_convergence_tolerance:
            issues.append("{} exceeds eikonal tolerance {}".format(
                key, args.eikonal_convergence_tolerance
            ))
    if reverse_error is not None and log_a_error is not None \
            and legacy_error is not None \
            and legacy_error != max(reverse_error, log_a_error):
        issues.append("legacy eikonal residual is not the maximum component estimate")

    map_radius = number(manifest.get("map_worldtube_radius"))
    evolution_radius = number(manifest.get("evolution_outer_radius"))
    legacy_radius = number(manifest.get("outer_boundary_radius"))
    analytic_extension = manifest.get("analytic_vacuum_extension")
    map_cells = manifest.get("map_radial_cell_count")
    evolution_cells = manifest.get("evolution_radial_cell_count")
    extension_start = manifest.get("analytic_vacuum_extension_start_index")
    if map_radius is None or map_radius <= 0.0 \
            or evolution_radius is None or evolution_radius < map_radius:
        issues.append("manifest has invalid R_B<=R_E map/evolution radii")
    elif legacy_radius is None or not close(legacy_radius, map_radius):
        issues.append("legacy outer_boundary_radius is not the map worldtube R_B")
    if isinstance(map_cells, bool) or not isinstance(map_cells, int) \
            or map_cells <= 0 \
            or isinstance(evolution_cells, bool) \
            or not isinstance(evolution_cells, int) \
            or evolution_cells < map_cells \
            or manifest.get("radial_cell_count") != evolution_cells:
        issues.append("manifest has invalid map/evolution radial cell counts")
    if isinstance(extension_start, bool) or not isinstance(extension_start, int) \
            or extension_start != map_cells:
        issues.append("analytic extension start is not the map prefix size")
    # The numerical map worldtube and the larger evolution domain are joined by a declared analytic vacuum extension.
    split = map_radius is not None and evolution_radius is not None \
        and evolution_radius > map_radius
    if not isinstance(analytic_extension, bool) \
            or analytic_extension != split \
            or (split and map_cells == evolution_cells):
        issues.append("analytic-vacuum extension flag is inconsistent with R_B<R_E")
    for key in ("grid_fingerprint", "map_grid_fingerprint"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 18 \
                or not value.startswith("0x") \
                or any(character not in "0123456789abcdef" for character in value[2:].lower()):
            issues.append("manifest {} is not a 64-bit hexadecimal fingerprint".format(key))

    required_clearance = number(manifest.get("required_worldtube_clearance"))
    matter_clearance = number(
        manifest.get("minimum_worldtube_matter_clearance")
    )
    crossing_clearance = number(
        manifest.get("minimum_crossing_radius_clearance")
    )
    max_donor_radius = number(manifest.get("maximum_donor_marker_radius"))
    max_crossing_radius = number(manifest.get("maximum_crossing_radius"))
    if required_clearance is None or required_clearance <= 0.0:
        issues.append("manifest lacks a positive required worldtube clearance")
    for key, clearance, maximum_radius in (
            ("matter", matter_clearance, max_donor_radius),
            ("crossing", crossing_clearance, max_crossing_radius)):
        if clearance is None or clearance <= 0.0 \
                or required_clearance is None \
                or clearance < required_clearance:
            issues.append("{} support lacks required worldtube clearance".format(key))
        if map_radius is None or maximum_radius is None \
                or not close(clearance, map_radius - maximum_radius):
            issues.append("{} clearance is inconsistent with maximum radius".format(key))

    analytic_residuals = {}
    for key in (
            "maximum_analytic_vacuum_field_residual",
            "maximum_analytic_vacuum_source",
            "maximum_analytic_inverse_map_residual"):
        value = number(manifest.get(key))
        analytic_residuals[key] = value
        if value is None or value < 0.0 \
                or value > args.analytic_vacuum_tolerance:
            issues.append("{} exceeds analytic-vacuum tolerance {}".format(
                key, args.analytic_vacuum_tolerance
            ))
    minimum_analytic_a = number(manifest.get("minimum_analytic_vacuum_a"))
    if minimum_analytic_a is None or minimum_analytic_a <= 0.0:
        issues.append("analytic vacuum extension is not strictly outside r=2M")

    residual_limits = {
        "maximum_marker_crossing_residual": args.crossing_residual_tolerance,
        "maximum_replay_time_error": args.composition_tolerance,
        "maximum_outer_b_normalization_error": args.composition_tolerance,
        "maximum_outer_source": args.outer_vacuum_tolerance,
    }
    residuals = {}
    for key, tolerance in residual_limits.items():
        value = number(manifest.get(key))
        residuals[key] = value
        if value is None or value < 0.0 or value > tolerance:
            issues.append("{} exceeds tolerance {}".format(key, tolerance))
    reported_b_tolerance = number(
        manifest.get("outer_b_normalization_tolerance")
    )
    b_error = residuals.get("maximum_outer_b_normalization_error")
    if reported_b_tolerance is None or reported_b_tolerance <= 0.0 \
            or reported_b_tolerance > args.composition_tolerance \
            or b_error is None or b_error > reported_b_tolerance:
        issues.append(
            "outer b normalization does not meet its authenticated mesh tolerance"
        )

    # Distinguish the family ADM target, donor initial mass and numerical null-slice ADM anchor when evaluating drift.
    target = number(manifest.get("target_initial_adm_mass"))
    donor = number(manifest.get("donor_initial_adm_mass"))
    null_slice = number(manifest.get("null_slice_adm_mass"))
    reported_drift = number(manifest.get("donor_adm_mass_relative_drift"))
    computed_drift = None
    if target is None or target <= 0.0 or donor is None or donor <= 0.0:
        issues.append("manifest lacks positive fixed-family/donor-t0 ADM masses")
    elif abs(donor - target) / target > args.fixed_adm_relative_tolerance:
        issues.append("donor-t0 ADM mass misses the fixed-family target")
    if donor is None or donor <= 0.0 or null_slice is None or null_slice <= 0.0:
        issues.append("manifest lacks a positive numerical null-slice ADM anchor")
    else:
        computed_drift = abs(null_slice - donor) / donor
        if computed_drift > args.donor_adm_drift_tolerance:
            issues.append("donor-to-null-slice ADM drift exceeds tolerance")
        if reported_drift is None or not close(reported_drift, computed_drift):
            issues.append("reported donor ADM drift is inconsistent")

    raw_artifacts = manifest.get("artifacts")
    artifact_audit = {}
    if not isinstance(raw_artifacts, dict):
        issues.append("manifest artifacts is not an object")
        raw_artifacts = {}
    missing = sorted(REQUIRED_ARTIFACTS.difference(raw_artifacts))
    if missing:
        issues.append("manifest lacks artifacts: {}".format(",".join(missing)))
    # Resolve each manifest path and hash the actual artifact before trusting its numerical metadata.
    for name, record in sorted(raw_artifacts.items()):
        if not isinstance(record, dict):
            issues.append("artifact {} has no path/SHA record".format(name))
            continue
        raw_path = record.get("path")
        expected = normalized_sha256(record.get("sha256"))
        if not isinstance(raw_path, str) or not raw_path or expected is None:
            issues.append("artifact {} has invalid path/SHA".format(name))
            continue
        path = pathlib.Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if not path.is_file():
            issues.append("artifact {} is missing: {}".format(name, path))
            continue
        actual = sha256_file(path)
        verified = actual == expected
        artifact_audit[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": verified,
        }
        if not verified:
            issues.append("artifact {} fails SHA-256 authentication".format(name))

    donor_history_audit = None
    donor_record = artifact_audit.get("donor_history")
    if isinstance(donor_record, dict) and donor_record.get("verified") is True:
        try:
            donor_history_audit = audit_donor_history(
                donor_record["path"], snapshots, evolution_cells
            )
        except (OSError, ValueError, struct.error) as error:
            issues.append("donor history schema fails: {}".format(error))
    else:
        issues.append("authenticated donor history is unavailable for schema audit")

    summary_audit = None
    # Join an optional evolution summary to this exact manifest and compare its deposited initial ADM mass to the null-slice anchor.
    if args.summary:
        summary_path = pathlib.Path(args.summary).resolve()
        with summary_path.open() as stream:
            summary = json.load(stream)
        high = summary.get("high_compactness") if isinstance(summary, dict) else None
        if not isinstance(high, dict):
            raise ValueError("summary lacks high_compactness object")
        summary_audit = {
            "path": str(summary_path),
            "coordinates": summary.get("coordinates"),
            "manifest_sha256": high.get("coordinate_map_manifest_sha256"),
            "fixed_family_adm_mass": high.get("fixed_family_adm_mass"),
            "null_slice_conserved_adm_mass": high.get(
                "null_slice_conserved_adm_mass"
            ),
            "initial_deposited_adm_mass": high.get(
                "initial_deposited_adm_mass"
            ),
        }
        if high.get("eikonal_mapped_initial_data") is not True:
            issues.append("summary does not assert solved-eikonal mapped data")
        if normalized_sha256(high.get("coordinate_map_manifest_sha256")) \
                != actual_manifest_sha:
            issues.append("summary does not authenticate this manifest")
        if marker_count != high.get("markers") \
                or manifest.get("radial_cell_count") != high.get("radial_cells"):
            issues.append("summary grid/marker counts disagree with manifest")
        for summary_key, manifest_key in (
                ("eikonal_boundary_radius", "map_worldtube_radius"),
                ("eikonal_evolution_outer_radius", "evolution_outer_radius"),
                ("eikonal_v_offset", "ef_v_offset"),
                ("fixed_family_adm_mass", "donor_initial_adm_mass"),
                ("null_slice_conserved_adm_mass", "null_slice_adm_mass")):
            left = number(high.get(summary_key))
            right = number(manifest.get(manifest_key))
            if left is None or right is None or not close(left, right):
                issues.append("summary {} disagrees with manifest {}".format(
                    summary_key, manifest_key
                ))
        if high.get("eikonal_analytic_vacuum_extension") \
                is not analytic_extension:
            issues.append("summary analytic-vacuum extension flag disagrees with manifest")
        deposited = number(high.get("initial_deposited_adm_mass"))
        if deposited is None or deposited <= 0.0:
            issues.append("summary lacks a positive deposited initial ADM mass")
        elif null_slice is not None and null_slice > 0.0:
            deposition_error = abs(deposited - null_slice) / null_slice
            summary_audit[
                "initial_deposition_adm_relative_error_from_null_slice"
            ] = deposition_error
            if deposition_error > args.initial_deposition_adm_tolerance:
                issues.append(
                    "initial EF deposition ADM mismatch exceeds tolerance"
                )

    return {
        "schema": "venus_wpic_null_map_preflight_v2",
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "manifest": {
            "path": str(manifest_path),
            "actual_sha256": actual_manifest_sha,
            "marker_count": marker_count,
            "crossing_count": crossing_count,
            "unique_crossing_count": unique_count,
            "eikonal_errors": {
                "reverse_time": reverse_error,
                "log_A": log_a_error,
                "legacy_maximum": legacy_error,
            },
            "residuals": residuals,
            "analytic_vacuum_residuals": analytic_residuals,
            "map_worldtube_radius": map_radius,
            "evolution_outer_radius": evolution_radius,
            "map_radial_cell_count": map_cells,
            "evolution_radial_cell_count": evolution_cells,
            "analytic_vacuum_extension": analytic_extension,
            "minimum_worldtube_matter_clearance": matter_clearance,
            "minimum_crossing_radius_clearance": crossing_clearance,
            "fixed_family_adm_mass": donor,
            "null_slice_conserved_adm_mass": null_slice,
            "donor_to_null_slice_relative_drift": computed_drift,
            "donor_history": donor_history_audit,
            "artifacts": artifact_audit,
        },
        "summary": summary_audit,
        "tolerances": {
            "eikonal_convergence": args.eikonal_convergence_tolerance,
            "crossing_residual": args.crossing_residual_tolerance,
            "composition": args.composition_tolerance,
            "outer_vacuum": args.outer_vacuum_tolerance,
            "fixed_adm_relative": args.fixed_adm_relative_tolerance,
            "donor_adm_drift": args.donor_adm_drift_tolerance,
            "initial_deposition_adm": args.initial_deposition_adm_tolerance,
            "analytic_vacuum": args.analytic_vacuum_tolerance,
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--eikonal-convergence-tolerance", type=float, default=2.0e-5
    )
    parser.add_argument(
        "--crossing-residual-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument("--composition-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--outer-vacuum-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--analytic-vacuum-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fixed-adm-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--donor-adm-drift-tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--initial-deposition-adm-tolerance", type=float, default=1.0e-3
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if name.endswith("tolerance") and (not math.isfinite(value) or value <= 0.0):
            parser.error("--{} must be positive and finite".format(
                name.replace("_", "-")
            ))
    try:
        result = validate(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("null-map preflight input error: {}".format(error), file=sys.stderr)
        return 1
    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.write_text(rendered)
    # Hash the serialized result itself so the acceptance record has an unambiguous byte identity.
    sidecar(output, hashlib.sha256(rendered.encode()).hexdigest())
    if result["status"] != "pass":
        for issue in result["issues"]:
            print("null-map preflight: {}".format(issue), file=sys.stderr)
        return INCONCLUSIVE
    print("authenticated null-map preflight: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
