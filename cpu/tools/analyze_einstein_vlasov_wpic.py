#!/usr/bin/env python3
# Validate relativistic WPIC outputs, compare compactness milestones and classify a sampled angular-momentum family.
# Inputs are saved summaries, ordered histories, exact source/map records and optional independent excision repeats in G=c=1 units.
# The output keeps numerical quality, physical outcome, coordinate evidence and incomplete cases distinct before reporting a threshold bracket.

"""Validate VENUS WPIC runs, compare a matched marker family, and bracket L.

The dynamic Schwarzschild/EF comparison aligns two strictly provenance-matched
marker families by the first upward crossing of gauge-invariant areal
compactness 2m/r, rather than by coordinate time.  It does not claim that the
coordinate-specific initial slices are identical physical hypersurfaces.
The threshold classifier calls a result a halo only after a full-duration,
explicit halo outcome; ambiguous or incomplete points remain undetermined.
An EF horizon is first only a ``black_hole_candidate`` after a finite
post-horizon interval, positive b and P, fixed-ADM accounting, and causal
interior excision all pass.  It becomes ``black_hole_certified`` only after
an authenticated pair of independent excision-trigger and buffer variations;
only that second tier may close a reported threshold bracket.
Threshold classification is gated by an authenticated strict same-event
dynamical-coordinate certificate with independent refinement/PIC convergence
evidence; the exploratory milestone comparison is not a substitute for that
certificate.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys


SCIENTIFIC_INCONCLUSIVE = 2
RECOGNIZED_OUTCOMES = {
    "initialized",
    "black_hole",
    "black_hole_threshold_proxy",
    "virialized_halo",
    "undetermined",
}
PRODUCTION_METRIC_INTERPOLATION = {
    "schwarzschild": (
        "production_center_face_lambda_monotone_hermite_mu"
    ),
    "eddington_finkelstein": "production_center_face_linear_a_b",
    "donor_history_format_version": 2,
    "production_equivalent": True,
}
SAVED_METRIC_INTERPOLATION = (
    "SC center-face lambda plus monotone-Hermite mu; EF center-face a,b"
)


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    position = fraction * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return (1.0 - weight) * ordered[left] + weight * ordered[right]


def linear_slope(times, values):
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("linear trend requires at least two paired samples")
    center_t = sum(times) / len(times)
    center_y = sum(values) / len(values)
    denominator = sum((time - center_t) ** 2 for time in times)
    if denominator <= 0.0:
        raise ValueError("linear trend has zero time baseline")
    return sum(
        (time - center_t) * (value - center_y)
        for time, value in zip(times, values)
    ) / denominator


def eprint(message):
    print(message, file=sys.stderr)


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(mapping, key):
    value = mapping.get(key)
    if not finite_number(value):
        raise ValueError("missing or non-finite numeric field: {}".format(key))
    return float(value)


def load_summary(run_dir):
    path = os.path.join(os.path.abspath(run_dir), "summary.json")
    with open(path, "r") as handle:
        summary = json.load(handle)
    if summary.get("module") != "weighted_einstein_vlasov_pic":
        raise ValueError("{} is not an Einstein--Vlasov WPIC summary".format(path))
    high = summary.get("high_compactness")
    if not isinstance(high, dict):
        raise ValueError("{} has no high_compactness object".format(path))
    return path, summary, high


def load_history(run_dir):
    path = os.path.join(os.path.abspath(run_dir), "high_compactness_history.csv")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        # Translate the Schwarzschild and EF history column names into one analysis schema while retaining coordinate-specific mass/excision channels.
        if {"time", "rest_mass"}.issubset(fieldnames):
            optional_source_names = {}
            source_names = {
                "step": "step",
                "time": "time",
                "rest_mass": "rest_mass",
                "adm_mass": "adm_mass",
                "max_compactness": "max_compactness",
                "r10": "r10",
                "r50": "r50",
                "r90": "r90",
                "virial_ratio": "virial_ratio",
            }
        elif {"advanced_time", "total_rest_weight"}.issubset(fieldnames):
            optional_source_names = {
                "active_rest_mass": "active_rest_weight",
                "excised_rest_mass": "excised_rest_weight",
                "escaped_rest_mass": "escaped_rest_weight",
                "min_a": "min_a",
                "min_b": "min_b",
                "apparent_horizon_radius": "apparent_horizon_radius",
                "excision_radius": "excision_radius",
            }
            source_names = {
                "step": "step",
                "time": "advanced_time",
                "rest_mass": "total_rest_weight",
                "adm_mass": "adm_mass",
                "max_compactness": "max_compactness",
                "r10": "r10",
                "r50": "r50",
                "r90": "r90",
                "virial_ratio": "virial_ratio",
            }
        else:
            raise ValueError(
                "{} is neither the Schwarzschild nor EF high-history schema".format(path)
            )
        missing = set(source_names.values()).difference(fieldnames)
        if missing:
            raise ValueError("{} lacks columns: {}".format(path, ", ".join(sorted(missing))))
        jeans_column = "anisotropic_jeans_relative_l1"
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {}
            try:
                for key, source_name in source_names.items():
                    row[key] = float(raw[source_name])
                for key, source_name in optional_source_names.items():
                    row[key] = (
                        float(raw[source_name])
                        if source_name in fieldnames else -1.0
                    )
                # An absent Jeans diagnostic is encoded as unavailable.
                # Black-hole and coordinate analyses can read these records;
                # halo classification requires the relativistic stress
                # balance to be measured explicitly.
                row[jeans_column] = (
                    float(raw[jeans_column])
                    if jeans_column in fieldnames else -1.0
                )
            except (TypeError, ValueError):
                raise ValueError("{}:{} contains a non-numeric diagnostic".format(path, line_number))
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError("{}:{} contains a non-finite diagnostic".format(path, line_number))
            rows.append(row)
    if not rows:
        raise ValueError("{} has no data rows".format(path))
    # The analysis requires strictly increasing saved physical coordinate times for trends and crossing searches.
    for first, second in zip(rows, rows[1:]):
        if second["time"] <= first["time"]:
            raise ValueError("{} has non-increasing output times".format(path))
    return path, rows


def last_history_time(run_dir):
    try:
        _path, rows = load_history(run_dir)
    except (IOError, OSError, ValueError):
        return None
    return rows[-1]["time"]


def parse_sha256_manifest(path):
    try:
        with open(path, "r") as handle:
            first = handle.readline().strip().split()
    except (IOError, OSError):
        return None
    if not first or len(first[0]) != 64:
        return None
    digest = first[0].lower()
    if any(character not in "0123456789abcdef" for character in digest):
        return None
    return digest


def file_sha256(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except (IOError, OSError):
        return None
    return digest.hexdigest()


def normalized_sha256(value):
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest):
        return None
    return digest


# Recheck the full saved null-map validation result, artifact identities and tolerances before using an EF run in a physical classification.
def authenticate_null_map_preflight(run_dir, high):
    """Authenticate the launcher's solved-null-map postflight record.

    The threshold classifier must not infer map validity from a summary flag.
    The launcher hashes a strict validator result after it has authenticated
    all five declared map artifacts.  Here we independently authenticate that
    result, bind it to this run's manifest and three ADM channels, and reject
    any relaxed eikonal/residual tolerance.
    """
    path = os.path.join(os.path.abspath(run_dir), "null_map_preflight.json")
    digest = file_sha256(path)
    if digest is None:
        raise ValueError("mapped EF run has no null_map_preflight.json")
    sidecar_path = path + ".sha256"
    try:
        with open(sidecar_path, "r") as handle:
            fields = handle.readline().strip().split()
    except (IOError, OSError):
        raise ValueError("mapped EF run has no null-map preflight SHA-256 sidecar")
    if len(fields) != 2 \
            or normalized_sha256(fields[0]) != digest \
            or fields[1].lstrip("*") != os.path.basename(path):
        raise ValueError("null-map preflight SHA-256 sidecar does not authenticate it")
    with open(path, "r") as handle:
        audit = json.load(handle)
    preflight_schema = audit.get("schema")
    if preflight_schema != "venus_wpic_null_map_preflight_v2" \
            or audit.get("status") != "pass" or audit.get("issues") != []:
        raise ValueError("null-map preflight is not an issue-free v2 pass")

    manifest = audit.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("null-map preflight has no manifest audit")
    manifest_digest = normalized_sha256(manifest.get("actual_sha256"))
    if manifest_digest is None or manifest_digest != normalized_sha256(
            high.get("coordinate_map_manifest_sha256")):
        raise ValueError("null-map preflight is not bound to the run manifest")
    artifacts = manifest.get("artifacts")
    required = {
        "null_initial_data", "crossings", "slice", "accepted_steps",
        "donor_history",
    }
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("null-map preflight lacks the complete artifact audit")
    for name in sorted(required):
        record = artifacts.get(name)
        if not isinstance(record, dict) or record.get("verified") is not True \
                or normalized_sha256(record.get("expected_sha256")) is None \
                or normalized_sha256(record.get("expected_sha256")) \
                    != normalized_sha256(record.get("actual_sha256")):
            raise ValueError(
                "null-map preflight has an unauthenticated artifact: {}".format(name)
            )
    donor_record = artifacts["donor_history"]
    donor_audit = manifest.get("donor_history")
    if not isinstance(donor_audit, dict) \
            or donor_audit.get("format_version") != 2 \
            or donor_audit.get("face_reconstruction") is not True \
            or not isinstance(donor_audit.get("path"), str) \
            or not isinstance(donor_record.get("path"), str) \
            or os.path.realpath(donor_audit["path"]) \
                != os.path.realpath(donor_record["path"]):
        raise ValueError(
            "null-map preflight lacks an authenticated face-aware v2 donor"
        )

    tolerances = audit.get("tolerances")
    # These are upper ceilings on the validation tolerances themselves, so a record made with looser criteria cannot qualify.
    maxima = {
        "eikonal_convergence": 2.0e-5,
        "crossing_residual": 1.0e-8,
        "composition": 5.0e-4,
        "outer_vacuum": 1.0e-12,
        "fixed_adm_relative": 1.0e-8,
        "donor_adm_drift": 1.0e-3,
        "initial_deposition_adm": 1.0e-3,
    }
    maxima["analytic_vacuum"] = 1.0e-12
    if not isinstance(tolerances, dict):
        raise ValueError("null-map preflight has no tolerance audit")
    for name, maximum in maxima.items():
        value = tolerances.get(name)
        if not finite_number(value) or not (0.0 < float(value) <= maximum):
            raise ValueError(
                "null-map preflight used a missing or relaxed {} tolerance".format(name)
            )
    errors = manifest.get("eikonal_errors")
    if not isinstance(errors, dict):
        raise ValueError("null-map preflight has no component eikonal errors")
    reverse = errors.get("reverse_time")
    log_a = errors.get("log_A")
    legacy = errors.get("legacy_maximum")
    if any(not finite_number(value) or float(value) < 0.0
           for value in (reverse, log_a, legacy)) \
            or any(float(value) > float(tolerances["eikonal_convergence"])
                   for value in (reverse, log_a, legacy)) \
            or float(legacy) != max(float(reverse), float(log_a)):
        raise ValueError("null-map preflight eikonal component audit is inconsistent")
    residuals = manifest.get("residuals")
    residual_limits = {
        "maximum_marker_crossing_residual": tolerances["crossing_residual"],
        "maximum_replay_time_error": tolerances["composition"],
        "maximum_outer_b_normalization_error": tolerances["composition"],
        "maximum_outer_source": tolerances["outer_vacuum"],
    }
    if not isinstance(residuals, dict):
        raise ValueError("null-map preflight has no residual audit")
    for name, limit in residual_limits.items():
        value = residuals.get(name)
        if not finite_number(value) or not (0.0 <= float(value) <= float(limit)):
            raise ValueError("null-map preflight residual fails: {}".format(name))

    analytic_residuals = manifest.get("analytic_vacuum_residuals")
    required_analytic = {
        "maximum_analytic_vacuum_field_residual",
        "maximum_analytic_vacuum_source",
        "maximum_analytic_inverse_map_residual",
    }
    if not isinstance(analytic_residuals, dict) \
            or not required_analytic.issubset(analytic_residuals):
        raise ValueError(
            "v2 null-map preflight lacks analytic-vacuum residuals"
        )
    for name in sorted(required_analytic):
        value = analytic_residuals.get(name)
        if not finite_number(value) or not (
                0.0 <= float(value)
                <= float(tolerances["analytic_vacuum"])):
            raise ValueError(
                "null-map analytic-vacuum residual fails: {}".format(name)
            )
    map_radius = manifest.get("map_worldtube_radius")
    evolution_radius = manifest.get("evolution_outer_radius")
    map_cells = manifest.get("map_radial_cell_count")
    evolution_cells = manifest.get("evolution_radial_cell_count")
    split = finite_number(map_radius) and finite_number(evolution_radius) \
        and float(evolution_radius) > float(map_radius)
    if not finite_number(map_radius) or float(map_radius) <= 0.0 \
            or not finite_number(evolution_radius) \
            or float(evolution_radius) < float(map_radius) \
            or isinstance(map_cells, bool) or not isinstance(map_cells, int) \
            or map_cells <= 0 or isinstance(evolution_cells, bool) \
            or not isinstance(evolution_cells, int) \
            or evolution_cells < map_cells \
            or manifest.get("analytic_vacuum_extension") is not split:
        raise ValueError(
            "v2 null-map preflight has inconsistent map/evolution domains"
        )

    # Join the validated map to the run's family ADM, null-slice ADM and deposited initial ADM channels separately.
    summary = audit.get("summary")
    if not isinstance(summary, dict) \
            or normalized_sha256(summary.get("manifest_sha256")) != manifest_digest:
        raise ValueError("null-map preflight has no run-summary binding")
    for summary_key, high_key in (
            ("fixed_family_adm_mass", "fixed_family_adm_mass"),
            ("null_slice_conserved_adm_mass", "null_slice_conserved_adm_mass"),
            ("initial_deposited_adm_mass", "initial_deposited_adm_mass")):
        audited = summary.get(summary_key)
        current = high.get(high_key)
        if not finite_number(audited) or not finite_number(current) \
                or abs(float(audited) - float(current)) \
                    > 2.0e-12 * max(1.0, abs(float(audited)), abs(float(current))):
            raise ValueError(
                "null-map preflight {} does not match summary".format(summary_key)
            )
    audited_fixed = manifest.get("fixed_family_adm_mass")
    audited_null = manifest.get("null_slice_conserved_adm_mass")
    audited_drift = manifest.get("donor_to_null_slice_relative_drift")
    if any(not finite_number(value)
           for value in (audited_fixed, audited_null, audited_drift)) \
            or float(audited_fixed) <= 0.0 or float(audited_null) <= 0.0 \
            or abs(float(audited_fixed) - float(high["fixed_family_adm_mass"])) \
                > 2.0e-12 * max(1.0, abs(float(audited_fixed))) \
            or abs(float(audited_null)
                   - float(high["null_slice_conserved_adm_mass"])) \
                > 2.0e-12 * max(1.0, abs(float(audited_null))) \
            or abs(float(audited_null) - float(audited_fixed)) \
                / float(audited_fixed) > float(tolerances["donor_adm_drift"]) \
            or abs(float(audited_drift)
                   - abs(float(audited_null) - float(audited_fixed))
                       / float(audited_fixed)) > 2.0e-12:
        raise ValueError("null-map preflight ADM donor/null audit is inconsistent")
    deposition_error = abs(
        float(high["initial_deposited_adm_mass"]) - float(audited_null)
    ) / float(audited_null)
    if deposition_error > float(tolerances["initial_deposition_adm"]) \
            or not finite_number(summary.get(
                "initial_deposition_adm_relative_error_from_null_slice"
            )) \
            or abs(float(summary[
                "initial_deposition_adm_relative_error_from_null_slice"
            ]) - deposition_error) > 2.0e-12:
        raise ValueError("null-map preflight EF deposition/null audit is inconsistent")
    return {
        "path": path,
        "sha256": digest,
        "sidecar": sidecar_path,
        "coordinate_map_manifest_sha256": manifest_digest,
        "status": "pass",
    }


# Authenticate the same-event coordinate certificate and its independent convergence evidence before any threshold conclusion.
def parity_certificate_gate(args):
    path = os.path.abspath(args.parity_certificate)
    with open(path, "r") as handle:
        certificate = json.load(handle)
    actual_digest = file_sha256(path)
    if actual_digest is None:
        raise ValueError("could not hash parity certificate: {}".format(path))

    explicit_digest = args.parity_certificate_sha256
    expected_digest = normalized_sha256(explicit_digest)
    digest_source = "--parity-certificate-sha256"
    if explicit_digest not in (None, "") and expected_digest is None:
        raise ValueError("invalid explicit parity-certificate SHA-256")
    if explicit_digest in (None, ""):
        sidecar = path + ".sha256"
        try:
            with open(sidecar, "r") as handle:
                fields = handle.readline().strip().split()
        except (IOError, OSError):
            raise ValueError(
                "parity certificate needs its .sha256 sidecar or an explicit digest"
            )
        if len(fields) != 2:
            raise ValueError("invalid parity certificate SHA-256 sidecar")
        expected_digest = normalized_sha256(fields[0])
        named_file = fields[1].lstrip("*")
        if expected_digest is None or named_file != os.path.basename(path):
            raise ValueError(
                "parity certificate sidecar digest/name does not identify the certificate"
            )
        digest_source = sidecar
    if actual_digest != expected_digest:
        raise ValueError("parity certificate SHA-256 mismatch")

    if certificate.get("schema") \
            != "venus_wpic_dynamic_coordinate_certificate_v2":
        raise ValueError("unsupported parity certificate schema")
    if certificate.get("status") != "pass":
        raise ValueError("parity certificate status is not pass")
    if certificate.get("issues") != []:
        raise ValueError("parity certificate contains unresolved issues")
    if certificate.get("failed_fields") != []:
        raise ValueError("parity certificate contains failed fields")
    if not isinstance(certificate.get("inverse_equation"), str) \
            or not certificate["inverse_equation"].strip():
        raise ValueError("parity certificate has no inverse eikonal equation")
    if certificate.get("metric_interpolation") \
            != PRODUCTION_METRIC_INTERPOLATION:
        raise ValueError(
            "parity certificate lacks production-equivalent center/face "
            "metric reconstruction"
        )
    certificate_eikonal_substeps = certificate.get("eikonal_substeps")
    if isinstance(certificate_eikonal_substeps, bool) \
            or not isinstance(certificate_eikonal_substeps, int) \
            or certificate_eikonal_substeps < 1:
        raise ValueError(
            "parity certificate has no valid eikonal substep contract"
        )
    certificate_certifier_sha256 = normalized_sha256(
        certificate.get("certifier_sha256")
    )
    if certificate_certifier_sha256 is None:
        raise ValueError("parity certificate has no valid certifier SHA-256")

    provenance = certificate.get("provenance")
    manifest = provenance.get("coordinate_map_manifest") \
        if isinstance(provenance, dict) else None
    if not isinstance(manifest, dict):
        raise ValueError("parity certificate has no coordinate-map provenance")
    if manifest.get("schema") != "venus_v3_sc_to_ef_null_map" \
            or manifest.get("schema_version") != 2 \
            or manifest.get("donor_history_format_version") != 2 \
            or manifest.get("saved_metric_interpolation") \
                != SAVED_METRIC_INTERPOLATION:
        raise ValueError(
            "parity certificate coordinate-map provenance lacks the "
            "face-aware v2 contract"
        )
    manifest_digest = normalized_sha256(manifest.get("actual_sha256"))
    if manifest_digest is None \
            or normalized_sha256(manifest.get("schwarzschild_summary_sha256")) \
                != manifest_digest \
            or normalized_sha256(manifest.get("eddington_finkelstein_summary_sha256")) \
                != manifest_digest:
        raise ValueError(
            "coordinate-map manifest hash is not authenticated by both run summaries"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("parity certificate has no authenticated map artifacts")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or record.get("verified") is not True:
            raise ValueError("unverified coordinate-map artifact: {}".format(name))
        expected = normalized_sha256(record.get("expected_sha256"))
        actual = normalized_sha256(record.get("actual_sha256"))
        if expected is None or expected != actual:
            raise ValueError("coordinate-map artifact hash mismatch: {}".format(name))
    donor_record = artifacts.get("donor_history")
    donor_audit = provenance.get("dense_time_reversed_donor_history")
    if not isinstance(donor_record, dict) \
            or not isinstance(donor_audit, dict) \
            or donor_audit.get("format_version") != 2 \
            or donor_audit.get("production_metric_reconstruction") is not True \
            or not isinstance(donor_record.get("path"), str) \
            or not isinstance(donor_audit.get("path"), str) \
            or os.path.realpath(donor_record["path"]) \
                != os.path.realpath(donor_audit["path"]):
        raise ValueError(
            "parity certificate does not bind its face-aware v2 donor history"
        )

    # Strict empirical phase-space evidence accompanies metric/stress comparisons because agreement of low moments alone leaves distribution differences unresolved.
    distribution = certificate.get("distribution")
    if not isinstance(distribution, dict) \
            or distribution.get("mode") != "strict" \
            or not isinstance(distribution.get("evidence"), list) \
            or not distribution["evidence"] \
            or distribution.get("failed_fields") != []:
        raise ValueError(
            "threshold gate requires strict, passing empirical-distribution evidence"
        )

    mapping = certificate.get("mapping")
    tolerances = certificate.get("tolerances")
    if not isinstance(mapping, dict) or not isinstance(tolerances, dict):
        raise ValueError("parity certificate lacks map errors/tolerances")
    inverse_error = mapping.get("inverse_maximum_step_refinement_error")
    composition_time = mapping.get("maximum_composition_time_error")
    composition_jacobian = mapping.get("maximum_composition_jacobian_error")
    inverse_tolerance = tolerances.get("eikonal_step_refinement")
    composition_tolerance = tolerances.get("forward_inverse_composition")
    if not all(finite_number(value) and float(value) >= 0.0 for value in (
            inverse_error, composition_time, composition_jacobian,
            inverse_tolerance, composition_tolerance)):
        raise ValueError("parity certificate has non-finite inverse/composition data")
    if float(inverse_error) > float(inverse_tolerance) or max(
            float(composition_time), float(composition_jacobian)) \
            > float(composition_tolerance):
        raise ValueError("parity certificate inverse/composition errors exceed tolerance")

    convergence = certificate.get("convergence")
    evidence = convergence.get("evidence") \
        if isinstance(convergence, dict) else None
    observed = convergence.get("observed_maximum_relative_error") \
        if isinstance(convergence, dict) else None
    if not isinstance(evidence, dict) or not isinstance(observed, dict):
        raise ValueError("parity certificate has no convergence audit")
    if not isinstance(evidence.get("resolution_levels"), int) \
            or evidence["resolution_levels"] < 2 \
            or not isinstance(evidence.get("independent_pic_realizations"), int) \
            or evidence["independent_pic_realizations"] < 2 \
            or evidence.get("monotone_refinement") is not True:
        raise ValueError(
            "parity certificate lacks two-resolution/two-realization refinement"
        )
    if evidence.get("metric_interpolation") \
            != PRODUCTION_METRIC_INTERPOLATION:
        raise ValueError(
            "parity convergence audit lacks the production metric contract"
        )
    if evidence.get("eikonal_substeps") != certificate_eikonal_substeps:
        raise ValueError(
            "parity convergence audit uses different eikonal substeps"
        )
    if normalized_sha256(evidence.get("diagnostic_certifier_sha256")) \
            != certificate_certifier_sha256:
        raise ValueError(
            "parity diagnostics and terminal certificate used different certifiers"
        )
    convergence_builder_sha256 = normalized_sha256(
        evidence.get("convergence_builder_sha256")
    )
    if convergence_builder_sha256 is None:
        raise ValueError("parity convergence audit has no valid builder SHA-256")
    evidence_actual = normalized_sha256(evidence.get("actual_sha256"))
    evidence_sidecar = normalized_sha256(evidence.get("sidecar_sha256"))
    evidence_path = evidence.get("path")
    if evidence_actual is None or evidence_actual != evidence_sidecar \
            or not isinstance(evidence_path, str) \
            or evidence.get("sidecar_named_file") \
                != os.path.basename(evidence_path):
        raise ValueError("parity convergence evidence is not SHA-authenticated")
    if file_sha256(evidence_path) != evidence_actual:
        raise ValueError("parity convergence evidence file has changed")
    try:
        with open(evidence_path, "r") as handle:
            evidence_document = json.load(handle)
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("parity convergence evidence cannot be read: {}".format(
            error))
    if evidence_document.get("schema") \
            != "venus_wpic_dynamic_coordinate_convergence_v1" \
            or evidence_document.get("status") != "pass" \
            or evidence_document.get("metric_interpolation") \
                != PRODUCTION_METRIC_INTERPOLATION \
            or evidence_document.get("eikonal_substeps") \
                != certificate_eikonal_substeps \
            or normalized_sha256(evidence_document.get(
                "diagnostic_certifier_sha256")) \
                != certificate_certifier_sha256 \
            or normalized_sha256(evidence_document.get(
                "convergence_builder_sha256")) \
                != convergence_builder_sha256 \
            or normalized_sha256(evidence_document.get(
                "coordinate_map_manifest_sha256")) != manifest_digest:
        raise ValueError(
            "parity convergence evidence targets the wrong coordinate map"
        )
    if evidence_document.get("resolution_levels") \
            != evidence["resolution_levels"] \
            or evidence_document.get("independent_pic_realizations") \
                != evidence["independent_pic_realizations"] \
            or evidence_document.get("monotone_refinement") \
                is not evidence["monotone_refinement"]:
        raise ValueError(
            "certificate convergence audit disagrees with its evidence file"
        )
    convergence_artifacts = evidence.get("artifacts")
    if not isinstance(convergence_artifacts, list) \
            or len(convergence_artifacts) < 2:
        raise ValueError("parity convergence audit has too few artifacts")
    document_artifacts = evidence_document.get("artifacts")
    if not isinstance(document_artifacts, list) \
            or len(document_artifacts) != len(convergence_artifacts):
        raise ValueError(
            "certificate convergence artifacts disagree with the evidence file"
        )
    evidence_root = os.path.dirname(os.path.abspath(evidence_path))
    # Join each convergence artifact to the underlying evidence file and hash the actual referenced input again.
    for index, (record, document_record) in enumerate(zip(
            convergence_artifacts, document_artifacts)):
        if not isinstance(record, dict) or record.get("verified") is not True:
            raise ValueError(
                "unverified convergence artifact {}".format(index))
        expected = normalized_sha256(record.get("expected_sha256"))
        actual = normalized_sha256(record.get("actual_sha256"))
        if not isinstance(document_record, dict):
            raise ValueError(
                "malformed convergence evidence artifact {}".format(index))
        document_expected = normalized_sha256(document_record.get("sha256"))
        document_path = document_record.get("path")
        if not isinstance(document_path, str):
            raise ValueError(
                "convergence evidence artifact {} has no path".format(index))
        resolved_document_path = os.path.abspath(
            document_path if os.path.isabs(document_path)
            else os.path.join(evidence_root, document_path)
        )
        if expected is None or expected != actual \
                or document_expected != expected \
                or os.path.realpath(record.get("path", "")) \
                    != os.path.realpath(resolved_document_path) \
                or file_sha256(resolved_document_path) != expected:
            raise ValueError(
                "convergence artifact hash mismatch: {}".format(index))
    uncertainty_bounds = evidence.get("uncertainty_bounds")
    if not isinstance(uncertainty_bounds, dict):
        raise ValueError("parity convergence audit has no uncertainty bounds")
    normalized_bounds = {}
    normalized_observed = {}
    document_bounds = evidence_document.get("uncertainty_bounds")
    if not isinstance(document_bounds, dict):
        raise ValueError("convergence evidence file has no uncertainty bounds")
    # Every measured category must lie within the same positive uncertainty bound stated by the authenticated convergence calculation.
    for category in ("metric", "stress", "moment", "distribution"):
        bound = uncertainty_bounds.get(category)
        document_bound = document_bounds.get(category)
        value = observed.get(category)
        if not finite_number(bound) or not float(bound) > 0.0 \
                or not finite_number(document_bound) \
                or float(document_bound) != float(bound) \
                or not finite_number(value) or not float(value) >= 0.0 \
                or float(value) > float(bound):
            raise ValueError(
                "parity {} error is outside its convergence bound".format(
                    category))
        normalized_bounds[category] = float(bound)
        normalized_observed[category] = float(value)

    return {
        "path": path,
        "schema": certificate["schema"],
        "status": certificate["status"],
        "actual_sha256": actual_digest,
        "expected_sha256": expected_digest,
        "digest_source": digest_source,
        "coordinate_map_manifest_sha256": manifest_digest,
        "authenticated_artifacts": sorted(artifacts),
        "strict_distribution_evidence_count": len(distribution["evidence"]),
        "inverse_maximum_step_refinement_error": float(inverse_error),
        "maximum_composition_time_error": float(composition_time),
        "maximum_composition_jacobian_error": float(composition_jacobian),
        "convergence_evidence_sha256": evidence_actual,
        "convergence_resolution_levels": evidence["resolution_levels"],
        "convergence_independent_pic_realizations": evidence[
            "independent_pic_realizations"],
        "convergence_authenticated_artifact_count": len(
            convergence_artifacts),
        "convergence_uncertainty_bounds": normalized_bounds,
        "convergence_observed_errors": normalized_observed,
        "eikonal_substeps": certificate_eikonal_substeps,
        "certifier_sha256": certificate_certifier_sha256,
        "convergence_builder_sha256": convergence_builder_sha256,
        "metric_interpolation": PRODUCTION_METRIC_INTERPOLATION,
    }


def validate_parity_certificate(args):
    """Expose the strict certificate gate as a retained workflow artifact."""
    try:
        result = parity_certificate_gate(args)
        result["issues"] = []
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        result = {
            "status": "fail",
            "path": os.path.abspath(args.parity_certificate),
            "issues": [str(error)],
        }
    emit_json(result, args.output)
    return 0 if result.get("status") == "pass" else SCIENTIFIC_INCONCLUSIVE


def run_provenance(run_dir):
    root = os.path.abspath(run_dir)
    return {
        "physical_ic_sha256": parse_sha256_manifest(os.path.join(root, "physical_ic.sha256")),
        "binary_sha256": parse_sha256_manifest(os.path.join(root, "binary.sha256")),
        "source_manifest_sha256": file_sha256(os.path.join(root, "source.sha256")),
    }


# Parse complete source and executable records recorded at launch. These identify the calculation even after its working checkout changes.
def authenticate_run_build_provenance(run_dir):
    """Authenticate the immutable source and executable records for one run.

    ``source.sha256`` is a multi-entry ``sha256sum`` manifest.  Its own
    content digest is the source identity because the launch checkout may no
    longer exist when a sensitivity certificate is audited.  ``binary.sha256``
    records the content digest of the executable used at launch.  Validate the
    complete record syntax before returning either identity; a digest-shaped
    first token by itself is not sufficient provenance.
    """
    root = os.path.abspath(run_dir)
    source_path = os.path.join(root, "source.sha256")
    binary_path = os.path.join(root, "binary.sha256")

    try:
        with open(source_path, "r") as handle:
            source_lines = handle.read().splitlines()
    except (IOError, OSError) as error:
        raise ValueError("run source provenance is missing: {}".format(error))
    if not source_lines:
        raise ValueError("run source provenance is empty")
    for line_number, line in enumerate(source_lines, start=1):
        fields = line.split(maxsplit=1)
        digest = normalized_sha256(fields[0]) if fields else None
        named_source = fields[1].lstrip("*") if len(fields) == 2 else ""
        if digest is None or not named_source or os.path.isabs(named_source):
            raise ValueError(
                "run source provenance line {} is malformed".format(
                    line_number
                )
            )

    try:
        with open(binary_path, "r") as handle:
            binary_lines = handle.read().splitlines()
    except (IOError, OSError) as error:
        raise ValueError("run binary provenance is missing: {}".format(error))
    if len(binary_lines) != 1:
        raise ValueError("run binary provenance must contain exactly one record")
    fields = binary_lines[0].split(maxsplit=1)
    binary_digest = normalized_sha256(fields[0]) if fields else None
    binary_name = fields[1].lstrip("*") if len(fields) == 2 else ""
    if binary_digest is None or not os.path.isabs(binary_name):
        raise ValueError("run binary provenance is malformed")

    source_manifest_digest = file_sha256(source_path)
    binary_record_digest = file_sha256(binary_path)
    if source_manifest_digest is None or binary_record_digest is None:
        raise ValueError("run build provenance could not be hashed")
    return {
        "source_manifest_path": source_path,
        "source_manifest_sha256": source_manifest_digest,
        "binary_record_path": binary_path,
        "binary_record_sha256": binary_record_digest,
        "binary_sha256": binary_digest,
        "binary_path_at_launch": binary_name,
    }


# Mass conservation and ADM drift are assessed before a physical outcome enters the classification.
def numerical_quality(high, max_adm_drift, max_rest_mass_error):
    reasons = []
    try:
        adm_drift = as_float(high, "adm_mass_relative_drift")
        rest_error = as_float(high, "rest_mass_relative_error")
    except ValueError as error:
        return False, [str(error)]
    if adm_drift < 0.0 or adm_drift >= max_adm_drift:
        reasons.append(
            "adm_mass_relative_drift={} is not below {}".format(adm_drift, max_adm_drift)
        )
    if rest_error < 0.0 or rest_error >= max_rest_mass_error:
        reasons.append(
            "rest_mass_relative_error={} is not below {}".format(rest_error, max_rest_mass_error)
        )
    return not reasons, reasons


def validate_run(args):
    try:
        summary_path, summary, high = load_summary(args.run_dir)
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        eprint("validation error: {}".format(error))
        return 1

    outcome = high.get("outcome")
    reasons = []
    if outcome not in RECOGNIZED_OUTCOMES:
        reasons.append("unrecognized outcome {!r}".format(outcome))
    quality_ok, quality_reasons = numerical_quality(
        high, args.max_adm_drift, args.max_rest_mass_error
    )
    reasons.extend(quality_reasons)

    horizon = high.get("horizon_formed") is True
    halo = high.get("virialized_halo") is True
    if args.expected == "initialized":
        expected_ok = outcome == "initialized"
    elif args.expected == "black_hole":
        expected_ok = outcome in {"black_hole", "black_hole_threshold_proxy"} and horizon
    # A halo outcome requires the explicit saved halo flag and completion of its configured time interval.
    elif args.expected == "halo":
        expected_ok = outcome == "virialized_halo" and halo and not horizon
        if expected_ok:
            end_time = high.get("end_time")
            final_time = last_history_time(args.run_dir)
            if not finite_number(end_time) or final_time is None \
                    or final_time < (1.0 - args.end_time_tolerance) * float(end_time):
                expected_ok = False
                reasons.append("halo run did not demonstrably reach its configured end_time")
    else:
        expected_ok = outcome in {
            "black_hole", "black_hole_threshold_proxy", "virialized_halo", "undetermined"
        }

    if not expected_ok:
        reasons.append("outcome {!r} does not satisfy expected={!r}".format(outcome, args.expected))
    valid = quality_ok and expected_ok and outcome in RECOGNIZED_OUTCOMES
    result = {
        "status": "valid" if valid else "invalid",
        "summary": summary_path,
        "coordinates": summary.get("coordinates"),
        "expected": args.expected,
        "outcome": outcome,
        "reasons": reasons,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if valid else SCIENTIFIC_INCONCLUSIVE


def parse_milestones(text):
    values = []
    for token in text.split(","):
        value = float(token.strip())
        if not math.isfinite(value) or not 0.0 < value < 1.5:
            raise ValueError("invalid compactness milestone {}".format(token))
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("at least one compactness milestone is required")
    return values


def first_upcrossing(rows, milestone):
    if rows[0]["max_compactness"] >= milestone:
        return dict(rows[0]), True
    previous = rows[0]
    for current in rows[1:]:
        c0 = previous["max_compactness"]
        c1 = current["max_compactness"]
        if c0 < milestone <= c1 and c1 > c0:
            # Interpolate diagnostic values at the first upward compactness crossing, preserving the distinction from equal-coordinate-time comparison.
            fraction = (milestone - c0) / (c1 - c0)
            interpolated = {}
            for key in previous:
                interpolated[key] = previous[key] + fraction * (current[key] - previous[key])
            interpolated["max_compactness"] = milestone
            return interpolated, False
        previous = current
    return None, False


def symmetric_relative_difference(left, right, floor=1.0e-14):
    return abs(left - right) / max(abs(left), abs(right), floor)


def angular_diagnostics(high):
    input_mass = as_float(high, "total_marker_mass")
    initial_adm_mass = as_float(high, "initial_adm_mass")
    if input_mass <= 0.0 or initial_adm_mass <= 0.0:
        raise ValueError("angular normalization masses must be positive")
    result = {
        "fixed_specific_angular_momentum": as_float(
            high, "fixed_specific_angular_momentum"
        ),
        "angular_momentum_scale": as_float(high, "angular_momentum_scale"),
        "input_marker_mass": input_mass,
        "initial_adm_mass": initial_adm_mass,
    }
    for name in (
            "mass_weighted_mean_specific_angular_momentum",
            "mass_weighted_rms_specific_angular_momentum",
            "maximum_specific_angular_momentum"):
        value = as_float(high, name)
        result[name] = value
        result[name + "_over_input_marker_mass"] = value / input_mass
        result[name + "_over_initial_adm_mass"] = value / initial_adm_mass
    return result


def compare_parity(args):
    try:
        milestones = parse_milestones(args.milestones)
        left_summary_path, left_summary, left_high = load_summary(args.schwarzschild)
        right_summary_path, right_summary, right_high = load_summary(args.ef)
        left_history_path, left_rows = load_history(args.schwarzschild)
        right_history_path, right_rows = load_history(args.ef)
        left_angular = angular_diagnostics(left_high)
        right_angular = angular_diagnostics(right_high)
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        eprint("parity analysis error: {}".format(error))
        return 1

    provenance_left = run_provenance(args.schwarzschild)
    provenance_right = run_provenance(args.ef)
    provenance_left["base_ic_fingerprint"] = left_high.get("base_ic_fingerprint")
    provenance_right["base_ic_fingerprint"] = right_high.get("base_ic_fingerprint")
    provenance_left["realized_ic_fingerprint"] = left_high.get("realized_ic_fingerprint")
    provenance_right["realized_ic_fingerprint"] = right_high.get("realized_ic_fingerprint")
    provenance_missing = []
    provenance_mismatch = []
    for key in sorted(provenance_left):
        if provenance_left[key] is None or provenance_right[key] is None:
            provenance_missing.append(key)
        elif provenance_left[key] != provenance_right[key]:
            provenance_mismatch.append(key)

    coordinate_issues = []
    if left_summary.get("coordinates") != "schwarzschild":
        coordinate_issues.append("left run is not schwarzschild")
    if right_summary.get("coordinates") != "eddington_finkelstein":
        coordinate_issues.append("right run is not eddington_finkelstein")

    rows_out = []
    compared = 0
    failed_milestones = 0
    # Initial states already beyond a milestone supply no dynamical crossing evidence for that milestone.
    initial_cutoff = max(
        left_rows[0]["max_compactness"], right_rows[0]["max_compactness"]
    )
    # Compare only milestones reached by both histories and above their initial compactness.
    for milestone in milestones:
        left, left_is_initial = first_upcrossing(left_rows, milestone)
        right, right_is_initial = first_upcrossing(right_rows, milestone)
        row = {
            "compactness": milestone,
            "status": "not_reached",
            "schwarzschild_time": None if left is None else left["time"],
            "ef_time": None if right is None else right["time"],
        }
        if left is None or right is None:
            rows_out.append(row)
            continue
        if milestone <= initial_cutoff or left_is_initial or right_is_initial:
            row["status"] = "at_or_below_initial_compactness"
            rows_out.append(row)
            continue

        differences = {
            "rest_mass_relative_difference": symmetric_relative_difference(
                left["rest_mass"], right["rest_mass"]
            ),
            "adm_mass_relative_difference": symmetric_relative_difference(
                left["adm_mass"], right["adm_mass"]
            ),
            "r10_relative_difference": symmetric_relative_difference(left["r10"], right["r10"]),
            "r50_relative_difference": symmetric_relative_difference(left["r50"], right["r50"]),
            "r90_relative_difference": symmetric_relative_difference(left["r90"], right["r90"]),
            "virial_ratio_absolute_difference": abs(
                left["virial_ratio"] - right["virial_ratio"]
            ),
        }
        row.update(differences)
        failures = []
        for key in ("rest_mass_relative_difference", "adm_mass_relative_difference"):
            if differences[key] > args.mass_relative_tolerance:
                failures.append(key)
        for key in ("r10_relative_difference", "r50_relative_difference", "r90_relative_difference"):
            if differences[key] > args.relative_tolerance:
                failures.append(key)
        if differences["virial_ratio_absolute_difference"] > args.virial_absolute_tolerance:
            failures.append("virial_ratio_absolute_difference")
        row["failed_diagnostics"] = failures
        row["status"] = "pass" if not failures else "fail"
        compared += 1
        if failures:
            failed_milestones += 1
        rows_out.append(row)

    if coordinate_issues or provenance_mismatch:
        status = "fail"
    elif provenance_missing and not args.allow_missing_provenance:
        status = "inconclusive"
    elif compared < args.minimum_milestones:
        status = "inconclusive"
    elif failed_milestones:
        status = "fail"
    else:
        status = "pass"

    result = {
        "status": status,
        "comparison_scope": (
            "matched marker-family dynamics; not an assertion that the "
            "coordinate-specific initial slices are identical physical hypersurfaces"
        ),
        "alignment": (
            "first upward crossing of gauge-invariant areal compactness 2m/r"
        ),
        "schwarzschild_summary": left_summary_path,
        "eddington_finkelstein_summary": right_summary_path,
        "schwarzschild_history": left_history_path,
        "eddington_finkelstein_history": right_history_path,
        "provenance": {
            "schwarzschild": provenance_left,
            "eddington_finkelstein": provenance_right,
            "missing": provenance_missing,
            "mismatch": provenance_mismatch,
        },
        "coordinate_issues": coordinate_issues,
        "angular_momentum": {
            "schwarzschild": left_angular,
            "eddington_finkelstein": right_angular,
        },
        "tolerances": {
            "mass_relative": args.mass_relative_tolerance,
            "radius_relative": args.relative_tolerance,
            "virial_absolute": args.virial_absolute_tolerance,
            "minimum_milestones": args.minimum_milestones,
        },
        "compared_milestones": compared,
        "failed_milestones": failed_milestones,
        "milestones": rows_out,
    }
    emit_json(result, args.output)
    if args.csv_output:
        parity_csv(rows_out, args.csv_output)
    return 0 if status == "pass" else SCIENTIFIC_INCONCLUSIVE


def parity_csv(rows, path):
    fields = [
        "compactness", "status", "schwarzschild_time", "ef_time",
        "rest_mass_relative_difference", "adm_mass_relative_difference",
        "r10_relative_difference", "r50_relative_difference",
        "r90_relative_difference", "virial_ratio_absolute_difference",
        "failed_diagnostics",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["failed_diagnostics"] = ";".join(row.get("failed_diagnostics", []))
            writer.writerow(row)


def emit_json(value, path):
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path:
        with open(path, "w") as handle:
            handle.write(rendered)
    else:
        sys.stdout.write(rendered)


def discover_summary_paths(campaign_root, run_dirs):
    paths = []
    if campaign_root:
        root = os.path.abspath(campaign_root)
        for directory, subdirectories, files in os.walk(root):
            subdirectories[:] = [name for name in subdirectories if name != ".build-locks"]
            if "summary.json" in files:
                paths.append(os.path.join(directory, "summary.json"))
    for run_dir in run_dirs or []:
        paths.append(os.path.join(os.path.abspath(run_dir), "summary.json"))
    return sorted(set(paths))


def missing_receipt_outputs(campaign_root):
    """Return scan rows that were submitted but have no summary yet."""
    if not campaign_root:
        return []
    receipt_path = os.path.join(os.path.abspath(campaign_root), "submission.tsv")
    try:
        handle = open(receipt_path, "r", newline="")
    except (IOError, OSError):
        return []
    missing = []
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("role") != "scan":
                continue
            output_dir = row.get("output_dir")
            if output_dir and not os.path.isfile(os.path.join(output_dir, "summary.json")):
                missing.append({
                    "summary": os.path.join(output_dir, "summary.json"),
                    "error": "submitted scan point has no summary.json",
                    "job_id": row.get("job_id"),
                })
    return missing


# A single EF run supplies intrinsic horizon and causal-excision evidence. Independent parameter sensitivity is checked separately.
def intrinsic_post_horizon_evidence(summary, high, history, args):
    """Audit the evidence intrinsic to one EF evolution.

    Passing this audit establishes only a ``black_hole_candidate``.  It does
    not establish excision-parameter independence, so it can never close the
    final angular-momentum bracket by itself.
    """
    coordinate = summary.get("coordinates")
    audit = high.get("ef_post_horizon_evidence")
    if not isinstance(audit, dict):
        audit = {}
    trapped_time = high.get("first_trapped_time", high.get("horizon_time"))
    trapped_time_value = float(trapped_time) if finite_number(trapped_time) else None
    final_time = history[-1]["time"]
    fixed_mass = high.get("fixed_family_adm_mass")
    fixed_mass_value = (
        float(fixed_mass)
        if finite_number(fixed_mass) and float(fixed_mass) > 0.0 else None
    )
    declared_post = high.get("post_horizon_advanced_time")
    declared_post_value = (
        float(declared_post)
        if finite_number(declared_post) and float(declared_post) >= 0.0 else None
    )
    computed_post = (
        final_time - trapped_time_value
        if trapped_time_value is not None else None
    )
    # Normalize post-horizon duration by the family ADM mass and enforce the documented minimum duration and sample count.
    minimum_mass_times = max(5.0, float(getattr(
        args, "minimum_post_horizon_mass_times", 5.0
    )))
    minimum_saved_samples = max(3, int(getattr(
        args, "minimum_post_horizon_saved_samples", 3
    )))

    post_rows = (
        [row for row in history if row["time"] > trapped_time_value]
        if trapped_time_value is not None else []
    )
    horizon_rows = (
        [row for row in history
         if row["time"] >= trapped_time_value
         and row.get("apparent_horizon_radius", -1.0) > 0.0]
        if trapped_time_value is not None else []
    )
    positive_excision_rows = [
        row for row in history if row.get("excision_radius", 0.0) > 0.0
    ]
    # Saved excision must remain strictly inside the apparent horizon, with actual removed interior rest weight accounted separately.
    causal_rows = [
        row for row in post_rows
        if row.get("apparent_horizon_radius", -1.0) > 0.0
        and row.get("excision_radius", 0.0) > 0.0
        and row["excision_radius"] < row["apparent_horizon_radius"]
    ]
    excised_rows = [
        row for row in causal_rows if row.get("excised_rest_mass", 0.0) > 0.0
    ]
    history_min_b = min(
        (row.get("min_b", -1.0) for row in history), default=-1.0
    )
    history_adm_max_relative_error = None
    if fixed_mass_value is not None:
        history_adm_max_relative_error = max(
            abs(row["adm_mass"] - fixed_mass_value) / fixed_mass_value
            for row in history
        )
    # Total rest weight closes against active, excised and escaped populations at every saved history epoch.
    mass_accounting_max_relative_error = max(
        abs(
            row["rest_mass"]
            - row.get("active_rest_mass", float("nan"))
            - row.get("excised_rest_mass", float("nan"))
            - row.get("escaped_rest_mass", float("nan"))
        ) / max(abs(row["rest_mass"]), 1.0e-300)
        for row in history
    )

    target_mass = high.get("target_initial_adm_mass")
    null_mass = high.get("null_slice_conserved_adm_mass")
    deposited_mass = high.get("initial_deposited_adm_mass")
    # Keep target, family, null-slice and deposited ADM references distinct before measuring their consistency.
    positive_mass_chain = all(
        finite_number(value) and float(value) > 0.0
        for value in (target_mass, fixed_mass, null_mass, deposited_mass)
    )
    fixed_target_error = None
    null_family_error = None
    deposition_error = None
    if positive_mass_chain:
        target_mass = float(target_mass)
        null_mass = float(null_mass)
        deposited_mass = float(deposited_mass)
        fixed_target_error = abs(fixed_mass_value - target_mass) / target_mass
        null_family_error = abs(null_mass - fixed_mass_value) / fixed_mass_value
        deposition_error = abs(deposited_mass - null_mass) / null_mass

    interval_consistent = (
        computed_post is not None and computed_post > 0.0
        and declared_post_value is not None
        and abs(declared_post_value - computed_post)
            <= float(getattr(args, "end_time_tolerance", 1.0e-6))
                * max(1.0, abs(final_time), abs(trapped_time_value))
    )
    duration_over_mass = (
        computed_post / fixed_mass_value
        if computed_post is not None and fixed_mass_value is not None else None
    )
    configured_excision_a = audit.get("configured_excision_a_threshold")
    configured_excision_buffer = audit.get(
        "configured_excision_buffer_cells"
    )
    configured_fraction = audit.get("configured_excision_horizon_fraction", 1.0)
    fraction_valid = finite_number(configured_fraction) and 0.0 < float(configured_fraction) <= 1.0
    checks = {
        "ingoing_ef_coordinates": coordinate == "eddington_finkelstein",
        "raw_horizon_outcome_consistent": (
            high.get("horizon_formed") is True
            and high.get("outcome") == "black_hole"
        ),
        "finite_first_trapped_time": (
            trapped_time_value is not None and trapped_time_value >= 0.0
        ),
        "declared_post_horizon_interval_consistent": interval_consistent,
        "driver_evolved_past_horizon_flag": (
            high.get("evolved_past_horizon") is True
        ),
        "minimum_dimensionless_post_horizon_duration": (
            duration_over_mass is not None
            and duration_over_mass >= minimum_mass_times
        ),
        "minimum_strictly_post_horizon_saved_samples": (
            len(post_rows) >= minimum_saved_samples
        ),
        "outer_apparent_horizon_saved": bool(horizon_rows),
        "accepted_step_audit_schema": (
            audit.get("schema")
                == "venus_wpic_ef_post_horizon_evidence_v1"
            and audit.get("available") is True
        ),
        "accepted_step_audit_complete": (
            audit.get("accepted_step_audit_complete") is True
            and isinstance(audit.get("accepted_step_audit_samples"), int)
            and audit.get("accepted_step_audit_samples") > 0
        ),
        "finite_fields_throughout": (
            audit.get("finite_fields_throughout") is True
        ),
        "b_strictly_positive_throughout": (
            finite_number(audit.get("minimum_b_over_evolution"))
            and float(audit["minimum_b_over_evolution"]) > 0.0
            and history_min_b > 0.0
        ),
        "p_r_strictly_positive_throughout": (
            finite_number(audit.get("minimum_p_r_over_evolution"))
            and float(audit["minimum_p_r_over_evolution"]) > 0.0
        ),
        "excision_controls_explicit_and_valid": (
            finite_number(configured_excision_a)
            and float(configured_excision_a) < 0.0
            and isinstance(configured_excision_buffer, int)
            and not isinstance(configured_excision_buffer, bool)
            and configured_excision_buffer >= 1
            and fraction_valid
        ),
        "excision_radius_cap_respected": (
            fraction_valid and all(
                row.get("excision_radius", 0.0)
                <= float(configured_fraction) * row.get("apparent_horizon_radius", 0.0) + 1e-12
                for row in positive_excision_rows
            )
        ),
        "fixed_adm_provenance_complete": positive_mass_chain,
        "fixed_adm_mass_matches_target": (
            fixed_target_error is not None
            and fixed_target_error <= args.fixed_adm_relative_tolerance
        ),
        "null_slice_adm_matches_fixed_family": (
            null_family_error is not None
            and null_family_error <= args.max_adm_drift
        ),
        "initial_deposition_adm_matches_null_slice": (
            deposition_error is not None
            and deposition_error <= args.max_adm_drift
        ),
        "fixed_adm_bookkeeping_throughout": (
            audit.get("fixed_adm_bookkeeping_throughout") is True
            and history_adm_max_relative_error is not None
            and history_adm_max_relative_error <= args.max_adm_drift
        ),
        "rest_weight_accounting_throughout": (
            finite_number(mass_accounting_max_relative_error)
            and mass_accounting_max_relative_error
                <= args.max_rest_mass_error
        ),
        "causal_excision_inside_horizon": (
            bool(positive_excision_rows)
            and len(causal_rows) == len(positive_excision_rows)
            and audit.get("causal_excision_valid_throughout") is True
            and audit.get("causal_excision_inside_horizon_seen") is True
            and finite_number(audit.get("maximum_excision_radius"))
            and float(audit["maximum_excision_radius"]) > 0.0
        ),
        "actual_excised_interior_weight": (
            bool(excised_rows)
            and audit.get("excised_interior_rest_weight_seen") is True
            and finite_number(audit.get("maximum_excised_rest_weight"))
            and float(audit["maximum_excised_rest_weight"]) > 0.0
        ),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": "venus_wpic_post_horizon_intrinsic_gate_v1",
        "status": "pass" if not failed_checks else "fail",
        "classification_if_passed": "black_hole_candidate",
        "checks": checks,
        "failed_checks": failed_checks,
        "thresholds": {
            "minimum_post_horizon_duration_over_fixed_family_adm_mass": (
                minimum_mass_times
            ),
            "minimum_strictly_post_horizon_saved_samples": minimum_saved_samples,
            "fixed_adm_relative_tolerance": args.fixed_adm_relative_tolerance,
            "adm_chain_relative_tolerance": args.max_adm_drift,
            "rest_weight_accounting_relative_tolerance": args.max_rest_mass_error,
        },
        "measurements": {
            "first_trapped_time": trapped_time_value,
            "final_saved_time": final_time,
            "declared_post_horizon_advanced_time": declared_post_value,
            "computed_post_horizon_advanced_time": computed_post,
            "post_horizon_duration_over_fixed_family_adm_mass": duration_over_mass,
            "strictly_post_horizon_saved_samples": len(post_rows),
            "saved_horizon_samples": len(horizon_rows),
            "saved_causal_excision_samples": len(causal_rows),
            "saved_positive_excision_samples": len(positive_excision_rows),
            "saved_excised_interior_samples": len(excised_rows),
            "minimum_saved_b": history_min_b,
            "minimum_accepted_step_p_r": audit.get(
                "minimum_p_r_over_evolution"
            ),
            "minimum_accepted_step_b": audit.get(
                "minimum_b_over_evolution"
            ),
            "maximum_excision_radius": audit.get(
                "maximum_excision_radius"
            ),
            "maximum_excised_rest_weight": audit.get(
                "maximum_excised_rest_weight"
            ),
            "maximum_history_adm_relative_error_from_fixed_family": (
                history_adm_max_relative_error
            ),
            "maximum_history_rest_weight_accounting_relative_error": (
                mass_accounting_max_relative_error
            ),
            "fixed_target_relative_error": fixed_target_error,
            "null_slice_family_relative_error": null_family_error,
            "initial_deposition_null_slice_relative_error": deposition_error,
        },
        "interpretation": (
            "pass establishes an intrinsic EF black-hole candidate only; "
            "a certified endpoint additionally requires an authenticated "
            "excision-threshold and buffer sensitivity repeat"
        ),
    }


def authenticate_file_sidecar(path, expected_sha256=None):
    actual = file_sha256(path)
    if actual is None:
        raise ValueError("cannot hash authenticated artifact: {}".format(path))
    expected = normalized_sha256(expected_sha256)
    digest_source = "argument"
    if expected_sha256 not in (None, "") and expected is None:
        raise ValueError("explicit authenticated-artifact SHA-256 is invalid")
    if expected is None:
        sidecar = path + ".sha256"
        try:
            with open(sidecar, "r") as handle:
                fields = handle.readline().strip().split()
        except (IOError, OSError):
            raise ValueError("authenticated artifact has no SHA-256 sidecar")
        if len(fields) != 2 or fields[1].lstrip("*") != os.path.basename(path):
            raise ValueError("authenticated artifact SHA-256 sidecar is malformed")
        expected = normalized_sha256(fields[0])
        digest_source = sidecar
    if expected is None or actual != expected:
        raise ValueError("authenticated artifact SHA-256 mismatch")
    return actual, digest_source


# Reclassify independent trigger and buffer variations from their own authenticated outputs instead of trusting a certificate's summary conclusion.
def excision_sensitivity_gate(primary_summary_path, primary_record, args):
    """Authenticate independent threshold and buffer variations.

    The certificate is a small immutable manifest.  The analyzer does not
    trust its scientific conclusion: it hashes every listed summary, reruns
    the intrinsic gate on each repeat, and computes the agreement below.

    Schema v1 is deliberately plain::

        {"schema": "venus_wpic_excision_sensitivity_certificate_v1",
         "status": "pass",
         "records": [{"primary_summary_sha256": "...",
                      "physical_ic_sha256": "...",
                      "binary_sha256": "...",
                      "source_manifest_sha256": "...",
                      "repeats": [{"summary": "path/summary.json",
                                   "summary_sha256": "..."}, ...]}]}

    Relative repeat paths are resolved beside the certificate.  One repeat
    must vary only the negative ``a`` trigger and another only the positive
    integer buffer.  Every summary is reclassified as an intrinsic candidate
    and must retain the same physical-IC, executable, source-manifest,
    angular-momentum, and fixed-ADM provenance.  The authenticated certificate
    binds the expected executable digest and exact source-manifest digest;
    every run's provenance records are parsed and hashed again during audit.
    """
    certificate_path = getattr(args, "excision_sensitivity_certificate", None)
    result = {
        "schema": "venus_wpic_excision_sensitivity_gate_v1",
        "status": "not_supplied",
        "certifies_black_hole": False,
        "certificate": None,
        "certificate_sha256": None,
        "failed_checks": ["authenticated_sensitivity_certificate_supplied"],
        "checks": {},
        "repeat_runs": [],
        "interpretation": (
            "a candidate is not a certified black-hole endpoint until both "
            "the excision a-threshold and buffer are varied independently"
        ),
    }
    if not certificate_path:
        return result
    certificate_path = os.path.abspath(certificate_path)
    try:
        digest, digest_source = authenticate_file_sidecar(
            certificate_path,
            getattr(args, "excision_sensitivity_certificate_sha256", None),
        )
        with open(certificate_path, "r") as handle:
            certificate = json.load(handle)
        if certificate.get("schema") \
                != "venus_wpic_excision_sensitivity_certificate_v1" \
                or certificate.get("status") != "pass":
            raise ValueError("excision sensitivity certificate is not status=pass v1")
        records = certificate.get("records")
        if not isinstance(records, list):
            raise ValueError("excision sensitivity certificate has no records list")
        primary_digest = file_sha256(primary_summary_path)
        matches = [
            record for record in records
            if isinstance(record, dict)
            and normalized_sha256(record.get("primary_summary_sha256"))
                == primary_digest
        ]
        if len(matches) != 1:
            raise ValueError(
                "certificate must contain exactly one record for the primary summary"
            )
        manifest_record = matches[0]
        manifest_physical_ic = normalized_sha256(
            manifest_record.get("physical_ic_sha256")
        )
        if manifest_physical_ic is None \
                or manifest_physical_ic \
                    != normalized_sha256(primary_record.get("physical_ic_sha256")):
            raise ValueError(
                "certificate is not bound to the primary physical IC"
            )
        primary_build_provenance = authenticate_run_build_provenance(
            os.path.dirname(os.path.abspath(primary_summary_path))
        )
        manifest_binary = normalized_sha256(
            manifest_record.get("binary_sha256")
        )
        if manifest_binary is None \
                or manifest_binary \
                    != primary_build_provenance["binary_sha256"]:
            raise ValueError(
                "certificate is not bound to the primary executable"
            )
        manifest_source = normalized_sha256(
            manifest_record.get("source_manifest_sha256")
        )
        if manifest_source is None \
                or manifest_source \
                    != primary_build_provenance["source_manifest_sha256"]:
            raise ValueError(
                "certificate is not bound to the primary source manifest"
            )
        repeat_specs = manifest_record.get("repeats")
        if not isinstance(repeat_specs, list) or len(repeat_specs) < 2:
            raise ValueError("certificate needs at least two sensitivity repeats")

        primary_audit = primary_record["evidence"]["post_horizon_evidence"]
        primary_high_audit = primary_record.get("ef_post_horizon_evidence", {})
        if normalized_sha256(primary_record.get("physical_ic_sha256")) is None \
                or not isinstance(primary_record.get("base_ic_fingerprint"), str) \
                or not primary_record["base_ic_fingerprint"] \
                or not isinstance(
                    primary_record.get("realized_ic_fingerprint"), str
                ) \
                or not primary_record["realized_ic_fingerprint"] \
                or normalized_sha256(
                    primary_record.get("coordinate_map_manifest_sha256")
                ) is None:
            raise ValueError("primary run lacks immutable physical IC provenance")
        primary_a = primary_high_audit.get("configured_excision_a_threshold")
        primary_buffer = primary_high_audit.get("configured_excision_buffer_cells")
        primary_fraction = primary_high_audit.get("configured_excision_horizon_fraction", 1.0)
        if not finite_number(primary_a) \
                or not isinstance(primary_buffer, int) \
                or isinstance(primary_buffer, bool):
            raise ValueError("primary summary lacks explicit excision controls")

        certificate_dir = os.path.dirname(certificate_path)
        repeats = []
        repeat_build_provenance = []
        repeat_digests = set()
        threshold_varied_independently = False
        buffer_varied_independently = False
        # Each sensitivity repeat requires unique summary bytes and the same executable, source and physical realization.
        for specification in repeat_specs:
            if not isinstance(specification, dict):
                raise ValueError("sensitivity repeat record is not an object")
            repeat_path = specification.get("summary")
            if not isinstance(repeat_path, str) or not repeat_path:
                raise ValueError("sensitivity repeat has no summary path")
            if not os.path.isabs(repeat_path):
                repeat_path = os.path.abspath(os.path.join(
                    certificate_dir, repeat_path
                ))
            expected_repeat_digest = normalized_sha256(
                specification.get("summary_sha256")
            )
            actual_repeat_digest = file_sha256(repeat_path)
            if expected_repeat_digest is None \
                    or actual_repeat_digest != expected_repeat_digest:
                raise ValueError(
                    "sensitivity repeat summary SHA-256 mismatch: {}".format(
                        repeat_path
                    )
                )
            if actual_repeat_digest in repeat_digests:
                raise ValueError("sensitivity certificate repeats one summary")
            repeat_digests.add(actual_repeat_digest)
            repeat_build = authenticate_run_build_provenance(
                os.path.dirname(repeat_path)
            )
            if repeat_build["binary_sha256"] \
                    != primary_build_provenance["binary_sha256"]:
                raise ValueError(
                    "sensitivity repeat changes executable provenance: {}".format(
                        repeat_path
                    )
                )
            if repeat_build["source_manifest_sha256"] \
                    != primary_build_provenance["source_manifest_sha256"]:
                raise ValueError(
                    "sensitivity repeat changes source provenance: {}".format(
                        repeat_path
                    )
                )
            repeat_record = classify_threshold_run(
                repeat_path, args, check_excision_sensitivity=False
            )
            if repeat_record is None \
                    or repeat_record["classification"] \
                        != "black_hole_candidate":
                raise ValueError(
                    "sensitivity repeat lacks intrinsic candidate evidence: {}".format(
                        repeat_path
                    )
                )
            repeat_audit = repeat_record.get("ef_post_horizon_evidence", {})
            repeat_a = repeat_audit.get("configured_excision_a_threshold")
            repeat_buffer = repeat_audit.get("configured_excision_buffer_cells")
            repeat_fraction = repeat_audit.get("configured_excision_horizon_fraction", 1.0)
            if not finite_number(primary_fraction) or not finite_number(repeat_fraction) \
                    or not 0.0 < float(primary_fraction) <= 1.0 \
                    or float(repeat_fraction) != float(primary_fraction):
                raise ValueError("threshold/buffer sensitivity repeat changes the interior radius cap")
            if not finite_number(repeat_a) \
                    or not isinstance(repeat_buffer, int) \
                    or isinstance(repeat_buffer, bool):
                raise ValueError("sensitivity repeat lacks explicit excision controls")
            # Require a trigger-only change and a buffer-only change while preserving the interior radius cap.
            threshold_varied_independently = threshold_varied_independently or (
                float(repeat_a) != float(primary_a)
                and repeat_buffer == primary_buffer
            )
            buffer_varied_independently = buffer_varied_independently or (
                float(repeat_a) == float(primary_a)
                and repeat_buffer != primary_buffer
            )
            if repeat_record.get("base_ic_fingerprint") \
                    != primary_record.get("base_ic_fingerprint") \
                    or repeat_record.get("realized_ic_fingerprint") \
                    != primary_record.get("realized_ic_fingerprint") \
                    or repeat_record.get("coordinate_map_manifest_sha256") \
                    != primary_record.get("coordinate_map_manifest_sha256") \
                    or repeat_record.get("physical_ic_sha256") \
                    != primary_record.get("physical_ic_sha256"):
                raise ValueError("sensitivity repeat changes physical IC provenance")
            angular_scale = max(1.0, abs(primary_record["angular_momentum"]))
            if abs(
                    repeat_record["angular_momentum"]
                    - primary_record["angular_momentum"]
                    ) > 1.0e-12 * angular_scale:
                raise ValueError("sensitivity repeat changes angular momentum")
            if abs(
                    repeat_record["fixed_family_adm_mass"]
                    - primary_record["fixed_family_adm_mass"]
                    ) / primary_record["fixed_family_adm_mass"] \
                    > args.fixed_adm_relative_tolerance:
                raise ValueError("sensitivity repeat changes fixed-family ADM mass")
            repeats.append(repeat_record)
            repeat_build_provenance.append(repeat_build)

        horizon_times = [
            primary_audit["measurements"]["first_trapped_time"]
        ] + [
            repeat["evidence"]["post_horizon_evidence"]["measurements"]
                ["first_trapped_time"]
            for repeat in repeats
        ]
        horizon_radii = [primary_record["apparent_horizon_radius"]] + [
            repeat["apparent_horizon_radius"] for repeat in repeats
        ]
        if any(not finite_number(value) or float(value) <= 0.0
               for value in horizon_radii):
            raise ValueError("sensitivity family lacks positive horizon radii")
        # Compare horizon timing in ADM-mass units and horizon radii relative to their family median.
        time_spread_over_mass = (
            max(horizon_times) - min(horizon_times)
        ) / primary_record["fixed_family_adm_mass"]
        radius_reference = percentile(horizon_radii, 0.5)
        radius_relative_spread = (
            max(horizon_radii) - min(horizon_radii)
        ) / radius_reference
        time_tolerance = min(1.0, float(getattr(
            args, "excision_sensitivity_max_horizon_time_spread_over_adm", 1.0
        )))
        radius_tolerance = min(0.05, float(getattr(
            args, "excision_sensitivity_max_horizon_radius_relative_spread", 0.05
        )))
        checks = {
            "independent_excision_a_threshold_variation": (
                threshold_varied_independently
            ),
            "independent_excision_buffer_variation": (
                buffer_varied_independently
            ),
            "horizon_time_agreement": time_spread_over_mass <= time_tolerance,
            "horizon_radius_agreement": radius_relative_spread <= radius_tolerance,
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        result.update({
            "status": "pass" if not failed else "fail",
            "certifies_black_hole": not failed,
            "certificate": certificate_path,
            "certificate_sha256": digest,
            "certificate_digest_source": digest_source,
            "failed_checks": failed,
            "checks": checks,
            "repeat_runs": [repeat["run_dir"] for repeat in repeats],
            "build_provenance": {
                "binary_sha256": primary_build_provenance["binary_sha256"],
                "source_manifest_sha256": primary_build_provenance[
                    "source_manifest_sha256"
                ],
                "primary": primary_build_provenance,
                "repeats": repeat_build_provenance,
            },
            "measurements": {
                "horizon_time_spread_over_fixed_family_adm_mass": (
                    time_spread_over_mass
                ),
                "horizon_radius_relative_spread": radius_relative_spread,
            },
            "thresholds": {
                "maximum_horizon_time_spread_over_fixed_family_adm_mass": (
                    time_tolerance
                ),
                "maximum_horizon_radius_relative_spread": radius_tolerance,
            },
        })
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        result.update({
            "status": "fail",
            "certifies_black_hole": False,
            "certificate": certificate_path,
            "failed_checks": ["certificate_authentication_and_reanalysis"],
            "error": str(error),
        })
    return result


# Build one physical outcome from numerical budgets, initial-family provenance, full duration and measured late or post-horizon behavior.
def classify_threshold_run(summary_path, args, check_excision_sensitivity=True):
    run_dir = os.path.dirname(summary_path)
    with open(summary_path, "r") as handle:
        summary = json.load(handle)
    high = summary.get("high_compactness")
    if summary.get("module") != "weighted_einstein_vlasov_pic" or not isinstance(high, dict):
        raise ValueError("not a high-compactness WPIC summary")
    outcome = high.get("outcome")
    if outcome == "initialized":
        return None
    coordinate = summary.get("coordinates")
    if args.coordinates != "any" and coordinate != args.coordinates:
        return None

    angular_momentum = as_float(high, args.angular_statistic)
    input_marker_mass = as_float(high, "total_marker_mass")
    initial_adm_mass = as_float(high, "initial_adm_mass")
    if input_marker_mass <= 0.0 or initial_adm_mass <= 0.0:
        raise ValueError("angular normalization masses must be positive")
    reasons = []
    quality_ok, quality_reasons = numerical_quality(
        high, args.max_adm_drift, args.max_rest_mass_error
    )
    reasons.extend(quality_reasons)
    target_initial_adm_mass = high.get("target_initial_adm_mass")
    fixed_family_adm_mass = high.get("fixed_family_adm_mass")
    initial_deposited_adm_mass = high.get(
        "initial_deposited_adm_mass", initial_adm_mass
    )
    null_slice_conserved_adm_mass = high.get(
        "null_slice_conserved_adm_mass"
    )
    fixed_family_rest_weight = high.get("fixed_family_rest_weight")
    fixed_family_binding_energy = high.get("fixed_family_binding_energy")
    fixed_family_binding_fraction = high.get("fixed_family_binding_fraction")
    if finite_number(initial_deposited_adm_mass) \
            and float(initial_deposited_adm_mass) > 0.0:
        initial_deposited_adm_mass = float(initial_deposited_adm_mass)
    else:
        initial_deposited_adm_mass = None
    if finite_number(null_slice_conserved_adm_mass) \
            and float(null_slice_conserved_adm_mass) > 0.0:
        null_slice_conserved_adm_mass = float(null_slice_conserved_adm_mass)
    else:
        null_slice_conserved_adm_mass = None
    if finite_number(fixed_family_adm_mass) \
            and float(fixed_family_adm_mass) > 0.0:
        fixed_family_adm_mass = float(fixed_family_adm_mass)
    elif coordinate == "schwarzschild":
        # Compatibility for pre-schema SC summaries is safe because their
        # initial slice is precisely the t=0 normalization slice.  The same
        # fallback would be wrong for mapped EF data and is forbidden below.
        fixed_family_adm_mass = initial_adm_mass
    else:
        fixed_family_adm_mass = None
    # Positive binding uses the family mass deficit M0-M_ADM and its rest-mass normalization, with algebraic consistency checked.
    binding_diagnostics_available = (
        finite_number(fixed_family_rest_weight)
        and float(fixed_family_rest_weight) > 0.0
        and finite_number(fixed_family_binding_energy)
        and finite_number(fixed_family_binding_fraction)
    )
    if binding_diagnostics_available:
        fixed_family_rest_weight = float(fixed_family_rest_weight)
        fixed_family_binding_energy = float(fixed_family_binding_energy)
        fixed_family_binding_fraction = float(fixed_family_binding_fraction)
        binding_scale = max(
            1.0, abs(fixed_family_rest_weight),
            abs(fixed_family_adm_mass or 0.0),
            abs(fixed_family_binding_energy),
        )
        binding_tolerance = 1.0e-12 * binding_scale
        binding_diagnostics_available = (
            fixed_family_adm_mass is not None
            and abs(
                fixed_family_binding_energy
                - (fixed_family_rest_weight - fixed_family_adm_mass)
            ) <= binding_tolerance
            and abs(
                fixed_family_binding_fraction
                - fixed_family_binding_energy / fixed_family_rest_weight
            ) <= 1.0e-12 * max(1.0, abs(fixed_family_binding_fraction))
            and high.get("fixed_family_binding_diagnostics_available") is True
            and high.get("fixed_family_positive_binding_energy") is (
                fixed_family_binding_fraction > 0.0
            )
        )
    if not binding_diagnostics_available:
        fixed_family_rest_weight = None
        fixed_family_binding_energy = None
        fixed_family_binding_fraction = None

    proxy_keys = (
        "initial_energy_at_infinity_proxy_minimum",
        "initial_energy_at_infinity_proxy_maximum",
        "initial_energy_at_infinity_proxy_rest_weighted_mean",
        "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one",
    )
    # Retain instantaneous frozen-metric particle-energy diagnostics with their own scope, separate from the family binding condition.
    energy_proxy = {
        key: (float(high[key]) if finite_number(high.get(key)) else None)
        for key in proxy_keys
    }
    energy_proxy["definition"] = high.get(
        "initial_energy_at_infinity_proxy_definition"
    )
    energy_proxy["interpretation"] = (
        "instantaneous Schwarzschild-t=0 frozen-metric proxy; not a "
        "conserved particle energy for dynamical data and not a halo gate"
    )
    proxy_values = [energy_proxy[key] for key in proxy_keys]
    energy_proxy["available"] = (
        all(value is not None for value in proxy_values)
        and energy_proxy[proxy_keys[0]] > 0.0
        and energy_proxy[proxy_keys[1]] >= energy_proxy[proxy_keys[0]]
        and energy_proxy[proxy_keys[0]]
            <= energy_proxy[proxy_keys[2]] <= energy_proxy[proxy_keys[1]]
        and 0.0 <= energy_proxy[proxy_keys[3]] <= 1.0
    )
    fixed_adm_relative_error = None
    coordinate_slice_adm_relative_error = None
    null_slice_adm_relative_drift_from_family = None
    initial_deposition_adm_relative_error_from_null_slice = None
    null_map_preflight = None
    require_fixed_adm = not getattr(args, "allow_unfixed_adm_family", False)
    if finite_number(target_initial_adm_mass) and float(target_initial_adm_mass) > 0.0:
        target_initial_adm_mass = float(target_initial_adm_mass)
        coordinate_slice_adm_relative_error = abs(
            initial_adm_mass - target_initial_adm_mass
        ) / target_initial_adm_mass
        if fixed_family_adm_mass is None:
            quality_ok = False
            reasons.append(
                "run has no positive fixed_family_adm_mass provenance"
            )
        else:
            fixed_adm_relative_error = abs(
                fixed_family_adm_mass - target_initial_adm_mass
            ) / target_initial_adm_mass
            if fixed_adm_relative_error > args.fixed_adm_relative_tolerance:
                quality_ok = False
                reasons.append(
                    "fixed-family ADM mass misses target by {:.6g}".format(
                        fixed_adm_relative_error
                    )
                )
    else:
        target_initial_adm_mass = None
        if require_fixed_adm:
            quality_ok = False
            reasons.append("run has no positive fixed target_initial_adm_mass")

    # EF classification requires the solved-null initial-data chain and its distinct ADM anchors.
    if coordinate == "eddington_finkelstein":
        if high.get("eikonal_mapped_initial_data") is not True:
            quality_ok = False
            reasons.append("EF threshold run lacks solved-eikonal initial data")
        manifest_sha = normalized_sha256(
            high.get("coordinate_map_manifest_sha256")
        )
        if manifest_sha is None:
            quality_ok = False
            reasons.append("EF threshold run lacks a valid map-manifest SHA-256")
        try:
            null_map_preflight = authenticate_null_map_preflight(run_dir, high)
        except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
            quality_ok = False
            reasons.append(
                "EF null-map preflight authentication failed: {}".format(error)
            )
        if null_slice_conserved_adm_mass is None:
            quality_ok = False
            reasons.append("EF run has no positive null-slice ADM anchor")
        if initial_deposited_adm_mass is None:
            quality_ok = False
            reasons.append("EF run has no positive deposited initial ADM mass")
        if fixed_family_adm_mass is not None \
                and null_slice_conserved_adm_mass is not None:
            null_slice_adm_relative_drift_from_family = abs(
                null_slice_conserved_adm_mass - fixed_family_adm_mass
            ) / fixed_family_adm_mass
            if null_slice_adm_relative_drift_from_family > args.max_adm_drift:
                quality_ok = False
                reasons.append(
                    "null-slice ADM drift from fixed family is {:.6g}".format(
                        null_slice_adm_relative_drift_from_family
                    )
                )
        if null_slice_conserved_adm_mass is not None \
                and initial_deposited_adm_mass is not None:
            initial_deposition_adm_relative_error_from_null_slice = abs(
                initial_deposited_adm_mass - null_slice_conserved_adm_mass
            ) / null_slice_conserved_adm_mass
            if initial_deposition_adm_relative_error_from_null_slice \
                    > args.max_adm_drift:
                quality_ok = False
                reasons.append(
                    "EF initial deposition ADM mismatch is {:.6g}".format(
                        initial_deposition_adm_relative_error_from_null_slice
                    )
                )

    history_path, history = load_history(run_dir)
    end_time = high.get("end_time")
    final_time = history[-1]["time"]
    completed = finite_number(end_time) and final_time >= (
        1.0 - args.end_time_tolerance
    ) * float(end_time)
    if not completed and high.get("horizon_formed") is not True:
        reasons.append("configured end_time was not reached")

    minimum_late_samples = max(2, int(args.minimum_late_samples))
    duration = final_time - history[0]["time"]
    # Define the late interval from the actual completed history span, then require enough independent saved samples for trends.
    cutoff = final_time - args.late_window_fraction * max(duration, 0.0)
    late = [row for row in history if row["time"] >= cutoff]
    late_series_available = len(late) >= minimum_late_samples
    if not late_series_available and high.get("horizon_formed") is not True:
        reasons.append(
            "late-time window has {} samples; need {}".format(
                len(late), minimum_late_samples
            )
        )

    horizon = high.get("horizon_formed") is True
    classification = "undetermined"
    evidence = {
        "history": history_path,
        "history_samples": len(history),
        "late_samples": len(late),
        "completed_end_time": completed,
        "late_window_fraction": args.late_window_fraction,
        "fixed_family_binding_diagnostics_available": (
            binding_diagnostics_available
        ),
        "fixed_family_binding_fraction": fixed_family_binding_fraction,
        "initial_energy_at_infinity_proxy": energy_proxy,
        "virial_ratio_interpretation": (
            "Newtonian 2K/abs(W) proxy, not a relativistic virial identity"
        ),
    }

    trapped_time = high.get("first_trapped_time", high.get("horizon_time"))
    maximum_history_compactness = max(row["max_compactness"] for row in history)
    black_hole_series_ok = (
        maximum_history_compactness >= args.black_hole_compactness_min
    )
    evidence["maximum_history_compactness"] = maximum_history_compactness
    evidence["black_hole_series_ok"] = black_hole_series_ok
    post_horizon_evidence = intrinsic_post_horizon_evidence(
        summary, high, history, args
    )
    evidence["post_horizon_evidence"] = post_horizon_evidence
    if quality_ok and horizon \
            and outcome == "black_hole" \
            and black_hole_series_ok \
            and post_horizon_evidence["status"] == "pass":
        classification = "black_hole_candidate"
    elif quality_ok and not horizon and completed and late_series_available:
        initial_r50 = as_float(high, "initial_r50")
        minimum_r50 = min(row["r50"] for row in history)
        if initial_r50 <= 0.0 or any(row["r50"] <= 0.0 for row in late):
            raise ValueError("r50 must remain positive for threshold classification")
        late_times = [row["time"] for row in late]
        late_r50 = [row["r50"] for row in late]
        late_r90 = [row["r90"] for row in late]
        late_virial = [row["virial_ratio"] for row in late]
        late_compactness = [row["max_compactness"] for row in late]
        # Use available relativistic anisotropic Jeans residuals to assess static stress balance over the same late interval.
        late_jeans_pairs = [
            (row["time"], row["anisotropic_jeans_relative_l1"])
            for row in late
            if row["anisotropic_jeans_relative_l1"] >= 0.0
        ]
        late_jeans = [value for _time, value in late_jeans_pairs]
        late_r50_median = percentile(late_r50, 0.5)
        late_r50_spread = (
            percentile(late_r50, 0.9) - percentile(late_r50, 0.1)
        ) / max(late_r50_median, 1.0e-300)
        baseline = late_times[-1] - late_times[0]
        # Normalize the fitted radius change by its initial scale and actual late-time baseline.
        r50_dimensionless_trend = linear_slope(
            late_times, late_r50
        ) * baseline / initial_r50
        r90_dimensionless_trend = linear_slope(
            late_times, late_r90
        ) * baseline / max(history[0]["r90"], 1.0e-300)
        compactness_absolute_spread = (
            percentile(late_compactness, 0.9)
            - percentile(late_compactness, 0.1)
        )
        compactness_absolute_trend = linear_slope(
            late_times, late_compactness
        ) * baseline
        minimum_late_jeans_samples = max(
            3,
            int(math.ceil(
                getattr(args, "minimum_late_jeans_fraction", 0.75)
                * len(late)
            )),
        )
        late_jeans_available = (
            len(late_jeans) >= minimum_late_jeans_samples
        )
        late_jeans_median = (
            percentile(late_jeans, 0.5) if late_jeans else None
        )
        late_jeans_p90 = (
            percentile(late_jeans, 0.9) if late_jeans else None
        )
        late_jeans_absolute_trend = (
            linear_slope(
                [time_value for time_value, _value in late_jeans_pairs],
                late_jeans,
            ) * (
                late_jeans_pairs[-1][0] - late_jeans_pairs[0][0]
            )
            if len(late_jeans_pairs) >= 2 else None
        )
        maximum_jeans_residual = getattr(
            args, "halo_max_anisotropic_jeans_relative_l1", 0.25
        )
        maximum_jeans_increase = getattr(
            args, "halo_max_anisotropic_jeans_late_increase", 0.05
        )
        virial_in_band = sum(
            args.late_virial_min <= value <= args.late_virial_max
            for value in late_virial
        ) / len(late_virial)
        late_virial_median = percentile(late_virial, 0.5)
        initial_rest_weight = high.get(
            "realized_initial_rest_weight", input_marker_mass
        )
        escaped_rest_mass = high.get("escaped_rest_mass", 0.0)
        if not finite_number(initial_rest_weight) or float(initial_rest_weight) <= 0.0:
            raise ValueError("realized initial rest weight must be positive")
        if not finite_number(escaped_rest_mass) or float(escaped_rest_mass) < 0.0:
            raise ValueError("escaped rest mass must be finite and nonnegative")
        escaped_fraction = float(escaped_rest_mass) / float(initial_rest_weight)
        # A halo requires sustained radial/metric behavior, resolved relativistic stress balance, positive family binding and a small escape budget.
        halo_checks = {
            "fixed_family_positive_binding_energy": binding_diagnostics_available
                and fixed_family_binding_fraction > 0.0,
            "driver_halo_flag": high.get("virialized_halo") is True
                and outcome == "virialized_halo",
            # A saved trace that has already reached the black-hole evidence
            # compactness is near trapped even if finite cadence or a
            # coordinate stopping tolerance has not set the horizon flag.
            # Fail closed instead of certifying such a state as a halo.
            "not_near_horizon_collapse": maximum_history_compactness
                < args.black_hole_compactness_min,
            "compression_occurred": minimum_r50
                < args.halo_compression_fraction * initial_r50,
            "late_radius_bounded": late_r50_median
                <= args.halo_max_late_r50_over_initial * initial_r50,
            "late_radius_nonsecular": abs(r50_dimensionless_trend)
                <= args.halo_max_dimensionless_r50_trend,
            "late_outer_profile_nonsecular": abs(r90_dimensionless_trend)
                <= getattr(
                    args, "halo_max_dimensionless_r90_trend", 0.25
                ),
            "late_radius_spread_bounded": late_r50_spread
                <= args.halo_max_r50_relative_spread,
            "late_metric_spread_bounded": compactness_absolute_spread
                <= getattr(
                    args, "halo_max_compactness_absolute_spread", 0.05
                ),
            "late_metric_nonsecular": abs(compactness_absolute_trend)
                <= getattr(
                    args, "halo_max_compactness_absolute_trend", 0.05
                ),
            "late_anisotropic_jeans_samples": late_jeans_available,
            "late_anisotropic_jeans_balance": late_jeans_available
                and late_jeans_median <= maximum_jeans_residual
                and late_jeans_p90 <= maximum_jeans_residual,
            "late_anisotropic_jeans_nonsecular": late_jeans_available
                and late_jeans_absolute_trend is not None
                and late_jeans_absolute_trend <= maximum_jeans_increase,
            "escape_small": escaped_fraction
                <= args.halo_max_escaped_fraction,
        }
        # Treat substantial escape or sustained expansion as its own physical outcome in a threshold family.
        dispersal_checks = {
            "large_escape": escaped_fraction
                >= args.dispersal_min_escaped_fraction,
            "sustained_radial_expansion": (
                late_r50_median
                    >= args.dispersal_min_late_r50_over_initial * initial_r50
                and r50_dimensionless_trend
                    >= args.dispersal_min_dimensionless_r50_trend
                and r90_dimensionless_trend > 0.0
            ),
        }
        evidence.update({
            "late_r50_median": late_r50_median,
            "late_r50_over_initial": late_r50_median / initial_r50,
            "late_r50_relative_spread": late_r50_spread,
            "late_r50_dimensionless_trend": r50_dimensionless_trend,
            "late_r90_dimensionless_trend": r90_dimensionless_trend,
            "late_compactness_absolute_spread": compactness_absolute_spread,
            "late_compactness_absolute_trend": compactness_absolute_trend,
            "late_anisotropic_jeans_samples": len(late_jeans),
            "minimum_late_anisotropic_jeans_samples": (
                minimum_late_jeans_samples
            ),
            "late_anisotropic_jeans_relative_l1_median": late_jeans_median,
            "late_anisotropic_jeans_relative_l1_p90": late_jeans_p90,
            "late_anisotropic_jeans_absolute_trend": (
                late_jeans_absolute_trend
            ),
            "halo_near_horizon_compactness": (
                args.black_hole_compactness_min
            ),
            "halo_max_anisotropic_jeans_relative_l1": (
                maximum_jeans_residual
            ),
            "halo_max_anisotropic_jeans_late_increase": (
                maximum_jeans_increase
            ),
            "anisotropic_jeans_interpretation": (
                "normalized L1 residual of p_r' + (rho+p_r)mu_r "
                "- (p_t_sum-2p_r)/r = 0 on resolved non-vacuum "
                "polar-areal cells; a necessary late static-balance "
                "diagnostic, not by itself proof of virialization"
            ),
            "late_virial_median": late_virial_median,
            "late_virial_in_band_fraction": virial_in_band,
            "late_virial_used_as_halo_gate": False,
            "escaped_rest_fraction": escaped_fraction,
            "halo_checks": halo_checks,
            "dispersal_checks": dispersal_checks,
        })
        if all(halo_checks.values()):
            classification = "halo"
        elif any(dispersal_checks.values()):
            classification = "dispersal"
        else:
            failed = sorted(key for key, value in halo_checks.items() if not value)
            reasons.append(
                "late-time halo evidence is incomplete: {}".format(",".join(failed))
            )
            if "fixed_family_positive_binding_energy" in failed:
                reasons.append(
                    "halo classification requires a present, internally "
                    "consistent positive fixed_family_binding_fraction; "
                    "the per-marker energy proxy cannot replace this gate"
                )
            if any(key.startswith("late_anisotropic_jeans") for key in failed):
                reasons.append(
                    "halo classification requires sustained resolved GR "
                    "anisotropic static-balance evidence; the Newtonian "
                    "2K/abs(W) proxy cannot replace this gate"
                )
    elif horizon and not black_hole_series_ok:
        reasons.append("horizon flag lacks compactness evidence in saved history")
    elif horizon and outcome not in {"black_hole", "black_hole_threshold_proxy"}:
        reasons.append("horizon flag is inconsistent with the reported outcome")
    elif horizon and post_horizon_evidence["status"] != "pass":
        reasons.append(
            "raw black-hole outcome lacks intrinsic post-horizon evidence: {}".format(
                ",".join(post_horizon_evidence["failed_checks"])
            )
        )
    if outcome not in RECOGNIZED_OUTCOMES:
        reasons.append("unrecognized outcome {!r}".format(outcome))

    record = {
        "run_dir": run_dir,
        "summary": summary_path,
        "coordinates": coordinate,
        "angular_momentum": angular_momentum,
        "specific_angular_momentum": angular_momentum,
        "angular_momentum_over_input_marker_mass": (
            angular_momentum / input_marker_mass
        ),
        "angular_momentum_over_initial_adm_mass": (
            angular_momentum / initial_adm_mass
        ),
        # Report the measured ADM denominator and prescribed family target
        # separately. Their agreement is assessed with the configured
        # fixed-ADM tolerance.
        "angular_momentum_over_measured_initial_adm_mass": (
            angular_momentum / initial_adm_mass
        ),
        "angular_momentum_over_fixed_family_adm_mass": (
            angular_momentum / fixed_family_adm_mass
            if fixed_family_adm_mass is not None else None
        ),
        "angular_momentum_over_target_initial_adm_mass": (
            angular_momentum / target_initial_adm_mass
            if target_initial_adm_mass is not None else None
        ),
        "input_marker_mass": input_marker_mass,
        "initial_adm_mass": initial_adm_mass,
        "initial_deposited_adm_mass": initial_deposited_adm_mass,
        "null_slice_conserved_adm_mass": null_slice_conserved_adm_mass,
        "fixed_family_adm_mass": fixed_family_adm_mass,
        "fixed_family_rest_weight": fixed_family_rest_weight,
        "fixed_family_binding_energy": fixed_family_binding_energy,
        "fixed_family_binding_fraction": fixed_family_binding_fraction,
        "fixed_family_binding_diagnostics_available": (
            binding_diagnostics_available
        ),
        "fixed_family_positive_binding_energy": (
            binding_diagnostics_available
            and fixed_family_binding_fraction > 0.0
        ),
        "initial_energy_at_infinity_proxy": energy_proxy,
        "initial_energy_at_infinity_proxy_minimum": energy_proxy[
            "initial_energy_at_infinity_proxy_minimum"
        ],
        "initial_energy_at_infinity_proxy_maximum": energy_proxy[
            "initial_energy_at_infinity_proxy_maximum"
        ],
        "initial_energy_at_infinity_proxy_rest_weighted_mean": energy_proxy[
            "initial_energy_at_infinity_proxy_rest_weighted_mean"
        ],
        "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one": (
            energy_proxy[
                "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one"
            ]
        ),
        "late_anisotropic_jeans_relative_l1_median": evidence.get(
            "late_anisotropic_jeans_relative_l1_median"
        ),
        "late_anisotropic_jeans_relative_l1_p90": evidence.get(
            "late_anisotropic_jeans_relative_l1_p90"
        ),
        "late_anisotropic_jeans_absolute_trend": evidence.get(
            "late_anisotropic_jeans_absolute_trend"
        ),
        "late_anisotropic_jeans_samples": evidence.get(
            "late_anisotropic_jeans_samples"
        ),
        "late_virial_median": evidence.get("late_virial_median"),
        "late_virial_in_band_fraction": evidence.get(
            "late_virial_in_band_fraction"
        ),
        "late_virial_used_as_halo_gate": evidence.get(
            "late_virial_used_as_halo_gate", False
        ),
        "target_initial_adm_mass": target_initial_adm_mass,
        "fixed_adm_relative_error": fixed_adm_relative_error,
        "coordinate_slice_adm_relative_error_from_target": (
            coordinate_slice_adm_relative_error
        ),
        "null_slice_adm_relative_drift_from_family": (
            null_slice_adm_relative_drift_from_family
        ),
        "initial_deposition_adm_relative_error_from_null_slice": (
            initial_deposition_adm_relative_error_from_null_slice
        ),
        "initial_rest_weight_scale": high.get("initial_rest_weight_scale"),
        "realized_initial_rest_weight": high.get("realized_initial_rest_weight"),
        "apparent_horizon_radius": (
            float(high["apparent_horizon_radius"])
            if finite_number(high.get("apparent_horizon_radius")) else None
        ),
        "ef_post_horizon_evidence": (
            high.get("ef_post_horizon_evidence")
            if isinstance(high.get("ef_post_horizon_evidence"), dict) else {}
        ),
        "post_horizon_intrinsic_status": post_horizon_evidence["status"],
        "post_horizon_duration_over_fixed_family_adm_mass": (
            post_horizon_evidence["measurements"][
                "post_horizon_duration_over_fixed_family_adm_mass"
            ]
        ),
        "strictly_post_horizon_saved_samples": (
            post_horizon_evidence["measurements"][
                "strictly_post_horizon_saved_samples"
            ]
        ),
        "minimum_accepted_step_p_r": post_horizon_evidence[
            "measurements"]["minimum_accepted_step_p_r"],
        "minimum_accepted_step_b": post_horizon_evidence[
            "measurements"]["minimum_accepted_step_b"],
        "angular_momentum_scale": high.get("angular_momentum_scale"),
        "fixed_specific_angular_momentum": high.get("fixed_specific_angular_momentum"),
        "base_ic_fingerprint": high.get("base_ic_fingerprint"),
        "realized_ic_fingerprint": high.get("realized_ic_fingerprint"),
        "coordinate_map_manifest_sha256": high.get(
            "coordinate_map_manifest_sha256"
        ),
        "null_map_preflight_path": (
            null_map_preflight["path"] if null_map_preflight else None
        ),
        "null_map_preflight_sha256": (
            null_map_preflight["sha256"] if null_map_preflight else None
        ),
        "physical_ic_sha256": parse_sha256_manifest(
            os.path.join(run_dir, "physical_ic.sha256")
        ),
        "outcome": outcome,
        "classification": classification,
        "evidence": evidence,
        "reasons": reasons,
    }
    # Only a passing independent excision sensitivity comparison upgrades intrinsic horizon evidence to the certified endpoint category.
    if classification == "black_hole_candidate" \
            and check_excision_sensitivity:
        sensitivity = excision_sensitivity_gate(summary_path, record, args)
        evidence["excision_sensitivity"] = sensitivity
        record["excision_sensitivity"] = sensitivity
        record["excision_sensitivity_status"] = sensitivity["status"]
        if sensitivity["certifies_black_hole"]:
            record["classification"] = "black_hole_certified"
        else:
            reasons.append(
                "intrinsic EF evidence establishes a black-hole candidate, "
                "not a certified endpoint: {}".format(
                    ",".join(sensitivity["failed_checks"])
                )
            )
    else:
        sensitivity = {
            "schema": "venus_wpic_excision_sensitivity_gate_v1",
            "status": (
                "not_evaluated_for_repeat"
                if not check_excision_sensitivity else "not_applicable"
            ),
            "certifies_black_hole": False,
            "failed_checks": [],
        }
        evidence["excision_sensitivity"] = sensitivity
        record["excision_sensitivity"] = sensitivity
        record["excision_sensitivity_status"] = sensitivity["status"]
    return record


# Authenticate coordinate evidence first, classify all available runs, then test the sampled family's ordering and completeness.
def classify_threshold(args):
    try:
        certificate_gate = parity_certificate_gate(args)
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        eprint("threshold parity-certificate gate failed: {}".format(error))
        return 1
    summary_paths = discover_summary_paths(args.campaign_root, args.run_dir)
    if not summary_paths:
        eprint("threshold analysis found no summary.json files")
        return 1
    records = []
    load_errors = missing_receipt_outputs(args.campaign_root)
    for path in summary_paths:
        try:
            record = classify_threshold_run(path, args)
            if record is not None:
                records.append(record)
        except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
            load_errors.append({"summary": path, "error": str(error)})
    records.sort(key=lambda value: (value["angular_momentum"], value["run_dir"]))

    black_holes = [
        record for record in records
        if record["classification"] == "black_hole_certified"
    ]
    black_hole_candidates = [
        record for record in records
        if record["classification"] == "black_hole_candidate"
    ]
    halos = [record for record in records if record["classification"] == "halo"]
    dispersals = [record for record in records if record["classification"] == "dispersal"]
    contradictions = []
    # Check whether sampled outcomes contradict the assumed angular-momentum ordering before constructing a two-endpoint bracket.
    for halo in halos:
        for black_hole in black_holes:
            if halo["angular_momentum"] > black_hole["angular_momentum"]:
                continue
            contradictions.append({
                "lower_L_halo": halo["run_dir"],
                "lower_L": halo["angular_momentum"],
                "higher_L_black_hole": black_hole["run_dir"],
                "higher_L": black_hole["angular_momentum"],
            })

    bracket = None
    status = "unbracketed"
    # Use the largest certified black-hole angular momentum and smallest qualifying halo angular momentum as the sampled bounds.
    if black_holes and halos and not contradictions:
        lower = max(black_holes, key=lambda value: value["angular_momentum"])
        upper = min(halos, key=lambda value: value["angular_momentum"])
        if lower["angular_momentum"] < upper["angular_momentum"]:
            bracket = {
                "black_hole_upper_L": lower["angular_momentum"],
                "black_hole_upper_L_over_input_marker_mass": (
                    lower["angular_momentum_over_input_marker_mass"]
                ),
                "black_hole_upper_L_over_initial_adm_mass": (
                    lower["angular_momentum_over_initial_adm_mass"]
                ),
                "black_hole_upper_L_over_measured_initial_adm_mass": (
                    lower["angular_momentum_over_measured_initial_adm_mass"]
                ),
                "black_hole_upper_L_over_fixed_family_adm_mass": (
                    lower["angular_momentum_over_fixed_family_adm_mass"]
                ),
                "black_hole_upper_L_over_target_initial_adm_mass": (
                    lower["angular_momentum_over_target_initial_adm_mass"]
                ),
                "black_hole_run": lower["run_dir"],
                "halo_lower_L": upper["angular_momentum"],
                "halo_lower_L_over_input_marker_mass": (
                    upper["angular_momentum_over_input_marker_mass"]
                ),
                "halo_lower_L_over_initial_adm_mass": (
                    upper["angular_momentum_over_initial_adm_mass"]
                ),
                "halo_lower_L_over_measured_initial_adm_mass": (
                    upper["angular_momentum_over_measured_initial_adm_mass"]
                ),
                "halo_lower_L_over_fixed_family_adm_mass": (
                    upper["angular_momentum_over_fixed_family_adm_mass"]
                ),
                "halo_lower_L_over_target_initial_adm_mass": (
                    upper["angular_momentum_over_target_initial_adm_mass"]
                ),
                "halo_run": upper["run_dir"],
                "width": upper["angular_momentum"] - lower["angular_momentum"],
            }
            status = "bracketed"
    if contradictions:
        status = "nonmonotonic_inconclusive"
    # A sampled third asymptotic outcome invalidates the binary premise
    # "black hole below one L, bound halo above it" even if a BH/halo endpoint
    # pair can be written down.  Preserve that candidate pair in `bracket` for
    # diagnosis, but never certify it as the requested two-outcome threshold.
    # A third asymptotic outcome prevents certification of a purely black-hole/halo threshold premise.
    if dispersals and not contradictions:
        status = "inconclusive_due_to_dispersal"
    if black_hole_candidates and halos and not black_holes \
            and not contradictions and not dispersals:
        status = "inconclusive_due_to_uncertified_black_hole_candidate"
    if load_errors and status == "bracketed":
        status = "inconclusive_due_to_unreadable_runs"
    base_fingerprints = {
        record["base_ic_fingerprint"] for record in records
        if record["base_ic_fingerprint"] is not None
    }
    physical_ic_hashes = {
        record["physical_ic_sha256"] for record in records
        if record["physical_ic_sha256"] is not None
    }
    missing_provenance = [
        record["run_dir"] for record in records
        if record["base_ic_fingerprint"] is None
        or record["physical_ic_sha256"] is None
    ]
    provenance_conflict = (
        len(base_fingerprints) > 1 or len(physical_ic_hashes) > 1
    )
    provenance_incomplete = (
        len(base_fingerprints) != 1 or len(physical_ic_hashes) != 1
        or bool(missing_provenance)
    )
    if status == "bracketed" and (
            provenance_conflict
            or (provenance_incomplete and not args.allow_missing_provenance)):
        status = "inconclusive_due_to_ic_provenance"

    missing_map_preflight = [
        record["run_dir"] for record in records
        if record["coordinates"] == "eddington_finkelstein"
        and (record["null_map_preflight_path"] is None
             or normalized_sha256(record["null_map_preflight_sha256"]) is None)
    ]
    if missing_map_preflight:
        status = "inconclusive_due_to_map_provenance"

    fixed_targets = [
        record["target_initial_adm_mass"] for record in records
        if record["target_initial_adm_mass"] is not None
    ]
    fixed_realized_errors = [
        record["fixed_adm_relative_error"] for record in records
        if record["fixed_adm_relative_error"] is not None
    ]
    coordinate_slice_errors = [
        record["coordinate_slice_adm_relative_error_from_target"]
        for record in records
        if record["coordinate_slice_adm_relative_error_from_target"] is not None
    ]
    null_slice_drifts = [
        record["null_slice_adm_relative_drift_from_family"]
        for record in records
        if record["null_slice_adm_relative_drift_from_family"] is not None
    ]
    deposition_errors = [
        record["initial_deposition_adm_relative_error_from_null_slice"]
        for record in records
        if record[
            "initial_deposition_adm_relative_error_from_null_slice"
        ] is not None
    ]
    # Require the prescribed ADM target and realized family errors to be consistent across every reported member.
    fixed_adm_family_ok = bool(records) and len(fixed_targets) == len(records)
    fixed_adm_family_ok = fixed_adm_family_ok \
        and len(fixed_realized_errors) == len(records)
    fixed_adm_reference = percentile(fixed_targets, 0.5) if fixed_targets else None
    fixed_adm_max_relative_spread = None
    fixed_adm_max_realized_relative_error = (
        max(fixed_realized_errors) if fixed_realized_errors else None
    )
    if fixed_targets:
        fixed_adm_max_relative_spread = max(
            abs(value - fixed_adm_reference)
            / max(fixed_adm_reference, 1.0e-300)
            for value in fixed_targets
        )
        fixed_adm_family_ok = fixed_adm_family_ok and (
            fixed_adm_max_relative_spread <= args.fixed_adm_relative_tolerance
        )
    if fixed_adm_max_realized_relative_error is not None:
        fixed_adm_family_ok = fixed_adm_family_ok and (
            fixed_adm_max_realized_relative_error
                <= args.fixed_adm_relative_tolerance
        )
    if status == "bracketed" and not fixed_adm_family_ok \
            and not args.allow_unfixed_adm_family:
        status = "inconclusive_due_to_nonuniform_initial_adm"

    result = {
        "status": status,
        "parity_certificate": certificate_gate,
        "coordinates_filter": args.coordinates,
        "angular_statistic": args.angular_statistic,
        "threshold_interpretation": (
            "only status=bracketed establishes a sampled fixed-initial-ADM "
            "black-hole/halo bracket; its black-hole endpoint must pass both "
            "the intrinsic post-horizon gate and authenticated independent "
            "excision-threshold/buffer repeats. Candidates never close the "
            "bracket, dispersal invalidates the binary threshold premise, "
            "and no interpolation is claimed"
        ),
        "bracket": bracket,
        "bracket_is_certified": status == "bracketed",
        "counts": {
            "black_hole_certified": len(black_holes),
            "black_hole_candidate": len(black_hole_candidates),
            "black_hole": len(black_holes),
            "halo": len(halos),
            "dispersal": len(dispersals),
            "undetermined": sum(record["classification"] == "undetermined" for record in records),
        },
        "fixed_initial_adm_family": {
            "required": not args.allow_unfixed_adm_family,
            "status": "pass" if fixed_adm_family_ok else "fail",
            "reference_target": fixed_adm_reference,
            "maximum_relative_target_spread": fixed_adm_max_relative_spread,
            "maximum_realized_relative_error": (
                fixed_adm_max_realized_relative_error
            ),
            "maximum_coordinate_slice_relative_error_from_target": (
                max(coordinate_slice_errors) if coordinate_slice_errors else None
            ),
            "maximum_null_slice_relative_drift_from_family": (
                max(null_slice_drifts) if null_slice_drifts else None
            ),
            "maximum_initial_deposition_relative_error_from_null_slice": (
                max(deposition_errors) if deposition_errors else None
            ),
            "slice_mismatches_are_convergence_diagnostics": True,
            "relative_tolerance": args.fixed_adm_relative_tolerance,
        },
        "contradictions": contradictions,
        "ic_provenance": {
            "base_ic_fingerprints": sorted(base_fingerprints),
            "physical_ic_sha256": sorted(physical_ic_hashes),
            "missing_runs": missing_provenance,
        },
        "null_map_preflight_authentication": {
            "required_for_mapped_ef": True,
            "status": "pass" if not missing_map_preflight else "fail",
            "missing_or_failed_runs": missing_map_preflight,
            "authenticated_sha256": sorted({
                record["null_map_preflight_sha256"] for record in records
                if record["null_map_preflight_sha256"] is not None
            }),
        },
        "load_errors": load_errors,
        "runs": records,
    }
    emit_json(result, args.output)
    if args.csv_output:
        threshold_csv(records, args.csv_output)
    return 0 if status == "bracketed" else SCIENTIFIC_INCONCLUSIVE


def threshold_csv(records, path):
    fields = [
        "angular_momentum", "specific_angular_momentum",
        "angular_momentum_over_input_marker_mass",
        "angular_momentum_over_initial_adm_mass",
        "angular_momentum_over_measured_initial_adm_mass",
        "angular_momentum_over_fixed_family_adm_mass",
        "angular_momentum_over_target_initial_adm_mass",
        "input_marker_mass", "initial_adm_mass",
        "initial_deposited_adm_mass", "null_slice_conserved_adm_mass",
        "fixed_family_adm_mass", "fixed_family_rest_weight",
        "fixed_family_binding_energy", "fixed_family_binding_fraction",
        "fixed_family_binding_diagnostics_available",
        "fixed_family_positive_binding_energy",
        "initial_energy_at_infinity_proxy_minimum",
        "initial_energy_at_infinity_proxy_maximum",
        "initial_energy_at_infinity_proxy_rest_weighted_mean",
        "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one",
        "late_anisotropic_jeans_relative_l1_median",
        "late_anisotropic_jeans_relative_l1_p90",
        "late_anisotropic_jeans_absolute_trend",
        "late_anisotropic_jeans_samples",
        "late_virial_median", "late_virial_in_band_fraction",
        "late_virial_used_as_halo_gate",
        "target_initial_adm_mass",
        "fixed_adm_relative_error",
        "coordinate_slice_adm_relative_error_from_target",
        "null_slice_adm_relative_drift_from_family",
        "initial_deposition_adm_relative_error_from_null_slice",
        "initial_rest_weight_scale", "realized_initial_rest_weight",
        "post_horizon_intrinsic_status",
        "post_horizon_duration_over_fixed_family_adm_mass",
        "strictly_post_horizon_saved_samples",
        "minimum_accepted_step_p_r", "minimum_accepted_step_b",
        "excision_sensitivity_status",
        "classification", "coordinates", "outcome",
        "angular_momentum_scale", "fixed_specific_angular_momentum",
        "base_ic_fingerprint", "realized_ic_fingerprint",
        "coordinate_map_manifest_sha256", "physical_ic_sha256",
        "null_map_preflight_path", "null_map_preflight_sha256",
        "run_dir", "summary", "reasons",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in records:
            row = dict(source)
            row["reasons"] = ";".join(row.get("reasons", []))
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate-run", help="validate one completed run")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument(
        "--expected", required=True,
        choices=("any", "halo", "black_hole", "initialized"),
    )
    validate.add_argument("--max-adm-drift", type=float, default=0.01)
    validate.add_argument("--max-rest-mass-error", type=float, default=1.0e-12)
    validate.add_argument("--end-time-tolerance", type=float, default=1.0e-6)
    validate.set_defaults(function=validate_run)

    certificate = subparsers.add_parser(
        "parity-certificate-gate",
        help="authenticate one strict solved-eikonal parity certificate",
    )
    certificate.add_argument("--parity-certificate", required=True)
    certificate.add_argument(
        "--parity-certificate-sha256",
        help="expected digest; otherwise PATH.sha256 is required",
    )
    certificate.add_argument("--output", required=True)
    certificate.set_defaults(function=validate_parity_certificate)

    parity = subparsers.add_parser(
        "parity", help="compare a matched Schwarzschild/EF marker family at compactness milestones"
    )
    parity.add_argument("--schwarzschild", required=True)
    parity.add_argument("--ef", required=True)
    parity.add_argument("--milestones", default="0.50,0.60,0.70,0.80,0.90,0.95,0.98")
    parity.add_argument("--mass-relative-tolerance", type=float, default=0.02)
    parity.add_argument("--relative-tolerance", type=float, default=0.05)
    parity.add_argument("--virial-absolute-tolerance", type=float, default=0.10)
    parity.add_argument("--minimum-milestones", type=int, default=3)
    parity.add_argument("--allow-missing-provenance", action="store_true")
    parity.add_argument("--output")
    parity.add_argument("--csv-output")
    parity.set_defaults(function=compare_parity)

    threshold = subparsers.add_parser(
        "threshold", help="classify a coarse scan and report a conservative L bracket"
    )
    threshold.add_argument("--campaign-root")
    threshold.add_argument("--run-dir", action="append")
    threshold.add_argument(
        "--parity-certificate", required=True,
        help=(
            "strict venus_wpic_dynamic_coordinate_certificate_v2 status=pass "
            "artifact with authenticated convergence evidence required before "
            "threshold classification"
        ),
    )
    threshold.add_argument(
        "--parity-certificate-sha256",
        help="expected certificate SHA-256; otherwise PATH.sha256 is required",
    )
    threshold.add_argument(
        "--coordinates", default="eddington_finkelstein",
        choices=("eddington_finkelstein", "schwarzschild", "any"),
    )
    threshold.add_argument(
        "--angular-statistic", default="mass_weighted_rms_specific_angular_momentum",
        choices=(
            "mass_weighted_mean_specific_angular_momentum",
            "mass_weighted_rms_specific_angular_momentum",
            "maximum_specific_angular_momentum",
        ),
    )
    threshold.add_argument("--max-adm-drift", type=float, default=0.01)
    threshold.add_argument("--max-rest-mass-error", type=float, default=1.0e-12)
    threshold.add_argument("--end-time-tolerance", type=float, default=1.0e-6)
    threshold.add_argument("--fixed-adm-relative-tolerance", type=float, default=1.0e-8)
    threshold.add_argument("--allow-unfixed-adm-family", action="store_true")
    threshold.add_argument("--black-hole-compactness-min", type=float, default=0.99)
    threshold.add_argument(
        "--minimum-post-horizon-mass-times", type=float, default=5.0,
        help=(
            "minimum (v_final-v_AH)/M_fixed for an intrinsic EF black-hole "
            "candidate; values below the documented floor 5 are rejected"
        ),
    )
    threshold.add_argument(
        "--minimum-post-horizon-saved-samples", type=int, default=3,
        help=(
            "minimum number of saved samples strictly after v_AH; values "
            "below the documented floor 3 are rejected"
        ),
    )
    threshold.add_argument(
        "--excision-sensitivity-certificate",
        help=(
            "authenticated venus_wpic_excision_sensitivity_certificate_v1 "
            "manifest; without it an intrinsic pass remains only a "
            "black_hole_candidate and cannot close the threshold bracket"
        ),
    )
    threshold.add_argument(
        "--excision-sensitivity-certificate-sha256",
        help=(
            "expected sensitivity-certificate SHA-256; otherwise the exact "
            "PATH.sha256 sidecar is required"
        ),
    )
    threshold.add_argument(
        "--excision-sensitivity-max-horizon-time-spread-over-adm",
        type=float, default=1.0,
        help=(
            "maximum (max(v_AH)-min(v_AH))/M_fixed across independent "
            "excision repeats; values above 1 are rejected"
        ),
    )
    threshold.add_argument(
        "--excision-sensitivity-max-horizon-radius-relative-spread",
        type=float, default=0.05,
        help=(
            "maximum relative apparent-horizon-radius spread across "
            "independent excision repeats; values above 0.05 are rejected"
        ),
    )
    threshold.add_argument("--late-window-fraction", type=float, default=0.30)
    threshold.add_argument("--minimum-late-samples", type=int, default=5)
    threshold.add_argument(
        "--late-virial-min", type=float, default=0.50,
        help="lower report-only band for the Newtonian 2K/abs(W) proxy",
    )
    threshold.add_argument(
        "--late-virial-max", type=float, default=1.50,
        help="upper report-only band for the Newtonian 2K/abs(W) proxy",
    )
    threshold.add_argument(
        "--minimum-late-virial-fraction", type=float, default=0.75,
        help=(
            "report-only reference fraction for 2K/abs(W); retained for "
            "output compatibility and never used as a halo gate"
        ),
    )
    threshold.add_argument("--halo-compression-fraction", type=float, default=0.95)
    threshold.add_argument("--halo-max-late-r50-over-initial", type=float, default=1.50)
    threshold.add_argument("--halo-max-r50-relative-spread", type=float, default=0.75)
    threshold.add_argument("--halo-max-dimensionless-r50-trend", type=float, default=0.25)
    threshold.add_argument("--halo-max-dimensionless-r90-trend", type=float, default=0.25)
    threshold.add_argument(
        "--halo-max-compactness-absolute-spread", type=float, default=0.05,
    )
    threshold.add_argument(
        "--halo-max-compactness-absolute-trend", type=float, default=0.05,
    )
    threshold.add_argument(
        "--halo-max-anisotropic-jeans-relative-l1", type=float, default=0.25,
        help=(
            "maximum late median and p90 normalized GR anisotropic "
            "static-balance residual"
        ),
    )
    threshold.add_argument(
        "--halo-max-anisotropic-jeans-late-increase", type=float, default=0.05,
        help=(
            "maximum fitted end-to-end increase of the late normalized GR "
            "anisotropic static-balance residual"
        ),
    )
    threshold.add_argument(
        "--minimum-late-jeans-fraction", type=float, default=0.75,
        help="minimum fraction of late outputs with an available GR balance residual",
    )
    threshold.add_argument("--halo-max-escaped-fraction", type=float, default=0.05)
    threshold.add_argument("--dispersal-min-escaped-fraction", type=float, default=0.20)
    threshold.add_argument("--dispersal-min-late-r50-over-initial", type=float, default=1.75)
    threshold.add_argument("--dispersal-min-dimensionless-r50-trend", type=float, default=0.20)
    threshold.add_argument("--allow-missing-provenance", action="store_true")
    threshold.add_argument("--output")
    threshold.add_argument("--csv-output")
    threshold.set_defaults(function=classify_threshold)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")
    if args.command == "threshold" and not args.campaign_root and not args.run_dir:
        parser.error("threshold requires --campaign-root or at least one --run-dir")
    if args.command == "threshold":
        if args.excision_sensitivity_certificate_sha256 \
                and not args.excision_sensitivity_certificate:
            parser.error(
                "--excision-sensitivity-certificate-sha256 requires "
                "--excision-sensitivity-certificate"
            )
        # Validate classifier controls before dispatch, including the documented floors on post-horizon evidence and sensitivity criteria.
        positive = (
            "fixed_adm_relative_tolerance", "black_hole_compactness_min",
            "minimum_post_horizon_mass_times",
            "excision_sensitivity_max_horizon_time_spread_over_adm",
            "excision_sensitivity_max_horizon_radius_relative_spread",
            "late_window_fraction", "late_virial_max",
            "minimum_late_virial_fraction", "halo_compression_fraction",
            "halo_max_late_r50_over_initial", "halo_max_r50_relative_spread",
            "halo_max_dimensionless_r50_trend",
            "halo_max_dimensionless_r90_trend",
            "halo_max_compactness_absolute_spread",
            "halo_max_compactness_absolute_trend",
            "halo_max_anisotropic_jeans_relative_l1",
            "halo_max_anisotropic_jeans_late_increase",
            "minimum_late_jeans_fraction", "halo_max_escaped_fraction",
            "dispersal_min_escaped_fraction",
            "dispersal_min_late_r50_over_initial",
            "dispersal_min_dimensionless_r50_trend",
        )
        if any(not finite_number(getattr(args, key)) or getattr(args, key) <= 0.0
               for key in positive):
            parser.error("threshold classifier tolerances must be finite and positive")
        if not (0.0 < args.late_window_fraction <= 1.0) \
                or not (0.0 < args.minimum_late_virial_fraction <= 1.0) \
                or not (0.0 < args.minimum_late_jeans_fraction <= 1.0) \
                or args.minimum_late_samples < 2 \
                or args.minimum_post_horizon_mass_times < 5.0 \
                or args.minimum_post_horizon_saved_samples < 3 \
                or args.excision_sensitivity_max_horizon_time_spread_over_adm \
                    > 1.0 \
                or args.excision_sensitivity_max_horizon_radius_relative_spread \
                    > 0.05 \
                or not finite_number(args.late_virial_min) \
                or args.late_virial_min >= args.late_virial_max:
            parser.error("invalid late-time threshold classifier window/bands")
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
