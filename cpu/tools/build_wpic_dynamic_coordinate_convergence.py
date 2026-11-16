#!/usr/bin/env python3
# Build uncertainty bounds from coordinate-comparison diagnostics at two resolutions and two independent quiet realizations.
# Inputs must share continuum physics, numerical source and executable while strictly refining radial/time/particle resolution.
# Output derives a conservative error envelope for fine-A from measured errors and realization spread.

"""Build fail-closed convergence evidence for SC/EF coordinate parity.

The four required inputs are *diagnostic* v2 coordinate audits: each must be
scientifically clean and inconclusive only because it has no convergence
evidence of its own.  Two resolutions and two distinct deterministic PIC
realizations are the minimum accepted matrix.  This tool authenticates that
matrix, checks refinement and realization provenance, requires non-increasing
coordinate-disagreement errors, and derives (rather than accepts) conservative
uncertainty bounds for the designated fine-A production pair.
"""

import argparse
import hashlib
import json
import math
import os
import sys


SCHEMA = "venus_wpic_dynamic_coordinate_convergence_v1"
AUDIT_SCHEMA = "venus_wpic_dynamic_coordinate_certificate_v2"
MISSING_CONVERGENCE_ISSUE = (
    "single-resolution agreement has no independent quantified "
    "discretization/PIC convergence evidence"
)
CATEGORIES = ("metric", "stress", "moment", "distribution")
PRODUCTION_METRIC_INTERPOLATION = {
    "schwarzschild": (
        "production_center_face_lambda_monotone_hermite_mu"
    ),
    "eddington_finkelstein": "production_center_face_linear_a_b",
    "donor_history_format_version": 2,
    "production_equivalent": True,
}
STATIC_SHELL_PHYSICS_KEYS = (
    "ansatz",
    "configured_amplitude_C",
    "configured_ell0",
    "configured_central_y",
    "configured_cutoff_power_k",
    "unscaled_adm_mass",
    "continuum_rescaled_to_target_adm_mass",
    "realized_amplitude_C",
    "realized_ell0",
    "realized_cutoff_power_k",
    "cutoff_energy_E0",
    "inner_matter_radius",
    "outer_matter_radius",
    "continuum_adm_mass",
    "continuum_rest_weight",
    "continuum_binding_energy",
    "continuum_binding_fraction",
    "central_lapse",
    "sampling_rule",
    "marker_weight_measure",
    "post_sampling_canonicalization_passes",
    "configured_inward_pair_mean_kick",
    "configured_kick_core_radius",
)
PHYSICS_KEYS = (
    "initial_condition",
    "inner_boundary",
    "radial_grid",
    "r_min",
    "r_max",
    "end_time",
    "gap_stop",
    "total_marker_mass",
    "target_initial_adm_mass",
    "scale_radius",
    "initial_velocity_factor",
    "initial_radial_infall",
    "infall_core_radius",
    "radial_importance_exponent",
    "shell_center",
    "shell_width",
    "shell_radial_importance_factor",
    "shell_radial_momentum",
    "shell_radial_momentum_dispersion",
    "shell_tangential_velocity_dispersion",
    "fixed_specific_angular_momentum",
    "angular_momentum_scale",
    "timestep_omitted_mass_fraction",
)
PAIR_KEYS = PHYSICS_KEYS + (
    "markers",
    "radial_cells",
    "dt_max",
    "output_steps",
    "sampling_realization",
    "mass_weighted_mean_specific_angular_momentum",
    "mass_weighted_rms_specific_angular_momentum",
    "maximum_specific_angular_momentum",
)


class EvidenceError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_sha256(value):
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest):
        return None
    return digest


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value)) and float(value) >= 0.0


def load_json(path, label):
    try:
        with open(path, "r") as stream:
            value = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise EvidenceError("{} cannot be read: {}".format(label, error))
    if not isinstance(value, dict):
        raise EvidenceError("{} root is not an object".format(label))
    return value


def authenticate_sidecar(path, label):
    actual = sha256_file(path)
    sidecar = path + ".sha256"
    try:
        with open(sidecar, "r") as stream:
            fields = stream.readline().strip().split()
    except OSError as error:
        raise EvidenceError("{} SHA-256 sidecar is missing: {}".format(
            label, error))
    if len(fields) != 2:
        raise EvidenceError("{} SHA-256 sidecar is malformed".format(label))
    expected = normalized_sha256(fields[0])
    named = fields[1].lstrip("*")
    if expected != actual or named != os.path.basename(path):
        raise EvidenceError(
            "{} SHA-256 sidecar does not authenticate its basename".format(
                label))
    return actual


def high_summary(path, expected_coordinates, label):
    document = load_json(path, label)
    if document.get("coordinates") != expected_coordinates:
        raise EvidenceError("{} has wrong coordinates".format(label))
    high = document.get("high_compactness")
    if not isinstance(high, dict):
        raise EvidenceError("{} has no high_compactness object".format(label))
    if high.get("coordinates") != expected_coordinates:
        raise EvidenceError("{} nested coordinates disagree".format(label))
    return high


# Compare recorded source and executable identities at launch, which remain meaningful after a working executable is rebuilt.
def load_run_provenance(summary_path, label):
    """Authenticate the immutable source/binary records staged with a run.

    ``source.sha256`` is itself a multi-entry sha256sum manifest.  Its content
    digest identifies the exact source capture without assuming that the
    checkout still exists.  ``binary.sha256`` names the executable used at
    launch; the executable may later be rebuilt, so its recorded digest—not
    the current path target—is the cross-run identity.
    """
    root = os.path.dirname(os.path.abspath(summary_path))
    source_path = os.path.join(root, "source.sha256")
    binary_path = os.path.join(root, "binary.sha256")
    try:
        with open(source_path, "r") as stream:
            source_lines = [line.rstrip("\n") for line in stream]
    except OSError as error:
        raise EvidenceError("{} source provenance is missing: {}".format(
            label, error))
    if not source_lines:
        raise EvidenceError("{} source provenance is empty".format(label))
    for line_number, line in enumerate(source_lines, start=1):
        fields = line.split(maxsplit=1)
        digest = normalized_sha256(fields[0]) if fields else None
        named = fields[1].lstrip("*") if len(fields) == 2 else ""
        if digest is None or not named or os.path.isabs(named):
            raise EvidenceError(
                "{} source provenance line {} is malformed".format(
                    label, line_number))

    try:
        with open(binary_path, "r") as stream:
            fields = stream.readline().strip().split(maxsplit=1)
    except OSError as error:
        raise EvidenceError("{} binary provenance is missing: {}".format(
            label, error))
    binary_digest = normalized_sha256(fields[0]) if fields else None
    binary_name = fields[1].lstrip("*") if len(fields) == 2 else ""
    if binary_digest is None or not os.path.isabs(binary_name):
        raise EvidenceError("{} binary provenance is malformed".format(label))
    return {
        "source_manifest_path": source_path,
        "source_manifest_sha256": sha256_file(source_path),
        "binary_record_path": binary_path,
        "binary_record_sha256": sha256_file(binary_path),
        "binary_sha256": binary_digest,
        "binary_path_at_launch": binary_name,
    }


def require_equal_fields(left, right, keys, label):
    missing = [key for key in keys if key not in left or key not in right]
    if missing:
        raise EvidenceError("{} lacks fields {}".format(label, missing))
    different = [key for key in keys if left[key] != right[key]]
    if different:
        raise EvidenceError("{} differs in fields {}".format(label, different))


# Separate the continuum distribution and invariant weight convention from resolution-dependent sampling errors.
def static_shell_physics(summary, label):
    """Return the resolution-independent static-shell physics signature.

    Sampling counts and quadrature errors deliberately remain outside this
    signature because they change under refinement.  The continuum ansatz,
    geometric normalization, invariant-weight convention, and fixed kick do
    not.  A certificate for ``energy_cutoff_shell`` is invalid unless every
    one of these fields is present and identical across coordinates,
    resolutions, and quiet-PIC realizations.
    """
    if summary.get("initial_condition") != "energy_cutoff_shell":
        return None
    shell = summary.get("energy_cutoff_shell")
    if not isinstance(shell, dict):
        raise EvidenceError("{} lacks energy_cutoff_shell physics".format(label))
    missing = [key for key in STATIC_SHELL_PHYSICS_KEYS if key not in shell]
    if missing:
        raise EvidenceError(
            "{} static-shell physics lacks fields {}".format(label, missing))
    return {key: shell[key] for key in STATIC_SHELL_PHYSICS_KEYS}


def require_equal_static_shell_physics(left, right, label):
    left_signature = static_shell_physics(left, label + " left")
    right_signature = static_shell_physics(right, label + " right")
    if left_signature != right_signature:
        raise EvidenceError("{} differs in static-shell physics".format(label))


def maximum_error(records, label):
    values = []
    for name, record in records:
        if not isinstance(record, dict):
            raise EvidenceError("{} field {} is malformed".format(label, name))
        for norm in ("relative_l2", "relative_linf"):
            value = record.get(norm)
            if not finite_nonnegative(value):
                raise EvidenceError(
                    "{} field {} has invalid {}".format(label, name, norm))
            values.append(float(value))
    if not values:
        raise EvidenceError("{} has no comparable fields".format(label))
    return max(values)


def category_errors(audit, label):
    errors = audit.get("errors")
    if not isinstance(errors, dict):
        raise EvidenceError("{} has no transformed-field errors".format(label))
    metric = []
    stress = []
    moment = []
    # Classify transformed quantities into metric, stress and moment errors before taking category-wide maximum norms.
    for name, record in errors.items():
        if name.startswith("metric_") \
                or name.startswith("coordinate_jacobian_") \
                or name == "misner_sharp_mass":
            metric.append((name, record))
        elif name.startswith("moment_"):
            moment.append((name, record))
        elif name.startswith("stress_"):
            stress.append((name, record))
        else:
            raise EvidenceError(
                "{} has an unclassified error field {}".format(label, name))

    initial_slice = audit.get("initial_mapped_slice")
    initial_errors = initial_slice.get("errors") \
        if isinstance(initial_slice, dict) else None
    if not isinstance(initial_errors, dict) or not initial_errors:
        raise EvidenceError(
            "{} has no direct mapped-initial-slice errors".format(label))
    for name, record in initial_errors.items():
        if name.startswith("initial_metric_") \
                or name == "initial_misner_sharp_mass":
            metric.append((name, record))
        elif name.startswith("initial_stress_"):
            stress.append((name, record))
        else:
            raise EvidenceError(
                "{} has an unclassified initial-slice error field {}".format(
                    label, name))

    distribution = audit.get("distribution")
    if not isinstance(distribution, dict) \
            or distribution.get("mode") != "strict" \
            or not isinstance(distribution.get("evidence"), list) \
            or not distribution["evidence"] \
            or distribution.get("failed_fields") not in ([], None):
        raise EvidenceError("{} lacks strict passing distribution evidence".format(
            label))
    distribution_records = []
    for index, record in enumerate(distribution["evidence"]):
        if not isinstance(record, dict) \
                or record.get("failed_fields") not in ([], None) \
                or not isinstance(record.get("errors"), dict):
            raise EvidenceError(
                "{} distribution evidence {} is malformed".format(label, index))
        distribution_records.extend(record["errors"].items())
    return {
        "metric": maximum_error(metric, label + " metric"),
        "stress": maximum_error(stress, label + " stress"),
        "moment": maximum_error(moment, label + " moment"),
        "distribution": maximum_error(
            distribution_records, label + " distribution"),
    }


def validate_mapping(audit, label):
    mapping = audit.get("mapping")
    tolerances = audit.get("tolerances")
    if not isinstance(mapping, dict) or not isinstance(tolerances, dict):
        raise EvidenceError("{} lacks map errors/tolerances".format(label))
    checks = (
        ("inverse_maximum_step_refinement_error", "eikonal_step_refinement"),
        ("maximum_composition_time_error", "forward_inverse_composition"),
        ("maximum_composition_jacobian_error", "forward_inverse_composition"),
    )
    for error_name, tolerance_name in checks:
        error = mapping.get(error_name)
        tolerance = tolerances.get(tolerance_name)
        if not finite_nonnegative(error) or not finite_nonnegative(tolerance) \
                or float(error) > float(tolerance):
            raise EvidenceError(
                "{} fails mapping check {}".format(label, error_name))


# Accept a single-resolution audit only when all physical comparisons pass and independent convergence is its sole outstanding requirement.
def load_pair(path, label, expected_eikonal_substeps,
              expected_certifier_sha256):
    path = os.path.abspath(path)
    audit_digest = authenticate_sidecar(path, label)
    audit = load_json(path, label)
    if audit.get("schema") != AUDIT_SCHEMA:
        raise EvidenceError("{} has unsupported schema".format(label))
    if audit.get("status") != "inconclusive" \
            or audit.get("failed_fields") not in ([], None) \
            or audit.get("issues") != [MISSING_CONVERGENCE_ISSUE]:
        raise EvidenceError(
            "{} must be inconclusive only for missing convergence evidence".format(
                label))
    if not isinstance(audit.get("inverse_equation"), str) \
            or not audit["inverse_equation"].strip():
        raise EvidenceError("{} has no inverse eikonal equation".format(label))
    if audit.get("metric_interpolation") \
            != PRODUCTION_METRIC_INTERPOLATION:
        raise EvidenceError(
            "{} does not establish production-equivalent center/face metric "
            "reconstruction".format(label)
        )
    if audit.get("eikonal_substeps") != expected_eikonal_substeps:
        raise EvidenceError(
            "{} eikonal substeps do not match the campaign contract {}".
            format(label, expected_eikonal_substeps)
        )
    if audit.get("certifier_sha256") != expected_certifier_sha256:
        raise EvidenceError(
            "{} was produced by a different parity certifier".format(label)
        )
    validate_mapping(audit, label)

    provenance = audit.get("provenance")
    manifest = provenance.get("coordinate_map_manifest") \
        if isinstance(provenance, dict) else None
    if not isinstance(manifest, dict):
        raise EvidenceError("{} has no coordinate-map provenance".format(label))
    if manifest.get("schema") != "venus_v3_sc_to_ef_null_map" \
            or manifest.get("schema_version") != 2 \
            or manifest.get("donor_history_format_version") != 2 \
            or manifest.get("saved_metric_interpolation") != (
                "SC center-face lambda plus monotone-Hermite mu; "
                "EF center-face a,b"
            ):
        raise EvidenceError(
            "{} coordinate-map manifest lacks the face-aware v2 contract".
            format(label)
        )
    manifest_digest = normalized_sha256(manifest.get("actual_sha256"))
    if manifest_digest is None \
            or normalized_sha256(manifest.get(
                "schwarzschild_summary_sha256")) != manifest_digest \
            or normalized_sha256(manifest.get(
                "eddington_finkelstein_summary_sha256")) != manifest_digest:
        raise EvidenceError("{} manifest is not bound to both summaries".format(
            label))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EvidenceError("{} has no map artifacts".format(label))
    for name, record in artifacts.items():
        if not isinstance(record, dict) or record.get("verified") is not True \
                or normalized_sha256(record.get("expected_sha256")) is None \
                or normalized_sha256(record.get("expected_sha256")) \
                    != normalized_sha256(record.get("actual_sha256")):
            raise EvidenceError(
                "{} has unauthenticated map artifact {}".format(label, name))
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
        raise EvidenceError(
            "{} does not bind the face-aware v2 donor history to its "
            "authenticated manifest artifact".format(label)
        )

    sc_path = provenance.get("schwarzschild_summary")
    ef_path = provenance.get("eddington_finkelstein_summary")
    if not isinstance(sc_path, str) or not isinstance(ef_path, str):
        raise EvidenceError("{} does not name both summaries".format(label))
    sc_path = os.path.abspath(sc_path)
    ef_path = os.path.abspath(ef_path)
    sc = high_summary(sc_path, "schwarzschild", label + " SC summary")
    ef = high_summary(
        ef_path, "eddington_finkelstein", label + " EF summary")
    # Require both coordinates to use the same scientific source and executable before their numerical errors enter a refinement sequence.
    sc_run_provenance = load_run_provenance(
        sc_path, label + " SC summary")
    ef_run_provenance = load_run_provenance(
        ef_path, label + " EF summary")
    if sc_run_provenance["source_manifest_sha256"] \
            != ef_run_provenance["source_manifest_sha256"]:
        raise EvidenceError("{} SC/EF source captures differ".format(label))
    if sc_run_provenance["binary_sha256"] \
            != ef_run_provenance["binary_sha256"]:
        raise EvidenceError("{} SC/EF executable digests differ".format(label))
    require_equal_fields(sc, ef, PAIR_KEYS, label + " SC/EF pair")
    require_equal_static_shell_physics(
        sc, ef, label + " SC/EF pair")
    target_adm = sc.get("target_initial_adm_mass")
    sc_adm = sc.get("initial_adm_mass")
    ef_adm = ef.get("initial_adm_mass")
    if not finite_nonnegative(target_adm) or not float(target_adm) > 0.0 \
            or not finite_nonnegative(sc_adm) or not float(sc_adm) > 0.0 \
            or not finite_nonnegative(ef_adm) or not float(ef_adm) > 0.0 \
            or abs(float(sc_adm) - float(target_adm)) \
                > 1.0e-9 * float(target_adm):
        raise EvidenceError(
            "{} does not normalize its SC donor to the common target initial "
            "ADM mass".format(label))
    if sc.get("base_ic_fingerprint") != ef.get("base_ic_fingerprint"):
        raise EvidenceError("{} SC/EF base fingerprints differ".format(label))
    if sc.get("coordinate_map_manifest_sha256") != manifest_digest \
            or ef.get("coordinate_map_manifest_sha256") != manifest_digest:
        raise EvidenceError("{} summaries do not name their manifest".format(label))
    if sc.get("eikonal_mapped_initial_data") is not True \
            or ef.get("eikonal_mapped_initial_data") is not True:
        raise EvidenceError("{} is not an eikonal-mapped pair".format(label))
    if sc.get("output_steps") != 1 or ef.get("output_steps") != 1:
        raise EvidenceError("{} does not save every accepted evolution step".format(
            label))
    return {
        "label": label,
        "audit_path": path,
        "audit_sha256": audit_digest,
        "audit": audit,
        "manifest_sha256": manifest_digest,
        "sc_summary_path": sc_path,
        "ef_summary_path": ef_path,
        "sc_summary_sha256": sha256_file(sc_path),
        "ef_summary_sha256": sha256_file(ef_path),
        "sc_run_provenance": sc_run_provenance,
        "ef_run_provenance": ef_run_provenance,
        "source_manifest_sha256": sc_run_provenance[
            "source_manifest_sha256"],
        "binary_sha256": sc_run_provenance["binary_sha256"],
        "summary": sc,
        "errors": category_errors(audit, label),
    }


# Two independent realizations at each level separate sampling variation from the systematic resolution trend.
def validate_matrix(pairs):
    levels = ["coarse"]
    if "medium_a" in pairs:
        levels.append("medium")
    levels.append("fine")
    for level in levels:
        left = pairs[level + "_a"]
        right = pairs[level + "_b"]
        require_equal_fields(
            left["summary"], right["summary"],
            tuple(key for key in PAIR_KEYS if key != "sampling_realization"),
            level + " realization pair",
        )
        require_equal_static_shell_physics(
            left["summary"], right["summary"],
            level + " realization pair")
        realization_a = left["summary"].get("sampling_realization")
        realization_b = right["summary"].get("sampling_realization")
        if not isinstance(realization_a, int) or isinstance(realization_a, bool) \
                or not isinstance(realization_b, int) or isinstance(realization_b, bool) \
                or realization_a == realization_b:
            raise EvidenceError(
                "{} A/B sampling realization IDs are not distinct integers".format(
                    level))
        if left["summary"].get("base_ic_fingerprint") \
                == right["summary"].get("base_ic_fingerprint"):
            raise EvidenceError(
                "{} A/B base IC fingerprints are not independent".format(level))

    for realization in ("a", "b"):
        ids = [pairs[level + "_" + realization]["summary"].get(
            "sampling_realization") for level in levels]
        if len(set(ids)) != 1:
            raise EvidenceError(
                "realization {} changes sampling ID across refinements".format(
                    realization.upper()))

    production_physics = pairs["fine_a"]["summary"]
    production_source = pairs["fine_a"]["source_manifest_sha256"]
    production_binary = pairs["fine_a"]["binary_sha256"]
    for name, pair in pairs.items():
        require_equal_fields(
            production_physics, pair["summary"], PHYSICS_KEYS,
            "{} versus production physics".format(name),
        )
        require_equal_static_shell_physics(
            production_physics, pair["summary"],
            "{} versus production physics".format(name))
        if pair["source_manifest_sha256"] != production_source:
            raise EvidenceError(
                "{} was not built from the production source capture".format(
                    name))
        if pair["binary_sha256"] != production_binary:
            raise EvidenceError(
                "{} did not use the production executable digest".format(name))

    # Each adjacent level increases radial cells and markers while reducing the timestep cap.
    for lower_name, upper_name in zip(levels[:-1], levels[1:]):
        lower = pairs[lower_name + "_a"]["summary"]
        upper = pairs[upper_name + "_a"]["summary"]
        if not int(upper["radial_cells"]) > int(lower["radial_cells"]) \
                or not int(upper["markers"]) > int(lower["markers"]) \
                or not float(upper["dt_max"]) < float(lower["dt_max"]):
            raise EvidenceError(
                "{} -> {} is not strict space/time/PIC refinement".format(
                    lower_name, upper_name))

    monotonicity = {}
    for realization in ("a", "b"):
        monotonicity[realization] = {}
        for category in CATEGORIES:
            sequence = [
                pairs[level + "_" + realization]["errors"][category]
                for level in levels
            ]
            # Allow only floating-point comparison slack when requiring coordinate-disagreement errors to decrease under refinement.
            slack = 32.0 * sys.float_info.epsilon * max(1.0, max(sequence))
            passed = all(
                upper <= lower + slack
                for lower, upper in zip(sequence[:-1], sequence[1:])
            )
            monotonicity[realization][category] = {
                "values": sequence,
                "passed": passed,
            }
            if not passed:
                raise EvidenceError(
                    "{} errors are not non-increasing for realization {}".format(
                        category, realization.upper()))
    return levels, monotonicity


# Use the weakest refinement among radial cells, timestep cap and square-root particle count as the conservative effective ratio.
def refinement_ratio(coarse, fine):
    ratios = (
        float(fine["radial_cells"]) / float(coarse["radial_cells"]),
        float(coarse["dt_max"]) / float(fine["dt_max"]),
        math.sqrt(float(fine["markers"]) / float(coarse["markers"])),
    )
    ratio = min(ratios)
    if not math.isfinite(ratio) or not ratio > 1.0:
        raise EvidenceError("effective refinement ratio is not greater than one")
    return ratio, ratios


def build(args):
    actual_builder_sha256 = sha256_file(os.path.realpath(__file__))
    if actual_builder_sha256 != args.convergence_builder_sha256:
        raise EvidenceError(
            "convergence builder SHA-256 changed after campaign submission"
        )
    paths = {
        "coarse_a": args.coarse_a,
        "coarse_b": args.coarse_b,
        "fine_a": args.fine_a,
        "fine_b": args.fine_b,
    }
    if bool(args.medium_a) != bool(args.medium_b):
        raise EvidenceError("--medium-a and --medium-b must be supplied together")
    if args.medium_a:
        paths["medium_a"] = args.medium_a
        paths["medium_b"] = args.medium_b
    pairs = {
        name: load_pair(
            path, name, args.eikonal_substeps,
            args.diagnostic_certifier_sha256,
        )
        for name, path in paths.items()
    }
    levels, monotonicity = validate_matrix(pairs)

    coarse = pairs["coarse_a"]["summary"]
    fine = pairs["fine_a"]["summary"]
    ratio, component_ratios = refinement_ratio(coarse, fine)
    ceilings = {
        "metric": args.metric_ceiling,
        "stress": args.stress_ceiling,
        "moment": args.moment_ceiling,
        "distribution": args.distribution_ceiling,
    }
    bounds = {}
    derivation = {}
    for category in CATEGORIES:
        coarse_a = pairs["coarse_a"]["errors"][category]
        coarse_b = pairs["coarse_b"]["errors"][category]
        fine_a = pairs["fine_a"]["errors"][category]
        fine_b = pairs["fine_b"]["errors"][category]
        # Form first-order refinement estimates separately for each realization, then include actual fine errors and realization spread in the envelope.
        richardson_a = abs(coarse_a - fine_a) / (ratio - 1.0)
        richardson_b = abs(coarse_b - fine_b) / (ratio - 1.0)
        ingredients = {
            "fine_a": fine_a,
            "fine_b": fine_b,
            "fine_realization_spread": abs(fine_a - fine_b),
            "first_order_richardson_a": richardson_a,
            "first_order_richardson_b": richardson_b,
        }
        raw = max(ingredients.values())
        # Multiply the largest measured ingredient by the required safety factor and positive floor before comparing with the final ceiling.
        bound = args.safety_factor * max(raw, args.positive_floor)
        if not bound <= ceilings[category]:
            raise EvidenceError(
                "derived {} bound {} exceeds final tolerance {}".format(
                    category, bound, ceilings[category]))
        bounds[category] = bound
        derivation[category] = {
            "ingredients": ingredients,
            "maximum_ingredient": raw,
            "bound": bound,
            "ceiling": ceilings[category],
        }

    # Preserve the full measured matrix and all input identities alongside the derived category bounds.
    pair_matrix = {}
    artifacts = []
    for name in sorted(pairs):
        pair = pairs[name]
        summary = pair["summary"]
        pair_matrix[name] = {
            "sampling_realization": summary["sampling_realization"],
            "radial_cells": summary["radial_cells"],
            "markers": summary["markers"],
            "dt_max": summary["dt_max"],
            "output_steps": summary["output_steps"],
            "initial_adm_mass": summary["initial_adm_mass"],
            "base_ic_fingerprint": summary["base_ic_fingerprint"],
            "coordinate_map_manifest_sha256": pair["manifest_sha256"],
            "audit_sha256": pair["audit_sha256"],
            "source_manifest_sha256": pair["source_manifest_sha256"],
            "binary_sha256": pair["binary_sha256"],
            "errors": pair["errors"],
        }
        for kind, path, digest in (
                ("diagnostic_audit", pair["audit_path"], pair["audit_sha256"]),
                ("schwarzschild_summary", pair["sc_summary_path"],
                 pair["sc_summary_sha256"]),
                ("eddington_finkelstein_summary", pair["ef_summary_path"],
                 pair["ef_summary_sha256"])):
            artifacts.append({
                "label": name,
                "kind": kind,
                "path": path,
                "sha256": digest,
            })
        for coordinates, provenance in (
                ("schwarzschild", pair["sc_run_provenance"]),
                ("eddington_finkelstein", pair["ef_run_provenance"])):
            for kind, path_key, digest_key in (
                    ("source_checksum_manifest", "source_manifest_path",
                     "source_manifest_sha256"),
                    ("binary_checksum_record", "binary_record_path",
                     "binary_record_sha256")):
                artifacts.append({
                    "label": name,
                    "coordinates": coordinates,
                    "kind": kind,
                    "path": provenance[path_key],
                    "sha256": provenance[digest_key],
                })

    return {
        "schema": SCHEMA,
        "status": "pass",
        "coordinate_map_manifest_sha256": pairs["fine_a"][
            "manifest_sha256"],
        "production_pair": "fine_a",
        "resolution_levels": len(levels),
        "independent_pic_realizations": 2,
        "monotone_refinement": True,
        "eikonal_substeps": args.eikonal_substeps,
        "diagnostic_certifier_sha256": args.diagnostic_certifier_sha256,
        "convergence_builder_sha256": actual_builder_sha256,
        "metric_interpolation": PRODUCTION_METRIC_INTERPOLATION,
        "uncertainty_method": (
            "two-realization coordinate-disagreement envelope with first-order "
            "Richardson estimate across strict radial/time/PIC refinement"
        ),
        "fine_resolution": {
            "radial_cells": fine["radial_cells"],
            "markers": fine["markers"],
        },
        "safety_factor": args.safety_factor,
        "effective_refinement_ratio": ratio,
        "component_refinement_ratios": {
            "radial_cells": component_ratios[0],
            "dt_max": component_ratios[1],
            "sqrt_markers": component_ratios[2],
        },
        "uncertainty_bounds": bounds,
        "derivation": derivation,
        "monotonicity": monotonicity,
        "physics_configuration": {
            **{key: fine[key] for key in PHYSICS_KEYS},
            **({
                "energy_cutoff_shell": static_shell_physics(
                    fine, "fine production physics")
            } if fine.get("initial_condition") == "energy_cutoff_shell"
                else {}),
        },
        "source_manifest_sha256": pairs["fine_a"][
            "source_manifest_sha256"],
        "binary_sha256": pairs["fine_a"]["binary_sha256"],
        "pair_matrix": pair_matrix,
        "artifacts": artifacts,
    }


def positive(value):
    number = float(value)
    if not math.isfinite(number) or not number > 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def positive_integer(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a positive integer")
    if str(number) != str(value).strip() or number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def sha256_digest(value):
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest):
        raise argparse.ArgumentTypeError("value must be a SHA-256 digest")
    return digest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--coarse-a", required=True)
    result.add_argument("--coarse-b", required=True)
    result.add_argument("--medium-a")
    result.add_argument("--medium-b")
    result.add_argument("--fine-a", required=True)
    result.add_argument("--fine-b", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--eikonal-substeps", type=positive_integer,
                        required=True)
    result.add_argument("--diagnostic-certifier-sha256", type=sha256_digest,
                        required=True)
    result.add_argument("--convergence-builder-sha256", type=sha256_digest,
                        required=True)
    result.add_argument("--safety-factor", type=positive, default=1.5)
    result.add_argument("--positive-floor", type=positive, default=1.0e-15)
    result.add_argument("--metric-ceiling", type=positive, default=0.05)
    result.add_argument("--stress-ceiling", type=positive, default=0.15)
    result.add_argument("--moment-ceiling", type=positive, default=0.15)
    result.add_argument(
        "--distribution-ceiling", type=positive, default=1.0e-10)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.safety_factor < 1.5:
        print("safety factor must be at least 1.5", file=sys.stderr)
        return 1
    output = os.path.abspath(args.output)
    if os.path.exists(output) or os.path.exists(output + ".sha256"):
        print("refusing to overwrite convergence evidence", file=sys.stderr)
        return 1
    try:
        result = build(args)
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as error:
        print("dynamic-coordinate convergence failed: {}".format(error),
              file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    digest = sha256_file(output)
    with open(output + ".sha256", "w") as stream:
        stream.write("{}  {}\n".format(digest, os.path.basename(output)))
    print(json.dumps({
        "status": "pass",
        "output": output,
        "sha256": digest,
        "resolution_levels": result["resolution_levels"],
        "independent_pic_realizations": 2,
        "uncertainty_bounds": result["uncertainty_bounds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
