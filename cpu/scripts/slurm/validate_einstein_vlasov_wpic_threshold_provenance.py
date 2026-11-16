#!/usr/bin/env python3
# Check that coordinate-parity evidence applies to the exact requested physical family and executable.
# The certificate supplies numerical comparison/convergence results. This reader joins its source, binary, continuum, perturbation and null-map records.
# Output is a machine-readable acceptance or a specific incompatible-input reason before a threshold or longer paired run is scheduled.

"""Authenticate a parity certificate against a requested threshold scan.

This semantic gate is deliberately additional to the full numerical parity
certificate gate in ``analyze_einstein_vlasov_wpic.py``.  It proves that
the certificate's production summaries came from the exact source manifest
and executable still deployed for the scan, and that their static continuum,
fixed kick, radial domain, and solved-null-map controls match the scan.  The
angular-momentum scale, resolution, realization, cadence, and end time are
allowed to vary because they are respectively the scanned control and the
convergence/production controls.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys


SCHEMA = "venus_v3_threshold_parity_provenance_v1"
PAIRED_LONG_SCHEMA = "venus_v3_paired_long_parity_provenance_v1"
CERTIFICATE_SCHEMA = "venus_wpic_dynamic_coordinate_certificate_v2"
CONVERGENCE_SCHEMA = "venus_wpic_dynamic_coordinate_convergence_v1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MANIFEST_LINE_RE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
COORDINATES = ("schwarzschild", "eddington_finkelstein")
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


class GateError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path, label):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise GateError("{} cannot be read: {}".format(label, error))
    if not isinstance(value, dict):
        raise GateError("{} root is not an object".format(label))
    return value


def resolve(path, root):
    if not isinstance(path, str) or not path:
        raise GateError("authenticated artifact path is missing")
    return os.path.realpath(path if os.path.isabs(path)
                            else os.path.join(root, path))


def digest(value, label):
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GateError("{} is not a SHA-256 digest".format(label))
    return value.lower()


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise GateError("{} is not a finite JSON number".format(label))
    return float(value)


def unsigned_integer(value, label, positive=False):
    number = finite(value, label)
    integer = int(number)
    if number != integer or integer < (1 if positive else 0):
        raise GateError("{} is not a{} unsigned integer".format(
            label, " positive" if positive else ""))
    return integer


def close(left, right):
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= 5.0e-12 * scale


def require_same(actual, requested, label):
    number = finite(actual, label)
    if not close(number, requested):
        raise GateError("{} mismatch: certificate={} scan={}".format(
            label, format(number, ".17g"), format(requested, ".17g")))
    return number


# Read relative source paths with one digest each, rejecting duplicates and parent-directory escapes.
def parse_sha256_manifest(path):
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    except OSError as error:
        raise GateError("source checksum manifest cannot be read: {}".format(error))
    if not lines:
        raise GateError("source checksum manifest is empty")
    for line_number, line in enumerate(lines, 1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise GateError("source checksum manifest line {} is malformed".format(
                line_number))
        name = match.group(2)
        if os.path.isabs(name) or name in ("", ".") \
                or ".." in name.replace("\\", "/").split("/"):
            raise GateError("source checksum manifest has unsafe path {!r}".format(name))
        if name in result:
            raise GateError("source checksum manifest repeats {!r}".format(name))
        result[name] = match.group(1).lower()
    return result


def source_inventory(cpu_root):
    cpu_root = os.path.realpath(cpu_root)
    names = ["Makefile"]
    for directory in ("include", "src"):
        root = os.path.join(cpu_root, directory)
        if not os.path.isdir(root):
            raise GateError("deployed source directory is missing: {}".format(root))
        for current, directories, files in os.walk(root):
            directories.sort()
            for filename in sorted(files):
                names.append(os.path.relpath(os.path.join(current, filename), cpu_root))
    names.extend((
        "scripts/slurm/ias_einstein_vlasov_wpic.sbatch",
        "scripts/slurm/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch",
        "scripts/slurm/ias_einstein_vlasov_wpic_parity_convergence.sbatch",
        "scripts/slurm/ias_einstein_vlasov_wpic_parity_postprocess.sbatch",
        "scripts/slurm/submit_einstein_vlasov_wpic_parity.sh",
        "scripts/slurm/submit_einstein_vlasov_wpic_angular_scan.sh",
        "scripts/slurm/submit_einstein_vlasov_wpic_paired_long.sh",
        "scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gate.sbatch",
        "scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py",
        "scripts/slurm/validate_einstein_vlasov_wpic_threshold_provenance.py",
        "tools/analyze_einstein_vlasov_wpic.py",
        "tools/compare_wpic_dynamic_coordinates.py",
        "tools/build_wpic_dynamic_coordinate_convergence.py",
        "tools/validate_wpic_null_map.py",
    ))
    if len(names) != len(set(names)):
        raise GateError("current source inventory contains duplicate paths")
    result = {}
    for name in sorted(names):
        path = os.path.join(cpu_root, name)
        if not os.path.isfile(path):
            raise GateError("deployed source artifact is missing: {}".format(path))
        result[name] = sha256_file(path)
    return result


def one_artifact(records, label, kind, coordinates=None):
    matches = [record for record in records
               if isinstance(record, dict)
               and record.get("label") == label
               and record.get("kind") == kind
               and (coordinates is None
                    or record.get("coordinates") == coordinates)]
    if len(matches) != 1:
        raise GateError("expected one {} artifact for {} {}, found {}".format(
            kind, label, coordinates or "", len(matches)))
    return matches[0]


def verified_artifact(record, evidence_root):
    path = resolve(record.get("path"), evidence_root)
    expected = digest(record.get("sha256"), "artifact digest")
    try:
        actual = sha256_file(path)
    except OSError as error:
        raise GateError("authenticated artifact cannot be read: {}".format(error))
    if actual != expected:
        raise GateError("authenticated artifact changed: {}".format(path))
    return path, actual


def parse_binary_record(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    except OSError as error:
        raise GateError("binary checksum record cannot be read: {}".format(error))
    if len(lines) != 1:
        raise GateError("binary checksum record must contain exactly one entry")
    match = MANIFEST_LINE_RE.fullmatch(lines[0])
    if match is None:
        raise GateError("binary checksum record is malformed")
    return match.group(1).lower()


# Compare saved coordinate summaries against the requested continuum, radial domain, perturbation and positive-binding definitions.
def static_summary_audit(summary, coordinates, requested):
    if summary.get("coordinates") != coordinates:
        raise GateError("{} top-level coordinate tag is wrong".format(coordinates))
    high = summary.get("high_compactness")
    if not isinstance(high, dict) or high.get("coordinates") != coordinates:
        raise GateError("{} summary lacks matching high_compactness data".format(
            coordinates))
    if high.get("initial_condition") != requested["initial_condition"]:
        raise GateError("{} initial condition does not match scan".format(coordinates))
    if high.get("inner_boundary") != "origin_parity" \
            or high.get("radial_grid") != "linear":
        raise GateError("{} parity run did not use origin_parity/linear".format(
            coordinates))
    require_same(high.get("r_min"), 0.0, "{}.r_min".format(coordinates))
    require_same(high.get("r_max"), requested["r_max"],
                 "{}.r_max".format(coordinates))
    require_same(high.get("target_initial_adm_mass"), requested["target_adm"],
                 "{}.target_initial_adm_mass".format(coordinates))
    require_same(high.get("initial_radial_infall"), requested["kick"],
                 "{}.initial_radial_infall".format(coordinates))
    require_same(high.get("infall_core_radius"), requested["kick_core"],
                 "{}.infall_core_radius".format(coordinates))
    if requested["map_radius"] is not None:
        require_same(high.get("eikonal_boundary_radius"), requested["map_radius"],
                     "{}.eikonal_boundary_radius".format(coordinates))
        require_same(high.get("eikonal_evolution_outer_radius"), requested["r_max"],
                     "{}.eikonal_evolution_outer_radius".format(coordinates))
    if high.get("eikonal_mapped_initial_data") is not True:
        raise GateError("{} summary does not identify solved-eikonal data".format(
            coordinates))
    if high.get("fixed_family_binding_diagnostics_available") is not True \
            or high.get("fixed_family_positive_binding_energy") is not True:
        raise GateError("{} certified family is not authenticated as bound".format(
            coordinates))
    rest = finite(high.get("fixed_family_rest_weight"),
                  "{}.fixed_family_rest_weight".format(coordinates))
    adm = finite(high.get("fixed_family_adm_mass"),
                 "{}.fixed_family_adm_mass".format(coordinates))
    energy = finite(high.get("fixed_family_binding_energy"),
                    "{}.fixed_family_binding_energy".format(coordinates))
    fraction = finite(high.get("fixed_family_binding_fraction"),
                      "{}.fixed_family_binding_fraction".format(coordinates))
    if not (rest > 0.0 and adm > 0.0 and energy > 0.0
            and 0.0 < fraction < 1.0
            and close(rest - adm, energy)
            and close(energy / rest, fraction)):
        raise GateError("{} fixed-family binding channels are inconsistent".format(
            coordinates))
    shell = high.get("energy_cutoff_shell")
    if not isinstance(shell, dict):
        raise GateError("{} summary has no energy_cutoff_shell object".format(
            coordinates))
    require_same(shell.get("configured_amplitude_C"), requested["amplitude"],
                 "{}.configured_amplitude_C".format(coordinates))
    require_same(shell.get("configured_ell0"), requested["ell0"],
                 "{}.configured_ell0".format(coordinates))
    require_same(shell.get("configured_central_y"), requested["central_y"],
                 "{}.configured_central_y".format(coordinates))
    return {
        "coordinates": coordinates,
        "binding_fraction": fraction,
        "reference_angular_momentum_scale": finite(
            high.get("angular_momentum_scale"),
            "{}.angular_momentum_scale".format(coordinates)),
        "markers": unsigned_integer(high.get("markers"),
                                    "{}.markers".format(coordinates), True),
        "radial_cells": unsigned_integer(
            high.get("radial_cells"),
            "{}.radial_cells".format(coordinates), True),
        "dt_max": finite(high.get("dt_max"),
                         "{}.dt_max".format(coordinates)),
        "sampling_realization": unsigned_integer(
            high.get("sampling_realization"),
            "{}.sampling_realization".format(coordinates)),
    }


def audit(args):
    certificate_path = os.path.realpath(args.certificate)
    certificate_root = os.path.dirname(certificate_path)
    certificate = load_json(certificate_path, "parity certificate")
    if certificate.get("schema") != CERTIFICATE_SCHEMA \
            or certificate.get("status") != "pass":
        raise GateError("parity certificate schema/status is not strict pass")
    if certificate.get("metric_interpolation") \
            != PRODUCTION_METRIC_INTERPOLATION:
        raise GateError(
            "parity certificate lacks production-equivalent center/face "
            "metric reconstruction"
        )
    certificate_eikonal_substeps = certificate.get("eikonal_substeps")
    if isinstance(certificate_eikonal_substeps, bool) \
            or not isinstance(certificate_eikonal_substeps, int) \
            or certificate_eikonal_substeps < 1:
        raise GateError(
            "parity certificate has no valid eikonal substep contract"
        )
    canonical_certifier = os.path.join(
        args.source_root, "cpu", "tools",
        "compare_wpic_dynamic_coordinates.py",
    )
    canonical_builder = os.path.join(
        args.source_root, "cpu", "tools",
        "build_wpic_dynamic_coordinate_convergence.py",
    )
    # The current comparison and convergence tools must be the same implementations that authored the supplied evidence.
    current_certifier_sha256 = sha256_file(canonical_certifier)
    current_builder_sha256 = sha256_file(canonical_builder)
    if digest(certificate.get("certifier_sha256"), "certifier digest") \
            != current_certifier_sha256:
        raise GateError(
            "parity certificate was produced by a different certifier"
        )

    provenance = certificate.get("provenance")
    if not isinstance(provenance, dict):
        raise GateError("parity certificate has no provenance object")
    convergence = certificate.get("convergence")
    evidence_record = convergence.get("evidence") \
        if isinstance(convergence, dict) else None
    if not isinstance(evidence_record, dict):
        raise GateError("parity certificate has no convergence-evidence record")
    if evidence_record.get("metric_interpolation") \
            != PRODUCTION_METRIC_INTERPOLATION:
        raise GateError(
            "certificate convergence audit lacks the production metric contract"
        )
    if evidence_record.get("eikonal_substeps") \
            != certificate_eikonal_substeps:
        raise GateError(
            "certificate convergence audit uses different eikonal substeps"
        )
    if digest(evidence_record.get("diagnostic_certifier_sha256"),
              "diagnostic certifier digest") != current_certifier_sha256:
        raise GateError(
            "convergence diagnostics used a different parity certifier"
        )
    if digest(evidence_record.get("convergence_builder_sha256"),
              "convergence builder digest") != current_builder_sha256:
        raise GateError("convergence evidence used a different builder")
    # Authenticate the complete convergence record before using its production-pair and physical-input references.
    evidence_path = resolve(evidence_record.get("path"), certificate_root)
    evidence_digest = digest(evidence_record.get("actual_sha256"),
                             "convergence-evidence digest")
    if sha256_file(evidence_path) != evidence_digest:
        raise GateError("convergence evidence changed after certification")
    evidence = load_json(evidence_path, "convergence evidence")
    if evidence.get("schema") != CONVERGENCE_SCHEMA \
            or evidence.get("status") != "pass" \
            or evidence.get("metric_interpolation") \
                != PRODUCTION_METRIC_INTERPOLATION \
            or evidence.get("eikonal_substeps") \
                != certificate_eikonal_substeps \
            or digest(evidence.get("diagnostic_certifier_sha256"),
                      "raw diagnostic certifier digest") \
                != current_certifier_sha256 \
            or digest(evidence.get("convergence_builder_sha256"),
                      "raw convergence builder digest") \
                != current_builder_sha256:
        raise GateError("convergence evidence schema/status is not pass")
    production = evidence.get("production_pair")
    if not isinstance(production, str) or not production:
        raise GateError("convergence evidence has no production pair")
    records = evidence.get("artifacts")
    if not isinstance(records, list):
        raise GateError("convergence evidence has no artifact list")
    evidence_root = os.path.dirname(evidence_path)

    summaries = {}
    source_inventory_digest = digest(evidence.get("source_manifest_sha256"),
                                     "production source-manifest digest")
    executable_digest = digest(evidence.get("binary_sha256"),
                               "production binary digest")
    current_sources = source_inventory(os.path.join(args.source_root, "cpu"))
    # Join each coordinate summary, recorded source inventory and binary digest to the same designated production pair.
    for coordinates, summary_kind in (
            ("schwarzschild", "schwarzschild_summary"),
            ("eddington_finkelstein", "eddington_finkelstein_summary")):
        summary_record = one_artifact(records, production, summary_kind)
        summary_path, _ = verified_artifact(summary_record, evidence_root)
        certificate_summary = resolve(
            provenance.get(summary_kind), certificate_root)
        if summary_path != certificate_summary:
            raise GateError("certificate/evidence {} paths disagree".format(
                summary_kind))
        summaries[coordinates] = load_json(summary_path, summary_kind)

        source_record = one_artifact(
            records, production, "source_checksum_manifest", coordinates)
        source_path, source_record_digest = verified_artifact(
            source_record, evidence_root)
        if source_record_digest != source_inventory_digest:
            raise GateError("{} source manifest is not the production manifest".format(
                coordinates))
        if parse_sha256_manifest(source_path) != current_sources:
            raise GateError("deployed source tree differs from certified production source")

        binary_record = one_artifact(
            records, production, "binary_checksum_record", coordinates)
        binary_path, _ = verified_artifact(binary_record, evidence_root)
        if parse_binary_record(binary_path) != executable_digest:
            raise GateError("{} binary record disagrees with convergence evidence".format(
                coordinates))

    executable_path = os.path.realpath(args.executable)
    if not os.path.isfile(executable_path) or not os.access(executable_path, os.X_OK):
        raise GateError("scan executable is missing or not executable: {}".format(
            executable_path))
    if sha256_file(executable_path) != executable_digest:
        raise GateError("scan executable differs from certified production binary")

    manifest_record = provenance.get("coordinate_map_manifest")
    if not isinstance(manifest_record, dict):
        raise GateError("certificate has no coordinate-map manifest record")
    if manifest_record.get("schema") != "venus_v3_sc_to_ef_null_map" \
            or manifest_record.get("schema_version") != 2 \
            or manifest_record.get("donor_history_format_version") != 2 \
            or manifest_record.get("saved_metric_interpolation") \
                != SAVED_METRIC_INTERPOLATION:
        raise GateError(
            "certificate coordinate-map provenance lacks the face-aware v2 "
            "contract"
        )
    manifest_artifacts = manifest_record.get("artifacts")
    donor_record = manifest_artifacts.get("donor_history") \
        if isinstance(manifest_artifacts, dict) else None
    donor_audit = provenance.get("dense_time_reversed_donor_history")
    if not isinstance(donor_record, dict) \
            or donor_record.get("verified") is not True \
            or not isinstance(donor_audit, dict) \
            or donor_audit.get("format_version") != 2 \
            or donor_audit.get("production_metric_reconstruction") is not True \
            or not isinstance(donor_record.get("path"), str) \
            or not isinstance(donor_audit.get("path"), str) \
            or os.path.realpath(donor_record["path"]) \
                != os.path.realpath(donor_audit["path"]):
        raise GateError(
            "certificate does not bind its face-aware v2 donor history"
        )
    manifest_path = resolve(manifest_record.get("path"), certificate_root)
    manifest_digest = digest(manifest_record.get("actual_sha256"),
                             "coordinate-map manifest digest")
    if sha256_file(manifest_path) != manifest_digest:
        raise GateError("coordinate-map manifest changed after certification")
    manifest = load_json(manifest_path, "coordinate-map manifest")
    if manifest.get("schema") != "venus_v3_sc_to_ef_null_map" \
            or manifest.get("schema_version") != 2 \
            or manifest.get("donor_history_format_version") != 2 \
            or manifest.get("saved_metric_interpolation") \
                != SAVED_METRIC_INTERPOLATION:
        raise GateError(
            "raw coordinate-map manifest lacks the face-aware v2 contract"
        )

    # A threshold family varies the angular scale by design. A longer paired run preserves the exact reference scale and realization.
    paired_long = args.purpose == "paired_long"
    scales = []
    for token in args.angular_momentum_scales.split():
        try:
            value = float(token)
        except ValueError:
            raise GateError("invalid angular-momentum scale {!r}".format(token))
        if not math.isfinite(value) or not value > 0.0:
            raise GateError("threshold scales must be finite and strictly positive")
        scales.append(value)
    if paired_long:
        if len(scales) != 1:
            raise GateError(
                "paired-long provenance needs exactly one reference scale")
    elif len(scales) < 2:
        raise GateError("threshold scan needs at least two angular-momentum scales")

    requested = {
        "initial_condition": args.initial_condition,
        "amplitude": args.amplitude,
        "ell0": args.ell0,
        "central_y": args.central_y,
        "kick": args.radial_infall,
        "kick_core": args.infall_core_radius,
        "target_adm": args.target_initial_adm_mass,
        "map_radius": args.map_worldtube_radius,
        "r_max": args.evolution_outer_radius,
    }
    # Resolution-independent distribution and perturbation controls remain equal throughout the evidence matrix and requested family.
    physics = evidence.get("physics_configuration")
    if not isinstance(physics, dict):
        raise GateError("convergence evidence has no 2x2 physics configuration")
    if physics.get("initial_condition") != args.initial_condition \
            or physics.get("inner_boundary") != "origin_parity" \
            or physics.get("radial_grid") != "linear":
        raise GateError("convergence 2x2 physics configuration is not the scan family")
    require_same(physics.get("r_min"), 0.0, "convergence.physics.r_min")
    require_same(physics.get("r_max"), args.evolution_outer_radius,
                 "convergence.physics.r_max")
    require_same(physics.get("target_initial_adm_mass"),
                 args.target_initial_adm_mass,
                 "convergence.physics.target_initial_adm_mass")
    require_same(physics.get("initial_radial_infall"), args.radial_infall,
                 "convergence.physics.initial_radial_infall")
    require_same(physics.get("infall_core_radius"), args.infall_core_radius,
                 "convergence.physics.infall_core_radius")
    physics_shell = physics.get("energy_cutoff_shell")
    if not isinstance(physics_shell, dict):
        raise GateError("convergence evidence has no 2x2 static-shell signature")
    require_same(physics_shell.get("configured_amplitude_C"), args.amplitude,
                 "convergence.physics.configured_amplitude_C")
    require_same(physics_shell.get("configured_ell0"), args.ell0,
                 "convergence.physics.configured_ell0")
    require_same(physics_shell.get("configured_central_y"), args.central_y,
                 "convergence.physics.configured_central_y")
    require_same(physics_shell.get("configured_inward_pair_mean_kick"),
                 args.radial_infall,
                 "convergence.physics.configured_inward_pair_mean_kick")
    require_same(physics_shell.get("configured_kick_core_radius"),
                 args.infall_core_radius,
                 "convergence.physics.configured_kick_core_radius")
    if physics_shell.get("continuum_rescaled_to_target_adm_mass") is not True:
        raise GateError("certified continuum was not normalized to fixed ADM mass")
    if physics_shell.get("marker_weight_measure") \
            != "invariant_schwarzschild_wpic_rest_weight" \
            or physics_shell.get("post_sampling_canonicalization_passes") != 0:
        raise GateError("certified marker-weight convention is not invariant WPIC")
    continuum_binding = finite(physics_shell.get("continuum_binding_energy"),
                               "convergence.physics.continuum_binding_energy")
    continuum_fraction = finite(physics_shell.get("continuum_binding_fraction"),
                                "convergence.physics.continuum_binding_fraction")
    if not continuum_binding > 0.0 or not 0.0 < continuum_fraction < 1.0:
        raise GateError("convergence 2x2 static-shell signature is not bound")
    summary_audits = [
        static_summary_audit(summaries[name], name, requested)
        for name in COORDINATES
    ]
    reference_scales = [entry["reference_angular_momentum_scale"]
                        for entry in summary_audits]
    if not all(value > 0.0 for value in reference_scales) \
            or not close(reference_scales[0], reference_scales[1]):
        raise GateError("SC/EF parity summaries disagree on their reference s_L")
    # Require exact production resolution, timestep cap and realization for this longer paired-run mode.
    if paired_long:
        if not close(scales[0], reference_scales[0]):
            raise GateError(
                "paired-long s_L must equal the certified fine-A reference")
        for entry in summary_audits:
            if entry["markers"] != args.markers:
                raise GateError(
                    "paired-long markers differ from certified fine-A")
            if entry["radial_cells"] != args.radial_cells:
                raise GateError(
                    "paired-long radial cells differ from certified fine-A")
            if not close(entry["dt_max"], args.dt_max):
                raise GateError(
                    "paired-long dt_max differs from certified fine-A")
            if entry["sampling_realization"] != args.sampling_realization:
                raise GateError(
                    "paired-long realization differs from certified fine-A")

        # Check the supplied physical and null initial-data paths against the certified fine-A artifacts.
        certified_sc_summary = resolve(
            provenance["schwarzschild_summary"], certificate_root)
        expected_physical_ic = os.path.realpath(os.path.join(
            os.path.dirname(os.path.dirname(certified_sc_summary)),
            "physical_ic.bin"))
        requested_physical_ic = os.path.realpath(args.physical_ic)
        if requested_physical_ic != expected_physical_ic \
                or not os.path.isfile(requested_physical_ic):
            raise GateError(
                "paired-long physical IC is not the certified fine-A artifact")
        null_record = manifest_artifacts.get("null_initial_data") \
            if isinstance(manifest_artifacts, dict) else None
        if not isinstance(null_record, dict) \
                or null_record.get("verified") is not True:
            raise GateError(
                "certificate has no verified fine-A null initial data")
        expected_null_ic = os.path.realpath(null_record.get("path", ""))
        requested_null_ic = os.path.realpath(args.null_initial_data)
        if requested_null_ic != expected_null_ic \
                or not os.path.isfile(requested_null_ic) \
                or sha256_file(requested_null_ic) \
                    != digest(null_record.get("actual_sha256"),
                              "null initial-data digest"):
            raise GateError(
                "paired-long null initial data is not the certified fine-A artifact")

    # The EF consumer also preserves the map worldtube, exterior domain and donor duration that define its null slice.
    if args.coordinates == "eddington_finkelstein":
        if args.map_worldtube_radius is None \
                or args.ef_null_donor_duration is None:
            raise GateError("EF provenance audit requires R_B and donor duration")
        require_same(manifest.get("map_worldtube_radius"),
                     args.map_worldtube_radius, "manifest.map_worldtube_radius")
        require_same(manifest.get("evolution_outer_radius"),
                     args.evolution_outer_radius, "manifest.evolution_outer_radius")
        require_same(manifest.get("eikonal_outer_reverse_time"),
                     args.ef_null_donor_duration,
                     "manifest.eikonal_outer_reverse_time")

    return {
        "schema": PAIRED_LONG_SCHEMA if paired_long else SCHEMA,
        "status": "pass",
        "certificate": certificate_path,
        "certificate_sha256": sha256_file(certificate_path),
        "convergence_evidence": evidence_path,
        "convergence_evidence_sha256": evidence_digest,
        "coordinate_map_manifest": manifest_path,
        "coordinate_map_manifest_sha256": manifest_digest,
        "certified_source_manifest_sha256": source_inventory_digest,
        "certified_binary_sha256": executable_digest,
        "eikonal_substeps": certificate_eikonal_substeps,
        "certifier_sha256": current_certifier_sha256,
        "convergence_builder_sha256": current_builder_sha256,
        "scan_executable": executable_path,
        "static_family": requested,
        "convergence_static_shell_signature": physics_shell,
        "threshold_angular_momentum_scales": scales if not paired_long else [],
        "paired_long_angular_momentum_scale": scales[0] if paired_long else None,
        "certified_reference_angular_momentum_scale": reference_scales[0],
        "allowed_differences": ([
            "output cadence",
            "evolution end time",
        ] if paired_long else [
            "angular_momentum_scale (threshold control)",
            "marker/radial/time resolution",
            "quiet-PIC realization",
            "output cadence",
            "evolution end time",
        ]),
        "summary_audits": summary_audits,
        "issues": [],
    }


def positive(value):
    number = float(value)
    if not math.isfinite(number) or not number > 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--purpose", choices=("threshold", "paired_long"),
        default="threshold",
        help=(
            "threshold preserves the two-or-more-scale scan gate; paired_long "
            "requires one exact fine-A reference scale/resolution/realization"
        ),
    )
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--coordinates", required=True, choices=COORDINATES)
    parser.add_argument("--initial-condition", required=True)
    parser.add_argument("--amplitude", required=True, type=positive)
    parser.add_argument("--ell0", required=True, type=positive)
    parser.add_argument("--central-y", required=True, type=positive)
    parser.add_argument("--radial-infall", required=True, type=nonnegative)
    parser.add_argument("--infall-core-radius", required=True, type=nonnegative)
    parser.add_argument("--target-initial-adm-mass", required=True, type=positive)
    parser.add_argument("--angular-momentum-scales", required=True)
    parser.add_argument("--markers", type=int)
    parser.add_argument("--radial-cells", type=int)
    parser.add_argument("--dt-max", type=positive)
    parser.add_argument("--sampling-realization", type=int)
    parser.add_argument("--physical-ic")
    parser.add_argument("--null-initial-data")
    parser.add_argument("--map-worldtube-radius", type=positive)
    parser.add_argument("--evolution-outer-radius", required=True, type=positive)
    parser.add_argument("--ef-null-donor-duration", type=positive)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.purpose == "paired_long":
        required = (
            "markers", "radial_cells", "dt_max", "sampling_realization",
            "physical_ic", "null_initial_data",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(
                "paired_long requires {}".format(
                    ", ".join("--" + name.replace("_", "-")
                              for name in missing)))
        if args.markers < 1 or args.radial_cells < 1 \
                or args.sampling_realization < 0:
            parser.error(
                "paired_long markers/cells must be positive and realization non-negative")

    try:
        result = audit(args)
    except (GateError, OSError) as error:
        result = {
            "schema": SCHEMA,
            "status": "fail",
            "certificate": os.path.realpath(args.certificate),
            "issues": [str(error)],
        }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp.{}".format(os.getpid())
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, output)
    if result["status"] != "pass":
        print("threshold provenance validation: {}".format(
            "; ".join(result["issues"])), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
