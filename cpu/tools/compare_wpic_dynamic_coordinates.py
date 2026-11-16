#!/usr/bin/env python3
# Compare Schwarzschild and ingoing EF descriptions at the same spacetime events.
# Saved face-aware metrics determine forward/inverse null-coordinate maps and their Jacobians in G=c=1 units.
# The analysis transforms metric/stress components, checks per-particle covectors and invariant rest weights, then joins independent resolution/sampling uncertainty before issuing a certificate.

"""Compare SC and ingoing-EF WPIC fields at the same spacetime events.

The tool solves the ingoing eikonal equation on the saved polar--areal
spacetime,

    v_r = exp(lambda-mu) v_tau,       v(tau,R) = tau + v_offset,

by tracing each event's ingoing-null characteristic to the outer worldtube.
It evolves log(A), A=v_tau>0, and the inverse log(F), F=tau_v>0, solves the
inverse EF eikonal equation, checks forward/inverse composition, and pulls
back the complete spherical metric
and covariant stress tensor, and samples the EF run at (v(tau,r),r).  A
manifest-hashed per-ID crossing audit checks the empirical scalar distribution
under P=exp(lambda)(w+epsilon).  Equal numeric time labels are never treated as
equal events.

This is a certification tool, not a curve-alignment heuristic.  It exits with
status 2 when the temporal profiles do not resolve radial light-crossing
times, the eikonal map lacks a sufficiently large pre-horizon overlap, or the
run metadata and extraction manifest do not establish eikonal-mapped initial
hypersurfaces.  Exit status zero is reserved for a fully passing certificate.
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import struct
import sys
from array import array


INCONCLUSIVE = 2
EIKONAL_INTEGRATION_METHOD = (
    "per_cell_RK4_step_doubling_n_vs_2n_accept_2n"
)
EIKONAL_RICHARDSON_ERROR_NORMALIZATION = (
    "abs(Y_2n-Y_n)/(15*max(1,abs(Y_in),abs(Y_2n)));"
    "Y={s_sigma,ln_A}"
)
SAVED_METRIC_INTERPOLATION = (
    "SC center-face lambda plus monotone-Hermite mu; EF center-face a,b"
)
PRODUCTION_METRIC_INTERPOLATION = {
    "schwarzschild": (
        "production_center_face_lambda_monotone_hermite_mu"
    ),
    "eddington_finkelstein": "production_center_face_linear_a_b",
    "donor_history_format_version": 2,
    "production_equivalent": True,
}

# These columns carry the cell-face data used by the production
# characteristic samplers.  Cell-centre profiles alone are not enough to
# reconstruct either sampler: the SC code uses centre--face linear
# interpolation for lambda and a slope-limited Hermite reconstruction for
# mu, while the EF code uses centre--face linear interpolation for a and b.
# Centre--centre interpolation provides a separate diagnostic reconstruction.
# A production parity certificate requires the centre--face characteristic
# samplers defined by the field arrays below.
SC_RECONSTRUCTION_COLUMNS = frozenset((
    "metric_sample_kind", "metric_cell", "face_left_r", "face_right_r",
    "face_left_lambda", "face_right_lambda", "center_mu_r",
    "face_left_mu", "face_right_mu", "face_left_mu_r", "face_right_mu_r",
))
EF_RECONSTRUCTION_COLUMNS = frozenset((
    "metric_sample_kind", "metric_cell", "face_left_r", "face_right_r",
    "face_left_a", "face_right_a", "face_left_b", "face_right_b",
))


class ProfileError(RuntimeError):
    pass


class OutsideProfile(ProfileError):
    pass


def finite(value):
    return math.isfinite(float(value))


def close(left, right, scale=1.0):
    return abs(left - right) <= 2.0e-12 * max(scale, abs(left), abs(right))


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


def normalized_fingerprint(value):
    if isinstance(value, int) and value >= 0:
        return "0x{:016x}".format(value)
    if not isinstance(value, str):
        return None
    result = value.strip().lower()
    if result.startswith("0x"):
        result = result[2:]
    if not result or any(character not in "0123456789abcdef" for character in result):
        return None
    return "0x" + result.zfill(16)


def nested_value(mapping, *paths):
    """Return the first present value among explicit dotted-key paths."""
    for path in paths:
        current = mapping
        present = True
        for component in path.split("."):
            if not isinstance(current, dict) or component not in current:
                present = False
                break
            current = current[component]
        if present:
            return current
    return None


def canonical_artifact_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def manifest_artifacts(manifest, manifest_path):
    """Read path+SHA records without silently trusting an unhashed artifact.

    The extractor's canonical representation is ``artifacts.<name>`` with a
    ``path`` and ``sha256`` member.  The top-level ``<name>_path`` /
    ``<name>_sha256`` spelling is accepted for early VENUS manifests, but
    every discovered artifact must still supply both values.
    """
    records = {}
    raw_artifacts = manifest.get("artifacts", {})
    if raw_artifacts is not None and not isinstance(raw_artifacts, dict):
        raise ProfileError("coordinate-map manifest artifacts is not an object")
    flat_artifacts = raw_artifacts or {}
    consumed_flat_keys = set()
    for raw_name, raw_record in flat_artifacts.items():
        if not raw_name.endswith("_path") or not isinstance(raw_record, str):
            continue
        stem = raw_name[:-5]
        digest_key = stem + "_sha256"
        records[canonical_artifact_name(stem)] = {
            "path": raw_record,
            "sha256": flat_artifacts.get(digest_key),
        }
        consumed_flat_keys.update((raw_name, digest_key))
    for raw_name, raw_record in flat_artifacts.items():
        if raw_name in consumed_flat_keys or raw_name.endswith("_sha256"):
            continue
        name = canonical_artifact_name(raw_name)
        if isinstance(raw_record, dict):
            records[name] = {
                "path": raw_record.get("path", raw_record.get("file")),
                "sha256": raw_record.get("sha256"),
            }
        elif isinstance(raw_record, str):
            records[name] = {"path": raw_record, "sha256": None}
        else:
            raise ProfileError(
                "coordinate-map manifest artifact {!r} has no path record".format(
                    raw_name
                )
            )

    for key, value in manifest.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        name = canonical_artifact_name(key[:-5])
        records.setdefault(name, {})["path"] = value
        digest_key = key[:-5] + "_sha256"
        if digest_key in manifest:
            records[name]["sha256"] = manifest[digest_key]

    # Resolve each declared map artifact relative to its manifest, requiring both its physical file and complete content digest.
    root = os.path.dirname(os.path.abspath(manifest_path))
    verified = {}
    for name, record in sorted(records.items()):
        raw_path = record.get("path")
        expected = normalized_sha256(record.get("sha256"))
        if not isinstance(raw_path, str) or not raw_path.strip() or expected is None:
            raise ProfileError(
                "coordinate-map artifact {!r} must have path and SHA-256".format(name)
            )
        path = raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise ProfileError(
                "coordinate-map artifact {!r} is missing: {}".format(name, path)
            )
        actual = sha256_file(path)
        verified[name] = {
            "path": path,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": actual == expected,
        }
    return verified


def artifact_by_alias(artifacts, aliases):
    for alias in aliases:
        key = canonical_artifact_name(alias)
        if key in artifacts:
            return key, artifacts[key]
    return None, None


class RectilinearProfile:
    def __init__(self, path, time_name, required_fields):
        self.path = os.path.abspath(path)
        self.outer_boundary_radius = None
        self.vacuum_coordinate_system = None
        self.metric_reconstruction_kind = None
        self.metric_reconstruction = None
        required_fields = tuple(sorted(required_fields))
        with open(self.path, "r", newline="") as stream:
            reader = csv.DictReader(stream)
            names = set(reader.fieldnames or [])
            required = {"step", time_name, "r"}.union(required_fields)
            missing = sorted(required.difference(names))
            if missing:
                raise ProfileError(
                    "{} lacks required columns: {}".format(
                        self.path, ", ".join(missing)
                    )
                )
            if SC_RECONSTRUCTION_COLUMNS.issubset(names):
                self.metric_reconstruction_kind = "schwarzschild"
            elif EF_RECONSTRUCTION_COLUMNS.issubset(names):
                self.metric_reconstruction_kind = "eddington_finkelstein"

            # Parse contiguous increasing radial slices directly into compact
            # binary64 arrays. Each saved row, clock and reconstruction value
            # remains present while memory follows the numerical payload size.
            self.times = []
            self.radii = []
            self.fields = {name: [] for name in required_fields}
            if self.metric_reconstruction_kind is not None:
                self.metric_reconstruction = []

            if self.metric_reconstruction_kind == "schwarzschild":
                reconstruction_names = (
                    "face_left_r", "face_right_r",
                    "face_left_lambda", "face_right_lambda",
                    "center_mu_r", "face_left_mu", "face_right_mu",
                    "face_left_mu_r", "face_right_mu_r",
                )
            elif self.metric_reconstruction_kind == "eddington_finkelstein":
                reconstruction_names = (
                    "face_left_r", "face_right_r",
                    "face_left_a", "face_right_a",
                    "face_left_b", "face_right_b",
                )
            else:
                reconstruction_names = ()

            current_step = None
            current_time = None
            current_radii = None
            current_fields = None
            current_reconstruction = None
            current_outer_face_rows = 0
            seen_steps = set()

            def begin_slice(step, time):
                return (
                    step,
                    time,
                    array("d"),
                    {name: array("d") for name in required_fields},
                    ({name: array("d") for name in reconstruction_names}
                     if reconstruction_names else None),
                    0,
                )

            # Close a radial slice only after its clock, ordered mesh and center/outer-face reconstruction form one complete physical state.
            def finish_slice():
                if current_step is None:
                    return
                if current_step in seen_steps:
                    raise ProfileError(
                        "{} repeats non-contiguous step {}".format(
                            self.path, current_step
                        )
                    )
                seen_steps.add(current_step)
                if self.times and not current_time > self.times[-1]:
                    raise ProfileError(
                        "{} coordinate times are not strictly increasing".
                        format(self.path)
                    )
                if len(current_radii) < 4 or current_radii[0] <= 0.0 or any(
                        current_radii[index + 1] <= current_radii[index]
                        for index in range(len(current_radii) - 1)):
                    raise ProfileError(
                        "{} needs at least four positive, increasing radii".
                        format(self.path)
                    )
                if not self.radii:
                    # The radial grid is tiny compared with the field history;
                    # keep it as a normal list for existing bisect call sites.
                    self.radii = list(current_radii)
                elif len(current_radii) != len(self.radii):
                    raise ProfileError(
                        "{} step {} is not a complete radial slice".format(
                            self.path, current_step
                        )
                    )
                else:
                    for index, radius in enumerate(current_radii):
                        if not close(radius, self.radii[index], scale=1.0):
                            raise ProfileError(
                                "{} changes radial grid at step {}".format(
                                    self.path, current_step
                                )
                            )

                # Each cell center must lie between its actual faces and neighboring cells must share the same interface.
                if current_reconstruction is not None:
                    record_count = len(
                        current_reconstruction["face_left_r"]
                    )
                    if current_outer_face_rows != 1 \
                            or record_count + 1 != len(current_radii):
                        raise ProfileError(
                            "{} step {} reconstruction does not match the "
                            "center-plus-outer-face profile grid".format(
                                self.path, current_step
                            )
                        )
                    for cell in range(record_count):
                        left = current_reconstruction["face_left_r"][cell]
                        center = current_radii[cell]
                        right = current_reconstruction["face_right_r"][cell]
                        if not left < center < right:
                            raise ProfileError(
                                "{} step {} cell {} has invalid face geometry".
                                format(self.path, current_step, cell)
                            )
                        if cell and not close(
                                current_reconstruction["face_right_r"][
                                    cell - 1],
                                left,
                                scale=1.0):
                            raise ProfileError(
                                "{} step {} has a discontinuous face grid".
                                format(self.path, current_step)
                            )
                    self.metric_reconstruction.append(
                        current_reconstruction
                    )

                self.times.append(current_time)
                for name in required_fields:
                    self.fields[name].append(current_fields[name])

            for line_number, raw in enumerate(reader, start=2):
                try:
                    step = int(raw["step"])
                    time = float(raw[time_name])
                    radius = float(raw["r"])
                    values = {
                        name: float(raw[name]) for name in required_fields
                    }
                except (TypeError, ValueError) as error:
                    raise ProfileError(
                        "{}:{} has an invalid numeric field ({})".format(
                            self.path, line_number, error
                        )
                    )
                if not finite(time) or not finite(radius) or not all(
                        finite(value) for value in values.values()):
                    raise ProfileError(
                        "{}:{} has a non-finite field".format(
                            self.path, line_number
                        )
                    )
                if current_step is None:
                    (current_step, current_time, current_radii,
                     current_fields, current_reconstruction,
                     current_outer_face_rows) = begin_slice(step, time)
                elif step != current_step:
                    finish_slice()
                    (current_step, current_time, current_radii,
                     current_fields, current_reconstruction,
                     current_outer_face_rows) = begin_slice(step, time)
                elif not close(current_time, time):
                    raise ProfileError(
                        "{} step {} has multiple coordinate times".format(
                            self.path, step
                        )
                    )
                # Append binary64 values directly to the current physical slice, retaining all saved samples.
                current_radii.append(radius)
                for name in required_fields:
                    current_fields[name].append(values[name])

                if self.metric_reconstruction_kind is not None:
                    kind = raw.get("metric_sample_kind", "").strip()
                    if kind not in ("cell_center", "outer_face"):
                        raise ProfileError(
                            "{}:{} has invalid metric_sample_kind {!r}".format(
                            self.path, line_number, kind
                        )
                    )
                    if kind == "cell_center":
                        if current_outer_face_rows:
                            raise ProfileError(
                                "{}:{} has a cell center after its outer face".
                                format(self.path, line_number)
                            )
                        try:
                            cell = int(raw["metric_cell"])
                            record = {
                                name: float(raw[name])
                                for name in reconstruction_names
                            }
                        except (KeyError, TypeError, ValueError) as error:
                            raise ProfileError(
                                "{}:{} has invalid metric reconstruction ({})".
                                format(self.path, line_number, error)
                            )
                        expected_cell = len(
                            current_reconstruction["face_left_r"]
                        )
                        if cell != expected_cell or not all(
                                finite(value) for value in record.values()):
                            raise ProfileError(
                                "{}:{} has non-finite metric reconstruction".
                                format(self.path, line_number)
                            )
                        for name in reconstruction_names:
                            current_reconstruction[name].append(record[name])
                    else:
                        current_outer_face_rows += 1

            finish_slice()

        if len(self.times) < 3:
            raise ProfileError(
                "{} needs at least three complete time slices".format(
                    self.path
                )
            )

    @property
    def max_time_gap(self):
        return max(
            self.times[index + 1] - self.times[index]
            for index in range(len(self.times) - 1)
        )

    @property
    def min_radial_gap(self):
        return min(
            self.radii[index + 1] - self.radii[index]
            for index in range(len(self.radii) - 1)
        )

    def _bracket(self, grid, value, label):
        tolerance = 8.0e-13 * max(1.0, abs(grid[0]), abs(grid[-1]))
        if value < grid[0] - tolerance or value > grid[-1] + tolerance:
            raise OutsideProfile(
                "{}={} lies outside [{},{}] in {}".format(
                    label, value, grid[0], grid[-1], self.path
                )
            )
        value = min(max(value, grid[0]), grid[-1])
        if value == grid[-1]:
            return len(grid) - 2, 1.0
        slot = bisect.bisect_right(grid, value) - 1
        slot = max(0, min(slot, len(grid) - 2))
        fraction = (value - grid[slot]) / (grid[slot + 1] - grid[slot])
        return slot, fraction

    def configure_vacuum_outer_continuation(self, boundary_radius, coordinates):
        if coordinates not in ("schwarzschild", "eddington_finkelstein"):
            raise ValueError("unknown vacuum continuation coordinates")
        if boundary_radius < self.radii[-1] or not finite(boundary_radius):
            raise ProfileError(
                "manifest outer boundary lies inside the profile center grid"
            )
        self.outer_boundary_radius = float(boundary_radius)
        self.vacuum_coordinate_system = coordinates

    # Outside the matter domain use the declared spherical vacuum metric and vanishing source moments with that slice's mass.
    def _vacuum_value_at_time_index(self, name, time_index, radius):
        mass = self.fields["m"][time_index][-1]
        a_value = 1.0 - 2.0 * mass / radius
        if not a_value > 0.0:
            raise ProfileError("vacuum outer continuation has a<=0")
        if name == "m":
            return mass
        if name == "a":
            return a_value
        if self.vacuum_coordinate_system == "schwarzschild":
            if name == "lambda":
                return -0.5 * math.log(a_value)
            if name == "mu":
                return 0.5 * math.log(a_value)
            if name == "eikonal_h":
                return 1.0 / a_value
            if name in (
                    "rho", "p_radial", "p_tangential_sum", "j",
                    "lambda_t"):
                return 0.0
        else:
            if name == "b":
                return 1.0
            if name in (
                    "T_rr", "S", "T_vv", "T_vr", "T_theta_theta",
                    "rest_weight_density"):
                return 0.0
        raise ProfileError(
            "no vacuum continuation formula for field {!r}".format(name)
        )

    def _vacuum_sample(self, name, time, radius):
        ti, ft = self._bracket(self.times, time, "time")
        lower = self._vacuum_value_at_time_index(name, ti, radius)
        upper = self._vacuum_value_at_time_index(name, ti + 1, radius)
        return (1.0 - ft) * lower + ft * upper

    @property
    def has_production_metric_reconstruction(self):
        return self.metric_reconstruction is not None

    # Locate a physical cell by its saved right faces before reconstructing center/face metric values.
    def _metric_reconstruction_cell(self, time_index, radius):
        records = self.metric_reconstruction[time_index]
        compact = isinstance(records, dict)
        if compact:
            record_count = len(records["face_left_r"])

            def record_value(cell, name):
                if name == "cell":
                    return cell
                if name == "center_r":
                    return self.radii[cell]
                if name.startswith("center_") and name != "center_mu_r":
                    return self.fields[name[7:]][time_index][cell]
                return records[name][cell]

            first_left = records["face_left_r"][0]
            last_right = records["face_right_r"][-1]
            right_faces = records["face_right_r"]
        else:
            record_count = len(records)

            def record_value(cell, name):
                return records[cell][name]

            first_left = records[0]["face_left_r"]
            last_right = records[-1]["face_right_r"]
            right_faces = [value["face_right_r"] for value in records]
        tolerance = 8.0e-13 * max(
            1.0,
            abs(first_left),
            abs(last_right),
        )
        if radius < first_left - tolerance \
                or radius > last_right + tolerance:
            raise OutsideProfile(
                "radius={} lies outside metric reconstruction in {}".format(
                    radius, self.path
                )
            )
        sampled = min(max(radius, first_left), last_right)
        cell = bisect.bisect_right(right_faces, sampled)
        cell = max(0, min(cell, record_count - 1))
        if not compact:
            return records[cell], sampled
        if self.metric_reconstruction_kind == "schwarzschild":
            names = (
                "cell", "center_r", "face_left_r", "face_right_r",
                "center_lambda", "face_left_lambda", "face_right_lambda",
                "center_mu", "center_mu_r", "face_left_mu",
                "face_right_mu", "face_left_mu_r", "face_right_mu_r",
            )
        else:
            names = (
                "cell", "center_r", "face_left_r", "face_right_r",
                "center_a", "face_left_a", "face_right_a",
                "center_b", "face_left_b", "face_right_b",
            )
        return {
            name: record_value(cell, name) for name in names
        }, sampled

    @staticmethod
    # Linear interpolation is performed separately on each face-to-center half-cell, reproducing the characteristic sampler geometry.
    def _center_face_linear(record, radius, name):
        center_r = record["center_r"]
        if radius <= center_r:
            left_r = record["face_left_r"]
            width = center_r - left_r
            fraction = (radius - left_r) / width if width > 0.0 else 0.0
            return ((1.0 - fraction) * record["face_left_" + name]
                    + fraction * record["center_" + name])
        right_r = record["face_right_r"]
        width = right_r - center_r
        fraction = (radius - center_r) / width if width > 0.0 else 0.0
        return ((1.0 - fraction) * record["center_" + name]
                + fraction * record["face_right_" + name])

    @staticmethod
    def _monotone_mu_hermite(record, radius):
        center_r = record["center_r"]
        if radius <= center_r:
            left_r = record["face_left_r"]
            right_r = center_r
            left_value = record["face_left_mu"]
            right_value = record["center_mu"]
            left_derivative = record["face_left_mu_r"]
            right_derivative = record["center_mu_r"]
        else:
            left_r = center_r
            right_r = record["face_right_r"]
            left_value = record["center_mu"]
            right_value = record["face_right_mu"]
            left_derivative = record["center_mu_r"]
            right_derivative = record["face_right_mu_r"]
        width = right_r - left_r
        if not width > 0.0:
            return left_value
        # The Hermite mu slopes follow the monotone secant and are limited jointly to prevent spurious interior extrema.
        secant = (right_value - left_value) / width
        if not secant > 0.0:
            left_derivative = 0.0
            right_derivative = 0.0
        else:
            left_derivative = max(0.0, left_derivative)
            right_derivative = max(0.0, right_derivative)
            alpha = left_derivative / secant
            beta = right_derivative / secant
            radius_squared = alpha * alpha + beta * beta
            if radius_squared > 9.0:
                scale = 3.0 / math.sqrt(radius_squared)
                left_derivative *= scale
                right_derivative *= scale
        t = (radius - left_r) / width
        t2 = t * t
        t3 = t2 * t
        return (
            (2.0 * t3 - 3.0 * t2 + 1.0) * left_value
            + (t3 - 2.0 * t2 + t) * width * left_derivative
            + (-2.0 * t3 + 3.0 * t2) * right_value
            + (t3 - t2) * width * right_derivative
        )

    # Select the exact spatial reconstruction appropriate to the coordinate chart: lambda/mu or a/b.
    def _production_spatial_metric_sample(self, name, time_index, radius):
        record, sampled = self._metric_reconstruction_cell(
            time_index, radius
        )
        if self.metric_reconstruction_kind == "schwarzschild":
            if name == "lambda":
                return self._center_face_linear(record, sampled, "lambda")
            if name == "mu":
                return self._monotone_mu_hermite(record, sampled)
        elif self.metric_reconstruction_kind == "eddington_finkelstein" \
                and name in ("a", "b"):
            return self._center_face_linear(record, sampled, name)
        raise ProfileError(
            "production metric reconstruction cannot sample {!r}".format(name)
        )

    def sample(self, name, time, radius):
        if radius > self.radii[-1]:
            if self.outer_boundary_radius is None \
                    or radius > self.outer_boundary_radius:
                raise OutsideProfile(
                    "radius={} lies outside vacuum continuation in {}".format(
                        radius, self.path
                    )
                )
            return self._vacuum_sample(name, time, radius)
        ti, ft = self._bracket(self.times, time, "time")
        ri, fr = self._bracket(self.radii, radius, "radius")
        values = self.fields[name]
        lower = (1.0 - fr) * values[ti][ri] + fr * values[ti][ri + 1]
        upper = (
            (1.0 - fr) * values[ti + 1][ri]
            + fr * values[ti + 1][ri + 1]
        )
        return (1.0 - ft) * lower + ft * upper

    # Use the same spatial endpoints for the interpolated field and its saved-time difference quotient.
    def sample_with_time_derivative(self, name, time, radius):
        if radius > self.radii[-1]:
            if self.outer_boundary_radius is None \
                    or radius > self.outer_boundary_radius:
                raise OutsideProfile(
                    "radius={} lies outside vacuum continuation in {}".format(
                        radius, self.path
                    )
                )
            ti, ft = self._bracket(self.times, time, "time")
            lower = self._vacuum_value_at_time_index(name, ti, radius)
            upper = self._vacuum_value_at_time_index(name, ti + 1, radius)
            dt = self.times[ti + 1] - self.times[ti]
            return (1.0 - ft) * lower + ft * upper, (upper - lower) / dt
        ti, ft = self._bracket(self.times, time, "time")
        if self.metric_reconstruction_kind == "eddington_finkelstein" \
                and self.metric_reconstruction is not None \
                and name in ("a", "b"):
            lower = self._production_spatial_metric_sample(
                name, ti, radius
            )
            upper = self._production_spatial_metric_sample(
                name, ti + 1, radius
            )
        else:
            ri, fr = self._bracket(self.radii, radius, "radius")
            values = self.fields[name]
            lower = (1.0 - fr) * values[ti][ri] + fr * values[ti][ri + 1]
            upper = (
                (1.0 - fr) * values[ti + 1][ri]
                + fr * values[ti + 1][ri + 1]
            )
        dt = self.times[ti + 1] - self.times[ti]
        return (1.0 - ft) * lower + ft * upper, (upper - lower) / dt

    def sample_eikonal_h_with_time_derivative(self, time, radius):
        """Sample h=exp(lambda-mu) in the production mapper's order.

        On each saved slice the production sampler linearly reconstructs
        lambda between a face and center, reconstructs mu with its limited
        face/center Hermite polynomial, and exponentiates lambda-mu.  Only
        those spatially reconstructed h values are interpolated in time.
        Reproducing both the spatial scheme and this ordering is part of the
        definition of the audited coordinate map.
        """
        if radius > self.radii[-1]:
            if self.outer_boundary_radius is None \
                    or radius > self.outer_boundary_radius:
                raise OutsideProfile(
                    "radius={} lies outside vacuum continuation in {}".format(
                        radius, self.path
                    )
                )
            ti, ft = self._bracket(self.times, time, "time")
            lower = self._vacuum_value_at_time_index(
                "eikonal_h", ti, radius
            )
            upper = self._vacuum_value_at_time_index(
                "eikonal_h", ti + 1, radius
            )
        elif self.metric_reconstruction_kind == "schwarzschild" \
                and self.metric_reconstruction is not None:
            ti, ft = self._bracket(self.times, time, "time")

            # Exponentiate the spatially reconstructed lambda-minus-mu on each slice before interpolating this characteristic coefficient in time.
            def h_at_time_index(index):
                log_h = (
                    self._production_spatial_metric_sample(
                        "lambda", index, radius
                    )
                    - self._production_spatial_metric_sample(
                        "mu", index, radius
                    )
                )
                try:
                    value = math.exp(log_h)
                except OverflowError:
                    raise ProfileError(
                        "eikonal coefficient is outside floating-point range"
                    )
                if not value > 0.0 or not finite(value):
                    raise ProfileError(
                        "eikonal coefficient is not positive and finite"
                    )
                return value

            lower = h_at_time_index(ti)
            upper = h_at_time_index(ti + 1)
        else:
            ti, ft = self._bracket(self.times, time, "time")
            ri, fr = self._bracket(self.radii, radius, "radius")

            def h_at_time_index(index):
                log_h = (
                    (1.0 - fr)
                    * (self.fields["lambda"][index][ri]
                       - self.fields["mu"][index][ri])
                    + fr
                    * (self.fields["lambda"][index][ri + 1]
                       - self.fields["mu"][index][ri + 1])
                )
                try:
                    value = math.exp(log_h)
                except OverflowError:
                    raise ProfileError(
                        "eikonal coefficient is outside floating-point range"
                    )
                if not value > 0.0 or not finite(value):
                    raise ProfileError(
                        "eikonal coefficient is not positive and finite"
                    )
                return value

            lower = h_at_time_index(ti)
            upper = h_at_time_index(ti + 1)
        dt = self.times[ti + 1] - self.times[ti]
        return (1.0 - ft) * lower + ft * upper, (upper - lower) / dt


# Validate the actual stored outer worldtube sources and metric normalization before treating it as a vacuum clock anchor.
def audit_saved_vacuum_anchor(profile, coordinates, source_tolerance,
                              normalization_tolerance, anchor_radius=None):
    """Verify the saved outer row before using it as a vacuum anchor."""
    radius = profile.radii[-1] if anchor_radius is None else anchor_radius
    maximum_source = 0.0
    maximum_normalization = 0.0
    if coordinates == "schwarzschild":
        source_names = ("rho", "p_radial", "p_tangential_sum", "j")
    elif coordinates == "eddington_finkelstein":
        source_names = ("T_rr", "S", "T_vv", "T_vr", "T_theta_theta")
    else:
        raise ValueError("unknown vacuum-anchor coordinates")
    for time_index, time in enumerate(profile.times):
        def value(name):
            if anchor_radius is None:
                return profile.fields[name][time_index][-1]
            return profile.sample(name, time, radius)

        mass = value("m")
        expected_a = 1.0 - 2.0 * mass / radius
        if not expected_a > 0.0:
            raise ProfileError("saved outer vacuum anchor has a<=0")
        maximum_normalization = max(
            maximum_normalization,
            abs(value("a") - expected_a),
        )
        if coordinates == "schwarzschild":
            maximum_normalization = max(
                maximum_normalization,
                abs(value("lambda")
                    + 0.5 * math.log(expected_a)),
                abs(value("mu")
                    - 0.5 * math.log(expected_a)),
            )
        else:
            maximum_normalization = max(
                maximum_normalization,
                abs(value("b") - 1.0),
            )
        maximum_source = max(
            maximum_source,
            *(abs(value(name))
              for name in source_names)
        )
    if maximum_source > source_tolerance:
        raise ProfileError(
            "saved outer profile row is not vacuum at tolerance {}".format(
                source_tolerance)
        )
    if maximum_normalization > normalization_tolerance:
        raise ProfileError(
            "saved outer profile row violates vacuum metric normalization at "
            "tolerance {}".format(normalization_tolerance)
        )
    return {
        "coordinates": coordinates,
        "saved_anchor_radius": radius,
        "time_slices": len(profile.times),
        "maximum_source_magnitude": maximum_source,
        "maximum_metric_normalization_residual": maximum_normalization,
        "source_tolerance": source_tolerance,
        "normalization_tolerance": normalization_tolerance,
    }


def audit_donor_v2_outer_vacuum_face(
        profile, values, reconstruction, serialized_adm_mass):
    """Check the mass and metric used to append the donor's outer face.

    VENUSSCHIST1 v2 does not store ``face_m`` separately.  The production
    writer instead stores the snapshot ADM mass plus the face lambda, mu and
    mu_r arrays.  In the authenticated vacuum tail, the last cell-centre mass
    must already equal that ADM mass, and the saved outer-face metric must be
    the Schwarzschild vacuum metric for the same mass.  Checking those facts
    before appending the exact outer profile row prevents a corrupt ADM scalar
    or face payload from silently defining a hybrid discrete spacetime.
    """
    if not finite(serialized_adm_mass) or serialized_adm_mass < 0.0:
        raise ProfileError(
            "donor v2 serialized ADM mass is not finite and nonnegative"
        )
    if not reconstruction or not values.get("m"):
        raise ProfileError("donor v2 has no outer-face mass reconstruction")

    boundary_radius = profile.radii[-1]
    outer = reconstruction[-1]
    if not finite(boundary_radius) or not boundary_radius > 0.0:
        raise ProfileError("donor v2 outer-face radius is invalid")
    if not close(outer["face_right_r"], boundary_radius, scale=1.0):
        raise ProfileError(
            "donor v2 outer-face radius disagrees with the saved SC profile"
        )

    # The serialized ADM scalar and last vacuum-cell mass must describe one metric before the donor outer face is appended.
    center_mass_residual = abs(values["m"][-1] - serialized_adm_mass)
    if not close(values["m"][-1], serialized_adm_mass, scale=1.0):
        raise ProfileError(
            "donor v2 vacuum-tail mass disagrees with serialized ADM mass"
        )

    boundary_a = 1.0 - 2.0 * serialized_adm_mass / boundary_radius
    if not finite(boundary_a) or not boundary_a > 0.0:
        raise ProfileError("donor v2 outer-face continuation has a<=0")
    expected_lambda = -0.5 * math.log(boundary_a)
    expected_mu = 0.5 * math.log(boundary_a)
    expected_mu_r = (
        serialized_adm_mass
        / (boundary_radius * boundary_radius * boundary_a)
    )
    residuals = {
        "serialized_adm_mass_vs_last_center_mass": center_mass_residual,
        "radius": abs(outer["face_right_r"] - boundary_radius),
        "lambda": abs(outer["face_right_lambda"] - expected_lambda),
        "mu": abs(outer["face_right_mu"] - expected_mu),
        "mu_r": abs(outer["face_right_mu_r"] - expected_mu_r),
    }
    expected = {
        "lambda": expected_lambda,
        "mu": expected_mu,
        "mu_r": expected_mu_r,
    }
    for name in ("lambda", "mu", "mu_r"):
        actual = outer["face_right_" + name]
        if not finite(actual) or not close(actual, expected[name], scale=1.0):
            raise ProfileError(
                "donor v2 outer-face {} is inconsistent with its serialized "
                "ADM mass".format(name)
            )
    return {
        "serialized_adm_mass": serialized_adm_mass,
        "boundary_radius": boundary_radius,
        "boundary_a": boundary_a,
        "residuals": residuals,
    }


def prepend_time_reversed_donor_history(profile, path):
    """Extend the SC profile to physical tau<0 from the hashed donor binary.

    The extractor stores outgoing fields versus s=-tau.  Scalar metric and
    pressure moments are even under time reversal, while j is odd.  Restoring
    that sign is essential before solving v(tau,r) through the matter region.
    """
    with open(path, "rb") as stream:
        payload = stream.read()
    if len(payload) < 36 or payload[:12] != b"VENUSSCHIST1":
        raise ProfileError("donor history has invalid VENUSSCHIST1 magic")
    offset = 16

    def unpack(code):
        nonlocal offset
        size = struct.calcsize("<" + code)
        if offset + size > len(payload):
            raise ProfileError("donor history is truncated")
        value = struct.unpack_from("<" + code, payload, offset)[0]
        offset += size
        return value

    version = unpack("I")
    snapshots = unpack("Q")
    cells = unpack("Q")
    has_exact_outer_face = cells + 1 == len(profile.radii)
    if version not in (1, 2) or snapshots < 2 or (
            cells != len(profile.radii) and not has_exact_outer_face):
        raise ProfileError(
            "donor history version/grid does not match the SC profiles"
        )
    names = ("lambda", "mu", "m", "rho", "p_radial", "p_tangential_sum", "j")
    history = []
    outer_face_audits = []
    for _index in range(snapshots):
        reverse_time = unpack("d")
        _rest_mass = unpack("d")
        adm_mass = unpack("d")
        _compactness = unpack("d")
        _min_gap = unpack("d")
        values = {}
        for name in names:
            values[name] = [unpack("d") for _cell in range(cells)]
        # Restore physical incoming time: radial current is odd under reversal while density, pressure and metric scalars are even.
        values["j"] = [-value for value in values["j"]]
        values["a"] = [math.exp(-2.0 * value) for value in values["lambda"]]
        values["lambda_t"] = [0.0] * cells
        reconstruction = None
        if version == 2:
            face_radius = [unpack("d") for _face in range(cells + 1)]
            face_lambda = [unpack("d") for _face in range(cells + 1)]
            face_mu = [unpack("d") for _face in range(cells + 1)]
            face_mu_r = [unpack("d") for _face in range(cells + 1)]
            center_mu_r = [unpack("d") for _cell in range(cells)]
            reconstruction = []
            for cell in range(cells):
                reconstruction.append({
                    "cell": cell,
                    "center_r": profile.radii[cell],
                    "face_left_r": face_radius[cell],
                    "face_right_r": face_radius[cell + 1],
                    "center_lambda": values["lambda"][cell],
                    "face_left_lambda": face_lambda[cell],
                    "face_right_lambda": face_lambda[cell + 1],
                    "center_mu": values["mu"][cell],
                    "center_mu_r": center_mu_r[cell],
                    "face_left_mu": face_mu[cell],
                    "face_right_mu": face_mu[cell + 1],
                    "face_left_mu_r": face_mu_r[cell],
                    "face_right_mu_r": face_mu_r[cell + 1],
                })
        if has_exact_outer_face:
            boundary_radius = profile.radii[-1]
            if version == 2:
                outer_face_audits.append(
                    audit_donor_v2_outer_vacuum_face(
                        profile, values, reconstruction, adm_mass
                    )
                )
                # face_m is represented by the serialized snapshot ADM mass
                # in VENUSSCHIST1 v2.  The audit above proves that it is also
                # the constant mass in the saved vacuum tail.
                mass = adm_mass
            else:
                mass = values["m"][-1]
            boundary_a = 1.0 - 2.0 * mass / boundary_radius
            if not boundary_a > 0.0:
                raise ProfileError("donor outer-face continuation has a<=0")
            values["lambda"].append(-0.5 * math.log(boundary_a))
            values["mu"].append(0.5 * math.log(boundary_a))
            values["m"].append(mass)
            values["rho"].append(0.0)
            values["p_radial"].append(0.0)
            values["p_tangential_sum"].append(0.0)
            values["j"].append(0.0)
            values["a"].append(boundary_a)
            values["lambda_t"].append(0.0)
        history.append((-reverse_time, values, reconstruction))
    # The donor binary must end exactly after its declared arrays, preventing a partial or mixed-format history from being accepted.
    if offset != len(payload):
        raise ProfileError("donor history has trailing bytes")
    history.sort(key=lambda item: item[0])
    cutoff = profile.times[0]
    retained = [item for item in history if item[0] < cutoff - 1.0e-14]
    if not retained:
        raise ProfileError("donor history supplies no physical tau<0 slices")
    if profile.metric_reconstruction is not None \
            and (version != 2 or any(item[2] is None for item in retained)):
        # Validate before mutating the profile.  A caught provenance error
        # must not leave the time/field arrays and reconstruction arrays with
        # different lengths.
        raise ProfileError(
            "production SC profiles require VENUSSCHIST1 version 2 "
            "face reconstruction in the donor history"
        )
    profile.times = [item[0] for item in retained] + profile.times
    for name in profile.fields:
        if name not in retained[0][1]:
            raise ProfileError(
                "donor history cannot supply required SC field {!r}".format(name)
            )
        profile.fields[name] = [item[1][name] for item in retained] \
            + profile.fields[name]
    if profile.metric_reconstruction is not None:
        profile.metric_reconstruction = [item[2] for item in retained] \
            + profile.metric_reconstruction
    if any(profile.times[index + 1] <= profile.times[index]
           for index in range(len(profile.times) - 1)):
        raise ProfileError("combined donor/SC physical times are not increasing")
    result = {
        "path": os.path.abspath(path),
        "physical_time_min": profile.times[0],
        "physical_time_max": profile.times[-1],
        "prepended_slices": len(retained),
        "format_version": version,
        "production_metric_reconstruction": (
            profile.metric_reconstruction is not None
        ),
        "time_reversal": "tau=-s; j(tau)=-j_reverse(s)",
    }
    if outer_face_audits:
        result["outer_vacuum_face"] = {
            "validated_snapshots": len(outer_face_audits),
            "mass_source": "serialized_snapshot_adm_mass",
            "maximum_residuals": {
                name: max(
                    audit["residuals"][name]
                    for audit in outer_face_audits
                )
                for name in outer_face_audits[0]["residuals"]
            },
        }
    return result


def load_summary(run_dir, expected_coordinates):
    path = os.path.join(os.path.abspath(run_dir), "summary.json")
    with open(path, "r") as stream:
        value = json.load(stream)
    if value.get("coordinates") != expected_coordinates:
        raise ProfileError(
            "{} does not describe a {} run".format(path, expected_coordinates)
        )
    high = value.get("high_compactness")
    if not isinstance(high, dict):
        raise ProfileError("{} has no high_compactness object".format(path))
    return path, high


def evenly_spaced_indices(size, requested):
    if requested >= size:
        return list(range(size))
    if requested <= 1:
        return [size - 1]
    return sorted(set(
        int(round(index * (size - 1) / float(requested - 1)))
        for index in range(requested)
    ))


# Along the ingoing SC null characteristic, d tau/dr=-h and d log(v_tau)/dr=-partial_tau h, with h=exp(lambda-mu).
def rk4_segment(sc, radius, time, log_sensitivity, width):
    """Advance the SC null ray and q_A=log(v_tau) by one RK4 step."""
    def rhs(sample_radius, sample_time):
        h, h_time = sc.sample_eikonal_h_with_time_derivative(
            sample_time, sample_radius
        )
        if not h > 0.0 or not finite(h) or not finite(h_time):
            raise ProfileError(
                "eikonal coefficient exp(lambda-mu) is not positive and finite"
            )
        return -h, -h_time

    k1_t, k1_log_a = rhs(radius, time)
    k2_t, k2_log_a = rhs(
        radius + 0.5 * width,
        time + 0.5 * width * k1_t,
    )
    k3_t, k3_log_a = rhs(
        radius + 0.5 * width,
        time + 0.5 * width * k2_t,
    )
    k4_t, k4_log_a = rhs(
        radius + width,
        time + width * k3_t,
    )
    next_time = (
        time + width * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t) / 6.0
    )
    next_log_a = (
        log_sensitivity
        + width
        * (k1_log_a + 2.0 * k2_log_a + 2.0 * k3_log_a + k4_log_a)
        / 6.0
    )
    if not finite(next_time) or not finite(next_log_a):
        raise ProfileError("forward eikonal RK4 stage is not finite")
    return next_time, next_log_a


# Logarithmic evolution preserves the positive Jacobian representation. Exponentiation still requires finite representable values.
def positive_jacobian_from_log(log_value, name):
    """Exponentiate an exact log-Jacobian without clipping or sign repair."""
    if not finite(log_value):
        raise ProfileError("{} log Jacobian is not finite".format(name))
    try:
        value = math.exp(log_value)
    except OverflowError:
        raise ProfileError(
            "{} Jacobian is outside floating-point range".format(name)
        )
    if not value > 0.0 or not finite(value):
        raise ProfileError(
            "{} Jacobian is outside floating-point range".format(name)
        )
    return value


def trace_to_outer_boundary(
        sc, time, radius, substeps, v_offset, boundary_radius=None):
    if substeps < 1:
        raise ValueError("substeps must be positive")
    outer = boundary_radius or sc.outer_boundary_radius or sc.radii[-1]
    if radius > outer or radius < sc.radii[0]:
        raise OutsideProfile("event radius is outside the SC profile")
    # Split the trace at saved radial intervals and use the requested RK4 substeps in every interval.
    breakpoints = [radius]
    breakpoints.extend(
        value for value in sc.radii if radius < value < outer
    )
    if not close(breakpoints[-1], outer, scale=1.0):
        breakpoints.append(outer)
    current_time = time
    log_sensitivity = 0.0
    current_radius = radius
    for endpoint in breakpoints[1:]:
        width = (endpoint - current_radius) / float(substeps)
        for _index in range(substeps):
            current_time, log_sensitivity = rk4_segment(
                sc, current_radius, current_time, log_sensitivity, width
            )
            current_radius += width
    sensitivity = positive_jacobian_from_log(log_sensitivity, "forward")
    return current_time + v_offset, sensitivity, log_sensitivity


def rk4_inverse_segment(ef, radius, advanced_time, log_sensitivity, width):
    """Advance an EF ingoing-null ray and its derivative with respect to v.

    The inverse eikonal equation is

        tau_r + (a b)^(-1) tau_v = 0.

    Thus tau is constant on rays satisfying dv/dr=(a b)^(-1).  The
    sensitivity propagated here becomes tau_v at the starting event after the
    ray reaches the outer boundary.
    """
    def rhs(sample_radius, sample_time):
        a_value, a_time = ef.sample_with_time_derivative(
            "a", sample_time, sample_radius
        )
        b_value, b_time = ef.sample_with_time_derivative(
            "b", sample_time, sample_radius
        )
        if not a_value > 0.0 or not b_value > 0.0:
            raise ProfileError(
                "inverse eikonal entered a<=0 or b<=0 outside its SC overlap"
            )
        q = 1.0 / (a_value * b_value)
        # Differentiate q=1/(a*b) at fixed radius to evolve the inverse characteristic sensitivity consistently.
        q_time = -q * (a_time / a_value + b_time / b_value)
        if not finite(q) or not finite(q_time):
            raise ProfileError("inverse eikonal RK4 stage is not finite")
        return q, q_time

    k1_v, k1_log_f = rhs(radius, advanced_time)
    k2_v, k2_log_f = rhs(
        radius + 0.5 * width,
        advanced_time + 0.5 * width * k1_v,
    )
    k3_v, k3_log_f = rhs(
        radius + 0.5 * width,
        advanced_time + 0.5 * width * k2_v,
    )
    k4_v, k4_log_f = rhs(
        radius + width,
        advanced_time + width * k3_v,
    )
    next_time = (
        advanced_time
        + width * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    )
    next_log_f = (
        log_sensitivity
        + width
        * (k1_log_f + 2.0 * k2_log_f + 2.0 * k3_log_f + k4_log_f)
        / 6.0
    )
    if not finite(next_time) or not finite(next_log_f):
        raise ProfileError("inverse eikonal RK4 stage is not finite")
    return next_time, next_log_f


# Use the raw n-versus-2n change for the strict saved-history comparison; Richardson scaling belongs to the separately stated extraction estimator.
def rk4_step_doubling_change(coarse, fine):
    """Return the raw n-versus-2n change used by the strict history gate."""
    change = abs(fine - coarse)
    if not finite(change):
        raise ProfileError("eikonal step-refinement change is not finite")
    return change


def jacobian_composition_error(log_a, log_f):
    """Return |A F-1| from logs, failing closed outside numeric range."""
    log_product = log_a + log_f
    if not finite(log_product):
        raise ProfileError("eikonal composition has a non-finite log Jacobian")
    try:
        # The forward/inverse Jacobians compose to unity. expm1 evaluates their small closure residual accurately.
        error = abs(math.expm1(log_product))
    except OverflowError:
        raise ProfileError("eikonal composition is outside floating-point range")
    if not finite(error):
        raise ProfileError("eikonal composition residual is not finite")
    return error


def inverse_trace_to_outer_boundary(
        ef, advanced_time, radius, substeps, v_offset,
        boundary_radius=None):
    if substeps < 1:
        raise ValueError("substeps must be positive")
    outer = boundary_radius or ef.outer_boundary_radius or ef.radii[-1]
    if radius > outer or radius < ef.radii[0]:
        raise OutsideProfile("event radius is outside the EF profile")
    breakpoints = [radius]
    breakpoints.extend(
        value for value in ef.radii if radius < value < outer
    )
    if not close(breakpoints[-1], outer, scale=1.0):
        breakpoints.append(outer)
    current_time = advanced_time
    log_sensitivity = 0.0
    current_radius = radius
    for endpoint in breakpoints[1:]:
        width = (endpoint - current_radius) / float(substeps)
        for _index in range(substeps):
            current_time, log_sensitivity = rk4_inverse_segment(
                ef, current_radius, current_time, log_sensitivity, width
            )
            current_radius += width
    sensitivity = positive_jacobian_from_log(log_sensitivity, "inverse")
    return current_time - v_offset, sensitivity, log_sensitivity


COMMON_DOMAIN_BOUNDARY_STATUSES = frozenset((
    "outside_sc_eikonal_domain",
    "outside_ef_domain",
    "at_or_beyond_horizon",
))

COMMON_DOMAIN_NUMERIC_DIAGNOSTICS = (
    "forward_step_refinement_error",
    "forward_time_step_refinement_error",
    "forward_log_A_step_refinement_error",
    "inverse_step_refinement_error",
    "inverse_time_step_refinement_error",
    "inverse_log_F_step_refinement_error",
    "composition_time_error",
    "composition_jacobian_error",
)


# Define comparison support geometrically: an outer-connected radial interval over one contiguous block of adequate saved times.
def audit_contiguous_common_domain(
        time_indices, times, radii, event_states,
        minimum_radial_coverage):
    """Find the maximal sampled exterior on which both charts can overlap.

    For each SC time, scan inward from the common outer radius.  An explicit
    history boundary or the a>a_min chart boundary may end the exterior.
    Numerical failures do not end it: they remain inside the domain and are
    reported as unexplained holes.  Among radially adequate slices, select the
    longest contiguous time block.  Any later re-entry is reported, never
    silently folded into a favorable coverage fraction.
    """
    radial_count = len(radii)
    slices = []
    disconnected_non_horizon_support_samples = 0
    for time_position, time_index in enumerate(time_indices):
        domain_slots_reversed = []
        boundary_status = None
        boundary_slot = None
        disconnected = 0
        for radial_slot in range(radial_count - 1, -1, -1):
            state = event_states.get((time_index, radial_slot), {})
            status = state.get("status", "missing_event_audit")
            if boundary_status is None and status in COMMON_DOMAIN_BOUNDARY_STATUSES:
                boundary_status = status
                boundary_slot = radial_slot
                continue
            if boundary_status is None:
                domain_slots_reversed.append(radial_slot)
            elif status not in COMMON_DOMAIN_BOUNDARY_STATUSES:
                disconnected += 1
        domain_slots = list(reversed(domain_slots_reversed))
        if boundary_status != "at_or_beyond_horizon":
            disconnected_non_horizon_support_samples += disconnected
        radial_coverage = (
            len(domain_slots) / float(radial_count) if radial_count else 0.0
        )
        # Numerical mapping failures inside that domain remain explicit holes; they cannot be reclassified as chart boundaries.
        hole_slots = [
            radial_slot for radial_slot in domain_slots
            if event_states.get((time_index, radial_slot), {}).get("status")
            != "mapped"
        ]
        slices.append({
            "time_position": time_position,
            "time_index": time_index,
            "schwarzschild_time": times[time_index],
            "domain_radial_slots": domain_slots,
            "domain_radial_points": len(domain_slots),
            "radial_coverage": radial_coverage,
            "qualifies_radially": (
                radial_coverage >= minimum_radial_coverage
            ),
            "complete": not hole_slots and bool(domain_slots),
            "hole_slots": hole_slots,
            "first_boundary_status": boundary_status,
            "first_boundary_radial_slot": boundary_slot,
            "disconnected_interior_samples": disconnected,
        })

    qualifying_positions = [
        position for position, value in enumerate(slices)
        if value["qualifies_radially"]
    ]
    runs = []
    for position in qualifying_positions:
        if not runs or position != runs[-1][-1] + 1:
            runs.append([position])
        else:
            runs[-1].append(position)
    # Choose the longest time-contiguous qualifying block and separately report all later re-entry or internal gaps.
    chosen_run = max(runs, key=lambda run: (len(run), -run[0])) if runs else []
    chosen_positions = set(chosen_run)
    domain_keys = set()
    hole_examples = []
    unexplained_holes = 0
    candidate_events = 0
    compared_events = 0
    complete_time_slices = 0
    for position in chosen_run:
        value = slices[position]
        candidate_events += value["domain_radial_points"]
        slice_compared = 0
        for radial_slot in value["domain_radial_slots"]:
            key = (value["time_index"], radial_slot)
            domain_keys.add(key)
            state = event_states.get(key, {})
            if state.get("status") == "mapped":
                compared_events += 1
                slice_compared += 1
                continue
            unexplained_holes += 1
            if len(hole_examples) < 16:
                example = {
                    "time_position": value["time_position"],
                    "time_index": value["time_index"],
                    "radial_slot": radial_slot,
                    "schwarzschild_time": value["schwarzschild_time"],
                    "r": radii[radial_slot],
                    "status": state.get("status", "missing_event_audit"),
                    "inner_neighbor_status": (
                        event_states.get(
                            (value["time_index"], radial_slot - 1), {}
                        ).get("status", "missing_event_audit")
                        if radial_slot > 0 else None
                    ),
                    "outer_neighbor_status": (
                        event_states.get(
                            (value["time_index"], radial_slot + 1), {}
                        ).get("status", "missing_event_audit")
                        if radial_slot + 1 < radial_count else None
                    ),
                    "first_inner_boundary_status": value[
                        "first_boundary_status"
                    ],
                    "first_inner_boundary_radial_slot": value[
                        "first_boundary_radial_slot"
                    ],
                }
                for name in COMMON_DOMAIN_NUMERIC_DIAGNOSTICS:
                    if name in state and finite(state[name]):
                        example[name] = state[name]
                if isinstance(state.get("error"), str):
                    example["error"] = state["error"]
                hole_examples.append(example)
        if slice_compared == value["domain_radial_points"] \
                and value["domain_radial_points"]:
            complete_time_slices += 1

    time_topology_gap_slices = 0
    if qualifying_positions:
        first_qualifying = qualifying_positions[0]
        last_qualifying = qualifying_positions[-1]
        time_topology_gap_slices = sum(
            not slices[position]["qualifies_radially"]
            for position in range(first_qualifying, last_qualifying + 1)
        )

    event_coverage = (
        compared_events / float(candidate_events) if candidate_events else 0.0
    )
    time_slice_coverage = (
        complete_time_slices / float(len(chosen_run)) if chosen_run else 0.0
    )
    minimum_slice_radial_coverage = (
        min(slices[position]["radial_coverage"] for position in chosen_run)
        if chosen_run else 0.0
    )
    first_time = (
        slices[chosen_run[0]]["schwarzschild_time"] if chosen_run else None
    )
    last_time = (
        slices[chosen_run[-1]]["schwarzschild_time"] if chosen_run else None
    )
    if not chosen_run:
        terminal_reason = "no_radially_complete_common_slice"
    elif chosen_run[-1] == len(slices) - 1:
        terminal_reason = "end_of_outer_clock_overlap"
    else:
        next_slice = slices[chosen_run[-1] + 1]
        if next_slice["first_boundary_status"] == "at_or_beyond_horizon":
            terminal_reason = "schwarzschild_or_ef_chart_cutoff"
        else:
            terminal_reason = "loss_of_common_support"

    public_slices = []
    for position, value in enumerate(slices):
        domain_slots = value["domain_radial_slots"]
        public_slices.append({
            "schwarzschild_time": value["schwarzschild_time"],
            "in_selected_time_block": position in chosen_positions,
            "domain_radial_points": value["domain_radial_points"],
            "radial_coverage": value["radial_coverage"],
            "inner_radius": radii[domain_slots[0]] if domain_slots else None,
            "outer_radius": radii[domain_slots[-1]] if domain_slots else None,
            "unexplained_holes": len(value["hole_slots"]),
            "first_boundary_status": value["first_boundary_status"],
        })

    return {
        "_domain_keys": domain_keys,
        "definition": (
            "maximal_contiguous_exterior_pre_schwarzschild_terminal"
        ),
        "radial_topology": "outer_connected_suffix",
        "sampled_time_slices_in_outer_clock_overlap": len(time_indices),
        "sampled_radial_points": radial_count,
        "required_minimum_radial_coverage": minimum_radial_coverage,
        "qualifying_time_runs": len(runs),
        "qualifying_time_slices": len(qualifying_positions),
        "initial_partial_overlap_time_slices": (
            chosen_run[0] if chosen_run else len(slices)
        ),
        "terminal_excluded_time_slices": (
            len(slices) - chosen_run[-1] - 1 if chosen_run else 0
        ),
        "candidate_events": candidate_events,
        "compared_events": compared_events,
        "event_coverage": event_coverage,
        "time_slices": len(chosen_run),
        "complete_time_slices": complete_time_slices,
        "time_slice_coverage": time_slice_coverage,
        "minimum_per_slice_radial_coverage": minimum_slice_radial_coverage,
        "unexplained_holes": unexplained_holes,
        "unexplained_hole_examples": hole_examples,
        "time_topology_gap_slices": time_topology_gap_slices,
        "disconnected_non_horizon_support_samples": (
            disconnected_non_horizon_support_samples
        ),
        "first_schwarzschild_time": first_time,
        "last_schwarzschild_time": last_time,
        "terminal_reason": terminal_reason,
        "slices": public_slices,
    }


# Reduce mapping errors on the declared common domain while retaining all-attempted diagnostics elsewhere in the result.
def summarize_domain_numeric_errors(event_states, domain_keys):
    """Reduce convergence/closure diagnostics only on the declared domain."""
    result = {}
    for name in COMMON_DOMAIN_NUMERIC_DIAGNOSTICS:
        values = [
            state[name] for key, state in event_states.items()
            if key in domain_keys and name in state and finite(state[name])
        ]
        result[name] = max(values) if values else None

    def worst(prefix):
        error_name = prefix + "_step_refinement_error"
        candidates = [
            state for key, state in event_states.items()
            if key in domain_keys and error_name in state
            and finite(state[error_name])
        ]
        if not candidates:
            return None
        return dict(max(candidates, key=lambda value: value[error_name]))

    result["forward_worst_step_refinement_event"] = worst("forward")
    result["inverse_worst_step_refinement_event"] = worst("inverse")
    return result


class ComparisonAccumulator:
    def __init__(self):
        self.pairs = {}

    def add(self, name, predicted, reference):
        if not finite(predicted) or not finite(reference):
            raise ProfileError("non-finite transformed {}".format(name))
        self.pairs.setdefault(name, []).append((predicted, reference))

    def errors(self, absolute_floor):
        result = {}
        for name, values in sorted(self.pairs.items()):
            # Normalize L2 and maximum errors symmetrically with both field amplitudes and the explicit absolute floor.
            difference_square = sum(
                (left - right) ** 2 for left, right in values
            )
            left_square = sum(left * left for left, _right in values)
            right_square = sum(right * right for _left, right in values)
            scale_l2 = max(
                math.sqrt(left_square),
                math.sqrt(right_square),
                absolute_floor * math.sqrt(len(values)),
            )
            scale_linf = max(
                max(abs(left) for left, _right in values),
                max(abs(right) for _left, right in values),
                absolute_floor,
            )
            result[name] = {
                "samples": len(values),
                "relative_l2": math.sqrt(difference_square) / scale_l2,
                "relative_linf": max(
                    abs(left - right) for left, right in values
                ) / scale_linf,
                "maximum_predicted_magnitude": max(
                    abs(left) for left, _right in values
                ),
                "maximum_reference_magnitude": max(
                    abs(right) for _left, right in values
                ),
            }
        return result


def discover_manifest_path(args, sc_summary, ef_summary):
    if args.manifest:
        return os.path.abspath(args.manifest)
    candidates = []
    for run_dir, summary in (
            (args.schwarzschild, sc_summary), (args.ef, ef_summary)):
        named = summary.get("coordinate_map_manifest_path")
        if isinstance(named, str) and named.strip():
            path = named if os.path.isabs(named) else os.path.join(run_dir, named)
            candidates.append(os.path.abspath(path))
    ef_root = os.path.abspath(args.ef)
    try:
        for name in os.listdir(ef_root):
            lower = name.lower()
            if lower.endswith(".manifest.json") and any(
                    token in lower for token in ("null", "eikonal", "coordinate", "map")):
                candidates.append(os.path.join(ef_root, name))
    except OSError:
        pass
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise ProfileError(
        "multiple coordinate-map manifests are discoverable; pass --manifest"
    )


def load_coordinate_manifest(args, sc_summary, ef_summary):
    """Load and authenticate the null-extraction provenance graph."""
    issues = []
    path = discover_manifest_path(args, sc_summary, ef_summary)
    if path is None:
        return None, {}, {}, [
            "no coordinate-map manifest is available; pass --manifest"
        ]
    try:
        with open(path, "r") as stream:
            manifest = json.load(stream)
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        return path, {}, {}, [
            "coordinate-map manifest cannot be read: {}".format(error)
        ]
    if not isinstance(manifest, dict):
        return path, {}, {}, ["coordinate-map manifest root is not an object"]
    actual_digest = sha256_file(path)
    expected_sc_digest = normalized_sha256(
        sc_summary.get("coordinate_map_manifest_sha256")
    )
    expected_ef_digest = normalized_sha256(
        ef_summary.get("coordinate_map_manifest_sha256")
    )
    if expected_sc_digest != actual_digest:
        issues.append(
            "Schwarzschild summary does not authenticate the supplied manifest"
        )
    if expected_ef_digest != actual_digest:
        issues.append("EF summary does not authenticate the supplied manifest")

    schema = manifest.get("schema")
    schema_version = manifest.get("schema_version")
    if schema != "venus_v3_sc_to_ef_null_map" \
            or schema_version not in (1, 2):
        issues.append(
            "coordinate-map manifest is not venus_v3_sc_to_ef_null_map v1/v2"
        )
    if schema_version != 2:
        issues.append(
            "production parity requires coordinate-map manifest schema v2"
        )
    completed = nested_value(
        manifest,
        "completed",
        "complete",
        "extraction.completed",
        "extraction.complete",
        "completeness.completed",
        "completeness.complete",
    )
    if completed is not True:
        issues.append("coordinate-map extraction is not marked complete")
    crossed_once = nested_value(
        manifest,
        "all_markers_crossed_once",
        "extraction.all_markers_crossed_once",
        "completeness.all_markers_crossed_once",
    )
    if crossed_once is not True:
        issues.append("manifest does not certify one unique crossing per source marker")
    if manifest.get("outer_worldtube_verified_vacuum") is not True:
        issues.append("manifest does not certify a vacuum outer normalization worldtube")
    donor_direction = manifest.get("donor_direction")
    if not isinstance(donor_direction, str) or "time_reversed" not in donor_direction:
        issues.append("manifest does not identify the East-style time-reversed donor")
    if not isinstance(manifest.get("eikonal_equation"), str) \
            or "exp(lambda-mu)" not in manifest["eikonal_equation"]:
        issues.append("manifest does not state the ingoing eikonal equation")
    # The null-slice scope, time reversal, outer clock, integration method and interpolation convention determine what the map means physically.
    exact_manifest_contract = {
        "mapping_scope": "initial_null_hypersurface",
        "time_reversal_momentum": "w_reverse=-w_schwarzschild",
        "outer_normalization": (
            "at R_B: v=tau, A=v_tau=1, F=tau_v=1, b=1"
            if schema_version == 2 else "v=tau and A=v_tau=1"
        ),
        "ef_time_coordinate": "v_bar=v+ef_v_offset",
        "eikonal_integration_method": EIKONAL_INTEGRATION_METHOD,
        "eikonal_richardson_error_normalization": (
            EIKONAL_RICHARDSON_ERROR_NORMALIZATION
        ),
    }
    if schema_version == 2:
        exact_manifest_contract.update({
            "donor_history_format_version": 2,
            "saved_metric_interpolation": SAVED_METRIC_INTERPOLATION,
        })
    for name, expected in exact_manifest_contract.items():
        if manifest.get(name) != expected:
            issues.append(
                "manifest {} does not equal {!r}".format(name, expected)
            )
    physical_run = manifest.get("physical_schwarzschild_run")
    if not isinstance(physical_run, str) or "unmodified" not in physical_run:
        issues.append("manifest does not preserve the physical incoming SC run")
    marker_map = manifest.get("marker_momentum_map")
    if not isinstance(marker_map, str) or "exp(lambda)" not in marker_map \
            or ">0" not in marker_map:
        issues.append("manifest does not certify the future-oriented marker map")

    source_count = nested_value(
        manifest,
        "source_marker_count",
        "marker_count",
        "counts.source_markers",
        "counts.source_marker_count",
        "extraction.source_marker_count",
    )
    mapped_count = nested_value(
        manifest,
        "mapped_marker_count",
        "crossing_count",
        "counts.mapped_markers",
        "counts.mapped_marker_count",
        "extraction.mapped_marker_count",
    )
    unique_count = nested_value(
        manifest,
        "unique_crossing_count",
        "counts.unique_crossing_count",
        "extraction.unique_crossing_count",
    )
    counts_valid = (
        isinstance(source_count, int)
        and not isinstance(source_count, bool)
        and isinstance(mapped_count, int)
        and not isinstance(mapped_count, bool)
        and source_count > 0
        and mapped_count == source_count
        and isinstance(unique_count, int)
        and not isinstance(unique_count, bool)
        and unique_count == source_count
    )
    if not counts_valid:
        issues.append("manifest marker counts do not prove complete extraction")

    extension_audit = {}
    if schema_version == 2:
        map_radius = manifest.get("map_worldtube_radius")
        evolution_radius = manifest.get("evolution_outer_radius")
        legacy_radius = manifest.get("outer_boundary_radius")
        map_cells = manifest.get("map_radial_cell_count")
        evolution_cells = manifest.get("evolution_radial_cell_count")
        extension_start = manifest.get(
            "analytic_vacuum_extension_start_index"
        )
        extension = manifest.get("analytic_vacuum_extension")
        radii_valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool) and finite(value)
            for value in (map_radius, evolution_radius, legacy_radius)
        ) and 0.0 < map_radius <= evolution_radius \
            and close(legacy_radius, map_radius, scale=1.0)
        counts_are_valid = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (map_cells, evolution_cells, extension_start)
        ) and 0 < map_cells <= evolution_cells \
            and extension_start == map_cells \
            and manifest.get("radial_cell_count") == evolution_cells
        split = radii_valid and evolution_radius > map_radius
        if not radii_valid or not counts_are_valid \
                or not isinstance(extension, bool) or extension != split \
                or (split and map_cells == evolution_cells):
            issues.append(
                "manifest v2 has inconsistent R_B/R_E analytic-extension metadata"
            )
        full_grid_fingerprint = normalized_fingerprint(
            manifest.get("grid_fingerprint")
        )
        map_grid_fingerprint = normalized_fingerprint(
            manifest.get("map_grid_fingerprint")
        )
        if full_grid_fingerprint is None or map_grid_fingerprint is None:
            issues.append("manifest v2 lacks map/full grid fingerprints")
        required_clearance = manifest.get("required_worldtube_clearance")
        matter_clearance = manifest.get(
            "minimum_worldtube_matter_clearance"
        )
        crossing_clearance = manifest.get(
            "minimum_crossing_radius_clearance"
        )
        max_donor = manifest.get("maximum_donor_marker_radius")
        max_crossing = manifest.get("maximum_crossing_radius")
        clearance_values = (
            required_clearance, matter_clearance, crossing_clearance,
            max_donor, max_crossing,
        )
        # Require source matter and crossing events to remain inside the map worldtube by the declared positive clearance.
        clearances_valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool) and finite(value)
            for value in clearance_values
        ) and required_clearance > 0.0 \
            and matter_clearance >= required_clearance \
            and crossing_clearance >= required_clearance \
            and close(matter_clearance, map_radius - max_donor, scale=1.0) \
            and close(crossing_clearance, map_radius - max_crossing, scale=1.0)
        if not clearances_valid:
            issues.append("manifest v2 worldtube matter/crossing clearance fails")
        analytic_limits = {
            "maximum_analytic_vacuum_field_residual": args.composition_tolerance,
            "maximum_analytic_vacuum_source": args.outer_vacuum_tolerance,
            "maximum_analytic_inverse_map_residual": args.composition_tolerance,
        }
        analytic_values = {}
        for name, tolerance in analytic_limits.items():
            value = manifest.get(name)
            analytic_values[name] = value
            if not isinstance(value, (int, float)) \
                    or isinstance(value, bool) or not finite(value) \
                    or value < 0.0 or value > tolerance:
                issues.append(
                    "manifest {} does not meet tolerance {}".format(
                        name, tolerance
                    )
                )
        minimum_analytic_a = manifest.get("minimum_analytic_vacuum_a")
        if not isinstance(minimum_analytic_a, (int, float)) \
                or isinstance(minimum_analytic_a, bool) \
                or not finite(minimum_analytic_a) \
                or minimum_analytic_a <= 0.0:
            issues.append("manifest v2 analytic vacuum crosses a<=0")
        b_error = manifest.get("maximum_outer_b_normalization_error")
        b_tolerance = manifest.get("outer_b_normalization_tolerance")
        if not isinstance(b_tolerance, (int, float)) \
                or isinstance(b_tolerance, bool) or not finite(b_tolerance) \
                or b_tolerance <= 0.0 \
                or b_tolerance > args.composition_tolerance \
                or not isinstance(b_error, (int, float)) \
                or isinstance(b_error, bool) or not finite(b_error) \
                or b_error > b_tolerance:
            issues.append(
                "manifest v2 b normalization fails its authenticated mesh tolerance"
            )
        extension_audit = {
            "map_worldtube_radius": map_radius,
            "evolution_outer_radius": evolution_radius,
            "map_radial_cell_count": map_cells,
            "evolution_radial_cell_count": evolution_cells,
            "analytic_vacuum_extension": extension,
            "analytic_vacuum_extension_start_index": extension_start,
            "map_grid_fingerprint": map_grid_fingerprint,
            "evolution_grid_fingerprint": full_grid_fingerprint,
            "minimum_worldtube_matter_clearance": matter_clearance,
            "minimum_crossing_radius_clearance": crossing_clearance,
            "analytic_residuals": analytic_values,
            "minimum_analytic_vacuum_a": minimum_analytic_a,
            "outer_b_normalization_tolerance": b_tolerance,
        }

    residual_checks = (
        (
            "maximum_eikonal_reverse_time_richardson_error",
            args.eikonal_convergence_tolerance,
        ),
        (
            "maximum_eikonal_log_A_richardson_error",
            args.eikonal_convergence_tolerance,
        ),
        (
            "maximum_eikonal_ode_residual",
            args.eikonal_convergence_tolerance,
        ),
        (
            "maximum_marker_crossing_residual",
            args.crossing_residual_tolerance,
        ),
        ("maximum_replay_time_error", args.composition_tolerance),
        (
            "maximum_outer_b_normalization_error",
            args.composition_tolerance,
        ),
        ("maximum_outer_source", args.outer_vacuum_tolerance),
    )
    extraction_residuals = {}
    for name, tolerance in residual_checks:
        value = manifest.get(name)
        extraction_residuals[name] = value
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not finite(value) or value < 0.0 or value > tolerance:
            issues.append(
                "manifest {} does not meet tolerance {}".format(
                    name, tolerance
                )
            )

    eikonal_reverse_time_error = extraction_residuals.get(
        "maximum_eikonal_reverse_time_richardson_error"
    )
    eikonal_log_A_error = extraction_residuals.get(
        "maximum_eikonal_log_A_richardson_error"
    )
    eikonal_legacy_maximum = extraction_residuals.get(
        "maximum_eikonal_ode_residual"
    )
    eikonal_components_are_finite = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and finite(value)
        for value in (
            eikonal_reverse_time_error,
            eikonal_log_A_error,
            eikonal_legacy_maximum,
        )
    )
    if eikonal_components_are_finite and eikonal_legacy_maximum != max(
            eikonal_reverse_time_error, eikonal_log_A_error):
        issues.append(
            "manifest maximum_eikonal_ode_residual is not the maximum "
            "of the two normalized RK4 Richardson component errors"
        )

    base_manifest = normalized_fingerprint(nested_value(
        manifest,
        "base_ic_fingerprint",
        "provenance.base_ic_fingerprint",
        "fingerprints.base_ic",
    ))
    source_realized = normalized_fingerprint(nested_value(
        manifest,
        "schwarzschild_realized_ic_fingerprint",
        "schwarzschild_realized_fingerprint",
        "source_realized_ic_fingerprint",
        "provenance.schwarzschild_realized_ic_fingerprint",
        "provenance.source_realized_ic_fingerprint",
        "fingerprints.schwarzschild_realized_ic",
    ))
    target_realized = normalized_fingerprint(nested_value(
        manifest,
        "eddington_finkelstein_realized_ic_fingerprint",
        "ef_realized_fingerprint",
        "target_realized_ic_fingerprint",
        "provenance.eddington_finkelstein_realized_ic_fingerprint",
        "provenance.target_realized_ic_fingerprint",
        "fingerprints.eddington_finkelstein_realized_ic",
    ))
    sc_base = normalized_fingerprint(sc_summary.get("base_ic_fingerprint"))
    ef_base = normalized_fingerprint(ef_summary.get("base_ic_fingerprint"))
    sc_realized = normalized_fingerprint(
        sc_summary.get("realized_ic_fingerprint")
    )
    ef_realized = normalized_fingerprint(
        ef_summary.get("realized_ic_fingerprint")
    )
    if base_manifest is None or sc_base != base_manifest or ef_base != base_manifest:
        issues.append("base-IC fingerprints do not match the map manifest")
    if source_realized is None or sc_realized != source_realized:
        issues.append(
            "Schwarzschild realized-IC fingerprint does not match its manifest field"
        )
    if target_realized is None or ef_realized != target_realized:
        issues.append(
            "EF realized-IC fingerprint does not match its separate manifest field"
        )

    try:
        artifacts = manifest_artifacts(manifest, path)
    except (IOError, OSError, ProfileError) as error:
        artifacts = {}
        issues.append(str(error))
    for name, record in artifacts.items():
        if not record["verified"]:
            issues.append(
                "coordinate-map artifact {!r} fails SHA-256 verification".format(
                    name
                )
            )
    required_aliases = {
        "binary null initial data": (
            "binary", "null_initial_data", "ef_null_initial_data", "map_binary",
            "null_map_binary"
        ),
        "marker crossings": (
            "crossings", "crossings_csv", "marker_crossings", "crossing_audit"
        ),
        "mapped null slice": (
            "slice", "slice_csv", "null_slice", "ef_null_slice"
        ),
        "accepted donor steps": (
            "accepted_steps", "accepted_steps_csv", "donor_accepted_steps",
            "time_steps"
        ),
        "donor spacetime history": (
            "history", "history_csv", "donor_history", "spacetime_history"
        ),
    }
    for label, aliases in required_aliases.items():
        _name, record = artifact_by_alias(artifacts, aliases)
        if record is None:
            issues.append("manifest has no hashed {} artifact".format(label))

    audit = {
        "path": os.path.abspath(path),
        "actual_sha256": actual_digest,
        "schwarzschild_summary_sha256": expected_sc_digest,
        "eddington_finkelstein_summary_sha256": expected_ef_digest,
        "schema": schema,
        "schema_version": schema_version,
        "completed": completed is True,
        "all_markers_crossed_once": crossed_once is True,
        "eikonal_integration_method": manifest.get(
            "eikonal_integration_method"
        ),
        "eikonal_richardson_error_normalization": manifest.get(
            "eikonal_richardson_error_normalization"
        ),
        "donor_history_format_version": manifest.get(
            "donor_history_format_version"
        ),
        "saved_metric_interpolation": manifest.get(
            "saved_metric_interpolation"
        ),
        "source_marker_count": source_count,
        "mapped_marker_count": mapped_count,
        "unique_crossing_count": unique_count,
        "base_ic_fingerprint": base_manifest,
        "schwarzschild_realized_ic_fingerprint": source_realized,
        "eddington_finkelstein_realized_ic_fingerprint": target_realized,
        "realized_fingerprints_are_coordinate_specific": True,
        "extraction_residuals": extraction_residuals,
        "analytic_vacuum_extension": extension_audit,
        "artifacts": artifacts,
    }
    return path, manifest, audit, issues


# Coordinate agreement at one resolution is a diagnostic. Independent space/time/particle refinement supplies its uncertainty.
def load_convergence_evidence(args, manifest_audit, sc_summary, ef_summary):
    """Authenticate independent resolution/PIC uncertainty evidence.

    A single coordinate pair can be an excellent regression diagnostic, but
    cannot by itself bound spatial/temporal truncation and particle sampling
    error.  Only this separate artifact can promote diagnostic agreement to a
    passing physics certificate.
    """
    if not args.convergence_evidence:
        return {}, [
            "single-resolution agreement has no independent quantified "
            "discretization/PIC convergence evidence"
        ]
    path = os.path.abspath(args.convergence_evidence)
    issues = []
    try:
        with open(path, "r") as stream:
            evidence = json.load(stream)
    except (IOError, OSError, ValueError, json.JSONDecodeError) as error:
        return {}, ["convergence evidence cannot be read: {}".format(error)]
    if not isinstance(evidence, dict):
        return {}, ["convergence evidence root is not an object"]
    actual_digest = sha256_file(path)
    sidecar_path = args.convergence_evidence_sha256 or path + ".sha256"
    expected_digest = None
    sidecar_name = None
    try:
        with open(sidecar_path, "r") as stream:
            fields = stream.readline().strip().split()
        if len(fields) == 2:
            expected_digest = normalized_sha256(fields[0])
            sidecar_name = fields[1].lstrip("*")
    except (IOError, OSError):
        pass
    if expected_digest != actual_digest \
            or sidecar_name != os.path.basename(path):
        issues.append(
            "convergence evidence SHA-256 sidecar is missing, stale, or "
            "names a different file"
        )
    if evidence.get("schema") != "venus_wpic_dynamic_coordinate_convergence_v1":
        issues.append("unrecognized dynamic-coordinate convergence schema")
    if evidence.get("status") != "pass":
        issues.append("convergence evidence is not passing")
    map_digest = normalized_sha256(
        evidence.get("coordinate_map_manifest_sha256")
    )
    if map_digest != manifest_audit.get("actual_sha256"):
        issues.append("convergence evidence targets a different coordinate map")
    resolution_levels = evidence.get("resolution_levels")
    pic_realizations = evidence.get("independent_pic_realizations")
    if not isinstance(resolution_levels, int) or resolution_levels < 2:
        issues.append("convergence evidence needs at least two resolution levels")
    if not isinstance(pic_realizations, int) or pic_realizations < 2:
        issues.append("convergence evidence needs at least two PIC realizations")
    if evidence.get("monotone_refinement") is not True:
        issues.append("convergence evidence does not establish monotone refinement")
    if evidence.get("metric_interpolation") != PRODUCTION_METRIC_INTERPOLATION:
        issues.append(
            "convergence evidence does not establish production-equivalent "
            "center/face metric reconstruction"
        )
    current_certifier_sha256 = sha256_file(os.path.realpath(__file__))
    builder_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "build_wpic_dynamic_coordinate_convergence.py",
    )
    try:
        current_builder_sha256 = sha256_file(builder_path)
    except (IOError, OSError):
        current_builder_sha256 = None
    if evidence.get("diagnostic_certifier_sha256") \
            != current_certifier_sha256:
        issues.append(
            "convergence diagnostics were produced by a different certifier"
        )
    if evidence.get("convergence_builder_sha256") \
            != current_builder_sha256:
        issues.append(
            "convergence evidence was produced by a different builder"
        )
    evidence_eikonal_substeps = evidence.get("eikonal_substeps")
    if isinstance(evidence_eikonal_substeps, bool) \
            or not isinstance(evidence_eikonal_substeps, int) \
            or evidence_eikonal_substeps < 1:
        issues.append(
            "convergence evidence has no valid eikonal substep count"
        )
    elif evidence_eikonal_substeps != args.eikonal_substeps:
        issues.append(
            "convergence evidence eikonal substeps do not match the "
            "certificate invocation"
        )
    method = evidence.get("uncertainty_method")
    if not isinstance(method, str) or not method.strip():
        issues.append("convergence evidence has no uncertainty method")

    # The uncertainty record must apply to the actual marker and radial resolution of both compared runs.
    fine = evidence.get("fine_resolution", {})
    sc_cells = sc_summary.get("radial_cells")
    ef_cells = ef_summary.get("radial_cells")
    sc_markers = sc_summary.get("markers")
    ef_markers = ef_summary.get("markers")
    if not isinstance(fine, dict) \
            or fine.get("radial_cells") != sc_cells \
            or fine.get("radial_cells") != ef_cells \
            or fine.get("markers") != sc_markers \
            or fine.get("markers") != ef_markers:
        issues.append("fine convergence resolution does not match both run summaries")

    bounds = evidence.get("uncertainty_bounds", {})
    configured = {
        "metric": args.metric_tolerance,
        "stress": args.stress_tolerance,
        "moment": args.moment_tolerance,
        "distribution": args.distribution_tolerance,
    }
    normalized_bounds = {}
    for name, ceiling in configured.items():
        value = bounds.get(name) if isinstance(bounds, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not finite(value) or not 0.0 < value <= ceiling:
            issues.append(
                "convergence {} uncertainty is absent or exceeds {}".format(
                    name, ceiling
                )
            )
        else:
            normalized_bounds[name] = float(value)

    raw_artifacts = evidence.get("artifacts")
    verified_artifacts = []
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) < 2:
        issues.append("convergence evidence needs at least two hashed pair audits")
    else:
        root = os.path.dirname(path)
        for index, record in enumerate(raw_artifacts):
            if not isinstance(record, dict):
                issues.append("convergence artifact {} is malformed".format(index))
                continue
            raw_path = record.get("path")
            expected = normalized_sha256(record.get("sha256"))
            if not isinstance(raw_path, str) or expected is None:
                issues.append("convergence artifact {} lacks path/SHA".format(index))
                continue
            artifact_path = (
                raw_path if os.path.isabs(raw_path)
                else os.path.join(root, raw_path)
            )
            artifact_path = os.path.abspath(artifact_path)
            try:
                actual = sha256_file(artifact_path)
            except (IOError, OSError):
                actual = None
            verified = actual == expected
            if not verified:
                issues.append(
                    "convergence artifact {} fails SHA-256 verification".format(index)
                )
            verified_artifacts.append({
                "path": artifact_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "verified": verified,
            })
    audit = {
        "path": path,
        "actual_sha256": actual_digest,
        "sidecar_sha256": expected_digest,
        "sidecar_named_file": sidecar_name,
        "resolution_levels": resolution_levels,
        "independent_pic_realizations": pic_realizations,
        "monotone_refinement": evidence.get("monotone_refinement") is True,
        "metric_interpolation": evidence.get("metric_interpolation"),
        "eikonal_substeps": evidence_eikonal_substeps,
        "diagnostic_certifier_sha256": evidence.get(
            "diagnostic_certifier_sha256"
        ),
        "convergence_builder_sha256": evidence.get(
            "convergence_builder_sha256"
        ),
        "uncertainty_method": method,
        "uncertainty_bounds": normalized_bounds,
        "artifacts": verified_artifacts,
    }
    return audit, issues


def read_numeric_csv(path):
    with open(path, "r", newline="") as stream:
        reader = csv.DictReader(stream)
        names = list(reader.fieldnames or [])
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            parsed = {}
            for name in names:
                if name in ("id", "source_id", "step", "slice") \
                        or name.endswith("_fingerprint"):
                    parsed[name] = raw[name]
                    continue
                try:
                    parsed[name] = float(raw[name])
                except (TypeError, ValueError):
                    raise ProfileError(
                        "{}:{} has non-numeric {}".format(path, line_number, name)
                    )
                if not finite(parsed[name]):
                    raise ProfileError(
                        "{}:{} has non-finite {}".format(path, line_number, name)
                    )
            rows.append(parsed)
    if not rows:
        raise ProfileError("{} has no audit rows".format(path))
    return set(names), rows


def first_column(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise ProfileError("marker audit lacks all aliases {}".format(names))


def audit_marker_distribution(path, args, allow_repeated_ids=False):
    """Certify the empirical scalar distribution under the covector map."""
    names, rows = read_numeric_csv(path)
    extraction_columns = {
        "id", "source_id", "rest_weight", "eta", "reverse_time",
        "schwarzschild_time", "r", "w_reverse", "w_schwarzschild",
        "p_r_ef", "lambda", "mu", "surface_reverse_time", "residual",
        "replay_step", "step_fraction", "source_marker_fingerprint",
        "ef_marker_fingerprint",
    }
    if not allow_repeated_ids:
        missing_extraction = sorted(extraction_columns.difference(names))
        if missing_extraction:
            raise ProfileError(
                "initial marker crossing audit lacks extractor columns {}".format(
                    missing_extraction)
            )
    required_groups = (
        ("id",),
        ("source_id",),
        ("r", "r_sc", "radius"),
        ("w_schwarzschild", "w_sc", "w"),
        ("p_r_ef", "P_ef", "P"),
        ("eta", "eta_sc"),
        ("rest_weight", "rest_weight_sc", "weight"),
        ("lambda", "lambda_sc"),
        ("tau_cross", "tau", "polar_time", "schwarzschild_time"),
        ("residual", "crossing_residual", "map_residual"),
    )
    missing = [group for group in required_groups if not any(
        name in names for name in group
    )]
    if missing:
        raise ProfileError(
            "marker audit {} lacks required groups {}".format(path, missing)
        )
    accumulator = ComparisonAccumulator()
    seen = set()
    seen_sources = set()
    slices = set()
    total_weight = 0.0
    max_residual = 0.0
    scalar_samples = 0
    for index, row in enumerate(rows):
        marker_id = str(row["id"])
        source_id = str(row["source_id"])
        slice_label = str(row.get(
            "slice", row.get("surface_s", row.get("v", "initial"))
        ))
        # Each initial marker crosses once. A later trajectory audit may reuse an ID on distinct explicitly identified slices.
        identity = (slice_label, marker_id) if allow_repeated_ids else marker_id
        if identity in seen:
            raise ProfileError(
                "marker audit repeats ID {!r} on slice {!r}".format(
                    marker_id, slice_label
                )
            )
        seen.add(identity)
        source_identity = (
            (slice_label, source_id) if allow_repeated_ids else source_id
        )
        if source_identity in seen_sources:
            raise ProfileError(
                "marker audit repeats source ID {!r} on slice {!r}".format(
                    source_id, slice_label)
            )
        seen_sources.add(source_identity)
        slices.add(slice_label)
        radius_sc = float(first_column(row, "r_sc", "r", "radius"))
        radius_ef = float(row.get("r_ef", radius_sc))
        w_sc = float(first_column(row, "w_schwarzschild", "w_sc", "w"))
        p_ef = float(first_column(row, "p_r_ef", "P_ef", "P"))
        eta_sc = float(first_column(row, "eta_sc", "eta"))
        eta_ef = float(row.get("eta_ef", eta_sc))
        weight_sc = float(first_column(
            row, "rest_weight_sc", "rest_weight", "weight"
        ))
        weight_ef = float(row.get("rest_weight_ef", weight_sc))
        lam = float(first_column(row, "lambda_sc", "lambda"))
        residual = abs(float(first_column(
            row, "residual", "crossing_residual", "map_residual"
        )))
        if not allow_repeated_ids:
            reverse_time = float(row["reverse_time"])
            schwarzschild_time = float(row["schwarzschild_time"])
            w_reverse = float(row["w_reverse"])
            surface_reverse_time = float(row["surface_reverse_time"])
            signed_residual = float(row["residual"])
            replay_step = float(row["replay_step"])
            step_fraction = float(row["step_fraction"])
            residual_identity_error = abs(
                reverse_time - surface_reverse_time - signed_residual
            )
            residual_identity_scale = max(
                1.0, abs(reverse_time), abs(surface_reverse_time)
            )
            if residual_identity_error > 5.0e-12 * residual_identity_scale:
                raise ProfileError(
                    "marker audit residual is not reverse_time-surface_reverse_time"
                )
            if reverse_time < -args.crossing_residual_tolerance:
                raise ProfileError("marker audit has negative donor reverse time")
            if replay_step < 0.0 or replay_step != math.floor(replay_step) \
                    or not 0.0 <= step_fraction <= 1.0:
                raise ProfileError("marker audit has invalid replay location")
            if normalized_fingerprint(row["source_marker_fingerprint"]) is None \
                    or normalized_fingerprint(
                        row["ef_marker_fingerprint"]) is None:
                raise ProfileError("marker audit has malformed marker fingerprint")
            accumulator.add(
                "distribution_time_reversal_tau",
                -reverse_time,
                schwarzschild_time,
            )
            accumulator.add(
                "distribution_time_reversal_w", -w_reverse, w_sc
            )
            accumulator.add(
                "distribution_null_surface_crossing",
                reverse_time,
                surface_reverse_time,
            )
        if not radius_sc > 0.0 or not radius_ef > 0.0:
            raise ProfileError("marker audit contains non-positive radius")
        if eta_sc < 0.0 or eta_ef < 0.0:
            raise ProfileError("marker audit contains negative eta")
        if not weight_sc > 0.0 or not weight_ef > 0.0:
            raise ProfileError("marker audit contains non-positive rest weight")
        chi = 1.0 + 2.0 * eta_sc / (radius_sc * radius_sc)
        epsilon = math.sqrt(chi + w_sc * w_sc)
        # The covector map P=exp(lambda)*(w+epsilon) uses the same event, invariant eta and future-oriented positive P.
        predicted_p = math.exp(lam) * (w_sc + epsilon)
        if not p_ef > 0.0 or not predicted_p > 0.0:
            raise ProfileError("marker audit violates future orientation P>0")
        x = math.exp(-lam) * p_ef
        # The mass-shell relation inverts the covector map, providing an independent radial-momentum comparison.
        inverse_w = 0.5 * (x - chi / x)
        accumulator.add("distribution_radius", radius_sc, radius_ef)
        accumulator.add("distribution_eta", eta_sc, eta_ef)
        accumulator.add("distribution_rest_weight", weight_sc, weight_ef)
        accumulator.add("distribution_p_r_covector_map", predicted_p, p_ef)
        accumulator.add("distribution_inverse_w", inverse_w, w_sc)
        f_sc = row.get("f_schwarzschild", row.get("f_sc"))
        f_ef = row.get("f_eddington_finkelstein", row.get("f_ef"))
        if f_sc is not None or f_ef is not None:
            if f_sc is None or f_ef is None:
                raise ProfileError("marker audit provides only one scalar-f value")
            accumulator.add("distribution_scalar_f", float(f_sc), float(f_ef))
            scalar_samples += 1
        total_weight += weight_sc
        max_residual = max(max_residual, residual)
    if len(rows) < args.minimum_marker_samples:
        raise ProfileError(
            "marker audit has {} samples; {} required".format(
                len(rows), args.minimum_marker_samples
            )
        )
    errors = accumulator.errors(args.absolute_floor)
    failures = sorted(
        name for name, diagnostics in errors.items()
        if diagnostics["relative_l2"] > args.distribution_tolerance
        or diagnostics["relative_linf"] > args.distribution_tolerance
    )
    if max_residual > args.crossing_residual_tolerance:
        failures.append("distribution_crossing_residual")
    return {
        "path": os.path.abspath(path),
        "samples": len(rows),
        "slices": len(slices),
        "unique_id_slice_pairs": len(seen),
        "unique_source_id_slice_pairs": len(seen_sources),
        "total_rest_weight": total_weight,
        "maximum_crossing_residual": max_residual,
        "explicit_scalar_f_samples": scalar_samples,
        "empirical_measure_rule": (
            "same ID/source ID, invariant rest weight and eta, same event, "
            "P=exp(lambda)*(w+epsilon)"
        ),
        "errors": errors,
        "failed_fields": sorted(set(failures)),
    }


# Check the directly saved initial mapped metric and covariant stress components, including each recorded difference identity.
def audit_initial_coordinate_map(path, args):
    names, rows = read_numeric_csv(path)
    fields = {
        "initial_metric_a": ("mapped_a", "ef_a", "delta_a", "metric"),
        "initial_metric_b": ("mapped_b", "ef_b", "delta_b", "metric"),
        "initial_misner_sharp_mass": (
            "mapped_m", "ef_m", "delta_m", "metric"
        ),
        "initial_stress_T_rr": (
            "mapped_T_rr", "ef_T_rr", "delta_T_rr", "stress"
        ),
        "initial_stress_S": ("mapped_S", "ef_S", "delta_S", "stress"),
        "initial_stress_T_vv": (
            "mapped_T_vv", "ef_T_vv", "delta_T_vv", "stress"
        ),
        "initial_stress_T_vr": (
            "mapped_T_vr", "ef_T_vr", "delta_T_vr", "stress"
        ),
        "initial_stress_T_theta_theta": (
            "mapped_T_theta_theta", "ef_T_theta_theta",
            "delta_T_theta_theta", "stress"
        ),
    }
    required = {"r"}
    for left, right, delta, _kind in fields.values():
        required.update((left, right, delta))
    missing = sorted(required.difference(names))
    if missing:
        raise ProfileError(
            "initial coordinate-map audit lacks {}".format(", ".join(missing))
        )
    accumulator = ComparisonAccumulator()
    for row in rows:
        if not row["r"] > 0.0:
            raise ProfileError("initial coordinate-map audit has r<=0")
        for name, (left, right, delta, _kind) in fields.items():
            predicted = float(row[left])
            reference = float(row[right])
            saved_delta = float(row[delta])
            if abs(saved_delta - (reference - predicted)) \
                    > 5.0e-12 * max(1.0, abs(predicted), abs(reference)):
                raise ProfileError(
                    "initial coordinate-map audit has inconsistent {}".format(delta)
                )
            accumulator.add(name, predicted, reference)
    errors = accumulator.errors(args.absolute_floor)
    failures = []
    for name, diagnostics in errors.items():
        tolerance = (
            args.metric_tolerance
            if fields[name][3] == "metric" else args.stress_tolerance
        )
        if diagnostics["relative_l2"] > tolerance \
                or diagnostics["relative_linf"] > tolerance:
            failures.append(name)
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "samples": len(rows),
        "errors": errors,
        "failed_fields": sorted(failures),
    }


def compare(args):
    certifier_sha256 = sha256_file(os.path.realpath(__file__))
    try:
        sc_summary_path, sc_summary = load_summary(
            args.schwarzschild, "schwarzschild"
        )
        ef_summary_path, ef_summary = load_summary(
            args.ef, "eddington_finkelstein"
        )
        sc = RectilinearProfile(
            os.path.join(args.schwarzschild, "high_compactness_profiles.csv"),
            "time",
            {
                "m", "a", "lambda", "mu", "lambda_t", "rho",
                "p_radial", "p_tangential_sum", "j",
            },
        )
        ef = RectilinearProfile(
            os.path.join(args.ef, "high_compactness_profiles.csv"),
            "advanced_time",
            {
                "a", "b", "m", "T_rr", "S", "T_vv", "T_vr",
                "T_theta_theta",
            },
        )
    except (IOError, OSError, ValueError, ProfileError, json.JSONDecodeError) as error:
        print("dynamic parity input error: {}".format(error), file=sys.stderr)
        return 1

    issues = []
    try:
        manifest_path, manifest, manifest_audit, manifest_issues = (
            load_coordinate_manifest(args, sc_summary, ef_summary)
        )
    except ProfileError as error:
        manifest_path, manifest, manifest_audit = None, {}, {}
        manifest_issues = [str(error)]
    issues.extend(manifest_issues)
    donor_history_audit = None
    _donor_name, donor_record = artifact_by_alias(
        manifest_audit.get("artifacts", {}),
        ("donor_history", "history", "spacetime_history"),
    )
    if donor_record is not None and donor_record.get("verified") is True:
        try:
            donor_history_audit = prepend_time_reversed_donor_history(
                sc, donor_record["path"]
            )
        except (IOError, OSError, ProfileError) as error:
            issues.append("dense donor history failed: {}".format(error))
    else:
        issues.append("no verified dense donor history is available for tau<0")
    interpolation_audit = {
        "schwarzschild": (
            PRODUCTION_METRIC_INTERPOLATION["schwarzschild"]
            if sc.has_production_metric_reconstruction
            else "legacy_center_center_linear"
        ),
        "eddington_finkelstein": (
            PRODUCTION_METRIC_INTERPOLATION["eddington_finkelstein"]
            if ef.has_production_metric_reconstruction
            else "legacy_center_center_linear"
        ),
        "donor_history_format_version": (
            donor_history_audit.get("format_version")
            if donor_history_audit is not None else None
        ),
        "production_equivalent": (
            sc.has_production_metric_reconstruction
            and ef.has_production_metric_reconstruction
            and donor_history_audit is not None
            and donor_history_audit.get("production_metric_reconstruction")
            is True
        ),
    }
    if not interpolation_audit["production_equivalent"]:
        issues.append(
            "saved spacetime lacks the complete center/face metric "
            "reconstruction used by the production SC and EF samplers"
        )
    mapped_flags = (
        sc_summary.get("eikonal_mapped_initial_data") is True
        and ef_summary.get("eikonal_mapped_initial_data") is True
    )
    if not mapped_flags:
        issues.append(
            "both run summaries must assert eikonal-mapped initial data"
        )
    sc_base = normalized_fingerprint(sc_summary.get("base_ic_fingerprint"))
    ef_base = normalized_fingerprint(ef_summary.get("base_ic_fingerprint"))
    provenance = {
        "schwarzschild_summary": sc_summary_path,
        "eddington_finkelstein_summary": ef_summary_path,
        "base_ic_fingerprint_match": sc_base is not None and sc_base == ef_base,
        "base_ic_fingerprint": sc_base,
        "schwarzschild_realized_ic_fingerprint": normalized_fingerprint(
            sc_summary.get("realized_ic_fingerprint")
        ),
        "eddington_finkelstein_realized_ic_fingerprint": normalized_fingerprint(
            ef_summary.get("realized_ic_fingerprint")
        ),
        "realized_ic_fingerprints_are_not_compared_for_equality": True,
        "realized_ic_reason": (
            "the coordinate covector arrays differ; each hash is matched to "
            "its own authenticated manifest field"
        ),
        "eikonal_mapped_initial_data_flags": mapped_flags,
        "coordinate_map_manifest": manifest_audit,
        "dense_time_reversed_donor_history": donor_history_audit,
    }
    if not provenance["base_ic_fingerprint_match"]:
        issues.append("base initial-condition fingerprints do not match")
    convergence_audit, convergence_issues = load_convergence_evidence(
        args, manifest_audit, sc_summary, ef_summary
    )
    issues.extend(convergence_issues)

    manifest_boundary = nested_value(
        manifest,
        "map_worldtube_radius",
        "eikonal_boundary_radius",
        "outer_boundary_radius",
        "outer_radius",
        "mapping.outer_radius",
        "eikonal.outer_radius",
        "boundary.radius",
    )
    boundary_values = (
        sc_summary.get("eikonal_boundary_radius"),
        ef_summary.get("eikonal_boundary_radius"),
        manifest_boundary,
    )
    if manifest.get("schema_version") == 2:
        evolution_values = (
            sc_summary.get("eikonal_evolution_outer_radius"),
            ef_summary.get("eikonal_evolution_outer_radius"),
            manifest.get("evolution_outer_radius"),
        )
        evolution_is_valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool) and finite(value)
            for value in evolution_values
        ) and all(close(evolution_values[0], value, scale=1.0)
                  for value in evolution_values[1:])
        extension_flags = (
            sc_summary.get("eikonal_analytic_vacuum_extension"),
            ef_summary.get("eikonal_analytic_vacuum_extension"),
            manifest.get("analytic_vacuum_extension"),
        )
        if not evolution_is_valid:
            issues.append(
                "coordinate-map v2 metadata has no common evolution boundary"
            )
        if len(set(extension_flags)) != 1 \
                or not isinstance(extension_flags[0], bool):
            issues.append(
                "coordinate-map v2 analytic-extension flags disagree"
            )
    manifest_offset = nested_value(
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
    manifest_offsets = (
        sc_summary.get("eikonal_v_offset"),
        ef_summary.get("eikonal_v_offset"),
        manifest_offset,
    )
    boundary_is_valid = all(
        isinstance(value, (int, float)) and finite(value)
        for value in boundary_values
    ) and all(close(boundary_values[0], value, scale=1.0)
              for value in boundary_values[1:])
    offset_is_valid = all(
        isinstance(value, (int, float)) and finite(value)
        for value in manifest_offsets
    ) and all(close(manifest_offsets[0], value, scale=1.0)
              for value in manifest_offsets[1:])
    if mapped_flags and not boundary_is_valid:
        issues.append("coordinate-map manifest metadata has no common boundary radius")
    if mapped_flags and not offset_is_valid:
        issues.append("coordinate-map manifest metadata has no common v offset")
    manifest_v_offset = float(manifest_offsets[0]) if offset_is_valid else 0.0
    selected_v_offset = (
        manifest_v_offset if args.v_offset is None else args.v_offset
    )
    if args.v_offset is not None and offset_is_valid and not close(
            args.v_offset, manifest_v_offset, scale=1.0):
        issues.append(
            "command-line v offset disagrees with the coordinate-map manifest"
        )

    outer_worldtube_mode = None
    outer_worldtube_audit = None
    if not close(sc.radii[-1], ef.radii[-1], scale=1.0):
        issues.append(
            "SC and EF outermost cell centers differ"
        )
    if boundary_is_valid:
        try:
            boundary_radius = float(boundary_values[0])
            profile_outer_radius = (
                float(manifest.get("evolution_outer_radius"))
                if manifest.get("schema_version") == 2
                else boundary_radius
            )
            outer_worldtube_mode = (
                "exact_saved_outer_face"
                if close(sc.radii[-1], boundary_radius, scale=1.0)
                and close(ef.radii[-1], boundary_radius, scale=1.0)
                else "interior_map_worldtube_with_authenticated_vacuum_tail"
            )
            sc.configure_vacuum_outer_continuation(
                profile_outer_radius, "schwarzschild"
            )
            ef.configure_vacuum_outer_continuation(
                profile_outer_radius, "eddington_finkelstein"
            )
            normalization_tolerance = max(
                1.0e-8, min(args.composition_tolerance, 1.0e-5)
            )
            outer_worldtube_audit = {
                "representation": outer_worldtube_mode,
                "manifest_boundary_radius": boundary_radius,
                "schwarzschild": audit_saved_vacuum_anchor(
                    sc, "schwarzschild", args.outer_vacuum_tolerance,
                    normalization_tolerance, boundary_radius,
                ),
                "eddington_finkelstein": audit_saved_vacuum_anchor(
                    ef, "eddington_finkelstein",
                    args.outer_vacuum_tolerance, normalization_tolerance,
                    boundary_radius,
                ),
            }
        except ProfileError as error:
            issues.append(str(error))

    common_inner = max(sc.radii[0], ef.radii[0], args.r_min)
    common_outer = min(sc.radii[-1], ef.radii[-1], args.r_max)
    if boundary_is_valid:
        common_outer = min(common_outer, float(boundary_values[0]))
    if not common_outer > common_inner:
        issues.append("SC and EF profiles have no requested radial overlap")

    distribution_audits = []
    distribution_failures = []
    if args.empirical_distribution != "strict":
        issues.append(
            "non-strict empirical-distribution mode is diagnostic only"
        )
    artifacts = manifest_audit.get("artifacts", {})
    crossing_name, crossing_record = artifact_by_alias(
        artifacts,
        ("crossings", "crossings_csv", "marker_crossings", "crossing_audit"),
    )
    crossing_path = (
        os.path.abspath(args.marker_crossings)
        if args.marker_crossings else (
            crossing_record["path"] if crossing_record is not None else None
        )
    )
    if args.marker_crossings and (
            crossing_record is None
            or os.path.realpath(crossing_path)
            != os.path.realpath(crossing_record["path"])):
        issues.append(
            "--marker-crossings is not the manifest-hashed crossing artifact"
        )
    if args.empirical_distribution == "strict" and crossing_path is None:
        issues.append(
            "strict empirical-f certification requires a hashed marker crossing audit"
        )
    if crossing_path is not None and args.empirical_distribution != "off":
        try:
            crossing_audit = audit_marker_distribution(crossing_path, args)
            crossing_audit["kind"] = "mapped_initial_null_hypersurface"
            crossing_audit["manifest_artifact"] = crossing_name
            distribution_audits.append(crossing_audit)
            distribution_failures.extend(crossing_audit["failed_fields"])
            expected_markers = manifest_audit.get("mapped_marker_count")
            if isinstance(expected_markers, int) \
                    and crossing_audit["samples"] != expected_markers:
                issues.append(
                    "crossing audit row count does not match the manifest marker count"
                )
        except (IOError, OSError, ProfileError) as error:
            issues.append("marker crossing audit failed: {}".format(error))

    trajectory_name, trajectory_record = artifact_by_alias(
        artifacts,
        (
            "sampled_marker_trajectories", "marker_trajectories",
            "trajectory_crossings", "dynamic_marker_audit",
        ),
    )
    if trajectory_record is not None and args.empirical_distribution != "off":
        try:
            trajectory_audit = audit_marker_distribution(
                trajectory_record["path"], args, allow_repeated_ids=True
            )
            trajectory_audit["kind"] = "later_mapped_null_hypersurfaces"
            trajectory_audit["manifest_artifact"] = trajectory_name
            distribution_audits.append(trajectory_audit)
            distribution_failures.extend(trajectory_audit["failed_fields"])
        except (IOError, OSError, ProfileError) as error:
            issues.append("sampled marker trajectory audit failed: {}".format(error))

    initial_map_audit = None
    initial_map_failures = []
    initial_map_path = os.path.join(
        os.path.abspath(args.ef), "initial_coordinate_map_residuals.csv"
    )
    try:
        initial_map_audit = audit_initial_coordinate_map(
            initial_map_path, args
        )
        initial_map_failures.extend(initial_map_audit["failed_fields"])
    except (IOError, OSError, ProfileError) as error:
        issues.append("initial mapped-slice audit failed: {}".format(error))

    cadence = {
        "schwarzschild_max_time_gap": sc.max_time_gap,
        "eddington_finkelstein_max_time_gap": ef.max_time_gap,
        "schwarzschild_min_radial_gap": sc.min_radial_gap,
        "eddington_finkelstein_min_radial_gap": ef.min_radial_gap,
    }
    # Compare saved-time spacing with radial light-crossing scales before trusting time-interpolated coordinate traces.
    cadence["schwarzschild_time_gap_over_dr"] = (
        sc.max_time_gap / sc.min_radial_gap
    )
    cadence["eddington_finkelstein_time_gap_over_dr"] = (
        ef.max_time_gap / ef.min_radial_gap
    )
    for name in (
            "schwarzschild_time_gap_over_dr",
            "eddington_finkelstein_time_gap_over_dr"):
        if cadence[name] > args.max_cadence_to_dr:
            issues.append(
                "{}={} exceeds {}".format(
                    name, cadence[name], args.max_cadence_to_dr
                )
            )

    raw_time_indices = evenly_spaced_indices(len(sc.times), args.time_samples)
    clock_tolerance = 8.0e-13 * max(
        1.0, abs(ef.times[0]), abs(ef.times[-1]), abs(selected_v_offset)
    )
    time_indices = [
        time_index for time_index in raw_time_indices
        if ef.times[0] - clock_tolerance
        <= sc.times[time_index] + selected_v_offset
        <= ef.times[-1] + clock_tolerance
    ]
    outer_clock_excluded_time_slices = (
        len(raw_time_indices) - len(time_indices)
    )
    eligible_radii = [
        (index, radius) for index, radius in enumerate(sc.radii)
        if common_inner <= radius <= common_outer
    ]
    radial_slots = evenly_spaced_indices(
        len(eligible_radii), min(args.radial_samples, len(eligible_radii))
    ) if eligible_radii else []
    selected_radii = [eligible_radii[index][1] for index in radial_slots]

    accumulator = ComparisonAccumulator()
    eikonal_errors = []
    eikonal_time_errors = []
    eikonal_log_a_errors = []
    inverse_eikonal_errors = []
    inverse_eikonal_time_errors = []
    inverse_eikonal_log_f_errors = []
    composition_time_errors = []
    composition_jacobian_errors = []
    event_states = {}
    successful_events = {}
    forward_worst_step_refinement_event = None
    inverse_worst_step_refinement_event = None
    excluded = {
        "outside_outer_clock_overlap": (
            outer_clock_excluded_time_slices * len(selected_radii)
        ),
        "outside_sc_eikonal_domain": 0,
        "outside_ef_domain": 0,
        "forward_eikonal_not_converged": 0,
        "forward_eikonal_error": 0,
        "inverse_eikonal_not_converged": 0,
        "inverse_eikonal_error": 0,
        "composition_not_closed": 0,
        "at_or_beyond_horizon": 0,
        "sc_metric_identity_failed": 0,
        "outside_maximal_contiguous_common_domain": 0,
    }
    # At every sampled SC event, obtain independent forward/inverse traces, quantify their refinement and composition, then sample EF at the mapped event.
    for time_index in time_indices:
        tau = sc.times[time_index]
        for radial_slot, radius in enumerate(selected_radii):
            event_key = (time_index, radial_slot)
            event_base = {
                "time_index": time_index,
                "radial_slot": radial_slot,
                "tau": tau,
                "r": radius,
            }
            try:
                coarse_v, coarse_a, coarse_log_a = trace_to_outer_boundary(
                    sc, tau, radius, args.eikonal_substeps,
                    selected_v_offset,
                    float(boundary_values[0]) if boundary_is_valid else None,
                )
                mapped_v, map_a, map_log_a = trace_to_outer_boundary(
                    sc, tau, radius, 2 * args.eikonal_substeps,
                    selected_v_offset,
                    float(boundary_values[0]) if boundary_is_valid else None,
                )
            except OutsideProfile:
                excluded["outside_sc_eikonal_domain"] += 1
                event_states[event_key] = dict(
                    event_base, status="outside_sc_eikonal_domain"
                )
                continue
            except ProfileError as error:
                issues.append(str(error))
                excluded["forward_eikonal_error"] += 1
                event_states[event_key] = dict(
                    event_base,
                    status="forward_eikonal_error",
                    error=str(error),
                )
                continue
            eikonal_time_error = rk4_step_doubling_change(
                coarse_v - selected_v_offset,
                mapped_v - selected_v_offset,
            )
            eikonal_log_a_error = rk4_step_doubling_change(
                coarse_log_a, map_log_a
            )
            eikonal_error = max(eikonal_time_error, eikonal_log_a_error)
            eikonal_time_errors.append(eikonal_time_error)
            eikonal_log_a_errors.append(eikonal_log_a_error)
            eikonal_errors.append(eikonal_error)
            diagnostic_h, _diagnostic_h_time = (
                sc.sample_eikonal_h_with_time_derivative(tau, radius)
            )
            forward_diagnostic = dict(
                event_base,
                a=sc.sample("a", tau, radius),
                h=diagnostic_h,
                coarse_v=coarse_v,
                fine_v=mapped_v,
                coarse_log_A=coarse_log_a,
                fine_log_A=map_log_a,
                forward_time_step_refinement_error=eikonal_time_error,
                forward_log_A_step_refinement_error=eikonal_log_a_error,
                forward_step_refinement_error=eikonal_error,
            )
            if forward_worst_step_refinement_event is None or (
                    eikonal_error
                    > forward_worst_step_refinement_event[
                        "forward_step_refinement_error"]):
                forward_worst_step_refinement_event = forward_diagnostic
            if eikonal_error > args.eikonal_convergence_tolerance:
                excluded["forward_eikonal_not_converged"] += 1
                event_states[event_key] = dict(
                    forward_diagnostic,
                    status="forward_eikonal_not_converged",
                )
                continue
            try:
                coarse_tau, coarse_f, coarse_log_f = (
                    inverse_trace_to_outer_boundary(
                    ef, mapped_v, radius, args.eikonal_substeps,
                    selected_v_offset,
                    float(boundary_values[0]) if boundary_is_valid else None,
                    )
                )
                mapped_tau, map_f, map_log_f = (
                    inverse_trace_to_outer_boundary(
                    ef, mapped_v, radius, 2 * args.eikonal_substeps,
                    selected_v_offset,
                    float(boundary_values[0]) if boundary_is_valid else None,
                    )
                )
            except OutsideProfile:
                excluded["outside_ef_domain"] += 1
                event_states[event_key] = dict(
                    forward_diagnostic, status="outside_ef_domain"
                )
                continue
            except ProfileError as error:
                if "a<=0 or b<=0" in str(error):
                    excluded["at_or_beyond_horizon"] += 1
                    event_status = "at_or_beyond_horizon"
                else:
                    issues.append(str(error))
                    excluded["inverse_eikonal_error"] += 1
                    event_status = "inverse_eikonal_error"
                event_states[event_key] = dict(
                    forward_diagnostic,
                    status=event_status,
                    error=str(error),
                )
                continue
            inverse_time_error = rk4_step_doubling_change(
                coarse_tau + selected_v_offset,
                mapped_tau + selected_v_offset,
            )
            inverse_log_f_error = rk4_step_doubling_change(
                coarse_log_f, map_log_f
            )
            inverse_error = max(inverse_time_error, inverse_log_f_error)
            inverse_eikonal_time_errors.append(inverse_time_error)
            inverse_eikonal_log_f_errors.append(inverse_log_f_error)
            inverse_eikonal_errors.append(inverse_error)
            inverse_diagnostic = dict(
                event_base,
                a=forward_diagnostic["a"],
                coarse_tau=coarse_tau,
                fine_tau=mapped_tau,
                coarse_log_F=coarse_log_f,
                fine_log_F=map_log_f,
                inverse_time_step_refinement_error=inverse_time_error,
                inverse_log_F_step_refinement_error=inverse_log_f_error,
                inverse_step_refinement_error=inverse_error,
            )
            if inverse_worst_step_refinement_event is None or (
                    inverse_error
                    > inverse_worst_step_refinement_event[
                        "inverse_step_refinement_error"]):
                inverse_worst_step_refinement_event = inverse_diagnostic
            if inverse_error > args.eikonal_convergence_tolerance:
                excluded["inverse_eikonal_not_converged"] += 1
                event_states[event_key] = dict(
                    forward_diagnostic,
                    status="inverse_eikonal_not_converged",
                    inverse_step_refinement_error=inverse_error,
                    inverse_time_step_refinement_error=inverse_time_error,
                    inverse_log_F_step_refinement_error=inverse_log_f_error,
                )
                continue
            time_scale = max(1.0, abs(tau), abs(mapped_tau))
            composition_time_error = abs(mapped_tau - tau) / time_scale
            composition_jacobian_error = jacobian_composition_error(
                map_log_a, map_log_f
            )
            composition_time_errors.append(composition_time_error)
            composition_jacobian_errors.append(composition_jacobian_error)
            if max(composition_time_error, composition_jacobian_error) \
                    > args.composition_tolerance:
                excluded["composition_not_closed"] += 1
                event_states[event_key] = dict(
                    forward_diagnostic,
                    status="composition_not_closed",
                    inverse_step_refinement_error=inverse_error,
                    inverse_time_step_refinement_error=inverse_time_error,
                    inverse_log_F_step_refinement_error=inverse_log_f_error,
                    composition_time_error=composition_time_error,
                    composition_jacobian_error=composition_jacobian_error,
                )
                continue
            try:
                lam = sc.sample("lambda", tau, radius)
                mu = sc.sample("mu", tau, radius)
                saved_a = sc.sample("a", tau, radius)
                mass = sc.sample("m", tau, radius)
                rho = sc.sample("rho", tau, radius)
                pressure = sc.sample("p_radial", tau, radius)
                tangential_pressure = sc.sample(
                    "p_tangential_sum", tau, radius
                )
                current = sc.sample("j", tau, radius)
                ef_a = ef.sample("a", mapped_v, radius)
                ef_b = ef.sample("b", mapped_v, radius)
                ef_mass = ef.sample("m", mapped_v, radius)
                ef_t_rr = ef.sample("T_rr", mapped_v, radius)
                ef_s = ef.sample("S", mapped_v, radius)
                ef_t_vv = ef.sample("T_vv", mapped_v, radius)
                ef_t_vr = ef.sample("T_vr", mapped_v, radius)
                ef_t_theta_theta = ef.sample(
                    "T_theta_theta", mapped_v, radius
                )
            except OutsideProfile:
                excluded["outside_ef_domain"] += 1
                event_states[event_key] = dict(
                    forward_diagnostic, status="outside_ef_domain"
                )
                continue

            sc_a = math.exp(-2.0 * lam)
            if min(sc_a, saved_a, ef_a) <= args.minimum_a:
                excluded["at_or_beyond_horizon"] += 1
                event_states[event_key] = dict(
                    forward_diagnostic, status="at_or_beyond_horizon"
                )
                continue
            if abs(saved_a - sc_a) > 5.0e-11 * max(1.0, abs(sc_a)):
                issues.append("SC profile violates a=exp(-2 lambda)")
                excluded["sc_metric_identity_failed"] += 1
                event_states[event_key] = dict(
                    forward_diagnostic, status="sc_metric_identity_failed"
                )
                continue
            # The coordinate Jacobian transforms the complete covariant metric. Zero rr and inverse vv components test the null-coordinate structure.
            predicted_b = math.exp(lam + mu) / map_a
            predicted_g_vv = -math.exp(2.0 * mu) / (map_a * map_a)
            h_value = math.exp(lam - mu)
            predicted_g_vr = (
                math.exp(2.0 * mu) * h_value / map_a
            )
            predicted_g_rr = (
                math.exp(2.0 * lam)
                - math.exp(2.0 * mu) * h_value * h_value
            )
            predicted_inverse_g_vv = (
                -math.exp(-2.0 * mu) * map_a * map_a
                + math.exp(-2.0 * lam)
                    * (h_value * map_a) * (h_value * map_a)
            )
            predicted_inverse_g_vr = (
                math.exp(-2.0 * lam) * h_value * map_a
            )
            reference_g_vv = -ef_a * ef_b * ef_b
            # Convert local observer moments into SC covariant stress components before applying the two coordinate Jacobian factors.
            sc_t_tautau = math.exp(2.0 * mu) * rho
            sc_t_taur = -math.exp(lam + mu) * current
            sc_t_rr = math.exp(2.0 * lam) * pressure
            map_f_forward = 1.0 / map_a
            predicted_t_rr = (
                sc_t_rr - 2.0 * h_value * sc_t_taur
                + h_value * h_value * sc_t_tautau
            )
            predicted_s = rho - pressure
            predicted_t_vv = map_f_forward * map_f_forward * sc_t_tautau
            predicted_t_vr = map_f_forward * (
                sc_t_taur - h_value * sc_t_tautau
            )
            # The saved tangential pressure is the sum of two equal spherical components, so one angular covariant component carries one-half r squared.
            predicted_t_theta_theta = (
                0.5 * radius * radius * tangential_pressure
            )

            accumulator.add("metric_a", sc_a, ef_a)
            accumulator.add("metric_b", predicted_b, ef_b)
            accumulator.add("metric_g_vv", predicted_g_vv, reference_g_vv)
            accumulator.add("metric_g_vr", predicted_g_vr, ef_b)
            accumulator.add("metric_g_rr", predicted_g_rr, 0.0)
            accumulator.add(
                "metric_inverse_g_vr", predicted_inverse_g_vr, 1.0 / ef_b
            )
            accumulator.add("metric_inverse_g_rr", sc_a, ef_a)
            accumulator.add(
                "metric_inverse_g_vv", predicted_inverse_g_vv, 0.0
            )
            accumulator.add(
                "coordinate_jacobian_tau_v", 1.0 / map_a, map_f
            )
            accumulator.add("misner_sharp_mass", mass, ef_mass)
            accumulator.add("stress_T_rr", predicted_t_rr, ef_t_rr)
            accumulator.add("stress_S", predicted_s, ef_s)
            accumulator.add("stress_T_vv", predicted_t_vv, ef_t_vv)
            accumulator.add("stress_T_vr", predicted_t_vr, ef_t_vr)
            accumulator.add(
                "stress_T_theta_theta",
                predicted_t_theta_theta,
                ef_t_theta_theta,
            )
            # Invert EF stress definitions into static-observer density, current and pressure for an independent moment-level comparison in the overlap.
            reconstructed_rho = ef_t_vv / (ef_a * ef_b * ef_b)
            reconstructed_current = -ef_t_vr / ef_b - reconstructed_rho
            reconstructed_pressure = reconstructed_rho - ef_s
            reconstructed_tangential_pressure = (
                2.0 * ef_t_theta_theta / (radius * radius)
            )
            accumulator.add("moment_rho", rho, reconstructed_rho)
            accumulator.add(
                "moment_p_radial", pressure, reconstructed_pressure
            )
            accumulator.add("moment_j", current, reconstructed_current)
            accumulator.add(
                "moment_p_tangential_sum",
                tangential_pressure,
                reconstructed_tangential_pressure,
            )
            event_states[event_key] = dict(
                forward_diagnostic,
                status="mapped",
                inverse_step_refinement_error=inverse_error,
                inverse_time_step_refinement_error=inverse_time_error,
                inverse_log_F_step_refinement_error=inverse_log_f_error,
                composition_time_error=composition_time_error,
                composition_jacobian_error=composition_jacobian_error,
            )
            successful_events[event_key] = {
                "event": {
                    "tau": tau,
                    "v": mapped_v,
                    "r": radius,
                    "v_tau": map_a,
                    "log_v_tau": map_log_a,
                    "tau_v": map_f,
                    "log_tau_v": map_log_f,
                    "inverse_tau": mapped_tau,
                },
                "pairs": {
                    name: values[-1]
                    for name, values in accumulator.pairs.items()
                },
            }

    # Resolve connected physical support before computing final comparison norms and coverage fractions.
    common_domain = audit_contiguous_common_domain(
        time_indices,
        sc.times,
        selected_radii,
        event_states,
        args.minimum_radial_coverage,
    )
    domain_keys = common_domain.pop("_domain_keys")
    domain_numeric = summarize_domain_numeric_errors(
        event_states, domain_keys
    )
    excluded["outside_maximal_contiguous_common_domain"] = sum(
        key not in domain_keys for key in successful_events
    )
    # Rebuild field comparisons using only events in the declared common domain, preserving excluded events in their separate diagnostics.
    accumulator = ComparisonAccumulator()
    mapped_events = []
    compared_times = set()
    compared_radii = set()
    for time_index in time_indices:
        for radial_slot, radius in enumerate(selected_radii):
            event_key = (time_index, radial_slot)
            if event_key not in domain_keys or event_key not in successful_events:
                continue
            successful = successful_events[event_key]
            for name, pair in successful["pairs"].items():
                accumulator.add(name, pair[0], pair[1])
            mapped_events.append(successful["event"])
            compared_times.add(sc.times[time_index])
            compared_radii.add(radius)

    if not mapped_events:
        issues.append("no same-event, pre-horizon samples were available")
    candidate_events = common_domain["candidate_events"]
    event_coverage = common_domain["event_coverage"]
    time_slice_coverage = common_domain["time_slice_coverage"]
    radial_coverage = common_domain["minimum_per_slice_radial_coverage"]
    if common_domain["unexplained_holes"]:
        issues.append(
            "maximal contiguous common exterior contains {} unexplained "
            "sample holes".format(common_domain["unexplained_holes"])
        )
    if common_domain["time_topology_gap_slices"]:
        issues.append(
            "radially complete common time domain has {} interior gap "
            "slices".format(common_domain["time_topology_gap_slices"])
        )
    if common_domain["disconnected_non_horizon_support_samples"]:
        issues.append(
            "common-domain support re-enters after a non-horizon boundary "
            "at {} samples".format(
                common_domain["disconnected_non_horizon_support_samples"]
            )
        )
    if common_domain["terminal_reason"] == "loss_of_common_support":
        issues.append(
            "maximal common time block ends before the SC chart terminal "
            "without a horizon/chart-cutoff explanation"
        )
    if len(mapped_events) < args.minimum_events:
        issues.append(
            "only {} mapped events are available; {} required".format(
                len(mapped_events), args.minimum_events
            )
        )
    if common_domain["complete_time_slices"] < args.minimum_time_slices:
        issues.append(
            "only {} SC time slices contribute; {} required".format(
                common_domain["complete_time_slices"],
                args.minimum_time_slices,
            )
        )
    if event_coverage < args.minimum_event_coverage:
        issues.append(
            "mapped event coverage {} is below {}".format(
                event_coverage, args.minimum_event_coverage
            )
        )
    if time_slice_coverage < args.minimum_time_coverage:
        issues.append(
            "mapped time-slice coverage {} is below {}".format(
                time_slice_coverage, args.minimum_time_coverage
            )
        )
    if radial_coverage < args.minimum_radial_coverage:
        issues.append(
            "mapped radial coverage {} is below {}".format(
                radial_coverage, args.minimum_radial_coverage
            )
        )
    all_attempted_maximum_eikonal_error = (
        max(eikonal_errors) if eikonal_errors else None
    )
    all_attempted_maximum_eikonal_time_error = (
        max(eikonal_time_errors) if eikonal_time_errors else None
    )
    all_attempted_maximum_eikonal_log_a_error = (
        max(eikonal_log_a_errors) if eikonal_log_a_errors else None
    )
    maximum_eikonal_error = domain_numeric[
        "forward_step_refinement_error"
    ]
    maximum_eikonal_time_error = domain_numeric[
        "forward_time_step_refinement_error"
    ]
    maximum_eikonal_log_a_error = domain_numeric[
        "forward_log_A_step_refinement_error"
    ]
    if maximum_eikonal_error is None or (
            maximum_eikonal_error > args.eikonal_convergence_tolerance):
        issues.append(
            "eikonal step refinement did not meet the requested tolerance"
        )
    all_attempted_maximum_inverse_eikonal_error = (
        max(inverse_eikonal_errors) if inverse_eikonal_errors else None
    )
    all_attempted_maximum_inverse_eikonal_time_error = (
        max(inverse_eikonal_time_errors)
        if inverse_eikonal_time_errors else None
    )
    all_attempted_maximum_inverse_eikonal_log_f_error = (
        max(inverse_eikonal_log_f_errors)
        if inverse_eikonal_log_f_errors else None
    )
    maximum_inverse_eikonal_error = domain_numeric[
        "inverse_step_refinement_error"
    ]
    maximum_inverse_eikonal_time_error = domain_numeric[
        "inverse_time_step_refinement_error"
    ]
    maximum_inverse_eikonal_log_f_error = domain_numeric[
        "inverse_log_F_step_refinement_error"
    ]
    if maximum_inverse_eikonal_error is None or (
            maximum_inverse_eikonal_error
            > args.eikonal_convergence_tolerance):
        issues.append(
            "inverse EF eikonal step refinement did not meet the requested tolerance"
        )
    all_attempted_maximum_composition_time_error = (
        max(composition_time_errors) if composition_time_errors else None
    )
    all_attempted_maximum_composition_jacobian_error = (
        max(composition_jacobian_errors)
        if composition_jacobian_errors else None
    )
    maximum_composition_time_error = domain_numeric[
        "composition_time_error"
    ]
    maximum_composition_jacobian_error = domain_numeric[
        "composition_jacobian_error"
    ]
    if maximum_composition_time_error is None \
            or maximum_composition_jacobian_error is None \
            or max(
                maximum_composition_time_error,
                maximum_composition_jacobian_error,
            ) > args.composition_tolerance:
        issues.append(
            "forward/inverse eikonal composition does not close at tolerance"
        )

    errors = accumulator.errors(args.absolute_floor)
    failures = []
    metric_names = {
        name for name in errors
        if name.startswith("metric_")
        or name.startswith("coordinate_jacobian_")
        or name == "misner_sharp_mass"
    }
    moment_names = {name for name in errors if name.startswith("moment_")}
    for name, diagnostics in errors.items():
        if name in metric_names:
            tolerance = args.metric_tolerance
        elif name in moment_names:
            tolerance = args.moment_tolerance
        else:
            tolerance = args.stress_tolerance
        if diagnostics["relative_l2"] > tolerance \
                or diagnostics["relative_linf"] > tolerance:
            failures.append(name)
    failures.extend(distribution_failures)
    failures.extend(initial_map_failures)
    failures = sorted(set(failures))
    # A meaningful matter comparison requires a resolved nonzero stress signal in addition to metric agreement.
    stress_signal = max(
        [
            max(
                diagnostics["maximum_predicted_magnitude"],
                diagnostics["maximum_reference_magnitude"],
            )
            for name, diagnostics in errors.items()
            if name.startswith("stress_")
        ] or [0.0]
    )
    if stress_signal < args.minimum_stress_signal:
        issues.append(
            "mapped overlap contains no resolved matter stress signal"
        )

    convergence_observed = {
        "metric": max([
            max(value["relative_l2"], value["relative_linf"])
            for name, value in errors.items() if name in metric_names
        ] + [
            max(value["relative_l2"], value["relative_linf"])
            for name, value in (
                initial_map_audit.get("errors", {}).items()
                if initial_map_audit else []
            ) if name.startswith("initial_metric_")
            or name == "initial_misner_sharp_mass"
        ] or [0.0]),
        "stress": max([
            max(value["relative_l2"], value["relative_linf"])
            for name, value in errors.items() if name.startswith("stress_")
        ] + [
            max(value["relative_l2"], value["relative_linf"])
            for name, value in (
                initial_map_audit.get("errors", {}).items()
                if initial_map_audit else []
            ) if name.startswith("initial_stress_")
        ] or [0.0]),
        "moment": max([
            max(value["relative_l2"], value["relative_linf"])
            for name, value in errors.items() if name in moment_names
        ] or [0.0]),
        "distribution": max([
            max(value["relative_l2"], value["relative_linf"])
            for audit in distribution_audits
            for value in audit["errors"].values()
        ] or [0.0]),
    }
    # Each observed category must satisfy its independently measured uncertainty bound as well as the configured physical comparison tolerance.
    for name, bound in convergence_audit.get("uncertainty_bounds", {}).items():
        if convergence_observed.get(name, float("inf")) > bound:
            failures.append("{}_outside_quantified_uncertainty".format(name))
    failures = sorted(set(failures))

    # Missing applicability or coverage evidence leaves the result inconclusive; complete comparisons distinguish numerical disagreement from a passing certificate.
    if issues:
        status = "inconclusive"
    elif failures:
        status = "fail"
    else:
        status = "pass"
    result = {
        "schema": "venus_wpic_dynamic_coordinate_certificate_v2",
        "status": status,
        "equation": "v_r=exp(lambda-mu)*v_tau",
        "inverse_equation": "tau_r+(a*b)^(-1)*tau_v=0",
        "advanced_root": True,
        "outer_boundary_condition": "v(tau,R)=tau+v_offset",
        "outer_worldtube_representation": outer_worldtube_mode,
        "outer_worldtube_validation": outer_worldtube_audit,
        "shifted_clock_interpretation": (
            "for East-style reverse donor s=-tau, the evolved EF clock is "
            "v_hat=v+T with v_offset=T=outer_reverse_time"
        ),
        "v_offset": selected_v_offset,
        "v_offset_source": (
            "coordinate_map_manifest" if args.v_offset is None
            else "command_line_checked_against_manifest"
        ),
        "same_event_rule": "compare SC(tau,r) with EF(v(tau,r),r)",
        "eikonal_substeps": args.eikonal_substeps,
        "certifier_sha256": certifier_sha256,
        "metric_interpolation": interpolation_audit,
        "certified_scope": (
            "pre-horizon same-event metric and stress/moment profiles plus "
            "the manifest-hashed empirical marker distribution; later marker "
            "slices are included when the extractor supplies them"
        ),
        "provenance": provenance,
        "cadence": cadence,
        "mapping": {
            "events": len(mapped_events),
            "candidate_events": candidate_events,
            "raw_sampled_grid_events": (
                len(raw_time_indices) * len(selected_radii)
            ),
            "outer_clock_overlap_sampled_events": (
                len(time_indices) * len(selected_radii)
            ),
            "time_slices": len(compared_times),
            "event_coverage": event_coverage,
            "time_slice_coverage": time_slice_coverage,
            "radial_coverage": radial_coverage,
            "common_domain": common_domain,
            "forward_maximum_step_refinement_error": maximum_eikonal_error,
            "forward_maximum_time_step_refinement_error": (
                maximum_eikonal_time_error
            ),
            "forward_maximum_log_A_step_refinement_error": (
                maximum_eikonal_log_a_error
            ),
            "forward_all_attempted_maximum_step_refinement_error": (
                all_attempted_maximum_eikonal_error
            ),
            "forward_all_attempted_maximum_time_step_refinement_error": (
                all_attempted_maximum_eikonal_time_error
            ),
            "forward_all_attempted_maximum_log_A_step_refinement_error": (
                all_attempted_maximum_eikonal_log_a_error
            ),
            "inverse_maximum_step_refinement_error": (
                maximum_inverse_eikonal_error
            ),
            "inverse_maximum_time_step_refinement_error": (
                maximum_inverse_eikonal_time_error
            ),
            "inverse_maximum_log_F_step_refinement_error": (
                maximum_inverse_eikonal_log_f_error
            ),
            "inverse_all_attempted_maximum_step_refinement_error": (
                all_attempted_maximum_inverse_eikonal_error
            ),
            "inverse_all_attempted_maximum_time_step_refinement_error": (
                all_attempted_maximum_inverse_eikonal_time_error
            ),
            "inverse_all_attempted_maximum_log_F_step_refinement_error": (
                all_attempted_maximum_inverse_eikonal_log_f_error
            ),
            "step_refinement_method": (
                "per-cell RK4 n-versus-2n; accept 2n"
            ),
            "step_refinement_error_definition": (
                "max(abs(delta coordinate time),abs(delta log Jacobian)); "
                "no Richardson division without demonstrated asymptotic order"
            ),
            "positive_jacobian_variables": (
                "log(A)=log(v_tau) and log(F)=log(tau_v)"
            ),
            "forward_worst_step_refinement_event": (
                domain_numeric["forward_worst_step_refinement_event"]
            ),
            "inverse_worst_step_refinement_event": (
                domain_numeric["inverse_worst_step_refinement_event"]
            ),
            "forward_all_attempted_worst_step_refinement_event": (
                forward_worst_step_refinement_event
            ),
            "inverse_all_attempted_worst_step_refinement_event": (
                inverse_worst_step_refinement_event
            ),
            "maximum_composition_time_error": (
                maximum_composition_time_error
            ),
            "maximum_composition_jacobian_error": (
                maximum_composition_jacobian_error
            ),
            "all_attempted_maximum_composition_time_error": (
                all_attempted_maximum_composition_time_error
            ),
            "all_attempted_maximum_composition_jacobian_error": (
                all_attempted_maximum_composition_jacobian_error
            ),
            "composition_rule": (
                "tau(v(tau,r),r)=tau and v_tau*tau_v=1"
            ),
            "excluded": excluded,
            "first_event": mapped_events[0] if mapped_events else None,
            "last_event": mapped_events[-1] if mapped_events else None,
        },
        "metric_pullback": {
            "tau_v": "F=1/A",
            "tau_r_at_fixed_v": "-exp(lambda-mu)",
            "g_vv": "-exp(2*mu)*F^2=-a*b^2",
            "g_vr": "exp(lambda+mu)*F=b",
            "g_rr": "0",
        },
        "stress_pullback": {
            "T_tautau": "exp(2*mu)*rho",
            "T_taur": "-exp(lambda+mu)*j",
            "T_rr_SC": "exp(2*lambda)*p_radial",
            "T_vv": "F^2*T_tautau",
            "T_vr": "F*(T_taur-exp(lambda-mu)*T_tautau)",
            "T_rr_EF": (
                "T_rr_SC-2*exp(lambda-mu)*T_taur"
                "+exp(2*(lambda-mu))*T_tautau"
            ),
            "T_theta_theta": "r^2*p_tangential_sum/2",
            "S": "rho-p_radial",
        },
        "initial_mapped_slice": initial_map_audit,
        "distribution": {
            "mode": args.empirical_distribution,
            "evidence": distribution_audits,
            "failed_fields": sorted(set(distribution_failures)),
            "later_trajectory_evidence_present": trajectory_record is not None,
        },
        "convergence": {
            "evidence": convergence_audit,
            "observed_maximum_relative_error": convergence_observed,
            "single_resolution_is_diagnostic_only": True,
        },
        "tolerances": {
            "metric_relative_l2_and_linf": args.metric_tolerance,
            "stress_relative_l2_and_linf": args.stress_tolerance,
            "moment_relative_l2_and_linf": args.moment_tolerance,
            "distribution_relative_l2_and_linf": (
                args.distribution_tolerance
            ),
            "crossing_residual": args.crossing_residual_tolerance,
            "eikonal_step_refinement": args.eikonal_convergence_tolerance,
            "forward_inverse_composition": args.composition_tolerance,
            "minimum_a": args.minimum_a,
            "minimum_event_coverage": args.minimum_event_coverage,
            "minimum_time_coverage": args.minimum_time_coverage,
            "minimum_radial_coverage": args.minimum_radial_coverage,
        },
        "errors": errors,
        "failed_fields": failures,
        "issues": sorted(set(issues)),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(rendered)
        digest = sha256_file(args.output)
        digest_path = args.sha256_output or args.output + ".sha256"
        with open(digest_path, "w") as stream:
            stream.write("{}  {}\n".format(
                digest, os.path.basename(os.path.abspath(args.output))
            ))
    else:
        sys.stdout.write(rendered)
    return 0 if status == "pass" else INCONCLUSIVE


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schwarzschild", required=True)
    parser.add_argument("--ef", required=True)
    parser.add_argument(
        "--manifest",
        help="SHA-authenticated null-extraction manifest (required for pass)",
    )
    parser.add_argument(
        "--marker-crossings",
        help="override path only when it resolves to the manifest artifact",
    )
    parser.add_argument(
        "--convergence-evidence",
        help=(
            "independent multi-resolution/PIC uncertainty certificate; "
            "without it this pair is diagnostic and cannot pass"
        ),
    )
    parser.add_argument("--convergence-evidence-sha256")
    parser.add_argument("--output")
    parser.add_argument("--sha256-output")
    parser.add_argument("--v-offset", type=float)
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=float("inf"))
    parser.add_argument("--time-samples", type=int, default=32)
    parser.add_argument("--radial-samples", type=int, default=64)
    parser.add_argument("--minimum-events", type=int, default=64)
    parser.add_argument("--minimum-time-slices", type=int, default=3)
    parser.add_argument("--minimum-event-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-time-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-radial-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-marker-samples", type=int, default=64)
    parser.add_argument("--minimum-a", type=float, default=1.0e-3)
    parser.add_argument("--minimum-stress-signal", type=float, default=1.0e-12)
    parser.add_argument("--absolute-floor", type=float, default=1.0e-12)
    parser.add_argument("--max-cadence-to-dr", type=float, default=0.75)
    parser.add_argument("--eikonal-substeps", type=int, default=2)
    parser.add_argument(
        "--eikonal-convergence-tolerance", type=float, default=2.0e-5
    )
    parser.add_argument("--composition-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--metric-tolerance", type=float, default=0.05)
    parser.add_argument("--stress-tolerance", type=float, default=0.15)
    parser.add_argument("--moment-tolerance", type=float, default=0.15)
    parser.add_argument("--distribution-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--crossing-residual-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument("--outer-vacuum-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--empirical-distribution",
        choices=("strict", "optional", "off"),
        default="strict",
        help=(
            "strict requires a manifest-hashed crossing audit; optional uses "
            "one when present; off cannot certify empirical f"
        ),
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.time_samples < 2 or args.radial_samples < 2:
        parser.error("time/radial sample counts must be at least two")
    if args.eikonal_substeps < 1:
        parser.error("--eikonal-substeps must be positive")
    for name in (
            "minimum_radial_coverage", "minimum_event_coverage",
            "minimum_time_coverage"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error("--{} must lie in [0,1]".format(name.replace("_", "-")))
    if args.minimum_marker_samples < 1:
        parser.error("--minimum-marker-samples must be positive")
    if args.sha256_output and not args.output:
        parser.error("--sha256-output requires --output")
    if args.convergence_evidence_sha256 and not args.convergence_evidence:
        parser.error(
            "--convergence-evidence-sha256 requires --convergence-evidence"
        )
    for name in (
            "eikonal_convergence_tolerance", "composition_tolerance",
            "metric_tolerance", "stress_tolerance", "moment_tolerance",
            "distribution_tolerance", "crossing_residual_tolerance",
            "outer_vacuum_tolerance"):
        if not getattr(args, name) > 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
