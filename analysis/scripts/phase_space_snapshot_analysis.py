#!/usr/bin/env python3
# Reduce complete WPIC-v6 particle/field groups into radial phase and orbital-energy histograms.
# Inputs include the actual constant-line-mass gravitational field and its background H(t).
# Outputs are compressed arrays and mass/count/source checks. All velocities in the primary energy are physical u=w+Hr.

"""Reduce complete native WPIC snapshots without changing the simulation.

The radial phase portrait is the eta projection of the mass distribution
function. Its finite-bin value is Mbin/(8*pi^2*Delta-r*Delta-u).
The energy projection uses physical radial velocity u=w+H*r and the saved
gravitational potential. Its weighted histogram is dM/dE. The supported
inputs are the complete version-6 GPU snapshot groups with active positive
mass weights, positive radii, and the saved constant-line-mass field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

NATIVE = np.dtype(
    [
        ("mass", "<f8"),
        ("r", "<f8"),
        ("w", "<f8"),
        ("eta", "<f8"),
        ("internal_energy", "<f8"),
        ("sigma", "<f8"),
        ("id", "<u8"),
        ("source_cell", "<u8"),
        ("state", "u1"),
        ("inner_speed", "<f8"),
        ("remaining_inner_time", "<f8"),
    ]
)
assert NATIVE.itemsize == 81
# Spherical spatial angles and tangential velocity angle give 8*pi^2.
# eta=L^2/2 absorbs L*dL, so dGamma=8*pi^2*dr*du*deta.
PHASE_MEASURE = 8.0 * math.pi**2


def sha(path: Path) -> str:
    """Return the complete SHA256 digest used to identify one input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SavedPotential:
    """Integrate the actual saved piecewise-constant dM/dr from the outer face.

    In a resolved cell M(r)=M_left+lambda*(r-r_left).  The integral of
    G*M(r)/r^2 is analytic.  The unresolved inner mass has the uniform-sphere
    continuation used by the solver, with a separate central point mass.
    """

    def __init__(self, field: dict):
        self.field = field
        self.edge = np.asarray(field["radius_edge"], dtype=float)
        self.center = np.asarray(field["radius_center"], dtype=float)
        self.mass = np.asarray(field["source_mass_edge"], dtype=float)
        self.line = np.asarray(field["source_line_density"], dtype=float)
        self.gravity = float(field["gravity"])
        self.point_mass = float(field["central_mass"])
        assert len(self.edge) == len(self.line) + 1 == len(self.mass)
        assert np.all(np.diff(self.edge) > 0) and self.edge[0] > 0
        # Within one cell, write enclosed mass as M(r)=constant+lambda*r, with lambda=dM/dr.
        self.constant = self.mass[:-1] - self.line * self.edge[:-1]
        # Long-double accumulation removes avoidable cancellation in this
        # analysis.  The source itself remains the saved binary64 field.
        a = self.edge[:-1].astype(np.longdouble)
        b = self.edge[1:].astype(np.longdouble)
        lam = self.line.astype(np.longdouble)
        c = self.mass[:-1].astype(np.longdouble) - lam * a
        # Integrate G*M(r)/r^2 analytically across each cell before accumulating inward from the outer vacuum value.
        integrals = self.gravity * (c * (1 / a - 1 / b) + lam * np.log(b / a))
        outer = -self.gravity * self.mass[-1] / self.edge[-1]
        suffix = np.cumsum(integrals[::-1], dtype=np.longdouble)[::-1]
        self.face_phi = np.asarray(np.r_[outer - suffix, outer], dtype=float)

    def potential(self, radius) -> np.ndarray:
        """Return Phi with its value fixed to zero at infinity."""
        r = np.asarray(radius, dtype=float)
        assert np.all(np.isfinite(r)) and np.all(r >= 0)
        result = np.empty_like(r)
        inner = r < self.edge[0]
        outer = r > self.edge[-1]
        resolved = ~(inner | outer)
        if np.any(resolved):
            q = r[resolved]
            i = np.clip(
                np.searchsorted(self.edge, q, side="right") - 1, 0, len(self.line) - 1
            )
            hi = self.edge[i + 1]
            integral = self.gravity * (
                self.constant[i] * (1 / q - 1 / hi) + self.line[i] * np.log(hi / q)
            )
            result[resolved] = self.face_phi[i + 1] - integral
        if np.any(inner):
            q = r[inner]
            r0 = self.edge[0]
            # Continue hidden inner matter as a uniform sphere. A prescribed central point mass contributes separately.
            coefficient = self.gravity * (self.mass[0] - self.point_mass) / r0**3
            point = np.zeros_like(q)
            if self.point_mass:
                assert np.all(q > 0), "Point-mass potential is singular at the origin"
                point = self.gravity * self.point_mass * (1 / r0 - 1 / q)
            result[inner] = (
                self.face_phi[0] + point + 0.5 * coefficient * (q * q - r0 * r0)
            )
        if np.any(outer):
            result[outer] = -self.gravity * self.mass[-1] / r[outer]
        return result

    def force(self, radius) -> np.ndarray:
        """Return positive dPhi/dr; radial gravitational acceleration is -dPhi/dr."""
        r = np.asarray(radius, dtype=float)
        result = np.empty_like(r)
        inner = r < self.edge[0]
        outer = r > self.edge[-1]
        resolved = ~(inner | outer)
        if np.any(resolved):
            q = r[resolved]
            i = np.clip(
                np.searchsorted(self.edge, q, side="right") - 1, 0, len(self.line) - 1
            )
            # Reconstruct the applied positive potential gradient from the same enclosed mass at the particle radius.
            result[resolved] = (
                self.gravity
                * (self.mass[i] + self.line[i] * (q - self.edge[i]))
                / (q * q)
            )
        if np.any(inner):
            q = r[inner]
            result[inner] = (
                self.gravity * (self.mass[0] - self.point_mass) * q / self.edge[0] ** 3
            )
            if self.point_mass:
                assert np.all(q > 0)
                result[inner] += self.gravity * self.point_mass / (q * q)
        if np.any(outer):
            result[outer] = self.gravity * self.mass[-1] / r[outer] ** 2
        return result

    def checks(self) -> dict:
        phi = self.potential(self.center)
        force = self.force(self.center)
        phi_ref = np.asarray(self.field["phi"])
        force_ref = np.asarray(self.field["phi_r"])
        scale = max(float(np.max(np.abs(phi_ref))), 1e-300)
        phi_error = float(np.max(np.abs(phi - phi_ref)) / scale)
        force_error = float(
            np.max(np.abs(force - force_ref)) / max(np.max(np.abs(force_ref)), 1e-300)
        )
        assert phi_error < 2e-12 and force_error < 2e-12
        # Use a separate finite-difference derivative to check consistency between potential and force interpolation.
        h = self.center * 1e-5
        derivative = (
            self.potential(self.center + h) - self.potential(self.center - h)
        ) / (2 * h)
        derivative_error = float(
            np.max(np.abs(derivative - force)) / max(np.max(np.abs(force)), 1e-300)
        )
        assert derivative_error < 2e-7
        return {
            "saved_center_potential_relative_max_error": phi_error,
            "saved_center_force_relative_max_error": force_error,
            "finite_difference_potential_derivative_relative_max_error": derivative_error,
            "outer_phi": float(self.face_phi[-1]),
            "inner_phi": float(self.face_phi[0]),
        }


def self_checks() -> dict:
    # Analytic eta projection for constant f on a rectangular phase domain.
    """Check the spherical phase measure and finite logarithmic-bin Jacobian."""
    redge = np.geomspace(1.0, 4.0, 33)
    xedge = np.log10(redge)
    uedge = np.linspace(-2.0, 3.0, 41)
    f, eta_width = 2.3, 7.0
    mass = (
        PHASE_MEASURE
        * f
        * eta_width
        * np.diff(redge)[:, None]
        * np.diff(uedge)[None, :]
    )
    recovered = mass / (
        PHASE_MEASURE * np.diff(redge)[:, None] * np.diff(uedge)[None, :]
    )
    # Changing the plotted radius coordinate to log10(r) changes the finite-bin Jacobian, not the physical phase measure.
    projected_log = mass / (np.diff(xedge)[:, None] * np.diff(uedge)[None, :])
    logarithmic_mean_r = np.diff(redge) / np.diff(np.log(redge))
    jacobian = PHASE_MEASURE * math.log(10) * logarithmic_mean_r[:, None] * recovered
    projection_error = float(np.max(np.abs(recovered / (f * eta_width) - 1)))
    log_error = float(np.max(np.abs(projected_log / jacobian - 1)))
    assert projection_error < 5e-15 and log_error < 5e-15
    # An independently integrated source checks potential interpolation.
    edge = np.array([1.0, 1.6, 2.8, 4.0])
    line = np.array([0.2, 0.5, 0.1])
    enclosed = np.r_[0.4, 0.4 + np.cumsum(line * np.diff(edge))]
    mock = {
        "radius_edge": edge.tolist(),
        "radius_center": ((edge[:-1] + edge[1:]) / 2).tolist(),
        "source_mass_edge": enclosed.tolist(),
        "source_line_density": line.tolist(),
        "central_mass": 0.1,
        "gravity": 1.0,
    }
    potential = SavedPotential(mock)
    points = np.array([0.2, 0.9, 1.2, 1.9, 3.4, 6.0])
    node, weight = np.polynomial.legendre.leggauss(48)
    direct = []
    for r in points:
        intervals = [r] + [float(a) for a in edge if a > r]
        if r < edge[-1]:
            value = -enclosed[-1] / edge[-1]
            for a, b in zip(intervals[:-1], intervals[1:]):
                q = 0.5 * (a + b) + 0.5 * (b - a) * node
                value -= 0.5 * (b - a) * float(np.sum(weight * potential.force(q)))
        else:
            value = -enclosed[-1] / r
        direct.append(value)
    quadrature_error = float(np.max(np.abs(potential.potential(points) - direct)))
    assert quadrature_error < 2e-12
    # Uniform radial velocities have p(epsilon) proportional to epsilon^-1/2.
    velocity = np.linspace(-2.0, 2.0, 65536, endpoint=False) + 2.0 / 65536
    energy = 0.5 * velocity * velocity
    eedge = np.linspace(0, 2, 65)
    histogram = np.histogram(
        energy, bins=eedge, weights=np.full(len(energy), 1 / len(energy))
    )[0]
    cdf_error = float(np.max(np.abs(np.cumsum(histogram) - np.sqrt(eedge[1:] / 2))))
    assert cdf_error <= 2 / len(energy)
    return {
        "constant_f_eta_projection_relative_error": projection_error,
        "finite_log_radius_Jacobian_relative_error": log_error,
        "potential_vs_independent_Gauss_quadrature_absolute_error": quadrature_error,
        "uniform_velocity_energy_CDF_max_error": cdf_error,
        "uniform_velocity_energy_CDF_error_limit": 2 / len(energy),
        "uniform_velocity_energy_mass": float(np.sum(histogram)),
        "phase_volume": "8*pi^2*dr*du*deta; u=w+H*r has unit Jacobian at fixed r,t",
    }


def require(condition, message):
    """Reject an unsupported or inconsistent input with a readable error."""
    if not condition:
        raise ValueError(message)


def recorded_digest(value, name):
    """Validate a SHA256 identity recorded by the snapshot writer."""
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Invalid recorded SHA256: " + name,
    )
    return value


def snapshot_file(directory, value):
    """Resolve a snapshot member without following an external path."""
    require(
        isinstance(value, str) and Path(value).name == value,
        "Snapshot members must use local filenames.",
    )
    path = directory / value
    require(
        path.is_file() and not path.is_symlink(), "Missing snapshot member: " + value
    )
    return path


def authenticate(directory: Path, expected=None, run_metadata=None):
    """Validate a complete GPU version-6 particle/field snapshot group.

    Recorded source identities describe the supplied calculation. Optional
    expected digests bind it to an executable, initial state or metadata chosen
    by the reader. The stored potential is checked independently below.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema") == "VENUS_phase_space_snapshot_1",
        "Unsupported phase-space snapshot schema.",
    )
    for key in (
        "particles_sha256",
        "field_sha256",
        "source_restart_sha256",
        "executable_sha256",
        "run_metadata_sha256",
    ):
        recorded_digest(manifest.get(key), key)
    for key, digest in (expected or {}).items():
        if digest is not None:
            require(
                manifest[key] == recorded_digest(digest, key),
                "Expected identity differs: " + key,
            )
    # A manifest records the generating metadata digest. Supplying that file enables an independent byte check.
    metadata_verified = run_metadata is not None
    if metadata_verified:
        require(
            sha(Path(run_metadata)) == manifest["run_metadata_sha256"],
            "Run metadata checksum differs from the snapshot record.",
        )
    native = snapshot_file(directory, manifest["particles_file"])
    field_path = snapshot_file(directory, manifest["field_file"])
    require(sha(native) == manifest["particles_sha256"], "Particle checksum mismatch.")
    require(sha(field_path) == manifest["field_sha256"], "Field checksum mismatch.")
    field = json.loads(field_path.read_text())
    require(
        field.get("schema") == "VENUS_phase_space_field_1", "Unsupported field schema."
    )
    require(
        field.get("radial_coordinate") == "physical_radius"
        and field.get("particle_angular_coordinate") == "eta=L^2/2",
        "Unsupported particle-coordinate convention.",
    )
    require(
        field.get("potential_normalization") == "zero_at_infinity"
        and field.get("poisson_reconstruction") == "constant_line_mass",
        "This reader requires the saved constant-line-mass potential.",
    )
    require(
        field.get("marker_weight") in ("density_contrast_mass", "total_mass")
        and not field.get("subtract_homogeneous_background"),
        "Use positive total-mass or directly supplied contrast-mass snapshots.",
    )
    for key in (
        "time",
        "H",
        "scale_factor",
        "a_ddot_over_a",
        "gravity",
        "central_mass",
        "deposited_mesh_mass",
        "hidden_inner_mass",
    ):
        require(np.isfinite(float(field[key])), "Nonfinite field scalar: " + key)
    require(
        field["gravity"] > 0
        and field["central_mass"] >= 0
        and field["scale_factor"] > 0,
        "Invalid gravitational/background scale.",
    )
    for key in (
        "radius_edge",
        "radius_center",
        "source_mass_edge",
        "source_line_density",
        "phi",
        "phi_r",
    ):
        require(
            np.asarray(field[key], dtype=float).ndim == 1
            and np.all(np.isfinite(np.asarray(field[key], dtype=float))),
            "Nonfinite field array: " + key,
        )
    # Validate radial geometry before associating any source or potential array with a cell.
    edge = np.asarray(field["radius_edge"], dtype=float)
    center = np.asarray(field["radius_center"], dtype=float)
    require(
        edge.ndim == center.ndim == 1
        and len(edge) == len(center) + 1
        and len(center) > 0
        and edge[0] > 0
        and np.all(np.diff(edge) > 0)
        and np.all((center > edge[:-1]) & (center < edge[1:])),
        "Invalid radial mesh.",
    )
    require(
        len(field["source_mass_edge"]) == len(edge)
        and all(
            len(field[k]) == len(center)
            for k in ("source_line_density", "phi", "phi_r")
        ),
        "Invalid field-array dimensions.",
    )
    source = np.asarray(field["source_mass_edge"], dtype=float)
    line = np.asarray(field["source_line_density"], dtype=float)
    # Allow roundoff at the source-mass scale while checking the same constant-line reconstruction across faces.
    tolerance = 128 * np.finfo(float).eps * max(abs(source).max(), 1e-300)
    require(
        np.all(line >= 0)
        and np.all(source >= field["central_mass"] - tolerance)
        and field["hidden_inner_mass"] >= 0
        and field["deposited_mesh_mass"] >= 0,
        "Supported positive weights require a nonnegative deposited source.",
    )
    require(
        np.all(np.abs(np.diff(source) - line * np.diff(edge)) <= tolerance),
        "Enclosed source mass is inconsistent with the cell line density.",
    )
    require(
        abs(source[0] - field["hidden_inner_mass"] - field["central_mass"])
        <= tolerance,
        "The inner source mass differs from the saved hidden and point masses.",
    )
    # The GPU aggregate ledger and face prefix use independent summation orders.
    ledger_tolerance = 2e-11 * max(
        abs(source[-1]), abs(field["deposited_mesh_mass"]), 1e-300
    )
    require(
        abs(source[-1] - source[0] - field["deposited_mesh_mass"]) <= ledger_tolerance,
        "The integrated resolved source differs from deposited marker mass.",
    )
    require(
        type(manifest["marker_count"]) is int
        and type(manifest["step"]) is int
        and np.isfinite(float(manifest["requested_time"]))
        and manifest["requested_time"] >= 0,
        "Invalid manifest count or clock.",
    )
    require(
        float(field["time"]) == float(manifest["time"]), "Field/manifest time mismatch."
    )
    # Read and validate the versioned native header before mapping its fixed-size packed records.
    with native.open("rb") as stream:
        head = stream.read(112)
    require(len(head) == 112, "Truncated version-6 native header.")
    magic, version, endian, count, time, step = struct.unpack_from("<16sIIQdq", head)
    require(
        magic == b"VENUSWVPIC1" + b"\0" * 5 and version == 6 and endian == 0x01020304,
        "Only GPU version-6 native snapshots are supported.",
    )
    require(
        count > 0 and np.isfinite(time) and time >= 0 and step >= 0,
        "Invalid native count or clock.",
    )
    require(
        count == manifest["marker_count"]
        and time == manifest["time"]
        and step == manifest["step"],
        "Native/manifest count or clock mismatch.",
    )
    require(
        native.stat().st_size == 112 + count * NATIVE.itemsize,
        "Native size differs from the complete version-6 payload.",
    )
    # Check carried event ages and positive timescale estimates as part of the complete state format.
    ages = struct.unpack_from("<4d", head, 80)
    require(
        all(np.isfinite(x) and x >= 0 for x in ages[:2])
        and all(x > 0 for x in ages[2:]),
        "Invalid native event clocks.",
    )
    # Map the immutable particle file for bounded block processing, retaining its packed little-endian values.
    particles = np.memmap(native, dtype=NATIVE, mode="r", offset=112, shape=(count,))
    mass_parts = []
    for block in blocks(particles):
        require(
            np.all(block["state"] == 0)
            and np.all(block["mass"] > 0)
            and np.all(block["r"] > 0)
            and np.all(block["eta"] >= 0)
            and np.all(block["r"] <= edge[-1] * (1 + 64 * np.finfo(float).eps)),
            "Only active positive-mass markers at positive radii are supported.",
        )
        for key in ("mass", "r", "w", "eta", "internal_energy"):
            require(np.all(np.isfinite(block[key])), "Nonfinite marker field: " + key)
        for key in ("sigma", "inner_speed", "remaining_inner_time"):
            require(
                np.all(np.isfinite(block[key])) and np.all(block[key] >= 0),
                "Invalid marker field: " + key,
            )
        require(np.all(block["internal_energy"] >= 0), "Negative internal energy.")
        mass_parts.append(float(np.sum(block["mass"], dtype=np.longdouble)))
    require(len(np.unique(particles["id"])) == count, "Repeated marker identity.")
    mass = math.fsum(mass_parts)
    # Close represented particle mass against resolved deposition plus the unresolved inner contribution.
    deposited = float(field["deposited_mesh_mass"]) + float(field["hidden_inner_mass"])
    require(
        deposited > 0 and abs(mass / deposited - 1) < 2e-11,
        "Particle/field mass mismatch.",
    )
    auth = dict(
        input_group=directory.name,
        manifest_sha256=sha(manifest_path),
        particles_sha256=manifest["particles_sha256"],
        field_sha256=manifest["field_sha256"],
        actual_time=time,
        requested_time=manifest["requested_time"],
        step=step,
        marker_count=count,
        native_version=version,
        source_input_sha256=manifest["source_restart_sha256"],
        executable_sha256=manifest["executable_sha256"],
        run_metadata_sha256=manifest["run_metadata_sha256"],
        run_metadata_file_verified=metadata_verified,
    )
    auth["potential_checks"] = SavedPotential(field).checks()
    checks = auth["potential_checks"]
    require(
        checks["saved_center_potential_relative_max_error"] < 2e-12
        and checks["saved_center_force_relative_max_error"] < 2e-12
        and checks["finite_difference_potential_derivative_relative_max_error"] < 2e-7,
        "Saved potential or force differs from the reconstructed source.",
    )
    return manifest, field, particles, auth


def recheck(directory, manifest, auth):
    """Require immutable snapshot bytes throughout both reduction passes."""
    require(
        sha(directory / "manifest.json") == auth["manifest_sha256"],
        "Manifest changed during reduction.",
    )
    for key in ("particles", "field"):
        require(
            sha(snapshot_file(directory, manifest[key + "_file"]))
            == auth[key + "_sha256"],
            key + " changed during reduction.",
        )


def snapshot_directories(paths, maximum_time):
    """Select complete groups from explicit directories or a phase_space parent."""
    groups = []
    for path in paths or []:
        path = Path(path)
        candidates = (
            [path]
            if (path / "manifest.json").is_file()
            else sorted(path.glob("snapshot_*"))
        )
        groups.extend(
            p
            for p in candidates
            if (p / "manifest.json").is_file()
            and float(json.loads((p / "manifest.json").read_text())["requested_time"])
            <= maximum_time
        )
    require(groups, "Supply a snapshot directory or its phase_space parent.")
    groups = sorted(
        groups,
        key=lambda p: float(json.loads((p / "manifest.json").read_text())["time"]),
    )
    require(len({p.resolve() for p in groups}) == len(groups), "Duplicate input group.")
    times = [
        float(json.loads((p / "manifest.json").read_text())["time"]) for p in groups
    ]
    require(
        np.all(np.isfinite(times)) and np.all(np.diff(times) > 0),
        "Snapshot times must increase.",
    )
    require(
        len({p.name for p in groups}) == len(groups),
        "Input group names must be unique.",
    )
    return groups


def add_input_arguments(parser):
    """Add the shared snapshot, identity and output controls."""
    parser.add_argument(
        "--snapshots",
        type=Path,
        nargs="+",
        help="GPU snapshot group(s), or phase_space parent",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-requested-time", type=float, default=float("inf"))
    parser.add_argument(
        "--run-metadata", type=Path, help="optional metadata.json to authenticate"
    )
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-binary-sha256")
    parser.add_argument("--expected-metadata-sha256")
    parser.add_argument(
        "--edges-from", type=Path, help="NPZ with fixed edges from a previous reduction"
    )
    parser.add_argument("--self-check-only", action="store_true")


def expected_identities(args):
    """Map optional reader-selected identities to the native manifest keys."""
    return {
        "source_restart_sha256": args.expected_input_sha256,
        "executable_sha256": args.expected_binary_sha256,
        "run_metadata_sha256": args.expected_metadata_sha256,
    }


def common_identity(records, key):
    """Summarize a shared identity while retaining each snapshot's own record."""
    values = {r[key] for r in records}
    return next(iter(values)) if len(values) == 1 else None


def blocks(particles):
    """Read complete native marker blocks without changing stored values."""
    for start in range(0, len(particles), 262144):
        a = particles[start : start + 262144]
        assert np.all(
            a["state"] == 0
        ), "A non-active state requires a separate carrier projection"
        for key in ("mass", "r", "w", "eta", "internal_energy"):
            assert np.all(np.isfinite(a[key]))
        assert np.all(a["mass"] > 0) and np.all(a["r"] > 0) and np.all(a["eta"] >= 0)
        yield a


def coordinates(a, field, potential):
    """Recover physical radius, radial velocity, and instantaneous orbital energy."""
    r = np.asarray(a["r"])
    w = np.asarray(a["w"])
    eta = np.asarray(a["eta"])
    # Recover physical radial motion before forming the orbital energy. The eta/r^2 term is tangential kinetic energy.
    u = w + float(field["H"]) * r
    phi = potential.potential(r)
    tangential = np.divide(eta, r * r)
    return (
        np.log10(r),
        u,
        0.5 * u * u + tangential + phi,
        0.5 * w * w + tangential + phi,
        phi,
    )


def scan(particles, field, potential) -> dict:
    """Measure full-population ranges, total marker mass, and internal energy."""
    low = np.full(4, np.inf)
    high = np.full(4, -np.inf)
    mass_parts = []
    energy_parts = []
    peculiar_parts = []
    internal_parts = []
    negative_parts = []
    negative_peculiar_parts = []
    for a in blocks(particles):
        x, u, energy, peculiar, phi = coordinates(a, field, potential)
        for j, q in enumerate((x, u, energy, peculiar)):
            assert np.all(np.isfinite(q))
            low[j] = min(low[j], float(q.min()))
            high[j] = max(high[j], float(q.max()))
        m = np.asarray(a["mass"])
        mass_parts.append(float(np.sum(m, dtype=np.longdouble)))
        energy_parts.append(float(np.sum(m * energy, dtype=np.longdouble)))
        peculiar_parts.append(float(np.sum(m * peculiar, dtype=np.longdouble)))
        internal_parts.append(float(np.sum(a["internal_energy"], dtype=np.longdouble)))
        negative_parts.append(float(np.sum(m[energy < 0], dtype=np.longdouble)))
        negative_peculiar_parts.append(
            float(np.sum(m[peculiar < 0], dtype=np.longdouble))
        )
    mass = math.fsum(mass_parts)
    deposited = float(field["deposited_mesh_mass"]) + float(field["hidden_inner_mass"])
    assert abs(mass / deposited - 1) < 2e-11
    return {
        "low": low.tolist(),
        "high": high.tolist(),
        "total_mass": mass,
        "weighted_halo_orbital_energy_sum": math.fsum(energy_parts),
        "weighted_peculiar_energy_sum": math.fsum(peculiar_parts),
        "stored_internal_energy": math.fsum(internal_parts),
        "negative_halo_orbital_energy_mass": math.fsum(negative_parts),
        "negative_peculiar_energy_mass": math.fsum(negative_peculiar_parts),
        "deposited_plus_hidden_mass": deposited,
        "particle_field_relative_mass_difference": mass / deposited - 1,
    }


def histogram(particles, field, potential, xedge, uedge, eedge, pecedge):
    """Bin marker masses in physical radial velocity and logarithmic radius.

    Divide by physical dr du and the spherical 8 pi^2 measure to recover
    the distribution integrated over eta; retain squared weights for N_eff."""
    shape = (len(xedge) - 1, len(uedge) - 1)
    size = int(np.prod(shape))
    mass = np.zeros(size)
    mass2 = np.zeros(size)
    count = np.zeros(size, dtype=np.int64)
    energy_mass = np.zeros(len(eedge) - 1)
    peculiar_mass = np.zeros(len(pecedge) - 1)
    energy_moment = np.zeros_like(energy_mass)
    for a in blocks(particles):
        x, u, energy, peculiar, phi = coordinates(a, field, potential)
        m = np.asarray(a["mass"])
        ix = np.searchsorted(xedge, x, side="right") - 1
        iu = np.searchsorted(uedge, u, side="right") - 1
        assert np.all((ix >= 0) & (ix < shape[0]) & (iu >= 0) & (iu < shape[1]))
        # Flatten the radial/velocity bin pair so bincount can accumulate complete weighted histograms.
        flat = ix * shape[1] + iu
        mass += np.bincount(flat, weights=m, minlength=size)
        # Squared weights give N_eff=(sum m)^2/sum(m^2), separately from the raw marker count.
        mass2 += np.bincount(flat, weights=m * m, minlength=size)
        count += np.bincount(flat, minlength=size)
        ie = np.searchsorted(eedge, energy, side="right") - 1
        ip = np.searchsorted(pecedge, peculiar, side="right") - 1
        assert np.all((ie >= 0) & (ie < len(energy_mass))) and np.all(
            (ip >= 0) & (ip < len(peculiar_mass))
        )
        energy_mass += np.bincount(ie, weights=m, minlength=len(energy_mass))
        energy_moment += np.bincount(ie, weights=m * energy, minlength=len(energy_mass))
        peculiar_mass += np.bincount(ip, weights=m, minlength=len(peculiar_mass))
    mass = mass.reshape(shape)
    mass2 = mass2.reshape(shape)
    # Use physical radial widths to recover the eta-integrated distribution from mass bins.
    dr = np.diff(10**xedge)
    du = np.diff(uedge)
    dx = np.diff(xedge)
    f_eta = mass / (PHASE_MEASURE * dr[:, None] * du[None, :])
    # Keep the mass density per log-radius interval separately from the distribution per physical dr.
    mass_log_radius = mass / (dx[:, None] * du[None, :])
    neff = np.divide(mass * mass, mass2, out=np.zeros_like(mass), where=mass2 > 0)
    return {
        "xedge": xedge,
        "redge": 10**xedge,
        "uedge": uedge,
        "energy_edge": eedge,
        "peculiar_energy_edge": pecedge,
        "mass_rw": mass,
        "mass_squared_rw": mass2,
        "count_rw": count.reshape(shape),
        "neff_rw": neff,
        "f_integrated_eta": f_eta,
        "mass_per_log10r_du": mass_log_radius,
        "energy_mass": energy_mass,
        "dM_denergy": energy_mass / np.diff(eedge),
        "energy_first_moment": energy_moment,
        "peculiar_energy_mass": peculiar_mass,
        "dM_dpeculiar_energy": peculiar_mass / np.diff(pecedge),
    }


def padded_edges(low, high, count):
    """Pad the measured range before constructing evenly spaced bin edges."""
    width = max(high - low, abs(low) * 1e-6, 1e-12)
    return np.linspace(low - 1e-9 * width, high + 1e-9 * width, count + 1)


def complete_edges(low, high, count):
    """Choose distinct edges even for a cold constant-coordinate population.

    Ordinary finite ranges use the same padding as the projection kernel.
    A representable fallback span resolves the requested number of bins when
    that padding collapses in binary64. Explicit --edges-from bins take
    precedence in the command-line path.
    """
    edge = padded_edges(low, high, count)
    if not np.all(np.diff(edge) > 0):
        span = max(abs(low), abs(high), 1.0) * max(
            1e-6, 4 * count * np.finfo(float).eps
        )
        center = 0.5 * low + 0.5 * high
        edge = np.linspace(center - 0.5 * span, center + 0.5 * span, count + 1)
    return edge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--radial-bins", type=int, default=360)
    parser.add_argument("--velocity-bins", type=int, default=240)
    parser.add_argument("--energy-bins", type=int, default=1024)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    require(
        min(args.radial_bins, args.velocity_bins, args.energy_bins) >= 2,
        "Use at least two bins.",
    )
    checks = self_checks()
    (args.output_directory / "analytic_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    if args.self_check_only:
        print(json.dumps(checks, indent=2))
        return
    datasets = []
    for directory in snapshot_directories(args.snapshots, args.maximum_requested_time):
        manifest, field, particles, auth = authenticate(
            directory, expected_identities(args), args.run_metadata
        )
        potential = SavedPotential(field)
        auth["potential_checks"] = potential.checks()
        limits = scan(particles, field, potential)
        datasets.append(
            (directory, manifest, field, particles, auth, potential, limits)
        )
    # Choose common histogram edges from all authenticated states so successive arrays share physical bins.
    low = np.min([d[-1]["low"] for d in datasets], axis=0)
    high = np.max([d[-1]["high"] for d in datasets], axis=0)
    if args.edges_from:
        with np.load(args.edges_from, allow_pickle=False) as fixed:
            xedge, uedge, eedge, pecedge = (
                fixed[k].copy()
                for k in ("xedge", "uedge", "energy_edge", "peculiar_energy_edge")
            )
    else:
        xedge = complete_edges(low[0], high[0], args.radial_bins)
        uedge = complete_edges(low[1], high[1], args.velocity_bins)
        eedge = complete_edges(low[2], high[2], args.energy_bins)
        pecedge = complete_edges(low[3], high[3], args.energy_bins)
    for edge in (xedge, uedge, eedge, pecedge):
        require(
            edge.ndim == 1
            and len(edge) >= 3
            and np.all(np.isfinite(edge))
            and np.all(np.diff(edge) > 0),
            "Invalid bin edges.",
        )
    reports = []
    for directory, manifest, field, particles, auth, potential, limits in datasets:
        arrays = histogram(particles, field, potential, xedge, uedge, eedge, pecedge)
        mass = limits["total_mass"]
        # Reintegrate each representation with its own bin widths to verify that the complete mass was retained.
        closures = {
            "phase_mass": float(np.sum(arrays["mass_rw"])),
            "eta_integrated_f_mass": float(
                PHASE_MEASURE
                * np.sum(
                    arrays["f_integrated_eta"]
                    * np.diff(10**xedge)[:, None]
                    * np.diff(uedge)[None, :]
                )
            ),
            "log_radius_mass": float(
                np.sum(
                    arrays["mass_per_log10r_du"]
                    * np.diff(xedge)[:, None]
                    * np.diff(uedge)[None, :]
                )
            ),
            "energy_mass": float(np.sum(arrays["dM_denergy"] * np.diff(eedge))),
            "peculiar_energy_mass": float(
                np.sum(arrays["dM_dpeculiar_energy"] * np.diff(pecedge))
            ),
        }
        relative = {key: value / mass - 1 for key, value in closures.items()}
        require(
            max(abs(x) for x in relative.values()) < 2e-12,
            "Phase/energy mass closure failed.",
        )
        require(
            int(np.sum(arrays["count_rw"])) == manifest["marker_count"],
            "Marker count closure failed.",
        )
        name = directory.name + ".npz"
        np.savez_compressed(args.output_directory / name, **arrays)
        # Rehash the input group after both passes so a changing snapshot cannot produce an accepted reduction.
        recheck(directory, manifest, auth)
        reports.append(
            dict(
                auth,
                **limits,
                cache_file=name,
                cache_sha256=sha(args.output_directory / name),
                mass_projection_closure=closures,
                relative_mass_projection_closure=relative,
                H=field["H"],
                scale_factor=field["scale_factor"],
                a_ddot_over_a=field["a_ddot_over_a"],
            )
        )
    report = dict(
        schema="VENUS_weighted_phase_space_analysis_1",
        analysis_script_sha256=sha(Path(__file__)),
        all_markers_used=True,
        source_input_sha256=common_identity(reports, "source_input_sha256"),
        binary_sha256=common_identity(reports, "executable_sha256"),
        phase_measure="8*pi^2*dr*du*deta; u=w+H*r has unit Jacobian at fixed r,t",
        primary_energy=(
            "E=(w+H*r)^2/2+eta/r^2+Phi_saved; instantaneous center-of-mass "
            "orbital energy"
        ),
        distribution_scope=(
            "Positive mass-weighted active marker distribution; stored internal "
            "energy is separate."
        ),
        energy_projection=(
            "M_bin/Delta_E; phase-space density additionally requires finite "
            "energy-bin phase volume."
        ),
        analytic_checks=checks,
        snapshots=reports,
    )
    (args.output_directory / "analysis_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "snapshots": len(reports),
                "actual_times": [r["actual_time"] for r in reports],
                "maximum_relative_mass_closure": max(
                    abs(x)
                    for r in reports
                    for x in r["relative_mass_projection_closure"].values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
