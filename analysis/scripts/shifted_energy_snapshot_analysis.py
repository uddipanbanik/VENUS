#!/usr/bin/env python3
# Reduce full particle states into shifted-energy mass spectra and phase-volume means.
# Use delta=E-Phi(0)>0 with the saved finite-center gravitational field and physical velocities.
# Outputs distinguish dM/delta, mean f on finite energy bins, sampling support and inner-continuation sensitivity.

"""Full-marker shifted-energy spectra and finite-bin phase-space means.

The companion reader authenticates each complete GPU snapshot group before
the spherical phase-volume integration. Positive attractive source fields
with a finite central potential are required. The native and field files
remain read-only; all reduced arrays are written to the selected directory.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import phase_space_snapshot_analysis as base


class PhaseVolume:
    """Integrate spherical phase volume for an arbitrary monotone potential.

    offset(r)=Phi(r)-Phi(0), edges include the physical origin and wall.
    Omega(delta)=16*pi^2/3 int r^2 [2(delta-offset(r))]_+^(3/2) dr.
    Its derivative is g=16*pi^2 int r^2 sqrt[2(delta-offset(r))]_+ dr.
    The last radial segment uses r=b-(b-a)*(1-z)^2, removing the square-root
    endpoint at an energy turning radius. All cell faces remain quadrature
    boundaries. Energy bins use differences of cumulative phase volumes.
    """

    def __init__(self, offset, edges):
        self.offset = offset
        self.edge = np.asarray(edges, dtype=float)
        assert self.edge[0] == 0 and np.all(np.diff(self.edge) > 0)
        self.face = np.asarray(offset(self.edge))
        assert abs(self.face[0]) < 1e-14
        assert np.all(np.diff(self.face) >= -2e-14 * max(1, float(self.face.max())))

    def turning_radius(self, delta):
        """Find the radial energy boundary by bisection, capped at the saved wall."""
        d = np.asarray(delta, dtype=float)
        assert np.all(d > 0)
        # Bracket each turning radius using monotone saved face potentials before bisection.
        i = np.clip(
            np.searchsorted(self.face, d, side="right") - 1, 0, len(self.edge) - 2
        )
        lo = self.edge[i].copy()
        hi = self.edge[i + 1].copy()
        full = d >= self.face[-1]
        for _ in range(58):
            mid = 0.5 * (lo + hi)
            left = self.offset(mid) < d
            lo = np.where(left, mid, lo)
            hi = np.where(left, hi, mid)
        return np.where(full, self.edge[-1], 0.5 * (lo + hi))

    def integrate(self, delta, order=32):
        """Integrate finite-domain phase volume and its energy derivative."""
        d = np.asarray(delta, dtype=float)
        turn = self.turning_radius(d)
        nodes, weights = np.polynomial.legendre.leggauss(order)
        z = 0.5 * (nodes + 1)
        weights = 0.5 * weights
        cumulative = []
        density = []
        for value, limit in zip(d, turn):
            # Retain every radial cell face as an integration boundary, stopping at the turning radius or physical wall.
            knots = np.r_[self.edge[self.edge < limit], limit]
            a = knots[:-1, None]
            b = knots[1:, None]
            r = a + (b - a) * z
            jac = np.broadcast_to(b - a, r.shape).copy()
            # The last segment ends at the turning radius (or physical wall).
            r[-1] = b[-1] - (b[-1] - a[-1]) * (1 - z) ** 2
            jac[-1] = 2 * (b[-1] - a[-1]) * (1 - z)
            # Only radii with positive available kinetic energy contribute to the velocity-ball phase volume.
            speed = np.sqrt(2 * np.maximum(value - self.offset(r), 0))
            # Combine the spherical radial measure with the coordinate-map Jacobian and quadrature weights.
            measure = r * r * jac * weights
            cumulative.append(
                (16 * math.pi**2 / 3)
                * float(np.sum(measure * speed**3, dtype=np.longdouble))
            )
            density.append(
                16 * math.pi**2 * float(np.sum(measure * speed, dtype=np.longdouble))
            )
        return np.asarray(cumulative), np.asarray(density), turn


def field_model(field):
    """Construct the saved attractive potential and its finite central offset."""
    pot = base.SavedPotential(field)
    assert pot.point_mass == 0, "A central point mass has no finite central potential."
    assert np.all(pot.mass >= 0) and np.all(pot.line >= 0)
    phi_c = float(pot.potential(np.array([0.0]))[0])
    # The uniform inner source gives an exactly harmonic potential difference near the origin.
    omega_squared = pot.gravity * pot.mass[0] / pot.edge[0] ** 3

    def offset(radius):
        r = np.asarray(radius, dtype=float)
        result = pot.potential(r) - phi_c
        # The exact inner difference avoids subtraction of the central offset.
        inner = r < pot.edge[0]
        # Evaluate the inner offset directly to avoid subtracting nearly equal central potential values.
        result = np.where(inner, 0.5 * omega_squared * r * r, result)
        return result

    volume = PhaseVolume(offset, np.r_[0.0, pot.edge])
    return pot, phi_c, offset, volume


def self_checks():
    # Independent known density of states in a finite constant potential.
    """Check flat and harmonic phase volumes and the finite energy-bin measure."""
    wall = 3.7
    flat = PhaseVolume(lambda r: np.zeros_like(np.asarray(r)), [0.0, 0.1, 1.0, wall])
    delta = np.geomspace(1e-8, 20, 19)
    omega, g, _ = flat.integrate(delta, 32)
    exact_omega = (16 * math.pi**2 / 9) * wall**3 * (2 * delta) ** 1.5
    exact_g = (16 * math.pi**2 / 3) * wall**3 * np.sqrt(2 * delta)
    flat_error = max(
        float(np.max(abs(omega / exact_omega - 1))), float(np.max(abs(g / exact_g - 1)))
    )
    # The 3D isotropic harmonic potential has a 6D ellipsoid below each E.
    # Its volume is (4*pi^3/3)*delta^3/omega_frequency^3.
    frequency = 1.3
    harmonic = PhaseVolume(
        lambda r: 0.5 * frequency**2 * np.asarray(r) ** 2,
        np.r_[0.0, np.geomspace(0.01, 100, 80)],
    )
    omega_h, g_h, turn = harmonic.integrate(delta, 32)
    exact_h = (4 * math.pi**3 / 3) * delta**3 / frequency**3
    exact_hg = 4 * math.pi**3 * delta**2 / frequency**3
    harmonic_error = max(
        float(np.max(abs(omega_h / exact_h - 1))),
        float(np.max(abs(g_h / exact_hg - 1))),
    )
    assert flat_error < 2e-12 and harmonic_error < 2e-10, (flat_error, harmonic_error)
    # Finite logarithmic energy bins retain dE, with dM/dlog10E related by
    # the exact bin-averaged Jacobian dE/dlog10E. Compare to analytic shell
    # volumes of the harmonic 6D ellipsoid.
    bins = np.geomspace(1e-7, 1.0, 81)
    omega_bins, _, _ = harmonic.integrate(bins, 32)
    volume = np.diff(omega_bins)
    exact_volume = (4 * math.pi**3 / 3) * np.diff(bins**3) / frequency**3
    bin_error = float(np.max(abs(volume / exact_volume - 1)))
    f_constant = 2.7
    mass = f_constant * exact_volume
    f_recovered = mass / volume
    mean_f_error = float(np.max(abs(f_recovered / f_constant - 1)))
    linear = mass / np.diff(bins)
    logarithmic = mass / np.diff(np.log10(bins))
    jacobian_error = float(
        np.max(
            abs(logarithmic / (linear * np.diff(bins) / np.diff(np.log10(bins))) - 1)
        )
    )
    # Direct reduced-coordinate integration also checks the 8*pi^2 measure:
    # int_du deta of a velocity ball = (2/3) r^2 [2psi]^(3/2).
    node, weight = np.polynomial.legendre.leggauss(48)
    psi = 0.7
    radius = 2.1
    u = np.sqrt(2 * psi) * node
    reduced = (
        8
        * math.pi**2
        * np.sqrt(2 * psi)
        * np.sum(weight * radius**2 * (psi - 0.5 * u * u))
    )
    independent = (16 * math.pi**2 / 3) * radius**2 * (2 * psi) ** 1.5
    measure_error = abs(float(reduced / independent - 1))
    assert max(bin_error, mean_f_error, jacobian_error, measure_error) < 2e-10
    return dict(
        flat_potential_relative_error=flat_error,
        harmonic_potential_relative_error=harmonic_error,
        finite_bin_phase_volume_relative_error=bin_error,
        constant_f_recovery_relative_error=mean_f_error,
        finite_log_energy_Jacobian_relative_error=jacobian_error,
        reduced_phase_measure_relative_error=measure_error,
    )


def marker_coordinates(a, field, offset):
    """Calculate E minus Phi(0) from physical center-of-mass motion."""
    r = np.asarray(a["r"])
    u = np.asarray(a["w"]) + float(field["H"]) * r
    # The physical radial and tangential kinetic energies add to the potential measured above Phi(0).
    delta = 0.5 * u * u + np.asarray(a["eta"]) / (r * r) + offset(r)
    assert np.all(np.isfinite(delta)) and np.all(delta > 0)
    return delta


def scan(particles, field, offset):
    """Measure shifted-energy limits and the complete marker mass budget."""
    low = np.inf
    high = 0.0
    masses = []
    internal = []
    for a in base.blocks(particles):
        delta = marker_coordinates(a, field, offset)
        low = min(low, float(delta.min()))
        high = max(high, float(delta.max()))
        masses.append(float(np.sum(a["mass"], dtype=np.longdouble)))
        internal.append(float(np.sum(a["internal_energy"], dtype=np.longdouble)))
    return dict(
        minimum_delta=low,
        maximum_delta=high,
        total_mass=math.fsum(masses),
        stored_internal_energy=math.fsum(internal),
    )


def histogram(particles, field, offset, edge):
    """Accumulate mass, squared mass, energy moment, and count in energy bins."""
    size = len(edge) - 1
    mass = np.zeros(size)
    squared = np.zeros(size)
    moment = np.zeros(size)
    count = np.zeros(size, dtype=np.int64)
    for a in base.blocks(particles):
        delta = marker_coordinates(a, field, offset)
        m = np.asarray(a["mass"])
        i = np.searchsorted(edge, delta, side="right") - 1
        assert np.all((i >= 0) & (i < size))
        mass += np.bincount(i, weights=m, minlength=size)
        # Retain squared mass weights and counts so energy-bin sampling can be assessed independently of mass.
        squared += np.bincount(i, weights=m * m, minlength=size)
        moment += np.bincount(i, weights=m * delta, minlength=size)
        count += np.bincount(i, minlength=size)
    return mass, squared, moment, count


def window_checks(x, values, valid):
    """Measure local slopes and curvature in adjacent 0.4-dex windows."""
    rows = []
    for left in np.arange(math.ceil(x[0] / 0.2) * 0.2, x[-1] - 0.4, 0.2):
        sel = (x >= left) & (x < left + 0.4)
        if sel.sum() < 5 or not np.all(valid[sel]):
            continue
        coeff = np.polyfit(x[sel], np.log10(values[sel]), 1)
        residual = np.log10(values[sel]) - np.polyval(coeff, x[sel])
        rows.append(
            dict(
                log10_delta_interval=[float(left), float(left + 0.4)],
                bins=int(sel.sum()),
                slope=float(coeff[0]),
                rms_log10=float(np.sqrt(np.mean(residual**2))),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_input_arguments(parser)
    parser.add_argument("--bins", type=int, default=192)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    base.require(args.bins >= 2, "Use at least two energy bins.")
    analytic = self_checks()
    (args.output_directory / "analytic_checks.json").write_text(
        json.dumps(analytic, indent=2) + "\n"
    )
    if args.self_check_only:
        print(json.dumps(analytic, indent=2))
        return
    datasets = []
    for directory in base.snapshot_directories(
        args.snapshots, args.maximum_requested_time
    ):
        manifest, field, particles, auth = base.authenticate(
            directory, base.expected_identities(args), args.run_metadata
        )
        base.require(
            field["central_mass"] == 0
            and np.all(np.asarray(field["source_mass_edge"]) >= 0)
            and np.all(np.asarray(field["source_line_density"]) >= 0),
            "Shifted energy requires a finite center and attractive monotone field.",
        )
        potential, phi_c, offset, volume = field_model(field)
        auth["potential_checks"] = potential.checks()
        limits = scan(particles, field, offset)
        base.require(
            np.isfinite(limits["minimum_delta"])
            and limits["minimum_delta"] > 0
            and np.isfinite(limits["maximum_delta"]),
            "Logarithmic shifted-energy bins require strictly positive finite energies.",
        )
        datasets.append(
            (
                directory,
                manifest,
                field,
                particles,
                auth,
                potential,
                phi_c,
                offset,
                volume,
                limits,
            )
        )
    assert datasets
    lower = min(d[-1]["minimum_delta"] for d in datasets)
    upper = max(d[-1]["maximum_delta"] for d in datasets)
    if args.edges_from:
        with np.load(args.edges_from, allow_pickle=False) as fixed:
            edge = fixed["delta_edge"].copy()
    else:
        # Use common logarithmic shifted-energy edges covering every authenticated input state.
        edge = np.geomspace(lower * (1 - 1e-8), upper * (1 + 1e-8), args.bins + 1)
    base.require(
        edge.ndim == 1
        and len(edge) >= 3
        and np.all(np.isfinite(edge))
        and np.all(edge > 0)
        and np.all(np.diff(edge) > 0),
        "Invalid shifted-energy edges.",
    )
    center = np.sqrt(edge[:-1] * edge[1:])
    records = []
    for (
        directory,
        manifest,
        field,
        particles,
        auth,
        pot,
        phi_c,
        offset,
        volume,
        limits,
    ) in datasets:
        mass, squared, moment, count = histogram(particles, field, offset, edge)
        # Compare two quadrature orders on each finite energy-bin phase volume.
        coarse, _, _ = volume.integrate(edge, 24)
        omega, _, turn_edge = volume.integrate(edge, 48)
        # A bin volume is the difference of cumulative phase volumes, rather than g at its center times bin width.
        delta_volume = np.diff(omega)
        coarse_volume = np.diff(coarse)
        assert np.all(delta_volume > 0)
        error = np.abs(delta_volume / coarse_volume - 1)
        assert np.max(error) < 1e-6, ("phase quadrature", float(np.max(error)))
        base.require(
            np.all(delta_volume > 0) and np.max(error) < 1e-6,
            "Finite-bin phase-volume quadrature failed.",
        )
        _, g_point, turn_center = volume.integrate(center, 48)
        neff = np.divide(
            mass * mass, squared, out=np.zeros_like(mass), where=squared > 0
        )
        dmass = mass / np.diff(edge)
        # This is the phase-volume mean on the instantaneous finite energy bin.
        # It applies to the sampled distribution during nonequilibrium evolution.
        # Divide the measured bin mass by its actual phase volume to obtain the bin-averaged distribution.
        mean_f = mass / delta_volume
        inner_delta = float(offset(np.array([pot.edge[0]]))[0])
        central = turn_edge[:-1] < pot.edge[0]
        # The fraction of each bin's phase volume inside the unresolved sphere
        # quantifies its contribution even when the turning radius is larger.
        inner_volume = PhaseVolume(offset, [0.0, pot.edge[0]])
        inner_omega, _, _ = inner_volume.integrate(edge, 48)
        inner_fraction = np.clip(np.diff(inner_omega) / delta_volume, 0, 1)
        # Track sampling and the unresolved-center phase-volume fraction as separate qualifications for fitted spectral windows.
        reliable = (neff >= 32) & ~central & (mass > 0) & (inner_fraction <= 0.01)
        arrays = dict(
            delta_edge=edge,
            delta_center=center,
            energy_edge=edge + phi_c,
            energy_mass=mass,
            energy_mass_squared=squared,
            energy_first_moment_shifted=moment,
            energy_count=count,
            neff_energy=neff,
            dM_ddelta=dmass,
            dM_dlog10delta=mass / np.diff(np.log10(edge)),
            cumulative_phase_volume=omega,
            bin_phase_volume=delta_volume,
            g_bin_average=delta_volume / np.diff(edge),
            g_at_center=g_point,
            mean_f_energy=mean_f,
            turning_radius_edge=turn_edge,
            turning_radius_center=turn_center,
            central_continuation_bin=central,
            reliable_energy_bin=reliable,
            inner_phase_volume_fraction=inner_fraction,
            phase_volume_quadrature_relative_error=error,
        )
        # Check both the energy histogram and mean-f times phase volume against the complete marker mass.
        closure = float(np.sum(mass) / limits["total_mass"] - 1)
        fclosure = float(np.sum(mean_f * delta_volume) / limits["total_mass"] - 1)
        assert (
            max(abs(closure), abs(fclosure)) < 2e-12
            and count.sum() == manifest["marker_count"]
        )
        base.require(
            max(abs(closure), abs(fclosure)) < 2e-12
            and count.sum() == manifest["marker_count"],
            "Energy-bin mass/count closure failed.",
        )
        name = directory.name + ".npz"
        np.savez_compressed(args.output_directory / name, **arrays)
        # Confirm all input bytes remain unchanged before recording the reduced cache as complete.
        base.recheck(directory, manifest, auth)
        report = dict(
            auth,
            **limits,
            phi_c=phi_c,
            first_field_face=float(pot.edge[0]),
            physical_wall=float(pot.edge[-1]),
            energy_at_first_face_minus_phi_c=inner_delta,
            zero_orbital_energy_delta=-phi_c,
            mass_closure_relative=closure,
            mean_f_mass_closure_relative=fclosure,
            maximum_phase_volume_quadrature_error=float(error.max()),
            mass_fraction_Neff_at_least32=float(
                mass[neff >= 32].sum() / limits["total_mass"]
            ),
            mass_fraction_in_central_turning_bins=float(
                mass[central].sum() / limits["total_mass"]
            ),
            mass_fraction_in_reliable_bins=float(
                mass[reliable].sum() / limits["total_mass"]
            ),
            cache_file=name,
            cache_sha256=base.sha(args.output_directory / name),
            slope_windows_dM_ddelta=window_checks(np.log10(center), dmass, reliable),
            slope_windows_mean_f=window_checks(np.log10(center), mean_f, reliable),
        )
        records.append(report)
        print(
            json.dumps(
                {
                    "time": auth["actual_time"],
                    "mass_closure": closure,
                    "quad_error": float(error.max()),
                    "cache": name,
                }
            ),
            flush=True,
        )
    receipt = dict(
        schema="VENUS_shifted_energy_analysis_1",
        source_input_sha256=base.common_identity(records, "source_input_sha256"),
        binary_sha256=base.common_identity(records, "executable_sha256"),
        analysis_script_sha256=base.sha(Path(__file__)),
        potential_and_native_reader_sha256=base.sha(Path(base.__file__)),
        all_weighted_markers_used=True,
        analytic_checks=analytic,
        shifted_energy="delta=E-Phi_c=u^2/2+eta/r^2+Phi_saved(r)-Phi_saved(0), physical u=w+H*r",
        central_potential=(
            "Actual saved uniform-central source continuation at r=0, with zero "
            "central point mass."
        ),
        energy_scope=(
            "Instantaneous center-of-mass orbital energy in the supplied saved "
            "field and background. Stored internal energy is separate."
        ),
        phase_measure="dGamma=8*pi^2 dr du deta; physical-u shift has unit Jacobian at fixed r,t.",
        density_of_states="g(E)=16*pi^2 integral_0^Rwall r^2 sqrt[2(E-Phi(r))]_+ dr",
        bin_phase_volume=(
            "DeltaOmega=16*pi^2/3 integral_0^Rwall "
            "r^2{[2(E_hi-Phi)]_+^(3/2)-[2(E_lo-Phi)]_+^(3/2)}dr"
        ),
        phase_mean=(
            "mean_f_bin=Mbin/DeltaOmega=(Mbin/DeltaE)/(DeltaOmega/DeltaE). This "
            "is a phase-volume weighted energy-bin mean for the measured, "
            "possibly anisotropic distribution."
        ),
        energy_spectrum=(
            "dM/ddelta=Mbin/Delta(delta); "
            "dM/dlog10(delta)=Mbin/Delta(log10(delta)) is cached separately."
        ),
        quadrature=(
            "All saved radial faces retained; turning radius bisected in the "
            "monotone potential. Last-interval quadratic map resolves the "
            "endpoint. Gauss-Legendre24/48 compared per finite energy-bin volume."
        ),
        reliability=(
            "Heavy curves require Neff>=32. A bin is central-continuation "
            "affected when its lower-edge turning radius is below the first field "
            "face. Power-law window checks additionally require at most1percent "
            "of their phase volume inside that face."
        ),
        slope_scope=(
            "Window slopes describe adjacent finite sampled intervals; their "
            "variation with energy and window measures spectral curvature."
        ),
        snapshots=records,
    )
    (args.output_directory / "analysis_manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "snapshots": len(records),
                "times": [r["actual_time"] for r in records],
                "output": str(args.output_directory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
