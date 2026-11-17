import csv
import hashlib
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


CPU_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = CPU_ROOT / "tools" / "compare_wpic_dynamic_coordinates.py"
BUILDER = CPU_ROOT / "tools" / "build_wpic_dynamic_coordinate_convergence.py"
TOOL_NAMESPACE = {
    "__name__": "compare_wpic_dynamic_coordinates",
    "__file__": str(TOOL),
}
exec(compile(TOOL.read_text(), str(TOOL), "exec"), TOOL_NAMESPACE)


class DynamicEikonalParityTest(unittest.TestCase):
    @staticmethod
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_summary(
            self, directory, coordinates, manifest_sha256, mapped=True):
        realized = (
            "0x1111222233334444" if coordinates == "schwarzschild"
            else "0xaaaabbbbccccdddd"
        )
        summary = {
            "module": "weighted_einstein_vlasov_pic",
            "coordinates": coordinates,
            "high_compactness": {
                "markers": 32,
                "radial_cells": 4,
                "total_marker_mass": 1.0,
                "base_ic_fingerprint": "0x0123456789abcdef",
                "realized_ic_fingerprint": realized,
                "eikonal_mapped_initial_data": mapped,
                "coordinate_map_manifest_sha256": (
                    manifest_sha256 if mapped else None
                ),
                "eikonal_boundary_radius": 3.0 if mapped else None,
                "eikonal_evolution_outer_radius": 3.0 if mapped else None,
                "eikonal_analytic_vacuum_extension": False,
                "eikonal_v_offset": 0.0 if mapped else None,
            },
        }
        (directory / "summary.json").write_text(json.dumps(summary))

    def refresh_manifest(self, manifest_path, sc_dir, ef_dir, artifact=None):
        manifest = json.loads(manifest_path.read_text())
        if artifact is not None:
            record = manifest["artifacts"][artifact]
            artifact_path = manifest_path.parent / record["path"]
            record["sha256"] = self.sha256(artifact_path)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        digest = self.sha256(manifest_path)
        for directory in (sc_dir, ef_dir):
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["high_compactness"]["coordinate_map_manifest_sha256"] = digest
            summary_path.write_text(json.dumps(summary))
        convergence_path = manifest_path.parent / "convergence.json"
        if convergence_path.is_file():
            convergence = json.loads(convergence_path.read_text())
            convergence["coordinate_map_manifest_sha256"] = digest
            convergence_path.write_text(json.dumps(convergence, sort_keys=True))
            (pathlib.Path(str(convergence_path) + ".sha256")).write_text(
                self.sha256(convergence_path) + "  convergence.json\n"
            )

    def write_coordinate_equivalent_profiles(self, root, mapped=True):
        sc_dir = root / "sc"
        ef_dir = root / "ef"
        sc_dir.mkdir()
        ef_dir.mkdir()

        radii = [1.0, 1.5, 2.0, 2.5, 3.0]
        times = [0.25 * index for index in range(17)]
        # Keep the end-to-end fixture flat so the independently reconstructed
        # production SC and EF spatial interpolants describe exactly the same
        # discrete spacetime.  Nontrivial face/Hermite interpolation is tested
        # separately below.
        mass_parameter = 0.0
        ef_b = 1.0
        center_radii = radii[:-1]
        face_radii = [0.75, 1.25, 1.75, 2.25, radii[-1]]

        def a_of_r(radius):
            return 1.0 - 2.0 * mass_parameter / radius

        def tortoise(radius):
            return radius + 2.0 * mass_parameter * math.log(
                radius - 2.0 * mass_parameter
            )

        def mapped_v(tau, radius):
            return tau + tortoise(radius) - tortoise(radii[-1])

        def rho_at(v_value, radius):
            radial_window = (radii[-1] - radius) / (
                radii[-1] - radii[0]
            )
            return 0.01 * radial_window * (
                1.0 + 0.03 * v_value + 0.02 * radius
            )

        sc_names = [
            "step", "time", "r", "rho", "compactness", "m", "a",
            "lambda", "mu", "lambda_t", "p_radial",
            "p_tangential_sum", "j", "metric_sample_kind", "metric_cell",
            "face_left_r", "face_right_r", "face_left_lambda",
            "face_right_lambda", "center_mu_r", "face_left_mu",
            "face_right_mu", "face_left_mu_r", "face_right_mu_r",
        ]
        with (sc_dir / "high_compactness_profiles.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sc_names)
            writer.writeheader()
            for step, tau in enumerate(times):
                for radial_index, radius in enumerate(radii):
                    a_value = a_of_r(radius)
                    lam = -0.5 * math.log(a_value)
                    mu = math.log(ef_b) + 0.5 * math.log(a_value)
                    mass = 0.5 * radius * (1.0 - a_value)
                    rho = rho_at(mapped_v(tau, radius), radius)
                    is_center = radial_index < len(center_radii)
                    cell = min(radial_index, len(center_radii) - 1)
                    writer.writerow({
                        "step": step,
                        "time": tau,
                        "r": radius,
                        "rho": rho,
                        "compactness": 1.0 - a_value,
                        "m": mass,
                        "a": a_value,
                        "lambda": lam,
                        "mu": mu,
                        "lambda_t": 0.0,
                        "p_radial": 0.2 * rho,
                        "p_tangential_sum": 0.3 * rho,
                        "j": -0.1 * rho,
                        "metric_sample_kind": (
                            "cell_center" if is_center else "outer_face"
                        ),
                        "metric_cell": (
                            radial_index if is_center else len(center_radii)
                        ),
                        "face_left_r": (
                            face_radii[cell] if is_center else radius
                        ),
                        "face_right_r": (
                            face_radii[cell + 1] if is_center else radius
                        ),
                        "face_left_lambda": 0.0,
                        "face_right_lambda": 0.0,
                        "center_mu_r": 0.0,
                        "face_left_mu": 0.0,
                        "face_right_mu": 0.0,
                        "face_left_mu_r": 0.0,
                        "face_right_mu_r": 0.0,
                    })

        ef_names = [
            "step", "advanced_time", "r", "a", "b", "m",
            "compactness", "T_rr", "S", "T_vv", "T_vr",
            "T_theta_theta", "rest_weight_density", "metric_sample_kind",
            "metric_cell", "face_left_r", "face_right_r", "face_left_a",
            "face_right_a", "face_left_b", "face_right_b",
        ]
        with (ef_dir / "high_compactness_profiles.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ef_names)
            writer.writeheader()
            for step, v_value in enumerate(times):
                for radial_index, radius in enumerate(radii):
                    a_value = a_of_r(radius)
                    mass = 0.5 * radius * (1.0 - a_value)
                    rho = rho_at(v_value, radius)
                    is_center = radial_index < len(center_radii)
                    cell = min(radial_index, len(center_radii) - 1)
                    writer.writerow({
                        "step": step,
                        "advanced_time": v_value,
                        "r": radius,
                        "a": a_value,
                        "b": ef_b,
                        "m": mass,
                        "compactness": 1.0 - a_value,
                        "T_rr": rho / a_value,
                        "S": 0.8 * rho,
                        "T_vv": ef_b * ef_b * a_value * rho,
                        "T_vr": -0.9 * ef_b * rho,
                        "T_theta_theta": 0.15 * radius * radius * rho,
                        "rest_weight_density": rho,
                        "metric_sample_kind": (
                            "cell_center" if is_center else "outer_face"
                        ),
                        "metric_cell": (
                            radial_index if is_center else len(center_radii)
                        ),
                        "face_left_r": (
                            face_radii[cell] if is_center else radius
                        ),
                        "face_right_r": (
                            face_radii[cell + 1] if is_center else radius
                        ),
                        "face_left_a": 1.0,
                        "face_right_a": 1.0,
                        "face_left_b": 1.0,
                        "face_right_b": 1.0,
                    })

        initial_names = [
            "r", "mapped_a", "ef_a", "delta_a", "mapped_b", "ef_b",
            "delta_b", "mapped_m", "ef_m", "delta_m", "mapped_T_rr",
            "ef_T_rr", "delta_T_rr", "mapped_S", "ef_S", "delta_S",
            "mapped_T_vv", "ef_T_vv", "delta_T_vv", "mapped_T_vr",
            "ef_T_vr", "delta_T_vr", "mapped_T_theta_theta",
            "ef_T_theta_theta", "delta_T_theta_theta",
        ]
        with (ef_dir / "initial_coordinate_map_residuals.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=initial_names)
            writer.writeheader()
            for radius in radii:
                a_value = a_of_r(radius)
                mass = 0.5 * radius * (1.0 - a_value)
                rho = rho_at(0.0, radius)
                values = {
                    "a": a_value,
                    "b": ef_b,
                    "m": mass,
                    "T_rr": rho / a_value,
                    "S": 0.8 * rho,
                    "T_vv": ef_b * ef_b * a_value * rho,
                    "T_vr": -0.9 * ef_b * rho,
                    "T_theta_theta": 0.15 * radius * radius * rho,
                }
                row = {"r": radius}
                for name, value in values.items():
                    row["mapped_" + name] = value
                    row["ef_" + name] = value
                    row["delta_" + name] = 0.0
                writer.writerow(row)

        crossing_path = root / "null_map.crossings.csv"
        crossing_names = [
            "id", "source_id", "rest_weight", "eta", "reverse_time",
            "schwarzschild_time", "r", "w_reverse", "w_schwarzschild",
            "p_r_ef", "lambda", "mu", "surface_reverse_time", "residual",
            "replay_step", "step_fraction", "source_marker_fingerprint",
            "ef_marker_fingerprint",
        ]
        with crossing_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=crossing_names)
            writer.writeheader()
            for marker_id in range(32):
                radius = radii[0] + (radii[-1] - radii[0]) * marker_id / 31.0
                a_value = a_of_r(radius)
                lam = -0.5 * math.log(a_value)
                mu = math.log(ef_b) + 0.5 * math.log(a_value)
                crossing_tau = -0.5 - 0.001 * marker_id
                eta = 0.01 + 0.001 * marker_id
                w_value = -0.2 + 0.002 * marker_id
                chi = 1.0 + 2.0 * eta / (radius * radius)
                epsilon = math.sqrt(chi + w_value * w_value)
                p_value = math.exp(lam) * (w_value + epsilon)
                writer.writerow({
                    "id": marker_id,
                    "source_id": 1000 + marker_id,
                    "rest_weight": 1.0 / 32.0,
                    "eta": eta,
                    "reverse_time": -crossing_tau,
                    "schwarzschild_time": crossing_tau,
                    "r": radius,
                    "w_reverse": -w_value,
                    "w_schwarzschild": w_value,
                    "p_r_ef": p_value,
                    "lambda": lam,
                    "mu": mu,
                    "surface_reverse_time": -crossing_tau - 1.0e-12,
                    "residual": 1.0e-12,
                    "replay_step": marker_id,
                    "step_fraction": 0.5,
                    "source_marker_fingerprint": "0x{:016x}".format(
                        marker_id + 1
                    ),
                    "ef_marker_fingerprint": "0x{:016x}".format(
                        marker_id + 100
                    ),
                })

        binary_path = root / "null_map.bin"
        binary_path.write_bytes(b"VENUS synthetic null map\0")
        slice_path = root / "null_map.slice.csv"
        slice_path.write_text(
            "r,s,tau,A,F,a,b,m,rho,p,j,T_vv,T_vr,T_rr,T_theta_theta,S\n"
            "3,0,0,1,1,0.85,1.2,0.225,0.01,0.002,-0.001,"
            "0.01224,-0.0108,0.0117647058824,0.0135,0.008\n"
        )
        steps_path = root / "null_map.accepted_steps.csv"
        steps_path.write_text("step,dt\n0,0.25\n")
        history_path = root / "null_map.donor_history.bin"
        payload = bytearray(b"VENUSSCHIST1\0\0\0\0")
        payload.extend(struct.pack(
            "<IQQ", 2, len(times), len(center_radii)
        ))
        for reverse_time in times:
            tau = -reverse_time
            # This flat fixture has M=0.  Keep the serialized ADM scalar
            # consistent with its centre and face metric payload; the
            # dedicated nonzero-mass regression below covers M>0.
            payload.extend(struct.pack("<5d", reverse_time, 1.0, 0.0, 0.0, 1.0))
            arrays = {name: [] for name in (
                "lambda", "mu", "m", "rho", "p", "pt", "j_reverse"
            )}
            for radius in center_radii:
                a_value = a_of_r(radius)
                rho = rho_at(mapped_v(tau, radius), radius)
                arrays["lambda"].append(-0.5 * math.log(a_value))
                arrays["mu"].append(math.log(ef_b) + 0.5 * math.log(a_value))
                arrays["m"].append(0.5 * radius * (1.0 - a_value))
                arrays["rho"].append(rho)
                arrays["p"].append(0.2 * rho)
                arrays["pt"].append(0.3 * rho)
                arrays["j_reverse"].append(0.1 * rho)
            for name in ("lambda", "mu", "m", "rho", "p", "pt", "j_reverse"):
                payload.extend(struct.pack(
                    "<{}d".format(len(center_radii)), *arrays[name]
                ))
            payload.extend(struct.pack("<{}d".format(len(face_radii)), *face_radii))
            payload.extend(struct.pack("<{}d".format(len(face_radii)),
                                       *([0.0] * len(face_radii))))
            payload.extend(struct.pack("<{}d".format(len(face_radii)),
                                       *([0.0] * len(face_radii))))
            payload.extend(struct.pack("<{}d".format(len(face_radii)),
                                       *([0.0] * len(face_radii))))
            payload.extend(struct.pack("<{}d".format(len(center_radii)),
                                       *([0.0] * len(center_radii))))
        history_path.write_bytes(payload)
        artifacts = {
            "null_initial_data": binary_path,
            "crossings": crossing_path,
            "slice": slice_path,
            "accepted_steps": steps_path,
            "donor_history": history_path,
        }
        manifest = {
            "schema": "venus_v3_sc_to_ef_null_map",
            "schema_version": 2,
            "completed": mapped,
            "all_markers_crossed_once": mapped,
            "outer_worldtube_verified_vacuum": mapped,
            "mapping_scope": "initial_null_hypersurface",
            "donor_direction": "East_time_reversed_outgoing_s=-tau",
            "physical_schwarzschild_run": (
                "incoming markers are unmodified; reverse donor is an independent copy"
            ),
            "time_reversal_momentum": "w_reverse=-w_schwarzschild",
            "eikonal_equation": "tau_sigma_r=-exp(lambda-mu)",
            "eikonal_integration_method": (
                "per_cell_RK4_step_doubling_n_vs_2n_accept_2n"
            ),
            "eikonal_richardson_error_normalization": (
                "abs(Y_2n-Y_n)/(15*max(1,abs(Y_in),abs(Y_2n)));"
                "Y={s_sigma,ln_A}"
            ),
            "saved_metric_interpolation": (
                "SC center-face lambda plus monotone-Hermite mu; "
                "EF center-face a,b"
            ),
            "marker_momentum_map": (
                "P=exp(lambda)*(epsilon+w_schwarzschild)>0"
            ),
            "outer_normalization": (
                "at R_B: v=tau, A=v_tau=1, F=tau_v=1, b=1"
            ),
            "ef_time_coordinate": "v_bar=v+ef_v_offset",
            "source_marker_count": 32,
            "mapped_marker_count": 32,
            "unique_crossing_count": 32,
            "eikonal_boundary_radius": 3.0,
            "outer_boundary_radius": 3.0,
            "map_worldtube_radius": 3.0,
            "evolution_outer_radius": 3.0,
            "radial_cell_count": 4,
            "map_radial_cell_count": 4,
            "evolution_radial_cell_count": 4,
            "analytic_vacuum_extension": False,
            "analytic_vacuum_extension_start_index": 4,
            "donor_history_format_version": 2,
            "grid_fingerprint": "0x9999888877776666",
            "map_grid_fingerprint": "0x9999888877776666",
            "required_worldtube_clearance": 0.1,
            "maximum_donor_marker_radius": 2.0,
            "maximum_crossing_radius": 2.2,
            "minimum_worldtube_matter_clearance": 1.0,
            "minimum_crossing_radius_clearance": 0.8,
            "maximum_analytic_vacuum_field_residual": 0.0,
            "maximum_analytic_vacuum_source": 0.0,
            "maximum_analytic_inverse_map_residual": 0.0,
            "minimum_analytic_vacuum_a": 1.0,
            "eikonal_v_offset": 0.0,
            "maximum_eikonal_reverse_time_richardson_error": 1.0e-12,
            "maximum_eikonal_log_A_richardson_error": 5.0e-13,
            "maximum_eikonal_ode_residual": 1.0e-12,
            "maximum_marker_crossing_residual": 1.0e-12,
            "maximum_replay_time_error": 1.0e-12,
            "maximum_outer_b_normalization_error": 1.0e-12,
            "outer_b_normalization_tolerance": 1.0e-8,
            "maximum_outer_source": 0.0,
            "provenance": {
                "base_ic_fingerprint": "0x0123456789abcdef",
                "schwarzschild_realized_ic_fingerprint": "0x1111222233334444",
                "eddington_finkelstein_realized_ic_fingerprint": "0xaaaabbbbccccdddd",
            },
            "artifacts": {
                name: {
                    "path": str(path.relative_to(root)),
                    "sha256": self.sha256(path),
                }
                for name, path in artifacts.items()
            },
        }
        manifest_path = root / "null_map.manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        manifest_digest = self.sha256(manifest_path)
        self.write_summary(
            sc_dir, "schwarzschild", manifest_digest, mapped=mapped
        )
        self.write_summary(
            ef_dir, "eddington_finkelstein", manifest_digest, mapped=mapped
        )
        coarse_audit = root / "coarse_pair_audit.json"
        fine_audit = root / "fine_pair_audit.json"
        coarse_audit.write_text('{"resolution":"coarse","complete":true}\n')
        fine_audit.write_text('{"resolution":"fine","complete":true}\n')
        convergence_path = root / "convergence.json"
        convergence = {
            "schema": "venus_wpic_dynamic_coordinate_convergence_v1",
            "status": "pass",
            "coordinate_map_manifest_sha256": manifest_digest,
            "resolution_levels": 2,
            "independent_pic_realizations": 2,
            "monotone_refinement": True,
            "eikonal_substeps": 2,
            "diagnostic_certifier_sha256": self.sha256(TOOL),
            "convergence_builder_sha256": self.sha256(BUILDER),
            "metric_interpolation": dict(
                TOOL_NAMESPACE["PRODUCTION_METRIC_INTERPOLATION"]
            ),
            "uncertainty_method": (
                "two-level radial/time refinement plus two correlated-PIC resamples"
            ),
            "fine_resolution": {"radial_cells": 4, "markers": 32},
            "uncertainty_bounds": {
                "metric": 0.005,
                "stress": 0.02,
                "moment": 0.02,
                "distribution": 1.0e-10,
            },
            "artifacts": [
                {
                    "path": path.name,
                    "sha256": self.sha256(path),
                }
                for path in (coarse_audit, fine_audit)
            ],
        }
        convergence_path.write_text(json.dumps(convergence, sort_keys=True))
        pathlib.Path(str(convergence_path) + ".sha256").write_text(
            self.sha256(convergence_path) + "  convergence.json\n"
        )
        return (
            sc_dir, ef_dir, manifest_path, crossing_path, convergence_path
        )

    def run_tool(
            self, sc_dir, ef_dir, manifest, convergence, output,
            extra_args=None):
        command = [
                sys.executable,
                str(TOOL),
                "--schwarzschild", str(sc_dir),
                "--ef", str(ef_dir),
                "--manifest", str(manifest),
                "--output", str(output),
                "--minimum-events", "8",
                "--minimum-time-slices", "2",
                "--minimum-radial-coverage", "0.95",
                "--minimum-event-coverage", "0.95",
                "--minimum-time-coverage", "0.95",
                "--minimum-marker-samples", "8",
                "--max-cadence-to-dr", "0.75",
                "--eikonal-convergence-tolerance", "2e-5",
                "--metric-tolerance", "0.005",
                "--stress-tolerance", "0.02",
                "--moment-tolerance", "0.02",
                "--distribution-tolerance", "1e-10",
            ]
        if convergence is not None:
            command.extend(("--convergence-evidence", str(convergence)))
        if extra_args:
            command.extend(extra_args)
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

    def test_profile_loader_uses_compact_binary64_slices(self):
        with tempfile.TemporaryDirectory(prefix="venus-profile-memory-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, _manifest, _crossings, _convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            profile_class = TOOL_NAMESPACE["RectilinearProfile"]
            array_class = TOOL_NAMESPACE["array"]
            sc = profile_class(
                sc_dir / "high_compactness_profiles.csv",
                "time",
                {
                    "m", "a", "lambda", "mu", "lambda_t", "rho",
                    "p_radial", "p_tangential_sum", "j",
                },
            )
            ef = profile_class(
                ef_dir / "high_compactness_profiles.csv",
                "advanced_time",
                {
                    "a", "b", "m", "T_rr", "S", "T_vv", "T_vr",
                    "T_theta_theta",
                },
            )
            for profile in (sc, ef):
                for rows in profile.fields.values():
                    self.assertTrue(rows)
                    self.assertIsInstance(rows[0], array_class)
                    self.assertEqual(rows[0].typecode, "d")
                self.assertIsInstance(
                    profile.metric_reconstruction[0], dict
                )
                for values in profile.metric_reconstruction[0].values():
                    self.assertIsInstance(values, array_class)
                    self.assertEqual(values.typecode, "d")
            self.assertTrue(math.isfinite(sc.sample("m", 0.5, 1.5)))
            self.assertGreater(
                sc.sample_eikonal_h_with_time_derivative(0.5, 1.5)[0],
                0.0,
            )
            self.assertGreater(
                ef.sample_with_time_derivative("a", 0.5, 1.5)[0],
                0.0,
            )

    def test_log_jacobian_eikonal_pushers_preserve_future_orientation(self):
        class AnalyticForwardProfile:
            radii = [0.0, 1.0]
            outer_boundary_radius = 1.0

            @staticmethod
            def sample_eikonal_h_with_time_derivative(time, _radius):
                value = math.exp(time)
                return value, value

        class AnalyticInverseProfile:
            radii = [0.0, 1.0]
            outer_boundary_radius = 1.0

            @staticmethod
            def sample_with_time_derivative(name, time, _radius):
                if name == "a":
                    value = math.exp(-time)
                    return value, -value
                if name == "b":
                    return 1.0, 0.0
                raise AssertionError("unexpected inverse field " + name)

        exact_v = -math.log(math.exp(-1.0) + 1.0)
        exact_a = 1.0 / (1.0 + math.e)
        previous_error = float("inf")
        for substeps in (1, 2, 4, 8, 16):
            mapped_v, map_a, log_a = (
                TOOL_NAMESPACE["trace_to_outer_boundary"](
                    AnalyticForwardProfile(), 1.0, 0.0, substeps, 0.0, 1.0
                )
            )
            error = max(abs(mapped_v - exact_v), abs(map_a - exact_a))
            self.assertGreater(map_a, 0.0)
            self.assertAlmostEqual(map_a, math.exp(log_a))
            self.assertLess(error, previous_error)
            previous_error = error

        mapped_tau, map_f, log_f = (
            TOOL_NAMESPACE["inverse_trace_to_outer_boundary"](
                AnalyticInverseProfile(), mapped_v, 0.0, 16, 0.0, 1.0
            )
        )
        self.assertGreater(map_f, 0.0)
        self.assertLess(abs(mapped_tau - 1.0), 5.0e-5)
        self.assertLess(abs(math.expm1(log_a + log_f)), 5.0e-5)

        # The superseded one-step direct-A RK4 update for this same smooth
        # problem crosses zero.  The exact Jacobian cannot do so.
        def direct_rhs(time, sensitivity):
            value = math.exp(time)
            return -value, -value * sensitivity

        k1_t, k1_a = direct_rhs(1.0, 1.0)
        k2_t, k2_a = direct_rhs(1.0 + 0.5 * k1_t, 1.0 + 0.5 * k1_a)
        k3_t, k3_a = direct_rhs(1.0 + 0.5 * k2_t, 1.0 + 0.5 * k2_a)
        _k4_t, k4_a = direct_rhs(1.0 + k3_t, 1.0 + k3_a)
        direct_a = 1.0 + (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a) / 6.0
        self.assertLess(direct_a, 0.0)

    def test_rk4_step_doubling_does_not_hide_log_jacobian_mismatch(self):
        change = TOOL_NAMESPACE["rk4_step_doubling_change"](
            math.log(1.0), math.log(1.03)
        )
        self.assertAlmostEqual(change, math.log(1.03))
        self.assertGreater(change, 2.0e-5)

    def test_legacy_center_interpolation_preserves_exponentiation_order(self):
        profile = object.__new__(TOOL_NAMESPACE["RectilinearProfile"])
        profile.path = "synthetic"
        profile.times = [0.0, 1.0]
        profile.radii = [1.0, 2.0]
        profile.outer_boundary_radius = None
        profile.vacuum_coordinate_system = None
        profile.metric_reconstruction_kind = None
        profile.metric_reconstruction = None
        profile.fields = {
            "lambda": [
                [0.0, math.log(4.0)],
                [math.log(2.0), math.log(8.0)],
            ],
            "mu": [[0.0, 0.0], [0.0, 0.0]],
        }
        value, derivative = (
            profile.sample_eikonal_h_with_time_derivative(0.25, 1.5)
        )
        self.assertAlmostEqual(value, 2.5)
        self.assertAlmostEqual(derivative, 2.0)
        # Even the diagnostic centre-only fallback must not exponentiate the nodes
        # before interpolating their logarithms; that would give 3.125 here
        # and define yet another map.
        self.assertNotAlmostEqual(value, 3.125)

    def test_eikonal_samplers_use_production_center_face_reconstruction(self):
        profile_class = TOOL_NAMESPACE["RectilinearProfile"]
        sc = object.__new__(profile_class)
        sc.path = "synthetic-sc"
        sc.times = [0.0, 1.0]
        sc.radii = [1.0, 2.0]
        sc.outer_boundary_radius = None
        sc.vacuum_coordinate_system = None
        sc.metric_reconstruction_kind = "schwarzschild"

        def sc_record(lambda_shift, mu_shift):
            return {
                "cell": 0,
                "center_r": 1.0,
                "face_left_r": 0.0,
                "face_right_r": 2.0,
                "center_lambda": 0.4 + lambda_shift,
                "face_left_lambda": 0.0 + lambda_shift,
                "face_right_lambda": 0.8 + lambda_shift,
                "center_mu": 0.15 + mu_shift,
                "center_mu_r": 0.2,
                "face_left_mu": 0.0 + mu_shift,
                "face_right_mu": 0.5 + mu_shift,
                "face_left_mu_r": 0.1,
                "face_right_mu_r": 0.4,
            }

        sc.metric_reconstruction = [
            [sc_record(0.0, 0.0)],
            [sc_record(0.3, 0.1)],
        ]
        sc.fields = {
            # Deliberately inconsistent centre-only values: using these
            # would select the separate centre-only interpolation path.
            "lambda": [[10.0, 10.0], [10.0, 10.0]],
            "mu": [[0.0, 0.0], [0.0, 0.0]],
        }

        t = 0.5
        mu_right_half = (
            (2.0 * t**3 - 3.0 * t**2 + 1.0) * 0.15
            + (t**3 - 2.0 * t**2 + t) * 0.2
            + (-2.0 * t**3 + 3.0 * t**2) * 0.5
            + (t**3 - t**2) * 0.4
        )
        h0 = math.exp(0.6 - mu_right_half)
        h1 = math.exp(0.9 - (mu_right_half + 0.1))
        h, h_time = sc.sample_eikonal_h_with_time_derivative(0.25, 1.5)
        self.assertAlmostEqual(h, 0.75 * h0 + 0.25 * h1)
        self.assertAlmostEqual(h_time, h1 - h0)
        self.assertNotAlmostEqual(h, math.exp(10.0))

        ef = object.__new__(profile_class)
        ef.path = "synthetic-ef"
        ef.times = [0.0, 1.0]
        ef.radii = [1.0, 2.0]
        ef.outer_boundary_radius = None
        ef.vacuum_coordinate_system = None
        ef.metric_reconstruction_kind = "eddington_finkelstein"

        def ef_record(a_shift, b_shift):
            return {
                "cell": 0,
                "center_r": 1.0,
                "face_left_r": 0.0,
                "face_right_r": 2.0,
                "center_a": 0.8 + a_shift,
                "face_left_a": 0.7 + a_shift,
                "face_right_a": 0.9 + a_shift,
                "center_b": 1.1 + b_shift,
                "face_left_b": 1.0 + b_shift,
                "face_right_b": 1.2 + b_shift,
            }

        ef.metric_reconstruction = [
            [ef_record(0.0, 0.0)],
            [ef_record(0.2, 0.4)],
        ]
        ef.fields = {
            "a": [[9.0, 9.0], [9.0, 9.0]],
            "b": [[8.0, 8.0], [8.0, 8.0]],
        }
        a_value, a_time = ef.sample_with_time_derivative("a", 0.25, 1.5)
        b_value, b_time = ef.sample_with_time_derivative("b", 0.25, 1.5)
        self.assertAlmostEqual(a_value, 0.85 + 0.25 * 0.2)
        self.assertAlmostEqual(a_time, 0.2)
        self.assertAlmostEqual(b_value, 1.15 + 0.25 * 0.4)
        self.assertAlmostEqual(b_time, 0.4)

    def test_nonzero_mass_donor_v2_outer_vacuum_face_is_authenticated(self):
        profile_class = TOOL_NAMESPACE["RectilinearProfile"]
        mass = 0.1
        center_radii = [1.0, 2.0, 3.0]
        face_radii = [0.5, 1.5, 2.5, 4.0]
        saved_radii = center_radii + [face_radii[-1]]

        def gap(radius):
            return 1.0 - 2.0 * mass / radius

        def lam(radius):
            return -0.5 * math.log(gap(radius))

        def mu(radius):
            return 0.5 * math.log(gap(radius))

        def mu_r(radius):
            return mass / (radius * radius * gap(radius))

        def make_profile():
            profile = object.__new__(profile_class)
            profile.path = "synthetic-nonzero-mass-sc"
            profile.times = [0.0, 1.0, 2.0]
            profile.radii = list(saved_radii)
            profile.outer_boundary_radius = None
            profile.vacuum_coordinate_system = None
            profile.metric_reconstruction_kind = "schwarzschild"

            records = []
            for cell, radius in enumerate(center_radii):
                records.append({
                    "cell": cell,
                    "center_r": radius,
                    "face_left_r": face_radii[cell],
                    "face_right_r": face_radii[cell + 1],
                    "center_lambda": lam(radius),
                    "face_left_lambda": lam(face_radii[cell]),
                    "face_right_lambda": lam(face_radii[cell + 1]),
                    "center_mu": mu(radius),
                    "center_mu_r": mu_r(radius),
                    "face_left_mu": mu(face_radii[cell]),
                    "face_right_mu": mu(face_radii[cell + 1]),
                    "face_left_mu_r": mu_r(face_radii[cell]),
                    "face_right_mu_r": mu_r(face_radii[cell + 1]),
                })
            profile.metric_reconstruction = [
                [dict(record) for record in records]
                for _time in profile.times
            ]
            row = {
                "lambda": [lam(radius) for radius in saved_radii],
                "mu": [mu(radius) for radius in saved_radii],
                "m": [mass] * len(saved_radii),
                "rho": [0.0] * len(saved_radii),
                "p_radial": [0.0] * len(saved_radii),
                "p_tangential_sum": [0.0] * len(saved_radii),
                "j": [0.0] * len(saved_radii),
                "a": [gap(radius) for radius in saved_radii],
                "lambda_t": [0.0] * len(saved_radii),
            }
            profile.fields = {
                name: [list(values) for _time in profile.times]
                for name, values in row.items()
            }
            return profile

        def write_history(path, serialized_adm_mass=mass,
                          outer_lambda_offset=0.0):
            payload = bytearray(b"VENUSSCHIST1\0\0\0\0")
            reverse_times = [0.5, 1.0]
            payload.extend(struct.pack(
                "<IQQ", 2, len(reverse_times), len(center_radii)
            ))
            center_arrays = (
                [lam(radius) for radius in center_radii],
                [mu(radius) for radius in center_radii],
                [mass] * len(center_radii),
                [0.0] * len(center_radii),
                [0.0] * len(center_radii),
                [0.0] * len(center_radii),
                [0.0] * len(center_radii),
            )
            face_lambda = [lam(radius) for radius in face_radii]
            face_lambda[-1] += outer_lambda_offset
            reconstruction_arrays = (
                face_radii,
                face_lambda,
                [mu(radius) for radius in face_radii],
                [mu_r(radius) for radius in face_radii],
                [mu_r(radius) for radius in center_radii],
            )
            for reverse_time in reverse_times:
                payload.extend(struct.pack(
                    "<5d", reverse_time, 1.0, serialized_adm_mass,
                    2.0 * mass / face_radii[0], gap(face_radii[0])
                ))
                for values in center_arrays + reconstruction_arrays:
                    payload.extend(struct.pack(
                        "<{}d".format(len(values)), *values
                    ))
            path.write_bytes(payload)

        with tempfile.TemporaryDirectory(prefix="venus-donor-v2-test-") as tmp:
            root = pathlib.Path(tmp)
            valid_path = root / "valid.bin"
            write_history(valid_path)
            profile = make_profile()
            audit = TOOL_NAMESPACE["prepend_time_reversed_donor_history"](
                profile, valid_path
            )
            face_audit = audit["outer_vacuum_face"]
            self.assertEqual(face_audit["validated_snapshots"], 2)
            self.assertEqual(
                face_audit["mass_source"],
                "serialized_snapshot_adm_mass",
            )
            self.assertTrue(all(
                residual == 0.0
                for residual in face_audit["maximum_residuals"].values()
            ))
            self.assertEqual(profile.fields["m"][0][-1], mass)
            self.assertAlmostEqual(
                profile.fields["lambda"][0][-1], lam(face_radii[-1])
            )
            self.assertAlmostEqual(
                profile.metric_reconstruction[0][-1]["face_right_mu_r"],
                mu_r(face_radii[-1]),
            )

            corruptions = (
                ("bad-adm.bin", mass + 1.0e-3, 0.0,
                 "vacuum-tail mass disagrees with serialized ADM mass"),
                ("bad-face.bin", mass, 1.0e-3,
                 "outer-face lambda is inconsistent"),
            )
            for name, adm_mass, lambda_offset, message in corruptions:
                with self.subTest(corruption=name):
                    path = root / name
                    write_history(path, adm_mass, lambda_offset)
                    corrupt_profile = make_profile()
                    original_times = list(corrupt_profile.times)
                    with self.assertRaisesRegex(
                            TOOL_NAMESPACE["ProfileError"], message):
                        TOOL_NAMESPACE[
                            "prepend_time_reversed_donor_history"
                        ](corrupt_profile, path)
                    # A rejected donor must not partially prepend its time or
                    # reconstruction arrays.
                    self.assertEqual(corrupt_profile.times, original_times)
                    self.assertEqual(
                        len(corrupt_profile.metric_reconstruction),
                        len(original_times),
                    )

    def test_log_jacobian_composition_overflow_fails_closed(self):
        with self.assertRaises(TOOL_NAMESPACE["ProfileError"]):
            TOOL_NAMESPACE["jacobian_composition_error"](700.0, 700.0)

    def test_common_domain_allows_only_explicit_inner_boundaries(self):
        times = [0.0, 1.0, 2.0]
        radii = [1.0, 2.0, 3.0, 4.0]
        states = {
            (time_index, radial_slot): {"status": "mapped"}
            for time_index in range(len(times))
            for radial_slot in range(len(radii))
        }
        for time_index in range(len(times)):
            states[(time_index, 0)] = {"status": "outside_ef_domain"}
        audit = TOOL_NAMESPACE["audit_contiguous_common_domain"](
            list(range(len(times))), times, radii, states, 0.75
        )
        audit.pop("_domain_keys")
        self.assertEqual(audit["candidate_events"], 9)
        self.assertEqual(audit["compared_events"], 9)
        self.assertEqual(audit["unexplained_holes"], 0)
        self.assertEqual(audit["time_topology_gap_slices"], 0)
        self.assertEqual(
            audit["disconnected_non_horizon_support_samples"], 0
        )
        self.assertAlmostEqual(audit["event_coverage"], 1.0)
        self.assertAlmostEqual(
            audit["minimum_per_slice_radial_coverage"], 0.75
        )

    def test_common_domain_exposes_numeric_hole_and_time_reentry(self):
        times = [0.0, 1.0, 2.0]
        radii = [1.0, 2.0, 3.0, 4.0]
        states = {
            (time_index, radial_slot): {"status": "mapped"}
            for time_index in range(len(times))
            for radial_slot in range(len(radii))
        }
        states[(0, 2)] = {"status": "forward_eikonal_not_converged"}
        states[(1, 0)] = {"status": "outside_ef_domain"}
        states[(1, 1)] = {"status": "outside_ef_domain"}
        audit = TOOL_NAMESPACE["audit_contiguous_common_domain"](
            list(range(len(times))), times, radii, states, 0.75
        )
        audit.pop("_domain_keys")
        self.assertEqual(audit["unexplained_holes"], 1)
        self.assertEqual(
            audit["unexplained_hole_examples"][0]["status"],
            "forward_eikonal_not_converged",
        )
        self.assertEqual(audit["time_topology_gap_slices"], 1)
        self.assertLess(audit["event_coverage"], 1.0)
        self.assertLess(audit["time_slice_coverage"], 1.0)

    def test_outer_clock_endpoint_does_not_excuse_composition_hole(self):
        """Reproduce the coarse Stage-A common-domain accounting exactly.

        Reaching the last sampled outer-clock time is a statement about the
        end of the requested time block.  It must not turn an in-domain
        numerical rejection into a chart boundary or a complete slice.
        """
        times = [float(index) for index in range(16)]
        radii = [float(index + 1) for index in range(64)]
        states = {
            (time_index, radial_slot): {"status": "mapped"}
            for time_index in range(len(times))
            for radial_slot in range(len(radii))
        }
        # Two innermost points on one slice are outside the stored EF chart.
        # They legitimately reduce that slice to a 62-point exterior suffix.
        states[(15, 0)] = {"status": "outside_ef_domain"}
        states[(15, 1)] = {"status": "outside_ef_domain"}
        # The next point is inside that suffix.  A failed composition test is
        # numerical evidence, not permission to shrink the claimed domain.
        states[(15, 2)] = {
            "status": "composition_not_closed",
            "composition_time_error": 1.0e-6,
            "composition_jacobian_error": 5.438e-4,
        }

        audit = TOOL_NAMESPACE["audit_contiguous_common_domain"](
            list(range(len(times))), times, radii, states, 0.95
        )
        domain_keys = audit.pop("_domain_keys")

        self.assertEqual(audit["candidate_events"], 1022)
        self.assertEqual(audit["compared_events"], 1021)
        self.assertAlmostEqual(audit["event_coverage"], 1021.0 / 1022.0)
        self.assertEqual(audit["time_slices"], 16)
        self.assertEqual(audit["complete_time_slices"], 15)
        self.assertAlmostEqual(audit["time_slice_coverage"], 15.0 / 16.0)
        self.assertAlmostEqual(
            audit["minimum_per_slice_radial_coverage"], 62.0 / 64.0
        )
        self.assertEqual(audit["unexplained_holes"], 1)
        hole = audit["unexplained_hole_examples"][0]
        self.assertEqual(hole["status"], "composition_not_closed")
        self.assertEqual(hole["time_position"], 15)
        self.assertEqual(hole["time_index"], 15)
        self.assertEqual(hole["radial_slot"], 2)
        self.assertEqual(hole["inner_neighbor_status"], "outside_ef_domain")
        self.assertEqual(hole["outer_neighbor_status"], "mapped")
        self.assertEqual(
            hole["first_inner_boundary_status"], "outside_ef_domain"
        )
        self.assertEqual(hole["first_inner_boundary_radial_slot"], 1)
        self.assertAlmostEqual(hole["composition_time_error"], 1.0e-6)
        self.assertAlmostEqual(
            hole["composition_jacobian_error"], 5.438e-4
        )
        self.assertIn((15, 2), domain_keys)
        self.assertEqual(audit["time_topology_gap_slices"], 0)
        self.assertEqual(
            audit["terminal_reason"], "end_of_outer_clock_overlap"
        )

        numeric = TOOL_NAMESPACE["summarize_domain_numeric_errors"](
            states, domain_keys
        )
        self.assertAlmostEqual(
            numeric["composition_jacobian_error"], 5.438e-4
        )

    def test_domain_numeric_maxima_ignore_only_true_chart_boundaries(self):
        times = [0.0]
        radii = [1.0, 2.0, 3.0]
        states = {
            (0, 0): {
                "status": "at_or_beyond_horizon",
                "forward_step_refinement_error": 1.0,
            },
            (0, 1): {
                "status": "mapped",
                "forward_step_refinement_error": 1.0e-8,
            },
            (0, 2): {
                "status": "mapped",
                "forward_step_refinement_error": 2.0e-8,
            },
        }
        audit = TOOL_NAMESPACE["audit_contiguous_common_domain"](
            [0], times, radii, states, 2.0 / 3.0
        )
        domain_keys = audit.pop("_domain_keys")
        summary = TOOL_NAMESPACE["summarize_domain_numeric_errors"](
            states, domain_keys
        )
        self.assertEqual(summary["forward_step_refinement_error"], 2.0e-8)

        states[(0, 1)] = {
            "status": "forward_eikonal_not_converged",
            "forward_step_refinement_error": 1.0,
        }
        audit = TOOL_NAMESPACE["audit_contiguous_common_domain"](
            [0], times, radii, states, 2.0 / 3.0
        )
        domain_keys = audit.pop("_domain_keys")
        summary = TOOL_NAMESPACE["summarize_domain_numeric_errors"](
            states, domain_keys
        )
        self.assertEqual(audit["unexplained_holes"], 1)
        self.assertEqual(summary["forward_step_refinement_error"], 1.0)

    def test_parser_uses_near_complete_coverage_defaults(self):
        args = TOOL_NAMESPACE["build_parser"]().parse_args([
            "--schwarzschild", "sc", "--ef", "ef"
        ])
        self.assertEqual(args.minimum_event_coverage, 0.95)
        self.assertEqual(args.minimum_time_coverage, 0.95)
        self.assertEqual(args.minimum_radial_coverage, 0.95)

    def test_static_dynamic_map_and_tensor_transform(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["eikonal_substeps"], 2)
            self.assertEqual(result["certifier_sha256"], self.sha256(TOOL))
            self.assertEqual(
                result["convergence"]["evidence"]["eikonal_substeps"], 2
            )
            self.assertTrue(
                result["metric_interpolation"]["production_equivalent"]
            )
            self.assertEqual(
                result["metric_interpolation"],
                TOOL_NAMESPACE["PRODUCTION_METRIC_INTERPOLATION"],
            )
            self.assertEqual(
                result["provenance"]["dense_time_reversed_donor_history"][
                    "format_version"
                ],
                2,
            )
            self.assertEqual(
                result["equation"], "v_r=exp(lambda-mu)*v_tau"
            )
            self.assertGreater(result["mapping"]["events"], 8)
            common_domain = result["mapping"]["common_domain"]
            self.assertEqual(
                common_domain["definition"],
                "maximal_contiguous_exterior_pre_schwarzschild_terminal",
            )
            self.assertEqual(common_domain["unexplained_holes"], 0)
            self.assertEqual(common_domain["time_topology_gap_slices"], 0)
            self.assertEqual(
                common_domain["disconnected_non_horizon_support_samples"],
                0,
            )
            self.assertGreaterEqual(
                result["mapping"]["event_coverage"], 0.95
            )
            self.assertGreaterEqual(
                result["mapping"]["time_slice_coverage"], 0.95
            )
            self.assertGreaterEqual(
                result["mapping"]["radial_coverage"], 0.95
            )
            # The field norms must contain only selected common-domain events,
            # never partial-overlap samples accumulated before domain choice.
            for diagnostic in result["errors"].values():
                self.assertEqual(
                    diagnostic["samples"], result["mapping"]["events"]
                )
            self.assertLess(
                result["errors"]["metric_b"]["relative_linf"], 5.0e-3
            )
            self.assertLess(
                result["errors"]["stress_T_rr"]["relative_l2"], 2.0e-2
            )
            self.assertEqual(
                result["inverse_equation"], "tau_r+(a*b)^(-1)*tau_v=0"
            )
            self.assertLess(
                result["mapping"]["maximum_composition_time_error"], 5.0e-4
            )
            self.assertLess(
                result["mapping"]["maximum_composition_jacobian_error"],
                5.0e-4,
            )
            self.assertEqual(
                result["mapping"]["positive_jacobian_variables"],
                "log(A)=log(v_tau) and log(F)=log(tau_v)",
            )
            self.assertEqual(
                result["mapping"]["step_refinement_error_definition"],
                (
                    "max(abs(delta coordinate time),abs(delta log Jacobian)); "
                    "no Richardson division without demonstrated asymptotic order"
                ),
            )
            self.assertEqual(len(result["distribution"]["evidence"]), 1)
            self.assertEqual(
                result["outer_worldtube_representation"],
                "exact_saved_outer_face",
            )
            self.assertEqual(
                result["outer_worldtube_validation"]["schwarzschild"][
                    "maximum_source_magnitude"],
                0.0,
            )
            crossing_errors = result["distribution"]["evidence"][0][
                "errors"]
            self.assertIn("distribution_time_reversal_tau", crossing_errors)
            self.assertIn("distribution_time_reversal_w", crossing_errors)
            self.assertIn("distribution_null_surface_crossing", crossing_errors)
            self.assertTrue((root / "comparison.json.sha256").is_file())
            self.assertNotEqual(
                result["provenance"]["schwarzschild_realized_ic_fingerprint"],
                result["provenance"][
                    "eddington_finkelstein_realized_ic_fingerprint"
                ],
            )
            map_audit = result["provenance"]["coordinate_map_manifest"]
            self.assertEqual(
                map_audit["eikonal_integration_method"],
                "per_cell_RK4_step_doubling_n_vs_2n_accept_2n",
            )
            self.assertEqual(
                map_audit["extraction_residuals"][
                    "maximum_eikonal_ode_residual"
                ],
                max(
                    map_audit["extraction_residuals"][
                        "maximum_eikonal_reverse_time_richardson_error"
                    ],
                    map_audit["extraction_residuals"][
                        "maximum_eikonal_log_A_richardson_error"
                    ],
                ),
            )

    def test_forward_nonconvergence_is_counted_and_located(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            # Give the SC production reconstruction a smooth non-polynomial
            # RK4 problem.  The default flat fixture integrates exactly and
            # therefore cannot exercise a deliberately microscopic
            # n-versus-2n rejection threshold.
            profile_path = sc_dir / "high_compactness_profiles.csv"
            with profile_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames
                rows = list(reader)
            for row in rows:
                radius = float(row["r"])
                lam = 1.0e-3 * radius * radius
                row["lambda"] = repr(lam)
                row["a"] = repr(math.exp(-2.0 * lam))
                row["face_left_lambda"] = repr(
                    1.0e-3 * float(row["face_left_r"]) ** 2
                )
                row["face_right_lambda"] = repr(
                    1.0e-3 * float(row["face_right_r"]) ** 2
                )
            with profile_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir,
                ef_dir,
                manifest,
                convergence,
                output,
                extra_args=("--eikonal-convergence-tolerance", "1e-30"),
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            mapping = result["mapping"]
            self.assertGreater(
                mapping["excluded"]["forward_eikonal_not_converged"], 0
            )
            worst = mapping["forward_worst_step_refinement_event"]
            self.assertIsNotNone(worst)
            self.assertTrue({
                "tau", "r", "a", "h", "coarse_v", "fine_v",
                "coarse_log_A", "fine_log_A",
                "forward_time_step_refinement_error",
                "forward_log_A_step_refinement_error",
                "forward_step_refinement_error",
            }.issubset(worst))
            self.assertGreater(
                worst["forward_step_refinement_error"], 1.0e-30
            )
            self.assertGreater(mapping["common_domain"]["unexplained_holes"], 0)
            self.assertTrue(any(
                "unexplained" in issue for issue in result["issues"]
            ))

    def test_center_only_metric_profiles_are_diagnostic_not_certifying(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            for directory in (sc_dir, ef_dir):
                profile_path = directory / "high_compactness_profiles.csv"
                with profile_path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    retained_names = [
                        name for name in reader.fieldnames
                        if not name.startswith("face_")
                        and name not in (
                            "metric_sample_kind", "metric_cell", "center_mu_r"
                        )
                    ]
                    rows = [
                        {name: row[name] for name in retained_names}
                        for row in reader
                    ]
                with profile_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=retained_names)
                    writer.writeheader()
                    writer.writerows(rows)
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertFalse(
                result["metric_interpolation"]["production_equivalent"]
            )
            self.assertTrue(any(
                "center/face metric reconstruction" in issue
                for issue in result["issues"]
            ))

    def test_legacy_convergence_evidence_cannot_promote_fresh_pair(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            evidence = json.loads(convergence.read_text())
            evidence.pop("metric_interpolation")
            convergence.write_text(json.dumps(evidence, sort_keys=True))
            pathlib.Path(str(convergence) + ".sha256").write_text(
                self.sha256(convergence) + "  convergence.json\n"
            )
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "convergence evidence does not establish production-equivalent"
                in issue for issue in result["issues"]
            ))

    def test_convergence_eikonal_substeps_must_match_certificate(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            evidence = json.loads(convergence.read_text())
            evidence["eikonal_substeps"] = 8
            convergence.write_text(json.dumps(evidence, sort_keys=True))
            pathlib.Path(str(convergence) + ".sha256").write_text(
                self.sha256(convergence) + "  convergence.json\n"
            )
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "eikonal substeps do not match" in issue
                for issue in result["issues"]
            ))

    def test_convergence_tool_identity_must_match_terminal_tools(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            evidence = json.loads(convergence.read_text())
            evidence["diagnostic_certifier_sha256"] = "f" * 64
            convergence.write_text(json.dumps(evidence, sort_keys=True))
            pathlib.Path(str(convergence) + ".sha256").write_text(
                self.sha256(convergence) + "  convergence.json\n"
            )
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertTrue(any(
                "different certifier" in issue for issue in result["issues"]
            ))

    def test_legacy_manifest_remains_diagnostic_only_with_fresh_profiles(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            payload = json.loads(manifest.read_text())
            payload["schema_version"] = 1
            payload.pop("donor_history_format_version")
            payload.pop("saved_metric_interpolation")
            manifest.write_text(json.dumps(payload, sort_keys=True))
            self.refresh_manifest(manifest, sc_dir, ef_dir)
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "production parity requires coordinate-map manifest schema v2"
                in issue for issue in result["issues"]
            ))

    def test_unmapped_initial_hypersurfaces_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root, mapped=False)
            )
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "mapped initial data" in issue for issue in result["issues"]
            ))

    def test_single_resolution_pair_is_diagnostic_not_certified(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, _convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, None, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "single-resolution" in issue for issue in result["issues"]
            ))

    def test_manifest_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            with crossings.open("a") as stream:
                stream.write("tampered\n")
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "SHA-256" in issue for issue in result["issues"]
            ))

    def test_eikonal_richardson_component_exceedance_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            payload = json.loads(manifest.read_text())
            payload["maximum_eikonal_reverse_time_richardson_error"] = 3.0e-5
            payload["maximum_eikonal_ode_residual"] = 3.0e-5
            manifest.write_text(json.dumps(payload, sort_keys=True))
            self.refresh_manifest(manifest, sc_dir, ef_dir)
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "maximum_eikonal_reverse_time_richardson_error" in issue
                for issue in result["issues"]
            ))

    def test_eikonal_legacy_maximum_must_equal_component_maximum(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, _crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            payload = json.loads(manifest.read_text())
            payload["maximum_eikonal_reverse_time_richardson_error"] = 1.0e-6
            payload["maximum_eikonal_log_A_richardson_error"] = 2.0e-6
            payload["maximum_eikonal_ode_residual"] = 1.0e-6
            manifest.write_text(json.dumps(payload, sort_keys=True))
            self.refresh_manifest(manifest, sc_dir, ef_dir)
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(
                "not the maximum" in issue for issue in result["issues"]
            ))

    def test_authenticated_bad_covector_map_is_a_parity_failure(self):
        with tempfile.TemporaryDirectory(prefix="venus-eikonal-test-") as tmp:
            root = pathlib.Path(tmp)
            sc_dir, ef_dir, manifest, crossings, convergence = (
                self.write_coordinate_equivalent_profiles(root)
            )
            with crossings.open("r", newline="") as stream:
                reader = csv.DictReader(stream)
                names = reader.fieldnames
                rows = list(reader)
            rows[0]["p_r_ef"] = str(1.1 * float(rows[0]["p_r_ef"]))
            with crossings.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                writer.writerows(rows)
            self.refresh_manifest(
                manifest, sc_dir, ef_dir, artifact="crossings"
            )
            output = root / "comparison.json"
            completed = self.run_tool(
                sc_dir, ef_dir, manifest, convergence, output
            )
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "fail")
            self.assertIn(
                "distribution_p_r_covector_map", result["failed_fields"]
            )


if __name__ == "__main__":
    unittest.main()
