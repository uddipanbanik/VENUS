# Read an INI configuration and command-line overrides into EVParams.
# File paths are resolved relative to the INI file. Numerical values use solver code units.

"""Configuration parser for the Python Einstein--Vlasov reference solver.

The C++ solver is the production path, but this Python layer is useful for
small reference runs and plotting-compatible experiments.  This file keeps the
same permissive key style as the C++ parser: underscores and hyphens are
treated equivalently, and command-line options override params.ini values.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .params import EVParams


# Section: argparse compatibility.
# Older Python installations do not provide BooleanOptionalAction, so the
# fallback below implements paired --flag/--no-flag options manually.
if hasattr(argparse, "BooleanOptionalAction"):
    BooleanOptionalAction = argparse.BooleanOptionalAction
else:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            strings = []
            for option in option_strings:
                strings.append(option)
                if option.startswith("--"):
                    strings.append("--no-" + option[2:])
            super().__init__(strings, dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not (option_string or "").startswith("--no-"))


# Section: config-file parsing helpers.
# These helpers normalize keys, strip comments, and coerce string values to the
# dataclass field types used by EVParams.

# Normalizes config and CLI keys for permissive underscore/hyphen handling.
def normalize_config_key(key: str) -> str:
    return key.strip().lstrip("-").replace("_", "-").lower()


# Removes unquoted inline comments from config lines.
def strip_config_comment(line: str) -> str:
    # Track quoted strings so a # or semicolon inside a path remains part of its value.
    in_single = False
    in_double = False
    for i, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double and char in "#;":
            return line[:i]
    return line


# Parses a human-readable boolean string from a config value.
def parse_bool_text(value: str) -> bool:
    token = normalize_config_key(value)
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


# Converts raw config text to the type of the corresponding default value.
def coerce_config_value(value: str, default: Any) -> Any:
    value = value.strip().strip('"').strip("'")
    if isinstance(default, bool):
        return parse_bool_text(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


# Section: config discovery and precedence.
# Defaults come from EVParams, then params.ini, then explicit CLI arguments.

# Loads a params.ini file into a defaults dictionary and resolves relative paths.
def load_config_defaults(path: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    # Map the accepted spelling of each option back to its dataclass field.
    key_to_field = {normalize_config_key(name): name for name in defaults}
    path_fields = {"initial_data_file", "output_dir", "restart_from"}
    out = dict(defaults)
    with path.open() as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = strip_config_comment(raw).strip()
            if not line or (line.startswith("[") and line.endswith("]")):
                continue
            sep = "=" if "=" in line else ":"
            if sep not in line:
                raise ValueError(f"Expected 'key = value' in {path}:{line_no}")
            key, value = (part.strip() for part in line.split(sep, 1))
            normalized = normalize_config_key(key)
            if normalized not in key_to_field:
                raise ValueError(f"Unknown Python-solver config key {key!r} in {path}:{line_no}")
            field = key_to_field[normalized]
            coerced = coerce_config_value(value, out[field])
            # Resolve relative data/output paths beside the configuration, where the user specified them.
            if field in path_fields and coerced:
                candidate = Path(str(coerced))
                if not candidate.is_absolute():
                    coerced = str((path.parent / candidate).resolve())
            out[field] = coerced
    return out


# Finds input/params.ini, falling back to root params.ini.
def default_config_path() -> Path | None:
    package_root = Path(__file__).resolve().parents[2]
    for candidate in (
        Path.cwd() / "input" / "params.ini",
        Path.cwd() / "params.ini",
        package_root / "input" / "params.ini",
        package_root / "params.ini",
        Path(__file__).resolve().with_name("params.ini"),
    ):
        if candidate.exists():
            return candidate
    return None


# Section: command-line interface.

# Combines defaults, config values, and CLI overrides into an EVParams object.
def parse_args() -> EVParams:
    """Read defaults, apply the selected INI file, then apply CLI overrides.

    The preliminary parser finds --config before the full parser is built.
    Relative input paths are resolved beside that configuration file.
    The returned EVParams is subsequently checked by solver_setup.
    """
    # Read --config first, so its values can become defaults for the full command-line parser.
    base_defaults = asdict(EVParams())
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, remaining = pre.parse_known_args()
    config_path = known.config if known.config else default_config_path()
    defaults = load_config_defaults(config_path, base_defaults) if config_path else base_defaults

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--physics-model", default=defaults["physics_model"])
    parser.add_argument("--vp-universe", default=defaults["vp_universe"])
    parser.add_argument("--vp-alpha", type=float, default=defaults["vp_alpha"])
    parser.add_argument("--vp-time-offset", type=float, default=defaults["vp_time_offset"])
    parser.add_argument("--vp-gravity", type=float, default=defaults["vp_gravity"])
    parser.add_argument("--vp-poisson-source", default=defaults["vp_poisson_source"])
    # Grid sizes count nodal values of f in radius, radial momentum and specific angular momentum.
    parser.add_argument("--Nr", type=int, default=defaults["Nr"])
    parser.add_argument("--Nw", type=int, default=defaults["Nw"])
    parser.add_argument("--NL", type=int, default=defaults["NL"])
    parser.add_argument("--r-min", type=float, default=defaults["r_min"])
    parser.add_argument("--r-max", type=float, default=defaults["r_max"])
    parser.add_argument(
        "--radial-grid",
        choices=("log", "linear", "piecewise-linear"),
        default=defaults["radial_grid"],
    )
    parser.add_argument(
        "--custom-radial-coordinate",
        choices=("linear", "log"),
        default=defaults["custom_radial_coordinate"],
    )
    parser.add_argument("--inner-refine-radius", type=float, default=defaults["inner_refine_radius"])
    parser.add_argument("--inner-refine-cells", type=int, default=defaults["inner_refine_cells"])
    parser.add_argument("--w-max", type=float, default=defaults["w_max"])
    parser.add_argument("--L-max", type=float, default=defaults["L_max"])
    parser.add_argument("--w-stretch", type=float, default=defaults["w_stretch"])
    parser.add_argument("--w-center", type=float, default=defaults["w_center"])
    parser.add_argument("--adaptive-w", action=BooleanOptionalAction, default=defaults["adaptive_w"])
    parser.add_argument("--adaptive-w-every", type=int, default=defaults["adaptive_w_every"])
    parser.add_argument("--adaptive-w-trigger", type=float, default=defaults["adaptive_w_trigger"])
    parser.add_argument("--adaptive-w-margin", type=float, default=defaults["adaptive_w_margin"])
    parser.add_argument("--adaptive-w-growth", type=float, default=defaults["adaptive_w_growth"])
    parser.add_argument("--adaptive-w-max", type=float, default=defaults["adaptive_w_max"])
    # Time limits and output cadences control evolution and saved diagnostics separately.
    parser.add_argument("--t-end", type=float, default=defaults["t_end"])
    parser.add_argument("--cfl", type=float, default=defaults["cfl"])
    parser.add_argument("--dt-max", type=float, default=defaults["dt_max"])
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument("--save-every", type=int, default=defaults["save_every"])
    parser.add_argument("--phase-every", type=int, default=defaults["phase_every"])
    parser.add_argument("--phase-r-stride", type=int, default=defaults["phase_r_stride"])
    parser.add_argument("--phase-w-stride", type=int, default=defaults["phase_w_stride"])
    parser.add_argument("--restart-every", type=int, default=defaults["restart_every"])
    parser.add_argument("--integrator", default=defaults["integrator"])
    parser.add_argument("--time-integrator-order", type=int, default=defaults["time_integrator_order"])
    parser.add_argument("--time-integrator-substeps", type=int, default=defaults["time_integrator_substeps"])
    parser.add_argument("--inner-boundary", default=defaults["inner_boundary"])
    parser.add_argument("--outer-boundary", default=defaults["outer_boundary"])
    parser.add_argument("--momentum-boundary", default=defaults["momentum_boundary"])
    parser.add_argument("--central-mass", type=float, default=defaults["central_mass"])
    # Choose the distribution source before setting its shape and mass-normalization controls.
    parser.add_argument("--initial-data", default=defaults["initial_data"])
    parser.add_argument("--initial-data-file", default=defaults["initial_data_file"])
    parser.add_argument(
        "--custom-ic-normalize-mass",
        action=BooleanOptionalAction,
        default=defaults["custom_ic_normalize_mass"],
    )
    parser.add_argument("--custom-ic-target-mass", type=float, default=defaults["custom_ic_target_mass"])
    parser.add_argument(
        "--shell-initial-df", action=BooleanOptionalAction, default=defaults["shell_initial_df"]
    )
    parser.add_argument("--R0", type=float, default=defaults["R0"])
    parser.add_argument("--dR", type=float, default=defaults["dR"])
    parser.add_argument("--top-hat-shell", action=BooleanOptionalAction, default=defaults["top_hat_shell"])
    parser.add_argument(
        "--top-hat-density-shell",
        action=BooleanOptionalAction,
        default=defaults["top_hat_density_shell"],
    )
    parser.add_argument(
        "--gaussian-density-shell",
        action=BooleanOptionalAction,
        default=defaults["gaussian_density_shell"],
    )
    parser.add_argument(
        "--double-gaussian-density-shell",
        action=BooleanOptionalAction,
        default=defaults["double_gaussian_density_shell"],
    )
    parser.add_argument("--R0-2", type=float, default=defaults["R0_2"])
    parser.add_argument("--dR-2", type=float, default=defaults["dR_2"])
    parser.add_argument("--amp-2", type=float, default=defaults["amp_2"])
    parser.add_argument("--w-in-2", type=float, default=defaults["w_in_2"])
    parser.add_argument("--beta", type=float, default=defaults["beta"])
    parser.add_argument("--amp", type=float, default=defaults["amp"])
    parser.add_argument("--w-in", type=float, default=defaults["w_in"])
    parser.add_argument("--initial-metric-iters", type=int, default=defaults["initial_metric_iters"])
    parser.add_argument("--metric-floor", type=float, default=defaults["metric_floor"])
    parser.add_argument("--gap-stop", type=float, default=defaults["gap_stop"])
    parser.add_argument("--active-floor-rel", type=float, default=defaults["active_floor_rel"])
    parser.add_argument("--active-floor-abs", type=float, default=defaults["active_floor_abs"])
    parser.add_argument("--f-floor", type=float, default=defaults["f_floor"])
    parser.add_argument("--threads", type=int, default=defaults["threads"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--restart-from", default=defaults["restart_from"])
    args = parser.parse_args(remaining)
    # Build one typed parameter record. Lower bounds on counts keep enabled cadences well defined.
    return EVParams(
        physics_model=args.physics_model,
        vp_universe=args.vp_universe,
        vp_alpha=args.vp_alpha,
        vp_time_offset=args.vp_time_offset,
        vp_gravity=args.vp_gravity,
        vp_poisson_source=args.vp_poisson_source,
        Nr=args.Nr,
        Nw=args.Nw,
        NL=args.NL,
        r_min=args.r_min,
        r_max=args.r_max,
        radial_grid=args.radial_grid,
        custom_radial_coordinate=args.custom_radial_coordinate,
        inner_refine_radius=args.inner_refine_radius,
        inner_refine_cells=args.inner_refine_cells,
        w_max=args.w_max,
        L_max=args.L_max,
        w_stretch=args.w_stretch,
        w_center=args.w_center,
        adaptive_w=args.adaptive_w,
        adaptive_w_every=max(1, args.adaptive_w_every),
        adaptive_w_trigger=args.adaptive_w_trigger,
        adaptive_w_margin=args.adaptive_w_margin,
        adaptive_w_growth=args.adaptive_w_growth,
        adaptive_w_max=args.adaptive_w_max,
        t_end=args.t_end,
        cfl=args.cfl,
        dt_max=args.dt_max,
        max_steps=args.max_steps,
        save_every=max(1, args.save_every),
        phase_every=max(0, args.phase_every),
        phase_r_stride=max(1, args.phase_r_stride),
        phase_w_stride=max(1, args.phase_w_stride),
        restart_every=max(0, args.restart_every),
        integrator=args.integrator,
        time_integrator_order=args.time_integrator_order,
        time_integrator_substeps=max(1, args.time_integrator_substeps),
        inner_boundary=args.inner_boundary,
        outer_boundary=args.outer_boundary,
        momentum_boundary=args.momentum_boundary,
        central_mass=args.central_mass,
        initial_data=args.initial_data,
        initial_data_file=args.initial_data_file,
        custom_ic_normalize_mass=args.custom_ic_normalize_mass,
        custom_ic_target_mass=args.custom_ic_target_mass,
        shell_initial_df=args.shell_initial_df
        or args.top_hat_shell
        or args.top_hat_density_shell
        or args.gaussian_density_shell,
        R0=args.R0,
        dR=args.dR,
        top_hat_shell=args.top_hat_shell,
        top_hat_density_shell=args.top_hat_density_shell,
        gaussian_density_shell=args.gaussian_density_shell,
        double_gaussian_density_shell=args.double_gaussian_density_shell,
        R0_2=args.R0_2,
        dR_2=args.dR_2,
        amp_2=args.amp_2,
        w_in_2=args.w_in_2,
        beta=args.beta,
        amp=args.amp,
        w_in=args.w_in,
        initial_metric_iters=max(0, args.initial_metric_iters),
        metric_floor=args.metric_floor,
        gap_stop=args.gap_stop,
        active_floor_rel=args.active_floor_rel,
        active_floor_abs=args.active_floor_abs,
        f_floor=args.f_floor,
        threads=max(1, args.threads),
        output_dir=args.output_dir,
        restart_from=args.restart_from,
    )
