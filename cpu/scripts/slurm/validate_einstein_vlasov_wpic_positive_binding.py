#!/usr/bin/env python3
# Check positive relativistic binding from a saved static-family evolution summary.
# The rest weight M0 and ADM mass use G=c=1; binding is M0 minus ADM and its fraction is normalized by M0.
# The output JSON records both exact model flags and independently checked finite numerical identities.

"""Fail-closed positive-binding audit for a VENUS WPIC summary.

The scheduled launcher uses this independent post-run check for every static
``energy_cutoff_shell`` parity and threshold job.  A driver's Boolean alone is
not sufficient: the rest mass, ADM mass, binding energy, and binding fraction
must also be finite, positive, and algebraically consistent.
"""

import argparse
import json
import math
import os
import sys


SCHEMA = "venus_v3_positive_binding_validation_v1"


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def close(left, right):
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= 5.0e-12 * scale


def audit(summary_path):
    issues = []
    try:
        with open(summary_path, "r", encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema": SCHEMA,
            "status": "fail",
            "summary": os.path.abspath(summary_path),
            "issues": ["summary cannot be read: {}".format(error)],
            "values": {},
        }

    high = summary.get("high_compactness")
    if not isinstance(high, dict):
        high = {}
        issues.append("summary has no high_compactness object")

    if high.get("initial_condition") != "energy_cutoff_shell":
        issues.append("initial_condition is not energy_cutoff_shell")
    if high.get("fixed_family_binding_diagnostics_available") is not True:
        issues.append("fixed-family binding diagnostics are unavailable")
    if high.get("fixed_family_positive_binding_energy") is not True:
        issues.append("fixed-family positive-binding flag is not true")

    names = (
        "fixed_family_rest_weight",
        "fixed_family_adm_mass",
        "fixed_family_binding_energy",
        "fixed_family_binding_fraction",
    )
    # Collect all four numerical quantities before testing their physical signs and algebraic relations.
    values = {name: high.get(name) for name in names}
    for name, value in values.items():
        if not finite_number(value):
            issues.append("{} is not a finite JSON number".format(name))

    shell = high.get("energy_cutoff_shell")
    if not isinstance(shell, dict):
        shell = {}
        issues.append("high_compactness has no energy_cutoff_shell object")
    # Require the realized energy-cutoff distribution power to equal the configured family parameter.
    configured_power = shell.get("configured_cutoff_power_k")
    realized_power = shell.get("realized_cutoff_power_k")
    values["configured_cutoff_power_k"] = configured_power
    values["realized_cutoff_power_k"] = realized_power
    if not finite_number(configured_power):
        issues.append("configured_cutoff_power_k is not a finite JSON number")
    elif not 1.0 <= float(configured_power) <= 8.0:
        issues.append("configured_cutoff_power_k is outside [1,8]")
    if not finite_number(realized_power):
        issues.append("realized_cutoff_power_k is not a finite JSON number")
    elif not 1.0 <= float(realized_power) <= 8.0:
        issues.append("realized_cutoff_power_k is outside [1,8]")
    if (finite_number(configured_power) and finite_number(realized_power)
            and not close(configured_power, realized_power)):
        issues.append("configured and realized cutoff powers differ")

    # A positive binding claim requires positive masses, positive mass deficit and the same deficit/fraction definitions.
    if all(finite_number(values[name]) for name in names):
        rest = float(values["fixed_family_rest_weight"])
        adm = float(values["fixed_family_adm_mass"])
        energy = float(values["fixed_family_binding_energy"])
        fraction = float(values["fixed_family_binding_fraction"])
        if not rest > 0.0:
            issues.append("fixed_family_rest_weight is not positive")
        if not adm > 0.0:
            issues.append("fixed_family_adm_mass is not positive")
        if not energy > 0.0:
            issues.append("fixed_family_binding_energy is not positive")
        if not 0.0 < fraction < 1.0:
            issues.append("fixed_family_binding_fraction is not in (0,1)")
        if not close(rest - adm, energy):
            issues.append("binding energy is inconsistent with M0-M_ADM")
        if rest > 0.0 and not close(energy / rest, fraction):
            issues.append("binding fraction is inconsistent with (M0-M_ADM)/M0")

    return {
        "schema": SCHEMA,
        "status": "pass" if not issues else "fail",
        "summary": os.path.abspath(summary_path),
        "definition": {
            "binding_energy": "M0-M_ADM",
            "binding_fraction": "(M0-M_ADM)/M0",
        },
        "values": values,
        "issues": issues,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = audit(args.summary)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # Write the complete audit to a temporary sibling and replace the requested report atomically.
    temporary = output + ".tmp.{}".format(os.getpid())
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, output)
    if result["status"] != "pass":
        for issue in result["issues"]:
            print("positive-binding validation: {}".format(issue), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
