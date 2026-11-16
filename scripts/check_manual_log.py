#!/usr/bin/env python3
# Check a completed LaTeX log for unresolved references, citations or duplicate labels.
# Input is one log path. Successful exit confirms the listed cross-reference errors are absent.

"""Reject unresolved references and duplicate labels after a LaTeX build."""
from pathlib import Path
import argparse
import re


def main() -> None:
    """Read the converged build log using Python available on every build host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    text = args.log.read_text(errors="replace")
    # Match full offending log lines so an error message identifies the reference problem to repair.
    failures = re.findall(
        r"[^\n]*(?:undefined references|undefined citations|Citation[^\n]*undefined|"
        r"Reference[^\n]*undefined|Label\(s\) may have changed|multiply[- ]defined labels|"
        r"Label[^\n]*multiply defined)[^\n]*", text, flags=re.IGNORECASE)
    if failures:
        raise SystemExit("Unresolved manual build:\n" + "\n".join(failures))
    print("Manual references and labels are resolved")


if __name__ == "__main__":
    main()
