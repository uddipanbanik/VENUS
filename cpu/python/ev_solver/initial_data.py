# Read EVICRWL1 nodal distributions or construct analytic shell data.
# Inputs use G=c=1 and unit particle rest mass. Return r,w,L and f with shape (Nr,Nw,NL).
# Density-shell options normalize their radial envelope through the actual momentum moment.

"""Initial-data readers and analytic distribution builders.

This module supports the EVICRWL1 binary format and the built-in analytic shell
families used by small Python reference runs.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from .grids import radial_coordinate
from .params import EVParams, PI


# Section: binary EVICRWL1 input.

# Reads and validates an EVICRWL1 binary f(r,w,L) initial-data file.
def read_initial_data_rwl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).split(b"\0", 1)[0]
        if magic != b"EVICRWL1":
            raise ValueError(f"{path} is not an EVICRWL1 initial-data file")
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"{path} ended before the initial-data header was complete")
        version, nr, nw, nl = struct.unpack("<iiii", header)
        if version != 1:
            raise ValueError(f"{path} has unsupported initial-data version {version}")
        if nr < 3 or nw < 3 or nl < 2:
            raise ValueError("Initial-data file needs Nr >= 3, Nw >= 3, and NL >= 2")
        # The payload contains three coordinate arrays followed by Nr*Nw*NL distribution values.
        count = nr + nw + nl + nr * nw * nl
        raw = np.fromfile(handle, dtype="<f8", count=count)
    if raw.size != count:
        raise ValueError(f"{path} ended before all grids and f values were read")
    # Advance through the concatenated arrays in exactly the order used by the binary writer.
    offset = 0
    r = np.asarray(raw[offset : offset + nr], dtype=float)
    offset += nr
    w = np.asarray(raw[offset : offset + nw], dtype=float)
    offset += nw
    L = np.asarray(raw[offset : offset + nl], dtype=float)
    offset += nl
    f = np.asarray(raw[offset:], dtype=float).reshape((nr, nw, nl))
    # Monotone coordinates give positive quadrature intervals and unambiguous interpolation brackets.
    if r[0] <= 0.0 or np.any(np.diff(r) <= 0.0):
        raise ValueError("Initial-data r grid must be positive and strictly increasing")
    if np.any(np.diff(w) <= 0.0):
        raise ValueError("Initial-data w grid must be strictly increasing")
    if L[0] < 0.0 or np.any(np.diff(L) <= 0.0):
        raise ValueError("Initial-data L grid must be nonnegative and strictly increasing")
    if np.any(~np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("Initial-data f values must be finite and nonnegative")
    return r, w, L, f


# Loads binary IC grids, updates parameter metadata, and returns grid spacings.
def grids_from_binary_initial_data(par: EVParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    r, w, L, f = read_initial_data_rwl(Path(par.initial_data_file))
    # The supplied file owns its grid. Update parameter metadata to describe that same array.
    par.Nr = int(r.size)
    par.Nw = int(w.size)
    par.NL = int(L.size)
    par.r_min = float(r[0])
    par.r_max = float(r[-1])
    par.L_max = float(L[-1])
    par.w_max = float(max(abs(w[0]), abs(w[-1])))
    par.radial_grid = "custom"
    x_r = np.asarray(radial_coordinate(r, par))
    return r, w, L, f, float(np.min(np.diff(x_r))), float(np.min(np.diff(w)))


# Section: analytic shell initial data.

# Evaluates the requested analytic radial shell envelope.
def shell_profile(r: np.ndarray, par: EVParams) -> np.ndarray:
    if par.top_hat_shell or par.top_hat_density_shell:
        return ((r >= par.R0 - 0.5 * par.dR) & (r <= par.R0 + 0.5 * par.dR)).astype(float)
    return np.exp(-0.5 * ((r - par.R0) / par.dR) ** 2)


# Initial distribution functions.  Density shell modes correct the radial
# envelope with the actual moment integral, avoiding a baked-in r^-2 density.
# Constructs the built-in analytic initial distribution on the full Python grid.
def initial_distribution(
    r: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    w_weights: np.ndarray,
    L_weights: np.ndarray,
    mu_seed: np.ndarray,
    par: EVParams,
) -> np.ndarray:
    """Evaluate the analytic distribution on the (r,w,L) nodal grid.

    Shell data use the lapse at the shell center to set the momentum
    distribution. Density-shell modes additionally divide out the actual
    radius-dependent energy-density moment, so their prescribed radial
    envelope describes density. The other analytic family uses the local
    lapse. solver_setup repeats this construction with the solved metric
    for the requested number of initial iterations.
    """
    rr = r[:, None, None]
    ww = w[None, :, None]
    LL = L[None, None, :]

    if par.shell_initial_df:
        rb = par.R0
        # Use the lapse at the shell center for a common momentum seed across its radial envelope.
        mu_b = float(np.interp(float(radial_coordinate(rb, par)), np.asarray(radial_coordinate(r, par)), mu_seed))
        b_shell = max(par.beta * math.exp(mu_b), np.finfo(float).tiny)
        norm = par.amp * math.sqrt(b_shell / (2.0 * PI)) * b_shell / (rb * rb)
        eps_shift = np.sqrt(1.0 + (ww - par.w_in) ** 2 + (LL / rb) ** 2)
        base = norm * np.exp(-b_shell * (eps_shift - 1.0))[0]
        shell = shell_profile(r, par)
        if par.top_hat_density_shell or par.gaussian_density_shell:
            eps_rb = np.sqrt(1.0 + w[:, None] ** 2 + (L[None, :] / rb) ** 2)
            # Compare energy-density integrals at r and R0. Their ratio removes the geometric r^-2 factor.
            I_rb = float(np.einsum("jk,jk,j,k,k->", eps_rb, base, w_weights, L_weights, L, optimize=True))
            for i, ri in enumerate(r):
                if shell[i] <= 0.0:
                    continue
                eps_r = np.sqrt(1.0 + w[:, None] ** 2 + (L[None, :] / ri) ** 2)
                I_r = max(
                    float(np.einsum("jk,jk,j,k,k->", eps_r, base, w_weights, L_weights, L, optimize=True)),
                    np.finfo(float).tiny,
                )
                # Rescale f so the requested shell envelope describes rho after momentum integration.
                shell[i] *= (ri * ri / (rb * rb)) * (I_rb / I_r)
        return shell[:, None, None] * base[None, :, :]

    # The local-lapse family varies its momentum scale with radius through beta*exp(mu).
    b = np.maximum(par.beta * np.exp(mu_seed), np.finfo(float).tiny)[:, None, None]
    norm = par.amp * np.sqrt(b / (2.0 * PI)) * b / (rr * rr)
    eps_shift = np.sqrt(1.0 + (ww - par.w_in) ** 2 + (LL / rr) ** 2)
    return norm * np.exp(-b * (eps_shift - 1.0))
