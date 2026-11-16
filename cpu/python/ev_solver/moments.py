# Integrate a spherical nodal distribution into rho, p_r and j in G=c=1 units.
# Use supplied w and L quadrature weights and the spherical momentum Jacobian 2*pi*L/r^2.

"""Matter-moment quadrature for the Python Einstein--Vlasov solver.

The module integrates one spherical ``f(r,w,L)`` array into energy density,
radial pressure, and radial current with the same momentum measure
used by the C++ metric constraints.
"""

from __future__ import annotations

import numpy as np

from .params import PI


# Section: momentum-space quadrature.

# Integrates f over w and L to obtain radial rho, p, and j profiles.
def compute_moments(
    f: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    w_weights: np.ndarray,
    L_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Unit material rest mass gives epsilon=sqrt(1+w^2+L^2/r^2).
    # The azimuthal integral contributes 2*pi*L*dL*dw/r^2.
    rr = r[:, None, None]
    ww = w[None, :, None]
    LL = L[None, None, :]
    eps = np.sqrt(1.0 + ww**2 + (LL / rr) ** 2)

    # Energy, squared radial momentum divided by energy, and radial momentum give rho, p_r and j.
    rho_int = eps * f * LL
    p_int = (ww**2 / eps) * f * LL
    j_int = ww * f * LL

    # Contract only the two momentum axes. One moment value remains at each radius.
    factor = 2.0 * PI / (r * r)
    rho = factor * np.einsum("ijk,j,k->i", rho_int, w_weights, L_weights, optimize=True)
    p = factor * np.einsum("ijk,j,k->i", p_int, w_weights, L_weights, optimize=True)
    j = factor * np.einsum("ijk,j,k->i", j_int, w_weights, L_weights, optimize=True)
    return rho, p, j
