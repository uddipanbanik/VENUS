# User-supplied f(r,w,L) for the gridded initial-data generator.
# Inputs are broadcastable radius, radial velocity and specific angular momentum.
# Output is a nonnegative distribution in the example's G=1 code units.

"""Example user-defined distribution for a static Vlasov--Poisson shell.

The IC generator passes broadcast-ready radius r, radial velocity w, and
specific angular momentum L arrays. The returned f must be nonnegative and
broadcast to (Nr, Nw, NL). This example has compact radial support 4 < r < 6,
mean inward velocity -0.3, radial dispersion 0.2, and angular-momentum scale
0.5, in the G=1 code units of vp_custom_shell.ini. Its amplitude sets the
initial mass; it describes an initial shell rather than an equilibrium.
"""

import numpy as np


def distribution(r, w, L):
    """Return f(r,w,L) for the supplied coordinate arrays.

    The radial polynomial vanishes at both shell edges. The two Gaussians
    set the radial-velocity and angular-momentum distributions. Edit these
    factors to specify a different initial population, then check its
    integrated moments and finite-grid support before evolving it.
    """
    # Center the radial envelope on r=5. Its polynomial reaches zero at both shell edges.
    offset = r - 5.0
    radial = np.where(np.abs(offset) < 1.0, (1.0 - offset**2)**2, 0.0)
    # Shift the Gaussian to an inward mean velocity. The angular factor controls tangential motion.
    velocity = np.exp(-0.5 * ((w + 0.3) / 0.2)**2)
    angular = np.exp(-0.5 * (L / 0.5)**2)
    # Multiply the three independent factors. The amplitude sets the integrated matter content.
    return 1.0e-5 * radial * velocity * angular
