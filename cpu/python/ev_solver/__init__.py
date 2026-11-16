# Import the parameter record and high-level reference solver through ev_solver.
# These exports return Python objects; importing the package starts no evolution.

"""Developer-friendly Einstein--Vlasov reference package.

The public surface deliberately exports only the parameter record and the
high-level run function.  Numerical kernels remain in focused submodules so a
developer can compare one physical operation with the modular C++ source.
"""

# Keep the small public API here. Numerical definitions remain in their respective modules.
from .params import EVParams
from .solver import run_simulation

__all__ = ["EVParams", "run_simulation"]
