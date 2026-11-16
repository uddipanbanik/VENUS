# Define the Python reference solver's configuration values.
# Lengths, times and momenta use G=c=1 with unit rest mass. config.py maps file/CLI values into this record.
# solver_setup validates which listed controls the Python implementation supports.

"""Parameter dataclass for the Python solver and compatible config files.

Defaults, derived grid properties, and validation live together so every
Python numerical module receives one already checked physical configuration.
The C++ ``Params`` record remains the production source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PI = math.pi


# Section: public parameter container.
# Keep field names aligned with config.py and the C++ Params structure where
# possible, so params.ini files transfer cleanly between Python tests and C++
# production runs.

# Stores every user-controlled parameter used by the Python solver and compatible config files.
@dataclass
class EVParams:
    # The shared option names also describe C++ modes. Python setup admits its implemented Einstein--Vlasov path.
    physics_model: str = "einstein_vlasov"
    vp_universe: str = "static"
    vp_alpha: float = 2.0 / 3.0
    vp_time_offset: float = 1.0
    vp_gravity: float = 1.0
    vp_poisson_source: str = "total_density"

    # Phase-space resolution.
    Nr: int = 260
    Nw: int = 240
    NL: int = 48

    # Radial grid and momentum-domain controls.
    r_min: float = 1.0e-3
    r_max: float = 60.0
    radial_grid: str = "log"
    custom_radial_coordinate: str = "linear"
    inner_refine_radius: float = 0.1
    inner_refine_cells: int = 96
    w_max: float = 8.0
    L_max: float = 6.0
    w_stretch: float = 2.5
    w_center: float = 0.0
    adaptive_w: bool = False
    adaptive_w_every: int = 20
    adaptive_w_trigger: float = 0.65
    adaptive_w_margin: float = 1.8
    adaptive_w_growth: float = 1.6
    adaptive_w_max: float = 0.0

    # Time stepping, output cadence, and implemented algorithms.
    t_end: float = 20.0
    cfl: float = 0.35
    dt_max: float = 0.04
    max_steps: int = 200000
    save_every: int = 20
    phase_every: int = 0
    phase_r_stride: int = 1
    phase_w_stride: int = 1
    restart_every: int = 0
    integrator: str = "semi_lagrangian_kick_drift_kick"
    time_integrator_order: int = 2
    time_integrator_substeps: int = 1
    inner_boundary: str = "reflecting"
    outer_boundary: str = "outflow_vacuum"
    momentum_boundary: str = "outflow_vacuum"

    # This prescribed inner mass is separate from the matter integrated from f.
    central_mass: float = 0.0

    # Initial-data source.  "analytic" keeps the built-in seed families.
    # "binary_rwl" reads f(r,w,L) plus all three grids from a binary file.
    initial_data: str = "analytic"
    initial_data_file: str = "initial_data/example_shell_rwl.bin"
    custom_ic_normalize_mass: bool = False
    custom_ic_target_mass: float = 1.0e-2

    # Initial data.  Shell modes evaluate the Maxwell-Juttner momentum seed at
    # R0 so the radial envelope is controlled independently of the momentum DF.
    shell_initial_df: bool = False
    R0: float = 5.0
    dR: float = 1.0
    top_hat_shell: bool = False
    top_hat_density_shell: bool = False
    gaussian_density_shell: bool = False
    double_gaussian_density_shell: bool = False
    R0_2: float = 10.0
    dR_2: float = 1.0
    amp_2: float = 1.0e-2
    w_in_2: float = -1.0
    beta: float = 4.0
    amp: float = 1.0e-2
    w_in: float = -0.6
    initial_metric_iters: int = 4

    # metric_floor protects division. gap_stop sets the physical compactness stopping condition.
    metric_floor: float = 1.0e-10
    gap_stop: float = 2.0e-5
    active_floor_rel: float = 1.0e-12
    active_floor_abs: float = 0.0
    f_floor: float = 0.0

    # File layout and restart control.
    threads: int = 1
    output_dir: str = "data/venus_einstein_vlasov_python"
    restart_from: str = ""
