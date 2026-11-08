"""Heatflow-MPM: Eulerian air (velocity + temperature) two-way coupled to thermal implicit MPM (NVIDIA Newton)."""
from .solver import HeatflowMPM, KAPPA_AIR  # noqa: F401
from .grid import GridConfig, AirGrid  # noqa: F401
from .heat import HeatField  # noqa: F401
from .heater import Heater  # noqa: F401
from .mpm import Material, ThermalMPM  # noqa: F401
from .shapes import load_mesh, sample_interior, mesh_particles, BUNNY  # noqa: F401

__version__ = "0.2.0"
