"""Heatflow-MPM: an Eulerian heat-and-airflow surroundings solver for thermal MPM systems.

The air is a Stable-Fluids grid (semi-Lagrangian advection, pressure projection, Boussinesq buoyancy) that moves and conducts heat;
the material is NVIDIA Newton's implicit MPM whose yield stress and viscosity follow the temperature it receives from the grid.
"""
from .solver import HeatflowMPM, ThermalMaterial, Domain  # noqa: F401
from .heater import Heater  # noqa: F401
from .geometry import load_mesh, sample_interior, mesh_particles  # noqa: F401

__version__ = "0.1.0"
