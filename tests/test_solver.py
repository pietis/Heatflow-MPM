"""GPU tests (a few minutes in total): run with `pytest -q tests` from the repository root.

conduction : implicit diffusion against the erfc half-space solution (no flow, no material)
plume      : the heater's hot air rises (buoyancy only)
fan        : a fan heater's stream reaches the material position and heats it
cold       : with the heater off a cold solid keeps its shape and nothing melts (needs NVIDIA Newton)
"""
import math, os, sys
import numpy as np
import pytest
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heatflow_mpm import HeatflowMPM, ThermalMaterial, Domain, Heater, mesh_particles  # noqa: E402
from heatflow_mpm.thermal import k_apply_fixed  # noqa: E402

wp.config.log_level = wp.LOG_WARNING
wp.init()
DEV = "cuda:0"
NO_PARTICLES = np.zeros((0, 3), np.float32)


def test_conduction_matches_erfc():
    sw = HeatflowMPM(NO_PARTICLES, ThermalMaterial(), Heater(on=False), Domain(L=0.30, walls="adiabatic", buoyancy=False), kappa_air=1e-3, device=DEV, mpm=False)
    n = sw.n; fixed = np.zeros((n, n, n), np.int32); fixed[:, :, 0] = 1
    sw.heat.set_fixed(fixed, np.where(fixed == 1, 70.0, 0.0).astype(np.float32))
    for _ in range(1000):
        sw.substep(2e-3)
    T = sw.grid_T()[n // 2, n // 2, :]; z = np.maximum((np.arange(n) + 0.5) * sw.h - 0.5 * sw.h, 0.0)
    ref = 25.0 + 45.0 * np.array([math.erfc(v / (2 * math.sqrt(1e-3 * 2.0))) for v in z])
    assert np.abs(T - ref)[: n // 2].max() < 0.1


def test_plume_rises():
    sw = HeatflowMPM(NO_PARTICLES, ThermalMaterial(), Heater(power=400.0, t_cap=120.0, fan=0.0), Domain(), device=DEV, mpm=False)
    for _ in range(1500):
        sw.substep(2e-3)
    T = sw.grid_T(); n = sw.n; hc = sw.heater_geom["centre"]; i, j = int(hc[0] / sw.h), int(hc[1] / sw.h)
    assert np.isfinite(T).all() and sw.grid.speed_max() > 0.2
    assert T[i, j, n - 2] > sw.dom.t_amb + 3.0        # ceiling over the heater is warm
    assert sw.grid.divergence_max() < 1e-3


def test_fan_heats_material_position():
    sw = HeatflowMPM(NO_PARTICLES, ThermalMaterial(), Heater(power=400.0, t_cap=120.0, fan=2.0), Domain(), device=DEV, mpm=False)
    for _ in range(1500):
        sw.substep(2e-3)
    T = sw.grid_T(); c = sw.n // 2; kb = int(0.05 / sw.h)
    assert T[c, c, kb] > 40.0 or T[c, c, kb + 4] > 40.0


def test_cold_solid_holds():
    pytest.importorskip("newton")
    P = mesh_particles(L=0.48, device=DEV)
    sw = HeatflowMPM(P, ThermalMaterial(), Heater(on=False), Domain(), t_init=20.0, device=DEV)
    for _ in range(500):
        sw.substep(2e-3)
    x = sw.positions(); tp = sw.temperatures()
    assert tp.max() < sw.mat.t_sol and sw.melt_fraction().max() == 0.0
    assert abs(x[:, 2].max() - 0.10) < 0.005
