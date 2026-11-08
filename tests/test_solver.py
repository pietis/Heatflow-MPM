"""GPU tests: python -m pytest -q tests"""
import math, os, sys
import numpy as np
import pytest
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heatflow_mpm import HeatflowMPM, Material, Heater, GridConfig, mesh_particles  # noqa: E402

wp.config.log_level = wp.LOG_WARNING
wp.init()
DEV = "cuda:0"
NONE = np.zeros((0, 3), np.float32)


def run(sim, seconds, dt=2e-3):
    for _ in range(int(seconds / dt)):
        sim.substep(dt)


def test_implicit_conduction_matches_erfc():
    sim = HeatflowMPM(NONE, heater=Heater(on=False), grid=GridConfig(L=0.30, buoyancy=False), kappa_air=1e-3, device=DEV)
    n = sim.grid.n
    fixed = np.zeros((n, n, n), bool); fixed[:, :, 0] = True          # hot floor, everything else free
    sim.heat.set_fixed(fixed, np.where(fixed, 70.0, 0.0))
    run(sim, 2.0)
    T = sim.grid_T()[n // 2, n // 2, :]
    z = np.maximum((np.arange(n) + 0.5) * sim.grid.h - sim.grid.h / 2, 0.0)
    ref = 25.0 + 45.0 * np.array([math.erfc(v / (2 * math.sqrt(1e-3 * 2.0))) for v in z])
    assert np.abs(T - ref)[: n // 2].max() < 0.1


def test_heater_plume_rises():
    sim = HeatflowMPM(NONE, heater=Heater(power=400.0, t_cap=120.0), device=DEV)
    run(sim, 3.0)
    T = sim.grid_T(); hc = sim.heater_geom["centre"]; i, j = int(hc[0] / sim.grid.h), int(hc[1] / sim.grid.h)
    assert np.isfinite(T).all() and sim.grid.speed_max() > 0.2
    assert T[i, j, sim.grid.n - 2] > sim.grid.cfg.t_amb + 3.0
    assert sim.grid.divergence_max() < 1e-3


def test_fan_heats_the_material_position():
    sim = HeatflowMPM(NONE, heater=Heater(power=400.0, t_cap=120.0, fan=2.0), device=DEV)
    run(sim, 3.0)
    T = sim.grid_T(); c = sim.grid.n // 2; kb = int(0.05 / sim.grid.h)
    assert max(T[c, c, kb], T[c, c, kb + 4]) > 40.0


def test_cold_solid_holds_its_shape():
    pytest.importorskip("newton")
    sim = HeatflowMPM(mesh_particles(device=DEV), heater=Heater(on=False), t_init=20.0, device=DEV)
    run(sim, 1.0)
    assert sim.temperatures().max() < sim.mat.t_sol and sim.melt_fraction().max() == 0.0
    assert abs(sim.positions()[:, 2].max() - 0.10) < 0.005
