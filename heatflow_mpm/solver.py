"""HeatflowMPM substep: particles -> grid (solid fraction, flux, heat) -> temperature (blend, source, advect, diffuse) -> particles
(temperature increment, melt fraction, material) -> air (buoyancy, obstacles, projection) -> MPM (contacts)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import warp as wp

from .fields import scatter
from .grid import GridConfig, AirGrid
from .heat import HeatField, k_scatter_heat, k_blend, k_advect, k_conductivity, k_apply_fixed, k_source, k_particle_update
from .heater import Heater
from .mpm import Material, ThermalMPM

KAPPA_AIR = 2.2e-5     # thermal diffusivity of air [m^2/s]
K_PARTICLE = 1.0e-3    # face relaxation weight per particle [kg/s]: the air inside the material follows it


@wp.kernel
def k_scatter_particles(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), vol: float, h: float, n: int, k0: float,
                        alpha: wp.array3d(dtype=float), qx: wp.array3d(dtype=float), qy: wp.array3d(dtype=float), qz: wp.array3d(dtype=float),
                        kx: wp.array3d(dtype=float), ky: wp.array3d(dtype=float), kz: wp.array3d(dtype=float),
                        mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float), mz: wp.array3d(dtype=float)):
    """Solid fraction and flux (for the projection) and relaxation weights (the air moves with the material)."""
    t = wp.tid()
    x = pos[t]
    vp = vel[t]
    frac = vol / (h * h * h)
    scatter(alpha, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n, frac)
    scatter(qx, x, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n, frac * vp[0])
    scatter(qy, x, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n, frac * vp[1])
    scatter(qz, x, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1, frac * vp[2])
    scatter(kx, x, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n, k0)
    scatter(ky, x, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n, k0)
    scatter(kz, x, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1, k0)
    scatter(mx, x, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n, k0 * vp[0])
    scatter(my, x, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n, k0 * vp[1])
    scatter(mz, x, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1, k0 * vp[2])


class HeatflowMPM:
    def __init__(self, particles: np.ndarray, mat: Material = Material(), heater: Heater = Heater(), grid: GridConfig = GridConfig(),
                 spacing: float = 2.5e-3, t_init: float = 20.0, kappa_air: float = KAPPA_AIR, device: str = "cuda:0"):
        self.mat, self.heater, self.dev, self.spacing, self.kappa_air = mat, heater, device, spacing, kappa_air
        self.grid = AirGrid(grid, device)
        n, h = self.grid.n, self.grid.h
        self.heat = HeatField(n, h, grid.t_amb, device)
        cell, solid, source, rate, k_faces, m_faces, self.heater_geom = heater.masks(n, h, (grid.L / 2, grid.L / 2), grid.rho)
        self.cell_type = cell
        fixed, tfix = (cell == 2), np.where(cell == 2, heater.t_rod, 0.0)
        wall = np.zeros((n, n, n), bool)
        wall[[0, -1], :, :] = wall[:, [0, -1], :] = wall[:, :, -1] = True
        fixed, tfix = fixed | wall, np.where(fixed, tfix, np.where(wall, grid.t_amb, 0.0))
        self.heat.set_fixed(fixed, tfix)
        self.heat.set_source(source, rate, heater.t_cap)
        self.grid.set_static(solid.astype(np.float32), k_faces, m_faces)
        self.t_max = max(heater.t_rod, heater.t_cap, grid.t_amb, t_init)
        self.n = len(particles)
        self.body = ThermalMPM(particles, spacing, mat, device) if self.n else None
        self.temp = wp.full(max(self.n, 1), float(t_init), dtype=float, device=device)
        self.melt = wp.zeros(max(self.n, 1), dtype=float, device=device)

    def substep(self, dt: float):
        g, H, b, n, h = self.grid, self.heat, self.body, self.grid.n, self.grid.h
        vol = self.spacing ** 3
        g.clear_obstacle_fields()
        H.alpha.zero_()
        H.tsum.zero_()
        if b:
            wp.launch(k_scatter_particles, dim=self.n, inputs=[b.pos, b.vel, vol, h, n, K_PARTICLE, g.alpha_s, g.qx, g.qy, g.qz, g.kx, g.ky, g.kz, g.mx, g.my, g.mz], device=self.dev)
            wp.launch(k_scatter_heat, dim=self.n, inputs=[b.pos, self.temp, vol, h, n, H.alpha, H.tsum], device=self.dev)
        g.add_static()
        d3 = (n, n, n)
        wp.launch(k_blend, dim=d3, inputs=[H.T, H.alpha, H.tsum, H.Tw], device=self.dev)
        wp.launch(k_apply_fixed, dim=d3, inputs=[H.Tw, H.fixed, H.tfix], device=self.dev)
        if H.rate > 0:
            wp.launch(k_source, dim=d3, inputs=[H.Tw, H.src, H.rate, H.t_cap, dt], device=self.dev)
        wp.launch(k_advect, dim=d3, inputs=[H.Tw, g.alpha_s, g.u, g.v, g.w, H.T2, h, n, dt], device=self.dev)
        wp.launch(k_conductivity, dim=d3, inputs=[g.alpha_s, H.kap, self.kappa_air, self.mat.kappa], device=self.dev)
        H.diffuse(dt)
        if b:
            m = self.mat
            wp.launch(k_particle_update, dim=self.n, inputs=[b.pos, H.Tw, H.T2, h, n, m.t_sol, m.t_liq, m.yield_solid, m.viscosity_solid, m.viscosity_liquid,
                                                            self.t_max, self.temp, self.melt, b.yield_stress, b.viscosity], device=self.dev)
        H.T, H.T2 = H.T2, H.T
        g.step(dt, H.T)
        if b:
            b.step(dt)

    # ---- state
    def positions(self): return self.body.positions() if self.body else np.zeros((0, 3), np.float32)
    def velocities(self): return self.body.velocities() if self.body else np.zeros((0, 3), np.float32)
    def temperatures(self): return self.temp.numpy()[: self.n]
    def melt_fraction(self): return self.melt.numpy()[: self.n]
    def grid_T(self): return self.heat.T.numpy()
    def air_velocity(self): return self.grid.cell_velocity()

    def static_dict(self) -> dict:
        hg = self.heater_geom
        return dict(L=self.grid.L, h=self.grid.h, n=self.grid.n, n_particles=self.n, spacing=self.spacing, t_amb=self.grid.cfg.t_amb,
                    cell_type=self.cell_type.astype(np.uint8), heater_on=int(self.heater.on), heater_centre=np.array(hg["centre"], np.float32),
                    heater_facing=np.array(hg["facing"], np.float32), heater_tilt=self.heater.tilt, heater_radius=self.heater.radius, heater_length=self.heater.length,
                    **{f"mat_{k}": v for k, v in asdict(self.mat).items() if not isinstance(v, str)})

    def frame_dict(self, t: float, with_air: bool = True) -> dict:
        d = dict(t=np.float32(t), pos=self.positions().astype(np.float32), vel=self.velocities().astype(np.float32), temp=self.temperatures().astype(np.float16),
                 melt=self.melt_fraction().astype(np.float16), grid_T=self.grid_T().reshape(-1).astype(np.float16))
        if with_air:
            d["air_u"] = self.air_velocity().astype(np.float16)
        return d
