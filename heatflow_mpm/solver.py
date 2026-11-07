"""HeatflowMPM: the coupled step.

Per substep (dt):
  1. material particles -> grid: volume fraction and volume flux (obstacle for the air), relaxation coefficients, heat (volume-weighted T)
  2. temperature: material cells take the particle temperature, Dirichlet cells are applied, the heater source heats the trough air,
     air cells are advected by the air, then the whole field is diffused implicitly (CG)
  3. particles receive the temperature increment at their position; melt fraction, yield stress and viscosity are updated
  4. air: Stable-Fluids step with buoyancy from the new temperature, obstacles and projection
  5. material: one implicit MPM step (NVIDIA Newton) with the updated per-particle material
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import warp as wp

from .grid import MacGrid, k_p2g, k_particle_block
from .thermal import HeatField, k_p2g_heat, k_blend_particles, k_advect_T, k_kappa, k_apply_fixed, k_heat_source, k_g2p_material
from .heater import Heater

GRAV = 9.81
RHO_AIR, KAPPA_AIR = 1.2, 2.2e-5     # air density [kg/m^3], thermal diffusivity [m^2/s] (physical)
K_PARTICLE = 1.0e-3                  # face relaxation coefficient per material particle [kg/s] (the air inside the material follows it)


@dataclass
class ThermalMaterial:
    """Viscoplastic material whose yield stress and viscosity follow its temperature (chocolate-like defaults)."""
    rho: float = 1300.0          # kg/m^3
    t_sol: float = 30.0          # C: below this fully solid (f = 0)
    t_liq: float = 36.0          # C: above this fully liquid (f = 1)
    yield_solid: float = 2.0e4   # Pa, yield stress at f = 0 (times (1 - f))
    viscosity_solid: float = 5.0   # Pa s at f = 0
    viscosity_liquid: float = 30.0  # Pa s at f = 1
    kappa: float = 2.0e-5        # thermal diffusivity inside the material [m^2/s] (graphics scale)
    ground_friction: float = 0.3
    strain_basis: str = "Q1"     # Newton MPM strain basis; Q1 keeps thin features of a cold solid together
    voxel_mult: float = 2.0      # MPM voxel = voxel_mult * particle spacing
    max_iterations: int = 250
    tolerance: float = 1e-5


@dataclass
class Domain:
    L: float = 0.48              # cubic box edge [m]
    h: float = 0.005             # grid cell [m]
    t_amb: float = 25.0          # room air temperature [C]
    walls: str = "fixed"         # 'fixed': side and top walls held at t_amb (heat leaves), 'adiabatic': closed box; the floor is adiabatic
    buoyancy: bool = True


class HeatflowMPM:
    def __init__(self, particles: np.ndarray, material: ThermalMaterial = ThermalMaterial(), heater: Heater = Heater(), domain: Domain = Domain(),
                 spacing: float = 2.5e-3, t_init: float = 20.0, kappa_air: float = KAPPA_AIR, device: str = "cuda:0", mpm: bool = True):
        self.mat, self.heater, self.dom, self.dev = material, heater, domain, device
        self.L, self.h, self.n = domain.L, domain.h, int(round(domain.L / domain.h))
        n, h = self.n, self.h
        self.spacing = spacing; self.v_p = spacing ** 3; self.m_p = material.rho * self.v_p
        self.n_p = len(particles); self.mpm = mpm and self.n_p > 0
        self.kappa_air = kappa_air
        self.grid = MacGrid(domain.L, h, RHO_AIR, device)
        self.heat = HeatField(n, h, domain.t_amb, device)
        # heater and walls
        cell, alpha_static, fixed, tfix, src, src_rate, faces, self.heater_geom = heater.masks(n, h, (domain.L / 2, domain.L / 2), RHO_AIR)
        if domain.walls == "fixed":
            wall = np.zeros((n, n, n), bool); wall[0, :, :] = wall[-1, :, :] = wall[:, 0, :] = wall[:, -1, :] = wall[:, :, -1] = True
            fixed = np.where(wall & (fixed == 0), 1, fixed); tfix = np.where(wall & (tfix == 0), domain.t_amb, tfix)
        self.cell_type = cell
        self.heat.set_fixed(fixed, tfix); self.heat.set_source(src, src_rate, heater.t_cap)
        W = lambda arr: wp.array(np.ascontiguousarray(arr, dtype=np.float32), dtype=float, device=device)
        self.alpha_static = W(alpha_static); self.k_static = [W(K) for K, _ in faces]; self.m_static = [W(M) for _, M in faces]
        self.t_max = max(heater.t_rod, heater.t_cap, domain.t_amb, t_init)
        # particles
        with wp.ScopedDevice(device):
            npad = max(self.n_p, 1)
            self.temp = wp.full(npad, float(t_init), dtype=float, device=device)
            self.melt = wp.zeros(npad, dtype=float, device=device)
            if self.mpm:
                import newton
                from newton.solvers import SolverImplicitMPM
                b = newton.ModelBuilder(); SolverImplicitMPM.register_custom_attributes(b)
                b.add_particles(pos=[wp.vec3(*p) for p in particles], vel=[wp.vec3(0.0)] * self.n_p, mass=[self.m_p] * self.n_p, radius=[spacing / 2] * self.n_p)
                b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=material.ground_friction))
                self.model = b.finalize(); self.model.set_gravity((0.0, 0.0, -GRAV))
                m = self.model.mpm
                m.friction.fill_(0.0); m.tensile_yield_ratio.fill_(1.0); m.yield_pressure.fill_(1.0e10)
                m.yield_stress.fill_(material.yield_solid); m.viscosity.fill_(material.viscosity_solid)
                cfg = SolverImplicitMPM.Config(); cfg.voxel_size = material.voxel_mult * spacing; cfg.grid_type = "sparse"; cfg.transfer_scheme = "apic"
                cfg.max_iterations = material.max_iterations; cfg.tolerance = material.tolerance; cfg.strain_basis = material.strain_basis
                self.solver = SolverImplicitMPM(self.model, config=cfg)
                self.s0, self.s1 = self.model.state(), self.model.state()
                self.ys, self.vis = self.model.mpm.yield_stress, self.model.mpm.viscosity
            else:
                P = np.asarray(particles, np.float32).reshape(-1, 3) if self.n_p else np.zeros((1, 3), np.float32)
                self.pos_arr = wp.array(P, dtype=wp.vec3, device=device); self.vel_arr = wp.zeros(len(P), dtype=wp.vec3, device=device)
                self.ys = wp.zeros(npad, dtype=float, device=device); self.vis = wp.zeros(npad, dtype=float, device=device)

    # ------------------------------------------------------------ state access
    def _pos_vel(self):
        return (self.s0.particle_q, self.s0.particle_qd) if self.mpm else (self.pos_arr, self.vel_arr)

    def positions(self): return self._pos_vel()[0].numpy()[: self.n_p]
    def velocities(self): return self._pos_vel()[1].numpy()[: self.n_p]
    def temperatures(self): return self.temp.numpy()[: self.n_p]
    def melt_fraction(self): return self.melt.numpy()[: self.n_p]
    def grid_T(self): return self.heat.T.numpy()
    def air_velocity(self): return self.grid.cell_velocity()

    # ------------------------------------------------------------ the step
    def substep(self, dt: float):
        dev, n, h, g, H, mat, dom = self.dev, self.n, self.h, self.grid, self.heat, self.mat, self.dom
        d3 = (n, n, n)
        pos, vel = self._pos_vel()
        g.clear_obstacle_fields(); H.alpha_m.zero_(); H.tsum.zero_()
        if self.n_p > 0:
            wp.launch(k_p2g, dim=self.n_p, inputs=[pos, vel, self.v_p, h, n, g.alpha_s, g.qx, g.qy, g.qz], device=dev)
            wp.launch(k_particle_block, dim=self.n_p, inputs=[pos, vel, h, n, K_PARTICLE, g.kx, g.ky, g.kz, g.mx, g.my, g.mz], device=dev)
            wp.launch(k_p2g_heat, dim=self.n_p, inputs=[pos, self.temp, self.v_p, h, n, H.alpha_m, H.tsum], device=dev)
        g.add_static(self.alpha_static, self.k_static, self.m_static)
        wp.launch(k_blend_particles, dim=d3, inputs=[H.T, H.alpha_m, H.tsum, H.Tw], device=dev)
        wp.launch(k_apply_fixed, dim=d3, inputs=[H.Tw, H.fixed, H.tfix], device=dev)
        if H.src_rate > 0:
            wp.launch(k_heat_source, dim=d3, inputs=[H.Tw, H.src, H.src_rate, H.t_cap, dt], device=dev)
        wp.launch(k_advect_T, dim=d3, inputs=[H.Tw, g.alpha_s, g.u, g.v, g.w, H.T2, h, n, dt], device=dev)
        wp.launch(k_kappa, dim=d3, inputs=[g.alpha_s, H.kap, self.kappa_air, mat.kappa], device=dev)
        H.diffuse_implicit(dt, H.T2, H.T2)
        if self.n_p > 0:
            wp.launch(k_g2p_material, dim=self.n_p, inputs=[pos, H.Tw, H.T2, h, n, mat.t_sol, mat.t_liq, mat.yield_solid, mat.viscosity_solid, mat.viscosity_liquid,
                                                           self.t_max, self.temp, self.melt, self.ys, self.vis], device=dev)
        H.T, H.T2 = H.T2, H.T
        g.step(dt, H.T, dom.t_amb, GRAV if dom.buoyancy else 0.0)
        if self.mpm:
            with wp.ScopedDevice(dev):
                self.solver.step(self.s0, self.s1, None, None, dt)
                self.solver.project_outside(self.s1, self.s1, dt)
            self.s0, self.s1 = self.s1, self.s0

    # ------------------------------------------------------------ output
    def static_dict(self):
        return dict(box=np.array([[0, 0, 0], [self.L] * 3], np.float32), grid_n=self.n, cell=self.h, n_particles=self.n_p, spacing=self.spacing,
                    particle_mass=self.m_p, cell_type=self.cell_type.astype(np.uint8), heater_centre=np.array(self.heater_geom["centre"], np.float32),
                    heater_facing=np.array(self.heater_geom["facing"], np.float32), heater_axis=np.array(self.heater_geom["axis"], np.float32),
                    heater_radius=self.heater.radius, heater_length=self.heater.length, heater_tilt=self.heater.tilt, heater_on=int(self.heater.on),
                    t_amb=self.dom.t_amb, **{f"mat_{k}": v for k, v in asdict(self.mat).items() if not isinstance(v, str)})

    def frame_dict(self, t: float, with_air: bool = True):
        d = dict(t=np.float32(t), pos=self.positions().astype(np.float32), vel=self.velocities().astype(np.float32),
                 temp=self.temperatures().astype(np.float16), melt=self.melt_fraction().astype(np.float16), grid_T=self.grid_T().reshape(-1).astype(np.float16))
        if with_air:
            d["air_u"] = self.air_velocity().astype(np.float16)
        return d
