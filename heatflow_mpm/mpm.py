"""Material particles as an implicit MPM body (NVIDIA Newton) whose yield stress and viscosity are set per particle each step."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import warp as wp


@dataclass
class Material:
    rho: float = 1300.0              # kg/m^3
    t_sol: float = 30.0              # fully solid below [C]
    t_liq: float = 36.0              # fully liquid above [C]
    yield_solid: float = 2.0e4       # Pa at f = 0
    viscosity_solid: float = 5.0     # Pa s at f = 0
    viscosity_liquid: float = 30.0   # Pa s at f = 1
    kappa: float = 2.0e-5            # thermal diffusivity in the material [m^2/s]
    ground_friction: float = 0.3
    strain_basis: str = "Q1"         # keeps thin features of a cold solid together
    voxel_mult: float = 2.0          # MPM voxel = voxel_mult x particle spacing
    max_iterations: int = 250
    tolerance: float = 1.0e-5


class ThermalMPM:
    def __init__(self, particles: np.ndarray, spacing: float, mat: Material, device: str):
        import newton
        from newton.solvers import SolverImplicitMPM
        self.dev, self.n = device, len(particles)
        mass = mat.rho * spacing ** 3
        with wp.ScopedDevice(device):
            b = newton.ModelBuilder()
            SolverImplicitMPM.register_custom_attributes(b)
            b.add_particles(pos=[wp.vec3(*p) for p in particles], vel=[wp.vec3(0.0)] * self.n, mass=[mass] * self.n, radius=[spacing / 2] * self.n)
            b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mat.ground_friction))
            self.model = b.finalize()
            self.model.set_gravity((0.0, 0.0, -9.81))
            m = self.model.mpm
            m.friction.fill_(0.0)
            m.tensile_yield_ratio.fill_(1.0)
            m.yield_pressure.fill_(1.0e10)
            m.yield_stress.fill_(mat.yield_solid)
            m.viscosity.fill_(mat.viscosity_solid)
            cfg = SolverImplicitMPM.Config()
            cfg.voxel_size = mat.voxel_mult * spacing
            cfg.grid_type = "sparse"
            cfg.transfer_scheme = "apic"
            cfg.max_iterations = mat.max_iterations
            cfg.tolerance = mat.tolerance
            cfg.strain_basis = mat.strain_basis
            self.solver = SolverImplicitMPM(self.model, config=cfg)
            self.s0, self.s1 = self.model.state(), self.model.state()

    @property
    def pos(self) -> wp.array:
        return self.s0.particle_q

    @property
    def vel(self) -> wp.array:
        return self.s0.particle_qd

    @property
    def yield_stress(self) -> wp.array:
        return self.model.mpm.yield_stress

    @property
    def viscosity(self) -> wp.array:
        return self.model.mpm.viscosity

    def step(self, dt: float):
        with wp.ScopedDevice(self.dev):
            self.solver.step(self.s0, self.s1, None, None, dt)
            self.solver.project_outside(self.s1, self.s1, dt)
        self.s0, self.s1 = self.s1, self.s0

    def positions(self) -> np.ndarray:
        return self.s0.particle_q.numpy()

    def velocities(self) -> np.ndarray:
        return self.s0.particle_qd.numpy()
