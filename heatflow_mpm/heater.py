"""Electric heater as grid masks: reflector shell and rods (solid), trough air as a power source, optional fan region."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

K_SOLID, K_FAN, CP_AIR = 1.0, 0.05, 1005.0   # face relaxation weights [kg/s]; air heat capacity [J/kg K]


@dataclass
class Heater:
    dist: float = 0.21               # distance from the box centre [m]
    direction: tuple = (-0.423, 0.906)   # horizontal unit direction from the centre to the heater
    z: float = 0.12                  # height of the trough axis [m]
    tilt: float = 15.0               # face tilted down [deg]
    radius: float = 0.055            # reflector radius [m]
    length: float = 0.22             # trough length [m]
    t_rod: float = 300.0             # rod temperature [C], Dirichlet
    power: float = 400.0             # heat injected into the trough air [W]
    t_cap: float = 120.0             # the heated air never exceeds this [C]
    fan: float = 0.0                 # outlet air speed [m/s], 0 = no fan
    on: bool = True

    def frame(self, centre_xy):
        """Head centre; unit vectors facing (toward the centre, tilted), axis (horizontal), up."""
        d = np.asarray(self.direction, float) / np.linalg.norm(self.direction)
        c = np.array([centre_xy[0] + self.dist * d[0], centre_xy[1] + self.dist * d[1], self.z])
        tl = np.deg2rad(self.tilt)
        f = np.array([-d[0] * np.cos(tl), -d[1] * np.cos(tl), -np.sin(tl)])
        a = np.array([d[1], -d[0], 0.0])
        return c, f, a, np.cross(a, f)

    def masks(self, n: int, h: float, centre_xy, rho_air: float):
        """Cell types (0 air, 1 shell, 2 rod, 3 fan, 4 trough air), solid/source masks, source rate [K/s], face weights and geometry."""
        c, f, a, up = self.frame(centre_xy)
        g = (np.arange(n) + 0.5) * h
        P = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1) - c
        s, q, r = P @ a, P @ f, P @ up
        rho = np.hypot(q, r)
        cell = np.zeros((n, n, n), np.int32)
        if self.on:
            R, L2 = self.radius, self.length / 2
            cell[(np.abs(s) <= L2) & (q <= 0) & (rho >= R - h / 2) & (rho <= R + 0.006 + h / 2)] = 1
            cell[(np.abs(np.abs(s) - L2) <= h / 2) & (q <= 0) & (rho <= R + 0.006)] = 1
            for rz in (-0.03, 0.03):
                cell[(np.abs(s) <= L2 - 0.015) & (np.hypot(q + 0.03, r - rz) <= 0.007 + h / 2)] = 2
            if self.fan > 0:
                cell[(cell == 0) & (np.abs(s) <= L2 - 0.02) & (np.abs(r) <= R - 0.01) & (q > 0) & (q <= 0.015)] = 3
            cell[(cell == 0) & (np.abs(s) <= L2 - 0.015) & (rho <= R - 0.004) & (q <= 0)] = 4
        solid = (cell == 1) | (cell == 2)
        source = (cell == 3) | (cell == 4)
        rate = self.power / (rho_air * CP_AIR * max(int(source.sum()), 1) * h ** 3) if self.on else 0.0
        k_faces, m_faces = [], []
        for axis in range(3):
            pad = [(0, 0)] * 3
            pad[axis] = (1, 1)
            sol, fan = np.pad(solid, pad), np.pad(cell == 3, pad)
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, n + 1), slice(1, n + 2)
            on_solid = sol[tuple(lo)] | sol[tuple(hi)]
            on_fan = (fan[tuple(lo)] | fan[tuple(hi)]) & ~on_solid
            k_faces.append(np.where(on_solid, K_SOLID, np.where(on_fan, K_FAN, 0.0)).astype(np.float32))
            m_faces.append(np.where(on_fan, K_FAN * self.fan * f[axis], 0.0).astype(np.float32))
        geom = dict(centre=c.tolist(), facing=f.tolist(), axis=a.tolist(), up=up.tolist())
        return cell, solid, source, rate, k_faces, m_faces, geom
