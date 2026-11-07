"""An electric heater as a static obstacle and heat source on the grid.

Geometry (metres): a half-cylinder reflector shell (solid, adiabatic) of radius `radius` and length `length`, open toward the material,
two rods inside the trough held at `t_rod` (Dirichlet, solid), the trough air as the source region of `power` watts capped at `t_cap`,
and an optional fan region in front of the rods with a prescribed air velocity `fan` (m/s) along the facing direction. The head stands
`dist` metres from the domain centre in the horizontal direction `direction`, at height `z`, facing the centre and tilted down by `tilt` deg.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

K_SOLID, K_FAN = 1.0, 0.05   # face relaxation coefficients [kg/s] for static solids and the fan region
CP_AIR = 1005.0              # J/(kg K)


@dataclass
class Heater:
    dist: float = 0.21
    direction: tuple = (-0.423, 0.906)
    z: float = 0.12
    tilt: float = 15.0
    radius: float = 0.055
    length: float = 0.22
    t_rod: float = 300.0
    power: float = 400.0
    t_cap: float = 120.0
    fan: float = 0.0
    on: bool = True

    def frame(self, centre_xy):
        """Head centre and unit vectors: facing f (toward the centre, tilted down), axis a (horizontal), up u = a x f."""
        d = np.asarray(self.direction, float); d /= np.linalg.norm(d)
        c = np.array([centre_xy[0] + self.dist * d[0], centre_xy[1] + self.dist * d[1], self.z])
        fh = -d
        tl = np.deg2rad(self.tilt)
        f = np.array([fh[0] * np.cos(tl), fh[1] * np.cos(tl), -np.sin(tl)])
        a = np.array([-fh[1], fh[0], 0.0])
        u = np.cross(a, f)
        return c, f, a, u

    def masks(self, n: int, h: float, centre_xy, rho_air: float):
        """Cell types (0 air, 1 shell, 2 rod, 3 fan, 4 trough air) and the static fields: alpha_static, fixed, tfix, src, src_rate, faces."""
        c, f, a, u = self.frame(centre_xy)
        g = (np.arange(n) + 0.5) * h
        P = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1) - c
        s, q, r = P @ a, P @ f, P @ u
        rho = np.sqrt(q ** 2 + r ** 2)
        cell = np.zeros((n, n, n), np.int32)
        if self.on:
            shell = (np.abs(s) <= self.length / 2) & (q <= 0.0) & (rho >= self.radius - 0.5 * h) & (rho <= self.radius + 0.006 + 0.5 * h)
            endcap = (np.abs(np.abs(s) - self.length / 2) <= 0.5 * h) & (q <= 0.0) & (rho <= self.radius + 0.006)
            cell[shell | endcap] = 1
            for rz in (-0.03, 0.03):
                rod = (np.abs(s) <= self.length / 2 - 0.015) & (np.sqrt((q + 0.03) ** 2 + (r - rz) ** 2) <= 0.007 + 0.5 * h)
                cell[rod] = 2
            if self.fan > 0:
                fan = (np.abs(s) <= self.length / 2 - 0.02) & (np.abs(r) <= self.radius - 0.01) & (q > 0.0) & (q <= 0.015)
                cell[fan & (cell == 0)] = 3
            trough = (np.abs(s) <= self.length / 2 - 0.015) & (rho <= self.radius - 0.004) & (q <= 0.0)
            cell[trough & (cell == 0)] = 4
        solid = (cell == 1) | (cell == 2)
        alpha_static = solid.astype(np.float32)
        fixed = (cell == 2).astype(np.int32); tfix = np.where(cell == 2, self.t_rod, 0.0).astype(np.float32)
        src = ((cell == 3) | (cell == 4)).astype(np.int32)
        src_rate = self.power / (rho_air * CP_AIR * max(int(src.sum()), 1) * h ** 3) if self.on else 0.0

        def face_fields(axis):
            shp = [n, n, n]; shp[axis] += 1
            K = np.zeros(shp, np.float32); M = np.zeros(shp, np.float32)
            pad = [(0, 0)] * 3; pad[axis] = (1, 1)
            sol = np.pad(solid, pad); fanc = np.pad(cell == 3, pad)
            sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3; sl0[axis] = slice(0, shp[axis]); sl1[axis] = slice(1, shp[axis] + 1)
            on_solid = sol[tuple(sl0)] | sol[tuple(sl1)]; on_fan = (fanc[tuple(sl0)] | fanc[tuple(sl1)]) & ~on_solid
            K[on_solid] = K_SOLID; K[on_fan] = K_FAN; M[on_fan] = K_FAN * self.fan * f[axis]
            return K, M
        faces = [face_fields(ax) for ax in range(3)]
        geom = dict(centre=c.tolist(), facing=f.tolist(), axis=a.tolist(), up=u.tolist(), **asdict(self))
        return cell, alpha_static, fixed, tfix, src, src_rate, faces, geom
