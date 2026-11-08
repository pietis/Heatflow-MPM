"""Air on a closed MAC grid: Stable-Fluids step with Boussinesq buoyancy, implicit relaxation toward obstacles and a pressure
projection with the solid fraction of the material. SI units; z is up."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import warp as wp
import warp.optim.linear as wpl

from .fields import sample, sample_velocity, face_position, face_offset, face_dims, face_axis_index, backtrace


@dataclass
class GridConfig:
    L: float = 0.48            # box edge [m]
    h: float = 0.005           # cell size [m]
    rho: float = 1.2           # air density [kg/m^3]
    t_amb: float = 25.0        # room temperature [C]; side and top walls are held here, the floor is adiabatic
    buoyancy: bool = True
    alpha_f_min: float = 0.05  # floor of the air fraction inside dense solids
    cg_tol: float = 1e-4
    cg_max_iter: int = 400


@wp.kernel
def k_advect_face(q: wp.array3d(dtype=float), u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                  out: wp.array3d(dtype=float), axis: int, h: float, n: int, dt: float):
    i, j, k = wp.tid()
    if face_axis_index(axis, i, j, k) == 0 or face_axis_index(axis, i, j, k) == n:
        out[i, j, k] = 0.0
        return
    d = face_dims(axis, n)
    out[i, j, k] = sample(q, backtrace(u, v, w, face_position(axis, i, j, k, h), h, n, dt), face_offset(axis), h, d[0], d[1], d[2])


@wp.kernel
def k_buoyancy(w: wp.array3d(dtype=float), T: wp.array3d(dtype=float), n: int, dt: float, g: float, t_amb: float):
    """a_z = g (T - T_amb) / T_amb[K] on the z faces."""
    i, j, k = wp.tid()
    if k == 0 or k == n:
        return
    tf = 0.5 * (T[i, j, k - 1] + T[i, j, k])
    w[i, j, k] = w[i, j, k] + dt * g * (tf - t_amb) / (t_amb + 273.15)


@wp.kernel
def k_face_fraction(alpha_s: wp.array3d(dtype=float), af: wp.array3d(dtype=float), axis: int, n: int, alpha_f_min: float):
    i, j, k = wp.tid()
    idx = face_axis_index(axis, i, j, k)
    if idx == 0 or idx == n:
        af[i, j, k] = 0.0
        return
    if axis == 0:
        a = 0.5 * (alpha_s[i - 1, j, k] + alpha_s[i, j, k])
    elif axis == 1:
        a = 0.5 * (alpha_s[i, j - 1, k] + alpha_s[i, j, k])
    else:
        a = 0.5 * (alpha_s[i, j, k - 1] + alpha_s[i, j, k])
    af[i, j, k] = wp.clamp(1.0 - a, alpha_f_min, 1.0)


@wp.kernel
def k_zero_wall(q: wp.array3d(dtype=float), axis: int, n: int):
    i, j, k = wp.tid()
    idx = face_axis_index(axis, i, j, k)
    if idx == 0 or idx == n:
        q[i, j, k] = 0.0


@wp.kernel
def k_relax(q: wp.array3d(dtype=float), kf: wp.array3d(dtype=float), mf: wp.array3d(dtype=float), af: wp.array3d(dtype=float),
            axis: int, h: float, n: int, dt: float, rho: float, alpha_f_min: float):
    """q = (q + dt c sum K v) / (1 + dt c sum K), c = 1 / (rho alpha_f h^3): the air follows the obstacle velocity."""
    i, j, k = wp.tid()
    idx = face_axis_index(axis, i, j, k)
    if idx == 0 or idx == n:
        return
    c = 1.0 / (rho * wp.max(af[i, j, k], alpha_f_min) * h * h * h)
    q[i, j, k] = (q[i, j, k] + dt * c * mf[i, j, k]) / (1.0 + dt * c * kf[i, j, k])


@wp.kernel
def k_divergence_rhs(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                     ax: wp.array3d(dtype=float), ay: wp.array3d(dtype=float), az: wp.array3d(dtype=float),
                     qx: wp.array3d(dtype=float), qy: wp.array3d(dtype=float), qz: wp.array3d(dtype=float),
                     rhs: wp.array3d(dtype=float), h: float, scale: float):
    i, j, k = wp.tid()
    div = (ax[i + 1, j, k] * u[i + 1, j, k] + qx[i + 1, j, k] - ax[i, j, k] * u[i, j, k] - qx[i, j, k]
           + ay[i, j + 1, k] * v[i, j + 1, k] + qy[i, j + 1, k] - ay[i, j, k] * v[i, j, k] - qy[i, j, k]
           + az[i, j, k + 1] * w[i, j, k + 1] + qz[i, j, k + 1] - az[i, j, k] * w[i, j, k] - qz[i, j, k]) / h
    rhs[i, j, k] = -scale * div


@wp.kernel
def k_laplacian(p: wp.array(dtype=float), ax: wp.array3d(dtype=float), ay: wp.array3d(dtype=float), az: wp.array3d(dtype=float),
                out: wp.array(dtype=float), y: wp.array(dtype=float), alpha: float, beta: float, h: float, n: int):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    pc = p[t]
    acc = float(0.0)
    if i > 0:
        acc += ax[i, j, k] * (pc - p[t - n * n])
    if i < n - 1:
        acc += ax[i + 1, j, k] * (pc - p[t + n * n])
    if j > 0:
        acc += ay[i, j, k] * (pc - p[t - n])
    if j < n - 1:
        acc += ay[i, j + 1, k] * (pc - p[t + n])
    if k > 0:
        acc += az[i, j, k] * (pc - p[t - 1])
    if k < n - 1:
        acc += az[i, j, k + 1] * (pc - p[t + 1])
    out[t] = alpha * acc / (h * h) + beta * y[t]


@wp.kernel
def k_jacobi(ax: wp.array3d(dtype=float), ay: wp.array3d(dtype=float), az: wp.array3d(dtype=float), d: wp.array(dtype=float), h: float, n: int):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    s = float(0.0)
    if i > 0:
        s += ax[i, j, k]
    if i < n - 1:
        s += ax[i + 1, j, k]
    if j > 0:
        s += ay[i, j, k]
    if j < n - 1:
        s += ay[i, j + 1, k]
    if k > 0:
        s += az[i, j, k]
    if k < n - 1:
        s += az[i, j, k + 1]
    d[t] = (h * h) / wp.max(s, 1.0e-12)


@wp.kernel
def k_project(q: wp.array3d(dtype=float), p: wp.array3d(dtype=float), axis: int, h: float, n: int, dt_over_rho: float):
    i, j, k = wp.tid()
    idx = face_axis_index(axis, i, j, k)
    if idx == 0 or idx == n:
        q[i, j, k] = 0.0
        return
    if axis == 0:
        gp = p[i, j, k] - p[i - 1, j, k]
    elif axis == 1:
        gp = p[i, j, k] - p[i, j - 1, k]
    else:
        gp = p[i, j, k] - p[i, j, k - 1]
    q[i, j, k] = q[i, j, k] - dt_over_rho * gp / h


@wp.kernel
def k_cell_velocity(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), out: wp.array(dtype=wp.vec3), n: int):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    out[t] = wp.vec3(0.5 * (u[i, j, k] + u[i + 1, j, k]), 0.5 * (v[i, j, k] + v[i, j + 1, k]), 0.5 * (w[i, j, k] + w[i, j, k + 1]))


@wp.kernel
def k_add(a: wp.array3d(dtype=float), b: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    a[i, j, k] = a[i, j, k] + b[i, j, k]


@wp.kernel
def k_sub_scalar(a: wp.array(dtype=float), s: float):
    t = wp.tid()
    a[t] = a[t] - s


class AirGrid:
    """Face velocities u, v, w and cell pressure p. Obstacle fields (solid fraction, solid flux, relaxation weights) are cleared
    with `clear_obstacle_fields`, filled by the material and `add_static`, and consumed by `step`."""

    def __init__(self, cfg: GridConfig, device: str):
        self.cfg, self.dev = cfg, device
        self.n = n = int(round(cfg.L / cfg.h))
        self.h, self.L = cfg.h, cfg.L
        Z = lambda *s: wp.zeros(s, dtype=float, device=device)
        self.u, self.v, self.w = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)
        self.scratch = [Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)]
        self.ax, self.ay, self.az = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)
        self.qx, self.qy, self.qz = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)
        self.kx, self.ky, self.kz = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)
        self.mx, self.my, self.mz = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)
        self.p, self.rhs, self.alpha_s = Z(n, n, n), Z(n, n, n), Z(n, n, n)
        self.diag = wp.zeros(n ** 3, dtype=float, device=device)
        self._cell_vel = wp.zeros(n ** 3, dtype=wp.vec3, device=device)
        self.static = None     # (alpha_static, [kx, ky, kz], [mx, my, mz])
        self.cg_iters = 0

    def set_static(self, alpha: np.ndarray, k_faces: list, m_faces: list):
        W = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float32), dtype=float, device=self.dev)
        self.static = (W(alpha), [W(a) for a in k_faces], [W(a) for a in m_faces])

    def clear_obstacle_fields(self):
        for a in (self.alpha_s, self.qx, self.qy, self.qz, self.kx, self.ky, self.kz, self.mx, self.my, self.mz):
            a.zero_()

    def add_static(self):
        if self.static is None:
            return
        alpha, ks, ms = self.static
        n = self.n
        wp.launch(k_add, dim=(n, n, n), inputs=[self.alpha_s, alpha], device=self.dev)
        for arr, st in zip((self.kx, self.ky, self.kz, self.mx, self.my, self.mz), ks + ms):
            wp.launch(k_add, dim=arr.shape, inputs=[arr, st], device=self.dev)

    def _faces(self):
        n = self.n
        return ((0, self.u, (n + 1, n, n)), (1, self.v, (n, n + 1, n)), (2, self.w, (n, n, n + 1)))

    def step(self, dt: float, T: wp.array, g: float = 9.81):
        n, h, dev, cfg = self.n, self.h, self.dev, self.cfg
        for axis, _, dim in self._faces():
            wp.launch(k_face_fraction, dim=dim, inputs=[self.alpha_s, (self.ax, self.ay, self.az)[axis], axis, n, cfg.alpha_f_min], device=dev)
            wp.launch(k_zero_wall, dim=dim, inputs=[(self.qx, self.qy, self.qz)[axis], axis, n], device=dev)
        u0, v0, w0 = self.u, self.v, self.w
        for axis, q, dim in self._faces():
            wp.launch(k_advect_face, dim=dim, inputs=[q, u0, v0, w0, self.scratch[axis], axis, h, n, dt], device=dev)
        self.u, self.v, self.w, self.scratch = self.scratch[0], self.scratch[1], self.scratch[2], [u0, v0, w0]
        if cfg.buoyancy:
            wp.launch(k_buoyancy, dim=(n, n, n + 1), inputs=[self.w, T, n, dt, g, cfg.t_amb], device=dev)
        for axis, q, dim in self._faces():
            wp.launch(k_relax, dim=dim, inputs=[q, (self.kx, self.ky, self.kz)[axis], (self.mx, self.my, self.mz)[axis], (self.ax, self.ay, self.az)[axis],
                                               axis, h, n, dt, cfg.rho, cfg.alpha_f_min], device=dev)
        self._project(dt)

    def _project(self, dt: float):
        n, h, dev, cfg = self.n, self.h, self.dev, self.cfg
        wp.launch(k_divergence_rhs, dim=(n, n, n), inputs=[self.u, self.v, self.w, self.ax, self.ay, self.az, self.qx, self.qy, self.qz, self.rhs, h, cfg.rho / dt], device=dev)
        rhs = self.rhs.reshape((n ** 3,))
        mean = float(rhs.numpy().mean())
        if mean != 0.0:
            wp.launch(k_sub_scalar, dim=n ** 3, inputs=[rhs, mean], device=dev)
        wp.launch(k_jacobi, dim=n ** 3, inputs=[self.ax, self.ay, self.az, self.diag, h, n], device=dev)
        ax, ay, az = self.ax, self.ay, self.az

        def matvec(x, y, z, alpha, beta):
            wp.launch(k_laplacian, dim=n ** 3, inputs=[x, ax, ay, az, z, y, float(alpha), float(beta), h, n], device=dev)
        A = wpl.LinearOperator(shape=(n ** 3, n ** 3), dtype=float, device=dev, matvec=matvec)
        it, _, _ = wpl.cg(A, rhs, self.p.reshape((n ** 3,)), tol=cfg.cg_tol, maxiter=cfg.cg_max_iter, M=wpl.aslinearoperator(self.diag), check_every=20)
        self.cg_iters = int(it)
        for axis, q, dim in self._faces():
            wp.launch(k_project, dim=dim, inputs=[q, self.p, axis, h, n, dt / cfg.rho], device=dev)

    def cell_velocity(self) -> np.ndarray:
        wp.launch(k_cell_velocity, dim=self.n ** 3, inputs=[self.u, self.v, self.w, self._cell_vel, self.n], device=self.dev)
        return self._cell_vel.numpy()

    def divergence_max(self) -> float:
        n = self.n
        wp.launch(k_divergence_rhs, dim=(n, n, n), inputs=[self.u, self.v, self.w, self.ax, self.ay, self.az, self.qx, self.qy, self.qz, self.rhs, self.h, 1.0], device=self.dev)
        return float(np.abs(self.rhs.numpy()).max()) * self.h

    def speed_max(self) -> float:
        return max(float(np.abs(a.numpy()).max()) for a in (self.u, self.v, self.w))
