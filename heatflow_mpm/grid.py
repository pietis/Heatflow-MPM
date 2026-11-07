"""Staggered (MAC) grid for the air: Stable-Fluids step (Stam 1999, Fedkiw et al. 2001).

Per step: semi-Lagrangian RK2 advection of the face velocities, Boussinesq buoyancy on the z faces, implicit relaxation of the faces
toward obstacle velocities (material particles, heater body, fan), then a pressure projection that enforces the mixture divergence
div(alpha_f u + alpha_s v_s) = 0 (alpha_s = solid volume fraction from the particles) with a Jacobi-preconditioned conjugate gradient.
Walls are closed (zero normal flux). Gravity is not applied: the pressure is relative to hydrostatic and buoyancy carries the density
difference. Units: SI.
"""
from __future__ import annotations

import numpy as np
import warp as wp
import warp.optim.linear as wpl

ALPHA_F_MIN = 0.05   # floor on the fluid fraction inside obstacles (keeps the projection well posed)


# ---------------------------------------------------------------- trilinear helpers
@wp.func
def _base(pos: wp.vec3, off: wp.vec3, h: float, n: int):
    g = pos / h - off
    i = wp.vec3i(int(wp.floor(g[0])), int(wp.floor(g[1])), int(wp.floor(g[2])))
    f = wp.vec3(g[0] - float(i[0]), g[1] - float(i[1]), g[2] - float(i[2]))
    return i, f


@wp.func
def _clampi(i: int, lo: int, hi: int):
    return wp.min(wp.max(i, lo), hi)


@wp.func
def sample3(a: wp.array3d(dtype=float), pos: wp.vec3, off: wp.vec3, h: float, nx: int, ny: int, nz: int):
    i, f = _base(pos, off, h, nx)
    s = float(0.0)
    for dx in range(2):
        wx = 1.0 - f[0]
        if dx == 1:
            wx = f[0]
        ix = _clampi(i[0] + dx, 0, nx - 1)
        for dy in range(2):
            wy = 1.0 - f[1]
            if dy == 1:
                wy = f[1]
            iy = _clampi(i[1] + dy, 0, ny - 1)
            for dz in range(2):
                wz = 1.0 - f[2]
                if dz == 1:
                    wz = f[2]
                iz = _clampi(i[2] + dz, 0, nz - 1)
                s += wx * wy * wz * a[ix, iy, iz]
    return s


@wp.func
def scatter3(a: wp.array3d(dtype=float), pos: wp.vec3, off: wp.vec3, h: float, nx: int, ny: int, nz: int, val: float):
    i, f = _base(pos, off, h, nx)
    for dx in range(2):
        wx = 1.0 - f[0]
        if dx == 1:
            wx = f[0]
        ix = _clampi(i[0] + dx, 0, nx - 1)
        for dy in range(2):
            wy = 1.0 - f[1]
            if dy == 1:
                wy = f[1]
            iy = _clampi(i[1] + dy, 0, ny - 1)
            for dz in range(2):
                wz = 1.0 - f[2]
                if dz == 1:
                    wz = f[2]
                iz = _clampi(i[2] + dz, 0, nz - 1)
                wp.atomic_add(a, ix, iy, iz, wx * wy * wz * val)


@wp.func
def sample_vel(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), pos: wp.vec3, h: float, n: int):
    return wp.vec3(sample3(u, pos, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n),
                   sample3(v, pos, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n),
                   sample3(w, pos, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1))


# ---------------------------------------------------------------- advection (semi-Lagrangian, RK2)
@wp.kernel
def k_advect_x(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), out: wp.array3d(dtype=float), h: float, n: int, dt: float):
    i, j, k = wp.tid()
    if i == 0 or i == n:
        out[i, j, k] = 0.0
        return
    x = wp.vec3(float(i) * h, (float(j) + 0.5) * h, (float(k) + 0.5) * h)
    vel = sample_vel(u, v, w, x, h, n)
    vel = sample_vel(u, v, w, x - 0.5 * dt * vel, h, n)
    out[i, j, k] = sample3(u, x - dt * vel, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n)


@wp.kernel
def k_advect_y(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), out: wp.array3d(dtype=float), h: float, n: int, dt: float):
    i, j, k = wp.tid()
    if j == 0 or j == n:
        out[i, j, k] = 0.0
        return
    x = wp.vec3((float(i) + 0.5) * h, float(j) * h, (float(k) + 0.5) * h)
    vel = sample_vel(u, v, w, x, h, n)
    vel = sample_vel(u, v, w, x - 0.5 * dt * vel, h, n)
    out[i, j, k] = sample3(v, x - dt * vel, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n)


@wp.kernel
def k_advect_z(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), out: wp.array3d(dtype=float), h: float, n: int, dt: float):
    i, j, k = wp.tid()
    if k == 0 or k == n:
        out[i, j, k] = 0.0
        return
    x = wp.vec3((float(i) + 0.5) * h, (float(j) + 0.5) * h, float(k) * h)
    vel = sample_vel(u, v, w, x, h, n)
    vel = sample_vel(u, v, w, x - 0.5 * dt * vel, h, n)
    out[i, j, k] = sample3(w, x - dt * vel, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1)


@wp.kernel
def k_buoyancy(w: wp.array3d(dtype=float), T: wp.array3d(dtype=float), n: int, dt: float, g: float, t_amb: float):
    """Boussinesq: a_z = g (T - T_amb) / T_amb[K] on the z faces (T in Celsius)."""
    i, j, k = wp.tid()
    if k == 0 or k == n:
        return
    tf = 0.5 * (T[i, j, k - 1] + T[i, j, k])
    w[i, j, k] = w[i, j, k] + dt * g * (tf - t_amb) / (t_amb + 273.15)


# ---------------------------------------------------------------- obstacles: particles and static bodies on the faces
@wp.kernel
def k_p2g(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), vsol: float, h: float, n: int,
          as_c: wp.array3d(dtype=float), qx: wp.array3d(dtype=float), qy: wp.array3d(dtype=float), qz: wp.array3d(dtype=float)):
    """Solid volume fraction on cell centres and the solid volume flux alpha_s v_s on the faces."""
    t = wp.tid()
    x = pos[t]
    vol = vsol / (h * h * h)
    scatter3(as_c, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n, vol)
    scatter3(qx, x, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n, vol * vel[t][0])
    scatter3(qy, x, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n, vol * vel[t][1])
    scatter3(qz, x, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1, vol * vel[t][2])


@wp.kernel
def k_particle_block(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), h: float, n: int, k0: float,
                     kx: wp.array3d(dtype=float), ky: wp.array3d(dtype=float), kz: wp.array3d(dtype=float),
                     mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float), mz: wp.array3d(dtype=float)):
    """Material particles as an obstacle: a relaxation coefficient k0 per particle (and k0 v_p) scattered to the faces."""
    t = wp.tid()
    x = pos[t]
    vp = vel[t]
    scatter3(kx, x, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n, k0)
    scatter3(ky, x, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n, k0)
    scatter3(kz, x, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1, k0)
    scatter3(mx, x, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n, k0 * vp[0])
    scatter3(my, x, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n, k0 * vp[1])
    scatter3(mz, x, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1, k0 * vp[2])


@wp.kernel
def k_face_relax(u: wp.array3d(dtype=float), kf: wp.array3d(dtype=float), mf: wp.array3d(dtype=float), af: wp.array3d(dtype=float),
                 axis: int, h: float, n: int, dt: float, rho: float):
    """Implicit relaxation toward the obstacle velocity: u = (u* + dt c sum(K v)) / (1 + dt c sum(K)), c = 1 / (rho alpha_f h^3)."""
    i, j, k = wp.tid()
    idx = k
    if axis == 0:
        idx = i
    elif axis == 1:
        idx = j
    if idx == 0 or idx == n:
        return
    c = 1.0 / (rho * wp.max(af[i, j, k], ALPHA_F_MIN) * h * h * h)
    u[i, j, k] = (u[i, j, k] + dt * c * mf[i, j, k]) / (1.0 + dt * c * kf[i, j, k])


# ---------------------------------------------------------------- projection
@wp.kernel
def k_face_alpha(as_: wp.array3d(dtype=float), af: wp.array3d(dtype=float), axis: int, n: int):
    i, j, k = wp.tid()
    idx = k
    if axis == 0:
        idx = i
    elif axis == 1:
        idx = j
    if idx == 0 or idx == n:
        af[i, j, k] = 0.0
        return
    if axis == 0:
        a = 0.5 * (as_[i - 1, j, k] + as_[i, j, k])
    elif axis == 1:
        a = 0.5 * (as_[i, j - 1, k] + as_[i, j, k])
    else:
        a = 0.5 * (as_[i, j, k - 1] + as_[i, j, k])
    af[i, j, k] = wp.clamp(1.0 - a, ALPHA_F_MIN, 1.0)


@wp.kernel
def k_zero_wall_flux(q: wp.array3d(dtype=float), axis: int, n: int):
    i, j, k = wp.tid()
    idx = k
    if axis == 0:
        idx = i
    elif axis == 1:
        idx = j
    if idx == 0 or idx == n:
        q[i, j, k] = 0.0


@wp.kernel
def k_rhs(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
          ax: wp.array3d(dtype=float), ay: wp.array3d(dtype=float), az: wp.array3d(dtype=float),
          qx: wp.array3d(dtype=float), qy: wp.array3d(dtype=float), qz: wp.array3d(dtype=float),
          rhs: wp.array3d(dtype=float), h: float, dt: float, rho: float):
    """rhs = -(rho/dt) [div(alpha_f u*) + div(q_s)] per cell."""
    i, j, k = wp.tid()
    fx1 = ax[i + 1, j, k] * u[i + 1, j, k] + qx[i + 1, j, k]
    fx0 = ax[i, j, k] * u[i, j, k] + qx[i, j, k]
    fy1 = ay[i, j + 1, k] * v[i, j + 1, k] + qy[i, j + 1, k]
    fy0 = ay[i, j, k] * v[i, j, k] + qy[i, j, k]
    fz1 = az[i, j, k + 1] * w[i, j, k + 1] + qz[i, j, k + 1]
    fz0 = az[i, j, k] * w[i, j, k] + qz[i, j, k]
    rhs[i, j, k] = -(rho / dt) * (fx1 - fx0 + fy1 - fy0 + fz1 - fz0) / h


@wp.kernel
def k_matvec(p: wp.array(dtype=float), ax: wp.array3d(dtype=float), ay: wp.array3d(dtype=float), az: wp.array3d(dtype=float),
             out: wp.array(dtype=float), y: wp.array(dtype=float), alpha: float, beta: float, h: float, n: int):
    """out = alpha (-div(alpha_f grad p)) + beta y; 7-point stencil, zero coefficients on the walls (Neumann)."""
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
def k_diag(ax: wp.array3d(dtype=float), ay: wp.array3d(dtype=float), az: wp.array3d(dtype=float), d: wp.array(dtype=float), h: float, n: int):
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
def k_project(u: wp.array3d(dtype=float), p: wp.array3d(dtype=float), axis: int, h: float, n: int, dt: float, rho: float):
    i, j, k = wp.tid()
    idx = k
    if axis == 0:
        idx = i
    elif axis == 1:
        idx = j
    if idx == 0 or idx == n:
        u[i, j, k] = 0.0
        return
    if axis == 0:
        gp = (p[i, j, k] - p[i - 1, j, k]) / h
    elif axis == 1:
        gp = (p[i, j, k] - p[i, j - 1, k]) / h
    else:
        gp = (p[i, j, k] - p[i, j, k - 1]) / h
    u[i, j, k] = u[i, j, k] - dt / rho * gp


@wp.kernel
def k_cell_vel(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), out: wp.array(dtype=wp.vec3), n: int):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    out[t] = wp.vec3(0.5 * (u[i, j, k] + u[i + 1, j, k]), 0.5 * (v[i, j, k] + v[i, j + 1, k]), 0.5 * (w[i, j, k] + w[i, j, k + 1]))


@wp.kernel
def k_sub_scalar(a: wp.array(dtype=float), s: float):
    t = wp.tid()
    a[t] = a[t] - s


@wp.kernel
def k_add3d(a: wp.array3d(dtype=float), b: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    a[i, j, k] = a[i, j, k] + b[i, j, k]


# ---------------------------------------------------------------- the grid
class MacGrid:
    """Closed cubic box of n^3 cells (edge L, cell h) with face velocities u, v, w and cell pressure p."""

    def __init__(self, L: float, h: float, rho: float, device: str):
        self.L, self.h, self.n, self.rho, self.dev = L, h, int(round(L / h)), rho, device
        n = self.n
        Z = lambda *s: wp.zeros(s, dtype=float, device=device)
        self.u, self.v, self.w = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)
        self.u2, self.v2, self.w2 = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)
        self.ax, self.ay, self.az = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)   # fluid fraction on the faces
        self.qx, self.qy, self.qz = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)   # solid volume flux on the faces
        self.kx, self.ky, self.kz = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)   # obstacle relaxation coefficient [kg/s]
        self.mx, self.my, self.mz = Z(n + 1, n, n), Z(n, n + 1, n), Z(n, n, n + 1)   # K-weighted obstacle velocity
        self.p, self.rhs, self.alpha_s = Z(n, n, n), Z(n, n, n), Z(n, n, n)
        self.diag = wp.zeros(n ** 3, dtype=float, device=device)
        self._cell_vel = wp.zeros(n ** 3, dtype=wp.vec3, device=device)
        self.cg_iters = 0

    def clear_obstacle_fields(self):
        for a in (self.alpha_s, self.qx, self.qy, self.qz, self.kx, self.ky, self.kz, self.mx, self.my, self.mz):
            a.zero_()

    def add_static(self, alpha_static, k_static, m_static):
        """Add the static obstacle fields (heater body, fan) after the particles were scattered."""
        n, dev = self.n, self.dev
        wp.launch(k_add3d, dim=(n, n, n), inputs=[self.alpha_s, alpha_static], device=dev)
        for arr, st in zip((self.kx, self.ky, self.kz), k_static):
            wp.launch(k_add3d, dim=arr.shape, inputs=[arr, st], device=dev)
        for arr, st in zip((self.mx, self.my, self.mz), m_static):
            wp.launch(k_add3d, dim=arr.shape, inputs=[arr, st], device=dev)

    def _operator(self):
        n, h, dev = self.n, self.h, self.dev
        ax, ay, az = self.ax, self.ay, self.az

        def mv(x, y, z, alpha, beta):
            wp.launch(k_matvec, dim=n ** 3, inputs=[x, ax, ay, az, z, y, float(alpha), float(beta), h, n], device=dev)
        return wpl.LinearOperator(shape=(n ** 3, n ** 3), dtype=float, device=dev, matvec=mv)

    def step(self, dt: float, T, t_amb: float, g: float = 9.81, cg_tol: float = 1e-4, cg_max: int = 400):
        """One Stable-Fluids step: advect, buoyancy from T, relax toward obstacles, project."""
        n, h, dev, rho = self.n, self.h, self.dev, self.rho
        wp.launch(k_face_alpha, dim=(n + 1, n, n), inputs=[self.alpha_s, self.ax, 0, n], device=dev)
        wp.launch(k_face_alpha, dim=(n, n + 1, n), inputs=[self.alpha_s, self.ay, 1, n], device=dev)
        wp.launch(k_face_alpha, dim=(n, n, n + 1), inputs=[self.alpha_s, self.az, 2, n], device=dev)
        for q, axis, dim in ((self.qx, 0, (n + 1, n, n)), (self.qy, 1, (n, n + 1, n)), (self.qz, 2, (n, n, n + 1))):
            wp.launch(k_zero_wall_flux, dim=dim, inputs=[q, axis, n], device=dev)
        wp.launch(k_advect_x, dim=(n + 1, n, n), inputs=[self.u, self.v, self.w, self.u2, h, n, dt], device=dev)
        wp.launch(k_advect_y, dim=(n, n + 1, n), inputs=[self.u, self.v, self.w, self.v2, h, n, dt], device=dev)
        wp.launch(k_advect_z, dim=(n, n, n + 1), inputs=[self.u, self.v, self.w, self.w2, h, n, dt], device=dev)
        self.u, self.u2 = self.u2, self.u
        self.v, self.v2 = self.v2, self.v
        self.w, self.w2 = self.w2, self.w
        if g != 0.0:
            wp.launch(k_buoyancy, dim=(n, n, n + 1), inputs=[self.w, T, n, dt, g, t_amb], device=dev)
        wp.launch(k_face_relax, dim=(n + 1, n, n), inputs=[self.u, self.kx, self.mx, self.ax, 0, h, n, dt, rho], device=dev)
        wp.launch(k_face_relax, dim=(n, n + 1, n), inputs=[self.v, self.ky, self.my, self.ay, 1, h, n, dt, rho], device=dev)
        wp.launch(k_face_relax, dim=(n, n, n + 1), inputs=[self.w, self.kz, self.mz, self.az, 2, h, n, dt, rho], device=dev)
        wp.launch(k_rhs, dim=(n, n, n), inputs=[self.u, self.v, self.w, self.ax, self.ay, self.az, self.qx, self.qy, self.qz, self.rhs, h, dt, rho], device=dev)
        rhs = self.rhs.reshape((n ** 3,))
        mean = float(np.mean(rhs.numpy()))   # pure-Neumann system: remove the incompatible part
        if abs(mean) > 0.0:
            wp.launch(k_sub_scalar, dim=n ** 3, inputs=[rhs, mean], device=dev)
        wp.launch(k_diag, dim=n ** 3, inputs=[self.ax, self.ay, self.az, self.diag, h, n], device=dev)
        pf = self.p.reshape((n ** 3,))
        it, _, _ = wpl.cg(self._operator(), rhs, pf, tol=cg_tol, maxiter=cg_max, M=wpl.aslinearoperator(self.diag), check_every=20, use_cuda_graph=not wp.config.verify_cuda)
        self.cg_iters = int(it)
        wp.launch(k_project, dim=(n + 1, n, n), inputs=[self.u, self.p, 0, h, n, dt, rho], device=dev)
        wp.launch(k_project, dim=(n, n + 1, n), inputs=[self.v, self.p, 1, h, n, dt, rho], device=dev)
        wp.launch(k_project, dim=(n, n, n + 1), inputs=[self.w, self.p, 2, h, n, dt, rho], device=dev)

    def divergence_max(self) -> float:
        """max |div(alpha_f u + q_s)| h after the projection (about the CG tolerance)."""
        n, h, dev = self.n, self.h, self.dev
        wp.launch(k_rhs, dim=(n, n, n), inputs=[self.u, self.v, self.w, self.ax, self.ay, self.az, self.qx, self.qy, self.qz, self.rhs, h, 1.0, 1.0], device=dev)
        return float(np.abs(self.rhs.numpy()).max()) * h

    def cell_velocity(self) -> np.ndarray:
        wp.launch(k_cell_vel, dim=self.n ** 3, inputs=[self.u, self.v, self.w, self._cell_vel, self.n], device=self.dev)
        return self._cell_vel.numpy()

    def speed_max(self) -> float:
        return max(float(np.abs(self.u.numpy()).max()), float(np.abs(self.v.numpy()).max()), float(np.abs(self.w.numpy()).max()))
