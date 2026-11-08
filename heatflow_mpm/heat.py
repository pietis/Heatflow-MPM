"""Temperature on the grid: advected by the air in air cells, rebuilt from the particles in material cells, diffused implicitly
(backward Euler, CG) with fixed cells and a capped volumetric source. Particles receive the increment at their position (FLIP)."""
from __future__ import annotations

import numpy as np
import warp as wp
import warp.optim.linear as wpl

from .fields import sample, scatter, sample_velocity


@wp.kernel
def k_scatter_heat(pos: wp.array(dtype=wp.vec3), temp: wp.array(dtype=float), vol: float, h: float, n: int,
                   alpha: wp.array3d(dtype=float), tsum: wp.array3d(dtype=float)):
    t = wp.tid()
    frac = vol / (h * h * h)
    scatter(alpha, pos[t], wp.vec3(0.5, 0.5, 0.5), h, n, n, n, frac)
    scatter(tsum, pos[t], wp.vec3(0.5, 0.5, 0.5), h, n, n, n, frac * temp[t])


@wp.kernel
def k_blend(T: wp.array3d(dtype=float), alpha: wp.array3d(dtype=float), tsum: wp.array3d(dtype=float), out: wp.array3d(dtype=float)):
    """Material cells take the particles' mean temperature, the other cells keep the grid value."""
    i, j, k = wp.tid()
    if alpha[i, j, k] > 0.05:
        out[i, j, k] = tsum[i, j, k] / alpha[i, j, k]
    else:
        out[i, j, k] = T[i, j, k]


@wp.kernel
def k_advect(T: wp.array3d(dtype=float), alpha: wp.array3d(dtype=float), u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
             out: wp.array3d(dtype=float), h: float, n: int, dt: float):
    i, j, k = wp.tid()
    if alpha[i, j, k] > 0.5:
        out[i, j, k] = T[i, j, k]
        return
    x = wp.vec3((float(i) + 0.5) * h, (float(j) + 0.5) * h, (float(k) + 0.5) * h)
    vel = sample_velocity(u, v, w, x, h, n)
    vel = sample_velocity(u, v, w, x - 0.5 * dt * vel, h, n)
    out[i, j, k] = sample(T, x - dt * vel, wp.vec3(0.5, 0.5, 0.5), h, n, n, n)


@wp.kernel
def k_conductivity(alpha: wp.array3d(dtype=float), kap: wp.array3d(dtype=float), kappa_air: float, kappa_mat: float):
    i, j, k = wp.tid()
    kap[i, j, k] = kappa_air + (kappa_mat - kappa_air) * wp.clamp(alpha[i, j, k] / 0.6, 0.0, 1.0)


@wp.func
def face_kappa(ka: float, kb: float):
    return 2.0 * ka * kb / (ka + kb + 1.0e-12)


@wp.func
def neighbour_terms(x: wp.array(dtype=float), kap: wp.array3d(dtype=float), fixed: wp.array3d(dtype=int), tfix: wp.array3d(dtype=float),
                    t: int, i: int, j: int, k: int, n: int, use_fixed: int):
    """Sum over the six faces of kappa_f (x_c - x_nb) for free neighbours; with use_fixed = 1 returns kappa_f T_fixed over fixed neighbours instead."""
    kc = kap[i, j, k]
    acc = float(0.0)
    xc = x[t]
    for s in range(6):
        di = 0
        dj = 0
        dk = 0
        if s == 0:
            di = -1
        elif s == 1:
            di = 1
        elif s == 2:
            dj = -1
        elif s == 3:
            dj = 1
        elif s == 4:
            dk = -1
        else:
            dk = 1
        ii = i + di
        jj = j + dj
        kk = k + dk
        if ii < 0 or jj < 0 or kk < 0 or ii >= n or jj >= n or kk >= n:
            continue
        kf = face_kappa(kc, kap[ii, jj, kk])
        if fixed[ii, jj, kk] == 1:
            if use_fixed == 1:
                acc += kf * tfix[ii, jj, kk]
            else:
                acc += kf * xc
        elif use_fixed == 0:
            acc += kf * (xc - x[t + di * n * n + dj * n + dk])
    return acc


@wp.kernel
def k_heat_matvec(x: wp.array(dtype=float), kap: wp.array3d(dtype=float), fixed: wp.array3d(dtype=int), tfix: wp.array3d(dtype=float),
                  out: wp.array(dtype=float), y: wp.array(dtype=float), alpha: float, beta: float, h: float, n: int, dt: float):
    """out = alpha A x + beta y, A = I - dt div(kappa grad) on free cells; fixed cells are identity rows (symmetric system)."""
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    if fixed[i, j, k] == 1:
        out[t] = alpha * x[t] + beta * y[t]
        return
    out[t] = alpha * (x[t] + dt / (h * h) * neighbour_terms(x, kap, fixed, tfix, t, i, j, k, n, 0)) + beta * y[t]


@wp.kernel
def k_heat_rhs(T: wp.array3d(dtype=float), kap: wp.array3d(dtype=float), fixed: wp.array3d(dtype=int), tfix: wp.array3d(dtype=float),
               x: wp.array(dtype=float), rhs: wp.array(dtype=float), h: float, n: int, dt: float):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    if fixed[i, j, k] == 1:
        rhs[t] = tfix[i, j, k]
        return
    rhs[t] = T[i, j, k] + dt / (h * h) * neighbour_terms(x, kap, fixed, tfix, t, i, j, k, n, 1)


@wp.kernel
def k_heat_diag(kap: wp.array3d(dtype=float), fixed: wp.array3d(dtype=int), d: wp.array(dtype=float), h: float, n: int, dt: float):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    if fixed[i, j, k] == 1:
        d[t] = 1.0
        return
    kc = kap[i, j, k]
    s = float(0.0)
    if i > 0:
        s += face_kappa(kc, kap[i - 1, j, k])
    if i < n - 1:
        s += face_kappa(kc, kap[i + 1, j, k])
    if j > 0:
        s += face_kappa(kc, kap[i, j - 1, k])
    if j < n - 1:
        s += face_kappa(kc, kap[i, j + 1, k])
    if k > 0:
        s += face_kappa(kc, kap[i, j, k - 1])
    if k < n - 1:
        s += face_kappa(kc, kap[i, j, k + 1])
    d[t] = 1.0 / (1.0 + dt / (h * h) * s)


@wp.kernel
def k_apply_fixed(T: wp.array3d(dtype=float), fixed: wp.array3d(dtype=int), tfix: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    if fixed[i, j, k] == 1:
        T[i, j, k] = tfix[i, j, k]


@wp.kernel
def k_source(T: wp.array3d(dtype=float), src: wp.array3d(dtype=int), rate: float, t_cap: float, dt: float):
    i, j, k = wp.tid()
    if src[i, j, k] == 1:
        T[i, j, k] = wp.min(T[i, j, k] + rate * dt, wp.max(T[i, j, k], t_cap))


@wp.kernel
def k_particle_update(pos: wp.array(dtype=wp.vec3), T_before: wp.array3d(dtype=float), T_after: wp.array3d(dtype=float), h: float, n: int,
                      t_sol: float, t_liq: float, y_solid: float, mu_solid: float, mu_liquid: float, t_max: float,
                      temp: wp.array(dtype=float), melt: wp.array(dtype=float), yield_stress: wp.array(dtype=float), viscosity: wp.array(dtype=float)):
    """T_p += grid increment at x_p; f = clamp((T - t_sol)/(t_liq - t_sol)); yield y (1 - f), viscosity mu_s (1 - f) + mu_l f."""
    t = wp.tid()
    x = pos[t]
    dT = sample(T_after, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n) - sample(T_before, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n)
    tp = wp.clamp(temp[t] + dT, 0.0, t_max)
    temp[t] = tp
    f = wp.clamp((tp - t_sol) / (t_liq - t_sol), 0.0, 1.0)
    melt[t] = f
    yield_stress[t] = y_solid * (1.0 - f)
    viscosity[t] = mu_solid * (1.0 - f) + mu_liquid * f


class HeatField:
    def __init__(self, n: int, h: float, t_init: float, device: str):
        self.n, self.h, self.dev = n, h, device
        Z = lambda: wp.zeros((n, n, n), dtype=float, device=device)
        self.T, self.T2, self.Tw, self.kap, self.alpha, self.tsum = Z(), Z(), Z(), Z(), Z(), Z()
        for a in (self.T, self.T2, self.Tw):
            a.fill_(t_init)
        self.fixed, self.tfix = wp.zeros((n, n, n), dtype=int, device=device), Z()
        self.src, self.rate, self.t_cap = wp.zeros((n, n, n), dtype=int, device=device), 0.0, 0.0
        self.rhs, self.diag = wp.zeros(n ** 3, dtype=float, device=device), wp.zeros(n ** 3, dtype=float, device=device)
        self.cg_iters = 0

    def set_fixed(self, fixed: np.ndarray, values: np.ndarray):
        self.fixed = wp.array(np.ascontiguousarray(fixed, dtype=np.int32), dtype=int, device=self.dev)
        self.tfix = wp.array(np.ascontiguousarray(values, dtype=np.float32), dtype=float, device=self.dev)
        wp.launch(k_apply_fixed, dim=(self.n,) * 3, inputs=[self.T, self.fixed, self.tfix], device=self.dev)

    def set_source(self, mask: np.ndarray, rate: float, t_cap: float):
        self.src = wp.array(np.ascontiguousarray(mask, dtype=np.int32), dtype=int, device=self.dev)
        self.rate, self.t_cap = float(rate), float(t_cap)

    def diffuse(self, dt: float, tol: float = 1e-6, maxiter: int = 200):
        """T2 <- (I - dt div kappa grad)^-1 T2 with the current kap, fixed and source fields."""
        n, h, dev = self.n, self.h, self.dev
        x = self.T2.reshape((n ** 3,))
        wp.launch(k_heat_rhs, dim=n ** 3, inputs=[self.T2, self.kap, self.fixed, self.tfix, x, self.rhs, h, n, dt], device=dev)
        wp.launch(k_heat_diag, dim=n ** 3, inputs=[self.kap, self.fixed, self.diag, h, n, dt], device=dev)
        kap, fixed, tfix = self.kap, self.fixed, self.tfix

        def matvec(a, y, z, alpha, beta):
            wp.launch(k_heat_matvec, dim=n ** 3, inputs=[a, kap, fixed, tfix, z, y, float(alpha), float(beta), h, n, dt], device=dev)
        A = wpl.LinearOperator(shape=(n ** 3, n ** 3), dtype=float, device=dev, matvec=matvec)
        it, _, _ = wpl.cg(A, self.rhs, x, tol=tol, maxiter=maxiter, M=wpl.aslinearoperator(self.diag), check_every=10)
        self.cg_iters = int(it)
