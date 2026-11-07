"""Temperature field on the grid and its exchange with the material particles.

The temperature is a cell quantity: in air cells it is advected by the air (semi-Lagrangian, as a passive scalar) and diffused; in
material cells it is rebuilt every step from the particles (volume-weighted mean), so the material carries its heat with it. Diffusion is
backward Euler, (I - dt div(kappa grad)) T = T*, solved by a Jacobi-preconditioned conjugate gradient; the face conductivity is the harmonic
mean of the two cells (kappa blended from air to material by the material volume fraction). Fixed cells (Dirichlet: heater rods, room
walls) are decoupled symmetrically; the remaining walls are adiabatic. A volumetric source (heater power) heats the chosen cells, capped at
a maximum temperature. After the solve each particle receives the temperature increment at its position (FLIP transfer, Stomakhin et al.
2014) and its melt fraction, yield stress and viscosity are updated.
"""
from __future__ import annotations

import warp as wp
import warp.optim.linear as wpl

from .grid import sample3, scatter3, sample_vel


@wp.kernel
def k_p2g_heat(pos: wp.array(dtype=wp.vec3), temp: wp.array(dtype=float), vol: float, h: float, n: int,
               alpha: wp.array3d(dtype=float), tsum: wp.array3d(dtype=float)):
    t = wp.tid()
    x = pos[t]
    w = vol / (h * h * h)
    scatter3(alpha, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n, w)
    scatter3(tsum, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n, w * temp[t])


@wp.kernel
def k_blend_particles(T: wp.array3d(dtype=float), alpha: wp.array3d(dtype=float), tsum: wp.array3d(dtype=float), work: wp.array3d(dtype=float)):
    """Cells holding material take the particles' mean temperature; the other cells keep the grid value."""
    i, j, k = wp.tid()
    a = alpha[i, j, k]
    if a > 0.05:
        work[i, j, k] = tsum[i, j, k] / a
    else:
        work[i, j, k] = T[i, j, k]


@wp.kernel
def k_advect_T(T: wp.array3d(dtype=float), alpha: wp.array3d(dtype=float), u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
               out: wp.array3d(dtype=float), h: float, n: int, dt: float):
    i, j, k = wp.tid()
    if alpha[i, j, k] > 0.5:
        out[i, j, k] = T[i, j, k]
        return
    x = wp.vec3((float(i) + 0.5) * h, (float(j) + 0.5) * h, (float(k) + 0.5) * h)
    vel = sample_vel(u, v, w, x, h, n)
    vel = sample_vel(u, v, w, x - 0.5 * dt * vel, h, n)
    out[i, j, k] = sample3(T, x - dt * vel, wp.vec3(0.5, 0.5, 0.5), h, n, n, n)


@wp.kernel
def k_kappa(alpha: wp.array3d(dtype=float), kap: wp.array3d(dtype=float), kappa_air: float, kappa_mat: float):
    i, j, k = wp.tid()
    kap[i, j, k] = kappa_air + (kappa_mat - kappa_air) * wp.clamp(alpha[i, j, k] / 0.6, 0.0, 1.0)


@wp.func
def face_kappa(ka: float, kb: float):
    return 2.0 * ka * kb / (ka + kb + 1.0e-12)


@wp.kernel
def k_heat_matvec(x: wp.array(dtype=float), kap: wp.array3d(dtype=float), fixed: wp.array3d(dtype=int), out: wp.array(dtype=float), y: wp.array(dtype=float),
                  alpha: float, beta: float, h: float, n: int, dt: float):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    xc = x[t]
    if fixed[i, j, k] == 1:
        out[t] = alpha * xc + beta * y[t]
        return
    kc = kap[i, j, k]
    acc = float(0.0)
    if i > 0:
        kf = face_kappa(kc, kap[i - 1, j, k])
        acc += kf * xc
        if fixed[i - 1, j, k] == 0:
            acc -= kf * x[t - n * n]
    if i < n - 1:
        kf = face_kappa(kc, kap[i + 1, j, k])
        acc += kf * xc
        if fixed[i + 1, j, k] == 0:
            acc -= kf * x[t + n * n]
    if j > 0:
        kf = face_kappa(kc, kap[i, j - 1, k])
        acc += kf * xc
        if fixed[i, j - 1, k] == 0:
            acc -= kf * x[t - n]
    if j < n - 1:
        kf = face_kappa(kc, kap[i, j + 1, k])
        acc += kf * xc
        if fixed[i, j + 1, k] == 0:
            acc -= kf * x[t + n]
    if k > 0:
        kf = face_kappa(kc, kap[i, j, k - 1])
        acc += kf * xc
        if fixed[i, j, k - 1] == 0:
            acc -= kf * x[t - 1]
    if k < n - 1:
        kf = face_kappa(kc, kap[i, j, k + 1])
        acc += kf * xc
        if fixed[i, j, k + 1] == 0:
            acc -= kf * x[t + 1]
    out[t] = alpha * (xc + dt / (h * h) * acc) + beta * y[t]


@wp.kernel
def k_heat_rhs(T: wp.array3d(dtype=float), kap: wp.array3d(dtype=float), fixed: wp.array3d(dtype=int), tfix: wp.array3d(dtype=float),
               rhs: wp.array(dtype=float), h: float, n: int, dt: float):
    t = wp.tid()
    k = t % n
    j = (t / n) % n
    i = t / (n * n)
    if fixed[i, j, k] == 1:
        rhs[t] = tfix[i, j, k]
        return
    kc = kap[i, j, k]
    acc = float(0.0)
    if i > 0 and fixed[i - 1, j, k] == 1:
        acc += face_kappa(kc, kap[i - 1, j, k]) * tfix[i - 1, j, k]
    if i < n - 1 and fixed[i + 1, j, k] == 1:
        acc += face_kappa(kc, kap[i + 1, j, k]) * tfix[i + 1, j, k]
    if j > 0 and fixed[i, j - 1, k] == 1:
        acc += face_kappa(kc, kap[i, j - 1, k]) * tfix[i, j - 1, k]
    if j < n - 1 and fixed[i, j + 1, k] == 1:
        acc += face_kappa(kc, kap[i, j + 1, k]) * tfix[i, j + 1, k]
    if k > 0 and fixed[i, j, k - 1] == 1:
        acc += face_kappa(kc, kap[i, j, k - 1]) * tfix[i, j, k - 1]
    if k < n - 1 and fixed[i, j, k + 1] == 1:
        acc += face_kappa(kc, kap[i, j, k + 1]) * tfix[i, j, k + 1]
    rhs[t] = T[i, j, k] + dt / (h * h) * acc


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
def k_heat_source(T: wp.array3d(dtype=float), src: wp.array3d(dtype=int), rate: float, t_cap: float, dt: float):
    """Volumetric heat source (K/s) in the source cells, never pushing a cell above t_cap."""
    i, j, k = wp.tid()
    if src[i, j, k] == 1:
        T[i, j, k] = wp.min(T[i, j, k] + rate * dt, wp.max(T[i, j, k], t_cap))


@wp.kernel
def k_g2p_material(pos: wp.array(dtype=wp.vec3), T_before: wp.array3d(dtype=float), T_after: wp.array3d(dtype=float), h: float, n: int,
                   t_sol: float, t_liq: float, y_solid: float, mu_solid: float, mu_liquid: float, t_max: float,
                   temp: wp.array(dtype=float), melt: wp.array(dtype=float), yield_stress: wp.array(dtype=float), viscosity: wp.array(dtype=float)):
    """Particle temperature += grid increment at its position; melt fraction f = clamp((T - t_sol) / (t_liq - t_sol));
    yield stress y_solid (1 - f), viscosity mu_solid (1 - f) + mu_liquid f."""
    t = wp.tid()
    x = pos[t]
    dT = sample3(T_after, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n) - sample3(T_before, x, wp.vec3(0.5, 0.5, 0.5), h, n, n, n)
    tp = wp.clamp(temp[t] + dT, 0.0, t_max)
    temp[t] = tp
    f = wp.clamp((tp - t_sol) / (t_liq - t_sol), 0.0, 1.0)
    melt[t] = f
    yield_stress[t] = y_solid * (1.0 - f)
    viscosity[t] = mu_solid * (1.0 - f) + mu_liquid * f


class HeatField:
    """Cell temperature with Dirichlet mask, source mask and the implicit diffusion solve."""

    def __init__(self, n: int, h: float, t_init: float, device: str):
        self.n, self.h, self.dev = n, h, device
        Z = lambda: wp.zeros((n, n, n), dtype=float, device=device)
        self.T, self.T2, self.Tw, self.kap, self.alpha_m, self.tsum = Z(), Z(), Z(), Z(), Z(), Z()
        for a in (self.T, self.T2, self.Tw):
            a.fill_(t_init)
        self.fixed = wp.zeros((n, n, n), dtype=int, device=device); self.tfix = Z()
        self.src = wp.zeros((n, n, n), dtype=int, device=device); self.src_rate = 0.0; self.t_cap = 0.0
        self.rhs = wp.zeros(n ** 3, dtype=float, device=device); self.diag = wp.zeros(n ** 3, dtype=float, device=device)
        self.cg_iters = 0

    def set_fixed(self, fixed, tfix):
        self.fixed = wp.array(fixed.astype("int32"), dtype=int, device=self.dev); self.tfix = wp.array(tfix.astype("float32"), dtype=float, device=self.dev)
        wp.launch(k_apply_fixed, dim=(self.n,) * 3, inputs=[self.T, self.fixed, self.tfix], device=self.dev)

    def set_source(self, src, rate: float, t_cap: float):
        self.src = wp.array(src.astype("int32"), dtype=int, device=self.dev); self.src_rate = float(rate); self.t_cap = float(t_cap)

    def _operator(self, dt):
        n, h, dev, kap, fixed = self.n, self.h, self.dev, self.kap, self.fixed

        def mv(x, y, z, alpha, beta):
            wp.launch(k_heat_matvec, dim=n ** 3, inputs=[x, kap, fixed, z, y, float(alpha), float(beta), h, n, dt], device=dev)
        return wpl.LinearOperator(shape=(n ** 3, n ** 3), dtype=float, device=dev, matvec=mv)

    def diffuse_implicit(self, dt: float, src_field, dst_field, tol: float = 1e-6, maxiter: int = 200):
        """dst = (I - dt div(kappa grad))^-1 src with the current kappa, fixed and tfix fields (dst may alias src's data after the call)."""
        n, h, dev = self.n, self.h, self.dev
        wp.launch(k_heat_rhs, dim=n ** 3, inputs=[src_field, self.kap, self.fixed, self.tfix, self.rhs, h, n, dt], device=dev)
        wp.launch(k_heat_diag, dim=n ** 3, inputs=[self.kap, self.fixed, self.diag, h, n, dt], device=dev)
        x = dst_field.reshape((n ** 3,))
        wp.copy(x, src_field.reshape((n ** 3,)))
        it, _, _ = wpl.cg(self._operator(dt), self.rhs, x, tol=tol, maxiter=maxiter, M=wpl.aslinearoperator(self.diag), check_every=10, use_cuda_graph=not wp.config.verify_cuda)
        self.cg_iters = int(it)
