"""Trilinear sampling and scattering on the staggered (MAC) grid. Warp device functions shared by the kernels."""
import warp as wp


@wp.func
def base_index(pos: wp.vec3, off: wp.vec3, h: float):
    """Lower lattice index and fractional offsets of `pos` on a lattice whose nodes sit at (i + off) h."""
    g = pos / h - off
    i = wp.vec3i(int(wp.floor(g[0])), int(wp.floor(g[1])), int(wp.floor(g[2])))
    return i, wp.vec3(g[0] - float(i[0]), g[1] - float(i[1]), g[2] - float(i[2]))


@wp.func
def clampi(i: int, lo: int, hi: int):
    return wp.min(wp.max(i, lo), hi)


@wp.func
def weight(f: float, d: int):
    if d == 1:
        return f
    return 1.0 - f


@wp.func
def sample(a: wp.array3d(dtype=float), pos: wp.vec3, off: wp.vec3, h: float, nx: int, ny: int, nz: int):
    i, f = base_index(pos, off, h)
    s = float(0.0)
    for dx in range(2):
        ix = clampi(i[0] + dx, 0, nx - 1)
        for dy in range(2):
            iy = clampi(i[1] + dy, 0, ny - 1)
            for dz in range(2):
                iz = clampi(i[2] + dz, 0, nz - 1)
                s += weight(f[0], dx) * weight(f[1], dy) * weight(f[2], dz) * a[ix, iy, iz]
    return s


@wp.func
def scatter(a: wp.array3d(dtype=float), pos: wp.vec3, off: wp.vec3, h: float, nx: int, ny: int, nz: int, val: float):
    i, f = base_index(pos, off, h)
    for dx in range(2):
        ix = clampi(i[0] + dx, 0, nx - 1)
        for dy in range(2):
            iy = clampi(i[1] + dy, 0, ny - 1)
            for dz in range(2):
                iz = clampi(i[2] + dz, 0, nz - 1)
                wp.atomic_add(a, ix, iy, iz, weight(f[0], dx) * weight(f[1], dy) * weight(f[2], dz) * val)


@wp.func
def value_range(a: wp.array3d(dtype=float), pos: wp.vec3, off: wp.vec3, h: float, nx: int, ny: int, nz: int):
    """Min and max of the eight lattice values around `pos` (MacCormack limiter)."""
    i, f = base_index(pos, off, h)
    lo = float(1.0e30)
    hi = float(-1.0e30)
    for dx in range(2):
        ix = clampi(i[0] + dx, 0, nx - 1)
        for dy in range(2):
            iy = clampi(i[1] + dy, 0, ny - 1)
            for dz in range(2):
                iz = clampi(i[2] + dz, 0, nz - 1)
                lo = wp.min(lo, a[ix, iy, iz])
                hi = wp.max(hi, a[ix, iy, iz])
    return wp.vec2(lo, hi)


@wp.func
def sample_velocity(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), pos: wp.vec3, h: float, n: int):
    return wp.vec3(sample(u, pos, wp.vec3(0.0, 0.5, 0.5), h, n + 1, n, n),
                   sample(v, pos, wp.vec3(0.5, 0.0, 0.5), h, n, n + 1, n),
                   sample(w, pos, wp.vec3(0.5, 0.5, 0.0), h, n, n, n + 1))


@wp.func
def face_position(axis: int, i: int, j: int, k: int, h: float):
    if axis == 0:
        return wp.vec3(float(i) * h, (float(j) + 0.5) * h, (float(k) + 0.5) * h)
    if axis == 1:
        return wp.vec3((float(i) + 0.5) * h, float(j) * h, (float(k) + 0.5) * h)
    return wp.vec3((float(i) + 0.5) * h, (float(j) + 0.5) * h, float(k) * h)


@wp.func
def face_offset(axis: int):
    if axis == 0:
        return wp.vec3(0.0, 0.5, 0.5)
    if axis == 1:
        return wp.vec3(0.5, 0.0, 0.5)
    return wp.vec3(0.5, 0.5, 0.0)


@wp.func
def face_dims(axis: int, n: int):
    if axis == 0:
        return wp.vec3i(n + 1, n, n)
    if axis == 1:
        return wp.vec3i(n, n + 1, n)
    return wp.vec3i(n, n, n + 1)


@wp.func
def face_axis_index(axis: int, i: int, j: int, k: int):
    if axis == 0:
        return i
    if axis == 1:
        return j
    return k


@wp.func
def backtrace(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), x: wp.vec3, h: float, n: int, dt: float):
    """Second-order (midpoint) departure point x - dt u(x)."""
    vel = sample_velocity(u, v, w, x, h, n)
    vel = sample_velocity(u, v, w, x - 0.5 * dt * vel, h, n)
    return x - dt * vel
