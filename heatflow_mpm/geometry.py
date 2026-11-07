"""Mesh loading and interior particle sampling (jittered lattice, winding-number inside test on the GPU)."""
from __future__ import annotations

import os
import numpy as np
import trimesh
import warp as wp

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DEFAULT_MESH = os.path.join(ASSETS, "bunny.obj")


@wp.kernel
def _inside_mesh(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), max_dist: float, inside: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh_id, points[i], max_dist, 2.0, 0.5)
    if q.result and q.sign < 0.0:
        inside[i] = 1
    else:
        inside[i] = 0


def load_mesh(path: str = DEFAULT_MESH, height: float = 0.10, yaw_deg: float = 0.0, y_up: bool = True) -> trimesh.Trimesh:
    """Load a mesh, convert Y-up to Z-up if asked, turn it by `yaw_deg` about z, scale it to `height` metres, centre it in xy, lowest point at z = 0."""
    mesh = trimesh.load(path, force="mesh", process=True)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if y_up:
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    c, s_ = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    v = np.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)
    v -= v.min(axis=0)
    v *= height / v[:, 2].max()
    bbox = v.max(axis=0)
    v[:, 0] -= 0.5 * bbox[0]
    v[:, 1] -= 0.5 * bbox[1]
    return trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), process=False)


def sample_interior(mesh: trimesh.Trimesh, spacing: float, jitter: float, rng: np.random.Generator, device: str = "cuda:0") -> np.ndarray:
    lo = mesh.vertices.min(axis=0) - spacing
    hi = mesh.vertices.max(axis=0) + spacing
    axes = [np.arange(lo[d] + 0.5 * spacing, hi[d], spacing) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid += rng.uniform(-jitter, jitter, size=grid.shape)
    with wp.ScopedDevice(device):
        wp_mesh = wp.Mesh(points=wp.array(mesh.vertices, dtype=wp.vec3), indices=wp.array(np.asarray(mesh.faces).reshape(-1), dtype=wp.int32), support_winding_number=True)
        pts = wp.array(grid.astype(np.float32), dtype=wp.vec3)
        flags = wp.zeros(len(grid), dtype=wp.int32)
        wp.launch(_inside_mesh, dim=len(grid), inputs=[wp_mesh.id, pts, float(np.max(hi - lo)), flags])
        mask = flags.numpy().astype(bool)
    return grid[mask].astype(np.float32)


def mesh_particles(path: str = DEFAULT_MESH, height: float = 0.10, spacing: float = 2.5e-3, L: float = 0.48, seed: int = 0, yaw_deg: float = 0.0,
                   jitter: float = 0.2, y_up: bool = True, device: str = "cuda:0") -> np.ndarray:
    """Particles filling the mesh, centred on the domain floor at (L/2, L/2), lowest particle half a spacing above z = 0."""
    mesh = load_mesh(path, height, yaw_deg, y_up)
    P = sample_interior(mesh, spacing, jitter * spacing, np.random.default_rng(seed), device)
    P[:, 0] += L / 2; P[:, 1] += L / 2; P[:, 2] += 0.5 * spacing
    return P.astype(np.float32)
