"""Particles filling a mesh (jittered lattice, winding-number inside test on the GPU)."""
from __future__ import annotations

import os
import numpy as np
import trimesh
import warp as wp

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
BUNNY = os.path.join(ASSETS, "bunny.obj")


@wp.kernel
def _inside(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), max_dist: float, flag: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh_id, points[i], max_dist, 2.0, 0.5)
    flag[i] = 1 if (q.result and q.sign < 0.0) else 0


def load_mesh(path: str = BUNNY, height: float = 0.10, yaw_deg: float = 0.0, y_up: bool = True) -> trimesh.Trimesh:
    """Mesh scaled to `height` [m], Z-up, yawed, centred in xy with its lowest point at z = 0."""
    m = trimesh.load(path, force="mesh", process=True)
    v = np.asarray(m.vertices, float)
    if y_up:
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    v = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
    v -= v.min(0)
    v *= height / v[:, 2].max()
    v[:, :2] -= v.max(0)[:2] / 2
    return trimesh.Trimesh(v, np.asarray(m.faces), process=False)


def sample_interior(mesh: trimesh.Trimesh, spacing: float, jitter: float, rng: np.random.Generator, device: str) -> np.ndarray:
    lo, hi = mesh.vertices.min(0) - spacing, mesh.vertices.max(0) + spacing
    axes = [np.arange(lo[d] + spacing / 2, hi[d], spacing) for d in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    P = P + rng.uniform(-jitter, jitter, P.shape) * spacing
    with wp.ScopedDevice(device):
        wm = wp.Mesh(points=wp.array(mesh.vertices, dtype=wp.vec3), indices=wp.array(np.asarray(mesh.faces).reshape(-1), dtype=wp.int32), support_winding_number=True)
        flag = wp.zeros(len(P), dtype=wp.int32)
        wp.launch(_inside, dim=len(P), inputs=[wm.id, wp.array(P.astype(np.float32), dtype=wp.vec3), float(np.max(hi - lo)), flag])
        keep = flag.numpy().astype(bool)
    return P[keep].astype(np.float32)


def mesh_particles(path: str = BUNNY, height: float = 0.10, spacing: float = 2.5e-3, L: float = 0.48, seed: int = 0, yaw_deg: float = 0.0,
                   device: str = "cuda:0") -> np.ndarray:
    """Particles filling the mesh, standing on the floor at the centre of a box of edge L."""
    P = sample_interior(load_mesh(path, height, yaw_deg), spacing, 0.2, np.random.default_rng(seed), device)
    return (P + np.array([L / 2, L / 2, spacing / 2], np.float32)).astype(np.float32)
