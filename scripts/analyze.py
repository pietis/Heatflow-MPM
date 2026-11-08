"""Melt timeline table and side views (particles by melt fraction over the air temperature slice).

    python scripts/analyze.py --run data/bunny_fan --every 100
"""
from __future__ import annotations

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--every", type=int, default=100); a = ap.parse_args()
    st = np.load(f"{a.run}/static.npz"); n = int(st["n"]); h = float(st["h"]); L = float(st["L"])
    frames = sorted(glob.glob(f"{a.run}/frames/frame_*.npz")); picks = []
    print("t(s) | height cm | rms radius cm | melted % | partly % | particle mean T | air T 2 cm above | air mean T | air speed max m/s")
    for k, f in enumerate(frames):
        if k % a.every and k != len(frames) - 1:
            continue
        d = np.load(f); P = d["pos"]; m = d["melt"].astype(np.float32); tp = d["temp"].astype(np.float32); G = d["grid_T"].astype(np.float32).reshape(n, n, n)
        c = P.mean(0); i, j = int(c[0] / h), int(c[1] / h); ktop = min(int(P[:, 2].max() / h) + 4, n - 1)
        vmax = float(np.linalg.norm(d["air_u"].astype(np.float32), axis=1).max()) if "air_u" in d.files else float("nan")
        air = G[st["cell_type"] == 0]
        print(f"{float(d['t']):5.1f} | {np.quantile(P[:, 2], 0.99) * 100:5.2f} | {np.sqrt(((P[:, :2] - c[:2]) ** 2).sum(1).mean()) * 100:5.2f} | {100 * (m >= 0.999).mean():5.1f} | "
              f"{100 * ((m > 0) & (m < 0.999)).mean():5.1f} | {tp.mean():5.1f} | {G[i, j, ktop]:5.1f} | {air.mean():5.1f} | {vmax:5.2f}")
        picks.append((float(d["t"]), P, m, G))
    cols = min(len(picks), 6); rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.6 * rows), squeeze=False)
    j0 = int(picks[0][1][:, 1].mean() / h); tmax = float(st["mat_t_liq"]) + 60.0
    for ax, (t, P, m, G) in zip(axes.flat, picks):
        ax.imshow(G[:, j0, :].T, origin="lower", extent=(0, L, 0, L), cmap="inferno", vmin=float(st["t_amb"]) - 5, vmax=tmax, alpha=0.7)
        ax.scatter(P[:, 0], P[:, 2], c=m, s=0.5, cmap="coolwarm", vmin=0, vmax=1)
        ax.set_xlim(0, L); ax.set_ylim(0, 0.5 * L); ax.set_aspect("equal"); ax.set_title(f"t = {t:.1f} s", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.suptitle(f"{os.path.basename(a.run)}: particles (blue solid, red melted) over the air temperature slice through the centre", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.run}/side_views.png", dpi=110); print("wrote", f"{a.run}/side_views.png")


if __name__ == "__main__":
    main()
