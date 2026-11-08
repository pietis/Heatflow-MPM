"""Melt a mesh of thermal material in front of an electric heater; write frames.

    python scripts/run_scene.py --out data/bunny_fan --fan 2.0 --power 400
Output: static.npz, meta.json, frames/frame_NNNN.npz (pos, vel, temp f16, melt f16, grid_T f16 [n^3], air_u f16 [n^3, 3]).
"""
import argparse, json, os, sys, time
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heatflow_mpm import HeatflowMPM, Material, Heater, GridConfig, mesh_particles, BUNNY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--frame-dt", type=float, default=0.02); ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--mesh", default=BUNNY); ap.add_argument("--height", type=float, default=0.10); ap.add_argument("--yaw", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spacing", type=float, default=2.5e-3); ap.add_argument("--L", type=float, default=0.48); ap.add_argument("--h", type=float, default=0.005)
    ap.add_argument("--t-amb", type=float, default=25.0); ap.add_argument("--t-init", type=float, default=20.0)
    ap.add_argument("--power", type=float, default=400.0); ap.add_argument("--t-cap", type=float, default=120.0); ap.add_argument("--t-rod", type=float, default=300.0); ap.add_argument("--fan", type=float, default=0.0)
    ap.add_argument("--heater-dist", type=float, default=0.21); ap.add_argument("--heater-z", type=float, default=0.12); ap.add_argument("--no-heater", action="store_true")
    ap.add_argument("--t-sol", type=float, default=30.0); ap.add_argument("--t-liq", type=float, default=36.0); ap.add_argument("--yield-solid", type=float, default=2e4)
    ap.add_argument("--viscosity-solid", type=float, default=5.0); ap.add_argument("--viscosity-liquid", type=float, default=30.0); ap.add_argument("--kappa", type=float, default=2e-5)
    ap.add_argument("--no-air-frames", action="store_true"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    wp.config.log_level = wp.LOG_WARNING; wp.init()
    mat = Material(t_sol=a.t_sol, t_liq=a.t_liq, yield_solid=a.yield_solid, viscosity_solid=a.viscosity_solid, viscosity_liquid=a.viscosity_liquid, kappa=a.kappa)
    heater = Heater(dist=a.heater_dist, z=a.heater_z, t_rod=a.t_rod, power=a.power, t_cap=a.t_cap, fan=a.fan, on=not a.no_heater)
    P = mesh_particles(a.mesh, a.height, a.spacing, a.L, a.seed, a.yaw, a.device)
    sim = HeatflowMPM(P, mat, heater, GridConfig(L=a.L, h=a.h, t_amb=a.t_amb), a.spacing, a.t_init, device=a.device)
    os.makedirs(f"{a.out}/frames", exist_ok=True)
    np.savez(f"{a.out}/static.npz", **sim.static_dict())
    json.dump(vars(a) | dict(n_particles=int(len(P)), heater=sim.heater_geom), open(f"{a.out}/meta.json", "w"), indent=1)
    sub, nf, t0 = int(round(a.frame_dt / a.dt)), int(round(a.seconds / a.frame_dt)), time.time()
    np.savez(f"{a.out}/frames/frame_0000.npz", **sim.frame_dict(0.0, not a.no_air_frames))
    print(f"{len(P)} particles, grid {sim.grid.n}^3", flush=True)
    for k in range(1, nf + 1):
        for _ in range(sub):
            sim.substep(a.dt)
        np.savez(f"{a.out}/frames/frame_{k:04d}.npz", **sim.frame_dict(k * a.frame_dt, not a.no_air_frames))
        if k % 50 == 0 or k == nf:
            x, f, T = sim.positions(), sim.melt_fraction(), sim.grid_T(); c = sim.grid.n // 2
            print(f"t {k * a.frame_dt:5.1f} s | height {x[:, 2].max() * 100:4.1f} cm | melted {100 * (f >= 1).mean():5.1f} % | "
                  f"air above the material {T[c, c, min(int(x[:, 2].max() / a.h) + 1, sim.grid.n - 1)]:5.1f} C | {(time.time() - t0) / k:.2f} s/frame", flush=True)
    print("done", a.out)


if __name__ == "__main__":
    main()
