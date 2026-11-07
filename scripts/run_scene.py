"""Run one scene: a mesh of thermal material on the floor, heated by the air from an electric heater.

    python scripts/run_scene.py --out data/bunny_fan --fan 2.0 --power 400 --seconds 20
    python scripts/run_scene.py --out data/bunny_conv --fan 0.0 --power 400 --seconds 20     # convection only

Output: <out>/static.npz (grid, heater geometry, material), <out>/meta.json, <out>/frames/frame_NNNN.npz
(t, pos, vel, temp f16, melt f16, grid_T f16 [n^3], air_u f16 [n^3, 3]).
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heatflow_mpm import HeatflowMPM, ThermalMaterial, Domain, Heater, mesh_particles  # noqa: E402
from heatflow_mpm.geometry import DEFAULT_MESH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--frame-dt", type=float, default=0.02); ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--mesh", default=DEFAULT_MESH); ap.add_argument("--height", type=float, default=0.10); ap.add_argument("--yaw", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spacing", type=float, default=2.5e-3); ap.add_argument("--L", type=float, default=0.48); ap.add_argument("--h", type=float, default=0.005)
    ap.add_argument("--t-amb", type=float, default=25.0); ap.add_argument("--t-init", type=float, default=20.0); ap.add_argument("--walls", choices=["fixed", "adiabatic"], default="fixed")
    ap.add_argument("--power", type=float, default=400.0); ap.add_argument("--t-cap", type=float, default=120.0); ap.add_argument("--t-rod", type=float, default=300.0); ap.add_argument("--fan", type=float, default=0.0)
    ap.add_argument("--heater-dist", type=float, default=0.21); ap.add_argument("--heater-z", type=float, default=0.12); ap.add_argument("--heater-tilt", type=float, default=15.0); ap.add_argument("--no-heater", action="store_true")
    ap.add_argument("--rho", type=float, default=1300.0); ap.add_argument("--t-sol", type=float, default=30.0); ap.add_argument("--t-liq", type=float, default=36.0)
    ap.add_argument("--yield-solid", type=float, default=2e4); ap.add_argument("--viscosity-solid", type=float, default=5.0); ap.add_argument("--viscosity-liquid", type=float, default=30.0)
    ap.add_argument("--kappa", type=float, default=2e-5); ap.add_argument("--kappa-air", type=float, default=2.2e-5); ap.add_argument("--ground-friction", type=float, default=0.3)
    ap.add_argument("--no-air-frames", action="store_true", help="do not store the air velocity per frame (saves 4 MB per frame at 96^3)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    wp.config.log_level = wp.LOG_WARNING; wp.init()
    mat = ThermalMaterial(rho=a.rho, t_sol=a.t_sol, t_liq=a.t_liq, yield_solid=a.yield_solid, viscosity_solid=a.viscosity_solid, viscosity_liquid=a.viscosity_liquid, kappa=a.kappa, ground_friction=a.ground_friction)
    heater = Heater(dist=a.heater_dist, z=a.heater_z, tilt=a.heater_tilt, t_rod=a.t_rod, power=a.power, t_cap=a.t_cap, fan=a.fan, on=not a.no_heater)
    dom = Domain(L=a.L, h=a.h, t_amb=a.t_amb, walls=a.walls)
    P = mesh_particles(a.mesh, a.height, a.spacing, a.L, a.seed, a.yaw, device=a.device)
    sw = HeatflowMPM(P, mat, heater, dom, spacing=a.spacing, t_init=a.t_init, kappa_air=a.kappa_air, device=a.device)
    os.makedirs(f"{a.out}/frames", exist_ok=True); n = sw.n
    np.savez(f"{a.out}/static.npz", **sw.static_dict())
    json.dump(vars(a) | dict(n_particles=int(len(P)), substeps=int(round(a.frame_dt / a.dt)), heater=sw.heater_geom), open(f"{a.out}/meta.json", "w"), indent=1)
    sub = int(round(a.frame_dt / a.dt)); nf = int(round(a.seconds / a.frame_dt)); t0 = time.time()
    save = lambda k: np.savez(f"{a.out}/frames/frame_{k:04d}.npz", **sw.frame_dict(k * a.frame_dt, with_air=not a.no_air_frames))
    save(0)
    print(f"{len(P)} particles, grid {n}^3 at {a.h * 1000:.0f} mm, heater {'off' if a.no_heater else f'{a.power:.0f} W cap {a.t_cap:.0f} C fan {a.fan} m/s'}", flush=True)
    for k in range(1, nf + 1):
        for _ in range(sub):
            sw.substep(a.dt)
        save(k)
        if k % 50 == 0 or k == nf:
            x = sw.positions(); f = sw.melt_fraction(); tp = sw.temperatures(); T = sw.grid_T(); c = n // 2
            print(f"frame {k:4d} t {k * a.frame_dt:5.1f} s | height {x[:, 2].max() * 100:5.1f} cm, rms radius {np.sqrt(((x[:, :2] - x[:, :2].mean(0)) ** 2).sum(1).mean()) * 100:4.1f} cm | "
                  f"melted {100 * (f >= 1).mean():4.1f} %, partly {100 * ((f > 0) & (f < 1)).mean():4.1f} % | particle T mean {tp.mean():5.1f} max {tp.max():5.1f} | "
                  f"air T above the material {T[c, c, min(int(x[:, 2].max() / a.h) + 1, n - 1)]:5.1f} mean {T[sw.cell_type == 0].mean():5.1f} C | air speed max {sw.grid.speed_max():.2f} m/s | {(time.time() - t0) / k:.2f} s/frame", flush=True)
    print("RUN DONE", a.out)


if __name__ == "__main__":
    main()
