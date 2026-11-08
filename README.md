# Heatflow-MPM

Eulerian air (velocity and temperature) two-way coupled to thermal implicit MPM, on the GPU (NVIDIA Warp + Newton).

A chocolate-like bunny stands in a box in front of an electric heater. The air is a Stable-Fluids MAC grid with Boussinesq
buoyancy that carries the heat; the material is a Newton implicit MPM body whose yield stress and viscosity follow the temperature
it receives from the grid; the material cools the air it touches and blocks its flow.

![melting bunny](docs/sheet.png)

*Stanford bunny of a chocolate-like material in front of a 400 W fan heater, 0 to 14 s (`scripts/run_scene.py --fan 2.0`).*

## Model

Air (grid, cell h): incompressible Euler with buoyancy, obstacles and the material's solid fraction,

    du/dt + (u . grad) u = -grad p / rho + g (T - T_amb) / T_amb z,    div(alpha_f u + alpha_s v_s) = 0

semi-Lagrangian advection, implicit relaxation of the faces toward the material and the heater body, mixture-divergence
projection (Jacobi-preconditioned CG).

Temperature (grid): `dT/dt + u . grad T = div(kappa grad T) + q`, advected in air cells, rebuilt from the particles in material
cells, diffused with backward Euler (CG), rods and room walls fixed, floor adiabatic, heater power injected into the trough air and
capped at an outlet temperature.

Material (Newton implicit MPM): each step the particles receive the temperature increment at their position (FLIP); the melt
fraction `f = clamp((T - T_sol)/(T_liq - T_sol))` sets yield stress `y (1 - f)` and viscosity `mu_s (1 - f) + mu_l f`.
Not modelled: radiation, latent heat.

## Install

    pip install -e ".[render,test]"
    # NVIDIA Newton (the MPM) is not on PyPI: https://github.com/newton-physics/newton

CUDA GPU required. Tested with Python 3.11, warp-lang 1.17, newton 1.7, mitsuba 3.9.

## Run

    python scripts/run_scene.py --out data/bunny_fan  --fan 2.0 --power 400 --t-cap 120     # fan heater
    python scripts/run_scene.py --out data/bunny_conv --fan 0.0 --power 400                 # convection only
    python scripts/analyze.py --run data/bunny_fan --every 100                               # table + side views
    python scripts/render.py  --run data/bunny_fan --times 0 2 4 6 8 10 12 14               # sheet; --every 5 --video for a movie

Defaults: bunny 10 cm at 2.5 mm spacing (13k particles), box 0.48 m with 5 mm cells, 2 ms substeps, frames every 20 ms, room 25 C,
material 20 C, heater 0.21 m from the material, rods 300 C, 400 W capped at 120 C. Any mesh: `--mesh path.obj --height 0.08`.
About 7 ms per air substep and 1 s per frame for the MPM on an RTX 5080. Output: `static.npz`, `meta.json`, `frames/frame_NNNN.npz`
with `pos`, `vel`, `temp`, `melt`, `grid_T` (n^3, f16) and `air_u` (n^3 x 3, f16).

```python
from heatflow_mpm import HeatflowMPM, Material, Heater, mesh_particles
sim = HeatflowMPM(mesh_particles(), Material(), Heater(power=400.0, t_cap=120.0, fan=2.0))
for _ in range(500):            # 1 s
    sim.substep(2e-3)
T_air, u_air = sim.grid_T(), sim.air_velocity()
x, T_p, f = sim.positions(), sim.temperatures(), sim.melt_fraction()
```

## Tests

    pytest -q tests

Implicit conduction against the erfc solution, a rising heater plume, a fan stream heating the material position, a cold solid
holding its shape.

## References

Stam 1999 (Stable Fluids); Fedkiw, Stam, Jensen 2001 (smoke, buoyancy); Stomakhin et al. 2014 (Augmented MPM, particle heat
exchange); NVIDIA Newton `SolverImplicitMPM`; Stanford bunny mesh.

MIT license.
