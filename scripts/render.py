"""Ray-traced renders (Mitsuba 3, CPU): the material surface rebuilt from the particles (mass splat, marching cubes), a pan, a table and the
heater where the physics has it. The camera aims and focuses on the material; the field of view is fitted so a chosen fraction of the
heater head is in frame.

    python scripts/render.py --run data/bunny_fan --times 0 2 4 6 8 10 12 14      # contact sheet
    python scripts/render.py --run data/bunny_fan --every 5 --video                # 10 fps video
"""
from __future__ import annotations

import argparse, glob, os, time
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix as rot
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from PIL import Image, ImageDraw, ImageFont
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")
T4 = mi.ScalarTransform4f

PAN_R, PAN_H, PAN_T = 0.145, 0.035, 0.004
MATERIAL = dict(type="principled", base_color=dict(type="rgb", value=[0.40, 0.19, 0.06]), roughness=0.22, specular=0.7, clearcoat=0.4, clearcoat_gloss=0.8)
PAN_IN = dict(type="principled", base_color=dict(type="rgb", value=[0.035, 0.035, 0.035]), metallic=0.4, roughness=0.45, specular=0.5)
PAN_OUT = dict(type="principled", base_color=dict(type="rgb", value=[0.04, 0.06, 0.55]), metallic=0.7, roughness=0.3, specular=0.6)
TABLE = dict(type="diffuse", reflectance=dict(type="rgb", value=[0.55, 0.55, 0.55]))
DARK_METAL = dict(type="principled", base_color=dict(type="rgb", value=[0.03, 0.03, 0.03]), metallic=0.5, roughness=0.4)
REFLECTOR = dict(type="principled", base_color=dict(type="rgb", value=[0.80, 0.80, 0.80]), metallic=0.9, roughness=0.18)
ROD_GLOW = dict(type="rgb", value=[16.0, 3.0, 0.6])


def two_sided(b): return dict(type="twosided", material=b)


def surface_mesh(P, spacing, sigma_cells=2.0, z_floor=0.0, taubin=8):
    cell = spacing / 2
    Pm = np.concatenate([P, P * np.array([1, 1, -1], np.float32) + np.array([0, 0, 2 * z_floor], np.float32)])   # mirrored below the floor
    lo = Pm.min(0) - 6 * cell; hi = Pm.max(0) + 6 * cell
    n = np.ceil((hi - lo) / cell).astype(int) + 1
    idx = np.floor((Pm - lo) / cell).astype(int)
    grid = np.zeros(n, np.float32); np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    grid = gaussian_filter(grid, sigma_cells)
    verts, faces, _, _ = marching_cubes(grid, level=0.5 * (cell / spacing) ** 3, spacing=(cell, cell, cell))
    m = trimesh.Trimesh(verts + lo, faces, process=True)
    if taubin:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=taubin)
    m = trimesh.intersections.slice_mesh_plane(m, plane_normal=[0, 0, 1], plane_origin=[0, 0, z_floor + 2e-4], cap=True)
    m.fix_normals()
    return m


def pan_meshes(centre, cam_az_deg):
    cx, cy = centre
    inner = trimesh.creation.annulus(PAN_R, PAN_R + PAN_T / 2, PAN_H, sections=160); inner.apply_translation([cx, cy, PAN_H / 2])
    outer = trimesh.creation.annulus(PAN_R + PAN_T / 2, PAN_R + PAN_T, PAN_H, sections=160); outer.apply_translation([cx, cy, PAN_H / 2])
    bottom = trimesh.creation.cylinder(PAN_R + PAN_T, 0.004, sections=160); bottom.apply_translation([cx, cy, -0.002])
    handle = trimesh.creation.box([0.19, 0.034, 0.011]); handle.apply_transform(rot(np.deg2rad(-10.0), [0, 1, 0])); handle.apply_translation([PAN_R + PAN_T + 0.085, 0, PAN_H - 0.006])
    bracket = trimesh.creation.box([0.03, 0.05, 0.02]); bracket.apply_translation([PAN_R + PAN_T + 0.012, 0, PAN_H - 0.012])
    ang = np.deg2rad(cam_az_deg + 180.0 - 45.0)   # radial handle: away from the camera, turned toward its right
    for m in (handle, bracket):
        m.apply_transform(rot(ang, [0, 0, 1])); m.apply_translation([cx, cy, 0])
    return [("pan_in", inner, PAN_IN, None), ("pan_out", outer, PAN_OUT, None), ("pan_bottom", bottom, PAN_IN, None), ("pan_handle", handle, PAN_IN, None), ("pan_bracket", bracket, PAN_IN, None)]


def heater_meshes(pos, yaw_deg, tilt_deg, radius=0.055, length=0.22):
    parts = []
    half = lambda m: trimesh.intersections.slice_mesh_plane(m, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0], cap=False)

    def xcyl(r, h, sections=96):
        c = trimesh.creation.cylinder(r, h, sections=sections); c.apply_transform(rot(np.pi / 2, [0, 1, 0])); return c
    parts.append(("heater_head_shell", half(xcyl(radius + 0.006, length)), DARK_METAL, None))
    parts.append(("heater_head_reflector", half(xcyl(radius, length)), REFLECTOR, None))
    for sx in (-1, 1):
        cap = trimesh.intersections.slice_mesh_plane(xcyl(radius + 0.006, 0.008), plane_normal=[0, 1, 0], plane_origin=[0, 0, 0], cap=True)
        cap.apply_translation([sx * (length / 2 + 0.004), 0, 0]); parts.append((f"heater_head_cap{sx + 1}", cap, DARK_METAL, None))
    for i, zc in enumerate((-0.03, 0.03)):
        rod = xcyl(0.007, length - 0.03, 48); rod.apply_translation([0, 0.03, zc]); parts.append((f"heater_head_rod{i}", rod, dict(type="diffuse"), ROD_GLOW))
    for i, xg in enumerate(np.linspace(-length / 2 + 0.02, length / 2 - 0.02, 9)):
        bar = trimesh.creation.cylinder(0.0015, 2 * radius, sections=12); bar.apply_translation([xg, -0.004, 0]); parts.append((f"heater_head_grille_v{i}", bar, DARK_METAL, None))
    for i, zg in enumerate((-0.045, 0.0, 0.045)):
        bar = xcyl(0.0015, length - 0.02, 12); bar.apply_translation([0, -0.004, zg]); parts.append((f"heater_head_grille_h{i}", bar, DARK_METAL, None))
    M = trimesh.transformations.translation_matrix(pos) @ rot(np.deg2rad(yaw_deg), [0, 0, 1]) @ rot(np.deg2rad(tilt_deg), [1, 0, 0])
    for _, m, _, _ in parts:
        m.apply_transform(M)
    post = trimesh.creation.cylinder(0.012, pos[2] - radius + 0.02, sections=32); post.apply_translation([pos[0], pos[1], (pos[2] - radius + 0.02) / 2])
    base = trimesh.creation.cylinder(0.05, 0.012, sections=96); base.apply_translation([pos[0], pos[1], 0.006])
    return parts + [("heater_post", post, DARK_METAL, None), ("heater_base", base, DARK_METAL, None)]


def camera_eye(centre, look_z, az_deg, el_deg, dist):
    cx, cy = centre; az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([cx + dist * np.cos(el) * np.cos(az), cy + dist * np.cos(el) * np.sin(az), look_z + dist * np.sin(el)]), np.array([cx, cy, look_z])


def visible_fraction(V, eye, look, fov_x_deg, width, height):
    f = look - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    d = V - eye; xc, yc, zc = d @ r, d @ u, d @ f
    tx = np.tan(np.deg2rad(fov_x_deg) / 2); ty = tx * height / width
    return float(((zc > 1e-3) & (np.abs(xc / (zc * tx)) <= 1) & (np.abs(yc / (zc * ty)) <= 1)).mean())


def fit_fov(V, eye, look, width, height, target=0.70, lo=15.0, hi=70.0):
    for fov in np.arange(hi, lo - 0.01, -0.25):
        if visible_fraction(V, eye, look, fov, width, height) <= target:
            return float(fov)
    return float(lo)


def scene_dict(surf_ply, static_shapes, centre, az, el, dist, width, height, spp, fov, look_z, aperture):
    cx, cy = centre; eye, look = camera_eye(centre, look_z, az, el, dist); eye = eye.tolist(); look = look.tolist()
    sensor = dict(type="thinlens", aperture_radius=aperture, focus_distance=dist) if aperture > 0 else dict(type="perspective")
    sensor.update(fov=fov, fov_axis="x", to_world=T4().look_at(origin=eye, target=look, up=[0, 0, 1]),
                  film=dict(type="hdrfilm", width=width, height=height, pixel_format="rgb", rfilter=dict(type="gaussian")), sampler=dict(type="independent", sample_count=spp))
    d = dict(type="scene", integrator=dict(type="path", max_depth=8), sensor=sensor,
             env=dict(type="constant", radiance=dict(type="rgb", value=[0.42, 0.42, 0.44])),
             key=dict(type="rectangle", to_world=T4().translate([cx - 0.25, cy - 0.35, 0.75]).rotate([1, 0, 0], 180).scale(0.25), emitter=dict(type="area", radiance=dict(type="rgb", value=[7.0, 7.0, 7.0])), bsdf=dict(type="diffuse")),
             fill=dict(type="rectangle", to_world=T4().translate([cx + 0.5, cy - 0.2, 0.45]).rotate([0, 1, 0], -120).scale(0.3), emitter=dict(type="area", radiance=dict(type="rgb", value=[2.0, 2.0, 2.2])), bsdf=dict(type="diffuse")),
             table=dict(type="rectangle", to_world=T4().translate([cx, cy, -0.0045]).scale(3.0), bsdf=TABLE),
             surface=dict(type="ply", filename=surf_ply, bsdf=two_sided(MATERIAL)))
    for name, ply, mat, glow in static_shapes:
        d[name] = dict(type="ply", filename=ply, bsdf=two_sided(mat)) if glow is None else dict(type="ply", filename=ply, bsdf=mat, emitter=dict(type="area", radiance=glow))
    return d


def to_png(img, path):
    a = np.clip(np.array(img), 0, None); a = np.clip(a / (1 + a) * 1.6, 0, 1)
    a = np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path)


def label(path, text):
    im = Image.open(path); dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(18, im.width // 30))
    except Exception:
        font = ImageFont.load_default()
    dr.text((im.width * 0.97 - dr.textlength(text, font=font), im.height * 0.03), text, fill=(20, 20, 20), font=font); im.save(path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--times", type=float, nargs="*"); ap.add_argument("--every", type=int, default=0); ap.add_argument("--video", action="store_true"); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--az", type=float, default=-110.0); ap.add_argument("--el", type=float, default=16.0); ap.add_argument("--dist", type=float, default=0.65); ap.add_argument("--look_z", type=float, default=0.05)
    ap.add_argument("--aperture", type=float, default=0.005); ap.add_argument("--fov", type=float, default=0.0, help="0 = fit so that --heater_visible of the heater head is in frame")
    ap.add_argument("--heater_visible", type=float, default=0.70); ap.add_argument("--no-pan", action="store_true")
    ap.add_argument("--width", type=int, default=1000); ap.add_argument("--height", type=int, default=700); ap.add_argument("--spp", type=int, default=160)
    a = ap.parse_args(); out = a.out or f"{a.run}/render"; os.makedirs(out, exist_ok=True)
    st = np.load(f"{a.run}/static.npz"); spacing = float(st["spacing"]); L = float(st["L"]); centre = (L / 2, L / 2)
    frames = sorted(glob.glob(f"{a.run}/frames/frame_*.npz")); fdt = float(np.load(frames[1])["t"]) - float(np.load(frames[0])["t"])
    ks = [int(round(t / fdt)) for t in a.times] if a.times else list(range(0, len(frames), max(1, a.every)))
    parts = [] if a.no_pan else pan_meshes(centre, a.az)
    heater = []
    if int(st["heater_on"]) == 1:
        hc = st["heater_centre"].astype(float); fh = st["heater_facing"].astype(float)[:2]; fh /= np.linalg.norm(fh)
        heater = heater_meshes(hc.tolist(), np.degrees(np.arctan2(fh[1], fh[0])) + 90.0, float(st["heater_tilt"]), float(st["heater_radius"]), float(st["heater_length"]))
    parts += heater
    fov = a.fov
    if fov <= 0:
        eye, look = camera_eye(centre, a.look_z, a.az, a.el, a.dist)
        V = np.concatenate([m.vertices for name, m, _, _ in heater if name.startswith("heater_head")]) if heater else None
        fov = fit_fov(V, eye, look, a.width, a.height, a.heater_visible) if V is not None else 38.0
        print(f"fov {fov:.2f} deg" + (f": heater head {100 * visible_fraction(V, eye, look, fov, a.width, a.height):.0f} % in frame" if V is not None else ""), flush=True)
    static_shapes = []
    for name, m, mat, glow in parts:
        p = f"{out}/{name}.ply"; m.export(p); static_shapes.append((name, p, mat, glow))
    pngs = []
    for k in ks:
        d = np.load(frames[k]); P = d["pos"].astype(np.float32); t = float(d["t"]); t0 = time.time()
        m = surface_mesh(P, spacing); ply = f"{out}/surface_{k:04d}.ply"; m.export(ply)
        img = mi.render(mi.load_dict(scene_dict(ply, static_shapes, centre, a.az, a.el, a.dist, a.width, a.height, a.spp, fov, a.look_z, a.aperture)), spp=a.spp)
        png = f"{out}/frame_{k:04d}.png"; to_png(img, png); label(png, f"t = {t:.1f} s"); pngs.append(png); os.remove(ply)
        print(f"frame {k} (t {t:.2f} s): {len(m.faces)} faces, volume {m.volume * 1e6:.0f} cm3, {time.time() - t0:.1f} s", flush=True)
    for _, p, _, _ in static_shapes:
        os.remove(p)
    if a.video:
        import imageio.v2 as iio
        with iio.get_writer(f"{out}/video.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=1) as w:
            for p in pngs:
                w.append_data(np.array(Image.open(p).convert("RGB")))
        print("wrote", f"{out}/video.mp4")
    elif len(pngs) > 1:
        ims = [Image.open(p) for p in pngs]; cols = min(4, len(ims)); rows = int(np.ceil(len(ims) / cols)); w, h = ims[0].size
        sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(f"{out}/sheet.png"); print("wrote", f"{out}/sheet.png")


if __name__ == "__main__":
    main()
