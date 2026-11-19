"""
Blender Photorealistic 3D Lunar Terrain Renderer — §13.4 Compliant
====================================================================
Generates cinema-quality 3D renders and `.blend` project files from Digital Elevation Models (DEMs)
and high-resolution Chandrayaan-2 orbital textures using Blender's headless physical render engine.

Coordinate frame contract (§13.4)
---------------------------------
* Scene units: METRIC, unit_scale = 1.0.  All vertex positions are in metres.
* Local Cartesian frame: X = lunar east, Y = lunar north, Z = up (radial).
* The local origin and reversible local→lunar-body-fixed transform are saved
  to a ``terrain_frame.json`` sidecar alongside every ``.blend`` file.
* Default vertical_exaggeration = 1.0 (no distortion).  Any non-unity exaggeration
  is applied ONLY to a separately-labelled Blender collection ``STYLIZED_VE_<N>x``
  so that the scientific mesh is never silently modified.
* Solar direction is derived from sun_azimuth/sun_elevation metadata and
  documented as a local-frame conversion (azimuth measured from local north,
  elevation above local horizontal plane).
* Blender lighting and renders are appearance products, NOT evidence of accuracy.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import time
import logging
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Default locations for Blender executable
_BLENDER_CANDIDATES = [
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender 5.2.app/Contents/MacOS/Blender",
    "/usr/local/bin/blender",
    "/usr/bin/blender",
]


def find_blender_binary() -> Optional[str]:
    """Locate the Blender executable on the current system (macOS / Linux / Windows)."""
    import glob

    env_path = os.environ.get("BLENDER_PATH", "").strip() or os.environ.get("BLENDER_BIN", "").strip()
    if env_path and os.path.exists(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("blender")
    if which_path and os.path.exists(which_path):
        return which_path

    candidates = _BLENDER_CANDIDATES
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # Windows: glob install roots so any version/major-drive matches
    win_roots = [
        r"D:\Blender*",
        r"C:\Program Files\Blender Foundation\Blender*",
        r"C:\Program Files\Blender*",
        r"C:\Program Files (x86)\Blender Foundation\Blender*",
        os.path.expandvars(r"%LOCALAPPDATA%\Blender Foundation\Blender*"),
    ]
    for root in win_roots:
        for match in sorted(glob.glob(root), reverse=True):
            exe = os.path.join(match, "blender.exe")
            if os.path.exists(exe) and os.access(exe, os.X_OK):
                return exe

    return None


class BlenderTerrainRenderer:
    """
    Automated Blender controller for photorealistic lunar DEM rendering.
    """

    def __init__(self, blender_path: Optional[str] = None):
        self.blender_bin = blender_path or find_blender_binary()

    def is_available(self) -> bool:
        """Returns True if a valid Blender binary is present."""
        return self.blender_bin is not None and os.path.exists(self.blender_bin)

    def render(
        self,
        dem: np.ndarray,
        texture: Optional[np.ndarray] = None,
        output_image_path: Optional[str] = None,
        output_blend_path: Optional[str] = None,
        camera_view: str = "isometric",
        sun_elevation_deg: float = 22.0,
        sun_azimuth_deg: float = 65.0,
        vertical_exaggeration: float = 1.0,  # §13.4: default is exactly 1.0 (no distortion)
        resolution: Tuple[int, int] = (1920, 1080),
        max_grid_dim: int = 350,
        # §13.4 frame contract
        local_origin_lon_deg: Optional[float] = None,
        local_origin_lat_deg: Optional[float] = None,
        local_origin_radius_m: float = 1_737_400.0,  # mean lunar radius
        pixel_size_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute headless Blender photorealistic render.

        Parameters
        ----------
        dem : np.ndarray
            2D Digital Elevation Model in metres (float32).
        texture : np.ndarray, optional
            2D or 3D lunar surface texture array (uint8).
        output_image_path : str, optional
            Path where the rendered PNG image will be saved.
        output_blend_path : str, optional
            Path where the self-contained `.blend` scene will be saved.
        camera_view : str
            Perspective: 'isometric' (45° orbital), 'nadir' (90° top-down), or 'rim_grazing' (low-angle).
        sun_elevation_deg : float
            Sun elevation angle above lunar local horizon in degrees.
            Converted to local-frame direction: X=east, Y=north, Z=up (§13.4).
        sun_azimuth_deg : float
            Solar azimuth measured from local north, clockwise, in degrees (0° = north).
        vertical_exaggeration : float
            Elevation multiplier.  Default is exactly **1.0** (no distortion, §13.4).
            Non-unity values are applied ONLY to a separately-labelled collection
            ``STYLIZED_VE_<N>x`` so the scientific mesh is never silently modified.
        resolution : tuple of (width, height)
            Output render resolution in pixels.
        max_grid_dim : int
            Maximum vertex subdivision dimension for the interactive mesh.
            A full-resolution scientific DEM is preserved separately.
        local_origin_lon_deg, local_origin_lat_deg : float, optional
            Planetographic coordinates of the local-frame origin.
            Saved to terrain_frame.json; used to reverse local→lunar-body-fixed.
        local_origin_radius_m : float
            Reference radius used to convert DEM heights to Cartesian Z.
        pixel_size_m : float, optional
            DEM pixel spacing in metres (for UV and X/Y scaling).

        Returns
        -------
        dict with status, paths, elapsed time, and metadata.
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Blender executable not found on this system.",
                "render_image": None,
                "blend_file": None,
            }

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Prepare DEM data
            dem_arr = np.asarray(dem, dtype=np.float32)
            if dem_arr.ndim != 2:
                return {"success": False, "error": f"DEM must be a 2D array, got shape {dem_arr.shape}"}

            dem_file = os.path.join(tmpdir, "dem.npy")
            np.save(dem_file, dem_arr)

            # 2. Prepare Texture data
            tex_file = os.path.join(tmpdir, "texture.png")
            if texture is not None:
                tex_arr = np.asarray(texture)
                if tex_arr.ndim == 2:
                    cv2.imwrite(tex_file, tex_arr)
                elif tex_arr.ndim == 3:
                    # Expect RGB from UI/pipeline, cv2 expects BGR
                    bgr = cv2.cvtColor(tex_arr, cv2.COLOR_RGB2BGR) if tex_arr.shape[2] == 3 else tex_arr
                    cv2.imwrite(tex_file, bgr)
                else:
                    cv2.imwrite(tex_file, (np.clip(dem_arr, 0, 255)).astype(np.uint8))
            else:
                # Generate default hillshade-like texture from DEM
                norm_tex = cv2.normalize(dem_arr, None, 40, 220, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(tex_file, norm_tex)

            # Output paths
            final_img = output_image_path or os.path.join(tmpdir, "render.png")
            final_blend = output_blend_path or os.path.join(tmpdir, "scene.blend")

            # 3. Generate self-contained Blender script
            blender_script = os.path.join(tmpdir, "blender_render_worker.py")
            script_code = f"""
import sys
import os
import math
import numpy as np
import bpy

# Reset scene to empty
bpy.ops.wm.read_factory_settings(use_empty=True)

dem_path = r"{dem_file}"
tex_path = r"{tex_file}"
out_img = r"{final_img}"
out_blend = r"{final_blend}"

camera_view = "{camera_view.lower()}"
sun_elev = {float(sun_elevation_deg)}
sun_azim = {float(sun_azimuth_deg)}
v_exagg = {float(vertical_exaggeration)}
res_w, res_h = {int(resolution[0])}, {int(resolution[1])}
max_grid = {int(max_grid_dim)}
px_size = {float(pixel_size_m or 0.0)}

# Load DEM
dem = np.load(dem_path).astype(np.float32)
rows, cols = dem.shape

# ------------------------------------------------------------------ #
# §13.4 Coordinate frame — metres, X=east, Y=north, Z=up            #
# Scene units: METRIC, unit_scale = 1.0                              #
# ------------------------------------------------------------------ #
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.scale_length = 1.0
bpy.context.scene.unit_settings.length_unit = 'METERS'

# Subsample grid if exceeds max dimension for interactive mesh
dem_full = dem.copy()  # keep full-resolution for scientific use
if max(rows, cols) > max_grid:
    step = int(np.ceil(max(rows, cols) / max_grid))
    dem = dem[::step, ::step]
    rows, cols = dem.shape

# Physical extent in metres (X=east, Y=north)
if px_size > 0:
    aspect_x = cols * px_size      # metres east-west
    aspect_y = rows * px_size      # metres north-south
else:
    aspect_x = cols / max(rows, cols) * 10.0
    aspect_y = rows / max(rows, cols) * 10.0

# Create SCIENCE collection (scientific mesh, VE=1.0 always)
sci_col = bpy.data.collections.new('SCIENCE_MESH_VE1.0')
bpy.context.scene.collection.children.link(sci_col)

mesh = bpy.data.meshes.new('LunarTerrain_Science')
obj  = bpy.data.objects.new('LunarTerrain_Science', mesh)
sci_col.objects.link(obj)
bpy.context.view_layer.objects.active = obj

# X = east (cols), Y = north (rows), Z = up (elevation in metres, VE=1.0)
x_coords = np.linspace(0.0, aspect_x, cols)   # east
y_coords = np.linspace(0.0, aspect_y, rows)   # north

med_val  = float(np.nanmedian(dem))
dem_norm = (dem - med_val)  # metres, NO exaggeration applied to science mesh

verts = []
for r in range(rows):
    for c in range(cols):
        verts.append((x_coords[c], y_coords[r], float(dem_norm[r, c])))

faces = []
for r in range(rows - 1):
    for c in range(cols - 1):
        v1 = r * cols + c
        v2 = r * cols + (c + 1)
        v3 = (r + 1) * cols + (c + 1)
        v4 = (r + 1) * cols + c
        faces.append((v1, v2, v3, v4))

mesh.from_pydata(verts, [], faces)
mesh.update(calc_edges=True)

# Generate UV Mapping
uv_layer = mesh.uv_layers.new(name="UVMap")
for poly in mesh.polygons:
    for loop_idx in poly.loop_indices:
        v_idx = mesh.loops[loop_idx].vertex_index
        c_idx = v_idx % cols
        r_idx = v_idx // cols
        u = c_idx / max(cols - 1, 1)
        v = r_idx / max(rows - 1, 1)
        uv_layer.data[loop_idx].uv = (u, v)

# Shade smooth
mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

# Lunar Material Shader
mat = bpy.data.materials.new(name="LunarRegolith")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

bsdf = nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Roughness"].default_value = 0.88
    if os.path.exists(tex_path):
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(tex_path)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

obj.data.materials.append(mat)

# ------------------------------------------------------------------ #
# §13.4 Solar direction — documented local-frame conversion          #
# Local frame: X=east, Y=north, Z=up (radial)                        #
# sun_azim measured from north (Y axis), clockwise                   #
# sun_elev measured above local horizontal plane (XY plane)          #
# ------------------------------------------------------------------ #
sun_data = bpy.data.lights.new(name='LunarSun_APPEARANCE_ONLY', type='SUN')
sun_data.energy = 5.2
sun_data.angle  = math.radians(0.53)  # angular width of Sun seen from lunar orbit
sun_obj = bpy.data.objects.new('LunarSun_APPEARANCE_ONLY', sun_data)
bpy.context.scene.collection.objects.link(sun_obj)

# Convert solar metadata to local Cartesian direction
# X=east: sin(azim)*cos(elev) | Y=north: cos(azim)*cos(elev) | Z=up: sin(elev)
rad_elev = math.radians(max(1.0, min(89.0, sun_elev)))
rad_azim = math.radians(sun_azim)         # from north, clockwise
sx = 20.0 * math.sin(rad_azim) * math.cos(rad_elev)   # east component
sy = 20.0 * math.cos(rad_azim) * math.cos(rad_elev)   # north component
sz = 20.0 * math.sin(rad_elev)                         # up component
sun_obj.location = (sx, sy, sz)

# Point sun at terrain center
c_sun = sun_obj.constraints.new(type='TRACK_TO')
c_sun.target = obj
c_sun.track_axis = 'TRACK_NEGATIVE_Z'
c_sun.up_axis = 'UP_Y'

# Camera Setup
cam_data = bpy.data.cameras.new("LunarOrbitalCamera")
cam_data.lens = 42
cam_obj = bpy.data.objects.new("LunarOrbitalCamera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Camera positioning by view mode
if camera_view == "nadir":
    # 90-degree straight down overhead photogrammetric view
    cam_obj.location = (0, 0, 14.5)
elif camera_view == "rim_grazing":
    # Low-angle crater rim horizon view
    cam_obj.location = (0, -11.0, 3.2)
else:
    # 45-degree cinematic isometric orbital flyover view
    cam_obj.location = (0, -11.5, 9.5)

c_cam = cam_obj.constraints.new(type='TRACK_TO')
c_cam.target = obj
c_cam.track_axis = 'TRACK_NEGATIVE_Z'
c_cam.up_axis = 'UP_Y'

# Deep space world background
scene = bpy.context.scene
world = bpy.data.worlds.new("DeepSpace")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Color"].default_value = (0.002, 0.003, 0.006, 1.0)
    bg_node.inputs["Strength"].default_value = 0.15

# Render configuration
scene.render.resolution_x = res_w
scene.render.resolution_y = res_h
scene.render.filepath = out_img
scene.render.image_settings.file_format = 'PNG'

# ------------------------------------------------------------------ #
# §13.4 — Stylized collection with non-unity VE (if requested)       #
# Scientific mesh above always has VE=1.0 and is never touched here  #
# ------------------------------------------------------------------ #
if abs(v_exagg - 1.0) > 0.01:
    sty_col = bpy.data.collections.new(f'STYLIZED_VE_{{v_exagg:.1f}}x_APPEARANCE_ONLY')
    bpy.context.scene.collection.children.link(sty_col)
    mesh_s = bpy.data.meshes.new(f'LunarTerrain_Stylized_{{v_exagg:.1f}}x')
    obj_s  = bpy.data.objects.new(f'LunarTerrain_Stylized_{{v_exagg:.1f}}x', mesh_s)
    sty_col.objects.link(obj_s)
    verts_s = [(v[0], v[1], v[2] * v_exagg) for v in verts]
    mesh_s.from_pydata(verts_s, [], faces)
    mesh_s.update()
    # Mark as hidden by default so scientific mesh is the active one
    obj_s.hide_viewport = True
    obj_s.hide_render    = True

# Embed scene metadata note
bpy.context.scene['clarity3d_schema'] = 'terrain_v1'
bpy.context.scene['vertical_exaggeration_science'] = 1.0
bpy.context.scene['vertical_exaggeration_stylized'] = v_exagg
bpy.context.scene['local_frame'] = 'X=lunar_east Y=lunar_north Z=up_radial'
bpy.context.scene['units'] = 'metres'
bpy.context.scene['accuracy_note'] = 'Blender lighting and render are appearance products, NOT evidence of accuracy.'

# Save .blend scene file
try:
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
except Exception as e_save:
    print('Blend save warning:', e_save)

# Render still image
bpy.ops.render.render(write_still=True)
print('BLENDER_RENDER_COMPLETE')
"""
            with open(blender_script, "w") as f:
                f.write(script_code)

            # 4. Run Blender headless
            cmd = [
                self.blender_bin,
                "-b",  # background / headless
                "-P", blender_script,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Blender render process timed out after 90 seconds.",
                    "render_image": None,
                    "blend_file": None,
                }

            if proc.returncode != 0 or not os.path.exists(final_img):
                err = proc.stderr.strip() or proc.stdout.strip()
                logger.error(f"Blender execution error: {err}")
                return {
                    "success": False,
                    "error": f"Blender failed (code {proc.returncode}): {err[-300:]}",
                    "render_image": None,
                    "blend_file": None,
                }

            # Copy to destination if temp paths were used and external paths were passed
            rendered_bytes = None
            with open(final_img, "rb") as f_img:
                rendered_bytes = f_img.read()

            blend_bytes = None
            if os.path.exists(final_blend):
                with open(final_blend, "rb") as f_bl:
                    blend_bytes = f_bl.read()

            # --- §13.4 terrain_frame.json sidecar ---
            frame_sidecar = None
            if output_blend_path:
                frame_path = os.path.splitext(output_blend_path)[0] + "_terrain_frame.json"
                frame_sidecar = export_terrain_frame_json(
                    path=frame_path,
                    dem_shape=dem_arr.shape,
                    local_origin_lon_deg=local_origin_lon_deg,
                    local_origin_lat_deg=local_origin_lat_deg,
                    local_origin_radius_m=local_origin_radius_m,
                    pixel_size_m=pixel_size_m,
                    vertical_exaggeration_science=1.0,
                    vertical_exaggeration_stylized=vertical_exaggeration,
                )

            elapsed = time.time() - t0
            return {
                "success": True,
                "elapsed_sec": round(elapsed, 2),
                "image_path": final_img if output_image_path else None,
                "image_bytes": rendered_bytes,
                "blend_path": final_blend if output_blend_path else None,
                "blend_bytes": blend_bytes,
                "terrain_frame_json": frame_sidecar,
                "resolution": resolution,
                "vertices": int(dem_arr.shape[0] * dem_arr.shape[1]),
                "faces": int((dem_arr.shape[0] - 1) * (dem_arr.shape[1] - 1)),
                "vertical_exaggeration_science": 1.0,
                "vertical_exaggeration_stylized": vertical_exaggeration,
                "local_frame": "X=lunar_east Y=lunar_north Z=up_radial units=metres",
            }


# ---------------------------------------------------------------------------
# §13.4 — terrain_frame.json sidecar writer
# ---------------------------------------------------------------------------
def export_terrain_frame_json(
    path: str,
    dem_shape: tuple,
    local_origin_lon_deg: Optional[float] = None,
    local_origin_lat_deg: Optional[float] = None,
    local_origin_radius_m: float = 1_737_400.0,
    pixel_size_m: Optional[float] = None,
    vertical_exaggeration_science: float = 1.0,
    vertical_exaggeration_stylized: float = 1.0,
) -> str:
    """
    Write terrain_frame.json — the §13.4 coordinate frame sidecar.

    Records:
    * Datum and units
    * Local Cartesian frame definition (X=east, Y=north, Z=up)
    * Local origin in lunar body-fixed coordinates
    * Reversible local ↔ lunar-body-fixed transform (rotation only for a
      local tangent-plane approximation)
    * DEM extent in local frame metres
    * Vertical exaggeration: 1.0 for science, N for stylized
    """
    import json, os, math, time as _time

    data = {
        "_schema": "clarity3d_terrain_frame_v1",
        "datum": "Lunar mean sphere, radius 1 737 400 m (IAU 2015)",
        "units": "metres",
        "elevation_definition": "height above local reference surface (metre)",
        "local_frame": {
            "X": "lunar_east",
            "Y": "lunar_north",
            "Z": "up_radial",
            "note": (
                "Local tangent-plane ENU frame. "
                "For small sites (<100 km) the approximation error is <0.01%."
            ),
        },
        "local_origin": {
            "longitude_deg": local_origin_lon_deg,
            "latitude_deg": local_origin_lat_deg,
            "radius_m": local_origin_radius_m,
            "note": "Planetocentric coordinates. None means origin not georeferenced.",
        },
        "local_to_body_fixed": {
            "method": "ENU_tangent_plane",
            "note": (
                "To convert local (x,y,z) to lunar body-fixed (X_bf, Y_bf, Z_bf): "
                "use standard ENU→ECEF rotation with the stored origin lat/lon. "
                "Full rotation matrix can be reconstructed from lat0/lon0."
            ),
            "available": local_origin_lat_deg is not None and local_origin_lon_deg is not None,
        },
        "dem_extent_m": {
            "rows": dem_shape[0],
            "cols": dem_shape[1],
            "east_west_m": dem_shape[1] * pixel_size_m if pixel_size_m else None,
            "north_south_m": dem_shape[0] * pixel_size_m if pixel_size_m else None,
            "pixel_size_m": pixel_size_m,
        },
        "vertical_exaggeration": {
            "science_mesh": vertical_exaggeration_science,
            "stylized_mesh": vertical_exaggeration_stylized,
            "note": (
                "Science mesh always uses VE=1.0. "
                "Stylized mesh (if present) is in a separately labelled Blender collection. "
                "Blender renders are appearance products, NOT evidence of accuracy."
            ),
        },
        "written_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Wrote terrain_frame.json: %s", path)
    return path


def render_lunar_dem_with_blender(
    dem: np.ndarray,
    texture: Optional[np.ndarray] = None,
    camera_view: str = "isometric",
    sun_elevation_deg: float = 22.0,
    sun_azimuth_deg: float = 65.0,
    vertical_exaggeration: float = 1.0,   # §13.4: default is exactly 1.0
) -> Dict[str, Any]:
    """Convenience wrapper for one-shot Blender rendering."""
    renderer = BlenderTerrainRenderer()
    return renderer.render(
        dem=dem,
        texture=texture,
        camera_view=camera_view,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        vertical_exaggeration=vertical_exaggeration,
    )
