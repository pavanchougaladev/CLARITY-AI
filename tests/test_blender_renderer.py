import os
import pytest
import numpy as np
from src.rendering.blender_renderer import BlenderTerrainRenderer, find_blender_binary

def test_find_blender_binary():
    bin_path = find_blender_binary()
    assert bin_path is not None
    assert os.path.exists(bin_path)


def test_blender_renderer_availability():
    renderer = BlenderTerrainRenderer()
    assert renderer.is_available() is True


def test_blender_render_synthetic_dem(tmp_path):
    renderer = BlenderTerrainRenderer()
    # Create small synthetic lunar DEM
    y, x = np.ogrid[-20:20, -20:20]
    dem = (40.0 - np.sqrt(x**2 + y**2) * 1.5).astype(np.float32)
    texture = np.full(dem.shape, 128, dtype=np.uint8)

    out_png = str(tmp_path / "test_render.png")
    out_blend = str(tmp_path / "test_scene.blend")

    res = renderer.render(
        dem=dem,
        texture=texture,
        output_image_path=out_png,
        output_blend_path=out_blend,
        camera_view="isometric",
        sun_elevation_deg=25.0,
        sun_azimuth_deg=45.0,
        resolution=(640, 360),
        max_grid_dim=100
    )

    assert res["success"] is True
    assert os.path.exists(out_png)
    assert os.path.exists(out_blend)
    assert res["image_bytes"] is not None and len(res["image_bytes"]) > 1000
    assert res["blend_bytes"] is not None and len(res["blend_bytes"]) > 1000


def test_generate_lunar_normal_map():
    import sys
    sys.path.insert(0, os.path.abspath("client"))
    import services

    z = np.full((32, 32), 10.0, dtype=np.float32)
    # Add a spherical crater depression
    y, x = np.ogrid[-16:16, -16:16]
    r = np.hypot(x, y)
    z[r < 12] -= (12 - r[r < 12]) * 0.8
    tex = np.full((32, 32), 140, dtype=np.uint8)

    norm_map = services.generate_lunar_normal_map(z, tex)
    assert norm_map.shape == (32, 32, 3)
    assert norm_map.dtype == np.uint8
    # Center flat terrain normal should be close to (127, 127, 255) pointing up (+Z)
    assert norm_map[0, 0, 2] > 200


def test_services_run_blender_generation_pillars():
    import sys
    sys.path.insert(0, os.path.abspath("client"))
    import services

    z = np.full((32, 32), 20.0, dtype=np.float32)
    tex = np.full((32, 32), 160, dtype=np.uint8)

    # Test with Nadir view, Hypsometric shader, and SPICE sun
    res = services.run_blender_generation(
        z=z,
        tex=tex,
        camera_view="nadir",
        render_engine="eevee",
        sun_elevation_deg=30.0,
        sun_azimuth_deg=90.0,
        vertical_exaggeration=1.0,
        shader_mode="hypsometric",
    )
    assert res["success"] is True
    assert os.path.exists(res["blend_path"])
    assert res["render_path"] is not None and os.path.exists(res["render_path"])
    assert os.path.exists(res["norm_path"])

