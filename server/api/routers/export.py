"""
Blender 3D Export Endpoints
===========================
ISRO LunarFusion Co-Registration Suite — 3D Visualization
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Export"])


@router.post("/api/v1/export/blender")
async def export_to_blender(ply_path: Optional[str] = None, background: bool = False):
    """
    Launch or compile a 3D Blender project file (.blend) for the LunarFusion terrain.
    """
    try:
        from scripts.launch_blender import launch_blender, find_blender_binary

        bin_path = find_blender_binary()
        if not bin_path:
            raise HTTPException(status_code=503, detail="Blender binary not found on server host.")
        proc = launch_blender(ply_path=ply_path, background=background)
        blend_file = os.path.abspath("data/outputs/lunar_terrain.blend")
        return {
            "status": "launched" if not background else "compiled",
            "blender_binary": bin_path,
            "blend_file": blend_file,
            "pid": proc.pid if proc else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
