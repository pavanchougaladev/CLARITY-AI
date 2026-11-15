"""
Triplet Cycle-Consistency Validation Endpoints
==============================================
ISRO LunarFusion Co-Registration Suite — Geometric Topology
"""

import numpy as np
from fastapi import APIRouter, HTTPException, Depends

from ..state import jobs
from ..dependencies import TripletValidationRequest, get_current_user
from src.evaluate.triplet_consistency import check_triplet_closure

router = APIRouter(tags=["Triplet Validation"])


@router.post("/api/v1/triplet-validate")
@router.post("/triplet-validate")
async def triplet_validate(req: TripletValidationRequest, user: dict = Depends(get_current_user)):
    """
    Validate multi-view cycle closure across 3 overlapping lunar frames (A -> B -> C -> A).
    Accepts either completed job IDs or explicit 3x3 homography matrices.
    """
    h_ab = req.H_AB
    h_bc = req.H_BC
    h_ca = req.H_CA

    # Look up from jobs registry if job IDs are provided
    if req.job_id_ab:
        job = jobs.get(req.job_id_ab)
        if not job or job.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Job {req.job_id_ab} not found or not completed.")
        h_ab = job.get("results", {}).get("H_fallback")
        if not h_ab:
            raise HTTPException(status_code=400, detail=f"Job {req.job_id_ab} did not produce a valid homography.")

    if req.job_id_bc:
        job = jobs.get(req.job_id_bc)
        if not job or job.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Job {req.job_id_bc} not found or not completed.")
        h_bc = job.get("results", {}).get("H_fallback")
        if not h_bc:
            raise HTTPException(status_code=400, detail=f"Job {req.job_id_bc} did not produce a valid homography.")

    if req.job_id_ca:
        job = jobs.get(req.job_id_ca)
        if not job or job.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Job {req.job_id_ca} not found or not completed.")
        h_ca = job.get("results", {}).get("H_fallback")
        if not h_ca:
            raise HTTPException(status_code=400, detail=f"Job {req.job_id_ca} did not produce a valid homography.")

    if h_ab is None or h_bc is None or h_ca is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either (job_id_ab, job_id_bc, job_id_ca) or explicit (H_AB, H_BC, H_CA) 3x3 matrices.",
        )

    try:
        report = check_triplet_closure(
            H_AB=np.array(h_ab, dtype=np.float64),
            H_BC=np.array(h_bc, dtype=np.float64),
            H_CA=np.array(h_ca, dtype=np.float64),
            image_shape=req.image_shape or (2048, 2048),
            tolerance_px=float(req.tolerance_px or 5.0),
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Triplet closure validation error: {e}")
