"""
Backend REST API Client Service
================================
ISRO LunarFusion Co-Registration Suite — UI API Client
Handles async submission, polling, artifact retrieval, and job lifecycle.
"""

import os
import time
import requests
import numpy as np
from typing import Optional, Any, Dict, List
import streamlit as st

try:
    from ui.config import BACKEND_API_URL, DEFAULT_TIMEOUT_SECONDS
except (ImportError, ValueError):
    from ..config import BACKEND_API_URL, DEFAULT_TIMEOUT_SECONDS


def is_backend_online(backend_url: Optional[str] = None) -> bool:
    """Check whether the FastAPI backend server is accessible."""
    base = backend_url or BACKEND_API_URL
    urls_to_try = [base]
    if "127.0.0.1" in base:
        urls_to_try.append(base.replace("127.0.0.1", "localhost"))
    elif "localhost" in base:
        urls_to_try.append(base.replace("localhost", "127.0.0.1"))

    for u in urls_to_try:
        try:
            resp = requests.get(f"{u}/api/v1/health", timeout=2.0)
            if resp.ok:
                return True
        except Exception:
            continue
    return False


def format_backend_results(results: dict) -> dict:
    """Format inliers and arrays from backend JSON payload, ensuring bidirectional schema compatibility."""
    if not results:
        raise RuntimeError("Backend completed without results payload")
    if "inliers_src" in results and results["inliers_src"] is not None:
        results["inliers_src"] = np.asarray(results["inliers_src"], dtype=np.float32)
    if "inliers_ref" in results and results["inliers_ref"] is not None:
        results["inliers_ref"] = np.asarray(results["inliers_ref"], dtype=np.float32)
    if "confidences" in results and results["confidences"] is not None:
        results["confidences"] = np.asarray(results["confidences"], dtype=np.float32)
    if "H_fallback" in results and results["H_fallback"] is not None:
        results["H_fallback"] = np.asarray(results["H_fallback"], dtype=np.float32)

    # Standardize field names across backend telemetry and UI inspectors
    n_inliers = len(results.get("inliers_src", [])) if results.get("inliers_src") is not None else results.get("inlier_count", 0)
    results.setdefault("inliers", n_inliers)
    results.setdefault("inlier_count", n_inliers)
    results.setdefault("rmse", results.get("rmse_pixels"))
    results.setdefault("rmse_pixels", results.get("rmse"))
    results.setdefault("method_used", results.get("method", "FastAPI Backend"))
    results.setdefault("method", results.get("method_used"))
    return results


def list_backend_jobs() -> List[dict]:
    """Fetch all active, completed, and historical jobs from FastAPI backend."""
    try:
        resp = requests.get(f"{BACKEND_API_URL}/api/v1/jobs", timeout=4.0)
        if resp.ok:
            return resp.json().get("jobs", [])
    except Exception:
        pass
    return []


def cancel_backend_job(job_id: str) -> bool:
    """Send cooperative abort signal to the FastAPI backend for a job."""
    try:
        resp = requests.post(f"{BACKEND_API_URL}/api/v1/jobs/{job_id}/cancel", timeout=4.0)
        return resp.ok
    except Exception:
        return False


def delete_backend_job(job_id: str) -> bool:
    """Delete a backend job and clean up its temporary scratch directory."""
    try:
        resp = requests.delete(f"{BACKEND_API_URL}/api/v1/jobs/{job_id}", timeout=4.0)
        return resp.ok
    except Exception:
        return False


def fetch_backend_artifact(job_id: str, artifact_type: str) -> Optional[bytes]:
    """Download artifact raw bytes from FastAPI backend."""
    try:
        resp = requests.get(f"{BACKEND_API_URL}/api/v1/jobs/{job_id}/artifacts/{artifact_type}", timeout=15.0)
        if resp.ok:
            return resp.content
    except Exception:
        pass
    return None


def trigger_backend_blender_export(ply_path: Optional[str] = None, background: bool = False) -> dict:
    """Trigger Blender export/render on the FastAPI backend."""
    try:
        params: Dict[str, Any] = {"background": background}
        if ply_path:
            params["ply_path"] = ply_path
        resp = requests.post(f"{BACKEND_API_URL}/api/v1/export/blender", params=params, timeout=10.0)
        if resp.ok:
            return resp.json()
        return {"status": "error", "detail": resp.text}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def fetch_backend_job(job_id: str) -> dict:
    """Query FastAPI backend for the results and downloadable artifacts of a specific job ID."""
    status_url = f"{BACKEND_API_URL}/api/v1/jobs/{job_id}"
    resp = requests.get(status_url, timeout=5.0)
    if not resp.ok:
        raise RuntimeError(f"Could not retrieve job {job_id} (HTTP {resp.status_code}): {resp.text}")
    data = resp.json()
    state = data.get("status")
    if state == "completed":
        results = format_backend_results(data.get("results"))
        results["backend_job_id"] = job_id
        results["backend_artifacts"] = data.get("artifacts", [])
        results["backend_elapsed"] = data.get("elapsed_seconds", 0.0)
        return results
    elif state == "failed":
        raise RuntimeError(data.get("error") or "Backend job failed")
    elif state in ("processing", "queued"):
        elapsed = int(time.time() - data.get("start_time", data.get("submitted_at", time.time())))
        raise RuntimeError(f"Backend job {job_id} is still {state.upper()} (Elapsed: {elapsed}s). Please wait a moment and click Fetch again.")
    else:
        raise RuntimeError(f"Unknown job state: {state}")


def submit_registration_to_backend(
    source_path: str,
    reference_path: str,
    matcher: str,
    max_dim: int = 2048,
    use_tps: bool = True,
    speed_mode: str = "balanced",
    status_placeholder: Optional[Any] = None,
    progress_bar: Optional[Any] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    pipeline_mode: Optional[str] = None,
    sensor_pair: Optional[str] = None,
    region_type: Optional[str] = None,
) -> dict:
    """Submit a registration job to the FastAPI backend and wait for completion with live status and progress."""
    if not os.path.exists(source_path) or not os.path.exists(reference_path):
        raise FileNotFoundError("Both input images must exist on disk before submitting to the backend.")

    response = None
    post_data = {
        "source_path": os.path.abspath(source_path),
        "reference_path": os.path.abspath(reference_path),
        "matcher": matcher,
        "max_dim": str(max_dim),
        "use_tps": str(use_tps).lower(),
        "speed_mode": speed_mode,
    }
    if pipeline_mode:
        post_data["pipeline_mode"] = pipeline_mode
    if sensor_pair:
        post_data["sensor_pair"] = sensor_pair
    if region_type:
        post_data["region_type"] = region_type

    try:
        response = requests.post(
            f"{BACKEND_API_URL}/api/v1/register",
            data=post_data,
            params={"matcher": matcher},
            timeout=15,
        )
    except Exception:
        response = None

    # Fallback to streaming multipart upload if form paths were rejected
    if response is None or response.status_code != 202:
        with open(source_path, "rb") as source_file, open(reference_path, "rb") as reference_file:
            multipart_data = {
                "matcher": matcher,
                "max_dim": str(max_dim),
                "use_tps": str(use_tps).lower(),
                "speed_mode": speed_mode,
            }
            if pipeline_mode:
                multipart_data["pipeline_mode"] = pipeline_mode
            if sensor_pair:
                multipart_data["sensor_pair"] = sensor_pair
            if region_type:
                multipart_data["region_type"] = region_type

            response = requests.post(
                f"{BACKEND_API_URL}/api/v1/register",
                files={
                    "source_image": (os.path.basename(source_path), source_file, "application/octet-stream"),
                    "reference_image": (os.path.basename(reference_path), reference_file, "application/octet-stream"),
                },
                data=multipart_data,
                params={"matcher": matcher},
                timeout=60,
            )

    if response.status_code != 202:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"Backend registration rejected: {detail}")

    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError("Backend registration response missing job_id")

    try:
        st.session_state["last_backend_job_id"] = job_id
    except Exception:
        pass

    status_url = f"{BACKEND_API_URL}/api/v1/jobs/{job_id}"
    last_status = None
    t_start = time.time()

    while time.time() - t_start < timeout_seconds:
        elapsed = int(time.time() - t_start)
        try:
            poll = requests.get(status_url, timeout=4.0)
            if poll.status_code == 404:
                raise RuntimeError(f"Backend job `{job_id}` was not found on the server (the service may have been restarted). Please resubmit the registration.")
            if poll.ok:
                last_status = poll.json()
                state = last_status.get("status")
                prog = last_status.get("progress") or {}
                p_step = prog.get("step", 0)
                p_total = max(prog.get("total", 8), 1)
                p_msg = prog.get("message", "Processing in FastAPI worker...")
                pct = min(1.0, max(0.0, p_step / p_total))

                if progress_bar:
                    progress_bar.progress(pct, text=f"⚡ [{p_step}/{p_total}] {p_msg} ({elapsed}s)")
                elif status_placeholder:
                    status_placeholder.info(
                        f"⏳ **FastAPI Backend Processing...** [{p_step}/{p_total}] {p_msg} | Job ID: `{job_id}` | Elapsed: **{elapsed}s**"
                    )

                if state == "completed":
                    results = format_backend_results(last_status.get("results"))
                    results["backend_job_id"] = job_id
                    results["backend_artifacts"] = last_status.get("artifacts", [])
                    results["backend_elapsed"] = last_status.get("elapsed_seconds", round(time.time() - t_start, 2))
                    return results
                if state == "failed":
                    raise RuntimeError(last_status.get("error") or "Backend job failed")
                if state == "cancelled":
                    raise RuntimeError("Backend job was cancelled by operator")
        except requests.RequestException:
            pass
        time.sleep(1.0)

    last_state = last_status.get("status") if last_status else "unknown"
    raise TimeoutError(
        f"Backend job {job_id} did not finish within {timeout_seconds}s timeout window (Current state: {last_state.upper()}). "
        f"The backend is still processing. You can reconnect using Job ID: `{job_id}`."
    )


# Backward compatibility aliases
_is_backend_online = is_backend_online
_format_backend_results = format_backend_results
_list_backend_jobs = list_backend_jobs
_cancel_backend_job = cancel_backend_job
_delete_backend_job = delete_backend_job
_fetch_backend_artifact = fetch_backend_artifact
_trigger_backend_blender_export = trigger_backend_blender_export
_fetch_backend_job = fetch_backend_job
_submit_registration_to_backend = submit_registration_to_backend
