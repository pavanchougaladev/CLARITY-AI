"""
Mission Audit Trail & FastAPI Cloud Vault View
==============================================
ISRO LunarFusion Co-Registration Suite — UI History View
"""

import os
import time
import tempfile
import requests
import streamlit as st
from typing import List, Dict, Any

try:
    from ui.config import BACKEND_API_URL
    from ui.services.api_client import (
        is_backend_online,
        list_backend_jobs,
        cancel_backend_job,
        delete_backend_job,
        fetch_backend_artifact,
        fetch_backend_job,
    )
except (ImportError, ValueError):
    from ..config import BACKEND_API_URL
    from ..services.api_client import (
        is_backend_online,
        list_backend_jobs,
        cancel_backend_job,
        delete_backend_job,
        fetch_backend_artifact,
        fetch_backend_job,
    )
from src.history import HistoryManager
from src.reporting.pdf_report import generate_pdf_report


@st.cache_data(ttl=30, max_entries=1)
def get_cached_history(storage_file: str = "registration_history.json") -> List[Dict[str, Any]]:
    """Cached loader for registration audit trail to prevent per-render disk I/O."""
    if os.path.exists(storage_file):
        try:
            import json
            with open(storage_file, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def render_history_tab(history_mgr: HistoryManager) -> None:
    """Render the Mission Audit Trail & FastAPI Cloud Job Vault interface."""
    hist_tab_local, hist_tab_cloud = st.tabs([
        "🖥️ Local Session Run Ledger",
        "📡 FastAPI Cloud Job Vault"
    ])

    with hist_tab_local:
        with st.container(border=True):
            st.markdown("""
            <div class="workflow-step-header">
                <span class="step-number">01</span>
                <div>
                    <div class="step-title">Persistent Registration Audit Trail</div>
                    <div style="font-size: 0.82rem; color: #CBD5E1;">Historical log of run telemetry, sub-pixel RMSE metrics, and algorithm performance benchmarks.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            hist = get_cached_history()
            if not hist:
                st.info("No registration history recorded yet. Run co-registration on the first tab!")
            else:
                df_data = []
                for h in reversed(hist):
                    m = h.get("metrics", {})
                    grade_val = m.get("grade", "N/A")
                    grade_icon = {"A": "🟢 Grade A", "B": "🟢 Grade B", "C": "🟡 Grade C", "D": "🟠 Grade D", "F": "🔴 Grade F"}.get(grade_val, grade_val)
                    rmse_val = m.get("rmse_pixels")
                    rmse_str = f"{rmse_val:.3f}" if (rmse_val is not None and isinstance(rmse_val, (int, float))) else "N/A"
                    w_rmse_val = m.get("weighted_rmse")
                    w_rmse_str = f"{w_rmse_val:.3f}" if (w_rmse_val is not None and isinstance(w_rmse_val, (int, float))) else (rmse_str if rmse_str != "N/A" else "N/A")
                    inliers_val = m.get("inlier_count", 0)
                    total_matches_val = m.get("total_matches", 0)
                    retention_val = m.get("inlier_ratio")
                    retention_str = f"{retention_val:.1%}" if (retention_val is not None and isinstance(retention_val, (int, float))) else "N/A"
                    coverage_val = m.get("coverage")
                    coverage_str = f"{coverage_val:.1%}" if (coverage_val is not None and isinstance(coverage_val, (int, float))) else "N/A"
                    df_data.append({
                        "Timestamp": h.get("date", "N/A"),
                        "Quality": grade_icon,
                        "Source Image": h.get("src_basename", "N/A"),
                        "Reference Image": h.get("ref_basename", "N/A"),
                        "Matcher": h.get("method", "N/A"),
                        "RMSE (px)": rmse_str,
                        "Weighted RMSE": w_rmse_str,
                        "Inliers": inliers_val,
                        "Total Matches": total_matches_val,
                        "Retention": retention_str,
                        "Coverage": coverage_str,
                        "Max Dim": h.get("params", {}).get("max_dim", "N/A")
                    })

                st.dataframe(df_data, width="stretch")

                # Historical PDF Report Exporter
                with st.expander("📄 Export Scientific PDF Report for Any Past Run", expanded=False):
                    run_options = [f"Run #{i+1} — {h.get('date', 'N/A')} — {h.get('method', 'N/A')} (Grade {h.get('metrics', {}).get('grade', 'N/A')})" for i, h in enumerate(reversed(hist))]
                    selected_run_idx = st.selectbox("Select Past Registration Run to Generate PDF:", range(len(run_options)), format_func=lambda i: run_options[i])
                    if selected_run_idx is not None:
                        h_selected = list(reversed(hist))[selected_run_idx]
                        m_sel = h_selected.get("metrics", {})
                        hist_results = {
                            "method_used": h_selected.get("method", "N/A"),
                            "total_matches": m_sel.get("total_matches", 0) or 0,
                            "inlier_count": m_sel.get("inlier_count", 0) or 0,
                            "inlier_ratio": m_sel.get("inlier_ratio") or 0.0,
                            "rmse_pixels": m_sel.get("rmse_pixels") or 0.0,
                            "weighted_rmse": m_sel.get("weighted_rmse") or m_sel.get("rmse_pixels") or 0.0,
                            "coverage": m_sel.get("coverage") or 0.0,
                            "scale_src": 1.0,
                            "scale_ref": 1.0,
                            "quality": {
                                "grade": h_selected["metrics"].get("grade", "B"),
                                "passed": h_selected["metrics"].get("passed", True),
                                "confidence": 0.92,
                                "warnings": [],
                                "details": {}
                            }
                        }
                        tmp_h_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp_h_pdf.close()
                        if generate_pdf_report(hist_results, h_selected["src_basename"], h_selected["ref_basename"], tmp_h_pdf.name) and os.path.exists(tmp_h_pdf.name):
                            with open(tmp_h_pdf.name, "rb") as f:
                                h_pdf_bytes = f.read()
                            st.download_button(
                                f"📥 Download Scientific PDF Report for Run #{selected_run_idx+1}",
                                data=h_pdf_bytes,
                                file_name=f"ISRO_Historical_Report_Run_{selected_run_idx+1}.pdf",
                                mime="application/pdf",
                                icon=":material/picture_as_pdf:",
                                width="stretch"
                            )

                st.write("")
                with st.expander("📊 Multi-Algorithm Accuracy & Benchmark Visualizations", expanded=True):
                    if os.path.exists("plots/algorithm_accuracy_dashboard.png"):
                        st.image("plots/algorithm_accuracy_dashboard.png", caption="ISRO Lunar Feature Matching — Accuracy & Precision Benchmark Dashboard", width="stretch")
                    if os.path.exists("plots/empirical_history_accuracy.png"):
                        st.image("plots/empirical_history_accuracy.png", caption="Empirical User History Registration Performance", width="stretch")

                    if os.path.exists("plots/accuracy_benchmark_interactive.html"):
                        with open("plots/accuracy_benchmark_interactive.html", "r", encoding="utf-8") as f:
                            html_content = f.read()
                        st.download_button(
                            label="📥 Download Interactive HTML Benchmark Report",
                            data=html_content,
                            file_name="lunar_matcher_accuracy_benchmark.html",
                            mime="text/html",
                            icon=":material/download:",
                        )

                st.write("")
                with st.popover("Clear Historical Database", icon=":material/delete:"):
                    st.write("Are you sure you want to delete all historical registration logs?")
                    if st.button("Confirm Delete All", type="primary", icon=":material/delete_forever:"):
                        history_mgr.clear()
                        get_cached_history.clear()
                        st.rerun()

    with hist_tab_cloud:
        with st.container(border=True):
            st.markdown("""
            <div class="workflow-step-header">
                <span class="step-number">02</span>
                <div>
                    <div class="step-title">FastAPI Backend Cloud Job Vault</div>
                    <div style="font-size: 0.82rem; color: #CBD5E1;">Inspect, manage, and download certified telemetry and rasters from remote asynchronous execution workers.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            backend_online = is_backend_online()
            if not backend_online:
                st.warning("⚠️ FastAPI Backend service is currently offline or unreachable at the configured URL.")
                st.info(f"Configured backend URL: `{BACKEND_API_URL}`\n\nTo start the background worker service, launch:\n```bash\n/usr/bin/python3 run_backend.py --host 0.0.0.0 --port 8000\n```")
            else:
                try:
                    h_resp = requests.get(f"{BACKEND_API_URL}/api/v1/health", timeout=3.0).json()
                except Exception:
                    h_resp = {}

                c_h1, c_h2, c_h3, c_h4 = st.columns(4)
                c_h1.metric("Cloud Service Status", "🟢 Healthy" if h_resp.get("status") == "healthy" else "🟡 Standby")
                c_h2.metric("Active Cloud Jobs", h_resp.get("active_jobs", 0))
                c_h3.metric("Total Jobs in Vault", h_resp.get("total_jobs", 0))
                c_h4.metric("Worker Concurrency", h_resp.get("max_concurrent_jobs", 2))

                col_rf1, col_rf2 = st.columns([5, 1])
                with col_rf2:
                    if st.button("🔄 Refresh Vault", key="btn_refresh_vault", width="stretch"):
                        st.rerun()

                b_jobs = list_backend_jobs()
                if not b_jobs:
                    st.info("No cloud jobs registered in the FastAPI ledger yet. Submit a registration run from Tab 1 with 'Use FastAPI backend' enabled.")
                else:
                    sorted_bjobs = sorted(b_jobs, key=lambda x: x.get("submitted_at", 0), reverse=True)
                    cloud_df_data = []
                    for j in sorted_bjobs:
                        st_val = j.get("status", "unknown")
                        st_badge = {
                            "completed": "🟢 Completed",
                            "processing": "⚡ Processing",
                            "queued": "⏳ Queued",
                            "failed": "🔴 Failed",
                            "cancelled": "🛑 Cancelled"
                        }.get(st_val, st_val)
                        cloud_df_data.append({
                            "Job ID": j.get("job_id", ""),
                            "Status": st_badge,
                            "Matcher": (j.get("matcher") or "unknown").upper(),
                            "Source Image": j.get("source_filename", "N/A"),
                            "Reference Image": j.get("reference_filename", "N/A"),
                            "Elapsed (s)": f"{float(j.get('elapsed_seconds') or 0.0):.2f}",
                            "Submitted At": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(j.get("submitted_at", time.time()))) if j.get("submitted_at") else "N/A"
                        })
                    st.dataframe(cloud_df_data, width="stretch")

                    st.markdown("---")
                    st.markdown("#### 🛠️ Cloud Job Action & Artifact Center")

                    job_choices = [
                        f"{'🟢' if j.get('status') == 'completed' else '⚡' if j.get('status') == 'processing' else '⏳' if j.get('status') == 'queued' else '🔴'} "
                        f"{j.get('job_id', '')[:8]}... | {(j.get('matcher') or 'unknown').upper()} | {j.get('status', '').upper()}"
                        for j in sorted_bjobs
                    ]
                    sel_job_idx = st.selectbox(
                        "Select Job to Inspect or Download Artifacts:",
                        range(len(job_choices)),
                        format_func=lambda i: job_choices[i],
                        key="cloud_vault_selected_job"
                    )
                    if sel_job_idx is not None:
                        sel_job = sorted_bjobs[sel_job_idx]
                        sel_jid = sel_job.get("job_id")
                        sel_status = sel_job.get("status")

                        st.write(f"**Selected Job ID:** `{sel_jid}` | **Status:** `{sel_status.upper()}` | **Engine:** `{(sel_job.get('matcher') or 'unknown').upper()}`")

                        if sel_status == "completed":
                            try:
                                c_results = fetch_backend_job(sel_jid)
                                cm1, cm2, cm3, cm4 = st.columns(4)
                                cm1.metric("Verified Inliers", f"{c_results.get('inlier_count', 0):,}")
                                rmse_c = c_results.get("rmse_pixels")
                                cm2.metric("Sub-Pixel RMSE", f"{rmse_c:.3f} px" if rmse_c is not None else "N/A")
                                cm3.metric("Quality Grade", c_results.get("quality_grade", c_results.get("quality", {}).get("grade", "N/A")))
                                cm4.metric("Runtime", f"{float(c_results.get('backend_elapsed') or 0.0):.2f}s")

                                c_act1, c_act2 = st.columns([2, 3])
                                with c_act1:
                                    if st.button("📥 Load into Active Session", key=f"load_session_{sel_jid}", width="stretch", icon=":material/visibility:"):
                                        st.session_state["active_results"] = c_results
                                        st.success(f"Job {sel_jid[:8]} loaded into Active Session! Head to Tab 1 to inspect.")

                                st.write("**Download Cloud-Certified Artifacts:**")
                                art_c1, art_c2, art_c3, art_c4 = st.columns(4)
                                with art_c1:
                                    g_bytes = fetch_backend_artifact(sel_jid, "geotiff")
                                    if g_bytes:
                                        st.download_button("🗺️ Warped GeoTIFF", data=g_bytes, file_name=f"ISRO_{sel_jid[:8]}_warped.tif", mime="image/tiff", width="stretch")
                                    else:
                                        st.button("GeoTIFF N/A", disabled=True, width="stretch")
                                with art_c2:
                                    t_bytes = fetch_backend_artifact(sel_jid, "tiepoints")
                                    if t_bytes:
                                        st.download_button("📍 ASP Tie-Points", data=t_bytes, file_name=f"ISRO_{sel_jid[:8]}_tiepoints.csv", mime="text/csv", width="stretch")
                                    else:
                                        st.button("ASP Tie-Points N/A", disabled=True, width="stretch")
                                with art_c3:
                                    p_bytes = fetch_backend_artifact(sel_jid, "pdf")
                                    if p_bytes:
                                        st.download_button("📑 Certification PDF", data=p_bytes, file_name=f"ISRO_{sel_jid[:8]}_report.pdf", mime="application/pdf", width="stretch")
                                    else:
                                        st.button("PDF N/A", disabled=True, width="stretch")
                                with art_c4:
                                    j_bytes = fetch_backend_artifact(sel_jid, "telemetry")
                                    if j_bytes:
                                        st.download_button("📊 Telemetry JSON", data=j_bytes, file_name=f"ISRO_{sel_jid[:8]}_telemetry.json", mime="application/json", width="stretch")
                                    else:
                                        st.button("Telemetry N/A", disabled=True, width="stretch")
                            except Exception as ex:
                                st.warning(f"Could not load full job details: {ex}")

                        elif sel_status in ("processing", "queued"):
                            st.info(f"⚡ Job `{sel_jid}` is currently executing on the FastAPI worker.")
                            if st.button("🛑 Abort / Cancel Job", key=f"cancel_job_{sel_jid}", width="stretch", icon=":material/stop:"):
                                if cancel_backend_job(sel_jid):
                                    st.warning(f"Job {sel_jid[:8]} cancellation requested.")
                                    st.rerun()

                        st.write("")
                        if st.button("🗑️ Purge Job & Clean Scratch Cache", key=f"del_job_{sel_jid}", icon=":material/delete:"):
                            if delete_backend_job(sel_jid):
                                st.success(f"Job {sel_jid[:8]} purged successfully.")
                                st.rerun()
                            else:
                                st.error("Failed to purge job.")
