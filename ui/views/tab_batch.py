"""
Batch Ingestion and Sequential Co-Registration View
===================================================
ISRO LunarFusion Co-Registration Suite — UI Batch View
"""

import os
import streamlit as st
import pandas as pd
from typing import Optional

try:
    from ui.config import DATA_DIR, WORKSPACE_ROOT
except (ImportError, ValueError):
    from ..config import DATA_DIR, WORKSPACE_ROOT
from src.batch import run_batch, discover_image_pairs


def render_batch_tab() -> None:
    """Render the Batch Directory Ingestion and sequential registration interface."""
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">01</span>
            <div>
                <div class="step-title">Batch Directory Ingestion</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Automated discovery and sequential sub-pixel registration across directory paths.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        b_col1, b_col2 = st.columns(2, gap="medium")
        batch_src_dir = b_col1.text_input("Source Imagery Directory", value=os.path.join(DATA_DIR))
        batch_ref_dir = b_col2.text_input("Reference Imagery Directory", value=os.path.join(DATA_DIR))

        st.write("")
        scan_btn = st.button("Scan & Discover Candidate Pairs", type="secondary", icon=":material/search:", width="stretch")
        if scan_btn:
            if os.path.exists(batch_src_dir) and os.path.exists(batch_ref_dir):
                pairs = discover_image_pairs(batch_src_dir, batch_ref_dir)
                st.session_state["batch_pairs"] = pairs
                st.success(f"Discovered {len(pairs)} candidate orbital pairs for batch execution.")
                if pairs:
                    st.dataframe(
                        [{"Pair ID": p["pair_name"], "Source Image": os.path.basename(p["src_path"]), "Reference Image": os.path.basename(p["ref_path"])} for p in pairs],
                        width="stretch"
                    )
            else:
                st.error("Specified directory path does not exist on disk.")

    if "batch_pairs" in st.session_state and len(st.session_state["batch_pairs"]) > 0:
        with st.container(border=True):
            st.markdown("""
            <div class="workflow-step-header">
                <span class="step-number">02</span>
                <div>
                    <div class="step-title">Batch Processing Execution</div>
                    <div style="font-size: 0.82rem; color: #CBD5E1;">Process all discovered pairs with automated outlier rejection and reporting.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Start Batch Registration Run", type="primary", icon=":material/rocket_launch:", width="stretch"):
                pairs = st.session_state["batch_pairs"]
                prog = st.progress(0, text="Initializing batch queue...")

                def batch_prog(idx, total, name, status):
                    prog.progress(idx / total, text=f"Processing [{idx}/{total}]: {name} — {status}")

                out_dir = os.path.join(WORKSPACE_ROOT, "outputs", "batch_run")
                try:
                    batch_res = run_batch(pairs, output_dir=out_dir, progress_callback=batch_prog)
                    prog.progress(1.0, text="✅ Batch Processing Run Completed!")

                    st.subheader("Batch Summary Statistics")
                    st.json(batch_res.get("summary", {}))

                    if batch_res.get("results"):
                        st.subheader("Results Table")
                        df_results = pd.DataFrame(batch_res["results"])
                        st.dataframe(df_results, width="stretch")

                        csv = df_results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Batch Results (CSV)",
                            data=csv,
                            file_name="batch_registration_results.csv",
                            mime="text/csv",
                            icon=":material/download:",
                            width="stretch"
                        )

                    if batch_res.get("failures"):
                        st.subheader("⚠️ Failed Pairs")
                        df_fails = pd.DataFrame(batch_res["failures"])
                        st.dataframe(df_fails, width="stretch")

                        csv_fails = df_fails.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Failures Log (CSV)",
                            data=csv_fails,
                            file_name="batch_failures_log.csv",
                            mime="text/csv",
                            icon=":material/download:",
                            width="stretch"
                        )
                except Exception as exc:
                    st.error(f"Batch processing encountered an error: {exc}")
