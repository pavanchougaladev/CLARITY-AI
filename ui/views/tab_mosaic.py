"""
Continuous Lunar Strip Mosaic Synthesizer View
==============================================
ISRO LunarFusion Co-Registration Suite — UI Mosaic View
"""

import os
import cv2
import tempfile
import traceback
import streamlit as st
from typing import List, Dict, Any

from src.mosaic import MosaicEngine
from src.pipeline import robust_read


def render_mosaic_tab(available_images: List[Dict[str, Any]]) -> None:
    """Render the Lunar Mosaic Assembly interface."""
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">01</span>
            <div>
                <div class="step-title">Continuous Lunar Strip Mosaic Synthesizer</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Stitch multiple overlapping lunar orbital strips into a continuous wide-area composite with feather blending.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if available_images:
            labels = [img["label"] for img in available_images]
            selected_mosaic_indices = st.multiselect(
                "Select 2 or more browse datasets to stitch into canvas:",
                range(len(labels)),
                default=[0, min(1, len(labels) - 1)] if len(labels) >= 2 else [0],
                format_func=lambda i: labels[i]
            )

            st.write("")
            if st.button("Build Composite Lunar Mosaic", type="primary", icon=":material/auto_awesome_mosaic:", width="stretch"):
                if len(selected_mosaic_indices) < 2:
                    st.warning("⚠️ Please select at least 2 overlapping images to generate a mosaic.")
                else:
                    m_prog = st.progress(0, text="Loading image frames into mosaic engine...")
                    try:
                        engine = MosaicEngine()
                        for idx, img_i in enumerate(selected_mosaic_indices):
                            path = available_images[img_i]["path"]
                            arr = robust_read(path)
                            if arr is not None:
                                engine.add_image(arr, name=available_images[img_i]["basename"])

                        m_prog.progress(0.5, text="Computing pairwise registrations and global canvas coordinates...")
                        mosaic_out, meta = engine.compute_mosaic(base_idx=0)
                        m_prog.progress(1.0, text="✅ Mosaic synthesis complete!")

                        st.subheader(f"Composite Lunar Mosaic ({meta.get('count', len(selected_mosaic_indices))} Strips Stitched)")
                        st.image(mosaic_out, caption=f"Continuous Mosaic Canvas ({mosaic_out.shape[1]} x {mosaic_out.shape[0]} px)", width="stretch")

                        tmp_m = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                        tmp_m.close()
                        cv2.imwrite(tmp_m.name, mosaic_out)
                        with open(tmp_m.name, "rb") as f:
                            mosaic_bytes = f.read()
                        st.download_button(
                            "Download Stitched Mosaic (TIFF)",
                            data=mosaic_bytes,
                            file_name="lunar_mosaic_composite.tif",
                            mime="image/tiff",
                            icon=":material/download:",
                            width="stretch"
                        )
                    except Exception as e:
                        m_prog.empty()
                        st.error(f"Mosaic creation error: {e}")
                        st.code(traceback.format_exc())
        else:
            st.info("No library images available.")
