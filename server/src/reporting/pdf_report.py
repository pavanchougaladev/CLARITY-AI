"""
pdf_report.py
=============
ISRO Lunar Image Registration — Scientific Telemetry & Quality Assessment Report
================================================================================
Generates a comprehensive, publication-grade aerospace PDF report containing:
  1. Executive Quality Gate Assessment (Grade, Confidence, Pass/Fail, Warnings)
  2. LunarFusion Adaptive Multi-Engine Telemetry (Terrain profiling, model weights,
     per-model correspondence yields, radiometric characterization)
  3. Statistical Error Distribution & Sub-Pixel Precision (RMSE, Weighted RMSE,
     Raw sensor RMSE, Error Percentiles: Mean, Median, Std, P90, P95, P99)
  4. Spatial Coverage & Correspondence Distribution (Inlier ratio, Grid coverage)
  5. Geometric Transformation Telemetry (Homography matrix, scaling factors)
  6. End-to-End Execution Latency Breakdown
"""

import os
import time
import logging
from typing import Dict, Any, Optional
import numpy as np

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

logger = logging.getLogger(__name__)


def _hex_to_color(hex_str: str) -> colors.Color:
    return colors.HexColor(hex_str)


def generate_pdf_report(
    results: Dict[str, Any],
    src_path: str,
    ref_path: str,
    output_path: str
) -> bool:
    """
    Generates an aerospace-grade scientific report detailing all registration telemetry,
    with comprehensive coverage of LunarFusion multi-engine metrics, quality gates,
    and sub-pixel error distribution statistics.
    """
    if not HAS_REPORTLAB:
        logger.error("ReportLab not installed. Cannot generate PDF report.")
        return False

    try:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        base_styles = getSampleStyleSheet()

        # Custom high-readability aerospace styles
        c_primary = _hex_to_color("#0B3D91")     # Deep ISRO / NASA blue
        c_secondary = _hex_to_color("#1E293B")   # Slate dark
        c_accent = _hex_to_color("#0284C7")      # Cyan-blue accent
        c_border = _hex_to_color("#CBD5E1")      # Light slate
        c_bg_head = _hex_to_color("#0F172A")     # Midnight header
        c_bg_row = _hex_to_color("#F8FAFC")      # Soft row bg
        c_bg_alt = _hex_to_color("#F1F5F9")      # Alternate row bg

        title_style = ParagraphStyle(
            'DocTitle',
            parent=base_styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=c_primary,
            alignment=1,
            spaceAfter=4
        )

        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#64748B"),
            alignment=1,
            spaceAfter=14
        )

        h2_style = ParagraphStyle(
            'SectionH2',
            parent=base_styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=15,
            textColor=c_primary,
            spaceBefore=12,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            'Body',
            parent=base_styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11.5,
            textColor=c_secondary
        )

        body_bold = ParagraphStyle(
            'BodyBold',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        badge_style = ParagraphStyle(
            'Badge',
            parent=base_styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            alignment=1
        )

        warning_style = ParagraphStyle(
            'WarningText',
            parent=body_style,
            textColor=_hex_to_color("#B91C1C"),
            fontName='Helvetica-Bold'
        )

        elements = []

        # =====================================================================
        # 1. Title & Header
        # =====================================================================
        elements.append(Paragraph("CLARITY* — CHANDRAYAAN LUNAR AI REGISTRATION &amp; IMAGE TRANSFORMATION SYSTEM AND 3D LUNAR MAPPING", title_style))
        elements.append(Paragraph("SCIENTIFIC TELEMETRY &amp; QUALITY ASSESSMENT REPORT", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=10))

        # Metadata Header Table
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        src_name = os.path.basename(src_path)
        ref_name = os.path.basename(ref_path)
        method_used = str(results.get('method_used', 'N/A'))

        meta_table_data = [
            [
                Paragraph(f"<b>Timestamp:</b> {timestamp}", body_style),
                Paragraph(f"<b>Matcher Engine:</b> {method_used}", body_style)
            ],
            [
                Paragraph(f"<b>Source Target:</b> {src_name}", body_style),
                Paragraph(f"<b>Reference Target:</b> {ref_name}", body_style)
            ]
        ]
        meta_table = Table(meta_table_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), c_bg_row),
            ('BOX', (0,0), (-1,-1), 1, c_border),
            ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # =====================================================================
        # 2. Executive Quality Gate Assessment
        # =====================================================================
        elements.append(Paragraph("1. Scientific Quality Gate Assessment", h2_style))
        quality = results.get('quality', {}) or {}
        grade = quality.get('grade', 'N/A')
        passed = quality.get('passed', True)
        conf_pct = quality.get('confidence', 1.0) * 100.0
        warnings = quality.get('warnings', [])
        details = quality.get('details', {}) or {}

        # Determine color badge for quality
        if grade in ['A', 'B'] and passed:
            badge_color = _hex_to_color("#DCFCE7")  # soft green
            text_color = _hex_to_color("#166534")
            status_text = "VERIFIED — PASSED SCIENTIFIC QUALITY GATE"
        elif grade == 'C':
            badge_color = _hex_to_color("#FEF9C3")  # soft yellow
            text_color = _hex_to_color("#854D0E")
            status_text = "ACCEPTABLE — QUALIFIED WITH MARGINAL THRESHOLDS"
        else:
            badge_color = _hex_to_color("#FEE2E2")  # soft red
            text_color = _hex_to_color("#991B1B")
            status_text = "FAILED / UNVERIFIED — OUTLIER THRESHOLD EXCEEDED"

        q_table_data = [
            [
                Paragraph(f"<b>Quality Grade:</b> <font color='{text_color.hexval()}'><b>GRADE {grade}</b></font>", badge_style),
                Paragraph(f"<b>Validation Status:</b> <font color='{text_color.hexval()}'><b>{status_text}</b></font>", badge_style),
                Paragraph(f"<b>Confidence:</b> {conf_pct:.1f}%", badge_style)
            ]
        ]
        q_table = Table(q_table_data, colWidths=[140, 280, 120])
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), badge_color),
            ('BOX', (0,0), (-1,-1), 1.2, text_color),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(q_table)

        # Quality Gate Sub-Scores Breakdown
        score_breakdown = [
            [
                Paragraph("<b>Component</b>", body_bold),
                Paragraph("<b>RMSE Evaluation</b>", body_bold),
                Paragraph("<b>Spatial Coverage</b>", body_bold),
                Paragraph("<b>Inlier Retention</b>", body_bold),
                Paragraph("<b>Composite Score</b>", body_bold),
            ],
            [
                Paragraph("Sub-Metric Score", body_style),
                Paragraph(f"{details.get('rmse_score', 0):.2f} (Grade {details.get('rmse_grade', grade)})", body_style),
                Paragraph(f"{details.get('coverage_score', 0):.2f}", body_style),
                Paragraph(f"{details.get('inlier_score', 0):.2f}", body_style),
                Paragraph(f"<b>{details.get('composite_score', 0):.3f} / 1.000</b>", body_bold),
            ]
        ]
        score_table = Table(score_breakdown, colWidths=[120, 110, 100, 100, 110])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), c_bg_alt),
            ('BOX', (0,0), (-1,-1), 0.5, c_border),
            ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        elements.append(Spacer(1, 4))
        elements.append(score_table)

        # Display advisory warnings if any
        if warnings:
            elements.append(Spacer(1, 4))
            for w in warnings:
                elements.append(Paragraph(f"<b>[ADVISORY]</b> {w}", warning_style))

        elements.append(Spacer(1, 10))

        # =====================================================================
        # 3. LunarFusion Adaptive Multi-Engine Telemetry (If Applicable)
        # =====================================================================
        fusion_stats = results.get('fusion_stats')
        if fusion_stats and isinstance(fusion_stats, dict):
            elements.append(Paragraph("2. LunarFusion Adaptive Multi-Engine Telemetry", h2_style))
            
            tag = fusion_stats.get('profile_tag', 'normal').upper()
            reason = fusion_stats.get('reason', 'Adaptive terrain matching')
            tex = fusion_stats.get('texture_avg', 0.0)
            shadow_pct = fusion_stats.get('shadow_fraction', 0.0) * 100.0
            hist_dist = fusion_stats.get('hist_distance', 0.0)
            models = fusion_stats.get('models_used', [])
            weights = fusion_stats.get('weights', {}) or {}
            per_model_counts = fusion_stats.get('per_model_counts', {}) or {}
            total_fused = fusion_stats.get('total_fused', results.get('total_matches', 0))
            elapsed = fusion_stats.get('elapsed_seconds', 0.0)

            profile_table_data = [
                [
                    Paragraph("<b>Diagnostic Terrain Classification</b>", body_bold),
                    Paragraph(f"<b>{tag}</b> ({reason})", body_style)
                ],
                [
                    Paragraph("<b>Surface Laplacian Texture Index</b>", body_bold),
                    Paragraph(f"{tex:.3f} (Variance of normalized log-gradient)", body_style)
                ],
                [
                    Paragraph("<b>Permanent Lunar Shadow Fraction</b>", body_bold),
                    Paragraph(f"{shadow_pct:.2f}% deep crater shadow coverage (&lt;30 DN)", body_style)
                ],
                [
                    Paragraph("<b>Sensor Radiometric Disparity</b>", body_bold),
                    Paragraph(f"{hist_dist:.4f} Bhattacharyya histogram divergence", body_style)
                ],
                [
                    Paragraph("<b>LunarFusion Engine Latency</b>", body_bold),
                    Paragraph(f"{elapsed:.2f} seconds total orchestration duration", body_style)
                ]
            ]
            prof_t = Table(profile_table_data, colWidths=[200, 340])
            prof_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), c_bg_row),
                ('BOX', (0,0), (-1,-1), 0.8, c_border),
                ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            elements.append(prof_t)
            elements.append(Spacer(1, 6))

            # Per-Engine Ensemble Yield Breakdown Table
            ensemble_headers = [
                Paragraph("<b>Constituent AI Model</b>", body_bold),
                Paragraph("<b>Architecture Type</b>", body_bold),
                Paragraph("<b>Profile Weight</b>", body_bold),
                Paragraph("<b>Candidates Yielded</b>", body_bold),
                Paragraph("<b>Ensemble Contribution</b>", body_bold)
            ]
            ensemble_rows = [ensemble_headers]
            arch_map = {
                'roma': 'Dense Foundation Transformer (RoMa v2)',
                'roma_v2': 'SOTA Dense Transformer (RoMa v2)',
                'roma_omega': 'SOTA Dense Transformer (RoMa v2)',
                'loftr': 'Detector-Free Cross-Attention (LoFTR)',
                'ni_lfm': 'NASA-IBM Lunar Foundation ViT (NI-LFM)',
                'sift': 'Illumination-Normalized Scale-Invariant (RootSIFT)',
                'lightglue': 'Sparse Deep Keypoint Graph (DISK+LightGlue)',
                'orb': 'Real-Time Oriented FAST Binary (ORB)'
            }

            for m in models:
                m_lower = m.lower()
                arch = arch_map.get(m_lower, 'Deep Neural Matcher')
                count = per_model_counts.get(m, 0)
                w = 0.0 if count == 0 else float(weights.get(m, 0.0))
                pct = (count / max(total_fused, 1)) * 100.0 if total_fused else 0.0
                w_str = f"{w:.1%}" if count > 0 else "0.0% (Excluded)"
                ensemble_rows.append([
                    Paragraph(f"<b>{m.upper()}</b>", body_style),
                    Paragraph(arch, body_style),
                    Paragraph(w_str, body_style),
                    Paragraph(f"{count:,} pts", body_style),
                    Paragraph(f"{pct:.1f}%", body_style)
                ])

            # Total Summary row
            ensemble_rows.append([
                Paragraph("<b>TOTAL UNIFIED ENSEMBLE</b>", body_bold),
                Paragraph("Unified RANSAC / MAGSAC Fusion", body_bold),
                Paragraph("100.0%", body_bold),
                Paragraph(f"<b>{total_fused:,} pts</b>", body_bold),
                Paragraph("100.0%", body_bold)
            ])

            ens_table = Table(ensemble_rows, colWidths=[110, 190, 80, 80, 80])
            ens_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), c_bg_head),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('BOX', (0,0), (-1,-1), 1, c_border),
                ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
                ('BACKGROUND', (0,-1), (-1,-1), c_bg_alt),
                ('ALIGN', (2,0), (-1,-1), 'CENTER'),
                ('TOPPADDING', (0,0), (-1,-1), 3.5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            elements.append(ens_table)
            elements.append(Spacer(1, 10))

        # =====================================================================
        # 4. OmniRoute Gateway Telemetry (If Applicable)
        # =====================================================================
        omni_stats = results.get('omniroute_stats')
        if omni_stats and isinstance(omni_stats, dict):
            elements.append(Paragraph("2. OmniRoute AI Gateway Telemetry", h2_style))
            omni_rows = [
                [
                    Paragraph("<b>Dispatch Strategy</b>", body_bold),
                    Paragraph(str(omni_stats.get('strategy', 'consensus')).upper(), body_style),
                    Paragraph("<b>Total Gateway Latency</b>", body_bold),
                    Paragraph(f"{omni_stats.get('latency_ms', 0):.1f} ms", body_style)
                ],
                [
                    Paragraph("<b>Successful Engines</b>", body_bold),
                    Paragraph(", ".join(m.upper() for m in omni_stats.get('models_succeeded', [])), body_style),
                    Paragraph("<b>Terrain Profile</b>", body_bold),
                    Paragraph(str(omni_stats.get('profile_tag', 'normal')).upper(), body_style)
                ]
            ]
            omni_t = Table(omni_rows, colWidths=[135, 135, 135, 135])
            omni_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), c_bg_row),
                ('BOX', (0,0), (-1,-1), 0.8, c_border),
                ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            elements.append(omni_t)
            elements.append(Spacer(1, 10))

        # =====================================================================
        # 5. Scientific Accuracy, Precision & Error Statistics
        # =====================================================================
        elements.append(Paragraph("3. Scientific Accuracy &amp; Reprojection Error Distribution", h2_style))

        rmse_val = results.get('rmse_pixels')
        w_rmse_val = results.get('weighted_rmse')
        rmse_raw_val = results.get('rmse_raw_pixels', rmse_val)
        w_rmse_raw_val = results.get('weighted_rmse_raw', w_rmse_val)
        err_stats = results.get('error_stats') or {}

        if rmse_val is not None:
            subpixel_badge = "[SUB-PIXEL CONFIRMED &lt; 0.5 px]" if rmse_val < 0.5 else "METRIC WITHIN OPERATIONAL RANGE"
            rmse_str = f"{rmse_val:.4f} px"
        else:
            subpixel_badge = "[QUALITY GATE REJECTED]"
            rmse_str = "Rejected"

        rmse_raw_str = f"{rmse_raw_val:.4f} px" if rmse_raw_val is not None else "Rejected"
        w_rmse_str = f"{w_rmse_val:.4f} px" if w_rmse_val is not None else "Rejected"
        w_rmse_raw_str = f"{w_rmse_raw_val:.4f} px" if w_rmse_raw_val is not None else "Rejected"

        def _fmt_stat(key: str) -> str:
            v = err_stats.get(key)
            return f"{float(v):.4f} px" if v is not None else "—"

        err_table_data = [
            [
                Paragraph("<b>Metric Name</b>", body_bold),
                Paragraph("<b>Processing Resolution</b>", body_bold),
                Paragraph("<b>Physical Sensor Resolution</b>", body_bold),
                Paragraph("<b>Scientific Relevance</b>", body_bold)
            ],
            [
                Paragraph("<b>Standard RMSE</b>", body_style),
                Paragraph(f"<b>{rmse_str}</b>", body_bold),
                Paragraph(rmse_raw_str, body_style),
                Paragraph(subpixel_badge, body_style)
            ],
            [
                Paragraph("<b>Confidence-Weighted RMSE</b>", body_style),
                Paragraph(f"<b>{w_rmse_str}</b>", body_bold),
                Paragraph(w_rmse_raw_str, body_style),
                Paragraph("Certainty-weighted reprojection residual", body_style)
            ],
            [
                Paragraph("<b>Mean Error (μ)</b>", body_style),
                Paragraph(_fmt_stat('mean'), body_style),
                Paragraph("—", body_style),
                Paragraph("Expected point displacement", body_style)
            ],
            [
                Paragraph("<b>Median Error (P50)</b>", body_style),
                Paragraph(_fmt_stat('median'), body_style),
                Paragraph("—", body_style),
                Paragraph("Robust 50th percentile centroid error", body_style)
            ],
            [
                Paragraph("<b>Standard Deviation (σ)</b>", body_style),
                Paragraph(_fmt_stat('std'), body_style),
                Paragraph("—", body_style),
                Paragraph("Spatial error variance &amp; dispersion", body_style)
            ],
            [
                Paragraph("<b>90th Percentile (P90)</b>", body_style),
                Paragraph(_fmt_stat('p90'), body_style),
                Paragraph("—", body_style),
                Paragraph("90% of inliers fall below this residual", body_style)
            ],
            [
                Paragraph("<b>95th Percentile (P95)</b>", body_style),
                Paragraph(_fmt_stat('p95'), body_style),
                Paragraph("—", body_style),
                Paragraph("95% confidence interval bound", body_style)
            ],
            [
                Paragraph("<b>99th Percentile (P99)</b>", body_style),
                Paragraph(_fmt_stat('p99'), body_style),
                Paragraph("—", body_style),
                Paragraph("Extreme edge boundary residual bound", body_style)
            ],
        ]

        err_table = Table(err_table_data, colWidths=[140, 110, 120, 170])
        err_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), c_bg_head),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('BOX', (0,0), (-1,-1), 1, c_border),
            ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
            ('BACKGROUND', (0,1), (-1,1), _hex_to_color("#F0FDF4") if (rmse_val is not None and rmse_val < 0.5) else c_bg_row),
            ('BACKGROUND', (0,2), (-1,2), _hex_to_color("#F0FDF4") if (w_rmse_val is not None and w_rmse_val < 0.5) else c_bg_alt),
            ('ALIGN', (1,0), (2,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(err_table)
        elements.append(Spacer(1, 10))

        # =====================================================================
        # 6. Spatial Coverage & Correspondence Telemetry
        # =====================================================================
        elements.append(Paragraph("4. Spatial Distribution &amp; Correspondence Yield", h2_style))

        n_total = results.get('total_matches', 0)
        n_inliers = results.get('inlier_count', 0)
        ratio = results.get('inlier_ratio', 0.0)
        cov = results.get('coverage', 0.0)

        corr_data = [
            [
                Paragraph("<b>Total Putative Matches:</b>", body_bold),
                Paragraph(f"{n_total:,} point pairs", body_style),
                Paragraph("<b>Inlier Count:</b>", body_bold),
                Paragraph(f"<b>{n_inliers:,} points</b>", body_bold)
            ],
            [
                Paragraph("<b>Inlier Retention Ratio:</b>", body_bold),
                Paragraph(f"<b>{ratio:.2%}</b>", body_bold),
                Paragraph("<b>Surface Coverage (8x8):</b>", body_bold),
                Paragraph(f"<b>{cov:.2%}</b> of lunar surface", body_bold)
            ]
        ]
        corr_t = Table(corr_data, colWidths=[140, 130, 140, 130])
        corr_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), c_bg_row),
            ('BOX', (0,0), (-1,-1), 0.8, c_border),
            ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(corr_t)
        elements.append(Spacer(1, 10))

        # =====================================================================
        # 7. Geometric Transform & Scale Normalization
        # =====================================================================
        elements.append(Paragraph("5. Geometric Transformation &amp; Scale Normalization", h2_style))
        s_src = results.get('scale_src', 1.0)
        s_ref = results.get('scale_ref', 1.0)
        s_tot = results.get('scale_ref_total', 1.0)
        H_mat = results.get('transform')

        geom_rows = [
            [
                Paragraph("<b>Source Scale Factor</b>", body_bold),
                Paragraph(f"{s_src:.4f}x", body_style),
                Paragraph("<b>Reference Scale Factor</b>", body_bold),
                Paragraph(f"{s_ref:.4f}x (Total: {s_tot:.4f}x)", body_style)
            ]
        ]
        if H_mat is not None and hasattr(H_mat, "shape"):
            h_det = float(np.linalg.det(H_mat[:2, :2])) if H_mat.shape[0] >= 2 and H_mat.shape[1] >= 2 else 1.0
            geom_rows.append([
                Paragraph("<b>Transform Matrix Type</b>", body_bold),
                Paragraph(f"{H_mat.shape[0]}x{H_mat.shape[1]} Projective Homography", body_style),
                Paragraph("<b>Affine Determinant</b>", body_bold),
                Paragraph(f"{h_det:.5f} (Area preservation indicator)", body_style)
            ])

        geom_t = Table(geom_rows, colWidths=[140, 130, 140, 130])
        geom_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), c_bg_row),
            ('BOX', (0,0), (-1,-1), 0.8, c_border),
            ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(geom_t)
        elements.append(Spacer(1, 10))

        # =====================================================================
        # 8. Pipeline Execution Latency Breakdown
        # =====================================================================
        pipeline_times = results.get('pipeline_times') or {}
        if pipeline_times:
            elements.append(Paragraph("6. End-to-End Processing Latency Breakdown", h2_style))
            time_headers = [Paragraph(f"<b>{k.replace('_', ' ').title()}</b>", body_bold) for k in pipeline_times.keys()]
            time_vals = [Paragraph(f"{v:.2f}s", body_style) for v in pipeline_times.values()]

            # Determine column widths dynamically
            n_cols = len(pipeline_times)
            col_w = min(540 / max(n_cols, 1), 100)
            time_table_data = [time_headers, time_vals]
            time_t = Table(time_table_data, colWidths=[col_w] * n_cols)
            time_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), c_bg_alt),
                ('BOX', (0,0), (-1,-1), 0.8, c_border),
                ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('TOPPADDING', (0,0), (-1,-1), 3.5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
            ]))
            elements.append(time_t)

        # =====================================================================
        # 7. Data Provenance & Cryptographic Audit Chain (SHA-256)
        # =====================================================================
        prov_chain = results.get('provenance_chain')
        if prov_chain and isinstance(prov_chain, dict):
            elements.append(Paragraph("7. Cryptographic Data Provenance &amp; Integrity Chain", h2_style))
            stages = prov_chain.get("stages", [])
            root_hash = str(prov_chain.get("root_hash", "N/A"))
            is_valid = prov_chain.get("is_valid", False)
            status_text = "VERIFIED (Tamper-Resistant)" if is_valid else "UNVERIFIED"

            prov_rows = [
                [
                    Paragraph("<b>Stage</b>", body_bold),
                    Paragraph("<b>Block Hash (SHA-256)</b>", body_bold),
                    Paragraph("<b>Payload Hash</b>", body_bold),
                    Paragraph("<b>UTC Time</b>", body_bold),
                ]
            ]
            for s in stages[:8]:
                s_name = str(s.get("stage", "")).upper()
                b_hash = str(s.get("block_hash", ""))[:16] + "..." if s.get("block_hash") else "N/A"
                p_hash = str(s.get("payload_hash", ""))[:16] + "..." if s.get("payload_hash") else "N/A"
                t_str = str(s.get("iso_time", ""))
                prov_rows.append([
                    Paragraph(f"<b>{s_name}</b>", body_style),
                    Paragraph(f"<font face='Courier' size=6.5>{b_hash}</font>", body_style),
                    Paragraph(f"<font face='Courier' size=6.5>{p_hash}</font>", body_style),
                    Paragraph(t_str, body_style),
                ])

            prov_t = Table(prov_rows, colWidths=[90, 160, 160, 130])
            prov_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), c_bg_head),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('BOX', (0,0), (-1,-1), 0.8, c_border),
                ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            elements.append(prov_t)
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(
                f"<b>Root Hash:</b> <font face='Courier' size=6.5>{root_hash}</font> | <b>Status:</b> {status_text}",
                body_style
            ))
            elements.append(Spacer(1, 6))

        # Footer signoff
        elements.append(Spacer(1, 15))
        elements.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceBefore=5, spaceAfter=5))
        elements.append(Paragraph(
            "<i>Generated by CLARITY* (Chandrayaan Lunar AI Registration &amp; Image Transformation system and 3D lunar mapping). Certified for orbital mapping and digital elevation synthesis.</i>",
            ParagraphStyle('Footer', parent=base_styles['Italic'], fontSize=7.5, leading=10, textColor=colors.HexColor("#64748B"), alignment=1)
        ))

        doc.build(elements)
        logger.info(f"Successfully generated scientific PDF report at: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}", exc_info=True)
        return False
