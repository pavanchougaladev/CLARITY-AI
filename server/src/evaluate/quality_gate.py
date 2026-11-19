"""
Registration Quality Gate — Automatic Failure Detection
=========================================================
Evaluates registration results against configurable thresholds
and assigns a quality grade (A/B/C/D/F) to prevent silent failures.

Addresses the problem where Run #8 returned 362.8 px RMSE
and was reported as "successful" with no warning.
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


MIN_REQUIRED_INLIERS = 15

# Default thresholds — can be overridden via config
DEFAULT_THRESHOLDS = {
    'rmse_excellent': 0.5,     # Grade A: sub-half-pixel
    'rmse_good': 1.0,          # Grade B: sub-pixel
    'rmse_acceptable': 3.0,    # Grade C: few-pixel
    'rmse_poor': 10.0,         # Grade D: poor but usable for coarse alignment
    # Grade F: RMSE > rmse_poor

    'coverage_excellent': 0.70,
    'coverage_good': 0.50,
    'coverage_minimum': 0.15,  # Below 15% = Severe Match Clustering failure

    'inlier_ratio_good': 0.50,
    'inlier_ratio_minimum': 0.30,  # Below 30% is not trustworthy for lunar co-registration
    'inlier_ratio_hard_fail': 0.30,  # Anything below this is mostly false matches; reject immediately

    'min_inlier_count': MIN_REQUIRED_INLIERS,  # Strict floor: at least 15 inliers required
}


def compute_spatial_entropy(points: np.ndarray, img_shape: tuple, grid: tuple = (8, 8)) -> float:
    """
    Compute normalized spatial information entropy over an M x N cell grid.
    Ranges strictly from 0.0 (all points clustered in one cell) to 1.0 (uniform spread).
    Unlike chi-squared, does not saturate for large tie-point counts (Tier 3 fix).
    """
    if len(points) == 0:
        return 0.0
    h, w = img_shape[:2]
    gx, gy = grid
    xs = np.clip((points[:, 0] / max(w, 1) * gx).astype(int), 0, gx - 1)
    ys = np.clip((points[:, 1] / max(h, 1) * gy).astype(int), 0, gy - 1)
    cell_indices = ys * gx + xs
    total_cells = gx * gy
    counts = np.bincount(cell_indices, minlength=total_cells)
    probs = counts[counts > 0] / len(points)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(total_cells)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def assess_quality(
    rmse_pixels: float,
    coverage: float,
    inlier_ratio: float,
    inlier_count: int,
    total_matches: int,
    thresholds: Dict[str, float] = None,
    geometric_model_type: str = "HOMOGRAPHY",
    fallback_code: str = None,
    condition_number: float = None,
    spatial_entropy: float = None,
    holdout_rmse: float = None,
    holdout_count: int = 0,
    symmetric_rmse: float = None,
    crater_coverage: float = None,
    crater_count: int = None,
) -> Dict[str, Any]:
    """
    Assess registration quality with Tier 3 industrial-grade failure taxonomy.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warnings = []
    details = {}
    reason_codes = []

    # A fitted residual is not independent evidence. Require the estimated
    # geometry to generalize to withheld correspondences before accepting it.
    holdout_failure = False
    if holdout_count > 0:
        details['holdout_rmse'] = float(holdout_rmse) if holdout_rmse is not None and np.isfinite(holdout_rmse) else None
        details['holdout_count'] = int(holdout_count)
        max_holdout_rmse = float(t.get('holdout_rmse_max', 5.0))
        max_holdout_ratio = float(t.get('holdout_to_fit_ratio_max', 4.0))
        # The ratio guard is only meaningful when the fit RMSE is an independent,
        # non-consensus residual. MAGSAC++/TPS inlier-fit residuals are small by
        # construction (consensus points), so holdout/fit ratios far above 4.0 are
        # NORMAL for healthy registrations (e.g., holdout 0.44 px vs fit 0.05 px).
        # The absolute holdout cap above remains the true overfit protection.
        ratio_guard_active = (
            rmse_pixels is not None and np.isfinite(rmse_pixels) and rmse_pixels >= 1.0
        )
        holdout_failure = (
            holdout_rmse is None
            or not np.isfinite(holdout_rmse)
            or holdout_rmse > max_holdout_rmse
            or (ratio_guard_active and holdout_rmse / rmse_pixels > max_holdout_ratio)
        )
        if holdout_failure:
            warnings.append(
                f"Holdout validation failed: withheld RMSE "
                f"{holdout_rmse:.2f} px exceeds independent-validation limits."
                if holdout_rmse is not None and np.isfinite(holdout_rmse)
                else "Holdout validation failed: no finite withheld-point RMSE."
            )
            reason_codes.append("HOLDOUT_VALIDATION_FAILURE")

    symmetric_failure = False
    if symmetric_rmse is not None:
        details['symmetric_rmse'] = float(symmetric_rmse) if np.isfinite(symmetric_rmse) else None
        symmetric_failure = not np.isfinite(symmetric_rmse) or symmetric_rmse > float(t.get('symmetric_rmse_max', 5.0))
        if symmetric_failure:
            warnings.append("Bidirectional geometry validation failed: forward and reverse transforms disagree.")
            reason_codes.append("SYMMETRIC_GEOMETRY_FAILURE")

    # --- Matrix Conditioning Check (Tier 3) ---
    cond_penalty = 1.0
    if condition_number is not None and not np.isnan(condition_number):
        details['condition_number'] = float(condition_number)
        # Condition number computed on dimension-normalized coordinates:
        # Well-spread points have condition number < 50; collinear points have cond > 1e4.
        if condition_number > 1e4:
            warnings.append(f"Ill-conditioned geometry: collinear/degenerate point distribution (condition number = {condition_number:.2e} > 1e4)")
            reason_codes.append("ILL_CONDITIONED_GEOMETRY")
            cond_penalty = 0.2
        elif condition_number > 500.0:
            warnings.append(f"Borderline conditioning: highly elongated point distribution (condition number = {condition_number:.2e})")
            cond_penalty = 0.8

    # --- RMSE Grade ---
    if rmse_pixels is None or np.isnan(rmse_pixels) or rmse_pixels > t['rmse_poor']:
        rmse_grade = 'F'
        rmse_score = 0.0
        rmse_disp = f"{rmse_pixels:.1f} px" if rmse_pixels is not None else "None (rejected)"
        warnings.append(f"RMSE ({rmse_disp}) exceeds failure threshold — images likely don't overlap")
        reason_codes.append("HIGH_RMSE_FAILURE")
    elif rmse_pixels <= t['rmse_excellent']:
        rmse_grade = 'A'
        rmse_score = 1.0
    elif rmse_pixels <= t['rmse_good']:
        rmse_grade = 'B'
        rmse_score = 0.8
    elif rmse_pixels <= t['rmse_acceptable']:
        rmse_grade = 'C'
        rmse_score = 0.5
    else:
        rmse_grade = 'D'
        rmse_score = 0.2
        warnings.append(f"High RMSE ({rmse_pixels:.2f} px) — registration may be inaccurate")
        reason_codes.append("ELEVATED_RMSE")

    details['rmse_grade'] = rmse_grade
    details['rmse_score'] = rmse_score
    rmse_failure = rmse_pixels is None or not np.isfinite(rmse_pixels) or rmse_pixels > t['rmse_poor']

    # --- Coverage & Spatial Entropy Assessment ---
    is_entropy_failure = False
    if spatial_entropy is not None:
        details['spatial_entropy'] = float(spatial_entropy)
        if spatial_entropy < 0.35:
            warnings.append(f"Severe match clustering (Spatial Entropy = {spatial_entropy:.2f} < 0.35)")
            reason_codes.append("CLUSTERED_TIE_POINTS")

    # Check for identical or near-identical image pairs (sub-half-pixel RMSE with high inlier ratio)
    is_identical_pair = (rmse_pixels is not None and rmse_pixels <= 0.05 and inlier_count >= 10 and inlier_ratio >= 0.60)

    if coverage >= t['coverage_excellent'] or is_identical_pair:
        coverage_score = 1.0
    elif coverage >= t['coverage_good']:
        coverage_score = 0.7
    elif coverage >= t['coverage_minimum']:
        # If RMSE is already excellent and inliers are abundant, don't over-penalize concentrated crater textures
        coverage_score = 0.6 if (rmse_pixels is not None and rmse_pixels <= t['rmse_excellent'] and inlier_count >= 50) else 0.4
        warnings.append(f"Low spatial coverage ({coverage:.1%}) — alignment unverified in {1-coverage:.0%} of the image")
        reason_codes.append("LOW_SPATIAL_COVERAGE")
    else:
        coverage_score = 0.1
        warnings.append(f"Very low coverage ({coverage:.1%}) — registration is unreliable across most of the image")
        reason_codes.append("CRITICAL_COVERAGE_FAILURE")

    details['coverage_score'] = coverage_score

    # --- Inlier Ratio Assessment ---
    # For dense multi-model ensembles, hundreds of verified inliers (>= 100) with sub-pixel RMSE
    # and valid coverage represent high statistical confidence even if the raw candidate pool was large.
    is_high_inlier_exception = (
        inlier_count >= 100
        and rmse_pixels is not None
        and rmse_pixels <= t.get('rmse_acceptable', 3.0)
        and coverage >= t.get('coverage_minimum', 0.15)
    )
    # Dense-inlier exception for pushbroom strips: when 200+ inliers are geometrically verified
    # with acceptable RMSE, relaxing the ratio floor is mathematically justified because
    # the statistical power of 200+ points dominates any ratio-based quality signal.
    is_dense_inlier_exception = (
        inlier_count >= 200
        and rmse_pixels is not None
        and rmse_pixels <= t.get('rmse_acceptable', 3.0)
    )
    # MAGSAC++ has already removed outliers, so statistical power comes from the
    # ABSOLUTE count of geometrically verified inliers (Gate 2, >= 15) — not from
    # the raw-pool ratio. Deep matcher pools (LoFTR / RoMa / tiled fusion) routinely
    # produce hundreds of candidates, so a 17–27% verified ratio with 50+ inliers
    # and sub-pixel RMSE is a healthy registration, not a matcher failure.
    # Keep the strict 30% floor only when the verified evidence is thin.
    inlier_ratio_hard_floor = float(t.get('inlier_ratio_hard_fail', 0.30))
    if inlier_count >= 50:
        inlier_ratio_hard_floor = min(inlier_ratio_hard_floor, 0.10)
    elif inlier_count >= 25:
        inlier_ratio_hard_floor = min(inlier_ratio_hard_floor, 0.15)
    details['inlier_ratio_hard_floor'] = inlier_ratio_hard_floor
    hard_inlier_failure = (
        (inlier_ratio < inlier_ratio_hard_floor)
        and not is_high_inlier_exception
        and not is_dense_inlier_exception
    )
    if inlier_ratio >= t['inlier_ratio_good']:
        inlier_score = 1.0
    elif inlier_ratio >= t['inlier_ratio_minimum']:
        inlier_score = 0.5
        warnings.append(f"Low inlier ratio ({inlier_ratio:.1%}) — many false matches detected")
        reason_codes.append("LOW_INLIER_RATIO")
    elif is_high_inlier_exception:
        inlier_score = 0.6
        warnings.append(
            f"Moderate inlier ratio ({inlier_ratio:.1%}) on dense candidate pool "
            f"({total_matches} candidates, {inlier_count} verified inliers)"
        )
    else:
        inlier_score = 0.1
        warnings.append(f"Very low inlier ratio ({inlier_ratio:.1%}) — matcher struggled with this image pair")
        reason_codes.append("CRITICAL_INLIER_RATIO")

    if hard_inlier_failure:
        warnings.append(f"Hard rejection: inlier ratio ({inlier_ratio:.1%}) below minimum trust floor ({inlier_ratio_hard_floor:.0%}).")
        reason_codes.insert(0, "INSUFFICIENT_INLIER_RATIO")

    details['inlier_score'] = inlier_score

    # --- CRITICAL PIPELINE CONSTRAINT: OVERFITTING PREVENTION ---
    # 1. The 4-Point Trap: Any registration yielding <= 4 inliers or inliers < 15 with zero residual (0.000 px RMSE)
    # MUST be immediately classified as an overfit matrix. Assign Grade F.
    min_floor = int(t.get('min_inlier_count', MIN_REQUIRED_INLIERS))
    is_four_point_trap = (inlier_count <= 4 and (rmse_pixels is None or rmse_pixels < 0.01)) or (inlier_count < min_floor and rmse_pixels is not None and rmse_pixels < 0.001)
    
    # Strict Inlier Floor Check (Minimum 15 inliers required for statistical redundancy)
    insufficient_inliers = (inlier_count < min_floor)
    
    if is_four_point_trap:
        warnings.append(f"THE 4-POINT TRAP DETECTED: Mathematical Illusion (Overfit) with {inlier_count} inliers and zero residual — mathematically overfit matrix with zero redundancy.")
        reason_codes.insert(0, "FOUR_POINT_OVERFIT_TRAP")
        if "GATE_1_FOUR_POINT_TRAP" not in reason_codes:
            reason_codes.append("GATE_1_FOUR_POINT_TRAP")
    elif insufficient_inliers:
        warnings.append(f"Insufficient inliers ({inlier_count} < {min_floor}) for robust homography. Minimum {min_floor} required.")
        reason_codes.insert(0, "INSUFFICIENT_INLIERS")
        if "GATE_2_MIN_VIABLE_REGISTRATION" not in reason_codes:
            reason_codes.append("GATE_2_MIN_VIABLE_REGISTRATION")

    # 2. Spatial Distribution Check: Coverage < 15% rejects registration due to Severe Match Clustering
    # Exception: if the inlier count is very high (>100) and RMSE is acceptable, spatial clustering
    # alone should not cause an F grade — the high inlier count IS the distributional evidence.
    is_clustering_failure = (coverage < t.get('coverage_minimum', 0.15))
    # Dense-verified exception: 200+ inliers with sub-3px RMSE cannot be clustered by definition
    if is_clustering_failure and is_dense_inlier_exception:
        is_clustering_failure = False  # Override: density implies adequate spatial sampling
    if is_clustering_failure and not is_identical_pair:
        warnings.append(f"Severe match clustering: spatial coverage ({coverage:.1%}) < 15% — registration rejected across unverified crater grid.")
        if "SEVERE_MATCH_CLUSTERING" not in reason_codes:
            reason_codes.insert(0, "SEVERE_MATCH_CLUSTERING")
        if "GATE_3_SEVERE_MATCH_CLUSTERING" not in reason_codes:
            reason_codes.append("GATE_3_SEVERE_MATCH_CLUSTERING")

    # Normalized Shannon spatial entropy check (H_s < 0.35 indicates severe spatial concentration)
    is_entropy_failure = (spatial_entropy is not None and spatial_entropy < 0.35 and not is_dense_inlier_exception and not is_identical_pair)
    if is_entropy_failure:
        warnings.append(f"Low spatial entropy ({spatial_entropy:.2f} < 0.35) — tie-points clustered in too few grid cells.")
        if "GATE_3_LOW_SPATIAL_ENTROPY" not in reason_codes:
            reason_codes.append("GATE_3_LOW_SPATIAL_ENTROPY")

    # 2b. Physical Lunar Crater Landmark Coverage (YOLO11 Ground-Truth Gate)
    if crater_coverage is not None:
        details['crater_coverage'] = float(crater_coverage)
        if crater_count is not None:
            details['crater_count'] = int(crater_count)
        if crater_count is not None and crater_count >= 5 and crater_coverage < 0.15 and not is_identical_pair:
            warnings.append(
                f"Physical crater coverage failure: only {crater_coverage:.1%} of {crater_count} detected craters "
                "have verified inlier tie-points (< 15% threshold)."
            )
            if "LOW_CRATER_COVERAGE" not in reason_codes:
                reason_codes.append("LOW_CRATER_COVERAGE")

    # 3. Auto-Recovery Instruction on low-inlier or clustering failures
    if rmse_failure or is_four_point_trap or insufficient_inliers or (is_clustering_failure and not is_identical_pair) or is_entropy_failure or holdout_failure or symmetric_failure:
        auto_rec_msg = "Auto-Recovery: Re-run extraction with 'Max processing dimension (px)' upscaled (e.g., jump from 2048 to 4096) to force extraction of higher-frequency terrain features."
        details['auto_recovery'] = auto_rec_msg
        warnings.append(auto_rec_msg)

    # --- Model Fallback Degradation (Tier 2/3) ---
    model_penalty = 1.0
    max_allowed_grade = 'A'
    if fallback_code is not None:
        details['fallback_code'] = fallback_code
        reason_codes.append(fallback_code)
        if "TRANSLATION" in fallback_code:
            model_penalty = 0.3
            max_allowed_grade = 'D'
            warnings.append("Degraded to translation-only model (2 DOF) — homography ill-conditioned")
        elif "PARTIAL" in fallback_code or "AFFINE" in fallback_code:
            model_penalty = 0.6
            max_allowed_grade = 'C'
            warnings.append(f"Degraded to {geometric_model_type} model — perspective geometry unresolvable")

    # Calculate overall confidence
    count_penalty = 0.0 if (is_four_point_trap or insufficient_inliers) else 1.0
    coverage_penalty = 0.1 if ((is_clustering_failure or is_entropy_failure) and not is_identical_pair) else 1.0

    confidence = (
        rmse_score * 0.45 +
        coverage_score * 0.30 +
        inlier_score * 0.25
    ) * count_penalty * coverage_penalty * cond_penalty * model_penalty

    confidence = max(0.0, min(1.0, confidence))

    # Assign base grade — hard Grade F on overfit, insufficient inliers, severe clustering, or low entropy
    if (rmse_failure or is_four_point_trap or insufficient_inliers or hard_inlier_failure or
        (is_clustering_failure and not is_identical_pair) or is_entropy_failure or holdout_failure or symmetric_failure):
        grade = 'F'
        confidence = 0.0
        passed = False
    elif confidence >= 0.85:
        grade = 'A'
        passed = True
    elif confidence >= 0.65:
        grade = 'B'
        passed = True
    elif confidence >= 0.45:
        grade = 'C'
        passed = True
    elif confidence >= 0.25:
        grade = 'D'
        passed = False
    else:
        grade = 'F'
        passed = False

    # Enforce model fallback ceiling if not already failed
    grade_order = ['A', 'B', 'C', 'D', 'F']
    if grade != 'F' and grade_order.index(grade) < grade_order.index(max_allowed_grade):
        grade = max_allowed_grade

    # Deterministic Gate Telemetry Tracking (Gates 1-5)
    gate_telemetry = {
        "GATE_1_4_POINT_TRAP": "BREACHED: Mathematical Illusion (Overfit)" if is_four_point_trap else "PASSED",
        "GATE_1_FOUR_POINT_TRAP": "BREACHED: Mathematical Illusion (Overfit)" if is_four_point_trap else "PASSED",
        "GATE_2_MIN_VIABLE_REGISTRATION": f"BREACHED: < {min_floor} Inliers (got {inlier_count})" if insufficient_inliers else "PASSED",
        "GATE_3_SPATIAL_DISTRIBUTION": (
            f"BREACHED: Severe Match Clustering (Coverage {coverage:.1%} < 15%)" if (is_clustering_failure and not is_identical_pair)
            else (f"BREACHED: Low Spatial Entropy (H_s = {spatial_entropy:.2f} < 0.35)" if is_entropy_failure else "PASSED")
        ),
        "GATE_3_SEVERE_MATCH_CLUSTERING": "BREACHED: Coverage < 15%" if (is_clustering_failure and not is_identical_pair) else "PASSED",
        "GATE_3_LOW_SPATIAL_ENTROPY": f"BREACHED: H_s = {spatial_entropy:.2f} < 0.35" if is_entropy_failure else "PASSED",
        "GATE_4_ZERO_HALLUCINATION": "ENFORCED (Strict 0% Weight on 0 Raw Matches)",
        "GATE_5_AUTONOMOUS_RECOVERY": "ACTION_REQUIRED" if (is_four_point_trap or insufficient_inliers or (is_clustering_failure and not is_identical_pair) or is_entropy_failure) else "NOMINAL",
    }
    details['gate_telemetry'] = gate_telemetry

    primary_reason = reason_codes[0] if reason_codes else "QUALITY_PASSED"

    for w in warnings:
        logger.warning(f"[Quality Gate] {w}")

    result = {
        'grade': grade,
        'confidence': round(confidence, 3),
        'passed': passed,
        'primary_reason': primary_reason,
        'reason_codes': reason_codes,
        'warnings': warnings,
        'details': details,
        'gate_telemetry': gate_telemetry,
    }

    level_emoji = {'A': '🟢', 'B': '🟢', 'C': '🟡', 'D': '🟠', 'F': '🔴'}
    rmse_txt = f"{rmse_pixels:.3f}px" if rmse_pixels is not None else "None (rejected)"
    logger.info(
        f"[Quality Gate] Grade: {level_emoji.get(grade, '?')} {grade} "
        f"(confidence={confidence:.1%}) [{primary_reason}] | "
        f"RMSE={rmse_txt}, Coverage={coverage:.1%}, "
        f"Inliers={inlier_count}/{total_matches} ({inlier_ratio:.1%})"
    )

    return result
