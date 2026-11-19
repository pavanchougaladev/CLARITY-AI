"""
Stable, JSON-safe registration telemetry shared by API, UI, and batch consumers.

§7 of the CLARITY3D brief requires keeping execution state and scientific
status strictly separate:

  ExecutionStatus       — what happened computationally
  ScientificStatus      — whether the result is scientifically valid
  AccuracyEvidenceClass — how the reported RMSE was computed

A job that *succeeds* computationally may still be scientifically *rejected*.
RMSE is reported as null (not 0.0) when measurements are absent.
Both source-pixel and reference-pixel RMSE are surfaced as distinct fields.
"""

from typing import Any, Dict, List, Literal, Optional
import math

from pydantic import BaseModel, Field


# Legacy alias — kept for backward compatibility with existing API consumers
Status = Literal["verified", "rejected", "error"]

# §7 execution state (queued → running → succeeded | failed | cancelled)
ExecutionStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

# §7 scientific status — separate from execution state
# A succeeded job may be scientifically rejected; a failed job cannot be validated
ScientificStatus = Literal[
    "passed_configured_checks",
    "insufficient_evidence",
    "rejected",
    "unsupported_input",
    "unknown",
]

# §7 accuracy evidence class
AccuracyEvidenceClass = Literal[
    "fit_only",                  # RMSE on inlier points used to fit the model
    "held_out",                  # RMSE on withheld correspondences
    "independently_validated",   # Externally verified landmarks / GCPs
    "unavailable",
]


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def json_safe(value: Any) -> Any:
    """
    Recursively coerce a result payload into JSON-serialisable Python types.

    Pipeline results can contain numpy scalars (e.g. np.float32 RMSE) and
    arrays. Those are not encodable by the API layer and would surface as a
    500 error when a job is polled, so every outward-facing payload passes
    through this function.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


class RegistrationTelemetry(BaseModel):
    # ------------------------------------------------------------------ #
    # Legacy / backward-compatible fields                                  #
    # ------------------------------------------------------------------ #
    status: Status = "error"
    method: str = "unknown"
    total_matches: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    rmse_pixels: Optional[float] = None         # reference-image pixel RMSE (fit)
    weighted_rmse: Optional[float] = None
    gt_rmse_pixels: Optional[float] = None
    gt_point_count: int = 0
    gt_validation_status: str = "not_provided"
    metadata_validation_status: str = "not_requested"
    mission_grade_certified: bool = False
    coverage: float = 0.0
    spatial_entropy: Optional[float] = None
    quality_grade: str = "F"
    quality_passed: bool = False
    quality_confidence: float = 0.0
    primary_reason: str = "UNKNOWN"
    reason_codes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    auto_recovery: Optional[str] = None
    provenance_chain: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # §7 — Execution vs. Scientific status separation (new fields)         #
    # ------------------------------------------------------------------ #
    execution_status: ExecutionStatus = "succeeded"
    """Computational result: queued | running | succeeded | failed | cancelled."""

    scientific_status: ScientificStatus = "unknown"
    """
    Scientific validity — independent of execution_status.
    A succeeded job may still be scientifically rejected when gates fail.
    """

    # ------------------------------------------------------------------ #
    # §7 — Source-pixel vs reference-pixel RMSE (distinct fields)          #
    # ------------------------------------------------------------------ #
    rmse_ref_px: Optional[float] = None
    """
    RMSE in reference-image pixel space (working / downscaled coordinates).
    Use this for display; do NOT equate it with original-sensor-pixel accuracy.
    """

    rmse_src_px: Optional[float] = None
    """
    RMSE in original source-image pixel space (before any downscaling).
    Required by Problem Statement 26166 — the accuracy target is stated in
    source pixels. Null when an inverse transform is unavailable.
    """

    # ------------------------------------------------------------------ #
    # §7 — Accuracy evidence class                                         #
    # ------------------------------------------------------------------ #
    accuracy_evidence_class: AccuracyEvidenceClass = "unavailable"
    """
    How the reported RMSE was computed:
      fit_only             — on points used to fit the model (least trusted)
      held_out             — on withheld correspondences (better)
      independently_validated — externally verified landmarks / GCPs (most trusted)
    Model-fitting residuals are NEVER labelled as independent accuracy evidence.
    """

    @classmethod
    def from_results(cls, results: Dict[str, Any]) -> "RegistrationTelemetry":
        quality = results.get("quality") or {}
        passed  = bool(quality.get("passed", False))

        # Execution status
        exec_status: ExecutionStatus = "succeeded"

        # Scientific status
        if passed:
            sci_status: ScientificStatus = "passed_configured_checks"
        else:
            pr = (quality.get("primary_reason") or "").upper()
            if "INSUFFICIENT" in pr or "FOUR_POINT" in pr:
                sci_status = "insufficient_evidence"
            elif "UNSUPPORTED" in pr or "LOW_OVERLAP" in pr:
                sci_status = "unsupported_input"
            else:
                sci_status = "rejected"

        # Evidence class
        gt_status = str(results.get("gt_validation_status", "not_provided"))
        ho_val = results.get("holdout_validation") or {}
        if gt_status == "valid" and results.get("gt_rmse_pixels") is not None:
            ev_class: AccuracyEvidenceClass = "independently_validated"
        elif ho_val.get("n_holdout", 0) > 0:
            ev_class = "held_out"
        elif results.get("rmse_pixels") is not None:
            ev_class = "fit_only"
        else:
            ev_class = "unavailable"

        return cls(
            # Legacy
            status="verified" if passed else "rejected",
            method=str(results.get("method_used", "unknown")),
            total_matches=int(results.get("total_matches", 0)),
            inlier_count=int(results.get("inlier_count", 0)),
            inlier_ratio=float(results.get("inlier_ratio", 0.0)),
            rmse_pixels=finite_float(results.get("rmse_pixels")),
            weighted_rmse=finite_float(results.get("weighted_rmse")),
            gt_rmse_pixels=finite_float(results.get("gt_rmse_pixels")),
            gt_point_count=int(results.get("gt_point_count", 0)),
            gt_validation_status=gt_status,
            metadata_validation_status=str(
                (results.get("metadata_validation") or {}).get("status", "not_requested")
            ),
            mission_grade_certified=bool(results.get("mission_grade_certified", False)),
            coverage=float(results.get("coverage", 0.0)),
            spatial_entropy=finite_float(results.get("spatial_entropy")),
            quality_grade=str(quality.get("grade", "F")),
            quality_passed=passed,
            quality_confidence=float(quality.get("confidence", 0.0)),
            primary_reason=str(quality.get("primary_reason", "UNKNOWN")),
            reason_codes=list(quality.get("reason_codes", [])),
            warnings=list(quality.get("warnings", [])),
            auto_recovery=results.get("auto_recovery"),
            provenance_chain=results.get("provenance_chain"),
            # §7 new fields
            execution_status=exec_status,
            scientific_status=sci_status,
            rmse_ref_px=finite_float(results.get("rmse_pixels")),
            rmse_src_px=finite_float(results.get("rmse_raw_pixels")),
            accuracy_evidence_class=ev_class,
        )

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()
