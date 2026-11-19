"""Certification reporting for lunar registration benchmark results.

This module converts a benchmark summary into an objective certification report
that can be used to state whether the pipeline meets the stated criteria.
"""

from __future__ import annotations

import json
from typing import Dict, Any


MIN_PASS_RATE = 0.90
MAX_MEDIAN_RMSE = 1.0
MIN_MEDIAN_INLIERS = 15


def certify_benchmark(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a certification verdict for a benchmark summary."""
    total_pairs = int(summary.get("total_pairs", 0))
    pass_rate = float(summary.get("pass_rate", 0.0))
    median_rmse = summary.get("median_rmse_px")
    median_inliers = summary.get("median_inlier_count")
    reason_counts = summary.get("reason_counts", {})

    pass_conditions = {
        "pass_rate_ok": total_pairs > 0 and pass_rate >= MIN_PASS_RATE,
        "rmse_ok": median_rmse is not None and median_rmse <= MAX_MEDIAN_RMSE,
        "inliers_ok": median_inliers is not None and median_inliers >= MIN_MEDIAN_INLIERS,
    }

    certified = all(pass_conditions.values())
    verdict = "CERTIFIED" if certified else "NOT_CERTIFIED"

    reason_summary = []
    if not pass_conditions["pass_rate_ok"]:
        reason_summary.append(f"Pass rate below {MIN_PASS_RATE:.0%} ({pass_rate:.1%})")
    if not pass_conditions["rmse_ok"]:
        reason_summary.append(f"Median RMSE exceeds {MAX_MEDIAN_RMSE} px ({median_rmse})")
    if not pass_conditions["inliers_ok"]:
        reason_summary.append(f"Median inliers below {MIN_MEDIAN_INLIERS} ({median_inliers})")

    if reason_counts:
        top_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        reason_summary.append("Top failure reasons: " + ", ".join(f"{k}={v}" for k, v in top_reasons))

    return {
        "verdict": verdict,
        "certified": certified,
        "thresholds": {
            "min_pass_rate": MIN_PASS_RATE,
            "max_median_rmse_px": MAX_MEDIAN_RMSE,
            "min_median_inliers": MIN_MEDIAN_INLIERS,
        },
        "metrics": {
            "total_pairs": total_pairs,
            "pass_rate": pass_rate,
            "median_rmse_px": median_rmse,
            "median_inlier_count": median_inliers,
        },
        "pass_conditions": pass_conditions,
        "reason_summary": reason_summary,
    }


def certify_ablation_matrix(matrix: Dict[str, Any], min_pass_rate: float = 0.85) -> Dict[str, Any]:
    """
    Certify a multi-algorithm ablation benchmark matrix.
    Evaluates whether candidate algorithms meet production flight thresholds.
    """
    algos = matrix.get("algorithms", {})
    certified_algos = []
    failing_algos = []

    for name, stats in algos.items():
        total = stats.get("total", 0)
        p_rate = stats.get("pass_rate", 0.0)
        rmse_val = stats.get("mean_rmse_px")
        inliers_val = stats.get("mean_inliers", 0)

        is_ok = (
            total > 0
            and p_rate >= min_pass_rate
            and (rmse_val is None or rmse_val <= MAX_MEDIAN_RMSE * 1.5)
            and inliers_val >= MIN_MEDIAN_INLIERS
        )
        if is_ok:
            certified_algos.append(name)
        else:
            failing_algos.append(name)

    overall_certified = len(certified_algos) > 0
    return {
        "verdict": "CERTIFIED" if overall_certified else "NOT_CERTIFIED",
        "certified": overall_certified,
        "certified_algorithms": certified_algos,
        "failing_algorithms": failing_algos,
        "min_required_pass_rate": min_pass_rate,
        "total_algorithms_tested": len(algos),
        "best_algorithm": matrix.get("best_algorithm"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Certify a benchmark summary")
    parser.add_argument("summary", help="Path to a JSON benchmark summary")
    args = parser.parse_args()

    with open(args.summary, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    print(json.dumps(certify_benchmark(summary), indent=2))
