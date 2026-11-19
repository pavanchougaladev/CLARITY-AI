"""Benchmark harness for real lunar registration validation.

This module provides a lightweight dataset-level evaluation that runs the
registration pipeline over a set of image pairs and summarizes success rates,
RMSE, inlier counts, and reasons for rejection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .pipeline import run_registration_pipeline

logger = logging.getLogger(__name__)


def run_benchmark_suite(
    pairs: Iterable[Tuple[str, str]],
    max_dim: int = 2048,
    auto_recover: bool = True,
    matcher_type: str = "sift",
    **kwargs,
) -> dict:
    """Run a benchmark over a list of (src_path, ref_path) pairs.

    Returns a dict with aggregate metrics and per-pair result entries.
    """
    results = []
    passed = 0
    failed = 0
    all_rmse = []
    all_inliers = []
    reasons = {}

    for src_path, ref_path in pairs:
        try:
            _, _, result = run_registration_pipeline(
                src_path,
                ref_path,
                matcher_type=matcher_type,
                max_dim=max_dim,
                auto_recover=auto_recover,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive benchmark guard
            logger.warning(f"Benchmark pair failed to run: {src_path} vs {ref_path}: {exc}")
            result = {
                "quality": {"passed": False, "primary_reason": "BENCHMARK_EXCEPTION"},
                "rmse_pixels": None,
                "inlier_count": 0,
                "coverage": 0.0,
                "method_used": matcher_type,
                "source": src_path,
                "reference": ref_path,
            }

        quality = result.get("quality", {})
        rmse = result.get("rmse_pixels")
        inliers = int(result.get("inlier_count", 0))
        coverage = float(result.get("coverage", 0.0))
        primary_reason = quality.get("primary_reason", "UNKNOWN")

        if quality.get("passed", False):
            passed += 1
        else:
            failed += 1
        reasons[primary_reason] = reasons.get(primary_reason, 0) + 1

        if rmse is not None and np.isfinite(rmse):
            all_rmse.append(float(rmse))
        all_inliers.append(inliers)

        results.append({
            "source": src_path,
            "reference": ref_path,
            "passed": bool(quality.get("passed", False)),
            "grade": quality.get("grade", "F"),
            "primary_reason": primary_reason,
            "rmse_pixels": rmse,
            "inlier_count": inliers,
            "coverage": coverage,
            "method_used": result.get("method_used", matcher_type),
        })

    median_rmse = float(np.median(all_rmse)) if all_rmse else None
    max_rmse = float(np.max(all_rmse)) if all_rmse else None
    median_inliers = float(np.median(all_inliers)) if all_inliers else None

    summary = {
        "total_pairs": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / len(results)) if results else 0.0,
        "median_rmse_px": median_rmse,
        "max_rmse_px": max_rmse,
        "median_inlier_count": median_inliers,
        "reason_counts": reasons,
        "results": results,
    }
    return summary


def benchmark_from_manifest(manifest_path: str, **kwargs) -> dict:
    """Load a JSON manifest of pairs and run the benchmark suite."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    pairs = []
    for item in manifest.get("pairs", []):
        if isinstance(item, dict):
            pairs.append((item["source"], item["reference"]))
        else:
            pairs.append(tuple(item))

    return run_benchmark_suite(pairs, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lunar registration benchmark runner")
    parser.add_argument("manifest", nargs="?", default=None, help="JSON manifest of image pairs")
    parser.add_argument("--matcher", default="sift")
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--pair", action="append", nargs=2, default=[], metavar=("SRC", "REF"))
    args = parser.parse_args()

    if args.manifest:
        summary = benchmark_from_manifest(args.manifest, matcher_type=args.matcher, max_dim=args.max_dim)
    elif args.pair:
        summary = run_benchmark_suite(args.pair, matcher_type=args.matcher, max_dim=args.max_dim)
    else:
        raise SystemExit("Provide either a manifest or --pair SRC REF")

    print(json.dumps(summary, indent=2, default=str))
