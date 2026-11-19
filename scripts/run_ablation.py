#!/usr/bin/env python3
"""
Automated Ablation Testing CLI
==============================
ISRO LunarFusion Co-Registration Suite — Research & Certification Tooling

Executes systematic parameter and algorithm sweeps across multi-sensor lunar datasets
(e.g., LoFTR vs. SIFT vs. SuperGlue vs. LunarFusion ensemble), aggregating pass rates,
sub-pixel RMSE, inlier yields, and certification compliance into a verified matrix.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List

# Ensure repository root and backend packages (server/) are on Python module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server")))

from src.batch import run_batch, discover_image_pairs
from src.certify import certify_ablation_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("clarity.ablation")


def load_pairs_from_manifest(manifest_path: str) -> List[Dict[str, str]]:
    """Load image pairs from JSON benchmark manifest."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both list of pairs or dictionary with 'pairs' key
    pairs_raw = data.get("pairs", data) if isinstance(data, dict) else data
    pairs = []
    for idx, item in enumerate(pairs_raw):
        s_path = item.get("src_path") or item.get("source")
        r_path = item.get("ref_path") or item.get("reference")
        name = item.get("pair_name") or item.get("name") or f"pair_{idx+1:03d}"
        if s_path and r_path:
            pairs.append({
                "src_path": s_path,
                "ref_path": r_path,
                "pair_name": name,
            })
    return pairs


def run_ablation_experiment(
    pairs: List[Dict[str, str]],
    algorithms: List[str],
    max_dim: int = 2048,
    output_dir: str = "data/outputs/ablation",
    rmse_threshold: float = 2.0,
    certify: bool = True,
) -> Dict[str, Any]:
    """Execute ablation testing matrix across multiple candidate algorithms."""
    os.makedirs(output_dir, exist_ok=True)
    algo_metrics: Dict[str, Any] = {}

    print("\n" + "=" * 70)
    print(f"🚀 STARTING LUNARFUSION ABLATION SUITE ({len(algorithms)} algorithms, {len(pairs)} pairs)")
    print("=" * 70)

    for algo in algorithms:
        print(f"\n[Ablation] Testing candidate algorithm: {algo.upper()} ...")
        algo_out_dir = os.path.join(output_dir, algo)
        os.makedirs(algo_out_dir, exist_ok=True)

        t_start = time.time()
        batch_summary = run_batch(
            pairs=pairs,
            matcher_type=algo,
            max_dim=max_dim,
            rmse_threshold=rmse_threshold,
            output_dir=algo_out_dir,
        )
        elapsed = time.time() - t_start

        results = batch_summary.get("results", [])
        total_p = len(results)
        passed_p = sum(1 for r in results if r.get("status") == "PASS")
        pass_rate = passed_p / max(total_p, 1)

        valid_rmses = [r["rmse_pixels"] for r in results if r.get("rmse_pixels") is not None]
        mean_rmse = float(sum(valid_rmses) / len(valid_rmses)) if valid_rmses else None
        median_rmse = float(sorted(valid_rmses)[len(valid_rmses) // 2]) if valid_rmses else None

        valid_inliers = [r["inlier_count"] for r in results if r.get("inlier_count") is not None]
        mean_inliers = float(sum(valid_inliers) / len(valid_inliers)) if valid_inliers else 0.0

        algo_metrics[algo] = {
            "total": total_p,
            "passed": passed_p,
            "pass_rate": round(pass_rate, 4),
            "mean_rmse_px": round(mean_rmse, 4) if mean_rmse is not None else None,
            "median_rmse_px": round(median_rmse, 4) if median_rmse is not None else None,
            "mean_inliers": round(mean_inliers, 1),
            "elapsed_total_s": round(elapsed, 2),
            "mean_time_per_pair_s": round(elapsed / max(total_p, 1), 2),
        }
        print(f"  ✓ {algo.upper()}: Pass Rate: {pass_rate:.1%}, Mean RMSE: {mean_rmse if mean_rmse is not None else 'N/A'} px, Mean Inliers: {mean_inliers:.0f}")

    # Determine best algorithm (prioritize pass rate, then lowest mean RMSE)
    def _rank_key(item):
        s = item[1]
        p_rate = s["pass_rate"]
        rmse_val = s["mean_rmse_px"] if s["mean_rmse_px"] is not None else 999.0
        return (-p_rate, rmse_val)

    ranked = sorted(algo_metrics.items(), key=_rank_key)
    best_algo = ranked[0][0] if ranked else None

    matrix = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_pairs_tested": len(pairs),
        "max_dim": max_dim,
        "rmse_threshold_px": rmse_threshold,
        "best_algorithm": best_algo,
        "algorithms": algo_metrics,
    }

    if certify:
        cert_report = certify_ablation_matrix(matrix)
        matrix["certification"] = cert_report
        print("\n" + "=" * 70)
        print(f"📜 ABLATION CERTIFICATION VERDICT: {cert_report['verdict']}")
        print(f"   Certified Engines: {', '.join(cert_report['certified_algorithms']) or 'None'}")
        print("=" * 70)

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Run Lunar Image Registration Ablation Experiments")
    parser.add_argument("--manifest", type=str, default="auto_discovered_benchmark_manifest.json",
                        help="Path to image pair JSON manifest")
    parser.add_argument("--algorithms", nargs="+", default=["loftr", "sift"],
                        help="Candidate matching algorithms to evaluate")
    parser.add_argument("--max-dim", type=int, default=2048,
                        help="Max processing resolution in pixels")
    parser.add_argument("--output-dir", type=str, default="data/outputs/ablation",
                        help="Directory to save per-algorithm logs and exports")
    parser.add_argument("--export-matrix", type=str, default="data/outputs/ablation/ablation_matrix.json",
                        help="Path to export consolidated JSON matrix")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Maximum RMSE threshold for certifying pairs")
    parser.add_argument("--no-certify", action="store_true",
                        help="Skip quality gate certification check")

    args = parser.parse_args()

    # If manifest not found at default, fall back to discovery or synthetic
    if not os.path.exists(args.manifest):
        logger.warning(f"Manifest {args.manifest} not found, searching data/ directory...")
        pairs = discover_image_pairs("data", "data")
        if not pairs:
            logger.error("No image pairs found in data/. Please provide a valid --manifest.")
            sys.exit(1)
    else:
        pairs = load_pairs_from_manifest(args.manifest)

    matrix = run_ablation_experiment(
        pairs=pairs,
        algorithms=args.algorithms,
        max_dim=args.max_dim,
        output_dir=args.output_dir,
        rmse_threshold=args.threshold,
        certify=not args.no_certify,
    )

    # Save output matrix
    export_path = os.path.abspath(args.export_matrix)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    logger.info(f"Saved ablation matrix to: {export_path}")


if __name__ == "__main__":
    main()
