#!/usr/bin/env python3
"""
Accuracy Benchmark — Three-Matcher Comparison
================================================
Runs LoFTR, DISK+LightGlue, and SIFT against multiple Chandrayaan
image pairs and collects per-matcher accuracy metrics:
  - RMSE, Weighted RMSE
  - Inlier ratio, Inlier count
  - Spatial coverage
  - Quality gate grade & confidence
  - Execution time

Results are saved to a JSON file for the documentation generator.
"""

import os
import sys
import json
import time
import logging
import traceback
import numpy as np


# Ensure project root and backend packages (server/) are in the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'server'))

from src.pipeline import run_registration_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ── Test Image Pairs ─────────────────────────────────────────────────────────
# We define pairs that exercise different scenarios:
#   Pair 1: CH2 TMC Nadir vs Aft (same orbit, same date → high overlap)
#   Pair 2: CH1 TMC Aft vs Fore  (same orbit, stereo pair → moderate overlap)
#   Pair 3: CH2 TMC Nadir vs Nadir (different date → challenging)
#   Pair 4: CH1 TMC Nadir vs Aft (same orbit, different viewing angle)
#   Pair 5: CH2 TMC Nadir g26 vs Aft g26 (same orbit stereo pair)

TEST_PAIRS = [
    {
        "name": "CH2-TMC Nadir vs Aft (Same Orbit, D18)",
        "category": "Same Orbit / High Overlap",
        "src": "data/raw/ch2_tmc_nrn_20200108T1153531115_d_img_d18/browse/raw/20200108/ch2_tmc_nrn_20200108T1153531115_b_brw_d18.png",
        "ref": "data/raw/ch2_tmc_nra_20200108T1153531115_d_img_d18/browse/raw/20200108/ch2_tmc_nra_20200108T1153531115_b_brw_d18.png",
    },
    {
        "name": "CH1-TMC Aft vs Fore (Stereo Pair, D18)",
        "category": "Stereo Pair / Moderate Overlap",
        "src": "data/raw/ch1_tmc_nca_20090529T0853239926_d_img_d18/browse/calibrated/20090529/ch1_tmc_nca_20090529T0853239926_b_brw_d18.png",
        "ref": "data/raw/ch1_tmc_ncf_20090529T0853239926_d_img_d18/browse/calibrated/20090529/ch1_tmc_ncf_20090529T0853239926_b_brw_d18.png",
    },
    {
        "name": "CH2-TMC Nadir vs Fore (Same Orbit, D18)",
        "category": "Same Orbit / Different View",
        "src": "data/raw/ch2_tmc_nrn_20200108T1153531115_d_img_d18/browse/raw/20200108/ch2_tmc_nrn_20200108T1153531115_b_brw_d18.png",
        "ref": "data/raw/ch2_tmc_nrf_20200108T1153531115_d_img_d18/browse/raw/20200108/ch2_tmc_nrf_20200108T1153531115_b_brw_d18.png",
    },
    {
        "name": "CH1-TMC Nadir vs Aft (Same Orbit, D18)",
        "category": "Same Orbit / Different View",
        "src": "data/raw/ch1_tmc_ncn_20090529T0853239798_d_img_d18/browse/calibrated/20090529/ch1_tmc_ncn_20090529T0853239798_b_brw_d18.png",
        "ref": "data/raw/ch1_tmc_nca_20090529T0853239926_d_img_d18/browse/calibrated/20090529/ch1_tmc_nca_20090529T0853239926_b_brw_d18.png",
    },
    {
        "name": "CH2-TMC Nadir vs Aft (Same Orbit, G26)",
        "category": "Same Orbit / High Overlap",
        "src": "data/raw/ch2_tmc_nrn_20191208T0221109733_d_img_g26/browse/raw/20191208/ch2_tmc_nrn_20191208T0221109733_b_brw_g26.png",
        "ref": "data/raw/ch2_tmc_nra_20191208T0221109733_d_img_g26/browse/raw/20191208/ch2_tmc_nra_20191208T0221109733_b_brw_g26.png",
    },
]

MATCHERS = ["fusion", "ensemble", "roma", "ni_lfm", "lightglue", "sift", "loftr"]

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "scripts", "accuracy_benchmark_results.json")


def run_single_test(pair: dict, matcher_type: str) -> dict:
    """Run a single matcher on a single image pair and collect metrics."""
    src_path = os.path.join(PROJECT_ROOT, pair["src"])
    ref_path = os.path.join(PROJECT_ROOT, pair["ref"])

    if not os.path.exists(src_path):
        return {"error": f"Source not found: {src_path}"}
    if not os.path.exists(ref_path):
        return {"error": f"Reference not found: {ref_path}"}

    t0 = time.time()
    try:
        _, _, results = run_registration_pipeline(
            src_path=src_path,
            ref_path=ref_path,
            matcher_type=matcher_type,
            max_dim=2048,
            use_tps=True,
            use_tile_matching=(matcher_type not in ["sift", "roma", "ensemble", "fusion", "lunar_fusion", "ni_lfm"]),
            tile_grid=(3, 3),
        )
        elapsed = time.time() - t0

        quality = results.get('quality', {})
        err_stats = results.get('error_stats', {})
        holdout_val = results.get('holdout_validation', {}) or {}
        error_budget = results.get('error_budget', {}) or {}
        h_rmse = holdout_val.get('holdout_rmse', float('nan'))
        tr_rmse = holdout_val.get('train_rmse', float('nan'))

        res_dict = {
            "status": "success",
            "matcher": matcher_type,
            "total_matches": int(results['total_matches']),
            "inlier_count": int(results['inlier_count']),
            "inlier_ratio": round(float(results['inlier_ratio']), 4),
            "rmse_pixels": round(float(results['rmse_pixels']), 4) if results.get('rmse_pixels') is not None else None,
            "weighted_rmse": round(float(results['weighted_rmse']), 4) if results.get('weighted_rmse') is not None else None,
            "holdout_rmse": round(float(h_rmse), 4) if not (isinstance(h_rmse, float) and np.isnan(h_rmse)) else None,
            "train_rmse": round(float(tr_rmse), 4) if not (isinstance(tr_rmse, float) and np.isnan(tr_rmse)) else None,
            "coverage": round(float(results['coverage']), 4),
            "spatial_entropy": round(float(results.get('spatial_entropy', 0.0)), 4),
            "condition_number": round(float(results['condition_number']), 2) if results.get('condition_number') is not None else None,
            "geometric_model_type": results.get('geometric_model_type', 'HOMOGRAPHY'),
            "fallback_code": results.get('fallback_code'),
            "quality_grade": quality.get('grade', 'N/A'),
            "quality_confidence": round(float(quality.get('confidence', 0)), 4),
            "quality_passed": quality.get('passed', False),
            "quality_reason_codes": quality.get('reason_codes', []),
            "quality_warnings": quality.get('warnings', []),
            "error_budget": {
                k: round(float(v), 4) for k, v in error_budget.items()
            } if error_budget else {},
            "method_used": results.get('method_used', matcher_type),
            "error_stats": {
                k: round(float(v), 4) if v != float('inf') else "inf"
                for k, v in err_stats.items()
            },
            "elapsed_seconds": round(elapsed, 2),
        }

        if 'fusion_stats' in results and results['fusion_stats']:
            res_dict['fusion_stats'] = {
                'profile_tag': results['fusion_stats'].get('profile_tag'),
                'models_used': results['fusion_stats'].get('models_used'),
                'weights': results['fusion_stats'].get('weights'),
                'reason': results['fusion_stats'].get('reason'),
                'total_fused': results['fusion_stats'].get('total_fused'),
            }

        return res_dict

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"FAILED: {pair['name']} with {matcher_type}: {e}")
        logger.debug(traceback.format_exc())
        return {
            "status": "failed",
            "matcher": matcher_type,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Accuracy Benchmark across lunar image matchers")
    parser.add_argument("--matchers", type=str, default=None, help="Comma-separated matchers to run")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated 1-based indices of pairs (e.g. 1,2)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if matcher already succeeded in output file")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Path to output JSON")
    args = parser.parse_args()

    active_matchers = [m.strip().lower() for m in args.matchers.split(",")] if args.matchers else MATCHERS
    target_pairs = TEST_PAIRS
    if args.pairs:
        indices = [int(p.strip()) - 1 for p in args.pairs.split(",")]
        target_pairs = [TEST_PAIRS[i] for i in indices if 0 <= i < len(TEST_PAIRS)]

    out_file = args.output

    # Load existing results if available
    existing_data = {}
    if os.path.exists(out_file) and not args.force:
        try:
            with open(out_file, 'r') as f:
                existing_data = json.load(f)
            logger.info(f"Loaded existing results from {out_file} ({len(existing_data.get('test_pairs', []))} pairs)")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    logger.info("=" * 70)
    logger.info(f"  ACCURACY BENCHMARK — Evaluating {len(active_matchers)} Matchers on {len(target_pairs)} Pairs")
    logger.info("=" * 70)

    # Initialize pair records from TEST_PAIRS
    pair_records = {}
    for p in TEST_PAIRS:
        pair_records[p["name"]] = {
            "pair_name": p["name"],
            "category": p["category"],
            "src": p["src"],
            "ref": p["ref"],
            "matcher_results": {},
        }

    # Pre-populate with existing results
    for p_entry in existing_data.get("test_pairs", []):
        p_name = p_entry.get("pair_name")
        if p_name in pair_records:
            for m, res in p_entry.get("matcher_results", {}).items():
                if res.get("status") == "success":
                    pair_records[p_name]["matcher_results"][m] = res

    # Run tests
    for pair_idx, pair in enumerate(target_pairs):
        p_name = pair["name"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Pair {pair_idx+1}/{len(target_pairs)}: {p_name}")
        logger.info(f"  Category: {pair['category']}")
        logger.info(f"{'─'*60}")

        for matcher in active_matchers:
            if not args.force and matcher in pair_records[p_name]["matcher_results"]:
                cached = pair_records[p_name]["matcher_results"][matcher]
                if cached.get("status") == "success":
                    logger.info(f"  ⚡ {matcher.upper()}: Using cached result (RMSE={cached['rmse_pixels']:.4f}px, Inliers={cached['inlier_count']})")
                    continue

            logger.info(f"\n  >>> Running {matcher.upper()} on '{p_name}'...")
            result = run_single_test(pair, matcher)
            pair_records[p_name]["matcher_results"][matcher] = result

            if result.get("status") == "success":
                logger.info(
                    f"  <<< {matcher.upper()}: Grade={result['quality_grade']} "
                    f"RMSE={result['rmse_pixels']:.4f}px "
                    f"Inliers={result['inlier_count']}/{result['total_matches']} "
                    f"Coverage={result['coverage']:.1%} "
                    f"Time={result['elapsed_seconds']:.1f}s"
                )
            else:
                logger.info(f"  <<< {matcher.upper()}: FAILED — {result.get('error', 'unknown')}")

            # Checkpoint intermediate results immediately to disk
            try:
                with open(out_file, 'w') as f:
                    json.dump({
                        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "matchers": MATCHERS,
                        "test_pairs": list(pair_records.values()),
                    }, f, indent=2)
            except Exception as e:
                logger.debug(f"Checkpoint save error: {e}")

    all_results = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matchers": MATCHERS,
        "test_pairs": list(pair_records.values()),
    }

    # ── Compute per-matcher aggregate stats ───────────────────────────────
    aggregate = {}
    for matcher in MATCHERS:
        rmses = []
        w_rmses = []
        holdout_rmses = []
        spatial_entropies = []
        coverages = []
        inlier_ratios = []
        grades = []
        times = []
        pass_count = 0
        fail_count = 0
        total_inliers = 0
        total_matches = 0

        for pair_result in all_results["test_pairs"]:
            r = pair_result["matcher_results"].get(matcher, {})
            if r.get("status") == "success":
                rmses.append(r["rmse_pixels"])
                w_rmses.append(r["weighted_rmse"])
                if r.get("holdout_rmse") is not None:
                    holdout_rmses.append(r["holdout_rmse"])
                if r.get("spatial_entropy") is not None:
                    spatial_entropies.append(r["spatial_entropy"])
                coverages.append(r["coverage"])
                inlier_ratios.append(r["inlier_ratio"])
                grades.append(r["quality_grade"])
                times.append(r["elapsed_seconds"])
                total_inliers += r["inlier_count"]
                total_matches += r["total_matches"]
                if r["quality_passed"]:
                    pass_count += 1
            else:
                fail_count += 1

        n = len(rmses)
        if n > 0:
            import numpy as np
            aggregate[matcher] = {
                "runs_succeeded": n,
                "runs_failed": fail_count,
                "pass_rate": round(pass_count / n, 4),
                "avg_rmse": round(float(np.mean(rmses)), 4),
                "median_rmse": round(float(np.median(rmses)), 4),
                "min_rmse": round(float(np.min(rmses)), 4),
                "max_rmse": round(float(np.max(rmses)), 4),
                "std_rmse": round(float(np.std(rmses)), 4),
                "avg_weighted_rmse": round(float(np.mean(w_rmses)), 4),
                "avg_holdout_rmse": round(float(np.mean(holdout_rmses)), 4) if holdout_rmses else None,
                "avg_spatial_entropy": round(float(np.mean(spatial_entropies)), 4) if spatial_entropies else 0.0,
                "avg_coverage": round(float(np.mean(coverages)), 4),
                "avg_inlier_ratio": round(float(np.mean(inlier_ratios)), 4),
                "total_inliers": total_inliers,
                "total_matches": total_matches,
                "avg_time_seconds": round(float(np.mean(times)), 2),
                "grade_distribution": {g: grades.count(g) for g in ['A', 'B', 'C', 'D', 'F']},
            }
        else:
            aggregate[matcher] = {"runs_succeeded": 0, "runs_failed": fail_count}

    all_results["aggregate"] = aggregate

    # ── Determine winner ──────────────────────────────────────────────────
    # Score: lower RMSE is better, higher coverage is better, higher inlier ratio is better
    # Composite score = 0.40*(1/avgRMSE) + 0.25*avgCoverage + 0.20*avgInlierRatio + 0.15*passRate
    best_matcher = None
    best_score = -1
    rankings = {}

    for matcher in MATCHERS:
        agg = aggregate.get(matcher, {})
        if agg.get("runs_succeeded", 0) == 0:
            rankings[matcher] = {"score": 0, "rank": len(MATCHERS)}
            continue

        avg_rmse = agg["avg_rmse"]
        rmse_score = 1.0 / (1.0 + avg_rmse)  # bounded [0,1], lower RMSE → higher score
        cov_score = agg["avg_coverage"]
        inlier_score = agg["avg_inlier_ratio"]
        pass_score = agg["pass_rate"]

        composite = (0.40 * rmse_score + 0.25 * cov_score + 0.20 * inlier_score + 0.15 * pass_score)
        rankings[matcher] = {"score": round(composite, 4)}

        if composite > best_score:
            best_score = composite
            best_matcher = matcher

    # Assign ranks
    sorted_matchers = sorted(rankings.keys(), key=lambda m: rankings[m]["score"], reverse=True)
    for rank, m in enumerate(sorted_matchers, 1):
        rankings[m]["rank"] = rank

    all_results["rankings"] = rankings
    all_results["best_matcher"] = best_matcher

    # ── Save results ──────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"  BENCHMARK COMPLETE — Results saved to {OUTPUT_FILE}")
    logger.info(f"  Best Matcher: {best_matcher.upper() if best_matcher else 'N/A'}")
    logger.info(f"{'='*70}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'MATCHER':<15} {'AVG RMSE':>10} {'HOLDOUT':>10} {'COVERAGE':>10} {'INLIER%':>10} {'ENTROPY':>10} {'RANK':>6}")
    print("-" * 90)
    for m in sorted_matchers:
        agg = aggregate.get(m, {})
        if agg.get("runs_succeeded", 0) > 0:
            h_str = f"{agg['avg_holdout_rmse']:.2f}px" if agg.get('avg_holdout_rmse') is not None else "N/A"
            r_badge = '🥇' if rankings[m]['rank']==1 else '🥈' if rankings[m]['rank']==2 else '🥉' if rankings[m]['rank']==3 else f"#{rankings[m]['rank']}"
            print(
                f"{m.upper():<15} "
                f"{agg['avg_rmse']:>9.4f}px "
                f"{h_str:>10} "
                f"{agg['avg_coverage']:>9.1%} "
                f"{agg['avg_inlier_ratio']:>9.1%} "
                f"{agg.get('avg_spatial_entropy', 0.0):>9.3f} "
                f"{r_badge:>6}"
            )
        else:
            print(f"{m.upper():<15} {'ALL FAILED':>10}")
    print("=" * 90)


if __name__ == "__main__":
    main()
