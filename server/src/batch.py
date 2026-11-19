"""
Batch Processing Engine
========================
Runs the registration pipeline on multiple image pairs, collects
metrics, and exports a consolidated summary report.
"""

import os
import time
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Callable

from .pipeline import run_registration_pipeline
from .registration.export import export_asp_compatible_tiepoints

logger = logging.getLogger(__name__)


def discover_image_pairs(
    src_dir: str,
    ref_dir: str,
    extensions: tuple = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.img'),
) -> List[Dict[str, str]]:
    """
    Auto-discovers matching image pairs from two directories.

    Pairing strategy:
      1. If filenames match exactly (e.g., region_A.tif in both), pair them.
      2. Otherwise, pair files alphabetically by index.

    Returns
    -------
    list of dict : Each dict has 'src_path', 'ref_path', and 'pair_name'.
    """
    def _list_images(directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for f in sorted(filenames):
                if os.path.splitext(f)[1].lower() in extensions:
                    files.append(os.path.join(root, f))
        return sorted(files)

    src_files = _list_images(src_dir)
    ref_files = _list_images(ref_dir)

    # Try name-based matching first
    src_names = {os.path.splitext(os.path.basename(f))[0]: f for f in src_files}
    ref_names = {os.path.splitext(os.path.basename(f))[0]: f for f in ref_files}

    common = set(src_names.keys()) & set(ref_names.keys())

    if common:
        pairs = []
        for name in sorted(common):
            pairs.append({
                'src_path': src_names[name],
                'ref_path': ref_names[name],
                'pair_name': name,
            })
        return pairs

    # Fallback: pair by index
    pairs = []
    for i, (s, r) in enumerate(zip(src_files, ref_files)):
        pairs.append({
            'src_path': s,
            'ref_path': r,
            'pair_name': f"pair_{i+1:03d}",
        })
    return pairs


def run_batch(
    pairs: List[Dict[str, str]],
    use_tps: bool = True,
    max_dim: int = 2048,
    rmse_threshold: float = 2.0,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    matcher_type: str = "loftr",
) -> Dict[str, Any]:
    """
    Run the registration pipeline on multiple image pairs.

    Parameters
    ----------
    pairs           : List of dicts with 'src_path', 'ref_path', 'pair_name'.
    use_tps         : Enable TPS jitter correction.
    max_dim         : Max processing dimension.
    rmse_threshold  : RMSE threshold for pass/fail classification (pixels).
    output_dir      : Directory to save per-pair CSV exports. None = skip export.
    progress_callback : Optional callable(pair_index, total_pairs, pair_name, status)

    Returns
    -------
    dict : Summary with 'results' list, 'summary' stats, and 'failures' list.
    """
    total = len(pairs)
    results = []
    failures = []

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for idx, pair in enumerate(pairs):
        pair_name = pair['pair_name']
        logger.info(f"[Batch {idx+1}/{total}] Processing: {pair_name}")

        if progress_callback:
            progress_callback(idx, total, pair_name, "running")

        t0 = time.time()
        try:
            img_src, img_ref, res = run_registration_pipeline(
                pair['src_path'], pair['ref_path'],
                use_tps=use_tps, max_dim=max_dim,
                matcher_type=matcher_type,
            )
            elapsed = time.time() - t0
            rmse_value = res.get('rmse_pixels')
            passed = bool(
                res.get('quality', {}).get('passed', False)
                and rmse_value is not None
                and rmse_value <= rmse_threshold
            )

            result_entry = {
                'pair_name': pair_name,
                'src_path': pair['src_path'],
                'ref_path': pair['ref_path'],
                'method': res['method_used'],
                'total_matches': res['total_matches'],
                'inlier_count': res['inlier_count'],
                'inlier_ratio': float(res['inlier_ratio']),
                'rmse_pixels': float(rmse_value) if rmse_value is not None else None,
                'weighted_rmse': float(res['weighted_rmse']) if res.get('weighted_rmse') is not None else None,
                'coverage': float(res['coverage']),
                'quality_grade': res.get('quality', {}).get('grade', 'N/A'),
                'quality_passed': res.get('quality', {}).get('passed', False),
                'elapsed_seconds': round(elapsed, 2),
                'status': 'PASS' if passed else 'FAIL',
            }
            results.append(result_entry)

            # Export tie-points per pair
            if output_dir:
                csv_path = os.path.join(output_dir, f"{pair_name}_tiepoints.csv")
                export_asp_compatible_tiepoints(
                    res['inliers_src'], res['inliers_ref'],
                    res['confidences'], csv_path,
                )

            if progress_callback:
                progress_callback(idx + 1, total, pair_name, result_entry['status'])

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[Batch] {pair_name} failed: {e}")
            failures.append({
                'pair_name': pair_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'elapsed_seconds': round(elapsed, 2),
            })
            if progress_callback:
                progress_callback(idx + 1, total, pair_name, "ERROR")

    # Summary statistics
    if results:
        rmse_values = [r['rmse_pixels'] for r in results if r['rmse_pixels'] is not None]
        pass_count = sum(1 for r in results if r['status'] == 'PASS')
        summary = {
            'total_pairs': total,
            'successful': len(results),
            'failed': len(failures),
            'passed': pass_count,
            'pass_rate': pass_count / max(len(results), 1),
            'mean_rmse': sum(rmse_values) / len(rmse_values) if rmse_values else None,
            'min_rmse': min(rmse_values) if rmse_values else None,
            'max_rmse': max(rmse_values) if rmse_values else None,
            'mean_inlier_ratio': sum(r['inlier_ratio'] for r in results) / len(results),
            'total_elapsed': sum(r['elapsed_seconds'] for r in results),
        }
    else:
        summary = {
            'total_pairs': total,
            'successful': 0,
            'failed': len(failures),
            'passed': 0,
            'pass_rate': 0.0,
        }

    batch_result = {
        'results': results,
        'failures': failures,
        'summary': summary,
    }

    # Save consolidated report
    if output_dir:
        report_path = os.path.join(output_dir, 'batch_report.json')
        with open(report_path, 'w') as f:
            json.dump(batch_result, f, indent=2)
        logger.info(f"[Batch] Report saved to {report_path}")

    return batch_result
