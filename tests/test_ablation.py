"""
Unit Tests for Automated Ablation Testing & Certification (Feature 5)
"""

import os
import json
import pytest
from src.certify import certify_ablation_matrix
from scripts.run_ablation import load_pairs_from_manifest, run_ablation_experiment


def test_certify_ablation_matrix():
    matrix = {
        "best_algorithm": "loftr",
        "algorithms": {
            "loftr": {
                "total": 10,
                "passed": 9,
                "pass_rate": 0.90,
                "mean_rmse_px": 0.65,
                "mean_inliers": 140,
            },
            "sift": {
                "total": 10,
                "passed": 5,
                "pass_rate": 0.50,
                "mean_rmse_px": 1.45,
                "mean_inliers": 25,
            },
        },
    }

    report = certify_ablation_matrix(matrix, min_pass_rate=0.80)
    assert report["certified"] is True
    assert report["verdict"] == "CERTIFIED"
    assert "loftr" in report["certified_algorithms"]
    assert "sift" in report["failing_algorithms"]


def test_certify_ablation_matrix_failure():
    matrix = {
        "best_algorithm": "sift",
        "algorithms": {
            "sift": {
                "total": 5,
                "passed": 2,
                "pass_rate": 0.40,
                "mean_rmse_px": 2.50,
                "mean_inliers": 8,
            }
        },
    }

    report = certify_ablation_matrix(matrix, min_pass_rate=0.85)
    assert report["certified"] is False
    assert report["verdict"] == "NOT_CERTIFIED"
    assert len(report["certified_algorithms"]) == 0


def test_load_pairs_from_manifest(tmp_path):
    manifest = tmp_path / "test_manifest.json"
    data = [
        {"source": "img1.png", "reference": "img2.png", "name": "pair_01"},
        {"src_path": "img3.png", "ref_path": "img4.png", "pair_name": "pair_02"},
    ]
    manifest.write_text(json.dumps(data))

    pairs = load_pairs_from_manifest(str(manifest))
    assert len(pairs) == 2
    assert pairs[0]["src_path"] == "img1.png"
    assert pairs[1]["ref_path"] == "img4.png"
