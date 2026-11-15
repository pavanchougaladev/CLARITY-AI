import cv2
import numpy as np

from src.benchmark import run_benchmark_suite


def test_run_benchmark_suite_returns_summary(tmp_path):
    src = np.zeros((256, 256), dtype=np.uint8)
    ref = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(src, (128, 128), 30, 255, -1)
    cv2.circle(ref, (138, 128), 30, 255, -1)

    src_path = tmp_path / "src.png"
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(src_path), src)
    cv2.imwrite(str(ref_path), ref)

    summary = run_benchmark_suite([(str(src_path), str(ref_path))], max_dim=256, auto_recover=False)

    assert summary["total_pairs"] == 1
    assert summary["passed"] + summary["failed"] == 1
    assert "median_rmse_px" in summary
    assert "results" in summary
    assert len(summary["results"]) == 1
