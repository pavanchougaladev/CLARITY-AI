import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.features.ensemble_matcher import fuse_keypoints, EnsembleMatcher


class TestEnsembleMatcher(unittest.TestCase):
    def test_fuse_keypoints_empty(self):
        matches = {}
        f_src, f_ref, f_conf, stats = fuse_keypoints(matches)
        self.assertEqual(len(f_src), 0)
        self.assertEqual(len(f_ref), 0)
        self.assertEqual(len(f_conf), 0)
        self.assertEqual(stats["total_fused"], 0)

    def test_fuse_keypoints_consensus_boosting(self):
        # Model 1 (e.g. RoMa) finds a match at (100, 100) -> (105, 105)
        # Model 2 (e.g. LoFTR) finds a match at (100.5, 99.8) -> (105.2, 104.9)
        matches = {
            "roma": (
                np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32),
                np.array([[105.0, 105.0], [210.0, 210.0]], dtype=np.float32),
                np.array([0.7, 0.8], dtype=np.float32),
            ),
            "loftr": (
                np.array([[100.5, 99.8], [300.0, 300.0]], dtype=np.float32),
                np.array([[105.2, 104.9], [315.0, 315.0]], dtype=np.float32),
                np.array([0.65, 0.9], dtype=np.float32),
            ),
        }

        f_src, f_ref, f_conf, stats = fuse_keypoints(
            matches, spatial_tolerance=3.0, consensus_boost=0.25
        )

        # 4 raw matches, 2 should merge (the one near (100, 100)), giving 3 total fused
        self.assertEqual(stats["total_raw"], 4)
        self.assertEqual(stats["total_fused"], 3)
        self.assertEqual(stats["consensus_count"], 1)

        # Find the consensus point (near 100, 100)
        dist_from_100 = np.linalg.norm(f_src - np.array([100.0, 100.0]), axis=1)
        consensus_idx = np.argmin(dist_from_100)

        # Confidence should be boosted above 0.7 + 0.25 = 0.95
        self.assertGreaterEqual(f_conf[consensus_idx], 0.90)

    def test_fuse_keypoints_conflict_resolution(self):
        # Two models pick same source point (100, 100) but predict divergent targets
        matches = {
            "model_a": (
                np.array([[100.0, 100.0]], dtype=np.float32),
                np.array([[110.0, 110.0]], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
            ),
            "model_b": (
                np.array([[100.2, 100.1]], dtype=np.float32),
                np.array([[500.0, 500.0]], dtype=np.float32),  # Incompatible target!
                np.array([0.4], dtype=np.float32),
            ),
            "model_c": (
                np.array([[50.0, 50.0], [60.0, 60.0], [70.0, 70.0]], dtype=np.float32),
                np.array([[55.0, 55.0], [65.0, 65.0], [75.0, 75.0]], dtype=np.float32),
                np.array([0.8, 0.8, 0.8], dtype=np.float32),
            ),
        }

        f_src, f_ref, f_conf, stats = fuse_keypoints(matches, spatial_tolerance=3.0)

        # The conflicting cluster should NOT boost consensus and should select the higher confidence target
        dist_from_100 = np.linalg.norm(f_src - np.array([100.0, 100.0]), axis=1)
        idx_100 = np.argmin(dist_from_100)
        # Check target is (110, 110), not (500, 500)
        np.testing.assert_allclose(f_ref[idx_100], [110.0, 110.0], atol=1.0)
        self.assertTrue(np.isclose(float(f_conf[idx_100]), 0.9, rtol=1e-4), f"Expected ~0.9, got {f_conf[idx_100]}")

    def test_ensemble_matcher_fault_isolation(self):
        # Test that if one sub-matcher throws, the remaining sub-matchers succeed
        ensemble = EnsembleMatcher(models=["failing_model", "succeeding_model"])

        img_dummy = np.zeros((100, 100), dtype=np.uint8)

        with patch("src.features.ensemble_matcher.LunarMatcher") as mock_lunar:
            def side_effect(matcher_type, **kwargs):
                instance = MagicMock()
                if matcher_type == "failing_model":
                    instance.match.side_effect = RuntimeError("CUDA Out of Memory")
                else:
                    instance.match.return_value = (
                        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [90.0, 100.0]], dtype=np.float32),
                        np.array([[12.0, 22.0], [32.0, 42.0], [52.0, 62.0], [72.0, 82.0], [92.0, 102.0]], dtype=np.float32),
                        np.array([0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32),
                    )
                return instance

            mock_lunar.side_effect = side_effect

            pts_s, pts_r, conf, succeeded, stats = ensemble.match(img_dummy, img_dummy)

            self.assertIn("succeeding_model", succeeded)
            self.assertNotIn("failing_model", succeeded)
            self.assertEqual(len(pts_s), 5)

    def test_pipeline_ensemble_integration(self):
        # Verify run_registration_pipeline accepts matcher_type='ensemble'
        import tempfile
        import os
        import cv2
        from src.pipeline import run_registration_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "s.png")
            p2 = os.path.join(tmpdir, "r.png")
            img = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(img, (50, 50), 20, 200, -1)
            cv2.imwrite(p1, img)
            cv2.imwrite(p2, img)

            # Mock EnsembleMatcher to return known valid correspondences
            with patch("src.pipeline.EnsembleMatcher") as mock_ens_cls:
                mock_ens = MagicMock()
                # 8 valid points with small identity shift
                pts = np.array([
                    [20.0, 20.0], [20.0, 80.0], [80.0, 20.0], [80.0, 80.0],
                    [30.0, 50.0], [50.0, 30.0], [70.0, 50.0], [50.0, 70.0],
                ], dtype=np.float32)
                mock_ens.match.return_value = (
                    pts, pts.copy(), np.ones(8, dtype=np.float32),
                    ["roma", "loftr"], {"consensus_count": 4}
                )
                mock_ens_cls.return_value = mock_ens

                s_out, r_out, res = run_registration_pipeline(
                    p1, p2, use_tps=False, matcher_type="ensemble"
                )
                self.assertIn("ENSEMBLE", res["method_used"])
                self.assertEqual(res["inlier_count"], 8)


if __name__ == "__main__":
    unittest.main()

