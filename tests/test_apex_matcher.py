"""
Tests for the APEX 6-Engine Super-Fusion matcher.

APEX = DISK+LightGlue + RoMa v2 + ResShift + RootSIFT + MAGSAC++
(SuperGlue is deliberately excluded from the APEX combination.)

These tests avoid downloading deep-model checkpoints: deep sub-matchers are
mocked via patching ``src.features.apex_matcher.LunarMatcher`` (same strategy
as ``test_ensemble_matcher.py``), and the classical RootSIFT engine plus the
MAGSAC++ / ResShift post-stages are exercised with real OpenCV math.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.apex_matcher import (
    ApexMatcher,
    magsac_filter,
    resshift_restore,
    rootsift_match,
)


def _make_lunar_pair(size: int = 200) -> tuple:
    """Synthetic lunar-like source + affine-shifted reference pair."""
    rng = np.random.default_rng(11)
    # Fractal-ish multi-octave regolith texture so RootSIFT finds keypoints
    base = rng.normal(0, 1, (size, size))
    img = 0.35 * np.ones((size, size))
    for sigma in (40.0, 16.0, 6.0, 2.5):
        img += 0.10 * cv2.GaussianBlur(base, (0, 0), sigma) / (sigma ** 0.5)
    # Craters
    xx, yy = np.mgrid[0:size, 0:size]
    img += 0.10 * np.exp(-((xx - 60) ** 2 + (yy - 70) ** 2) / 300.0)
    img -= 0.12 * np.exp(-((xx - 140) ** 2 + (yy - 120) ** 2) / 600.0)
    src = np.clip(img * 255, 0, 255).astype(np.uint8)
    M = np.float32([[1, 0, 6], [0, 1, -4]])
    ref = cv2.warpAffine(src, M, (size, size))
    return src, ref


class TestResShiftStage(unittest.TestCase):
    def test_deterministic_fallback_runs(self):
        src, _ = _make_lunar_pair()
        restored, mode = resshift_restore(src, steps=3)
        self.assertEqual(mode, "deterministic_fallback")
        self.assertEqual(restored.shape, src.shape)
        self.assertEqual(restored.dtype, np.uint8)

    def test_empty_input_passthrough(self):
        empty = np.empty((0,), dtype=np.uint8)
        restored, mode = resshift_restore(empty)
        self.assertEqual(mode, "skipped_empty_input")

    def test_no_structure_hallucination(self):
        # Restoration of a flat field must stay flat within tolerance
        # (Gate 1 safe: no hallucinated structure).
        flat = np.full((64, 64), 100, dtype=np.uint8)
        restored, _ = resshift_restore(flat, steps=2)
        self.assertLessEqual(int(np.ptp(restored.astype(int))), 3)


class TestRootSIFTEngine(unittest.TestCase):
    def test_real_pair_matches(self):
        src, ref = _make_lunar_pair()
        pts_s, pts_r, conf = rootsift_match(src, ref)
        self.assertGreaterEqual(len(pts_s), 4)
        self.assertEqual(len(pts_s), len(pts_r))
        self.assertEqual(len(conf), len(pts_s))
        self.assertTrue(np.all((conf >= 0.0) & (conf <= 1.0)))

    def test_flat_image_zero_matches(self):
        flat = np.full((64, 64), 128, dtype=np.uint8)
        pts_s, pts_r, conf = rootsift_match(flat, flat)
        self.assertEqual(len(pts_s), 0)


class TestMAGSACStage(unittest.TestCase):
    def test_inlier_filtering(self):
        rng = np.random.default_rng(7)
        # 30 clean correspondences + 20 gross outliers
        pts_s = rng.uniform(0, 200, (50, 2)).astype(np.float32)
        pts_r = pts_s + np.array([5.0, -3.0], dtype=np.float32)
        pts_r[30:] += rng.uniform(60, 120, (20, 2)).astype(np.float32)  # outliers
        conf = np.ones(50, dtype=np.float32)

        f_s, f_r, f_c, stats = magsac_filter(pts_s, pts_r, conf)
        self.assertTrue(stats["magsac_applied"])
        # Survivors must be dominated by the clean set
        survivors = f_s if len(f_s) == len(f_c) else f_s
        self.assertLessEqual(len(survivors), 50)
        # Outliers near the shifted cluster should be rejected:
        # check the inlier count is close to the 30 clean matches.
        self.assertGreaterEqual(stats["magsac_inliers"], 25)
        self.assertLessEqual(stats["magsac_inliers"], 40)

    def test_too_few_matches_skips(self):
        pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
        f_s, f_r, f_c, stats = magsac_filter(pts, pts, np.ones(3, dtype=np.float32))
        self.assertFalse(stats["magsac_applied"])
        self.assertIn("too_few_matches", stats.get("magsac_skipped_reason", ""))


class TestApexMatcher(unittest.TestCase):
    def test_gate4_zero_match_engine_gets_zero_weight(self):
        apex = ApexMatcher(engines=["rootsift"], enable_resshift=True, enable_magsac=False)
        flat = np.full((64, 64), 10, dtype=np.uint8)
        _, _, _, stats = apex.match_with_stats(flat, flat)
        self.assertEqual(stats["raw_counts"]["rootsift"], 0)
        self.assertEqual(stats["engine_weights"]["rootsift"], 0.0)
        self.assertIsNone(stats.get("dominant_engine"))
        self.assertEqual(stats["total_fused"], 0)

    def test_fault_isolation(self):
        """A crashing deep engine must not prevent the remaining engines."""
        apex = ApexMatcher(engines=["lightglue", "roma_v2", "rootsift"],
                           enable_resshift=False, enable_magsac=False)
        src, ref = _make_lunar_pair()

        with patch("src.features.apex_matcher.LunarMatcher") as mock_lunar:
            def side_effect(matcher_type, **kwargs):
                instance = MagicMock()
                if matcher_type == "lightglue":
                    instance.match.side_effect = RuntimeError("CUDA OOM")
                else:
                    instance.match.return_value = (
                        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0],
                                  [70.0, 80.0], [90.0, 100.0]], dtype=np.float32),
                        np.array([[12.0, 22.0], [32.0, 42.0], [52.0, 62.0],
                                  [72.0, 82.0], [92.0, 102.0]], dtype=np.float32),
                        np.array([0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32),
                    )
                return instance

            mock_lunar.side_effect = side_effect
            pts_s, pts_r, conf, stats = apex.match_with_stats(src, ref)

        self.assertEqual(stats["raw_counts"]["lightglue"], 0)
        self.assertEqual(stats["engine_weights"]["lightglue"], 0.0)
        self.assertGreater(len(pts_s), 0)

    def test_superglue_excluded_from_default_stack(self):
        """SuperGlue must not be part of the APEX combination by default."""
        apex = ApexMatcher(engines=None)
        self.assertNotIn("superglue", apex.engines)
        self.assertNotIn("superpoint_superglue", apex.engines)

    def test_superglue_excluded_even_if_requested(self):
        """Explicitly requesting SuperGlue must not inject it into APEX."""
        apex = ApexMatcher(engines=["superglue", "lightglue", "rootsift"],
                           enable_resshift=False, enable_magsac=False)
        self.assertEqual(apex.engines, ["lightglue", "rootsift"])

    def test_superglue_never_runs_or_contributes(self):
        """APEX stats must never report SuperGlue as run/contributing/weighted."""
        apex = ApexMatcher(engines=["superglue", "rootsift"],
                           enable_resshift=False, enable_magsac=False)
        src, ref = _make_lunar_pair()
        _, _, _, stats = apex.match_with_stats(src, ref)

        self.assertNotIn("superglue", stats["raw_counts"])
        self.assertNotIn("superglue", stats["engines_run"])
        self.assertNotIn("superglue", stats["engines_contributing"])
        self.assertNotIn("superglue", stats["engine_weights"])
        self.assertNotEqual(stats.get("dominant_engine"), "superglue")
        self.assertNotIn("SuperGlue", stats["stack"])

    def test_engine_weights_sum_to_100(self):
        apex = ApexMatcher(engines=["rootsift"], enable_resshift=False, enable_magsac=False)
        src, ref = _make_lunar_pair()
        _, _, _, stats = apex.match_with_stats(src, ref)
        self.assertGreater(stats["raw_counts"]["rootsift"], 0)
        # Weights are distributed across every engine that produced raw matches
        # (including the Gate 5 deterministic fallback, when it engaged).
        self.assertAlmostEqual(sum(stats["engine_weights"].values()), 100.0, places=1)
        for engine, weight in stats["engine_weights"].items():
            if stats["raw_counts"].get(engine, 0) == 0:
                self.assertEqual(weight, 0.0)
        self.assertIn(stats["dominant_engine"], stats["engines_contributing"])

    def test_cpu_engine_list_excludes_roma(self):
        """On a CPU-only machine RoMa v2 must be dropped (1GB checkpoint)."""
        with patch("src.features.apex_matcher.HAS_TORCH", False):
            apex = ApexMatcher(engines=None)
        self.assertNotIn("roma_v2", apex.engines)

    def test_match_accepts_tensors(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        apex = ApexMatcher(engines=["rootsift"], enable_resshift=False, enable_magsac=False)
        src, ref = _make_lunar_pair()
        t_src = torch.from_numpy(src).float().unsqueeze(0).unsqueeze(0) / 255.0
        t_ref = torch.from_numpy(ref).float().unsqueeze(0).unsqueeze(0) / 255.0
        pts_s, pts_r, conf = apex.match(t_src, t_ref)
        self.assertEqual(pts_s.shape[1], 2)
        self.assertGreaterEqual(len(pts_s), 1)


class TestGate5Recovery(unittest.TestCase):
    """Gate 5: deterministic CPU fallback when the engine pool is hopeless."""

    @staticmethod
    def _fake(n: int):
        base = np.stack([np.linspace(10, 190, n), np.linspace(10, 190, n)], axis=1)
        return (base.astype(np.float32),
                (base + np.array([5.0, -3.0], dtype=np.float32)).astype(np.float32),
                np.full(n, 0.9, dtype=np.float32))

    def test_sift_recovery_engages_below_mvr_floor(self):
        apex = ApexMatcher(engines=["rootsift"], enable_resshift=False, enable_magsac=False)
        src, ref = _make_lunar_pair()

        with patch("src.features.apex_matcher.rootsift_match", return_value=self._fake(5)), \
             patch("src.features.apex_matcher.sift_recovery_match", return_value=self._fake(40)):
            _, _, _, stats = apex.match_with_stats(src, ref)

        self.assertTrue(stats["recovery"]["triggered"])
        self.assertEqual(stats["recovery"]["raw_matches"], 40)
        self.assertEqual(stats["raw_counts"]["sift_recovery"], 40)
        self.assertIn("sift_recovery", stats["engines_contributing"])
        self.assertAlmostEqual(sum(stats["engine_weights"].values()), 100.0, places=1)

    def test_sift_recovery_skipped_when_pool_is_sufficient(self):
        apex = ApexMatcher(engines=["rootsift"], enable_resshift=False, enable_magsac=False)
        src, ref = _make_lunar_pair()

        with patch("src.features.apex_matcher.rootsift_match", return_value=self._fake(60)), \
             patch("src.features.apex_matcher.sift_recovery_match") as mock_recovery:
            _, _, _, stats = apex.match_with_stats(src, ref)

        mock_recovery.assert_not_called()
        self.assertFalse(stats["recovery"]["triggered"])
        self.assertNotIn("sift_recovery", stats["raw_counts"])

    def test_pipeline_geometric_rejection_returns_graded_failure(self):
        """All matches rejected as outliers must yield Grade-F telemetry, not a crash."""
        from src import pipeline as pipeline_mod

        src, _ = _make_lunar_pair(180)
        ref = src.copy()

        with tempfile.TemporaryDirectory() as td:
            p_src = os.path.join(td, "src.png")
            p_ref = os.path.join(td, "ref.png")
            cv2.imwrite(p_src, src)
            cv2.imwrite(p_ref, ref)

            def _reject_all(pts_src, pts_ref, **kwargs):
                return None, np.zeros(len(pts_src), dtype=bool)

            with patch.object(pipeline_mod, "estimate_transform", side_effect=_reject_all):
                _, _, result = pipeline_mod.run_registration_pipeline(
                    p_src, p_ref, use_tps=False, matcher_type="sift",
                    use_tile_matching=False, auto_recover=False,
                )

        self.assertFalse(result["quality"]["passed"])
        self.assertEqual(result["quality"]["primary_reason"], "INSUFFICIENT_INLIERS")
        self.assertEqual(result["pipeline_status"], "FAILED QUALITY GATE")
        self.assertEqual(result["quality_grade"], "Grade F")
        self.assertEqual(result["inlier_count"], 0)


class TestPipelineWiring(unittest.TestCase):
    def test_pipeline_accepts_apex(self):
        """run_registration_pipeline must accept matcher_type='apex'."""
        from src.pipeline import run_registration_pipeline

        size = 160
        src, _ = _make_lunar_pair(size)
        # Identical reference maximizes the deterministic overlap precheck
        # score so the low-overlap SIFT guard does not hijack the APEX branch.
        ref = src.copy()
        # Dense 5x5 tie-point grid with sub-pixel jitter so the quality gates
        # (>= 15 inliers, non-zero RMSE) can pass.
        rng = np.random.default_rng(5)
        xs, ys = np.meshgrid(np.linspace(20, size - 20, 5), np.linspace(20, size - 20, 5))
        pts_s = np.stack([xs.ravel() + rng.normal(0, 0.25, xs.size),
                          ys.ravel() + rng.normal(0, 0.25, ys.size)], axis=1).astype(np.float32)
        pts_r = pts_s + np.array([6.0, -4.0], dtype=np.float32)  # known shift
        pts_r += rng.normal(0, 0.18, pts_r.shape).astype(np.float32)

        # Mock the ApexMatcher to avoid any deep-model checkpoint downloads.
        with patch("src.features.apex_matcher.ApexMatcher") as mock_apex_cls:
            instance = MagicMock()
            instance.match_with_stats.return_value = (
                pts_s,
                pts_r,
                np.full(len(pts_s), 0.9, dtype=np.float32),
                {"engine": "apex", "raw_counts": {"rootsift": len(pts_s)},
                 "engines_run": ["rootsift"], "engines_contributing": ["rootsift"],
                 "engine_weights": {"rootsift": 100.0}, "magsac": {},
                 "total_fused": len(pts_s), "dominant_engine": "rootsift"},
            )
            mock_apex_cls.return_value = instance

            with tempfile.TemporaryDirectory() as td:
                p_src = os.path.join(td, "src.png")
                p_ref = os.path.join(td, "ref.png")
                cv2.imwrite(p_src, src)
                cv2.imwrite(p_ref, ref)
                _, _, results = run_registration_pipeline(
                    p_src, p_ref, use_tps=False, matcher_type="apex", auto_recover=False
                )

        self.assertIn("apex_stats", results)
        self.assertEqual(results["apex_stats"]["engine"], "apex")
        self.assertIn("APEX", results.get("method_used") or "")


if __name__ == "__main__":
    unittest.main()
