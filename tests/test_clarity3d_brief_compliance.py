"""
Tests for §3, §7, §9 and §13.4 CLARITY3D brief implementations:
  - ImageDescriptor typed container
  - Artifact exporter (matches.csv, transform.json, quality.json,
    provenance.json, valid_mask.tif, support_mask.tif)
  - result_schema ExecutionStatus / ScientificStatus / AccuracyEvidenceClass
  - Blender renderer terrain_frame.json sidecar
  - Rejected-product guard (registered_source.tif must NOT be written when
    quality gate fails)
"""

import json
import os
import tempfile
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ImageDescriptor (§3)
# ---------------------------------------------------------------------------
class TestImageDescriptor:
    def _make_desc(self, path="dummy.tif"):
        from src.io.image_descriptor import ImageDescriptor
        return ImageDescriptor(image_id="test", path=path, sensor="TMC-2",
                               dims=(1000, 2000), dtype="uint16")

    def test_to_dict_is_json_serialisable(self):
        d = self._make_desc()
        blob = json.dumps(d.to_dict())
        assert "test" in blob

    def test_from_dict_round_trip(self):
        from src.io.image_descriptor import ImageDescriptor
        d = self._make_desc()
        d2 = ImageDescriptor.from_dict(d.to_dict())
        assert d2.sensor == "TMC-2"
        assert d2.dims == (1000, 2000)

    def test_dims_is_tuple_after_from_dict(self):
        from src.io.image_descriptor import ImageDescriptor
        d = self._make_desc()
        blob = d.to_dict()
        assert isinstance(blob["dims"], list)        # serialised as list
        d2 = ImageDescriptor.from_dict(blob)
        assert isinstance(d2.dims, tuple)            # restored as tuple

    def test_no_invented_crs_for_jpeg(self, tmp_path):
        """JPEG inputs must not receive a fabricated CRS."""
        from src.io.image_descriptor import ImageDescriptor
        img_path = str(tmp_path / "test.jpg")
        # Write a tiny JPEG so file exists
        import cv2
        cv2.imwrite(img_path, np.zeros((64, 64), dtype=np.uint8))
        d = ImageDescriptor.from_path(img_path)
        assert d.crs_wkt is None, "CRS must not be invented for JPEG"

    def test_sha256_is_none_for_missing_file(self):
        from src.io.image_descriptor import ImageDescriptor
        d = ImageDescriptor(path="/nonexistent/file.tif")
        result = d.compute_checksum()
        assert result is None

    def test_pair_descriptor_round_trip(self):
        from src.io.image_descriptor import ImageDescriptor, ImagePairDescriptor
        pair = ImagePairDescriptor(
            fixed=ImageDescriptor(image_id="ref"),
            moving=ImageDescriptor(image_id="src"),
            gsd_ratio=2.5,
        )
        d = pair.to_dict()
        assert d["fixed"]["image_id"] == "ref"
        assert d["gsd_ratio"] == 2.5


# ---------------------------------------------------------------------------
# result_schema — §7 status separation
# ---------------------------------------------------------------------------
class TestResultSchema:
    def _make_results(self, passed=True, inliers=25, rmse=0.45):
        return {
            "quality": {
                "passed": passed,
                "grade": "A" if passed else "F",
                "confidence": 0.9 if passed else 0.0,
                "primary_reason": "QUALITY_PASSED" if passed else "INSUFFICIENT_INLIERS",
                "reason_codes": [] if passed else ["INSUFFICIENT_INLIERS"],
                "warnings": [],
            },
            "method_used": "SIFT",
            "total_matches": 100,
            "inlier_count": inliers,
            "inlier_ratio": inliers / 100,
            "rmse_pixels": rmse,
            "rmse_raw_pixels": rmse * 2,
            "coverage": 0.6,
            "spatial_entropy": 0.7,
            "auto_recovery": None,
            "gt_validation_status": "not_provided",
            "gt_rmse_pixels": None,
            "gt_point_count": 0,
            "metadata_validation": {"status": "not_requested"},
            "mission_grade_certified": False,
            "holdout_validation": {"n_holdout": 5, "holdout_rmse": 0.55},
        }

    def test_execution_status_is_succeeded(self):
        from src.result_schema import RegistrationTelemetry
        t = RegistrationTelemetry.from_results(self._make_results(passed=True))
        assert t.execution_status == "succeeded"

    def test_scientific_status_passed(self):
        from src.result_schema import RegistrationTelemetry
        t = RegistrationTelemetry.from_results(self._make_results(passed=True))
        assert t.scientific_status == "passed_configured_checks"

    def test_scientific_status_rejected(self):
        from src.result_schema import RegistrationTelemetry
        t = RegistrationTelemetry.from_results(self._make_results(passed=False))
        assert t.scientific_status in (
            "rejected", "insufficient_evidence", "unsupported_input"
        )

    def test_rmse_ref_px_is_populated(self):
        from src.result_schema import RegistrationTelemetry
        t = RegistrationTelemetry.from_results(self._make_results())
        assert t.rmse_ref_px == pytest.approx(0.45, abs=1e-4)

    def test_rmse_src_px_is_distinct(self):
        from src.result_schema import RegistrationTelemetry
        t = RegistrationTelemetry.from_results(self._make_results())
        # rmse_raw_pixels = rmse * 2 in mock data
        assert t.rmse_src_px == pytest.approx(0.90, abs=1e-4)
        assert t.rmse_ref_px != t.rmse_src_px

    def test_evidence_class_held_out_when_holdout_present(self):
        from src.result_schema import RegistrationTelemetry
        t = RegistrationTelemetry.from_results(self._make_results())
        assert t.accuracy_evidence_class == "held_out"

    def test_evidence_class_fit_only_without_holdout(self):
        from src.result_schema import RegistrationTelemetry
        r = self._make_results()
        r["holdout_validation"] = {"n_holdout": 0}
        t = RegistrationTelemetry.from_results(r)
        assert t.accuracy_evidence_class == "fit_only"

    def test_evidence_class_independent_with_gt(self):
        from src.result_schema import RegistrationTelemetry
        r = self._make_results()
        r["gt_validation_status"] = "valid"
        r["gt_rmse_pixels"] = 0.3
        r["gt_point_count"] = 20
        t = RegistrationTelemetry.from_results(r)
        assert t.accuracy_evidence_class == "independently_validated"

    def test_legacy_status_field_backward_compat(self):
        from src.result_schema import RegistrationTelemetry
        t = RegistrationTelemetry.from_results(self._make_results(passed=True))
        assert t.status == "verified"   # backward compat
        t2 = RegistrationTelemetry.from_results(self._make_results(passed=False))
        assert t2.status == "rejected"

    def test_to_dict_contains_new_fields(self):
        from src.result_schema import RegistrationTelemetry
        d = RegistrationTelemetry.from_results(self._make_results()).to_dict()
        assert "execution_status" in d
        assert "scientific_status" in d
        assert "rmse_ref_px" in d
        assert "rmse_src_px" in d
        assert "accuracy_evidence_class" in d


# ---------------------------------------------------------------------------
# Artifact exporter (§9)
# ---------------------------------------------------------------------------
def _fake_results(passed=True, n_matches=30):
    """Minimal results dict for testing the artifact exporter."""
    inliers = n_matches if passed else 3
    pts = np.random.default_rng(0).random((n_matches, 2)).astype(np.float32) * 256
    mask = np.ones(n_matches, dtype=bool)
    if not passed:
        mask[5:] = False
    H = np.eye(3, dtype=np.float64)
    return {
        "quality": {
            "passed": passed,
            "grade": "A" if passed else "F",
            "confidence": 0.85 if passed else 0.0,
            "primary_reason": "QUALITY_PASSED" if passed else "INSUFFICIENT_INLIERS",
            "reason_codes": [],
            "warnings": [],
            "gate_telemetry": {},
            "details": {},
        },
        "method_used": "SIFT",
        "total_matches": n_matches,
        "inlier_count": inliers,
        "inlier_ratio": inliers / n_matches,
        "rmse_pixels": 0.4 if passed else None,
        "rmse_raw_pixels": 0.8 if passed else None,
        "coverage": 0.6,
        "spatial_entropy": 0.72,
        "auto_recovery": None,
        "gt_validation_status": "not_provided",
        "gt_rmse_pixels": None,
        "gt_point_count": 0,
        "metadata_validation": {"status": "not_requested"},
        "mission_grade_certified": False,
        "holdout_validation": {"n_holdout": 6, "n_train": 14, "holdout_rmse": 0.5},
        "error_stats": {},
        "error_budget": {},
        "pipeline_times": {},
        "provenance": {},
        "recovery_attempts": [],
        "symmetric_rmse": None,
        "condition_number": None,
        "geometric_model_type": "HOMOGRAPHY",
        "fallback_code": None,
        "pts_src": pts,
        "pts_ref": pts + 0.3,
        "inliers_src": pts[:inliers],
        "inliers_ref": (pts + 0.3)[:inliers],
        "mask": mask,
        "confidences": np.ones(n_matches, dtype=np.float32),
        "H_fallback": H,
        "transform": H,
    }


class TestArtifactExporter:
    def test_matches_csv_written(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        r = _fake_results()
        arts = export_all_artifacts(str(tmp_path), r)
        csv_path = arts.get("matches.csv")
        assert csv_path and os.path.exists(csv_path)
        with open(csv_path) as f:
            header = f.readline()
        assert "x_moving_px" in header
        assert "is_inlier" in header
        assert "engine" in header
        assert "residual_ref_px" in header
        assert "validation_split" in header

    def test_transform_json_written(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        r = _fake_results()
        arts = export_all_artifacts(str(tmp_path), r)
        tj = arts.get("transform.json")
        assert tj and os.path.exists(tj)
        with open(tj) as f:
            data = json.load(f)
        assert data["direction"] == "moving_to_fixed"
        assert "coordinate_convention" in data
        assert "inversion" in data

    def test_quality_json_evidence_class_labelled(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        r = _fake_results()
        arts = export_all_artifacts(str(tmp_path), r)
        qj = arts.get("quality.json")
        assert qj and os.path.exists(qj)
        with open(qj) as f:
            data = json.load(f)
        ae = data["accuracy_evidence"]
        assert "class" in ae
        assert ae["class"] in ("fit_only", "held_out", "independently_validated")
        # Source-pixel and reference-pixel RMSE must be separate named fields
        assert "fit_rmse_ref_px" in ae
        assert "fit_rmse_src_px" in ae
        # Must have limitations listed
        assert len(data.get("limitations", [])) >= 3

    def test_provenance_json_written(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        r = _fake_results()
        arts = export_all_artifacts(str(tmp_path), r)
        pj = arts.get("provenance.json")
        assert pj and os.path.exists(pj)
        with open(pj) as f:
            data = json.load(f)
        assert "_schema" in data

    def test_registered_source_written_when_passed(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        warped = np.zeros((64, 64), dtype=np.uint8)
        r = _fake_results(passed=True)
        arts = export_all_artifacts(str(tmp_path), r, warped_img=warped)
        rs = arts.get("registered_source.tif")
        assert rs is not None and os.path.exists(rs), \
            "registered_source.tif must be written for accepted results"

    def test_registered_source_NOT_written_when_failed(self, tmp_path):
        """§9: rejected results must NOT produce a registered product."""
        from src.registration.artifact_exporter import export_all_artifacts
        warped = np.zeros((64, 64), dtype=np.uint8)
        r = _fake_results(passed=False)
        arts = export_all_artifacts(str(tmp_path), r, warped_img=warped,
                                    rejected_preview=False)
        rs = arts.get("registered_source.tif")
        assert rs is None, \
            "registered_source.tif must NOT be written when quality gate fails"
        # The file must also not physically exist
        assert not os.path.exists(str(tmp_path / "registered_source.tif"))

    def test_unvalidated_preview_written_when_flag_set(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        warped = np.zeros((64, 64), dtype=np.uint8)
        r = _fake_results(passed=False)
        arts = export_all_artifacts(str(tmp_path), r, warped_img=warped,
                                    rejected_preview=True)
        preview = arts.get("registered_source_UNVALIDATED.tif")
        assert preview and os.path.exists(preview), \
            "UNVALIDATED preview must be written when rejected_preview=True"
        # The canonical accepted product must still NOT exist
        assert not os.path.exists(str(tmp_path / "registered_source.tif"))

    def test_valid_mask_written_for_accepted(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        warped = np.full((64, 64), 128, dtype=np.uint8)
        r = _fake_results(passed=True)
        arts = export_all_artifacts(str(tmp_path), r, warped_img=warped)
        vm = arts.get("valid_mask.tif")
        assert vm and os.path.exists(vm)

    def test_support_mask_written(self, tmp_path):
        from src.registration.artifact_exporter import export_all_artifacts
        warped = np.full((64, 64), 128, dtype=np.uint8)
        r = _fake_results(passed=True)
        arts = export_all_artifacts(str(tmp_path), r, warped_img=warped)
        sm = arts.get("registration_support_mask.tif")
        assert sm and os.path.exists(sm)

    def test_empty_matches_csv_has_header(self, tmp_path):
        from src.registration.artifact_exporter import export_matches_csv
        p = str(tmp_path / "empty.csv")
        export_matches_csv(p, np.empty((0, 2)), np.empty((0, 2)))
        with open(p) as f:
            header = f.readline()
        # The empty-path writes a shorter header; verify expected columns are present
        assert "x_moving" in header
        assert "is_inlier" in header
        assert "engine" in header



# ---------------------------------------------------------------------------
# Blender renderer — terrain_frame.json (§13.4)
# ---------------------------------------------------------------------------
class TestBlenderTerrainFrame:
    def test_terrain_frame_json_content(self, tmp_path):
        from src.rendering.blender_renderer import export_terrain_frame_json
        path = str(tmp_path / "terrain_frame.json")
        export_terrain_frame_json(
            path=path,
            dem_shape=(256, 512),
            local_origin_lon_deg=42.3,
            local_origin_lat_deg=-15.1,
            pixel_size_m=5.0,
            vertical_exaggeration_science=1.0,
            vertical_exaggeration_stylized=2.0,
        )
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["units"] == "metres"
        assert data["local_frame"]["X"] == "lunar_east"
        assert data["local_frame"]["Y"] == "lunar_north"
        assert data["local_frame"]["Z"] == "up_radial"
        assert data["local_origin"]["longitude_deg"] == pytest.approx(42.3)
        assert data["vertical_exaggeration"]["science_mesh"] == 1.0
        assert data["vertical_exaggeration"]["stylized_mesh"] == 2.0
        assert data["local_to_body_fixed"]["available"] is True

    def test_terrain_frame_json_ungeoreferenced(self, tmp_path):
        from src.rendering.blender_renderer import export_terrain_frame_json
        path = str(tmp_path / "frame.json")
        export_terrain_frame_json(path=path, dem_shape=(128, 128))
        with open(path) as f:
            data = json.load(f)
        assert data["local_origin"]["longitude_deg"] is None
        assert data["local_to_body_fixed"]["available"] is False

    def test_default_vertical_exaggeration_is_1(self):
        """§13.4: the public API default must be exactly 1.0, not 1.8."""
        import inspect
        from src.rendering.blender_renderer import render_lunar_dem_with_blender
        sig = inspect.signature(render_lunar_dem_with_blender)
        ve_default = sig.parameters["vertical_exaggeration"].default
        assert ve_default == 1.0, (
            f"§13.4 requires default vertical_exaggeration=1.0; got {ve_default}"
        )


# ---------------------------------------------------------------------------
# Pipeline §9 Artifact Export Integration Tests
# ---------------------------------------------------------------------------
class TestPipelineArtifactExport:
    def test_pipeline_exports_artifacts_when_output_dir_provided(self, tmp_path):
        import cv2
        from src.pipeline import run_registration_pipeline
        # Create a synthetic image with distinct textured patterns
        img = np.zeros((200, 200), dtype=np.uint8)
        for i in range(10, 190, 20):
            for j in range(10, 190, 20):
                cv2.circle(img, (i, j), 4, 255, -1)

        src_file = str(tmp_path / "src.png")
        ref_file = str(tmp_path / "ref.png")
        cv2.imwrite(src_file, img)
        cv2.imwrite(ref_file, img)

        out_dir = str(tmp_path / "artifacts")
        _, _, results = run_registration_pipeline(
            src_path=src_file,
            ref_path=ref_file,
            matcher_type="sift",
            max_dim=256,
            auto_recover=False,
            output_dir=out_dir,
        )

        assert "exported_artifacts" in results
        exported = results["exported_artifacts"]
        assert "matches.csv" in exported and os.path.exists(exported["matches.csv"])
        assert "transform.json" in exported and os.path.exists(exported["transform.json"])
        assert "quality.json" in exported and os.path.exists(exported["quality.json"])
        assert "provenance.json" in exported and os.path.exists(exported["provenance.json"])

    def test_pipeline_rejected_pair_does_not_write_registered_source(self, tmp_path):
        import cv2
        from src.pipeline import run_registration_pipeline
        # Featureless blank images ensure failure
        src_file = str(tmp_path / "blank_src.png")
        ref_file = str(tmp_path / "blank_ref.png")
        cv2.imwrite(src_file, np.zeros((100, 100), dtype=np.uint8))
        cv2.imwrite(ref_file, np.zeros((100, 100), dtype=np.uint8))

        out_dir = str(tmp_path / "rejected_artifacts")
        _, _, results = run_registration_pipeline(
            src_path=src_file,
            ref_path=ref_file,
            matcher_type="sift",
            max_dim=128,
            auto_recover=False,
            output_dir=out_dir,
            rejected_preview=False,
        )
        assert results["quality"]["passed"] is False
        registered_tif = os.path.join(out_dir, "registered_source.tif")
        assert not os.path.exists(registered_tif), "Rejected registration must not write registered_source.tif"

