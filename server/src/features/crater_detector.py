"""
ISRO LunarFusion — Lunar Crater & Hazard Detector (YOLO11)
===========================================================
Provides fast, deep learning-based detection of lunar craters and boulders
using fine-tuned Ultralytics YOLO11 models.

Features:
- Sub-10ms inference per tile using MPS (Apple Silicon Metal) or CUDA
- Centroid and physical diameter extraction in meters using GSD
- Spatial tie-point coverage calculation for AGENTS.md Gate 3 validation
- Visual annotation rendering for Streamlit dashboard overlays
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import cv2
import torch

# Optional heavy dependency — loaded lazily so the module can be imported
# in test environments that do not have ultralytics installed.
try:
    from ultralytics import YOLO as _YOLO  # type: ignore
    _ULTRALYTICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YOLO = None  # type: ignore
    _ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/yolo11_lunar_crater.pt"


class LunarCraterDetector:
    """
    Lunar Crater and Hazard Detector based on fine-tuned YOLO11.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize the detector with weights and hardware acceleration.
        """
        self.conf = conf
        self.iou = iou
        self.model_path = model_path

        # Determine target device
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "0"
        else:
            self.device = "cpu"

        # Check model path
        if not os.path.exists(model_path):
            logger.warning(
                f"Trained model '{model_path}' not found. Falling back to base 'yolo11n.pt'."
            )
            self.model_path = "yolo11n.pt"

        # Guard: require ultralytics at instantiation time, not import time.
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "LunarCraterDetector requires 'ultralytics>=8.3.0'. "
                "Install it with: pip install 'ultralytics>=8.3.0'"
            )

        try:
            self.model = _YOLO(self.model_path)
            logger.info(f"Loaded Lunar YOLO11 model from {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}. Attempting fallback.")
            self.model = _YOLO("yolo11n.pt")

    def detect(
        self,
        image: Union[str, np.ndarray],
        conf: Optional[float] = None,
        imgsz: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Detect craters and hazards in a single image or crop.

        Parameters
        ----------
        image : str or np.ndarray
            File path or BGR/Grayscale image array.
        conf : float, optional
            Override confidence threshold.
        imgsz : int
            Inference resolution.

        Returns
        -------
        List of dicts with box, centroid, confidence, diameter_px, class_name.
        """
        confidence = conf if conf is not None else self.conf

        # Convert grayscale to 3-channel if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 1:
                img_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img_input = image
        else:
            img_input = image

        try:
            results = self.model.predict(
                source=img_input,
                conf=confidence,
                iou=self.iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning(f"Inference on {self.device} failed ({e}). Falling back to CPU.")
                self.device = "cpu"
                results = self.model.predict(
                    source=img_input,
                    conf=confidence,
                    iou=self.iou,
                    imgsz=imgsz,
                    device="cpu",
                    verbose=False
                )
            else:
                raise e

        detections = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = res.names
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = float(confs[i])
            cls_id = int(classes[i])
            cls_name = names.get(cls_id, "crater")

            w = float(x2 - x1)
            h = float(y2 - y1)
            cx = float(x1 + w / 2.0)
            cy = float(y1 + h / 2.0)
            diameter = max(w, h)

            detections.append({
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "centroid": (cx, cy),
                "width": w,
                "height": h,
                "diameter_px": diameter,
                "confidence": score,
                "class_id": cls_id,
                "class_name": cls_name,
            })

        return detections

    def get_crater_centroids(
        self,
        image: Union[str, np.ndarray],
        conf: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract only crater centroid coordinates (x, y) for geometric hashing or constellation matching.

        Returns
        -------
        np.ndarray of shape (N, 2), or empty array of shape (0, 2).
        """
        detections = self.detect(image, conf=conf)
        craters = [d for d in detections if d["class_name"].lower() == "crater"]
        if not craters:
            return np.empty((0, 2), dtype=np.float32)

        coords = np.array([c["centroid"] for c in craters], dtype=np.float32)
        return coords

    def compute_crater_coverage(
        self,
        tie_points: np.ndarray,
        craters: List[Dict[str, Any]],
        radius_multiplier: float = 1.0,
        min_inliers_per_crater: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate physical crater coverage for AGENTS.md Gate 3 validation.
        Verifies whether tie-points are anchored to actual lunar craters rather
        than clustered in a single flat region.

        Parameters
        ----------
        tie_points : np.ndarray
            (N, 2) array of tie-point coordinates (x, y).
        craters : List[Dict[str, Any]]
            Detections returned by `detect()`.
        radius_multiplier : float
            Tolerance multiplier for crater bounding circle.
        min_inliers_per_crater : int
            Minimum tie-points required inside a crater to count it as covered.

        Returns
        -------
        Dict with covered_count, total_craters, coverage_ratio, passed_gate_3.
        """
        crater_list = [c for c in craters if c["class_name"].lower() == "crater"]
        total_craters = len(crater_list)

        if total_craters == 0:
            return {
                "total_craters": 0,
                "covered_craters": 0,
                "coverage_ratio": 0.0,
                "passed_gate_3": True,  # Non-blocking if scene has zero detectable craters
                "note": "No craters detected in reference frame"
            }

        if len(tie_points) == 0:
            return {
                "total_craters": total_craters,
                "covered_craters": 0,
                "coverage_ratio": 0.0,
                "passed_gate_3": False,
                "note": "Zero tie-points available"
            }

        pts = np.asarray(tie_points, dtype=np.float32)
        covered_indices = []

        for idx, cr in enumerate(crater_list):
            cx, cy = cr["centroid"]
            radius = (cr["diameter_px"] / 2.0) * radius_multiplier
            # Distance from all points to this centroid
            dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            inliers_inside = np.sum(dists <= radius)
            if inliers_inside >= min_inliers_per_crater:
                covered_indices.append(idx)

        covered_count = len(covered_indices)
        coverage_ratio = covered_count / total_craters
        # Gate 3 threshold: >= 15% coverage of the crater grid/landscape
        passed_gate_3 = coverage_ratio >= 0.15

        return {
            "total_craters": total_craters,
            "covered_craters": covered_count,
            "coverage_ratio": float(coverage_ratio),
            "passed_gate_3": bool(passed_gate_3),
            "covered_indices": covered_indices
        }

    def compute_crater_diameters_m(
        self,
        craters: List[Dict[str, Any]],
        gsd_m: float
    ) -> List[Dict[str, Any]]:
        """
        Convert pixel crater diameters into true metric meters using ground sampling distance.
        """
        updated = []
        for c in craters:
            item = dict(c)
            item["diameter_m"] = float(item["diameter_px"] * gsd_m)
            updated.append(item)
        return updated

    def draw_craters(
        self,
        image: np.ndarray,
        craters: List[Dict[str, Any]],
        tie_points: Optional[np.ndarray] = None,
        covered_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Draw crater bounding ellipses, centroids, and tie-point status on an RGB canvas.
        """
        if len(image.shape) == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        covered_set = set(covered_indices) if covered_indices is not None else set()

        for idx, cr in enumerate(craters):
            x1, y1, x2, y2 = [int(v) for v in cr["box"]]
            cx, cy = int(cr["centroid"][0]), int(cr["centroid"][1])
            is_covered = idx in covered_set

            # Covered craters in green, uncovered in cyan/yellow
            color = (0, 255, 0) if is_covered else (255, 200, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)

            # Label
            conf = cr["confidence"]
            cls_name = cr["class_name"]
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                canvas,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )

        # Draw tie-points if provided
        if tie_points is not None and len(tie_points) > 0:
            for pt in tie_points:
                px, py = int(pt[0]), int(pt[1])
                cv2.drawMarker(
                    canvas,
                    (px, py),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    6,
                    1
                )

        return canvas
