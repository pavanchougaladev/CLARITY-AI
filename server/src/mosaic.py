"""
Multi-Image Mosaicking Engine
==============================
Registers and stitches multiple overlapping lunar images into a single
continuous mosaic using feature matching, global canvas projection,
and distance-weighted feather blending.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


from .matching.outlier_rejection import estimate_transform
from .features.root_sift import root_sift_descriptors


class MosaicEngine:
    def __init__(self):
        self.images: List[np.ndarray] = []
        self.image_names: List[str] = []
        self.transforms: List[np.ndarray] = []  # Transform H_i mapping image i to base image 0

    def add_image(self, img: np.ndarray, name: str = ""):
        """Add an image array (grayscale uint8) to the mosaic collection."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.images.append(img.astype(np.uint8))
        self.image_names.append(name or f"image_{len(self.images)}")

    def _match_pair(self, img_src: np.ndarray, img_ref: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates homography mapping img_src coordinates to img_ref coordinates
        using CLAHE contrast normalization and the progressive MAGSAC++ model cascade.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        src_proc = clahe.apply(img_src)
        ref_proc = clahe.apply(img_ref)

        sift = cv2.SIFT_create(nfeatures=5000)
        kp1, des1 = sift.detectAndCompute(src_proc, None)
        kp2, des2 = sift.detectAndCompute(ref_proc, None)
        des1 = root_sift_descriptors(des1)  # Upgrade to RootSIFT (Hellinger metric)
        des2 = root_sift_descriptors(des2)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m_list in matches:
            if len(m_list) == 2 and m_list[0].distance < 0.80 * m_list[1].distance:
                good.append(m_list[0])

        if len(good) < 4:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        ref_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        H, inliers = estimate_transform(src_pts, ref_pts, use_tps=False)
        if H is not None and inliers is not None and np.sum(inliers) >= 4:
            if hasattr(H, 'shape') and H.shape == (3, 3):
                return H
        return None

    def compute_mosaic(self, base_idx: int = 0) -> Tuple[np.ndarray, dict]:
        """
        Calculates pairwise alignments relative to base image,
        allocates a unified canvas, and blends images using distance feathering.
        """
        n = len(self.images)
        if n == 0:
            raise ValueError("No images provided for mosaic generation.")
        if n == 1:
            return self.images[0], {"success": True, "count": 1, "canvas_size": self.images[0].shape}

        # Step 1: Compute transformations mapping each image to base_idx (Image 0)
        transforms = [None] * n
        transforms[base_idx] = np.eye(3, dtype=np.float64)

        for i in range(n):
            if i == base_idx:
                continue
            # Try matching with base image directly first
            H = self._match_pair(self.images[i], self.images[base_idx])
            if H is None and i > 0 and transforms[i - 1] is not None:
                # Chain transform via neighbor (i-1)
                H_neighbor = self._match_pair(self.images[i], self.images[i - 1])
                if H_neighbor is not None:
                    H = np.dot(transforms[i - 1], H_neighbor)

            if H is not None:
                transforms[i] = H
            else:
                logger.warning(f"Could not align image {i} ({self.image_names[i]}) to mosaic base.")

        # Step 2: Compute overall bounding box in reference frame
        corners_list = []
        valid_indices = []

        for i in range(n):
            H = transforms[i]
            if H is None:
                continue
            valid_indices.append(i)
            h, w = self.images[i].shape[:2]
            corners = np.array([
                [0, 0, 1],
                [w, 0, 1],
                [w, h, 1],
                [0, h, 1]
            ], dtype=np.float64).T  # 3x4

            warped_corners = np.dot(H, corners)
            warped_corners /= warped_corners[2:3, :]  # Normalize homogenous
            corners_list.append(warped_corners[:2, :])

        if not corners_list:
            return self.images[0], {"success": False, "count": 1, "msg": "Alignment failed"}

        all_corners = np.hstack(corners_list)
        min_x = np.min(all_corners[0, :])
        max_x = np.max(all_corners[0, :])
        min_y = np.min(all_corners[1, :])
        max_y = np.max(all_corners[1, :])

        # Canvas bounds with margin
        canvas_w = int(np.ceil(max_x - min_x))
        canvas_h = int(np.ceil(max_y - min_y))

        # Safety bounds check to avoid OOM crashes on degenerate homographies
        max_limit = 16384
        if canvas_w <= 0 or canvas_h <= 0 or canvas_w > max_limit or canvas_h > max_limit:
            logger.warning(
                f"Mosaic canvas dimensions ({canvas_w}x{canvas_h}) out of valid range (1..{max_limit}). "
                "Limiting canvas to safe dimensions."
            )
            canvas_w = min(max(canvas_w, 256), max_limit)
            canvas_h = min(max(canvas_h, 256), max_limit)

        # Translation matrix to shift origin to (0, 0)
        T = np.array([
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Step 3: Warp each image onto the master canvas and accumulate with weights
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for idx in valid_indices:
            img = self.images[idx]
            H_final = np.dot(T, transforms[idx])
            warped = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h))
            
            # Mask of valid warped pixels
            mask = (warped > 0).astype(np.uint8)
            
            # Distance transform for feathering (smooth border transitions)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            max_d = np.max(dist)
            if max_d > 0:
                norm_d = dist / max_d
                # Cosine roll-off produces seamless blending without linear boundary creases
                weight = 0.5 * (1.0 - np.cos(np.pi * norm_d))
            else:
                weight = mask.astype(np.float32)

            canvas += warped.astype(np.float32) * weight
            weight_sum += weight

        # Step 4: Normalize blended pixels
        valid_pixels = weight_sum > 1e-4
        canvas[valid_pixels] /= weight_sum[valid_pixels]
        mosaic_out = np.clip(canvas, 0, 255).astype(np.uint8)

        meta = {
            "success": True,
            "count": len(valid_indices),
            "total_images": n,
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
            "aligned_images": [self.image_names[i] for i in valid_indices]
        }

        return mosaic_out, meta
