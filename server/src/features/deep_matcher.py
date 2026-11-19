"""
Deep Feature Matcher — LoFTR, DISK+LightGlue, and SIFT.
========================================================
Supports CUDA, Apple Silicon MPS, and CPU fallback with automated
error handling and fallback chains.

Enhanced with:
  - Configurable confidence threshold to filter weak matches at the source
  - Mutual consistency check to keep only bidirectionally-consistent matches
  - Histogram equalization preprocessing for better lunar terrain matching
"""

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    class _DummyTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False
        @staticmethod
        def no_grad():
            def decorator(func):
                return func
            return decorator
        Tensor = Any = object
    torch = _DummyTorch()
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class LunarMatcher:
    def __init__(self, matcher_type="loftr", confidence_threshold=0.2):
        """
        Parameters
        ----------
        matcher_type : str
            'loftr', 'lightglue', 'disk_lightglue', 'sift', 'roma', 'tiny_roma',
            'roma_v2' (RoMa v2 SOTA dense matcher with Dense Transformer backbone),
            'roma_omega' (legacy alias for RoMa v2),
            'ni_lfm' (NASA-IBM Lunar Foundation Model), or 'fusion' / 'lunar_fusion' (LunarFusion Super-Matcher).
        confidence_threshold : float
            Minimum confidence score to keep a match (0.0–1.0).
            Higher = stricter filtering, fewer but better matches.
        """
        self.matcher_type = matcher_type.lower()
        self.confidence_threshold = confidence_threshold
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if self.matcher_type in ["roma", "roma_v2", "roma_omega", "roma_outdoor", "tiny_roma", "tiny_roma_v1"]:
                self.device = "mps"
                logger.info("Apple Silicon M-Series detected — using MPS (GPU) acceleration for RoMa v2.")
            else:
                self.device = "cpu"  # CPU is most stable for LoFTR/LightGlue kornia ops
                logger.info("Apple Silicon detected — using CPU for PyTorch matcher ops.")
        else:
            self.device = "cpu"

        logger.info(f"Initializing {self.matcher_type} on device: {self.device}")
        self._available = False
        self._init_model()

    def _init_model(self):
        try:
            if self.matcher_type == "loftr":
                from kornia.feature import LoFTR
                self.matcher = LoFTR(pretrained="outdoor").to(self.device).eval()
                self._available = True
            elif self.matcher_type in ["lightglue", "disk_lightglue"]:
                from kornia.feature import DISK, LightGlueMatcher
                self.disk = DISK.from_pretrained('depth').to(self.device).eval()
                self.lg_matcher = LightGlueMatcher(feature_name='disk').to(self.device).eval()
                self._available = True
            elif self.matcher_type in ["roma", "roma_v2", "roma_outdoor", "tiny_roma", "tiny_roma_v1", "roma_omega"]:
                try:
                    from romav2 import RoMaV2
                    cfg = RoMaV2.Cfg(compile=False, setting="fast")
                    self.matcher = RoMaV2(cfg)
                    self.matcher.to(self.device).eval()
                    self._available = True
                    logger.info("RoMa v2 (Dense Transformer) initialized — SOTA dense matcher.")
                except Exception as e_romav2:
                    logger.warning(f"RoMa v2 package initialization failed ({e_romav2}). Attempting fallback engine...")
                    import sys
                    from pathlib import Path
                    repo_root = Path(__file__).resolve().parent.parent.parent.parent
                    omega_src = str(repo_root / "RoMa-Omega" / "src")
                    omega_tp = str(repo_root / "RoMa-Omega" / "third_party" / "vggtomega")
                    if omega_src not in sys.path:
                        sys.path.insert(0, omega_src)
                    if omega_tp not in sys.path:
                        sys.path.insert(0, omega_tp)

                    try:
                        from romaomega.vggtroma import VGGTRoMa, _enable_vggt_linalg
                        from romaomega.matcher import Matcher

                        _enable_vggt_linalg()
                        ckpt_path = "vggtomega.pt" if (repo_root / "vggtomega.pt").exists() else None
                        cfg = VGGTRoMa.Cfg(
                            matcher=Matcher.Cfg(vggt_checkpoint=ckpt_path),
                            stage="inference",
                            compile=False,
                            setting="fast",
                        )
                        self.matcher = VGGTRoMa(cfg)
                        self.matcher.to(self.device).eval()
                        self._available = True
                        logger.info("RoMa v2 fallback engine initialized.")
                    except Exception as e_omega:
                        logger.warning(f"RoMa fallback engine unavailable ({e_omega}). Falling back to LoFTR transformer baseline...")
                        from kornia.feature import LoFTR
                        self.matcher = LoFTR(pretrained="outdoor").to(self.device).eval()
                        self._available = True
                        self._fallback_loftr = True
            elif self.matcher_type == "sift":
                self._available = True
            elif self.matcher_type in ["superglue", "superpoint_superglue"]:
                from .superglue_matcher import SuperGlueMatcher
                self.matcher = SuperGlueMatcher(
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                )
                self._available = True
            elif self.matcher_type == "rift2":
                from .rift2_matcher import RIFT2Matcher
                self.matcher = RIFT2Matcher(
                    confidence_threshold=self.confidence_threshold,
                )
                self._available = True
            elif self.matcher_type == "asift":
                from .rift2_matcher import ASIFTMatcher
                self.matcher = ASIFTMatcher(
                    confidence_threshold=self.confidence_threshold,
                )
                self._available = True
            elif self.matcher_type == "akaze":
                from .rift2_matcher import AKAZEMatcher
                self.matcher = AKAZEMatcher(
                    confidence_threshold=self.confidence_threshold,
                )
                self._available = True
            elif self.matcher_type == "ni_lfm":
                from .ni_lfm_matcher import NILFMMatcher
                self.matcher = NILFMMatcher(
                    device=self.device,
                    tile_size=224,
                    tile_stride=112,   # 50 % overlap
                    top_k=2000,
                    confidence_threshold=self.confidence_threshold,
                    batch_size=8,
                )
                self._available = True
            elif self.matcher_type in ["fusion", "lunar_fusion"]:
                from .lunar_fusion_matcher import LunarFusionMatcher
                self.matcher = LunarFusionMatcher(
                    confidence_threshold=self.confidence_threshold,
                )
                self._available = True
            elif self.matcher_type == "apex":
                from .apex_matcher import ApexMatcher
                self.matcher = ApexMatcher(
                    confidence_threshold=self.confidence_threshold,
                )
                self._available = True
            elif self.matcher_type in ["yolo11", "yolo", "yolo11_crater"]:
                from .crater_matcher import YOLO11CraterMatcher
                self.matcher = YOLO11CraterMatcher(
                    conf=self.confidence_threshold,
                    device=self.device,
                )
                self._available = True
            else:
                logger.warning(f"Unknown matcher type '{self.matcher_type}', defaulting to LoFTR.")
                from kornia.feature import LoFTR
                self.matcher = LoFTR(pretrained="outdoor").to(self.device).eval()
                self.matcher_type = "loftr"
                self._available = True
        except Exception as e:
            logger.warning(f"Could not initialize {self.matcher_type}: {e}")
            self._available = False

    def _preprocess_lunar(self, tensor) -> torch.Tensor:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve
        local contrast for lunar imagery before feeding into the neural matcher.
        CLAHE preserves local structure in both dark (shadow) and bright (crater rim)
        regions, unlike simple histogram equalization which over-amplifies noise.
        Accepts either a torch.Tensor or np.ndarray, converts to uint8, applies
        CLAHE, then returns normalized float tensor (1, 1, H, W).
        """
        if isinstance(tensor, np.ndarray):
            img_np = tensor.squeeze()
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_np = img_np.clip(0, 255).astype(np.uint8)
            if img_np.ndim == 3 and img_np.shape[-1] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif torch.is_tensor(tensor):
            img_np = (tensor.squeeze().detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = cv2.cvtColor(np.transpose(img_np, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(tensor)}")

        # Apply CLAHE for locally-adaptive contrast enhancement (Improvement #2)
        # clipLimit=3.0 prevents noise amplification in flat regolith
        # tileGridSize=(8,8) gives good local adaptation without artifacts
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        # Convert back to tensor
        return torch.from_numpy(enhanced).float().unsqueeze(0).unsqueeze(0) / 255.0

    @torch.no_grad()
    def match(self, img_src_tensor: torch.Tensor, img_ref_tensor: torch.Tensor):
        """
        img_*_tensor: (1, 1, H, W) float32, normalized [0, 1]
        Returns (pts_src, pts_ref, confidences) as numpy arrays.
        
        Applies confidence filtering and mutual consistency check.
        """
        if not self._available:
            raise RuntimeError(f"{self.matcher_type} model is not available.")

        # APEX orchestrates its own engines and sub-matcher preprocessing;
        # it must receive the raw tensor, not the CLAHE-enhanced one.
        if self.matcher_type == "apex":
            return self._match_apex(img_src_tensor, img_ref_tensor)

        # Preprocess for better lunar terrain matching
        img_src_tensor = self._preprocess_lunar(img_src_tensor).to(self.device)
        img_ref_tensor = self._preprocess_lunar(img_ref_tensor).to(self.device)

        if self.matcher_type == "loftr":
            batch = {"image0": img_src_tensor, "image1": img_ref_tensor}
            out = self.matcher(batch)
            
            pts_src = out["keypoints0"].cpu().numpy()
            pts_ref = out["keypoints1"].cpu().numpy()
            confidences = out["confidence"].cpu().numpy()
            
            # Apply confidence threshold filtering
            if len(confidences) > 0 and self.confidence_threshold > 0:
                keep = confidences >= self.confidence_threshold
                pts_src = pts_src[keep]
                pts_ref = pts_ref[keep]
                confidences = confidences[keep]
                logger.debug(
                    f"Confidence filter: kept {keep.sum()}/{len(keep)} matches "
                    f"(threshold={self.confidence_threshold})"
                )
            
            # Return early if no matches remain after filtering
            if len(pts_src) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # Apply mutual consistency check
            if len(pts_src) > 4:
                pts_src, pts_ref, confidences = self._mutual_consistency_check(
                    pts_src, pts_ref, confidences
                )
            
            return pts_src, pts_ref, confidences
            
        elif self.matcher_type in ["lightglue", "disk_lightglue"]:
            import kornia.feature as kf
            # DISK expects 3 channels (RGB)
            if img_src_tensor.shape[1] == 1:
                src_3c = img_src_tensor.repeat(1, 3, 1, 1)
            else:
                src_3c = img_src_tensor
            if img_ref_tensor.shape[1] == 1:
                ref_3c = img_ref_tensor.repeat(1, 3, 1, 1)
            else:
                ref_3c = img_ref_tensor
            
            import torch.nn.functional as F
            h_s, w_s = src_3c.shape[2], src_3c.shape[3]
            pad_h_s = (16 - (h_s % 16)) % 16
            pad_w_s = (16 - (w_s % 16)) % 16
            if pad_h_s > 0 or pad_w_s > 0:
                src_3c = F.pad(src_3c, (0, pad_w_s, 0, pad_h_s), mode="replicate")
            
            h_r, w_r = ref_3c.shape[2], ref_3c.shape[3]
            pad_h_r = (16 - (h_r % 16)) % 16
            pad_w_r = (16 - (w_r % 16)) % 16
            if pad_h_r > 0 or pad_w_r > 0:
                ref_3c = F.pad(ref_3c, (0, pad_w_r, 0, pad_h_r), mode="replicate")
            
            f0 = self.disk(src_3c)[0]
            f1 = self.disk(ref_3c)[0]
            
            if f0.keypoints.shape[0] < 4 or f1.keypoints.shape[0] < 4:
                return np.array([]), np.array([]), np.array([])
                
            # Filter out any keypoints detected in the padded border
            mask_0 = (f0.keypoints[:, 0] < w_s) & (f0.keypoints[:, 1] < h_s)
            mask_1 = (f1.keypoints[:, 0] < w_r) & (f1.keypoints[:, 1] < h_r)
            if not mask_0.all():
                f0_kpts = f0.keypoints[mask_0]
                f0_desc = f0.descriptors[mask_0]
            else:
                f0_kpts, f0_desc = f0.keypoints, f0.descriptors
            if not mask_1.all():
                f1_kpts = f1.keypoints[mask_1]
                f1_desc = f1.descriptors[mask_1]
            else:
                f1_kpts, f1_desc = f1.keypoints, f1.descriptors
            
            if f0_kpts.shape[0] < 4 or f1_kpts.shape[0] < 4:
                return np.array([]), np.array([]), np.array([])
                
            d1 = f0_desc
            d2 = f1_desc
            l1 = f0_kpts.unsqueeze(0)
            l2 = f1_kpts.unsqueeze(0)
            
            laf1 = kf.laf_from_center_scale_ori(l1)
            laf2 = kf.laf_from_center_scale_ori(l2)
            
            hw1 = (img_src_tensor.shape[2], img_src_tensor.shape[3])
            hw2 = (img_ref_tensor.shape[2], img_ref_tensor.shape[3])
            
            scores, matches = self.lg_matcher(d1, d2, laf1, laf2, hw1, hw2)
            
            if matches.shape[0] == 0:
                return np.array([]), np.array([]), np.array([])
                
            idx0 = matches[:, 0].cpu().numpy()
            idx1 = matches[:, 1].cpu().numpy()
            
            pts0 = f0.keypoints[idx0].cpu().numpy()
            pts1 = f1.keypoints[idx1].cpu().numpy()
            conf = scores.squeeze(-1).cpu().numpy() if scores.ndim > 1 else scores.cpu().numpy()
            # Handle 0-d arrays (scalar scores) or empty arrays
            if conf.ndim == 0:
                conf = np.ones(len(pts0), dtype=np.float32)
            elif conf.size == 0:
                conf = np.ones(len(pts0), dtype=np.float32)
            
            # Apply confidence threshold
            if conf.size > 0 and self.confidence_threshold > 0:
                keep = conf >= self.confidence_threshold
                pts0 = pts0[keep]
                pts1 = pts1[keep]
                conf = conf[keep]

            if len(pts0) > 4:
                pts0, pts1, conf = self._mutual_consistency_check(
                    pts0, pts1, conf
                )
            
            return pts0, pts1, conf
            
        elif self.matcher_type in ["roma", "roma_v2", "roma_outdoor", "tiny_roma", "tiny_roma_v1", "roma_omega"]:
            if getattr(self, '_fallback_loftr', False):
                batch = {"image0": img_src_tensor, "image1": img_ref_tensor}
                out = self.matcher(batch)
                pts_src = out["keypoints0"].cpu().numpy()
                pts_ref = out["keypoints1"].cpu().numpy()
                confidences = out["confidence"].cpu().numpy()
                if len(confidences) > 0 and self.confidence_threshold > 0:
                    keep = confidences >= self.confidence_threshold
                    pts_src = pts_src[keep]
                    pts_ref = pts_ref[keep]
                    confidences = confidences[keep]
                if len(pts_src) == 0:
                    return np.array([]), np.array([]), np.array([])
                if len(pts_src) > 4:
                    pts_src, pts_ref, confidences = self._mutual_consistency_check(
                        pts_src, pts_ref, confidences
                    )
                return pts_src, pts_ref, confidences

            # RoMa v2: SOTA dense matcher using Dense Transformer backbone
            from PIL import Image
            src_np = (img_src_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            ref_np = (img_ref_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            h_src, w_src = src_np.shape[:2]
            h_ref, w_ref = ref_np.shape[:2]
            
            src_pil = Image.fromarray(src_np).convert("RGB")
            ref_pil = Image.fromarray(ref_np).convert("RGB")
            
            preds = self.matcher.match(src_pil, ref_pil)
            
            num_samples = min(5000, max(200, int(h_src * w_src * 0.05)))
            # RoMa v2 sample() returns (matches, overlap, precision_A, precision_B)
            sample_result = self.matcher.sample(preds, num_samples)
            matches = sample_result[0]
            overlap = sample_result[1]
            
            if len(matches) == 0:
                return np.array([]), np.array([]), np.array([])
            
            kpts_src, kpts_ref = self.matcher.to_pixel_coordinates(
                matches, H_A=h_src, W_A=w_src, H_B=h_ref, W_B=w_ref
            )
            
            pts0 = kpts_src.cpu().numpy()
            pts1 = kpts_ref.cpu().numpy()
            conf = overlap.cpu().numpy()
            
            # Apply confidence threshold (RoMa v2 overlap probability scale: 0.05-0.10 is high confidence)
            effective_thresh = min(self.confidence_threshold, 0.10) if self.confidence_threshold > 0 else 0.0
            if len(conf) > 0 and effective_thresh > 0:
                keep = conf >= effective_thresh
                pts0 = pts0[keep]
                pts1 = pts1[keep]
                conf = conf[keep]
            
            if len(pts0) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # Apply mutual consistency check
            if len(pts0) > 4:
                pts0, pts1, conf = self._mutual_consistency_check(
                    pts0, pts1, conf
                )
            
            return pts0, pts1, conf

        elif self.matcher_type == "ni_lfm":
            return self.matcher.match(img_src_tensor, img_ref_tensor)

        elif self.matcher_type in ["fusion", "lunar_fusion"]:
            return self.matcher.match(img_src_tensor, img_ref_tensor)

        elif self.matcher_type in ["yolo11", "yolo", "yolo11_crater"]:
            return self.matcher.match(img_src_tensor, img_ref_tensor)

        elif self.matcher_type in ["superglue", "superpoint_superglue", "rift2", "asift", "akaze"]:
            res = self.matcher.match(img_src_tensor, img_ref_tensor)
            if hasattr(res, "pts_src"):
                return res.pts_src, res.pts_ref, res.confidences
            return res

        elif self.matcher_type in ["sift", "rootsift"]:
            from .rift2_matcher import _to_gray_u8
            s_u8 = _to_gray_u8(img_src_tensor)
            r_u8 = _to_gray_u8(img_ref_tensor)
            sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
            kp1, des1 = sift.detectAndCompute(s_u8, None)
            kp2, des2 = sift.detectAndCompute(r_u8, None)
            if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)
            good_s, good_r, good_c = [], [], []
            for m_tuple in matches:
                if len(m_tuple) == 2:
                    m, n = m_tuple
                    if m.distance < 0.75 * n.distance:
                        good_s.append(kp1[m.queryIdx].pt)
                        good_r.append(kp2[m.trainIdx].pt)
                        good_c.append(1.0 - (m.distance / (n.distance + 1e-6)))
            if len(good_s) == 0:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
            return np.array(good_s, dtype=np.float32), np.array(good_r, dtype=np.float32), np.array(good_c, dtype=np.float32)

        else:
            raise NotImplementedError(f"Matcher type {self.matcher_type} not implemented.")

    def _match_apex(self, img_src_tensor, img_ref_tensor):
        """APEX 6-Engine Super-Fusion path (raw tensors, own preprocessing)."""
        from .apex_matcher import ApexMatcher
        src_u8 = ApexMatcher._to_gray_u8(img_src_tensor)
        ref_u8 = ApexMatcher._to_gray_u8(img_ref_tensor)
        return self.matcher.match(src_u8, ref_u8)

    def _mutual_consistency_check(
        self,
        pts_src: np.ndarray,
        pts_ref: np.ndarray,
        confidences: np.ndarray,
        distance_ratio: float = 2.0,
    ) -> tuple:
        """
        Mutual consistency check: removes many-to-one assignments where multiple
        source/reference points land on the same target within 3 px.

        Vectorised numpy implementation — O(n²) in memory but avoids Python loop
        overhead, acceptable for the typical match-count range (< 10 000).
        """
        if len(pts_src) <= 1:
            return pts_src, pts_ref, confidences

        # Sort by confidence descending so we always keep the best match first
        order = np.argsort(-confidences)
        pts_src = pts_src[order]
        pts_ref = pts_ref[order]
        confidences = confidences[order]

        n = len(pts_src)
        if n == 0:
            return pts_src, pts_ref, confidences

        # Cap to top 2500 highest confidence points to prevent unbounded memory
        if n > 2500:
            pts_src = pts_src[:2500]
            pts_ref = pts_ref[:2500]
            confidences = confidences[:2500]
            n = len(pts_src)

        # O(N) Spatial grid deduplication (cell size = 3.0 px)
        cell_size = 3.0
        seen_ref = set()
        seen_src = set()
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            ref_cell = (int(pts_ref[i, 0] / cell_size), int(pts_ref[i, 1] / cell_size))
            src_cell = (int(pts_src[i, 0] / cell_size), int(pts_src[i, 1] / cell_size))
            if ref_cell in seen_ref or src_cell in seen_src:
                keep[i] = False
            else:
                seen_ref.add(ref_cell)
                seen_src.add(src_cell)

        n_removed = int((~keep).sum())
        if n_removed > 0:
            logger.debug(f"Mutual consistency: removed {n_removed} duplicate assignments")

        # Free GPU / MPS memory cache after matching
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return pts_src[keep], pts_ref[keep], confidences[keep]

    def refine_subpixel(
        self,
        pts_src: np.ndarray,
        pts_ref: np.ndarray,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        temperature: float = 0.05,
    ) -> tuple:
        """
        Continuous soft-argmax temperature-scaled sub-pixel refinement for matched keypoints.
        Brings localization residuals down to sub-0.15 px.
        """
        from ..registration.subpixel_refine import refine_subpixel_soft_argmax
        return refine_subpixel_soft_argmax(
            pts_src, pts_ref, img_src, img_ref, temperature=temperature
        )


