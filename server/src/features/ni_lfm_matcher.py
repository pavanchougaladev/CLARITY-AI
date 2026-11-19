"""
ni_lfm_matcher.py
=================
NASA-IBM Lunar Foundation Model (NI-LFM) based feature matcher.

Uses the LunarFM ViT-Base backbone (pretrained on 20 lunar modalities) to
extract patch-level embeddings from greyscale pushbroom strips, then matches
tiles across two images via cosine similarity.

Integration:
    Called by LunarMatcher(matcher_type="ni_lfm") in deep_matcher.py.
    Returns (pts_src, pts_ref, confidences) — same API as LoFTR / RoMa.

Tiling strategy:
    Pushbroom strips (e.g. 18,302 × 400 px) are split into overlapping
    tile_size × tile_size patches.  Each tile's centre pixel becomes the
    representative keypoint coordinate.  After matching, only cross-tile
    pairs whose cosine similarity exceeds `confidence_threshold` are kept.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    class _DummyTorch:
        @staticmethod
        def no_grad():
            def decorator(func):
                return func
            return decorator
        Tensor = Any = object
    torch = _DummyTorch()
    F = object()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve the NI-LFM source tree (sibling of this file's project root)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent   # project root (server/src/features/ → ../../..)
_NILFM_ROOT   = _PROJECT_ROOT / "NASA-IBM-Lunar-Foundation-Model-main"

if str(_NILFM_ROOT) not in sys.path:
    sys.path.insert(0, str(_NILFM_ROOT))

_BACKBONE_CKPT = _PROJECT_ROOT / "backbone" / "backbone" / "checkpoint.pt"
_BACKBONE_CFG  = _PROJECT_ROOT / "backbone" / "backbone" / "config.yaml"


# ---------------------------------------------------------------------------
# NILFMMatcher
# ---------------------------------------------------------------------------

class NILFMMatcher:
    """
    Tile-based lunar image matcher powered by the NASA-IBM Lunar Foundation
    Model backbone.

    Parameters
    ----------
    checkpoint_path : str | Path, optional
        Path to ``checkpoint.pt``.  Defaults to the project's
        ``backbone/backbone/checkpoint.pt``.
    config_path : str | Path, optional
        Path to ``config.yaml`` shipped with the checkpoint.  Defaults to
        ``backbone/backbone/config.yaml``.
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.
    tile_size : int
        Side length (px) of each square tile fed to the backbone.
        Default: 224 (NI-LFM's native input size).
    tile_stride : int
        Stride between consecutive tiles.  ``tile_stride = tile_size // 2``
        gives 50 % overlap.  Default: 112.
    top_k : int
        Maximum number of matches to return per image pair.  Default: 2000.
    confidence_threshold : float
        Minimum cosine similarity to keep a match.  Range [0, 1].
        Default: 0.5.
    batch_size : int
        Number of tiles to process per forward pass.  Reduce if you hit OOM.
        Default: 8.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path:     str | Path | None = None,
        device:          str = "cpu",
        tile_size:       int = 224,
        tile_stride:     int = 112,
        top_k:           int = 2000,
        confidence_threshold: float = 0.5,
        batch_size:      int = 8,
    ) -> None:
        self.device              = device
        self.tile_size           = tile_size
        self.tile_stride         = tile_stride
        self.top_k               = top_k
        self.confidence_threshold = confidence_threshold
        self.batch_size          = batch_size

        ckpt_path = Path(checkpoint_path) if checkpoint_path else _BACKBONE_CKPT
        cfg_path  = Path(config_path)     if config_path     else _BACKBONE_CFG

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"NI-LFM checkpoint not found: {ckpt_path}\n"
                "Run: python scripts/download_nasa_ibm_backbone.py"
            )
        if not cfg_path.exists():
            raise FileNotFoundError(f"NI-LFM config not found: {cfg_path}")

        logger.info(f"Loading NI-LFM backbone from {ckpt_path} on {device} …")
        self._model, self._modality_info = self._load_backbone(ckpt_path, cfg_path)
        self._input_size = 256   # NAC modality native input size
        logger.info(f"NI-LFM backbone loaded (input_size={self._input_size}).")

    # -----------------------------------------------------------------------
    # Backbone loading
    # -----------------------------------------------------------------------

    def _load_backbone(self, ckpt_path: Path, cfg_path: Path):
        """Load NILunarFM and return (model, modality_info) for 'nac' (1-channel)."""
        try:
            from omegaconf import OmegaConf
            from ni_lfm.data.modality_info import MODALITY_INFO
            from ni_lfm.models.lfm_utils import init_enc_dec_embeddings
            from ni_lfm.models.ni_lfm import MODEL_CONFIGS, NILunarFM
        except ImportError as e:
            raise ImportError(
                f"Could not import ni_lfm: {e}\n"
                f"Expected source at: {_NILFM_ROOT}"
            ) from e

        cfg = OmegaConf.load(str(cfg_path))
        loaded_info = OmegaConf.to_container(cfg.data.domains, resolve=True)

        # 'nac' = 1-channel Narrow Angle Camera imagery — matches Chandrayaan pushbroom strips
        mod = "nac"
        structural = MODALITY_INFO.get(mod, {})
        info_dict  = {**structural, **loaded_info[mod]}
        modality_info = {mod: info_dict}

        self._modality_name = mod
        self._patch_size    = info_dict.get("patch_size", 16)   # 16
        self._embed_dim     = MODEL_CONFIGS["base"]["dim"]       # 768
        # num_tokens per tile = (input_size / patch_size)^2 = (256/16)^2 = 256
        self._num_tokens    = (info_dict.get("input_size", 256) // self._patch_size) ** 2

        enc_emb, dec_emb = init_enc_dec_embeddings(
            cfg=cfg,
            modality_info=modality_info,
            in_domains=[mod],
            out_domains=[mod],
        )

        model_cfg = MODEL_CONFIGS["base"].copy()
        model_cfg["num_register_tokens"] = 0

        model = NILunarFM(
            encoder_embeddings=enc_emb,
            decoder_embeddings=dec_emb,
            modality_info=modality_info,
            **model_cfg,
        )

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.debug(f"NI-LFM: {len(missing)} missing keys (decoder weights not used in encoder-only mode)")
        if unexpected:
            logger.warning(f"NI-LFM: {len(unexpected)} unexpected keys in checkpoint")

        model = model.to(self.device).eval()
        return model, modality_info

    # -----------------------------------------------------------------------
    # Tiling utilities
    # -----------------------------------------------------------------------

    def _tile_image(self, img_np: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Split a 2-D grayscale image into overlapping tiles.

        Parameters
        ----------
        img_np : np.ndarray  shape (H, W), float32 in [0, 1]

        Returns
        -------
        tiles      : np.ndarray  shape (N, tile_size, tile_size)
        centers    : list of (cx, cy) in image pixel coordinates
        """
        H, W   = img_np.shape
        ts, ss = self.tile_size, self.tile_stride
        tiles, centers = [], []

        for y in range(0, max(H - ts + 1, 1), ss):
            for x in range(0, max(W - ts + 1, 1), ss):
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)
                patch = img_np[y:y_end, x:x_end]

                # Zero-pad if the patch is smaller than tile_size
                if patch.shape != (ts, ts):
                    padded = np.zeros((ts, ts), dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded

                tiles.append(patch)
                cx = x + (x_end - x) / 2.0
                cy = y + (y_end - y) / 2.0
                centers.append((cx, cy))

        # Ensure at least one tile even if image is smaller than tile_size
        if not tiles:
            padded = np.zeros((ts, ts), dtype=np.float32)
            ph, pw = min(H, ts), min(W, ts)
            padded[:ph, :pw] = img_np[:ph, :pw]
            tiles.append(padded)
            centers.append((W / 2.0, H / 2.0))

        return np.array(tiles, dtype=np.float32), centers

    # -----------------------------------------------------------------------
    # Embedding extraction
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _embed_tiles(self, tiles: np.ndarray) -> torch.Tensor:
        """
        Run the NI-LFM encoder over tiles in mini-batches.

        Parameters
        ----------
        tiles : np.ndarray  shape (N, tile_size, tile_size)

        Returns
        -------
        descriptors : torch.Tensor  shape (N, embed_dim)  L2-normalised
        """
        N     = len(tiles)
        descs = []

        # CLAHE for better lunar contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for i in range(0, N, self.batch_size):
            batch_np = tiles[i : i + self.batch_size]          # (B, H, W)

            # CLAHE equalisation per tile
            eq = []
            for t in batch_np:
                t_u8 = (t * 255).clip(0, 255).astype(np.uint8)
                t_eq = clahe.apply(t_u8).astype(np.float32) / 255.0
                eq.append(t_eq)
            batch_np = np.stack(eq, axis=0)

            # Resize to model's native input_size (256 × 256)
            if batch_np.shape[1] != self._input_size or batch_np.shape[2] != self._input_size:
                resized = np.stack(
                    [cv2.resize(t, (self._input_size, self._input_size), interpolation=cv2.INTER_AREA)
                     for t in batch_np],
                    axis=0,
                )
            else:
                resized = batch_np

            # (B, 1, input_size, input_size)
            batch_t = torch.from_numpy(resized).unsqueeze(1).to(self.device)

            # Run encoder-only forward pass
            tokens = self._encoder_forward(batch_t)   # (B, N_tok, D)

            # Mean-pool spatial tokens → (B, D)  then L2-normalise
            desc = tokens.mean(dim=1)
            desc = F.normalize(desc, dim=-1)
            descs.append(desc.cpu())

        return torch.cat(descs, dim=0)             # (N, embed_dim)

    def _encoder_forward(self, batch_t: torch.Tensor) -> torch.Tensor:
        """
        Run encoder blocks directly on a (B, 1, H, W) image tensor.
        Returns (B, N_tokens, D) — no decoder involved.

        Uses the NILunarFM internal API:
          encoder_embedding({'tensor': img_t}) → dict with 'x', 'emb'
          We add 'input_mask' (all zeros = all tokens valid) manually.
          forward_mask_encoder(mod_dict, N_tok) → tokens, emb, mask, mod_mask
          forward_encoder(tokens + emb, mask)   → (B, N, D)
        """
        model = self._model
        mod   = self._modality_name   # 'nac'

        # encoder_embeddings[mod] expects a dict with key 'tensor'
        emb_out = model.encoder_embeddings[mod]({"tensor": batch_t})
        # emb_out: {'x': (B, N_tok, D), 'emb': (B, N_tok, D)}

        B, N_tok, D = emb_out["x"].shape

        # cat_encoder_tensors() reads 'input_mask' (B, N_tok) — 0 = valid, 1 = masked.
        # During inference we keep all tokens, so set all to 0 (valid).
        emb_out["input_mask"] = torch.zeros(B, N_tok, dtype=torch.bool,
                                            device=batch_t.device)

        encoder_mod_dict = {mod: emb_out}

        # Gather all tokens with no masking
        tokens, enc_emb, enc_mask, _ = model.forward_mask_encoder(
            encoder_mod_dict, num_encoder_tokens=N_tok
        )

        # Add positional embeddings and run transformer blocks
        x = tokens + enc_emb
        x = model.forward_encoder(x, encoder_mask=enc_mask)  # (B, N, D)
        return x

    # -----------------------------------------------------------------------
    # Main match interface
    # -----------------------------------------------------------------------

    def match(
        self,
        img_src_tensor: torch.Tensor,
        img_ref_tensor: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match two images using NI-LFM tile embeddings.

        Parameters
        ----------
        img_src_tensor : torch.Tensor  shape (1, 1, H, W), float32 [0, 1]
        img_ref_tensor : torch.Tensor  shape (1, 1, H, W), float32 [0, 1]

        Returns
        -------
        pts_src       : np.ndarray  shape (M, 2)  (x, y) pixel coords
        pts_ref       : np.ndarray  shape (M, 2)
        confidences   : np.ndarray  shape (M,)    cosine similarity scores
        """
        # Convert tensors → numpy float32 [0, 1]
        src_np = img_src_tensor.squeeze().cpu().numpy().astype(np.float32)
        ref_np = img_ref_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Tile both images
        src_tiles, src_centers = self._tile_image(src_np)
        ref_tiles, ref_centers = self._tile_image(ref_np)

        if len(src_tiles) == 0 or len(ref_tiles) == 0:
            return np.array([]), np.array([]), np.array([])

        logger.debug(
            f"NI-LFM: {len(src_tiles)} src tiles, {len(ref_tiles)} ref tiles "
            f"(tile={self.tile_size}px, stride={self.tile_stride}px)"
        )

        # Embed all tiles
        src_desc = self._embed_tiles(src_tiles)   # (N_src, D)
        ref_desc = self._embed_tiles(ref_tiles)   # (N_ref, D)

        # Cosine similarity matrix (both are L2-normalised → dot product = cosine)
        sim_matrix = torch.mm(src_desc, ref_desc.T)   # (N_src, N_ref)

        # Best ref match per src tile (mutual nearest-neighbour)
        best_ref_scores, best_ref_idx = sim_matrix.max(dim=1)   # (N_src,)
        best_src_scores, best_src_idx = sim_matrix.max(dim=0)   # (N_ref,)

        # Mutual consistency: src→ref and ref→src must agree
        n_src = len(src_tiles)
        mutual_mask = torch.arange(n_src) == best_src_idx[best_ref_idx]

        conf_np     = best_ref_scores.numpy()
        ref_idx_np  = best_ref_idx.numpy()
        mutual_np   = mutual_mask.numpy()

        # Apply confidence threshold + mutual consistency
        keep = mutual_np & (conf_np >= self.confidence_threshold)
        src_idx_keep = np.where(keep)[0]

        if len(src_idx_keep) == 0:
            # Relax to top-K without mutual check if nothing passes
            order = np.argsort(-conf_np)[:self.top_k]
            thresh_pass = conf_np[order] >= self.confidence_threshold
            src_idx_keep = order[thresh_pass]
            if len(src_idx_keep) == 0:
                return np.array([]), np.array([]), np.array([])

        ref_idx_keep  = ref_idx_np[src_idx_keep]
        conf_keep     = conf_np[src_idx_keep]

        # Sort by confidence, keep top-K
        order = np.argsort(-conf_keep)[: self.top_k]
        src_idx_keep  = src_idx_keep[order]
        ref_idx_keep  = ref_idx_keep[order]
        conf_keep     = conf_keep[order]

        # Convert tile indices → pixel coordinates
        pts_src = np.array([src_centers[i] for i in src_idx_keep], dtype=np.float32)
        pts_ref = np.array([ref_centers[i] for i in ref_idx_keep], dtype=np.float32)

        logger.debug(
            f"NI-LFM: {len(pts_src)} matches after filtering "
            f"(threshold={self.confidence_threshold}, top_k={self.top_k})"
        )

        return pts_src, pts_ref, conf_keep
