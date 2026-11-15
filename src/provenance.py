"""
Cryptographic Provenance & Data Integrity Chain
===============================================
ISRO LunarFusion Co-Registration Suite — Planetary Data Architecture

Implements a deterministic, cryptographically linked audit trail (SHA-256)
tracking every image transform, feature matching stage, quality gate evaluation,
and export product to guarantee non-repudiation and photogrammetric reproducibility.
"""

import os
import time
import json
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np


def compute_sha256(data: Any) -> str:
    """
    Compute deterministic SHA-256 hex digest for diverse payload types.
    Supports file paths, raw bytes, numpy arrays, strings, and dictionaries.
    """
    hasher = hashlib.sha256()

    if isinstance(data, str):
        if os.path.exists(data) and os.path.isfile(data):
            # Stream file in 64KB blocks for memory safety on gigapixel rasters
            with open(data, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        else:
            hasher.update(data.encode("utf-8"))
            return hasher.hexdigest()

    elif isinstance(data, (bytes, bytearray, memoryview)):
        hasher.update(data)
        return hasher.hexdigest()

    elif isinstance(data, np.ndarray):
        # Hash array header (shape, dtype) + raw byte content
        header = f"{data.shape}:{data.dtype}:".encode("utf-8")
        hasher.update(header)
        # Use contiguous buffer view
        hasher.update(np.ascontiguousarray(data).tobytes())
        return hasher.hexdigest()

    elif isinstance(data, dict):
        canonical_json = json.dumps(data, sort_keys=True, default=str)
        hasher.update(canonical_json.encode("utf-8"))
        return hasher.hexdigest()

    elif data is None:
        hasher.update(b"NULL")
        return hasher.hexdigest()

    else:
        hasher.update(str(data).encode("utf-8"))
        return hasher.hexdigest()


class ProvenanceBlock:
    """A single immutable node in the lunar telemetry provenance chain."""

    def __init__(
        self,
        index: int,
        stage: str,
        timestamp: float,
        prev_hash: str,
        details: Dict[str, Any],
        payload_hash: str,
        block_hash: Optional[str] = None,
    ):
        self.index = index
        self.stage = stage
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.details = details
        self.payload_hash = payload_hash
        self.block_hash = block_hash or self.compute_hash()

    def compute_hash(self) -> str:
        """Compute block hash linking previous block hash and payload."""
        canonical_details = json.dumps(self.details, sort_keys=True, default=str)
        raw_string = f"{self.index}:{self.stage}:{self.timestamp:.4f}:{self.prev_hash}:{self.payload_hash}:{canonical_details}"
        return hashlib.sha256(raw_string.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "prev_hash": self.prev_hash,
            "payload_hash": self.payload_hash,
            "details": self.details,
            "block_hash": self.block_hash,
        }


class ProvenanceChain:
    """
    Immutable audit chain verifying all transitions throughout the co-registration lifecycle.
    """

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{int(time.time() * 1000)}"
        self.blocks: List[ProvenanceBlock] = []
        self._sealed: bool = False

    @property
    def latest_hash(self) -> str:
        return self.blocks[-1].block_hash if self.blocks else self.GENESIS_HASH

    def record(
        self,
        stage: str,
        details: Optional[Dict[str, Any]] = None,
        payload: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Record a pipeline stage into the cryptographically linked ledger.

        Parameters
        ----------
        stage : str
            Stage identifier ('ingest', 'preprocessing', 'matching', 'quality_gate', 'export', 'certification')
        details : dict, optional
            Stage metadata (parameters, gate status, metrics)
        payload : any, optional
            Data object (file path, image array, or raw bytes) to hash
        """
        if self._sealed:
            raise RuntimeError("Cannot record into a sealed ProvenanceChain.")

        idx = len(self.blocks)
        prev_h = self.latest_hash
        p_hash = compute_sha256(payload) if payload is not None else compute_sha256(details or {})
        clean_details = details or {}

        block = ProvenanceBlock(
            index=idx,
            stage=stage,
            timestamp=time.time(),
            prev_hash=prev_h,
            details=clean_details,
            payload_hash=p_hash,
        )
        self.blocks.append(block)
        return block.to_dict()

    def seal(self, certifier: str = "ISRO LunarFusion Quality Gate") -> str:
        """
        Finalize and seal the provenance chain with a certification block.
        """
        if not self._sealed:
            self.record(
                stage="certification",
                details={
                    "certifier": certifier,
                    "total_stages": len(self.blocks),
                    "integrity_verified": self.verify(),
                },
                payload=self.latest_hash,
            )
            self._sealed = True
        return self.latest_hash

    def verify(self) -> bool:
        """
        Audit the entire chain from genesis to head.
        Returns True if every hash and back-pointer is mathematically valid.
        """
        if not self.blocks:
            return True

        expected_prev = self.GENESIS_HASH
        for block in self.blocks:
            if block.prev_hash != expected_prev:
                return False
            # Recompute block hash
            if block.compute_hash() != block.block_hash:
                return False
            expected_prev = block.block_hash

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Export serialized provenance chain report."""
        verified = self.verify()
        return {
            "run_id": self.run_id,
            "root_hash": self.latest_hash,
            "is_valid": verified,
            "sealed": self._sealed,
            "total_blocks": len(self.blocks),
            "stages": [b.to_dict() for b in self.blocks],
        }
