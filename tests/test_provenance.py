"""
Unit Tests for Cryptographic Provenance & Data Integrity Chain (Feature 4)
"""

import pytest
import numpy as np
from src.provenance import ProvenanceChain, ProvenanceBlock, compute_sha256
from src.result_schema import RegistrationTelemetry


def test_compute_sha256_types(tmp_path):
    # Test string hashing
    h_str = compute_sha256("Chandrayaan-2 TMC")
    assert isinstance(h_str, str) and len(h_str) == 64

    # Test file hashing
    p = tmp_path / "sample.bin"
    p.write_bytes(b"ISRO_LUNAR_DATA_BLOCK")
    h_file = compute_sha256(str(p))
    assert len(h_file) == 64

    # Test numpy array hashing
    arr = np.zeros((10, 10), dtype=np.uint8)
    h_arr1 = compute_sha256(arr)
    arr[0, 0] = 255
    h_arr2 = compute_sha256(arr)
    assert h_arr1 != h_arr2

    # Test dict hashing
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}
    assert compute_sha256(d1) == compute_sha256(d2)


def test_provenance_chain_lifecycle():
    chain = ProvenanceChain(run_id="test_run_001")
    assert chain.latest_hash == ProvenanceChain.GENESIS_HASH

    # Record ingest
    b1 = chain.record("ingest", details={"sensor": "TMC", "res": 5.0}, payload=b"raw_pixels")
    assert b1["index"] == 0
    assert b1["prev_hash"] == ProvenanceChain.GENESIS_HASH

    # Record preprocessing
    b2 = chain.record("preprocessing", details={"clahe": True}, payload="processed")
    assert b2["index"] == 1
    assert b2["prev_hash"] == b1["block_hash"]

    # Record matching
    b3 = chain.record("matching", details={"inliers": 150, "rmse": 0.45})
    assert b3["index"] == 2
    assert b3["prev_hash"] == b2["block_hash"]

    # Verify validity before seal
    assert chain.verify() is True

    # Seal chain
    root = chain.seal()
    assert isinstance(root, str) and len(root) == 64
    assert chain.verify() is True

    # Cannot record into sealed chain
    with pytest.raises(RuntimeError):
        chain.record("extra_stage", {})

    # Export dictionary
    d = chain.to_dict()
    assert d["run_id"] == "test_run_001"
    assert d["is_valid"] is True
    assert d["sealed"] is True
    assert len(d["stages"]) == 4  # ingest, preprocessing, matching, certification


def test_provenance_tamper_detection():
    chain = ProvenanceChain(run_id="tamper_test")
    chain.record("stage1", details={"val": 1})
    chain.record("stage2", details={"val": 2})
    chain.seal()

    assert chain.verify() is True

    # Tamper with an internal block
    chain.blocks[0].details["val"] = 999
    assert chain.verify() is False


def test_registration_telemetry_provenance_binding():
    chain = ProvenanceChain(run_id="telemetry_test")
    chain.record("ingest", details={"sensor": "OHRC"})
    chain.seal()

    results = {
        "method_used": "loftr",
        "total_matches": 100,
        "inlier_count": 80,
        "rmse_pixels": 0.35,
        "quality": {"passed": True, "grade": "A", "confidence": 0.95},
        "provenance_chain": chain.to_dict(),
    }

    telem = RegistrationTelemetry.from_results(results)
    assert telem.provenance_chain is not None
    assert telem.provenance_chain["run_id"] == "telemetry_test"
    assert telem.provenance_chain["is_valid"] is True
