from __future__ import annotations
"""Persistent on-disk storage for the operator's active uploaded pair.

Streamlit session state lives in server memory only. It survives navigation
inside a tab, but a second tab, the other port (8501 and 8502 each run their
own session), or a server restart silently drops every upload — after which
the app falls back to the bundled Chandrayaan/Apollo presets. That looked
like "the app registers different images".

This module keeps exactly one active pair on disk so the same pair is active
in every tab, on every port, and across restarts, until the operator clears
it explicitly (Clear uploaded images button, or the ✕ on an upload slot).
"""
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

WORKSPACE_DIR = Path(
    os.environ.get(
        "CLARITY_WORKSPACE",
        Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CLARITY-AI" / "workspace",
    )
)
PAIR_DIR = WORKSPACE_DIR / "current_pair"
_META = "meta.json"
_SRC = "source.bin"
_REF = "reference.bin"


def _read_bytes(name: str) -> Optional[bytes]:
    p = PAIR_DIR / name
    try:
        return p.read_bytes()
    except OSError:
        return None


def sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def save_pair(src_name: str, src_bytes: bytes, ref_name: str, ref_bytes: bytes) -> None:
    """Persist both sides of the active pair (either side may be empty/None)."""
    try:
        PAIR_DIR.mkdir(parents=True, exist_ok=True)
        meta = {}
        if src_bytes:
            (PAIR_DIR / _SRC).write_bytes(src_bytes)
            meta["src_name"] = src_name
            meta["src_sha"] = sha16(src_bytes)
        if ref_bytes:
            (PAIR_DIR / _REF).write_bytes(ref_bytes)
            meta["ref_name"] = ref_name
            meta["ref_sha"] = sha16(ref_bytes)
        (PAIR_DIR / _META).write_text(json.dumps(meta), encoding="utf-8")
        # Drop stale side files so a half-cleared pair does not resurrect.
        if not src_bytes:
            (PAIR_DIR / _SRC).unlink(missing_ok=True)
        if not ref_bytes:
            (PAIR_DIR / _REF).unlink(missing_ok=True)
    except OSError:
        # Persistence is a convenience; never break registration over it.
        shutil.rmtree(PAIR_DIR, ignore_errors=True)


def clear() -> None:
    shutil.rmtree(PAIR_DIR, ignore_errors=True)


def load_pair() -> Optional[Tuple[Tuple[str, bytes], Tuple[str, bytes]]]:
    """Return ((src_name, src_bytes), (ref_name, ref_bytes)) or None.

    Only sides that exist on disk are returned; a side recorded on disk but
    failing its stored SHA-256 is discarded (corrupted workspace).
    """
    if not PAIR_DIR.exists():
        return None
    try:
        meta = json.loads((PAIR_DIR / _META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    src = ref = None
    src_bytes = _read_bytes(_SRC)
    if src_bytes and meta.get("src_sha") == sha16(src_bytes):
        src = (meta.get("src_name", "source"), src_bytes)
    ref_bytes = _read_bytes(_REF)
    if ref_bytes and meta.get("ref_sha") == sha16(ref_bytes):
        ref = (meta.get("ref_name", "reference"), ref_bytes)
    if src is None and ref is None:
        return None
    return src, ref
