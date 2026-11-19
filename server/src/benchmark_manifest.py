"""Utilities for creating manifest-driven benchmark runs.

This module keeps the benchmark workflow deterministic and easy to use with
real lunar image datasets.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .pipeline import estimate_pair_overlap


def build_manifest_from_pairs(pairs: Iterable[Tuple[str, str]], output_path: str) -> str:
    """Write a JSON benchmark manifest from a list of (src, ref) pairs."""
    manifest = {"pairs": [{"source": src, "reference": ref} for src, ref in pairs]}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path)


def load_manifest(manifest_path: str) -> List[Tuple[str, str]]:
    """Load a manifest file and return a list of (source, reference) pairs."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    pairs = []
    for item in manifest.get("pairs", []):
        if isinstance(item, dict):
            pairs.append((item["source"], item["reference"]))
        else:
            pairs.append(tuple(item))
    return pairs


def filter_manifest_by_overlap(
    manifest_path: str,
    min_overlap: float = 0.15,
    output_path: str = None,
) -> List[Tuple[str, str]]:
    """Discard weak candidate pairs from a manifest using a deterministic overlap proxy."""
    selected = []
    for src_path, ref_path in load_manifest(manifest_path):
        try:
            src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if src is None or ref is None:
                continue
            score = estimate_pair_overlap(src, ref)
            if score >= float(min_overlap):
                selected.append((src_path, ref_path))
        except Exception:
            continue

    if output_path is not None:
        build_manifest_from_pairs(selected, output_path)

    return selected


def _candidate_group_key(image_path: str) -> tuple[str, str] | None:
    """Return a stable sensor/date key for a lunar browse image."""
    name = Path(image_path).name
    match = re.match(r'^(ch\d+_[a-z0-9]+_[a-z0-9]+)_(\d{8})[A-Za-z0-9_]*_b_brw_.*?\.png$', name)
    if not match:
        return None
    family, date_code = match.groups()
    return family, date_code


def discover_overlapping_pairs(
    data_root: str,
    min_overlap: float = 0.25,
    max_pairs: int = 20,
    same_date_only: bool = True,
) -> List[Tuple[str, str]]:
    """Auto-discover the strongest valid lunar image pairs under a data root."""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in Path(data_root).rglob("*.png"):
        key = _candidate_group_key(str(path))
        if key is not None:
            grouped[key].append(str(path))

    ranked: list[tuple[float, str, str]] = []
    for paths in grouped.values():
        if len(paths) < 2:
            continue
        for idx, src_path in enumerate(paths):
            for ref_path in paths[idx + 1:]:
                try:
                    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                    if src is None or ref is None:
                        continue
                    score = estimate_pair_overlap(src, ref)
                    if score >= float(min_overlap):
                        ranked.append((score, src_path, ref_path))
                except Exception:
                    continue

    ranked.sort(key=lambda item: item[0], reverse=True)
    unique_pairs: list[tuple[str, str]] = []
    seen = set()
    for _, src_path, ref_path in ranked:
        pair = tuple(sorted((src_path, ref_path)))
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append((src_path, ref_path))
        if max_pairs is not None and len(unique_pairs) >= max_pairs:
            break

    return unique_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create or inspect a benchmark manifest")
    parser.add_argument("--pair", action="append", nargs=2, default=[], metavar=("SRC", "REF"))
    parser.add_argument("--output", default="benchmark_manifest.json")
    parser.add_argument("--filter-overlap", type=float, default=None, help="Discard pairs with overlap proxy below this value")
    parser.add_argument("--discover-root", default=None, help="Auto-discover overlapping benchmark pairs under a dataset root")
    parser.add_argument("--min-overlap", type=float, default=0.25, help="Minimum overlap score required for discovery")
    parser.add_argument("--max-pairs", type=int, default=20, help="Maximum pairs to retain after discovery")
    args = parser.parse_args()

    if args.discover_root is not None:
        selected = discover_overlapping_pairs(args.discover_root, min_overlap=args.min_overlap, max_pairs=args.max_pairs)
        path = build_manifest_from_pairs(selected, args.output)
        print(json.dumps({"pairs": [{"source": src, "reference": ref} for src, ref in selected]}, indent=2))
        print(f"Manifest written to {path}")
    elif args.filter_overlap is not None:
        if not args.pair:
            selected = filter_manifest_by_overlap(args.output, min_overlap=args.filter_overlap, output_path=args.output)
            print(json.dumps({"pairs": [{"source": src, "reference": ref} for src, ref in selected]}, indent=2))
        else:
            out = build_manifest_from_pairs(args.pair, args.output)
            selected = filter_manifest_by_overlap(out, min_overlap=args.filter_overlap, output_path=out)
            print(json.dumps({"pairs": [{"source": src, "reference": ref} for src, ref in selected]}, indent=2))
    elif args.pair:
        out = build_manifest_from_pairs(args.pair, args.output)
        print(out)
    else:
        print(json.dumps({"pairs": []}, indent=2))
