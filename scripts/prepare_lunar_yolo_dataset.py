"""
ISRO LunarFusion — Multi-Mission Lunar Crater Dataset Generator
===============================================================
Harvests and harmonizes lunar imagery across all sources in the data/ directory:
1. Ground-truth pre-annotated crater datasets (data/chandrayan-2 dataset/archive/craters)
2. Chandrayaan-2 OHRC high-resolution browse strips (data/chandrayan-2 dataset/)
3. Chandrayaan-1 & Chandrayaan-2 TMC/OHRC multi-view strips (data/raw/)
4. ISRO Multi-Mission Apollo 11, 12, 14 and Chandrayaan-3 archives (data/ISRO/)
5. NASA LROC NAC EDR high-resolution planetary strips (data/lroc_edr/)
6. Full-resolution Gigapixel Chandrayaan-2 GeoTIFF (data/raw/ohrc_image.tif)

Annotates unlabelled crops using the calibrated ISRO teacher detector
(models/teacher_yolov5l_ohrc.pt), structures unified YOLO format datasets,
and writes data/lunar_crater_yolo/lunar_craters.yaml for YOLO11 training.
"""

import os
import glob
import random
import shutil
import cv2
import numpy as np
import torch

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def extract_craters_from_crop(teacher, crop: np.ndarray, tile_size: int = 640, min_conf: float = 0.40):
    """Run teacher detector on crop and return normalized YOLO boxes."""
    if len(crop.shape) == 2:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.shape[2] == 1:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = crop

    results = teacher(crop_bgr)
    preds = results.pandas().xyxy[0]

    if len(preds) == 0:
        return []

    craters = []
    for _, row in preds.iterrows():
        if row["confidence"] < min_conf:
            continue
        bw = row["xmax"] - row["xmin"]
        bh = row["ymax"] - row["ymin"]
        if bw < 6 or bh < 6:
            continue

        xc = (row["xmin"] + row["xmax"]) / 2.0 / tile_size
        yc = (row["ymin"] + row["ymax"]) / 2.0 / tile_size
        wn = bw / tile_size
        hn = bh / tile_size

        xc = max(0.0, min(1.0, float(xc)))
        yc = max(0.0, min(1.0, float(yc)))
        wn = max(0.0, min(1.0, float(wn)))
        hn = max(0.0, min(1.0, float(hn)))

        craters.append((0, xc, yc, wn, hn))

    return craters


def prepare_lunar_dataset(
    data_root: str = "data",
    output_dir: str = "data/lunar_crater_yolo",
    teacher_weights: str = "models/teacher_yolov5l_ohrc.pt",
    tile_size: int = 640,
    stride: int = 512,
    max_tiles_per_strip: int = 25,
    max_tiles_lroc: int = 15,
    max_tiles_geotiff: int = 60,
    min_confidence: float = 0.38,
    val_ratio: float = 0.15,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("🌕 INITIATING MULTI-MISSION LUNAR CRATER DATASET HARVEST")
    print(f"  • Root Directory:       {data_root}")
    print(f"  • Output Directory:     {output_dir}")
    print(f"  • Teacher Weights:      {teacher_weights}")
    print(f"  • Tile Dimensions:      {tile_size}x{tile_size}")
    print("=" * 70)

    # Clean and setup target directories
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Ingest Pre-Annotated Roboflow Ground-Truth Archive
    # -----------------------------------------------------------------------
    archive_base = os.path.join(data_root, "chandrayan-2 dataset", "archive", "craters")
    gold_standard_samples = []

    if os.path.exists(archive_base):
        print("\n[1/6] Ingesting Ground-Truth Pre-Annotated Crater Dataset...")
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(archive_base, split, "images")
            lbl_dir = os.path.join(archive_base, split, "labels")
            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                continue

            for img_file in glob.glob(os.path.join(img_dir, "*.jpg")):
                base = os.path.splitext(os.path.basename(img_file))[0]
                lbl_file = os.path.join(lbl_dir, f"{base}.txt")
                if not os.path.exists(lbl_file):
                    continue

                img = cv2.imread(img_file)
                if img is None:
                    continue

                boxes = []
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            boxes.append((0, xc, yc, w, h))

                if boxes:
                    tile_id = f"gold_{base}"
                    gold_standard_samples.append((tile_id, img, boxes))

        print(f"  ✔ Loaded {len(gold_standard_samples)} verified ground-truth crater tiles from archive.")
    else:
        print("\n[1/6] No pre-annotated archive directory found; proceeding with raw telemetry.")

    # -----------------------------------------------------------------------
    # Step 2: Load Calibrated Teacher Detector
    # -----------------------------------------------------------------------
    print(f"\n[2/6] Loading Teacher Detector from {teacher_weights}...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    teacher = torch.hub.load("ultralytics/yolov5", "custom", path=teacher_weights, force_reload=False, trust_repo=True)
    teacher.to(device)
    teacher.conf = min_confidence
    teacher.iou = 0.45
    print(f"  ✔ Teacher detector active on {device.upper()} with confidence threshold {min_confidence}.")

    generated_samples = []

    # -----------------------------------------------------------------------
    # Step 3: Harvest Standard Browse Strips (Chandrayaan-1/2, Apollo, TMC)
    # -----------------------------------------------------------------------
    print("\n[3/6] Harvesting Browse Strips from Chandrayaan-1/2, TMC, Apollo & ISRO...")
    all_pngs = []
    for search_dir in [
        os.path.join(data_root, "chandrayan-2 dataset"),
        os.path.join(data_root, "raw"),
        os.path.join(data_root, "ISRO"),
    ]:
        if os.path.exists(search_dir):
            for ext in ["*.png", "*.PNG", "*_b_brw_*.png"]:
                all_pngs.extend(glob.glob(os.path.join(search_dir, "**", ext), recursive=True))

    # De-duplicate and exclude output/archive directories
    browse_images = sorted(list(set(
        p for p in all_pngs
        if "lunar_crater_yolo" not in p and "archive" not in p
    )))
    print(f"  ✔ Located {len(browse_images)} multi-mission browse strips.")

    for img_idx, img_path in enumerate(browse_images):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if w < tile_size or h < tile_size:
            continue

        strip_tiles = 0
        y_coords = list(range(0, h - tile_size + 1, stride))
        x_coords = list(range(0, w - tile_size + 1, stride))
        candidates = [(y, x) for y in y_coords for x in x_coords]
        random.shuffle(candidates)
        candidates = candidates[:max_tiles_per_strip]

        for y, x in candidates:
            crop = img[y:y+tile_size, x:x+tile_size]
            if np.std(crop) < 12.0:
                continue

            craters = extract_craters_from_crop(teacher, crop, tile_size, min_confidence)
            if not craters:
                continue

            tile_id = f"brw_{base_name}_y{y}_x{x}"
            generated_samples.append((tile_id, crop, craters))
            strip_tiles += 1

        print(f"  [{img_idx+1:02d}/{len(browse_images):02d}] {base_name[:32]}... -> {strip_tiles} valid tiles")

    # -----------------------------------------------------------------------
    # Step 4: Stream Windowed Crops from Full-Res Gigapixel GeoTIFF
    # -----------------------------------------------------------------------
    geotiff_path = os.path.join(data_root, "raw", "ohrc_image.tif")
    if HAS_RASTERIO and os.path.exists(geotiff_path):
        print(f"\n[4/6] Streaming Windowed Crops from 1.12 Gigapixel GeoTIFF: {geotiff_path}...")
        try:
            with rasterio.open(geotiff_path) as src:
                gh, gw = src.height, src.width
                print(f"  GeoTIFF Native Dimensions: {gw} x {gh}")
                valid_tiff_tiles = 0
                attempts = 0
                max_attempts = max_tiles_geotiff * 4

                while valid_tiff_tiles < max_tiles_geotiff and attempts < max_attempts:
                    attempts += 1
                    rx = random.randint(0, max(0, gw - tile_size))
                    ry = random.randint(0, max(0, gh - tile_size))

                    win = Window(rx, ry, tile_size, tile_size)
                    crop_raw = src.read(1, window=win)

                    if crop_raw.std() < 12.0:
                        continue

                    # Normalize to 8-bit BGR
                    if crop_raw.dtype != np.uint8:
                        crop_u8 = cv2.normalize(crop_raw.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    else:
                        crop_u8 = crop_raw

                    crop_bgr = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2BGR)
                    craters = extract_craters_from_crop(teacher, crop_bgr, tile_size, min_confidence)
                    if not craters:
                        continue

                    tile_id = f"geotiff_y{ry}_x{rx}"
                    generated_samples.append((tile_id, crop_bgr, craters))
                    valid_tiff_tiles += 1

                print(f"  ✔ Extracted {valid_tiff_tiles} high-contrast crater tiles from Gigapixel GeoTIFF.")
        except Exception as e:
            print(f"  ⚠ GeoTIFF window extraction skipped: {e}")
    else:
        print("\n[4/6] Gigapixel GeoTIFF streaming skipped (file or rasterio absent).")

    # -----------------------------------------------------------------------
    # Step 5: Stream Windowed Crops from NASA LROC NAC EDR Strips
    # -----------------------------------------------------------------------
    lroc_files = sorted(glob.glob(os.path.join(data_root, "lroc_edr", "*.IMG")))
    if HAS_RASTERIO and lroc_files:
        print(f"\n[5/6] Streaming Windowed Crops from {len(lroc_files)} NASA LROC NAC EDR Cameras...")
        total_lroc_tiles = 0
        for l_idx, lroc_path in enumerate(lroc_files):
            base_lroc = os.path.splitext(os.path.basename(lroc_path))[0]
            try:
                with rasterio.open(lroc_path) as src:
                    lh, lw = src.height, src.width
                    file_tiles = 0
                    attempts = 0
                    max_attempts = max_tiles_lroc * 3

                    while file_tiles < max_tiles_lroc and attempts < max_attempts:
                        attempts += 1
                        rx = random.randint(0, max(0, lw - tile_size))
                        ry = random.randint(0, max(0, lh - tile_size))

                        win = Window(rx, ry, tile_size, tile_size)
                        crop_raw = src.read(1, window=win)

                        if crop_raw.std() < 12.0:
                            continue

                        if crop_raw.dtype != np.uint8:
                            crop_u8 = cv2.normalize(crop_raw.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        else:
                            crop_u8 = crop_raw

                        crop_bgr = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2BGR)
                        craters = extract_craters_from_crop(teacher, crop_bgr, tile_size, min_confidence)
                        if not craters:
                            continue

                        tile_id = f"lroc_{base_lroc}_y{ry}_x{rx}"
                        generated_samples.append((tile_id, crop_bgr, craters))
                        file_tiles += 1
                        total_lroc_tiles += 1

                    print(f"  [{l_idx+1:02d}/{len(lroc_files):02d}] {base_lroc} ({lw}x{lh}) -> {file_tiles} valid tiles")
            except Exception as e:
                print(f"  ⚠ LROC file {base_lroc} read error: {e}")
        print(f"  ✔ Extracted {total_lroc_tiles} crater tiles across NASA LROC NAC strips.")
    else:
        print("\n[5/6] LROC NAC EDR extraction skipped.")

    # -----------------------------------------------------------------------
    # Step 6: Consolidate, Split, and Write YOLO Dataset
    # -----------------------------------------------------------------------
    all_samples = gold_standard_samples + generated_samples
    random.shuffle(all_samples)

    total_craters = sum(len(c) for _, _, c in all_samples)
    val_count = max(1, int(len(all_samples) * val_ratio))
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    print(f"\n[6/6] Writing Unified YOLO Dataset to {output_dir}...")
    print(f"  • Total Images:      {len(all_samples)}")
    print(f"  • Training Split:    {len(train_samples)} tiles")
    print(f"  • Validation Split:  {len(val_samples)} tiles")
    print(f"  • Total Annotations: {total_craters} craters")

    def save_split(samples, img_dir, lbl_dir):
        for tile_id, crop, annotations in samples:
            img_file = os.path.join(img_dir, f"{tile_id}.jpg")
            lbl_file = os.path.join(lbl_dir, f"{tile_id}.txt")

            cv2.imwrite(img_file, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lbl_file, "w") as f:
                for cls_id, xc, yc, wn, hn in annotations:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    save_split(train_samples, train_img_dir, train_lbl_dir)
    save_split(val_samples, val_img_dir, val_lbl_dir)

    # Clean any stale caches
    for cache_f in [
        os.path.join(output_dir, "labels", "train.cache"),
        os.path.join(output_dir, "labels", "val.cache")
    ]:
        if os.path.exists(cache_f):
            os.remove(cache_f)

    # Write dataset YAML
    yaml_content = f"""# ISRO LunarFusion Multi-Mission Crater Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: crater
"""
    yaml_path = os.path.join(output_dir, "lunar_craters.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n" + "=" * 70)
    print("✅ MULTI-MISSION LUNAR CRATER YOLO DATASET PREPARED")
    print(f"  • Config YAML:      {yaml_path}")
    print(f"  • Total Samples:    {len(all_samples)}")
    print(f"  • Total Craters:    {total_craters}")
    print("=" * 70)
    return yaml_path


if __name__ == "__main__":
    prepare_lunar_dataset()
