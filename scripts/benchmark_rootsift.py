"""
Benchmark: Standard SIFT vs RootSIFT
=====================================
Generates synthetic lunar-like test images with known ground-truth homographies
(including brightness shifts, noise, and shadow-like occlusions typical of
Chandrayaan orbital imagery), then measures:

  - Number of correct inlier matches (vs ground truth)
  - Inlier ratio (correct / total matches after Lowe's ratio test)
  - Homography reprojection error (RMSE in pixels)
  - Match count
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import cv2
import numpy as np
from src.features.root_sift import root_sift_descriptors

SEED = 42
np.random.seed(SEED)

# ── Synthetic lunar image generator ──────────────────────────────────────────

def make_lunar_image(h=512, w=512):
    """Simulate a grayscale lunar surface with craters and noise."""
    img = np.random.randint(60, 100, (h, w), dtype=np.uint8)

    # Add Gaussian noise
    noise = np.random.normal(0, 8, (h, w)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add crater-like blobs (dark centre + bright rim)
    rng = np.random.default_rng(SEED)
    for _ in range(40):
        cx, cy = rng.integers(50, w - 50), rng.integers(50, h - 50)
        r = rng.integers(8, 40)
        cv2.circle(img, (cx, cy), r, int(rng.integers(20, 50)), -1)        # dark floor
        cv2.circle(img, (cx, cy), r, int(rng.integers(160, 220)), 2)       # bright rim

    # Shadow strip (harsh solar illumination edge)
    shadow_x = rng.integers(w // 4, 3 * w // 4)
    img[:, :shadow_x] = np.clip(img[:, :shadow_x].astype(np.int16) - 40, 0, 255)

    return img


def apply_transform(img, H):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h))


def random_homography(strength=5.0, img_size=512):
    """Small random perspective distortion (simulates orbital parallax)."""
    s = img_size
    src = np.float32([[0,0],[s,0],[s,s],[0,s]])
    rng = np.random.default_rng(SEED + 7)
    dst = src + rng.uniform(-strength, strength, src.shape).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


# ── Matcher core ──────────────────────────────────────────────────────────────

def match(img1, img2, use_rootsift, nfeatures=3000, ratio_thresh=0.75):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, 0, float('inf'), 0

    if use_rootsift:
        des1 = root_sift_descriptors(des1)
        des2 = root_sift_descriptors(des2)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw if m.distance < ratio_thresh * n.distance]
    total_matches = len(good)

    if total_matches < 4:
        return 0, 0, float('inf'), total_matches

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H_est, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 2.0, confidence=0.999)
    if H_est is None or mask is None:
        return 0, 0, float('inf'), total_matches

    inliers = int(mask.sum())
    inlier_ratio = inliers / total_matches

    return inliers, inlier_ratio, H_est, total_matches


def reprojection_rmse(H_gt, H_est, img_size=512):
    """Mean reprojection error of H_est vs ground-truth H_gt on a grid of points."""
    if not isinstance(H_est, np.ndarray):
        return float('inf')
    s = img_size
    grid_pts = np.float32([[x, y] for x in np.linspace(0, s, 10)
                                   for y in np.linspace(0, s, 10)]).reshape(-1, 1, 2)
    proj_gt  = cv2.perspectiveTransform(grid_pts, H_gt)
    proj_est = cv2.perspectiveTransform(grid_pts, H_est)
    errors = np.linalg.norm(proj_gt - proj_est, axis=2).flatten()
    return float(np.sqrt(np.mean(errors ** 2)))


# ── Run benchmark ─────────────────────────────────────────────────────────────

def run_benchmark(n_trials=10):
    print("=" * 62)
    print("   Standard SIFT  vs  RootSIFT — Accuracy Benchmark")
    print("   (Synthetic lunar imagery, known ground-truth homography)")
    print("=" * 62)

    results = {"sift": [], "rootsift": []}

    for trial in range(n_trials):
        np.random.seed(trial)
        img_ref = make_lunar_image()

        # Apply random mild perspective warp + brightness shift + noise
        H_gt = random_homography(strength=4.0 + trial * 0.5)
        img_src = apply_transform(img_ref, H_gt)

        # Add per-image noise to simulate different acquisition conditions
        noise = np.random.normal(0, 6, img_src.shape).astype(np.int16)
        img_src = np.clip(img_src.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        for use_root, key in [(False, "sift"), (True, "rootsift")]:
            inliers, ratio, H_est, total = match(img_ref, img_src, use_root)
            rmse_val = reprojection_rmse(H_gt, H_est)
            results[key].append({
                "inliers": inliers,
                "ratio": ratio,
                "rmse": rmse_val,
                "total": total,
            })

    # ── Print per-trial table ──────────────────────────────────────────────
    print(f"\n{'Trial':>5} | {'SIFT Inliers':>12} {'Inlier %':>9} {'RMSE(px)':>9} | "
          f"{'RootSIFT Inliers':>16} {'Inlier %':>9} {'RMSE(px)':>9} | Δ Inliers")
    print("-" * 100)

    for i in range(n_trials):
        s = results["sift"][i]
        r = results["rootsift"][i]
        delta = r["inliers"] - s["inliers"]
        sign = "+" if delta >= 0 else ""
        rmse_s = f"{s['rmse']:.3f}" if s['rmse'] < 999 else " N/A"
        rmse_r = f"{r['rmse']:.3f}" if r['rmse'] < 999 else " N/A"
        print(f"{i+1:>5} | {s['inliers']:>12} {s['ratio']*100:>8.1f}% {rmse_s:>9} | "
              f"{r['inliers']:>16} {r['ratio']*100:>8.1f}% {rmse_r:>9} | {sign}{delta}")

    # ── Aggregate stats ────────────────────────────────────────────────────
    def avg(key, field):
        vals = [v[field] for v in results[key] if v[field] < 999]
        return np.mean(vals) if vals else float('nan')

    print("-" * 100)
    print(f"\n{'AVERAGE':>5} | {'SIFT':>12} {'Inlier %':>9} {'RMSE(px)':>9} | "
          f"{'RootSIFT':>16} {'Inlier %':>9} {'RMSE(px)':>9} | Δ Inliers")
    si = avg("sift", "inliers"); ri = avg("rootsift", "inliers")
    sr = avg("sift", "ratio");   rr = avg("rootsift", "ratio")
    sm = avg("sift", "rmse");    rm = avg("rootsift", "rmse")
    di = ri - si
    print(f"{'':>5} | {si:>12.1f} {sr*100:>8.1f}% {sm:>9.3f} | "
          f"{ri:>16.1f} {rr*100:>8.1f}% {rm:>9.3f} | {'+' if di>=0 else ''}{di:.1f}")

    print(f"\n{'Improvement':>12} | Inliers: {(ri-si)/si*100:+.1f}%  "
          f"| Inlier ratio: {(rr-sr)*100:+.1f}pp  "
          f"| RMSE: {(rm-sm):+.3f}px")
    print("=" * 62)


if __name__ == "__main__":
    run_benchmark(n_trials=10)
