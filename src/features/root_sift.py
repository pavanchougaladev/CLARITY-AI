"""
RootSIFT — Improved SIFT Descriptor via Hellinger Normalization
===============================================================
Applies the RootSIFT transformation (Arandjelović & Zisserman, CVPR 2012)
on top of standard OpenCV SIFT descriptors.

Why RootSIFT?
  Standard SIFT uses L2 distance on L2-normalized gradient histograms.
  SIFT descriptors are histograms — i.e., probability distributions.
  For histograms, the Hellinger (Bhattacharyya) metric is theoretically
  superior to Euclidean distance.

  RootSIFT achieves this at zero extra cost:
    1. L1-normalize each descriptor  →  turns it into a true probability dist
    2. Take element-wise square root  →  maps to Hellinger space
    3. Use L2 distance in this space  ≡  Hellinger distance in original space

Usage:
    sift = cv2.SIFT_create(nfeatures=5000)
    kp, des = sift.detectAndCompute(img, None)
    des = root_sift_descriptors(des)   # in-place upgrade to RootSIFT
    # Now use des with BFMatcher(cv2.NORM_L2) as usual
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def root_sift_descriptors(des: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Apply the RootSIFT transformation to a matrix of SIFT descriptors.

    Parameters
    ----------
    des : np.ndarray, shape (N, 128), dtype float32
        Raw SIFT descriptors as returned by cv2.SIFT_create().detectAndCompute().
    eps : float
        Small constant for numerical stability during L1 normalisation.

    Returns
    -------
    np.ndarray, shape (N, 128), dtype float32
        RootSIFT descriptors.  Drop-in replacement for the original ``des``;
        still compatible with BFMatcher(cv2.NORM_L2).

    Notes
    -----
    If ``des`` is None or empty, it is returned unchanged.
    The transformation is applied out-of-place (original array untouched).
    """
    if des is None or len(des) == 0:
        return des

    des = des.astype(np.float32, copy=True)

    # Step 1: L1-normalise → each descriptor sums to 1 (probability simplex)
    l1_norms = np.sum(np.abs(des), axis=1, keepdims=True)
    des /= (l1_norms + eps)

    # Step 2: Element-wise square root → Hellinger embedding
    np.sqrt(des, out=des)

    return des
