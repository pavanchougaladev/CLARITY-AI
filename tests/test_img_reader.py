"""
Unit tests for direct planetary .img raster loading and PDS4 integration.
"""
import os
import pytest
import numpy as np
from src.pipeline import robust_read
from src.ingest.pds4_parser import load_img_raster, find_matching_xml_label
from ui.app import _find_archive_metadata, get_safe_preview


def test_find_matching_xml_label():
    p_img = "data/chandrayan-2 dataset/ch2_ohr_ncp_20230823T1647285315_d_img_n18.img"
    if os.path.exists(p_img):
        xml_label = find_matching_xml_label(p_img)
        assert xml_label is not None
        assert os.path.exists(xml_label)
        assert xml_label.endswith(".xml")


def test_load_img_raster_decimated():
    p_img = "data/chandrayan-2 dataset/ch2_ohr_ncp_20230823T1647285315_d_img_n18.img"
    if os.path.exists(p_img):
        arr = load_img_raster(p_img, max_dim=512)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8
        assert max(arr.shape) <= 512
        assert arr.shape[0] > 0 and arr.shape[1] > 0


def test_robust_read_img():
    p_img = "data/chandrayan-2 dataset/ch2_ohr_ncp_20230823T1647285315_d_img_n18.img"
    if os.path.exists(p_img):
        arr = robust_read(p_img, max_dim=400)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8
        assert max(arr.shape) <= 400


def test_get_safe_preview_img():
    p_img = "data/chandrayan-2 dataset/ch2_ohr_ncp_20230823T1647285315_d_img_n18.img"
    if os.path.exists(p_img):
        preview = get_safe_preview(p_img, max_dim=300)
        assert preview is not None
        assert isinstance(preview, np.ndarray)
        assert max(preview.shape) <= 300


def test_find_archive_metadata_for_img():
    p_img = "data/chandrayan-2 dataset/ch2_ohr_ncp_20230823T1647285315_d_img_n18.img"
    if os.path.exists(p_img):
        meta = _find_archive_metadata(p_img)
        assert "label" in meta
        assert "grid" in meta
        assert os.path.exists(meta["label"])
        assert os.path.exists(meta["grid"])
