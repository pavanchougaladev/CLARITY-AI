import os
import argparse
from typing import Optional
import numpy as np
import cv2

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import pds4_tools
except ImportError:
    pds4_tools = None


def find_matching_xml_label(img_path: str) -> Optional[str]:
    """
    Search for an accompanying PDS4 XML label for the given .img file.
    Checks:
      1. Sibling file with .xml extension
      2. Any .xml in the same folder mentioning the image basename
      3. Global search in project data/ directory
    """
    if not img_path:
        return None

    img_dir = os.path.dirname(os.path.abspath(img_path))
    base = os.path.splitext(os.path.basename(img_path))[0]

    # 1. Exact sibling .xml
    direct_xml = os.path.join(img_dir, f"{base}.xml")
    if os.path.exists(direct_xml):
        return direct_xml

    # 2. Check files in same directory
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.endswith('.xml') and (base in f or f.startswith(base.rsplit('_', 1)[0])):
                cand = os.path.join(img_dir, f)
                return cand

    # 3. Search under project data directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data")
    if os.path.exists(data_dir):
        target_xml = f"{base}.xml"
        for root, dirs, files in os.walk(data_dir):
            if target_xml in files:
                return os.path.join(root, target_xml)

    return None


def load_img_raster(img_path: str, max_dim: Optional[int] = None, xml_path: Optional[str] = None) -> np.ndarray:
    """
    Directly loads a planetary .img raster (PDS4, Chandrayaan-2 OHRC/TMC, or raw binary)
    into an 8-bit normalized grayscale numpy array. Supports high-speed decimated
    sub-sampling via rasterio or memmap when max_dim is specified.
    """
    if not img_path or not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # If the user passed the XML directly as img_path, swap
    if img_path.lower().endswith('.xml'):
        xml_path = img_path

    if not xml_path:
        xml_path = find_matching_xml_label(img_path)

    # Strategy A: Read through PDS4 XML label with rasterio (fast decimated read)
    if xml_path and os.path.exists(xml_path) and rasterio is not None:
        try:
            with rasterio.open(xml_path) as ds:
                h, w = ds.height, ds.width
                if max_dim is not None and max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
                    arr = ds.read(1, out_shape=(nh, nw))
                else:
                    arr = ds.read(1)

                if arr.dtype != np.uint8:
                    p1, p99 = np.percentile(arr, [1, 99])
                    if p99 > p1:
                        arr = np.clip((arr.astype(np.float32) - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)
                    else:
                        arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return arr
        except Exception:
            pass

    # Strategy B: PDS4 tools fallback if rasterio had an issue
    if xml_path and os.path.exists(xml_path) and pds4_tools is not None:
        try:
            structures = pds4_tools.read(xml_path, lazy_load=True)
            for structure in structures:
                if structure.is_array():
                    data = structure.data
                    if len(data.shape) >= 2:
                        h, w = data.shape[:2]
                        if max_dim is not None and max(h, w) > max_dim:
                            scale = max_dim / max(h, w)
                            step = max(1, int(1.0 / scale))
                            arr = np.array(data[::step, ::step])
                        else:
                            arr = np.array(data)

                        if arr.dtype != np.uint8:
                            p1, p99 = np.percentile(arr, [1, 99])
                            if p99 > p1:
                                arr = np.clip((arr.astype(np.float32) - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)
                            else:
                                arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        return arr
        except Exception:
            pass

    # Strategy C: Raw binary direct memory mapping deduction for known ISRO / lunar sensors
    sz = os.path.getsize(img_path)
    if sz % 12000 == 0:
        # Chandrayaan-2 OHRC (12,000 samples, 1 byte uint8)
        w = 12000
        h = sz // 12000
        dtype = np.uint8
    elif sz % 8000 == 0:
        # Chandrayaan TMC (4,000 samples, 2 bytes uint16)
        w = 4000
        h = sz // 8000
        dtype = np.uint16
    else:
        # Fallback to square or standard aspect ratio check
        w = int(np.sqrt(sz))
        if w > 0 and sz % w == 0:
            h = sz // w
            dtype = np.uint8
        else:
            raise ValueError(
                f"Could not determine dimensions for binary .img file ({sz} bytes). "
                f"Please ensure its matching PDS4 .xml label is also present."
            )

    m = np.memmap(img_path, dtype=dtype, mode='r', shape=(h, w))
    if max_dim is not None and max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        step = max(1, int(1.0 / scale))
        arr = np.array(m[::step, ::step])
    else:
        arr = np.array(m)

    if arr.dtype != np.uint8:
        p1, p99 = np.percentile(arr, [1, 99])
        if p99 > p1:
            arr = np.clip((arr.astype(np.float32) - p1) * (255.0 / max(1.0, p99 - p1)), 0, 255).astype(np.uint8)
        else:
            arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return arr


def convert_pds4_to_tiff(xml_path: str, output_path: str):
    """
    Parses a PDS4 XML/IMG bundle and saves the image array as a standard TIFF 
    so it can be used by the registration pipeline.
    """
    print(f"Reading PDS4 label: {xml_path}")
    try:
        if pds4_tools is None:
            raise ImportError("pds4_tools is not installed.")
        structures = pds4_tools.read(xml_path, lazy_load=True)
        image_data = None
        for structure in structures:
            if structure.is_array():
                data = structure.data
                if len(data.shape) >= 2:
                    image_data = data
                    break
                    
        if image_data is None:
            raise ValueError("No 2D image data found in the PDS4 bundle.")
            
        print(f"Found image data with shape: {image_data.shape}, dtype: {image_data.dtype}")
        
        if image_data.dtype == np.uint8 or image_data.dtype == np.uint16 or image_data.dtype == np.float32:
            export_data = image_data
        else:
            export_data = image_data.astype(np.float32)
            
        export_data = np.squeeze(export_data)
        print(f"Saving to: {output_path}")
        cv2.imwrite(output_path, export_data)
        print("Conversion successful!")
        
    except Exception as e:
        print(f"Failed to convert PDS4 data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ISRO PDS4 to standard TIFF")
    parser.add_argument("xml_file", help="Path to the PDS4 .xml label file")
    parser.add_argument("output_file", help="Path to save the output .tif file")
    args = parser.parse_args()
    
    convert_pds4_to_tiff(args.xml_file, args.output_file)
