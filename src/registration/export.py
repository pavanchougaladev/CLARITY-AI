import pandas as pd
import numpy as np
import os
import logging
import cv2

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger(__name__)

def export_asp_compatible_tiepoints(pts_src: np.ndarray, pts_ref: np.ndarray, confidences: np.ndarray, filepath: str):
    """
    Exports tie-points in a format ready for NASA Ames Stereo Pipeline (ASP) 
    bundle_adjust, bridging our 2D matcher with 3D photogrammetry pipelines.
    """
    if len(pts_src) == 0:
        logger.warning("No tie points to export.")
        return
        
    df = pd.DataFrame({
        'x_src': pts_src[:, 0], 
        'y_src': pts_src[:, 1],
        'x_ref': pts_ref[:, 0], 
        'y_ref': pts_ref[:, 1],
        'confidence': confidences
    })
    
    # ASP expects specific header/column structures for external tie-points
    df.to_csv(filepath, index=False, header=['x1', 'y1', 'x2', 'y2', 'weight'])
    logger.info(f"Exported {len(df)} tie points to {filepath}")
    return filepath

def export_asp_match_file(pts_src: np.ndarray, pts_ref: np.ndarray, filepath: str):
    """
    Exports tie-points in ASP's plain text .match format.
    Format: col1 row1 col2 row2
    """
    if len(pts_src) == 0:
        logger.warning("No tie points to export.")
        return
        
    with open(filepath, 'w') as f:
        for (x1, y1), (x2, y2) in zip(pts_src, pts_ref):
            f.write(f"{x1} {y1} {x2} {y2}\n")
    
    logger.info(f"Exported {len(pts_src)} tie points to {filepath}")

def export_geotiff(
    img: np.ndarray,
    filepath: str,
    reference_tif: str = None,
    transform_matrix = None,
    already_warped: bool = True,
):
    """
    Export image as a GeoTIFF.

    Parameters
    ----------
    img : np.ndarray
        Raster array (2D grayscale or 3D multi-channel).
    filepath : str
        Destination path (.tif/.tiff).
    reference_tif : str, optional
        Path to reference GeoTIFF providing CRS and affine georeferencing transform.
    transform_matrix : array-like, optional
        3x3 transformation matrix mapping src -> ref coordinates.
    already_warped : bool, default True
        If True, `img` has already been resampled/warped into the reference frame coordinates,
        meaning its pixel indices directly map to `reference_tif`'s affine transform.
        If False, `img` is in unwarped source coordinates, and `transform_matrix` is composed
        with the reference georeference transform (Geo = T_ref * H).
    """
    if not HAS_RASTERIO:
        logger.error("Rasterio is not installed. GeoTIFF export is unavailable.")
        # Fallback to standard TIFF
        cv2.imwrite(filepath, img)
        return
        
    try:
        if len(img.shape) == 2:
            count = 1
            height, width = img.shape
        else:
            height, width, count = img.shape
            
        kwargs = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': img.dtype,
            'compress': 'lzw'
        }
        
        if reference_tif and os.path.exists(reference_tif):
            with rasterio.open(reference_tif) as src:
                kwargs.update({
                    'crs': src.crs,
                    'transform': src.transform
                })
                # If img is unwarped, compose transform: Geo = T_ref * H
                # If img is ALREADY warped to reference raster dimensions, do not double-transform!
                if not already_warped and transform_matrix is not None and kwargs.get('transform') is not None:
                    try:
                        from rasterio.transform import Affine
                        H = np.array(transform_matrix, dtype=np.float64)
                        if H.shape == (3, 3):
                            h_affine = Affine(H[0, 0], H[0, 1], H[0, 2],
                                              H[1, 0], H[1, 1], H[1, 2])
                            kwargs['transform'] = kwargs['transform'] * h_affine
                            logger.info("Composed affine geospatial transform with reference GeoTIFF.")
                    except Exception as e_trans:
                        logger.warning(f"Could not compose spatial transform: {e_trans}")
                
        with rasterio.open(filepath, 'w', **kwargs) as dst:
            if count == 1:
                dst.write(img, 1)
            else:
                for i in range(count):
                    dst.write(img[:,:,i], i+1)
                    
        logger.info(f"Successfully exported GeoTIFF: {filepath}")
    except Exception as e:
        logger.error(f"Failed to export GeoTIFF: {e}")
        cv2.imwrite(filepath, img) # Fallback
