"""Image preprocessing utilities using OpenCV."""
import cv2 
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(
    image_path: str,
    size: Tuple[int, int] = (640, 640),
    keep_aspect: bool = False
) -> np.ndarray:
    """
    Load an image, resize, optionally keep aspect ratio with padding, 
    and convert to RGB numpy array.

    Args:
        image_path (str): Path to the input image.
        size (Tuple[int, int]): Desired output size (width, height).
        keep_aspect (bool): Whether to preserve the original aspect ratio.

    Returns:
        np.ndarray: Preprocessed RGB image.
    """
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"Invalid size argument: {size}. Must be a tuple/list of 2 integers.")

    width, height = int(size[0]), int(size[1])
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    h0, w0 = img.shape[:2]
    logger.info(f"Original image size: {w0}x{h0}")

    if keep_aspect:
        # Compute scale to maintain aspect ratio
        scale = min(width / w0, height / h0)
        nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))  # avoid zero size
        img_resized = cv2.resize(img, (nw, nh))
        
        # Compute padding
        pad_w, pad_h = width - nw, height - nh
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        img = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))
        logger.info(f"Resized with aspect ratio: {nw}x{nh}, padded to {width}x{height}")
    else:
        img = cv2.resize(img, (width, height))
        logger.info(f"Resized without aspect ratio: {width}x{height}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
