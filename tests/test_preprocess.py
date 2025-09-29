from pipeline.preprocess import preprocess_image
import pytest
import os
import cv2
import numpy as np
def test_preprocess_image(tmp_path):

    dummy = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(path), dummy)
    result = preprocess_image(str(path))
    assert result.shape == (640, 640, 3)
