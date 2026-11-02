"""RTMDet 板端前处理，只依赖 NumPy 和 OpenCV。"""

import cv2
import numpy as np


DETECTOR_SIZE = (640, 640)  # (宽, 高)
DETECTOR_MEAN = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
DETECTOR_STD = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)


def preprocess_detector_image(image_bgr):
    """按 RTMDet test pipeline 等比例缩放、补边和归一化图像。"""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('image_bgr 必须是 HWC 三通道图像')
    original_height, original_width = image_bgr.shape[:2]
    scale = min(DETECTOR_SIZE[0] / original_width,
                DETECTOR_SIZE[1] / original_height)
    resized_width = int(original_width * scale + 0.5)
    resized_height = int(original_height * scale + 0.5)
    resized = cv2.resize(
        image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((DETECTOR_SIZE[1], DETECTOR_SIZE[0], 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    normalized = (padded.astype(np.float32) - DETECTOR_MEAN) / DETECTOR_STD
    return normalized.transpose(2, 0, 1)[None].astype(np.float32), {
        'scale_factor': np.asarray([
            resized_width / original_width,
            resized_height / original_height], dtype=np.float32),
        'img_shape': (DETECTOR_SIZE[1], DETECTOR_SIZE[0]),
        'content_shape': (resized_height, resized_width),
        'original_shape': (original_height, original_width),
    }
