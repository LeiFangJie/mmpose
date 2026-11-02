"""RTMPose 板端前处理，只依赖 NumPy 和 OpenCV。"""

import cv2
import numpy as np


POSE_INPUT_SIZE = (192, 256)  # (宽, 高)
POSE_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def _bbox_center_scale(bbox_xyxy):
    """复现 MMPose GetBBoxCenterScale 的 padding=1.25。"""
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    center = (bbox[:2] + bbox[2:]) * 0.5
    scale = (bbox[2:] - bbox[:2]) * 1.25
    return center, scale


def _fix_aspect_ratio(scale, aspect_ratio):
    """按 TopdownAffine 规则将框调整到输入宽高比。"""
    width, height = float(scale[0]), float(scale[1])
    if width > height * aspect_ratio:
        return np.asarray([width, width / aspect_ratio], dtype=np.float32)
    return np.asarray([height * aspect_ratio, height], dtype=np.float32)


def _warp_matrix(center, scale, output_size):
    """计算 use_udp=False 时 MMPose 使用的三点仿射矩阵。"""
    dst_width, dst_height = output_size
    source_direction = np.asarray([-scale[0] * 0.5, 0.], dtype=np.float32)
    source_point = center + source_direction
    destination_center = np.asarray(
        [dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    destination_point = destination_center + np.asarray(
        [-dst_width * 0.5, 0.], dtype=np.float32)
    source_third = source_point + np.asarray(
        [source_direction[1], -source_direction[0]], dtype=np.float32)
    destination_direction = destination_point - destination_center
    destination_third = destination_point + np.asarray(
        [destination_direction[1], -destination_direction[0]], dtype=np.float32)
    source = np.asarray([center, source_point, source_third], dtype=np.float32)
    destination = np.asarray(
        [destination_center, destination_point, destination_third],
        dtype=np.float32)
    return cv2.getAffineTransform(source, destination)


def preprocess_pose_image(image_bgr, bbox_xyxy):
    """裁剪并归一化一个人体框。

    参数:
        image_bgr: HWC、uint8、BGR 图像。
        bbox_xyxy: 原图坐标中的 ``[x1, y1, x2, y2]``。

    返回:
        tuple: ``(NCHW float32 输入, center, scale)``。
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('image_bgr 必须是 HWC 三通道图像')
    center, scale = _bbox_center_scale(bbox_xyxy)
    scale = _fix_aspect_ratio(scale, POSE_INPUT_SIZE[0] / POSE_INPUT_SIZE[1])
    matrix = _warp_matrix(center, scale, POSE_INPUT_SIZE)
    cropped = cv2.warpAffine(
        image_bgr, matrix, POSE_INPUT_SIZE, flags=cv2.INTER_LINEAR)
    rgb = cropped[..., ::-1].astype(np.float32)
    normalized = (rgb - POSE_MEAN) / POSE_STD
    return normalized.transpose(2, 0, 1)[None].astype(np.float32), center, scale
