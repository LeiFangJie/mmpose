"""RTMO raw ONNX 的板端前处理、图外解码和结果还原。

运行时只依赖 NumPy 和 OpenCV。模型输入、网格顺序、bbox 解码、DCC
坐标分类和 MMPose 的输入到原图坐标变换均与源码保持一致。
"""

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np


OUTPUT_NAMES = (
    'cls_s16', 'bbox_s16', 'vis_s16', 'pose_s16',
    'cls_s32', 'bbox_s32', 'vis_s32', 'pose_s32')
STRIDES = (16, 32)
NUM_KEYPOINTS = 17
EPS = 1e-8


def sigmoid(values: np.ndarray) -> np.ndarray:
    """计算数值稳定的 sigmoid。"""
    values = np.asarray(values, dtype=np.float32)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _warp_matrix(center: np.ndarray, scale: np.ndarray,
                 output_size: Tuple[int, int]) -> np.ndarray:
    """复现 MMPose get_warp_matrix 的无旋转、保持纵横比变换。"""
    src_width, src_height = scale
    dst_width, dst_height = output_size
    src = np.array([
        center,
        center + np.array([-src_width * 0.5, 0.0], dtype=np.float32),
        center + np.array([-src_width * 0.5, -src_height * 0.5], dtype=np.float32)
    ], dtype=np.float32)
    dst = np.array([
        [dst_width * 0.5, dst_height * 0.5],
        [dst_width * 0.0, dst_height * 0.5],
        [dst_width * 0.0, dst_height * 0.0],
    ], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


class RTMOPreProcessor:
    """执行 RTMO 配置中的 BottomupResize 和输入张量整理。"""

    def __init__(self, input_size: Tuple[int, int] = (640, 640),
                 pad_value: Tuple[int, int, int] = (114, 114, 114)):
        self.input_size = tuple(input_size)
        self.pad_value = tuple(pad_value)
        if self.input_size != (640, 640):
            raise ValueError('当前 raw ONNX 契约固定为 640x640')

    def process(self, image: np.ndarray):
        """将 BGR uint8 图像变为 ONNX 的 NCHW float32 输入。

        返回值中的 metadata 必须原样传给后处理，用于恢复原图坐标。
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('输入图像必须是 HWC 三通道 BGR 图像')
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError('输入图像尺寸必须大于零')

        input_width, input_height = self.input_size
        ratio = width / float(height)
        resized_width = min(input_width, input_height * ratio)
        resized_height = min(input_height, input_width / ratio)
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = np.array([
            width * input_width / resized_width,
            height * input_height / resized_height,
        ], dtype=np.float32)
        matrix = _warp_matrix(center, scale, self.input_size)
        warped = cv2.warpAffine(
            image, matrix, self.input_size, flags=cv2.INTER_LINEAR,
            borderValue=self.pad_value)
        tensor = np.ascontiguousarray(warped.transpose(2, 0, 1), dtype=np.float32)
        metadata = {
            'original_size': (width, height),
            'input_size': self.input_size,
            'input_center': center,
            'input_scale': scale,
            'warp_matrix': matrix,
        }
        return tensor[None], metadata


def build_grid(height: int, width: int, stride: int) -> np.ndarray:
    """按 MlvlPointGenerator 的 row-major 顺序构造中心化网格。"""
    # centralize_points=True：先用 offset=0，再加 (stride-1)/2。
    center_offset = (stride - 1.0) / 2.0
    x = np.arange(width, dtype=np.float32) * stride + center_offset
    y = np.arange(height, dtype=np.float32) * stride + center_offset
    grid_x, grid_y = np.meshgrid(x, y)
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1)


def flatten_head(tensor: np.ndarray) -> np.ndarray:
    """将 NCHW raw head 转成源码使用的 B,N,C 排列。"""
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f'raw head 必须为 [1,C,H,W]，实际为 {tensor.shape}')
    return tensor.transpose(0, 2, 3, 1).reshape(1, -1, tensor.shape[1])


def decode_bbox(pred_bboxes: np.ndarray, priors: np.ndarray,
                strides: np.ndarray) -> np.ndarray:
    """复现 YOLOXPoseHead.decode_bbox。"""
    pred_bboxes = np.asarray(pred_bboxes, dtype=np.float32)
    priors = np.asarray(priors, dtype=np.float32)
    strides = np.asarray(strides, dtype=np.float32).reshape(1, -1, 1)
    centers = pred_bboxes[..., :2] * strides + priors.reshape(1, -1, 2)
    with np.errstate(over='ignore', under='ignore'):
        sizes = np.exp(pred_bboxes[..., 2:]) * strides
    return np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=-1)


def _bbox_xyxy2cs(boxes: np.ndarray, padding: float = 1.25):
    """复现 bbox_xyxy2cs。"""
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    scale = (boxes[..., 2:] - boxes[..., :2]) * padding
    return np.concatenate((center, scale), axis=-1)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算与源码 bbox_overlaps 一致的 IoU。"""
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = wh[:, 0] * wh[:, 1]
    area_box = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    area_boxes = np.maximum((boxes[:, 2] - boxes[:, 0]) *
                            (boxes[:, 3] - boxes[:, 1]), 0.0)
    return intersection / np.maximum(area_box + area_boxes - intersection, 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """复现 mmpose.evaluation.functional.nms_torch 的贪心 IoU NMS。"""
    order = np.argsort(-scores, kind='stable')
    keep = []
    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        overlaps = _iou(boxes[current], boxes[order[1:]])
        # 源码中 NaN IoU 不满足 ``> threshold``，因此会被保留。
        order = order[1:][np.isnan(overlaps) | (overlaps <= threshold)]
    return np.asarray(keep, dtype=np.int64)


class RTMODCC:
    """使用 deploy 后 DCC 参数执行 RTMO 的图外坐标分类。"""

    def __init__(self, parameters: Mapping[str, np.ndarray]):
        required = {
            'pose_to_kpts_weight', 'pose_to_kpts_bias', 'gau_ln_g',
            'gau_uv_weight', 'gau_uv_bias', 'gau_gamma', 'gau_beta',
            'gau_o_weight', 'gau_o_bias', 'x_fc_weight', 'x_fc_bias',
            'y_fc_weight', 'y_fc_bias', 'x_bins', 'y_bins', 'dim_t'}
        missing = sorted(required - set(parameters))
        if missing:
            raise ValueError(f'DCC 参数缺失: {missing}')
        self.p = {key: np.asarray(value, dtype=np.float32)
                  for key, value in parameters.items()}

    def decode(self, pose_features: np.ndarray, boxes: np.ndarray,
               grids: np.ndarray) -> np.ndarray:
        """将 pose feature、bbox 和 grid 解码为输入图坐标关键点。"""
        p = self.p
        kpt_features = pose_features @ p['pose_to_kpts_weight'].T + p['pose_to_kpts_bias']
        kpt_features = kpt_features.reshape(-1, NUM_KEYPOINTS, 128)
        residual = kpt_features
        norm = np.linalg.norm(kpt_features, axis=-1, keepdims=True) * (128.0 ** -0.5)
        features = kpt_features / np.maximum(norm, 1e-5) * p['gau_ln_g']
        uv = features @ p['gau_uv_weight'].T + p['gau_uv_bias']
        uv = np.maximum(uv, 0.0)
        u, v, base = np.split(uv, (256, 512), axis=-1)
        base = base[None]
        u, v = u[None], v[None]
        q = base * p['gau_gamma'][0][None, None, None, :] + p['gau_beta'][0][None, None]
        k = base * p['gau_gamma'][1][None, None, None, :] + p['gau_beta'][1][None, None]
        kernel = np.maximum(q @ np.swapaxes(k, -1, -2) / np.sqrt(128.0), 0.0) ** 2
        gau_output = (u * (kernel @ v)) @ p['gau_o_weight'].T + p['gau_o_bias']
        features = (p['gau_res_scale'][None, None, None, :] * residual[None] +
                    gau_output)

        bbox_cs = _bbox_xyxy2cs(boxes)
        center, scale = bbox_cs[:, :2], bbox_cs[:, 2:]
        relative_center = center - grids
        x_bins = p['x_bins'][None, :] * scale[:, 0:1] + relative_center[:, 0:1]
        y_bins = p['y_bins'][None, :] * scale[:, 1:2] + relative_center[:, 1:2]
        x_enc = self._encode_bins(x_bins, p['x_fc_weight'], p['x_fc_bias'], p['dim_t'])
        y_enc = self._encode_bins(y_bins, p['y_fc_weight'], p['y_fc_bias'], p['dim_t'])
        x_logits = features @ np.swapaxes(x_enc, -1, -2)
        y_logits = features @ np.swapaxes(y_enc, -1, -2)
        x_prob = _softmax(x_logits)
        y_prob = _softmax(y_logits)
        x = (x_prob * x_bins[None, :, None, :]).sum(axis=-1) + grids[None, :, 0:1]
        y = (y_prob * y_bins[None, :, None, :]).sum(axis=-1) + grids[None, :, 1:2]
        return np.stack((x, y), axis=-1)[0]

    @staticmethod
    def _encode_bins(bins, weight, bias, dim_t):
        frequency = bins[:, :, None] / dim_t[None, None, :]
        sine = np.concatenate((np.cos(frequency), np.sin(frequency)), axis=-1)
        return sine @ weight.T + bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    """复现 DCC 的 clamp、减最大值和归一化。"""
    logits = np.clip(logits, -5e4, 5e4)
    logits = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / (values.sum(axis=-1, keepdims=True) + EPS)


def export_dcc_parameters(model, output_path: str) -> None:
    """从已加载并已 switch_to_deploy 的 RTMO 模型导出 NumPy DCC 参数。"""
    dcc = model.head.dcc
    gau = dcc.gau
    pose_fc = dcc.pose_to_kpts
    x_fc, y_fc = dcc.x_fc, dcc.y_fc
    values = {
        'pose_to_kpts_weight': pose_fc.weight.detach().cpu().numpy(),
        'pose_to_kpts_bias': pose_fc.bias.detach().cpu().numpy(),
        'gau_ln_g': gau.ln.g.detach().cpu().numpy(),
        'gau_uv_weight': gau.uv.weight.detach().cpu().numpy(),
        'gau_uv_bias': np.zeros(gau.uv.out_features, dtype=np.float32)
        if gau.uv.bias is None else gau.uv.bias.detach().cpu().numpy(),
        'gau_gamma': gau.gamma.detach().cpu().numpy(),
        'gau_beta': gau.beta.detach().cpu().numpy(),
        'gau_o_weight': gau.o.weight.detach().cpu().numpy(),
        'gau_o_bias': np.zeros(gau.o.out_features, dtype=np.float32)
        if gau.o.bias is None else gau.o.bias.detach().cpu().numpy(),
        'x_fc_weight': x_fc.weight.detach().cpu().numpy(),
        'x_fc_bias': x_fc.bias.detach().cpu().numpy(),
        'y_fc_weight': y_fc.weight.detach().cpu().numpy(),
        'y_fc_bias': y_fc.bias.detach().cpu().numpy(),
        'x_bins': dcc.x_bins.detach().cpu().numpy(),
        'y_bins': dcc.y_bins.detach().cpu().numpy(),
        'dim_t': dcc.spe.dim_t.detach().cpu().numpy(),
        'gau_res_scale': gau.res_scale.scale.detach().cpu().numpy(),
    }
    if hasattr(dcc, 'pos_enc'):
        values['gau_beta'] = values['gau_beta'][:, None, :] + dcc.pos_enc.detach().cpu().numpy()[None]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **values)


class RTMOPostProcessor:
    """解码八个 raw 输出并还原到原图坐标。"""

    def __init__(self, parameters: Mapping[str, np.ndarray],
                 score_threshold: float = 0.1, nms_threshold: float = 0.65):
        self.dcc = RTMODCC(parameters) if parameters else None
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def process(self, raw_outputs: Mapping[str, np.ndarray], metadata: Mapping):
        """执行源码顺序的筛选、NMS、DCC 和坐标还原。"""
        missing = [name for name in OUTPUT_NAMES if name not in raw_outputs]
        if missing:
            raise ValueError(f'缺少 raw 输出: {missing}')
        levels = []
        for level, stride in enumerate(STRIDES):
            suffix = f's{stride}'
            cls = flatten_head(raw_outputs[f'cls_{suffix}'])
            boxes = flatten_head(raw_outputs[f'bbox_{suffix}'])
            visibility = flatten_head(raw_outputs[f'vis_{suffix}'])
            pose = flatten_head(raw_outputs[f'pose_{suffix}'])
            height, width = raw_outputs[f'cls_{suffix}'].shape[-2:]
            grid = build_grid(height, width, stride)
            levels.append((cls, boxes, visibility, pose, grid,
                           np.full((grid.shape[0],), stride, dtype=np.float32)))
        cls = np.concatenate([item[0] for item in levels], axis=1)[0]
        boxes = np.concatenate([item[1] for item in levels], axis=1)
        visibility = np.concatenate([item[2] for item in levels], axis=1)[0]
        pose = np.concatenate([item[3] for item in levels], axis=1)[0]
        grids = np.concatenate([item[4] for item in levels], axis=0)
        strides = np.concatenate([item[5] for item in levels], axis=0)
        scores = sigmoid(cls[:, 0])
        keep = scores > self.score_threshold
        boxes = decode_bbox(boxes, grids, strides)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        visibility = sigmoid(visibility[keep])
        pose = pose[keep]
        grids = grids[keep]
        if boxes.shape[0] == 0:
            return self._empty_result()
        keep_nms = nms(boxes, scores, self.nms_threshold)
        boxes, scores = boxes[keep_nms], scores[keep_nms]
        visibility, pose, grids = visibility[keep_nms], pose[keep_nms], grids[keep_nms]
        input_width, input_height = metadata['input_size']
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, input_width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, input_height)
        if self.dcc is None:
            keypoints = np.zeros((boxes.shape[0], NUM_KEYPOINTS, 2), dtype=np.float32)
        else:
            keypoints = self.dcc.decode(pose, boxes, grids)
        keypoints = self._to_original(keypoints, metadata)
        boxes = self._to_original_boxes(boxes, metadata)
        return {
            'bboxes': boxes, 'scores': scores, 'keypoints': keypoints,
            'keypoint_scores': visibility,
        }

    @staticmethod
    def _to_original(points, metadata):
        input_size = np.asarray(metadata['input_size'], dtype=np.float32)
        scale = np.asarray(metadata['input_scale'], dtype=np.float32)
        center = np.asarray(metadata['input_center'], dtype=np.float32)
        return points / input_size * scale + center - 0.5 * scale

    def _to_original_boxes(self, boxes, metadata):
        return self._to_original(boxes.reshape(-1, 2, 2), metadata).reshape(-1, 4)

    @staticmethod
    def _empty_result():
        return {
            'bboxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros((0,), dtype=np.float32),
            'keypoints': np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32),
            'keypoint_scores': np.zeros((0, NUM_KEYPOINTS), dtype=np.float32),
        }


def load_dcc_parameters(path: str) -> Dict[str, np.ndarray]:
    """加载 export_dcc_parameters 生成的 NumPy 参数文件。"""
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}
