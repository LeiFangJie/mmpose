"""RTMDet raw 输出后处理和 pose 衔接，只依赖 NumPy。"""

import numpy as np


def _nms(boxes, scores, threshold):
    """执行单类 xyxy NMS。"""
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        current = order[0]
        kept.append(current)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[current, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[current, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[current, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[current, 3], boxes[order[1:], 3])
        intersection = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        area = ((boxes[current, 2] - boxes[current, 0]) *
                (boxes[current, 3] - boxes[current, 1]))
        other_area = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
                      (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = intersection / np.maximum(area + other_area - intersection, 1e-12)
        order = order[1:][iou <= threshold]
    return np.asarray(kept, dtype=np.int64)


def decode_detector_outputs(outputs, metadata, score_threshold=0.001,
                            iou_threshold=0.65, nms_pre=30000,
                            max_per_image=300):
    """复现 RTMDetHead 的 sigmoid、格点、DistancePointBBoxCoder 和 NMS。"""
    if len(outputs) != 6:
        raise ValueError('RTMDet 输出必须按 cls,bbox 交错包含 6 个张量')
    boxes, scores, labels = [], [], []
    for level, stride in enumerate((8., 16., 32.)):
        cls_logits = outputs[level * 2][0].transpose(1, 2, 0).reshape(-1, 80)
        bbox_logits = outputs[level * 2 + 1][0].transpose(1, 2, 0).reshape(-1, 4)
        height, width = outputs[level * 2].shape[-2:]
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32), indexing='ij')
        points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1) * stride
        distances = bbox_logits * stride
        level_boxes = np.stack([
            points[:, 0] - distances[:, 0], points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2], points[:, 1] + distances[:, 3]], axis=1)
        image_height, image_width = metadata['img_shape']
        level_boxes[:, 0::2] = np.clip(level_boxes[:, 0::2], 0, image_width)
        level_boxes[:, 1::2] = np.clip(level_boxes[:, 1::2], 0, image_height)
        level_scores = 1. / (1. + np.exp(-cls_logits))
        candidate = np.argwhere(level_scores > score_threshold)
        if nms_pre > 0 and candidate.shape[0] > nms_pre:
            candidate_scores = level_scores[candidate[:, 0], candidate[:, 1]]
            top_indexes = np.argpartition(candidate_scores, -nms_pre)[-nms_pre:]
            candidate = candidate[top_indexes]
        boxes.append(level_boxes[candidate[:, 0]])
        scores.append(level_scores[candidate[:, 0], candidate[:, 1]])
        labels.append(candidate[:, 1])
    boxes = np.concatenate(boxes, axis=0)
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    kept = []
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        kept.extend(indexes[_nms(boxes[indexes], scores[indexes], iou_threshold)])
    kept = np.asarray(kept, dtype=np.int64)
    kept = kept[np.argsort(scores[kept])[::-1]][:max_per_image]
    scale = metadata['scale_factor']
    boxes = boxes[kept] / np.asarray([scale[0], scale[1], scale[0], scale[1]])
    return boxes, scores[kept], labels[kept]


def select_person_boxes(boxes, scores, labels, person_label=0):
    """选择 COCO/HumanArt 的 person 类作为 RTMPose 输入框。"""
    mask = labels == person_label
    return boxes[mask], scores[mask]
