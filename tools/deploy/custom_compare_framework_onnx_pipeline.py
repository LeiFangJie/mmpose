"""对比 PyTorch checkpoint 与 ONNX 独立部署链路。

脚本分开报告三类误差：

1. 使用完全相同输入 tensor 的 PyTorch/ONNX raw 模型误差；
2. 使用完全相同特征的 PyTorch RTMCC/SimCC head 与 NumPy 后处理误差；
3. 两条链路分别使用自己的检测框、前处理和后处理后的端到端误差。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
import onnxruntime as ort
import torch
from mmcv.transforms import Compose

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from RTMDet.custom_rtmdet_postprocess import (  # noqa: E402
    decode_detector_outputs, select_person_boxes)
from RTMDet.custom_rtmdet_preprocess import (  # noqa: E402
    preprocess_detector_image)
from RTMPose.custom_rtmpose_postprocess import (  # noqa: E402
    decode_simcc, load_postprocess_weights, simcc_from_feature)
from RTMPose.custom_rtmpose_preprocess import (  # noqa: E402
    preprocess_pose_image)


def _max_error(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个数组的最大绝对误差。"""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.size == 0 and second.size == 0:
        return 0.0
    if first.shape != second.shape:
        return float('inf')
    return float(np.max(np.abs(first.astype(np.float64) -
                             second.astype(np.float64))))


def _per_tensor_error(first: Iterable[np.ndarray],
                      second: Iterable[np.ndarray]) -> Dict[str, float]:
    """计算多输出模型逐输出和整体最大绝对误差。"""
    first = list(first)
    second = list(second)
    if len(first) != len(second):
        raise ValueError(
            f'输出数量不一致: PyTorch={len(first)}, ONNX={len(second)}')
    errors = {
        f'output_{index}_max_absolute_error': _max_error(left, right)
        for index, (left, right) in enumerate(zip(first, second))
    }
    errors['max_absolute_error'] = max(errors.values(), default=0.0)
    return errors


def _load_framework_detector(checkpoint):
    """加载 MMDetection 原生 RTMDet checkpoint。"""
    import mmdet
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    config_path = Path(mmdet.__file__).parent / \
        '.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
    return init_detector(str(config_path), checkpoint, device='cpu')


def _run_framework_detector(image, checkpoint):
    """运行 PyTorch RTMDet 原生前处理、推理和后处理。"""
    from mmdet.apis import inference_detector

    model = _load_framework_detector(checkpoint)
    result = inference_detector(model, image)
    instances = result.pred_instances
    return (instances.bboxes.detach().cpu().numpy(),
            instances.scores.detach().cpu().numpy(),
            instances.labels.detach().cpu().numpy(), model)


def _run_framework_detector_raw(model, detector_input):
    """在指定 detector input 上运行 PyTorch raw RTMDet head。"""
    head = model.bbox_head
    input_tensor = torch.from_numpy(detector_input)
    with torch.no_grad():
        features = model.extract_feat(input_tensor)
        outputs = []
        for level, feature in enumerate(features):
            cls_feature = feature
            reg_feature = feature
            for layer in head.cls_convs[level]:
                cls_feature = layer(cls_feature)
            for layer in head.reg_convs[level]:
                reg_feature = layer(reg_feature)
            outputs.extend([
                head.rtm_cls[level](cls_feature),
                head.rtm_reg[level](reg_feature)])
            if head.with_objectness:
                outputs.append(head.rtm_obj[level](reg_feature))
    return tuple(output.detach().cpu().numpy() for output in outputs)


def _load_framework_pose(checkpoint, config_path):
    """加载 MMPose 原生 RTMPose checkpoint。"""
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    torch.manual_seed(0)
    return init_model(config_path, checkpoint, device='cpu')


def _framework_pose_input(model, image, bbox):
    """运行 MMPose 原生 pipeline，返回姿态模型输入 tensor。"""
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    data = pipeline({
        'img': image,
        'bbox': np.asarray([bbox], dtype=np.float32),
        'bbox_score': np.ones(1, dtype=np.float32),
        **model.dataset_meta,
    })
    prepared = model.data_preprocessor(
        {'inputs': [data['inputs']], 'data_samples': [data['data_samples']]},
        training=False)
    return prepared['inputs'].detach().cpu().numpy()


def _run_framework_pose(model, image, bbox):
    """运行 PyTorch RTMPose 原生前处理、head 和 SimCC decoder。"""
    from mmpose.apis import inference_topdown

    result = inference_topdown(
        model, image, np.asarray([bbox], dtype=np.float32))[0]
    instances = result.pred_instances
    return (instances.keypoints.copy(),
            instances.keypoint_scores.copy())


def _run_framework_pose_raw(model, pose_input):
    """在指定 pose input 上运行 PyTorch backbone、7x7 Conv 和完整 head。"""
    input_tensor = torch.from_numpy(pose_input)
    with torch.no_grad():
        backbone_feature = model.backbone(input_tensor)[-1]
        feature = model.head.final_layer(backbone_feature)
        simcc_x, simcc_y = model.head((backbone_feature,))
    return (
        backbone_feature.detach().cpu().numpy(),
        feature.detach().cpu().numpy(),
        simcc_x.detach().cpu().numpy(),
        simcc_y.detach().cpu().numpy(),
    )


def _run_framework_pose_head_from_feature(model, feature):
    """只运行 PyTorch head 的 7x7 Conv 后续部分。"""
    feature_tensor = torch.from_numpy(feature)
    with torch.no_grad():
        tokens = model.head.mlp(feature_tensor.flatten(2))
        tokens = model.head.gau(tokens)
        simcc_x = model.head.cls_x(tokens)
        simcc_y = model.head.cls_y(tokens)
    return simcc_x.detach().cpu().numpy(), simcc_y.detach().cpu().numpy()


def _select_top_person(boxes, scores, labels):
    """选择分数最高的 person 框，并返回全部 person 结果。"""
    person_boxes, person_scores = select_person_boxes(
        boxes, scores, labels)
    if not len(person_boxes):
        raise RuntimeError('检测结果中没有 person 框')
    top_index = int(np.argmax(person_scores))
    return person_boxes, person_scores, person_boxes[top_index], top_index


def compare(args):
    """执行 raw、后处理和独立端到端三层对照。"""
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f'图像无法读取: {args.image}')

    # PyTorch detector 使用原生完整链路，ONNX detector 使用独立部署链路。
    framework_boxes, framework_scores, framework_labels, detector_model = \
        _run_framework_detector(image, args.det_checkpoint)
    detector_input, detector_meta = preprocess_detector_image(image)
    detector_session = ort.InferenceSession(
        args.det_onnx, providers=['CPUExecutionProvider'])
    detector_outputs = detector_session.run(
        None, {detector_session.get_inputs()[0].name: detector_input})
    torch_detector_outputs = _run_framework_detector_raw(
        detector_model, detector_input)
    detector_raw_errors = _per_tensor_error(
        torch_detector_outputs, detector_outputs)
    custom_boxes, custom_scores, custom_labels = decode_detector_outputs(
        detector_outputs, detector_meta)
    framework_person_boxes, framework_person_scores, framework_box, \
        framework_top_index = _select_top_person(
            framework_boxes, framework_scores, framework_labels)
    custom_person_boxes, custom_person_scores, custom_box, custom_top_index = \
        _select_top_person(custom_boxes, custom_scores, custom_labels)

    # 同一个 custom pose input 用于 PyTorch/ONNX raw 对照，隔离模型转换误差。
    custom_pose_input, custom_center, custom_scale = preprocess_pose_image(
        image, custom_box)
    pose_session = ort.InferenceSession(
        args.pose_onnx, providers=['CPUExecutionProvider'])
    custom_feature = pose_session.run(
        None, {pose_session.get_inputs()[0].name: custom_pose_input})[0]
    pose_model = _load_framework_pose(args.pose_checkpoint, args.pose_config)
    torch_backbone, torch_feature, torch_x, torch_y = _run_framework_pose_raw(
        pose_model, custom_pose_input)
    weights = load_postprocess_weights(args.pose_weights)
    custom_x, custom_y = simcc_from_feature(custom_feature, weights)
    torch_custom_feature_x, torch_custom_feature_y = \
        _run_framework_pose_head_from_feature(pose_model, custom_feature)

    # 两条端到端链路分别使用自己的 detector bbox。
    framework_keypoints, framework_keypoint_scores = _run_framework_pose(
        pose_model, image, framework_box)
    custom_keypoints, custom_keypoint_scores = decode_simcc(
        custom_x, custom_y, custom_center, custom_scale)

    framework_pose_input = _framework_pose_input(
        pose_model, image, framework_box)
    custom_pose_input_for_framework_box, _, _ = preprocess_pose_image(
        image, framework_box)
    framework_input_error = _max_error(
        framework_pose_input, custom_pose_input_for_framework_box)

    report = {
        'image': str(Path(args.image).resolve()),
        'checkpoints': {
            'detector': str(Path(args.det_checkpoint).resolve()),
            'pose': str(Path(args.pose_checkpoint).resolve()),
        },
        'artifacts': {
            'detector_onnx': str(Path(args.det_onnx).resolve()),
            'pose_onnx': str(Path(args.pose_onnx).resolve()),
            'pose_postprocess_weights': str(
                Path(args.pose_weights).resolve()),
        },
        'detector': {
            'raw_model_same_input': detector_raw_errors,
            'framework_person_count': int(len(framework_person_boxes)),
            'custom_person_count': int(len(custom_person_boxes)),
            'framework_top_person_index': framework_top_index,
            'custom_top_person_index': custom_top_index,
            'framework_top_person_box': framework_box.tolist(),
            'custom_top_person_box': custom_box.tolist(),
            'top_person_box_max_absolute_error': _max_error(
                framework_box, custom_box),
            'top_person_score_absolute_error': _max_error(
                framework_person_scores[framework_top_index],
                custom_person_scores[custom_top_index]),
            'input_shape': list(detector_input.shape),
        },
        'pose': {
            'raw_model_same_input': {
                'feature_max_absolute_error': _max_error(
                    torch_feature, custom_feature),
                'simcc_x_max_absolute_error': _max_error(
                    torch_x, custom_x),
                'simcc_y_max_absolute_error': _max_error(
                    torch_y, custom_y),
            },
            'postprocess_same_feature': {
                'simcc_x_max_absolute_error': _max_error(
                    torch_custom_feature_x, custom_x),
                'simcc_y_max_absolute_error': _max_error(
                    torch_custom_feature_y, custom_y),
            },
            'native_preprocess_same_framework_box_max_absolute_error': (
                framework_input_error),
            'framework_box': framework_box.tolist(),
            'custom_box': custom_box.tolist(),
            'framework_keypoint_max_absolute_error': _max_error(
                framework_keypoints, custom_keypoints),
            'framework_keypoint_score_max_absolute_error': _max_error(
                framework_keypoint_scores, custom_keypoint_scores),
            'custom_pose_input_shape': list(custom_pose_input.shape),
            'framework_pose_input_shape': list(framework_pose_input.shape),
            'feature_shape': list(custom_feature.shape),
        },
        'contracts': {
            'detector_input': list(detector_input.shape),
            'pose_input': list(custom_pose_input.shape),
            'pose_feature': list(custom_feature.shape),
        },
    }
    return report


def main():
    """解析命令行并写出分阶段对照报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--pose-checkpoint', required=True)
    parser.add_argument('--pose-config', default=(
        'configs/body_2d_keypoint/rtmpose/custom/'
        'custom_rtmpose-t_8xb32-420e_private-256x192.py'))
    parser.add_argument('--det-onnx', default=(
        'RTMDet/custom_rtmdet_tiny_humanart_raw_opset9.onnx'))
    parser.add_argument('--pose-onnx', default=(
        'RTMPose/custom_rtmpose_t_raw_feature_opset9.onnx'))
    parser.add_argument('--pose-weights', default=(
        'RTMPose/custom_rtmpose_postprocess_weights.npz'))
    parser.add_argument('--report', default=(
        'RTMPose/custom_framework_onnx_prepost_compare.json'))
    args = parser.parse_args()
    report = compare(args)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
