"""开发机 smoke：比较 PyTorch raw 输出、ONNX raw 输出和图外后处理。"""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmpose.registry import MODELS
from mmpose.structures import PoseDataSample
from mmpose.utils import register_all_modules
from tools.deploy.rtmo.custom_export_raw_onnx import RTMORawExportWrapper
from RTMO.custom_rtmo_preprocess_postprocess import (
    OUTPUT_NAMES, RTMOPostProcessor, RTMOPreProcessor, export_dcc_parameters)


def _read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return image


def _as_numpy(value):
    """兼容 MMPose 返回 Tensor 或 numpy.ndarray 的两种版本。"""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    """运行单张图的 PyTorch/ONNX raw 数值比对。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--dcc-params', default='RTMO/custom_rtmo_dcc_params.npz')
    parser.add_argument('--report', default='RTMO/custom_rtmo_smoke_report.json')
    args = parser.parse_args()

    register_all_modules()
    config = Config.fromfile(args.config)
    imports = config.get('custom_imports', {})
    import_modules_from_strings(imports=imports.get('imports', []),
                                allow_failed_imports=False)
    model = MODELS.build(config.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    model.switch_to_deploy()
    wrapper = RTMORawExportWrapper(model, (640, 640)).eval()
    image = _read_image(args.image)
    tensor, metadata = RTMOPreProcessor().process(image)
    torch_input = torch.from_numpy(tensor)
    with torch.no_grad():
        torch_outputs = wrapper(torch_input)
    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    onnx_values = session.run(list(OUTPUT_NAMES), {'images': tensor})
    raw_errors = {
        name: float(np.max(np.abs(torch_value.numpy() - onnx_value)))
        for name, torch_value, onnx_value in zip(OUTPUT_NAMES, torch_outputs, onnx_values)
    }
    raw_relative_errors = {
        name: float(np.max(np.abs(torch_value.numpy() - onnx_value)) /
                    max(float(np.max(np.abs(torch_value.numpy()))), 1e-6))
        for name, torch_value, onnx_value in zip(OUTPUT_NAMES, torch_outputs, onnx_values)
    }
    export_dcc_parameters(model, args.dcc_params)
    postprocessor = RTMOPostProcessor(dict(np.load(args.dcc_params)))
    result_torch = postprocessor.process(
        dict(zip(OUTPUT_NAMES, [value.numpy() for value in torch_outputs])), metadata)
    result_onnx = postprocessor.process(dict(zip(OUTPUT_NAMES, onnx_values)), metadata)
    data_sample = PoseDataSample(metainfo={
        'ori_shape': image.shape[:2],
        'img_shape': (640, 640),
        'input_size': metadata['input_size'],
        'input_center': metadata['input_center'],
        'input_scale': metadata['input_scale'],
    })
    with torch.no_grad():
        source_result = model.predict(torch_input, [data_sample])[0].pred_instances
    source_arrays = {
        'bboxes': _as_numpy(source_result.bboxes),
        'scores': _as_numpy(source_result.scores),
        'keypoints': _as_numpy(source_result.keypoints),
        'keypoint_scores': _as_numpy(source_result.keypoint_scores),
    }
    rng = np.random.default_rng(20261028)
    probe_features = rng.normal(size=(3, 256)).astype(np.float32)
    probe_boxes = np.array([
        [32.0, 48.0, 180.0, 220.0],
        [210.0, 100.0, 390.0, 410.0],
        [400.0, 60.0, 590.0, 500.0],
    ], dtype=np.float32)
    probe_grids = np.array([[7.5, 7.5], [167.5, 199.5], [391.5, 71.5]],
                           dtype=np.float32)
    probe_boxes_cs = torch.from_numpy(
        np.concatenate(((probe_boxes[:, :2] + probe_boxes[:, 2:]) * 0.5,
                        (probe_boxes[:, 2:] - probe_boxes[:, :2]) * 1.25), axis=1))
    with torch.no_grad():
        probe_source = model.head.dcc.forward_test(
            torch.from_numpy(probe_features), probe_boxes_cs,
            torch.from_numpy(probe_grids)).numpy()
    probe_deploy = postprocessor.dcc.decode(
        probe_features, probe_boxes, probe_grids)
    dcc_probe_max_abs_error = float(np.max(np.abs(probe_source - probe_deploy)))
    postprocess_errors = {}
    postprocess_finite_errors = {}
    same_count = result_torch['bboxes'].shape[0] == source_arrays['bboxes'].shape[0]
    source_shapes = {key: list(value.shape) for key, value in source_arrays.items()}
    torch_shapes = {key: list(value.shape) for key, value in result_torch.items()}
    if same_count:
        for name, values in source_arrays.items():
            if values.shape == result_torch[name].shape and values.size:
                difference = np.abs(values - result_torch[name])
                raw_error = float(np.max(difference))
                postprocess_errors[name] = raw_error if np.isfinite(raw_error) else None
                finite = np.isfinite(difference)
                postprocess_finite_errors[name] = (
                    float(np.max(difference[finite])) if finite.any() else None)
            elif values.shape != result_torch[name].shape:
                same_count = False
    report = {
        'input_shape': list(tensor.shape),
        'raw_max_abs_error': raw_errors,
        'raw_max_abs_error_all': max(raw_errors.values()),
        'raw_relative_error': raw_relative_errors,
        'torch_postprocess_instances': int(result_torch['bboxes'].shape[0]),
        'onnx_postprocess_instances': int(result_onnx['bboxes'].shape[0]),
        'matches_source_instance_count': same_count,
        'source_output': source_shapes,
        'torch_output': torch_shapes,
        'postprocess_max_abs_error': postprocess_errors,
        'postprocess_finite_max_abs_error': postprocess_finite_errors,
        'dcc_probe_max_abs_error': dcc_probe_max_abs_error,
        'torch_output': {key: list(value.shape) for key, value in result_torch.items()},
        'onnx_output': {key: list(value.shape) for key, value in result_onnx.items()},
        'onnx': str(Path(args.onnx)),
        'dcc_parameters': str(Path(args.dcc_params)),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
