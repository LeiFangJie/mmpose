"""导出 RTMO 两尺度原始预测张量，不包含 DCC 和候选后处理。"""

import argparse
import copy
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmpose.registry import MODELS
from mmpose.utils import register_all_modules


class RTMORawExportWrapper(nn.Module):
    """固定输入尺寸，直接输出 RTMO head 的四类原始预测张量。

    参数:
        model: 已构建并加载权重的 BottomupPoseEstimator。
        input_size: 固定输入的 (height, width)。

    返回:
        按 stride 16、stride 32 排列的 cls、bbox、visibility、pose 张量。
    """

    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (640, 640)):
        super().__init__()
        self.model = model
        self.input_size = tuple(input_size)
        self._focus_conv = self._build_fused_focus_conv(model.backbone.stem)

    @staticmethod
    def _build_fused_focus_conv(focus: nn.Module) -> nn.Module:
        """把 Focus 重排和 3x3 卷积精确合并为首层 7x7 卷积。"""
        conv_module = copy.deepcopy(focus.conv)
        source_conv = conv_module.conv
        if not isinstance(source_conv, nn.Conv2d):
            raise TypeError('Focus stem 必须使用 nn.Conv2d')
        if (source_conv.kernel_size != (3, 3)
                or source_conv.stride != (1, 1)
                or source_conv.padding != (1, 1)
                or source_conv.dilation != (1, 1)
                or source_conv.groups != 1
                or source_conv.padding_mode != 'zeros'):
            raise ValueError('Focus stem 仅支持 3x3、stride=1、padding=1 的普通卷积')

        input_channels = focus.conv.conv.in_channels // 4
        if source_conv.in_channels != input_channels * 4:
            raise ValueError('Focus stem 输入通道数必须是原图通道数的 4 倍')
        fused_conv = nn.Conv2d(
            input_channels,
            source_conv.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=source_conv.bias is not None,
            padding_mode=source_conv.padding_mode,
            device=source_conv.weight.device,
            dtype=source_conv.weight.dtype)

        # 在 7x7 核左上预留一行一列零，使其和 6x6、padding=2 的等价卷积对齐。
        with torch.no_grad():
            fused_conv.weight.zero_()
            source_weight = source_conv.weight
            for offset_index, (offset_y, offset_x) in enumerate(
                    ((0, 0), (1, 0), (0, 1), (1, 1))):
                for kernel_y in range(3):
                    for kernel_x in range(3):
                        target_y = 2 * kernel_y + offset_y + 1
                        target_x = 2 * kernel_x + offset_x + 1
                        channel_start = offset_index * input_channels
                        channel_end = channel_start + input_channels
                        fused_conv.weight[:, :, target_y, target_x].copy_(
                            source_weight[:, channel_start:channel_end,
                                          kernel_y, kernel_x])
            if source_conv.bias is not None:
                fused_conv.bias.copy_(source_conv.bias)

        conv_module.conv = fused_conv
        return conv_module

    def _run_focus(self, images: torch.Tensor) -> torch.Tensor:
        """直接运行等价卷积，避免首节点为 Permute。"""
        return self._focus_conv(images)

    @staticmethod
    def _split_class_pose_features(feature: torch.Tensor):
        """按通道拆分分类与姿态特征，并为 ONNX Slice 指定有限结束索引。"""
        channel_count = feature.shape[1]
        half_channels = channel_count // 2
        return (feature[:, :half_channels],
                feature[:, half_channels:channel_count])

    @property
    def backbone(self):
        """返回原始 backbone，便于核对 Focus 部署路径。"""
        return self.model.backbone

    def forward(self, images: torch.Tensor):
        expected_shape = (1, 3, self.input_size[0], self.input_size[1])
        if tuple(images.shape) != expected_shape:
            raise ValueError(f'输入必须固定为 {list(expected_shape)}，实际为 {list(images.shape)}')

        backbone = self.model.backbone
        features = []
        x = images
        for layer_index, layer_name in enumerate(backbone.layers):
            if layer_name == 'stem':
                x = self._run_focus(x)
            else:
                x = getattr(backbone, layer_name)(x)
            if layer_index in backbone.out_indices:
                features.append(x)

        if getattr(self.model, 'with_neck', False):
            features = self.model.neck(tuple(features))
        else:
            features = tuple(features)

        head_module = self.model.head.head_module
        cls_outputs, bbox_outputs = [], []
        visibility_outputs, pose_outputs = [], []
        for level, feature in enumerate(features):
            cls_feature, pose_feature = self._split_class_pose_features(feature)
            cls_feature = head_module.conv_cls[level](cls_feature)
            pose_feature = head_module.conv_pose[level](pose_feature)
            cls_outputs.append(head_module.out_cls[level](cls_feature))
            bbox_outputs.append(head_module.out_bbox[level](pose_feature))
            visibility_outputs.append(
                head_module.out_kpt_vis[level](pose_feature))
            if getattr(head_module, 'pose_vec_channels', 0) > 0 or hasattr(
                    head_module, 'out_pose'):
                pose_feature = head_module.out_pose[level](pose_feature)
            pose_outputs.append(pose_feature)

        outputs = []
        for level in range(len(features)):
            outputs.extend((cls_outputs[level], bbox_outputs[level],
                            visibility_outputs[level], pose_outputs[level]))
        return tuple(outputs)


def parse_args():
    """解析导出命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='模型配置文件')
    parser.add_argument('--checkpoint', required=True, help='训练 checkpoint')
    parser.add_argument('--output', required=True, help='输出 ONNX 路径')
    parser.add_argument('--audit', action='store_true', help='导出后执行静态算子审计')
    return parser.parse_args()


def main():
    """构建模型、导出固定 opset 9 图并可选执行审计。"""
    args = parse_args()
    register_all_modules()
    config = Config.fromfile(args.config)
    custom_imports = config.get('custom_imports', {})
    import_modules_from_strings(
        imports=custom_imports.get('imports', []),
        allow_failed_imports=custom_imports.get('allow_failed_imports', False))
    model = MODELS.build(config.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    model.switch_to_deploy()

    input_size = tuple(config.model.get('test_cfg', {}).get('input_size', (640, 640)))
    if input_size != (640, 640):
        raise ValueError(f'本导出器固定要求 640x640，配置为 {input_size}')
    wrapper = RTMORawExportWrapper(model, input_size).eval()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros((1, 3, *input_size), dtype=torch.float32)
    output_names = [
        'cls_s16', 'bbox_s16', 'vis_s16', 'pose_s16',
        'cls_s32', 'bbox_s32', 'vis_s32', 'pose_s32'
    ]
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            opset_version=9,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamo=False)

    import onnx

    try:
        from tools.deploy.rtmo.custom_onnx_audit import remove_identity_nodes
    except ModuleNotFoundError:
        from custom_onnx_audit import remove_identity_nodes
    exported_graph = onnx.load(str(output_path))
    removed_identity_count = remove_identity_nodes(exported_graph)
    onnx.checker.check_model(exported_graph)
    onnx.save(exported_graph, str(output_path))
    print(f'已消除 Identity 节点: {removed_identity_count}')

    if args.audit:
        import sys

        try:
            from tools.deploy.rtmo.custom_onnx_audit import audit_onnx
        except ModuleNotFoundError:
            from custom_onnx_audit import audit_onnx
        report = audit_onnx(output_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report['static_onnx_contract_passed']:
            sys.exit(1)


if __name__ == '__main__':
    main()
