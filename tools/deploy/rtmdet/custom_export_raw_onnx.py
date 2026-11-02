"""导出 RTMDet-Tiny 未解码的多尺度分类和框回归张量。"""

import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS


class RawDetectionModel(torch.nn.Module):
    """仅运行 backbone、neck 和检测 head。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        """返回每个尺度的分类 logits、框回归 logits 和 objectness。"""
        features = self.model.extract_feat(images)
        head = self.model.bbox_head
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
        return tuple(outputs)


def main():
    """依据 HumanArt RTMDet-Tiny 配置导出固定 640x640 ONNX。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='RTMDet/custom_rtmdet_tiny_humanart_raw_opset9.onnx')
    args = parser.parse_args()
    register_all_modules(init_default_scope=True)
    if args.config:
        cfg = Config.fromfile(args.config)
    else:
        import mmdet
        config_path = Path(mmdet.__file__).parent / '.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
        cfg = Config.fromfile(str(config_path))
        cfg.model.bbox_head.num_classes = 80
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    wrapper = RawDetectionModel(model).eval()
    sample = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = wrapper(sample)
    output_names = [f'raw_output_{index}' for index in range(len(outputs))]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, sample, str(output_path), opset_version=9,
        input_names=['images'], output_names=output_names,
        do_constant_folding=True, dynamo=False)
    print('输出张量: ' + ', '.join(
        f'{name}={tuple(value.shape)}' for name, value in zip(output_names, outputs)))
    print(f'导出完成: {output_path.resolve()}')


if __name__ == '__main__':
    main()
