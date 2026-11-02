"""导出 RTMPose 的 7x7 final_layer 特征图 ONNX。"""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
from onnx.external_data_helper import convert_model_from_external_data
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.registry import MODELS
from mmpose.utils import register_all_modules


class RawFeatureModel(torch.nn.Module):
    """截断于 RTMCCHead final_layer 的推理包装器。"""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.final_layer = model.head.final_layer

    def forward(self, images):
        """返回 [B, 17, 8, 6] 的 7x7 卷积输出。"""
        return self.final_layer(self.backbone(images)[-1])


def remove_identity_nodes(path):
    """折叠导出器为共享权重生成的 Identity 占位节点。"""
    model = onnx.load(str(path))
    replacements = {}
    kept_nodes = []
    for node in model.graph.node:
        if node.op_type == 'Identity' and len(node.input) == 1 and len(node.output) == 1:
            replacements[node.output[0]] = node.input[0]
        else:
            kept_nodes.append(node)
    for node in kept_nodes:
        for index, value in enumerate(node.input):
            node.input[index] = replacements.get(value, value)
    for output in model.graph.output:
        output_name = output.name
        output.name = replacements.get(output_name, output_name)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def export_postprocess_weights(head, path):
    """导出板外 NumPy 续算 RTMCC/SimCC 所需参数。"""
    gau = head.gau
    weights = {
        'mlp_norm_g': head.mlp[0].g,
        'mlp_weight': head.mlp[1].weight,
        'gau_ln_g': gau.ln.g,
        'gau_uv': gau.uv.weight,
        'gau_gamma': gau.gamma,
        'gau_beta': gau.beta,
        'gau_o': gau.o.weight,
        'gau_res_scale': gau.res_scale.scale,
        'gau_s': np.asarray(gau.sqrt_s, dtype=np.float32),
        'cls_x': head.cls_x.weight,
        'cls_y': head.cls_y.weight,
    }
    if gau.use_rel_bias:
        weights['gau_w'] = gau.w
    np.savez_compressed(
        path, **{name: value.detach().cpu().numpy()
                 if isinstance(value, torch.Tensor) else value
                 for name, value in weights.items()})


def embed_external_data(path):
    """将导出器生成的 external data 转回单文件 ONNX。"""
    model = onnx.load(str(path), load_external_data=True)
    convert_model_from_external_data(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(path), save_as_external_data=False)
    sidecar = Path(str(path) + '.data')
    if sidecar.exists():
        sidecar.unlink()


def main():
    """加载完整姿态模型并生成固定 batch=1、opset=9 的 ONNX。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='RTMPose/custom_rtmpose_t_raw_feature_opset9.onnx')
    parser.add_argument('--weights-output', default='RTMPose/custom_rtmpose_postprocess_weights.npz')
    args = parser.parse_args()
    register_all_modules(init_default_scope=True)
    import mmpose.models.backbones.custom_cspnext
    torch.manual_seed(0)
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    wrapper = RawFeatureModel(model).eval()
    export_postprocess_weights(model.head, args.weights_output)
    sample = torch.randn(1, 3, 256, 192)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, sample, str(output_path), opset_version=9,
        input_names=['images'], output_names=['head_feature'],
        do_constant_folding=True, dynamo=False, external_data=False)
    embed_external_data(output_path)
    remove_identity_nodes(output_path)
    print(f'导出完成: {output_path.resolve()}')


if __name__ == '__main__':
    main()
