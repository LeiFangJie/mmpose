"""固定 RTMO raw ONNX 图的 shape、opset 和 CDK 算子静态审计。"""

from collections import Counter
from pathlib import Path

import onnx


# 第 5.2 节 ONNX 列中标为 v0.5 的算子；版本受限项单独列出。
SUPPORTED_OPS = {
    'Gemm', 'Relu', 'PRelu', 'Sigmoid', 'Add', 'Flatten', 'Concat',
    'BatchNormalization', 'Softmax', 'Upsample', 'LpNormalization',
    'Reshape', 'LeakyRelu', 'Clip', 'SpaceToDepth', 'Conv', 'MaxPool',
    'AveragePool', 'Transpose', 'Slice', 'GlobalMaxPool', 'GlobalAveragePool'
}
VERSION_LIMITED_OPS = {
    'ConvTranspose': '手册标注 v0.9-RC',
    'Mul': '手册标注 v0.9-RC',
    'Round': '手册标注 opset 11，当前导出为 opset 9',
    'Pad': '手册标注 v0.9-RC'
}
STRUCTURAL_OPS = {'Constant', 'Identity', 'Shape'}
FORBIDDEN_POSTPROCESS_OPS = {
    'NonMaxSuppression', 'TopK', 'Gather', 'GatherElements', 'Erf', 'Sin',
    'Cos', 'MatMul', 'Div', 'Exp', 'Sqrt'
}
OUTPUT_SHAPES = [
    [1, 1, 40, 40], [1, 4, 40, 40], [1, 17, 40, 40], [1, 256, 40, 40],
    [1, 1, 20, 20], [1, 4, 20, 20], [1, 17, 20, 20], [1, 256, 20, 20]
]
OUTPUT_NAMES = [
    'cls_s16', 'bbox_s16', 'vis_s16', 'pose_s16',
    'cls_s32', 'bbox_s32', 'vis_s32', 'pose_s32'
]


def remove_identity_nodes(model):
    """删除 ONNX 图中的无副作用 Identity 直通节点。

    参数:
        model: 已加载的 ONNX ModelProto。

    返回:
        实际删除的 Identity 节点数量。
    """
    graph = model.graph
    removed_count = 0
    kept_nodes = []
    replacements = {}
    for node in graph.node:
        if (node.op_type == 'Identity' and len(node.input) == 1 and
                len(node.output) == 1):
            replacements[node.output[0]] = node.input[0]
            removed_count += 1
        else:
            kept_nodes.append(node)

    def resolve(value_name):
        """解析可能经过多层 Identity 的输入别名。"""
        visited = set()
        while value_name in replacements and value_name not in visited:
            visited.add(value_name)
            value_name = replacements[value_name]
        return value_name

    for node in kept_nodes:
        for index, value_name in enumerate(node.input):
            node.input[index] = resolve(value_name)
    for output in graph.output:
        output.name = resolve(output.name)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    kept_value_info = [
        value for value in graph.value_info
        if value.name not in replacements
    ]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)
    return removed_count


def _tensor_shape(value_info):
    """读取 ONNX value info 中的静态 shape。"""
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField('dim_value'):
            return None
        dims.append(dim.dim_value)
    return dims


def _tensor_dtype(value_info):
    """读取 ONNX tensor 的元素类型。"""
    return value_info.type.tensor_type.elem_type


def classify_operators(operation_counts):
    """按手册支持列表分类图算子。"""
    operation_types = set(operation_counts)
    limited = sorted(operation_types & set(VERSION_LIMITED_OPS))
    return {
        'supported_ops': sorted(operation_types & SUPPORTED_OPS),
        'structural_ops': sorted(operation_types & STRUCTURAL_OPS),
        'version_limited_ops': {
            op: VERSION_LIMITED_OPS[op] for op in limited
        },
        'unlisted_ops': sorted(
            operation_types - SUPPORTED_OPS - STRUCTURAL_OPS -
            set(VERSION_LIMITED_OPS))
    }


def audit_onnx(path):
    """审计固定输入输出契约、ONNX 算子及卷积分组。

    参数:
        path: 待审计 ONNX 文件路径。

    返回:
        包含 shape、op 计数、问题节点和静态结论的字典。
    """
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    opsets = {entry.domain or 'ai.onnx': entry.version
              for entry in inferred.opset_import}
    default_opset = opsets.get('ai.onnx')
    values = {value.name: value for value in
              list(inferred.graph.input) + list(inferred.graph.value_info) +
              list(inferred.graph.output)}
    input_shapes = {value.name: _tensor_shape(value)
                    for value in inferred.graph.input}
    output_shapes = {value.name: _tensor_shape(value)
                     for value in inferred.graph.output}
    operation_counts = Counter(node.op_type for node in inferred.graph.node)
    operator_report = classify_operators(operation_counts)
    unlisted_ops = operator_report['unlisted_ops']
    version_limited = list(operator_report['version_limited_ops'])
    postprocess_ops = sorted(set(operation_counts) & FORBIDDEN_POSTPROCESS_OPS)
    conv_issues = []
    pooling_issues = []
    custom_domain_nodes = []
    upsample_issues = []
    large_kernel_warnings = []
    initializers = {tensor.name: tensor for tensor in inferred.graph.initializer}

    for index, node in enumerate(inferred.graph.node):
        if node.domain not in ('', 'ai.onnx'):
            custom_domain_nodes.append({
                'node': node.name or str(index),
                'domain': node.domain,
                'op_type': node.op_type
            })
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute)
                      for attribute in node.attribute}
        if node.op_type == 'MaxPool':
            kernel_shape = attributes.get('kernel_shape', [])
            pads = attributes.get('pads', [0, 0, 0, 0])
            if (len(kernel_shape) != 2 or max(kernel_shape) > 8 or
                    kernel_shape[0] != kernel_shape[1] or max(pads) > 2 or
                    pads[:2] != pads[2:]):
                pooling_issues.append({
                    'node': node.name or str(index),
                    'kernel_shape': list(kernel_shape),
                    'pads': list(pads)
                })
        if node.op_type == 'Upsample' and attributes.get('mode') != b'nearest':
            upsample_issues.append({
                'node': node.name or str(index),
                'mode': attributes.get('mode')
            })
        if node.op_type != 'Conv':
            continue
        groups = int(attributes.get('group', 1))
        weights = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if weights is None:
            conv_issues.append({'node': node.name or index,
                                'issue': '卷积权重 shape 无法静态审计'})
            continue
        weight_shape = list(weights.dims)
        output_channels = weight_shape[0]
        input_channels = weight_shape[1] * groups
        kernel_shape = attributes.get('kernel_shape', weight_shape[2:])
        if max(kernel_shape) > 5:
            large_kernel_warnings.append({
                'node': node.name or str(index),
                'kernel_shape': list(kernel_shape)
            })
        if groups != 1 and not (
                groups == input_channels == output_channels and
                weight_shape[1] == 1):
            conv_issues.append({
                'node': node.name or str(index),
                'group': groups,
                'input_channels': input_channels,
                'output_channels': output_channels})

    violations = []
    if default_opset != 9:
        violations.append(f'ONNX opset 必须为 9，实际为 {default_opset}')
    if input_shapes != {'images': [1, 3, 640, 640]}:
        violations.append(f'输入 shape 不符合固定契约: {input_shapes}')
    input_dtypes = {value.name: _tensor_dtype(value)
                    for value in inferred.graph.input}
    if input_dtypes != {'images': onnx.TensorProto.FLOAT}:
        violations.append(f'输入必须是 float32: {input_dtypes}')
    if list(output_shapes) != OUTPUT_NAMES:
        violations.append(f'输出名称或顺序不符合契约: {list(output_shapes)}')
    if [output_shapes.get(name) for name in OUTPUT_NAMES] != OUTPUT_SHAPES:
        violations.append(f'输出 shape 不符合固定契约: {output_shapes}')
    output_dtypes = {value.name: _tensor_dtype(value)
                     for value in inferred.graph.output}
    if any(dtype != onnx.TensorProto.FLOAT for dtype in output_dtypes.values()):
        violations.append(f'输出必须是 float32: {output_dtypes}')
    if unlisted_ops:
        violations.append(f'包含手册第 5.2 节未列出的算子: {unlisted_ops}')
    if version_limited:
        violations.append('包含手册标注版本受限的算子')
    if postprocess_ops:
        violations.append(f'包含应在图外执行的解码/后处理算子: {postprocess_ops}')
    if conv_issues:
        violations.append('存在不满足 Conv 分组限制或无法审计的节点')
    if pooling_issues:
        violations.append('存在不符合手册 MaxPool 参数限制的节点')
    if custom_domain_nodes:
        violations.append('包含非标准 ONNX domain 节点')
    if upsample_issues:
        violations.append('存在非 nearest 模式的 Upsample 节点')
    if 'Identity' in operation_counts:
        violations.append('包含未消除的 Identity 占位节点')

    return {
        'path': str(Path(path)),
        'opset': default_opset,
        'input_shapes': input_shapes,
        'input_dtypes': input_dtypes,
        'output_shapes': output_shapes,
        'output_dtypes': output_dtypes,
        'operator_counts': dict(sorted(operation_counts.items())),
        'supported_ops': operator_report['supported_ops'],
        'structural_ops': operator_report['structural_ops'],
        'version_limited_ops': operator_report['version_limited_ops'],
        'unlisted_ops': unlisted_ops,
        'forbidden_postprocess_ops': postprocess_ops,
        'conv_group_issues': conv_issues,
        'maxpool_issues': pooling_issues,
        'custom_domain_nodes': custom_domain_nodes,
        'upsample_issues': upsample_issues,
        'conv_large_kernel_warnings': large_kernel_warnings,
        'violations': violations,
        'static_onnx_contract_passed': not violations
    }


def main():
    """输出 JSON 格式的 ONNX 审计报告，并在存在问题时返回失败码。"""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('onnx_path', help='待审计 ONNX 文件')
    args = parser.parse_args()
    report = audit_onnx(args.onnx_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['static_onnx_contract_passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
