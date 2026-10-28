import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch
import torch.nn as nn

if importlib.util.find_spec('onnxruntime') is not None:
    import onnxruntime as ort
else:
    ort = None

from mmpose.models.backbones.custom_csp_darknet import Focus
from tools.deploy.rtmo.custom_export_raw_onnx import (
    RTMORawExportWrapper, SpaceToDepth)
from tools.deploy.rtmo.custom_onnx_audit import (
    classify_operators, remove_identity_nodes)


class TinyBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = Focus(
            3,
            8,
            kernel_size=3,
            stride=1,
            norm_cfg=None,
            act_cfg=dict(type='ReLU'))
        self.stage1 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.layers = ['stem', 'stage1']
        self.out_indices = (0, 1)


class TinyHeadModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_cls = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.conv_pose = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.out_cls = nn.ModuleList([nn.Conv2d(4, 1, 1) for _ in range(2)])
        self.out_bbox = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(2)])
        self.out_kpt_vis = nn.ModuleList(
            [nn.Conv2d(4, 17, 1) for _ in range(2)])
        self.out_pose = nn.ModuleList(
            [nn.Conv2d(4, 256, 1) for _ in range(2)])


class TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.neck = nn.Identity()
        self.head = SimpleNamespace(head_module=TinyHeadModule())


class TestRTMORawExport(unittest.TestCase):

    def setUp(self):
        self.wrapper = RTMORawExportWrapper(
            TinyModel(), input_size=(32, 32)).eval()

    def test_wrapper_returns_eight_four_dimensional_raw_heads(self):
        outputs = self.wrapper(torch.randn(1, 3, 32, 32))

        self.assertEqual(len(outputs), 8)
        self.assertEqual([tuple(output.shape) for output in outputs], [
            (1, 1, 16, 16),
            (1, 4, 16, 16),
            (1, 17, 16, 16),
            (1, 256, 16, 16),
            (1, 1, 8, 8),
            (1, 4, 8, 8),
            (1, 17, 8, 8),
            (1, 256, 8, 8),
        ])

    def test_wrapper_rejects_non_static_input_shape(self):
        with self.assertRaisesRegex(ValueError, r'\[1, 3, 32, 32\]'):
            self.wrapper(torch.randn(2, 3, 32, 32))

    def test_space_to_depth_matches_focus_and_exports_as_opset9_node(self):
        inputs = torch.randn(1, 3, 8, 8)
        expected = self.wrapper.backbone.stem(inputs)
        actual = self.wrapper._run_focus(inputs)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

        class SpaceToDepthOnly(nn.Module):

            def forward(self, tensor):
                return SpaceToDepth.apply(tensor, 2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            onnx_path = Path(temporary_directory) / 'space_to_depth.onnx'
            torch.onnx.export(
                SpaceToDepthOnly(),
                inputs,
                str(onnx_path),
                opset_version=9,
                dynamo=False)
            graph = onnx.load(str(onnx_path))
        self.assertEqual(graph.opset_import[0].version, 9)
        self.assertIn('SpaceToDepth', [node.op_type for node in graph.graph.node])

    @unittest.skipIf(ort is None, 'ONNX Runtime 未安装，跳过运行时数值对照')
    def test_onnxruntime_focus_matches_original_focus(self):
        inputs = torch.randn(1, 3, 8, 8)
        expected = self.wrapper.backbone.stem(inputs)

        class FocusOnly(nn.Module):

            def __init__(self, wrapper):
                super().__init__()
                self.wrapper = wrapper

            def forward(self, tensor):
                return self.wrapper._run_focus(tensor)

        with tempfile.TemporaryDirectory() as temporary_directory:
            focus_path = Path(temporary_directory) / 'focus.onnx'
            torch.onnx.export(
                FocusOnly(self.wrapper),
                inputs,
                str(focus_path),
                opset_version=9,
                dynamo=False)
            session = ort.InferenceSession(
                str(focus_path), providers=['CPUExecutionProvider'])
            onnx_output = session.run(
                None, {session.get_inputs()[0].name: inputs.numpy()})[0]

        self.assertTrue(torch.allclose(
            expected, torch.from_numpy(onnx_output), atol=1e-6, rtol=1e-6))

    def test_operator_audit_flags_ops_outside_the_manual(self):
        report = classify_operators({'Conv': 3, 'SpaceToDepth': 1,
                                     'Mul': 1, 'MatMul': 1})

        self.assertEqual(report['supported_ops'], ['Conv', 'SpaceToDepth'])
        self.assertEqual(report['unlisted_ops'], ['MatMul'])
        self.assertIn('Mul', report['version_limited_ops'])

    def test_identity_nodes_are_removed_without_changing_consumers(self):
        tensor_info = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 1, 1, 1])
        output_info = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 1, 1, 1])
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node('Identity', ['input'], ['alias']),
                onnx.helper.make_node('Relu', ['alias'], ['output']),
            ],
            'identity_graph', [tensor_info], [output_info])
        model = onnx.helper.make_model(graph, opset_imports=[
            onnx.helper.make_opsetid('', 9)])

        removed = remove_identity_nodes(model)

        self.assertEqual(removed, 1)
        self.assertEqual([node.op_type for node in model.graph.node], ['Relu'])
        self.assertEqual(model.graph.node[0].input[0], 'input')


if __name__ == '__main__':
    unittest.main()
