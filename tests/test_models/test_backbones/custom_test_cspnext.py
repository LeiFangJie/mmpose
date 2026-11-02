import unittest

import torch
from torch.nn import MaxPool2d, ReLU

from mmpose.registry import MODELS
import mmpose.models.backbones.custom_cspnext


class TestCustomCSPNeXt(unittest.TestCase):
    """验证私有导出主干的结构与输入输出契约。"""

    def test_private_backbone_uses_relu_and_convolutional_pyramid(self):
        backbone = MODELS.build(dict(
            type='CustomCSPNeXt',
            arch='P5',
            deepen_factor=0.167,
            widen_factor=0.375,
            out_indices=(4,),
            channel_attention=False,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN')))
        self.assertFalse(any(isinstance(layer, MaxPool2d)
                             for layer in backbone.modules()))
        self.assertTrue(all(isinstance(layer, ReLU)
                            for layer in backbone.modules()
                            if layer.__class__.__name__ == 'ReLU'))
        self.assertEqual(backbone.stage4[1].conv1.conv.in_channels, 384)
        output = backbone(torch.randn(1, 3, 256, 192))[-1]
        self.assertEqual(tuple(output.shape), (1, 384, 8, 6))


if __name__ == '__main__':
    unittest.main()
