"""适配受限部署算子的 RTMPose 私有 CSPNeXt。"""

from torch import cat
from torch.nn import ModuleList
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpose.registry import MODELS
from .cspnext import CSPNeXt


class CustomSPPConvPyramid(BaseModule):
    """使用连续 depthwise 卷积构造空间金字塔特征。"""

    def __init__(self, in_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.conv1 = ConvModule(in_channels, in_channels // 2, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.pyramid_convs = ModuleList([
            ConvModule(in_channels // 2, in_channels // 2, 5, padding=2,
                       groups=in_channels // 2, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(3)
        ])
        self.conv2 = ConvModule(
            (in_channels // 2) * 4, in_channels, 1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, inputs):
        """生成输入及三层卷积感受野特征并融合。"""
        base = self.conv1(inputs)
        features = [base]
        for conv in self.pyramid_convs:
            features.append(conv(features[-1]))
        return self.conv2(cat(features, dim=1))


@MODELS.register_module()
class CustomCSPNeXt(CSPNeXt):
    """关闭通道注意力并替换 SPP 池化的 CSPNeXt。"""

    def __init__(self, channel_attention=False, act_cfg=dict(type='ReLU'),
                 **kwargs):
        kwargs.pop('channel_attention', None)
        kwargs.pop('act_cfg', None)
        super().__init__(channel_attention=False, act_cfg=act_cfg, **kwargs)
        stage = getattr(self, self.layers[-1])
        spp = stage[1]
        norm_cfg = kwargs.get('norm_cfg', dict(type='BN'))
        conv_cfg = kwargs.get('conv_cfg')
        stage[1] = CustomSPPConvPyramid(
            spp.conv1.conv.in_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
