import unittest
from pathlib import Path

import torch
from mmengine.config import Config

from mmpose.models.heads.hybrid_heads.rtmo_head import RTMOHeadModule


class TestRTMOHeadModule(unittest.TestCase):

    def test_private_config_selects_ovm_compatible_structure(self):
        repo_root = Path(__file__).resolve().parents[4]
        config_path = repo_root / (
            'configs/body_2d_keypoint/rtmo/custom/'
            'custom_rtmo-s_8xb32-600e_private-640x640.py')
        config = Config.fromfile(str(config_path))

        self.assertEqual(config.model.neck.use_encoder_idx, [])
        self.assertEqual(config.model.neck.num_encoder_layers, 0)
        self.assertEqual(config.model.head.head_module_cfg.num_groups, 1)
        self.assertEqual(
            config.model.head.head_module_cfg.channels_per_group, 288)

    def test_pose_branch_group_one_keeps_raw_head_shapes(self):
        module = RTMOHeadModule(
            num_keypoints=17,
            in_channels=256,
            num_classes=1,
            widen_factor=0.5,
            cls_feat_channels=256,
            num_groups=1,
            channels_per_group=288,
            pose_vec_channels=256,
            featmap_strides=(16, 32),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        module.eval()

        for level in module.conv_pose:
            for layer in level:
                self.assertEqual(layer.conv.groups, 1)
                self.assertEqual(layer.conv.out_channels, 144)

        with torch.no_grad():
            cls, bbox, offsets, visibility, pose = module((
                torch.randn(1, 256, 40, 40),
                torch.randn(1, 256, 20, 20),
            ))

        self.assertEqual([tensor.shape for tensor in cls], [
            torch.Size((1, 1, 40, 40)),
            torch.Size((1, 1, 20, 20)),
        ])
        self.assertEqual([tensor.shape for tensor in bbox], [
            torch.Size((1, 4, 40, 40)),
            torch.Size((1, 4, 20, 20)),
        ])
        self.assertEqual(offsets, [])
        self.assertEqual([tensor.shape for tensor in visibility], [
            torch.Size((1, 17, 40, 40)),
            torch.Size((1, 17, 20, 20)),
        ])
        self.assertEqual([tensor.shape for tensor in pose], [
            torch.Size((1, 256, 40, 40)),
            torch.Size((1, 256, 20, 20)),
        ])


if __name__ == '__main__':
    unittest.main()
