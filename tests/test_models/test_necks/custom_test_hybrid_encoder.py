import unittest

import torch

from mmpose.models.necks.custom_hybrid_encoder import CustomHybridEncoder


class TestHybridEncoder(unittest.TestCase):

    def test_empty_encoder_indices_skip_attention(self):
        encoder = CustomHybridEncoder(
            in_channels=[8, 16, 32],
            hidden_dim=16,
            use_encoder_idx=[],
            num_encoder_layers=0,
            output_indices=[1, 2],
            deepen_factor=0.33,
            widen_factor=0.5,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        encoder.eval()

        with torch.no_grad():
            outputs = encoder((
                torch.randn(1, 8, 16, 16),
                torch.randn(1, 16, 8, 8),
                torch.randn(1, 32, 4, 4),
            ))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, torch.Size((1, 16, 8, 8)))
        self.assertEqual(outputs[1].shape, torch.Size((1, 16, 4, 4)))
        self.assertFalse(hasattr(encoder, 'encoder'))

        encoder.switch_to_deploy(test_cfg=dict(input_size=(128, 128)))


if __name__ == '__main__':
    unittest.main()
