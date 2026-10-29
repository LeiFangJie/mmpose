import unittest

import numpy as np

from RTMO.custom_rtmo_preprocess_postprocess import (
    RTMOPostProcessor, RTMOPreProcessor, build_grid, decode_bbox, sigmoid)


class TestRTMOPreprocessPostprocess(unittest.TestCase):

    def test_preprocess_matches_fit_letterbox_contract(self):
        image = np.zeros((320, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 20
        image[:, :, 1] = 80
        image[:, :, 2] = 200

        tensor, metadata = RTMOPreProcessor().process(image)

        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(float(tensor.max()), 200.0)
        self.assertEqual(float(tensor.min()), 20.0)
        self.assertEqual(metadata['input_size'], (640, 640))
        self.assertTrue(np.allclose(metadata['input_center'], [320, 160]))
        self.assertTrue(np.allclose(metadata['input_scale'], [640, 640]))

    def test_decode_bbox_matches_rtmo_center_log_wh_formula(self):
        logits = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        priors = np.array([[8.0, 16.0]], dtype=np.float32)
        strides = np.array([16.0], dtype=np.float32)

        boxes = decode_bbox(logits, priors, strides)

        np.testing.assert_allclose(boxes, [[[0.0, 8.0, 16.0, 24.0]]])

    def test_sigmoid_is_stable_for_extreme_logits(self):
        result = sigmoid(np.array([-1000.0, 0.0, 1000.0], dtype=np.float32))

        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-7)

    def test_grid_uses_centralized_prior_offset(self):
        grid = build_grid(2, 2, 16)

        np.testing.assert_allclose(grid, [
            [7.5, 7.5], [23.5, 7.5], [7.5, 23.5], [23.5, 23.5]])

    def test_postprocessor_preserves_empty_output_shapes(self):
        processor = RTMOPostProcessor(parameters={})
        raw_outputs = {
            'cls_s16': np.full((1, 1, 40, 40), -100.0, dtype=np.float32),
            'bbox_s16': np.zeros((1, 4, 40, 40), dtype=np.float32),
            'vis_s16': np.zeros((1, 17, 40, 40), dtype=np.float32),
            'pose_s16': np.zeros((1, 256, 40, 40), dtype=np.float32),
            'cls_s32': np.full((1, 1, 20, 20), -100.0, dtype=np.float32),
            'bbox_s32': np.zeros((1, 4, 20, 20), dtype=np.float32),
            'vis_s32': np.zeros((1, 17, 20, 20), dtype=np.float32),
            'pose_s32': np.zeros((1, 256, 20, 20), dtype=np.float32),
        }
        metadata = {
            'input_size': (640, 640),
            'input_center': np.array([320.0, 320.0]),
            'input_scale': np.array([640.0, 640.0]),
            'original_size': (640, 640),
        }

        result = processor.process(raw_outputs, metadata)

        self.assertEqual(result['bboxes'].shape, (0, 4))
        self.assertEqual(result['keypoints'].shape, (0, 17, 2))
        self.assertEqual(result['keypoint_scores'].shape, (0, 17))


if __name__ == '__main__':
    unittest.main()
