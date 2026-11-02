"""验证 NumPy/OpenCV 前后处理与 OpenMMLab 原生实现的一致性。"""

import os
import tempfile
import unittest

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmpose.registry import MODELS, TRANSFORMS
from mmpose.utils import register_all_modules
from mmengine.registry import DefaultScope

from RTMDet.custom_rtmdet_postprocess import decode_detector_outputs
from RTMDet.custom_rtmdet_preprocess import preprocess_detector_image
from RTMPose.custom_rtmpose_postprocess import (
    decode_simcc, load_postprocess_weights, simcc_from_feature)
from RTMPose.custom_rtmpose_preprocess import preprocess_pose_image
from tools.deploy.rtmpose.custom_export_raw_feature_onnx import export_postprocess_weights


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class TestDeploymentPrePost(unittest.TestCase):
    """覆盖 affine、归一化、RTMCC continuation 和检测解码。"""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261102)
        register_all_modules(init_default_scope=True)
        import mmpose.models.backbones.custom_cspnext
        cls.pose_config = Config.fromfile(
            'configs/body_2d_keypoint/rtmpose/custom/'
            'custom_rtmpose-t_8xb32-420e_private-256x192.py')
        cls.pose_head = MODELS.build(cls.pose_config.model.head).eval()
        cls.det_config = Config.fromfile(
            'C:/Users/admin/miniforge3/envs/rtmpose/Lib/site-packages/'
            'mmdet/.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py')
        cls.det_model = None
        import mmdet.datasets.transforms
        import mmdet.models

    def test_pose_preprocess_matches_framework_affine(self):
        """验证部署 crop 和 MMPose TopdownAffine 的矩阵及像素一致。"""
        image = np.random.default_rng(9).integers(
            0, 256, (301, 517, 3), dtype=np.uint8)
        bbox = np.asarray([43., 19., 470., 280.], dtype=np.float32)
        actual, center, scale = preprocess_pose_image(image, bbox)
        DefaultScope.get_instance('mmpose-test', scope_name='mmpose')
        import mmpose.datasets.transforms
        framework = TRANSFORMS.build(dict(
            type='TopdownAffine', input_size=(192, 256), use_udp=False))
        framework_scale = (bbox[2:] - bbox[:2]) * 1.25
        framework_center = (bbox[:2] + bbox[2:]) * 0.5
        ratio = 192 / 256
        if framework_scale[0] > framework_scale[1] * ratio:
            framework_scale[1] = framework_scale[0] / ratio
        else:
            framework_scale[0] = framework_scale[1] * ratio
        framework_results = framework(dict(
            img=image.copy(), bbox_center=framework_center[None],
            bbox_scale=framework_scale[None],
            bbox_rotation=np.asarray([0.], dtype=np.float32)))
        framework_crop = framework_results['img']
        expected_rgb = framework_crop[..., ::-1].astype(np.float32)
        expected = ((expected_rgb - np.asarray([123.675, 116.28, 103.53])) /
                    np.asarray([58.395, 57.12, 57.375]))
        np.testing.assert_allclose(actual[0].transpose(1, 2, 0), expected,
                                   atol=1e-6)

    def test_detector_preprocess_matches_framework_transforms(self):
        """验证 Resize/Pad 与 MMDetection 原生 transforms 的像素一致。"""
        from mmdet.datasets.transforms import Pad, Resize
        from mmcv.transforms import Compose

        image = np.random.default_rng(13).integers(
            0, 256, (377, 853, 3), dtype=np.uint8)
        actual, metadata = preprocess_detector_image(image)
        transforms = Compose([
            Resize(scale=(640, 640), keep_ratio=True),
            Pad(size=(640, 640), pad_val=dict(img=(114, 114, 114)))])
        expected = transforms(dict(img=image.copy(), img_shape=image.shape[:2]))
        resized = expected['img'].astype(np.float32)
        normalized = ((resized - np.asarray([103.53, 116.28, 123.675])) /
                      np.asarray([57.375, 57.12, 58.395]))
        np.testing.assert_allclose(actual[0].transpose(1, 2, 0), normalized,
                                   atol=1e-6)
        self.assertEqual(metadata['img_shape'], expected['img_shape'])

    def test_numpy_rtmcc_matches_framework_head_and_decoder(self):
        """验证 NumPy 续算 head 与原生 RTMCCHead/SimCC 输出一致。"""
        # 初始化值为 1 时无法覆盖 residual scale 的广播和分支顺序。
        with torch.no_grad():
            residual_scale = torch.linspace(
                0.25, 1.75, self.pose_head.gau.res_scale.scale.numel())
            self.pose_head.gau.res_scale.scale.copy_(residual_scale)
        final_input = torch.randn(1, 384, 8, 6)
        feature = self.pose_head.final_layer(final_input)
        with tempfile.TemporaryDirectory() as directory:
            weights_path = os.path.join(directory, 'weights.npz')
            export_postprocess_weights(self.pose_head, weights_path)
            weights = load_postprocess_weights(weights_path)
            with torch.no_grad():
                expected_x, expected_y = self.pose_head((final_input,))
        actual_x, actual_y = simcc_from_feature(feature.detach().numpy(), weights)
        np.testing.assert_allclose(actual_x, expected_x.numpy(), rtol=2e-4,
                                   atol=2e-4)
        np.testing.assert_allclose(actual_y, expected_y.numpy(), rtol=2e-4,
                                   atol=2e-4)
        centers = np.asarray([[200., 150.]], dtype=np.float32)
        scales = np.asarray([[300., 400.]], dtype=np.float32)
        with torch.no_grad():
            framework_keypoints, framework_scores = self.pose_head.decoder.decode(
                expected_x[0].numpy(), expected_y[0].numpy())
        expected_keypoints = (framework_keypoints / np.asarray([192., 256.]) *
                              scales[:, None, :] + centers[:, None, :] -
                              0.5 * scales[:, None, :])
        actual_keypoints, actual_scores = decode_simcc(
            actual_x, actual_y, centers[0], scales[0])
        np.testing.assert_allclose(actual_keypoints, expected_keypoints,
                                   atol=1e-5)
        np.testing.assert_allclose(actual_scores, framework_scores,
                                   atol=1e-5)

    def test_numpy_detector_decode_matches_mmdet_head(self):
        """验证类别分数、distance box、裁剪与 NMS 跟 MMDetection 一致。"""
        DefaultScope.get_instance('mmdet-test', scope_name='mmdet')
        from mmdet.registry import MODELS as DET_MODELS
        self.det_model = DET_MODELS.build(self.det_config.model).eval()
        head = self.det_model.bbox_head
        shapes = [(80, 80), (40, 40), (20, 20)]
        raw_outputs = []
        framework_scores = []
        framework_boxes = []
        for level, (height, width) in enumerate(shapes):
            cls_logits = np.full((1, 80, height, width), -12., np.float32)
            bbox_logits = np.full((1, 4, height, width), 1.25, np.float32)
            cls_logits[0, 0, 10 + level, 12 + level] = 3.0 - level
            raw_outputs.extend([cls_logits, bbox_logits])
            framework_scores.append(torch.from_numpy(cls_logits.copy()))
            framework_boxes.append(torch.from_numpy(
                bbox_logits.copy() * (8. * (2 ** level))))
        metadata = dict(img_shape=(640, 640), ori_shape=(640, 640),
                        scale_factor=np.asarray([1., 1., 1., 1.]))
        with torch.no_grad():
            expected = head.predict_by_feat(
                framework_scores, framework_boxes,
                batch_img_metas=[metadata], rescale=False)[0]
        actual_boxes, actual_scores, actual_labels = decode_detector_outputs(
            raw_outputs, dict(img_shape=(640, 640),
                              scale_factor=np.asarray([1., 1.])),
            score_threshold=0.001, iou_threshold=0.65, nms_pre=30000,
            max_per_image=300)
        np.testing.assert_allclose(actual_boxes, expected.bboxes.numpy(),
                                   atol=1e-5)
        np.testing.assert_allclose(actual_scores, expected.scores.numpy(),
                                   atol=1e-6)
        np.testing.assert_array_equal(actual_labels, expected.labels.numpy())


if __name__ == '__main__':
    unittest.main()
