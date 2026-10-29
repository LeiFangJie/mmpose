# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Dict, Tuple

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner.checkpoint import CheckpointLoader

from mmpose.registry import HOOKS


@HOOKS.register_module()
class CustomCompatiblePretrainedHook(Hook):
    """按参数名和形状加载预训练模型中兼容的权重。"""

    def __init__(self,
                 checkpoint: str,
                 prefixes: Tuple[str, ...] = ('backbone.', 'neck.', 'head.')):
        self.checkpoint = checkpoint
        self.prefixes = prefixes

    def before_train(self, runner) -> None:
        """在模型及优化器构建后加载兼容权重，并跳过续训覆盖。"""
        if getattr(runner, '_resume', False) or getattr(
                runner, '_load_from', None):
            runner.logger.info('检测到续训或显式 load_from，跳过预训练权重加载。')
            return

        model = runner.model.module if is_model_wrapper(runner.model) \
            else runner.model
        checkpoint = CheckpointLoader.load_checkpoint(
            self.checkpoint, map_location='cpu', logger=runner.logger)
        source_state = checkpoint.get('state_dict', checkpoint)
        compatible_state, report = self._filter_compatible_weights(
            model.state_dict(), source_state)
        model.load_state_dict(compatible_state, strict=False)

        summary = ', '.join(
            f'{name}: {counts[0]} 已加载/{counts[1]} 形状不匹配/'
            f'{counts[2]} 当前模型无对应参数'
            for name, counts in report.items())
        runner.logger.info(
            f'从 {self.checkpoint} 加载兼容预训练权重（strict=False）：{summary}')

    def _filter_compatible_weights(
            self, model_state: Dict[str, torch.Tensor],
            source_state: Dict[str, torch.Tensor]):
        """筛出名称、形状一致的权重，避免尺寸不符时加载失败。"""
        compatible_state = {}
        report = defaultdict(lambda: [0, 0, 0])
        for key, value in source_state.items():
            group = next((prefix[:-1] for prefix in self.prefixes
                          if key.startswith(prefix)), None)
            if group is None:
                continue
            if key not in model_state:
                report[group][2] += 1
            elif not torch.is_tensor(value) or \
                    value.shape != model_state[key].shape:
                report[group][1] += 1
            else:
                compatible_state[key] = value
                report[group][0] += 1
        return compatible_state, dict(report)
