# MMPose 修改指南 - 每一层能改什么

## 🎯 核心原则

```
配置层 (Config)    → 修改实验参数、组合方式
实现层 (mmpose/)   → 修改算法逻辑、添加新组件
执行层 (tools/)    → 修改训练流程、添加功能
```

---

## 第 1 层：Config 配置层修改

### ✅ 能修改什么

**1. 模型结构组合**
```python
# 原始配置
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(type='HRNet', ...),
    head=dict(type='HeatmapHead', ...)
)

# 修改1：换骨干网络
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(type='ResNet', depth=50),  # HRNet → ResNet
    head=dict(type='HeatmapHead', ...)
)

# 修改2：添加 Neck
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(type='HRNet', ...),
    neck=dict(type='FeatureFusionNeck'),  # 新增特征融合层
    head=dict(type='HeatmapHead', ...)
)

# 修改3：换预测头
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(type='HRNet', ...),
    head=dict(type='RegressionHead', ...)  # Heatmap → Regression
)
```

**2. 超参数调整**
```python
# 学习率
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001,  # 原来是 0.0005
    )
)

# Batch Size
train_dataloader = dict(
    batch_size=64,  # 原来是 32
    ...
)

# 训练轮数
train_cfg = dict(
    max_epochs=300,  # 原来是 210
    val_interval=5   # 每 5 轮验证一次
)

# 学习率策略
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000  # warmup 步数
    ),
    dict(
        type='CosineAnnealingLR',  # 余弦退火
        T_max=300,
        by_epoch=True,
        begin=0,
        end=300
    )
]
```

**3. 数据增强策略**
```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    
    # 修改：增加数据增强
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=0.3),  # 调整尺度
    
    # 新增：颜色增强
    dict(type='Albumentation',
         transforms=[
             dict(type='ColorJitter', p=0.5),
             dict(type='GaussianBlur', p=0.3),
         ]),
    
    dict(type='TopdownAffine', input_size=(256, 256)),  # 改变输入尺寸
    dict(type='GenerateTarget', ...),
    dict(type='PackPoseInputs'),
]
```

**4. 损失函数配置**
```python
# 原始
head = dict(
    type='HeatmapHead',
    loss=dict(type='KeypointMSELoss', use_target_weight=True)
)

# 修改：换损失函数
head = dict(
    type='HeatmapHead',
    loss=dict(type='AdaptiveWingLoss', alpha=2.1, omega=14)
)

# 修改：多损失组合
head = dict(
    type='HeatmapHead',
    loss=dict(
        type='CombinedLoss',
        losses=[
            dict(type='KeypointMSELoss', weight=1.0),
            dict(type='KeypointOHKMMSELoss', weight=0.5)
        ]
    )
)
```

**5. 数据集路径和配置**
```python
# 修改数据集路径
data_root = '/path/to/your/dataset/'  # 改成你的路径

# 修改标注文件
train_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/my_train.json',  # 自定义标注
        data_prefix=dict(img='images/train/'),
    )
)

# 使用不同数据集
train_dataloader = dict(
    dataset=dict(
        type='MPIIDataset',  # COCO → MPII
        data_root='data/mpii/',
        ...
    )
)
```

**6. 评估指标**
```python
# 原始
val_evaluator = dict(
    type='CocoMetric',
    ann_file='annotations/val.json'
)

# 修改：自定义评估指标
val_evaluator = [
    dict(type='CocoMetric', ann_file='annotations/val.json'),
    dict(type='PCKAccuracy', thr=0.2),  # 添加 PCK 指标
    dict(type='NME')  # 添加归一化误差
]
```

**7. 运行时配置**
```python
# 日志频率
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),  # 每 20 次迭代打印
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # 每 5 轮保存
        max_keep_ckpts=3,  # 最多保存 3 个
        save_best='coco/AP',  # 保存最佳模型
    ),
    visualization=dict(
        type='PoseVisualizationHook',
        enable=True,  # 启用可视化
        interval=100
    )
)

# 可视化后端
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),  # 添加 Tensorboard
    dict(type='WandbVisBackend',  # 添加 W&B
         init_kwargs=dict(project='mmpose', name='my_exp'))
]
```

### ⚠️ 配置层的限制

**不能做的事**：
- ❌ 添加新的模块类型（必须在 mmpose/ 中实现）
- ❌ 修改算法逻辑（必须在 mmpose/ 中修改）
- ❌ 改变数据处理的底层实现
- ❌ 修改模型的前向传播流程

**只能做的事**：
- ✅ 组合已有的模块
- ✅ 调整超参数
- ✅ 配置数据流程
- ✅ 选择损失函数和优化器

---

## 第 2 层：mmpose/ 实现层修改

### ✅ 能修改什么

#### 场景 1：添加新的骨干网络

```python
# mmpose/models/backbones/my_backbone.py

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

@MODELS.register_module()  # 注册到 MODELS
class MyBackbone(BaseBackbone):
    """自定义骨干网络"""
    
    def __init__(self, 
                 in_channels=3,
                 depth=50,
                 custom_param=10):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        
        # 构建网络层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7)
        self.layer1 = self._make_layer(64, 128, depth)
        # ...
    
    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.layer1(x)
        # ...
        return x  # 返回特征图
```

然后在配置文件中使用：
```python
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(
        type='MyBackbone',  # 使用你的骨干网络
        depth=50,
        custom_param=20
    ),
    ...
)
```

#### 场景 2：修改现有模型的逻辑

```python
# 修改 mmpose/models/heads/heatmap_head.py

@MODELS.register_module()
class HeatmapHead(BaseHead):
    
    def forward(self, feats):
        """前向传播 - 可以修改"""
        # 原始实现
        # x = self.final_layer(feats)
        
        # 修改：添加注意力机制
        x = self.attention(feats)  # 新增
        x = self.final_layer(x)
        return x
    
    def loss(self, feats, data_samples, train_cfg):
        """损失计算 - 可以修改"""
        pred_heatmaps = self.forward(feats)
        gt_heatmaps = torch.stack([d.gt_heatmaps for d in data_samples])
        
        # 原始损失
        loss = self.loss_module(pred_heatmaps, gt_heatmaps)
        
        # 修改：添加正则化
        reg_loss = self._compute_regularization()  # 新增
        
        return dict(
            loss_kpt=loss,
            loss_reg=reg_loss  # 新增
        )
```

#### 场景 3：添加新的损失函数

```python
# mmpose/models/losses/my_loss.py

from mmpose.registry import MODELS
import torch.nn as nn

@MODELS.register_module()
class MyCustomLoss(nn.Module):
    """自定义损失函数"""
    
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        """计算损失"""
        # 你的损失计算逻辑
        diff = pred - target
        loss = torch.mean(diff ** 2)  # 简化示例
        
        return loss * self.weight
```

配置文件使用：
```python
head = dict(
    type='HeatmapHead',
    loss=dict(type='MyCustomLoss', weight=2.0)
)
```

#### 场景 4：添加新的数据变换

```python
# mmpose/datasets/transforms/my_transform.py

from mmpose.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register_module()
class MyAugmentation:
    """自定义数据增强"""
    
    def __init__(self, prob=0.5, param=10):
        self.prob = prob
        self.param = param
    
    def transform(self, results):
        """执行数据变换"""
        if np.random.rand() < self.prob:
            # 修改 results 中的数据
            img = results['img']
            # 你的增强逻辑
            img = self._my_augment(img)
            results['img'] = img
        
        return results
    
    def _my_augment(self, img):
        # 具体的增强实现
        return img
```

配置文件使用：
```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='MyAugmentation', prob=0.7, param=15),  # 使用自定义变换
    dict(type='RandomFlip'),
    ...
]
```

#### 场景 5：添加新的数据集

```python
# mmpose/datasets/datasets/body/my_dataset.py

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset

@DATASETS.register_module()
class MyCustomDataset(BaseCocoStyleDataset):
    """自定义数据集
    
    数据格式：
    - images/
    - annotations/
        - train.json
        - val.json
    """
    
    METAINFO: dict = dict(
        dataset_name='my_dataset',
        paper_info=dict(...),
        keypoint_info={
            0: dict(name='nose', id=0, color=[255, 0, 0]),
            1: dict(name='left_eye', id=1, color=[255, 0, 0]),
            # ... 定义你的关键点
        },
        skeleton_info={
            0: dict(link=('nose', 'left_eye'), id=0, color=[0, 255, 0]),
            # ... 定义骨架连接
        }
    )
    
    def parse_data_info(self, raw_data_info):
        """解析单个样本的标注信息"""
        data_info = super().parse_data_info(raw_data_info)
        
        # 添加自定义字段
        data_info['custom_field'] = raw_data_info.get('custom', None)
        
        return data_info
```

配置文件使用：
```python
train_dataloader = dict(
    dataset=dict(
        type='MyCustomDataset',  # 使用自定义数据集
        data_root='data/my_dataset/',
        ann_file='annotations/train.json',
        ...
    )
)
```

#### 场景 6：添加新的评估指标

```python
# mmpose/evaluation/metrics/my_metric.py

from mmpose.registry import METRICS
from .base import BaseMetric

@METRICS.register_module()
class MyCustomMetric(BaseMetric):
    """自定义评估指标"""
    
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def process(self, data_batch, data_samples):
        """处理一个批次的预测结果"""
        for data_sample in data_samples:
            pred_coords = data_sample.pred_instances.keypoints
            gt_coords = data_sample.gt_instances.keypoints
            
            # 保存中间结果
            result = {
                'pred': pred_coords,
                'gt': gt_coords,
            }
            self.results.append(result)
    
    def compute_metrics(self, results):
        """计算最终指标"""
        # 根据 self.results 计算指标
        accuracy = self._calculate_accuracy(results)
        
        return dict(my_accuracy=accuracy)
```

配置文件使用：
```python
val_evaluator = [
    dict(type='CocoMetric'),
    dict(type='MyCustomMetric', threshold=0.3)  # 自定义指标
]
```

### 🔧 修改现有组件的步骤

**步骤 1：找到要修改的文件**
```bash
# 例如要修改 HRNet
grep -r "class HRNet" mmpose/models/backbones/
# 找到: mmpose/models/backbones/hrnet.py
```

**步骤 2：阅读并理解代码**
```python
# mmpose/models/backbones/hrnet.py
@MODELS.register_module()
class HRNet(BaseBackbone):
    def __init__(self, ...):
        # 理解初始化参数
        pass
    
    def forward(self, x):
        # 理解前向传播流程
        pass
```

**步骤 3：修改代码**
```python
# 修改 forward 方法
def forward(self, x):
    x = self.conv1(x)
    
    # 新增：添加你的逻辑
    if self.use_custom_feature:
        x = self.custom_process(x)
    
    x = self.layer1(x)
    return x
```

**步骤 4：测试修改**
```bash
# 单元测试
pytest tests/test_models/test_backbones/test_hrnet.py

# 训练测试
python tools/train.py configs/test_config.py
```

---

## 第 3 层：tools/ 执行层修改

### ✅ 能修改什么

#### 场景 1：修改训练流程

```python
# tools/train.py

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # 修改1：添加自定义初始化
    init_my_custom_module()
    
    # 修改2：在构建 Runner 前处理配置
    if args.auto_scale_lr:
        cfg = auto_scale_lr_config(cfg, args.batch_size)
    
    runner = Runner.from_cfg(cfg)
    
    # 修改3：注册自定义钩子
    runner.register_hook(MyCustomHook())
    
    # 修改4：自定义训练前的操作
    prepare_training_environment(cfg)
    
    runner.train()
    
    # 修改5：训练后处理
    post_training_analysis(runner)
```

#### 场景 2：添加新的训练脚本

```python
# tools/my_custom_train.py

"""
自定义训练脚本
用途：支持特殊的训练需求，如：
- 多阶段训练
- 课程学习
- 元学习
"""

import argparse
from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--stage', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 阶段 1：训练骨干网络
    if args.stage == 1:
        cfg = Config.fromfile(args.config)
        cfg.model.head.requires_grad = False  # 冻结 head
        runner = Runner.from_cfg(cfg)
        runner.train()
    
    # 阶段 2：微调全部网络
    elif args.stage == 2:
        cfg = Config.fromfile(args.config)
        cfg.load_from = 'work_dirs/stage1/latest.pth'
        cfg.optim_wrapper.optimizer.lr = 0.0001  # 降低学习率
        runner = Runner.from_cfg(cfg)
        runner.train()

if __name__ == '__main__':
    main()
```

使用：
```bash
# 阶段 1
python tools/my_custom_train.py configs/my_config.py --stage 1

# 阶段 2
python tools/my_custom_train.py configs/my_config.py --stage 2
```

#### 场景 3：添加分析工具

```python
# tools/analysis_tools/my_analysis.py

"""
自定义分析工具
用途：分析模型在特定数据上的表现
"""

import argparse
import torch
from mmengine.config import Config
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import MODELS

def analyze_model_on_hard_samples(config, checkpoint, data_file):
    """分析模型在困难样本上的表现"""
    
    # 加载模型
    model = init_model(config, checkpoint, device='cuda:0')
    
    # 加载困难样本
    hard_samples = load_hard_samples(data_file)
    
    results = []
    for sample in hard_samples:
        # 推理
        pred = inference_topdown(model, sample['img'])
        
        # 分析
        error = compute_error(pred, sample['gt'])
        results.append({
            'sample_id': sample['id'],
            'error': error,
            'pred': pred,
            'gt': sample['gt']
        })
    
    # 生成报告
    generate_analysis_report(results)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    
    analyze_model_on_hard_samples(
        args.config, 
        args.checkpoint, 
        args.data
    )

if __name__ == '__main__':
    main()
```

#### 场景 4：修改测试脚本

```python
# tools/test.py 的修改示例

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # 修改1：添加测试时增强（TTA）
    if args.use_tta:
        cfg = enable_test_time_augmentation(cfg)
    
    # 修改2：自定义推理配置
    if args.output_heatmaps:
        cfg.model.test_cfg['output_heatmaps'] = True
    
    runner = Runner.from_cfg(cfg)
    
    # 修改3：自定义测试逻辑
    if args.visualize_failures:
        runner.register_hook(VisualizeFailureCasesHook())
    
    # 运行测试
    metrics = runner.test()
    
    # 修改4：额外的结果处理
    if args.save_predictions:
        save_predictions_to_file(metrics, args.output_file)
    
    return metrics
```

#### 场景 5：添加自定义钩子

```python
# tools/hooks/my_hook.py

from mmengine.hooks import Hook
from mmpose.registry import HOOKS

@HOOKS.register_module()
class MyCustomHook(Hook):
    """自定义钩子
    
    用途：在训练过程的特定时刻执行自定义逻辑
    """
    
    def __init__(self, interval=100):
        self.interval = interval
    
    def before_train(self, runner):
        """训练开始前"""
        print("Training is about to start!")
    
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """每次迭代后"""
        if runner.iter % self.interval == 0:
            # 自定义逻辑：记录额外信息
            self.log_custom_info(runner, outputs)
    
    def after_train_epoch(self, runner):
        """每个 epoch 后"""
        # 自定义逻辑：保存中间结果
        self.save_intermediate_results(runner)
    
    def log_custom_info(self, runner, outputs):
        """记录自定义信息"""
        # 例如：记录梯度范数
        grad_norm = compute_grad_norm(runner.model)
        runner.logger.info(f'Grad norm: {grad_norm:.4f}')
```

配置文件中使用：
```python
custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='MyCustomHook', interval=50)  # 添加自定义钩子
]
```

---

## 🔄 完整修改流程示例

### 场景：实现一个新的姿态估计算法

**需求**：实现一个新算法，使用自定义的 Backbone + Head + Loss

#### 步骤 1：在 mmpose/ 中实现组件

```python
# 1. 自定义 Backbone
# mmpose/models/backbones/my_net.py
@MODELS.register_module()
class MyNet(BaseBackbone):
    def __init__(self, layers=[2, 2, 2, 2]):
        super().__init__()
        # 实现网络结构
    
    def forward(self, x):
        # 前向传播
        return feats

# 2. 自定义 Head
# mmpose/models/heads/my_head.py
@MODELS.register_module()
class MyHead(BaseHead):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 实现预测头
    
    def forward(self, feats):
        return predictions
    
    def loss(self, feats, data_samples, train_cfg):
        # 计算损失
        return dict(loss_kpt=loss)

# 3. 自定义 Loss
# mmpose/models/losses/my_loss.py
@MODELS.register_module()
class MyLoss(nn.Module):
    def forward(self, pred, target):
        # 损失计算
        return loss
```

#### 步骤 2：在 __init__.py 中注册

```python
# mmpose/models/backbones/__init__.py
from .my_net import MyNet
__all__ = [..., 'MyNet']

# mmpose/models/heads/__init__.py
from .my_head import MyHead
__all__ = [..., 'MyHead']

# mmpose/models/losses/__init__.py
from .my_loss import MyLoss
__all__ = [..., 'MyLoss']
```

#### 步骤 3：编写配置文件

```python
# configs/my_algorithm/my_net_coco.py

_base_ = ['../_base_/default_runtime.py']

# 模型配置
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(
        type='MyNet',  # 使用自定义 Backbone
        layers=[3, 4, 6, 3]
    ),
    head=dict(
        type='MyHead',  # 使用自定义 Head
        in_channels=2048,
        out_channels=17,
        loss=dict(type='MyLoss')  # 使用自定义 Loss
    )
)

# 数据配置
data_root = 'data/coco/'
train_dataloader = dict(...)
val_dataloader = dict(...)

# 训练配置
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01)
)

train_cfg = dict(max_epochs=200, val_interval=10)
```

#### 步骤 4：训练和测试

```bash
# 训练
python tools/train.py configs/my_algorithm/my_net_coco.py \
    --work-dir work_dirs/my_net

# 测试
python tools/test.py configs/my_algorithm/my_net_coco.py \
    work_dirs/my_net/best_coco_AP_epoch_150.pth

# 推理
python demo/image_demo.py \
    demo.jpg \
    configs/my_algorithm/my_net_coco.py \
    work_dirs/my_net/best_coco_AP_epoch_150.pth
```

---

## 💡 修改建议

### 最佳实践

1. **从配置开始**
   - 先尝试在配置层解决问题
   - 只有配置无法满足时才修改代码

2. **小步迭代**
   - 每次只修改一个组件
   - 立即测试验证效果

3. **继承而非重写**
   ```python
   # ✅ 推荐：继承基类
   @MODELS.register_module()
   class MyHead(HeatmapHead):  # 继承现有的 HeatmapHead
       def forward(self, x):
           x = super().forward(x)  # 复用父类逻辑
           x = self.my_custom_layer(x)  # 添加自己的逻辑
           return x
   
   # ❌ 不推荐：从头实现
   @MODELS.register_module()
   class MyHead(nn.Module):  # 所有逻辑都要自己写
       def forward(self, x):
           # 从头实现所有功能...
   ```

4. **使用 Projects/**
   - 实验性代码放在 `projects/my_project/`
   - 稳定后再考虑合并到 `mmpose/`

5. **版本控制**
   ```bash
   # 在 Git 中创建分支
   git checkout -b feature/my-custom-algorithm
   
   # 提交修改
   git add mmpose/models/backbones/my_net.py
   git commit -m "Add MyNet backbone"
   ```

### 调试技巧

```python
# 1. 使用 pdb 调试
import pdb; pdb.set_trace()

# 2. 打印中间结果
print(f"Feature shape: {feats.shape}")
print(f"Loss value: {loss.item()}")

# 3. 可视化
from mmpose.visualization import PoseLocalVisualizer
visualizer = PoseLocalVisualizer()
visualizer.add_datasample(...)

# 4. 单元测试
# tests/test_models/test_backbones/test_my_net.py
def test_my_net_forward():
    model = MyNet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 2048, 8, 8)
```

---

## 📝 总结

| 层次 | 修改内容 | 难度 | 影响范围 |
|------|---------|------|---------|
| **Config** | 超参数、数据增强、模型组合 | ⭐ | 单个实验 |
| **mmpose/** | 算法实现、新组件 | ⭐⭐⭐ | 整个框架 |
| **tools/** | 训练流程、工具脚本 | ⭐⭐ | 执行方式 |

**记住**：
- 90% 的实验可以通过修改配置完成
- 只有需要新功能时才修改 mmpose/
- tools/ 的修改主要是为了特殊的训练/测试需求

需要帮你实现具体的修改吗？告诉我你想做什么！
