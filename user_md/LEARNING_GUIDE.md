# MMPose 仓库学习指南

## 📋 项目概览

**MMPose** 是 OpenMMLab 开源的姿态估计工具箱，基于 PyTorch 构建。

### 核心信息
- **版本**: v1.3.0
- **Python 支持**: ≥ 3.7
- **PyTorch 支持**: ≥ 1.8
- **许可证**: Apache 2.0

### 主要功能
- 2D/3D 人体姿态估计
- 手部关键点检测
- 人脸关键点检测
- 全身姿态估计（133 个关键点）
- 动物姿态估计
- 服饰关键点检测

---

## 🗂️ 目录结构

```
mmpose/
├── configs/              # 配置文件（按任务类型分类）
│   ├── _base_/          # 基础配置（数据集、模型、训练策略）
│   ├── body_2d_keypoint/    # 2D 人体姿态
│   ├── body_3d_keypoint/    # 3D 人体姿态
│   ├── hand_2d_keypoint/    # 手部关键点
│   ├── face_2d_keypoint/    # 人脸关键点
│   ├── wholebody_2d_keypoint/ # 全身姿态
│   └── animal_2d_keypoint/  # 动物姿态
│
├── mmpose/              # 核心代码库
│   ├── apis/           # 高级 API（推理、训练接口）
│   ├── models/         # 模型定义
│   │   ├── backbones/     # 骨干网络（ResNet, HRNet 等）
│   │   ├── necks/         # 特征融合层
│   │   ├── heads/         # 预测头
│   │   ├── losses/        # 损失函数
│   │   └── pose_estimators/ # 完整姿态估计器
│   ├── datasets/       # 数据集定义与数据加载
│   │   ├── datasets/      # 数据集类
│   │   └── transforms/    # 数据增强
│   ├── codecs/         # 编解码器（坐标编码/解码）
│   ├── evaluation/     # 评估指标
│   ├── structures/     # 数据结构（bbox, keypoint）
│   ├── engine/         # 训练引擎（hooks, 优化器）
│   ├── utils/          # 工具函数
│   └── visualization/  # 可视化工具
│
├── tools/              # 训练和测试脚本
│   ├── train.py           # 训练入口
│   ├── test.py            # 测试入口
│   ├── dist_train.sh      # 分布式训练脚本
│   ├── analysis_tools/    # 分析工具
│   └── dataset_converters/ # 数据集转换工具
│
├── demo/               # 示例代码
│   ├── image_demo.py      # 图像推理
│   ├── inferencer_demo.py # 推理器示例
│   └── topdown_demo_with_mmdet.py # 结合目标检测
│
├── projects/           # 社区项目（RTMPose, RTMO 等）
├── tests/              # 单元测试
└── docs/               # 文档

```

---

## 🚀 快速开始流程

### 1. 环境安装
```bash
# 创建虚拟环境（推荐）
conda create -n mmpose python=3.9 -y
conda activate mmpose

# 安装 PyTorch（根据 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装 MMPose
pip install -U openmim
mim install mmengine mmcv mmdet
pip install -e .
```

### 2. 运行第一个 Demo
```bash
# 使用预训练模型进行推理
python demo/image_demo.py \
    demo/resources/human_pose.jpg \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --out-file vis_results.jpg
```

### 3. 训练自己的模型
```bash
# 单 GPU 训练
python tools/train.py configs/xxx.py

# 多 GPU 训练
bash tools/dist_train.sh configs/xxx.py 8
```

---

## 🧩 核心概念

### 1. 配置系统
MMPose 使用继承式配置文件系统：

```python
# 示例：configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py

_base_ = [
    '../../../_base_/default_runtime.py',  # 运行时配置
    '../../../_base_/datasets/coco.py'     # 数据集配置
]

# 模型配置
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(type='HRNet', ...),
    head=dict(type='HeatmapHead', ...)
)

# 训练配置
train_cfg = dict(max_epochs=210, ...)
```

**配置继承优先级**: 具体配置 > _base_ 配置

### 2. 模型架构
典型的姿态估计模型由以下部分组成：

```
输入图像
   ↓
Backbone (ResNet/HRNet/...)  # 特征提取
   ↓
Neck (可选)                   # 特征融合
   ↓
Head (Heatmap/Regression/...) # 关键点预测
   ↓
输出关键点坐标
```

### 3. Top-down vs Bottom-up
- **Top-down**: 先检测人 → 再预测关键点（精度高，适合多人场景）
- **Bottom-up**: 先检测所有关键点 → 再组装成人（速度快）

### 4. Codec（编解码器）
负责关键点的编码和解码：
- **编码**: 将坐标转换为热图或其他表示
- **解码**: 将模型输出转回坐标

常见类型：
- `MSRAHeatmap`: 高斯热图编码
- `SimCCLabel`: SimCC 方法的标签编码
- `RegressionLabel`: 直接回归坐标

---

## 📚 学习路径

### 阶段 1：基础使用（1-2 天）
**目标**: 能运行 Demo 和使用预训练模型

1. ✅ 阅读 [README_CN.md](README_CN.md) 了解项目概况
2. ✅ 安装环境并运行 [demo/image_demo.py](demo/image_demo.py)
3. ✅ 尝试不同的预训练模型（在 [Model Zoo](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo.html) 中查找）
4. ✅ 学习使用 Inferencer API（最简单的推理接口）

**关键文件**:
- [demo/inferencer_demo.py](demo/inferencer_demo.py) - 推理器示例
- [mmpose/apis/inferencers/](mmpose/apis/inferencers/) - 推理器实现

### 阶段 2：理解配置系统（2-3 天）
**目标**: 能读懂和修改配置文件

1. ✅ 阅读 [配置文件教程](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/configs.html)
2. ✅ 分析 [configs/_base_/](configs/_base_/) 中的基础配置
3. ✅ 理解配置继承机制
4. ✅ 修改一个配置文件并训练

**关键文件**:
- [configs/_base_/default_runtime.py](configs/_base_/default_runtime.py) - 默认运行时配置
- [configs/_base_/datasets/](configs/_base_/datasets/) - 数据集配置
- [configs/_base_/schedules/](configs/_base_/schedules/) - 训练策略

### 阶段 3：数据准备与训练（3-5 天）
**目标**: 能在自己的数据集上训练模型

1. ✅ 了解数据集格式（COCO 格式）
2. ✅ 准备自定义数据集
3. ✅ 编写数据集配置文件
4. ✅ 训练并评估模型

**关键文件**:
- [tools/dataset_converters/](tools/dataset_converters/) - 数据格式转换
- [mmpose/datasets/datasets/](mmpose/datasets/datasets/) - 数据集实现
- [docs/zh_cn/user_guides/prepare_datasets.html](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/prepare_datasets.html)

### 阶段 4：模型架构深入（5-7 天）
**目标**: 理解模型实现细节

1. ✅ 学习 [mmpose/models/](mmpose/models/) 目录结构
2. ✅ 阅读关键模块源码：
   - Backbone: [mmpose/models/backbones/hrnet.py](mmpose/models/backbones/hrnet.py)
   - Head: [mmpose/models/heads/heatmap_head.py](mmpose/models/heads/heatmap_head.py)
   - Loss: [mmpose/models/losses/mse_loss.py](mmpose/models/losses/mse_loss.py)
3. ✅ 理解前向传播流程
4. ✅ 学习如何自定义模块

**关键文件**:
- [mmpose/models/pose_estimators/topdown.py](mmpose/models/pose_estimators/topdown.py) - Top-down 模型
- [mmpose/models/heads/](mmpose/models/heads/) - 各种预测头

### 阶段 5：高级功能（7+ 天）
**目标**: 能实现论文和自定义算法

1. ✅ 学习 Codec 机制
2. ✅ 实现自定义数据增强
3. ✅ 实现自定义损失函数
4. ✅ 实现自定义模型
5. ✅ 模型部署（ONNX/TensorRT）

**关键文件**:
- [mmpose/codecs/](mmpose/codecs/) - 编解码器
- [mmpose/datasets/transforms/](mmpose/datasets/transforms/) - 数据变换
- [docs/zh_cn/advanced_guides/](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/)

---

## 🔑 关键 API 与工具

### 1. Inferencer（推荐用于推理）
```python
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d='rtmpose-m',  # 模型名称或配置文件路径
    pose2d_weights='checkpoint.pth'  # 权重文件
)

result = inferencer('demo.jpg', show=True)
```

### 2. 训练与测试
```bash
# 训练
python tools/train.py CONFIG_FILE [--work-dir WORK_DIR]

# 测试
python tools/test.py CONFIG_FILE CHECKPOINT_FILE [--out OUTPUT_FILE]
```

### 3. 模型分析
```bash
# 计算 FLOPs 和参数量
python tools/analysis_tools/get_flops.py CONFIG_FILE

# 可视化模型
python tools/analysis_tools/visualize_network.py CONFIG_FILE --output model.png
```

---

## 📖 重要文档链接

### 官方文档
- [中文文档首页](https://mmpose.readthedocs.io/zh_CN/latest/)
- [20 分钟上手教程](https://mmpose.readthedocs.io/zh_CN/latest/guide_to_framework.html)
- [模型库](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo.html)
- [数据集准备](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/prepare_datasets.html)

### 进阶教程
- [实现新模型](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/implement_new_models.html)
- [自定义数据集](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/customize_datasets.html)
- [编解码器详解](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/codecs.html)

### 社区资源
- [GitHub Issues](https://github.com/open-mmlab/mmpose/issues)
- [论文复现列表](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html)
- [RTMPose 项目](projects/rtmpose/) - 最新的实时姿态估计模型

---

## 💡 学习建议

### 1. 动手实践
- 不要只看文档，一定要运行代码
- 修改配置文件参数，观察效果变化
- 在小数据集上快速实验

### 2. 阅读源码技巧
- 从配置文件入手，找到对应的模型类
- 使用 IDE 的跳转功能追踪函数调用
- 重点关注 `__init__` 和 `forward` 方法

### 3. 调试技巧
```python
# 在配置文件中设置日志级别
log_level = 'DEBUG'

# 可视化数据增强结果
python tools/analysis_tools/browse_dataset.py CONFIG_FILE

# 单步调试训练流程
python -m pdb tools/train.py CONFIG_FILE
```

### 4. 常见问题排查
- **内存不足**: 减小 `batch_size`
- **训练不收敛**: 检查学习率、数据标注
- **推理速度慢**: 使用轻量级模型或模型量化

---

## 🎯 推荐项目

这些是 MMPose 仓库中值得重点关注的项目：

### 1. RTMPose
- 路径: [projects/rtmpose/](projects/rtmpose/)
- 特点: 实时高精度姿态估计
- 适合: 工业应用、移动端部署

### 2. RTMO
- 路径: [projects/rtmo/](projects/rtmo/)
- 特点: 单阶段实时多人姿态估计
- 适合: 多人场景、实时应用

### 3. PoseAnything
- 路径: [projects/pose_anything/](projects/pose_anything/)
- 特点: 零样本姿态估计
- 适合: 研究新任务

---

## 🔧 实用命令速查

```bash
# 查看配置文件最终内容（继承后）
python tools/misc/print_config.py CONFIG_FILE

# 测试数据加载速度
python tools/analysis_tools/analyze_data_loading.py CONFIG_FILE

# 可视化数据集
python tools/misc/browse_dataset.py CONFIG_FILE

# 转换模型为 ONNX
python tools/deployment/pytorch2onnx.py CONFIG_FILE CHECKPOINT_FILE

# 日志可视化（需要 tensorboard）
tensorboard --logdir work_dirs/
```

---

## 📝 下一步行动

根据你的目标选择：

**如果你想快速使用**:
1. 运行 [demo/inferencer_demo.py](demo/inferencer_demo.py)
2. 浏览 [Model Zoo](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo.html) 选择合适的预训练模型
3. 使用 Inferencer API 集成到项目中

**如果你想训练模型**:
1. 准备数据集（COCO 格式）
2. 参考 [configs/body_2d_keypoint/](configs/body_2d_keypoint/) 修改配置
3. 运行 `python tools/train.py YOUR_CONFIG.py`

**如果你想研究算法**:
1. 阅读 [mmpose/models/](mmpose/models/) 源码
2. 研究论文对应的配置文件
3. 在 [projects/](projects/) 中贡献新算法

---

## ❓ 遇到问题？

1. 查看 [FAQ](https://mmpose.readthedocs.io/zh_CN/latest/faq.html)
2. 搜索 [GitHub Issues](https://github.com/open-mmlab/mmpose/issues)
3. 加入社区微信群（见 [README_CN.md](README_CN.md) 底部）

---

**最后更新**: 2026-10-09
**版本**: MMPose v1.3.0
