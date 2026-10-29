# RTMO-S 私有数据 OVM 友好 Raw ONNX 阶段说明

## 1. 背景

本项目使用 MMPose/RTMO-S 训练私有人体关键点数据。目标部署平台的算子支持范围受限，用户提供的 OmniVision OA08000/NPU 文档第 5.2 节只支持有限的 ONNX 算子，并要求导出图使用 ONNX opset 9。

原始 RTMO 推理流程包含以下部署不需要放入网络图的逻辑：

- neck 中的 Transformer attention；
- 分类、框和关键点的 sigmoid、网格解码及坐标解码；
- DCC 的 bin encoding、GAU、softmax 和坐标积分；
- TopK、阈值筛选和 NMS。

此外，原始网络还有几项与目标算子约束不匹配的结构：

- Focus 的切片拼接可能被导出为不希望出现的 Slice/Gather 组合；
- SPP 使用 9x9、13x13 MaxPool，超过手册对 MaxPool 的核大小限制；
- RTMO pose 分支默认使用分组卷积，组数不符合手册中 Conv 的 Group=1 或 Group=C 限制；
- PyTorch ONNX 导出器会为共享卷积参数生成多个无计算意义的 Identity 直通节点。

## 2. 阶段目标

本阶段建立一条独立的 `custom_` 导出路径，用于生成可审查的 raw ONNX 图：

1. 关闭 neck attention，仅保留 FPN/PAN 特征融合；
2. 将 Focus 部署路径导出为 ONNX `SpaceToDepth`，并重排 stem Conv 权重保持数值等价；
3. 将超过限制的 SPP 大核池化拆分为等效的小核 MaxPool；
4. 将 pose 分支改为 Group=1；
5. 直接导出两尺度的分类、框、可见度和 pose feature 原始张量；
6. 将 DCC、NMS 和所有预测后处理留在 ONNX 图外；
7. 固定 batch=1、输入 640x640、float32、opset 9；
8. 导出后删除所有无用 Identity 节点，并对图执行静态审计。

## 3. 参考文档和依据

### 3.1 算子支持文档

主要依据是用户提供的：

`E:/cdk-v1.7.4-g-WM1111/docs/Document/User Guide/NPU_Solution_for_Deep_Learning_Inference.pdf`

重点使用第 5.2 节 `Supported Neural Network (NN) Operators`。本阶段按该表检查：

- `Conv` 的分组限制；
- `MaxPool` 的对称 padding 和最大核大小；
- `SpaceToDepth`、`Slice`、`Concat`、`Add`、`LeakyRelu`、`Upsample` 等导出算子；
- `Mul`、`Pad`、`ConvTranspose` 等版本受限算子；
- opset 9 下不可接受的 `Round` 及图外后处理相关算子。

### 3.2 项目和实现文档

- [custom RTMO-S 配置](configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py)
- [设计说明](docs/superpowers/specs/custom_2026-10-28-rtmo-ovm-raw-export-design.md)
- [实施计划与验证记录](docs/superpowers/plans/custom_2026-10-28-rtmo-ovm-raw-export.md)

## 4. 修改内容

所有本阶段新增或修改的实现文件均使用 `custom_` 前缀。原始 MMPose 的核心文件已恢复，不直接覆盖原始类和默认配置。

### 4.1 自定义模型实现

- `mmpose/models/backbones/custom_csp_darknet.py`
  - 新增 `CustomCSPDarknet` 注册类；
  - 保留 Focus 训练实现；
  - 将大于手册限制的 SPP 池化核拆分为多个 5x5/3x3 MaxPool；
  - 保持原始 SPP 的有效感受野和 Max 运算语义。

- `mmpose/models/necks/custom_hybrid_encoder.py`
  - 新增 `CustomHybridEncoder` 注册类；
  - `use_encoder_idx=[]` 且 `num_encoder_layers=0` 时不创建 encoder 和位置编码；
  - 保留其余 FPN/PAN 路径。

- `configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py`
  - 通过 `custom_imports` 加载自定义 backbone 和 neck；
  - 使用 `CustomCSPDarknet` 和 `CustomHybridEncoder`；
  - 显式关闭 neck attention；
  - pose branch 设置 `num_groups=1`、`channels_per_group=288`，在 `widen_factor=0.5` 下得到 144 个中间通道；
  - 保留私有数据集 COCO 配置和 `configs/_base_/datasets/oms_private.py` 元信息入口。

### 4.2 Raw ONNX 导出器

- `tools/deploy/rtmo/custom_export_raw_onnx.py`
  - 固定输入为 `[1, 3, 640, 640]`；
  - 固定导出 opset 9，不使用 dynamic axes；
  - 使用自定义 `SpaceToDepth` symbolic；
  - 对 stem Conv 权重进行通道重排，匹配 ONNX SpaceToDepth 的空间偏移顺序；
  - 直接调用 head module 的分类、框、可见度和 pose feature 分支；
  - 不调用 `RTMOHead.predict()`，因此 DCC、NMS 和解码不进入图；
  - 导出后自动消除 Identity 节点并重新运行 ONNX checker 和审计。

### 4.3 ONNX 审计器

- `tools/deploy/rtmo/custom_onnx_audit.py`
  - 检查 opset、输入输出名称、shape 和 float32 dtype；
  - 检查 ONNX checker 和 shape inference；
  - 按手册第 5.2 节分类支持算子、结构节点和版本受限算子；
  - 检查 Conv 分组、MaxPool 参数和 nearest Upsample；
  - 拒绝未知算子、非标准 domain、Identity、NMS、TopK、Gather、MatMul、Erf、Sin、Cos、Div 等图内节点；
  - 提供 `remove_identity_nodes()`，将 Identity 消费者直接连接到原输入。

### 4.4 测试和过程文档

- `tests/test_models/test_backbones/custom_test_csp_darknet.py`
- `tests/test_models/test_necks/custom_test_hybrid_encoder.py`
- `tests/test_models/test_heads/test_hybrid_heads/custom_test_rtmo_head.py`
- `tests/test_tools/custom_test_rtmo_raw_export.py`
- `docs/superpowers/specs/custom_2026-10-28-rtmo-ovm-raw-export-design.md`
- `docs/superpowers/plans/custom_2026-10-28-rtmo-ovm-raw-export.md`

测试覆盖 attention 关闭、SPP 等效性、pose 分支分组、Focus/SpaceToDepth 数值一致性、Identity 删除、raw 输出 shape 和算子审计。

## 5. 最终 ONNX 输出契约

最终 ONNX 文件为：

`RTMO/custom_rtmo_private_raw_opset9.onnx`

输入：

| 名称 | dtype | shape |
| --- | --- | --- |
| `images` | float32 | `[1, 3, 640, 640]` |

输出顺序固定为：

| 名称 | stride | shape | 含义 |
| --- | --- | --- | --- |
| `cls_s16` | 16 | `[1, 1, 40, 40]` | 分类 logits |
| `bbox_s16` | 16 | `[1, 4, 40, 40]` | 框 raw 输出 |
| `vis_s16` | 16 | `[1, 17, 40, 40]` | 关键点可见度 logits |
| `pose_s16` | 16 | `[1, 256, 40, 40]` | pose feature |
| `cls_s32` | 32 | `[1, 1, 20, 20]` | 分类 logits |
| `bbox_s32` | 32 | `[1, 4, 20, 20]` | 框 raw 输出 |
| `vis_s32` | 32 | `[1, 17, 20, 20]` | 关键点可见度 logits |
| `pose_s32` | 32 | `[1, 256, 20, 20]` | pose feature |

板端或外部后处理程序需要根据这些 raw 输出执行 sigmoid、框解码、DCC/关键点解码、阈值筛选和 NMS。

## 6. 冒烟测试结果

使用随机初始化的 smoke checkpoint：

`RTMO/custom_rtmo_random_smoke.pth`

执行了完整流程：

1. 加载自定义配置和 checkpoint；
2. 构建 `CustomCSPDarknet`、`CustomHybridEncoder` 和 RTMO head；
3. 执行 `switch_to_deploy()`；
4. 导出 opset 9 ONNX；
5. 删除 66 个 Identity 节点；
6. 执行 ONNX checker 和静态算子审计；
7. 使用 ONNX Runtime 检查输出 shape 和数值。

审计报告为：

`RTMO/custom_rtmo_private_raw_opset9_audit.json`

最终图的主要算子计数如下：

```text
Conv=80
LeakyRelu=67
MaxPool=6
Upsample=2
Slice=4
Concat=9
Add=11
SpaceToDepth=1
Constant=2
Identity=0
```

审计结果：

- opset：9；
- 输入输出 shape：通过；
- 输入输出 dtype：float32；
- 未列出算子：无；
- 版本受限算子：无；
- NMS/DCC/attention 指示节点：无；
- Conv 分组问题：无；
- MaxPool/Upsample 属性问题：无；
- ONNX Runtime 与 PyTorch wrapper 最大绝对误差：约 `2.1e-3`；
- 自定义回归测试：18 passed。

## 7. 当前边界

- 当前 ONNX 使用随机 smoke checkpoint，不能代表私有训练模型的精度；
- pose branch 结构从默认分组卷积改为 Group=1，已有旧 pose-head 权重不能直接视为完全兼容，需要重新训练或微调；
- 本阶段只验证 ONNX 静态图和本地 ONNX Runtime 数值语义，没有执行 NPUC 或板端编译；
- 完成私有数据训练后，应使用真实 checkpoint 重新执行导出命令和审计命令，并保存新的 `custom_` ONNX 与审计报告。

## 8. 使用命令

```powershell
C:\Users\admin\miniforge3\envs\rtmpose\python.exe tools/deploy/rtmo/custom_export_raw_onnx.py `
  --config configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py `
  --checkpoint <训练完成的checkpoint> `
  --output RTMO/custom_rtmo_private_raw_opset9.onnx `
  --audit
```

独立审计命令：

```powershell
C:\Users\admin\miniforge3\envs\rtmpose\python.exe tools/deploy/rtmo/custom_onnx_audit.py `
  RTMO/custom_rtmo_private_raw_opset9.onnx
```
