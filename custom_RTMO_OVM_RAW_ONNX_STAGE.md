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

## 9. 前处理与后处理

raw ONNX 图只包含 backbone、neck 特征融合和分类/框/可见度/pose feature 输出。部署侧必须在图外完成图像预处理和预测解码；这部分实现位于：

- `RTMO/custom_rtmo_preprocess_postprocess.py`
- 使用说明：`RTMO/custom_RTMO_PREPOST_USAGE.md`
- PyTorch、ONNX Runtime 对照工具：`RTMO/custom_rtmo_smoke_test.py`

板端运行模块仅依赖 NumPy 和 OpenCV，不依赖 PyTorch、MMCV、MMEngine、MMPose 或 ONNX Runtime。ONNX Runtime 不是板端模块依赖；若板端 runtime 由其他 API 提供模型输出，只需将八个输出数组按名称传入后处理器。

### 9.1 前处理

前处理对应配置中的 `BottomupResize(input_size=(640,640), pad_val=(114,114,114))` 和 `PoseDataPreprocessor`：

- 输入图像为 OpenCV BGR 三通道图像；
- 保持原图宽高比，将图像仿射缩放并居中填充到 640x640，填充值为 114；
- 不做 BGR/RGB 转换、不除以 255、不做 mean/std 归一化；
- 输出为 float32 NCHW，shape 为 `[1,3,640,640]`，像素范围仍为 0..255；
- 同时返回输入中心、尺度和原图尺寸元数据，供后处理恢复原图坐标。

### 9.2 后处理

后处理按源码 RTMO head 的推理顺序执行：

1. 按 stride 16、32 的中心化 prior 网格顺序展平 raw 张量；
2. 对分类 logits 和关键点可见度 logits 执行 sigmoid；
3. 按 RTMO bbox 的中心偏移、log 宽高公式解码框；
4. 按 `score_thr=0.1` 筛选候选，并按 `nms_thr=0.65` 执行源码一致的 IoU NMS；
5. 用 bbox padding=1.25 和候选网格坐标执行 DCC 完整解码，包括 pose feature 投影、GAU、动态坐标 bins、sin/cos 编码、softmax 和积分；
6. 将关键点和框从 640x640 输入坐标系映射回原图坐标系。

DCC 的 NumPy 参数必须由**正在部署的同一个训练 checkpoint**导出。不同 checkpoint 间不可复用 `custom_rtmo_dcc_params.npz`，也不要把当前随机 smoke checkpoint 生成的参数用于正式模型。后处理结果包含原图坐标系的 `bboxes [N,4]`、`scores [N]`、`keypoints [N,17,2]` 和 `keypoint_scores [N,17]`。

## 10. 正式训练、评估与部署导出

以下命令均从 MMPose 仓库根目录的 PowerShell 执行，使用本项目已验证的 `rtmpose` Conda 环境。正式运行前确认 `datasets/coco_annotations/{train,val,test}.json`、标注所引用的图片以及元信息文件 `configs/_base_/datasets/oms_private.py` 均存在且路径正确。

### 10.1 启动训练

配置采用 600 epoch 训练计划，训练日志和 checkpoint 输出到 `work_dirs/custom_rtmo_private`：

```powershell
& 'C:\Users\admin\miniforge3\envs\rtmpose\python.exe' tools/train.py `
  configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py `
  --work-dir work_dirs/custom_rtmo_private
```

中断后从该目录自动恢复训练：

```powershell
& 'C:\Users\admin\miniforge3\envs\rtmpose\python.exe' tools/train.py `
  configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py `
  --work-dir work_dirs/custom_rtmo_private `
  --resume
```

训练配置中的 batch size 为每卡 32。单卡显存不足时应先按机器资源调整 batch size，并相应核对学习率缩放策略；不要直接把配置文件名中的 `8xb32` 当作本地自动启动八卡的命令。多卡训练时需使用 MMEngine 支持的分布式启动方式。

### 10.2 评估训练 checkpoint

训练完成后，先用验证/测试入口核对 checkpoint 和数据链路。下面以最终 epoch checkpoint 为例；若训练目录内实际 checkpoint 文件名不同，请将路径替换为实际文件：

```powershell
& 'C:\Users\admin\miniforge3\envs\rtmpose\python.exe' tools/test.py `
  configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py `
  work_dirs/custom_rtmo_private/epoch_600.pth `
  --work-dir work_dirs/custom_rtmo_private/test_results
```

### 10.3 导出并审计正式 ONNX

将 `<训练完成的checkpoint>` 替换为评估通过的实际 checkpoint 路径。ONNX 和审计报告输出到 `RTMO/`：

```powershell
& 'C:\Users\admin\miniforge3\envs\rtmpose\python.exe' tools/deploy/rtmo/custom_export_raw_onnx.py `
  --config configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py `
  --checkpoint <训练完成的checkpoint> `
  --output RTMO/custom_rtmo_private_raw_opset9.onnx `
  --audit
```

成功后应检查 `RTMO/custom_rtmo_private_raw_opset9_audit.json`，确认输入仍为 `[1,3,640,640]`、八个输出 shape 和顺序符合第 5 节契约，opset 为 9，`Identity=0`，未列出算子和违反项均为空。

### 10.4 导出同 checkpoint 的 DCC 参数并完成联调

使用相同 checkpoint 和实际测试图运行 smoke 工具。该命令会重新加载模型，生成与 checkpoint 对应的 `RTMO/custom_rtmo_dcc_params.npz`，并将 raw ONNX/PyTorch 数值对照和 DCC 对照结果写入 `RTMO/custom_rtmo_smoke_report.json`：

```powershell
& 'C:\Users\admin\miniforge3\envs\rtmpose\python.exe' RTMO/custom_rtmo_smoke_test.py `
  --config configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py `
  --checkpoint <与ONNX相同的训练checkpoint> `
  --onnx RTMO/custom_rtmo_private_raw_opset9.onnx `
  --image <用于联调的真实测试图像> `
  --dcc-params RTMO/custom_rtmo_dcc_params.npz `
  --report RTMO/custom_rtmo_smoke_report.json
```

检查报告中的八路 raw tensor 误差、`dcc_probe_max_abs_error`、源码后处理实例数和输出 shape。若更换 checkpoint，必须重新导出 ONNX 并重新生成 DCC 参数，再执行联调；不要将 ONNX 与 DCC 参数混用不同 checkpoint 的产物。

本流程完成的是训练配置的实际训练、raw ONNX 导出/审计及 NumPy 前后处理联调。它不包含 NPUC 转换、模拟器运行或板端编译；这些属于后续部署验证阶段。
