# RTMO-S OVM 友好 Raw ONNX 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为私有 RTMO-S 建立固定 batch=1、640x640、opset 9 的 OVM 友好 raw-head ONNX 导出路径，并将 neck attention、NMS 与 DCC 解码排除在计算图之外。

**架构：** 让 HybridEncoder 支持空 transformer encoder 配置，SPP 大核以重复 5x5 池化精确分解；训练配置将 RTMO pose branch 改为普通卷积。新增导出 wrapper 使用 SpaceToDepth 和 stem 权重重排保留 Focus 数值行为，直接输出两尺度的 cls、bbox、visibility、pose feature 张量；静态图审计逐项报告算子和 shape。

**技术栈：** PyTorch、MMEngine/MMPose、ONNX opset 9、pytest、ONNX Runtime。

**设计说明：** `docs/superpowers/specs/2026-10-28-rtmo-ovm-raw-export-design.md`

## 全局约束

- 输入固定为 float32 NCHW `[1,3,640,640]`，不导出 dynamic axes。
- 输出固定为 8 个 NCHW 四维 tensor：两尺度各含 cls `[1,1,H,W]`、bbox `[1,4,H,W]`、kpt visibility `[1,17,H,W]`、pose feature `[1,256,H,W]`；`H,W` 分别为 40、40 和 20、20。
- ONNX opset 必须为 9；导出失败或有未解释的表外节点时，不得标记为 OVM 兼容。
- NMS、阈值筛选、bbox decode、DCC 和关键点坐标解码必须在 ONNX 图外。
- 改动私有配置和框架源码时，不覆盖工作区既有的用户文件；只修改本计划列出的文件。
- 算子结论以用户提供手册第 5.2 节和导出 ONNX 静态图为准；本轮不执行 NPUC。

---

### 任务 1：让 HybridEncoder 支持关闭 attention

**文件：**
- 修改：`mmpose/models/necks/hybrid_encoder.py`
- 测试：`tests/test_models/test_necks/custom_test_hybrid_encoder.py`
- 修改：`configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py`

**接口：**
- 输入配置：`use_encoder_idx=[]` 且 `num_encoder_layers=0`。
- 输出行为：输入特征直接进入现有 FPN/PAN；不构造 `encoder` 或 positional encoding，不产生 attention。

- [x] **步骤 1：先添加失败测试**

测试构建三层输入、`hidden_dim=32`、`use_encoder_idx=[]`、`num_encoder_layers=0` 的 HybridEncoder，断言 forward 输出层数/空间分辨率正确且模块没有 `encoder`。

- [x] **步骤 2：运行测试确认按预期失败**

运行：`C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m pytest tests/test_models/test_necks/custom_test_hybrid_encoder.py -q`

预期：当前实现因 `pos_enc_dim` 未初始化而失败。

- [x] **步骤 3：最小实现**

只在 `use_encoder_idx` 非空时初始化位置编码和 transformer encoder；`forward()` 与 `switch_to_deploy()` 保持 encoder 相关逻辑受 `num_encoder_layers > 0` 控制。私有 RTMO 配置显式增加 `use_encoder_idx=[]`、`num_encoder_layers=0`。

- [x] **步骤 4：重跑测试并确认通过**

运行上述 pytest 命令；再加载私有配置并构建模型，确认 neck 无 `MultiheadAttention`、无 `encoder`，输出特征 shape 与 attention 开启时一致。

### 任务 2：将 SPP 大核池化精确分解为 5x5 池化

**文件：**
- 修改：`mmpose/models/backbones/csp_darknet.py`
- 测试：`tests/test_models/test_backbones/custom_test_csp_darknet.py`

**接口：** `SPPBottleneck(kernel_sizes=(5,9,13))` 仍返回与现有模块相同的 shape 和数值；实际池化节点的 kernel 最大为 5。

- [x] **步骤 1：先添加失败测试**

用固定 tensor 分别比较单个 `MaxPool2d(k, stride=1, padding=k//2)` 与 5x5 池化重复 `1+(k-5)//4` 次的结果，覆盖 `k=5,9,13` 和边界像素；另断言 SPP 模块中没有大于 5 的池化核。

- [x] **步骤 2：运行测试确认失败**

运行：`C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m pytest tests/test_models/test_backbones/custom_test_csp_darknet.py -q -k spp`

预期：当前模块仍含 9x9 和 13x13 池化。

- [x] **步骤 3：最小实现**

将大于 8 的奇数核对应池化替换为 `nn.Sequential`：重复 `MaxPool2d(5, stride=1, padding=2)` 共 `(kernel_size-1)//4` 次；当余数 `(kernel_size-1)%4` 为 2 时，再追加一次 `MaxPool2d(3, stride=1, padding=1)`。不大于 8 的核保留原核。保持并行金字塔分支输出个数与通道拼接顺序不变。

- [x] **步骤 4：运行 focused 和原测试**

运行 `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m pytest tests/test_models/test_backbones/custom_test_csp_darknet.py -q`，确认池化数值等价以及 P5/P6 shape 测试通过。

### 任务 3：修改私有 RTMO pose branch 为 Group=1

**文件：**
- 修改：`configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py`
- 测试：`tests/test_models/test_heads/test_hybrid_heads/custom_test_rtmo_head.py`

**接口：** `head_module_cfg` 设置 `num_groups=1`、`channels_per_group=288`；在 `widen_factor=0.5` 下 pose 中间通道仍为 144，pose 分支所有 Conv 的 groups 均为 1。

- [x] **步骤 1：先添加失败测试**

按配置构建轻量 RTMOHeadModule，断言每层 `conv_pose` 的卷积 `groups == 1`、输出通道为 144，且 `forward()` 返回的分类、框、visibility、pose feature 两层尺度 shape 匹配输入特征。

- [x] **步骤 2：运行测试确认失败**

运行：`C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m pytest tests/test_models/test_heads/test_hybrid_heads/custom_test_rtmo_head.py -q -k group`

预期：配置默认仍为 `num_groups=8`。

- [x] **步骤 3：更新私有配置**

仅在私有配置 `head_module_cfg` 设置 `num_groups=1`、`channels_per_group=288`；不改通用 RTMO 默认值。保留分类/visibility 使用的 sigmoid 在模型图外处理。

- [x] **步骤 4：验证 shape 和 groups**

运行同一 pytest 命令，并用私有配置构造模型确认 pose 分支中间通道 144、groups=1。记录 pose 分支旧权重 shape 不兼容，需要重新初始化/微调。

### 任务 4：新增 raw ONNX exporter 和算子审计

**文件：**
- 新建：`tools/deploy/rtmo/custom_export_raw_onnx.py`
- 新建：`tools/deploy/rtmo/custom_onnx_audit.py`
- 新建：`tests/test_tools/custom_test_rtmo_raw_export.py`
- 更新：`docs/superpowers/specs/2026-10-28-rtmo-ovm-raw-export-design.md`

**接口：**
- `RTMORawExportWrapper.forward(images) -> tuple[Tensor, ...]` 返回顺序固定为 `cls_s16, bbox_s16, vis_s16, pose_s16, cls_s32, bbox_s32, vis_s32, pose_s32`。
- CLI 接受配置、checkpoint、输出路径；固定 dummy input 为 `[1,3,640,640]`，`opset_version=9`，无 dynamic axes。
- `audit_onnx(path)` 报告 opset、输入/输出 shape、op type 计数、Conv groups 和未列入 CDK v1.7.4 ONNX 表的计算节点。

- [x] **步骤 1：先添加 exporter 行为测试**

用轻量假的 backbone/neck/head module 构造 wrapper，断言输出 tuple 的 8 个 tensor 均为 4D 且顺序/形状正确；断言固定输入错误时抛出带期望 shape 的 `ValueError`。

- [x] **步骤 2：先添加 SpaceToDepth parity 和 ONNX 测试**

测试部署 Focus 分支在固定图像上与原 `Focus` 输出 allclose；使用 legacy exporter 导出只含该变换的模块，断言 opset 9 图含 `SpaceToDepth` 节点，而不是动态 Slice/Gather。

- [x] **步骤 3：运行测试确认失败**

运行：`C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m pytest tests/test_tools/custom_test_rtmo_raw_export.py -q`

预期：raw wrapper、export symbolic、审计函数尚不存在。

- [x] **步骤 4：实现 raw wrapper**

wrapper 固定遍历 CSPDarknet 的 stage；stem 输入使用自定义 autograd `SpaceToDepth` symbolic，forward 将 PixelUnshuffle 结果改排成 ONNX 的空间偏移优先顺序，backward 执行逆变换；建立 stem Conv 副本并按 ONNX 与旧 Focus 的通道次序重排权重。然后调用 neck 和 head module 的原始预测分支，只返回 cls、bbox、kpt visibility、pose feature 两尺度输出，不调用 `RTMOHead.predict()`、NMS 或 DCC。

- [x] **步骤 5：实现 CLI 导出**

读取 config/checkpoint 后将模型置 eval，调用 `model.switch_to_deploy()`，构造固定输入 wrapper，使用 PyTorch legacy `torch.onnx.export(..., opset_version=9, do_constant_folding=True)` 导出；显式设置 8 个 output names，不传 dynamic axes。

- [x] **步骤 6：实现 ONNX audit**

使用 `onnx.load`、`onnx.checker.check_model` 和 shape inference；验证输入 `[1,3,640,640]`、八个输出的固定 shape、opset=9；统计节点，检查 `NonMaxSuppression`、`TopK`、`Sin`、`Cos`、`MatMul`、`Erf` 等 DCC/attention 指示节点不在图中；检查所有 Conv `group==1` 或 `group==input_channels==output_channels`。将 Constant/Shape/Identity 等结构节点单列，计算算子依据手册白名单判定。

- [x] **步骤 7：重跑 exporter/audit 测试**

运行：`C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m pytest tests/test_tools/custom_test_rtmo_raw_export.py -q`。

- [x] **步骤 8：执行完整配置的随机权重导出与静态检查**

命令：`C:\Users\admin\miniforge3\envs\rtmpose\python.exe tools/deploy/rtmo/custom_export_raw_onnx.py --config configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py --checkpoint <微调后checkpoint> --output RTMO/custom_rtmo_private_raw_opset9.onnx --audit`

结果：使用私有配置随机初始化模型完成导出，输入输出契约与手册算子审计通过。当前没有使用私有训练 checkpoint，因此这不是训练权重产物。

### 任务 5：验证 raw 输出语义及部署编译入口

**文件：**
- 修改：`tests/test_tools/custom_test_rtmo_raw_export.py`
- 更新：`docs/superpowers/specs/2026-10-28-rtmo-ovm-raw-export-design.md`
- 更新：`docs/superpowers/plans/2026-10-28-rtmo-ovm-raw-export.md`

- [x] **步骤 1：数值对照 raw wrapper 和 head module**

固定随机输入，以 `model.extract_feat(x)` 和 `model.head.head_module(features)` 作为参考，逐项比较 wrapper 的 8 个结果；Focus 输出单独与 SpaceToDepth/权重重排参考比较。

- [x] **步骤 2：检查 ONNX 图输出契约**

重新运行 full export/audit，将算子计数、输入输出 shape 和违规项结果记录到设计文档；明确任何仍在表外的节点和对应源头，不将它们静默忽略。

- [x] **步骤 3：运行相关回归测试**

运行：`C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m pytest tests/test_models/test_backbones/custom_test_csp_darknet.py tests/test_models/test_necks/custom_test_hybrid_encoder.py tests/test_models/test_heads/test_hybrid_heads/custom_test_rtmo_head.py tests/test_tools/custom_test_rtmo_raw_export.py -q`。

- [x] **步骤 4：按用户要求不执行 NPUC 编译**

本轮验收范围仅为 ONNX 图符合手册算子约束和固定 raw 输出契约；不执行 NPUC，不据此声称已经完成板端编译。

## 执行结果

- 相关回归测试：18 passed。
- 完整私有配置随机权重 ONNX：opset 9、固定输入和八个 4D 输出均符合约束；移除了 66 个 Identity 节点，未出现手册外算子、NMS/DCC/attention 节点或违规 Conv 分组。
- ONNX Runtime 与 PyTorch wrapper 最大绝对误差：约 `2.1e-3`；Focus 的 SpaceToDepth 部署路径另有独立数值对照。
- 当前没有使用私有训练 checkpoint；需要训练完成后以实际 checkpoint 再执行导出和审计。

## 自审记录

- 设计说明覆盖 attention、Focus、SPP、Group=1、DCC/NMS 外置、输出契约和 opset；实施任务逐项覆盖。
- SpaceToDepth 使用自定义 ONNX symbolic，避免当前 PyTorch `PixelUnshuffle` 自动导出成 Reshape/Transpose 链；ONNX Runtime 回归验证了空间偏移通道次序。
- SPP 通过重复 5x5 MaxPool 保持数学等价，避免将池化替成非等价卷积金字塔造成额外精度变化。
- Group=1 会改变 pose head 参数和计算量，计划明确要求重新初始化/微调并记录资源风险。
- 按用户要求，本任务验收以手册算子表和 ONNX 静态图/本地数值语义为界，不包含 NPUC 编译。
