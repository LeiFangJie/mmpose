# RTMO-S OVM 友好 Raw ONNX 设计

## 目标

为私有数据训练得到的 RTMO-S 提供可审查的 OVM 友好模型图：固定输入 `1x3x640x640`、ONNX opset 9，网络只输出两个尺度上的分类、框、关键点可见度和 pose feature 四维张量；候选筛选、NMS 与 DCC 解码留在 ONNX 图外。

## 依据与当前问题

- 当前 RTMO-S 使用 `CSPDarknet + HybridEncoder + RTMOHead`。
- `HybridEncoder` 可通过 `use_encoder_idx` 选择 transformer encoder 特征层，但空列表目前会在位置编码初始化路径出错；neck attention 因而没有可直接使用的关闭配置。
- CSPDarknet Focus 使用步长切片后拼接；已知 opset 9 导出曾在这一路径失败。导出图可用 ONNX `SpaceToDepth` 表达，并用重排后的 stem 卷积权重保持 Focus 原有通道顺序。
- SPP 默认使用 5、9、13 的 MaxPool。手册第 5.2 节列出对称 MaxPool，限制最大核为 8；9 和 13 核有兼容性风险。stride=1 的 9、13 池化可分别精确分解成 2、3 次 5x5 池化。
- RTMO pose branch 的后续卷积使用 `groups=8`。手册对 Conv 的限制为 Group=1 或 Group=C；当前组数不符合该条件。训练配置将该分支改为 Group=1，并保持中间通道数 144。
- `RTMOHead.predict()` 内执行分类/框解码、候选筛选、NMS、DCC 和关键点解码；raw 导出 wrapper 将直接调用 head module，避开这段预测后处理。
- `switch_to_deploy()` 仅执行模型已有的部署改写/固定位置编码预计算，不作为算子兼容保证。

## 设计

### 网络结构和训练

在 `HybridEncoder` 中让 `use_encoder_idx=[]` 成为合法配置；该设置跳过 transformer encoder 和位置编码，但保留原 FPN/PAN。私有 RTMO-S 配置明确设置空 encoder 索引、零 encoder 层。

SPP 保留手册范围内（不大于 8）的核；超过 8 的奇数核分解为 5x5 池化串联，并在余数为 3 时追加 3x3 池化，均为 stride=1、对称 padding。组合后有效感受野与原核一致，保持 max 运算语义。Focus 的训练实现保持现状；仅部署 wrapper 将 Focus 输入变换为 SpaceToDepth，再对 stem 卷积权重重排，确保与训练图的 Focus 通道次序一致。这样不需更改训练检查点中 stem 的参数布局。

私有训练配置将 RTMO pose 分支设置为 `num_groups=1`、`channels_per_group=288`、`widen_factor=0.5`，因此 `out_channels=144` 且每层卷积 Group=1。该变化会改变 pose 分支参数形状，已有 pose-head 权重不直接兼容；backbone/neck 和其他兼容参数可沿用，pose 分支需重新初始化并在私有数据上微调。

### 导出契约

新增固定输入 raw-export 工具。wrapper 顺序执行 backbone（Focus 使用 SpaceToDepth 导出分支）、neck、head module，并输出每个尺度的四类 NCHW tensor。640x640 输入、17 点、256 pose 通道时，输出为：

| 输出 | stride 16 | stride 32 |
| --- | --- | --- |
| cls logits | `[1, 1, 40, 40]` | `[1, 1, 20, 20]` |
| bbox raw | `[1, 4, 40, 40]` | `[1, 4, 20, 20]` |
| keypoint visibility logits | `[1, 17, 40, 40]` | `[1, 17, 20, 20]` |
| pose feature | `[1, 256, 40, 40]` | `[1, 256, 20, 20]` |

共 8 个固定形状的四维输出。导出图不含 sigmoid、bbox/grid 解码、TopK/阈值筛选、NMS、DCC 的 bin encoding/GAU/坐标积分。外部程序根据输出顺序和 stride 执行这些步骤。

导出输入是已按配置完成颜色顺序/归一化的 float32 NCHW tensor；该模型配置 mean=0、std=1。导出不设置 dynamic axes，batch 与空间维度固定。

### 静态审计实现与本地 smoke test

审计器按手册 5.2 的 ONNX 列分类算子：v0.5 算子作为支持项；`Mul`、`Pad`、`ConvTranspose` 等标注为 v0.9-RC 的算子单独标记为版本受限；`Round` 标注为 opset 11；`Constant`、`Identity`、`Shape` 单独作为结构节点报告。其他未知算子、非标准 domain、NMS/DCC/attention 指示节点都会导致审计失败。审计同时检查 float32 输入输出、固定 shape/opset、Conv 分组和 MaxPool 的对称 padding、核大小限制。

本地随机权重 smoke test 已成功导出完整 640x640 图。初始导出图含 66 个 PyTorch exporter 为共享参数生成的 Identity 直通节点；导出器现会重连消费者并移除这些无副作用节点，审计也会拒绝包含 Identity 的图。清理后算子计数为 `Conv=80`、`LeakyRelu=67`、`MaxPool=6`、`Upsample=2`、`Slice=4`、`Concat=9`、`Add=11`、`SpaceToDepth=1`，另有 `Constant=2`；没有未列出算子、版本受限算子、非标准 domain、违规 Conv 分组或后处理节点。输入为 `[1,3,640,640]`，八个输出 shape 符合上表。清理后的随机图在 CPU ONNX Runtime 对照中相对 PyTorch wrapper 的最大绝对误差约 `2.1e-3`；SpaceToDepth + stem Conv 也单独通过 Focus 数值对照。

该 smoke test 使用随机初始化模型，不代表私有训练 checkpoint 的导出结果或精度。本轮按用户要求只验收 ONNX 静态图及本地数值语义，不执行 NPUC 编译。

### 算子审计与验收

导出工具需检查模型输入输出 shape、opset、节点类型、Conv group、输出名和是否包含已移出图的 NMS/DCC 专用节点。审计报告将把手册第 5.2 节列出的计算算子与图结构节点区分开；任何未列出的实际算子都报告为未满足，不能用“图可被 ONNX 加载”替代 OVM 支持结论。

本轮验收范围包括两项：

1. 单元测试：空 attention 配置、SPP 池化等价、Focus/SpaceToDepth 数值一致、raw 输出的固定形状。
2. ONNX 验证：opset 9、固定 batch/input、raw 输出形状、ONNX checker 通过、算子审计不含未解释项，并用本地 ONNX Runtime 对照 raw 输出数值。

## 非目标

- 不训练模型、不承诺 AP；需要用户在该结构上训练或微调。
- 不实现板端 C/C++ 后处理和 DCC 数值内核；本次只定义 raw 张量契约并保证这些解码节点不进入网络图。
- 不改动用户已有数据转换、推理脚本和数据文件。
- 不把手册静态算子审计表述为 NPUC 或板端编译结果。

## 主要风险

- ONNX opset 9 的 PyTorch 导出器可能将固定形状操作仍表示成表外节点；导出后审计是硬门槛。
- SpaceToDepth 通道顺序和 stem 权重重排必须经数值等价测试。
- Group=1 会增加 pose branch 的 MAC 和权重规模，需在目标硬件上测量性能/资源。
- attention 关闭和 group conv 改动会影响精度，需重新训练/微调和评估。
- opset 9 的 PyTorch 导出器会提示 Upsample 通用兼容性警告；当前固定 nearest x2 图已通过本机 ONNX Runtime 数值对照。
- 随机初始化 smoke test 只验证结构和数值路径，不代表训练 checkpoint 的精度或最终模型表现。
