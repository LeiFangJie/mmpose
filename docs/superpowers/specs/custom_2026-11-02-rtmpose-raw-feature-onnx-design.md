# RTMPose 私有数据 Raw Feature ONNX 设计

## 背景

本分支用于在现有 MMPose/RTMPose 工程中微调 RTMPose-T，并导出适配受限部署算子的 ONNX。预训练姿态 checkpoint 为 `E:/OMS/RTMpose/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth`。私有 COCO 标注位于 `datasets/coco_annotations/`，图片相对路径位于 `datasets/images/`，使用 17 个 COCO 顺序关键点。

部署边界按已确认方案定义：训练时仍运行完整 RTMPose，包括 7x7 `final_layer`、展平、ScaleNorm、MLP、GAU、SimCC 两路分类和原 SimCC 损失；ONNX 导出仅包含 backbone 和 7x7 `final_layer`，外部模块从同一训练 checkpoint 获取其余 head 权重并执行后续计算与解码。ONNX 输出是特征图，不是坐标。

## 目标与接口

- 训练网络输入尺寸为 `(W,H)=(192,256)`，ONNX NCHW 输入固定为 `[1,3,256,192]`。
- ONNX 输出唯一张量 `head_feature`，float32，固定 shape `[1,17,8,6]`。
- ONNX opset 固定为 9，batch 固定为 1，不导出动态轴。
- 继续使用标准 `RTMCCHead` 的完整 forward/loss，训练优化不截断。
- 部署图使用自定义 CSPNeXt：关闭 ChannelAttention、backbone 激活改为 ReLU、SPP MaxPool pyramid 改为 5x5 depthwise convolution pyramid。
- SPP 卷积替代不是 MaxPool 的数值等价替代；新增卷积参数随机初始化，模型必须在私有数据上微调。
- 7x7 `Conv` 是手册第 5.2 节列出的 ONNX 算子；手册将大于 5x5 的 kernel 标为次优表现，因此保留该层但在审计中作为性能提示，不把它判为算子不支持。
- 预处理、检测、仿射 crop、RTMCC 后续计算、SimCC decode 和坐标还原均不放入 RTMPose ONNX 图。另行使用用户提供的 RTMDet-Tiny HumanArt checkpoint 导出独立 640x640 raw-head ONNX，供检测推理链路使用；RTMDet 不参与姿态训练。

## 模型适配

新增 `CustomCSPNeXt` 继承 MMDetection/MMPose CSPNeXt 的层级和权重命名，保持主干结构以兼容已有预训练权重。构造原有 CSPNeXt 后，将最后一个 stage 的 SPP 模块替换为卷积 pyramid 实现：沿用原 SPP 的 `conv1`、`conv2` 参数名和张量 shape，在四个 concat 分支中使用输入及 5x5 depthwise Conv 串联 1、2、3 次的结果。构建 backbone 时传入 `channel_attention=False` 和 ReLU 激活。这样可复用兼容 checkpoint 参数；被移除的通道注意力权重忽略，新 pyramid depthwise 权重重新初始化。

定制训练配置复制官方 RTMPose-T 256x192 COCO 配置的优化器、pipeline、SimCC 编码和完整 RTMCC head，改为私有 dataset root、标注、元信息以及 `CustomCSPNeXt`。从本地 RTMPose-T checkpoint 加载兼容权重，保留原 head 的 17 点/7x7/SimCC 形状。

## ONNX 导出

自定义 wrapper 显式计算 `model.backbone(images)`，并返回 `model.head.final_layer(features[-1])`。不调用 `model.predict()`、head decoder 或完整 SimCC forward。导出器固定输入输出 shape、dtype 和 opset，并保存到分支内的 `RTMPose/` 目录。独立审计检查 ONNX checker、opset、静态接口、算子白名单、Conv group 和 7x7 Conv 提示。

smoke test 使用同一 checkpoint 构建训练模型和 wrapper，比较 PyTorch wrapper 与 ONNX Runtime 输出；同时验证训练模型完整 head 仍可产生 `(simcc_x, simcc_y)`，且 ONNX 输出逐项等于 PyTorch 的 `final_layer(backbone(images))`。该测试只证明结构和数值等价，不代表私有训练精度或目标 OVM 编译成功。

RTMDet 使用仓库 RTMDet-Tiny HumanArt 配置和用户给定 checkpoint 单独导出。导出边界为检测网络 raw multi-level head tensors，不包含 decode、score filtering 或 NMS；输入固定 shape 和 raw output shapes 由实际配置解析并在导出审计中固化。该 ONNX 与 RTMPose ONNX 相互独立，检测框到 pose crop 的前后处理属于后续推理集成。

## 独立部署前后处理

部署辅助代码不导入 MMPose、MMDetection、MMCV、MMEngine 或 PyTorch；RTMPose 与 RTMDet 各自提供清晰命名的预处理和后处理模块。运行依赖限制为 NumPy 与 OpenCV。框架对照程序单独使用 OpenMMLab 原生 pipeline/model，不调用部署辅助函数来产生参考值。

- RTMDet 前处理读取 BGR 图像，按官方 HumanArt test pipeline 的 `(640,640)` 等比例缩放和右下补边 114，再按 `DetDataPreprocessor` 的 BGR mean/std 归一化，输出静态 NCHW float32。保留 `scale_factor` 与未补边的 `img_shape` 供后处理还原。
- RTMDet 后处理接收 ONNX 输出的 3 层分类 logits 和未缩放回归 logits，分别 sigmoid 分类分数、依据 stride `[8,16,32]` 变换回归量、按 prior generator `offset=0` 生成格点、执行距离框解码、边界裁剪、置信度筛选、逐类 NMS 和全图 top-300；最后按 resize scale 还原原图坐标，并可筛出 COCO `person` 类供姿态模型使用。阈值来自实际合并配置，不能复制其他模型默认值。
- RTMPose 前处理读取 BGR 图像和 detector 原图 xyxy 框，按 `GetBBoxCenterScale(padding=1.25)` 计算 center/scale，按 `TopdownAffine(input_size=(192,256), use_udp=False)` 做宽高比调整和 OpenCV affine crop，再执行 BGR->RGB、mean/std 归一化，输出 `[1,3,256,192]` 与映射元数据。
- RTMPose ONNX 后处理将 `[1,17,8,6]` 特征按源码继续执行 flatten、ScaleNorm、无偏 Linear、RTMCC GAU、两路 SimCC Linear，使用原生 SimCC argmax/分数规则解码并按 center/scale 映射回原图。GAU、激活和矩阵运算用 NumPy 直接实现，不调用框架。它从训练 checkpoint 单独导出 `custom_rtmpose_postprocess_weights.npz`；该参数包与自包含的单文件 ONNX 是两个职责不同的工件，不能跨 checkpoint 混用。

## 端到端对照

`custom_compare_framework_onnx_pipeline.py` 分开执行两条完整路径。参考路径使用 OpenMMLab 原生 RTMDet test pipeline、检测 `test_step`、MMPose `inference_topdown`、原生 RTMCC head 与 codec。部署路径只使用独立 OpenCV/NumPy 前后处理以及 ONNX Runtime。对照同一输入图像，分别报告 detector 预处理张量误差、框/分数/类别差异、pose crop 误差、raw feature 误差、SimCC 输出误差和最终关键点坐标差；不会让两条路径共用被测的前后处理实现。烟测输出需带 checkpoint、ONNX 和输入图片标识。

## 归一化层

私有 RTMPose 配置使用普通 BN。此前沿用官方多人多卡配置的 SyncBN 是复制配置时未针对当前单机私有微调场景调整；BN 与 SyncBN 的 state_dict 参数名/形状兼容，可从预训练 checkpoint 载入 running statistics。RTMDet 的配置和权重保持官方 checkpoint 定义，不因姿态模型的 BN 决策而改变。

## 不在本阶段范围

- NPUC、OVM 转换、模拟器或板端验证。
- RTMDet 训练；仅导出独立 raw-head ONNX。
- ONNX 中的目标检测、人体框选择、Affine 变换、RTMCC 后续 head 和 SimCC 解码。
- 修改 MMPose 原生 RTMO/RTMPose 源码或共享默认配置。

## 文件边界

- 新建 `mmpose/models/backbones/custom_cspnext.py`：注册定制 CSPNeXt 和卷积 SPP。
- 新建 `configs/body_2d_keypoint/rtmpose/custom/custom_rtmpose-t_8xb32-420e_private-256x192.py`：私有数据完整训练配置。
- 新建 `tools/deploy/rtmpose/custom_export_raw_feature_onnx.py`：固定接口导出器。
- 新建 `tools/deploy/rtmpose/custom_onnx_audit.py`：静态图契约审计。
- 新建 `tools/deploy/rtmdet/custom_export_raw_onnx.py`：RTMDet-Tiny raw-head ONNX 导出。
- 新建 `tools/deploy/rtmdet/custom_onnx_audit.py`：RTMDet ONNX 静态审计。
- 新建 `RTMPose/custom_rtmpose_preprocess.py`、`RTMPose/custom_rtmpose_postprocess.py`：纯 NumPy/OpenCV 姿态前后处理。
- 新建 `RTMDet/custom_rtmdet_preprocess.py`、`RTMDet/custom_rtmdet_postprocess.py`：纯 NumPy/OpenCV 检测前后处理。
- 更新导出器：姿态 ONNX 保存成单个自包含文件，并导出板外 RTMCC/SimCC 权重 NPZ。
- 新建 `tools/deploy/custom_compare_framework_onnx_pipeline.py`：框架路径与独立部署路径端到端数值对照。
- 新建 `tests/test_models/test_backbones/custom_test_cspnext.py` 与 `tests/test_tools/custom_test_rtmpose_raw_feature_export.py`：SPP/注意力/激活、输出契约、图审计和 PyTorch/ONNX 数值对照。
- 新建 `RTMPose/custom_RTMPOSE_RAW_FEATURE_ONNX_USAGE.md`：训练、评估、两个模型导出和验证命令。

## 风险和验证边界

卷积 pyramid 改变了 backbone 表达，原 checkpoint 只能提供兼容参数的初始化，精度需要私有训练评估确认。7x7 Conv 的 ONNX 表达与算子支持不等于目标硬件上的最优性能或编译成功。外部 head 必须和 ONNX 使用同一 checkpoint，且必须严格复现 RTMCC/SimCC 源码顺序。RTMDet raw-head ONNX 仍需由后续推理集成完成解码和 NMS。本阶段静态 ONNX 和 ONNX Runtime smoke 不替代 OVM 工程的导入、性能和板端测试。
