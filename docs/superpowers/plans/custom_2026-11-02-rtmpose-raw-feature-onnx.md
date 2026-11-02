# RTMPose Raw Feature ONNX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在私有数据上完整微调 RTMPose-T，并生成截断在 7x7 head Conv 的 opset 9 ONNX；另将现有 RTMDet-Tiny checkpoint 导出为独立 raw-head ONNX。

**Architecture:** 新增兼容 RTMPose-T checkpoint 命名的 CustomCSPNeXt，关闭 ChannelAttention、使用 ReLU 并以可训练 depthwise Conv pyramid 替换 MaxPool SPP；训练仍保留原生 RTMCCHead。RTMPose 导出 wrapper 在 7x7 `final_layer` 后截断图。RTMDet-Tiny 使用独立 exporter 输出 raw 多尺度 head tensors。两套图分别审计并以 ORT 对比 PyTorch。

**Tech Stack:** Python 3.10、PyTorch 2.14、MMEngine、MMDetection 3.3、MMPose 1.3.2、ONNX 1.17、ONNX Runtime 1.23。

**Spec:** `docs/superpowers/specs/custom_2026-11-02-rtmpose-raw-feature-onnx-design.md`

## Global Constraints

- 私有关键点为 17 个 COCO 顺序关键点；模型输入为宽高 `(192,256)`。
- ONNX 固定 batch=1、float32、opset=9、输入 `[1,3,256,192]`、输出 `[1,17,8,6]`。
- 7x7 Conv 保留并按性能次优提示报告，不得列为不支持算子。
- 完整 RTMCC/SimCC head 和原 loss 仍参与训练；ONNX 在 head 的 `final_layer` 输出处截断。
- 新增文件使用 `custom_` 前缀；不修改共享 MMPose 源码/默认配置。
- 使用 `C:\Users\admin\miniforge3\envs\rtmpose\python.exe`；不安装全局依赖。
- 不执行 NPUC、OVM 转换或板端测试。
- RTMDet 使用输入 `[1,3,640,640]`，仅导出 raw multi-level head outputs，不含 decode/NMS。

---

### Task 1: Custom CSPNeXt 与卷积 SPP

**Files:**
- Create: `mmpose/models/backbones/custom_cspnext.py`
- Test: `tests/test_models/test_backbones/custom_test_cspnext.py`

**Interfaces:**
- Consumes: MMPose `CSPNeXt`, `CSPLayer`, `ConvModule`, `DepthwiseSeparableConvModule`。
- Produces: 注册模块 `CustomCSPNeXt`，接收标准 CSPNeXt 参数及 `spp_kernel_sizes`，返回与 RTMPose-T 兼容的末级 `[B,384,8,6]` 特征。

- [ ] **Step 1: Add failing backbone tests**

测试以 RTMPose-T 参数构建模型并断言：最后 stage 不含 MaxPool、无 ChannelAttention、Conv 激活均为 ReLU、卷积 SPP 输出 shape 保持不变、depthwise pyramid 的卷积 groups 等于通道数，且 checkpoint 中主干/PPM 公共参数名称 shape 兼容。

- [ ] **Step 2: Run tests to confirm expected failure**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_models.test_backbones.custom_test_cspnext -v`
Expected: FAIL because `CustomCSPNeXt` is not registered.

- [ ] **Step 3: Implement minimal custom backbone**

继承现有 CSPNeXt 建立原骨干后，以相同 `conv1`/`conv2` 定义替换最后 stage SPP；分支为原输入和逐层 5x5 depthwise Conv 的一阶、二阶、三阶输出。构造时强制 `channel_attention=False`、backbone `act_cfg=ReLU`，不改 RTMCC head。

- [ ] **Step 4: Run backbone tests**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_models.test_backbones.custom_test_cspnext -v`
Expected: PASS，且训练尺寸下输出为 `[1,384,8,6]`。

### Task 2: 私有 RTMPose 完整训练配置

**Files:**
- Create: `configs/body_2d_keypoint/rtmpose/custom/custom_rtmpose-t_8xb32-420e_private-256x192.py`
- Test: `tests/test_tools/custom_test_rtmpose_config.py`

**Interfaces:**
- Consumes: Task 1 的 `CustomCSPNeXt`、17 点元信息 `configs/_base_/datasets/oms_private.py`、COCO JSON 和图片。
- Produces: 完整 `TopdownPoseEstimator`，调用原生 `RTMCCHead.forward/loss`，从本地 RTMPose-T checkpoint 加载兼容参数。

- [ ] **Step 1: Add failing config assertions**

检查模型类型、输入尺寸、17 keypoints、7x7 head、完整 SimCC decoder、私有 train/val/test JSON 和 17 点元信息路径，并确保 checkpoint 路径解析到用户给出的 RTMPose-T 权重。

- [ ] **Step 2: Run config test to confirm expected failure**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_tools.custom_test_rtmpose_config -v`
Expected: FAIL because the private RTMPose config does not exist.

- [ ] **Step 3: Add isolated private config**

复制官方 RTMPose-T COCO 256x192 配置到 `custom/`，保留 SimCC、loss、augmentation 和 pipeline switch；用自定义 backbone、ReLU、`datasets/` root、`coco_annotations/{train,val,test}.json`、`data_prefix=dict(img='')`、oms 元信息以及本地 checkpoint 覆盖配置。关闭 flip test 只影响推理设置，不改变训练目标。

- [ ] **Step 4: Run config and dataset smoke tests**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_tools.custom_test_rtmpose_config -v`
Expected: PASS；配置构建和一个样本 pipeline 可执行，训练模型 forward 返回 `(simcc_x, simcc_y)`，loss 可反向传播。

### Task 3: Raw 7x7 Conv ONNX 导出与审计

**Files:**
- Create: `tools/deploy/rtmpose/custom_export_raw_feature_onnx.py`
- Create: `tools/deploy/rtmpose/custom_onnx_audit.py`
- Test: `tests/test_tools/custom_test_rtmpose_raw_feature_export.py`

**Interfaces:**
- Consumes: 完整训练配置和 RTMPose checkpoint。
- Produces: ONNX `images -> head_feature`，固定 shape `[1,3,256,192] -> [1,17,8,6]`，opset 9；输出保存到 `RTMPose/`。

- [ ] **Step 1: Write failing wrapper/export/audit tests**

覆盖 wrapper 输出与 PyTorch `head.final_layer(backbone(images))` 一致、ONNX opset/shape/dtype 固定、图中不存在 RTMCC 后续节点，并在算子审计中将 7x7 Conv 记为性能提示而非不支持。

- [ ] **Step 2: Run export tests to confirm expected failure**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_tools.custom_test_rtmpose_raw_feature_export -v`
Expected: FAIL because the RTMPose raw-feature exporter is not available.

- [ ] **Step 3: Implement fixed-shape exporter and audit**

Wrapper 运行 backbone 和 `head.final_layer`；导出固定 opset 9 且无 dynamic axes。审计 ONNX checker、接口、维度、算子支持表、Conv group 和 kernel size；kernel 大于 5 只输出性能 warning。

- [ ] **Step 4: Run export tests and ONNX Runtime parity**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_tools.custom_test_rtmpose_raw_feature_export -v`
Expected: PASS；ORT 和 PyTorch wrapper 输出满足设定误差，输出 shape 为 `[1,17,8,6]`。

### Task 4: RTMDet-Tiny Raw ONNX 导出

**Files:**
- Create: `tools/deploy/rtmdet/custom_export_raw_onnx.py`
- Create: `tools/deploy/rtmdet/custom_onnx_audit.py`
- Test: `tests/test_tools/custom_test_rtmdet_raw_export.py`

**Interfaces:**
- Consumes: `projects/rtmpose/rtmdet/person/rtmdet_tiny_8xb32-300e_humanart.py` 與 `E:/OMS/RTMpose/rtmdet_tiny_8xb32-300e_humanart-7da5554e.pth`。
- Produces: 固定 `[1,3,640,640]` 输入、三层 raw cls/bbox outputs、opset 9 ONNX；不含 decode、top-k 或 NMS。

- [ ] **Step 1: Write failing RTMDet exporter tests**

覆盖 exporter wrapper 输出与 `model.forward(..., mode='tensor')` 的 head tensors 一致，静态 shape 固定、没有 NMS/TopK/Gather 解码节点，opset 为 9。

- [ ] **Step 2: Run tests to confirm expected failure**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_tools.custom_test_rtmdet_raw_export -v`
Expected: FAIL because the custom RTMDet raw exporter does not exist.

- [ ] **Step 3: Implement standalone RTMDet raw exporter and audit**

读取模型自带 `bbox_head` 输出，建立不调用 `predict()` 的 wrapper；使用 RTMDet-Tiny 官方配置和 checkpoint 导出固定 opset 9 图，并检查 ONNX checker、输入、输出 shape/dtype 和算子统计。

- [ ] **Step 4: Run PyTorch/ORT parity smoke**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_tools.custom_test_rtmdet_raw_export -v`
Expected: PASS；ORT 各 raw output 与 PyTorch 对齐，NMS/decode 留在图外。

### Task 5: 使用说明与全量验证

**Files:**
- Create: `RTMPose/custom_RTMPOSE_RAW_FEATURE_ONNX_USAGE.md`

- [ ] **Step 1: Document train/eval/export commands**

记录 RTMPose 训练、恢复训练、评估、导出与 ORT smoke 命令，以及 RTMDet-Tiny ONNX 独立导出命令；说明两模型输出边界均为 raw tensors，decode/NMS 留在图外，RTMCC 后续权重必须与 RTMPose ONNX 来自同一 checkpoint。

- [ ] **Step 2: Run focused and repository checks**

Run: `C:\Users\admin\miniforge3\envs\rtmpose\python.exe -m unittest tests.test_models.test_backbones.custom_test_cspnext tests.test_tools.custom_test_rtmpose_config tests.test_tools.custom_test_rtmpose_raw_feature_export tests.test_tools.custom_test_rtmdet_raw_export -v`
Expected: all focused tests pass；再以本地 pose 和 detector checkpoints 分别执行真实固定输入 ONNX/ORT smoke 与静态审计。

- [ ] **Step 3: Review output artifacts and scope**

确认 pose ONNX 位于 `RTMPose/`、detector ONNX 位于 `RTMDet/`，各自 shape/opset 正确，Identity 无残留，pose 7x7 Conv 提示被正确分类，并报告没有执行 OVM/板端验证。
