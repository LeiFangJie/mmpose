# RTMPose 私有数据训练、导出与部署说明

本文档覆盖本分支的完整 RTMPose 流程：COCO 私有数据、训练配置、训练/评估、RTMDet 人体框衔接、RTMPose 前处理、7x7 Conv raw ONNX、板外 RTMCC/SimCC 后处理和数值验证。

推理链路为：

```text
BGR 图像 -> RTMDet 前处理 -> RTMDet raw ONNX -> 框解码/NMS/person 筛选
          -> RTMPose bbox crop -> RTMPose raw ONNX -> NPZ head 续算
          -> SimCC 解码 -> 原图坐标关键点
```

RTMDet 只提供人体框，不参与 RTMPose 训练。OVM/NPUC 转换和板端运行不在本阶段范围内。

## 1. 输出契约

### RTMPose

训练时保留完整 RTMPose：CSPNeXt、7x7 `RTMCCHead.final_layer`、flatten、ScaleNorm、Linear、GAU、两路 SimCC 和原始 SimCC 损失。

导出时只保留 backbone 和 `final_layer`：

| 项目 | 契约 |
| --- | --- |
| 输入 | `float32 [1,3,256,192]`，NCHW |
| 输出 | `head_feature: float32 [1,17,8,6]` |
| ONNX opset | 9 |
| batch | 固定为 1，无动态轴 |
| 图外计算 | RTMCC/SimCC head、解码、坐标还原 |

7x7 Conv 按算子文档可以保留；大于 5x5 是性能提示，不是本阶段的不支持算子。

### RTMDet

输入固定为 `float32 [1,3,640,640]`，输出 raw 多尺度分类和框回归：

```text
raw_output_0: [1,80,80,80]    raw_output_1: [1,4,80,80]
raw_output_2: [1,80,40,40]    raw_output_3: [1,4,40,40]
raw_output_4: [1,80,20,20]    raw_output_5: [1,4,20,20]
```

检测 ONNX 不包含预处理、框解码、置信度筛选或 NMS。

## 2. 运行方式

所有命令从仓库根目录执行：

```text
E:/OMS/RTMpose/mmpose
```

不需要设置 PowerShell 变量，也不需要先激活环境，直接调用专用解释器：

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe --version
```

该解释器应包含 PyTorch、MMPose、MMDetection、MMCV、MMEngine、NumPy、OpenCV、ONNX 和 ONNX Runtime。不要把依赖安装到系统 Python。

预训练文件：

```text
E:/OMS/RTMpose/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth
E:/OMS/RTMpose/rtmdet_tiny_8xb32-300e_humanart-7da5554e.pth
```

## 3. 数据格式

目录结构：

```text
datasets/
  coco_annotations/train.json
  coco_annotations/val.json
  coco_annotations/test.json
  images/train/
  images/val/
  images/test/
```

配置使用 `data_root='datasets/'`、`data_prefix=dict(img='')`，所以 JSON 的 `images[].file_name` 必须相对于 `datasets/`，例如：

```json
"file_name": "images/train/frame_2026-08-20_16-00-58_000000.jpg"
```

COCO 标注要求：

- 只有 `person` 类，`category_id=1`；
- `bbox` 为 `[x,y,width,height]`；
- `keypoints` 为 17 个关键点，每个点为 `[x,y,visibility]`，总长度 51；
- `num_keypoints` 等于 visibility 大于 0 的点数；
- 关键点顺序为：

```text
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
```

元信息位于 [oms_private.py](../configs/_base_/datasets/oms_private.py)，包含 `keypoint_info`、`skeleton_info`、`joint_weights` 和 `sigmas`。改变关键点数量或顺序时，必须同步修改 JSON、元信息、模型输出通道和后处理 shape。

## 4. 配置与模型改造

训练配置为 [custom_rtmpose-t_8xb32-420e_private-256x192.py](../configs/body_2d_keypoint/rtmpose/custom/custom_rtmpose-t_8xb32-420e_private-256x192.py)。该文件是自包含配置，不依赖 `_base_` 合并；官方 RTMPose-T 256x192 的训练超参数、优化器、学习率调度、codec、数据增强 pipeline、stage2 pipeline、runtime hooks 和 evaluator 都已直接写入文件。私有改动主要是：

- 普通 `BN`，不使用 `SyncBN`；
- backbone 激活函数为 `ReLU`；
- 关闭 Channel Attention；
- 最后 stage 的 SPP MaxPool pyramid 替换为连续 5x5 depthwise Conv pyramid；
- 保留原生 RTMCCHead、SimCCLabel 和损失；
- 输入尺寸为 `(192,256)`，关键点数为 17；
- 测试配置 `flip_test=False`。

卷积 SPP pyramid 是新增结构，原始 checkpoint 没有对应参数。加载时出现 pyramid 参数缺失是预期现象，必须通过私有数据微调后再作为最终模型。

## 5. 训练与评估

直接启动训练：

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe tools/train.py configs/body_2d_keypoint/rtmpose/custom/custom_rtmpose-t_8xb32-420e_private-256x192.py --work-dir work_dirs/custom_rtmpose_t_private
```

恢复训练：

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe tools/train.py configs/body_2d_keypoint/rtmpose/custom/custom_rtmpose-t_8xb32-420e_private-256x192.py --work-dir work_dirs/custom_rtmpose_t_private --resume
```

评估指定 checkpoint：

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe tools/test.py configs/body_2d_keypoint/rtmpose/custom/custom_rtmpose-t_8xb32-420e_private-256x192.py work_dirs/custom_rtmpose_t_private/best_coco_AP.pth
```

用于部署的 checkpoint 必须包含 backbone、7x7 final layer 和完整 RTMCC/SimCC head。

## 6. ONNX 导出

### 6.1 RTMPose

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe tools/deploy/rtmpose/custom_export_raw_feature_onnx.py --config configs/body_2d_keypoint/rtmpose/custom/custom_rtmpose-t_8xb32-420e_private-256x192.py --checkpoint work_dirs/custom_rtmpose_t_private/best_coco_AP.pth --output RTMPose/custom_rtmpose_t_raw_feature_opset9.onnx --weights-output RTMPose/custom_rtmpose_postprocess_weights.npz
```

输出：

```text
RTMPose/custom_rtmpose_t_raw_feature_opset9.onnx
RTMPose/custom_rtmpose_postprocess_weights.npz
```

导出器固定 batch=1、opset=9、输入 `[1,3,256,192]`，只导出 backbone + 7x7 final layer，保存单个自包含 ONNX，并删除 Identity 节点。NPZ 包含 MLP、GAU、`cls_x` 和 `cls_y` 权重，必须与 ONNX 来自同一个 checkpoint。

### 6.2 RTMDet

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe tools/deploy/rtmdet/custom_export_raw_onnx.py --checkpoint E:/OMS/RTMpose/rtmdet_tiny_8xb32-300e_humanart-7da5554e.pth --output RTMDet/custom_rtmdet_tiny_humanart_raw_opset9.onnx
```

输出：

```text
RTMDet/custom_rtmdet_tiny_humanart_raw_opset9.onnx
```

## 7. 独立前处理与后处理

部署代码不导入 MMPose、MMDetection、MMCV、MMEngine 或 PyTorch，只依赖 NumPy 和 OpenCV；ONNX Runtime 负责执行 ONNX 图。

### 7.1 RTMDet

文件：

- [custom_rtmdet_preprocess.py](../RTMDet/custom_rtmdet_preprocess.py)
- [custom_rtmdet_postprocess.py](../RTMDet/custom_rtmdet_postprocess.py)

```python
import cv2
import onnxruntime as ort
from RTMDet.custom_rtmdet_preprocess import preprocess_detector_image
from RTMDet.custom_rtmdet_postprocess import decode_detector_outputs, select_person_boxes

image_bgr = cv2.imread('image.jpg')
det_input, det_meta = preprocess_detector_image(image_bgr)
session = ort.InferenceSession('RTMDet/custom_rtmdet_tiny_humanart_raw_opset9.onnx', providers=['CPUExecutionProvider'])
raw_outputs = session.run(None, {session.get_inputs()[0].name: det_input})
boxes, scores, labels = decode_detector_outputs(raw_outputs, det_meta)
person_boxes, person_scores = select_person_boxes(boxes, scores, labels)
```

前处理复现等比例缩放、640 补边 114、BGR mean/std 归一化，输出 `[1,3,640,640]`。后处理复现 stride `[8,16,32]` 格点、DistancePointBBoxCoder、sigmoid、置信度筛选、逐类 NMS、top-300 和原图坐标还原。`person_boxes` 是原图坐标 `[x1,y1,x2,y2]`，可直接传给 RTMPose；多个人体需要逐框推理。

### 7.2 RTMPose

文件：

- [custom_rtmpose_preprocess.py](custom_rtmpose_preprocess.py)
- [custom_rtmpose_postprocess.py](custom_rtmpose_postprocess.py)

```python
import onnxruntime as ort
from RTMPose.custom_rtmpose_preprocess import preprocess_pose_image
from RTMPose.custom_rtmpose_postprocess import decode_simcc, load_postprocess_weights, simcc_from_feature

pose_input, center, scale = preprocess_pose_image(image_bgr, person_boxes[0])
session = ort.InferenceSession('RTMPose/custom_rtmpose_t_raw_feature_opset9.onnx', providers=['CPUExecutionProvider'])
feature = session.run(None, {session.get_inputs()[0].name: pose_input})[0]
weights = load_postprocess_weights('RTMPose/custom_rtmpose_postprocess_weights.npz')
simcc_x, simcc_y = simcc_from_feature(feature, weights)
keypoints, keypoint_scores = decode_simcc(simcc_x, simcc_y, center, scale)
```

前处理复现 `GetBBoxCenterScale(padding=1.25)`、宽高比调整、`TopdownAffine(use_udp=False)`、BGR 转 RGB 和 mean/std 归一化，输出 `[1,3,256,192]`。

后处理顺序为：

```text
[1,17,8,6] -> flatten -> ScaleNorm/Linear -> SiLU/GAU
-> cls_x/cls_y -> SimCC argmax/score -> center/scale 坐标还原
```

返回 `keypoints` 为原图坐标 `[1,17,2]`，`keypoint_scores` 为 `[1,17]`。GAU shortcut 使用源码一致的 `res_scale(tokens) + main_branch`。

## 8. 验证

前后处理单元测试：

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe -m unittest discover -s tests/test_tools -p "custom_test_deployment_prepost.py" -v
```

backbone 结构测试：

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe -m unittest discover -s tests/test_models/test_backbones -p "custom_test_cspnext.py" -v
```

完整框架/ONNX 对照：

```powershell
C:/Users/admin/miniforge3/envs/rtmpose/python.exe tools/deploy/custom_compare_framework_onnx_pipeline.py --image datasets/images/train/frame_2026-08-20_16-00-58_000000.jpg --det-checkpoint E:/OMS/RTMpose/rtmdet_tiny_8xb32-300e_humanart-7da5554e.pth --pose-checkpoint E:/OMS/RTMpose/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth
```

参考路径使用 MMPose/MMDetection 原生 pipeline、模型、RTMCC head 和 decoder；部署路径使用 NumPy/OpenCV 前后处理、ONNX Runtime 和 NPZ head，两条路径不共用被测实现。报告写入 `RTMPose/custom_framework_onnx_prepost_compare.json`。

当前预训练 checkpoint 的一次验证结果：

```text
RTMDet person 框数量：框架 300，部署路径 300
top person 框最大误差：1.2207e-4
RTMPose raw feature 最大误差：0.00342
SimCC X/Y 最大误差：0.01221 / 0.01147
最终关键点最大误差：1.2207e-4
```

导出器和对照脚本固定 `torch.manual_seed(0)`，用于保证未微调的卷积 SPP pyramid 初始化一致。训练完成后必须重新导出 ONNX/NPZ 并重新验证。

## 9. 常见问题

### ONNX 输出为什么不是坐标？

ONNX 在 7x7 Conv 后截断，输出 `[1,17,8,6]`。flatten、GAU、SimCC 和坐标解码在图外执行，所以部署时必须同时提供 NPZ head 权重和后处理代码。

### ONNX 和 NPZ 为什么必须同 checkpoint？

ONNX 使用 backbone/final layer 参数，NPZ 使用 MLP/GAU/SimCC 参数。跨 checkpoint 混用会导致特征与 head 不匹配。

### missing keys 是否错误？

新增卷积 SPP 没有预训练参数，缺失对应卷积和 BN 参数是预期现象；必须微调后再评估精度。

### 为什么用 BN？

当前配置面向单机私有数据训练，使用普通 BN；RTMDet 仍使用其 checkpoint 对应的原始配置。

## 10. 本阶段边界

已覆盖数据接口、训练配置、RTMDet 检测衔接、RTMPose 前处理、raw ONNX、NPZ head、SimCC 解码和 PyTorch/ONNX 数值验证。

未覆盖：

- NPUC/OVM 编译和算子模拟；
- 板端运行、延迟、内存和功耗；
- 私有数据训练后的最终精度；
- RTMDet 训练；
- 将前处理、检测解码或姿态后处理塞回 ONNX 图。

只有在私有数据微调、重新导出并完成端到端对照后，新的 checkpoint 才能作为部署版本使用。


