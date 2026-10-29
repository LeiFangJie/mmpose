# RTMO 板端前后处理

## 目标与边界

本模块衔接 `custom_rtmo_private_raw_opset9.onnx` 的固定八输出接口。板端运行代码只依赖 NumPy 和 OpenCV；开发机验证脚本额外依赖 PyTorch、MMPose 和 ONNX Runtime。DCC 解码参数必须从部署所用 checkpoint 导出，不能跨 checkpoint 复用。

## 源码对应关系

- 前处理对应 `BottomupResize(input_size=(640,640), pad_val=(114,114,114))`：保持纵横比、中心仿射填充。
- 数据预处理器没有启用 BGR/RGB 转换、`/255` 或 mean/std 归一化。因此输入为 BGR、像素量纲 0..255，输出 `[1,3,640,640]` float32。
- 网格对应 `MlvlPointGenerator(centralize_points=True, strides=[16,32])`，中心位置为 `(index + 0.5) * stride - 0.5`。
- bbox 解码对应 `YOLOXPoseHead.decode_bbox`；关键点对应 `DCC.forward_test`，含部署态 FC+BN 融合、GAU、位置编码并入 beta、残差缩放、动态 bins、softmax 积分。
- 候选筛选、IoU NMS、bbox 网络边界裁剪和原图坐标变换对应 `RTMOHead.predict` 与 `BottomupPoseEstimator.add_pred_to_datasample`。

## 接口

```python
import cv2
import onnxruntime as ort

from RTMO.custom_rtmo_preprocess_postprocess import (
    RTMOPostProcessor,
    RTMOPreProcessor,
    load_dcc_parameters,
)

image = cv2.imread('input.jpg')
input_tensor, metadata = RTMOPreProcessor().process(image)
session = ort.InferenceSession('custom_rtmo_private_raw_opset9.onnx')
raw_values = session.run(None, {'images': input_tensor})
output_names = [output.name for output in session.get_outputs()]
raw_outputs = dict(zip(output_names, raw_values))
processor = RTMOPostProcessor(load_dcc_parameters('custom_rtmo_dcc_params.npz'))
result = processor.process(raw_outputs, metadata)
```

`result` 包含原图坐标系的 `bboxes [N,4]`、`scores [N]`、`keypoints [N,17,2]`、`keypoint_scores [N,17]`。如板端运行时不提供 ONNX Runtime，可直接把其八个输出数组按输出名传给 `process`。

## 导出参数与冒烟验证

```powershell
C:\Users\admin\miniforge3\envs\rtmpose\python.exe RTMO/custom_rtmo_smoke_test.py `
  --config configs/body_2d_keypoint/rtmo/custom/custom_rtmo-s_8xb32-600e_private-640x640.py `
  --checkpoint <训练checkpoint> `
  --onnx RTMO/custom_rtmo_private_raw_opset9.onnx `
  --image <测试图像>
```

脚本从传入 checkpoint 导出 `RTMO/custom_rtmo_dcc_params.npz`，检查八个 ONNX raw tensor 与 PyTorch wrapper 的误差，并单独对照 NumPy DCC 和源码部署态 DCC。默认产物和 JSON 报告均写入 `RTMO/`。当前提交内的随机 smoke checkpoint 仅适合验证图和代码路径，不能代表训练精度；请在正式部署时使用训练后的 checkpoint 和由它导出的 ONNX。
