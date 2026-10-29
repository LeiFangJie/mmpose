# RTMO 前后处理实施计划

## 目标

为 `RTMO/custom_rtmo_private_raw_opset9.onnx` 提供可用于后续板端衔接的源码一致前处理和后处理。板端运行代码仅依赖 NumPy、OpenCV；开发机冒烟工具可使用 PyTorch、MMPose、ONNX Runtime 做数值核验。

## 源码契约

- `BottomupResize`：640x640 fit、保持宽高比、仿射居中、pad 值 114。
- `PoseDataPreprocessor`：不做通道变换、缩放或归一化，ONNX 输入为 BGR float32，数值范围仍为 0..255。
- RTMO head：stride 16/32 网格，分类和可见性使用 sigmoid；bbox 为中心偏移及 log 宽高；bbox 解码后以 1.25 padding 形成 DCC 坐标范围。
- DCC：使用训练权重执行 BN 融合后的 pose 投影、GAU、动态 sin/cos bins、x/y 矩阵乘、softmax 积分解码。
- 推理后处理：按 `score_thr` 筛选，按 `nms_thr` 执行 IoU NMS，bbox clamp 到网络输入范围，再根据 Bottomup 元数据映射回原图。

## 实施步骤

1. 新增预处理和后处理测试，覆盖 letterbox 几何/像素、输出 shape/dtype、bbox 解码、DCC 参数化接口、筛选/NMS、坐标恢复和异常输入。
2. 新增纯 NumPy/OpenCV 运行模块及从已构建 deploy 模型导出 DCC 参数的函数。
3. 新增命令行冒烟工具：用同一 checkpoint、同一预处理输入分别运行 PyTorch raw wrapper 和 ONNX Runtime，对照八个 raw tensor；并对照源码 `model.predict` 和 NumPy 图外解码结果。
4. 执行单元测试、现有 ONNX 审计和随机 smoke checkpoint 冒烟，记录误差和产物位置。

## 验收条件

- 板端模块导入不依赖 PyTorch、MMCV、MMEngine、MMPose 或 ONNX Runtime。
- 前处理输出严格为 `[1,3,640,640]` float32，颜色顺序和像素量纲符合配置。
- 八个 raw 输出逐项与 PyTorch 输出比较；后处理关键点、bbox、scores 与 MMPose 源码预测比较。
- 输出报告区分随机初始化 smoke 权重与训练 checkpoint，不能将随机权重精度误报为训练效果。
