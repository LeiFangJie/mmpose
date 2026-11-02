"""RTMPose 板端后处理，只依赖 NumPy。"""

import numpy as np


def _scale_norm(values, gain, eps=1e-5):
    """复现 RTMCC 的 ScaleNorm。"""
    dim_scale = values.shape[-1] ** -0.5
    norm = np.linalg.norm(values, axis=-1, keepdims=True) * dim_scale
    return values / np.maximum(norm, eps) * gain


def _silu(values):
    """计算 SiLU，使用稳定形式避免大负值溢出。"""
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = values[positive] / (1. + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = values[~positive] * exp_values / (1. + exp_values)
    return output


def _relative_position_bias(weights, token_count):
    """复现 RTMCCBlock.rel_pos_bias 的窗口切片。"""
    values = np.pad(weights[:2 * token_count - 1], (0, token_count))
    values = np.tile(values, token_count)[:-token_count]
    values = values.reshape(token_count, token_count, 3 * token_count - 2)
    radius = (2 * token_count - 1) // 2
    return values[..., radius:-radius]


def _gau(tokens, weights):
    """用 NumPy 执行 RTMCCBlock 的 self-attention GAU。"""
    normalized = _scale_norm(tokens, weights['gau_ln_g'])
    uv = _silu(np.matmul(normalized, weights['gau_uv'].T))
    expansion = weights['gau_o'].shape[1]
    u, value, base = np.split(uv, [expansion, expansion * 2], axis=2)
    base = base[:, :, None, :] * weights['gau_gamma'][None, None, :, :]
    base += weights['gau_beta'][None, None, :, :]
    query, key = np.split(base, 2, axis=2)
    query = query[:, :, 0, :]
    key = key[:, :, 0, :]
    logits = np.matmul(query, np.swapaxes(key, 1, 2))
    if 'gau_w' in weights:
        logits += _relative_position_bias(weights['gau_w'], tokens.shape[1])
    kernel = np.square(np.maximum(logits / weights['gau_s'], 0.))
    output = u * np.matmul(kernel, value)
    output = np.matmul(output, weights['gau_o'].T)
    # RTMCCBlock 的 shortcut 分支是缩放输入残差，再叠加 GAU 主分支。
    return output + tokens * weights['gau_res_scale']


def simcc_from_feature(feature, weights):
    """从 `[B,17,8,6]` 特征执行板外 RTMCC/SimCC。"""
    if feature.ndim != 4 or feature.shape[1:] != (17, 8, 6):
        raise ValueError(f'feature shape 必须为 [B,17,8,6]，实际为 {feature.shape}')
    tokens = feature.reshape(feature.shape[0], 17, -1)
    tokens = _scale_norm(tokens, weights['mlp_norm_g'])
    tokens = np.matmul(tokens, weights['mlp_weight'].T)
    tokens = _gau(tokens, weights)
    return (np.matmul(tokens, weights['cls_x'].T),
            np.matmul(tokens, weights['cls_y'].T))


def decode_simcc(simcc_x, simcc_y, center, scale, split_ratio=2.0):
    """按 MMPose SimCCLabel.decode 和 TopdownPoseEstimator 还原坐标。"""
    x_index = np.argmax(simcc_x, axis=-1).astype(np.float32)
    y_index = np.argmax(simcc_y, axis=-1).astype(np.float32)
    score = np.minimum(
        np.max(simcc_x, axis=-1), np.max(simcc_y, axis=-1))
    keypoints = np.stack([x_index, y_index], axis=-1) / split_ratio
    input_size = np.asarray([192., 256.], dtype=np.float32)
    keypoints = keypoints / input_size * np.asarray(scale)[None, None, :]
    keypoints += np.asarray(center)[None, None, :] - 0.5 * np.asarray(scale)[None, None, :]
    return keypoints, score


def load_postprocess_weights(path):
    """读取导出器生成的 RTMCC/SimCC NPZ 权重。"""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}
