"""
流式推理脚本 - 使用 MMPose Inferencer
针对私有数据集的批量推理
"""

from mmpose.apis import MMPoseInferencer
from pathlib import Path
import json
import time


def stream_inference(
    input_folder,
    output_folder='inference_results',
    model='rtmpose-m',
    device='cuda:0',
    save_vis=True,
    save_pred=True
):
    """
    流式推理文件夹中的所有图片

    参数:
        input_folder: 输入图片文件夹路径
        output_folder: 输出结果文件夹路径
        model: 模型名称 (rtmpose-t/s/m/l/x 或 'human')
        device: 设备 ('cuda:0' 或 'cpu')
        save_vis: 是否保存可视化结果
        save_pred: 是否保存预测结果
    """

    print("=" * 60)
    print("MMPose 流式推理")
    print("=" * 60)

    # 初始化推理器
    print(f"\n[1/3] 初始化推理器...")
    print(f"  - 模型: {model}")
    print(f"  - 设备: {device}")

    inferencer = MMPoseInferencer(
        pose2d=model,
        device=device,
        show_progress=True  # 显示进度条
    )

    print("  ✓ 推理器初始化完成")

    # 准备输出目录
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"\n❌ 错误: 输入文件夹不存在: {input_folder}")
        return

    print(f"\n[2/3] 准备推理...")
    print(f"  - 输入目录: {input_path.absolute()}")
    print(f"  - 输出目录: {output_path.absolute()}")

    # 流式推理 (Inferencer 会自动遍历文件夹并逐张处理)
    print(f"\n[3/3] 开始流式推理...")

    vis_dir = str(output_path / 'visualizations') if save_vis else None
    pred_dir = str(output_path / 'predictions') if save_pred else None

    start_time = time.time()

    # 统计信息
    stats = {
        'total_images': 0,
        'total_people': 0,
        'image_results': []
    }

    # 流式推理 - Inferencer 返回 Generator，逐张 yield
    for result in inferencer(
        str(input_path),
        show=False,
        vis_out_dir=vis_dir,
        pred_out_dir=pred_dir,
    ):
        # 处理当前图片的结果
        stats['total_images'] += 1

        if 'predictions' in result:
            num_people = len(result['predictions'])
            stats['total_people'] += num_people

            # 记录结果摘要
            img_name = result.get('img_path', f'image_{stats["total_images"]}')
            if isinstance(img_name, str):
                img_name = Path(img_name).name

            stats['image_results'].append({
                'image': img_name,
                'num_people': num_people
            })

    elapsed_time = time.time() - start_time

    # 打印统计信息
    print("\n" + "=" * 60)
    print("推理完成")
    print("=" * 60)
    print(f"总图片数: {stats['total_images']}")
    print(f"检测总人数: {stats['total_people']}")

    if stats['total_images'] > 0:
        avg_people = stats['total_people'] / stats['total_images']
        print(f"平均每张图片人数: {avg_people:.2f}")
        print(f"处理速度: {stats['total_images'] / elapsed_time:.2f} 张/秒")

    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"\n结果保存位置:")
    if save_vis:
        print(f"  - 可视化: {output_path / 'visualizations'}")
    if save_pred:
        print(f"  - 预测结果: {output_path / 'predictions'}")

    # 保存统计摘要
    summary_file = output_path / 'summary.json'
    output_path.mkdir(exist_ok=True, parents=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'model': model,
                'device': device,
                'input_folder': str(input_path),
            },
            'statistics': {
                'total_images': stats['total_images'],
                'total_people': stats['total_people'],
                'elapsed_time': elapsed_time,
            },
            'results': stats['image_results']
        }, f, indent=2, ensure_ascii=False)

    print(f"  - 统计摘要: {summary_file}")
    print("=" * 60)


if __name__ == '__main__':
    # 配置参数
    INPUT_FOLDER = r'E:\OMS\dataset_test\smoke_phone_open_closed_222\images\val'
    OUTPUT_FOLDER = 'inference_output'

    # 可用的模型别名：
    # - 'human' 或 'body'  → RTMPose-M (推荐，平衡速度和精度)
    # - 'rtmpose-l'        → RTMPose-L (精度更高，速度较慢)
    #
    # 或使用完整的模型名称：
    # - 'rtmpose-t_8xb256-420e_body8-256x192'  → 最快
    # - 'rtmpose-s_8xb256-420e_body8-256x192'  → 很快
    # - 'rtmpose-m_8xb256-420e_body8-256x192'  → 平衡（推荐）
    # - 'rtmpose-l_8xb256-420e_body8-256x192'  → 精度高
    MODEL = 'rtmpose-s_8xb256-420e_body8-256x192'  # 推荐使用 'human'

    DEVICE = 'cuda:0'    # 如果没有 GPU，改为 'cpu'

    # 执行推理
    stream_inference(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        model=MODEL,
        device=DEVICE,
        save_vis=True,   # 保存可视化结果
        save_pred=True   # 保存预测坐标（JSON）
    )
