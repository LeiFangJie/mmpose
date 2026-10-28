# Copyright (c) OpenMMLab. All rights reserved.
"""Convert a split YOLO Pose dataset to COCO keypoint annotations.

Expected layout::

    dataset_root/
      images/{train,val,test}/image.jpg
      labels/{train,val,test}/image.txt

Each non-empty label line contains class, normalized xywh, and normalized
(x, y, visibility) triples. Images without a label file are treated as
negative images; empty label files are also valid.
"""

import argparse
import json
from pathlib import Path

IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
NUM_KEYPOINTS = 17


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--out-dir', default='datasets/coco_annotations')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument(
        '--strict-labels',
        action='store_true',
        help='Fail if an image has no corresponding label file.')
    return parser.parse_args()


def _load_image_size(image_path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Pillow is required to read image dimensions.') from exc

    with Image.open(image_path) as image:
        return image.width, image.height


def _parse_label_line(line, label_path, line_number, width, height):
    values = line.split()
    expected = 5 + NUM_KEYPOINTS * 3
    if len(values) != expected:
        raise ValueError(
            f'{label_path}:{line_number}: expected {expected} values '
            f'(class, xywh, 17 keypoints), got {len(values)}')

    try:
        class_id = int(values[0])
        numbers = [float(value) for value in values[1:]]
    except ValueError as exc:
        raise ValueError(
            f'{label_path}:{line_number}: labels must contain numeric values') from exc

    if class_id != 0:
        raise ValueError(
            f'{label_path}:{line_number}: expected person class id 0, got {class_id}')
    if any(value < 0 or value > 1 for value in numbers[:4]):
        raise ValueError(
            f'{label_path}:{line_number}: bbox values must be in [0, 1]')

    cx, cy, box_width, box_height = numbers[:4]
    bbox_x = (cx - box_width / 2) * width
    bbox_y = (cy - box_height / 2) * height
    bbox_width = box_width * width
    bbox_height = box_height * height

    keypoints = []
    num_keypoints = 0
    for index in range(NUM_KEYPOINTS):
        x, y, visibility = numbers[4 + index * 3:7 + index * 3]
        if visibility not in (0, 1, 2):
            raise ValueError(
                f'{label_path}:{line_number}: keypoint visibility must be 0, 1, or 2')
        if visibility > 0:
            num_keypoints += 1
        keypoints.extend([x * width, y * height, int(visibility)])

    return {
        'bbox': [bbox_x, bbox_y, bbox_width, bbox_height],
        'area': bbox_width * bbox_height,
        'keypoints': keypoints,
        'num_keypoints': num_keypoints,
    }


def convert_split(data_root, out_dir, split, strict_labels=False):
    image_dir = data_root / 'images' / split
    label_dir = data_root / 'labels' / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image split directory does not exist: {image_dir}')
    if not label_dir.is_dir():
        raise FileNotFoundError(f'Label split directory does not exist: {label_dir}')

    images = []
    annotations = []
    missing_labels = []
    image_id = 1
    annotation_id = 1

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        width, height = _load_image_size(image_path)
        relative_image_path = image_path.relative_to(data_root).as_posix()
        images.append({
            'id': image_id,
            'file_name': relative_image_path,
            'width': width,
            'height': height,
        })

        label_path = label_dir / f'{image_path.stem}.txt'
        if not label_path.is_file():
            missing_labels.append(str(label_path))
        else:
            with label_path.open('r', encoding='utf-8-sig') as label_file:
                for line_number, line in enumerate(label_file, start=1):
                    if not line.strip():
                        continue
                    instance = _parse_label_line(
                        line, label_path, line_number, width, height)
                    bbox = instance['bbox']
                    annotations.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': 1,
                        'bbox': bbox,
                        'area': instance['area'],
                        'keypoints': instance['keypoints'],
                        'num_keypoints': instance['num_keypoints'],
                        'iscrowd': 0,
                        'segmentation': [],
                    })
                    annotation_id += 1

        image_id += 1

    if strict_labels and missing_labels:
        preview = '\n'.join(missing_labels[:10])
        raise ValueError(
            f'{len(missing_labels)} images have no label file:\n{preview}')

    result = {
        'info': {'description': f'YOLO Pose converted dataset ({split})'},
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'person', 'keypoints': [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
            'skeleton': []}],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f'{split}.json'
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        f'{split}: {len(images)} images, {len(annotations)} persons, '
        f'{len(missing_labels)} missing label files -> {output_path}')
    return output_path


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    for split in args.splits:
        convert_split(data_root, out_dir, split, args.strict_labels)


if __name__ == '__main__':
    main()
