"""A script that performs cropping and resizing of images to a given size."""

import argparse
from pathlib import Path
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from ultralytics import YOLO
from tqdm import tqdm
from imgaug.augmentables.bbs import (
    BoundingBox,
    BoundingBoxesOnImage
)


PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '-i', '--input', type=Path, required=True,
        help='Path to images dataset'
    )
    parser.add_argument(
        '-o', '--output', type=Path, required=True,
        help='Path to output'
    )
    parser.add_argument(
        '-m', '--yolo-model',
        choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
        default='yolo11s', help='Type of yolo model.'
    )
    parser.add_argument(
        '-s', '--target-size', type=int, nargs='+',
        default=(1024, 1024), help='Output image size.'
    )
    parser.add_argument(
        '-l', '--target-label', type=int,
        default=0, help='Target class label.'
    )
    parser.add_argument(
        '-c', '--max-object-count', type=int,
        default=1, help='Maximal objects count per image.'
    )
    parser.add_argument(
        '--limit-per-subdir', type=int,
        default=0, help='Maximal images per sub directory.'
    )
    parser.add_argument(
        '--min-obj-size', type=int, nargs='+',
        default=(512, 512), help='Minimal object size.'
    )
    parser.add_argument(
        '--force', action='store_true', help='Forced run.'
    )

    return parser.parse_args()


class PersonFinder:
    def __init__(self, opts: argparse.Namespace):
        self.__person_detector = YOLO('yolo11s.pt')
        self.__face_detector = YOLO(
            PROJECT_ROOT / 'data' / 'models' / 'yolov11n-face.pt'
        )
        self.__opts = opts
        self.__paths = self.__init_paths(opts)

    def __init_paths(self, args: argparse.Namespace):
        source_dir: Path = args.input
        output_dir: Path = args.output

        if not args.force and source_dir.name != output_dir.name:
            logger.warning(
                'Final name of input dir and final name of '
                f'output dir is not matched: '
                f'{source_dir.name} and {output_dir.name}'
            )
            logger.warning('Do you want continue? (y/n)')
            if input() != 'y':
                exit(0)

        if not args.force and output_dir.is_dir():
            logger.warning(f'Output folder already exist: {output_dir}')
            logger.warning('Do you want continue? (y/n)')
            if input() != 'y':
                exit(0)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info('Prepare list of images...')
        image_extensions = ['.jpg', '.png', '.jpeg', '.webp']
        image_extensions = (
                image_extensions + [x.upper() for x in image_extensions]
        )
        image_paths: list[Path] = [
            x for x in source_dir.rglob('*')
            if x.is_file() and x.suffix in image_extensions
        ]
        logger.info(f'Total images: {len(image_paths)}')

        if args.limit_per_subdir < 1:
            input_data: list[Path] = image_paths
        else:
            logger.info('Split list of images into sub dirs...')
            sub_dirs = set([x.parent for x in image_paths])
            sub_dirs = {str(k): [] for k in sub_dirs}
            for path in image_paths:
                sub_dirs[str(path.parent)].append(path)
            input_data: dict[str, list[Path]] = sub_dirs
            logger.info(f'Total sub dirs: {len(input_data)}')
        return input_data

    def __process_image(
            self,
            img_path: Path,
            dst_path: Path,
            target_label: int,
            max_objects_count: int,
            obj_min_size: tuple[int, int],
            target_shape: tuple[int, int],
    ):
        image = cv2.imread(str(img_path))
        results = self.__person_detector(image, verbose=False)
        for result in results:
            labels = result.boxes.cls.detach().cpu().numpy().astype(int)
            bboxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
            bboxes_wh = result.boxes.xywh.detach().cpu().numpy().astype(int)

            filtered_bboxes = self.__filter_boxes(
                bboxes, bboxes_wh, labels,
                target_label, obj_min_size
            )

            if (
                    len(filtered_bboxes) == 0 or
                    len(filtered_bboxes) > max_objects_count
            ):
                continue

            squared_bboxes = self.__to_square(filtered_bboxes, image.shape)

            for box_idx, box in enumerate(squared_bboxes):
                sl_data = (
                    slice(box[0, 1], box[1, 1]),
                    slice(box[0, 0], box[1, 0]),
                    slice(None)
                )
                cropped_image = image[sl_data].copy()
                cropped_image = Image.fromarray(cropped_image)
                cropped_image.thumbnail(
                    target_shape, Image.Resampling.LANCZOS
                )
                cropped_image = np.asarray(cropped_image)

                parent_dir = img_path.parent.name
                file_name = (
                    f'{parent_dir}_{img_path.stem}_{str(box_idx).zfill(3)}.png'
                )
                file_name = str(dst_path / file_name)
                cv2.imwrite(file_name, cropped_image)

    def __filter_boxes(
            self,
            bboxes: np.ndarray,
            bboxes_wh: np.ndarray,
            labels: np.ndarray,
            target_label: int,
            shape_thresh_min: tuple[int, int]
    ) -> np.ndarray:
        bboxes = bboxes[labels == target_label]
        bboxes_wh = bboxes_wh[labels == target_label]

        # Check min size of object
        indices = [
            i for i, x in enumerate(bboxes)
            if ((x[2] - x[0] >= shape_thresh_min[0]) and
                (x[3] - x[1] >= shape_thresh_min[1])
                )
        ]
        if len(indices) == 0:
            return np.array([])
        bboxes = np.array([bboxes[i] for i in indices])
        bboxes_wh = np.array([bboxes_wh[i] for i in indices])

        areas = np.array([x[1] * x[2] for x in bboxes_wh])
        area_thresh = np.max(areas) * 0.1
        return bboxes[areas >= area_thresh]

    def __to_square(
            self,
            bboxes: np.ndarray,
            image_shape: tuple[int, ...]
    ) -> np.ndarray:
        for box in bboxes:
            wh = box[2:] - box[:2]
            if wh[0] > wh[1]:
                diff = wh[0] - wh[1]
                shift_0 = diff // 2
                shift_1 = diff - shift_0
                box[1] -= shift_0
                box[3] += shift_1
            elif wh[1] > wh[0]:
                diff = wh[1] - wh[0]
                shift_0 = diff // 2
                shift_1 = diff - shift_0
                box[0] -= shift_0
                box[2] += shift_1

        bboxes = BoundingBoxesOnImage(
            [BoundingBox(*x) for x in bboxes],
            shape=image_shape
        )
        bboxes = bboxes.remove_out_of_image().clip_out_of_image()
        bboxes = [x.coords for x in bboxes.bounding_boxes]
        return np.array(bboxes, dtype=int)

    def run(self):
        output_dir: Path = self.__opts.output

        target_label = self.__opts.target_label
        max_objects_count = self.__opts.max_object_count

        target_shape: tuple[int, ...] = tuple(self.__opts.target_size)
        obj_min_size: tuple[int, ...] = tuple(self.__opts.min_obj_size)

        if len(target_shape) > 2:
            raise ValueError('Target shape must be a size 2.')
        if len(obj_min_size) > 2:
            raise ValueError('Obj min size must be a size 2.')

        if len(target_shape) == 1:
            target_shape = target_shape[0], target_shape[0]
        if len(obj_min_size) == 1:
            obj_min_size = obj_min_size[0], obj_min_size[0]

        final_images_path = []
        if isinstance(self.__paths, list):
            final_images_path = self.__paths
        else:
            logger.info(
                f'Images per folder: {self.__opts.limit_per_subdir}'
            )
            logger.info('Prepare finished list of images...')
            for sub_dir in tqdm(self.__paths):
                image_paths: list[Path] = self.__paths[sub_dir]
                shuffle(image_paths)
                final_images_path.extend(
                    image_paths[:self.__opts.limit_per_subdir]
                )
            logger.info(f'The final image number: {len(final_images_path)}')

        logger.info('Run cropping...')
        for img_path in tqdm(final_images_path):
            try:
                self.__process_image(
                    img_path, output_dir, target_label,
                    max_objects_count, obj_min_size, target_shape
                )
            except Exception as e:
                logger.warning(
                    f'Failed to process image: {img_path}; verbose: {e}'
                )


def main():
    args = parse_args()
    image_handler = PersonFinder(args)
    image_handler.run()
    logger.info('Cropping process has been finished.')


if __name__ == '__main__':
    main()