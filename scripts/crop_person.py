"""A script that performs cropping and resizing of images to a given size."""

import argparse
from pathlib import Path

from loguru import logger
from ds_gen import PersonFinder, PFParams


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


def main():
    args = parse_args()
    params = PFParams(**args.__dict__)
    image_handler = PersonFinder(params)
    image_handler.run()
    logger.info('Cropping process has been finished.')


if __name__ == '__main__':
    main()