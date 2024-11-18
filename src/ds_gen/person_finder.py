"""A class that crops the area of an image containing a person."""

from enum import Enum
from pathlib import Path
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
from ultralytics import YOLO
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).parent.parent.parent


class YoloModelType(Enum):
    """Type of yolo model."""

    yolo11n = 'yolo11n'
    yolo11s = 'yolo11s'
    yolo11m = 'yolo11m'
    yolo11l = 'yolo11l'
    yolo11x = 'yolo11x'


class PFParams(BaseModel):
    """Parameters for Person Finder class."""

    input: Path = Field(
        description='Path to images dataset.'
    )

    output: Path = Field(
        description='Path where cropped images will be saved.'
    )

    yolo_model: YoloModelType = Field(
        default=YoloModelType.yolo11s,
        description='Type of yolo model for person detection.'
    )

    target_size: list[int] = Field(
        default=[1024, 1024],
        description='Output image size.'
    )

    target_label: int = Field(
        ge=0,
        default=0,
        description='Target class label.'
    )

    max_object_count: int = Field(
        gt=0,
        default=1,
        description='Maximal objects count per image.'
    )

    limit_per_subdir: int = Field(
        ge=0,
        default=0,
        description='Maximal images per sub directory.'
    )

    min_obj_size: list[int] = Field(
        default=[512, 512],
        description='Minimal object size.'
    )

    force: bool = Field(
        default=False,
        description='Forced run.'
    )


class PersonFinder:
    """A class that crops the area of an image containing a person.

    Each input image is searched for a region containing a person using
    the yolo detector. The found region is then reduced to square proportions,
    preserving the person's face in the given region (if there is a face
    in the image).
    """

    def __init__(self, params: PFParams):
        """Initializes the Person Finder object.

        Parameters
        ----------
        params : PFParams
            A PFParams object containing configuration parameters for the
            person finder.

        """
        yolo_model = f'{params.yolo_model.value}.pt'
        self.__person_detector: YOLO = YOLO(
            PROJECT_ROOT / 'data' / 'models' /yolo_model
        )
        self.__face_detector: YOLO = YOLO(
            PROJECT_ROOT / 'data' / 'models' / 'yolov11n-face.pt'
        )
        self.__params: PFParams = params
        self.__paths: list[Path] | dict[str, list[Path]] = (
            self.__init_paths(params)
        )

    def __init_paths(
            self, params: PFParams
    ) -> list[Path] | dict[str, list[Path]]:
        """Initializes the paths for input and output based on user parameters.

        This method prepares the necessary file paths for image processing,
        handling both force override and existing directory warnings.

        Parameters
        ----------
        params : PFParams
            An instance of the PFParams class containing
            user-defined parameters.

        Returns
        -------
        list[Path] | dict[str, list[Path]]
            If `params.limit_per_subdir` is less than 1: Returns a sorted list
            of all image paths within the input directory. Otherwise: Returns
            a dictionary where keys are subdirectory names and values are lists
            of image paths within each subdirectory.

        """
        source_dir: Path = params.input
        output_dir: Path = params.output

        if not params.force and source_dir.name != output_dir.name:
            logger.warning(
                'Final name of input dir and final name of '
                f'output dir is not matched: '
                f'{source_dir.name} and {output_dir.name}'
            )
            logger.warning('Do you want continue? (y/n)')
            if input() != 'y':
                exit(0)

        if not params.force and output_dir.is_dir():
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
        image_paths = sorted(image_paths)
        logger.info(f'Total images: {len(image_paths)}')

        if params.limit_per_subdir < 1:
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
        """Processes an image to extract and save cropped images.

        Parameters
        ----------
        img_path : Path
            Path to the input image file.
        dst_path : Path
            Path to the directory where cropped images will be saved.
        target_label : int
            The label representing a person in the detection model's output.
        max_objects_count : int
            Maximum number of detected persons to process.
        obj_min_size : tuple[int, int]
            Minimum size (width, height) for a detected object to
            be considered.
        target_shape : tuple[int, int]
            Desired shape (width, height) for the cropped images.

        """
        image = cv2.imread(str(img_path))
        result = self.__person_detector(image, verbose=False)[0]
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
            return None

        squared_bboxes = self.__to_square(image, filtered_bboxes)

        for box_idx, box in enumerate(squared_bboxes):
            sl_data = (
                slice(box[1], box[3]),
                slice(box[0], box[2]),
                slice(None)
            )
            cropped_image = image[sl_data].copy()
            cropped_image = cv2.resize(
                cropped_image, target_shape, interpolation=cv2.INTER_CUBIC
            )

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
        """Filters bounding boxes based on label and size.

        This method filters a set of bounding boxes based on the
        following criteria:

        1. **Label:** Only bounding boxes with the specified `target_label`
        are kept.
        2. **Size:** Bounding boxes must have a minimum area defined by
        `shape_thresh_min`. The area is calculated as the product of width
        and height. Bounding boxes smaller than 10% of the maximum area
        are also removed.

        Parameters
        ----------
        bboxes : np.ndarray
            A numpy array of bounding boxes in format [xmin, ymin, xmax, ymax].
        bboxes_wh : np.ndarray
            A numpy array of bounding box widths and heights in
            format [width, height].
        labels : np.ndarray
            A numpy array of labels corresponding to each bounding box.
        target_label : int
            The label to filter for.
        shape_thresh_min : tuple[int, int]
            A tuple representing the minimum width and height for a
            bounding box to be kept.

        Returns
        -------
        np.ndarray
            A numpy array of filtered bounding boxes in
            format [xmin, ymin, xmax, ymax].

        """
        bboxes = bboxes[labels == target_label]
        bboxes_wh = bboxes_wh[labels == target_label]

        # Check min size of object
        indices = [
            i for i, x in enumerate(bboxes)
            if (
                (x[2] - x[0] >= shape_thresh_min[0]) and
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
            image: np.ndarray,
            bboxes: np.ndarray,
    ) -> np.ndarray:
        """Ensures all bounding boxes in `bboxes` are square proprotions.

        This method iterates through each bounding box (`bbox`) in `bboxes`,
        calculating its width and height.

        If the width is greater than the height, it calculates the difference
        and shifts the top and bottom coordinates to center the box.
        Conversely, if the height is greater than the width, it shifts
        the left and right coordinates.

        After squaring the bounding boxes, it checks if they are within
        the image boundaries and adjusts them accordingly.
        Then, it utilizes the `_get_faces` method to detect faces within
        each squared bounding box. Finally, it calls the
        `_shift_boxes_wrt_face` method to further shift the bounding
        boxes based on detected face locations.

        Parameters
        ----------
        image : np.ndarray
            The input image as a NumPy array.
        bboxes : np.ndarray
            A NumPy array of shape (N, 4) representing N bounding
            boxes in format (x1, y1, x2, y2).

        Returns
        -------
        np.ndarray
            A NumPy array of shape (N, 4) representing N bounding boxes
            in format (x1, y1, x2, y2).

        """
        # Get square proportions
        for box in bboxes:
            wh = box[2:] - box[:2]
            if wh[0] > wh[1]:
                diff = wh[0] - wh[1]
                shifts = [-diff // 2, diff - diff // 2]
                box[[1, 3]] += np.array(shifts, dtype=box.dtype)
            elif wh[1] > wh[0]:
                diff = wh[1] - wh[0]
                shifts = [-diff // 2, diff - diff // 2]
                box[[0, 2]] += np.array(shifts, dtype=box.dtype)

        # Check out of bounds
        bboxes[:, :2] = np.maximum(bboxes[:, :2], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], image.shape[1])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], image.shape[0])

        faces: dict[int, np.ndarray | None] = self._get_faces(image, bboxes)
        shifted_bboxes: np.ndarray = self._shift_boxes_wrt_face(bboxes, faces)

        return shifted_bboxes

    def _get_faces(
            self,
            image: np.ndarray,
            bboxes: np.ndarray
    ) -> dict[int, np.ndarray | None]:
        """Detects faces within each bounding box provided.

        This method iterates through the given bounding boxes (bboxes) and
        extracts the corresponding image patches. It then utilizes the
        internal face detector to detect faces within each patch.

        Parameters
        ----------
        image : np.ndarray
            The input image as a NumPy array.
        bboxes : np.ndarray
            A NumPy array of shape (N, 4) representing bounding boxes, where
            N is the number of bounding boxes. Each row represents a box
            in the format [x_min, y_min, x_max, y_max].

        Returns
        -------
        dict[int, np.ndarray | None]
            A dictionary where keys are indices corresponding to
            the input bounding boxes and values are NumPy arrays of detected
            face bounding boxes. If no faces are detected within a
            bounding box, the value will be `None`.

        """
        faces = {i: None for i in range(len(bboxes))}
        for i, bbox in enumerate(bboxes):
            box_image = image[
                bbox[1]:bbox[3], bbox[0]:bbox[2], :
            ]
            result = self.__face_detector(box_image, verbose=False)[0]
            face_bboxes = (
                result.boxes.xyxy.detach().cpu().numpy().astype(int)
            )
            if len(face_bboxes) == 0:
                continue
            s_indices = np.argsort(face_bboxes[:, 1])[0]
            faces[i] = face_bboxes[s_indices]
        return faces

    def _shift_boxes_wrt_face(
            self,
            bboxes: np.ndarray,
            faces: dict[int, np.ndarray | None]
    ) -> np.ndarray:
        """Shifts bounding boxes to better align with detected faces.

        This method iterates through the provided bounding boxes (bboxes) and
        correspondingly detected face bounding boxes (faces). For each box:

        1. **Calculates aspect ratio:** Determines the width-to-height
        ratio of the bounding box.

        2. **Handles missing faces:** If no face is detected within a bounding
           box, it adjusts the box size to be roughly square based on its
           original aspect ratio.

        3. **Shifts boxes based on face location:** If a face is detected, it
           adjusts the bounding box coordinates to better align with the face's
           position within the box. The shift direction and magnitude are
           determined by comparing the width and height of the bounding box
           and the face's position relative to the box's edges.

        Parameters
        ----------
        bboxes : np.ndarray
            A NumPy array of shape (N, 4) representing bounding boxes, where
            N is the number of bounding boxes. Each row represents a box
            in the format [x_min, y_min, x_max, y_max].
        faces : dict[int, np.ndarray  |  None]
            A dictionary mapping bounding box indices to detected face
            bounding boxes (as NumPy arrays).

        Returns
        -------
        np.ndarray
            The modified bounding box array with shifted coordinates.

        """
        shifted_bboxes = bboxes.copy()
        for person_idx in faces:
            person_box = shifted_bboxes[person_idx]
            wh = person_box[2:] - person_box[:2]
            diff = abs(wh[1] - wh[0])
            if faces[person_idx] is None:
                if wh[0] > wh[1]:
                    shifts = [diff // 2, -(diff - diff // 2)]
                    person_box[[0, 2]] += np.array(
                        shifts, dtype=person_box.dtype
                    )
                elif wh[1] > wh[0]:
                    shifts = [diff // 2, -(diff - diff // 2)]
                    person_box[[1, 3]] += np.array(
                        shifts, dtype=person_box.dtype
                    )
            else:
                face_box = faces[person_idx]
                if wh[1] > wh[0]:
                    border_dist = face_box[1], wh[1] - face_box[1]
                    if border_dist[0] < border_dist[1]:
                        person_box[3] -= diff
                    elif border_dist[0] > border_dist[1]:
                        person_box[1] += diff
                else:
                    border_dist = face_box[0], wh[0] - face_box[0]
                    if border_dist[0] < border_dist[1]:
                        person_box[2] -= diff
                    elif border_dist[0] > border_dist[1]:
                        person_box[0] += diff

        return shifted_bboxes

    def run(self):
        """Runs the person finder pipeline, cropping images and saving outputs.

        This method processes a list of image paths, extracts persons
        from each image, and saves the cropped person regions to the specified
        output directory.
        """
        output_dir: Path = self.__params.output

        target_label = self.__params.target_label
        max_objects_count = self.__params.max_object_count

        target_shape: tuple[int, ...] = tuple(self.__params.target_size)
        obj_min_size: tuple[int, ...] = tuple(self.__params.min_obj_size)

        if len(target_shape) > 2:
            raise ValueError('Target shape must be a size 2.')
        if len(obj_min_size) > 2:
            raise ValueError('Obj min size must be a size 2.')

        if len(target_shape) == 1:
            target_shape = target_shape[0], target_shape[0]
        if len(obj_min_size) == 1:
            obj_min_size = obj_min_size[0], obj_min_size[0]

        final_images_path: list[Path] = []
        if isinstance(self.__paths, list):
            final_images_path = self.__paths
        else:
            logger.info(
                f'Images per folder: {self.__params.limit_per_subdir}'
            )
            logger.info('Prepare finished list of images...')
            for sub_dir in tqdm(self.__paths):
                image_paths: list[Path] = self.__paths[sub_dir]
                shuffle(image_paths)
                final_images_path.extend(
                    image_paths[:self.__params.limit_per_subdir]
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
