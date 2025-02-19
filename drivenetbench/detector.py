import cv2
import numpy as np
import numpy.typing as npt
from ultralytics import YOLO

import drivenetbench.utilities.config as configs
from drivenetbench.utilities.view_transformer import ViewTransformer


class Detector:
    def __init__(self):
        model_path = configs.get_config(
            "benchmarker.detection_model.model_path"
        )
        self.model = YOLO(model_path)

        source_keypoints = configs.get_config("view_transformer.source_path")
        target_keypoints = configs.get_config("view_transformer.target_path")
        self.view_transformer = ViewTransformer(
            source=source_keypoints, target=target_keypoints
        )

        reference_path_file_path = configs.get_config(
            "benchmarker.reference_track_npy_path"
        )
        reference_path = np.load(reference_path_file_path, dtype=np.int32)

        track_image_path = configs.get_config("benchmarker.track_image_path")
        self.track_image = cv2.imread(track_image_path)

        for point in reference_path:
            cv2.circle(self.track_image, tuple(point), 5, (0, 255, 0), -1)

        self.cap_path = configs.get_config("benchmarker.video_path")

        self.max_frames = configs.get_config(
            "actions.early_stop_after_x_frames"
        )

    def detect_robot(self) -> npt.ndarray:
        cap = cv2.VideoCapture(self.cap_path)

        if not cap.isOpened():
            raise IOError("Error opening video stream or file")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.max_frames is not None and current_frame > self.max_frames:
                break
