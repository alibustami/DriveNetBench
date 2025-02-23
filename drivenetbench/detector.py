from typing import Union

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics import YOLO
from ultralytics.engine.results import Results

import drivenetbench.utilities.config as configs


class Detector:
    """Class for detecting the robot in the video."""

    def __init__(self):
        """Initialize the Detector."""
        self.frame_height = None
        model_path = configs.get_config(
            "benchmarker.detection_model.model_path"
        )
        self.model = YOLO(model_path)

        self.confidence_threshold = configs.get_config(
            "benchmarker.detection_model.conf_threshold"
        )
        self.shift_ratio = configs.get_config(
            "benchmarker.detection_model.shift_ratio"
        )

    def detect_robot(self, frame: npt.NDArray) -> Union[npt.NDArray, None]:
        """Detect the robot in the frame.

        Parameters
        ----------
        frame : npt.NDArray
            The frame to detect the robot in.

        Returns
        -------
        Union[npt.NDArray, None]
            The centroid of the robot.
        """
        results, *_ = self.model(frame)
        centriod = self._post_process(results=results)

        return centriod

    def _post_process(self, results: Results) -> Union[npt.NDArray, None]:
        """Post process the results.

        Parameters
        ----------
        results : Results
            The results to post process.

        Returns
        -------
        Union[npt.NDArray, None]
            The shifted centroid of the robot.
        """
        if results.obb is None:
            return None

        shifted_centroid = None
        for box in results.obb:
            if box.conf[0] < self.confidence_threshold:
                continue

            corners = box.xyxyxyxy.cpu().numpy().squeeze()
            self.points = np.array(
                [
                    [int(corners[0][0]), int(corners[0][1])],
                    [int(corners[1][0]), int(corners[1][1])],
                    [int(corners[2][0]), int(corners[2][1])],
                    [int(corners[3][0]), int(corners[3][1])],
                ],
                dtype=np.int32,
            )

            current_shifted_centroid = self._extract_centroid(self.points)
            if current_shifted_centroid is not None:
                shifted_centroid = current_shifted_centroid
                break
            if shifted_centroid is None:
                return None

        return shifted_centroid

    def _extract_centroid(self, points: npt.NDArray) -> npt.NDArray:
        """Extract the centroid of the robot.

        Parameters
        ----------
        points : npt.NDArray
            The points to extract the centroid from.

        Returns
        -------
        npt.NDArray
            The centroid of the robot.
        """
        centroid = np.mean(points, axis=0)

        if centroid[0] > 2700 and centroid[1] < 100:
            return None

        y_diff = self.frame_height - centroid[1]
        shift = int(y_diff * self.shift_ratio)

        shited_centroid = np.array([centroid[0], centroid[1] + shift])

        return shited_centroid
