"""Benchmarker class for running benchmarks on DriveNet models."""

import os

import cv2
import numpy as np
import numpy.typing as npt
import toml

import drivenetbench.utilities.config as configs
from drivenetbench.detector import Detector
from drivenetbench.similarity_calculator import SimilarityCalculator
from drivenetbench.utilities.utils import load_and_preprocess_points
from drivenetbench.utilities.view_transformer import ViewTransformer


class BenchMarker:
    """Class for running benchmarks on DriveNet models."""

    def __init__(self, config_file_path: str):
        """Initialize the BenchMarker.

        Parameters
        ----------
        config_file_path : str
            The path to the configuration file.
        """
        self.config_file_path = config_file_path

        os.mkdir("results") if not os.path.exists("results") else None

        configs.load_config(config_file_path)

        self.detector = Detector()
        self.similarity_calculator = SimilarityCalculator()

        self.view_transformer = None
        self._load_view_transformer()

        self.reference_path = None
        self.track_image = None
        self._annotate_path_to_track()

        video_path = configs.get_config("benchmarker.video_path")
        self.video_name = self.experiment_name = os.path.basename(
            video_path
        ).split(".")[0]

        self._create_results_directory()

        self.cap = cv2.VideoCapture(video_path)
        self.detector.frame_height = self.frame_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def _load_view_transformer(self):
        """Load the view transformer from the configuration file."""
        source_keypoints_path = configs.get_config(
            "view_transformer.source_path"
        )
        target_keypoints_path = configs.get_config(
            "view_transformer.target_path"
        )

        source_keypoints = load_and_preprocess_points(source_keypoints_path)
        target_keypoints = load_and_preprocess_points(target_keypoints_path)

        self.view_transformer = ViewTransformer(
            source=source_keypoints, target=target_keypoints
        )

    def _annotate_path_to_track(self):
        """Annotate the reference path to the track image."""
        reference_path_file_path = configs.get_config(
            "benchmarker.reference_track_npy_path"
        )
        self.reference_path = np.load(reference_path_file_path)

        track_image_path = configs.get_config("benchmarker.track_image_path")
        self.track_image = cv2.imread(track_image_path)
        for point in self.reference_path:
            cv2.circle(self.track_image, tuple(point), 5, (76, 39, 0), -1)

    def _create_results_directory(self):
        """Create the results directory."""
        self.results_dir = os.path.join("results", self.experiment_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
            return

        RESULTS_COMPONENETS = [
            "results.toml",
            "robot_path.npy",
            "final.jpg",
        ]
        self.save_video = configs.get_config("actions.save_live_to_disk")
        if self.save_video:
            RESULTS_COMPONENETS.append("final.mp4")
        all_present = all(
            os.path.exists(os.path.join(self.results_dir, comp))
            for comp in RESULTS_COMPONENETS
        )
        if all_present:
            raise FileExistsError(
                f"Results directory {self.results_dir} already exists."
            )

        os.makedirs(self.results_dir, exist_ok=True)

    def run(self):
        """Run the benchmarker."""
        robot_path = (
            self._detect_robot_path()
            if not os.path.exists(
                os.path.join(self.results_dir, "robot_path.npy")
            )
            else np.load(os.path.join(self.results_dir, "robot_path.npy"))
        )
        robot_path = robot_path[~np.all(robot_path == -1, axis=1)]
        distance_threshold = configs.get_config(
            "benchmarker.time.distance_threshold_in_pixels"
        )
        skip_seconds = configs.get_config(
            "benchmarker.time.skip_first_x_seconds"
        )

        time_to_return = self._calculate_time_for_full_rotation(
            robot_path,
            fps=self.video_fps,
            distance_threshold=distance_threshold,
            skip_seconds=skip_seconds,
        )

        similarity_percentage = (
            self.similarity_calculator.calculate_path_similarity(
                robot_path, self.reference_path
            )
        )

        results = {
            "time_to_return": time_to_return,
            "similarity_percentage": similarity_percentage,
        }
        with open(os.path.join(self.results_dir, "results.toml"), "w") as f:
            toml.dump(results, f)

    def _detect_robot_path(self) -> npt.NDArray:
        """Detect the robot path in the video.

        Returns
        -------
        npt.NDArray
            The robot path.

        Raises
        ------
        IOError
            If the video stream or file cannot be opened.
        """
        if not self.cap.isOpened():
            raise IOError("Error opening video stream or file")

        self.show_live = configs.get_config("actions.show_live")

        video_writer = (
            cv2.VideoWriter(
                os.path.join(self.results_dir, "final.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (self.frame_width, self.frame_height),
            )
            if self.save_video
            else None
        )
        robot_path = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            max_seconds = configs.get_config(
                "actions.early_stop_after_x_seconds"
            )

            max_frames = None
            if max_seconds:
                max_frames = max_seconds * self.video_fps
            if max_frames is not None and current_frame > max_frames:
                break

            robot_centroid: npt.NDArray = self.detector.detect_robot(
                frame=frame
            )
            if robot_centroid is not None:
                robot_path.append(robot_centroid)
            else:
                robot_path.append(np.array([-1, -1]))
                continue

            transformed_centroid = self._get_transformed_point(robot_centroid)

            if self.show_live:
                cv2.circle(
                    self.track_image,
                    tuple(transformed_centroid.astype(int)),
                    20,
                    (5, 203, 255),
                    -1,
                )
                cv2.polylines(
                    frame,
                    [self.detector.points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                cv2.circle(
                    frame,
                    tuple(robot_centroid.astype(int)),
                    5,
                    (0, 255, 0),
                    -1,
                )

                cv2.namedWindow("Robot Detection", cv2.WINDOW_NORMAL)
                cv2.imshow("Robot Detection", frame)

                cv2.namedWindow("Track View", cv2.WINDOW_NORMAL)
                cv2.imshow("Track View", self.track_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("User requested exit. Stopping early...")
                    break
            if video_writer is not None:
                video_writer.write(frame)
        cv2.imwrite(
            os.path.join(self.results_dir, "final.jpg"), self.track_image
        )
        video_writer.release() if self.save_video else None
        self.cap.release()
        cv2.destroyAllWindows()

        robot_path = np.array(robot_path, dtype=np.float32)
        np.save(os.path.join(self.results_dir, "robot_path.npy"), robot_path)

        return robot_path

    def _get_transformed_point(self, point: npt.NDArray) -> npt.NDArray:
        """Get the transformed point.

        Parameters
        ----------
        point : npt.NDArray
            The point to transform.

        Returns
        -------
        npt.NDArray
            The transformed point.
        """
        centroid_2d = point[np.newaxis, ...]
        transformed_centroid = self.view_transformer.transform_points(
            centroid_2d
        )
        squeezed_transformed_centroid = np.squeeze(
            transformed_centroid, axis=0
        )
        return squeezed_transformed_centroid

    def _calculate_time_for_full_rotation(
        self,
        robot_path: npt.NDArray,
        fps: int,
        distance_threshold: float = 50.0,
        skip_seconds: int = 5,
    ) -> float:
        """
        Calculates how long it takes for the robot to make a "full rotation" and return close
        to the first point in its path.

        Parameters
        ----------
        robot_path : npt.NDArray, shape (N, 2)
            The robot's path in track coordinates, in chronological order.
        fps : int
            Frames per second of the source video, used to convert frames to seconds.
        distance_threshold : float, optional
            If the robot comes within this Euclidean distance of the start point, we
            consider it has returned.
        start_frame : int, optional

        Returns
        -------
        float
            Time in seconds it took to complete the rotation. If it never returns, returns -1.
        """
        if len(robot_path) < 2:
            raise ValueError("Robot path must have at least two points.")

        start_frame = int(skip_seconds * fps)
        start_point = robot_path[0]
        for i in range(start_frame, len(robot_path)):
            dist = np.linalg.norm(robot_path[i] - start_point)
            if dist <= distance_threshold:
                # i => the frame index of the path, time => i / fps, assuming each frame has a robot position
                return i / fps

        return -1.0  # never returned to start


if __name__ == "__main__":
    benchmarker = BenchMarker("config.yaml")
    benchmarker.run()
