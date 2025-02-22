from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from drivenetbench.utilities.utils import (
    load_and_preprocess_points,
    path_checker,
    path_fixer,
)
from drivenetbench.utilities.view_transformer import ViewTransformer

SOURCE_KPTS = "keypoints/new_track/keypoints_from_camera.npy"
TARGET_KPTS = "keypoints/new_track/keypoints_from_diagram.npy"


def detect_robot_path(
    video_path: str,
    model_path: str,
    track_path: str,
    source_keypoints_path: str,
    target_keypoints_path: str,
    confidence_threshold: float = 0.85,
    shift_ratio: float = 0.02,
    max_frames: Optional[int] = None,
    show_live: bool = False,
) -> np.ndarray:
    """
    Detects the robot in each frame of a video, applies a homography transform
    to map detections onto a track coordinate space, and returns the resulting path.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    model_path : str
        Path to the YOLO model weights.
    source_keypoints_path : str
        Path to the .npy file with source (camera) keypoints.
    target_keypoints_path : str
        Path to the .npy file with target (track) keypoints.
    confidence_threshold : float, optional
        Minimum confidence for YOLO detections, defaults to 0.85.
    shift_ratio : float, optional
        A vertical offset ratio for the detected bounding box centroid (heuristic).
    max_frames : int, optional
        If provided, limits the video reading to the first 'max_frames' frames.
    show_live : bool, optional
        If True, displays a live preview of each processed frame with OpenCV.

    Returns
    -------
    np.ndarray
        An array of shape (N, 2) with the robot centroid track coordinates for each frame
        where detection occurs.
    """

    # 1. Load model
    model = YOLO(model_path)
    model.conf = confidence_threshold

    # 2. Create ViewTransformer using source/target keypoints
    source_keypoints = load_and_preprocess_points(
        path_checker(path_fixer(source_keypoints_path))[1]
    )
    target_keypoints = load_and_preprocess_points(
        path_checker(path_fixer(target_keypoints_path))[1]
    )
    view_transformer = ViewTransformer(
        source=source_keypoints, target=target_keypoints
    )
    track_image = cv2.imread(track_path)
    # 3. Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video at {video_path}")

    frame_index = 0
    robot_path = []  # 4. Collect path in the track coordinate system
    # centroid_deque = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional early stop
        if max_frames is not None and frame_index >= max_frames:
            break
        frame_index += 1

        # Run YOLO detection
        results = model(frame)

        if results[0].obb is not None:
            for box in results[0].obb:
                if box.conf[0] < confidence_threshold:
                    continue

                corners = box.xyxyxyxy.cpu().numpy().squeeze()
                points = np.array(
                    [
                        [int(corners[0][0]), int(corners[0][1])],
                        [int(corners[1][0]), int(corners[1][1])],
                        [int(corners[2][0]), int(corners[2][1])],
                        [int(corners[3][0]), int(corners[3][1])],
                    ],
                    dtype=np.int32,
                )
                centroid = np.mean(points, axis=0)
                if centroid[0] > 2700 and centroid[1] < 100:
                    continue

                # Apply optional "shift" (heuristic)
                y_diff = frame.shape[0] - centroid[1]
                shift = int(y_diff * shift_ratio)
                centroid[1] += shift

                # (A) Draw on the live frame if requested
                if show_live:
                    cv2.polylines(
                        frame,
                        [points],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    # Mark the shifted centroid
                    cv2.circle(
                        frame, tuple(centroid.astype(int)), 5, (0, 0, 255), -1
                    )

                # (B) Transform to track coordinates
                centroid_2d = centroid[np.newaxis, ...]  # shape (1,2)
                track_coords = view_transformer.transform_points(centroid_2d)
                track_coords = np.squeeze(track_coords, axis=0)
                cv2.circle(
                    track_image,
                    tuple(track_coords.astype(int)),
                    20,
                    (255, 0, 0),
                    -1,
                )

                robot_path.append(track_coords)

        # Show live feed if requested
        if show_live:
            cv2.namedWindow("Robot Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Robot Detection", frame)

            cv2.namedWindow("Track View", cv2.WINDOW_NORMAL)
            cv2.imshow("Track View", track_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User requested exit. Stopping early...")
                break

    cap.release()
    cv2.destroyAllWindows()

    robot_path = np.array(robot_path, dtype=np.float32)  # shape (N, 2)
    return robot_path


if __name__ == "__main__":

    VIDEO_PATH = "assets/new_track/driver_4.MOV"
    MODEL_PATH = "weights/best.pt"
    TRACK_PATH = "assets/new_track/track-v2.jpg"
    SOURCE_KPTS = "keypoints/new_track/keypoints_from_camera.npy"
    TARGET_KPTS = "keypoints/new_track/keypoints_from_diagram.npy"
    REF_PATH = "keypoints/center_path.npy"  # from track_processor.py
    OUTPUT_VIDEO = "assets/benchmark_output.mp4"

    robot_path = detect_robot_path(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        track_path=TRACK_PATH,
        source_keypoints_path=SOURCE_KPTS,
        target_keypoints_path=TARGET_KPTS,
        confidence_threshold=0.85,
        shift_ratio=0.0075,
        show_live=True,
    )
