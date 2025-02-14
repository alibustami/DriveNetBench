import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from drivenetbench.utilities.config import get_config
from drivenetbench.utilities.utils import (
    load_and_preprocess_points,
    path_checker,
    path_fixer,
)

# Import your existing utilities and classes
from drivenetbench.view_transformer import ViewTransformer

###############################################################################
# Helper Functions
###############################################################################


def detect_robot_path(
    video_path: str,
    model_path: str,
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

    # 3. Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video at {video_path}")

    frame_index = 0
    robot_path = []  # 4. Collect path in the track coordinate system

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
                centroid_2d = centroid[np.newaxis, ...].astype(
                    np.float32
                )  # shape (1,2)
                track_coords = view_transformer.transform_points(centroid_2d)
                track_coords = np.squeeze(track_coords, axis=0)
                robot_path.append(track_coords)

        # Show live feed if requested
        if show_live:
            cv2.imshow("Robot Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User requested exit. Stopping early...")
                break

    cap.release()
    cv2.destroyAllWindows()

    robot_path = np.array(robot_path, dtype=np.float32)  # shape (N, 2)
    return robot_path


def calculate_time_for_full_rotation(
    path_xy: np.ndarray, fps: float, distance_threshold: float = 30.0
) -> float:
    """
    Calculates how long it takes for the robot to make a "full rotation" and return close
    to the first point in its path.

    Parameters
    ----------
    path_xy : np.ndarray, shape (N, 2)
        The robot's path in track coordinates, in chronological order.
    fps : float
        Frames per second of the source video, used to convert frames to seconds.
    distance_threshold : float, optional
        If the robot comes within this Euclidean distance of the start point, we
        consider it has returned.

    Returns
    -------
    float
        Time in seconds it took to complete the rotation. If it never returns, returns -1.
    """
    if len(path_xy) < 2:
        return -1.0

    start_point = path_xy[0]
    for i in range(1, len(path_xy)):
        dist = np.linalg.norm(path_xy[i] - start_point)
        if dist <= distance_threshold:
            # i => the frame index of the path, time => i / fps
            return i / fps

    return -1.0  # never returned to start


def calculate_path_similarity(
    robot_path: np.ndarray, reference_path: np.ndarray, method: str = "dtw"
) -> float:
    """
    Computes a similarity score (%), between 0 and 100, comparing the
    robot’s path to a reference path from `TrackProcessor`.

    Parameters
    ----------
    robot_path : np.ndarray, shape (N,2)
        The robot’s path in track coordinates.
    reference_path : np.ndarray, shape (M,2)
        The reference track path from a .npy file (e.g., skeleton or offset contour).
    method : str, optional
        The algorithm to use for path matching. Choose from ["dtw", "frechet"].

    Returns
    -------
    float
        A percentage in [0..100], where 100 means a perfect match.
    """
    if robot_path.size == 0 or reference_path.size == 0:
        return 0.0

    # Compute a distance measure with the chosen method
    if method.lower() == "dtw":
        score = _dtw_distance(robot_path, reference_path)
    elif method.lower() == "frechet":
        score = _frechet_distance(robot_path, reference_path)
    else:
        raise ValueError("Unsupported method. Use 'dtw' or 'frechet'.")

    # Convert raw distance to a percentage: smaller distance => higher similarity
    # This is a heuristic. Adjust 'some_scale' to your typical track dimensions.
    some_scale = 5000.0
    similarity_percent = 100.0 * max(0.0, 1.0 - (score / some_scale))
    similarity_percent = min(similarity_percent, 100.0)

    return similarity_percent


def _dtw_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """
    Computes a basic DTW (Dynamic Time Warping) distance between two 2D paths.
    Returns a distance value: lower = more similar.
    """
    n, m = len(path_a), len(path_b)
    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(path_a[i - 1] - path_b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # deletion
                dtw_matrix[i, j - 1],  # insertion
                dtw_matrix[i - 1, j - 1],  # match
            )
    return float(dtw_matrix[n, m])


def _frechet_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """
    Computes a simplified version of the Frechet distance between two 2D paths.
    Returns a distance value: lower = more similar.
    """
    ca = -np.ones((len(path_a), len(path_b)), dtype=np.float32)

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        dist = np.linalg.norm(path_a[i] - path_b[j])
        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), dist)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), dist)
        else:
            ca[i, j] = max(
                min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), dist
            )
        return ca[i, j]

    return _c(len(path_a) - 1, len(path_b) - 1)


###############################################################################
# Main Benchmarking Function
###############################################################################


def benchmark_robot_performance(
    video_path: str,
    detection_model_path: str,
    source_keypoints_path: str,
    target_keypoints_path: str,
    reference_track_npy_path: str,
    output_video_path: str = "benchmarked_output.mp4",
    confidence_threshold: float = 0.85,
    shift_ratio: float = 0.0275,
    path_similarity_method: str = "dtw",
    show_live: bool = False,
) -> Tuple[float, float]:
    """
    Main function to benchmark the robot’s performance by:
      1) Detecting the robot’s path from a video (via YOLO + homography).
      2) Calculating time for a full rotation (frames -> seconds).
      3) Calculating how well the robot’s path matches a reference path.

    Parameters
    ----------
    video_path : str
        Path to the robot’s navigation video.
    detection_model_path : str
        Path to YOLO detection model weights.
    source_keypoints_path : str
        Path to .npy with source (camera) keypoints for homography.
    target_keypoints_path : str
        Path to .npy with target (track) keypoints for homography.
    reference_track_npy_path : str
        Path to .npy with the reference path (e.g., from TrackProcessor).
    output_video_path : str, optional
        Path for saving annotated output video. Currently not fully implemented,
        but you can adapt from `main.py` if you want side-by-side frames.
    confidence_threshold : float, optional
        YOLO confidence threshold.
    shift_ratio : float, optional
        Vertical shift heuristic.
    path_similarity_method : str, optional
        Algorithm used for path comparison. Options: "dtw" or "frechet".
    show_live : bool, optional
        If True, shows real-time processing frames (press 'q' to quit early).

    Returns
    -------
    Tuple[float, float]
        (full_rotation_time_in_seconds, path_similarity_percent).
        If the robot never returns to start, `full_rotation_time_in_seconds` = -1.
    """

    # --- 1) Detect robot path in track coordinates ---
    robot_path = detect_robot_path(
        video_path=video_path,
        model_path=detection_model_path,
        source_keypoints_path=source_keypoints_path,
        target_keypoints_path=target_keypoints_path,
        confidence_threshold=confidence_threshold,
        shift_ratio=shift_ratio,
        show_live=show_live,
    )

    # If you want to produce an annotated output video like in main.py,
    # you could do that here. For brevity, we skip it.

    # --- 2) Compute time for a full rotation ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    full_rotation_time = calculate_time_for_full_rotation(robot_path, fps)

    # --- 3) Compute path similarity to reference track path ---
    reference_path = np.load(reference_track_npy_path)
    similarity_percent = calculate_path_similarity(
        robot_path, reference_path, method=path_similarity_method
    )

    return (full_rotation_time, similarity_percent)


if __name__ == "__main__":
    # VIDEO_PATH = "assets/pos3.mov"
    # MODEL_PATH = "weights/best.pt"
    # SOURCE_KPTS = "keypoints/pos3_keypoints_from_camera.npy"
    # TARGET_KPTS = "keypoints/old_keypoints_from_digital.npy"
    # REF_PATH = "keypoints/all_path.npy"  # from track_processor.py
    # OUTPUT_VIDEO = "assets/benchmark_output.mp4"

    VIDEO_PATH = "assets/new_track/driver_4.mov"
    MODEL_PATH = "weights/best.pt"
    SOURCE_KPTS = "keypoints/new_track/keypoints_from_camera.npy"
    TARGET_KPTS = "keypoints/new_track/keypoints_from_diagram.npy"
    REF_PATH = "keypoints/center_path.npy"  # from track_processor.py
    OUTPUT_VIDEO = "assets/benchmark_output.mp4"

    rotation_time, sim_score = benchmark_robot_performance(
        video_path=VIDEO_PATH,
        detection_model_path=MODEL_PATH,
        source_keypoints_path=SOURCE_KPTS,
        target_keypoints_path=TARGET_KPTS,
        reference_track_npy_path=REF_PATH,
        output_video_path=OUTPUT_VIDEO,
        confidence_threshold=0.85,
        shift_ratio=0.0075,
        path_similarity_method="dtw",
        show_live=True,  # <-- Enable live visualization
    )

    if rotation_time < 0:
        print("Robot did not return to start point within threshold.")
    else:
        print(f"Time for full rotation: {rotation_time:.2f} seconds")

    print(f"Path similarity: {sim_score:.2f}%")
