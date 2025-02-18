import glob
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


def detect_robot_path(
    video_path: str,
    model_path: str,
    track_img_path: str,
    reference_track_npy_path: str,
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
    model = YOLO(model_path)
    model.conf = confidence_threshold

    source_keypoints = load_and_preprocess_points(
        path_checker(path_fixer(source_keypoints_path))[1]
    )
    target_keypoints = load_and_preprocess_points(
        path_checker(path_fixer(target_keypoints_path))[1]
    )
    view_transformer = ViewTransformer(
        source=source_keypoints, target=target_keypoints
    )
    track_image = cv2.imread(track_img_path)
    reference_path = np.load(reference_track_npy_path)
    for point in reference_path:
        x, y = point
        cv2.circle(track_image, (int(x), int(y)), 5, (76, 39, 0), -1)

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split(".")[0]
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
                centroid_2d = centroid[np.newaxis, ...]  # shape (1,2)
                track_coords = view_transformer.transform_points(centroid_2d)
                track_coords = np.squeeze(track_coords, axis=0)
                # print(f"[DEBUG] Frame {frame_index}, detection centroid => track_coords = {track_coords}")

                cv2.circle(
                    track_image,
                    tuple(track_coords.astype(int)),
                    20,
                    (5, 203, 255),
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
    np.save(f"{video_name}_robot_path.npy", robot_path)
    return robot_path


def calculate_time_for_full_rotation(
    path_xy: np.ndarray,
    fps: float,
    distance_threshold: float = 50.0,
    skip_seconds: int = 5,
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
    start_frame : int, optional

    Returns
    -------
    float
        Time in seconds it took to complete the rotation. If it never returns, returns -1.
    """
    if len(path_xy) < 2:
        return -1.0

    # skip the first 5 seconds
    start_frame = int(5 * fps)
    start_point = path_xy[0]
    for i in range(start_frame, len(path_xy)):
        dist = np.linalg.norm(path_xy[i] - start_point)
        if dist <= distance_threshold:
            # i => the frame index of the path, time => i / fps
            return i / fps

    return -1.0  # never returned to start


def calculate_path_similarity(
    robot_path: np.ndarray,
    reference_path: np.ndarray,
    method: str = "dtw",
    auto_tune: bool = True,
    clamp_percentage: float = 0.05,
    clamp_dist: float = 300.0,
    distance_baseline: float = 3500.0,
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
    auto_tune : bool, optional
        If True, automatically tune distance_baseline and clamp_dist based on the paths.
    clamp_percentage : float, optional
        Percentage of the bounding box diagonal to use for clamping. Defaults to 0.05.
    clamp_dist : float, optional
        The max distance for a "soft match" (0%). Defaults to 300.0. If auto_tune=True, this is ignored.
    distance_baseline : float, optional
        The max distance for a "perfect match" (100%). If auto_tune=True, this is ignored.

    Returns
    -------
    float
        A percentage in [0..100], where 100 means a perfect match.
    """
    if robot_path.size == 0 or reference_path.size == 0:
        return 0.0

    # --- Auto-tune if requested ---
    if auto_tune:
        distance_baseline, clamp_dist = auto_tune_parameters(
            robot_path, reference_path, clamp_percentage
        )

    # --- Compute raw distance via chosen method ---
    if method.lower() == "dtw":
        dist_value = _dtw_distance(robot_path, reference_path, clamp_dist)
    elif method.lower() == "frechet":
        dist_value = _frechet_distance(robot_path, reference_path, clamp_dist)
    else:
        raise ValueError("Unsupported method. Use 'dtw' or 'frechet'.")

    # print("[DEBUG] dist_value (DTW or Frechet) =", dist_value)
    # print("[DEBUG] distance_baseline =", distance_baseline)

    # --- Convert distance -> similarity in [0..100] ---
    raw_ratio = 1.0 - (dist_value / distance_baseline)
    similarity_percent = 100.0 * max(0.0, raw_ratio)
    # print("[DEBUG] raw_ratio =", raw_ratio)
    # print("[DEBUG] final similarity_percent =", similarity_percent)

    similarity_percent = min(similarity_percent, 100.0)

    return similarity_percent


def _dtw_distance(
    path_a: np.ndarray, path_b: np.ndarray, clamp_dist: float = None
) -> float:
    """
    Computes DTW distance between two 2D paths (N vs M). Lower = more similar.

    Parameters
    ----------
    path_a : (N, 2) array
    path_b : (M, 2) array
    clamp_dist : float, optional
        If provided, each pairwise distance is clamped to this max.

    Returns
    -------
    float
        Total DTW cost.
    """
    n, m = len(path_a), len(path_b)
    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(path_a[i - 1] - path_b[j - 1])
            if clamp_dist is not None:
                cost = min(cost, clamp_dist)

            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # deletion
                dtw_matrix[i, j - 1],  # insertion
                dtw_matrix[i - 1, j - 1],  # match
            )
    # The raw sum cost:
    raw_sum = float(dtw_matrix[n, m])
    # Normalize it by the path size:
    normalized_dtw = raw_sum / (n + m)
    return normalized_dtw


def _frechet_distance(
    path_a: np.ndarray, path_b: np.ndarray, clamp_dist: float = None
) -> float:
    """
    Iterative Frechet distance between two 2D paths. Lower = more similar.

    Parameters
    ----------
    path_a : (N, 2) array
    path_b : (M, 2) array
    clamp_dist : float, optional
        If not None, will clamp each pairwise distance to at most this value
        to soften the penalty for large differences.

    Returns
    -------
    float
        Frechet distance.
    """
    n, m = len(path_a), len(path_b)
    # dp[i,j] will hold the Frechet distance up to path_a[:i+1], path_b[:j+1].
    dp = np.full((n, m), -1.0, dtype=np.float32)

    # Helper to compute local cost with clamp
    def local_dist(i, j):
        d = np.linalg.norm(path_a[i] - path_b[j])
        return min(d, clamp_dist)

    # Initialize first cell
    dp[0, 0] = local_dist(0, 0)

    # First row
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], local_dist(0, j))

    # First column
    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], local_dist(i, 0))

    # Fill the rest
    for i in range(1, n):
        for j in range(1, m):
            cost_ij = local_dist(i, j)
            dp[i, j] = max(
                min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]), cost_ij
            )

    return float(dp[n - 1, m - 1])


def benchmark_robot_performance(
    video_path: str,
    detection_model_path: str,
    track_img_path: str,
    source_keypoints_path: str,
    target_keypoints_path: str,
    reference_track_npy_path: str,
    output_video_path: str = "benchmarked_output.mp4",
    confidence_threshold: float = 0.85,
    shift_ratio: float = 0.0275,
    path_similarity_method: str = "dtw",
    show_live: bool = False,
    auto_tune: bool = True,
    clamp_percentage: float = 0.05,
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
    track_img_path : str
        Path to the track image for homography.
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
    auto_tune : bool, optional
        If True, automatically tunes distance_baseline and clamp_dist based on the paths.
    clamp_percentage : float, optional
        Percentage of the bounding box diagonal to use for clamping.

    Returns
    -------
    Tuple[float, float]
        (full_rotation_time_in_seconds, path_similarity_percent).
        If the robot never returns to start, `full_rotation_time_in_seconds` = -1.
    """

    video_name = os.path.basename(video_path).split(".")[0]
    robot_path = (
        detect_robot_path(
            video_path=video_path,
            model_path=detection_model_path,
            track_img_path=track_img_path,
            reference_track_npy_path=reference_track_npy_path,
            source_keypoints_path=source_keypoints_path,
            target_keypoints_path=target_keypoints_path,
            confidence_threshold=confidence_threshold,
            shift_ratio=shift_ratio,
            show_live=show_live,
        )
        if not os.path.exists(f"{video_name}_robot_path.npy")
        else np.load(f"{video_name}_robot_path.npy")
    )
    # robot_path = detect_robot_path(
    #         video_path=video_path,
    #         model_path=detection_model_path,
    #         track_img_path=track_img_path,
    #         reference_track_npy_path=reference_track_npy_path,
    #         source_keypoints_path=source_keypoints_path,
    #         target_keypoints_path=target_keypoints_path,
    #         confidence_threshold=confidence_threshold,
    #         shift_ratio=shift_ratio,
    #         show_live=show_live,
    #     )

    # If you want to produce an annotated output video like in main.py,
    # you could do that here. For brevity, we skip it.

    # --- 2) Compute time for a full rotation ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # print("[DEBUG] robot_path shape:", robot_path.shape)
    # if robot_path.size > 0:
    #     print("[DEBUG] robot_path min, max:\n",
    #         robot_path.min(axis=0),
    #         robot_path.max(axis=0))
    #     print("[DEBUG] first 10 robot_path points:\n", robot_path[:10])
    full_rotation_time = calculate_time_for_full_rotation(robot_path, fps)

    # --- 3) Compute path similarity to reference track path ---
    reference_path = np.load(reference_track_npy_path)
    # print("[DEBUG] reference_path shape:", reference_path.shape)
    # print("[DEBUG] reference_path min, max:\n",
    #     reference_path.min(axis=0),
    #     reference_path.max(axis=0))

    similarity_percent = calculate_path_similarity(
        robot_path,
        reference_path,
        method=path_similarity_method,
        auto_tune=auto_tune,
        clamp_percentage=clamp_percentage,
    )

    return (full_rotation_time, similarity_percent)


def auto_tune_parameters(
    robot_path: np.ndarray,
    reference_path: np.ndarray,
    clamp_percentage: float = 0.05,
) -> tuple:
    """
    Automates picking distance_baseline and clamp_dist by combining both paths' bounding boxes.

    Parameters
    ----------
    robot_path : np.ndarray
        The robot's path in track coordinates.
    reference_path : np.ndarray
        The reference track path from a .npy file (e.g., skeleton or offset contour).
    clamp_percentage : float, optional
        Percentage of the bounding box diagonal to use for clamping.

    Returns
    -------
    (distance_baseline, clamp_dist) : (float, float)
    """
    if robot_path.size == 0 or reference_path.size == 0:
        return 1000.0, 100.0

    combined = np.vstack([robot_path, reference_path])
    combined_min = combined.min(axis=0)  # [x_min, y_min]
    combined_max = combined.max(axis=0)  # [x_max, y_max]

    # Diagonal length
    bounding_diagonal = np.linalg.norm(combined_max - combined_min)

    if bounding_diagonal < 1.0:
        # If the bounding box is extremely tiny, fallback:
        return 1000.0, 100.0

    # Let's pick the baseline as the bounding box diagonal:
    distance_baseline = bounding_diagonal
    # And clamp_dist as ~10% of that diagonal, so large outliers don't explode the cost:
    clamp_dist = clamp_percentage * bounding_diagonal
    # print("[DEBUG] bounding_diagonal =", bounding_diagonal)
    # print("[DEBUG] distance_baseline (before returning) =", distance_baseline)
    # print("[DEBUG] clamp_dist (before returning) =", clamp_dist)

    return distance_baseline, clamp_dist


if __name__ == "__main__":
    # VIDEO_PATH = "assets/pos3.mov"
    # MODEL_PATH = "weights/best.pt"
    # SOURCE_KPTS = "keypoints/pos3_keypoints_from_camera.npy"
    # TARGET_KPTS = "keypoints/old_keypoints_from_digital.npy"
    # REF_PATH = "keypoints/all_path.npy"  # from track_processor.py
    # OUTPUT_VIDEO = "assets/benchmark_output.mp4"

    VIDEO_PATH = "assets/new_track/driver_4.MOV"
    MODEL_PATH = "weights/best.pt"
    TRACK_img_PATH = "assets/new_track/track-v2.jpg"
    SOURCE_KPTS = "keypoints/new_track/keypoints_from_camera.npy"
    TARGET_KPTS = "keypoints/new_track/keypoints_from_diagram.npy"
    REF_PATH = (
        "keypoints/new_track/all_path.npy"  # track_processor module output
    )
    OUTPUT_VIDEO = "assets/benchmark_output.mp4"

    rotation_time, sim_score = benchmark_robot_performance(
        video_path=VIDEO_PATH,
        detection_model_path=MODEL_PATH,
        track_img_path=TRACK_img_PATH,
        source_keypoints_path=SOURCE_KPTS,
        target_keypoints_path=TARGET_KPTS,
        reference_track_npy_path=REF_PATH,
        output_video_path=OUTPUT_VIDEO,
        confidence_threshold=0.85,
        shift_ratio=0.003,
        path_similarity_method="dtw",
        show_live=True,
        auto_tune=True,
        clamp_percentage=0.2,
    )
    video_name = os.path.basename(VIDEO_PATH).split(".")[0]
    print(f"Benchmarking for {video_name} -> ")
    if rotation_time < 0:
        print("Robot did not return to start point within threshold.")
    else:
        print(f"Time for full rotation: {rotation_time:.2f} seconds")

    print(f"Path similarity: {sim_score:.2f}%")
