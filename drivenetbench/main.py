"""Main module for the DriveNet benchmarking application."""

import os
from collections import deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from drivenetbench.utilities.config import get_config
from drivenetbench.utilities.utils import (
    load_and_preprocess_points,
    path_checker,
    path_fixer,
)
from drivenetbench.view_transformer import ViewTransformer

_, MODEL_PATH = path_checker(path_fixer(get_config("detection_model_path")))
detection_model = YOLO(MODEL_PATH)
detection_model.conf = 0.85

box_annotator = sv.OrientedBoxAnnotator(color=sv.Color.GREEN)
label_annotator = sv.LabelAnnotator(
    color=sv.Color.GREEN, text_position=sv.Position.TOP_LEFT
)

source_keypoints = load_and_preprocess_points(
    path_checker(path_fixer(get_config("keypoints_npy_path_source")))[1]
)
target_keypoints = load_and_preprocess_points(
    path_checker(path_fixer(get_config("keypoints_npy_path_target")))[1]
)
view_transformer = ViewTransformer(
    source=source_keypoints, target=target_keypoints
)

cap = cv2.VideoCapture("assets/pos3.mov")
cap_fps = cap.get(cv2.CAP_PROP_FPS)
# centroid_deque = deque(maxlen=100)
centroid_deque = []
track = cv2.imread(
    path_checker(path_fixer(get_config("digital_design_path")))[1]
)
track_height, track_width = track.shape[:2]
cap_height, cap_width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH
)

TRACK_SCALE = cap_height / track_height

output_frame_size = (
    int(cap_width + track_width * TRACK_SCALE),
    int(cap_height),
)  # (height, width)

SHIFT_RATIO = 0.0275

video_writer = cv2.VideoWriter(
    os.path.join("assets", f"pos3_output_shifted_{SHIFT_RATIO}.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    cap_fps,
    output_frame_size,
)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = detection_model(frame)
    annotated_track = track.copy()
    if results[0].obb is not None:
        for box in results[0].obb:
            if box.conf[0] < detection_model.conf:
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
            cv2.polylines(
                frame, [points], isClosed=True, color=(0, 255, 0), thickness=2
            )
            polylines_centoid = np.mean(points, axis=0)
            y_dif = frame.shape[0] - polylines_centoid[1]
            shift = int(y_dif * SHIFT_RATIO)
            polylines_centoid[1] += shift
            cv2.circle(
                frame,
                tuple(polylines_centoid.astype(int)),
                10,
                (0, 255, 0),
                -1,
            )
            transformed_centroid = view_transformer.transform_points(
                polylines_centoid[np.newaxis, ...]
            )
            centroid_deque.append(np.squeeze(transformed_centroid))
            colors = (
                [(x, 0, 0) for x in np.linspace(0, 255, len(centroid_deque))]
                if len(centroid_deque) > 1
                else [(255, 0, 0)]
            )
            for i, centroid in enumerate(centroid_deque):
                cv2.circle(
                    annotated_track,
                    tuple(centroid.astype(int)),
                    20,
                    colors[i],
                    # (255, 0, 0),
                    -1,
                )
            cv2.circle(
                annotated_track,
                tuple(np.squeeze(transformed_centroid).astype(int)),
                50,
                (0, 255, 0),
                -1,
            )

            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{detection_model.names[cls]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    resized_track = cv2.resize(
        annotated_track,
        (
            int(track_width * TRACK_SCALE),
            int(cap_height),
        ),
    )
    output_frame = np.concatenate((frame, resized_track), axis=1)
    assert (
        output_frame.shape[1],
        output_frame.shape[0],
    ) == output_frame_size, (
        f"Expected frame size {output_frame_size}, "
        f"got {(output_frame.shape[1], output_frame.shape[0])}."
    )
    cv2.imshow("Track", output_frame)
    video_writer.write(output_frame)
    # cv2.imshow("Track", annotated_track)
    # cv2.imshow("YOLOv11 OBB Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
video_writer.release()
cap.release()
