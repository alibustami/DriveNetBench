import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(
    "C:/Users/beto/Documents/YOLOv8/ackbotNano/runs/obb/train/weights/best.pt"
)
model.conf = 0.90
model.iou = 0.3

video_path = "C:/Users/beto/Documents/YOLOv8/ackbotNano/videos/test.mov"
track_image_path = "C:/Users/beto/Documents/YOLOv8/ackbotNano/videos/track.jpg"

track_image = cv2.imread(track_image_path)
if track_image is None:
    print("Error: Could not load track image.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

target_width = 640
target_height = 480

track_image = cv2.resize(track_image, (target_width, target_height))

# Perspective control values
top_width_factor = 1.3
bottom_width_factor = 0.55
# Use 1 to disable warps
# top_width_factor = 1
# bottom_width_factor = 1
strength_factor = 0.033


def get_destination_points():
    return np.float32(
        [
            [target_width * (1 - top_width_factor) / 2, 0],
            [target_width * (1 + top_width_factor) / 2, 0],
            [target_width * (1 - bottom_width_factor) / 2, target_height],
            [target_width * (1 + bottom_width_factor) / 2, target_height],
        ]
    )


source_points = np.float32(
    [
        [0, 0],
        [target_width, 0],
        [0, target_height],
        [target_width, target_height],
    ]
)

destination_points = get_destination_points()
homography_matrix = cv2.getPerspectiveTransform(
    source_points, destination_points
)

anchor_point = np.array([target_width // 2, target_height])

video_points = []
track_points = []
selecting_points = True


def select_points(event, x, y, flags, param):
    global selecting_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if param == "video" and len(video_points) < 4:
            video_points.append((x, y))
            print(f"Video point selected: {x}, {y}")
        elif param == "track" and len(track_points) < 4:
            track_points.append((x, y))
            print(f"Track point selected: {x}, {y}")
        if len(video_points) == 4 and len(track_points) == 4:
            selecting_points = False
            cv2.destroyWindow("Select Points on Video")
            cv2.destroyWindow("Select Points on Track")


ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from video.")
    cap.release()
    exit()

frame = cv2.resize(frame, (target_width, target_height))
warped_frame = cv2.warpPerspective(
    frame, homography_matrix, (target_width, target_height)
)

cv2.imshow("Select Points on Video", warped_frame)
cv2.imshow("Select Points on Track", track_image)
cv2.setMouseCallback("Select Points on Video", select_points, param="video")
cv2.setMouseCallback("Select Points on Track", select_points, param="track")

print(
    "Click 4 points on the video and 4 corresponding points on the track image."
)
while selecting_points:
    cv2.imshow("Select Points on Video", warped_frame)
    cv2.imshow("Select Points on Track", track_image)
    cv2.waitKey(1)

video_points_np = np.float32(video_points)
track_points_np = np.float32(track_points)
video_to_track_homography, _ = cv2.findHomography(
    video_points_np, track_points_np
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (target_width, target_height))
    warped_frame = cv2.warpPerspective(
        frame, homography_matrix, (target_width, target_height)
    )
    track_display = track_image.copy()

    results = model(warped_frame)
    if results[0].obb is not None:
        for box in results[0].obb:
            corners = box.xyxyxyxy.cpu().numpy().squeeze()
            x_center = int(np.mean(corners[:, 0]))
            y_center = int(np.mean(corners[:, 1]))
            car_center = np.array([x_center, y_center])

            line_vector = anchor_point - car_center
            line_length = np.linalg.norm(line_vector)
            line_direction = line_vector / line_length

            if y_center < target_height * 0.5:
                scaling_factor = 1 - (y_center / (target_height * 0.5))
                shifted_point = (
                    car_center
                    + (scaling_factor * strength_factor) * line_vector
                )
            else:
                shifted_point = car_center

            cv2.circle(
                warped_frame,
                (int(shifted_point[0]), int(shifted_point[1])),
                5,
                (0, 255, 0),
                -1,
            )
            cv2.line(
                warped_frame,
                tuple(anchor_point),
                tuple(car_center),
                (255, 0, 0),
                2,
            )

            shifted_point_homogeneous = np.array(
                [shifted_point[0], shifted_point[1], 1.0]
            )
            mapped_point = (
                video_to_track_homography @ shifted_point_homogeneous
            )
            mapped_point /= mapped_point[2]
            mapped_point = (int(mapped_point[0]), int(mapped_point[1]))

            cv2.circle(track_display, mapped_point, 5, (0, 0, 255), -1)

    combined_display = np.hstack((warped_frame, track_display))
    cv2.imshow("Video and Track Side by Side", combined_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
