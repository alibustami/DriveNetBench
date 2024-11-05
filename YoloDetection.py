import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Broverette dimensions in meters
height_m = 0.1311  # height in meters
width_m = 0.1685   # width in meters
length_m = 0.23529 # length in meters

# Load YOLOv8 model with OBB support
model = YOLO('C:/Users/beto/Documents/YOLOv8/ackbotNano/runs/obb/train/weights/best.pt')
model.conf = 0.85  
model.iou = 0.3    

video_path = 'C:/Users/beto/Documents/YOLOv8/ackbotNano/videos/test.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
defined_point = (frame_width / 2, frame_height - 10)  #Point is used to try to find which side is visible 

center_points = deque(maxlen=5)  # Adjustable for smoother or faster transitions
prev_avg_center = None
scale_factor = 0.9  # Scale factor to shrink OBB closer to the object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    if results[0].obb is not None:
        for box in results[0].obb:
            corners = box.xyxyxyxy.cpu().numpy().squeeze()
            points = np.array([
                [int(corners[0][0]), int(corners[0][1])],
                [int(corners[1][0]), int(corners[1][1])],
                [int(corners[2][0]), int(corners[2][1])],
                [int(corners[3][0]), int(corners[3][1])]
            ], dtype=np.int32)

            # Calculate the center of the OBB
            x_center = int(np.mean(corners[:, 0]))
            y_center = int(np.mean(corners[:, 1]))
            obb_center = np.array([x_center, y_center])

            # Adjust each corner toward the center by the scale factor for a tighter fit
            adjusted_points = []
            for point in points:
                adjusted_point = obb_center + scale_factor * (point - obb_center)
                adjusted_points.append(adjusted_point.astype(int))

            adjusted_points = np.array(adjusted_points, dtype=np.int32)

            # Draw a dot at each adjusted corner without connecting lines
            for (x, y) in adjusted_points:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1) 

            # Store the current center point
            car_center = (x_center, y_center)
            center_points.append(car_center)
            avg_center = (
                int(np.mean([pt[0] for pt in center_points])),
                int(np.mean([pt[1] for pt in center_points]))
            )

            # Calculate direction vector or use a default if no direction exists (Still testing)
            if prev_avg_center is not None:
                direction_vector = (avg_center[0] - prev_avg_center[0], avg_center[1] - prev_avg_center[1])
            else:
                direction_vector = (1, 0)  

            bbox_length = int(np.linalg.norm(adjusted_points[0] - adjusted_points[2]))
            width_pixels = bbox_length * (width_m / length_m)

            length = np.hypot(*direction_vector)
            if length != 0:
                norm_direction = (direction_vector[0] / length, direction_vector[1] / length)
                perp_vector = (-norm_direction[1], norm_direction[0])  # Perpendicular to direction

                base_center = avg_center

                f1 = (int(base_center[0] + norm_direction[0] * (bbox_length / 2) + perp_vector[0] * (width_pixels / 2)),
                      int(base_center[1] + norm_direction[1] * (bbox_length / 2) + perp_vector[1] * (width_pixels / 2)))
                f2 = (int(base_center[0] + norm_direction[0] * (bbox_length / 2) - perp_vector[0] * (width_pixels / 2)),
                      int(base_center[1] + norm_direction[1] * (bbox_length / 2) - perp_vector[1] * (width_pixels / 2)))
                r1 = (int(base_center[0] - norm_direction[0] * (bbox_length / 2) + perp_vector[0] * (width_pixels / 2)),
                      int(base_center[1] - norm_direction[1] * (bbox_length / 2) + perp_vector[1] * (width_pixels / 2)))
                r2 = (int(base_center[0] - norm_direction[0] * (bbox_length / 2) - perp_vector[0] * (width_pixels / 2)),
                      int(base_center[1] - norm_direction[1] * (bbox_length / 2) - perp_vector[1] * (width_pixels / 2)))

                base_points = {'f1': f1, 'f2': f2, 'r1': r1, 'r2': r2}
                for key, base_point in base_points.items():
                    min_distance = float('inf')
                    closest_corner = None
                    for corner in adjusted_points:
                        distance = np.linalg.norm(np.array(base_point) - corner)
                        if distance < min_distance:
                            min_distance = distance
                            closest_corner = corner
                    base_points[key] = tuple(closest_corner)

                front_center = ((base_points['f1'][0] + base_points['f2'][0]) // 2, (base_points['f1'][1] + base_points['f2'][1]) // 2)
                rear_center = ((base_points['r1'][0] + base_points['r2'][0]) // 2, (base_points['r1'][1] + base_points['r2'][1]) // 2)
                left_center = ((base_points['f2'][0] + base_points['r2'][0]) // 2, (base_points['f2'][1] + base_points['r2'][1]) // 2)
                right_center = ((base_points['f1'][0] + base_points['r1'][0]) // 2, (base_points['f1'][1] + base_points['r1'][1]) // 2)

                distances = {
                    'front': np.linalg.norm(np.array(front_center) - defined_point),
                    'rear': np.linalg.norm(np.array(rear_center) - defined_point),
                    'left': np.linalg.norm(np.array(left_center) - defined_point),
                    'right': np.linalg.norm(np.array(right_center) - defined_point)
                }

                # Determine the closest side
                closest_side = min(distances, key=distances.get)
                if closest_side == 'front':
                    center_point = front_center
                elif closest_side == 'rear':
                    center_point = rear_center
                elif closest_side == 'left':
                    center_point = left_center
                else:
                    center_point = right_center

                # Draw the center green dot on the closest side
                cv2.circle(frame, center_point, 5, (0, 255, 0), -1)  

            prev_avg_center = avg_center

    cv2.imshow('YOLOv8 OBBD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
