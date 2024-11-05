import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("weights/best.pt")

model.conf = 0.75

video_path = "assets/IMG_4183.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    if results[0].obb is not None:
        for box in results[0].obb:
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

            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    cv2.imshow("YOLOv11 OBB Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
