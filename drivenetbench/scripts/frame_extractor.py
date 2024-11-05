import cv2

VIDEO_PATH = "assets/pos2.mov"

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

cv2.imwrite("assets/pos2_frame_capture.png", frame)
