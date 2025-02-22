import cv2
import numpy as np

track_path = "assets/new_track/track-v2.jpg"
track = cv2.imread(track_path)

points_path = "keypoints/new_track/all_path.npy"
points = np.load(points_path)

for i, point in enumerate(points):
    x, y = point
    cv2.circle(track, (int(x), int(y)), 3, (255, 0, 0), -1)


while True:
    cv2.namedWindow("Track", cv2.WINDOW_NORMAL)
    cv2.imshow("Track", track)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
