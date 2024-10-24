import cv2
import numpy as np

def skeletonize_binary_mask(binary_mask):
    """
    Morphologically thins a binary mask (255=foreground, 0=background) until we
    get a skeleton (1-pixel-wide centerline).
    """
    skeleton = np.zeros_like(binary_mask, dtype=np.uint8)
    temp = binary_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(temp, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp2 = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, temp2)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton

def find_contour_and_draw_centerline(
    image_path,
    output_path="annotated_track.png"
):
    """
    1. Reads an image of a pink track (approx. H=330, S=23, V=84 in [0..360,0..100,0..100] color picker).
    2. Converts that color to OpenCV's HSV (H=165, S=~59, V=~214 in [0..179, 0..255, 0..255]).
    3. Uses a single HSV range around that color to create a mask of the pink region.
    4. Finds the largest outer contour (the track boundary).
    5. Skeletonizes the mask to find the track's centerline.
    6. Draws the outer contour (green) & centerline (red) on the image, then saves.
    """

    #----------------------------------------------------------
    # 1. Read the image and convert from BGR to HSV (OpenCV)
    #----------------------------------------------------------
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #--------------------------------------------------------------------
    # 2. Convert your picker's H=330, S=23, V=84 into OpenCV's HSV scale
    #
    #   In your picker:  H in [0..360],  S in [0..100],  V in [0..100]
    #   In OpenCV:       H in [0..179],  S in [0..255],  V in [0..255]
    #
    #   - H(opencv) = H(picker) * 179/360  =>  330*(179/360) ~ 164
    #   - S(opencv) = S(picker) * 255/100  =>  23 * 2.55 = ~58.65
    #   - V(opencv) = V(picker) * 255/100  =>  84 * 2.55 = ~214
    #
    # We'll pick a range around ~H=165, S=59, V=214 to detect the pink track.
    # Adjust these bounds if parts of the track are still missing.
    #--------------------------------------------------------------------
    center_h = 165
    center_s = 60
    center_v = 210

    # A +/- range around those values:
    h_range = 15
    s_min   = 30
    v_min   = 100

    # Lower and upper bounds in OpenCV HSV
    lower_pink = np.array([
        max(center_h - h_range, 0),   # H
        s_min,                        # S
        v_min                         # V
    ])
    upper_pink = np.array([
        min(center_h + h_range, 179),
        255,
        255
    ])

    #------------------------------------------------------------------
    # 3. Threshold the HSV image to isolate the pink track region
    #------------------------------------------------------------------
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Optionally apply morphological "closing" to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #--------------------------
    # 4. Find the track contour
    #--------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found for the pink track!")

    # Take largest by area
    largest_contour = max(contours, key=cv2.contourArea)
    contour_xy = largest_contour.reshape(-1, 2)

    #-------------------------
    # 5. Skeletonize the mask
    #-------------------------
    skeleton = skeletonize_binary_mask(mask)
    center_pixels = np.column_stack(np.where(skeleton > 0))  # (row, col)
    centerline_points = center_pixels[:, ::-1]               # (x, y)

    #------------------------------------------------
    # Draw the outer contour (green) and center (red)
    #------------------------------------------------
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    for (x, y) in centerline_points:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    #-------------------------------
    # 6. Save annotated result
    #-------------------------------
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to: {output_path}")

    return contour_xy, centerline_points

if __name__ == "__main__":
    input_image  = "assets/track-v2.jpg"           # <-- Put your image path here
    output_image = "assets/annotated_track.png"
    contour_pts, center_pts = find_contour_and_draw_centerline(input_image, output_image)
    print(f"Contour has {contour_pts.shape[0]} points.")
    print(f"Centerline has {len(center_pts)} points.")
