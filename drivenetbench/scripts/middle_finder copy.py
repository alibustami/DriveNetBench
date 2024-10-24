import cv2
import numpy as np

def skeletonize_binary_mask(binary_mask):
    """
    Morphologically thins a binary mask (255=foreground, 0=background) until
    we get a skeleton (1-pixel-wide centerline).
    """
    skeleton = np.zeros_like(binary_mask, dtype=np.uint8)
    temp = binary_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(temp, kernel)
        opened = cv2.dilate(eroded, kernel)
        # Pixels that disappear after an opening are added to skeleton
        temp2 = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, temp2)
        temp = eroded.copy()

        if cv2.countNonZero(temp) == 0:
            break

    return skeleton

def estimate_track_width(mask):
    """
    Estimates the average track width in pixels for the given track mask.

    Steps:
    1. Skeletonize the mask => a thin centerline.
    2. Distance transform (L2) on the mask => each pixel = distance to boundary.
    3. For each skeleton pixel, distance ~ half the local track thickness.
    4. Return 2 * median(distance at skeleton) as the overall track width.
    """
    # Distance transform of the track region
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Skeleton: center line of the track
    skel = skeletonize_binary_mask(mask)

    # Gather distance values at skeleton pixels
    skel_pixels = np.where(skel > 0)
    dist_vals = dist[skel_pixels]  # array of distances at center line

    if len(dist_vals) == 0:
        # No skeleton => track not found or too thin
        return 0.0

    # Local thickness = 2 * distance at skeleton pixel
    # We take the median for robustness
    median_dist = np.median(dist_vals)
    track_width_est = 2.0 * median_dist

    return track_width_est

def find_contour_and_offset_centerline(image_path, output_path="annotated_track.png"):
    """
    1. Reads an image of a pink/red track on a light background.
    2. Thresholds in HSV to isolate the track.
    3. Finds largest outer contour => main track boundary.
    4. Estimates the track width from the distance transform at the skeleton.
    5. Offsets (erodes) the outer boundary by ~half that width => center path.
    6. Draws both the outer contour (green) and the offset path (red).
    7. Saves the annotated image.

    Returns
    -------
    contour_xy : (N,2) np.ndarray
        Outer boundary points (x,y).
    center_xy : (M,2) np.ndarray
        Inset center path from the offset boundary (x,y).
    track_width_est : float
        Estimated track width in pixels.
    """
    #----------------------------------------------------
    # 1. Read image, convert to HSV
    #----------------------------------------------------
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #----------------------------------------------------
    # 2. Create a binary mask for the pink/red region
    #    (Adjust these bounds to match your track color!)
    #----------------------------------------------------
    lower_pink = np.array([150, 40,  80])    # e.g. ~H=150..179, S=40..255, V=80..255
    upper_pink = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Morphological close => fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #----------------------------------------------------
    # 3. Find the largest outer contour => track boundary
    #----------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found for the pink track!")
    largest_contour = max(contours, key=cv2.contourArea)
    contour_xy = largest_contour.reshape(-1, 2)

    #----------------------------------------------------
    # 4. Estimate track width automatically
    #----------------------------------------------------
    track_width_est = estimate_track_width(mask)
    if track_width_est <= 1.0:
        print("Warning: Estimated track width is very small! Defaulting to 1 pixel.")
        track_width_est = 1.0
    half_width = int(round(track_width_est / 2))

    #----------------------------------------------------
    # 5. Offset (erode) the outer boundary by half width
    #----------------------------------------------------
    # Draw only the largest contour onto a blank mask => outer region
    outer_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(outer_mask, [largest_contour], -1, 255, -1)

    # Erode by half_width
    # We'll use an ellipse kernel about that size
    # (we add +1 to ensure it's an odd dimension)
    erode_ksize = 2 * half_width + 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    offset_mask = cv2.erode(outer_mask, erode_kernel, iterations=1)

    # Retrieve offset boundary
    offset_contours, _ = cv2.findContours(offset_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not offset_contours:
        raise ValueError("Offset boundary not found. Possibly track is too narrow for that offset!")
    # Pick largest offset contour (in case multiple pieces remain)
    offset_contour = max(offset_contours, key=cv2.contourArea)
    center_xy = offset_contour.reshape(-1, 2)

    #----------------------------------------------------
    # 6. Draw the outer boundary (green) + offset (red)
    #----------------------------------------------------
    # Outer boundary
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    # Offset (center) contour
    for (x, y) in center_xy:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    #----------------------------------------------------
    # 7. Save annotated image
    #----------------------------------------------------
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to: {output_path}")
    print(f"Estimated track width: {track_width_est:.2f} pixels.")

    return contour_xy, center_xy, track_width_est

if __name__ == "__main__":
    # Example usage
    input_img  = "assets/track-v2.jpg"            # Replace with your track image path
    output_img = "assets/annotated_track2.png"

    outer_pts, center_pts, est_width = find_contour_and_offset_centerline(
        image_path=input_img,
        output_path=output_img
    )

    print(f"Outer contour has {outer_pts.shape[0]} points.")
    print(f"Offset 'center' contour has {center_pts.shape[0]} points.")
    print(f"Track width was estimated as ~{est_width:.2f} px.")
