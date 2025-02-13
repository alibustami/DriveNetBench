import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrackProcessor:
    """
    A class to process a track image in order to extract either:
      - All center lines (via skeletonization), OR
      - The single offset center path of the outer boundary (via erosion).

    Usage:
    ------
        processor = TrackProcessor(
            image_path="track.png",
            color_hsv=(330, 23, 84),        # H,S,V in [0..360,0..100,0..100]
            output_image_path="annotated.png",
            output_npy_path="center_path.npy",
            only_offset_outer=False
        )
        processor.process()

    This will produce:
      - An annotated image on disk with the outer boundary (green) and center line(s) (red).
      - A .npy file containing the center path as an (N,2) array of (x, y) coordinates.
    """

    def __init__(
        self,
        image_path: str,
        color_hsv: tuple,
        output_image_path: str,
        output_npy_path: str,
        only_offset_outer: bool = False,
    ):
        """
        Parameters
        ----------
        image_path : str
            Path to the input track image.
        color_hsv : tuple
            The HSV color of the track in the color-picker scale:
            (H in [0..360], S in [0..100], V in [0..100]).
        output_image_path : str
            Path to save the annotated result (PNG or JPG).
        output_npy_path : str
            Path to save the center path as a .npy 2D array.
        only_offset_outer : bool
            If False, skeletonize the entire track to get all center lines.
            If True, offset just the largest outer boundary to find its center path.
        """
        self.image_path = image_path
        self.color_hsv_picker = color_hsv  # e.g. (330, 23, 84)
        self.output_image_path = output_image_path
        self.output_npy_path = output_npy_path
        self.only_offset_outer = only_offset_outer

    @staticmethod
    def _skeletonize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
        """
        Morphologically thins a binary mask (255=foreground, 0=background)
        to obtain a 1-pixel-wide skeleton.
        """
        skeleton = np.zeros_like(binary_mask, dtype=np.uint8)
        temp = binary_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(temp, kernel)
            opened = cv2.dilate(eroded, kernel)
            # Pixels that disappear after an opening => part of the skeleton
            temp2 = cv2.subtract(temp, opened)
            skeleton = cv2.bitwise_or(skeleton, temp2)
            temp = eroded.copy()

            if cv2.countNonZero(temp) == 0:
                break

        return skeleton

    @staticmethod
    def _estimate_track_width(mask: np.ndarray) -> float:
        """
        Estimates the average track width in pixels for the given track mask.

        1. Skeletonize the mask => center lines.
        2. Distance transform => each pixel distance to boundary.
        3. At skeleton pixels, distance ~ half local thickness.
        4. Return 2 * median(distance at skeleton) => approx. track width.
        """
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        skeleton = TrackProcessor._skeletonize_binary_mask(mask)

        skel_pixels = np.where(skeleton > 0)
        dist_vals = dist[skel_pixels]

        if len(dist_vals) == 0:
            # No skeleton => track not found or too thin
            return 0.0

        median_dist = np.median(dist_vals)
        track_width_est = 2.0 * median_dist
        return track_width_est

    @staticmethod
    def _convert_picker_hsv_to_opencv(hsv_picker: tuple) -> tuple:
        """
        Converts HSV from color picker scale (H in [0..360], S in [0..100], V in [0..100])
        into OpenCV's HSV scale (H in [0..179], S in [0..255], V in [0..255]).

        Returns (H_cv, S_cv, V_cv).
        """
        h_picker, s_picker, v_picker = hsv_picker

        # Hue:   [0..360] -> [0..179]
        h_cv = int(np.clip(h_picker * 179.0 / 360.0, 0, 179))
        # Saturation: [0..100] -> [0..255]
        s_cv = int(np.clip(s_picker * 2.55, 0, 255))
        # Value: [0..100] -> [0..255]
        v_cv = int(np.clip(v_picker * 2.55, 0, 255))

        return (h_cv, s_cv, v_cv)

    def process(self):
        """
        Orchestrates the track detection, either skeletonizing for all center lines
        or offsetting for the outer boundary center path. Saves results to disk.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(
                f"Could not read image from path: {self.image_path}"
            )

        (h_cv, s_cv, v_cv) = self._convert_picker_hsv_to_opencv(
            self.color_hsv_picker
        )

        # a small hue margin; adjust if needed
        h_margin = 15
        # We'll define minimal saturation/value floors to avoid overshooting
        s_min = max(s_cv - 40, 0)
        v_min = max(v_cv - 40, 0)
        lower_bound = np.array([max(h_cv - h_margin, 0), s_min, v_min])
        upper_bound = np.array([min(h_cv + h_margin, 179), 255, 255])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Optional
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            raise ValueError(
                "No contours found for the specified track color!"
            )
        largest_contour = max(contours, key=cv2.contourArea)
        outer_contour_xy = largest_contour.reshape(-1, 2)  # shape (N,2)

        if not self.only_offset_outer:
            # --- (A) Skeletonize for all center lines ---
            skeleton = self._skeletonize_binary_mask(mask)
            center_pixels = np.column_stack(
                np.where(skeleton > 0)
            )  # (row, col)
            center_line_xy = center_pixels[:, ::-1]  # (x,y)
        else:
            # --- (B) Offset only the outer boundary by half the track width ---
            track_width_est = self._estimate_track_width(mask)
            if track_width_est < 1.0:
                print(
                    "Warning: Estimated track width is extremely small. Defaulting to 1 pixel."
                )
                track_width_est = 1.0

            half_width = int(round(track_width_est / 2))
            outer_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(outer_mask, [largest_contour], -1, 255, -1)

            erode_ksize = 2 * half_width + 1
            erode_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize)
            )
            offset_mask = cv2.erode(outer_mask, erode_kernel, iterations=1)
            offset_contours, _ = cv2.findContours(
                offset_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not offset_contours:
                raise ValueError(
                    "Offset contour not found. Track may be too narrow for that offset."
                )
            offset_contour = max(offset_contours, key=cv2.contourArea)
            center_line_xy = offset_contour.reshape(-1, 2)

        cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
        for x, y in center_line_xy:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.imwrite(self.output_image_path, image)
        print(f"Annotated image saved to: {self.output_image_path}")

        np.save(self.output_npy_path, center_line_xy)
        print(
            f"Center path (shape {center_line_xy.shape}) saved to: {self.output_npy_path}"
        )

        return outer_contour_xy, center_line_xy


if __name__ == "__main__":
    # Example color: H=330, S=23, V=84 from color picker scale
    # We want all skeleton lines, not just the offset.
    # this tool was used: https://pinetools.com/image-color-picker to get the HSV values

    processor = TrackProcessor(
        image_path="assets/track-v2.jpg",
        color_hsv=(330, 23, 84),  # (H,S,V) in [0..360,0..100,0..100]
        output_image_path="assets/annotated.png",
        output_npy_path="keypoints/all_path.npy",
        only_offset_outer=False,
    )
    outer_points, center_points = processor.process()
    print(f"Outer contour has {len(outer_points)} points.")
    print(f"Center path has {len(center_points)} points.")
