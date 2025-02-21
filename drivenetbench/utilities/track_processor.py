import logging
from ast import literal_eval
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import drivenetbench.utilities.config as configs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrackProcessor:
    """A class to process a track image in order to extract either:
    - All center lines (via skeletonization), OR
    - The single offset center path of the outer boundary (via erosion),
    THEN cluster the resulting points to remove noise.
    """

    def __init__(
        self,
        image_path: Optional[str] = None,
        color_hsv: Optional[tuple] = None,
        output_image_path: Optional[str] = None,
        output_npy_path: Optional[str] = None,
        only_offset_outer: Optional[bool] = None,
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
        self.image_path = (
            image_path
            if image_path is not None
            else configs.get_config("track_processor.image_path")
        )
        self.color_hsv_picker = (
            color_hsv
            if color_hsv is not None
            else configs.get_config("track_processor.color_hsv")
        )  # e.g. (330, 23, 84)
        self.color_hsv_picker = literal_eval(self.color_hsv_picker)
        self.output_image_path = (
            output_image_path
            if output_image_path is not None
            else configs.get_config("track_processor.output_image_path_export")
        )
        self.output_npy_path = (
            output_npy_path
            if output_npy_path is not None
            else configs.get_config("track_processor.output_npy_path_export")
        )
        self.only_offset_outer = (
            only_offset_outer
            if only_offset_outer is not None
            else configs.get_config("track_processor.only_offset_the_outer")
        )

    @staticmethod
    def _skeletonize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
        """Morphologically thins a binary mask (255=foreground, 0=background) to obtain a 1-pixel-wide skeleton.

        Parameters
        ----------
        binary_mask : np.ndarray
            A binary mask of the image to be skeletonized.

        Returns
        -------
        np.ndarray
            The skeletonized image.
        """
        # Ensure strictly 0 or 255
        binary_mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)

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
        """Estimates the average track width in pixels for the given track mask.

        Parameters
        ----------
        mask : np.ndarray
            A binary mask of the track (255=track, 0=background).

        Returns
        -------
        float
            The estimated track width in pixels.
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
        Converts HSV from color-picker scale (H in [0..360], S in [0..100], V in [0..100]) into OpenCV's HSV scale (H in [0..179], S in [0..255], V in [0..255]).

        Parameters
        ----------
        hsv_picker : tuple
            The HSV color in the color-picker scale.

        Returns
        -------
        tuple
            The HSV color in OpenCV's scale.
        """
        h_picker, s_picker, v_picker = hsv_picker

        # Hue:   [0..360] -> [0..179]
        h_cv = int(np.clip(h_picker * 179.0 / 360.0, 0, 179))
        # Saturation: [0..100] -> [0..255]
        s_cv = int(np.clip(s_picker * 2.55, 0, 255))
        # Value: [0..100] -> [0..255]
        v_cv = int(np.clip(v_picker * 2.55, 0, 255))

        return (h_cv, s_cv, v_cv)

    @staticmethod
    def _cluster_and_filter(
        points: np.ndarray,
        eps: float = 5.0,
        min_samples: int = 10,
        cluster_size_threshold: int = 30,
    ) -> np.ndarray:
        """Cluster 2D points (x,y) with DBSCAN, and filter out small clusters or noise.

        Parameters
        ----------
        points : np.ndarray
            The 2D points to be clustered.
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        cluster_size_threshold : int
            The minimum number of points required to keep a cluster.

        Returns
        -------
        np.ndarray
            The filtered points.
        """
        if len(points) == 0:
            return points

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(points)

        unique_labels, counts = np.unique(labels, return_counts=True)
        filtered_points = []

        for lbl, count in zip(unique_labels, counts):
            if lbl == -1:
                continue  # -1 is "noise" in DBSCAN
            if count >= cluster_size_threshold:
                filtered_points.append(points[labels == lbl])

        if not filtered_points:
            # If everything is small or noise, fallback to returning original
            return points

        return np.vstack(filtered_points)

    def process(self) -> tuple:
        """
        Orchestrates the track detection, either skeletonizing for all center lines or offsetting for the outer boundary center path. Then clusters the resulting points to remove noise. Saves results to disk.

        Returns
        -------
        tuple
            A tuple containing the outer contour points and the center path points.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(
                f"Could not read image from path: {self.image_path}"
            )

        (h_cv, s_cv, v_cv) = self._convert_picker_hsv_to_opencv(
            self.color_hsv_picker
        )

        h_margin = 15
        s_min = max(s_cv - 40, 0)
        v_min = max(v_cv - 40, 0)
        lower_bound = np.array([max(h_cv - h_margin, 0), s_min, v_min])
        upper_bound = np.array([min(h_cv + h_margin, 179), 255, 255])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

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
            skeleton = self._skeletonize_binary_mask(mask)
            center_pixels = np.column_stack(
                np.where(skeleton > 0)
            )  # (row, col)
            center_line_xy = center_pixels[:, ::-1]  # (x, y)
        else:
            track_width_est = self._estimate_track_width(mask)
            if track_width_est < 1.0:
                logger.warning("Estimated track width < 1. Defaulting to 1.")
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

        eps = configs.get_config("track_processor.dbscan.eps")
        min_samples = configs.get_config("track_processor.dbscan.min_samples")
        cluster_size_threshold = configs.get_config(
            "track_processor.dbscan.cluster_size_threshold"
        )
        center_line_xy = self._cluster_and_filter(
            center_line_xy,
            eps=eps,
            min_samples=min_samples,
            cluster_size_threshold=cluster_size_threshold,
        )

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

    # processor = TrackProcessor(
    #     image_path="assets/new_track/track-v2.jpg",
    #     color_hsv=(330, 23, 84),  # (H,S,V) in [0..360,0..100,0..100]
    #     output_image_path="assets/annotated.png",
    #     output_npy_path="keypoints/new_track/all_path.npy",
    #     only_offset_outer=False,
    # )
    # outer_points, center_points = processor.process()
    # print(f"Outer contour has {len(outer_points)} points.")
    # print(f"Center path has {len(center_points)} points.")

    processor = TrackProcessor()
    processor.process()
