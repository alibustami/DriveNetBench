"""View transformer module."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


class ViewTransformer:
    def __init__(
        self, source: npt.NDArray[np.float32], target: npt.NDArray[np.float32]
    ) -> None:
        """Initialize the ViewTransformer.

        Parameters
        ----------
        source : npt.NDArray[np.float32]
            The source points.
        target : npt.NDArray[np.float32]
            The target points.
        """
        if source.xy.shape != target.xy.shape:
            raise ValueError("Source and target must have the same shape.")

        source_pts = source.xy.copy()
        target_pts = target.xy.copy()
        source_pts = source_pts.squeeze(axis=0)
        target_pts = target_pts.squeeze(axis=0)

        if source_pts.shape[1] != 2:
            raise ValueError(
                "Source and target points must be 2D coordinates."
            )

        self.source_pts = source_pts.astype(np.float32)
        self.target_pts = target_pts.astype(np.float32)
        self.m, _ = cv2.findHomography(self.source_pts, self.target_pts)

        if self.m is None:
            raise ValueError("Homography matrix must could not be calculated.")
        self.avg_error = None
        self.accumulated_error = None

    def transform_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray:
        """Transform the points.

        Parameters
        ----------
        points : npt.NDArray[np.float32]
            The points to transform.

        Returns
        -------
        npt.NDArray
            The transformed points.
        """
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_image(
        self, image: npt.NDArray[np.uint8], resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """Transform the image.

        Parameters
        ----------
        image : npt.NDArray[np.uint8]
            The image to transform.
        resolution_wh : Tuple[int, int]
            The resolution of the image.

        Returns
        -------
        npt.NDArray
            The transformed image.
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or color.")

        return cv2.warpPerspective(image, self.m, resolution_wh)

    def calculate_transformation_error(
        self,
        source: Optional[npt.NDArray] = None,
        target: Optional[npt.NDArray] = None,
        is_source_transformed: bool = False,
    ) -> Tuple[float, float]:
        """Calculate the transformation error.

        Parameters
        ----------
        source : npt.NDArray, optional
            The source points (default is None), if None, the source points used in the initialization are used.
        target : npt.NDArray, optional
            The target points (default is None), if None, the target points used in the initialization are used
        is_source_transformed : bool, optional
            Whether the source points are transformed (default is False).

        Returns
        -------
        Tuple[float, float]
            The average error and the accumulated error.
        """
        source = source if isinstance(source, np.ndarray) else self.source_pts
        target = target if isinstance(target, np.ndarray) else self.target_pts

        source = (
            source if is_source_transformed else self.transform_points(source)
        )

        self.accumulated_error = 0
        for source_pt, target_pt in zip(source, target):
            error = np.linalg.norm(source_pt - target_pt)
            self.accumulated_error += error

        self.avg_error = self.accumulated_error / len(source)
        return self.avg_error, self.accumulated_error


if __name__ == "__main__":
    import numpy as np
    import supervision as sv

    original_source_keypoints = np.load(
        "/Users/aalbustami/UMD/BIMI/projects/broverette/DriveNetBench/keypoints/keypoints_from_camera.npy"
    )
    target_keypoints = np.load(
        "/Users/aalbustami/UMD/BIMI/projects/broverette/DriveNetBench/keypoints/keypoints_from_digital.npy"
    )
    source_keypoints = original_source_keypoints[np.newaxis, ...]
    target_keypoints = target_keypoints[np.newaxis, ...]

    source_keypoints = sv.KeyPoints(source_keypoints)
    target_keypoints = sv.KeyPoints(target_keypoints)
    transformer = ViewTransformer(
        source=source_keypoints, target=target_keypoints
    )

    print(transformer.calculate_transformation_error())
