from enum import Enum

import numpy as np
import numpy.typing as npt

import drivenetbench.utilities.config as configs


class Methods(Enum):
    """Enum for the methods of path similarity."""

    DTW = 1
    FRECHET = 2


class SimilarityCalculator:
    """Class for calculating the similarity between two paths."""

    def __init__(self):
        """Initialize the SimilarityCalculator."""
        method = configs.get_config("benchmarker.path_similarity.method")
        self.method = Methods[method.upper()]

        self.auto_tune = configs.get_config(
            "benchmarker.path_similarity.auto_tune"
        )
        if self.auto_tune:
            self.clamp_percentage = configs.get_config(
                "benchmarker.path_similarity.clamp_percentage"
            )
            self.clamp_distance = None
            self.distance_baseline = None
        else:
            self.clamp_distance = configs.get_config(
                "benchmarker.path_similarity.clamp_distance"
            )
            self.distance_baseline = configs.get_config(
                "benchmarker.path_similarity.distance_baseline"
            )

    def calculate_path_similarity(
        self,
        robot_path: npt.NDArray,
        reference_path: npt.NDArray,
    ) -> float:
        """Calculate the similarity between two paths.

        Parameters
        ----------
        robot_path : npt.NDArray
            The robot path.
        reference_path : npt.NDArray
            The reference path.

        Returns
        -------
        float
            The similarity percentage.
        """
        self.robot_path = robot_path
        self.reference_path = reference_path

        if self.auto_tune:
            # sets the clamp_distance and distance_baseline
            self._auto_tune_parameters()

        functions_map = {
            Methods.DTW: self._dtw_distance,
            Methods.FRECHET: self._frechet_distance,
        }

        distance = functions_map[self.method](
            self.robot_path, self.reference_path, self.clamp_distance
        )

        raw_ratio = 1.0 - (distance / self.distance_baseline)
        similarity_percent = 100.0 * max(0.0, raw_ratio)

        similarity_percent = min(similarity_percent, 100.0)

        return similarity_percent

    def _auto_tune_parameters(self):
        """Automatically tune the parameters for the similarity calculation."""
        if self.robot_path.size == 0 or self.reference_path.size == 0:
            return 1000.0, 100.0

        combined = np.vstack([self.robot_path, self.reference_path])
        combined_min = combined.min(axis=0)
        combined_max = combined.max(axis=0)

        bounding_diagonal = np.linalg.norm(combined_max - combined_min)

        if bounding_diagonal < 1.0:
            # If the bounding box is extremely tiny, fallback:
            return 1000.0, 100.0

        self.distance_baseline = bounding_diagonal
        # And clamp_distance as ~10% of that diagonal, so large outliers don't explode the cost:
        self.clamp_distance = self.clamp_percentage * bounding_diagonal

    @staticmethod
    def _dtw_distance(
        path_a: npt.NDArray, path_b: npt.NDArray, clamp_dist: float = None
    ) -> float:
        """Compute DTW distance between two 2D paths (N vs M). Lower = more similar.

        Parameters
        ----------
        path_a : npt.NDArray
            First path.
        path_b : npt.NDArray
            Second path.
        clamp_dist : float, optional
            If provided, each pairwise distance is clamped to this max.

        Returns
        -------
        float
            Total DTW cost.
        """
        n, m = len(path_a), len(path_b)
        dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(path_a[i - 1] - path_b[j - 1])
                if clamp_dist is not None:
                    cost = min(cost, clamp_dist)

                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # deletion
                    dtw_matrix[i, j - 1],  # insertion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        # The raw sum cost:
        raw_sum = float(dtw_matrix[n, m])
        # Normalize it by the path size:
        normalized_dtw = raw_sum / (n + m)

        return normalized_dtw

    @staticmethod
    def _frechet_distance(
        path_a: npt.NDArray, path_b: npt.NDArray, clamp_dist: float = None
    ) -> float:
        """Iterative Frechet distance between two 2D paths. Lower = more similar.

        Parameters
        ----------
        path_a : npt.NDArray
            First path.
        path_b : npt.NDArray
            Second path.
        clamp_dist : float, optional
            If not None, will clamp each pairwise distance to at most this value
            to soften the penalty for large differences.

        Returns
        -------
        float
            Frechet distance.
        """
        n, m = len(path_a), len(path_b)
        # dp[i,j] will hold the Frechet distance up to path_a[:i+1], path_b[:j+1].
        dp = np.full((n, m), -1.0, dtype=np.float32)

        # Helper to compute local cost with clamp
        def local_dist(i, j):
            d = np.linalg.norm(path_a[i] - path_b[j])
            return min(d, clamp_dist)

        # Initialize first cell
        dp[0, 0] = local_dist(0, 0)

        # First row
        for j in range(1, m):
            dp[0, j] = max(dp[0, j - 1], local_dist(0, j))

        # First column
        for i in range(1, n):
            dp[i, 0] = max(dp[i - 1, 0], local_dist(i, 0))

        # Fill the rest
        for i in range(1, n):
            for j in range(1, m):
                cost_ij = local_dist(i, j)
                dp[i, j] = max(
                    min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]), cost_ij
                )

        return float(dp[n - 1, m - 1])
