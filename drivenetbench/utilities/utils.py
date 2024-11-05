"""Utility functions for the project."""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import supervision as sv


def path_fixer(path: Path, return_root: bool = True) -> str:
    """Fix the path to be compatible with the OS.

    Parameters
    ----------
    path : Path
        The path to fix.
    return_root : bool
        Whether to return the root path or not.

    Returns
    -------
    str
        The fixed path.
    """
    assert os.path.exists(path), f"Path {path} does not exist"
    if return_root:
        current_dir = Path(os.path.dirname(__file__))
        root = current_dir.parent.parent
        input_path = Path(os.path.abspath(path))
        input_path = str(input_path).split(os.sep)
        root_index = input_path.index("DriveNetBench")
        return os.path.join(root, os.sep.join(input_path[root_index + 1 :]))
    return (
        path.replace("/", os.sep).replace("//", os.sep).replace("\\", os.sep)
    )


def path_checker(
    path: Union[Path, List[Union[Path, str]]], break_if_not_found: bool = True
) -> Tuple[bool, Union[Path, List[Union[Path, str]]]]:
    """Check if the path exists.

    Parameters
    ----------
    path : Union[Path, List[Union[Path, str]]]
        The path to check, which can be a single path or a list of paths.
    break_if_not_found : bool
        Whether to break if the path is not found.

    Returns
    -------
    Tuple[bool, Union[Path, List[Union[Path, str]]]]
        Whether the path exists and the path itself.
    """

    def _check_single_path(path: Path):
        if not os.path.exists(path):
            if break_if_not_found:
                raise FileNotFoundError(f"Path {path} does not exist")
            return False, path
        return True, path

    if isinstance(path, list):
        return all(_check_single_path(p) for p in path), path
    return _check_single_path(path)


def check_if_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook.

    Returns
    -------
    bool
        Whether the code is running in a Jupyter notebook.
    """
    return "ipykernel" in sys.modules


def load_and_preprocess_points(points_path: Path) -> sv.KeyPoints:
    """Load and preprocess the points.

    Parameters
    ----------
    points_path : Path
        The path to the points file.

    Returns
    -------
    sv.KeyPoints
        The preprocessed points.
    """
    points = np.load(points_path)
    points = points[np.newaxis, ...]
    points = sv.KeyPoints(points)
    return points
