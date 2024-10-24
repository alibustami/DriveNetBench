"""Utility functions for the project."""

import os
from pathlib import Path


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
        root = current_dir.parent
        input_path = Path(os.path.abspath(path))
        input_path = str(input_path).split(os.sep)
        root_index = input_path.index("mem-leak-miner")
        return os.path.join(root, os.sep.join(input_path[root_index + 1 :]))
    return (
        path.replace("/", os.sep).replace("//", os.sep).replace("\\", os.sep)
    )
