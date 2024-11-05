"""Command Line Interface for DriveNetBench."""

from pathlib import Path
from typing import Optional

from typer import Typer

from drivenetbench.geometry_definer import GeometryDefiner
from drivenetbench.utilities.config import get_config
from drivenetbench.utilities.utils import path_checker, path_fixer

app = Typer()


@app.command()
def define_polygon(
    output_name: str,
    source: Path,
    frame_num: int = 1,
    polygon: bool = False,
    override_if_exists: bool = False,
):
    """Define Polygons GUI."""
    # if source is None:
    #     source = get_config("digital_design_path")
    _, fixed_source = path_checker(path_fixer(source))

    geometry_definer = GeometryDefiner(
        source=fixed_source,
        output_name=output_name,
        polygon=polygon,
        frame_num=frame_num,
        override_if_exists=override_if_exists,
    )
    geometry_definer.run()


@app.command()
def test():
    """Test DriveNetBench."""
    print("Testing DriveNetBench.")


if __name__ == "__main__":
    # run define_polygon() when the script is run
    define_polygon(
        "keypoints/pos3_keypoints_from_camera",
        "assets/pos3_frame_capture.png",
        override_if_exists=True,
    )
    # define_polygon("keypoints/keypoints_from_digital", "assets/track.jpg", override_if_exists=True)
