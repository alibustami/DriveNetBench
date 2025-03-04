"""Command Line Interface for DriveNetBench."""

from typing import Optional

from typer import Typer

import drivenetbench.utilities.config as configs
from drivenetbench.benchmarker import BenchMarker
from drivenetbench.keypoints_definer import KeyPointsDefiner
from drivenetbench.utilities.track_processor import TrackProcessor

app = Typer()


@app.command()
def define_keypoints(config_file_path: Optional[str] = "config.yaml"):
    """Define Keypoints."""
    configs.load_config(config_file_path)
    keypoints_definer = KeyPointsDefiner()
    keypoints_definer.run()


@app.command()
def extract_track(config_file_path: Optional[str] = "config.yaml"):
    """Extract the track."""
    configs.load_config(config_file_path)
    track_processor = TrackProcessor()
    track_processor.process()


@app.command()
def benchmark(config_file_path: Optional[str] = "config.yaml"):
    """Benchmark the DriveNetBench."""
    bench_marker = BenchMarker(config_file_path)
    bench_marker.run()


if __name__ == "__main__":
    app()
