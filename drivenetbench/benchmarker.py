"""Benchmarker class for running benchmarks on DriveNet models."""

import drivenetbench.utilities.config as configs
from drivenetbench.detector import Detector


class BenchMarker:
    def __init__(self, config_file_path: str):
        self.config_file_path = config_file_path

        configs.load_config()
        self.detector = Detector()

    def run(self):
        robot_path = self.detector.detect_robot()
