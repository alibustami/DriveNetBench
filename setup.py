"""Package setup file."""

import os

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="DriveNetBench",
    version="0.1",
    packages=find_packages(include=["drivenetbench", "drivenetbench.*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "drivenetbench=drivenetbench.apps.cli:app",
        ],
    },
)
