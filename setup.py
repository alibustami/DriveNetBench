"""Package setup file."""

import os

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="DriveNetBench",
    version="0.1",
    short_description="A benchmarking tool using a single camera for robot navigation in predefined tracks.",
    packages=find_packages(include=["drivenetbench", "drivenetbench.*"]),
    author="Ali Albustami",
    author_email="abustami@umich.edu",
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "drivenetbench=drivenetbench.apps.cli:app",
        ],
    },
)
