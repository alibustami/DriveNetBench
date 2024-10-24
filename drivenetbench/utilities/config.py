"""Utility functions for the project."""

import os
from typing import Any

from yaml import safe_load

GLOBAL_CONFIGS = None


def load_config() -> dict:
    """Load the config.yaml file and return the configs.

    Returns
    -------
    dict
        The configs from the config.yaml file.
    """
    assert os.path.exists(
        "config.yaml"
    ), "Config.yaml file not found in the root directory"
    with open("config.yaml") as f:
        global GLOBAL_CONFIGS
        GLOBAL_CONFIGS = safe_load(f)
        return GLOBAL_CONFIGS


def get_config(key: str) -> Any:
    """Get the value of the key from the config.yaml file.

    Parameters
    ----------
    key : str
        The key to get the value of from the config.yaml file.

    Returns
    -------
    Any
        The value of the key from the config.yaml file.
    """
    global GLOBAL_CONFIGS
    if GLOBAL_CONFIGS:
        assert key in GLOBAL_CONFIGS, f"Key {key} not found in config.yaml"
        return GLOBAL_CONFIGS.get(key)
    else:
        GLOBAL_CONFIGS = load_config()
        return get_config(key)
