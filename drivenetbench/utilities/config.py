"""Utility functions for the project."""

import os
from typing import Any, Dict, Optional

from yaml import safe_load

GLOBAL_CONFIGS = None


def load_config(config_file_path: Optional[str] = None) -> Dict:
    """Load the config.yaml file and return the configs.

    Returns
    -------
    Dict
        The configs from the config.yaml file.
    """
    if config_file_path:
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(
                f"Config file {config_file_path} not found"
            )
    elif os.path.exists("config.yaml"):
        config_file_path = "config.yaml"
    else:
        raise FileNotFoundError("Config file not found")
    with open(config_file_path) as f:
        global GLOBAL_CONFIGS
        GLOBAL_CONFIGS = _process_configs(safe_load(f))
        # return GLOBAL_CONFIGS


def _process_configs(configs: Dict) -> Dict:
    """Process the configs by flattening them so that nested dictionaries
    can be accessed using dot-notation keys like 'foo.bar.baz'.

    Parameters
    ----------
    configs : Dict
        The configs to process.

    Returns
    -------
    Dict
        The flattened configs.
    """

    def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """Recursively flattens a nested dict.

        Parameters
        ----------
        d : Dict
            The dictionary to flatten.
        parent_key : str
            The base key string for recursion.
        sep : str
            The separator to use when creating the flattened keys.

        Returns
        -------
        Dict
            A single-level dictionary with dot-joined keys.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(_flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    flattned_configs = _flatten_dict(configs)

    for key, value in flattned_configs.items():
        if not key.endswith("_path"):
            continue
        if not os.path.exists(value):
            raise FileNotFoundError(
                f"Path {value} associated with {key} not found"
            )

    return flattned_configs


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
        if key not in GLOBAL_CONFIGS:
            raise KeyError(f"Key {key} not found in config.yaml")
        return GLOBAL_CONFIGS.get(key)
    else:
        load_config()
        return get_config(key)
