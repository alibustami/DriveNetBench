"""Utility functions for the project."""

import os
from typing import Any, Dict

from yaml import safe_load

GLOBAL_CONFIGS = None


def load_config() -> Dict:
    """Load the config.yaml file and return the configs.

    Returns
    -------
    Dict
        The configs from the config.yaml file.
    """
    assert os.path.exists(
        "config.yaml"
    ), "Config.yaml file not found in the root directory"
    with open("config.yaml") as f:
        global GLOBAL_CONFIGS
        GLOBAL_CONFIGS = _process_configs(safe_load(f))
        return GLOBAL_CONFIGS


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

    return _flatten_dict(configs)


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
