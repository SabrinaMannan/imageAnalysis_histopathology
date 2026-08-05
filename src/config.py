"""
Configuration loader.

Author: Sabrina
Project: EED-Villus-Morphometry
"""

from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load project configuration.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    dict
    """

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found."
        )

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config