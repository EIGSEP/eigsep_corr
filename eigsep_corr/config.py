from pathlib import Path
import yaml

from .utils import get_config_path


def load_config(name):
    """
    Load the configuration file.

    Parameters
    ----------
    name : str
        Name of the configuration file.

    Returns
    -------
    dict
        Configuration parameters.
    """
    p = Path(name)
    if p.is_absolute():
        config_path = p
    else:
        config_path = get_config_path(name)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
