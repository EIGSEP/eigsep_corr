from pathlib import Path
import yaml

from .utils import calc_inttime, get_config_path


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
    print("Loading configuration from:", config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # useful computed quantities
    sample_rate = config["sample_rate"]
    corr_acc_len = config["corr_acc_len"]
    acc_bins = config["acc_bins"]
    t_int = calc_inttime(
        sample_rate * 1e6,  # in Hz
        corr_acc_len,
        acc_bins=acc_bins,
    )
    ntimes = config["ntimes"]
    file_time = t_int * ntimes
    config["integration_time"] = t_int
    config["file_time"] = file_time
    return config
