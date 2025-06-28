from pathlib import Path
import yaml

from .utils import calc_inttime


def load_config(name, compute_inttime=True):
    """
    Load the configuration file.

    Parameters
    ----------
    name : str or Path
        Path to the configuration file.
    compute_inttime : bool
        Compute integration time and file time if True.

    Returns
    -------
    dict
        Configuration parameters.

    """
    config_path = Path(name)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    if compute_inttime:
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
