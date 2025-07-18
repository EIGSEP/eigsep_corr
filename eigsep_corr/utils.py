from importlib import resources
import numpy as np


def get_path(dirname=None, fname=None):
    """
    Get the path to a directory or file within the package.
    Default returns path to the package, <pkg_path>.
    If `dirname` is provided, return path to <pkg_path>/<dirname>.
    If `fname` is provided, return the full path to that file.

    Parameters
    ----------
    dirname : str or Path
        Name of the directory within the package.
    fname : str or Path
        Name of the file within the package or directory.

    Returns
    -------
    Path
        The path to the specified directory or file within the package.
    """
    path = resources.files(__package__)
    if dirname is not None:
        path = path.joinpath(dirname)
    if fname is not None:
        path = path.joinpath(fname)
    return path


def get_data_path(fname=None):
    """
    Get the path to the data directory within the package.
    If `fname` is provided, return the full path to that file.
    """
    return get_path(dirname="data", fname=fname)


def get_config_path(fname=None):
    """
    Get the path to the configuration directory within the package.
    If `fname` is provided, return the full path to that file.
    """
    return get_path(dirname="config", fname=fname)


def calc_freqs_dfreq(sample_rate_Hz, nchan):
    """Return frequencies and delta between frequencies for real-sampled
    spectra from the SNAP spectrometer/correlator."""
    dfreq = sample_rate_Hz / (2 * nchan)  # assumes real sampling
    freqs = np.arange(nchan) * dfreq
    return freqs, dfreq


def calc_inttime(sample_rate_Hz, acc_len, acc_bins=2):
    """Calculate time per integration [s] from sample_freq and acc_len."""
    inttime = 1 / sample_rate_Hz * acc_len * acc_bins
    return inttime


def calc_times(acc_cnt, inttime, sync_time):
    """Calculate integration times [s] from acc_cnt using sync time."""
    times = acc_cnt * inttime + sync_time  # XXX acc_cnt + 1?
    return times


def calc_integration_len(itemsize, acc_bins, nchan, pairs):
    """
    Calculate the number of bytes for an integration of ``acc_bins`` bins.
    Cross-correlations have double length since there's a real and imaginary
    part.

    Parameters
    ----------
    itemsize : int
        Size of data type in bytes.
    acc_bins : int
        Number of accumulations per integration.
    nchan : int
        Number of frequency channels per spectrum.
    pairs : list of str
        List of correlation pairs. Lenght 1 for autos, 2 for cross.

    Returns
    -------
    int_len : int
        Number of bytes for an integration of ``acc_bins`` bins.

    """
    n_auto = len([p for p in pairs if len(p) == 1])
    n_cross = len(pairs) - n_auto
    return itemsize * acc_bins * nchan * (n_auto + 2 * n_cross)
