'''Utility functions for EIGSEP'''

import numpy as np


def calc_freqs_dfreq(sample_rate_Hz, nchan):
    """Return frequencies and delta between frequencies for real-sampled
    spectra from the SNAP spectrometer/correlator."""
    dfreq = sample_rate_Hz / (2 * nchan)  # assumes real sampling
    freqs = np.arange(nchan) * dfreq
    return freqs, dfreq
    

def calc_inttime(sample_rate_Hz, acc_len):
    '''Calculate time per integration [s] from sample_freq and acc_len.'''
    inttime = 1 / sample_rate_Hz * acc_len
    return inttime


def calc_times(acc_cnt, inttime, sync_time):
    '''Calculate integration times [s] from acc_cnt using sync time.'''
    times = acc_cnt * inttime + sync_time
    return times

def calc_integration_len(itemsize, acc_bins, nchan, pairs):
    # XXX implement and update for real/imag calculation in pairs
    pass

