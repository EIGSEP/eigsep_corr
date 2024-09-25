from eigsep_corr.data import DATA_PATH
from eigsep_corr import io
import numpy as np


def read_dat(module, id_num, keys=None):
    """
    Parameters
    ----------
    module : str
        ``fem'', ``pam'', ``snap'', or ``fiber''
    id_num : str
        The id number of the module. Labelled on the module itself.
    keys : list of str
        Which keys to read from the data (e.g. polarization or input channel).
        If None, all keys are read.

    Returns
    -------
    data : dict
        The data read from the module. Keys refer to polarization or input
        channel in the case of the snap module.

    """
    # default data type is big-endian int32
    dtype = io.build_dtype("int32", ">")
    data = {}
    d = np.load(f"{DATA_PATH}/gain_cal/{module}/{id_num}.npz")
    if keys:
        d = {k: v for k, v in d.items() if k in keys}
    for k, v in d.items():
        data[k] = np.frombuffer(v, dtype=dtype).astype(float)
    return data


class SignalChain:

    # the inputs used for the reference signal to calibrate to
    ref_config = {
        "fem": ("032", "north"),
        "pam": ("375", "north"),
        "snap": ("C000091", "E6"),
        "fiber": ("4", "north"),
    }
    ref_pam_atten = 8

    # the signal to calibrate to
    ref_snap, ref_snap_inp = ref_config["snap"]
    ref_signal = read_dat("snap", ref_snap, keys=ref_snap_inp)

    def __init__(self):
        self.get_ref_gains()
        self.modules = {}
        self.gain_ratios = {}

    def get_ref_gains(self):
        self.ref_gains = {}
        for module, (id_num, inp) in self.ref_config.items():
            data = read_dat(module, id_num)
            self.ref_gains[module] = data[inp]

    def add_module(self, module, id_num, keys=None):
        self.modules[module] = id_num
        data = read_dat(module, id_num, keys=keys)
        self.gain_ratios[module] = {}
        for inp in data.keys():
            r = data[inp] / self.ref_gains[module]
            if module == "snap":
                key = inp[0].lower()
                if key == "e":
                    key = "east"
                elif key == "n":
                    key = "north"
            else:
                key = inp
            self.gain_ratios[module][key] = r
