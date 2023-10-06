import json
import numpy as np

# XXX make sure header, metadata, data all have size multiple of 64 bytes
# can use sys.getsizeof to check size of object in bytes

HEADER = {
    "header_size": 640,  # XXX size of this object in bytes
    "nfiles": 60,  # number of integrations per file
    "dtype": "int32",  # data type
    "byteorder": ">",  # endianess of data
    "snap_ip": "10.10.10.236",  # IP address of SNAP board
    "nchan": 1024,  # number of frequency channels
    "fpg_file": "eigsep_fengine_1g_v2_0_2023-09-30_1811.fpg",
    "fpg_version": 0x1000,
    "sample_rate": 500,  # in MHz
    "gain": 4,  # gain of ADC
    "corr_acc_len": 2**28,  # number of samples to accumulate
    "corr_scalar": 2**9,  # 2^9 = 1, using 8 bits after binary point
    "pol0_delay": 0,  # delay in sample clocks of input 0
    "pol1_delay": 0,  # delay in sample clocks of input 0
    "fft_shift": 0x0055,
    "inputs": [0, 1, 2, 3, 4, 5],  # inputs used
}

DTYPE = np.dtype(HEADER["dtype"]).newbyteorder(HEADER["byteorder"])


def get_header_size(fh, dtype):
    """
    Read size of header assuming it is the first word in a file and that it
    is encoded in 4 bytes.

    Parameters
    ----------
    fh : file handle
        File handle to read from.
    dtype : numpy.dtype or str
        Data type.

    Returns
    -------
    header_size : int
        Size of header in bytes.

    """
    fh.seek(0, 0)  # go to beginning of file
    return np.frombuffer(fh.read(4), dtype=dtype)[0]


class File:
    def __init__(self, fname, time, sync_time):
        self.fname = fname
        self.header = HEADER
        self.time = time  # 32 bit integer, #XXX check endianess
        self.sync_time = sync_time
        self.dtype = DTYPE
        self.data = {
            "header": self.header,
            "time": self.time,
            "sync_time": self.sync_time,
        }
        self.max_cnt = None

    def add_data(self, data, cnt):
        self.data[f"{cnt}"] = data
        if self.max_cnt is None:  # this is the first file
            self.max_cnt = cnt + HEADER["nfiles"]

    def write(self):
        with open(self.fname, "w") as fh:  # XXX why w not wb
            json.dump(self.data, fh)

    def read_header(self):
        """
        Read header from file.

        Returns
        -------
        header : dict
            Header information.

        """
        with open(self.fname, "rb") as fh:
            header_size = get_header_size(fh, self.dtype)
            fh.seek(0, 0)  # go to beginning of file
            header = fh.read(header_size)
        return json.loads(header)

    def read_data(self):
        with open(self.fname, "rb") as fh:
            header_size = get_header_size(fh, self.dtype)
            fh.seek(
                header_size, 0
            )  # go to beginning of data #XXX need to add size of time (32 bit)
            data = json.load(fh)
        return data
