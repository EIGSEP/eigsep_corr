import json
import os
import numpy as np
import struct
from . import utils

# XXX make sure header, metadata, data all have size multiple of 64 bytes

HEADER_LEN_BYTES = 8
HEADER_LEN_DTYPE = '>Q'

DEFAULT_HEADER = {
    "dtype": ("int32", ">"),  # data type, endianess of data
    "infochan": 2,  # number of frequency channels used to track acc_cnt
    "nchan": 1024,  # number of frequency channels
    "acc_bins": 2,  # number of accumulation bins per integration
    "fpg_file": "eigsep_fengine_1g_v2_0_2023-09-30_1811.fpg",
    "fpg_version": 0x1000,
    "sample_rate": int(500e6),  # in Hz
    "gain": 4,  # gain of ADC
    "corr_acc_len": 2**28,  # number of samples to accumulate
    "corr_scalar": 2**9,  # 2^9 = 1, using 8 bits after binary point
    "pol01_delay": 0,  # delay in sample clocks of inputs 0/1
    "pam_atten": {0: (8, 8), 1: (8, 8), 2: (8, 8)},  # PAM attenuations
    "fft_shift": 0x0055,
    "pairs": ['0', '1', '2', '3', '4', '5',
              '02_r', '02_i', '04_r', '04_i', '24_r', '24_i'
              '13_r', '13_i', '15_r', '15_i', '35_r', '35_i'],
    "acc_cnt": np.arange(60),
    "sync_time": 0.0,
}


def build_dtype(dtype, endian):
    return np.dtype(dtype).newbyteorder(endian)


def unpack_raw_header(buf, header_size=None):
    if header_size is None:
        header_size = len(buf)
    else:
        buf = buf[:header_size]
    header = json.loads(buf)  # trim trailing nulls
    dt = build_dtype(*header['dtype'])
    data_start = header_size + HEADER_LEN_BYTES + (8 - (header_size % 8))
    header['header_size'] = header_size
    header['data_start'] = data_start
    header['pam_atten'] = {int(k): v for k, v in header['pam_atten'].items()}
    header['acc_cnt'] = np.frombuffer(header['acc_cnt'].encode('utf-8'), dtype=dt)
    return header


def pack_raw_header(header):
    dt = build_dtype(*header['dtype'])
    # filter to official header keys
    header = {k: v for k, v in header.items() if k in DEFAULT_HEADER}
    header['acc_cnt'] = np.array(header['acc_cnt'], dtype=dt).tobytes().decode('utf-8')
    buf = json.dumps(header)
    return buf


def _read_header_size(fh):
    """Read size of header from first word in file."""
    fh.seek(0, 0)  # go to beginning of file
    return struct.unpack(HEADER_LEN_DTYPE, fh.read(HEADER_LEN_BYTES))[0]


def _read_raw_header(fh):
    header_size = _read_header_size(fh) # leaves us after ``header size''
    header = unpack_raw_header(fh.read(header_size).decode('utf-8'))
    return header


def _write_raw_header(fh, header):
    buf = pack_raw_header(header).encode('utf-8')
    header_size = len(buf)
    fh.write(struct.pack(HEADER_LEN_DTYPE, header_size))
    fh.write(buf)
    fh.write((8 - (header_size % 8)) * b'\x00')  # pad with trailing nulls


def read_header(filename):
    with open(filename, 'rb') as fh:
        h = _read_raw_header(fh)
    # augment raw header with useful calculated values
    h['filename'] = filename
    h['filesize'] = filesize = os.path.getsize(filename)
    dt = build_dtype(*h['dtype'])
    integration_len = dt.itemsize * h['acc_bins'] * h['nchan'] * len(h['pairs'])
    h['nspec'] = nspec = (filesize - h['data_start']) // integration_len
    assert nspec == len(h['acc_cnt']), "Check that file size matches integration cnts"
    h['freqs'], h['dfreq'] = utils.calc_freqs_dfreq(h['sample_rate'], h['nchan'])
    h['inttime'] = inttime = utils.calc_inttime(h['sample_rate'], h['corr_acc_len'])
    h['times'] = utils.calc_times(h['acc_cnt'], inttime, h['sync_time'])
    return h
    

def unpack_raw_data(buf, npairs=18, acc_bins=2, nchan=1024, dtype=('int32', '>')):
    dt = build_dtype(*dtype)
    spec_len = dt.itemsize * acc_bins * nchan
    data = np.frombuffer(buf, dtype=dt)
    data.shape = (-1, npairs, acc_bins, nchan)
    return data


def pack_raw_data(data, dtype=('int32', '>')):
    dt = build_dtype(*dtype)
    buf = data.astype(dt).tobytes()
    return buf


def unpack_data(fh_buf, h, nspec=-1, skip=0):
    dt = build_dtype(*h['dtype'])
    integration_len = dt.itemsize * h['acc_bins'] * h['nchan'] * len(h['pairs'])
    if type(fh_buf) is bytes:
        buf = fh_buf
    else:
        fh = fh_buf
        start = h['data_start'] + skip * integration_len
        fh.seek(start, 0)
        if nspec < 0:
            buf = fh.read()
        else:
            buf = fh.read(nspec * integration_len)
    data = unpack_raw_data(buf,
                           len(h['pairs']),
                           acc_bins=h['acc_bins'],
                           nchan=h['nchan'],
                           dtype=h['dtype']
                          )
    data = {p: data[:,i] for i, p in enumerate(h['pairs'])}
    return data


def pack_data(data, h):
    data = np.array([data[k] for k in h['pairs']]).transpose(1, 0, 2, 3)
    buf = pack_raw_data(data, dtype=h['dtype'])
    return buf
    

def pack_corr_data(dict_list, h):
    """For use with a list of dicts of binary data read straight from correlator."""
    buf = ''.join([d[k] for d in dict_list for k in h['pairs']])
    return buf

    
def read_file(filename, header=None, nspec=-1, skip=0):
    if header is None:
        header = read_header(filename)
    with open(filename, 'rb') as fh:
        data = unpack_data(fh, header, nspec=nspec, skip=skip)
    return header, data
                              

def write_file(filename, header, data):
    with open(filename, 'wb') as fh:
        _write_raw_header(fh, header)
        fh.write(pack_data(data, header))


class File:
    # XXX how are even/odd being treated
    # XXX don't use json per integration: just <int32 unused> <int32 acc_cnt> <NCHAN * int32 data>
    def __init__(self, fname, sync_time):
        self.fname = fname
        self.header = HEADER
        self.header["sync_time"] = sync_time  # XXX account for this in size
        self.time = 0  # XXX compute from sync_time and count
        self.sync_time = sync_time
        self.dtype = DTYPE
        self.data = {"header": self.header, "time": self.time}
        self.max_cnt = None

    def add_data(self, data, cnt):
        self.data[f"{cnt}"] = data
        if self.max_cnt is None:  # this is the first file
            self.max_cnt = cnt + HEADER["nfiles"]

    def write(self):
        with open(self.fname, "w") as fh:  # XXX why w not wb
            json.dump(self.data, fh)

    def read_header(self, header_size=None):
        """
        Read header from file.

        Returns
        -------
        header : dict
            Header information.

        """
        with open(self.fname, "rb") as fh:
            if header_size is None:
                header_size = get_header_size(fh)  # seeks to 0
            else:
                fh.seek(HEADER_LEN_SIZE, 0)  # go to beginning of header, after ``header size''
            hdata = fh.read(header_size)
        self.header = json.loads(hdata)
        return self.header

    def read_data(self, see):
        with open(self.fname, "rb") as fh:
            header_size = get_header_size(fh, self.dtype)
            # move past ``header size'', header, and ``time''
            data_start = 4 + header_size + 4
            fh.seek(data_start, 0)  # go to beginning of data
            data = json.load(fh)
        return data
