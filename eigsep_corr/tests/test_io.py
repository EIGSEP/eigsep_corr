"""Tests for eigsep_corr.io"""

import copy
import numpy as np
import pytest

from eigsep_corr import io

NTIMES = 60
TEST_HEADER = {
    "dtype": ("int32", ">"),
    "infochan": 2,
    "nchan": 1024,
    "acc_bins": 2,
    "fpg_file": "eigsep_fengine_1g_v2_3_2024-07-08_1858.fpg",
    "fpg_version": 0x2003,
    "sample_rate": int(500e6),
    "gain": 4,
    "corr_acc_len": 2**28,
    "corr_scalar": 2**9,
    "pol01_delay": 0,
    "pol23_delay": 0,
    "pol45_delay": 0,
    "pam_atten": {0: (8, 8), 1: (8, 8), 2: (8, 8)},
    "fft_shift": 0x0055,
    "pairs": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "02",
        "04",
        "24",
        "13",
        "15",
        "35",
    ],
    "acc_cnt": np.arange(NTIMES),
    "sync_time": 0.0,
}


@pytest.fixture
def header():
    return copy.deepcopy(TEST_HEADER)


class TestFileIO:
    def test_pack_unpack_header(self, header):
        buf = io.pack_raw_header(header)
        h2 = io.unpack_raw_header(buf)
        assert h2["data_start"] % 8 == 0
        for k, v in header.items():
            if type(v) is tuple or type(v) is list:
                assert tuple(v) == tuple(h2[k])
            elif type(v) is dict:
                for _k, _v in v.items():
                    assert tuple(_v) == tuple(h2[k][_k])
            elif type(v) is np.ndarray:
                np.testing.assert_allclose(v, h2[k])
            else:
                assert v == h2[k]

    def test_pack_unpack_raw_data(self):
        dt = io.build_dtype("int32", ">")
        d1 = np.ones((10, 2, 1024, 1), dtype=dt)
        buf = io.pack_raw_data(d1)
        d2 = io.unpack_raw_data(buf, "0")
        np.testing.assert_allclose(d1, d2)
        d1 = np.ones((10, 2, 1024, 2), dtype=dt)
        buf = io.pack_raw_data(d1)
        d2 = io.unpack_raw_data(buf, "02")
        np.testing.assert_allclose(d1, d2)

    def test_pack_unpack_data(self, header):
        pairs = header["pairs"]
        d1 = {
            p: (
                np.ones((10, 2, 1024, 1))
                if len(p) == 1
                else np.ones((10, 2, 1024, 2))
            )
            for p in pairs
        }
        buf = io.pack_data(d1, header)
        d2 = io.unpack_data(buf, header)
        for k, v in d1.items():
            np.testing.assert_allclose(v, d2[k])

    def test_write_read_file(self, tmp_path, header):
        filename = tmp_path / "test.eig"
        pairs = header["pairs"]
        d1 = {
            p: (
                np.ones((len(header["acc_cnt"]), 2, 1024, 1))
                if len(p) == 1
                else np.ones((len(header["acc_cnt"]), 2, 1024, 2))
            )
            for p in pairs
        }
        io.write_file(filename, header, d1)
        h2, d2 = io.read_file(filename)
        for k, v in d1.items():
            np.testing.assert_allclose(v, d2[k])
        for k, v in header.items():
            if type(v) is tuple or type(v) is list:
                assert tuple(v) == tuple(h2[k])
            elif type(v) is dict:
                for _k, _v in v.items():
                    assert tuple(_v) == tuple(h2[k][_k])
            elif type(v) is np.ndarray:
                np.testing.assert_allclose(v, h2[k])
            else:
                assert v == h2[k]
