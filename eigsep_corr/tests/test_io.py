'''Tests for limbo.io'''
import pytest
import os
import numpy as np
import copy

from eigsep_corr import io
#from eigsep_corr.data import DATA_PATH

class TestFileIO(object):
    def test_pack_unpack_header(self):
        h1 = io.DEFAULT_HEADER
        buf = io.pack_raw_header(h1)
        h2 = io.unpack_raw_header(buf)
        assert h2['data_start'] % 8 == 0
        for k, v in h1.items():
            if type(v) is tuple or type(v) is list:
                assert tuple(v) == tuple(h2[k])
            elif type(v) is dict:
                print(k, v, h2[k])
                for _k, _v in v.items():
                    assert tuple(_v) == tuple(h2[k][_k])
            elif type(v) is np.ndarray:
                np.testing.assert_allclose(v, h2[k])
            else:
                assert v == h2[k]

    def test_pack_unpack_raw_data(self):
        dt = io.build_dtype('int32', '>')
        d1 = np.ones((10, 18, 2, 1024), dtype=dt)
        buf = io.pack_raw_data(d1)
        d2 = io.unpack_raw_data(buf)
        np.testing.assert_allclose(d1, d2)

    def test_pack_unpack_data(self):
        h = copy.deepcopy(io.DEFAULT_HEADER)
        pairs = h['pairs']
        d1 = {p: np.ones((10, 2, 1024)) for p in pairs}
        buf = io.pack_data(d1, h)
        d2 = io.unpack_data(buf, h)
        for k, v in d1.items():
            np.testing.assert_allclose(v, d2[k])

    def test_write_read_file(self, tmp_path):
        filename = tmp_path / 'test.eig'
        h1 = copy.deepcopy(io.DEFAULT_HEADER)
        pairs = h1['pairs']
        d1 = {p: np.ones((len(h1['acc_cnt']), 2, 1024)) for p in pairs}
        io.write_file(filename, h1, d1)
        h2, d2 = io.read_file(filename)
        for k, v in d1.items():
            np.testing.assert_allclose(v, d2[k])
        for k, v in h1.items():
            if type(v) is tuple or type(v) is list:
                assert tuple(v) == tuple(h2[k])
            elif type(v) is dict:
                print(k, v, h2[k])
                for _k, _v in v.items():
                    assert tuple(_v) == tuple(h2[k][_k])
            elif type(v) is np.ndarray:
                np.testing.assert_allclose(v, h2[k])
            else:
                assert v == h2[k]

    
    #def setup_method(self):
    #    self.filename = os.path.join(DATA_PATH, 'test.dat')

    #def test_read_header(self):
    #    header = io.read_header(self.filename)
    #    assert header['filename'] == self.filename
    #    assert header['fpg'] == 'limbo_500_2022-12-03_1749.fpg'
    #    assert header['inttime'] == 2e-9 * 127 * 4096
    #    assert header['freqs'].size == 2048
    #    assert header['freqs'][0] == 1350e6

    #def test_read_raw_data(self):
    #    data = io.read_raw_data(self.filename)
    #    assert data.shape == (4, 2048 + 12)
    #    assert data.dtype == np.dtype('>u2')
    #    data = io.read_raw_data(self.filename, nspec=2)
    #    assert data.shape == (2, 2048 + 12)

    #def test_read_file(self):
    #    hdr, data = io.read_file(self.filename)
    #    assert data.shape == (4, 2048)
    #    assert hdr['times'].size == 4
    #    assert hdr['times'][0] == hdr['Time']
    #    np.testing.assert_almost_equal(np.diff(hdr['times']), hdr['inttime'], 6)
    #    assert hdr['jds'].size == 4
    #    hdr, data = io.read_file(self.filename, nspec=2)
    #    assert data.shape == (2, 2048)
    #    assert hdr['times'].size == 2
    #    assert hdr['jds'].size == 2
