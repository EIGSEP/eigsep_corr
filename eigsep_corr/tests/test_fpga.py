import numpy as np
import pytest

from eigsep_corr.testing import DummyEigsepFpga


@pytest.fixture
def fpga():
    """Fixture to create a DummyFpga instance."""
    return DummyEigsepFpga()


@pytest.fixture
def expected_auto_raw():
    """
    Expected data. See DummyFpga for details.

    Parameters
    ----------
    auto : bool
        If True, return data for autos, otherwise for cross.

    Returns
    -------
    data : bytes
        Bytes representing the expected data.
    """
    nbytes = 2 * 1024 * 4  # 2 for even/odd, 1024 channels, 4 bytes per int
    return b"\x12" * nbytes


@pytest.fixture
def expected_cross_raw():
    """
    Expected data for cross-correlations.

    Returns
    -------
    data : bytes
        Bytes representing the expected cross-correlation data.
    """
    nbytes = 2 * 2 * 1024 * 4
    return b"\x12" * nbytes


@pytest.fixture
def expected_auto(expected_auto_raw):
    return np.frombuffer(expected_auto_raw, dtype=">i4")


@pytest.fixture
def expected_cross(expected_cross_raw):
    return np.frombuffer(expected_cross_raw, dtype=">i4")


def test_read_autos(fpga, expected_auto_raw, expected_auto):
    """Test reading auto-correlations."""
    d = fpga.read_auto()
    autos = [str(i) for i in range(6)]
    assert set(d.keys()) == set(autos)
    for k in autos:
        assert d[k] == expected_auto_raw

    d = fpga.read_auto(unpack=True)
    assert set(d.keys()) == set(autos)
    for k in autos:
        np.testing.assert_array_equal(d[k], expected_auto)

    autos = ["1", "2", "3"]
    d = fpga.read_auto(i=autos)
    assert set(d.keys()) == set(autos)

    d = fpga.read_auto(i="1")
    assert set(d.keys()) == set(["1"])


def test_read_cross(fpga, expected_cross_raw, expected_cross):
    """Test reading cross-correlations."""
    d = fpga.read_cross()
    cross = ["02", "04", "13", "15", "24", "35"]
    assert set(d.keys()) == set(cross)
    for k in cross:
        assert d[k] == expected_cross_raw

    d = fpga.read_cross(unpack=True)
    assert set(d.keys()) == set(cross)
    for k in cross:
        np.testing.assert_array_equal(d[k], expected_cross)

    cross = ["02", "04"]
    d = fpga.read_cross(ij=cross)
    assert set(d.keys()) == set(cross)

    d = fpga.read_cross(ij="02")
    assert set(d.keys()) == set(["02"])


def test_read_data(fpga, expected_auto_raw, expected_cross_raw):
    """Test reading both auto and cross-correlations."""
    d = fpga.read_data()
    autos = [str(i) for i in range(6)]
    cross = ["02", "04", "13", "15", "24", "35"]
    assert set(d.keys()) == set(autos + cross)

    for k in autos:
        assert d[k] == expected_auto_raw
    for k in cross:
        assert d[k] == expected_cross_raw

    d = fpga.read_data(unpack=True)
    assert set(d.keys()) == set(autos + cross)

    for k in autos:
        np.testing.assert_array_equal(
            d[k], np.frombuffer(expected_auto_raw, dtype=">i4")
        )
    for k in cross:
        np.testing.assert_array_equal(
            d[k], np.frombuffer(expected_cross_raw, dtype=">i4")
        )
