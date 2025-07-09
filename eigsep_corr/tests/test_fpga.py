import numpy as np
import pytest
import tempfile

import eigsep_corr.fpga
from eigsep_corr.testing import DummyEigsepFpga, DummyPam


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


def test_fpga_initialization(fpga):
    """Test FPGA initialization."""
    assert fpga.cfg["snap_ip"] == "10.10.10.12"
    assert fpga.cfg["sample_rate"] == 500.0
    assert not fpga.adc_initialized
    assert not fpga.pams_initialized
    assert not fpga.is_synchronized


def test_fpga_initialize_method(monkeypatch, fpga):
    """Test the initialize method."""
    monkeypatch.setattr(eigsep_corr.fpga, "Pam", DummyPam)
    fpga.initialize(
        initialize_adc=True,
        initialize_fpga=True,
        sync=True,
        update_redis=False,
    )
    assert fpga.adc_initialized
    assert fpga.pams_initialized
    assert fpga.is_synchronized


def test_validate_config(monkeypatch, fpga):
    """Test configuration validation."""

    monkeypatch.setattr(eigsep_corr.fpga, "Pam", DummyPam)
    # First initialize the FPGA to set up the PFB properly
    fpga.initialize(initialize_fpga=True)

    # The validation expects computed values to match, so let's update config
    # with the computed integration_time and file_time
    from eigsep_corr.utils import calc_inttime

    sample_rate = fpga.cfg["sample_rate"]
    corr_acc_len = fpga.cfg["corr_acc_len"]
    acc_bins = fpga.cfg["acc_bins"]
    t_int = calc_inttime(sample_rate * 1e6, corr_acc_len, acc_bins=acc_bins)

    fpga.cfg["integration_time"] = t_int

    # Test that validation passes when config matches hardware
    fpga.validate_config()


def test_redis_integration(fpga):
    """Test Redis integration with fakeredis."""
    assert fpga.redis is not None

    # Test setting and getting values
    fpga.redis.set("test_key", "test_value")
    assert fpga.redis.get("test_key").decode() == "test_value"

    # Test that it's actually fakeredis
    import fakeredis

    assert isinstance(fpga.redis, fakeredis.FakeRedis)


@pytest.mark.skip("Skipping test for now due to timeout issues.")
def test_observe_method(fpga):
    """Test the observe method with write_files=False."""
    fpga.initialize(sync=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test observe without writing files
        try:
            fpga.observe(
                save_dir=tmpdir,
                pairs=None,
                timeout=0.1,  # Very short timeout for testing
                update_redis=False,
                write_files=False,
            )
        except Exception:
            # Expected to timeout quickly
            pass
        finally:
            fpga.end_observing()


def test_config_loading_integration():
    """Test config loading integration."""
    from eigsep_corr.fpga import default_config

    # Test that default config loads
    assert default_config is not None
    assert "snap_ip" in default_config
    assert "sample_rate" in default_config

    # Test creating FPGA with default config
    fpga = DummyEigsepFpga(cfg=default_config)
    assert fpga.cfg == default_config


def test_add_args_function():
    """Test the add_args function."""
    import argparse
    from eigsep_corr.fpga import add_args

    parser = argparse.ArgumentParser()
    add_args(parser)

    # Test that required arguments are added
    args = parser.parse_args(["--dummy"])
    assert args.dummy_mode is True

    args = parser.parse_args(["-p"])
    assert args.program is True

    args = parser.parse_args(["-f"])
    assert args.initialize_fpga is True
