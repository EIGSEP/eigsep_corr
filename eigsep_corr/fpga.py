"""
Module for interfacing to a 6-antpol xx/yy correlator for EIGSEP.
This is nominally uses a 4-tap, 2048 real sample (1024 ch) PFB,
with a direct correlation and vector accumulation of 2048 samples,
producing odd/even data sets that are jackknifed every spectrum
(which is faster than ideal: the PFB correlates adjacent spectra).

The important bit widths are:
(ADC) 8_7 (PFB_FIR) 18_17 (FFT) 18_17 (CORR) 18_17 (SCALAR) 18_7 (VACC) 32_7
Affecting the signal level are the FFT_SHIFT (0b00001010101) and the
CORR_SCALAR (18_8).
"""

import datetime
import logging
import numpy as np
from pathlib import Path
from queue import Queue
import time
from threading import Event, Thread

import redis

USE_CASPERFPGA = True
try:
    import casperfpga
    from casperfpga.transport_tapcp import TapcpTransport
except ImportError:
    USE_CASPERFPGA = False
    TapcpTransport = None

from . import io
from .blocks import Input, NoiseGen, Pam, Pfb, Sync
from .config import load_config
from .utils import calc_inttime, get_data_path

logger = logging.getLogger(__name__)
if not USE_CASPERFPGA:
    logger.warning("Running without casperfpga installed")

default_config_file = "config.yaml"
default_config = load_config(default_config_file)


def add_args(parser, default_config_file=default_config_file):
    """
    Add command-line arguments for EigsepFpga.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to.

    """
    parser.add_argument(
        "-p",
        dest="program",
        action="store_true",
        default=False,
        help="Program Eigsep correlator.",
    )
    parser.add_argument(
        "-P",
        dest="force_program",
        action="store_true",
        default=False,
        help="Force program Eigsep correlator.",
    )
    parser.add_argument(
        "-a",
        dest="initialize_adc",
        action="store_true",
        default=False,
        help="Initialize ADCs.",
    )
    parser.add_argument(
        "-f",
        dest="initialize_fpga",
        action="store_true",
        default=False,
        help="Initialize Eigsep correlator.",
    )
    parser.add_argument(
        "-s",
        dest="sync",
        action="store_true",
        default=False,
        help="Sync Eigsep correlator.",
    )
    parser.add_argument(
        "--config_file",
        dest="config_file",
        default=default_config_file,
        help="Configuration file for Eigsep Fpga.",
    )
    parser.add_argument(
        "--dummy",
        dest="dummy_mode",
        action="store_true",
        default=False,
        help="Run with a dummy SNAP interface",
    )


class EigsepFpga:

    def __init__(self, cfg=default_config, program=False):
        """
        Class for interfacing with the SNAP board.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary. See `config/config.yaml` for
            details.
        program : bool or str
            Whether to program the SNAP with the fpg file. Options are
            True (program if fpg_file is different from the one in
            flash), False (do not program), 'force' (always program).

        """
        self.logger = logger
        self.logger.debug("Initializing EigsepFpga")
        self.cfg = cfg
        self.autos = ["0", "1", "2", "3", "4", "5"]
        self.crosses = ["02", "13", "24", "35", "04", "15"]

        # redis instance
        rcfg = self.cfg["redis"]
        self.redis = self._create_redis(rcfg["host"], rcfg["port"])

        fpg_file = Path(self.cfg["fpg_file"])
        if not fpg_file.is_absolute():
            self.fpg_file = get_data_path(fname=fpg_file)
        else:
            self.fpg_file = fpg_file

        self.fpga = casperfpga.CasperFpga(
            self.cfg["snap_ip"], transport=TapcpTransport
        )
        if program:
            force = program == "force"
            self.fpga.upload_to_ram_and_program(self.fpg_file, force=force)

        if cfg["use_ref"]:
            ref = 10
        else:
            ref = None

        # blocks
        self.adc = casperfpga.snapadc.SnapAdc(
            self.fpga, num_chans=2, resolution=8, ref=ref
        )
        self.sync = Sync(self.fpga, "sync")
        self.noise = NoiseGen(self.fpga, "noise", nstreams=6)
        self.inp = Input(self.fpga, "input", nstreams=12)
        self.pfb = Pfb(self.fpga, "pfb")
        self.blocks = [self.sync, self.noise, self.inp, self.pfb]

        self.adc_initialized = False
        self.pams_initialized = False
        self.is_synchronized = False

    @staticmethod
    def _create_redis(host, port):
        """
        Create a Redis instance.

        Parameters
        ----------
        host : str
            Redis server hostname.
        port : int
            Redis server port.

        Returns
        -------
        redis.Redis
            Redis instance connected to the specified host and port.

        """
        return redis.Redis(host=host, port=port)

    @property
    def version(self):
        val = self.fpga.read_uint("version_version")
        major = val >> 16
        minor = val & 0xFFFF
        return [major, minor]

    @property
    def header(self):
        """
        Generate a file header. This is a copy of the configuration
        dictionary, but replaced with the actual values from the
        SNAP when available.

        Returns
        -------
        dict
            Dictionary with keys as configuration parameters and values
            as the actual values from the SNAP board.

        """
        if self.adc_initialized:
            sample_rate = self.adc.sample_rate / 1e6  # in MHz
            adc_gain = self.adc.gain
        else:
            sample_rate = self.cfg["sample_rate"]
            adc_gain = self.cfg["adc_gain"]
        if self.pams_initialized:
            pam_atten = {
                int(i): list(p.get_attenuation())
                for i, p in enumerate(self.pams)
            }
        else:
            pam_atten = self.cfg["pam_atten"]
        if self.is_synchronized:
            sync_time = self.sync_time
        else:
            sync_time = 0

        corr_acc_len = self.fpga.read_uint("corr_acc_len")
        acc_bins = self.cfg["acc_bins"]
        t_int = calc_inttime(
            sample_rate * 1e6,  # in Hz
            corr_acc_len,
            acc_bins=acc_bins,
        )
        ntimes = self.cfg["ntimes"]
        file_time = t_int * ntimes
        m = {
            "snap_ip": self.cfg["snap_ip"],
            "fpg_file": str(self.fpg_file),
            "fpg_version": self.version,
            "sample_rate": sample_rate,
            "nchan": self.cfg["nchan"],
            "use_ref": self.cfg["use_ref"]
            "use_noise": self.cfg["use_noise"],
            "adc_gain": adc_gain,
            "fft_shift": self.pfb.get_fft_shift(),
            "corr_acc_len": corr_acc_len,
            "corr_scalar": self.fpga.read_uint("corr_scalar"),
            "corr_word": self.cfg["corr_word"],
            "acc_bins": acc_bins,
            "dtype": self.cfg["dtype"],
            "pam_atten": pam_atten,
            "pol_delay": {
                "01": self.fpga.read_uint("pfb_pol01_delay"),
                "23": self.fpga.read_uint("pfb_pol23_delay"),
                "45": self.fpga.read_uint("pfb_pol45_delay"),
            },
            "ntimes": ntimes,
            "save_dir": self.cfg["save_dir"],
            "redis": self.cfg["redis"],
            "sync_time": sync_time,
            "integration_time": t_int,
            "file_time": file_time,
        }
        return m

    def validate_config(self):
        """
        Ensure that the configuration in `self.cfg` matches the
        actual hardware setup, from `self.header`.

        Raises
        ------
        RuntimeError
            If the configuration does not match the hardware setup.

        """
        fails = []
        for key, value in self.header.items():
            cfg_value = self.cfg.get(key)
            if key == "sync_time":
                continue  # sync_time is not in cfg
            if key in ("pam_atten", "pol_delay"):
                if set(value.keys()) != set(cfg_value.keys()):
                    fails.append(key)
                elif any(value[k] != cfg_value[k] for k in value.keys()):
                    fails.append(key)
            elif value != cfg_value:
                fails.append(key)
        if len(fails) > 0:
            raise RuntimeError(
                "Configuration does not match hardware setup: "
                + ", ".join(fails)
            )

    def initialize(
        self,
        initialize_adc=True,
        initialize_fpga=True,
        sync=True,
        update_redis=True,
    ):
        """
        Initialize the Eigsep correlator.

        Parameters
        ----------
        initialize_adc : bool
            Initialize the ADCs.
        initialize_fpga : bool
            Initialize the FPGA.
        sync : bool
            Synchronize the correlator clock.
        update_redis : bool
            Update Redis with sync time. Only used if `sync` is True.

        Notes
        -----
        This is a convenience method that calls the methods
            - `initialize_adc`
            - `initialize_fpga`
            - `set_input`
            - `synchronize`
        in the specified order with their default parameters.

        """
        if initialize_adc:
            self.logger.debug("Initializing ADCs.")
            self.initialize_adc()
        if initialize_fpga:
            self.logger.debug("Initializing FPGA.")
            self.initialize_fpga()
        self.set_input()
        if sync:
            self.logger.debug("Synchronizing.")
            self.synchronize(update_redis=update_redis)

    def _run_adc_test(self, test, n_tries):
        """
        Run a test and retry if it fails.

        Parameters
        ----------
        test : callable
            The test to run. Must return a list of failed tests.
        n_tries : int
            Number of attempts at each test before giving up.

        Raises
        ------
        RuntimeError
            If the tests do not pass after n_tries attempts.

        """
        fails = test()
        tries = 1
        while len(fails) > 0:
            self.logger.warning(f" {test.__name__} failed on: " + str(fails))
            fails = test()
            tries += 1
            if tries > n_tries:
                raise RuntimeError(f"test failed after {tries} tries")

    def initialize_adc(self, n_tries=10):
        """
        Initialize the ADC. Aligns the clock and data lanes, and runs a ramp
        test.

        Parameters
        ----------
        n_tries : int
            Number of attempts at each test before giving up. Default 10.

        Raises
        ------
        RuntimeError
            If the tests do not pass after n_tries attempts.

        Notes
        -----
        This is called by `initialize` but can be called separately.

        """
        sample_rate = self.cfg["sample_rate"]
        gain = self.cfg["adc_gain"]

        self.logger.info("Initializing ADCs")
        self.adc.init(sample_rate=sample_rate)

        self._run_adc_test(self.adc.alignLineClock, n_tries=n_tries)
        self._run_adc_test(self.adc.alignFrameClock, n_tries=n_tries)
        self._run_adc_test(self.adc.rampTest, n_tries=n_tries)

        self.adc.selectADC()
        self.adc.adc.selectInput([1, 1, 3, 3])  # XXX allow as input arg?
        self.adc.set_gain(gain)

        self.adc.sample_rate = int(sample_rate * 1e6)  # in Hz
        self.adc.gain = gain
        self.adc_initialized = True

    def initialize_fpga(self, verify=False):
        """
        Initialize the correlator.

        Notes
        -----
        This is called by `initialize` but can be called separately.

        """
        fft_shift = self.cfg["fft_shift"]
        corr_acc_len = self.cfg["corr_acc_len"]
        corr_scalar = self.cfg["corr_scalar"]
        pol_delay = self.cfg["pol_delay"]

        for blk in self.blocks:
            blk.initialize()
        try:
            # initialize pams
            self.initialize_pams()
        except OSError:
            self.logger.warn("Couldn't initialize PAMs.")
            pass
        self.logger.info(f"Setting FFT_SHIFT: {fft_shift}")
        self.pfb.set_fft_shift(fft_shift)
        self.logger.info(f"Setting CORR_ACC_LEN: {corr_acc_len}")
        self.fpga.write_int("corr_acc_len", corr_acc_len)
        self.logger.info(f"Setting CORR_SCALAR: {corr_scalar}")
        self.fpga.write_int("corr_scalar", corr_scalar)
        if verify:
            assert self.fpga.read_uint("corr_acc_len") == corr_acc_len
            assert self.fpga.read_uint("corr_scalar") == corr_scalar
        self.set_pol_delay(pol_delay, verify=verify)

    def set_input(self):
        """
        Set the input to either noise or ADC based on the configuration.
        This method is called after initializing the ADC and FPGA.

        Notes
        -----
        This is called by `initialize`. Can be called separately
        to change the input after initialization.
        """
        self.noise.set_seed(stream=None, seed=0)
        if self.cfg["use_noise"]:
            self.logger.warning("Switching to noise input.")
            self.inp.use_noise(stream=None)
            self.sync.arm_noise()
            for i in range(3):
                self.sync.sw_sync()
            self.logger.info("Synchronized noise")
        else:
            self.logger.info("Switching to ADC input.")
            self.inp.use_adc(stream=None)

    def set_pol_delay(self, delay, verify=False):
        """
        Delay one or more input channels. The same delay is applied to both
        polarizations, so it can be set for 01, 23, and 45.

        Parameters
        ----------
        delay : dict
            Keys are "01", "23", and "45". Values (int) are the delay in
            clock cycles. Max 1024 (2^10).

        Notes
        -----
        This is called by `initialize_fpga`. Can be called separately
        to change the delay after initialization.

        """
        for key in ["01", "23", "45"]:
            dly = delay.get(key, 0)
            self.logger.info(f"Setting POL{key}_DELAY: {dly}")
            self.fpga.write_int(f"pfb_pol{key}_delay", dly)
            if verify:
                assert self.fpga.read_uint(f"pfb_pol{key}_delay") == dly

    def initialize_pams(self):
        """
        Initialize the PAMs.

        Parameters
        ----------
        attenuation : dict
            Dictionary of attenuation values for each PAM. Keys are antenna
            numbers, values are tuples of (east, north) attenuation values.

        Notes
        -----
        This is called by `initialize_fpga`. Can be called separately
        to change the attenuation after initialization.

        """
        attenuation = self.cfg["pam_atten"]

        self.pams = []
        for p, (att_e, att_n) in attenuation.items():
            pam = Pam(self.fpga, f"i2c_ant{p}")
            pam.initialize()
            self.logger.info(
                f"Setting pam{p} attenuation to ({att_e},{att_n})"
            )
            pam.set_attenuation(att_e, att_n)
            self.pams.append(pam)
        self.blocks.extend(self.pams)
        self.pams_initialized = True

    def synchronize(self, delay=0, update_redis=True):
        """
        Synchronize the correlator clock.

        Parameters
        ----------
        delay : int
            Dealy in FPGA clock ticks between arrival of an external
            sync pulse and the issuing of an internal trigger.
        update_redis : bool
            Whether to update Redis with the synchronization time.

        """
        self.sync.set_delay(delay)
        self.sync.arm_sync()
        for i in range(3):
            self.sync.sw_sync()
            sync_time = time.time()  # not an int unless 1PPS is provided
            self.logger.info(f"Synchronized at {sync_time}.")
        self.sync_time = sync_time
        if update_redis:
            self.redis.set("SYNC_TIME", str(sync_time))
            self.redis.set(
                "SYNC_DATE",
                datetime.datetime.fromtimestamp(sync_time).isoformat(),
            )
        self.is_synchronized = True

    def unpack_data(self, data):
        """
        Unpack raw correlation data into numpy arrays.

        Parameters
        ----------
        data : dict
            Dictionary with keys as the input pairs and values as raw
            correlation data in bytes.

        Returns
        -------
        dict
            Dictionary with keys as the input pairs and values as numpy arrays
            of unpacked correlation data.

        """
        dt = self.cfg["dtype"]
        return {k: np.frombuffer(v, dtype=dt) for k, v in data.items()}

    def _read_spec(self, spec_type, i, unpack):
        """
        Read a single spectrum from the FPGA. This is a helper method for
        read_auto and read_cross and should not be called directly.

        Parameters
        ----------
        spec_type : str
            The type of spectrum to read, either 'auto' or 'cross'.
        i : str, list of str, or None
            The identifier of the spectrum to read, e.g. '0', '02'. If None,
            reads all spectra of the specified type.
        unpack : bool
            Whether to unpack the data into numpy arrays. Default is False,
            which returns raw bytes.

        Returns
        -------
        spec : bytes or numpy array
            The spectrum data. If unpack is True, returns a numpy array of
            integers, otherwise returns raw bytes.

        """
        if i is None:
            if spec_type == "auto":
                i = self.autos
            else:
                i = self.crosses
        elif isinstance(i, str):
            i = [i]
        if len(i) == 0:
            return {}
        # total number of bytes to read, factor of 2 is for odd/even
        nbytes = self.cfg["corr_word"] * 2 * self.cfg["nchan"]
        if spec_type == "cross":
            nbytes *= 2  # real/imag for cross correlations
        spec = {}
        for k in i:
            key = f"corr_{spec_type}_{k}_dout"
            data = self.fpga.read(key, nbytes)
            spec[k] = data
        if unpack:
            spec = self.unpack_data(spec)
        return spec

    def read_auto(self, i=None, unpack=False):
        """
        Read the i'th (counting from 0) autocorrelation spectrum.

        Parameters
        ----------
        i : str
            Which autocorrelation to read. Default is None, which reads all
            autocorrelations.
        unpack : bool
            Whether to unpack the data into numpy arrays. Default is False,
            which returns raw bytes.

        Returns
        -------
        spec : dict
            Dictionary with keys as autocorrelation identifiers and values as
            the corresponding spectra. If unpack is True, values are numpy
            arrays of integers.

        Notes
        -----
        The first half of the data is the 'even' integration, and the second
        half is the 'odd' integration.

        """
        return self._read_spec("auto", i, unpack)

    def read_cross(self, ij=None, unpack=False):
        """
        Read the cross-correlation spectrum between inputs i and j.

        Parameters
        ----------
        ij : str
            Which cross-correlation to read. Default is None, which reads all
            cross-correlations.
        unpack : bool
            Whether to unpack the data into numpy arrays. Default is False,
            which returns raw bytes.

        Returns
        -------
        spec : dict
            Dictionary with keys as autocorrelation identifiers and values as
            the corresponding spectra. If unpack is True, values are numpy
            arrays of integers.

        Notes
        -----
        The first half of the data is the 'even' integration, and the second
        half is the 'odd' integration. Real and imaginary parts are
        interleaved, with every other sample being the real part and the
        following sample being the imaginary part.

        """
        return self._read_spec("cross", ij, unpack)

    def read_data(self, pairs=None, unpack=False):
        """
        Read even/odd spectra for correlations specified in pairs.

        Parameters
        ----------
        pairs : str, list of str or None
            List of pairs to read. If None, reads all pairs. If a string,
            reads the single pair specified.
        unpack : bool
            Whether to unpack the data into numpy arrays. Default is False,
            which returns raw bytes.

        Returns
        -------
        data : dict
            Dictionary with keys as correlation identifiers and values as
            the corresponding spectra. If unpack is True, values are numpy
            arrays of integers.

        """
        data = {}
        if pairs is None:
            pairs = self.autos + self.crosses
        elif type(pairs) is str:
            pairs = [pairs]
        data = self.read_auto([p for p in pairs if len(p) == 1], unpack=unpack)
        data.update(
            self.read_cross([p for p in pairs if len(p) != 1], unpack=unpack)
        )
        return data

    def update_redis(self, data, cnt):
        """
        Update redis database with data from first half ("even")
        integrations.

        Parameters
        ----------
        data : dict
            Dictionary with keys as correlation identifiers and values as
            the corresponding spectra.
        cnt : int
            The current integration count.

        """
        corr_word = self.cfg["corr_word"]
        nchan = self.cfg["nchan"]
        spec_len = corr_word * nchan
        for p, d in data.items():
            if len(p) == 1:
                d = d[:spec_len]  # only one spectrum for auto
            else:
                d = d[: 2 * spec_len]  # two for real/imag
            self.redis.set(f"data:{p}", d)
        self.redis.set("ACC_CNT", cnt)
        self.redis.set("updated_unix", int(time.time()))
        self.redis.set("updated_date", datetime.datetime.now().isoformat())

    def _read_integrations(self, pairs, timeout=10):
        """
        Read integrated correlations from the SNAP board.

        Parameters
        ----------
        pairs : list
            List of pairs to read.
        timeout : float
            Number of seconds to wait for a new integration before timing out.

        """
        cnt = self.fpga.read_int("corr_acc_cnt")
        t = time.time()

        while time.time() < t + timeout and not self.event.is_set():
            new_cnt = self.fpga.read_int("corr_acc_cnt")
            if new_cnt == cnt:
                time.sleep(0.01)
                continue
            if new_cnt > cnt + 1:
                self.logger.warning(
                    f"Missed {new_cnt - cnt - 1} integration(s)."
                )
            cnt = new_cnt
            self.logger.info(f"Reading acc_cnt={cnt}")
            data = self.read_data(pairs=pairs, unpack=False)
            if cnt != self.fpga.read_int("corr_acc_cnt"):
                self.logger.error(
                    f"Read of acc_cnt={cnt} FAILED to complete before "
                    "next integration."
                )
            self.queue.put({"data": data, "cnt": cnt})
            t = time.time()

    def end_observing(self):
        try:
            self.event.set()
            self.queue.put(None)  # signals end of observing
        except AttributeError:
            pass

    def observe(
        self,
        save_dir,
        pairs=None,
        timeout=10,
        update_redis=True,
        write_files=True,
        header=io.DEFAULT_HEADER,
    ):
        """
        Observe continuously.

        Parameters
        ----------
        save_dir: str
            Destination directory to write files.
        pairs : list
            List of pairs to read. Default is None, which reads all pairs.
        timeout : float
            Number of seconds to wait for a new integration before returning.
        update_redis : bool
            Whether to update redis.
        write_files : bool
            Whether to write data to files.
        ntimes : int
            Number of integrations to write to each file. Default is
            io.DEFAULT_NTIMES (60).
        header : dict
            Header to write to each file. Default is io.DEFAULT_HEADER.
        """
        self.file = None
        self.queue = Queue(maxsize=0)  # XXX infinite size
        self.event = Event()
        ntimes = self.cfg["ntimes"]
        if pairs is None:
            pairs = self.autos + self.crosses

        self.logger.debug("Start reading integrations.")
        thd = Thread(
            target=self._read_integrations,
            args=(pairs,),
            kwargs={"timeout": timeout},
        )
        thd.start()

        if write_files:
            # update header
            for k, v in self.header.items():
                header[k] = v
            self.file = io.File(save_dir, ntimes=ntimes, header=header)

        while not self.event.is_set() or not self.queue.empty():
            d = self.queue.get()
            if d is None:
                if self.event.is_set():
                    self.logger.info("End of queue, processing finished.")
                    break
                else:
                    continue
            data = d["data"]
            cnt = d["cnt"]
            if update_redis:
                self.update_redis(data, cnt)
            if write_files:
                filename = self.file.add_data(data, cnt)
                if filename is not None:
                    self.logger.info(f"Wrote file {filename}")
        if self.file is not None:
            if len(self.file) > 0:
                self.logger.info("Writing short final file.")
                self.file.corr_write()

        thd.join()
        self.logger.info("Done observing.")
