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
import queue
import time
from threading import Event, Thread
import numpy as np

try:
    import casperfpga
    from casperfpga.transport_tapcp import TapcpTransport
except ImportError:
    logging.warning("Running without casperfpga installed")
    TapcpTransport = None

from eigsep_observing import io
from eigsep_observing import EigsepRedis
from eigsep_observing.config import default_corr_config
from .blocks import Input, NoiseGen, Pam, Pfb, Sync


def add_args(parser, eig_observing=False):
    """
    Add command line arguments to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to.
    eig_observing : bool
        If True, arguments are for the new eigsep_observing code.
        This removes the -r flag because it is impossible to run
        eigsep_observing without redis.

    """
    parser.add_argument(
        "--dummy",
        dest="dummy_mode",
        action="store_true",
        default=False,
        help="Run with a dummy SNAP interface",
    )
    parser.add_argument(
        "-p",
        dest="program",
        action="store_true",
        default=False,
        help="Program the SNAP with the fpg file",
    )
    parser.add_argument(
        "-P",
        dest="force_program",
        action="store_true",
        default=False,
        help="Force programming the SNAP even if fpg file is the same",
    )
    parser.add_argument(
        "--fpg",
        dest="fpg_file",
        default=default_corr_config.fpg_file,
        help="Path to the fpg file",
    )
    parser.add_argument(
        "-a",
        dest="initialize_adc",
        action="store_true",
        default=False,
        help="Initialize the ADCs",
    )
    parser.add_argument(
        "-f",
        dest="initialize_fpga",
        action="store_true",
        default=False,
        help="Initialize the FPGA",
    )
    parser.add_argument(
        "-s",
        dest="sync",
        action="store_true",
        default=False,
        help="Synchronize the FPGA",
    )
    parser.add_argument(
        "-w",
        dest="write_files",
        action="store_true",
        default=False,
        help="Write files",
    )
    parser.add_argument(
        "--ntimes",
        dest="ntimes",
        type=int,
        default=default_corr_config.ntimes,
        help="Number of integrations to write to each file",
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        default=default_corr_config.save_dir,
        help="Directory to save files",
    )

    if not eig_observing:
        parser.add_argument(
            "-r",
            dest="update_redis",
            action="store_true",
            default=False,
            help="Update redis",
        )


class EigsepFpga:
    def __init__(
        self,
        cfg=default_corr_config,
        program=False,
        transport=TapcpTransport,
        logger=None,
        force_program=False,
    ):
        """
        Class for interfacing with the SNAP board.

        Parameters
        ----------
        cfg : eigsep_corr.config.CorrConfig
            The configuration object containing settings for the SNAP.
        program : bool
            Whether to program the SNAP with the fpg file.
        transport : casperfpga.transport_tapcp.TapcpTransport
            The transport protocol to use. The default is TapcpTransport.
        logger : logging.Logger
            The logger to use. If None, creates a new logger.
        force_program : bool
            If program is True, decide whether to force casperfpga to program
            or not. By default, casperfpga skips the programming if the
            filename is the same, but this flag overrides that.

        """
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
        self.logger = logger

        self.cfg = cfg
        self.fpga = casperfpga.CasperFpga(
            self.cfg.snap_ip, transport=transport
        )
        if program:
            self.fpga.upload_to_ram_and_program(
                self.cfg.fpg_file, force=force_program
            )

        if self.cfg.use_ref:
            ref = 10  # use 10 MHz reference clock to generate ADC clocks
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

        self.autos = ["0", "1", "2", "3", "4", "5"]
        self.crosses = ["02", "13", "24", "35", "04", "15"]
        self.data_type = io.build_dtype(*self.cfg.dtype)

        self.redis = EigsepRedis()

        self.adc_initialized = False
        self.pams_initialized = False
        self.is_synchronized = False

        self.file = None
        self.queue = None
        self.pause_event = None
        self.stop_event = None

    @property
    def version(self):
        val = self.fpga.read_uint("version_version")
        major = val >> 16
        minor = val & 0xFFFF
        return (major, minor)

    def check_version(self):
        assert self.version == self.cfg.fpg_version, (
            f"FPGA version {self.version} does not match expected version "
            f"{self.cfg.fpg_version}"
        )

    @property
    def header(self):
        """
        This attribute only includes metadata that is not changing during
        observation. Live metadata (from sensors) is pulled from Redis.

        """
        m = {
            "dtype": self.cfg.dtype,
            "acc_bins": self.cfg.acc_bins,
            "nchan": self.cfg.nchan,
            "fpg_file": self.cfg.fpg_file,
            "fpg_version": self.version,
            "corr_acc_len": self.fpga.read_uint("corr_acc_len"),
            "corr_scalar": self.fpga.read_uint("corr_scalar"),
            "pol01_delay": self.fpga.read_uint("pfb_pol01_delay"),
            "pol23_delay": self.fpga.read_uint("pfb_pol23_delay"),
            "pol45_delay": self.fpga.read_uint("pfb_pol45_delay"),
            "fft_shift": self.pfb.get_fft_shift(),
        }
        if self.adc_initialized:
            m["sample_rate"] = self.adc.sample_rate
            m["gain"] = self.adc.gain
        if self.pams_initialized:
            m["pam_atten"] = {
                str(i): p.get_attenuation() for i, p in enumerate(self.pams)
            }
        if self.is_synchronized:
            m["sync_time"] = self.sync_time
        return m

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

    def initialize_adc(self, sample_rate=None, gain=None, n_tries=10):
        """
        Initialize the ADC. Aligns the clock and data lanes, and runs a ramp
        test. Dedaults are in self.cfg.

        Parameters
        ----------
        sample_rate : int
            The sample rate in MHz.
        gain : int
            The gain of the ADC.
        n_tries : int
            Number of attempts at each test before giving up. Default 10.

        Raises
        ------
        RuntimeError
            If the tests do not pass after n_tries attempts.
        """
        if sample_rate is None:
            sample_rate = self.cfg.sample_rate
        if gain is None:
            gain = self.cfg.adc_gain

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

    def initialize_fpga(
        self,
        fft_shift=None,
        corr_acc_len=None,
        corr_scalar=None,
        pol_delay=None,
        pam_atten=None,
        verify=False,
    ):
        """
        Initialize the correlator. Defaults are in self.cfg.

        Parameters
        ----------
        fft_shift : int
        corr_acc_len : int (power of 2)
            The accumulation length.
        corr_scalar : int (power of 2)
            Scalar that is multiplied to each correlation.
        pol_delay : dict
            Keys are "01", "23", and "45". Values (int) are the delay.
        pam_atten : dict
            Keys are antenna numbers, values are tuples of (east, north).

        """
        if fft_shift is None:
            fft_shift = self.cfg.fft_shift
        if corr_acc_len is None:
            corr_acc_len = self.cfg.corr_acc_len
        if corr_scalar is None:
            corr_scalar = self.cfg.corr_scalar
        if pol_delay is None:
            pol_delay = self.cfg.pol_delay
        if pam_atten is None:
            pam_atten = self.cfg.pam_atten

        for blk in self.blocks:
            blk.initialize()
        try:
            # initialize pams
            self.initialize_pams(attenuation=pam_atten)
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
        self.set_pol_delay(delay=pol_delay, verify=verify)

    def set_input(self):
        """
        Set the input to either noise or ADC based on the configuration.
        This method is called after initializing the ADC and FPGA.
        """
        self.noise.set_seed(stream=None, seed=0)
        if self.cfg.use_noise:
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

        """
        for key in ["01", "23", "45"]:
            dly = delay.get(key, 0)
            if dly != 0:
                self.logger.info(f"Setting POL{key}_DELAY: {dly}")
            self.fpga.write_int(f"pfb_pol{key}_delay", dly)
            if verify:
                assert self.fpga.read_uint(f"pfb_pol{key}_delay") == dly

    def initialize_pams(self, attenuation=None):
        """
        Initialize the PAMs. Defaults are in self.cfg.

        Parameters
        ----------
        attenuation : dict
            Dictionary of attenuation values for each PAM. Keys are antenna
            numbers, values are tuples of (east, north) attenuation values.

        """
        if attenuation is None:
            attenuation = self.cfg.pam_atten

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
        self.sync.set_delay(delay)
        self.sync.arm_sync()
        for i in range(3):
            self.sync.sw_sync()
            sync_time = time.time()  # not an int unless 1PPS is provided
            self.logger.info(f"Synchronized at {sync_time}.")
        self.sync_time = sync_time
        if update_redis:
            self.redis.add_raw("SYNC_TIME", str(sync_time))
            self.redis.add_raw(
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
        return {
            k: np.frombuffer(v, dtype=self.data_type) for k, v in data.items()
        }

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
        nbytes = self.cfg.corr_word * 2 * self.cfg.nchan
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
        for p, d in data.items():
            if len(p) == 1:
                d = d[: self.cfg.corr_word * self.cfg.nchan]
            else:
                d = d[
                    : self.cfg.corr_word * 2 * self.cfg.nchan
                ]  # two for real/imag
            self.redis.add_raw(f"data:{p}", d)
        self.redis.add_metadata("acc_cnt", cnt)
        self.redis.add_metadata("updated_unix", int(time.time()))
        self.redis.add_metadata(
            "updated_date", datetime.datetime.now().isoformat()
        )

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

        try:
            while time.time() < t + timeout and not self.stop_event.is_set():
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
        finally:
            self.end_observing()

    def end_observing(self):
        """End observing thread."""
        self.stop_event.set()
        self.queue.put(None)  # signals end of observing

    def observe(
        self,
        pairs=None,
        timeout=10,
        update_redis=True,
        write_files=True,
    ):
        """
        Observe continuously.

        Parameters
        ----------
        pairs : list
            List of pairs to read. Default is None, which reads all pairs.
        timeout : float
            Number of seconds to wait for a new integration before returning.
        n_ints : int
            Number of integrations to read. Default is None, which reads
            indefinitely.
        update_redis : bool
            Whether to update redis.
        write_files : bool
            Whether to write data to files.
        """
        self.logger.warning(
            "This function is deprecated, use EigObserver.observe from "
            "eigsep_observing instead."
        )
        if pairs is None:
            pairs = self.autos + self.crosses

        self.queue = queue.Queue(maxsize=0)
        self.stop_event = Event()

        thd = Thread(
            target=self._read_integrations,
            args=(pairs),
            kwargs={"timeout": timeout},
        )
        thd.start()

        if write_files:
            self.file = io.File(
                self.cfg.save_dir,
                pairs,
                self.cfg.ntimes,
                self.header,
            )

        while not self.stop_event.is_set():
            try:
                d = self.queue.get(block=True, timeout=timeout)
            except queue.Empty:
                self.logger.warning(
                    f"Queue empty after {timeout} seconds, continuing to "
                    "wait for data."
                )
                continue
            if d is None:
                if self.stop_event.is_set():
                    self.logger.info("End of queue, processing finished.")
                    break
                continue
            data = d["data"]
            cnt = d["cnt"]
            if update_redis:
                self.update_redis(data, cnt)
            if write_files:
                # unpack data from bytes for writing to file
                unpacked_data = self.unpack_data(data)
                filename = self.file.add_data(unpacked_data)
                if filename is not None:
                    self.logger.info(f"Wrote file {filename}")
        if self.file is not None:
            if len(self.file) > 0:
                self.logger.info("Writing short final file.")
                self.file.corr_write()

        thd.join()
        self.logger.info("Done observing.")
