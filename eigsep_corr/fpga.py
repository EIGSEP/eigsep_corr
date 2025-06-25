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

try:
    import casperfpga
    from casperfpga.transport_tapcp import TapcpTransport
except ImportError:
    logging.warning("Running without casperfpga installed")
    TapcpTransport = None

from . import io
from .blocks import Input, NoiseGen, Pam, Pfb, Sync
from .data import DATA_PATH

logger = logging.getLogger(__name__)
SNAP_IP = "10.10.10.236"
FPG_FILE = Path(DATA_PATH) / "eigsep_fengine_1g_v2_3_2024-07-08_1858.fpg"


class EigsepFpga:

    # defaults
    defaults = {
        "sample_rate": 500,  # MHz
        "fpg_version": (2, 3),  # major, minor
        "adc_gain": 4,  # ADC gain
        "fft_shift": 0x0055,  # FFT shift
        "corr_acc_len": 2**28,  # makes corr_acc_cnt increment by ~1 per second
        "corr_scalar": 2
        ** 9,  # 2**9 = 1, correlator uses 8 bits after binary point
        "pam_attenuation": {
            0: (8, 8),
            1: (8, 8),
            2: (8, 8),
        },
        "pol_delay": {
            "01": 0,
            "23": 0,
            "45": 0,
        },
        "nchan": 1024,  # number of channels
    }

    corr_word = 4  # number of bytes per correlation word
    data_type = ">i4"  # numpy data type for correlation data
    redis_host = "localhost"
    redis_port = 6379

    def __init__(
        self,
        snap_ip=SNAP_IP,
        fpg_file=FPG_FILE,
        program=False,
        use_ref=True,
        transport=TapcpTransport,
        force_program=False,
    ):
        """
        Class for interfacing with the SNAP board.

        Parameters
        ----------
        snap_ip : str
            The IP address of the SNAP board. The two used for EIGSEP are
            10.10.10.13 and 10.10.10.18.
        fpg_file : str
            The path to the fpg file to program the SNAP with.
        program : bool
            Whether to program the SNAP with the fpg file.
        use_ref : False
            Supply 10 MHz reference and let SNAP generate sample clock.
            If False, supply 500 MHz clock directly to the SNAP board.
        transport : casperfpga.transport_tapcp.TapcpTransport
            The transport protocol to use. The default is TapcpTransport.
        logger : logging.Logger
            The logger to use. If None, creates a new logger.
        read_accelerometer : bool
            Whether to read accelerometer data from the platform
            FEM. Default is False.
        force_program : bool
            If program is True, decide whether to force casperfpga to program
            or not. By default, casperfpga skips the programming if the
            filename is the same, but this flag overrides that.

        """
        self.logger = logger

        self.fpg_file = fpg_file
        self.fpga = casperfpga.CasperFpga(snap_ip, transport=transport)
        if program:
            self.fpga.upload_to_ram_and_program(
                self.fpg_file, force=force_program
            )

        if use_ref:
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

        self.autos = ["0", "1", "2", "3", "4", "5"]
        self.crosses = ["02", "13", "24", "35", "04", "15"]

        self.redis = redis.Redis(self.redis_host, port=self.redis_port)

        self.adc_initialized = False
        self.pams_initialized = False
        self.is_synchronized = False

        self.file = None
        self.queue = None
        self.event = None

    @property
    def version(self):
        val = self.fpga.read_uint("version_version")
        major = val >> 16
        minor = val & 0xFFFF
        return (major, minor)

    def check_version(self, expected_version=None):
        if expected_version is None:
            expected_version = self.defaults["fpg_version"]
        assert self.version == expected_version

    @property
    def metadata(self):
        """
        This attribute only includes metadata that is not changing during
        observation.

        """
        m = {
            "nchan": self.nchan,
            "fpg_file": self.fpg_file,
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
                int(i): p.get_attenuation() for i, p in enumerate(self.pams)
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
        test.

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
            sample_rate = self.defaults["sample_rate"]
        if gain is None:
            gain = self.defaults["adc_gain"]

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
        Initialize the correlator.

        Parameters
        ----------
        fft_shift : int
        corr_acc_len : int (power of 2)
            The accumulation length.  Default CORR_ACC_LEN.
        corr_scalar : int (power of 2)
            Scalar that is multiplied to each correlation.
        pol_delay : dict
            Keys are "01", "23", and "45". Values (int) are the delay.
        pam_atten : dict
            Keys are antenna numbers, values are tuples of (east, north).

        """
        if fft_shift is None:
            fft_shift = self.defaults["fft_shift"]
        if corr_acc_len is None:
            corr_acc_len = self.defaults["corr_acc_len"]
        if corr_scalar is None:
            corr_scalar = self.defaults["corr_scalar"]
        if pol_delay is None:
            pol_delay = self.defaults["pol_delay"]
        if pam_atten is None:
            pam_atten = self.defaults["pam_attenuation"]

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

    def set_input(self, use_noise=False):
        """
        Set the input to either noise or ADC based on the configuration.
        This method is called after initializing the ADC and FPGA.
        """
        self.noise.set_seed(stream=None, seed=0)
        if use_noise:
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
        Initialize the PAMs.

        Parameters
        ----------
        attenuation : dict
            Dictionary of attenuation values for each PAM. Keys are antenna
            numbers, values are tuples of (east, north) attenuation values.

        """
        if attenuation is None:
            attenuation = self.defaults["pam_attenuation"]

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
        nbytes = self.corr_word * 2 * self.nchan
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
                d = d[: self.corr_word * self.nchan]
            else:
                d = d[: self.corr_word * 2 * self.nchan]  # two for real/imag
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
        ntimes=io.DEFAULT_NTIMES,
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
        self.queue = Queue(maxsize=0)  # XXX infinite size
        self.event = Event()

        thd = Thread(
            target=self._read_integrations,
            args=(pairs,),
            kwargs={"timeout": timeout},
        )
        thd.start()

        if write_files:
            # update header
            for k, v in self.metadata.items():
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
