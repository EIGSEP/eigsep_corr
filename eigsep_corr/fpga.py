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
import redis
import struct
import os
import time
from threading import Event, Thread
from queue import Queue
import numpy as np

try:
    import casperfpga
    from casperfpga.transport_tapcp import TapcpTransport
except ImportError:
    logging.warning("Running without casperfpga installed")
    TapcpTransport = None

from . import io
from .blocks import Input, Fem, NoiseGen, Pam, Pfb, Sync
from .data import DATA_PATH

SNAP_IP = "10.10.10.236"
SAMPLE_RATE = 500  # MHz
FPG_FILE = os.path.join(
    DATA_PATH, "eigsep_fengine_1g_v2_3_2024-07-08_1858.fpg"
)
FPG_VERSION = (2, 3)  # major, minor
ADC_GAIN = 4
FFT_SHIFT = 0x0055
CORR_ACC_LEN = 2**28  # makes corr_acc_cnt increment by ~1 per second
CORR_SCALAR = 2**9  # correlator uses 8 bits after binary point so 2**9 = 1
CORR_WORD = 4  # bytes
DEFAULT_PAM_ATTEN = {0: (8, 8), 1: (8, 8), 2: (8, 8)}
DEFAULT_POL_DELAY = {"01": 0, "23": 0, "45": 0}
N_FEMS = 0  # set to 0 since they're not initialized from SNAP
NCHAN = 1024
REDIS_HOST = "localhost"
REDIS_PORT = 6379


class EigsepFpga:
    def __init__(
        self,
        snap_ip=SNAP_IP,
        fpg_file=FPG_FILE,
        program=False,
        ref=None,
        transport=TapcpTransport,
        logger=None,
        read_accelerometer=False,
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
        ref : int
            The reference clock frequency in MHz. If None, uses the
            500 MHz clock on the SNAP board. Typically set to None or 10.
        transport : casperfpga.transport_tapcp.TapcpTransport
            The transport protocol to use. The default is TapcpTransport.
        logger : logging.Logger
            The logger to use. If None, creates a new logger.
        read_accelerometer : bool
            Whether to read accelerometer data from the platform
            FEM. Default is False.
        force_program : bool
            If program is True, decide whether to force casperfpga to program or not. By
            default, casperfpga skips the programming if the filename is the same, but
            this flag overrides that.
        read_accelerometer : bool
            Whether to read accelerometer data from the platform
            FEM. Default is False.
        force_program : bool
            If program is True, decide whether to force casperfpga to program or not. By
            default, casperfpga skips the programming if the filename is the same, but
            this flag overrides that.

        """
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
        self.logger = logger

        self.fpg_file = fpg_file
        self.fpga = casperfpga.CasperFpga(snap_ip, transport=transport)
        if program:
            self.fpga.upload_to_ram_and_program(self.fpg_file, force=force_program)

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

        self.redis = redis.Redis(REDIS_HOST, port=REDIS_PORT)

        self.adc_initialized = False
        self.pams_initialized = False
        self.is_synchronized = False

        self.file = None
        self.queue = None
        self.event = None

        # accelerometer data from FEM on platform
        if read_accelerometer:
            self.platform_redis = redis.Redis(host="10.10.10.12", port=6379)
        else:
            self.platform_redis = None

    @property
    def version(self):
        val = self.fpga.read_uint("version_version")
        major = val >> 16
        minor = val & 0xFFFF
        return (major, minor)

    def check_version(self):
        assert self.version == FPG_VERSION

    @property
    def metadata(self):
        m = {
            "nchan": NCHAN,
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
        if self.platform_redis is not None:
            theta = str(self.platform_redis.get("theta"))
            phi = str(self.platform_redis.get("phi"))
            print(theta, phi)
            accel = {
                "theta": theta,
                "phi": phi,
            }
            m["accelerometer"] = accel
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

    def initialize_adc(
        self, sample_rate=SAMPLE_RATE, gain=ADC_GAIN, n_tries=10
    ):
        """
        Initialize the ADC. Aligns the clock and data lanes, and runs a ramp
        test.

        Parameters
        ----------
        sample_rate : int
            The sample rate in MHz. Default SAMPLE_RATE.
        gain : int
            The gain of the ADC. Default ADC_GAIN.
        n_tries : int
            Number of attempts at each test before giving up. Default 10.

        Raises
        ------
        RuntimeError
            If the tests do not pass after n_tries attempts.
        """
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
        fft_shift=FFT_SHIFT,
        corr_acc_len=CORR_ACC_LEN,
        corr_scalar=CORR_SCALAR,
        pol_delay=DEFAULT_POL_DELAY,
        pam_atten=DEFAULT_PAM_ATTEN,
        n_fems=N_FEMS,
        verify=False,
    ):
        """
        Initialize the correlator.

        Parameters
        ----------
        corr_acc_len : int (power of 2)
            The accumulation length.  Default CORR_ACC_LEN.
        corr_scalar : int (power of 2)
            Scalar that is multiplied to each correlation. Default CORR_SCALAR.

        """
        for blk in self.blocks:
            blk.initialize()
        try:
            # initialize pams
            self.initialize_pams(attenuation=pam_atten)
            # initialize fems
            self.initialize_fems(N=n_fems)
        except OSError:
            self.logger.warn("Couldn't initialize PAMs and FEMs")
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

    def initialize_pams(self, attenuation=DEFAULT_PAM_ATTEN):
        """
        Initialize the PAMs.

        Parameters
        ----------
        attenuation : dict
            Dictionary of attenuation values for each PAM. Keys are antenna
            numbers, values are tuples of (east, north) attenuation values.

        """
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

    def initialize_fems(self, N=N_FEMS):
        """
        Initialize the FEMs.

        Parameters
        ----------
        N : int
           Number of FEMs to initialize.

        """
        self.logger.info(f"Attaching {N} FEMs")
        self.fems = [Fem(self.fpga, f"i2c_ant{i}") for i in range(N)]
        for fem in self.fems:
            fem.initialize()
        self.blocks.extend(self.fems)
        self.fems_initialized = True

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

    # XXX check read_auto(i=[])
    def read_auto(self, i=None, unpack=False):
        """
        Read the i'th (counting from 0) autocorrelation spectrum.

        Parameters
        ----------
        i : str
            Which autocorrelation to read. Default is None, which reads all
            autocorrelations.
        """
        if i is None:
            i = self.autos
        elif type(i) is str:
            i = [i]
        nbytes = CORR_WORD * 2 * NCHAN  # odd/even
        spec = {k: self.fpga.read(f"corr_auto_{k}_dout", nbytes) for k in i}
        if unpack:
            spec = {
                k: np.array(struct.unpack(f">{nbytes // CORR_WORD}l", v))
                for k, v in spec.items()
            }
        return spec

    # XXX check read_cross(ij=[])
    def read_cross(self, ij=None, unpack=False):
        """
        Read the cross correlation spectrum between inputs i and j.

        Parameters
        ----------
        ij : str
            Which correlation to read, e.g. "02". Assuming N<M. Default is
            None, which reads all cross correlations.
        """
        if ij is None:
            ij = self.crosses
        elif type(ij) is str:
            ij = [ij]
        nbytes = CORR_WORD * 2 * 2 * NCHAN  # odd/even, real/imag
        spec = {k: self.fpga.read(f"corr_cross_{k}_dout", nbytes) for k in ij}
        if unpack:
            spec = {
                k: np.array(struct.unpack(f">{nbytes // CORR_WORD}l", spec))
                for k, v in spec.items()
            }
        return spec

    def read_data(self, pairs=None, unpack=False):
        """
        Read even/odd spectra for correlations specified in pairs.
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
        """Update redis database with data from first half ("even")
        integrations."""
        for p, d in data.items():
            if len(p) == 1:
                d = d[: CORR_WORD * NCHAN]
            else:
                d = d[: CORR_WORD * 2 * NCHAN]  # two for real/imag
            self.redis.set(f"data:{p}", d)
        self.redis.set("ACC_CNT", cnt)
        self.redis.set("updated_unix", int(time.time()))
        self.redis.set("updated_date", datetime.datetime.now().isoformat())

    def _read_integrations(self, pairs, timeout=10):
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
        read_accelerometer=False,
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
        read_accelerometer : bool
            Grab accelerometer data from redis and write to file. Default is
            False.

        """
        self.queue = Queue(maxsize=0)  # XXX infinite size
        self.event = Event()

        thd = Thread(target=self._read_integrations, args=(pairs, timeout))
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
        if self.file is not None and len(self.file) > 0:
            self.logger.info("Writing short final file.")
            self.file.corr_write()

        thd.join()
        self.logger.info("Done observing.")
