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
import time
import numpy as np
import casperfpga
from casperfpga.transport_tapcp import TapcpTransport

from . import io
from .blocks import Input, Fem, NoiseGen, Pam, Pfb, Sync

SNAP_IP = "10.10.10.236"
SAMPLE_RATE = 500  # MHz
FPG_FILE = (
    "/home/eigsep/eigsep_corr/eigsep_fengine_1g_v2_0_2023-09-30_1811.fpg"
)
ADC_GAIN = 4
FFT_SHIFT = 0x0055  #
CORR_ACC_LEN = 2**28  # makes corr_acc_cnt increment by ~1 per second
CORR_SCALAR = 2**9  # correlator uses 8 bits after binary point so 2**9 = 1
CORR_WORD = 4  # bytes
DEFAULT_PAM_ATTEN = {0: (8, 8), 1: (8, 8), 2: (8, 8)}
DEFAULT_POL0_DELAY = 0
N_FEMS = 0  # set to 0 since they're not initialized from SNAP
NCHAN = 1024
REDIS_HOST = "localhost"
REDIS_PORT = 6379
# XXX suggest making the below part of observing script, not module
# DATA_PATH = "/media/eigsep/T7/data"


class EigsepFpga:
    def __init__(
        self,
        snap_ip=SNAP_IP,
        fpg_file=FPG_FILE,
        program=False,
        ref=None,
        transport=TapcpTransport,
        logger=None,
    ):
        """
        Class for interfacing with the SNAP board.

        Parameters
        ----------
        snap_ip : str
            The IP address of the SNAP board. The two used for EIGSEP are
            10.10.10.13 and 10.10.10.236.
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

        """
        if logger is None:
            logger = logging.getLogger(__name__)
            logging.basicConfig(filename="snap.log", level=logging.DEBUG)
        self.logger = logger

        self.fpg_file = fpg_file
        self.fpga = casperfpga.CasperFpga(snap_ip, transport=transport)
        if program:
            self.fpga.upload_to_ram_and_program(self.fpg_file)

        # blocks
        # self.synth = casperfpga.synth.LMX2581(self.fpga, "synth", fosc=10)
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

        # io.File object for saving data (gets instantiated in write_file)
        self.savefile = None

    @property
    def metadata(self):
        m = {
            "snap_ip": self.fpga.host,
            "fpg_version": self.fpga.read_int("version_version"),
            "fpg_file": self.fpg_file,
            "nchan": NCHAN,
            "adc_sample_rate": self.adc.sample_rate,
            "adc_gain": self.adc.gain,
            "fft_shift": self.pfb.fft_shift,
            "corr_acc_len": self.fpga.read_uint("corr_acc_len"),
            "corr_scalar": self.fpga.read_uint("corr_scalar"),
            "pol0_delay": self.fpga.read_uint("pfb_pol0_delay"),
            "n_pams": len(self.pams),
            "n_fems": len(self.fems),
            "sync_time": self.sync_time,
        }
        for i, pam in self.pams:
            m[f"pam{i}_attenuation"] = pam.get_attenuation()
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

    def initialize_fpga(
        self,
        fft_shift=FFT_SHIFT,
        corr_acc_len=CORR_ACC_LEN,
        corr_scalar=CORR_SCALAR,
        pol0_delay=DEFAULT_POL0_DELAY,
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
        # initialize pams
        self.initialize_pams(pam_atten=pam_atten)
        # initialize fems
        self.initialize_fems(N=n_fems)
        self.logger.info(f"Setting FFT_SHIFT: {fft_shift}")
        self.pfb.set_fft_shift(fft_shift)
        self.logger.info(f"Setting CORR_ACC_LEN: {corr_acc_len}")
        self.fpga.write_int("corr_acc_len", corr_acc_len)
        self.logger.info(f"Setting CORR_SCALAR: {corr_scalar}")
        self.fpga.write_int("corr_scalar", corr_scalar)
        if verify:
            assert self.fpga.read_uint("corr_acc_len") == corr_acc_len
            assert self.fpga.read_uint("corr_scalar") == corr_scalar
        self.set_pol0_delay(delay=pol0_delay, verify=verify)

    def set_pol0_delay(self, delay, verify=False):
        """
        Set the delay for the pol0 and pol1 inputs.

        Parameters
        ----------
        delay : int
            The delay in clock cycles.

        """
        self.logger.info(f"Setting POL0_DELAY: {delay}")
        self.fpga.write_int("pfb_pol0_delay", delay)
        if verify:
            assert self.fpga.read_uint("pfb_pol0_delay") == delay

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

    @property
    def sync_time(self):
        try:
            return self.redis.get("SYNC_TIME")
        except (redis.RedisError, KeyError):  # XXX check error
            return None

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

    def write_file(self, data, cnt, save_dir):
        """
        Write data to file. Stores data to a buffer until the buffer is full,
        then writes to file and instantiates a new savefile object.

        Parameters
        ----------
        data : dict
            The data to write to file.
        cnt : int
            The acc count of the data.
        save_dir : str
            The directory to save the data to.

        """
        # XXX any chance of 2 files in same second?
        # check if we need to instantiate a new savefile object
        if self.savefile is None:
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{save_dir}/{date}.json"
            self.savefile = io.File(fname, self.sync_time)
        # add data to buffer
        self.savefile.add_data(data, cnt)
        # write to file if buffer is full
        if cnt > self.savefile.max_cnt:
            self.savefile.write()
            self.savefile = None  # reset savefile object

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

    def observe(
        self,
        save_dir,
        pairs=None,
        timeout=10,
        update_redis=True,
        write_files=True,
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

        """
        # XXX make threaded, push/pop integrations from Queue, write integrations
        # as they arrive and finish files/start new ones
        cnt = self.fpga.read_int("corr_acc_cnt")
        t = time.time()

        while time.time() < t + timeout:
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
                    f"Read of acc_cnt={cnt} FAILED to complete before next integration."
                )

            # XXX move these into separate thread to avoid missing integrations
            if update_redis:
                self.update_redis(data, cnt)
            if write_files:
                self.savefile = None
                self.write_file(data, cnt, save_dir)
            t = time.time()
