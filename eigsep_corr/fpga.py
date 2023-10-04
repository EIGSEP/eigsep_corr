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
SAMPLE_RATE = 500
ADC_GAIN = 4
FFT_SHIFT = 0x0055
CORR_ACC_LEN = 2**28  # makes corr_acc_cnt increment by ~1 per second
CORR_SCALAR = 2**9  # correlator uses 8 bits after binary point so 2**9 = 1
N_PAMS = 3
N_FEMS = 0  # set to 0 since they're not initialized from SNAP
NCHAN = 1024
REDIS_HOST = "localhost"
REDIS_PORT = 6379
DATA_PATH = "/media/eigsep/T7/data"


class EigsepFpga:
    def __init__(
        self,
        snap_ip=SNAP_IP,
        fpg_file=None,
        transport=TapcpTransport,
        logger=None,
    ):
        if logger is None:
            logging.getLogger().setLevel(logging.DEBUG)
            logger = logging.getLogger(__name__)
        self.logger = logger

        self.fpga = casperfpga.CasperFpga(snap_ip, transport=transport)
        if fpg_file is not None:
            self.fpg_file = fpg_file
            self.fpga.upload_to_ram_and_program(self.fpg_file)

        # blocks
        self.synth = casperfpga.synth.LMX2581(self.fpga, "synth", fosc=10)
        self.adc = casperfpga.snapadc.SnapAdc(
            self.fpga, num_chans=2, resolution=8, ref=10
        )
        self.sync = Sync(self.fpga, "sync")
        self.noise = NoiseGen(self.fpga, "noise", nstreams=6)
        self.inp = Input(self.fpga, "input", nstreams=12)
        self.pfb = Pfb(self.fpga, "pfb")
        self.blocks = [self.sync, self.noise, self.inp, self.pfb]

        self.autos = ["0", "1", "2", "3", "4", "5"]
        self.crosses = ["02", "13", "24", "35", "04", "15"]

        self.redis = redis.Redis(REDIS_HOST, port=REDIS_PORT)
        self.redis.set("sync_time", 0)  # set to 0 until sync is run
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
            "pfb_fft_shift": self.pfb.fft_shift,
            "corr_acc_len": self.fpga.read_uint("corr_acc_len"),
            "corr_scalar": self.fpga.read_uint("corr_scalar"),
            "n_pams": len(self.pams),
            "n_fems": len(self.fems),
            "pam_attenuation": self.pams[0].get_attenuation(),
            "data_path": DATA_PATH,
            "sync_time": self.sync_time,
        }
        # only v2_1 has pfb_pol0_delay
        if "pfb_pol0_delay" in self.fpga.listdev():
            m["pol0_delay"] = self.fpga.read_uint("pfb_pol0_delay")
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

    def initialize_adc(self, sample_rate, gain, n_tries=10):
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
            Number of attempts at each test before giving up.

        Raises
        ------
        RuntimeError
            If the tests do not pass after n_tries attempts.
        """
        self.adc.init(sample_rate=sample_rate)

        self._run_adc_test(self.adc.alignLineClock, n_tries=n_tries)
        self._run_adc_test(self.adc.alignFrameClock, n_tries=n_tries)
        self._run_adc_test(self.adc.rampTest, n_tries=n_tries)

        self.adc.selectADC()
        self.adc.adc.selectInput([1, 1, 3, 3])  # XXX allow as input arg?
        self.adc.set_gain(gain)

    def initialize_fpga(self, corr_acc_len, corr_scalar):
        """
        Initialize the correlator.

        Parameters
        ----------
        corr_acc_len : int (power of 2)
            The accumulation length.
        corr_scalar : int (power of 2)
            Scalar that is multiplied to each correlation.

        """
        self.fpga.write_int("corr_acc_len", corr_acc_len)
        self.fpga.write_int("corr_scalar", corr_scalar)

    def set_pol0_delay(self, delay=0):
        """
        Set the delay for the pol0 input.

        Parameters
        ----------
        delay : int
            The delay in clock cycles.

        """
        self.fpga.write_int("pfb_pol0_delay", delay)

    def initialize_pams(self, N):
        """
        Initialize the PAMs.

        Parameters
        ----------
        N : int
           Number of PAMs to initialize.

        """
        self.pams = [Pam(self.fpga, f"i2c_ant{i}") for i in range(N)]
        for pam in self.pams:
            pam.initialize()
            pam.set_attenuation(8, 8)  # XXX

    def initialize_fems(self, N):
        """
        Initialize the FEMs.

        Parameters
        ----------
        N : int
           Number of FEMs to initialize.

        """
        self.fems = [Fem(self.fpga, f"i2c_ant{i}") for i in range(N)]
        for fem in self.fems:
            fem.initialize()

    def initialize(
        self,
        adc_sample_rate=SAMPLE_RATE,
        adc_gain=ADC_GAIN,
        pfb_fft_shift=FFT_SHIFT,
        corr_acc_len=CORR_ACC_LEN,
        corr_scalar=CORR_SCALAR,
        n_pams=N_PAMS,
        n_fems=N_FEMS,
    ):
        self.initialize_adc(adc_sample_rate, adc_gain)
        for blk in self.blocks:
            blk.initialize()
        self.initialize_fpga(corr_acc_len, corr_scalar)
        # initialize pams
        if n_pams > 0:
            self.initialize_pams(N=n_pams)
            self.blocks.extend(self.pams)
        # initialize fems
        if n_fems > 0:
            self.initialize_fems(N=n_fems)
            self.blocks.extend(self.fems)
        self.synchronize()
        self.pfb.set_fft_shift(pfb_fft_shift)

    def synchronize(self, delay=0):
        self.sync.set_delay(delay)
        self.sync.arm_sync()
        for i in range(3):
            self.sync.sw_sync()
            sync_time = int(time.time())
            self.logger.info(f"Synchronized at {sync_time}.")
        self.redis.set("sync_time", sync_time)

    @property
    def sync_time(self):
        return self.redis.get("sync_time")

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
            return np.array([self.read_auto(i=a) for a in self.autos])
        name = f"corr_auto_{i}_dout"
        spec = self.fpga.read(name, 4 * 2 * NCHAN)
        if unpack:
            spec = np.array(struct.unpack(f">{2 * NCHAN}l", spec))
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
            return np.array([self.read_cross(ij=x) for x in self.crosses])
        name = f"corr_cross_{ij}_dout"
        spec = self.fpga.read(name, 4 * 2 * 2 * NCHAN)
        if unpack:
            spec = np.array(struct.unpack(f">{2*2*NCHAN}l", spec))
        return spec

    def read_data(self, pairs=None, unpack=False):
        data = {}
        if pairs is None:
            pairs = self.autos + self.crosses
        for p in pairs:
            if len(p) == 1:
                data[p] = self.read_auto(p, unpack=unpack)
            else:
                data[p] = self.read_cross(p, unpack=unpack)
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
        # check if we need to instantiate a new savefile object
        if self.savefile is None:
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{save_dir}/{date}.json"
            self.savefile = io.File(fname, self.metadata)
        # add data to buffer
        self.savefile.add_data(data, cnt)
        # write to file if buffer is full
        if cnt > self.savefile.max_cnt:
            self.savefile.write()
            self.savefile = None  # reset savefile object

    def update_redis(self, data, cnt):
        for p, d in data.items():
            if len(p) == 1:
                d = d[: 4 * NCHAN]
            else:
                d = d[: 4 * 2 * NCHAN]
            self.redis.set(f"data:{p}", d)
        self.redis.set("ACC_CNT", cnt)
        self.redis.set("updated_unix", int(time.time()))
        self.redis.set("updated_date", datetime.datetime.now().isoformat())

    def observe(
        self,
        pairs=None,
        timeout=10,
        update_redis=True,
        write_files=True,
        save_dir=DATA_PATH,
    ):
        """
        Observe continuously.

        Parameters
        ----------
        pairs : list
            List of pairs to read. Default is None, which reads all pairs.
        timeout : float
            Number of seconds to wait for a new integration before returning.
        update_redis : bool
            Whether to update redis.
        write_files : bool
            Whether to write data to files.
        save_dir : str
            The directory to save the data to. Only used if write_files=True.

        """
        cnt = self.fpga.read_int("corr_acc_cnt")
        t = time.time()
        while time.time() < t + timeout:
            new_cnt = self.fpga.read_int("corr_acc_cnt")
            if new_cnt == cnt:
                time.sleep(0.01)
                continue
            if new_cnt > cnt + 1:
                self.logger.warning(
                    f"Missed {new_cnt - cnt - 1} integrations."
                )
            cnt = new_cnt
            data = self.read_data(pairs=pairs, unpack=False)
            assert cnt == self.fpga.read_int(
                "corr_acc_cnt"
            ), "Ensure read completes before new integration"
            if update_redis:
                self.update_redis(data, cnt)
            if write_files:
                self.savefile = None
                self.write_file(data, cnt, save_dir)
            t = time.time()
