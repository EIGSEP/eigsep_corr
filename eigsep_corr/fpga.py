import datetime
import logging
import redis
import struct
import time
import numpy as np
import casperfpga
from casperfpga.transport_tapcp import TapcpTransport
from eigsep_corr.blocks import Input, Fem, NoiseGen, Pam, Pfb, Sync

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
DATA_PATH = "/media/eigsep/T7/data"  # XXX need one for each ssd
NSPEC = 60  # number of spectra to accumulate before writing to disk


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

    def initialize_adc(self, sample_rate, gain):
        self.adc.init(sample_rate=sample_rate)

        # Align clock and data lanes of ADC.
        fails = self.adc.alignLineClock()
        while len(fails) > 0:
            self.logger.warning("alignLineClock failed on: " + str(fails))
            fails = self.adc.alignLineClock()
        fails = self.adc.alignFrameClock()
        while len(fails) > 0:
            self.logger.warning("alignFrameClock failed on: " + str(fails))
            fails = self.adc.alignFrameClock()
        fails = self.adc.rampTest()
        while len(fails) > 0:
            self.logger.warning("rampTest failed on: " + str(fails))
            fails = self.adc.rampTest()

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

    def write_file(self, data, cnt, nspec, save_dir):
        """
        Write the data to a file.

        Parameters
        ----------
        data : dict
            Dictionary of data to write.
        cnt : int
            Correlation accumulation count.
        nspec : int
            Number of spectra to write to file before creating a new file.
        save_dir : str
            Directory to save data to.

        """
        self.buffer[f"{cnt}"] = data
        if cnt > self.save_cnt + nspec:
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{save_dir}/{date}.npz"
            np.savez(fname, **self.buffer)
            self.buffer = {}
            self.save_cnt = cnt

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
        nspec=NSPEC,
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
        nspec : int
            Number of spectra to write to file before creating a new file.

        """
        cnt = self.fpga.read_int("corr_acc_cnt")
        t = time.time()
        if write_files:
            self.save_cnt = cnt
            self.buffer = {}
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
                self.write_file(data, cnt, nspec, DATA_PATH)
            t = time.time()
