import logging
import struct
import time
import numpy as np
import casperfpga
from casperfpga.transport_tapcp import TapcpTransport
from eigsep_corr.blocks import Input, Fem, NoiseGen, Pam, Pfb, Sync, Synth


class EigsepFpga:
    def __init__(
        self, snap_ip, fpg_file=None, transport=TapcpTransport, logger=None
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
        self.synth = Synth(self.fpga, "synth")
        self.adc = casperfpga.snapadc.SnapAdc(
            self.fpga, num_chans=2, resolution=8, ref=10
        )
        self.sync = Sync(self.fpga, "sync")
        self.noise = NoiseGen(self.fpga, "noise", nstreams=6)
        self.inp = Input(self.fpga, "input", nstreams=12)
        # self.delay = Delay(self.fpga, "delay", nstreams=6)
        self.pfb = Pfb(self.fpga, "pfb")
        # self.eq = Eq(self.fpga, "eq_core", nstreams=6, ncoeffs=2**10)
        # self.reorder = ChanReorder(self.fpga, "chan_reorder", nchans=2**10)
        # self.packetizer = Packetizer(self.fpga, "packetizer", n_time_demux=2)
        # self.eth = Eth(self.fpga, "eth")
        # self.corr = Corr(self.fpga, "corr_0")
        # self.phase_switch = PhaseSwitch(self.fpga, "phase_switch")

        self.blocks = [
            self.synth,
            self.sync,
            self.noise,
            self.inp,
            # self.delay,
            self.pfb,
            # self.eq,
            # self.reorder,
            # self.packetizer,
            # self.eth,
            # self.corr,
            # self.phase_switch,
        ]

        self.autos = [0, 1, 2, 3, 4, 5]
        self.crosses = ["02", "13", "24", "35", "04", "15"]

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

    def initialize_fpga(self, corr_acc_len=2**28, corr_scalar=2**9):
        """
        Parameters that must be set for the correlator

        Parameters
        ----------
        corr_acc_len : int (power of 2)
            The accumulation length. Default value ensures that the
            corr_acc_cnt goes up by 1 per ~1 second
        corr_scalar : int (power of 2)
            Scalar that is multiplied to each correlation. Default value is
            2**9 since the values have 8 bits after the binary point,
            hence 2**9 = 1

        """
        self.fpga.write_int("corr_acc_len", corr_acc_len)
        self.fpga.write_int("corr_scalar", corr_scalar)

    def initialize_pams(self, N=3):
        """
        Initialize the PAMs.

        Parameters
        ----------
        N : int
           Number of PAMs to initialize. Default is 3.

        """
        self.pams = [Pam(self.fpga, f"i2c_ant{i}") for i in range(N)]
        for pam in self.pams:
            pam.initialize()
            pam.set_attenuation(8, 8)  # XXX

    def initialize_fems(self, N=3):
        """
        Initialize the FEMs.

        Parameters
        ----------
        N : int
           Number of FEMs to initialize. Default is 3.

        """
        self.fems = [Fem(self.fpga, f"i2c_ant{i}") for i in range(N)]
        for fem in self.fems:
            fem.initialize()

    def initialize(
        self,
        adc_sample_rate,
        adc_gain=4,
        pfb_fft_shift=0xFFFF,
        corr_acc_len=2**28,
        corr_scalar=2**9,
        n_pams=3,
        n_fems=3,
    ):
        self.initialize_fpga(corr_acc_len, corr_scalar)
        self.initialize_adc(adc_sample_rate, adc_gain)
        for blk in self.blocks:
            blk.initialize()
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

    def read_auto(self, i=None):
        """
        Read the i'th (counting from 0) autocorrelation spectrum.

        Parameters
        ----------
        i : int
            Which autocorrelation to read. Default is None, which reads all
            autocorrelations.
        """
        if i is None:
            return np.array([self.read_auto(i=a) for a in self.autos])
        name = "corr_auto_%d_dout" % i
        spec = np.array(struct.unpack(">2048l", self.fpga.read(name, 8192)))
        return spec

    def read_cross(self, ij=None):
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
        name = "corr_cross_%s_dout" % ij
        spec = np.array(struct.unpack(">4096l", self.fpga.read(name, 16384)))
        return spec

    def time_read_corrs(self):
        """
        Measure how long it takes to read all corrs
        """
        cnt = self.fpga.read_int("corr_acc_cnt")
        while self.fpga.read_int("corr_acc_cnt") == cnt:
            pass
        start = time.time()
        _ = self.read_auto()
        _ = self.read_cross()
        dt = time.time() - start
        assert self.fpga.read_int("corr_acc_cnt") == cnt + 1
        return dt
