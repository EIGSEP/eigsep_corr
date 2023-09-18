import casperfpga
from casperfpga.transport_tapcp import TapcpTransport
from eigsep_corr.blocks import Input, NoiseGen, Pam, Pfb, Sync
import logging
import numpy as np
import struct
import time

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
        
        self.synth = casperfpga.synth.LMX2581(self.fpga, "synth")
        self.adc = casperfpga.snapadc.SnapAdc(
            self.fpga, num_chans=2, resolution=8, ref=10
        )
        self.sync = Sync(self.fpga, "sync")
        self.inp = Input(self.fpga, "input", nstreams=12)
        self.noise = NoiseGen(self.fpga, "noise", nstreams=6)
        self.pfb = Pfb(self.fpga, "pfb")

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

    def initialize_pams(self):
        self.pams = [Pam(self.fpga, f"i2c_ant{i}") for i in range(3)]
        for pam in self.pams:
            pam.initialize()
            pam.set_attenuation(8, 8) # XXX

    def initialize_blocks(
        self,
        adc_sample_rate,
        adc_gain=4,
        pfb_fft_shift=0xffff,
        corr_acc_len=2**28,
        corr_scalar=2**9,
        pams=True,
    ):
        
        self.initialize_adc(adc_sample_rate, adc_gain)
        self.sync.initialize()
        self.inp.initialize()
        self.noise.initialize()
        self.pfb.initialize()
        self.pfb.set_fft_shift(pfb_fft_shift)
        self.initialize_fpga(corr_acc_len, corr_scalar)
        if pams:
            self.initialize_pams()
    

    def synchronize(self, delay=0):
        self.sync.set_delay(delay)
        self.sync.arm_sync()
        for i in range(3):
            self.sync.sw_sync()
            sync_time = int(time.time())
            self.logger.info(f"Synchronized at {sync_time}.")

    def read_auto(self, N):
        """
        Read the Nth (counting from 0) autocorrelation spectrum
        """
        name = "corr_auto_%d_dout"%N
        spec = np.array(struct.unpack(">2048l", self.fpga.read(name, 8192)))
        return spec

    def read_cross(self, NM):
        """
        Read the NM cross correlation spectrum

        Parameters
        ----------
        NM : str
            Which correlation to read, e.g. "02". Assuming N<M.
        """
        name = "corr_cross_%s_dout"%NM
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
        for auto in self.autos:
            self.read_auto(auto)
        for cross in self.crosses:
            self.read_cross(cross)
        dt = time.time() - start
        assert self.fpga.read_int("corr_acc_cnt") == cnt + 1
        return dt


    def test_corr_noise(self):
        self.initialize_blocks(500, pams=False)
        self.noise.set_seed()  # all feeds get same seed
        self.inp.use_noise()
        self.sync.arm_noise()
        for i in range(3):
            self.sync.sw_sync()
        self.synchronize()

        #XXX clear buffer (appears necessary). 5 seems to work, but why?
        cnt = self.fpga.read_int("corr_acc_cnt")
        while self.fpga.read_int("corr_acc_cnt") < cnt + 5:
            pass
        auto_spec = [self.read_auto(N) for N in self.autos]
        cross_spec = [self.read_cross(NM) for NM in self.crosses]
        # read a second time and see we get all the same
        auto_spec2 = [self.read_auto(N) for N in self.autos]
        cross_spec2 = [self.read_cross(NM) for NM in self.crosses]
        assert np.allclose(auto_spec, auto_spec2)
        assert np.allclose(cross_spec, cross_spec2)
        # all spectra should be the same since the noise is the same
        assert np.all(auto_spec == auto_spec[0])
        assert np.all(cross_spec == cross_spec[0])
        # cross corr should have real part = autos and im part = 0
        assert np.all(cross_spec[0][::2] == auto_spec[0])
        assert np.all(cross_spec[0][1::2] == 0)

        # use a different seed for each stream
        for i in range(len(self.autos)):
            self.noise.set_seed(stream=i, seed=i)
        self.inp.use_noise()
        self.sync.arm_noise()
        for i in range(3):
            self.sync.sw_sync()
        self.synchronize()
        cnt = self.fpga.read_int("corr_acc_cnt")
        while self.fpga.read_int("corr_acc_cnt") < cnt + 5:
            pass
        auto_spec = [self.read_auto(N) for N in self.autos]
        cross_spec = [self.read_cross(NM) for NM in self.crosses]
        # some autos are hardwired to be the same (0 == 1, 2 == 3, 4 == 5)
        assert np.all(auto_spec[0] == auto_spec[1])
        assert np.all(auto_spec[2] == auto_spec[3])
        assert np.all(auto_spec[4] == auto_spec[5])
        # the others are different
        assert np.any(auto_spec[0] != auto_spec[2])
        assert np.any(auto_spec[0] != auto_spec[4])
        assert np.any(auto_spec[2] != auto_spec[4])
        # certain cross corrs must be the same by the above hardwiring
        assert np.all(cross_spec[0] == cross_spec[1])  # 02 == 13
        assert np.all(cross_spec[2] == cross_spec[3])  # 24 == 35
        assert np.all(cross_spec[4] == cross_spec[5])  # 04 == 15
        # the others are different
        assert np.any(cross_spec[0] != cross_spec[2])
        assert np.any(cross_spec[0] != cross_spec[4])
        assert np.any(cross_spec[2] != cross_spec[4])
        # there's no reason for all imag parts to be 0 anymore
        for i in range(3):
            assert np.any(cross_spec[2*i][1::2] != 0)
