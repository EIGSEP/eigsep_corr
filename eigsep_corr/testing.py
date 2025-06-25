import time
import logging
import redis
from math import floor

from .fpga import EigsepFpga
from .fpga import SNAP_IP, FPG_FILE


class DummyBlock:
    def __init__(self, fpga, attrs=[]):
        self.fpga = fpga
        for attr in attrs:
            self.fpga.regs[attr] = None
        self._attributes = {}

    def init(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        pass

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return self

    def __call__(self, *args, **kwargs):
        return None


class DummyFpga(DummyBlock):
    def __init__(self, regs, **kwargs):
        self.sync_time = None
        self.cnt_period = kwargs.pop("cnt_period", 2**28 / (500 * 1e6))
        self.regs = {r: None for r in regs}
        self.regs["version_version"] = 0x20003

    def upload_to_ram_and_program(self, fpg_file, force=False):
        pass

    def write_int(self, reg, val):
        self.regs[reg] = val

    def read_int(self, reg):
        if reg == "corr_acc_cnt":
            acc_cnt = (time.time() - self.sync_time) / self.cnt_period
            acc_cnt = int(floor(acc_cnt))
            return acc_cnt
        else:
            return self.regs[reg]

    read_uint = read_int

    def read(self, reg, nbytes):
        return b"\x12" * nbytes


class DummyAdcAdc:

    def selectInput(self, inp):
        pass


class DummyAdc(DummyBlock):

    def __init__(self, fpga, num_chans=2, resolution=8, ref=None):
        super().__init__(fpga)

    def init(self, sample_rate=500):
        self.adc = DummyAdcAdc()

    def alignLineClock(self):
        return []

    def alignFrameClock(self):
        return []

    def rampTest(self):
        return []

    def selectAdc(self):
        pass

    def set_gain(self, gain):
        pass


class DummyPfb(DummyBlock):

    def set_fft_shift(self, fft_shift):
        pass


class DummyPam(DummyBlock):

    def set_attenuation(self, att_e, att_n):
        pass


class DummySync(DummyBlock):

    def set_delay(self, delay):
        pass

    def arm_sync(self):
        pass

    def arm_noise(self):
        pass

    def sw_sync(self):
        self.fpga.sync_time = time.time()


class DummyNoise(DummyBlock):

    def set_seed(self, steam=None, seed=0):
        pass


class DummyInput(DummyBlock):

    def use_noise(self, stream=None):
        pass

    def use_adc(self, stream=None):
        pass


class DummyEigsepFpga(EigsepFpga):
    def __init__(
        self,
        snap_ip=SNAP_IP,
        fpg_file=FPG_FILE,
        program=False,
        use_ref=False,
        transport=None,
        logger=None,
        force_program=False,
    ):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
        self.logger = logger

        self.fpg_file = fpg_file
        cnt_period = self.defaults["corr_acc_len"] / (
            self.defaults["sample_rate"] * 1e6
        )
        self.fpga = DummyFpga(
            [], snap_ip=snap_ip, transport=transport, cnt_period=cnt_period
        )
        if program:
            self.fpga.upload_to_ram_and_program(fpg_file, force=force_program)

        if use_ref:
            ref = 10
        else:
            ref = None

        self.adc = DummyAdc(self.fpga, ref=ref)
        self.sync = DummySync(self.fpga)
        self.noise = DummyNoise(self.fpga)
        self.inp = DummyInput(self.fpga)
        self.pfb = DummyPfb(self.fpga)
        self.blocks = [self.sync, self.noise, self.inp, self.pfb]

        self.autos = ["0", "1", "2", "3", "4", "5"]
        self.crosses = ["02", "13", "24", "35", "04", "15"]

        self.redis = redis.Redis(self.redis_host, port=self.redis_port)

        self.adc_initialized = False
        self.pams_initialized = False
        self.is_synchronized = False

        self.sample_rate = self.defaults["sample_rate"]
        self.nchan = self.defaults["nchan"]
        self.acc_len = self.defaults["corr_acc_len"]

        self.file = None
        self.queue = None
        self.event = None

    def initialize_pams(self, attenuation=(8, 8)):

        self.pams = []
        for p, (att_e, att_n) in attenuation.items():
            pam = DummyPam(self.fpga, f"i2c_ant{p}")
            pam.initialize()
            self.logger.info(
                f"Setting pam{p} attenuation to ({att_e}, {att_n})"
            )
            pam.set_attenuation(att_e, att_n)
            self.pams.append(pam)
        self.blocks.extend(self.pams)
        self.pams_initialized = True
