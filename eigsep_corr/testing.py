import time
import logging
from math import floor

import fakeredis

from .fpga import EigsepFpga, default_config

logger = logging.getLogger(__name__)


class DummyBlock:
    def __init__(self, fpga):
        self.fpga = fpga

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
    def __init__(self, **kwargs):
        self.sync_time = None
        self.cnt_period = kwargs.pop("cnt_period", 2**28 / (500 * 1e6))
        self.regs = {}
        self.regs["version_version"] = 0x20003
        self.regs["corr_acc_len"] = kwargs.get("corr_acc_len", 67108864)
        self.regs["corr_scalar"] = kwargs.get("corr_scalar", 512)
        self.regs["fft_shift"] = kwargs.get("fft_shift", 0x0FF)
        self.regs["pfb_pol01_delay"] = 0
        self.regs["pfb_pol23_delay"] = 0
        self.regs["pfb_pol45_delay"] = 0

    def upload_to_ram_and_program(self, fpg_file, force=False):
        pass

    def write_int(self, reg, val):
        logger.debug(f"Writing {val} to {reg}")
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fft_shift = None

    def set_fft_shift(self, fft_shift):
        self.fft_shift = fft_shift

    def get_fft_shift(self):
        return self.fft_shift


class DummyPam(DummyBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attenuation = (0, 0)

    def set_attenuation(self, att_e, att_n):
        self.attenuation = (att_e, att_n)

    def get_attenuation(self):
        return self.attenuation


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

    def set_seed(self, stream=None, seed=0):
        pass


class DummyInput(DummyBlock):

    def use_noise(self, stream=None):
        pass

    def use_adc(self, stream=None):
        pass


class DummyEigsepFpga(EigsepFpga):
    def __init__(self, cfg=default_config, program=False):
        self.logger = logger
        self.cfg = cfg

        self.fpg_file = self.cfg["fpg_file"]
        corr_acc_len = self.cfg["corr_acc_len"]
        sample_rate = self.cfg["sample_rate"]
        cnt_period = corr_acc_len / (sample_rate * 1e6)
        self.fpga = DummyFpga(
            snap_ip=self.cfg["snap_ip"],
            transport=None,
            cnt_period=cnt_period,
            corr_acc_len=corr_acc_len,
            corr_scalar=self.cfg["corr_scalar"],
            fft_shift=self.cfg["fft_shift"],
        )
        if program:
            force = program == "force"
            self.fpga.upload_to_ram_and_program(self.fpg_file, force=force)

        if cfg["use_ref"]:
            ref = 10
        else:
            ref = None

        self.logger.debug("Adding dummy blocks to FPGA")
        self.adc = DummyAdc(self.fpga, ref=ref)
        self.sync = DummySync(self.fpga)
        self.noise = DummyNoise(self.fpga)
        self.inp = DummyInput(self.fpga)
        self.pfb = DummyPfb(self.fpga)
        self.blocks = [self.sync, self.noise, self.inp, self.pfb]

        self.autos = ["0", "1", "2", "3", "4", "5"]
        self.crosses = ["02", "13", "24", "35", "04", "15"]

        self.logger.debug("Initializing dummy Redis")
        self.redis = fakeredis.FakeRedis()

        self.adc_initialized = False
        self.pams_initialized = False
        self.is_synchronized = False

    def initialize_pams(self):
        attenuation = self.cfg["pam_atten"]
        self.pams = []
        for p, (att_e, att_n) in attenuation.items():
            pam = DummyPam(self.fpga)
            pam.initialize()
            self.logger.info(
                f"Setting pam{p} attenuation to ({att_e}, {att_n})"
            )
            pam.set_attenuation(att_e, att_n)
            self.pams.append(pam)
        self.blocks.extend(self.pams)
        self.pams_initialized = True
