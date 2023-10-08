import time
import datetime
import logging
import redis
import numpy as np
from math import floor

from .fpga import EigsepFpga
from .fpga import SNAP_IP, FPG_FILE
from .fpga import NCHAN, CORR_ACC_LEN, SAMPLE_RATE
from .fpga import REDIS_HOST, REDIS_PORT


class DummyBlock(object):
    def __init__(self, fpga, attrs=[]):
        self.fpga = fpga
        for attr in attrs:
            self.fpga.regs[attr] = None
        self._attributes = {}
    def init(self, *args, **kwargs):
        pass
    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except(AttributeError):
            return self
    def __call__(self, *args, **kwargs):
        return None


class DummyFpga(DummyBlock):
    def __init__(self, regs, **kwargs):
        self.sync_time = None
        self.cnt_period = CORR_ACC_LEN / (SAMPLE_RATE * 1e6)
        self.regs = {r: None for r in regs}
        self.regs['version_version'] = 0x20002

    def write_int(self, reg, val):
        self.regs[reg] = val

    def read_int(self, reg):
        if reg == 'corr_acc_cnt':
            return int(floor((time.time() - self.sync_time) / self.cnt_period))
        else:
            return self.regs[reg]

    read_uint = read_int

    def read(self, reg, nbytes):
        return "\x12" * nbytes


class DummyEigsepFpga(EigsepFpga):
    def __init__(
        self,
        snap_ip=SNAP_IP,
        fpg_file=FPG_FILE,
        program=False,
        ref=None,
        transport=None,
        logger=None,
        sample_rate=SAMPLE_RATE,
        nchan=NCHAN,
        acc_len=CORR_ACC_LEN,
        **kwargs,
    ):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
        self.logger = logger

        self.fpg_file = fpg_file
        self.fpga = DummyFpga([], snap_ip=snap_ip, transport=transport)

        self.adc = DummyBlock(self.fpga)
        self.sync = DummyBlock(self.fpga)
        self.noise = DummyBlock(self.fpga)
        self.inp = DummyBlock(self.fpga)
        self.pfb = DummyBlock(self.fpga)
        self.blocks = [self.sync, self.noise, self.inp, self.pfb]

        self.autos = ["0", "1", "2", "3", "4", "5"]
        self.crosses = ["02", "13", "24", "35", "04", "15"]

        self.redis = redis.Redis(REDIS_HOST, port=REDIS_PORT)

        self.adc_initialized = False
        self.pams_initialized = False
        self.is_synchronized = False

        self.sample_rate = sample_rate
        self.nchan = nchan
        self.acc_len = acc_len

    def initialize_adc(self, *args, **kwargs):
        self.adc_initialized = True

    def initialize_pams(self, *args, **kwargs):
        self.pams = [DummyBlock(self.fpga), DummyBlock(self.fpga), DummyBlock(self.fpga)]
        self.pams_initialized = True

    def initialize_fems(self, *args, **kwargs):
        self.fems_initialized = True

    def synchronize(self, delay=0, update_redis=True):
        self.fpga.sync_time = self.sync_time = time.time()
        if update_redis:
            self.redis.set("SYNC_TIME", str(self.sync_time))
            self.redis.set(
                "SYNC_DATE",
                datetime.datetime.fromtimestamp(self.sync_time).isoformat(),
            )
        self.is_synchronized = True

