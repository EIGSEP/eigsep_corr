import casperfpga
from casperfpga.transport_tapcp import TapcpTransport
from eigsep_corr.blocks import Sync, Input, NoiseGen, Pfb, Pam
import logging
import time

SNAP_IP = '10.10.10.236'
FPGFILE = 'eigsep_fengine_1g_v1_0_2002-08-26_1007.fpg'
FPG_VERSION = 0x10000
SAMPLE_RATE = 500 # MHz
CORR_ACC_LEN = 2**28
CORR_SCALAR = 2**9
FFT_SHIFT = 0xffff
USE_NOISE = False # use digital noise instead of ADC data
LOG_LEVEL = logging.DEBUG
REUPLOAD_FPG = False

logging.getLogger().setLevel(LOG_LEVEL)
logger = logging.getLogger(__name__)

fpga = casperfpga.CasperFpga(SNAP_IP, transport=TapcpTransport)
if REUPLOAD_FPG:
    fpga.upload_to_ram_and_program(FPGFILE)

# check version
assert fpga.read_int('version_version') == FPG_VERSION

# set up block interfaces
synth = casperfpga.synth.LMX2581(fpga, 'synth')
adc = casperfpga.snapadc.SnapAdc(fpga, num_chans=2, resolution=8, ref=10)
sync = Sync(fpga, 'sync')
inp = Input(fpga, 'input', nstreams=12)
noise = NoiseGen(fpga, 'noise', nstreams=6)
pfb = Pfb(fpga, 'pfb')

# initialize adc
#synth.initialize() # XXX is this necessary?
adc.init(sample_rate=SAMPLE_RATE)

# Align clock and data lanes of ADC.
fails = adc.alignLineClock()
if len(fails) > 0:
    logger.warning("alignLineClock failed on: " + str(fails))
fails = adc.alignFrameClock()
if len(fails) > 0:
    logger.warning("alignFrameClock failed on: " + str(fails))
fails = adc.rampTest()
if len(fails) > 0:
    logger.warning("rampTest failed on: " + str(fails))

# Otherwise, finish up here.
adc.selectADC()
adc.adc.selectInput([1, 1, 3, 3]) 
adc.set_gain(4)

# set register values
for blk in [sync, inp, noise, pfb]:
    blk.initialize()

pfb.set_fft_shift(FFT_SHIFT)
fpga.write_int('corr_acc_len', CORR_ACC_LEN)
fpga.write_int('corr_scalar', CORR_SCALAR)

# set input
if USE_NOISE:
    logger.warn("Switching to noise input")
    noise.set_seed(stream=None, seed=0)
    inp.use_noise(stream=None)
    sync.arm_noise()
    for i in range(3):
        sync.sw_sync()
    logger.info('Synchronized noise.')
else:
    logger.info('Switching to ADC input')
    inp.use_adc(stream=None)

# initialize pams
pams = [Pam(fpga, 'i2c_ant%d' % i) for i in range(3)]
for pam in pams:
    pam.initialize()
    pam.set_attenuation(8, 8)

# synchronize
sync.set_delay(0)
sync.arm_sync()
for i in range(3):
    sync.sw_sync()
    sync_time = int(time.time())
    logger.info(f'Synchronized at {sync_time}')

import IPython; IPython.embed()
