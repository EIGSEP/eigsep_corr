import logging
import IPython
from eigsep_corr.fpga import EigsepFpga

SNAP_IP = "10.10.10.236"
FPGFILE = (
    "/home/eigsep/eigsep/eigsep_corr/"
    "eigsep_fengine_1g_v1_0_2022-08-26_1007.fpg"
)
FPG_VERSION = 0x10000
SAMPLE_RATE = 500  # MHz
GAIN = 4  # ADC gain
CORR_ACC_LEN = 2**28
CORR_SCALAR = 2**9
FFT_SHIFT = 0xFFFF
USE_NOISE = False  # use digital noise instead of ADC data
LOG_LEVEL = logging.DEBUG
REUPLOAD_FPG = False
N_PAMS = 1  # number of PAMs to initialize (0-3)
N_FEMS = 1  # number of FEMs to initialize (0-3)

logging.getLogger().setLevel(LOG_LEVEL)
logger = logging.getLogger(__name__)

if REUPLOAD_FPG:
    fpga = EigsepFpga(SNAP_IP, fpg_file=FPGFILE, logger=logger)
else:
    fpga = EigsepFpga(SNAP_IP, logger=logger)

# check version
assert fpga.fpga.read_int("version_version") == FPG_VERSION

fpga.initialize(
    SAMPLE_RATE,
    adc_gain=GAIN,
    pfb_fft_shift=FFT_SHIFT,
    corr_acc_len=CORR_ACC_LEN,
    corr_scalar=CORR_SCALAR,
    n_pams=N_PAMS,
    n_fems=N_FEMS,
)

# set input
if USE_NOISE:
    fpga.logger.warning("Switching to noise input")
    fpga.noise.set_seed(stream=None, seed=0)
    fpga.inp.use_noise(stream=None)
    fpga.sync.arm_noise()
    for i in range(3):
        fpga.sync.sw_sync()
    fpga.logger.info("Synchronized noise.")
else:
    fpga.logger.info("Switching to ADC input")
    fpga.inp.use_adc(stream=None)

# synchronize
fpga.synchronize(delay=0)

IPython.embed()
