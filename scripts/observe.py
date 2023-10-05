import argparse
import logging
import IPython
from eigsep_corr.fpga import EigsepFpga

SNAP_IP = "10.10.10.13"
#SNAP_IP = "10.10.10.236"
fpg_filename = "eigsep_fengine_1g_v2_1_2023-10-05_1148.fpg"
FPG_FILE = "/home/eigsep/eigsep/eigsep_corr/" + fpg_filename
FPG_VERSION = 0x20001
SAMPLE_RATE = 500  # MHz
GAIN = 4  # ADC gain
CORR_ACC_LEN = 2**28
CORR_SCALAR = 2**9
INPUT_DELAY = 0
FFT_SHIFT = 0x0055
USE_NOISE = False  # use digital noise instead of ADC data
LOG_LEVEL = logging.DEBUG
N_PAMS = 0  # number of PAMs to initialize (0-3)
N_FEMS = 0  # number of FEMs to initialize (0-3)

parser = argparse.ArgumentParser(
    description="Eigsep Correlator",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-p",
    dest="program",
    action="store_true",
    default=False,
    help="program eigsep correlator",
)
parser.add_argument(
    "-i",
    dest="initialize",
    action="store_true",
    default=False,
    help="initialize eigsep correlator",
)
parser.add_argument(
    "-s",
    dest="sync",
    action="store_true",
    default=False,
    help="sync eigsep correlator",
)
parser.add_argument(
    "-r",
    dest="update_redis",
    action="store_true",
    default=False,
    help="update redis",
)
parser.add_argument(
    "-w",
    dest="write_files",
    action="store_true",
    default=False,
    help="write data to file",
)
args = parser.parse_args()

logging.getLogger().setLevel(LOG_LEVEL)
logger = logging.getLogger(__name__)

fpga = EigsepFpga(
    SNAP_IP, fpg_file=FPG_FILE, program=args.program, logger=logger
)

# check version
#print(fpga.fpga.read_int("version_version"))
#assert fpga.fpga.read_int("version_version") == FPG_VERSION

if args.initialize:
    fpga.initialize(
        SAMPLE_RATE,
        adc_gain=GAIN,
        pfb_fft_shift=FFT_SHIFT,
        corr_acc_len=CORR_ACC_LEN,
        corr_scalar=CORR_SCALAR,
        input_delay=INPUT_DELAY,
        n_pams=N_PAMS,
        n_fems=N_FEMS,
    )

# set input
fpga.noise.set_seed(stream=None, seed=0)
if USE_NOISE:
    fpga.logger.warning("Switching to noise input")
    fpga.inp.use_noise(stream=None)
    fpga.sync.arm_noise()
    for i in range(3):
        fpga.sync.sw_sync()
    fpga.logger.info("Synchronized noise.")
else:
    fpga.logger.info("Switching to ADC input")
    fpga.inp.use_adc(stream=None)

# synchronize
if args.sync:
    fpga.synchronize(delay=0)

print("observing ...")
try:
    fpga.observe(
        update_redis=args.update_redis,
        write_files=args.write_files,
        timeout=10,
    )
except KeyboardInterrupt:
    pass
IPython.embed()
