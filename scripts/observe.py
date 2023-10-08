import argparse
import logging

from eigsep_corr.fpga import EigsepFpga

# SNAP_IP = "10.10.10.13"
SNAP_IP = "10.10.10.236"
fpg_filename = "eigsep_fengine_1g_v2_2_2023-10-06_1806.fpg"
FPG_FILE = "/home/eigsep/eigsep/eigsep_corr/" + fpg_filename
FPG_VERSION = (2, 2)  # major, minor
SAMPLE_RATE = 500  # MHz
GAIN = 4  # ADC gain
CORR_ACC_LEN = 2**28
CORR_SCALAR = 2**9
POL0_DELAY = 0
FFT_SHIFT = 0x0055
USE_REF = False  # use reference input
USE_NOISE = False  # use digital noise instead of ADC data
PAM_ATTEN = {"0": (8, 8), "1": (8, 8), "2": (8, 8)}
N_FEMS = 0  # number of FEMs to initialize (0-3)
SAVE_DIR = "/media/eigsep/T7/data"
LOG_LEVEL = logging.DEBUG

parser = argparse.ArgumentParser(
    description="Eigsep Correlator",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dummy",
    dest="dummy_mode",
    action="store_true",
    default=False,
    help="Run with a dummy SNAP interface",
)
parser.add_argument(
    "-p",
    dest="program",
    action="store_true",
    default=False,
    help="program eigsep correlator",
)
parser.add_argument(
    "-a",
    dest="initialize_adc",
    action="store_true",
    default=False,
    help="initialize ADCs",
)
parser.add_argument(
    "-f",
    dest="initialize_fpga",
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

logger = logging.getLogger(__name__)
logging.basicConfig(filename="snap.log", level=LOG_LEVEL)

if USE_REF:
    ref = 10
else:
    ref = None

if args.dummy_mode:
    logging.warning("Running in DUMMY mode")
    from eigsep_corr.testing import DummyEigsepFpga as EigsepFpga

fpga = EigsepFpga(
    SNAP_IP, fpg_file=FPG_FILE, program=args.program, ref=ref, logger=logger
)


if args.initialize_adc:
    fpga.initialize_adc(sample_rate=SAMPLE_RATE, gain=GAIN)

if args.initialize_fpga:
    fpga.initialize_fpga(
        fft_shift=FFT_SHIFT,
        corr_acc_len=CORR_ACC_LEN,
        corr_scalar=CORR_SCALAR,
        pol0_delay=POL0_DELAY,
        pam_atten=PAM_ATTEN,
        n_fems=N_FEMS,
    )

# check version
assert fpga.version == FPG_VERSION

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
    fpga.synchronize(delay=0, update_redis=args.update_redis)

print("Observing ...")
try:
    fpga.observe(
        SAVE_DIR,
        pairs=None,
        timeout=10,
        update_redis=args.update_redis,
        write_files=args.write_files,
    )
except KeyboardInterrupt:
    print("Exiting.")
finally:
    fpga.end_observing()
