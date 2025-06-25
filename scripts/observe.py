import argparse
import logging

from eigsep_corr.fpga import EigsepFpga, FPG_FILE
from eigsep_corr.testing import DummyEigsepFpga

SNAP_IP = "10.10.10.13"  # C00091
# SNAP_IP = "10.10.10.18"  # C00069
SAMPLE_RATE = 500  # MHz
GAIN = 4  # ADC gain
CORR_ACC_LEN = 2**28
CORR_SCALAR = 2**9
POL_DELAY = {"01": 0, "23": 0, "45": 0}
FFT_SHIFT = 0x00FF
USE_REF = False  # use synth to generate adc clock from 10 MHz
USE_NOISE = False  # use digital noise instead of ADC data
PAM_ATTEN = {"0": (8, 8), "1": (8, 8), "2": (8, 8)}  # order is EAST, NORTH
SAVE_DIR = "/media/eigsep/T7/data"
LOG_LEVEL = logging.INFO

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
    "-P",
    dest="force_program",
    action="store_true",
    default=False,
    help="force program eigsep correlator even if fpg file is the same",
)
parser.add_argument(
    "--fpg",
    dest="fpg_file",
    default=FPG_FILE,
    help="FPG file for eigsep correlator",
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
parser.add_argument(
    "--ntimes",
    dest="ntimes",
    type=int,
    default=60,
    help="Number of integrations to write per file.",
)
parser.add_argument(
    "--save_dir",
    dest="save_dir",
    default=SAVE_DIR,
    help="Directory to save files.",
)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

force_program = args.force_program
program = args.program or force_program
fpga_kwargs = {
    "snap_ip": SNAP_IP,
    "fpg_file": args.fpg_file,
    "program": program,
    "use_ref": USE_REF,
    "logger": logger,
    "force_program": force_program,
}

if args.dummy_mode:
    logger.warning("Running in DUMMY mode")
    fpga = DummyEigsepFpga(**fpga_kwargs)
else:
    logger.info(f"Connecting to Eigsep correlator at {SNAP_IP}")
    fpga = EigsepFpga(**fpga_kwargs)

if args.initialize_adc:
    fpga.initialize_adc(sample_rate=SAMPLE_RATE, gain=GAIN)

if args.initialize_fpga:
    fpga.initialize_fpga(
        fft_shift=FFT_SHIFT,
        corr_acc_len=CORR_ACC_LEN,
        corr_scalar=CORR_SCALAR,
        pol_delay=POL_DELAY,
        pam_atten=PAM_ATTEN,
    )

fpga.check_version()

# set input
fpga.set_input(use_noise=USE_NOISE)

# synchronize
if args.sync:
    fpga.synchronize(delay=0, update_redis=args.update_redis)

logger.info("Observing ...")
try:
    fpga.observe(
        args.save_dir,
        pairs=None,
        timeout=10,
        update_redis=args.update_redis,
        write_files=args.write_files,
        ntimes=args.ntimes,
    )
except KeyboardInterrupt:
    logger.info("Exiting.")
finally:
    fpga.end_observing()
