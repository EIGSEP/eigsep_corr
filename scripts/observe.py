import argparse
import logging

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)

from eigsep_corr.config import load_config
from eigsep_corr.fpga import EigsepFpga
from eigsep_corr.testing import DummyEigsepFpga

logger = logging.getLogger(__name__)
CONFIG_FILE = "config.yaml"  # relative to config directory
CFG = load_config(CONFIG_FILE)

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
    "--save_dir",
    dest="save_dir",
    default=CFG["save_dir"],
    help="Directory to save files.",
)
args = parser.parse_args()

if args.force_program:
    program = "force"
else:
    program = args.program

if args.dummy_mode:
    logger.warning("Running in DUMMY mode")
    fpga = DummyEigsepFpga(cfg=CFG, program=program)
else:
    snap_ip = CFG["snap_ip"]
    logger.info(f"Connecting to Eigsep correlator at {snap_ip}")
    fpga = EigsepFpga(cfg=CFG, program=program)

if args.initialize_adc:
    logger.debug("Initializing ADCs")
    fpga.initialize_adc()

if args.initialize_fpga:
    logger.debug("Initializing FPGA")
    fpga.initialize_fpga()

# validate configuration
fpga.validate_config()

# set input
fpga.set_input()

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
    )
except KeyboardInterrupt:
    logger.info("Exiting.")
finally:
    fpga.end_observing()
