import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from eigsep_corr.config import load_config
from eigsep_corr.fpga import add_args, EigsepFpga
from eigsep_corr.testing import DummyEigsepFpga


parser = argparse.ArgumentParser(
    description="Eigsep Correlator",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
add_args(parser)
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
    help="Write data to file.",
)
parser.add_argument(
    "--save_dir",
    dest="save_dir",
    default=None,
    help="Directory to save files. Overrides the default in the config file.",
)
parser.add_argument(
    "--dummy",
    dest="dummy_mode",
    action="store_true",
    default=False,
    help="Run with a dummy SNAP interface",
)
args = parser.parse_args()
cfg = load_config(args.config_file)

if args.force_program:
    program = "force"
else:
    program = args.program

if args.dummy_mode:
    logger.warning("Running in DUMMY mode")
    fpga = DummyEigsepFpga(cfg=cfg, program=program)
else:
    snap_ip = cfg["snap_ip"]
    logger.info(f"Connecting to Eigsep correlator at {snap_ip}")
    fpga = EigsepFpga(cfg=cfg, program=program)

# initialize SNAP
fpga.initialize(
    initialize_adc=args.initialize_adc,
    initialize_fpga=args.initialize_fpga,
    sync=args.sync,
    update_redis=args.update_redis,
)

# validate configuration
fpga.validate_config()

if args.save_dir is None:
    save_dir = cfg["save_dir"]
else:
    logger.info(f"Using save directory: {args.save_dir}")
    save_dir = args.save_dir

logger.info("Observing.")
try:
    fpga.observe(
        save_dir,
        pairs=None,
        timeout=10,
        update_redis=args.update_redis,
        write_files=args.write_files,
    )
except KeyboardInterrupt:
    logger.info("Exiting.")
finally:
    fpga.end_observing()
