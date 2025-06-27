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
