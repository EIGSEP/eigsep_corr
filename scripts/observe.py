import argparse
import logging

from eigsep_corr.fpga import add_args
from eigsep_corr.config import CorrConfig

SNAP_IP = "10.10.10.13"  # C00091
# SNAP_IP = "10.10.10.18"  # C00069
GAIN = 4  # ADC gain
FFT_SHIFT = 0x00FF
PAM_ATTEN = {"0": (8, 8), "1": (8, 8), "2": (8, 8)}  # order is EAST, NORTH
FPG_VERSION = (2, 3)

USE_REF = False  # use synth to generate adc clock from 10 MHz
USE_NOISE = False  # use digital noise instead of ADC data
LOG_LEVEL = logging.INFO

parser = argparse.ArgumentParser(
    description="Eigsep Correlator",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
add_args(parser)
args = parser.parse_args()

# see eigsep_corr.config for default values
# parameters are: snap_ip, sample_rate, fpg_file, fpg_version, adc_gain,
# fft_shift, corr_acc_len, corr_scalar, corr_word, pam_atten, pol_delay,
# nchan, redis_host, redis_port, save_dir
cfg = CorrConfig(
    snap_ip=SNAP_IP,
    fpg_file=args.fpg_file,
    fpg_version=FPG_VERSION,
    adc_gain=GAIN,
    fft_shift=FFT_SHIFT,
    pam_atten=PAM_ATTEN,
    save_dir=args.save_dir,
    ntimes=args.ntimes,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

if USE_REF:
    ref = 10
else:
    ref = None

if args.dummy_mode:
    logger.warning("Running in DUMMY mode")
    from eigsep_corr.testing import DummyEigsepFpga

    fpga = DummyEigsepFpga(ref=ref, logger=logger)
else:
    from eigsep_corr.fpga import EigsepFpga

    if args.force_program:
        program = True
        force_program = True
    elif args.program:
        program = True
        force_program = False
    else:
        program = False
        force_program = False
    fpga = EigsepFpga(
        cfg=cfg,
        program=program,
        ref=ref,
        logger=logger,
        force_program=force_program,
    )


if args.initialize_adc:
    fpga.initialize_adc()

if args.initialize_fpga:
    fpga.initialize_fpga()

fpga.check_version()

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

logger.info("Observing ...")
try:
    fpga.observe(
        pairs=None,
        timeout=10,
        update_redis=args.update_redis,
        write_files=args.write_files,
    )
except KeyboardInterrupt:
    logger.info("Exiting.")
finally:
    fpga.end_observing()
