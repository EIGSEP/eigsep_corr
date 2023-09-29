import numpy as np
from eigsep_corr.fpga import EigsepFpga

SNAP_IP = "10.10.10.236"
FPG_FILE = (
    "/home/eigsep/eigsep/eigsep_corr/"
    "eigsep_fengine_1g_v1_0_2022-08-26_1007.fpg"
)
SAMPLE_RATE = 500
GAIN = 4
CORR_ACC_LEN = 2**28
CORR_SCALAR = 2**9
FFT_SHIFT = 0xFFFF
REUPLOAD_FPG = False

if REUPLOAD_FPG:
    fpga = EigsepFpga(SNAP_IP, fpg_file=FPG_FILE)
else:
    fpga = EigsepFpga(SNAP_IP)

fpga.initialize(
    SAMPLE_RATE,
    adc_gain=GAIN,
    pfb_fft_shift=FFT_SHIFT,
    corr_acc_len=CORR_ACC_LEN,
    corr_scalar=CORR_SCALAR,
    n_pams=0,
    n_fems=0,
)

fpga.logger.warning("Switching to noise input")
fpga.noise.set_seed()  # all feeds get same seed
fpga.inp.use_noise()
fpga.sync.arm_noise()
for i in range(3):
    fpga.sync.sw_sync()
fpga.synchronize()

# XXX clear buffer (appears necessary). 5 seems to work, but why?
cnt = fpga.fpga.read_int("corr_acc_cnt")
while fpga.fpga.read_int("corr_acc_cnt") < cnt + 5:
    pass
auto_spec = fpga.read_auto()
cross_spec = fpga.read_cross()
# read a second time and see we get all the same
auto_spec2 = fpga.read_auto()
cross_spec2 = fpga.read_cross()
assert np.allclose(auto_spec, auto_spec2)
assert np.allclose(cross_spec, cross_spec2)
# all spectra should be the same since the noise is the same
assert np.allclose(auto_spec, auto_spec[0])
assert np.allclose(cross_spec, cross_spec[0])
# cross corr should have real part = autos and im part = 0
assert np.allclose(cross_spec[0, ::2], auto_spec[0])
assert np.allclose(cross_spec[0, 1::2], 0)

# use a different seed for each stream
for i in range(len(fpga.autos)):
    fpga.noise.set_seed(stream=i, seed=i)
fpga.inp.use_noise()
fpga.sync.arm_noise()
for i in range(3):
    fpga.sync.sw_sync()
fpga.synchronize()
cnt = fpga.fpga.read_int("corr_acc_cnt")
while fpga.fpga.read_int("corr_acc_cnt") < cnt + 5:
    pass
auto_spec = fpga.read_auto()
cross_spec = fpga.read_cross()
# some autos are hardwired to be the same (0 == 1, 2 == 3, 4 == 5)
assert np.allclose(auto_spec[0], auto_spec[1])
assert np.allclose(auto_spec[2], auto_spec[3])
assert np.allclose(auto_spec[4], auto_spec[5])
# the others are different
assert np.any(auto_spec[0] != auto_spec[2])
assert np.any(auto_spec[0] != auto_spec[4])
assert np.any(auto_spec[2] != auto_spec[4])
# certain cross corrs must be the same by the above hardwiring
assert np.allclose(cross_spec[0], cross_spec[1])  # 02 == 13
assert np.allclose(cross_spec[2], cross_spec[3])  # 24 == 35
assert np.allclose(cross_spec[4], cross_spec[5])  # 04 == 15
# the others are different
assert np.any(cross_spec[0] != cross_spec[2])
assert np.any(cross_spec[0] != cross_spec[4])
assert np.any(cross_spec[2] != cross_spec[4])
# there's no reason for all imag parts to be 0 anymore
for i in range(3):
    assert np.any(cross_spec[2 * i][1::2] != 0)
