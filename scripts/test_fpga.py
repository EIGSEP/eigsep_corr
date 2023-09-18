from eigsep_corr.fpga import EigsepFpga

SNAP_IP = "10.10.10.236"
DIR = "/home/eigsep/eigsep/eigsep_corr/"
FPG_FILE = DIR + "eigsep_fengine_1g_v1_0_2022-08-26_1007.fpg"
REUPLOAD = False
if REUPLOAD:
    fpga = EigsepFpga(SNAP_IP, fpg_file=FPG_FILE)
else:
    fpga = EigsepFpga(SNAP_IP)
fpga.initialize_fpga(corr_acc_len=2**28, corr_scalar=2**9)
print(fpga.time_read_corrs())
fpga.test_corr_noise()
