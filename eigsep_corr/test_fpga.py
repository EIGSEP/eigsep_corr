from eigsep_corr.fpga import EigsepFpga

# fpg file to upload (none if no upload)
fname = "/home/eigsep/eigsep/eigsep_corr/eigsep_fengine_1g_v1_0_2022-08-26_1007.fpg"
#fname = None
eig_fpga = EigsepFpga("10.10.236", fname)
eig_fpga.fpga.write_int("corr_acc_len", 2**28)
eig_fpga.fpga.write_int("corr_scalar", 2**9)
eig_fpga.test_corr_noise()

