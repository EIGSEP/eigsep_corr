import casperfpga
from casperfpga.transport_tapcp import TapcpTransport

SNAP_IP = '10.10.10.236'
FPGFILE = 'eigsep_fengine_1g_v1_0_2002-08-26_1007.fpg'
SAMPLE_RATE = 500 # MHz

fpga = casperfpga.CasperFpga(SNAP_IP, transport=TapcpTransport)
fpga.upload_to_ram_and_program(FPGFILE)
synth = casperfpga.synth.LMX2581(fpga, 'synth')
adc = casperfpga.snapadc.SnapAdc(fpga, num_chans=2, resolution=8, ref=10)
#synth.initialize() # XXX is this necessary?
adc.init(sample_rate=SAMPLE_RATE)

import IPython; IPython.embed()
