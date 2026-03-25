# eigsep_corr

Low-level SNAP FPGA correlator driver for EIGSEP.

## Installation

```bash
pip install eigsep_corr
```

### Hardware dependency

Controlling SNAP hardware requires
[casperfpga](https://github.com/EIGSEP/casperfpga), which is **not** included
in the PyPI package because it must be installed from source. The pinned version
is tracked in `hardware-requirements.txt`:

```bash
pip install -r hardware-requirements.txt
```

See that file for the current tag (currently **v0.6.0**).