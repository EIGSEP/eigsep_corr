# eigsep_corr

> **This package is archived at version 2.0.1 and is no longer actively
> developed.** The live EIGSEP correlator driver has moved to
> [`eigsep_observing`](https://github.com/EIGSEP/eigsep_observing), which
> absorbed `fpga.py`, `blocks.py`, `testing.py`, the config loader, the
> correlator math utilities, and the `.fpg` bitstream in-tree. New work
> should land there.
>
> This repository is retained — and remains installable via
> `pip install eigsep_corr` — because `eigsep_corr.io.read_file` is the
> only reader for legacy `.eig` binary data files produced before the
> HDF5 migration, and because `eigsep_corr/data/gain_cal/` holds the
> per-component calibration tree referenced by historical analysis
> notebooks. Bug fixes to those two code paths may still be accepted;
> everything else should go upstream to `eigsep_observing`.

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

Note that new SNAP-side development has moved to `eigsep_observing`; the
casperfpga pin in this repository is frozen at v0.6.0 and will not be
updated here. See
[`eigsep_observing/hardware-requirements.txt`](https://github.com/EIGSEP/eigsep_observing/blob/main/hardware-requirements.txt)
for the current pin.