# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status: archived at 2.0.1

**This package is archived and is no longer actively developed.** The live
SNAP FPGA driver, register blocks, testing dummies, config loader, and
correlator math helpers have all moved to
[`eigsep_observing`](https://github.com/EIGSEP/eigsep_observing). New
work on the correlator belongs there.

This repository is retained because two of its components have no
in-tree equivalent in `eigsep_observing` and are still needed for
historical-data analysis:

1. **`eigsep_corr.io`** — the reader (`read_file`) for the legacy `.eig`
   binary format, used by analysis notebooks that predate the HDF5
   migration in `eigsep_observing`.
2. **`eigsep_corr/data/gain_cal/`** — per-component `.npz` calibration
   files (PAM, FEM, fiber, box_fem, power, SNAP) referenced by those
   same notebooks.

Bug fixes to those two code paths may still be accepted. Anything else
(fpga.py, blocks.py, testing.py, config loading, correlator utilities)
should go upstream to `eigsep_observing`.

Release automation has been removed from this repository: there is no
`release-please` workflow and no PyPI publish step. The frozen 2.0.1
release remains installable via `pip install eigsep_corr`.

## Project Overview

eigsep_corr is the low-level SNAP FPGA correlator driver for EIGSEP — a 6-antenna radio interferometer. It provides hardware abstraction for ADC, FFT/PFB, PAM, and correlator blocks, plus binary I/O for correlation data. High-level orchestration lives in the separate `eigsep_observing` repository.

## Development Commands

```bash
# Install with dev dependencies
pip install -e .[dev]

# Lint and format
.venv/bin/ruff check .
.venv/bin/ruff format --check .    # check only
.venv/bin/ruff format .            # apply formatting

# Run all tests
.venv/bin/pytest

# Run a single test file or test
.venv/bin/pytest eigsep_corr/tests/test_fpga.py
.venv/bin/pytest eigsep_corr/tests/test_fpga.py::test_name
```

## Code Style

- Line length: 79 characters
- Formatter/linter: ruff (E203 ignored globally; F401 ignored in `__init__.py`)
- Excludes: `notebooks/`, `snap_fengines/`, `build/`

## Architecture

The core abstraction is `EigsepFpga` (fpga.py), which composes low-level register block objects defined in blocks.py:

- **fpga.py** — `EigsepFpga`: main control class. Wraps a `casperfpga.CasperFpga` connection and exposes methods for initialization, synchronization, PAM attenuation, and reading auto/cross-correlation spectra. Uses Redis for state persistence (sync time). Runs integration collection on a background thread (`_read_integrations`).
- **blocks.py** — Register block classes (`Sync`, `NoiseGen`, `Input`, `Pfb`, `Pam`, `Fem`, `Delay`, `Eq`, `EqTvg`, `Eth`) each inheriting from `Block`. These map directly to named FPGA register groups and handle low-level reads/writes via `casperfpga`.
- **io.py** — Custom binary file format: JSON header + raw correlation data. Functions for pack/unpack of headers and data, file read/write, and byte-offset calculation for correlation pairs.
- **config.py** — Loads YAML config from `eigsep_corr/config/config.yaml`. Key parameters: SNAP IP, sample rate, nchan (1024), accumulation length, FFT shift, correlation pairs, antenna/PAM mapping.
- **utils.py** — Path helpers, frequency/time calculation utilities.
- **testing.py** — Dummy/mock implementations (`DummyEigsepFpga`, `DummyFpga`, etc.) using `fakeredis` for testing without FPGA hardware.

## Testing Notes

- Tests run without hardware using mock objects from `testing.py` and `fakeredis`.
- CI matrix: Python 3.10–3.12.
- pytest-timeout is set to 60s per test.
- `casperfpga` is an optional dependency — the package handles its absence gracefully.
