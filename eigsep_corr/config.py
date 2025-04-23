from dataclasses import dataclass, field
from pathlib import Path
from .data import DATA_PATH


@dataclass
class CorrConfig:
    """
    Configuration for EigsepFpga and observing with the SNAP correlator.

    """

    snap_ip: str = "10.10.10.13"
    sample_rate: float = 500  # in MHz
    fpg_file: str = str(
        (
            Path(DATA_PATH) / "eigsep_fengine_1g_v2_3_2024-07-08_1858.fpg"
        ).resolve()
    )
    fpg_version: tuple[int, int] = (2, 3)  # major, minor
    adc_gain: float = 4
    fft_shift: int = 0x055
    corr_acc_len: int = 2**28  # increment corr_acc_cnt by ~1/second
    corr_scalar: int = 2**9  # 8 bits after binary point so 2**9 = 1
    corr_word: int = 4  # 4 bytes per word
    pam_atten: dict[int, tuple[int, int]] = field(
        default_factory=lambda: {0: (8, 8), 1: (8, 8), 2: (8, 8)}
    )
    pol_delay: dict[str, int] = field(
        default_factory=lambda: {"01": 0, "23": 0, "45": 0}
    )
    nchan: int = 1024
    redis_host: str = "localhost"
    redis_port: int = 6379
    save_dir: str = "media/eigsep/T7/data"


default_corr_config = CorrConfig()

# config for Dummy SNAP interface (not connected to SNAP)
dummy_corr_config = CorrConfig(
    snap_ip="",
    fpg_file="",
    fpg_version=(0, 0),
    save_dir="./test_data",
)
