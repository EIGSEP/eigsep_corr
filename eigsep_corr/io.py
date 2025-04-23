import warnings

warnings.warn(
    "The 'io' module is deprecated and will be removed in future versions. "
    "Please import from eigsep_observing.io instead.",
    DeprecationWarning,
    stacklevel=2,
)
from eigsep_observing.io import *
