import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numba101  # noqa: E402,F401  -- pins BLAS threads before numpy loads
