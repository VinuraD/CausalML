"""Precision requirements for constraint-based causal discovery -- experiment harness.

Importing this package pins BLAS threading. It must therefore be imported BEFORE numpy
in every script, driver and test:

    import precision_pc          # pins OMP/OPENBLAS/MKL/... to 1 thread
    import numpy as np           # only now

The reverse order silently invalidates E0.1 and everything downstream of it.
"""

from . import env as env

env.pin()

__all__ = ["env"]
__version__ = "0.1.0"
