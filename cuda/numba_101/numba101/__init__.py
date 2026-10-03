"""Lesson 1 scaffold: Numba, by building the CI test.

Importing this package pins BLAS to a single thread (see numba101.env).
Always `import numba101` (or `from numba101 import env`) BEFORE `import numpy`.
"""

from . import env as env  # noqa: F401

env.pin()

__all__ = ["env"]
