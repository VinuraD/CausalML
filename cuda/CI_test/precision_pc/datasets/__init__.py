"""Data generating processes.

Each module returns a correlation matrix plus the effective sample size, because the
correlation matrix is the only input the CI-test engine ever sees.
"""

from . import var as var  # noqa: F401

__all__ = ["var"]
