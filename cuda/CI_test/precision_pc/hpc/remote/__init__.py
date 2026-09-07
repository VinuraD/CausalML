"""Remote scripts, shipped as package data.

This package exists only so `array.sbatch` and `stage.sh` are installed alongside
`worker.py` and can be located with `Path(__file__).parent`. The scripts are executed
on the cluster by path, never imported.
"""
