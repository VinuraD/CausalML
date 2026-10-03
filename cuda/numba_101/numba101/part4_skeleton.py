"""Part 4 - The skeleton loop (PC-stable level 0) and the flip-rate harness.

Given to you:  level0.
Yours:         level0_numpy, flip_rate, and the alpha = 1.0 experiment driver.
"""

import numpy as np
from numba import njit

from .part1_fisher_z import fisher_z


@njit(cache=True)
def level0(corr, n, alpha):
    """Marginal-independence pass over every pair.

    Returns BOTH the adjacency matrix and the full p-value matrix. Returning all
    p-values is deliberate - see Exercise 4b. `fisher_z` is called from inside a
    compiled function, so Numba inlines it: no Python call overhead, which is the
    main reason to prefer @njit over hand-vectorising.
    """
    N = corr.shape[0]
    adj = np.ones((N, N), dtype=np.uint8)
    np.fill_diagonal(adj, 0)
    pvals = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            p = fisher_z(corr[i, j], n, 0)
            pvals[i, j] = p
            pvals[j, i] = p
            if p > alpha:  # independent -> remove edge
                adj[i, j] = 0
                adj[j, i] = 0
    return adj, pvals


def level0_numpy(corr, n, alpha):
    """EXERCISE 4a. Fully vectorised NumPy level 0, same two return values.

    Use numba101.part1_fisher_z.fisher_z_numpy on the whole matrix at once. Mask
    the diagonal (r = 1 makes the transform blow up). Then time this against
    level0 at N = 500, 1000, 2000 and explain the crossover - or its absence.
    """
    raise NotImplementedError("Exercise 4a: vectorised level 0")


def flip_rate(pvals_a, pvals_b, alpha: float) -> float:
    """EXERCISE 4c. Fraction of tests where two runs disagree on accept/reject.

    Compare only the upper triangle above the diagonal - every test appears twice
    in a symmetric p-value matrix, and the diagonal is not a test at all. Counting
    both triangles does not change the ratio, but it will bite you the moment you
    start reporting absolute counts of flipped edges.

    E0.1 acceptance criterion: this must return exactly 0.0 when the two inputs
    come from identical float64 runs. Nothing measured later means anything until
    that passes.
    """
    raise NotImplementedError("Exercise 4c: decision flip rate")


def paired_pvalues(corr, n, formats, round_matrix_fn, parcorr_fn=None):
    """EXERCISE 4b. The alpha = 1.0 device, from the SSDBM paper, repurposed.

    Run level 0 at alpha = 1.0 (so no edge is ever removed) once per precision
    format, and keep every p-value. Thresholds are applied afterwards, in
    flip_rate, at whatever alpha you like.

    Why it matters: if a flip at level 0 actually removed an edge, the set of
    tests performed at level 1 would differ between precisions, and the runs would
    no longer be paired. Fixing the test set up front is what makes the
    comparison exact.

    Returns {format_name: pvalue_matrix}.
    """
    raise NotImplementedError("Exercise 4b: alpha = 1.0 paired harness")
