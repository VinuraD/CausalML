"""Flip accounting, and the amplification metric.

Two things in here are corrections to the proposal rather than implementations of it, and
both are deliberate.

**1. Undefined is not "the same decision".**
A NaN statistic compared against a threshold evaluates to False -- "not rejected" -- which
is the same decision as a well-behaved test that was also not rejected. Folding those
together biases every flip rate DOWNWARD, systematically, in exactly the low-precision
formats the paper is about, and would manufacture a reassuring negative result. So
`flip_counts` returns `n_undefined` as its own bucket and REFUSES to run unless the caller
states a policy for it.

**2. `SHD / flip_rate` is dimensionally incoherent.**
SHD is a count of edges; flip rate is dimensionless. Their ratio has units of edges, and
it moves by orders of magnitude with N for no scientific reason (the denominator falls as
~N^-2 while SHD grows as ~N). It is not a factor and it is not comparable across settings.
The quantity contribution C1 actually claims is *graph edges changed per flipped
decision*, which is `edges_changed / n_flipped_tests` -- dimensionless, interpretable, and
roughly scale-stable. Both are reported; only the second is called amplification.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import stats

#: What to do with tests whose decision is undefined (NaN statistic).
UNDEFINED_POLICIES = ("error", "exclude", "count_as_flip")


class PairingError(ValueError):
    """Two results were compared that do not describe the same tests."""


@dataclass(frozen=True)
class FlipCounts:
    alpha: float
    n_tests: int
    n_compared: int
    n_flip: int
    n_same: int
    n_undefined: int
    n_dep_to_indep: int  # reference rejected, variant did not (edge LOST)
    n_indep_to_dep: int  # reference did not reject, variant did (edge GAINED)
    n_reject_ref: int
    n_reject_var: int

    @property
    def flip_rate(self) -> Optional[float]:
        """Fraction of comparable tests that changed decision. None if nothing comparable."""
        if self.n_compared == 0:
            return None
        return self.n_flip / self.n_compared

    def as_dict(self) -> dict:
        d = asdict(self)
        d["flip_rate"] = self.flip_rate
        return d


def flip_counts(
    ref,
    var,
    alpha: float,
    *,
    undefined_policy: str = "error",
    check_pairing: bool = True,
) -> FlipCounts:
    """Compare two CellResults at one alpha.

    `ref` and `var` must have been computed against the same TestSet. That is checked by
    digest, not by shape: two different test sets can easily have the same length, and
    comparing test #7134 of one against test #7134 of the other manufactures flips out of
    nothing while leaving no trace in the output.
    """
    if undefined_policy not in UNDEFINED_POLICIES:
        raise ValueError(
            "undefined_policy must be one of %r, got %r" % (UNDEFINED_POLICIES, undefined_policy)
        )
    if check_pairing and ref.testset_digest != var.testset_digest:
        raise PairingError(
            "refusing to compare cells computed against different test sets: "
            + ref.testset_digest[:12]
            + " vs "
            + var.testset_digest[:12]
            + ". Flip rates across mismatched test sets are meaningless."
        )
    a = np.asarray(ref.stat, dtype=np.float64)
    b = np.asarray(var.stat, dtype=np.float64)
    if a.shape != b.shape:
        raise PairingError("stat arrays differ in shape: %r vs %r" % (a.shape, b.shape))

    c = stats.crit(alpha)
    undef = np.isnan(a) | np.isnan(b)
    n_undef = int(undef.sum())

    if n_undef and undefined_policy == "error":
        raise ValueError(
            "%d of %d tests have an undefined decision (NaN statistic) at alpha=%g. "
            "These are NOT 'not rejected' -- treating them as such biases the flip rate "
            "downward. Pass undefined_policy='exclude' or 'count_as_flip' and report "
            "n_undefined alongside the rate." % (n_undef, a.size, alpha)
        )

    rej_a = a > c
    rej_b = b > c
    differs = rej_a != rej_b

    if undefined_policy == "count_as_flip":
        comparable = np.ones_like(undef, dtype=bool)
        differs = differs | undef
    else:  # exclude (or no undefined at all)
        comparable = ~undef
        differs = differs & comparable

    n_cmp = int(comparable.sum())
    n_flip = int(differs.sum())
    lost = int((rej_a & ~rej_b & comparable).sum())
    gained = int((~rej_a & rej_b & comparable).sum())

    return FlipCounts(
        alpha=float(alpha),
        n_tests=int(a.size),
        n_compared=n_cmp,
        n_flip=n_flip,
        n_same=n_cmp - n_flip,
        n_undefined=n_undef,
        n_dep_to_indep=lost,
        n_indep_to_dep=gained,
        n_reject_ref=int((rej_a & comparable).sum()),
        n_reject_var=int((rej_b & comparable).sum()),
    )


def is_bitwise_identical(ref, var) -> bool:
    """Bitwise identity of the statistic vectors, NaN patterns included.

    This, not "flip rate == 0", is the E0.1 criterion. Bitwise identity implies zero
    flips; the converse does not hold, and bitwise is far more debuggable -- a flip rate
    of zero tells you nothing about where two runs began to diverge.
    """
    a = np.asarray(ref.stat, dtype=np.float64)
    b = np.asarray(var.stat, dtype=np.float64)
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    m = ~np.isnan(a)
    return a[m].tobytes() == b[m].tobytes()


def near_threshold_mass(result, alpha: float, width: float = 2.0) -> float:
    """Fraction of tests with p in [alpha/width, alpha*width] -- the E4.3 diagnostic.

    Computed on the statistic scale, where the band is [crit(alpha*width),
    crit(alpha/width)]. These are the only tests that can plausibly flip, so this is the
    one-line pre-flight check a practitioner can run before choosing a format.
    """
    s = np.asarray(result.stat, dtype=np.float64)
    lo = stats.crit(min(alpha * width, 0.999999))
    hi = stats.crit(alpha / width)
    ok = ~np.isnan(s)
    if not ok.any():
        return float("nan")
    return float(((s >= lo) & (s <= hi) & ok).sum() / ok.sum())


def kappa_histogram(result, bins=None):
    """log10(kappa_proxy) histogram -- the coverage evidence for a cell.

    A reviewer's real worry about any of the pairing devices is "did your test set even
    contain the ill-conditioned matrices?". This answers it numerically instead of
    rhetorically, and it is what lets E1.1 bound the trajectory bias rather than argue it.
    """
    if bins is None:
        bins = np.arange(-1.0, 21.0, 1.0)
    k = np.asarray(result.kappa_proxy, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lk = np.log10(k)
    lk = lk[np.isfinite(lk)]
    counts, edges = np.histogram(lk, bins=bins)
    return counts.astype(np.int64), edges


# ------------------------------------------------------------------- amplification
@dataclass(frozen=True)
class Amplification:
    n_flipped_tests: int
    edges_changed: Optional[int]
    shd: Optional[int]
    flip_rate: Optional[float]

    @property
    def amplification(self) -> Optional[float]:
        """Graph edges changed per flipped decision. Dimensionless. This is C1.

        None when no test flipped: the ratio is UNDEFINED, not infinite and not zero.
        A run with zero flips and a non-zero SHD would mean something has gone wrong
        upstream, not that amplification is unbounded.
        """
        if self.n_flipped_tests == 0 or self.edges_changed is None:
            return None
        return self.edges_changed / self.n_flipped_tests

    @property
    def shd_per_flip_rate(self) -> Optional[float]:
        """The proposal's literal SHD / flip-rate ratio, kept for continuity.

        Reported under this name, never as "amplification": it has units of edges and
        moves by orders of magnitude with N for reasons that are not scientific.
        """
        if not self.flip_rate or self.shd is None:
            return None
        return self.shd / self.flip_rate

    def as_dict(self) -> dict:
        d = asdict(self)
        d["amplification"] = self.amplification
        d["shd_per_flip_rate"] = self.shd_per_flip_rate
        return d


# ------------------------------------------------------------------ graph comparison
@dataclass(frozen=True)
class SkeletonDiff:
    n_edges_ref: int
    n_edges_var: int
    edges_changed: int  # symmetric difference == skeleton SHD
    edges_lost: int  # in ref, not in var
    edges_gained: int  # in var, not in ref

    def as_dict(self) -> dict:
        return asdict(self)


def skeleton_diff(ref, var) -> SkeletonDiff:
    """Symmetric difference of two undirected skeletons.

    For skeletons this IS the structural Hamming distance: with no orientations to
    disagree about, SHD reduces to counting edges present in one graph and not the other.
    causal-learn's SHD is used for the CPDAG comparison in E2.3, where orientation makes
    the two quantities differ.
    """
    a = ref.edge_set() if hasattr(ref, "edge_set") else set(ref)
    b = var.edge_set() if hasattr(var, "edge_set") else set(var)
    return SkeletonDiff(
        n_edges_ref=len(a),
        n_edges_var=len(b),
        edges_changed=len(a ^ b),
        edges_lost=len(a - b),
        edges_gained=len(b - a),
    )


def shd_vs_truth(result, true_skeleton) -> int:
    """Skeleton SHD against a known ground-truth adjacency (E4.1).

    Distinct from precision-vs-precision: near-threshold coin flips that do not degrade
    accuracy are a far softer finding than systematic loss of true edges, and a lower
    precision scoring BETTER on some seeds is a real possibility that must be reported
    rather than smoothed away.
    """
    import numpy as _np

    tru = {(int(a), int(b)) for a, b in _np.argwhere(_np.triu(true_skeleton, 1))}
    return len(result.edge_set() ^ tru)


@dataclass(frozen=True)
class SepsetDivergence:
    n_common_removed: int  # edges removed by BOTH runs
    n_sepset_differ: int  # ... whose recorded separating set differs
    n_only_ref: int
    n_only_var: int

    @property
    def rate(self):
        if self.n_common_removed == 0:
            return None
        return self.n_sepset_differ / self.n_common_removed

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rate"] = self.rate
        return d


def sepset_divergence(ref, var) -> SepsetDivergence:
    """E2.3. How often the two runs removed the same edge but recorded a DIFFERENT
    separating set.

    This channel is currently unmeasured by anyone. Separating sets determine
    v-structures and therefore orientation, so precision can change the CPDAG even when
    the skeleton is bit-for-bit identical -- which a skeleton-only comparison would score
    as perfect agreement.
    """
    a, b = ref.sepsets, var.sepsets
    common = set(a) & set(b)
    differ = sum(1 for k in common if tuple(a[k]) != tuple(b[k]))
    return SepsetDivergence(
        n_common_removed=len(common),
        n_sepset_differ=differ,
        n_only_ref=len(set(a) - set(b)),
        n_only_var=len(set(b) - set(a)),
    )


def trajectory_jaccard(ref, var) -> float:
    """Overlap of the two executed test sets -- how far the search paths diverged.

    This is the empirical bound on replay mode's bias: if the trajectories are nearly
    identical, freezing one of them costs almost nothing.
    """
    return ref.trajectory.jaccard(var.trajectory)
