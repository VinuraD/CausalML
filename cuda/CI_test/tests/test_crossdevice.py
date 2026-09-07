"""E0.2 restructured: the two arms, and the objection that motivated the split."""

from __future__ import annotations

import numpy as np
import pytest

from precision_pc import corr, crossdevice as X, stats
from precision_pc import testset as T


def test_disagreement_is_disjoint_from_the_decision_boundary():
    """The quantitative form of the objection to E0.2 as specified.

    At alpha = 0.01 the critical value is 2.5758, where normal-tail implementations agree
    to ~1e-14. They diverge only above stat ~ 8, where p < 1e-15. The region where
    implementations disagree does not overlap the region where decisions are marginal, so
    the experiment as written was predisposed to a null.
    """
    rows = {r["stat"]: r for r in X.tail_profile()}
    at_alpha = rows[2.5758]["rel_disagreement"]
    deep_tail = rows[8.0]["rel_disagreement"]
    assert at_alpha < 1e-13, "implementations disagree at the decision boundary?"
    assert deep_tail > 1e-3, "no disagreement even in the deep tail?"
    assert deep_tail / at_alpha > 1e9, "the two regions are not separated"


def test_transform_implementations_agree_where_decisions_are_made():
    rng = np.random.default_rng(0)
    rho = rng.uniform(-0.99, 0.99, 5000)
    impls = X.transcendental_arm(rho, 4000, k=1)
    base = impls["log1p_numpy"]
    c = stats.crit(0.01)
    for name, v in impls.items():
        m = np.isfinite(base) & np.isfinite(v)
        assert int(((base > c) != (v > c))[m].sum()) == 0, "%s flipped a decision" % name


def test_sensitivity_matches_the_analytic_prediction():
    """atanh's derivative is 1/(1-r^2), so a rho perturbation is amplified by
    sqrt(n-k-3)/(1-r^2). That is why |r| -> 1 is the danger zone, and why the tail at
    alpha = 0.01 was the wrong place to look."""
    rows = X.near_unit_correlation_sweep(n_obs=4000, k=1)
    for r in rows:
        assert r["sensitivity"] > 0
    # Sensitivity must grow by orders of magnitude as r approaches 1.
    assert rows[-1]["sensitivity"] / rows[0]["sensitivity"] > 1e6
    # And so must the distance one ulp of rho travels.
    assert rows[-1]["dstat_per_ulp"] / rows[0]["dstat_per_ulp"] > 1e6


def test_linalg_arm_holds_the_transform_fixed():
    """Arm B varies only how rho is computed. Every backend forms the statistic the same
    way -- host-side float64 log1p -- so a difference here is linear algebra, not the
    transform. That separation is what makes the arms independently attributable."""
    rng = np.random.default_rng(0)
    C = corr.correlation(rng.standard_normal((400, 12)))
    adj = np.ones((12, 12), dtype=bool)
    np.fill_diagonal(adj, False)
    ts = T.from_adjacency(adj, 1)
    res = X.linalg_arm(C, ts, 400, backends=("inv", "regression"))
    assert len(res) == 2
    for r in res:
        assert r.n_tests == len(ts)
        # Two routes of the same algorithm at FP64 disagree at ulp level, never in a
        # decision -- the E0.3 finding, reproduced here as the baseline for arm B.
        assert r.max_abs_drho < 1e-10
        assert r.flips[0.01] == 0


def test_cross_machine_floor_is_recorded():
    """Nothing at or below the floor two CPUs already show is evidence about a device."""
    assert 1e-14 < X.CROSS_MACHINE_FLOOR_REL < 1e-11
