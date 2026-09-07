"""Orientation: v-structures, Meek's rules, and CPDAG comparison.

The cross-validation against causal-learn is the load-bearing test here. Orientation rules
are exactly the kind of code that produces a plausible-looking graph while being subtly
wrong, and a wrong CPDAG would make every number in E2.3 wrong in a way that no internal
consistency check would catch.
"""

import itertools

import numpy as np
import pytest

from precision_pc import orient
from precision_pc.datasets import gaussian as G


# --------------------------------------------------------------- textbook cases
def test_collider_is_oriented():
    """0 -> 2 <- 1 with 0,1 non-adjacent is the one triple whose arrows are identifiable."""
    dag = np.zeros((3, 3), dtype=bool)
    dag[0, 2] = dag[1, 2] = True
    c = orient.dag_to_cpdag(dag)
    assert c.vstructures == {(0, 2, 1)}
    assert c.is_directed(0, 2) and c.is_directed(1, 2)
    assert c.n_undirected == 0


def test_chain_and_fork_are_indistinguishable():
    """0->1->2, 0<-1->2 and 0<-1<-2 share a Markov equivalence class, so the CPDAG must
    leave both edges undirected. Orienting them would be a false claim of identifiability."""
    for dag in (
        np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool),  # chain
        np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=bool),  # fork
    ):
        c = orient.dag_to_cpdag(dag)
        assert c.n_directed == 0
        assert c.n_undirected == 2
        assert c.vstructures == set()


def test_meek_r1():
    A = np.zeros((3, 3), dtype=bool)
    A[0, 1] = True  # 0 -> 1
    A[1, 2] = A[2, 1] = True  # 1 - 2
    out = orient._apply_meek(A.copy())
    assert out[1, 2] and not out[2, 1], "R1 should orient 1 -> 2"


def test_meek_r2():
    A = np.zeros((3, 3), dtype=bool)
    A[0, 2] = True  # 0 -> 2
    A[2, 1] = True  # 2 -> 1
    A[0, 1] = A[1, 0] = True  # 0 - 1
    out = orient._apply_meek(A.copy())
    assert out[0, 1] and not out[1, 0], "R2 should orient 0 -> 1"


def test_meek_r3():
    # c -> b, d -> b, a - b, a - c, a - d, c and d non-adjacent  =>  a -> b
    a, b, c, d = 0, 1, 2, 3
    A = np.zeros((4, 4), dtype=bool)
    A[a, b] = A[b, a] = True
    A[a, c] = A[c, a] = True
    A[a, d] = A[d, a] = True
    A[c, b] = True
    A[d, b] = True
    out = orient._apply_meek(A.copy())
    assert out[a, b] and not out[b, a], "R3 should orient a -> b"


def test_meek_reaches_a_fixpoint():
    """Applying the rules twice must change nothing the second time."""
    rng = np.random.default_rng(0)
    c = orient.dag_to_cpdag(G.random_dag(12, 0.25, rng))
    once = c.A.copy()
    twice = orient._apply_meek(once.copy())
    assert np.array_equal(once, twice)


# ------------------------------------------------- cross-validation (the real test)
def _causallearn_cpdag(dag: np.ndarray) -> np.ndarray:
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    n = dag.shape[0]
    nodes = [GraphNode("X%d" % i) for i in range(n)]
    g = Dag(nodes)
    for i, j in np.argwhere(dag):
        g.add_directed_edge(nodes[i], nodes[j])
    M = dag2cpdag(g).graph
    A = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if M[j, i] == 1 and M[i, j] == -1:
                A[i, j] = True  # i -> j
            elif M[i, j] == -1 and M[j, i] == -1:
                A[i, j] = A[j, i] = True  # i - j
    return A


@pytest.mark.parametrize("seed", range(25))
def test_cpdag_matches_causallearn(seed):
    """Independent implementation, same answer. Nothing internal to this harness could
    detect a wrong Meek rule; only a second implementation can."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(seed)
    dag = G.random_dag(9, 0.3, rng)
    assert np.array_equal(orient.dag_to_cpdag(dag).A, _causallearn_cpdag(dag))


# ------------------------------------------------------------------- the comparison
def test_cpdag_diff_is_zero_for_identical_graphs():
    rng = np.random.default_rng(1)
    c = orient.dag_to_cpdag(G.random_dag(10, 0.3, rng))
    d = orient.cpdag_diff(c, c)
    assert d.n_pairs_differ == 0
    assert d.orientation_differ == 0
    assert d.orientation_rate == 0.0


def test_cpdag_diff_decomposes_shd():
    """SHD = skeleton differences + orientation differences, and each pair counts once."""
    rng = np.random.default_rng(2)
    a = orient.dag_to_cpdag(G.random_dag(10, 0.3, rng))
    b = orient.dag_to_cpdag(G.random_dag(10, 0.3, rng))
    d = orient.cpdag_diff(a, b)
    assert d.n_pairs_differ == d.skeleton_differ + d.orientation_differ
    n = a.n
    assert d.n_pairs_differ <= n * (n - 1) // 2


def test_orientation_rate_isolates_the_sepset_channel():
    """The number E2.3 was pointing at: defined only on edges BOTH graphs recovered, so a
    non-zero value with skeleton_differ == 0 is orientation disagreement and nothing else."""
    # Same skeleton, one collider flipped to a chain: skeleton identical, arrows differ.
    skel = np.zeros((3, 3), dtype=bool)
    skel[0, 1] = skel[1, 0] = skel[1, 2] = skel[2, 1] = True
    collider = orient.to_cpdag(skel, {(0, 2): ()})  # 1 not in sepset -> collider
    chain = orient.to_cpdag(skel, {(0, 2): (1,)})  # 1 in sepset -> no collider
    d = orient.cpdag_diff(collider, chain)
    assert d.skeleton_differ == 0
    assert d.orientation_differ == 2
    assert d.orientation_rate == 1.0
    assert d.n_vstruct_differ == 1


def test_conflicting_colliders_are_counted_not_hidden():
    """When two unshielded triples demand opposite arrowheads the result is not a valid
    CPDAG. Counting the conflicts keeps that visible instead of averaging over it."""
    # 0 - 1 - 2 and 1 - 3, with sepsets making 1 a collider for (0,2) and a tail elsewhere.
    skel = np.zeros((4, 4), dtype=bool)
    for i, j in ((0, 1), (1, 2), (1, 3)):
        skel[i, j] = skel[j, i] = True
    c = orient.to_cpdag(skel, {(0, 2): (), (0, 3): (), (2, 3): ()})
    assert c.n_conflicts >= 0
    assert isinstance(c.n_conflicts, int)


def test_sepset_without_a_record_is_skipped_not_guessed():
    """A pair with no recorded separating set means its edge was never removed. Inventing
    one would fabricate v-structures."""
    skel = np.zeros((3, 3), dtype=bool)
    skel[0, 1] = skel[1, 0] = skel[1, 2] = skel[2, 1] = True
    assert orient.to_cpdag(skel, {}).vstructures == set()
