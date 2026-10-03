"""Helpers for moving between pywhyllm's edge dictionaries and causal-learn graphs."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.PDAG2DAG import pdag2dag

EdgeVotes = Mapping[Tuple[str, str], int]


def pywhyllm_edges_to_causal_graph(
    model_edges,
    node_names: Sequence[str],
    min_votes: int = 1,
    name_map: Mapping[str, str] | None = None,
    conflict: str = "majority",
    warn_on_cycle: bool = True,
) -> CausalGraph:
    """Convert pywhyllm's suggested edges into a causal-learn CausalGraph.

    ModelSuggester.suggest_relationships returns Dict[(cause, effect), votes],
    where votes counts how many experts proposed that edge. causal-learn's PC
    returns a CausalGraph whose .G holds an endpoint matrix. This bridges the
    two so an LLM-suggested graph can be drawn and compared against the one PC
    discovered, using the same calls.

    Args:
        model_edges: The dict returned by suggest_relationships. The Confounder
            strategy returns (edges, confounders) instead of a bare dict; a
            2-tuple is accepted and its first element used.
        node_names: Every node the graph should contain, in the order the
            returned graph should use. Nodes with no surviving edge are kept as
            isolated nodes, so the result stays comparable to a PC graph over
            the same variables.
        min_votes: Drop edges proposed by fewer than this many experts. The
            default keeps everything; raise it to trust only consensus edges.
        name_map: Optional alias -> original name lookup (e.g. des_to_col),
            applied to the edge endpoints before matching against node_names.
        conflict: How to handle a pair suggested in both directions, which
            happens when experts disagree.
            "majority" - the direction with more votes wins; a tie becomes an
                undirected edge.
            "undirected" - always an undirected edge.
            "bidirected" - always a bidirected (<->) edge, reading the
                disagreement as an unmeasured common cause.
        warn_on_cycle: Warn if the directed edges contain a cycle. Nothing in
            causal-learn forbids one, but DoWhy needs a DAG, so it is worth
            knowing before the graph is passed downstream.

    Returns:
        A CausalGraph with .G populated and .draw_pydot_graph() available.
    """
    if conflict not in {"majority", "undirected", "bidirected"}:
        raise ValueError(f"conflict must be majority/undirected/bidirected, got {conflict!r}")

    # suggest_relationships returns (edges, confounders) under the Confounder strategy
    if isinstance(model_edges, tuple) and len(model_edges) == 2:
        model_edges = model_edges[0]
    if model_edges is None:
        raise ValueError(
            "model_edges is None. suggest_relationships returns None for the IV and "
            "Mediator strategies, which pywhyllm defines in RelationshipStrategy but "
            "never dispatches on."
        )

    node_names = list(node_names)
    if len(set(node_names)) != len(node_names):
        raise ValueError("node_names must be unique")

    resolved = _resolve_votes(model_edges, node_names, min_votes, name_map, conflict)

    nodes = [GraphNode(name) for name in node_names]
    index = {name: node for name, node in zip(node_names, nodes)}
    graph = GeneralGraph(nodes)

    directed: List[Tuple[str, str]] = []
    for (cause, effect), kind in resolved.items():
        if kind == "directed":
            graph.add_directed_edge(index[cause], index[effect])
            directed.append((cause, effect))
        elif kind == "undirected":
            graph.add_edge(Edge(index[cause], index[effect], Endpoint.TAIL, Endpoint.TAIL))
        else:  # bidirected
            graph.add_edge(Edge(index[cause], index[effect], Endpoint.ARROW, Endpoint.ARROW))

    if warn_on_cycle and directed:
        cycles = list(nx.simple_cycles(nx.DiGraph(directed)))
        if cycles:
            warnings.warn(
                f"{len(cycles)} cycle(s) in the suggested edges, e.g. {cycles[0]}. "
                "causal-learn accepts this, but DoWhy requires a DAG.",
                stacklevel=2,
            )

    cg = CausalGraph(len(nodes), node_names)
    cg.G = graph
    return cg


def _resolve_votes(
    model_edges: EdgeVotes,
    node_names: Sequence[str],
    min_votes: int,
    name_map: Mapping[str, str] | None,
    conflict: str,
) -> Dict[Tuple[str, str], str]:
    """Filter, rename and de-conflict the raw votes into one entry per node pair.

    Returns {(cause, effect): "directed" | "undirected" | "bidirected"}. For an
    undirected or bidirected result the key order carries no meaning.
    """
    known = set(node_names)
    votes: Dict[Tuple[str, str], int] = {}

    for edge, count in model_edges.items():
        if not (isinstance(edge, tuple) and len(edge) == 2):
            raise TypeError(f"Expected (cause, effect) keys, got {edge!r}")
        cause, effect = (_rename(end, name_map) for end in edge)

        if count < min_votes or cause == effect:
            continue
        missing = {cause, effect} - known
        if missing:
            warnings.warn(f"Dropping edge {edge}: {sorted(missing)} not in node_names", stacklevel=3)
            continue
        votes[(cause, effect)] = votes.get((cause, effect), 0) + count

    resolved: Dict[Tuple[str, str], str] = {}
    for (cause, effect), count in votes.items():
        if (effect, cause) in resolved or (cause, effect) in resolved:
            continue  # pair already settled from the other direction

        reverse = votes.get((effect, cause))
        if reverse is None:
            resolved[(cause, effect)] = "directed"
        elif conflict == "undirected":
            resolved[(cause, effect)] = "undirected"
        elif conflict == "bidirected":
            resolved[(cause, effect)] = "bidirected"
        elif count > reverse:
            resolved[(cause, effect)] = "directed"
        elif reverse > count:
            resolved[(effect, cause)] = "directed"
        else:
            resolved[(cause, effect)] = "undirected"

    return resolved


def _rename(name: str, name_map: Mapping[str, str] | None) -> str:
    return name_map.get(name, name) if name_map else name


def compare_graphs(
    graph_a,
    graph_b,
    labels: Tuple[str, str] = ("a", "b"),
) -> Dict[str, object]:
    """Quantify agreement between two causal-learn graphs, symmetrically.

    causal-learn's SHD, AdjacencyConfusion and ArrowConfusion all take a
    (truth, est) pair, which is the wrong frame for "how much do PC and the LLM
    agree" -- neither is truth. This reports symmetric statistics instead, split
    into two levels because the graphs mean different things by an undirected
    edge: PC's `---` means the orientation is not identifiable from data, while
    the LLM's means the experts split. Scoring those against each other as if
    they disagreed would penalise PC for being honest about non-identifiability.

        skeleton    - adjacency only, orientation ignored. The fair headline.
        orientation - direction agreement among pairs BOTH graphs (a) call
                      adjacent and (b) commit to a direction on.

    Args:
        graph_a, graph_b: CausalGraph or GeneralGraph objects over the same
            node names. Node order may differ; nodes are matched by name.
        labels: Names for the two graphs, used in the "only_in_*" keys.

    Returns:
        A flat dict of counts and rates. Notable entries:
            skeleton_jaccard - shared edges / union of edges. 1.0 is identical.
            skeleton_kappa   - Cohen's kappa on the adjacency indicator over all
                n*(n-1)/2 pairs, correcting for the agreement two graphs of
                these densities would reach by chance. This is the number to
                quote: raw agreement is inflated when both graphs are sparse.
            orientation_agreement - of the pairs where both commit to an arrow,
                the fraction pointing the same way. None if there are none.
            shd - causal-learn's Structural Hamming Distance, as a distance.
    """
    mat_a, names_a = _endpoint_matrix(graph_a)
    mat_b, names_b = _endpoint_matrix(graph_b)

    if set(names_a) != set(names_b):
        only_a = sorted(set(names_a) - set(names_b))
        only_b = sorted(set(names_b) - set(names_a))
        raise ValueError(f"Graphs have different node sets. Only in {labels[0]}: {only_a}; only in {labels[1]}: {only_b}")

    names = list(names_a)
    idx_a = {n: i for i, n in enumerate(names_a)}
    idx_b = {n: i for i, n in enumerate(names_b)}

    both = only_in_a = only_in_b = neither = 0
    same_dir = opposite_dir = uncommitted = 0

    for u, v in ((u, v) for k, u in enumerate(names) for v in names[k + 1:]):
        state_a = _pair_state(mat_a, idx_a[u], idx_a[v])
        state_b = _pair_state(mat_b, idx_b[u], idx_b[v])
        adj_a, adj_b = state_a is not None, state_b is not None

        if adj_a and adj_b:
            both += 1
            if state_a in ("forward", "backward") and state_b in ("forward", "backward"):
                if state_a == state_b:
                    same_dir += 1
                else:
                    opposite_dir += 1
            else:
                uncommitted += 1
        elif adj_a:
            only_in_a += 1
        elif adj_b:
            only_in_b += 1
        else:
            neither += 1

    n_pairs = both + only_in_a + only_in_b + neither
    union = both + only_in_a + only_in_b
    committed = same_dir + opposite_dir

    # Cohen's kappa on the adjacency indicator across all node pairs
    p_observed = (both + neither) / n_pairs if n_pairs else 0.0
    rate_a = (both + only_in_a) / n_pairs if n_pairs else 0.0
    rate_b = (both + only_in_b) / n_pairs if n_pairs else 0.0
    p_chance = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)
    kappa = (p_observed - p_chance) / (1 - p_chance) if p_chance < 1 else None

    return {
        "n_nodes": len(names),
        "n_pairs": n_pairs,
        f"edges_{labels[0]}": both + only_in_a,
        f"edges_{labels[1]}": both + only_in_b,
        "skeleton_shared": both,
        f"skeleton_only_in_{labels[0]}": only_in_a,
        f"skeleton_only_in_{labels[1]}": only_in_b,
        "skeleton_jaccard": both / union if union else None,
        "skeleton_raw_agreement": p_observed,
        "skeleton_kappa": kappa,
        "orientation_same": same_dir,
        "orientation_opposite": opposite_dir,
        "orientation_uncommitted": uncommitted,
        "orientation_agreement": same_dir / committed if committed else None,
        "shd": SHD(_as_general_graph(graph_a), _as_general_graph(graph_b)).get_shd(),
    }


def _as_general_graph(graph) -> GeneralGraph:
    """Accept either a CausalGraph wrapper or a bare GeneralGraph."""
    return graph.G if hasattr(graph, "G") else graph


def _endpoint_matrix(graph):
    gg = _as_general_graph(graph)
    return gg.graph, gg.get_node_names()


def _pair_state(matrix, i: int, j: int):
    """Classify the edge between i and j, or None when they are not adjacent."""
    ij, ji = matrix[i, j], matrix[j, i]
    if ij == 0 and ji == 0:
        return None
    if ij == -1 and ji == 1:
        return "forward"   # i --> j
    if ji == -1 and ij == 1:
        return "backward"  # j --> i
    if ij == -1 and ji == -1:
        return "undirected"
    return "other"         # <->, circle endpoints, and anything else adjacent


def same_markov_equivalence_class(
    graph_a,
    graph_b,
    labels: Tuple[str, str] = ("a", "b"),
) -> Dict[str, object]:
    """Test whether two graphs encode the same Markov equivalence class.

    Two DAGs are Markov equivalent iff they share a skeleton and a set of
    v-structures (Verma & Pearl 1990), and the CPDAG is the unique canonical
    representative of that class. So the test reduces to: canonicalise both to
    CPDAGs, then compare endpoint matrices exactly.

    causal-learn ships mec_check(DAG1, DAG2) in causallearn.utils.MECCheck, but
    it takes two *DAGs*. Neither graph here is necessarily one: PC returns a
    CPDAG, and a pywhyllm vote graph is an arbitrary digraph that may be cyclic
    or carry undirected edges from expert disagreement. This canonicalises each
    input first and reports why, rather than failing opaquely.

    Args:
        graph_a, graph_b: CausalGraph or GeneralGraph over the same node names.
        labels: Names for the two graphs, used in the returned reasons.

    Returns:
        A dict with:
            same_mec - True, False, or None when the question is not well posed
                for one of the inputs (see reason).
            reason   - Plain-text explanation, always populated.
            cpdag_a, cpdag_b - The canonical CPDAGs, or None. Compare these to
                see *where* two non-equivalent graphs diverge.
            shd - SHD between the two CPDAGs, or None. Zero iff same_mec.
    """
    out: Dict[str, object] = {"same_mec": None, "reason": "", "cpdag_a": None, "cpdag_b": None, "shd": None}

    gg_a, gg_b = _as_general_graph(graph_a), _as_general_graph(graph_b)
    if set(gg_a.get_node_names()) != set(gg_b.get_node_names()):
        out["reason"] = "Graphs have different node sets; Markov equivalence is undefined."
        return out

    cpdags = {}
    for label, graph in zip(labels, (gg_a, gg_b)):
        cpdag, why = _to_cpdag(graph)
        if cpdag is None:
            out["reason"] = f"{label}: {why}"
            return out
        cpdags[label] = cpdag

    cpdag_a, cpdag_b = cpdags[labels[0]], cpdags[labels[1]]
    out["cpdag_a"], out["cpdag_b"] = cpdag_a, cpdag_b

    # dag2cpdag preserves node order, so the matrices are directly comparable
    same = bool(np.array_equal(cpdag_a.graph, cpdag_b.graph))
    out["same_mec"] = same
    out["shd"] = SHD(cpdag_a, cpdag_b).get_shd()
    out["reason"] = (
        "Canonical CPDAGs are identical: same skeleton and same v-structures."
        if same
        else f"Canonical CPDAGs differ (SHD {out['shd']}); the graphs imply different conditional independencies."
    )
    return out


def _to_cpdag(graph: GeneralGraph):
    """Canonicalise a DAG or PDAG to its CPDAG. Returns (cpdag, reason_if_None)."""
    names = graph.get_node_names()
    matrix = graph.graph
    directed: List[Tuple[str, str]] = []
    has_undirected = False

    for i, u in enumerate(names):
        for j in range(i + 1, len(names)):
            state = _pair_state(matrix, i, j)
            if state is None:
                continue
            if state == "forward":
                directed.append((u, names[j]))
            elif state == "backward":
                directed.append((names[j], u))
            elif state == "undirected":
                has_undirected = True
            else:
                return None, (
                    f"edge {u} <-> {names[j]} is bidirected or has a circle endpoint. "
                    "Markov equivalence classes are defined over DAGs; a graph with "
                    "latent-variable endpoints belongs to a PAG, not a CPDAG."
                )

    # Both dag2cpdag and pdag2dag loop until every edge is ordered, so a cycle
    # would hang them rather than raise. Check before handing anything over.
    cycles = list(nx.simple_cycles(nx.DiGraph(directed))) if directed else []
    if cycles:
        return None, (
            f"the directed edges contain a cycle, e.g. {cycles[0]}. Markov "
            "equivalence is only defined for acyclic graphs."
        )

    if has_undirected:
        # A PDAG must be extended to some consistent DAG before canonicalising.
        # Any consistent extension yields the same CPDAG.
        graph = pdag2dag(graph)

    return dag2cpdag(graph), ""
