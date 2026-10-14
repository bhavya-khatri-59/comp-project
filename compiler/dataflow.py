"""Generic iterative (worklist) dataflow solver over a CFG.

Replaces enumeration of all simple paths, which is exponential in the number of
branches and only follows each loop once. A worklist analysis visits each block
a bounded number of times and reaches a fixpoint that covers every path,
including any number of loop iterations.
"""

from collections import deque
from typing import Any, Callable, Dict, Iterable, Tuple

import networkx as nx


class AnalysisLimitError(RuntimeError):
    """Raised when an analysis does not converge within its iteration budget."""


def solve_forward(
    cfg: nx.DiGraph,
    entry_state: Any,
    transfer: Callable[[int, Any], Any],
    meet: Callable[[Iterable[Any]], Any],
    max_visits_per_node: int = 200,
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """Forward analysis. Returns ``(IN, OUT)`` maps for every reachable block.

    Blocks whose predecessors have not produced a state yet are skipped (their
    contribution is "no path yet"), which gives optimistic, precise results for
    loops.
    """
    entry = cfg.graph["entry"]
    order = list(nx.dfs_preorder_nodes(cfg, entry))
    rank = {n: i for i, n in enumerate(order)}
    IN: Dict[int, Any] = {}
    OUT: Dict[int, Any] = {}
    worklist = deque(order)
    queued = set(order)
    budget = max_visits_per_node * max(len(order), 1)

    while worklist:
        budget -= 1
        if budget < 0:
            raise AnalysisLimitError(f"dataflow analysis of '{cfg.graph.get('name')}' did not converge")
        n = worklist.popleft()
        queued.discard(n)
        if n == entry:
            state = entry_state
        else:
            preds = [OUT[p] for p in cfg.predecessors(n) if p in OUT]
            if not preds:
                continue
            state = meet(preds)
        IN[n] = state
        out = transfer(n, state)
        if n not in OUT or OUT[n] != out:
            OUT[n] = out
            for s in sorted(cfg.successors(n), key=lambda x: rank.get(x, 0)):
                if s not in queued:
                    worklist.append(s)
                    queued.add(s)
    return IN, OUT


def solve_backward(
    cfg: nx.DiGraph,
    exit_state: Any,
    transfer: Callable[[int, Any], Any],
    meet: Callable[[Iterable[Any]], Any],
    bottom: Any,
    max_visits_per_node: int = 200,
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """Backward analysis (e.g. liveness). Returns ``(IN, OUT)`` maps, where OUT is at block end."""
    exit_node = cfg.graph["exit"]
    nodes = list(cfg.nodes)
    IN: Dict[int, Any] = {n: bottom for n in nodes}
    OUT: Dict[int, Any] = {n: bottom for n in nodes}
    worklist = deque(nx.dfs_postorder_nodes(cfg, cfg.graph["entry"]))
    seen = set(worklist)
    worklist.extend(n for n in nodes if n not in seen)
    queued = set(worklist)
    budget = max_visits_per_node * max(len(nodes), 1)

    while worklist:
        budget -= 1
        if budget < 0:
            raise AnalysisLimitError(f"backward analysis of '{cfg.graph.get('name')}' did not converge")
        n = worklist.popleft()
        queued.discard(n)
        if n == exit_node:
            out = exit_state
        else:
            succ_states = [IN[s] for s in cfg.successors(n)]
            out = meet(succ_states) if succ_states else bottom
        OUT[n] = out
        new_in = transfer(n, out)
        if new_in != IN[n]:
            IN[n] = new_in
            for p in cfg.predecessors(n):
                if p not in queued:
                    worklist.append(p)
                    queued.add(p)
    return IN, OUT
