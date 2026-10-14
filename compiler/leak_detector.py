"""Resource-leak detection as a forward "may-be-open" dataflow analysis.

State at each program point:
  * bindings  - which allocation sites each local variable may refer to (aliases)
  * open      - allocation sites that may still be open

Transfer per CFG item:
  1. releases   ``x.close()``, ``free(x)``, ``os.close(x)``, ``with x:``, ``x.release()``
  2. escapes    ``return x``, ``yield x``, ``self.x = x``, ``lst.append(x)``: ownership
                moves to the caller/another object, so it is not a leak here
  3. bindings   ``x = open(..)`` creates a site, ``y = x`` aliases, any other
                assignment or ``del`` drops the binding
  4. lost handles  an open site that no variable refers to any more can never be
                closed: reported immediately (CWE-772, "handle overwritten")

At the function's Exit node, every site still in ``open`` leaks on at least one
path (normal, early-return, break/continue, or exception edge). A witness path is
reconstructed for the UI.

Allocation sites are tracked rather than variable names, so ``g = f; g.close()``
correctly closes ``f``.
"""

import ast
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import networkx as nx

from .ast_utils import (
    ImportResolver,
    defined_names,
    evaluated_parts,
    name_matches,
    scope_declared_nonlocal,
    walk_scope,
)
from .dataflow import solve_forward
from .findings import Finding

# Allocator pattern -> resource kind. Patterns use ``name_matches`` semantics.
ALLOCATORS: Dict[str, str] = {
    "open": "file", "io.open": "file", "codecs.open": "file", "gzip.open": "file", "bz2.open": "file",
    "lzma.open": "file", "tarfile.open": "archive", "zipfile.ZipFile": "archive",
    "tempfile.TemporaryFile": "file", "tempfile.NamedTemporaryFile": "file",
    "socket": "socket", "socket.socket": "socket", "socket.create_connection": "socket",
    "connect": "connection", "sqlite3.connect": "database connection", "psycopg2.connect": "database connection",
    "pymysql.connect": "database connection", "mysql.connector.connect": "database connection",
    "urlopen": "http response", "urllib.request.urlopen": "http response",
    "requests.Session": "http session", "malloc": "memory", "os.open": "file descriptor",
    "multiprocessing.Pool": "process pool", "ThreadPoolExecutor": "thread pool",
    "ProcessPoolExecutor": "process pool",
}
# For backwards compatibility with the original module's public constants.
TRACKED_ALLOCATORS = set(ALLOCATORS)
RELEASE_METHODS = {"close", "disconnect", "free", "release", "shutdown_and_close", "terminate", "__exit__"}
TRACKED_DEALLOCATORS = RELEASE_METHODS
RELEASE_FUNCTIONS = ["free", "os.close", "contextlib.closing", "closing"]
ESCAPE_METHODS = {"append", "add", "put", "put_nowait", "extend", "insert", "setdefault", "register", "push", "enter_context"}

_LEAK_CWE = {"lock": "CWE-667"}
_SEVERITY = {"lock": "high"}

State = Tuple[FrozenSet[Tuple[str, str]], FrozenSet[str]]


def _value_names(expr: Optional[ast.AST]) -> set:
    """Names whose *object* flows into the value of ``expr`` (not just a method/attr of it)."""
    out = set()
    if expr is None:
        return out
    if isinstance(expr, ast.Name):
        out.add(expr.id)
    elif isinstance(expr, ast.Attribute):
        pass  # ``f.name``: an attribute of the handle, not the handle itself
    elif isinstance(expr, ast.Call):
        if not isinstance(expr.func, ast.Attribute):
            out |= _value_names(expr.func)
        for a in expr.args:
            out |= _value_names(a)
        for k in expr.keywords:
            out |= _value_names(k.value)
    else:
        for child in ast.iter_child_nodes(expr):
            out |= _value_names(child)
    return out


class LeakDetector:
    def __init__(self, cfg: nx.DiGraph, resolver: Optional[ImportResolver] = None):
        self.cfg = cfg
        self.function = cfg.graph.get("name", "<module>")
        scope = cfg.graph.get("scope")
        self.resolver = resolver or ImportResolver(scope if scope is not None else ast.Module(body=[], type_ignores=[]))
        self.globals = scope_declared_nonlocal(scope) if scope is not None else set()
        self.sites: Dict[str, Dict[str, Any]] = {}
        self._findings: Optional[List[Finding]] = None

    # ----------------------------------------------------------------- helpers
    def _allocator_kind(self, call: ast.AST) -> Optional[str]:
        if not isinstance(call, ast.Call):
            return None
        name = self.resolver.resolve(call.func)
        if not name_matches(name, ALLOCATORS):
            return None
        best = max((p for p in ALLOCATORS if name_matches(name, [p])), key=len)
        return ALLOCATORS[best]

    def _site(self, item: ast.AST, var: str, kind: str, alloc: str) -> str:
        key = f"{item.lineno}:{item.col_offset}:{var}"
        self.sites.setdefault(key, {"var": var, "kind": kind, "allocator": alloc, "line": item.lineno,
                                    "end_line": getattr(item, "end_lineno", item.lineno), "item": item})
        return key

    # ---------------------------------------------------------------- transfer
    def _transfer_item(self, item: ast.AST, state: State, events: Optional[List] = None) -> State:
        binds: Dict[str, FrozenSet[str]] = {}
        for var, site in state[0]:
            binds[var] = binds.get(var, frozenset()) | {site}
        open_ = set(state[1])

        def sites_of(name):
            return binds.get(name, frozenset())

        parts = evaluated_parts(item)

        # 1. releases ------------------------------------------------------
        for part in parts:
            for n in walk_scope(part):
                if not isinstance(n, ast.Call):
                    continue
                if isinstance(n.func, ast.Attribute) and n.func.attr in RELEASE_METHODS and isinstance(n.func.value, ast.Name):
                    open_ -= sites_of(n.func.value.id)
                elif name_matches(self.resolver.resolve(n.func), RELEASE_FUNCTIONS) and n.args and isinstance(n.args[0], ast.Name):
                    open_ -= sites_of(n.args[0].id)
                elif isinstance(n.func, ast.Attribute) and n.func.attr in ESCAPE_METHODS:
                    for a in list(n.args) + [k.value for k in n.keywords]:
                        for v in _value_names(a):
                            open_ -= sites_of(v)
        if isinstance(item, (ast.With, ast.AsyncWith)):
            for wi in item.items:
                if isinstance(wi.context_expr, ast.Name):
                    open_ -= sites_of(wi.context_expr.id)  # ``with f:`` closes f on exit

        # 2. escapes -------------------------------------------------------
        escaping = set()
        if isinstance(item, ast.Return):
            escaping |= _value_names(item.value)
        for part in parts:
            for n in walk_scope(part):
                if isinstance(n, (ast.Yield, ast.YieldFrom)):
                    escaping |= _value_names(n.value)
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            if any(not isinstance(t, ast.Name) or t.id in self.globals for t in targets):
                escaping |= _value_names(item.value)
        for v in escaping:
            open_ -= sites_of(v)

        # 3. bindings ------------------------------------------------------
        new_binding: Dict[str, FrozenSet[str]] = {}
        if isinstance(item, (ast.Assign, ast.AnnAssign)) and getattr(item, "value", None) is not None:
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            kind = self._allocator_kind(item.value)
            if kind and names:
                alloc = self.resolver.resolve(item.value.func) or "?"
                site = self._site(item, names[0], kind, alloc)
                open_.add(site)
                for nm in names:
                    new_binding[nm] = frozenset({site})
            elif isinstance(item.value, ast.Name) and sites_of(item.value.id):
                for nm in names:
                    new_binding[nm] = sites_of(item.value.id)
        if (isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                and isinstance(item.value.func, ast.Attribute) and item.value.func.attr == "acquire"
                and isinstance(item.value.func.value, ast.Name)):
            var = item.value.func.value.id
            site = self._site(item, var, "lock", f"{var}.acquire")
            open_.add(site)
            new_binding[var] = sites_of(var) | {site}

        for name in defined_names(item):
            binds.pop(name, None)
        binds.update(new_binding)

        # 4. lost handles --------------------------------------------------
        referenced = set().union(*binds.values()) if binds else set()
        for site in sorted(open_ - referenced):
            open_.discard(site)
            if events is not None:
                events.append(("lost", site, item))

        pairs = frozenset((v, s) for v, ss in binds.items() for s in ss)
        return pairs, frozenset(open_)

    def _transfer(self, node: int, state: State, events=None) -> State:
        for item in self.cfg.nodes[node]["block"].statements:
            state = self._transfer_item(item, state, events)
        return state

    @staticmethod
    def _meet(states) -> State:
        pairs, opened = frozenset(), frozenset()
        for p, o in states:
            pairs |= p
            opened |= o
        return pairs, opened

    # ----------------------------------------------------------------- results
    def _witness(self, site: str, end: int, OUT: Dict[int, State]) -> List[int]:
        entry = self.cfg.graph["entry"]
        start = next((n for n, d in self.cfg.nodes(data=True)
                      if any(it is self.sites[site]["item"] for it in d["block"].statements)), entry)
        try:
            prefix = nx.shortest_path(self.cfg, entry, start)
        except nx.NetworkXNoPath:
            prefix = [start]
        parent = {start: None}
        queue = [start]
        while queue:
            n = queue.pop(0)
            if n == end:
                break
            for s in self.cfg.successors(n):
                if s not in parent and (s == end or site in OUT.get(s, (frozenset(), frozenset()))[1]):
                    parent[s] = n
                    queue.append(s)
        if end not in parent:
            return prefix + [end]
        tail, n = [], end
        while n is not None:
            tail.append(n)
            n = parent[n]
        return prefix[:-1] + list(reversed(tail))

    def analyze(self) -> List[Finding]:
        if self._findings is not None:
            return self._findings
        IN, OUT = solve_forward(self.cfg, (frozenset(), frozenset()), self._transfer, self._meet)
        exit_node = self.cfg.graph["exit"]
        findings: List[Finding] = []
        seen = set()

        # Lost handles: replay each block with event collection switched on.
        for n in sorted(IN):
            events: List = []
            self._transfer(n, IN[n], events)
            for _, site, item in events:
                if ("lost", site) in seen:
                    continue
                seen.add(("lost", site))
                path = self._witness(site, n, OUT)
                findings.append(self._finding(self.sites[site], site, path, lost_at=item.lineno))

        # Sites that may still be open when the function exits.
        for site in sorted(IN.get(exit_node, (frozenset(), frozenset()))[1]):
            if ("lost", site) in seen:
                continue
            path = self._witness(site, exit_node, OUT)
            findings.append(self._finding(self.sites[site], site, path))
        self._findings = findings
        return findings

    def _finding(self, meta, site, path, lost_at: Optional[int] = None) -> Finding:
        var, kind = meta["var"], meta["kind"]
        via_exception = any(self.cfg.edges[u, v].get("label", "").find("exception") >= 0
                            or "raise" in self.cfg.edges[u, v].get("label", "")
                            for u, v in zip(path, path[1:]) if self.cfg.has_edge(u, v))
        if kind == "lock":
            title = "Lock acquired but not released"
            msg = (f"Lock '{var}' acquired at line {meta['line']} is still held when the function exits on "
                   f"execution path {path}. Other threads will block forever (deadlock / denial of service).")
            hint = f"Use 'with {var}:' instead of acquire()/release()."
        elif lost_at is not None:
            title = "Resource handle lost while open"
            msg = (f"Resource handle '{var}' ({kind}) allocated at line {meta['line']} is overwritten or deleted at "
                   f"line {lost_at} while still open; it can never be closed.")
            hint = f"Close '{var}' before reassigning it, or use a 'with' block per resource."
        else:
            title = "Resource leak"
            how = " (only when an exception is raised)" if via_exception else ""
            msg = (f"Resource handle '{var}' allocated at line {meta['line']} is never closed on execution path "
                   f"{path}{how}.")
            hint = f"Wrap the allocation in 'with {meta['allocator']}(...) as {var}:' or close it in a 'finally' block."
        module_level = self.function == "<module>"
        return Finding(
            rule_id="LOCK-NOT-RELEASED" if kind == "lock" else ("RESOURCE-LOST" if lost_at else "RESOURCE-LEAK"),
            title=title, cwe=_LEAK_CWE.get(kind, "CWE-772"),
            severity="low" if module_level else _SEVERITY.get(kind, "medium"),
            confidence="medium" if via_exception else "high",
            category="resource", message=msg, line=meta["line"], end_line=meta["end_line"],
            function=self.function, path=path, fix_hint=hint,
            extra={"variable": var, "resource_type": kind, "allocator": meta["allocator"],
                   "via_exception": via_exception, "lost_at": lost_at},
        )

    def detect_leaks(self) -> List[Dict[str, Any]]:
        """Backwards-compatible dict view used by the UI and the patcher."""
        out = []
        for f in self.analyze():
            out.append({
                "variable": f.extra["variable"],
                "resource_type": f.extra["allocator"],
                "allocated_line": f.line,
                "execution_path": f.path,
                "message": f.message,
                "cwe": f.cwe,
                "function": f.function,
                "rule_id": f.rule_id,
            })
        return out
