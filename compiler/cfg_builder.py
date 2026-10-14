"""Control Flow Graph construction.

One CFG is built per scope (module body or function). Every basic block holds at
most one *CFG item*: a simple statement, or the header of a compound statement
(``if`` test, loop header, ``with`` items, ``except`` clause, ``match`` case).
Blocks carry the source line span of their item so the UI can map graph nodes
back to the editor.

Modelled control flow:
  * ``if`` / ``elif`` / ``else`` and ``match`` / ``case``
  * ``while`` / ``for`` (incl. ``else`` clauses) with ``break`` and ``continue``
  * ``return`` and ``raise``
  * ``try`` / ``except`` / ``else`` / ``finally``: every block inside a ``try``
    body has an exception edge to each handler, and ``finally`` bodies are
    duplicated per exit kind (normal, return, break, continue, raise), exactly
    like CPython's code generator does, so paths stay precise.
  * Branches whose condition is a compile-time constant are not generated, so
    unreachable code never gets a block (this is what dead-code elimination
    relies on).
"""

import ast
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .ast_utils import FUNC_NODES, TRY_NODES, iter_scopes, short_source


class BasicBlock:
    def __init__(self, block_id: int, label: str, kind: str = "stmt"):
        self.block_id = block_id
        self.label = label
        self.kind = kind
        self.statements: List[ast.AST] = []

    @property
    def lineno(self) -> Optional[int]:
        return getattr(self.statements[0], "lineno", None) if self.statements else None

    @property
    def end_lineno(self) -> Optional[int]:
        if not self.statements:
            return None
        item = self.statements[-1]
        # Compound statements only "own" their header line, not their body.
        if isinstance(item, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
                             ast.ExceptHandler, ast.Match, ast.match_case) + TRY_NODES + FUNC_NODES + (ast.ClassDef,)):
            return getattr(item, "lineno", None)
        return getattr(item, "end_lineno", None) or getattr(item, "lineno", None)

    def __str__(self):
        return f"Block {self.block_id}: {self.label}"


class _FinallyCtx:
    def __init__(self, body: List[ast.stmt]):
        self.body = body
        self.copies: Dict[Tuple[str, int], int] = {}
        self.copy_blocks: set = set()  # blocks of the duplicated finally bodies


def _const_truth(expr: ast.AST) -> Optional[bool]:
    if isinstance(expr, ast.Constant):
        try:
            return bool(expr.value)
        except Exception:
            return None
    return None


class CFGBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.block_counter = 0
        self._loops: List[Tuple[int, int, int]] = []  # (header, exit, finally depth)
        self._finally: List[_FinallyCtx] = []
        self._covered: List[bool] = []  # enclosing try regions that catch exceptions
        self.exit_block = -1

    # ------------------------------------------------------------------ blocks
    def new_block(self, label: str = "", kind: str = "stmt", item: Optional[ast.AST] = None) -> int:
        bid = self.block_counter
        self.block_counter += 1
        block = BasicBlock(bid, label, kind)
        if item is not None:
            block.statements.append(item)
        self.graph.add_node(bid, block=block, label=f"B{bid}: {label}" if label else f"B{bid}", kind=kind)
        return bid

    def _edge(self, u: int, v: int, label: str = ""):
        if self.graph.has_edge(u, v):
            old = self.graph.edges[u, v].get("label", "")
            if label and label not in old.split("/"):
                self.graph.edges[u, v]["label"] = f"{old}/{label}" if old else label
        else:
            self.graph.add_edge(u, v, label=label)

    # ------------------------------------------------------------------- build
    def build_cfg(self, scope: ast.AST, name: Optional[str] = None) -> nx.DiGraph:
        """Build the CFG of one scope (a ``Module`` or a function definition)."""
        if isinstance(scope, FUNC_NODES):
            name = name or scope.name
            body = scope.body
        else:
            name = name or "<module>"
            body = getattr(scope, "body", [])

        entry = self.new_block("Entry", kind="entry")
        self.exit_block = self.new_block("Exit", kind="exit")
        end = self._body(body, entry)
        if end is not None:
            self._edge(end, self.exit_block)

        self._prune_unreachable(entry)
        self._compact()
        for n, data in self.graph.nodes(data=True):
            blk = data["block"]
            data["lineno"] = blk.lineno
            data["end_lineno"] = blk.end_lineno
        self.graph.graph.update(entry=entry, exit=self.exit_block, name=name, scope=scope)
        return self.graph

    def _body(self, stmts: List[ast.stmt], current: Optional[int]) -> Optional[int]:
        """Process a statement list. Returns the open block or ``None`` if control cannot fall through."""
        for stmt in stmts:
            if current is None:
                break  # everything after a terminator is unreachable
            current = self._stmt(stmt, current)
        return current

    def _simple(self, node: ast.AST, current: int, kind: str = "stmt", label: Optional[str] = None) -> int:
        label = label or f"L{getattr(node, 'lineno', '?')}: {short_source(node)}"
        b = self.new_block(label, kind=kind, item=node)
        self._edge(current, b)
        return b

    def _stmt(self, node: ast.stmt, current: int) -> Optional[int]:
        if isinstance(node, ast.If):
            return self._if(node, current)
        if isinstance(node, (ast.While, ast.For, ast.AsyncFor)):
            return self._loop(node, current)
        if isinstance(node, TRY_NODES):
            return self._try(node, current)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            items = ", ".join(short_source(i, 40) for i in node.items)
            b = self._simple(node, current, kind="with", label=f"L{node.lineno}: with {items}")
            return self._body(node.body, b)
        if isinstance(node, ast.Match):
            return self._match(node, current)
        if isinstance(node, ast.Return):
            b = self._simple(node, current, kind="return")
            self._edge(b, self._through_finally("return", self.exit_block, len(self._finally), 0))
            return None
        if isinstance(node, ast.Raise):
            b = self._simple(node, current, kind="raise")
            if not any(self._covered):
                # Uncaught: leaves the function (through any pending finally bodies).
                self._edge(b, self._through_finally("raise", self.exit_block, len(self._finally), 0), "raise")
            # Otherwise the enclosing try adds exception edges from this block.
            return None
        if isinstance(node, (ast.Break, ast.Continue)):
            kind = "break" if isinstance(node, ast.Break) else "continue"
            b = self._simple(node, current, kind=kind)
            if self._loops:
                header, loop_exit, depth = self._loops[-1]
                target = loop_exit if kind == "break" else header
                self._edge(b, self._through_finally(kind, target, len(self._finally), depth), kind)
            return None
        if isinstance(node, FUNC_NODES + (ast.ClassDef,)):
            word = "class" if isinstance(node, ast.ClassDef) else "def"
            return self._simple(node, current, label=f"L{node.lineno}: {word} {node.name}")
        return self._simple(node, current)

    # ------------------------------------------------------------- statements
    def _if(self, node: ast.If, current: int) -> Optional[int]:
        cond = self._simple(node, current, kind="branch", label=f"L{node.lineno}: if {short_source(node.test)}")
        truth = _const_truth(node.test)
        ends = []
        if truth is not False:
            then = self.new_block("then", kind="empty")
            self._edge(cond, then, "True")
            ends.append(self._body(node.body, then))
        if truth is not True:
            other = self.new_block("else", kind="empty")
            self._edge(cond, other, "False")
            ends.append(self._body(node.orelse, other))
        return self._join([e for e in ends if e is not None], "join")

    def _join(self, ends: List[int], label: str) -> Optional[int]:
        if not ends:
            return None
        if len(ends) == 1:
            return ends[0]
        j = self.new_block(label, kind="empty")
        for e in ends:
            self._edge(e, j)
        return j

    def _loop(self, node, current: int) -> Optional[int]:
        if isinstance(node, ast.While):
            label = f"L{node.lineno}: while {short_source(node.test)}"
            truth = _const_truth(node.test)
        else:
            label = f"L{node.lineno}: for {short_source(node.target)} in {short_source(node.iter)}"
            truth = None
        header = self._simple(node, current, kind="loop", label=label)
        loop_exit = self.new_block("loop exit", kind="empty")

        if truth is not False:
            body = self.new_block("body", kind="empty")
            self._edge(header, body, "True" if isinstance(node, ast.While) else "next")
            self._loops.append((header, loop_exit, len(self._finally)))
            end = self._body(node.body, body)
            self._loops.pop()
            if end is not None:
                self._edge(end, header, "loop")

        if truth is not True:
            done = self.new_block("loop done", kind="empty")
            self._edge(header, done, "False" if isinstance(node, ast.While) else "done")
            end = self._body(node.orelse, done)  # ``else`` runs only when the loop was not broken
            if end is not None:
                self._edge(end, loop_exit)

        if self.graph.in_degree(loop_exit) == 0:
            self.graph.remove_node(loop_exit)  # e.g. ``while True`` without ``break``
            return None
        return loop_exit

    def _match(self, node: ast.Match, current: int) -> Optional[int]:
        head = self._simple(node, current, kind="branch", label=f"L{node.lineno}: match {short_source(node.subject)}")
        ends = []
        irrefutable = False
        for case in node.cases:
            lbl = f"L{case.pattern.lineno}: case {short_source(case.pattern, 40)}"
            cb = self.new_block(lbl, kind="branch", item=case)
            self._edge(head, cb, "case")
            ends.append(self._body(case.body, cb))
            if case.guard is None and isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None:
                irrefutable = True
        if not irrefutable:
            nomatch = self.new_block("no match", kind="empty")
            self._edge(head, nomatch, "no match")
            ends.append(nomatch)
        return self._join([e for e in ends if e is not None], "join")

    def _try(self, node, current: int) -> Optional[int]:
        fctx = _FinallyCtx(node.finalbody) if node.finalbody else None
        if fctx:
            self._finally.append(fctx)

        handler_entries = []
        for h in node.handlers:
            what = short_source(h.type, 40) if h.type is not None else ""
            alias = f" as {h.name}" if h.name else ""
            handler_entries.append(self.new_block(f"L{h.lineno}: except {what}{alias}".rstrip(), kind="except", item=h))

        start = self.block_counter
        try_block = self.new_block(f"L{node.lineno}: try", kind="try", item=node)
        self._edge(current, try_block)

        self._covered.append(bool(node.handlers) or fctx is not None)
        end = self._body(node.body, try_block)
        self._covered.pop()
        own_copies = fctx.copy_blocks if fctx else set()
        body_blocks = [b for b in range(start, self.block_counter) if b not in own_copies]
        for b in body_blocks:
            if self.graph.has_node(b):
                for h in handler_entries:
                    self._edge(b, h, "exception")

        self._covered.append(fctx is not None)
        if end is not None and node.orelse:
            end = self._body(node.orelse, end)
        ends = [end]
        for h_node, h_entry in zip(node.handlers, handler_entries):
            ends.append(self._body(h_node.body, h_entry))
        self._covered.pop()
        # Exceptions raised inside this try's own finally copies are not caught by it.
        region = [b for b in range(start, self.block_counter) if b not in own_copies] + handler_entries

        ends = [e for e in ends if e is not None]
        if fctx is None:
            return self._join(ends, "end try")

        # Exceptions escaping the try/except region still run ``finally``.
        level = len(self._finally)
        raise_entry = self._through_finally("raise", self.exit_block, level, 0)
        for b in region:
            if self.graph.has_node(b) and b != raise_entry:
                self._edge(b, raise_entry, "exception")
        result = None
        if ends:
            after = self.new_block("end try", kind="empty")
            normal = self._through_finally("normal", after, level, level - 1)
            for e in ends:
                self._edge(e, normal)
            result = after if self.graph.in_degree(after) else None
        self._finally.pop()
        return result

    def _through_finally(self, kind: str, target: int, level: int, depth: int) -> int:
        """Entry block of a jump that must first run the ``finally`` bodies above ``depth``."""
        if level <= depth:
            return target
        ctx = self._finally[level - 1]
        key = (kind, target)
        if key not in ctx.copies:
            first = self.block_counter
            entry = self.new_block(f"finally ({kind})", kind="finally")
            ctx.copies[key] = entry
            saved_finally, saved_covered = self._finally, self._covered
            self._finally, self._covered = saved_finally[: level - 1], []
            end = self._body(ctx.body, entry)
            self._finally, self._covered = saved_finally, saved_covered
            ctx.copy_blocks.update(range(first, self.block_counter))
            if end is not None:
                nxt = self._through_finally(kind, target, level - 1, depth)
                self._edge(end, nxt, "raise" if kind == "raise" else "")
        return ctx.copies[key]

    # ---------------------------------------------------------------- cleanup
    def _prune_unreachable(self, entry: int):
        reachable = nx.descendants(self.graph, entry) | {entry, self.exit_block}
        self.graph.remove_nodes_from([n for n in list(self.graph.nodes) if n not in reachable])

    def _compact(self):
        """Remove empty helper blocks, rewiring predecessors to successors."""
        for n in list(self.graph.nodes):
            if self.graph.nodes[n]["kind"] != "empty":
                continue
            succs = list(self.graph.successors(n))
            if n in succs:
                continue
            preds = list(self.graph.predecessors(n))
            for p in preds:
                in_label = self.graph.edges[p, n].get("label", "")
                for s in succs:
                    out_label = self.graph.edges[n, s].get("label", "")
                    self._edge(p, s, in_label or out_label)
            self.graph.remove_node(n)


def build_all_cfgs(tree: ast.Module) -> Dict[str, nx.DiGraph]:
    """One CFG for the module body plus one per (nested) function or method."""
    cfgs: Dict[str, nx.DiGraph] = {}
    module_stmts = [s for s in tree.body if not isinstance(s, (ast.Import, ast.ImportFrom, ast.Expr))
                    or (isinstance(s, ast.Expr) and not isinstance(s.value, ast.Constant))]
    if any(not isinstance(s, FUNC_NODES + (ast.ClassDef,)) for s in module_stmts):
        cfgs["<module>"] = CFGBuilder().build_cfg(tree, "<module>")
    for qual, fn in iter_scopes(tree):
        cfgs[qual] = CFGBuilder().build_cfg(fn, qual)
    return cfgs
