"""Shared AST helpers: scope-aware traversal, def/use sets and dotted call names."""

import ast
from typing import Dict, Iterator, List, Optional, Set

# Nodes that open a new Python scope. Names bound inside them do not affect
# the enclosing function, so most passes must not descend into them.
SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
TRY_NODES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())

# Calls that can observe or mutate a frame's locals dynamically. Functions that
# use them are excluded from optimizations that reason about local variables.
DYNAMIC_SCOPE_CALLS = {"locals", "vars", "eval", "exec", "dir", "globals"}


def walk_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Like ``ast.walk`` but does not descend into nested scopes.

    Nested scope nodes themselves are yielded (so callers can see them), their
    bodies are not.
    """
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, SCOPE_NODES):
                yield child
            else:
                stack.append(child)


def evaluated_parts(item: ast.AST) -> List[ast.AST]:
    """Sub-trees of a CFG item that are evaluated when control enters its block.

    Compound statements only evaluate their header (``if`` test, ``for``
    iterable, ...); their bodies are separate CFG blocks.
    """
    if isinstance(item, (ast.If, ast.While)):
        return [item.test]
    if isinstance(item, (ast.For, ast.AsyncFor)):
        return [item.iter, item.target]
    if isinstance(item, (ast.With, ast.AsyncWith)):
        parts: List[ast.AST] = []
        for wi in item.items:
            parts.append(wi.context_expr)
            if wi.optional_vars is not None:
                parts.append(wi.optional_vars)
        return parts
    if isinstance(item, ast.ExceptHandler):
        return [item.type] if item.type is not None else []
    if isinstance(item, ast.Match):
        return [item.subject]
    if isinstance(item, ast.match_case):
        return [item.guard] if item.guard is not None else []
    if isinstance(item, TRY_NODES):
        return []
    if isinstance(item, FUNC_NODES):
        args = item.args
        return list(item.decorator_list) + list(args.defaults) + [d for d in args.kw_defaults if d is not None]
    if isinstance(item, ast.ClassDef):
        return list(item.decorator_list) + list(item.bases) + [k.value for k in item.keywords]
    return [item]


def _pattern_names(pattern: ast.AST) -> Set[str]:
    out = set()
    for n in ast.walk(pattern):
        if isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
    return out


def defined_names(item: ast.AST) -> Set[str]:
    """Names (re)bound or deleted by a CFG item in the current scope."""
    out: Set[str] = set()
    if isinstance(item, FUNC_NODES + (ast.ClassDef,)):
        out.add(item.name)
    elif isinstance(item, ast.ExceptHandler):
        if item.name:
            out.add(item.name)
    elif isinstance(item, (ast.Import, ast.ImportFrom)):
        for alias in item.names:
            if alias.name != "*":
                out.add((alias.asname or alias.name).split(".")[0])
    elif isinstance(item, ast.match_case):
        out |= _pattern_names(item.pattern)
    for part in evaluated_parts(item):
        for n in walk_scope(part):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)
    return out


def used_names(item: ast.AST) -> Set[str]:
    """Names read by a CFG item (over-approximated through nested scopes)."""
    out: Set[str] = set()
    roots = [item] if isinstance(item, FUNC_NODES + (ast.ClassDef, ast.Lambda)) else evaluated_parts(item)
    for part in roots:
        for n in ast.walk(part):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                out.add(n.id)
    if isinstance(item, ast.AugAssign) and isinstance(item.target, ast.Name):
        out.add(item.target.id)
    return out


def scope_declared_nonlocal(scope: ast.AST) -> Set[str]:
    """Names declared ``global``/``nonlocal`` anywhere inside ``scope``."""
    out: Set[str] = set()
    for n in ast.walk(scope):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def names_in_nested_scopes(scope: ast.AST) -> Set[str]:
    """All names referenced inside scopes nested in ``scope`` (closure candidates)."""
    out: Set[str] = set()
    for node in walk_scope(scope):
        if node is not scope and isinstance(node, SCOPE_NODES):
            for n in ast.walk(node):
                if isinstance(n, ast.Name):
                    out.add(n.id)
    return out


def uses_dynamic_scope(scope: ast.AST) -> bool:
    for n in walk_scope(scope):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in DYNAMIC_SCOPE_CALLS:
            return True
    return False


def bound_names_anywhere(tree: ast.AST) -> Set[str]:
    """Every name bound anywhere in the module (used to detect shadowed builtins)."""
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, FUNC_NODES + (ast.ClassDef,)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                out.add((alias.asname or alias.name).split(".")[0])
    return out


def dotted_name(expr: ast.AST) -> Optional[str]:
    """Render ``a.b.c`` / ``a.b().c`` style call targets as a dotted string."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = dotted_name(expr.value)
        return f"{base or '?'}.{expr.attr}"
    if isinstance(expr, ast.Call):
        base = dotted_name(expr.func)
        return f"{base or '?'}()"
    if isinstance(expr, ast.Subscript):
        base = dotted_name(expr.value)
        return f"{base or '?'}[]"
    return None


class ImportResolver:
    """Maps local aliases back to fully qualified names.

    ``import subprocess as sp`` makes ``sp.run`` resolve to ``subprocess.run``;
    ``from os import system`` makes ``system`` resolve to ``os.system``.
    """

    def __init__(self, tree: ast.AST):
        self.aliases: Dict[str, str] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        self.aliases[a.asname] = a.name
            elif isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    if a.name != "*":
                        self.aliases[a.asname or a.name] = f"{n.module}.{a.name}"

    def resolve(self, expr: ast.AST) -> Optional[str]:
        name = dotted_name(expr)
        if not name:
            return None
        head, _, rest = name.partition(".")
        if head in self.aliases:
            head = self.aliases[head]
        return f"{head}.{rest}" if rest else head


def name_matches(name: Optional[str], patterns) -> bool:
    """Match a resolved dotted name against patterns.

    ``"os.system"`` matches ``os.system`` and ``x.os.system``; ``"*.execute"``
    matches any method call named ``execute``.
    """
    if not name:
        return False
    for p in patterns:
        if p.startswith("*."):
            if "." in name and name.rsplit(".", 1)[1] == p[2:]:
                return True
        elif name == p or name.endswith("." + p):
            return True
    return False


def short_source(node: ast.AST, limit: int = 60) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = type(node).__name__
    text = text.splitlines()[0] if text else ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def iter_scopes(tree: ast.AST, prefix: str = ""):
    """Yield ``(qualified_name, node)`` for every function in the module."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, FUNC_NODES):
            qual = f"{prefix}{node.name}"
            yield qual, node
            yield from iter_scopes(node, qual + ".")
        elif isinstance(node, ast.ClassDef):
            yield from iter_scopes(node, f"{prefix}{node.name}.")
        elif not isinstance(node, SCOPE_NODES):
            yield from iter_scopes(node, prefix)
