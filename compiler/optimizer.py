"""Global (CFG-based) machine-independent optimizations and the pass driver.

Passes, run to a fixpoint:
  1. constant folding / algebraic simplification   (``ast_analyzer``)
  2. constant propagation        forward dataflow, lattice UNBOUND < const < NAC
  3. dead code elimination       constant branches + CFG-unreachable statements
  4. dead store elimination      backward liveness analysis
then once:
  5. strength reduction          ``i * c`` for range loop induction variables

All passes keep the original line numbers on the nodes they keep or create, so
diagnostics on the optimized program still point at the user's source.
"""

import ast
import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from .ast_analyzer import ConstantFolder, FoldRefused, fold_binop, fold_compare, fold_unary, is_pure
from .ast_utils import (
    FUNC_NODES,
    SCOPE_NODES,
    bound_names_anywhere,
    defined_names,
    iter_scopes,
    names_in_nested_scopes,
    scope_declared_nonlocal,
    short_source,
    used_names,
    uses_dynamic_scope,
    walk_scope,
)
from .cfg_builder import CFGBuilder
from .dataflow import solve_backward, solve_forward

ALL_PASSES = (
    "constant-folding",
    "constant-propagation",
    "dead-code-elimination",
    "dead-store-elimination",
    "strength-reduction",
)
_BUILTINS_WE_REASON_ABOUT = {"range", "len", "int", "float"}
_MAX_PROPAGATED_LEN = 256


@dataclass
class OptimizationResult:
    tree: ast.Module
    code: str
    log: List[Dict] = field(default_factory=list)
    rounds: int = 0


def _scopes(tree: ast.Module):
    yield "<module>", tree
    yield from iter_scopes(tree)


def _record(log, pass_name, node, before, after):
    log.append({"pass": pass_name, "line": getattr(node, "lineno", None), "before": before, "after": after})


# ---------------------------------------------------------------------------
# Constant propagation
# ---------------------------------------------------------------------------
class _NAC:
    def __repr__(self):
        return "NAC"


NAC = _NAC()  # "not a constant": several values reach this point
_UNKNOWN = object()


class Const:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Const) and type(self.value) is type(other.value) and repr(self.value) == repr(other.value)

    def __hash__(self):
        return hash((type(self.value), repr(self.value)))

    def __repr__(self):
        return f"Const({self.value!r})"


def _propagatable(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, complex)):
        if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 256:
            return False
        return value == value  # excludes NaN
    if isinstance(value, (str, bytes)):
        return len(value) <= _MAX_PROPAGATED_LEN
    return False


def const_eval(expr: ast.AST, env: Dict):
    """Evaluate an expression over the constant environment, or return ``_UNKNOWN``."""
    try:
        if isinstance(expr, ast.Constant):
            return expr.value
        if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load):
            v = env.get(expr.id)
            return v.value if isinstance(v, Const) else _UNKNOWN
        if isinstance(expr, ast.BinOp):
            left, right = const_eval(expr.left, env), const_eval(expr.right, env)
            if left is _UNKNOWN or right is _UNKNOWN:
                return _UNKNOWN
            return fold_binop(expr.op, left, right)
        if isinstance(expr, ast.UnaryOp):
            v = const_eval(expr.operand, env)
            return _UNKNOWN if v is _UNKNOWN else fold_unary(expr.op, v)
        if isinstance(expr, ast.Compare):
            vals = [const_eval(e, env) for e in [expr.left] + expr.comparators]
            if any(v is _UNKNOWN for v in vals):
                return _UNKNOWN
            return fold_compare(vals[0], expr.ops, vals[1:])
        if isinstance(expr, ast.BoolOp):
            is_and = isinstance(expr.op, ast.And)
            last = _UNKNOWN
            for sub in expr.values:
                last = const_eval(sub, env)
                if last is _UNKNOWN:
                    return _UNKNOWN
                if bool(last) != is_and:
                    return last
            return last
    except FoldRefused:
        return _UNKNOWN
    return _UNKNOWN


def cp_meet(states):
    states = list(states)
    out = {}
    for key in set().union(*states):
        vals = [s.get(key, NAC) for s in states]
        out[key] = vals[0] if all(v == vals[0] for v in vals) else NAC
    return out


def cp_transfer_item(item: ast.AST, env: Dict, excluded: Set[str]) -> Dict:
    defs = defined_names(item)
    if not defs:
        return env
    new = dict(env)
    for name in defs:
        if name not in excluded:
            new[name] = NAC
    if isinstance(item, ast.Delete):
        for t in item.targets:
            if isinstance(t, ast.Name):
                new.pop(t.id, None)
    target, value = None, _UNKNOWN
    if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
        target, value = item.targets[0].id, const_eval(item.value, env)
    elif isinstance(item, ast.AnnAssign) and item.value is not None and isinstance(item.target, ast.Name):
        target, value = item.target.id, const_eval(item.value, env)
    elif isinstance(item, ast.AugAssign) and isinstance(item.target, ast.Name):
        target = item.target.id
        cur, rhs = env.get(target), const_eval(item.value, env)
        if isinstance(cur, Const) and rhs is not _UNKNOWN:
            try:
                value = fold_binop(item.op, cur.value, rhs)
            except FoldRefused:
                value = _UNKNOWN
    if target is not None and target not in excluded and value is not _UNKNOWN and _propagatable(value):
        new[target] = Const(value)
    return new


class _Substituter(ast.NodeTransformer):
    def __init__(self, consts: Dict[str, Const], log, pass_name="constant-propagation"):
        self.consts = consts
        self.log = log
        self.pass_name = pass_name
        self.changed = False

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.consts:
            new = ast.copy_location(ast.Constant(value=self.consts[node.id].value), node)
            _record(self.log, self.pass_name, node, node.id, short_source(new))
            self.changed = True
            return new
        return node

    def generic_visit(self, node):
        if isinstance(node, SCOPE_NODES):
            return node  # names inside nested scopes may refer to something else
        return super().generic_visit(node)

    visit_Lambda = visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = lambda self, n: n


def _substitute_item(item: ast.AST, sub: _Substituter):
    if isinstance(item, (ast.If, ast.While)):
        item.test = sub.visit(item.test)
    elif isinstance(item, (ast.For, ast.AsyncFor)):
        item.iter = sub.visit(item.iter)
    elif isinstance(item, (ast.With, ast.AsyncWith)):
        for wi in item.items:
            wi.context_expr = sub.visit(wi.context_expr)
    elif isinstance(item, ast.Match):
        item.subject = sub.visit(item.subject)
    elif isinstance(item, ast.match_case):
        if item.guard is not None:
            item.guard = sub.visit(item.guard)
    elif isinstance(item, ast.ExceptHandler):
        if item.type is not None:
            item.type = sub.visit(item.type)
    elif isinstance(item, (ast.Try, ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.Delete)
                    + FUNC_NODES + (ast.ClassDef,)) or (hasattr(ast, "TryStar") and isinstance(item, ast.TryStar)):
        return
    elif isinstance(item, ast.AugAssign):
        item.value = sub.visit(item.value)  # the target is both read and written
    else:
        sub.generic_visit(item)


def _entry_env(scope) -> Dict:
    env = {}
    if isinstance(scope, FUNC_NODES):
        a = scope.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            env[arg.arg] = NAC
    return env


def propagate_constants(tree: ast.Module, log: List[Dict]) -> bool:
    excluded_everywhere = scope_declared_nonlocal(tree)
    changed = False
    for _, scope in _scopes(tree):
        cfg = CFGBuilder().build_cfg(scope)
        excluded = excluded_everywhere

        def transfer(n, env):
            for item in cfg.nodes[n]["block"].statements:
                env = cp_transfer_item(item, env, excluded)
            return env

        IN, _ = solve_forward(cfg, _entry_env(scope), transfer, cp_meet)
        per_item: Dict[int, tuple] = {}
        for n, env in IN.items():
            for item in cfg.nodes[n]["block"].statements:
                if id(item) in per_item:
                    per_item[id(item)] = (item, cp_meet([per_item[id(item)][1], env]))
                else:
                    per_item[id(item)] = (item, env)
                env = cp_transfer_item(item, env, excluded)
        for item, env in per_item.values():
            consts = {k: v for k, v in env.items() if isinstance(v, Const) and k not in excluded}
            if not consts or not (used_names(item) & consts.keys()):
                continue
            sub = _Substituter(consts, log)
            _substitute_item(item, sub)
            changed |= sub.changed
    return changed


# ---------------------------------------------------------------------------
# Statement removal helpers (shared by DCE and DSE)
# ---------------------------------------------------------------------------
def _binding_counter(nodes) -> Counter:
    """Count binding occurrences per name in ``nodes`` (not descending into nested scopes)."""
    counts: Counter = Counter()
    for root in nodes:
        if isinstance(root, FUNC_NODES + (ast.ClassDef,)):
            counts[root.name] += 1
            continue
        for n in walk_scope(root):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                counts[n.id] += 1
            elif isinstance(n, FUNC_NODES + (ast.ClassDef,)):
                counts[n.name] += 1
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    counts[(a.asname or a.name).split(".")[0]] += 1
            elif isinstance(n, ast.ExceptHandler) and n.name:
                counts[n.name] += 1
    return counts


class _ScopeGuard:
    """Decides whether statements can be deleted without changing name resolution.

    Deleting the only binding of a name that is still read would turn it from
    a local into a global lookup, and deleting a ``yield`` would turn a generator into a plain
    function, so such statements are kept.
    """

    def __init__(self, scope: ast.AST):
        body = scope.body
        self.counts = _binding_counter(body)
        self.loaded = {n.id for s in body for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if isinstance(scope, FUNC_NODES):
            a = scope.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
                self.counts[arg.arg] += 1

    def try_delete(self, stmts: List[ast.stmt]) -> bool:
        for s in stmts:
            for n in walk_scope(s):
                if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)):
                    return False
        removed = _binding_counter(stmts)
        if any(self.counts[name] - c <= 0 and name in self.loaded for name, c in removed.items()):
            return False
        self.counts -= removed
        return True


_BODY_FIELDS = ("body", "orelse", "finalbody")


def _filter_statements(scope: ast.AST, should_remove: Callable[[ast.stmt], bool], guard: _ScopeGuard,
                       log: List[Dict], pass_name: str, describe: Callable[[ast.stmt], str]) -> bool:
    changed = False

    def visit_list(stmts: List[ast.stmt]) -> List[ast.stmt]:
        nonlocal changed
        kept = []
        for s in stmts:
            if should_remove(s) and guard.try_delete([s]):
                _record(log, pass_name, s, short_source(s, 80), describe(s))
                changed = True
                continue
            visit_stmt(s)
            kept.append(s)
        return kept

    def visit_stmt(s: ast.stmt):
        if isinstance(s, SCOPE_NODES):
            return
        for f in _BODY_FIELDS:
            if isinstance(getattr(s, f, None), list):
                setattr(s, f, visit_list(getattr(s, f)))
        for h in getattr(s, "handlers", []) or []:
            h.body = visit_list(h.body)
        for c in getattr(s, "cases", []) or []:
            c.body = visit_list(c.body)

    scope.body = visit_list(scope.body)
    return changed


# ---------------------------------------------------------------------------
# Dead code elimination
# ---------------------------------------------------------------------------
class _BranchFolder(ast.NodeTransformer):
    def __init__(self, log):
        self.log = log
        self.changed = False
        self.guards: List[_ScopeGuard] = []

    def visit_Module(self, node):
        self.guards.append(_ScopeGuard(node))
        self.generic_visit(node)
        self.guards.pop()
        return node

    def visit_FunctionDef(self, node):
        self.guards.append(_ScopeGuard(node))
        self.generic_visit(node)
        self.guards.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        return node  # class bodies are not optimized

    def _fold(self, node, keep, drop, what):
        if not self.guards[-1].try_delete(drop):
            return node
        self.changed = True
        _record(self.log, "dead-code-elimination", node, what, "removed dead branch" if keep else "removed statement")
        return keep

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            taken = bool(node.test.value)
            keep, drop = (node.body, node.orelse) if taken else (node.orelse, node.body)
            return self._fold(node, keep, drop, f"if {short_source(node.test)}: ...")
        return node

    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and not node.test.value:
            return self._fold(node, node.orelse, node.body, f"while {short_source(node.test)}: ...")
        return node


def fold_branches(tree: ast.Module, log: List[Dict]) -> bool:
    bf = _BranchFolder(log)
    bf.visit(tree)
    return bf.changed


def remove_unreachable(tree: ast.Module, log: List[Dict]) -> bool:
    changed = False
    for _, scope in _scopes(tree):
        cfg = CFGBuilder().build_cfg(scope)
        covered = {id(item) for _, d in cfg.nodes(data=True) for item in d["block"].statements}
        changed |= _filter_statements(
            scope, lambda s: id(s) not in covered, _ScopeGuard(scope), log,
            "dead-code-elimination", lambda s: "removed unreachable statement")
    return changed


# ---------------------------------------------------------------------------
# Dead store elimination (liveness)
# ---------------------------------------------------------------------------
def _kills(item: ast.AST) -> Set[str]:
    if isinstance(item, (ast.For, ast.AsyncFor)):
        return set()  # the target is not rebound when the loop is exhausted
    return defined_names(item)


def _removable_store(item: ast.AST, params: Set[str]) -> Optional[Set[str]]:
    def pure_value(v):
        if isinstance(v, ast.Name):
            return v.id in params
        if isinstance(v, ast.Lambda):
            return True
        return is_pure(v) and not any(isinstance(n, ast.Name) for n in ast.walk(v))

    if isinstance(item, ast.Assign) and all(isinstance(t, ast.Name) for t in item.targets) and pure_value(item.value):
        return {t.id for t in item.targets}
    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None and item.simple \
            and pure_value(item.value):
        return {item.target.id}
    return None


def eliminate_dead_stores(tree: ast.Module, log: List[Dict]) -> bool:
    changed = False
    globals_ = scope_declared_nonlocal(tree)
    for _, fn in iter_scopes(tree):
        if uses_dynamic_scope(fn):
            continue
        excluded = globals_ | names_in_nested_scopes(fn)
        a = fn.args
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [v for v in (a.vararg, a.kwarg) if v]}
        cfg = CFGBuilder().build_cfg(fn)

        def transfer(n, live):
            for item in reversed(cfg.nodes[n]["block"].statements):
                live = (live - _kills(item)) | used_names(item)
            return live

        _, OUT = solve_backward(cfg, frozenset(), lambda n, s: frozenset(transfer(n, s)),
                                lambda states: frozenset().union(*states), frozenset())
        dead: Dict[int, bool] = {}
        for n, live in OUT.items():
            for item in reversed(cfg.nodes[n]["block"].statements):
                targets = _removable_store(item, params)
                is_dead = bool(targets) and not (targets & live) and not (targets & excluded)
                dead[id(item)] = dead.get(id(item), True) and is_dead
                live = (live - _kills(item)) | used_names(item)

        changed |= _filter_statements(
            fn, lambda s: dead.get(id(s), False), _ScopeGuard(fn), log,
            "dead-store-elimination", lambda s: "removed (value never read)")
    return changed


# ---------------------------------------------------------------------------
# Strength reduction of loop induction variables
# ---------------------------------------------------------------------------
def _int_const(node) -> Optional[int]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _int_const(node.operand)
        return -v if v is not None else None
    return None


class _StrengthReducer(ast.NodeTransformer):
    """``for i in range(a, b, s): ... i * c ...``  ->  a running ``t += s * c``.

    ``i`` is an int (range only yields ints) and is never reassigned in the
    body, so the rewrite is exact. The increment is placed at the top of the
    body so ``continue`` cannot skip it.
    """

    def __init__(self, log, taken_names: Set[str]):
        self.log = log
        self.taken = taken_names
        self.changed = False

    def visit_FunctionDef(self, node):
        return node  # each function is reduced separately (see strength_reduce)

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef

    def _fresh(self, base: str) -> str:
        name, k = base, 1
        while name in self.taken:
            k += 1
            name = f"{base}_{k}"
        self.taken.add(name)
        return name

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        it = node.iter
        if not (isinstance(node.target, ast.Name) and isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "range" and 1 <= len(it.args) <= 3 and not it.keywords):
            return node
        var = node.target.id
        step = 1 if len(it.args) < 3 else _int_const(it.args[2])
        if not step:
            return node
        start = ast.Constant(value=0) if len(it.args) == 1 else it.args[0]
        if not (_int_const(start) is not None or (isinstance(start, ast.Name) and start.id != var)):
            return node
        for s in node.body:
            for n in walk_scope(s):
                if isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, (ast.Store, ast.Del)):
                    return node

        products: Dict[int, List[ast.BinOp]] = {}
        for s in node.body:
            for n in walk_scope(s):
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
                    for a, b in ((n.left, n.right), (n.right, n.left)):
                        c = _int_const(b)
                        if isinstance(a, ast.Name) and a.id == var and c is not None and c not in (0, 1, -1):
                            products.setdefault(c, []).append(n)
                            break
        if not products:
            return node

        prelude = []
        for c, uses in products.items():
            temp = self._fresh(f"{var}_x{c}" if c > 0 else f"{var}_xm{-c}")
            replace = {id(u) for u in uses}

            class _Swap(ast.NodeTransformer):
                def visit_BinOp(self, b):
                    self.generic_visit(b)
                    if id(b) in replace:
                        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), b)
                    return b

                def generic_visit(self, n):
                    return n if isinstance(n, SCOPE_NODES) else super().generic_visit(n)

            node.body = [_Swap().visit(s) for s in node.body]
            init_value = ast.BinOp(left=ast.BinOp(left=copy.deepcopy(start), op=ast.Sub(), right=ast.Constant(step)),
                                   op=ast.Mult(), right=ast.Constant(c))
            init = ast.copy_location(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=init_value), node)
            bump = ast.copy_location(ast.AugAssign(target=ast.Name(id=temp, ctx=ast.Store()), op=ast.Add(),
                                                   value=ast.Constant(step * c)), node.body[0])
            prelude.append(init)
            node.body.insert(0, bump)
            self.changed = True
            _record(self.log, "strength-reduction", node, f"{var} * {c}  ({len(uses)} use(s) in loop)",
                    f"{temp} += {step * c} each iteration")
        for stmt in prelude + [node]:
            ast.fix_missing_locations(stmt)
        return prelude + [node]


def strength_reduce(tree: ast.Module, log: List[Dict]) -> bool:
    if "range" in bound_names_anywhere(tree):
        return False
    changed = False
    for _, fn in iter_scopes(tree):
        taken = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
        sr = _StrengthReducer(log, taken)
        new_body = []
        for s in fn.body:
            out = sr.visit(s)
            new_body.extend(out if isinstance(out, list) else [out])
        fn.body = new_body
        changed |= sr.changed
    return changed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _ensure_bodies(tree: ast.AST):
    for node in ast.walk(tree):
        for f in ("body",):
            body = getattr(node, f, None)
            if isinstance(body, list) and not body and not isinstance(node, (ast.Module, ast.Lambda, ast.IfExp)):
                setattr(node, f, [ast.copy_location(ast.Pass(), node)])
        if isinstance(node, ast.Try) and not node.handlers and not node.finalbody:
            node.finalbody = [ast.copy_location(ast.Pass(), node)]


def optimize_tree(tree: ast.Module, passes=ALL_PASSES, max_rounds: int = 8) -> OptimizationResult:
    """Run the enabled passes on a *copy* of ``tree`` until nothing changes."""
    passes = set(passes)
    tree = copy.deepcopy(tree)
    log: List[Dict] = []
    shadowed = bound_names_anywhere(tree) & _BUILTINS_WE_REASON_ABOUT
    rounds = 0

    def fold() -> bool:
        nonlocal tree
        folder = ConstantFolder(log, shadowed)
        tree = folder.visit(tree)
        return folder.changed

    for rounds in range(1, max_rounds + 1):
        changed = False
        if "constant-folding" in passes:
            changed |= fold()
        if "constant-propagation" in passes:
            changed |= propagate_constants(tree, log)
        if "dead-code-elimination" in passes:
            changed |= fold_branches(tree, log)
            _ensure_bodies(tree)
            changed |= remove_unreachable(tree, log)
        if "dead-store-elimination" in passes:
            changed |= eliminate_dead_stores(tree, log)
        _ensure_bodies(tree)
        ast.fix_missing_locations(tree)
        if not changed:
            break

    if "strength-reduction" in passes and strength_reduce(tree, log):
        if "constant-folding" in passes:
            fold()
        ast.fix_missing_locations(tree)

    return OptimizationResult(tree=tree, code=ast.unparse(tree), log=log, rounds=rounds)
