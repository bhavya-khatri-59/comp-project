"""Local machine-independent optimizations: constant folding and algebraic simplification.

Every rewrite here must preserve Python semantics exactly:

* Folding only evaluates operators on literal constants, never calls, and
  refuses results above a size cap. ``"A" * 10 ** 10`` would otherwise make the
  *analyzer* allocate 10 GB (a denial of service against the tool, CWE-400).
* Identities such as ``x * 1 -> x`` are only applied when ``x`` is provably an
  ``int``: for strings/lists ``x * 1`` makes a copy, for floats ``x + 0`` turns
  ``-0.0`` into ``0.0``, ``True * 1`` is ``1``, and ``f() * 0`` must still call
  ``f``.
"""

import ast
import math
import operator
from typing import Dict, List, Optional, Set

from .ast_utils import FUNC_NODES, short_source, walk_scope

MAX_FOLDED_LEN = 4096
MAX_FOLDED_INT_BITS = 4096
FOLDABLE_TYPES = (int, float, complex, str, bytes, bool)


class FoldRefused(Exception):
    """The expression must be left for the runtime (would raise, or is too large)."""


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.BitAnd: operator.and_,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Invert: operator.invert, ast.Not: operator.not_}
_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}


def _check_result(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value.bit_length() > MAX_FOLDED_INT_BITS:
        raise FoldRefused("integer result exceeds the folding size cap")
    if isinstance(value, (str, bytes)) and len(value) > MAX_FOLDED_LEN:
        raise FoldRefused("sequence result exceeds the folding size cap")
    if isinstance(value, float) and not math.isfinite(value):
        raise FoldRefused("non-finite float result")
    if isinstance(value, complex) and not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise FoldRefused("non-finite complex result")
    if not isinstance(value, FOLDABLE_TYPES):
        raise FoldRefused("unsupported result type")
    return value


def _require_foldable(*values):
    for v in values:
        if not isinstance(v, FOLDABLE_TYPES):
            raise FoldRefused("operand is not a foldable literal")


def fold_binop(op: ast.operator, left, right):
    _require_foldable(left, right)
    if isinstance(op, ast.Mult):
        for seq, count in ((left, right), (right, left)):
            if isinstance(seq, (str, bytes)) and isinstance(count, int) and len(seq) * max(count, 0) > MAX_FOLDED_LEN:
                raise FoldRefused("sequence repetition exceeds the folding size cap")
    elif isinstance(op, ast.Pow) and isinstance(left, int) and isinstance(right, int):
        if right > 0 and abs(left) > 1 and right * (abs(left).bit_length() - 1) > MAX_FOLDED_INT_BITS:
            raise FoldRefused("power exceeds the folding size cap")
    elif isinstance(op, ast.LShift) and isinstance(left, int) and isinstance(right, int):
        if left != 0 and right + left.bit_length() > MAX_FOLDED_INT_BITS:
            raise FoldRefused("shift exceeds the folding size cap")
    elif isinstance(op, ast.Mod) and isinstance(left, (str, bytes)):
        raise FoldRefused("printf-style formatting is not folded")
    fn = _BINOPS.get(type(op))
    if fn is None:
        raise FoldRefused("unsupported operator")
    try:
        result = fn(left, right)
    except Exception as exc:  # ZeroDivisionError, TypeError, OverflowError, ...
        raise FoldRefused(f"evaluation raises {type(exc).__name__}") from None
    return _check_result(result)


def fold_unary(op: ast.unaryop, operand):
    _require_foldable(operand)
    try:
        return _check_result(_UNARY[type(op)](operand))
    except FoldRefused:
        raise
    except Exception as exc:
        raise FoldRefused(f"evaluation raises {type(exc).__name__}") from None


def fold_compare(left, ops, comparators):
    _require_foldable(left, *comparators)
    cur = left
    for op, right in zip(ops, comparators):
        fn = _COMPARE.get(type(op))
        if fn is None:
            raise FoldRefused("identity comparisons are not folded")
        try:
            ok = fn(cur, right)
        except Exception as exc:
            raise FoldRefused(f"evaluation raises {type(exc).__name__}") from None
        if not ok:
            return False
        cur = right
    return True


# ---------------------------------------------------------------------------
# Lightweight numeric type inference (just enough to justify identities)
# ---------------------------------------------------------------------------
_INT_PRESERVING = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.LShift,
                   ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor)


def numeric_kind(expr: ast.AST, int_names: Set[str], shadowed: Set[str]) -> Optional[str]:
    """Return ``"int"``/``"float"`` if the expression provably has that type."""
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, bool):
            return None
        if isinstance(expr.value, int):
            return "int"
        if isinstance(expr.value, float):
            return "float"
        return None
    if isinstance(expr, ast.Name):
        return "int" if expr.id in int_names else None
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub, ast.Invert)):
        kind = numeric_kind(expr.operand, int_names, shadowed)
        return kind if not (isinstance(expr.op, ast.Invert) and kind != "int") else None
    if isinstance(expr, ast.BinOp):
        lk = numeric_kind(expr.left, int_names, shadowed)
        rk = numeric_kind(expr.right, int_names, shadowed)
        if lk is None or rk is None:
            return None
        if lk == rk == "int" and isinstance(expr.op, _INT_PRESERVING):
            return "int"
        if isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            return "float"
        return None
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and len(expr.args) == 1
            and not expr.keywords and expr.func.id not in shadowed):
        return {"len": "int", "int": "int", "float": "float"}.get(expr.func.id)
    return None


def is_pure(expr: ast.AST) -> bool:
    """No calls or other operations that could run user code or bind names."""
    for n in ast.walk(expr):
        if isinstance(n, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Subscript, ast.Attribute)):
            return False
    return True


def range_int_names(fn: ast.AST, shadowed: Set[str]) -> Set[str]:
    """Names whose every binding is ``for <name> in range(...)`` (hence always ints)."""
    if "range" in shadowed:
        return set()
    stores: Dict[str, int] = {}
    range_bindings: Dict[str, int] = {}
    for n in walk_scope(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            stores[n.id] = stores.get(n.id, 0) + 1
        elif (isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.target, ast.Name)
              and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range"):
            range_bindings[n.target.id] = range_bindings.get(n.target.id, 0) + 1
    params = set()
    if isinstance(fn, FUNC_NODES):
        a = fn.args
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [v for v in (a.vararg, a.kwarg) if v]}
    return {name for name, count in range_bindings.items() if stores.get(name) == count and name not in params}


# ---------------------------------------------------------------------------
# The folding / simplification pass
# ---------------------------------------------------------------------------
def _const(value, like: ast.AST) -> ast.Constant:
    return ast.copy_location(ast.Constant(value=value), like)


class ConstantFolder(ast.NodeTransformer):
    """Constant folding + provably-safe algebraic simplification."""

    def __init__(self, log: Optional[List[Dict]] = None, shadowed: Optional[Set[str]] = None):
        self.log = log if log is not None else []
        self.shadowed = shadowed or set()
        self.changed = False
        self._int_names: List[Set[str]] = [set()]

    def _record(self, pass_name: str, old: ast.AST, new: ast.AST):
        self.changed = True
        self.log.append({"pass": pass_name, "line": getattr(old, "lineno", None),
                         "before": short_source(old, 80), "after": short_source(new, 80)})
        return new

    def _refused(self, node: ast.AST, reason: str):
        if "size cap" in reason and not getattr(node, "_fold_guard_logged", False):
            node._fold_guard_logged = True
            self.log.append({"pass": "folding-guard", "line": getattr(node, "lineno", None),
                             "before": short_source(node, 80), "after": f"not folded: {reason}"})

    def visit_FunctionDef(self, node):
        self._int_names.append(range_int_names(node, self.shadowed))
        self.generic_visit(node)
        self._int_names.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._int_names.append(set())
        self.generic_visit(node)
        self._int_names.pop()
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            try:
                value = fold_binop(node.op, node.left.value, node.right.value)
                return self._record("constant-folding", node, _const(value, node))
            except FoldRefused as exc:
                self._refused(node, str(exc))
                return node
        return self._simplify(node)

    def _simplify(self, node: ast.BinOp) -> ast.AST:
        ints = self._int_names[-1]

        def is_int(e):
            return numeric_kind(e, ints, self.shadowed) == "int"

        def lit(e, value):
            return isinstance(e, ast.Constant) and type(e.value) is int and e.value == value

        op, left, right = node.op, node.left, node.right
        identity_right = {ast.Add: 0, ast.Sub: 0, ast.Mult: 1, ast.FloorDiv: 1, ast.Pow: 1,
                          ast.BitOr: 0, ast.BitXor: 0, ast.LShift: 0, ast.RShift: 0}
        identity_left = {ast.Add: 0, ast.Mult: 1, ast.BitOr: 0, ast.BitXor: 0}
        t = type(op)
        if t in identity_right and lit(right, identity_right[t]) and is_int(left):
            return self._record("algebraic-simplification", node, left)
        if t in identity_left and lit(left, identity_left[t]) and is_int(right):
            return self._record("algebraic-simplification", node, right)
        if t in (ast.Mult, ast.BitAnd):
            for zero, other in ((right, left), (left, right)):
                if lit(zero, 0) and is_int(other) and is_pure(other):
                    return self._record("algebraic-simplification", node, _const(0, node))
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.operand, ast.Constant):
            try:
                return self._record("constant-folding", node, _const(fold_unary(node.op, node.operand.value), node))
            except FoldRefused:
                return node
        return node

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        parts = [node.left] + node.comparators
        if all(isinstance(p, ast.Constant) for p in parts):
            try:
                value = fold_compare(node.left.value, node.ops, [c.value for c in node.comparators])
                return self._record("constant-folding", node, _const(value, node))
            except FoldRefused:
                return node
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        values = list(node.values)
        is_and = isinstance(node.op, ast.And)
        while len(values) > 1 and isinstance(values[0], ast.Constant):
            truthy = bool(values[0].value)
            if truthy != is_and:  # ``falsy and ...`` / ``truthy or ...`` short-circuits
                return self._record("constant-folding", node, values[0])
            values.pop(0)
        if len(values) == 1:
            return self._record("constant-folding", node, values[0])
        if len(values) != len(node.values):
            new = ast.copy_location(ast.BoolOp(op=node.op, values=values), node)
            return self._record("constant-folding", node, new)
        return node

    def visit_IfExp(self, node: ast.IfExp):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return self._record("constant-folding", node, node.body if node.test.value else node.orelse)
        return node


# Backwards-compatible name used by earlier versions of the project.
ASTOptimizer = ConstantFolder


def optimize_source(source_code: str) -> str:
    """Parse, run every optimization pass, and unparse back to code."""
    from .optimizer import optimize_tree  # local import: optimizer depends on this module

    return optimize_tree(ast.parse(source_code)).code
