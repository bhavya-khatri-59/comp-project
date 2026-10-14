import ast
import time

from compiler.optimizer import optimize_tree
from testcases import EXAMPLES


def opt(src):
    return optimize_tree(ast.parse(src))


def run(src, fn, *args):
    ns = {}
    exec(compile(src, "<test>", "exec"), ns)
    return ns[fn](*args)


def test_identity_not_applied_to_unknown_types():
    code = opt("def f(s, g):\n    return (s * 1, s + 0, g() * 0)\n").code
    assert "s * 1" in code and "s + 0" in code and "g() * 0" in code


def test_identity_applied_to_proven_ints():
    code = opt("def f(n):\n    t = 0\n    for i in range(n):\n        t += i * 1 + 0\n    return t\n").code
    assert "i * 1" not in code and "+ 0" not in code


def test_folding_bomb_is_refused_quickly():
    start = time.perf_counter()
    result = opt('x = "A" * 10 ** 10\ny = 2 ** 10 ** 8\n')
    assert time.perf_counter() - start < 2
    assert any(e["pass"] == "folding-guard" for e in result.log)
    assert "10000000000" in result.code


def test_division_by_zero_is_left_for_runtime():
    assert "1 / 0" in opt("x = 1 / 0\n").code


def test_dce_keeps_generator_semantics():
    code = opt("def g():\n    if False:\n        yield 1\n    return 2\n").code
    assert "yield" in code


def test_dce_keeps_only_binding_of_a_read_name():
    src = "def f(c):\n    if c:\n        y = 1\n    return y\n"
    assert "y = 1" in opt(src).code


def test_for_target_is_not_a_definite_kill():
    src = "def f():\n    x = 1\n    for x in []:\n        pass\n    return x\n"
    out = opt(src).code
    assert run(src, "f") == run(out, "f") == 1


def test_globals_are_not_propagated():
    src = "x = 1\ndef bump():\n    global x\n    x = 2\nbump()\ny = x\n"
    out = opt(src).code
    ns1, ns2 = {}, {}
    exec(src, ns1)
    exec(out, ns2)
    assert ns1["y"] == ns2["y"] == 2


DIFFERENTIAL = [
    ("""
def f(items):
    debug = False
    scale = 60 * 60
    limit = scale * 2
    if debug:
        items = []
    total = 0
    for x in items:
        total += x * limit
    return total
""", "f", [([1, 2, 3],), ([],)]),
    ("""
def f(data, n):
    out = []
    for i in range(2, n, 3):
        if i * 5 > 40:
            continue
        out.append(data[i * 5 % len(data)] + i * 5)
    return out
""", "f", [(list(range(50)), 20), (list(range(7)), 30)]),
    ("""
def f(n):
    k = 3
    acc = 0
    i = 0
    while i < n:
        acc += k * i
        if acc > 100:
            break
        i += 1
    else:
        acc = -1
    return acc
""", "f", [(5,), (50,)]),
]


def test_differential_optimized_programs_behave_identically():
    for src, fn, inputs in DIFFERENTIAL:
        out = opt(src).code
        for args in inputs:
            assert run(src, fn, *args) == run(out, fn, *args), out


def test_examples_11_and_12_are_optimized_and_equivalent():
    src11 = EXAMPLES["11. Constant propagation + DCE"]["source"]
    out11 = opt(src11).code
    assert "unused" not in out11 and "debug" not in out11 and "7200" in out11
    assert run(src11, "test_constant_propagation_dce", [1, 2]) == run(out11, "test_constant_propagation_dce", [1, 2])

    src12 = EXAMPLES["12. Strength reduction"]["source"]
    out12 = opt(src12).code
    assert "i * 4" not in out12
    data = list(range(100))
    assert run(src12, "test_strength_reduction", data, 10) == run(out12, "test_strength_reduction", data, 10)
