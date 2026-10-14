import ast

from compiler.cfg_builder import CFGBuilder


def cfg_of(src):
    fn = ast.parse(src).body[0]
    return CFGBuilder().build_cfg(fn)


def block_with(cfg, text):
    for n, d in cfg.nodes(data=True):
        if text in d["label"]:
            return n
    raise AssertionError(f"no block containing {text!r}: {[d['label'] for _, d in cfg.nodes(data=True)]}")


def test_break_jumps_to_loop_exit_and_continue_to_header():
    cfg = cfg_of("""
def f(xs):
    for x in xs:
        if x:
            break
        if not x:
            continue
        y = 1
    after = 2
""")
    brk, cont = block_with(cfg, "break"), block_with(cfg, "continue")
    header, after = block_with(cfg, "for x in xs"), block_with(cfg, "after = 2")
    assert list(cfg.successors(brk)) == [after]
    assert list(cfg.successors(cont)) == [header]


def test_while_true_without_break_never_reaches_following_code():
    cfg = cfg_of("""
def f():
    while True:
        work()
    unreachable()
""")
    labels = [d["label"] for _, d in cfg.nodes(data=True)]
    assert not any("unreachable" in l for l in labels)
    assert cfg.in_degree(cfg.graph["exit"]) == 0


def test_return_inside_try_runs_finally_then_exits():
    cfg = cfg_of("""
def f():
    try:
        return 1
    finally:
        cleanup()
    after()
""")
    ret = block_with(cfg, "return 1")
    normal = [s for s in cfg.successors(ret) if "exception" not in cfg.edges[ret, s].get("label", "")]
    (fin,) = normal
    assert "finally (return)" in cfg.nodes[fin]["label"]
    (cleanup,) = cfg.successors(fin)
    assert "cleanup()" in cfg.nodes[cleanup]["label"]
    assert cfg.graph["exit"] in cfg.successors(cleanup)
    assert not any("after()" in d["label"] for _, d in cfg.nodes(data=True))


def test_try_body_has_exception_edges_to_handlers():
    cfg = cfg_of("""
def f():
    try:
        a()
        b()
    except ValueError:
        handle()
""")
    handler = block_with(cfg, "except ValueError")
    for text in ("a()", "b()"):
        assert cfg.has_edge(block_with(cfg, text), handler)


def test_constant_condition_prunes_dead_branch():
    cfg = cfg_of("""
def f():
    if False:
        dead()
    live()
""")
    assert not any("dead()" in d["label"] for _, d in cfg.nodes(data=True))


def test_loop_else_skipped_by_break():
    cfg = cfg_of("""
def f(xs):
    for x in xs:
        if x:
            break
    else:
        not_found()
    done()
""")
    brk, nf, done = block_with(cfg, "break"), block_with(cfg, "not_found"), block_with(cfg, "done()")
    assert cfg.has_edge(brk, done)
    assert not cfg.has_edge(brk, nf)


def test_match_statement_cases():
    cfg = cfg_of("""
def f(cmd):
    match cmd:
        case "a":
            x = 1
        case _:
            x = 2
    return x
""")
    head = block_with(cfg, "match cmd")
    assert len(list(cfg.successors(head))) == 2  # irrefutable `case _` => no implicit fall-through


def test_blocks_carry_line_numbers():
    cfg = cfg_of("""
def f():
    x = 1
    y = 2
""")
    assert cfg.nodes[block_with(cfg, "y = 2")]["lineno"] == 4
