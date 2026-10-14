import pytest

from compiler import analyze
from testcases import EXAMPLES


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_findings(name):
    ex = EXAMPLES[name]
    result = analyze(ex["source"])
    got = {f.rule_id for f in result.findings if f.confidence != "low"}
    assert got == ex["expect"]
    passes = {e["pass"] for e in result.optimization_log}
    assert ex["expect_opt"] <= passes


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_findings_map_to_source_lines(name):
    """Every finding and every CFG block points at a real line of the user's input."""
    ex = EXAMPLES[name]
    result = analyze(ex["source"])
    n_lines = len(ex["source"].splitlines())
    for f in result.findings:
        assert 1 <= f.line <= n_lines
    for cfg in result.cfgs.values():
        for _, data in cfg.nodes(data=True):
            if data.get("lineno") is not None:
                assert 1 <= data["lineno"] <= n_lines
