import json

from compiler import analyze, to_sarif
from compiler.pipeline import AnalysisError


def rules(src, min_conf=("high", "medium")):
    return {f.rule_id for f in analyze(src).findings if f.confidence in min_conf}


def test_sanitizers_are_vulnerability_specific():
    src = """
def f(cur):
    name = shlex.quote(input())
    cur.execute("SELECT * FROM t WHERE n = '" + name + "'")
"""
    assert "SQL-INJECTION" in rules(src)


def test_import_aliases_are_resolved():
    src = """
import subprocess as sp
from os import system as run_cmd
def f():
    cmd = input()
    sp.run(cmd, shell=True)
    run_cmd(cmd)
"""
    findings = [f for f in analyze(src).findings if f.rule_id == "CMD-INJECTION"]
    assert len(findings) == 2


def test_subprocess_argument_list_is_not_command_injection():
    assert "CMD-INJECTION" not in rules("def f():\n    subprocess.run(['ping', input()])\n")


def test_taint_through_containers_and_loops():
    src = """
def f():
    parts = []
    for line in sys.stdin.readline().split():
        parts.append(line)
    eval(" ".join(parts))
"""
    assert "CODE-INJECTION" in rules(src)


def test_parameters_are_low_confidence_sources():
    findings = analyze("def f(cmd):\n    os.system(cmd)\n").findings
    assert [(f.rule_id, f.confidence) for f in findings if f.category == "taint"] == [("CMD-INJECTION", "low")]


def test_reassignment_kills_taint():
    assert "CMD-INJECTION" not in rules("def f():\n    c = input()\n    c = 'ls'\n    os.system(c)\n")


def test_yaml_safe_loader_is_accepted():
    src = "def f():\n    yaml.load(input(), Loader=yaml.SafeLoader)\n"
    assert not rules(src)


def test_secret_rule_ignores_placeholders_and_env():
    src = 'password = os.environ["PW"]\napi_key = "changeme"\ntoken_url = "https://x/token"\n'
    assert "HARDCODED-SECRET" not in rules(src)


def test_sarif_export_is_valid():
    result = analyze("def f():\n    os.system(input())\n")
    sarif = json.loads(to_sarif(result.findings, "demo.py"))
    run = sarif["runs"][0]
    assert sarif["version"] == "2.1.0"
    assert run["results"][0]["ruleId"] == "CMD-INJECTION"
    assert run["results"][0]["locations"][0]["physicalLocation"]["region"]["startLine"] == 2


def test_oversized_input_rejected():
    try:
        analyze("x = 1\n" * 50_000)
    except AnalysisError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("expected AnalysisError")


def test_deeply_nested_input_rejected_gracefully():
    try:
        analyze("x = " + "(" * 500 + "1" + ")" * 500 + "\n")
    except AnalysisError:
        pass


def test_path_explosion_is_linear():
    import time
    body = "".join(f"    if c{i}:\n        x{i} = 1\n" for i in range(60))  # 2**60 paths
    src = "def f(" + ", ".join(f"c{i}" for i in range(60)) + "):\n    h = open('p')\n" + body + "    h.close()\n"
    start = time.perf_counter()
    result = analyze(src)
    assert time.perf_counter() - start < 10
    assert not result.leak_findings
