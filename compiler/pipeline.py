"""End-to-end analysis pipeline used by the UI and by the patch verifier.

    source --parse--> AST --optimize--> optimized AST --CFG per scope-->
        leak detection (dataflow) + taint analysis (dataflow)   [optimized AST]
        security pattern rules                                   [original AST]

The analyzer processes untrusted input, so it enforces resource limits itself:
source size, AST size and depth, CFG size, and dataflow iteration budgets.
Code is never executed.
"""

import ast
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import networkx as nx

from .ast_utils import ImportResolver
from .cfg_builder import build_all_cfgs
from .dataflow import AnalysisLimitError
from .findings import Finding, sort_findings
from .leak_detector import LeakDetector
from .optimizer import ALL_PASSES, optimize_tree
from .security_rules import run_security_rules
from .taint import TaintAnalyzer

MAX_SOURCE_BYTES = 200_000
MAX_AST_NODES = 60_000
MAX_CFG_NODES = 5_000


class AnalysisError(Exception):
    """The input could not be analyzed (syntax error or a resource limit)."""

    def __init__(self, message: str, line: Optional[int] = None):
        super().__init__(message)
        self.line = line


@dataclass
class AnalysisOptions:
    passes: Sequence[str] = ALL_PASSES
    run_taint: bool = True
    run_rules: bool = True


@dataclass
class AnalysisResult:
    source: str
    tree: ast.Module
    optimized_tree: ast.Module
    optimized_code: str
    optimization_log: List[Dict]
    cfgs: Dict[str, nx.DiGraph]
    findings: List[Finding]
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def leak_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.category == "resource"]

    @property
    def security_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.category != "resource"]

    @property
    def leaks(self) -> List[Dict]:
        """Leak diagnostics in the original dict format (used by the LLM payload)."""
        return [{
            "variable": f.extra.get("variable"), "resource_type": f.extra.get("allocator"),
            "allocated_line": f.line, "execution_path": f.path, "message": f.message,
            "cwe": f.cwe, "function": f.function, "rule_id": f.rule_id,
        } for f in self.leak_findings]


def parse_source(source: str) -> ast.Module:
    if len(source.encode("utf-8", "replace")) > MAX_SOURCE_BYTES:
        raise AnalysisError(f"Input exceeds the {MAX_SOURCE_BYTES // 1000} KB analysis limit.")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AnalysisError(f"Syntax error: {exc.msg}", exc.lineno) from None
    except (RecursionError, MemoryError, ValueError) as exc:
        raise AnalysisError(f"Input rejected: {type(exc).__name__} while parsing (too deeply nested?)") from None
    count = sum(1 for _ in ast.walk(tree))
    if count > MAX_AST_NODES:
        raise AnalysisError(f"Input has {count} AST nodes; the limit is {MAX_AST_NODES}.")
    return tree


def analyze(source: str, options: Optional[AnalysisOptions] = None) -> AnalysisResult:
    options = options or AnalysisOptions()
    timings: Dict[str, float] = {}
    warnings: List[str] = []
    t0 = time.perf_counter()
    tree = parse_source(source)
    timings["parse"] = time.perf_counter() - t0

    try:
        t = time.perf_counter()
        try:
            opt = optimize_tree(tree, options.passes)
            opt_tree, opt_code, opt_log = opt.tree, opt.code, opt.log
        except AnalysisLimitError as exc:
            warnings.append(f"Optimizer stopped early: {exc}. Analyzing the unoptimized program.")
            opt_tree, opt_code, opt_log = tree, ast.unparse(tree), []
        timings["optimize"] = time.perf_counter() - t

        t = time.perf_counter()
        cfgs = build_all_cfgs(opt_tree)
        total = sum(g.number_of_nodes() for g in cfgs.values())
        if total > MAX_CFG_NODES:
            raise AnalysisError(f"Control-flow graphs have {total} blocks; the limit is {MAX_CFG_NODES}.")
        timings["cfg"] = time.perf_counter() - t

        resolver = ImportResolver(tree)
        findings: List[Finding] = []
        t = time.perf_counter()
        for name, cfg in cfgs.items():
            try:
                findings.extend(LeakDetector(cfg, resolver).analyze())
                if options.run_taint:
                    findings.extend(TaintAnalyzer(cfg, resolver).analyze())
            except AnalysisLimitError as exc:
                warnings.append(f"{name}: {exc}")
        timings["dataflow"] = time.perf_counter() - t

        if options.run_rules:
            t = time.perf_counter()
            # A syntactic match is redundant where the dataflow analysis already
            # proved the flow at the same location.
            proven = {(f.line, f.cwe) for f in findings if f.category == "taint"}
            findings.extend(f for f in run_security_rules(tree) if (f.line, f.cwe) not in proven)
            timings["rules"] = time.perf_counter() - t
    except RecursionError:
        raise AnalysisError("Input rejected: expression nesting too deep to analyze safely.") from None

    timings["total"] = time.perf_counter() - t0
    return AnalysisResult(
        source=source, tree=tree, optimized_tree=opt_tree, optimized_code=opt_code,
        optimization_log=opt_log, cfgs=cfgs, findings=sort_findings(findings),
        warnings=warnings, timings=timings,
    )
