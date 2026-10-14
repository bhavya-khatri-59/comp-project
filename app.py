import difflib
import html
import json
import os

import graphviz
import streamlit as st

from agent.patcher import DEFAULT_MODEL, generate_and_verify_patch, screen_for_prompt_injection, select_targets
from compiler import AnalysisError, AnalysisOptions, analyze, to_sarif
from compiler.findings import CWE_NAMES, SEVERITY_ORDER, risk_score
from compiler.optimizer import ALL_PASSES
from testcases import EXAMPLES

try:  # rich editor with gutter annotations and line markers
    from streamlit_ace import st_ace
except ImportError:  # pragma: no cover
    st_ace = None
try:  # clickable graph
    from streamlit_agraph import Config, Edge, Node, agraph
except ImportError:  # pragma: no cover
    agraph = None

st.set_page_config(page_title="Static Resource Leak & Security Analyzer", page_icon="🛡️", layout="wide")

SEVERITY_COLOR = {"critical": "#b91c1c", "high": "#dc2626", "medium": "#d97706", "low": "#2563eb", "info": "#6b7280"}
SEVERITY_ICON = {"critical": "🟥", "high": "🔴", "medium": "🟠", "low": "🔵", "info": "⚪"}

VIEW_FINDINGS, VIEW_CFG, VIEW_OPT, VIEW_FIX = ("🛡️ Findings", "🔀 Control-flow graph", "⚙️ Optimizer",
                                               "🤖 Verified auto-repair")
VIEWS = [VIEW_FINDINGS, VIEW_CFG, VIEW_OPT, VIEW_FIX]

ss = st.session_state
ss.setdefault("view", VIEW_FINDINGS)
FIRST_EXAMPLE = next(iter(EXAMPLES))
ss.setdefault("code", EXAMPLES[FIRST_EXAMPLE]["source"])
ss.setdefault("editor_version", 0)
ss.setdefault("result", None)
ss.setdefault("error", None)
ss.setdefault("selection", None)  # {"function", "nodes", "lines", "title"}
ss.setdefault("patch", None)
ss.setdefault("last_graph_click", None)


def load_example():
    ss.code = EXAMPLES[ss.example_pick]["source"]
    ss.editor_version += 1  # forces the editor component to reload its content
    ss.result, ss.error, ss.selection, ss.patch = None, None, None, None


def lines_of_nodes(cfg, nodes):
    lines = set()
    for n in nodes:
        if n in cfg.nodes:
            start, end = cfg.nodes[n].get("lineno"), cfg.nodes[n].get("end_lineno")
            if start:
                lines.update(range(start, (end or start) + 1))
    return lines


def select_finding(finding):
    cfg = ss.result.cfgs.get(finding.function)
    lines = lines_of_nodes(cfg, finding.path) if cfg is not None else set()
    lines.update(range(finding.line, (finding.end_line or finding.line) + 1))
    ss.selection = {"function": finding.function, "nodes": list(finding.path), "lines": lines,
                    "focus": finding.line, "title": f"{finding.title} ({finding.cwe}), line {finding.line}"}


def trace_finding(finding):
    """Button callback: runs before the rerun, so it may switch the view widget."""
    select_finding(finding)
    ss.view = VIEW_CFG


def render_code(source: str, highlight: set, focus=None, flagged=None) -> str:
    """Line-numbered code view. Input is HTML-escaped: analyzed code is untrusted (no XSS via the report)."""
    flagged = flagged or {}
    rows = []
    for i, line in enumerate(source.splitlines(), start=1):
        style = ""
        if i == focus:
            style = "background:rgba(220,38,38,.30);"
        elif i in highlight:
            style = "background:rgba(250,204,21,.28);"
        mark = flagged.get(i, "")
        rows.append(f'<div style="{style}padding:0 6px;white-space:pre;">'
                    f'<span style="opacity:.45;display:inline-block;width:3.2em;user-select:none;">{i}</span>'
                    f'<span style="display:inline-block;width:1.4em;">{mark}</span>{html.escape(line) or " "}</div>')
    return ('<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;line-height:1.5;'
            'border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:6px 0;overflow-x:auto;max-height:560px;'
            f'overflow-y:auto;">{"".join(rows)}</div>')


def visible_findings(result, show_low):
    return [f for f in result.findings if show_low or f.confidence != "low"]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Examples")
    st.selectbox("Load a test case", list(EXAMPLES), key="example_pick", on_change=load_example,
                 index=list(EXAMPLES).index(FIRST_EXAMPLE))
    st.header("Analysis options")
    passes = st.multiselect("Optimization passes", list(ALL_PASSES), default=list(ALL_PASSES))
    show_low = st.checkbox("Show low-confidence findings", value=False,
                           help="Flows that start at a function parameter: exploitable only if callers pass "
                                "attacker-controlled data.")
    static_graph = st.checkbox("Static Graphviz CFG (not clickable)", value=agraph is None)
    st.header("Auto-repair")
    max_attempts = st.slider("Max LLM attempts", 1, 5, 3)
    model = st.text_input("Groq model", value=DEFAULT_MODEL)
    st.caption("API key: " + ("✅ GROQ_API_KEY found" if os.environ.get("GROQ_API_KEY") else "⚠️ GROQ_API_KEY not set"))
    with st.expander("What is detected?"):
        st.markdown("\n".join(f"- **{k}** {v}" for k, v in sorted(CWE_NAMES.items(), key=lambda kv: int(kv[0][4:]))))

st.title("🛡️ Static Resource Leak & Security Analyzer")
st.caption("A deterministic compiler pipeline (AST optimizer → CFG → dataflow analyses) with an LLM patcher "
           "whose output is only accepted after static verification.")

# ---------------------------------------------------------------- editor
result = ss.result if ss.result is not None and ss.result.source == ss.code else None
selection = ss.selection if result is not None else None

col_editor, col_summary = st.columns([3, 2], gap="large")
with col_editor:
    st.subheader("1. Source code")
    annotations, markers = [], []
    if result is not None:
        for f in visible_findings(result, show_low):
            kind = "error" if SEVERITY_ORDER[f.severity] >= 3 else ("warning" if f.severity == "medium" else "info")
            annotations.append({"row": f.line - 1, "column": 0, "type": kind,
                                "text": f"[{f.severity}] {f.cwe} {f.title}: {f.message}"})
    if selection:
        for ln in sorted(selection["lines"]):
            markers.append({"startRow": ln - 1, "startCol": 0, "endRow": ln - 1, "endCol": 1,
                            "className": "ace_step", "type": "fullLine"})
    if st_ace is not None:
        new_code = st_ace(value=ss.code, language="python", theme="chrome", key=f"ace-{ss.editor_version}",
                          min_lines=20, max_lines=40, font_size=14, tab_size=4, show_gutter=True,
                          auto_update=True, annotations=annotations, markers=markers)
    else:
        new_code = st.text_area("Python code", value=ss.code, height=420, key=f"ta-{ss.editor_version}")
    if new_code is not None and new_code != ss.code:
        ss.code = new_code
        st.rerun()
    if ss.result is not None and result is None:
        st.info("The code changed since the last analysis. Run it again to refresh the results.")
    if st.button("🚀 Run compiler analysis", type="primary", use_container_width=True):
        try:
            ss.result = analyze(ss.code, AnalysisOptions(passes=tuple(passes)))
            ss.error = None
        except AnalysisError as exc:
            ss.result, ss.error = None, f"{exc}" + (f" (line {exc.line})" if exc.line else "")
        ss.selection, ss.patch, ss.last_graph_click = None, None, None
        st.rerun()
    if ss.error:
        st.error(f"❌ {ss.error}")

with col_summary:
    st.subheader("2. Summary")
    if result is None:
        st.markdown("Run the analysis to see diagnostics. Pick a test case in the sidebar to try the leak, "
                    "injection, optimizer and AI-patcher scenarios.")
    else:
        shown = visible_findings(result, show_low)
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk score", f"{risk_score(shown)}/100")
        m2.metric("Security findings", len([f for f in shown if f.category != "resource"]))
        m3.metric("Resource leaks", len([f for f in shown if f.category == "resource"]))
        counts = {s: sum(1 for f in shown if f.severity == s) for s in SEVERITY_ORDER}
        st.markdown(" · ".join(f"{SEVERITY_ICON[s]} {counts[s]} {s}" for s in SEVERITY_ORDER if counts[s]) or
                    "✅ No findings with medium or high confidence.")
        st.caption(f"Analyzed {len(result.cfgs)} scope(s) in {result.timings.get('total', 0) * 1000:.0f} ms · "
                   f"{len(result.optimization_log)} optimization(s) applied")
        for w in result.warnings:
            st.warning(w)
        if selection:
            st.success(f"Tracing: {selection['title']}. Highlighted lines: "
                       f"{', '.join(map(str, sorted(selection['lines']))) or '-'}")
            if st.button("Clear trace"):
                ss.selection, ss.last_graph_click = None, None
                st.rerun()

if result is None:
    st.stop()

# Only the active view is rendered (st.tabs renders hidden panels at zero width,
# which breaks the graph component's auto-fit), and "Trace" can switch views.
st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
view = ss.view

# ---------------------------------------------------------------- findings
if view == VIEW_FINDINGS:
    main = [f for f in result.findings if f.confidence != "low"]
    low = [f for f in result.findings if f.confidence == "low"]

    def show_finding(f, idx):
        header = (f"{SEVERITY_ICON[f.severity]} **{f.title}** · {f.cwe} · line {f.line} · `{f.function}` "
                  f"· {f.confidence} confidence")
        with st.expander(header, expanded=False):
            st.markdown(f.message)
            if f.cwe_name:
                st.caption(f"{f.cwe}: {f.cwe_name} · rule {f.rule_id} · {f.category}")
            if f.fix_hint:
                st.markdown(f"**Fix:** {f.fix_hint}")
            st.button("🔎 Trace in editor & CFG", key=f"trace-{idx}", on_click=trace_finding, args=(f,))

    if not main:
        st.success("✅ No resource leaks or security issues with medium/high confidence.")
    for i, f in enumerate(main):
        show_finding(f, i)
    if low:
        with st.expander(f"Potential issues if parameters are attacker-controlled ({len(low)})"):
            for i, f in enumerate(low):
                show_finding(f, 10_000 + i)

    st.divider()
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ SARIF report (GitHub code scanning)", to_sarif(result.findings, "input.py"),
                       file_name="analysis.sarif", mime="application/json", use_container_width=True)
    c2.download_button("⬇️ JSON report", json.dumps([f.to_dict() for f in result.findings], indent=2),
                       file_name="analysis.json", mime="application/json", use_container_width=True)

# ---------------------------------------------------------------- CFG
if view == VIEW_CFG:
    names = list(result.cfgs)
    default = names.index(selection["function"]) if selection and selection["function"] in names else 0
    fn = st.selectbox("Scope", names, index=default, key=f"scope-{bool(selection)}-{default}")
    cfg = result.cfgs[fn]
    path_nodes = set(selection["nodes"]) if selection and selection["function"] == fn else set()
    path_edges = set(zip(selection["nodes"], selection["nodes"][1:])) if path_nodes else set()
    finding_lines = {f.line for f in visible_findings(result, show_low) if f.function == fn}

    def node_color(n, data):
        if n in path_nodes:
            return "#fecaca", "#b91c1c"
        if data.get("kind") in ("entry", "exit"):
            return "#e5e7eb", "#374151"
        if data.get("lineno") in finding_lines:
            return "#fde68a", "#b45309"
        if data.get("kind") == "finally":
            return "#ddd6fe", "#6d28d9"
        return "#dbeafe", "#1d4ed8"

    gcol, ccol = st.columns([3, 2], gap="medium")
    with gcol:
        st.caption("Click a block to highlight its source lines. Red = traced path, amber = block with a finding, "
                   "violet = duplicated `finally` body, dashed = exception edge.")
        if static_graph or agraph is None:
            dot = graphviz.Digraph(node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica"})
            for n, data in cfg.nodes(data=True):
                fill, line = node_color(n, data)
                dot.node(str(n), data.get("label", str(n)), fillcolor=fill, color=line)
            for u, v, data in cfg.edges(data=True):
                on_path = (u, v) in path_edges
                dot.edge(str(u), str(v), label=data.get("label", ""), color="#b91c1c" if on_path else "#6b7280",
                         penwidth="2.2" if on_path else "1", style="dashed" if "exception" in data.get("label", "") else "solid")
            st.graphviz_chart(dot)
        else:
            nodes = []
            for n, data in cfg.nodes(data=True):
                fill, line = node_color(n, data)
                nodes.append(Node(id=str(n), label=data.get("label", str(n)), shape="box", color={"background": fill, "border": line},
                                  font={"face": "monospace", "size": 13, "color": "#111827"}, borderWidth=2 if n in path_nodes else 1,
                                  title=f"lines {data.get('lineno')}–{data.get('end_lineno')}" if data.get("lineno") else data.get("kind")))
            edges = []
            for u, v, data in cfg.edges(data=True):
                label = data.get("label", "")
                on_path = (u, v) in path_edges
                edges.append(Edge(source=str(u), target=str(v), label=label, color="#b91c1c" if on_path else "#9ca3af",
                                  width=3 if on_path else 1, dashes="exception" in label or label == "raise"))
            # Hierarchical physics only runs during stabilization; it is what makes vis.js fit the view.
            config = Config(width=600, height=640, directed=True, physics=True, hierarchical=True,
                            solver="hierarchicalRepulsion", stabilization=True, fit=True,
                            direction="UD", sortMethod="directed", levelSeparation=80, nodeSpacing=260,
                            shakeTowards="roots", interaction={"zoomView": True, "dragView": True, "navigationButtons": True})
            clicked = agraph(nodes=nodes, edges=edges, config=config)
            if clicked is not None and clicked != ss.last_graph_click and str(clicked).isdigit():
                ss.last_graph_click = clicked
                n = int(clicked)
                if n in cfg.nodes:
                    lines = lines_of_nodes(cfg, [n])
                    ss.selection = {"function": fn, "nodes": [n], "lines": lines,
                                    "focus": min(lines) if lines else None, "title": cfg.nodes[n]["label"]}
                    st.rerun()
    with ccol:
        flagged = {f.line: SEVERITY_ICON[f.severity] for f in visible_findings(result, show_low)}
        hl = selection["lines"] if selection else set()
        st.markdown(render_code(result.source, hl, selection.get("focus") if selection else None, flagged),
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- optimizer
if view == VIEW_OPT:
    o1, o2 = st.columns(2)
    with o1:
        st.markdown("**Original**")
        st.code(result.source, language="python", line_numbers=True)
    with o2:
        st.markdown("**Optimized** (what the analyses run on)")
        st.code(result.optimized_code, language="python", line_numbers=True)
    if result.optimization_log:
        by_pass = {}
        for e in result.optimization_log:
            by_pass[e["pass"]] = by_pass.get(e["pass"], 0) + 1
        st.markdown(" · ".join(f"**{k}**: {v}" for k, v in by_pass.items()))
        st.dataframe([{"pass": e["pass"], "line": e["line"], "before": e["before"], "after": e["after"]}
                      for e in result.optimization_log], use_container_width=True, hide_index=True)
    else:
        st.info("Nothing to optimize in this program.")

# ---------------------------------------------------------------- auto-repair
if view == VIEW_FIX:
    injection = screen_for_prompt_injection(result.source)
    if injection:
        st.warning("⚠️ **Possible prompt injection** in the source. These comments/strings look like instructions "
                   "aimed at an AI model. They are passed to the LLM strictly as data, and any patch is still "
                   "subject to every verification gate:\n\n" +
                   "\n".join(f"- line {h['line']}: `{h['text']}`" for h in injection))
    targets = select_targets(result)
    st.markdown(f"**{len(targets)}** finding(s) will be sent to the model. A patch is accepted only if it passes "
                "every blocking gate: syntax, analysis limits, zero leaks, targeted findings fixed, no new findings, "
                "same interface, same operations, no new capabilities, no new literals in dangerous calls. "
                "Patches are **never executed**.")
    if st.button("🤖 Generate & verify patch", disabled=not targets, use_container_width=True):
        with st.spinner("Querying the model and verifying candidates…"):
            ss.patch = generate_and_verify_patch(result.source, result, max_attempts=max_attempts, model=model)
    patch = ss.patch
    if patch is not None:
        (st.success if patch.verified else st.error)(("✅ " if patch.verified else "❌ ") + patch.message)
        for att in patch.attempts:
            with st.expander(f"Attempt {att.number}: {'passed' if att.passed else 'rejected'}", expanded=not patch.verified):
                if att.error:
                    st.error(att.error)
                for g in att.gates:
                    icon = "✅" if g.passed else ("❌" if g.blocking else "⚠️")
                    st.markdown(f"{icon} **{g.name}**: {g.detail}")
        if patch.attempts and patch.attempts[-1].code:
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Patched code**" + ("" if patch.verified else " (rejected, shown for inspection)"))
                st.code(patch.code, language="python")
            with p2:
                st.markdown("**Diff**")
                diff = "".join(difflib.unified_diff(result.source.splitlines(True), patch.code.splitlines(True),
                                                    "original.py", "patched.py"))
                st.code(diff or "(no changes)", language="diff")
