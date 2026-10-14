"""Intraprocedural taint analysis (source -> sanitizer -> sink) over the CFG.

This reuses the same worklist engine as the leak detector: instead of tracking
"which handles are open" it tracks "which variables may hold attacker-controlled
data". A finding is reported when tainted data reaches a dangerous API (sink)
without passing through a sanitizer for that vulnerability class.

Each taint fact remembers where the data came from and which vulnerability
classes it has been sanitized for, e.g. ``shlex.quote(x)`` clears CWE-78 but
not CWE-89, while ``int(x)`` clears everything.

Data that comes from a *function parameter* is tainted with lower confidence:
whether it is attacker-controlled depends on the callers. Parameters of web
route handlers (``@app.route``, ``@app.get``, ...) are treated as real sources.
"""

import ast
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import networkx as nx

from .ast_utils import FUNC_NODES, ImportResolver, defined_names, evaluated_parts, name_matches, walk_scope
from .dataflow import solve_forward
from .findings import Finding

ALL_CLASSES = frozenset({"CWE-78", "CWE-89", "CWE-95", "CWE-22", "CWE-502", "CWE-918", "CWE-79", "CWE-601", "CWE-1333"})

# Attribute reads that yield attacker-controlled data.
SOURCE_ATTRIBUTES = [
    "sys.argv", "os.environ",
    "request.args", "request.form", "request.values", "request.json", "request.data", "request.cookies",
    "request.headers", "request.files", "request.GET", "request.POST", "request.body", "request.query_params",
]
# Calls whose return value is attacker-controlled.
SOURCE_CALLS = [
    "input", "raw_input", "sys.stdin.read", "sys.stdin.readline", "*.recv", "*.recvfrom", "*.recv_into",
    "os.getenv", "os.environ.get", "request.get_json", "request.get_data", "*.readline_from_client",
]
ROUTE_DECORATORS = ["*.route", "*.get", "*.post", "*.put", "*.delete", "*.patch", "api_view", "*.websocket"]

# Sanitizer -> vulnerability classes it neutralizes.
SANITIZERS: Dict[str, FrozenSet[str]] = {
    "int": ALL_CLASSES, "float": ALL_CLASSES, "bool": ALL_CLASSES, "len": ALL_CLASSES, "abs": ALL_CLASSES,
    "ord": ALL_CLASSES, "hash": ALL_CLASSES, "uuid.UUID": ALL_CLASSES,
    "shlex.quote": frozenset({"CWE-78"}), "pipes.quote": frozenset({"CWE-78"}),
    "os.path.basename": frozenset({"CWE-22"}), "secure_filename": frozenset({"CWE-22"}),
    "werkzeug.utils.secure_filename": frozenset({"CWE-22"}),
    "html.escape": frozenset({"CWE-79"}), "markupsafe.escape": frozenset({"CWE-79"}), "escape": frozenset({"CWE-79"}),
    "bleach.clean": frozenset({"CWE-79"}), "re.escape": frozenset({"CWE-1333"}),
    "url_has_allowed_host_and_scheme": frozenset({"CWE-601", "CWE-918"}),
}


@dataclass(frozen=True)
class SinkSpec:
    rule_id: str
    cwe: str
    title: str
    severity: str
    patterns: Tuple[str, ...]
    args: Tuple[int, ...] = (0,)
    keywords: Tuple[str, ...] = ()
    require_shell: bool = False
    hint: str = ""


SINKS: List[SinkSpec] = [
    SinkSpec("CMD-INJECTION", "CWE-78", "OS command injection", "critical",
             ("os.system", "os.popen", "commands.getoutput", "subprocess.getoutput", "subprocess.getstatusoutput"),
             keywords=("cmd", "command"),
             hint="Pass an argument list to subprocess.run([...]) without shell=True, or quote with shlex.quote()."),
    SinkSpec("CMD-INJECTION", "CWE-78", "OS command injection (shell=True)", "critical",
             ("subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call"),
             keywords=("args",), require_shell=True,
             hint="Drop shell=True and pass an argument list."),
    SinkSpec("SQL-INJECTION", "CWE-89", "SQL injection", "critical",
             ("*.execute", "*.executemany", "*.executescript", "*.raw", "*.mogrify"), keywords=("sql", "query", "operation"),
             hint="Use a parameterized query: cursor.execute('... WHERE x = ?', (value,))."),
    SinkSpec("CODE-INJECTION", "CWE-95", "Code injection", "critical",
             ("eval", "exec", "compile", "execfile", "builtins.eval", "builtins.exec"), keywords=("source",),
             hint="Never evaluate untrusted input; use ast.literal_eval() or a dedicated parser."),
    SinkSpec("PATH-TRAVERSAL", "CWE-22", "Path traversal", "high",
             ("open", "io.open", "os.remove", "os.unlink", "os.rmdir", "os.rename", "shutil.rmtree", "shutil.copy",
              "shutil.copyfile", "shutil.move", "send_file", "flask.send_file", "send_from_directory_unsafe"),
             keywords=("file", "path", "filename"),
             hint="Normalize with os.path.realpath() and check the result stays inside an allow-listed base directory."),
    SinkSpec("UNSAFE-DESERIALIZATION", "CWE-502", "Deserialization of untrusted data", "critical",
             ("pickle.loads", "pickle.load", "cPickle.loads", "marshal.loads", "dill.loads", "jsonpickle.decode",
              "yaml.unsafe_load", "yaml.load", "shelve.open"), keywords=("data",),
             hint="Use a data-only format such as json.loads() or yaml.safe_load()."),
    SinkSpec("SSRF", "CWE-918", "Server-side request forgery", "high",
             ("requests.get", "requests.post", "requests.put", "requests.delete", "requests.head", "requests.patch",
              "urllib.request.urlopen", "urlopen", "httpx.get", "httpx.post", "urllib.request.Request"),
             keywords=("url",),
             hint="Validate the URL's scheme and host against an allow-list before requesting it."),
    SinkSpec("SSRF", "CWE-918", "Server-side request forgery", "high", ("requests.request",), args=(1,), keywords=("url",),
             hint="Validate the URL's scheme and host against an allow-list before requesting it."),
    SinkSpec("XSS", "CWE-79", "Cross-site scripting", "high",
             ("render_template_string", "flask.render_template_string", "Markup", "markupsafe.Markup", "mark_safe"),
             hint="Render with a template that auto-escapes, or escape with html.escape()."),
    SinkSpec("OPEN-REDIRECT", "CWE-601", "Open redirect", "medium",
             ("redirect", "flask.redirect", "HttpResponseRedirect"), keywords=("location",),
             hint="Only redirect to relative paths or allow-listed hosts."),
    SinkSpec("REGEX-INJECTION", "CWE-1333", "Regular expression injection (ReDoS)", "medium",
             ("re.compile", "re.match", "re.search", "re.fullmatch", "re.findall", "re.sub", "re.split"),
             keywords=("pattern",), hint="Escape user input with re.escape() before building a pattern."),
]

MUTATING_METHODS = {"append", "extend", "insert", "add", "update", "setdefault", "write", "writelines", "put"}


@dataclass(frozen=True)
class Fact:
    source: str  # human readable description of the source
    line: int
    origin: str  # "source" (real untrusted input) | "parameter" (depends on callers)
    cleared: FrozenSet[str] = frozenset()


Taint = FrozenSet[Fact]
State = Dict[str, Taint]
EMPTY: Taint = frozenset()


class TaintAnalyzer:
    def __init__(self, cfg: nx.DiGraph, resolver: Optional[ImportResolver] = None):
        self.cfg = cfg
        self.scope = cfg.graph.get("scope")
        self.function = cfg.graph.get("name", "<module>")
        self.resolver = resolver or ImportResolver(self.scope if self.scope is not None else ast.Module(body=[], type_ignores=[]))

    # ----------------------------------------------------------- expressions
    def _resolve(self, expr) -> Optional[str]:
        return self.resolver.resolve(expr)

    def expr_taint(self, expr: Optional[ast.AST], state: State) -> Taint:
        if expr is None or isinstance(expr, ast.Constant):
            return EMPTY
        if isinstance(expr, ast.Name):
            return state.get(expr.id, EMPTY)
        if isinstance(expr, ast.Attribute):
            name = self._resolve(expr)
            if name_matches(name, SOURCE_ATTRIBUTES):
                return frozenset({Fact(name, expr.lineno, "source")})
            return self.expr_taint(expr.value, state)
        if isinstance(expr, ast.Subscript):
            return self.expr_taint(expr.value, state)
        if isinstance(expr, ast.Call):
            name = self._resolve(expr.func)
            if name_matches(name, SOURCE_CALLS):
                return frozenset({Fact(f"{name}()", expr.lineno, "source")})
            arg_taint = EMPTY
            for a in expr.args:
                arg_taint |= self.expr_taint(a, state)
            for k in expr.keywords:
                arg_taint |= self.expr_taint(k.value, state)
            cleared = frozenset().union(*(c for p, c in SANITIZERS.items() if name_matches(name, [p])))
            if cleared:
                return frozenset(Fact(f.source, f.line, f.origin, f.cleared | cleared) for f in arg_taint)
            receiver = self.expr_taint(expr.func.value, state) if isinstance(expr.func, ast.Attribute) else EMPTY
            return arg_taint | receiver  # unknown call: result derives from its inputs
        if isinstance(expr, ast.Compare):
            return EMPTY  # booleans carry no attacker-controlled payload
        if isinstance(expr, (ast.Lambda,)):
            return EMPTY
        out = EMPTY
        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.AST) and not isinstance(child, (ast.expr_context, ast.operator, ast.boolop,
                                                                      ast.unaryop, ast.cmpop)):
                out |= self.expr_taint(child, state)
        return out

    # -------------------------------------------------------------- transfer
    def _assign(self, target: ast.AST, taint: Taint, state: Dict[str, Taint], strong: bool = True):
        if isinstance(target, ast.Name):
            state[target.id] = taint if strong else state.get(target.id, EMPTY) | taint
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign(elt.value if isinstance(elt, ast.Starred) else elt, taint, state, strong)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            base = target.value
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if isinstance(base, ast.Name) and taint:
                state[base.id] = state.get(base.id, EMPTY) | taint  # weak update of the container

    def transfer_item(self, item: ast.AST, state: State) -> State:
        new = dict(state)
        for name in defined_names(item):
            new.pop(name, None)
        if isinstance(item, ast.Assign):
            t = self.expr_taint(item.value, state)
            for target in item.targets:
                self._assign(target, t, new)
        elif isinstance(item, ast.AnnAssign) and item.value is not None:
            self._assign(item.target, self.expr_taint(item.value, state), new)
        elif isinstance(item, ast.AugAssign):
            t = self.expr_taint(item.value, state) | self.expr_taint(item.target, state)
            self._assign(item.target, t, new)
        elif isinstance(item, (ast.For, ast.AsyncFor)):
            self._assign(item.target, self.expr_taint(item.iter, state), new)
        elif isinstance(item, (ast.With, ast.AsyncWith)):
            for wi in item.items:
                if wi.optional_vars is not None:
                    self._assign(wi.optional_vars, self.expr_taint(wi.context_expr, state), new)
        for part in evaluated_parts(item):
            for n in walk_scope(part):
                if isinstance(n, ast.NamedExpr):
                    self._assign(n.target, self.expr_taint(n.value, state), new)
                elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr in MUTATING_METHODS and isinstance(n.func.value, ast.Name)):
                    t = EMPTY
                    for a in n.args:
                        t |= self.expr_taint(a, state)
                    if t:
                        new[n.func.value.id] = new.get(n.func.value.id, EMPTY) | t
        return {k: v for k, v in new.items() if v}

    @staticmethod
    def _meet(states: Iterable[State]) -> State:
        out: Dict[str, Taint] = {}
        for s in states:
            for k, v in s.items():
                out[k] = out.get(k, EMPTY) | v
        return out

    def _entry_state(self) -> State:
        state: State = {}
        if isinstance(self.scope, FUNC_NODES):
            a = self.scope.args
            is_route = any(name_matches(self._resolve(d.func if isinstance(d, ast.Call) else d), ROUTE_DECORATORS)
                           for d in self.scope.decorator_list)
            for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
                if arg.arg in ("self", "cls"):
                    continue
                origin = "source" if is_route else "parameter"
                desc = f"route parameter '{arg.arg}'" if is_route else f"parameter '{arg.arg}'"
                state[arg.arg] = frozenset({Fact(desc, self.scope.lineno, origin)})
        return state

    # ----------------------------------------------------------------- sinks
    def _sink_args(self, call: ast.Call, spec: SinkSpec) -> List[ast.AST]:
        exprs = [call.args[i] for i in spec.args if i < len(call.args) and not isinstance(call.args[i], ast.Starred)]
        exprs += [k.value for k in call.keywords if k.arg in spec.keywords]
        return exprs

    @staticmethod
    def _shell_true(call: ast.Call) -> bool:
        return any(k.arg == "shell" and not (isinstance(k.value, ast.Constant) and not k.value.value)
                   for k in call.keywords)

    def _check_sinks(self, node: int, item: ast.AST, state: State, out: List[Finding], seen: set):
        for part in evaluated_parts(item):
            for call in walk_scope(part):
                if not isinstance(call, ast.Call):
                    continue
                name = self._resolve(call.func)
                for spec in SINKS:
                    if not name_matches(name, spec.patterns):
                        continue
                    if spec.require_shell and not self._shell_true(call):
                        continue
                    if spec.rule_id == "UNSAFE-DESERIALIZATION" and name_matches(name, ["yaml.load"]) and any(
                            k.arg == "Loader" and "Safe" in (self._resolve(k.value) or "") for k in call.keywords):
                        continue
                    facts = EMPTY
                    for e in self._sink_args(call, spec):
                        facts |= self.expr_taint(e, state)
                    live = [f for f in facts if spec.cwe not in f.cleared]
                    if not live:
                        continue
                    real = [f for f in live if f.origin == "source"]
                    chosen = sorted(real or live, key=lambda f: (f.line, f.source))[0]
                    key = (spec.rule_id, call.lineno, call.col_offset)
                    if key in seen:
                        continue
                    seen.add(key)
                    confidence = "high" if real else "low"
                    out.append(Finding(
                        rule_id=spec.rule_id, title=spec.title, cwe=spec.cwe, severity=spec.severity,
                        confidence=confidence, category="taint",
                        message=(f"Untrusted data from {chosen.source} (line {chosen.line}) reaches "
                                 f"{name}() at line {call.lineno} without sanitization."
                                 + ("" if real else " Exploitable if callers pass attacker-controlled values.")),
                        line=call.lineno, end_line=getattr(call, "end_lineno", call.lineno), col=call.col_offset,
                        function=self.function, source_line=chosen.line, fix_hint=spec.hint,
                        extra={"sink": name, "source": chosen.source, "node": node},
                    ))

    def analyze(self) -> List[Finding]:
        IN, _ = solve_forward(self.cfg, self._entry_state(), self._transfer, self._meet)
        findings: List[Finding] = []
        seen: set = set()
        for n in sorted(IN):
            state = IN[n]
            for item in self.cfg.nodes[n]["block"].statements:
                self._check_sinks(n, item, state, findings, seen)
                state = self.transfer_item(item, state)
        for f in findings:
            f.path = self._path_to(f.extra["node"])
        return findings

    def _transfer(self, node: int, state: State) -> State:
        for item in self.cfg.nodes[node]["block"].statements:
            state = self.transfer_item(item, state)
        return state

    def _path_to(self, node: int) -> List[int]:
        try:
            return nx.shortest_path(self.cfg, self.cfg.graph["entry"], node)
        except nx.NetworkXNoPath:
            return [node]
