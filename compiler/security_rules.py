"""Syntactic security rules (AST pattern matching).

These complement the dataflow analyses with checks that need no flow
information: insecure configuration, weak cryptography, hard-coded secrets,
dangerous APIs. They run on the *original* tree, because a secret is a problem
as long as it is in the source code, even in dead code.
"""

import ast
import re
from typing import List, Optional

from .ast_utils import ImportResolver, name_matches
from .findings import Finding

SECRET_NAME = re.compile(r"(pass(wd|word)?|pwd|secret|token|api_?key|apikey|private_?key|access_?key|auth|credential)",
                         re.IGNORECASE)
RANDOM_SENSITIVE_NAME = re.compile(r"(token|password|passwd|secret|key|nonce|salt|otp|session|csrf|reset)", re.IGNORECASE)
PLACEHOLDER = re.compile(r"^(|x+|\*+|changeme|change_me|todo|none|null|example|placeholder|<.*>|\$\{.*\}|%\(.*\)s)$",
                         re.IGNORECASE)

WEAK_HASHES = ["hashlib.md5", "hashlib.sha1", "md5", "sha1"]
WEAK_CIPHERS = ["Crypto.Cipher.DES.new", "Crypto.Cipher.ARC4.new", "Crypto.Cipher.Blowfish.new", "DES.new", "ARC4.new"]
INSECURE_RANDOM = ["random.random", "random.randint", "random.choice", "random.randrange", "random.getrandbits",
                   "random.choices", "random.sample", "random.uniform"]
UNSAFE_DESERIALIZERS = ["pickle.loads", "pickle.load", "marshal.loads", "marshal.load", "dill.loads", "shelve.open",
                        "yaml.unsafe_load", "jsonpickle.decode"]
XML_PARSERS = ["xml.etree.ElementTree.parse", "xml.etree.ElementTree.fromstring", "ElementTree.parse",
               "ElementTree.fromstring", "ET.parse", "ET.fromstring", "xml.dom.minidom.parse",
               "xml.dom.minidom.parseString", "minidom.parseString", "xml.sax.parse", "lxml.etree.parse",
               "lxml.etree.fromstring"]
CLEARTEXT_PROTOCOLS = ["ftplib.FTP", "telnetlib.Telnet", "smtplib.SMTP"]


class _RuleVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.Module):
        self.resolver = ImportResolver(tree)
        self.findings: List[Finding] = []
        self.scope: List[str] = []

    # ----------------------------------------------------------- utilities
    def _fn(self) -> str:
        return ".".join(self.scope) or "<module>"

    def add(self, node, rule_id, title, cwe, severity, message, hint, confidence="high"):
        self.findings.append(Finding(
            rule_id=rule_id, title=title, cwe=cwe, severity=severity, confidence=confidence, category="pattern",
            message=message, line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
            col=node.col_offset, function=self._fn(), fix_hint=hint))

    def _name(self, expr) -> Optional[str]:
        return self.resolver.resolve(expr)

    @staticmethod
    def _kw(call: ast.Call, name: str) -> Optional[ast.AST]:
        return next((k.value for k in call.keywords if k.arg == name), None)

    @staticmethod
    def _is_const(expr, value) -> bool:
        return isinstance(expr, ast.Constant) and expr.value is value

    # -------------------------------------------------------------- scopes
    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    # ---------------------------------------------------------- assignments
    def _check_secret(self, node, target_name: str, value: ast.AST):
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            return
        text = value.value.strip()
        if len(text) < 4 or PLACEHOLDER.match(text) or not SECRET_NAME.search(target_name):
            return
        if re.search(r"(_url|_file|_path|_name|_field|_header|_type|_len|_length|prompt)$", target_name, re.IGNORECASE):
            return
        self.add(node, "HARDCODED-SECRET", "Hard-coded credential", "CWE-798", "high",
                 f"'{target_name}' is assigned a hard-coded secret literal. Anyone with access to the source "
                 f"(or the repository history) can read it.",
                 "Load secrets from the environment or a secret manager, e.g. os.environ['API_KEY'].")

    def _check_random(self, node, target_name: str, value: ast.AST):
        for n in ast.walk(value):
            if isinstance(n, ast.Call) and name_matches(self._name(n.func), INSECURE_RANDOM) \
                    and RANDOM_SENSITIVE_NAME.search(target_name):
                self.add(node, "INSECURE-RANDOM", "Predictable random value for a security token", "CWE-330", "medium",
                         f"'{target_name}' is generated with the non-cryptographic 'random' module, whose output "
                         f"can be predicted.", "Use the 'secrets' module (secrets.token_urlsafe(), secrets.randbelow()).")
                return

    def visit_Assign(self, node: ast.Assign):
        for t in node.targets:
            name = t.id if isinstance(t, ast.Name) else (t.attr if isinstance(t, ast.Attribute) else None)
            if name:
                self._check_secret(node, name, node.value)
                self._check_random(node, name, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None and isinstance(node.target, ast.Name):
            self._check_secret(node, node.target.id, node.value)
            self._check_random(node, node.target.id, node.value)
        self.generic_visit(node)

    # --------------------------------------------------------------- calls
    def visit_Call(self, node: ast.Call):
        name = self._name(node.func) or ""

        for k in node.keywords:
            if k.arg and SECRET_NAME.search(k.arg):
                self._check_secret(node, k.arg, k.value)

        if name_matches(name, WEAK_HASHES) and not self._is_const(self._kw(node, "usedforsecurity"), False):
            self.add(node, "WEAK-HASH", "Weak hash algorithm", "CWE-328", "medium",
                     f"{name}() is cryptographically broken (practical collision attacks).",
                     "Use hashlib.sha256() (or bcrypt/argon2/hashlib.scrypt for passwords).")
        elif name_matches(name, ["hashlib.new"]) and node.args and isinstance(node.args[0], ast.Constant) \
                and str(node.args[0].value).lower() in ("md5", "sha1", "md4"):
            self.add(node, "WEAK-HASH", "Weak hash algorithm", "CWE-328", "medium",
                     f"hashlib.new('{node.args[0].value}') uses a broken hash algorithm.", "Use 'sha256' or stronger.")

        if name_matches(name, WEAK_CIPHERS):
            self.add(node, "WEAK-CIPHER", "Broken cipher", "CWE-327", "high",
                     f"{name}() uses an obsolete cipher.", "Use AES-GCM or ChaCha20-Poly1305 (e.g. cryptography's AESGCM).")
        for arg in list(node.args) + [k.value for k in node.keywords]:
            if isinstance(arg, ast.Attribute) and arg.attr == "MODE_ECB":
                self.add(node, "WEAK-CIPHER", "ECB block cipher mode", "CWE-327", "high",
                         "ECB mode leaks patterns of the plaintext.", "Use an authenticated mode such as GCM.")

        verify = self._kw(node, "verify")
        if verify is not None and self._is_const(verify, False):
            self.add(node, "TLS-VERIFY-DISABLED", "TLS certificate verification disabled", "CWE-295", "high",
                     f"{name or 'call'}(verify=False) accepts any certificate, enabling man-in-the-middle attacks.",
                     "Remove verify=False (or point verify= at a CA bundle).")
        if name_matches(name, ["ssl._create_unverified_context"]):
            self.add(node, "TLS-VERIFY-DISABLED", "TLS certificate verification disabled", "CWE-295", "high",
                     "ssl._create_unverified_context() disables certificate checks.", "Use ssl.create_default_context().")

        if name_matches(name, ["tempfile.mktemp", "os.tempnam", "os.tmpnam"]):
            self.add(node, "INSECURE-TEMPFILE", "Insecure temporary file", "CWE-377", "medium",
                     f"{name}() returns a name that an attacker can create first (race condition).",
                     "Use tempfile.NamedTemporaryFile() or tempfile.mkstemp().")

        if name_matches(name, ["*.run"]) and self._is_const(self._kw(node, "debug"), True):
            self.add(node, "DEBUG-ENABLED", "Debug mode enabled", "CWE-215", "medium",
                     "Running the application with debug=True exposes an interactive debugger / stack traces.",
                     "Disable debug in production; read it from configuration.")

        if name_matches(name, ["os.chmod", "os.fchmod", "os.mkdir", "os.makedirs"]):
            mode = node.args[1] if len(node.args) > 1 else self._kw(node, "mode")
            if isinstance(mode, ast.Constant) and isinstance(mode.value, int) and mode.value & 0o002:
                self.add(node, "WORLD-WRITABLE", "World-writable permissions", "CWE-732", "medium",
                         f"Mode {oct(mode.value)} lets any local user modify the file.", "Use 0o600 / 0o700 or 0o644.")

        if name_matches(name, ["subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_output",
                               "subprocess.check_call"]) and self._kw(node, "shell") is not None \
                and not self._is_const(self._kw(node, "shell"), False):
            self.add(node, "SHELL-TRUE", "Subprocess launched through the shell", "CWE-78", "low",
                     f"{name}(shell=True) interprets shell metacharacters; dangerous if any part of the command is "
                     f"user-controlled.", "Pass an argument list and drop shell=True.", confidence="medium")

        if name_matches(name, ["eval", "exec"]) and node.args and not isinstance(node.args[0], ast.Constant):
            self.add(node, "DYNAMIC-EVAL", "Dynamic code evaluation", "CWE-95", "medium",
                     f"{name}() executes a dynamically built string as code.",
                     "Use ast.literal_eval() for data, or explicit dispatch tables.", confidence="medium")

        if name_matches(name, UNSAFE_DESERIALIZERS) or (
                name_matches(name, ["yaml.load"]) and "Safe" not in (self._name(self._kw(node, "Loader")) or "")):
            self.add(node, "UNSAFE-DESERIALIZER-API", "Unsafe deserialization API", "CWE-502", "medium",
                     f"{name}() can execute arbitrary code when given crafted input.",
                     "Use json, or yaml.safe_load(); never unpickle data from an untrusted source.", confidence="medium")

        if name_matches(name, XML_PARSERS):
            self.add(node, "XXE", "XML parser vulnerable to entity expansion", "CWE-611", "low",
                     f"{name}() does not protect against XML bombs / external entities in all configurations.",
                     "Use the 'defusedxml' package.", confidence="medium")

        if name_matches(name, CLEARTEXT_PROTOCOLS):
            self.add(node, "CLEARTEXT-PROTOCOL", "Cleartext protocol", "CWE-319", "low",
                     f"{name} transmits data (and credentials) unencrypted.", "Use FTP_TLS / SMTP_SSL / SSH.",
                     confidence="medium")

        host = self._kw(node, "host")
        if isinstance(host, ast.Constant) and host.value == "0.0.0.0":
            self.add(node, "BIND-ALL", "Service bound to all interfaces", "CWE-605", "low",
                     "Listening on 0.0.0.0 exposes the service on every network interface.",
                     "Bind to 127.0.0.1 unless external access is intended.", confidence="medium")
        if name_matches(name, ["*.bind"]) and node.args and isinstance(node.args[0], ast.Tuple) \
                and node.args[0].elts and isinstance(node.args[0].elts[0], ast.Constant) \
                and node.args[0].elts[0].value in ("0.0.0.0", ""):
            self.add(node, "BIND-ALL", "Socket bound to all interfaces", "CWE-605", "low",
                     "Binding to 0.0.0.0 exposes the socket on every network interface.",
                     "Bind to 127.0.0.1 unless external access is intended.", confidence="medium")
        self.generic_visit(node)

    # ----------------------------------------------------------- exceptions
    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"))
        if broad and len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue)):
            self.add(node, "SWALLOWED-EXCEPTION", "Exception silently swallowed", "CWE-390", "low",
                     "A broad 'except' that does nothing hides failures, including security-relevant ones.",
                     "Catch specific exceptions and log or re-raise them.", confidence="medium")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        text = ast.unparse(node.test).lower()
        if any(w in text for w in ("auth", "admin", "permission", "is_staff", "logged_in", "role")):
            self.add(node, "ASSERT-AS-GUARD", "assert used for an access-control check", "CWE-617", "medium",
                     "assert statements are removed when Python runs with -O, disabling this check.",
                     "Use an explicit 'if not ...: raise PermissionError'.")
        self.generic_visit(node)


def run_security_rules(tree: ast.Module) -> List[Finding]:
    v = _RuleVisitor(tree)
    v.visit(tree)
    return v.findings
