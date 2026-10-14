"""Common finding model, severity ordering and SARIF 2.1.0 export."""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
SARIF_LEVEL = {"critical": "error", "high": "error", "medium": "warning", "low": "note", "info": "note"}

CWE_NAMES = {
    "CWE-22": "Path Traversal",
    "CWE-78": "OS Command Injection",
    "CWE-79": "Cross-site Scripting",
    "CWE-89": "SQL Injection",
    "CWE-95": "Code Injection (eval)",
    "CWE-215": "Debug Information Exposure",
    "CWE-295": "Improper Certificate Validation",
    "CWE-327": "Broken or Risky Cryptographic Algorithm",
    "CWE-328": "Weak Hash",
    "CWE-330": "Insufficiently Random Values",
    "CWE-377": "Insecure Temporary File",
    "CWE-390": "Error Swallowed Without Action",
    "CWE-400": "Uncontrolled Resource Consumption",
    "CWE-404": "Improper Resource Shutdown or Release",
    "CWE-502": "Deserialization of Untrusted Data",
    "CWE-601": "Open Redirect",
    "CWE-319": "Cleartext Transmission",
    "CWE-605": "Binding to All Interfaces",
    "CWE-611": "XML External Entity",
    "CWE-617": "Reachable Assertion",
    "CWE-667": "Improper Locking",
    "CWE-732": "Incorrect Permission Assignment",
    "CWE-772": "Missing Release of Resource after Effective Lifetime",
    "CWE-798": "Hard-coded Credentials",
    "CWE-918": "Server-Side Request Forgery",
    "CWE-1333": "Inefficient Regular Expression (ReDoS)",
}


@dataclass
class Finding:
    rule_id: str
    title: str
    cwe: str
    severity: str  # critical | high | medium | low | info
    confidence: str  # high | medium | low
    category: str  # resource | taint | pattern
    message: str
    line: int
    function: str = "<module>"
    end_line: Optional[int] = None
    col: int = 0
    path: List[int] = field(default_factory=list)  # CFG node ids (in ``function``'s CFG)
    source_line: Optional[int] = None  # taint: where the untrusted data entered
    fix_hint: str = ""
    extra: Dict = field(default_factory=dict)

    @property
    def cwe_name(self) -> str:
        return CWE_NAMES.get(self.cwe, "")

    @property
    def key(self):
        """Line-independent identity, used to compare findings across program versions."""
        return (self.rule_id, self.function, self.extra.get("variable", ""))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["cwe_name"] = self.cwe_name
        return d


def sort_findings(findings: List[Finding]) -> List[Finding]:
    return sorted(findings, key=lambda f: (-SEVERITY_ORDER.get(f.severity, 0), f.line, f.rule_id))


def risk_score(findings: List[Finding]) -> int:
    """Simple 0-100 score: weighted by severity, discounted by confidence."""
    weights = {"critical": 40, "high": 20, "medium": 8, "low": 2, "info": 0}
    conf = {"high": 1.0, "medium": 0.6, "low": 0.25}
    total = sum(weights.get(f.severity, 0) * conf.get(f.confidence, 0.5) for f in findings)
    return min(100, round(total))


def to_sarif(findings: List[Finding], file_name: str = "input.py", tool_version: str = "2.0.0") -> str:
    rules: Dict[str, Dict] = {}
    results = []
    for f in findings:
        if f.rule_id not in rules:
            rules[f.rule_id] = {
                "id": f.rule_id,
                "name": f.title,
                "shortDescription": {"text": f.title},
                "properties": {"tags": ["security", f.cwe], "precision": f.confidence},
                "helpUri": f"https://cwe.mitre.org/data/definitions/{f.cwe.split('-')[-1]}.html" if f.cwe else None,
            }
        results.append({
            "ruleId": f.rule_id,
            "level": SARIF_LEVEL.get(f.severity, "warning"),
            "message": {"text": f.message},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": file_name},
                    "region": {"startLine": f.line, "endLine": f.end_line or f.line, "startColumn": f.col + 1},
                },
                "logicalLocations": [{"name": f.function, "kind": "function"}],
            }],
            "properties": {"severity": f.severity, "confidence": f.confidence, "cwe": f.cwe},
        })
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "Static Resource Leak & Security Analyzer", "version": tool_version,
                                "rules": list(rules.values())}},
            "results": results,
        }],
    }
    return json.dumps(sarif, indent=2)
