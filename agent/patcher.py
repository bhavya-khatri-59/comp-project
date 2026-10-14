import ast
import json
import os
import re
from typing import Any, Dict, List, Tuple
from groq import Groq
from compiler.ast_analyzer import optimize_source
from compiler.cfg_builder import CFGBuilder
from compiler.leak_detector import LeakDetector
from dotenv import load_dotenv

load_dotenv()

client = Groq()


def _clean_llm_code_response(raw_response: str) -> str:
    """Extracts raw Python code from markdown blocks or unstructured LLM output."""
    raw_response = raw_response.strip()

    # Match code inside ```python ... ``` or ``` ... ``` blocks
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Fallback if no markdown fences were used
    return raw_response


def generate_and_verify_patch(
    source_code: str, leaks: List[Dict[str, Any]]
) -> Tuple[str, bool, str]:
    """Sends compiler diagnostics to Groq Llama to generate a refactored patch,

    then re-runs the compiler pipeline to verify 0 remaining leaks.

    Returns:
        Tuple[str, bool, str]: (refactored_code, is_verified, status_message)
    """
    if not leaks:
        return source_code, True, "No leaks detected by compiler."

    prompt = f"""You are a formal compiler refactoring engine.
You are given target Python code along with a structured JSON payload of compiler diagnostics indicating resource leaks.

TARGET SOURCE CODE:
```python
{source_code}
```

COMPILER DIAGNOSTICS PAYLOAD:
{json.dumps(leaks, indent=2)}

TASK:
1. Refactor the code to guarantee that ALL allocated resources (e.g. files, sockets, database handles) are properly closed or safely managed using Python context managers (`with` blocks) or `try/finally` blocks.
2. Preserve exact mathematical logic, variable outputs, and execution semantics.
3. Return ONLY valid, executable Python code inside a single ```python ``` block. Do NOT include explanations, preambles, or conversational filler.
"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise automated compiler refactoring engine. Output code only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )
        raw_output = completion.choices[0].message.content or ""
        candidate_code = _clean_llm_code_response(raw_output)
    except Exception as e:
        return (
            source_code,
            False,
            f"Groq API Error: Failed to query model API ({str(e)})",
        )

    # -------------------------------------------------------------
    # Compiler Verification Loop
    # Re-run the generated code through the AST parser, CFG, & Leak Detector
    # -------------------------------------------------------------
    try:
        # Step 1: Syntax / AST Parsing Verification
        patch_tree = ast.parse(candidate_code)

        # Step 2: CFG Construction Pass
        cfg_builder = CFGBuilder()
        patch_cfg = cfg_builder.build_cfg(patch_tree)

        # Step 3: Data-Flow Leak Analysis
        detector = LeakDetector(patch_cfg)
        remaining_leaks = detector.detect_leaks()

        if len(remaining_leaks) == 0:
            return (
                candidate_code,
                True,
                "✅ Verification Passed: Compiler confirmed 0 remaining resource leaks.",
            )
        else:
            return (
                candidate_code,
                False,
                f"❌ Verification Failed: Compiler detected {len(remaining_leaks)} remaining leak(s).",
            )

    except SyntaxError as syn_err:
        return (
            candidate_code,
            False,
            f"❌ Verification Failed: LLM generated invalid syntax at line {syn_err.lineno}: {syn_err.msg}",
        )
    except Exception as ver_err:
        return (
            candidate_code,
            False,
            f"❌ Verification Failed: Compiler pipeline error ({str(ver_err)})",
        )