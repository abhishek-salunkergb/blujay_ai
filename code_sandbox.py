"""
Live Code Execution Sandbox & Test Runner (Feature 13)
======================================================
Provides safe, isolated, and bounded subprocess code execution for Python and JavaScript
to verify algorithm solutions, test cases, and edge cases live during technical coding rounds.
"""

import re
import sys
import ast
import time
import subprocess
from typing import Dict, Any, List, Tuple

RESTRICTED_SANDBOX_MODULES = {
    "os", "subprocess", "shutil", "socket", "pty",
    "ctypes", "posix", "nt", "winreg", "msvcrt", "signal"
}

RESTRICTED_SANDBOX_CALLS = {
    "remove", "unlink", "rmdir", "rmtree", "system", "popen",
    "spawn", "kill", "terminate"
}


def validate_code_safety(code: str, language: str = "python") -> Tuple[bool, str]:
    """
    Statically analyzes code to disallow execution of dangerous system-level calls
    and destructive filesystem modifications inside the interview sandbox.
    """
    lang = language.lower()
    if lang in ["python", "py", "python3"]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base_mod = alias.name.split(".")[0].lower()
                        if base_mod in RESTRICTED_SANDBOX_MODULES:
                            return False, f"Sandbox Security Violation: import of '{base_mod}' is restricted."
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_mod = node.module.split(".")[0].lower()
                        if base_mod in RESTRICTED_SANDBOX_MODULES:
                            return False, f"Sandbox Security Violation: import from '{base_mod}' is restricted."
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in RESTRICTED_SANDBOX_CALLS:
                            return False, f"Sandbox Security Violation: call to '{node.func.id}()' is restricted."
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in RESTRICTED_SANDBOX_CALLS:
                            return False, f"Sandbox Security Violation: call to '.{node.func.attr}()' is restricted."
        except SyntaxError:
            pass
        except Exception:
            pass

    return True, ""


def extract_code_from_markdown(text: str) -> List[Dict[str, str]]:
    """
    Extracts all fenced code blocks from markdown text with language tags.
    """
    pattern = r"```([a-zA-Z0-9_\+\#\.\-]*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for lang, code in matches:
        clean_lang = lang.strip().lower() or "python"
        clean_code = code.strip()
        if clean_code:
            results.append({
                "language": clean_lang,
                "code": clean_code
            })
    return results


def execute_code_snippet(
    code: str,
    language: str = "python",
    stdin_input: str = "",
    timeout_sec: float = 3.5
) -> Dict[str, Any]:
    """
    Executes a code snippet in an isolated subprocess with bounded CPU timeout.
    """
    is_safe, violation_msg = validate_code_safety(code, language)
    if not is_safe:
        return {
            "success": False,
            "stdout": "",
            "stderr": violation_msg,
            "returncode": -1,
            "duration_sec": 0.0,
            "timed_out": False
        }

    lang = language.lower()
    start_time = time.perf_counter()

    if lang in ["python", "py", "python3"]:
        cmd = [sys.executable, "-c", code]
    elif lang in ["javascript", "js", "node"]:
        cmd = ["node", "-e", code]
    else:
        # Fallback to python
        cmd = [sys.executable, "-c", code]

    try:
        proc = subprocess.run(
            cmd,
            input=stdin_input,
            text=True,
            capture_output=True,
            timeout=timeout_sec
        )
        duration = round(time.perf_counter() - start_time, 4)
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
            "duration_sec": duration,
            "timed_out": False
        }
    except subprocess.TimeoutExpired:
        duration = round(time.perf_counter() - start_time, 4)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution Timed Out (exceeded {timeout_sec}s limit - possible infinite loop).",
            "returncode": -1,
            "duration_sec": duration,
            "timed_out": True
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Runtime interpreter for '{language}' not found on system PATH.",
            "returncode": 127,
            "duration_sec": 0.0,
            "timed_out": False
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Sandbox execution error: {e}",
            "returncode": 1,
            "duration_sec": 0.0,
            "timed_out": False
        }


def run_test_cases(
    code: str,
    test_cases: List[Dict[str, str]],
    language: str = "python"
) -> Dict[str, Any]:
    """
    Runs code against a suite of input/expected_output test cases.
    """
    passed = 0
    total = len(test_cases)
    results = []

    for idx, tc in enumerate(test_cases, 1):
        inp = tc.get("input", "")
        expected = tc.get("expected", "").strip()
        res = execute_code_snippet(code, language=language, stdin_input=inp)
        actual = res.get("stdout", "").strip()
        is_match = (actual == expected) if expected else res.get("success", False)
        if is_match:
            passed += 1
        results.append({
            "test_number": idx,
            "input": inp,
            "expected": expected,
            "actual": actual,
            "passed": is_match,
            "error": res.get("stderr", ""),
            "duration_sec": res.get("duration_sec", 0.0)
        })

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "all_passed": passed == total and total > 0,
        "details": results
    }
