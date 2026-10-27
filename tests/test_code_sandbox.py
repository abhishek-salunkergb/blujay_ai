"""
Unit Test Suite for Live Code Execution Sandbox (Feature 13)
============================================================
Validates:
1. Markdown code block extraction
2. Python execution with stdout and return code
3. Stderr and syntax error handling
4. Infinite loop timeout protection
5. Test cases runner evaluation
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
from code_sandbox import (
    extract_code_from_markdown,
    execute_code_snippet,
    run_test_cases
)


class TestCodeSandbox(unittest.TestCase):

    def test_01_extract_code_from_markdown(self):
        """Verify extractor identifies python code blocks from markdown."""
        md_text = """Here is the optimal O(N) Two Sum solution:
```python
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in seen:
            return [seen[diff], i]
        seen[n] = i
    return []

print(twoSum([2, 7, 11, 15], 9))
```
Complexity is O(N) time and O(N) space.
"""
        blocks = extract_code_from_markdown(md_text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["language"], "python")
        self.assertIn("twoSum", blocks[0]["code"])

    def test_02_execute_python_code_success(self):
        """Verify execution of valid Python code."""
        code = "a = 5\nb = 10\nprint(f'Sum: {a + b}')"
        res = execute_code_snippet(code, language="python")
        self.assertTrue(res["success"])
        self.assertEqual(res["stdout"], "Sum: 15")
        self.assertEqual(res["returncode"], 0)
        self.assertFalse(res["timed_out"])

    def test_03_execute_python_runtime_error(self):
        """Verify execution captures exceptions and stderr."""
        code = "print(10 / 0)"
        res = execute_code_snippet(code, language="python")
        self.assertFalse(res["success"])
        self.assertIn("ZeroDivisionError", res["stderr"])
        self.assertNotEqual(res["returncode"], 0)

    def test_04_timeout_infinite_loop(self):
        """Verify sandbox terminates infinite loop after timeout."""
        code = "while True:\n    pass"
        res = execute_code_snippet(code, language="python", timeout_sec=0.4)
        self.assertFalse(res["success"])
        self.assertTrue(res["timed_out"])
        self.assertIn("Timed Out", res["stderr"])

    def test_05_run_test_cases_suite(self):
        """Verify test case evaluation."""
        code = """import sys
lines = sys.stdin.read().split()
if lines:
    n = int(lines[0])
    print(n * 2)
"""
        test_cases = [
            {"input": "5", "expected": "10"},
            {"input": "12", "expected": "24"}
        ]
        res = run_test_cases(code, test_cases, language="python")
        self.assertEqual(res["total"], 2)
        self.assertEqual(res["passed"], 2)
        self.assertTrue(res["all_passed"])

    def test_06_sandbox_safety_blocks_dangerous_calls(self):
        """Verify sandbox statically intercepts dangerous system calls and modules."""
        # 1. Attempt to import os
        dangerous_code_1 = "import os\nos.system('echo hacked')"
        res1 = execute_code_snippet(dangerous_code_1, language="python")
        self.assertFalse(res1["success"])
        self.assertIn("Sandbox Security Violation", res1["stderr"])
        self.assertIn("os", res1["stderr"])

        # 2. Attempt to import subprocess
        dangerous_code_2 = "import subprocess\nsubprocess.Popen(['calc'])"
        res2 = execute_code_snippet(dangerous_code_2, language="python")
        self.assertFalse(res2["success"])
        self.assertIn("Sandbox Security Violation", res2["stderr"])


if __name__ == "__main__":
    unittest.main()

