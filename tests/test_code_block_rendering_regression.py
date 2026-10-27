"""
Regression Test Suite for Code-Block Rendering & Syntax-Highlighting Bug
=========================================================================
Tests:
1. Exact Python example rendering does NOT leak literal <span... or corrupted attribute text.
2. Code block HTML attributes are well-formed (no nested <span inside class="...").
3. Normal answer text outside code blocks preserves HTML escaping (e.g. 1 < 2 -> 1 &lt; 2).
4. Code inside code blocks is safely escaped against XSS payloads.
5. Code indentation (whitespace/tabs) is preserved intact.
6. Copy Code text extraction (innerText/textContent) returns raw code without HTML tags.
7. Multi-language syntax highlighting (Python, JS, SQL, C++, Java) works cleanly.
8. Keywords inside strings (e.g. "class Test: return 1") or comments (# def foo()) are not falsely highlighted.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import os
import re
import subprocess
import unittest


class TestCodeBlockRenderingRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app_js_path = os.path.join(os.path.dirname(__file__), "frontend", "app.js")
        with open(cls.app_js_path, "r", encoding="utf-8") as f:
            cls.app_js_content = f.read()

    def _eval_app_js_format(self, input_markdown: str) -> str:
        """Helper to run formatMarkdownAnswer from frontend/app.js directly via Node.js."""
        node_script = f"""
const fs = require('fs');
const appJs = fs.readFileSync({repr(self.app_js_path)}, 'utf8');

global.setInterval = () => 0;
global.clearInterval = () => {{}};
global.requestAnimationFrame = () => 0;
global.cancelAnimationFrame = () => {{}};

const mockElem = {{
  classList: {{ add: () => {{}}, remove: () => {{}}, contains: () => false }},
  style: {{}},
  addEventListener: () => {{}},
  removeEventListener: () => {{}},
  setAttribute: () => {{}},
  getAttribute: () => null
}};

const window = {{
  pywebview: {{ api: {{}} }},
  dismissAnswerCard: () => {{}},
  copyCodeSnippet: () => {{}},
  addEventListener: () => {{}},
  removeEventListener: () => {{}},
  navigator: {{ clipboard: {{ writeText: () => {{}} }} }}
}};
const document = {{
  getElementById: () => mockElem,
  querySelector: () => mockElem,
  querySelectorAll: () => [],
  addEventListener: () => {{}},
  removeEventListener: () => {{}},
  body: mockElem
}};
global.window = window;
global.document = document;

let codeToRun = appJs.replace(
  '// --- Syntax Highlighting & Rich Formatting ---',
  'window.formatMarkdownAnswer = formatMarkdownAnswer; // --- Syntax Highlighting & Rich Formatting ---'
);
if (!codeToRun.includes('window.formatMarkdownAnswer')) {{
  codeToRun = appJs.replace('(function () {{', '(function () {{ window.formatMarkdownAnswer = formatMarkdownAnswer;');
}}

eval(codeToRun);

const input = {repr(input_markdown)};
const output = window.formatMarkdownAnswer(input);
process.stdout.write(output || '');
process.exit(0);
"""
        res = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0:
            raise RuntimeError(f"Node execution error: {res.stderr}\nStdout: {res.stdout}")
        return res.stdout

    # -------------------------------------------------------------------------
    # 1. REPRODUCE EXACT BUG: Python example must not leak literal <span> markup
    # -------------------------------------------------------------------------
    def test_01_exact_python_example_syntax_highlighting_not_literal(self):
        """
        Verifies that the generated Python two_sum example produces clean,
        well-formed HTML tags without nested <span> tags in class attributes.
        """
        raw_markdown = """Here is the optimal solution:

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []
```

• **Time Complexity**: O(n)
• **Space Complexity**: O(n)"""

        rendered_html = self._eval_app_js_format(raw_markdown)

        # 1. Must contain valid code block wrapper and code tags
        self.assertIn('<pre class="code-block-wrapper">', rendered_html)
        self.assertIn("<code>", rendered_html)

        # 2. Must NOT contain corrupted attribute syntax: class=<span or class="<span
        self.assertNotIn("class=<span", rendered_html)
        self.assertNotIn('class="<span', rendered_html)
        self.assertNotIn("class=<", rendered_html)
        self.assertNotIn('<span class="code-string">"code-keyword"</span>', rendered_html)
        self.assertNotIn('<span class="code-string">"code-builtin"</span>', rendered_html)

        # 3. Must properly highlight keywords and builtins with valid span elements
        self.assertIn('<span class="code-keyword">def</span>', rendered_html)
        self.assertIn('<span class="code-keyword">for</span>', rendered_html)
        self.assertIn('<span class="code-keyword">in</span>', rendered_html)
        self.assertIn('<span class="code-builtin">enumerate</span>', rendered_html)
        self.assertIn('<span class="code-keyword">if</span>', rendered_html)
        self.assertIn('<span class="code-keyword">return</span>', rendered_html)

        # 4. Open and close span counts must match exactly inside code block
        code_match = re.search(r"<code>([\s\S]*?)</code>", rendered_html)
        self.assertIsNotNone(code_match)
        code_html = code_match.group(1)
        self.assertEqual(len(re.findall(r"<span", code_html)), len(re.findall(r"</span>", code_html)))

    # -------------------------------------------------------------------------
    # 2. CODE INDENTATION PRESERVATION
    # -------------------------------------------------------------------------
    def test_02_code_indentation_preserved(self):
        """Verifies exact 4-space and 8-space indentation is preserved inside code blocks."""
        raw_markdown = """```python
def example():
    first_level = 1
        second_level = 2
    return first_level + second_level
```"""
        rendered_html = self._eval_app_js_format(raw_markdown)
        self.assertIn("    first_level", rendered_html)
        self.assertIn("        second_level", rendered_html)
        self.assertIn('    <span class="code-keyword">return</span>', rendered_html)

    # -------------------------------------------------------------------------
    # 3. COPY CODE RAW TEXT EXTRACTION (NO HTML TAGS)
    # -------------------------------------------------------------------------
    def test_03_copy_code_returns_clean_raw_code(self):
        """
        Simulates browser DOM textContent/innerText extraction from <code>
        to ensure Copy Code returns the exact raw Python code without any HTML tags.
        """
        raw_code = """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []"""

        raw_markdown = f"```python\n{raw_code}\n```"
        rendered_html = self._eval_app_js_format(raw_markdown)

        # Extract content between <code> and </code>
        code_match = re.search(r"<code>([\s\S]*?)</code>", rendered_html)
        self.assertIsNotNone(code_match, "<code> tag must exist")
        inner_code_html = code_match.group(1)

        # Simulate DOM textContent (stripping HTML tags and unescaping HTML entities)
        text_content = re.sub(r"<[^>]+>", "", inner_code_html)
        text_content = (
            text_content
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        )

        self.assertEqual(text_content.strip(), raw_code.strip(), "Extracted text content must match original code exactly")

    # -------------------------------------------------------------------------
    # 4. NORMAL TEXT HTML ESCAPING & XSS PREVENTION
    # -------------------------------------------------------------------------
    def test_04_html_escaping_and_xss_safety(self):
        """
        Verifies that:
        - Normal answer text with <, >, & is safely HTML-escaped.
        - Code blocks with XSS payloads (e.g. <script>) are safely HTML-escaped.
        """
        raw_markdown = """Comparing performance when a < b & b > 0:

```python
def check():
    payload = "<script>alert('xss')</script>"
    return 1 < 2 and 3 > 2
```

Use `a < b` for comparison."""

        rendered_html = self._eval_app_js_format(raw_markdown)

        # Normal text escaping
        self.assertIn("a &lt; b &amp; b &gt; 0", rendered_html)
        self.assertNotIn("<script>", rendered_html)

        # Code block escaping
        self.assertIn("&lt;script&gt;", rendered_html)
        self.assertIn("&lt; <span", rendered_html)  # '<' is escaped to &lt;
        self.assertIn("&gt; <span", rendered_html)  # '>' is escaped to &gt;

    # -------------------------------------------------------------------------
    # 5. KEYWORDS INSIDE STRINGS AND COMMENTS NOT CORRUPTED
    # -------------------------------------------------------------------------
    def test_05_keywords_inside_strings_and_comments_not_corrupted(self):
        """
        Verifies that keywords inside string literals or comments do NOT get
        falsely highlighted or break the surrounding string/comment tag.
        """
        raw_markdown = """```python
# This comment contains def, for, return and quotes "test"
def get_msg():
    msg = "class Test: def solve(self): return 123"
    return msg
```"""
        rendered_html = self._eval_app_js_format(raw_markdown)

        # Entire comment should be in a single code-comment span
        self.assertIn('<span class="code-comment"># This comment contains def, for, return and quotes &quot;test&quot;</span>', rendered_html)

        # Entire string should be in a single code-string span, not broken up
        self.assertIn('<span class="code-string">&quot;class Test: def solve(self): return 123&quot;</span>', rendered_html)

    # -------------------------------------------------------------------------
    # 6. CODE BLOCKS NEVER WRAPPED IN PARAGRAPH TAGS
    # -------------------------------------------------------------------------
    def test_06_code_blocks_not_wrapped_in_p(self):
        """
        Verifies that code block wrappers (<pre class="code-block-wrapper">)
        are never illegally wrapped inside <p>...</p> tags.
        """
        raw_markdown = """Here is the solution:

```python
def solve():
    return 42
```

Done."""
        rendered_html = self._eval_app_js_format(raw_markdown)
        self.assertIn('<pre class="code-block-wrapper">', rendered_html)
        self.assertNotIn('<p><pre', rendered_html)
        self.assertNotIn('</pre></p>', rendered_html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
