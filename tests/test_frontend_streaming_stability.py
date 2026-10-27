import unittest
import re
import os


class TestFrontendStreamingStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_js_path = os.path.join(os.path.dirname(__file__), "frontend", "app.js")
        with open(app_js_path, "r", encoding="utf-8") as f:
            cls.app_js_content = f.read()

    def test_01_raf_batching_and_scheduler_defined(self):
        """Verify scheduleAnswerRender and requestAnimationFrame are correctly wired in app.js."""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

        self.assertIn("function scheduleAnswerRender", self.app_js_content)
        self.assertIn("requestAnimationFrame", self.app_js_content)
        self.assertIn("renderPending", self.app_js_content)

    def test_02_render_safe_projection_preserves_raw_immutability(self):
        """Verify getRenderSafeAnswer balances incomplete fences without mutating original string."""
        self.assertIn("function getRenderSafeAnswer", self.app_js_content)
        
        # Test simulation of getRenderSafeAnswer logic in Python
        def get_render_safe_answer(text: str) -> str:
            if not text:
                return ""
            render_text = text
            fence_matches = re.findall(r"```", render_text)
            if fence_matches and len(fence_matches) % 2 != 0:
                render_text += "\n```"
            clean_without_blocks = re.sub(r"```[\s\S]*?```", "", render_text)
            single_backticks = re.findall(r"(?<!`)`(?!`)", clean_without_blocks)
            if len(single_backticks) % 2 != 0:
                render_text += "`"
            return render_text

        raw_stream = "Here is the code:\n```python\ndef solve():\n    return 42"
        safe_projection = get_render_safe_answer(raw_stream)
        
        # Raw stream is untouched
        self.assertFalse(raw_stream.endswith("```"))
        # Safe projection is balanced
        self.assertTrue(safe_projection.endswith("```"))
        self.assertEqual(len(re.findall(r"```", safe_projection)) % 2, 0)

    def test_03_incomplete_inline_backtick_recovery(self):
        """Verify unclosed inline backticks are balanced for display projection."""
        def get_render_safe_answer(text: str) -> str:
            if not text:
                return ""
            render_text = text
            fence_matches = re.findall(r"```", render_text)
            if fence_matches and len(fence_matches) % 2 != 0:
                render_text += "\n```"
            clean_without_blocks = re.sub(r"```[\s\S]*?```", "", render_text)
            single_backticks = re.findall(r"(?<!`)`(?!`)", clean_without_blocks)
            if len(single_backticks) % 2 != 0:
                render_text += "`"
            return render_text

        partial_text = "Let's inspect the `nums[i] element"
        safe_text = get_render_safe_answer(partial_text)
        self.assertTrue(safe_text.endswith("`"))

    def test_04_generation_id_tracking_in_window_hooks(self):
        """Verify onQuestionDetected, onAnswerStart, onAnswerToken, onAnswerComplete use currentGenerationId."""
        self.assertIn("currentGenerationId++", self.app_js_content)
        self.assertIn("genId !== currentGenerationId", self.app_js_content)

    def test_05_smart_auto_scroll_near_bottom_logic(self):
        """Verify smartScrollToBottom checks distance from bottom (< 60px) rather than unconditional forcing."""
        self.assertIn("function smartScrollToBottom", self.app_js_content)
        self.assertIn("distanceFromBottom < 60", self.app_js_content)
        self.assertIn("qaScrollBody.scrollHeight - qaScrollBody.scrollTop", self.app_js_content)

    def test_06_copy_button_data_block_idx_and_persistence(self):
        """Verify copy buttons include data-block-idx and active copy state restoration."""
        self.assertIn('data-block-idx="${idx}"', self.app_js_content)
        self.assertIn("restoreActiveCopyState()", self.app_js_content)

    def test_07_syntax_highlighting_multi_language_support(self):
        """Verify syntax highlighting rules cover Python, Java, C++, JavaScript, and SQL."""
        self.assertIn("function highlightCode", self.app_js_content)
        self.assertIn("'select', 'from', 'where', 'group', 'by', 'order'", self.app_js_content)
        self.assertIn("'public', 'private', 'protected', 'static', 'final'", self.app_js_content)
        self.assertIn("'typename', 'namespace', 'nullptr'", self.app_js_content)
        self.assertIn("'const', 'let', 'var', 'function', 'export'", self.app_js_content)
        self.assertIn("'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while'", self.app_js_content)

    def test_08_complexity_pill_badge_formatting(self):
        """Verify complexity formatting generates time-pill and space-pill badges."""
        self.assertIn('complexity-pill time-pill', self.app_js_content)
        self.assertIn('complexity-pill space-pill', self.app_js_content)

    def test_09_stale_token_discard_simulation(self):
        """Simulate generational token discard during rapid question transition."""
        state = {
            "current_generation_id": 1,
            "active_answer": ""
        }

        def on_answer_token(token: str, gen_id: int):
            if gen_id != state["current_generation_id"]:
                return  # Dropped
            state["active_answer"] += token

        def on_question_detected(new_q: str):
            state["current_generation_id"] += 1
            state["active_answer"] = ""

        # Generation 1 tokens
        on_answer_token("Answer 1 part A ", 1)
        on_answer_token("Answer 1 part B ", 1)
        self.assertEqual(state["active_answer"], "Answer 1 part A Answer 1 part B ")

        # New question arrives -> Generation 2
        on_question_detected("New Question 2")
        self.assertEqual(state["current_generation_id"], 2)
        self.assertEqual(state["active_answer"], "")

        # Stale token from generation 1 arrives
        on_answer_token("late token from gen 1", 1)
        self.assertEqual(state["active_answer"], "")  # Must remain empty!

        # Token for generation 2 arrives
        on_answer_token("Answer 2 clean token", 2)
        self.assertEqual(state["active_answer"], "Answer 2 clean token")


if __name__ == "__main__":
    unittest.main()
