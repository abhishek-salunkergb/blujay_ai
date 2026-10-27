"""
Test suite validating end-to-end follow-up conversation lifecycle
after a coding problem is solved via screenshot or region snip.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import unittest
import time
from unittest.mock import MagicMock, patch

from interview_engine import (
    InterviewSession,
    InterviewTurnState,
    classify_answer_relationship,
    classify_and_understand_utterance,
    generate_interview_answer
)
from web_copilot_view import WebCopilotView


class TestScreenshotFollowUpLifecycle(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()
        self.mock_client = MagicMock()

    @patch("web_copilot_view.ContinuousTranscriber")
    @patch("web_copilot_view.GlobalHotkeyManager")
    @patch("web_copilot_view.solve_screen_coding_problem")
    def test_screen_solve_populates_active_coding_and_anchors_turn2_followup(
        self, mock_solve_screen, mock_hotkey, mock_transcriber
    ):
        """
        Simulates:
        Turn 1: User triggers screen solver on a coding problem.
        Turn 2: Interviewer says "explain it" -> verify it anchors to the screen code solution.
        Turn 3: Interviewer says "what's the time complexity?" -> verify complexity follow-up anchors.
        Turn 4: Interviewer says "can you optimize it?" -> verify optimization anchors.
        """
        view = WebCopilotView(client=self.mock_client)
        view.session = self.session
        view.window = MagicMock()
        view._eval_js = MagicMock()

        screen_solution = (
            "1. 💡 **Problem Summary & Key Edge Cases**:\n"
            "- Two Sum: Given an array of integers and a target, find the two indices that add up to target.\n\n"
            "2. 🧠 **Optimal Approach & Intuition**:\n"
            "- Use a single-pass hash map to achieve O(N) lookup.\n\n"
            "3. 💻 **Clean, Production-Ready Code**:\n"
            "```python\n"
            "def twoSum(nums: list[int], target: int) -> list[int]:\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        diff = target - num\n"
            "        if diff in seen:\n"
            "            return [seen[diff], i]\n"
            "        seen[num] = i\n"
            "    return []\n"
            "```\n\n"
            "4. ⏱️ **Complexity Analysis**:\n"
            "- • **Time Complexity**: O(N)\n"
            "- • **Space Complexity**: O(N)\n"
        )
        mock_solve_screen.return_value = screen_solution

        # Turn 1: Trigger screenshot solver
        view.trigger_code_screen_solver()

        # Wait briefly for worker thread to complete
        time.sleep(0.1)

        # Verify Turn 1 state in session
        self.assertIn("def twoSum", self.session.active_coding_solution)
        self.assertTrue(len(self.session.qa_pairs) > 0)
        self.assertEqual(self.session.turn_state, InterviewTurnState.LISTENING)
        self.assertEqual(view.active_raw_answer, screen_solution)

        # Turn 2: Follow-up question via spoken speech: "explain it"
        analysis = classify_and_understand_utterance("explain it", session=self.session, source="system")
        self.assertTrue(analysis["should_generate_answer"])
        self.assertIn("Explain the algorithmic approach and logic", analysis["refined_question"])

        # Verify generation prompt anchors the solved screen code
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Here is the explanation..."
        self.mock_client.chat.completions.create.return_value = [mock_chunk]

        generate_interview_answer(
            client=self.mock_client,
            session=self.session,
            question=analysis["refined_question"],
            on_token=lambda t: None,
            question_type=analysis.get("utterance_type", "technical")
        )
        call_args = self.mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_prompt = messages[-1]["content"]
        self.assertIn("[PRECEDING ACTIVE CODE SOLUTION TO EXPLAIN]:", user_prompt)
        self.assertIn("def twoSum", user_prompt)

        # Turn 3: Follow-up via manual typed input: "what is the time complexity?"
        rel3 = classify_answer_relationship("what is the time complexity?", session=self.session)
        self.assertEqual(rel3["strategy_key"], "coding_complexity")
        self.assertTrue(rel3["is_follow_up"])

        # Turn 4: Follow-up via manual typed input: "can you optimize it?"
        rel4 = classify_answer_relationship("can you optimize it?", session=self.session)
        self.assertEqual(rel4["strategy_key"], "coding_optimization")
        self.assertTrue(rel4["is_follow_up"])

    @patch("web_copilot_view.ContinuousTranscriber")
    @patch("web_copilot_view.GlobalHotkeyManager")
    def test_submit_manual_question_uses_active_coding_followup(
        self, mock_hotkey, mock_transcriber
    ):
        """Verify that typing 'explain it' in the input box also anchors to the active code."""
        view = WebCopilotView(client=self.mock_client)
        view.session = self.session
        view.window = MagicMock()
        view._eval_js = MagicMock()
        view._handle_detected_question = MagicMock()

        self.session.record_qa(
            "Screen Problem: Reverse Linked List",
            "```python\ndef reverseList(head): pass\n```",
            category="coding"
        )
        view.submit_manual_question("explain it")

        view._handle_detected_question.assert_called_once()
        called_args = view._handle_detected_question.call_args
        called_q = called_args[0][0]
        self.assertIn("Explain the algorithmic approach and logic", called_q)


if __name__ == "__main__":
    unittest.main()
