"""
Comprehensive Test Suite for Phase 3 — Screen-Grounded Interview Questions
===========================================================================
Verifies that when the interviewer refers to a question on screen:
1. Spoken reference is detected as a visual reference.
2. Screen context is retrieved from the session cache.
3. Active coding problem is established on the InterviewSession.
4. Voice and screen signals fuse into a screen-grounded question for answer generation.
5. Non-visual questions bypass screen analysis and do not trigger false captures.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
from interview_engine import (
    InterviewSession,
    classify_and_understand_utterance
)
import screen_analyzer


class TestPhase3ScreenGroundedQuestions(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()
        self.mock_screen_data = {
            "is_coding_related": True,
            "has_coding_problem": True,
            "language": "Python",
            "problem_title": "Two Sum",
            "problem_statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            "code": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        pass",
            "error_message": "",
            "question_type": "coding",
            "confidence": 0.98
        }
        self.session.mock_screen_data = self.mock_screen_data

    # -------------------------------------------------------------------------
    # 1. POSITIVE VISUAL REFERENCE UTTERANCES
    # -------------------------------------------------------------------------
    def test_01_solve_question_on_the_screen(self):
        """Verify 'Solve the question on the screen.' fuses with screen problem."""
        utterance = "Solve the question on the screen."
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Two Sum", self.session.active_coding_problem)
        self.assertIn("Two Sum", turn["refined_question"])
        self.assertEqual(turn["utterance_type"], "CODING")

    def test_01b_please_solve_question_given_on_the_screen(self):
        """Verify 'Please solve the question given on the screen.' fuses with screen problem."""
        utterance = "Please solve the question given on the screen."
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Two Sum", self.session.active_coding_problem)
        self.assertIn("Two Sum", turn["refined_question"])

    def test_02_solve_this(self):
        """Verify 'Solve this.' fuses with screen problem."""
        utterance = "Solve this."
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Two Sum", self.session.active_coding_problem)
        self.assertIn("Two Sum", turn["refined_question"])

    def test_03_implement_this(self):
        """Verify 'Implement this.' fuses with screen problem."""
        utterance = "Implement this."
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Two Sum", self.session.active_coding_problem)
        self.assertIn("Two Sum", turn["refined_question"])

    def test_04_write_the_solution(self):
        """Verify 'Write the solution.' fuses with screen problem."""
        utterance = "Write the solution."
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Two Sum", self.session.active_coding_problem)
        self.assertIn("Two Sum", turn["refined_question"])

    def test_05_explain_this_problem(self):
        """Verify 'Explain this problem.' fuses with screen problem."""
        utterance = "Explain this problem."
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Two Sum", self.session.active_coding_problem)
        self.assertIn("Two Sum", turn["refined_question"])

    def test_06_what_is_the_output(self):
        """Verify 'What is the output?' fuses with debugging/screen code."""
        debug_screen = {
            "is_coding_related": True,
            "has_coding_problem": True,
            "language": "Python",
            "problem_title": "Output Trace",
            "problem_statement": "What is the output of print([i*2 for i in range(3)])?",
            "code": "print([i*2 for i in range(3)])",
            "error_message": "",
            "question_type": "debugging",
            "confidence": 0.95
        }
        self.session.mock_screen_data = debug_screen

        utterance = "What is the output?"
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Debug and explain", turn["refined_question"])
        self.assertEqual(turn["utterance_type"], "DEBUGGING")

    def test_07_give_me_pseudocode_for_this(self):
        """Verify 'Give me pseudocode for this.' fuses with screen problem."""
        utterance = "Give me pseudocode for this."
        self.assertTrue(screen_analyzer.is_visual_reference_question(utterance))

        turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertIn("Two Sum", self.session.active_coding_problem)

    # -------------------------------------------------------------------------
    # 2. NEGATIVE CONTROLS (NON-VISUAL SPOKEN QUESTIONS)
    # -------------------------------------------------------------------------
    def test_08_normal_voice_questions_bypass_screen_capture(self):
        """Verify normal technical / behavioral questions do NOT trigger visual reference."""
        non_visual_cases = [
            "Why did you choose Python?",
            "Explain REST API.",
            "What is polymorphism?"
        ]
        for utterance in non_visual_cases:
            with self.subTest(utterance=utterance):
                # Must not match visual reference
                self.assertFalse(
                    screen_analyzer.is_visual_reference_question(utterance),
                    f"Utterance '{utterance}' falsely triggered visual reference!"
                )
                
                # Session should not establish active coding problem from unrelated question
                fresh_session = InterviewSession()
                classify_and_understand_utterance(utterance, session=fresh_session, source="system")
                self.assertEqual(fresh_session.active_coding_problem, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
