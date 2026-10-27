"""
Comprehensive Regression Test Suite for Phase 5 — Surgical Failure Fixes
=========================================================================
Tests the exact failure cases identified and resolved in Phase 4:
1. 'Solve it.', 'Implement it.', 'Code it.' triggered as visual reference coding questions.
2. Declarative interviewer monologues ('So how we handle this...', 'Why we chose Redis was because...') are ignored.
3. Rephrase/retraction statements ('Actually, let me rephrase that.') are ignored without generating answers.
4. Genuine questions ('How did you handle Redis caching?', 'Why did you choose Redis?') continue to generate high-confidence answers.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
from interview_engine import (
    InterviewSession,
    classify_and_understand_utterance,
    classify_answer_relationship
)
import screen_analyzer


class TestPhase5StressFixes(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()

    # -------------------------------------------------------------------------
    # 1. FIX 1: Screen-Grounded 'Solve it' / 'Implement it' / 'Code it'
    # -------------------------------------------------------------------------
    def test_01_solve_it_visual_reference_and_grounding(self):
        """Verify 'Solve it.' is recognized as a visual reference and grounds to screen context."""
        mock_screen = {
            "is_coding_related": True,
            "has_coding_problem": True,
            "language": "Python",
            "problem_title": "LRU Cache",
            "problem_statement": "Design a data structure for LRU cache.",
            "code": "class LRUCache: pass",
            "error_message": "",
            "question_type": "coding",
            "confidence": 0.98
        }
        self.session.mock_screen_data = mock_screen

        for cmd in ["Solve it.", "Implement it.", "Code it."]:
            with self.subTest(cmd=cmd):
                self.assertTrue(screen_analyzer.is_visual_reference_question(cmd))
                turn = classify_and_understand_utterance(cmd, session=self.session, source="system")
                self.assertTrue(turn["should_generate_answer"])
                self.assertEqual(turn["utterance_type"], "CODING")
                self.assertIn("LRU", self.session.active_coding_problem)

    # -------------------------------------------------------------------------
    # 2. FIX 2: Declarative Interviewer Monologues Bypassed
    # -------------------------------------------------------------------------
    def test_02_interviewer_monologues_do_not_generate_answers(self):
        """Verify first-person plural explanatory statements do NOT trigger answers."""
        monologues = [
            "So how we handle this in production is by using Kubernetes.",
            "Why we chose Redis was because of its fast in-memory key-value lookups.",
            "How we deploy our services is via a multi-stage CI/CD pipeline.",
            "What we normally do is partition the table by date."
        ]
        for monologue in monologues:
            with self.subTest(monologue=monologue):
                turn = classify_and_understand_utterance(monologue, session=self.session, source="system")
                self.assertFalse(
                    turn["should_generate_answer"],
                    f"Monologue '{monologue}' falsely triggered an answer!"
                )
                self.assertIn(turn["question_state"], {"LOW", "NON_QUESTION"})

    # -------------------------------------------------------------------------
    # 3. FIX 3: Rephrasing & Retraction Statements Bypassed
    # -------------------------------------------------------------------------
    def test_03_rephrase_and_retraction_statements_do_not_generate(self):
        """Verify rephrasing statements do NOT trigger answers."""
        rephrase_statements = [
            "Actually, let me rephrase that.",
            "Let me ask that differently.",
            "Scratch that, let's talk about databases instead.",
            "Never mind that question."
        ]
        for statement in rephrase_statements:
            with self.subTest(statement=statement):
                turn = classify_and_understand_utterance(statement, session=self.session, source="system")
                self.assertFalse(
                    turn["should_generate_answer"],
                    f"Rephrase statement '{statement}' falsely triggered an answer!"
                )

    # -------------------------------------------------------------------------
    # 4. Genuine Questions Continue to Generate High-Confidence Answers
    # -------------------------------------------------------------------------
    def test_04_genuine_questions_generate_high_confidence_answers(self):
        """Verify genuine candidate-directed questions produce high-confidence answers."""
        genuine_questions = [
            ("How did you handle Redis caching?", "FOLLOW_UP"),
            ("Why did you choose Redis?", "FOLLOW_UP"),
            ("How do you handle race conditions in distributed systems?", "TECHNICAL"),
            ("Why did you pick PostgreSQL over MySQL?", "TECHNICAL")
        ]
        for query, expected_type in genuine_questions:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertTrue(
                    turn["should_generate_answer"],
                    f"Genuine question '{query}' failed to trigger an answer!"
                )
                self.assertEqual(turn["question_state"], "HIGH")
                self.assertGreaterEqual(turn["question_confidence"], 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
