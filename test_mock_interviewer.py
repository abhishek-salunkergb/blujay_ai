"""
Unit Test Suite for AI Mock Interviewer & Simulation Mode (Feature 9)
====================================================================
Validates:
1. Session initialization across roles, difficulties, and personas
2. First question generation tailored to tracks (backend, frontend, system_design, behavioral)
3. Incremental answer submission, rubric scoring, and feedback
4. Multi-turn simulation progression until target questions reached
5. Final performance scorecard generation and category metrics
"""

import unittest
from mock_interviewer import MockInterviewSession, ROLE_TRACKS, DIFFICULTY_LEVELS, PERSONAS


class TestMockInterviewer(unittest.TestCase):

    def test_01_session_initialization(self):
        """Verify initialization with custom parameters and default fallbacks."""
        session = MockInterviewSession(
            role="backend",
            difficulty="senior",
            persona="bar_raiser",
            target_questions=4
        )
        info = session.get_session_info()
        self.assertEqual(info["role"], "backend")
        self.assertEqual(info["difficulty"], "senior")
        self.assertEqual(info["persona"], "bar_raiser")
        self.assertEqual(info["total_questions"], 4)
        self.assertFalse(info["is_completed"])

    def test_02_first_question_generation(self):
        """Verify first question generation for different tracks."""
        # Backend
        s_be = MockInterviewSession(role="backend")
        q_be = s_be.generate_first_question()
        self.assertIn("Question 1", q_be["interviewer_text"])
        self.assertIn("pooling", q_be["question_text"])

        # System Design
        s_sd = MockInterviewSession(role="system_design")
        q_sd = s_sd.generate_first_question()
        self.assertIn("rate limiter", q_sd["question_text"])

        # Behavioral
        s_beh = MockInterviewSession(role="behavioral")
        q_beh = s_beh.generate_first_question()
        self.assertIn("challenging project", q_beh["question_text"])

    def test_03_submit_answer_scoring_and_rubric(self):
        """Verify submitting an answer evaluates depth and returns rubric."""
        session = MockInterviewSession(role="backend", target_questions=3)
        session.generate_first_question()

        # Detailed answer
        ans = (
            "We use HikariCP for connection pooling with max pool size tuned based on PostgreSQL CPU cores. "
            "For query optimization, we create composite B-tree indexes, avoid N+1 queries via batch fetching, "
            "and use read replicas with PgBouncer to offload read traffic."
        )
        res = session.submit_answer(candidate_answer=ans, elapsed_seconds=45)

        self.assertFalse(res["is_completed"])
        self.assertGreaterEqual(res["evaluation"]["score"], 80)
        self.assertIn("technical_accuracy", res["evaluation"]["rubric"])
        self.assertIsNotNone(res["next_question"])

    def test_04_full_simulation_lifecycle_to_scorecard(self):
        """Verify completing all questions generates comprehensive scorecard."""
        session = MockInterviewSession(role="backend", target_questions=2)
        session.generate_first_question()

        # Answer Q1
        res1 = session.submit_answer("First answer with solid detail and trade-offs.", elapsed_seconds=30)
        self.assertFalse(res1["is_completed"])

        # Answer Q2 (final)
        res2 = session.submit_answer("Second answer explaining optimistic locking vs pessimistic locking with version columns.", elapsed_seconds=40)
        self.assertTrue(res2["is_completed"])
        self.assertIsNotNone(res2["scorecard"])

        sc = res2["scorecard"]
        self.assertIn("overall_score", sc)
        self.assertIn("recommendation", sc)
        self.assertIn("category_scores", sc)
        self.assertEqual(len(sc["questions_summary"]), 2)
        self.assertGreaterEqual(len(sc["strengths"]), 1)


if __name__ == "__main__":
    unittest.main()
