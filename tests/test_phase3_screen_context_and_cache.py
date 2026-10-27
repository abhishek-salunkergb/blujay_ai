import unittest
import time
from unittest.mock import patch, MagicMock

from interview_engine import (
    InterviewSession,
    classify_and_understand_utterance,
    classify_answer_relationship
)
import screen_analyzer
import classifier


class TestPhase3ScreenContextAndCache(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()

    def test_01_screen_problem_detection(self):
        """Verify is_visual_reference_question identifies screen coding references."""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

        self.assertTrue(screen_analyzer.is_visual_reference_question("Solve this."))
        self.assertTrue(screen_analyzer.is_visual_reference_question("Implement this problem."))
        self.assertTrue(screen_analyzer.is_visual_reference_question("What's wrong with this?"))
        self.assertTrue(screen_analyzer.is_visual_reference_question("Can you explain this problem on screen?"))

    def test_02_screen_problem_extraction_and_binding(self):
        """Verify screen problem statement and language are extracted and bound to session."""
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Write a function to find the first non-repeating character in a string.",
            "code": "def firstUniqChar(s: str) -> int: pass",
            "question_type": "coding",
            "confidence": 0.95
        }
        self.session.mock_screen_data = mock_screen
        
        result = classify_and_understand_utterance(
            utterance="Solve this.",
            session=self.session,
            speaker_hint="Interviewer"
        )
        
        # Check utterance processing result
        self.assertEqual(result["utterance_type"], "CODING")
        self.assertIn("first non-repeating character", result["refined_question"])
        
        # Check session state binding
        self.assertEqual(self.session.active_coding_problem, "Write a function to find the first non-repeating character in a string.")
        self.assertEqual(self.session.active_coding_language, "Python")
        
        # Check session cache write
        self.assertIsNotNone(self.session.screen_context_cache)
        self.assertEqual(self.session.screen_context_cache["data"]["problem_statement"], mock_screen["problem_statement"])

    def test_03_session_cache_write_and_read(self):
        """Verify session.screen_context_cache stores and returns cached data."""
        mock_screen = {
            "is_coding_related": True,
            "language": "Java",
            "problem_statement": "Reverse a Singly Linked List.",
            "code": "ListNode reverseList(ListNode head)",
            "question_type": "coding"
        }
        # Direct write to session cache
        self.session.screen_context_cache = {
            "data": mock_screen,
            "timestamp": time.time()
        }
        
        # analyze_visible_screen should return cached data without capturing
        with patch("screen_analyzer.capture_screen_image") as mock_capture:
            data = screen_analyzer.analyze_visible_screen(session=self.session)
            mock_capture.assert_not_called()
            self.assertEqual(data["problem_statement"], "Reverse a Singly Linked List.")

    def test_04_cache_ttl_validity_at_19_seconds(self):
        """Verify cache is valid and reused at t = 19.9s (within 20s TTL)."""
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Valid Parentheses."
        }
        current_time = 1000.0
        self.session.screen_context_cache = {
            "data": mock_screen,
            "timestamp": current_time
        }
        
        # Mock time at current_time + 19.9s
        with patch("time.time", return_value=current_time + 19.9):
            with patch("screen_analyzer.capture_screen_image") as mock_capture:
                data = screen_analyzer.analyze_visible_screen(session=self.session)
                mock_capture.assert_not_called()
                self.assertEqual(data["problem_statement"], "Valid Parentheses.")

    def test_05_cache_ttl_expiration_at_20_seconds(self):
        """Verify cache is expired and ignored at t = 20.1s."""
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Stale Problem."
        }
        current_time = 1000.0
        self.session.screen_context_cache = {
            "data": mock_screen,
            "timestamp": current_time
        }
        
        # Mock time at current_time + 20.1s (expired)
        with patch("time.time", return_value=current_time + 20.1):
            with patch("screen_analyzer.capture_screen_image") as mock_capture:
                mock_capture.side_effect = Exception("Stop after capture check")
                data = screen_analyzer.analyze_visible_screen(session=self.session)
                # Capture was attempted because cache was expired
                mock_capture.assert_called_once()

    def test_06_cache_replacement_with_new_problem(self):
        """Verify a new valid screen problem replaces the old cached problem."""
        problem_a = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Problem A: Two Sum",
            "question_type": "coding"
        }
        problem_b = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Problem B: LRU Cache",
            "question_type": "coding"
        }
        
        # Turn 1 with Problem A
        self.session.mock_screen_data = problem_a
        classify_and_understand_utterance("Solve this.", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(self.session.active_coding_problem, "Problem A: Two Sum")
        
        # Turn 2 with Problem B (new screen data)
        self.session.mock_screen_data = problem_b
        classify_and_understand_utterance("Solve this.", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(self.session.active_coding_problem, "Problem B: LRU Cache")
        self.assertEqual(self.session.screen_context_cache["data"]["problem_statement"], "Problem B: LRU Cache")

    def test_07_solve_this_intent_with_screen_context(self):
        """Verify 'Solve this.' correctly fuses with active screen context."""
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Find median of two sorted arrays.",
            "question_type": "coding"
        }
        self.session.mock_screen_data = mock_screen
        res = classify_and_understand_utterance("Solve this.", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(res["utterance_type"], "CODING")
        self.assertIn("Find median of two sorted arrays", res["refined_question"])

    def test_08_implement_this_intent_with_screen_context(self):
        """Verify 'Implement this.' fuses with active screen context."""
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Construct Binary Tree from Preorder and Inorder Traversal.",
            "question_type": "coding"
        }
        self.session.mock_screen_data = mock_screen
        res = classify_and_understand_utterance("Implement this.", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(res["utterance_type"], "CODING")
        self.assertIn("Construct Binary Tree", res["refined_question"])

    def test_09_explain_this_intent_with_screen_context(self):
        """Verify 'Explain this problem.' uses screen context."""
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Word Break Problem using Dynamic Programming.",
            "question_type": "coding"
        }
        self.session.mock_screen_data = mock_screen
        res = classify_and_understand_utterance("Explain this problem on screen.", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(res["utterance_type"], "CODING")
        self.assertIn("Word Break Problem", res["refined_question"])

    def test_10_optimize_this_intent_with_screen_context(self):
        """Verify 'Optimize this solution.' triggers coding optimization."""
        self.session.active_coding_problem = "Two Sum O(N^2) brute force"
        self.session.active_coding_solution = "def twoSum(nums, target): pass"
        rel = classify_answer_relationship("Optimize this solution.", session=self.session)
        self.assertEqual(rel["strategy_key"], "coding_optimization")

    def test_11_give_me_pseudocode_with_screen_context(self):
        """Verify 'Give me pseudocode for this.' resolves to pseudocode with active problem."""
        self.session.active_coding_problem = "Merge K Sorted Lists"
        rel = classify_answer_relationship("Give me pseudocode for this.", session=self.session)
        self.assertEqual(rel["strategy_key"], "pseudocode")
        self.assertIn("Merge K Sorted Lists", rel["fused_question"])

    def test_12_multi_turn_screen_code_explain_pseudocode_complexity_optimize(self):
        """
        Verify the full 5-turn sequence with screen grounding:
        Turn 1: Screen problem -> 'Solve this' -> Code generated & cached
        Turn 2: 'Explain your approach' -> Uses cached problem & canonical solution
        Turn 3: 'Give me pseudocode' -> Uses cached problem & canonical solution
        Turn 4: 'What's the time complexity?' -> Uses cached problem & canonical solution
        Turn 5: 'Can you optimize it?' -> Uses canonical solution as optimization base
        """
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Find first non-repeating character in string.",
            "question_type": "coding"
        }
        self.session.mock_screen_data = mock_screen
        
        # Turn 1: Screen + Solve this
        res1 = classify_and_understand_utterance("Solve this.", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(res1["utterance_type"], "CODING")
        self.assertEqual(self.session.active_coding_problem, "Find first non-repeating character in string.")
        
        # Simulated LLM code answer
        code_ans = (
            "```python\n"
            "def firstUniqChar(s: str) -> int:\n"
            "    count = Counter(s)\n"
            "    for i, c in enumerate(s):\n"
            "        if count[c] == 1:\n"
            "            return i\n"
            "    return -1\n"
            "```\n"
            "• **Complexity**: Time: O(N) | Space: O(1)"
        )
        self.session.record_qa("Solve this.", code_ans, category="coding")
        self.assertIn("def firstUniqChar", self.session.active_coding_solution)
        
        # Clear mock screen so subsequent turns MUST use session cache
        self.session.mock_screen_data = None

        # Turn 2: Explain your approach
        rel2 = classify_answer_relationship("Explain your approach.", session=self.session)
        self.assertEqual(rel2["strategy_key"], "code_explanation")
        self.session.record_qa("Explain your approach.", "I use a hash map to count character frequencies.", category="code_explanation")
        self.assertIn("def firstUniqChar", self.session.active_coding_solution)

        # Turn 3: Give me pseudocode
        rel3 = classify_answer_relationship("Give me pseudocode.", session=self.session)
        self.assertEqual(rel3["strategy_key"], "pseudocode")
        self.session.record_qa("Give me pseudocode.", "```text\nCount frequencies\nFind first with count 1\n```", category="pseudocode")
        self.assertIn("def firstUniqChar", self.session.active_coding_solution)

        # Turn 4: What's the time complexity?
        rel4 = classify_answer_relationship("What's the time complexity?", session=self.session)
        self.assertEqual(rel4["strategy_key"], "coding_complexity")
        self.session.record_qa("What's the time complexity?", "Time O(N) Space O(1)", category="coding_complexity")
        self.assertIn("def firstUniqChar", self.session.active_coding_solution)

        # Turn 5: Can you optimize it?
        rel5 = classify_answer_relationship("Can you optimize it?", session=self.session)
        self.assertEqual(rel5["strategy_key"], "coding_optimization")

    def test_13_canonical_solution_preservation(self):
        """Verify canonical solution is never overwritten by prose or pseudocode."""
        self.session.active_coding_solution = "def canonicalSolution(): pass"
        self.session.record_qa("Explain it", "Just basic logic explanation", category="code_explanation")
        self.assertEqual(self.session.active_coding_solution, "def canonicalSolution(): pass")
        
        self.session.record_qa("Give pseudocode", "```text\nstep 1\nstep 2\n```", category="pseudocode")
        self.assertEqual(self.session.active_coding_solution, "def canonicalSolution(): pass")

    def test_14_normal_voice_bypasses_screen_capture(self):
        """Verify normal voice question ('What is polymorphism?') does NOT trigger screen capture."""
        with patch("screen_analyzer.capture_screen_image") as mock_capture:
            res = classify_and_understand_utterance(
                utterance="What is polymorphism in Java?",
                session=self.session,
                speaker_hint="Interviewer"
            )
            # Screen capture must NOT be called
            mock_capture.assert_not_called()
            self.assertNotEqual(res["utterance_type"], "CODING")

    def test_15_stale_context_protection_for_voice_questions(self):
        """Verify old screen problem does NOT contaminate subsequent unrelated voice questions."""
        # Set old screen problem
        self.session.screen_context_cache = {
            "data": {
                "is_coding_related": True,
                "problem_statement": "Two Sum Algorithm",
                "code": "def twoSum(): pass",
                "question_type": "coding"
            },
            "timestamp": time.time()
        }
        self.session.active_coding_problem = "Two Sum Algorithm"
        
        # Candidate asks ordinary technical question
        fused = screen_analyzer.fuse_voice_and_screen_question(
            spoken_question="Explain Python decorators.",
            screen_context=self.session.screen_context_cache["data"],
            session=self.session
        )
        
        # Screen context MUST NOT be used
        self.assertFalse(fused["screen_used"])
        self.assertFalse(fused["is_coding"])
        self.assertEqual(fused["refined_question"], "Explain Python decorators.")
        self.assertNotIn("Two Sum", fused["refined_question"])

    def test_16_session_isolation(self):
        """Verify Session A and Session B have completely isolated screen caches."""
        session_a = InterviewSession()
        session_b = InterviewSession()
        
        session_a.screen_context_cache = {
            "data": {"problem_statement": "Problem A: Two Sum"},
            "timestamp": time.time()
        }
        session_b.screen_context_cache = {
            "data": {"problem_statement": "Problem B: Binary Tree Inorder"},
            "timestamp": time.time()
        }
        
        # Verify Session A sees Problem A
        data_a = screen_analyzer.analyze_visible_screen(session=session_a)
        self.assertEqual(data_a["problem_statement"], "Problem A: Two Sum")
        
        # Verify Session B sees Problem B
        data_b = screen_analyzer.analyze_visible_screen(session=session_b)
        self.assertEqual(data_b["problem_statement"], "Problem B: Binary Tree Inorder")
        
        # Clear Session A
        session_a.clear()
        self.assertIsNone(session_a.screen_context_cache)
        
        # Verify Session B is unaffected
        self.assertIsNotNone(session_b.screen_context_cache)
        data_b_after = screen_analyzer.analyze_visible_screen(session=session_b)
        self.assertEqual(data_b_after["problem_statement"], "Problem B: Binary Tree Inorder")

    def test_17_session_clear_invalidates_cache_and_state(self):
        """Verify session.clear() completely wipes screen cache and coding state."""
        self.session.screen_context_cache = {"data": {"problem_statement": "Some Problem"}, "timestamp": time.time()}
        self.session.active_coding_problem = "Some Problem"
        self.session.active_coding_solution = "def solve(): pass"
        self.session.mock_screen_data = {"data": "mock"}
        
        self.session.clear()
        
        self.assertIsNone(self.session.screen_context_cache)
        self.assertIsNone(self.session.mock_screen_data)
        self.assertEqual(self.session.active_coding_problem, "")
        self.assertEqual(self.session.active_coding_solution, "")

    def test_18_no_unnecessary_screen_recapture_during_followups(self):
        """Verify follow-up turns do NOT call capture_screen_image if cache is valid."""
        self.session.screen_context_cache = {
            "data": {
                "is_coding_related": True,
                "problem_statement": "Reverse String",
                "code": "def reverseString(s): pass",
                "question_type": "coding"
            },
            "timestamp": time.time()
        }
        
        with patch("screen_analyzer.capture_screen_image") as mock_capture:
            data = screen_analyzer.analyze_visible_screen(session=self.session)
            mock_capture.assert_not_called()
            self.assertEqual(data["problem_statement"], "Reverse String")

    def test_19_no_global_cache_leakage_when_session_cache_is_none(self):
        """Verify when session exists with empty cache, it does NOT accidentally pick up global cache."""
        session_c = InterviewSession()
        session_c.screen_context_cache = None
        
        # Set global cache
        screen_analyzer._SCREEN_CACHE = {
            "data": {"problem_statement": "Global Leaked Problem"},
            "timestamp": time.time()
        }
        
        with patch("screen_analyzer.capture_screen_image") as mock_capture:
            mock_capture.side_effect = Exception("Stop after capture attempt")
            # Because session_c has no cache, it MUST attempt fresh capture rather than inheriting global cache
            try:
                screen_analyzer.analyze_visible_screen(session=session_c)
            except Exception:
                pass
            mock_capture.assert_called_once()

    def test_20_spelling_and_sql_screen_fusions(self):
        """Verify SQL queries visible on screen are correctly recognized."""
        mock_sql = {
            "is_coding_related": True,
            "language": "SQL",
            "problem_statement": "Find highest salary in each department.",
            "code": "SELECT dept_id, MAX(salary) FROM employees GROUP BY dept_id;",
            "question_type": "sql"
        }
        self.session.mock_screen_data = mock_sql
        res = classify_and_understand_utterance("Can you write a query for this?", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(res["utterance_type"], "SQL")
        self.assertIn("highest salary", res["refined_question"])


if __name__ == "__main__":
    unittest.main()
