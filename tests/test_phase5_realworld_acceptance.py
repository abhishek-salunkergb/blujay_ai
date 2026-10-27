import unittest
import time
from unittest.mock import patch, MagicMock

import interview_engine
from interview_engine import (
    InterviewSession,
    classify_and_understand_utterance,
    classify_answer_relationship,
    generate_interview_answer
)
import screen_analyzer
import classifier


class TestPhase5RealWorldAcceptance(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()

    # =========================================================================
    # SCENARIO 1: FULL 7-TURN REAL-WORLD MULTI-TURN INTERVIEW
    # =========================================================================
    def test_01_full_seven_turn_interview_sequence(self):
        """
        Turn 1: Screen Problem -> 'Solve this problem' -> Grounding & Screen Cache Populated
        Turn 2: 'Give me the code' -> CODE_IMPLEMENTATION / Coding -> Canonical solution created
        Turn 3: 'Explain your approach' -> CODE_EXPLANATION -> Same active problem, solution preserved
        Turn 4: 'Give me pseudocode' -> PSEUDOCODE -> Same active problem, solution preserved
        Turn 5: "What's the time and space complexity?" -> COMPLEXITY -> Same solution
        Turn 6: 'Can you optimize it?' -> OPTIMIZATION -> Same solution base
        Turn 7: 'Explain Python decorators.' -> NEW_TOPIC -> Stale screen context NOT injected
        """

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

        screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Given an array of integers nums and an integer target, return indices of the two numbers such赋 that they add up to target.",
            "problem_title": "Two Sum",
            "code": "def twoSum(nums: List[int], target: int) -> List[int]:\n    pass",
            "question_type": "coding"
        }
        self.session.mock_screen_data = screen_data

        # TURN 1: SCREEN PROBLEM GROUNDING
        res1 = classify_and_understand_utterance("Solve this problem.", session=self.session, speaker_hint="Interviewer")
        self.assertEqual(res1["utterance_type"], "CODING")
        self.assertIn("Two Sum", self.session.active_coding_problem)
        self.assertIsNotNone(self.session.screen_context_cache)
        self.assertEqual(self.session.active_coding_language, "Python")

        # TURN 2: CODE IMPLEMENTATION
        # Screen was captured once; clear mock so subsequent turns MUST use session cache
        self.session.mock_screen_data = None
        rel2 = classify_answer_relationship("Give me the code.", session=self.session)
        self.assertEqual(rel2["relationship_type"], "CODE_IMPLEMENTATION")
        self.assertIn("Two Sum", rel2["fused_question"])

        canonical_code = (
            "Here is the optimal O(N) Python solution:\n"
            "```python\n"
            "def twoSum(nums: list[int], target: int) -> list[int]:\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        comp = target - num\n"
            "        if comp in seen:\n"
            "            return [seen[comp], i]\n"
            "        seen[num] = i\n"
            "    return []\n"
            "```\n"
            "• **Complexity**: Time: O(N) | Space: O(N)"
        )
        self.session.record_qa(rel2["fused_question"], canonical_code, category="coding")
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # TURN 3: EXPLAIN APPROACH
        rel3 = classify_answer_relationship("Explain your approach.", session=self.session)
        self.assertEqual(rel3["relationship_type"], "CODE_EXPLANATION")
        self.assertIn("Two Sum", rel3["fused_question"])
        explanation_prose = "I use a single-pass hash map to store previously seen numbers and look up complements in O(1) time."
        self.session.record_qa(rel3["fused_question"], explanation_prose, category="code_explanation")
        # Canonical solution must NOT be overwritten
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # TURN 4: PSEUDOCODE
        rel4 = classify_answer_relationship("Give me pseudocode.", session=self.session)
        self.assertEqual(rel4["relationship_type"], "PSEUDOCODE")
        self.assertIn("Two Sum", rel4["fused_question"])
        pseudo_prose = "```text\nCreate hash map\nFor index, val in enumerate(nums):\n  if target-val in map: return [map[target-val], index]\n  map[val] = index\n```"
        self.session.record_qa(rel4["fused_question"], pseudo_prose, category="pseudocode")
        # Canonical solution must NOT be overwritten
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # TURN 5: COMPLEXITY
        rel5 = classify_answer_relationship("What's the time and space complexity?", session=self.session)
        self.assertEqual(rel5["relationship_type"], "COMPLEXITY")
        self.assertIn("Two Sum", rel5["fused_question"])
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # TURN 6: OPTIMIZATION
        rel6 = classify_answer_relationship("Can you optimize it?", session=self.session)
        self.assertEqual(rel6["relationship_type"], "OPTIMIZATION")
        self.assertIn("Two Sum", rel6["fused_question"])
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # TURN 7: NEW UNRELATED TOPIC (STALE CONTEXT ACCEPTANCE TEST)
        res7 = classify_and_understand_utterance("Explain Python decorators.", session=self.session, speaker_hint="Interviewer")
        self.assertNotEqual(res7["utterance_type"], "CODING")
        self.assertNotIn("Two Sum", res7["refined_question"])
        self.assertIn("Explain Python decorators", res7["refined_question"])

    # =========================================================================
    # SCENARIO 2: NORMAL VOICE INTERVIEW QUESTIONS (ZERO SCREEN CAPTURES)
    # =========================================================================
    def test_02_normal_voice_interview_bypasses_all_visual_analysis(self):
        """Verify normal voice questions trigger 0 screen captures and 0 visual analyses."""
        voice_questions = [
            "What is polymorphism?",
            "Explain Python decorators.",
            "What is dependency injection in software architecture?",
            "How does garbage collection work in modern runtimes?",
            "Tell me about REST APIs versus GraphQL."
        ]
        
        with patch("screen_analyzer.capture_screen_image") as mock_capture:
            with patch("screen_analyzer.analyze_visible_screen") as mock_analyze:
                for q in voice_questions:
                    res = classify_and_understand_utterance(q, session=self.session, speaker_hint="Interviewer")
                    self.assertFalse(res.get("screen_used", False))
                mock_capture.assert_not_called()
                mock_analyze.assert_not_called()

    # =========================================================================
    # SCENARIO 3: VISUAL REFERENCE VS CONVERSATIONAL FOLLOW-UP DISCRIMINATION
    # =========================================================================
    def test_03_visual_reference_discrimination(self):
        """Verify explicit visual references trigger screen detection while ordinary follow-ups do not."""
        # Visual references
        self.assertTrue(screen_analyzer.is_visual_reference_question("Solve this."))
        self.assertTrue(screen_analyzer.is_visual_reference_question("Implement this problem."))
        self.assertTrue(screen_analyzer.is_visual_reference_question("Look at the screen and explain this code."))
        self.assertTrue(screen_analyzer.is_visual_reference_question("What's wrong with this?"))

        # Non-visual conversational sentences (even containing pronouns)
        self.assertFalse(screen_analyzer.is_visual_reference_question("What is this design pattern?"))
        self.assertFalse(screen_analyzer.is_visual_reference_question("How does this algorithm work in theory?"))
        self.assertFalse(screen_analyzer.is_visual_reference_question("Why did you choose that database?"))

    # =========================================================================
    # SCENARIO 4: RAPID QUESTION INTERRUPTION & GENERATION ID ISOLATION
    # =========================================================================
    def test_04_rapid_question_interruption_packet_discard(self):
        """Simulate generation ID tracking during rapid question interruption."""
        gen_state = {
            "active_generation_id": 100,
            "buffer": ""
        }

        def stream_token(token: str, gen_id: int):
            if gen_id != gen_state["active_generation_id"]:
                return  # Dropped late token
            gen_state["buffer"] += token

        # Generation 100 streaming
        stream_token("Answer 100 token A ", 100)
        stream_token("Answer 100 token B ", 100)
        self.assertEqual(gen_state["buffer"], "Answer 100 token A Answer 100 token B ")

        # Rapid Interruption: Question 101 arrives
        gen_state["active_generation_id"] = 101
        gen_state["buffer"] = ""

        # Late token from Generation 100 arrives
        stream_token("late token from 100", 100)
        self.assertEqual(gen_state["buffer"], "")  # Intact and empty

        # Token for Generation 101 arrives
        stream_token("Answer 101 first token", 101)
        self.assertEqual(gen_state["buffer"], "Answer 101 first token")

    # =========================================================================
    # SCENARIO 5: SESSION ISOLATION & SESSION CLEAR RESET
    # =========================================================================
    def test_05_session_isolation_and_clear_reset(self):
        """Verify Session A and Session B never leak state, and clear() resets completely."""
        session_a = InterviewSession()
        session_b = InterviewSession()

        session_a.screen_context_cache = {"data": {"problem_statement": "Problem A: LRU Cache"}, "timestamp": time.time()}
        session_a.active_coding_problem = "Problem A: LRU Cache"
        session_a.active_coding_solution = "def lru(): pass"

        session_b.screen_context_cache = {"data": {"problem_statement": "Problem B: Trie Tree"}, "timestamp": time.time()}
        session_b.active_coding_problem = "Problem B: Trie Tree"
        session_b.active_coding_solution = "def trie(): pass"

        # Verify isolation
        self.assertEqual(session_a.active_coding_problem, "Problem A: LRU Cache")
        self.assertEqual(session_b.active_coding_problem, "Problem B: Trie Tree")

        # Clear Session A
        session_a.clear()
        self.assertIsNone(session_a.screen_context_cache)
        self.assertEqual(session_a.active_coding_problem, "")
        self.assertEqual(session_a.active_coding_solution, "")

        # Session B must remain completely intact
        self.assertIsNotNone(session_b.screen_context_cache)
        self.assertEqual(session_b.active_coding_problem, "Problem B: Trie Tree")
        self.assertEqual(session_b.active_coding_solution, "def trie(): pass")

    # =========================================================================
    # SCENARIO 6: 20-SECOND SCREEN CACHE TTL BEHAVIOR
    # =========================================================================
    def test_06_screen_cache_ttl_and_recapture(self):
        """Verify cache reuse within 20s and fresh capture after 20s."""
        base_time = 5000.0
        self.session.screen_context_cache = {
            "data": {"problem_statement": "Fibonacci Dynamic Programming"},
            "timestamp": base_time
        }

        # Test at t = 10.0s (Reuse)
        with patch("time.time", return_value=base_time + 10.0):
            with patch("screen_analyzer.capture_screen_image") as mock_cap:
                data = screen_analyzer.analyze_visible_screen(session=self.session)
                mock_cap.assert_not_called()
                self.assertEqual(data["problem_statement"], "Fibonacci Dynamic Programming")

        # Test at t = 19.9s (Reuse)
        with patch("time.time", return_value=base_time + 19.9):
            with patch("screen_analyzer.capture_screen_image") as mock_cap:
                data = screen_analyzer.analyze_visible_screen(session=self.session)
                mock_cap.assert_not_called()
                self.assertEqual(data["problem_statement"], "Fibonacci Dynamic Programming")

        # Test at t = 20.1s (Expired -> Recaptures)
        with patch("time.time", return_value=base_time + 20.1):
            with patch("screen_analyzer.capture_screen_image") as mock_cap:
                mock_cap.side_effect = Exception("Stop after capture")
                try:
                    screen_analyzer.analyze_visible_screen(session=self.session)
                except Exception:
                    pass
                mock_cap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
