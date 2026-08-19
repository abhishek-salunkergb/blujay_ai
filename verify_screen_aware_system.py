"""
Comprehensive Automated, Simulated, and Code-Level Verification Suite
for Parakit Screen-Aware Coding Question Detection & Turn-Taking Engine
=====================================================================
Validates Sections 1 through 18 of the test protocol.
"""

import sys
import os
import time
import json
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine
import classifier
import screen_analyzer
import overlay


class TestScreenAwareCodingSystem(unittest.TestCase):

    def setUp(self):
        """Initialize fresh session before each test."""
        self.session = interview_engine.InterviewSession()
        self.session.screen_aware_coding = True
        self.session.mock_screen_data = None
        # Reset screen cache
        screen_analyzer._SCREEN_CACHE = {"data": None, "timestamp": 0.0}

    # ============================================================
    # SECTION 1: COMPLETE PIPELINE TRACE & CONNECTION VERIFICATION
    # ============================================================
    def test_01_complete_pipeline_trace(self):
        """Traces the real end-to-end call flow from transcript to prompt/answer."""
        raw_transcript = "Can you solve this?"
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Reverse a Singly Linked List in place.",
            "code": "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next",
            "error_message": "",
            "visible_question": "Reverse Linked List",
            "question_type": "coding",
            "confidence": 0.98
        }
        self.session.mock_screen_data = mock_screen

        # Stage 1: Question Detection & Speaker Identification
        speaker_res = interview_engine.detect_speaker_turn(raw_transcript, session=self.session)
        self.assertEqual(speaker_res["speaker"], "INTERVIEWER")

        # Stage 2 & 3: Understanding & Multimodal Voice + Screen Fusion
        fused = interview_engine.classify_and_understand_utterance(raw_transcript, session=self.session)
        self.assertTrue(fused["should_generate_answer"])
        self.assertEqual(fused["utterance_type"], "CODING")
        self.assertIn("Reverse a Singly Linked List", fused["refined_question"])

        # Stage 4: Strategy Mapping
        strategy = classifier.STRATEGY_MAP.get(fused["utterance_type"].lower())
        self.assertIsNotNone(strategy)
        self.assertIn("CODING INTERVIEW STRATEGY", strategy)

        # Stage 5: System Prompt Construction
        system_prompt = interview_engine.build_interview_system_prompt(self.session, fused["utterance_type"].lower())
        self.assertIn("CODING INTERVIEW STRATEGY", system_prompt)
        self.assertIn("APPROACH", system_prompt)
        self.assertIn("COMPLEXITY", system_prompt)

    # ============================================================
    # SECTION 2: TEST VOICE-ONLY CODING QUESTIONS
    # ============================================================
    def test_02_voice_only_coding_questions(self):
        """Tests all 8 specified voice-only coding questions without screen dependence."""
        test_cases = [
            ("Write a function to reverse a linked list.", "CODING", "coding"),
            ("Find the second largest element in an array.", "CODING", "coding"),
            ("Implement binary search.", "CODING", "coding"),
            ("How would you solve this dynamic programming problem?", "CODING", "coding"),
            ("Write a SQL query to find the second highest salary.", "SQL", "sql"),
            ("What is the time complexity of binary search?", "CODING_COMPLEXITY", "coding_complexity"),
            ("What's wrong with this code?", "DEBUGGING", "debugging"),
            ("Can you optimize this solution?", "CODING_OPTIMIZATION", "coding_optimization"),
        ]

        for utterance, expected_type, expected_strat_key in test_cases:
            with self.subTest(utterance=utterance):
                # Ensure no mock screen is provided
                self.session.mock_screen_data = None
                
                # Check visual trigger requirement
                is_visual = screen_analyzer.is_visual_reference_question(utterance)
                # Self-contained voice questions like "Write a function...", "Implement binary search..." do NOT require screen
                if "what's wrong" not in utterance.lower():
                    self.assertFalse(is_visual, f"'{utterance}' should not require screen capture")

                res = interview_engine.classify_and_understand_utterance(utterance, session=self.session)
                self.assertTrue(res["should_generate_answer"], f"Failed to detect question for: {utterance}")
                self.assertEqual(res["utterance_type"], expected_type)
                
                # Verify correct strategy exists
                strategy = classifier.STRATEGY_MAP.get(expected_strat_key)
                self.assertIsNotNone(strategy, f"Missing strategy for: {expected_strat_key}")

    # ============================================================
    # SECTION 3: TEST SCREEN-DEPENDENT QUESTIONS
    # ============================================================
    def test_03_screen_dependent_questions(self):
        """Tests Tests A, B, C with simulated screen contexts."""
        # Test A: "Can you solve this?" + Longest Substring on screen
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Find the longest substring without repeating characters.",
            "code": "def lengthOfLongestSubstring(s: str) -> int:",
            "error_message": "",
            "visible_question": "Longest Substring Without Repeating Characters",
            "question_type": "coding",
            "confidence": 0.97
        }
        res_a = interview_engine.classify_and_understand_utterance("Can you solve this?", session=self.session)
        self.assertTrue(res_a["should_generate_answer"])
        self.assertEqual(res_a["utterance_type"], "CODING")
        self.assertIn("longest substring", res_a["refined_question"].lower())

        # Test B: "What's wrong with this?" + two_sum error on screen
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Two Sum problem",
            "code": "def two_sum(nums, target):\n    for i in range(len(nums)):\n        if nums[i] == target: return i",
            "error_message": "IndexError: list index out of range",
            "visible_question": "Two Sum bug",
            "question_type": "debugging",
            "confidence": 0.95
        }
        res_b = interview_engine.classify_and_understand_utterance("What's wrong with this?", session=self.session)
        self.assertTrue(res_b["should_generate_answer"])
        self.assertEqual(res_b["utterance_type"], "DEBUGGING")
        self.assertIn("IndexError", res_b["refined_question"])

        # Test C: "How would you get the second highest salary?" + SQL on screen
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "language": "SQL",
            "problem_statement": "Employee salary table",
            "code": "SELECT * FROM employees;",
            "error_message": "",
            "visible_question": "Second highest salary query",
            "question_type": "sql",
            "confidence": 0.96
        }
        res_c = interview_engine.classify_and_understand_utterance("How would you get the second highest salary?", session=self.session)
        self.assertTrue(res_c["should_generate_answer"])
        self.assertEqual(res_c["utterance_type"], "SQL")

    # ============================================================
    # SECTION 4: TEST NON-CODING QUESTIONS WITH CODE VISIBLE (PRIORITY)
    # ============================================================
    def test_04_non_coding_questions_with_code_visible(self):
        """Verifies that spoken intent strictly overrides visible background code."""
        # Code visible on screen
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Calculate order total with discount",
            "code": "def calculate_total(items, discount):\n    return sum(items) * (1 - discount)",
            "error_message": "",
            "visible_question": "Calculate Total",
            "question_type": "coding",
            "confidence": 0.95
        }

        non_coding_prompts = [
            ("Why did you choose Django for this project?", "PROJECT"),
            ("Tell me about yourself.", "HR"),
            ("Why did you choose React?", "TECHNICAL"),
            ("Describe a difficult problem you solved.", "BEHAVIORAL")
        ]

        for utterance, expected_family in non_coding_prompts:
            with self.subTest(utterance=utterance):
                res = interview_engine.classify_and_understand_utterance(utterance, session=self.session)
                self.assertTrue(res["should_generate_answer"])
                self.assertNotEqual(res["utterance_type"], "CODING", f"'{utterance}' must NOT be overridden to CODING!")
                self.assertNotEqual(res["utterance_type"], "DEBUGGING")
                self.assertNotEqual(res["utterance_type"], "SQL")

    # ============================================================
    # SECTION 5: TEST CANDIDATE SPEECH (PROTECTION)
    # ============================================================
    def test_05_candidate_speech_protection(self):
        """Verifies candidate answers never trigger AI generation or screen analysis."""
        candidate_utterances = [
            "I would use a hash map for this.",
            "I implemented this using Redis.",
            "We used Django REST Framework.",
            "I would solve this using dynamic programming.",
            "In my previous company, I built the recommendation engine.",
            "Yes, I chose PostgreSQL for ACID compliance."
        ]

        for cand_text in candidate_utterances:
            with self.subTest(cand_text=cand_text):
                res = interview_engine.classify_and_understand_utterance(cand_text, session=self.session)
                self.assertFalse(res["should_generate_answer"], f"Candidate speech triggered answer: {cand_text}")
                self.assertEqual(res["speaker"], "Candidate")
                # Ensure candidate response was buffered as context
                self.assertIn(cand_text, self.session.last_candidate_response)

    # ============================================================
    # SECTION 6: TEST RAPID FOLLOW-UP CODING QUESTIONS
    # ============================================================
    def test_06_rapid_follow_up_coding_questions(self):
        """Tests full multi-turn coding sequence with context reuse."""
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Find the second largest element in an array.",
            "code": "def second_largest(arr):",
            "error_message": "",
            "visible_question": "Second largest element",
            "question_type": "coding",
            "confidence": 0.95
        }

        # Turn 1: Interviewer: "How would you solve this?"
        t1 = interview_engine.classify_and_understand_utterance("How would you solve this?", session=self.session)
        self.assertTrue(t1["should_generate_answer"])
        self.assertEqual(t1["utterance_type"], "CODING")
        self.session.qa_pairs.append({"question": t1["refined_question"], "answer": "I would track the largest and second largest elements in a single O(N) pass."})

        # Turn 2: Candidate: "I'd use a hash map."
        t2 = interview_engine.classify_and_understand_utterance("I'd use a hash map.", session=self.session)
        self.assertFalse(t2["should_generate_answer"])
        self.assertEqual(t2["speaker"], "Candidate")

        # Turn 3: Interviewer: "What is the time complexity?"
        t3 = interview_engine.classify_and_understand_utterance("What is the time complexity?", session=self.session)
        self.assertTrue(t3["should_generate_answer"])
        self.assertEqual(t3["utterance_type"], "CODING_COMPLEXITY")

        # Turn 4: Interviewer: "Can you optimize it?"
        t4 = interview_engine.classify_and_understand_utterance("Can you optimize it?", session=self.session)
        self.assertTrue(t4["should_generate_answer"])
        self.assertEqual(t4["utterance_type"], "CODING_OPTIMIZATION")

    # ============================================================
    # SECTION 7: TEST SCREEN CACHE TTL (PROGRAMMATIC)
    # ============================================================
    def test_07_screen_cache_ttl(self):
        """Verifies programmatic screen cache TTL (20 seconds)."""
        screen_analyzer._SCREEN_CACHE = {"data": None, "timestamp": 0.0}
        
        sample_screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Cache Test Problem",
            "code": "def test(): pass",
            "error_message": "",
            "visible_question": "Cache Test",
            "question_type": "coding",
            "confidence": 0.99
        }

        # t = 0: Initial capture & cache
        t0 = time.time()
        with patch("time.time", return_value=t0):
            res_t0 = screen_analyzer.analyze_visible_screen(mock_screen_data=sample_screen_data)
            self.assertEqual(res_t0["problem_statement"], "Cache Test Problem")
            self.assertEqual(screen_analyzer._SCREEN_CACHE["timestamp"], t0)

        # t = 5: Should reuse cache
        with patch("time.time", return_value=t0 + 5.0):
            res_t5 = screen_analyzer.analyze_visible_screen(client=None)
            self.assertEqual(res_t5["problem_statement"], "Cache Test Problem")

        # t = 19: Should still reuse cache (19s < 20s TTL)
        with patch("time.time", return_value=t0 + 19.0):
            res_t19 = screen_analyzer.analyze_visible_screen(client=None)
            self.assertEqual(res_t19["problem_statement"], "Cache Test Problem")

        # t = 21: Cache expired! (21s > 20s TTL)
        with patch("time.time", return_value=t0 + 21.0):
            # Without client/mock, returns empty default result
            res_t21 = screen_analyzer.analyze_visible_screen(client=None)
            self.assertFalse(res_t21["is_coding_related"])
            self.assertEqual(res_t21["problem_statement"], "")

    # ============================================================
    # SECTION 8: TEST PARAKIT SELF-CAPTURE EXCLUSION
    # ============================================================
    def test_08_parakit_self_capture_exclusion(self):
        """Verifies SetWindowDisplayAffinity and WDA_EXCLUDEFROMCAPTURE."""
        self.assertEqual(overlay.WDA_EXCLUDEFROMCAPTURE, 0x00000011)
        self.assertEqual(overlay.WDA_MONITOR, 0x00000001)

        # Verify hide_from_screen_capture function is properly defined and handles missing HWND safely
        self.assertFalse(overlay.hide_from_screen_capture(None))

    # ============================================================
    # SECTION 9: TEST SCREEN-CAPTURE FAILURE FALLBACK
    # ============================================================
    def test_09_screen_capture_failure_fallback(self):
        """Verifies graceful degradation when screen capture throws exception."""
        with patch("screen_analyzer.capture_screen_image", side_effect=RuntimeError("Display grab failed")):
            # Should not throw, returns fallback dict
            res = screen_analyzer.analyze_visible_screen(client=MagicMock())
            self.assertFalse(res["is_coding_related"])

        # End-to-end question processing continues via voice context
        res_q = interview_engine.classify_and_understand_utterance("Write a function to reverse a linked list.", session=self.session)
        self.assertTrue(res_q["should_generate_answer"])
        self.assertEqual(res_q["utterance_type"], "CODING")

    # ============================================================
    # SECTION 10: TEST VISION / OCR FAILURE FALLBACK
    # ============================================================
    def test_10_vision_ocr_failure_fallback(self):
        """Verifies voice-only fallback when vision API returns invalid or empty JSON."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Vision API timeout")

        with patch("screen_analyzer.capture_screen_image", return_value="dummy.png"), \
             patch("screen_analyzer.encode_image_base64", return_value="abc"), \
             patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            res = screen_analyzer.analyze_visible_screen(client=mock_client)
            self.assertFalse(res["is_coding_related"])

        # Voice question processing remains resilient
        res_q = interview_engine.classify_and_understand_utterance("Explain the concept of REST API.", session=self.session)
        self.assertTrue(res_q["should_generate_answer"])
        self.assertEqual(res_q["utterance_type"], "TECHNICAL")

    # ============================================================
    # SECTION 11: TEST SCREEN-AWARE CODING TOGGLE & PERSISTENCE
    # ============================================================
    def test_11_screen_aware_coding_toggle_and_persistence(self):
        """Verifies toggle ON/OFF behavior and configuration persistence."""
        # Toggle OFF
        self.session.set_screen_aware_coding(False)
        self.assertFalse(self.session.screen_aware_coding)
        
        # Verify persistence
        saved_cfg = interview_engine.load_user_config()
        self.assertFalse(saved_cfg.get("screen_aware_coding", True))

        # With toggle OFF, screen analysis is bypassed
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "problem_statement": "Find second largest element",
            "question_type": "coding"
        }
        res_off = interview_engine.classify_and_understand_utterance("Tell me about yourself.", session=self.session)
        self.assertEqual(res_off["utterance_type"], "HR")

        # Toggle back ON
        self.session.set_screen_aware_coding(True)
        self.assertTrue(self.session.screen_aware_coding)
        saved_cfg_on = interview_engine.load_user_config()
        self.assertTrue(saved_cfg_on.get("screen_aware_coding", False))

    # ============================================================
    # SECTION 12: TEST QUESTION PRIORITY MATRIX
    # ============================================================
    def test_12_question_priority_matrix(self):
        """Verifies voice intent vs screen context priority rules."""
        # Rule 1: Clear spoken question + irrelevant screen code -> Voice intent wins
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Random LeetCode Problem",
            "code": "def solve(): pass",
            "question_type": "coding"
        }
        res_voice_wins = interview_engine.classify_and_understand_utterance("Why did you choose PostgreSQL over MongoDB?", session=self.session)
        self.assertEqual(res_voice_wins["utterance_type"], "DATABASE")

        # Rule 2: Ambiguous spoken question + coding screen -> Screen context supplements voice
        res_screen_supp = interview_engine.classify_and_understand_utterance("Can you solve this?", session=self.session)
        self.assertEqual(res_screen_supp["utterance_type"], "CODING")

    # ============================================================
    # SECTION 14: TEST CONTINUOUS INTERVIEW SIMULATION
    # ============================================================
    def test_14_continuous_interview_simulation(self):
        """Simulates full realistic multi-turn interview sequence."""
        # Turn 1: Interviewer: "Tell me about your project."
        t1 = interview_engine.classify_and_understand_utterance("Tell me about your project.", session=self.session)
        self.assertTrue(t1["should_generate_answer"])
        self.assertEqual(t1["utterance_type"], "PROJECT")

        # Turn 2: Candidate: "I built a React and Django application."
        t2 = interview_engine.classify_and_understand_utterance("I built a React and Django application.", session=self.session)
        self.assertFalse(t2["should_generate_answer"])
        self.assertEqual(t2["speaker"], "Candidate")

        # Turn 3: Interviewer: "Why React?"
        t3 = interview_engine.classify_and_understand_utterance("Why React?", session=self.session)
        self.assertTrue(t3["should_generate_answer"])
        self.assertIn("React", t3["refined_question"])

        # Turn 4: Candidate: "I chose React because of virtual DOM performance."
        t4 = interview_engine.classify_and_understand_utterance("I chose React because of virtual DOM performance.", session=self.session)
        self.assertFalse(t4["should_generate_answer"])

        # Turn 5: Interviewer: "Can you solve this?" (Screen contains problem)
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Find the second largest element in an array.",
            "code": "def second_largest(arr):",
            "error_message": "",
            "question_type": "coding",
            "confidence": 0.96
        }
        t5 = interview_engine.classify_and_understand_utterance("Can you solve this?", session=self.session)
        self.assertTrue(t5["should_generate_answer"])
        self.assertEqual(t5["utterance_type"], "CODING")
        self.assertIn("second largest", t5["refined_question"].lower())

        # Turn 6: Candidate: "I would sort it or use two pointers."
        t6 = interview_engine.classify_and_understand_utterance("I would sort it or use two pointers.", session=self.session)
        self.assertFalse(t6["should_generate_answer"])

        # Turn 7: Interviewer: "What is the time complexity?"
        t7 = interview_engine.classify_and_understand_utterance("What is the time complexity?", session=self.session)
        self.assertTrue(t7["should_generate_answer"])
        self.assertEqual(t7["utterance_type"], "CODING_COMPLEXITY")

    # ============================================================
    # SECTION 15: VERIFY GENERATED PROMPTS & STRATEGY INJECTION
    # ============================================================
    def test_15_verify_generated_prompts(self):
        """Verifies runtime prompt construction uses exact strategy templates."""
        strategies = {
            "coding": "CODING INTERVIEW STRATEGY",
            "debugging": "DEBUGGING & CODE REVIEW STRATEGY",
            "sql": "SQL QUERY STRATEGY",
            "coding_complexity": "CODING COMPLEXITY ONLY STRATEGY",
            "coding_optimization": "CODING OPTIMIZATION STRATEGY",
            "technical": "TECHNICAL STRATEGY",
            "behavioral": "BEHAVIORAL (STAR) STRATEGY"
        }

        for q_type, expected_header in strategies.items():
            with self.subTest(q_type=q_type):
                prompt = interview_engine.build_interview_system_prompt(self.session, q_type)
                self.assertIn(expected_header, prompt)

    # ============================================================
    # SECTION 16: TEST DUPLICATE PREVENTION
    # ============================================================
    def test_16_duplicate_prevention(self):
        """Verifies identical and slightly variant questions are deduplicated."""
        self.session.mock_screen_data = {
            "is_coding_related": True,
            "problem_statement": "Reverse string",
            "question_type": "coding"
        }

        # First question
        q1 = interview_engine.classify_and_understand_utterance("Can you solve this?", session=self.session)
        q1_id = q1["question_id"]
        self.session.latest_detected_question = q1["refined_question"]

        # Duplicate questions with punctuation differences
        q2 = interview_engine.classify_and_understand_utterance("Can you solve this", session=self.session)
        q3 = interview_engine.classify_and_understand_utterance("Can you solve this...", session=self.session)

        self.assertEqual(q1["refined_question"], q2["refined_question"])
        self.assertEqual(q1["refined_question"], q3["refined_question"])

    # ============================================================
    # SECTION 17: TEST STALE ANSWER CANCELLATION MECHANISM
    # ============================================================
    def test_17_stale_answer_cancellation(self):
        """Verifies generation cancellation flag when a newer question arrives."""
        generation_cancelled = False

        def is_cancelled_check():
            return generation_cancelled

        # Stream Question A
        mock_client = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="Processing Answer A..."))]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        tokens_received = []
        def on_token(t):
            tokens_received.append(t)

        # Start streaming Answer A
        ans_a = interview_engine.generate_interview_answer(
            client=mock_client,
            session=self.session,
            question="Question A",
            on_token=on_token,
            is_cancelled_callback=is_cancelled_check
        )
        self.assertIn("Processing Answer A", ans_a)

        # Now simulate arrival of Question B (which cancels Question A)
        generation_cancelled = True
        ans_cancelled = interview_engine.generate_interview_answer(
            client=mock_client,
            session=self.session,
            question="Question B",
            on_token=on_token,
            is_cancelled_callback=is_cancelled_check
        )
        # Should terminate immediately upon seeing cancellation callback return True
        self.assertEqual(ans_cancelled, "")

    # ============================================================
    # SECTION 18: PERFORMANCE & OVERHEAD BENCHMARK
    # ============================================================
    def test_18_performance_and_overhead_benchmark(self):
        """Benchmarks decision latency with screen_aware_coding ON vs OFF."""
        normal_q = "Why did you choose Django for this project?"
        
        # Latency with screen_aware_coding = OFF
        self.session.screen_aware_coding = False
        t_start = time.perf_counter()
        for _ in range(100):
            interview_engine.classify_and_understand_utterance(normal_q, session=self.session)
        t_off_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

        # Latency with screen_aware_coding = ON (normal question should bypass screen capture)
        self.session.screen_aware_coding = True
        t_start = time.perf_counter()
        for _ in range(100):
            interview_engine.classify_and_understand_utterance(normal_q, session=self.session)
        t_on_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

        print(f"\n[Performance Benchmark]")
        print(f"  Normal Question Latency (Screen-Aware OFF): {t_off_ms:.4f} ms")
        print(f"  Normal Question Latency (Screen-Aware ON) : {t_on_ms:.4f} ms")
        
        # Decision latency should be sub-millisecond
        self.assertLess(t_on_ms, 5.0, "Decision latency must be < 5ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
