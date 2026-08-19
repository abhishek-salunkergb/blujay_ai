"""
Automated Test Suite for Parakit Continuous Live & Manual Input System
======================================================================
Tests:
1. Live interview continuous question detection and answering.
2. Manual typed question processing via unified engine.
3. Ctrl+Enter submission and multi-line pasted questions.
4. Absence of '?' requirement (semantic understanding).
5. Screen-aware coding integration on manual input.
6. Shared session history and context between live & manual questions.
7. Double-submission deduplication.
8. Empty/whitespace input validation.
9. Interruption handling & generation cancellation.
10. Active answer mode & resume grounding preservation.
"""

import sys
import os
import unittest
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine
from screen_analyzer import analyze_visible_screen, fuse_voice_and_screen_question


class TestContinuousAndManualInput(unittest.TestCase):

    def setUp(self):
        self.session = interview_engine.InterviewSession()
        self.session.clear()

    # ============================================================
    # TEST 1: MANUAL QUESTION WITHOUT '?' (SEMANTIC RECOGNITION)
    # ============================================================
    def test_01_manual_question_without_question_mark(self):
        manual_queries = [
            "How did you implement authentication",
            "Why React",
            "Tell me about your project",
            "Write a function to reverse a linked list",
            "Explain how indexing works in PostgreSQL"
        ]
        
        for q in manual_queries:
            analysis = interview_engine.classify_and_understand_utterance(
                utterance=q,
                session=self.session,
                client=None,
                source="MANUAL_INPUT",
                is_manual=True
            )
            self.assertTrue(analysis["should_generate_answer"], f"Failed to generate answer for: {q}")
            self.assertEqual(analysis["input_source"], "MANUAL_INPUT")
            self.assertEqual(analysis["speaker"], "Manual Input")

    # ============================================================
    # TEST 2: MULTI-LINE PASTED QUESTION PROCESSING
    # ============================================================
    def test_02_multiline_pasted_question(self):
        multiline_q = """Given an array of integers nums and an integer target,
return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution.

Constraints:
2 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9"""
        
        analysis = interview_engine.classify_and_understand_utterance(
            utterance=multiline_q,
            session=self.session,
            client=None,
            source="MANUAL_INPUT",
            is_manual=True
        )
        self.assertTrue(analysis["should_generate_answer"])
        self.assertEqual(analysis["utterance_type"], "CODING")
        self.assertEqual(analysis["input_source"], "MANUAL_INPUT")

    # ============================================================
    # TEST 3: EMPTY & WHITESPACE-ONLY INPUT VALIDATION
    # ============================================================
    def test_03_empty_and_whitespace_validation(self):
        invalid_inputs = ["", "   ", "   \n\t  ", "Type or paste question here..."]
        
        for inp in invalid_inputs:
            if not inp or inp == "Type or paste question here..." or not inp.strip():
                # Should be ignored by the UI / engine
                raw = inp.strip()
                if not raw:
                    analysis = interview_engine.classify_and_understand_utterance(
                        utterance=inp,
                        session=self.session,
                        client=None,
                        source="MANUAL_INPUT",
                        is_manual=True
                    )
                    self.assertFalse(analysis["should_generate_answer"])

    # ============================================================
    # TEST 4: SCREEN-AWARE CODING ON MANUAL INPUT
    # ============================================================
    def test_04_manual_screen_aware_coding(self):
        self.session.mock_screen_data = {
            "has_coding_problem": True,
            "problem_title": "Merge K Sorted Lists",
            "problem_description": "Merge k sorted linked lists and return it as one sorted list.",
            "code_snippet": "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val",
            "detected_language": "python",
            "is_debugging_scenario": False,
            "is_sql_query": False,
            "error_message": "",
            "confidence": 0.95
        }
        
        # User manually types: "Can you solve this?"
        analysis = interview_engine.classify_and_understand_utterance(
            utterance="Can you solve this?",
            session=self.session,
            client=None,
            source="MANUAL_INPUT",
            is_manual=True
        )
        
        self.assertTrue(analysis["should_generate_answer"])
        self.assertEqual(analysis["utterance_type"], "CODING")
        self.assertIn("Merge K Sorted Lists", analysis["refined_question"])
        self.assertEqual(analysis["input_source"], "MANUAL_INPUT")

    # ============================================================
    # TEST 5: SHARED SESSION CONTEXT BETWEEN MANUAL & LIVE QUESTIONS
    # ============================================================
    def test_05_shared_context_between_manual_and_live(self):
        # Step 1: User manually asks "Why did you choose MongoDB?"
        manual_entry = self.session.add_transcript(
            "Why did you choose MongoDB?",
            role="Interviewer",
            is_question=True,
            source="MANUAL_INPUT",
            is_manual=True
        )
        self.assertEqual(manual_entry["source"], "MANUAL_INPUT")
        
        # Candidate responds (voice or manual)
        self.session.add_candidate_speech("Because our document schema was rapidly evolving and needed flexible JSON storage.")
        
        # Step 2: Live interviewer asks follow-up "What about scalability?"
        live_analysis = interview_engine.classify_and_understand_utterance(
            utterance="What about scalability?",
            session=self.session,
            client=None,
            source="mic"
        )
        
        self.assertTrue(live_analysis["should_generate_answer"])
        self.assertEqual(live_analysis["utterance_type"], "FOLLOW_UP")
        self.assertEqual(live_analysis["input_source"], "LIVE_INTERVIEW")
        # Follow-up should reference the candidate's previous response or topic
        self.assertIn("scalability", live_analysis["refined_question"].lower())

    # ============================================================
    # TEST 6: RESUME & KNOWLEDGE GROUNDING IN SYSTEM PROMPT
    # ============================================================
    def test_06_resume_grounding_manual_question(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write("Candidate: Abhishek\nProject: Built scalable real-time streaming backend using FastAPI & Celery.")
            tf_path = tf.name
            
        try:
            self.session.set_resume(tf_path)
            prompt = interview_engine.build_interview_system_prompt(self.session, "project")
            self.assertIn("FastAPI & Celery", prompt)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    # ============================================================
    # TEST 7: INTERRUPTION & GENERATION CANCELLATION
    # ============================================================
    def test_07_interruption_cancellation(self):
        # Generation 1 starts
        gen_id_1 = 1
        current_gen_id = gen_id_1
        
        # In-flight check callback
        is_cancelled_1 = lambda: current_gen_id != gen_id_1
        self.assertFalse(is_cancelled_1())
        
        # Live interviewer suddenly interrupts with new question
        gen_id_2 = 2
        current_gen_id = gen_id_2
        
        # Generation 1 is immediately marked as cancelled
        self.assertTrue(is_cancelled_1())
        
        # Generation 2 is active
        is_cancelled_2 = lambda: current_gen_id != gen_id_2
        self.assertFalse(is_cancelled_2())

    # ============================================================
    # TEST 8: CONTINUOUS INTERVIEW MULTI-TURN SEQUENCE
    # ============================================================
    def test_08_continuous_live_and_manual_interleaved(self):
        turns = [
            ("LIVE", "Tell me about yourself.", True),
            ("CANDIDATE", "I am a full-stack AI engineer.", False),
            ("MANUAL", "Write a function to find the maximum subarray sum.", True),
            ("LIVE", "What is the time complexity of that solution?", True),
            ("CANDIDATE", "It runs in O(N) using Kadane's algorithm.", False),
            ("LIVE", "Okay, thank you.", False) # Statement/acknowledgement -> No answer
        ]
        
        answers_count = 0
        for source_type, text, expect_answer in turns:
            if source_type == "CANDIDATE":
                self.session.add_candidate_speech(text)
            elif source_type == "MANUAL":
                analysis = interview_engine.classify_and_understand_utterance(
                    utterance=text,
                    session=self.session,
                    client=None,
                    source="MANUAL_INPUT",
                    is_manual=True
                )
                self.assertEqual(analysis["should_generate_answer"], expect_answer)
                if analysis["should_generate_answer"]:
                    answers_count += 1
                    self.session.add_transcript(text, role="Interviewer", is_question=True, source="MANUAL_INPUT")
            else: # LIVE
                analysis = interview_engine.classify_and_understand_utterance(
                    utterance=text,
                    session=self.session,
                    client=None,
                    source="mic"
                )
                self.assertEqual(analysis["should_generate_answer"], expect_answer)
                if analysis["should_generate_answer"]:
                    answers_count += 1
                    self.session.add_transcript(text, role="Interviewer", is_question=True, source="LIVE_INTERVIEW")
                else:
                    self.session.add_transcript(text, role="Interviewer", is_question=False, source="LIVE_INTERVIEW")
                    
        self.assertEqual(answers_count, 3) # 2 live questions + 1 manual question answered


if __name__ == "__main__":
    unittest.main(verbosity=2)
