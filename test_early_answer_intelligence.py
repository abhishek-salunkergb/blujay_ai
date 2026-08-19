"""
Automated Test Suite for Parakit Early-Answer Intelligence & Question Completeness Engine
========================================================================================
Tests:
1. Test 1: Complete Technical Question (Early Start Trigger)
2. Test 2: Progressive Question Evolution & Normalized Semantic Stability
3. Test 3: Incomplete Trailing Stem Protection (No Premature Trigger)
4. Test 4: Context-Resolvable Follow-up Early Start
5. Test 5: Fast Follow-up (Why? / How?) with Immediate Grounding
6. Test 6: Coding Problem Completeness
7. Test 7: Candidate Speech Protection During Interim STT
8. Test 8: Semantic Reconciliation on Final ASR (Zero Duplicate Generation)
9. Test 9: Rapid Question Sequence
10. Test 10: Performance & Completeness Decision Latency
"""

import sys
import os
import unittest
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine
from interview_engine import (
    evaluate_question_completeness,
    compute_question_likelihood,
    get_normalized_semantic_key,
    classify_and_understand_utterance,
    InterviewTurnState
)


class TestEarlyAnswerIntelligence(unittest.TestCase):

    def setUp(self):
        self.session = interview_engine.InterviewSession()
        self.session.clear()

    # ============================================================
    # TEST 1: COMPLETE TECHNICAL QUESTION
    # ============================================================
    def test_01_complete_technical_question(self):
        q = "How did you implement authentication in Django?"
        eval_res = evaluate_question_completeness(q, session=self.session)
        self.assertTrue(eval_res["is_semantically_complete"])
        self.assertTrue(eval_res["can_start_early"])
        self.assertGreaterEqual(eval_res["completeness"], 0.90)

    # ============================================================
    # TEST 2: PROGRESSIVE QUESTION & NORMALIZED SEMANTIC KEY
    # ============================================================
    def test_02_progressive_question_candidate(self):
        p1 = "How"
        p2 = "How did you"
        p3 = "How did you implement"
        p4 = "How did you implement authentication"
        p5 = "How did you implement authentication in Django?"

        # Early fragments should not trigger early start
        self.assertFalse(evaluate_question_completeness(p1, session=self.session)["can_start_early"])
        self.assertFalse(evaluate_question_completeness(p2, session=self.session)["can_start_early"])
        self.assertFalse(evaluate_question_completeness(p3, session=self.session)["can_start_early"])

        # Meaningful semantic completion triggers early start
        e4 = evaluate_question_completeness(p4, session=self.session)
        self.assertTrue(e4["can_start_early"])

        e5 = evaluate_question_completeness(p5, session=self.session)
        self.assertTrue(e5["can_start_early"])

        # Semantic keys share core tokens
        key4 = get_normalized_semantic_key(p4)
        key5 = get_normalized_semantic_key(p5)
        self.assertIn("implement authentication", key4)
        self.assertIn("implement authentication", key5)

    # ============================================================
    # TEST 3: INCOMPLETE TRAILING STEM PROTECTION
    # ============================================================
    def test_03_incomplete_trailing_stem_protection(self):
        incomplete_queries = [
            "Why did you choose",
            "How did you implement",
            "What is the difference between",
            "Tell me about your",
            "Can you explain"
        ]
        for q in incomplete_queries:
            eval_res = evaluate_question_completeness(q, session=self.session)
            self.assertFalse(eval_res["can_start_early"], f"Should not start early for incomplete stem: '{q}'")
            self.assertFalse(eval_res["is_semantically_complete"])
            self.assertLess(eval_res["completeness"], 0.75)

    # ============================================================
    # TEST 4: CONTEXT-RESOLVABLE FOLLOW-UP
    # ============================================================
    def test_04_context_resolvable_followup(self):
        self.session.active_topic = "MongoDB"
        self.session.add_transcript("Why did you choose MongoDB?", role="Interviewer", is_question=True)
        self.session.add_candidate_speech("Because of our flexible document schema.")

        q = "What about scalability..."
        eval_res = evaluate_question_completeness(q, session=self.session)
        self.assertTrue(eval_res["can_start_early"])
        self.assertTrue(eval_res["is_semantically_complete"])

    # ============================================================
    # TEST 5: FAST FOLLOW-UP (WHY? / HOW?)
    # ============================================================
    def test_05_fast_followup(self):
        self.session.active_topic = "Redis"
        
        eval_why = evaluate_question_completeness("Why?", session=self.session)
        self.assertTrue(eval_why["can_start_early"])
        self.assertEqual(eval_why["completeness"], 1.0)

        eval_how = evaluate_question_completeness("How?", session=self.session)
        self.assertTrue(eval_how["can_start_early"])
        self.assertEqual(eval_how["completeness"], 1.0)

    # ============================================================
    # TEST 6: CODING PROBLEM COMPLETENESS
    # ============================================================
    def test_06_coding_problem_completeness(self):
        # Incomplete stem
        incomplete_coding = "Write a function to"
        eval_inc = evaluate_question_completeness(incomplete_coding, session=self.session)
        self.assertFalse(eval_inc["can_start_early"])

        # Complete task
        complete_coding = "Write a function to reverse a linked list"
        eval_comp = evaluate_question_completeness(complete_coding, session=self.session)
        self.assertTrue(eval_comp["can_start_early"])
        self.assertGreaterEqual(eval_comp["completeness"], 0.90)

    # ============================================================
    # TEST 7: CANDIDATE SPEECH PROTECTION
    # ============================================================
    def test_07_candidate_speech_protection(self):
        self.session.turn_state = InterviewTurnState.CANDIDATE_SPEAKING
        cand_speech = "I implemented JWT authentication with refresh tokens stored in HttpOnly cookies."
        
        analysis = classify_and_understand_utterance(cand_speech, session=self.session)
        self.assertEqual(analysis["speaker"], "Candidate")
        self.assertFalse(analysis["should_generate_answer"])

    # ============================================================
    # TEST 8: SEMANTIC RECONCILIATION ON FINAL ASR
    # ============================================================
    def test_08_semantic_reconciliation_zero_duplicates(self):
        partial_q = "How did you implement authentication"
        final_q = "How did you implement authentication in Django?"

        sem_partial = get_normalized_semantic_key(partial_q)
        sem_final = get_normalized_semantic_key(final_q)

        # Core semantic tokens match
        self.assertTrue(sem_partial in sem_final or sem_final in sem_partial)

    # ============================================================
    # TEST 9: RAPID QUESTION SEQUENCE
    # ============================================================
    def test_09_rapid_question_sequence(self):
        q1 = "Explain indexing in PostgreSQL."
        eval1 = evaluate_question_completeness(q1, session=self.session)
        self.assertTrue(eval1["can_start_early"])

        q2 = "And what is sharding?"
        eval2 = evaluate_question_completeness(q2, session=self.session)
        self.assertTrue(eval2["can_start_early"])

        self.assertNotEqual(eval1["semantic_key"], eval2["semantic_key"])

    # ============================================================
    # TEST 10: PERFORMANCE BENCHMARK
    # ============================================================
    def test_10_completeness_decision_latency(self):
        start = time.perf_counter()
        iterations = 500
        for _ in range(iterations):
            evaluate_question_completeness("How did you implement authentication in Django?", session=self.session)
        elapsed = (time.perf_counter() - start) * 1000.0 / iterations
        
        print(f"\n[Question Completeness Benchmark] Average decision latency: {elapsed:.4f} ms")
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
