"""
Automated Test Suite for Parakit Real-Time Speech Capture & Progressive Question Detection Pipeline
===================================================================================================
Tests:
1. Test 1: Normal Question Progressive Likelihood Stream
2. Test 2: Non-Question Statement Protection (Transcript only, NO answer)
3. Test 3: Progressive Contextual Follow-up Detection
4. Test 4: Progressive Coding Imperative Detection
5. Test 5: Candidate Speech Protection During Live STT
6. Test 6: Speaker Transition & Interruption Priority
7. Test 7: Rapid Question Sequence
8. Test 8: Deduplication of Progressive Interim Updates (1 Final Record & 1 Answer)
9. Test 9: STT Provider Abstraction Architecture
10. Test 10: Performance & Latency Benchmark
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import sys
import os
import unittest
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine
from interview_engine import (
    compute_question_likelihood,
    QuestionStabilityState,
    classify_and_understand_utterance,
    InterviewTurnState,
    STTProvider,
    ContinuousTranscriber
)


class MockSTTProvider(STTProvider):
    """Mock STT Provider simulating fast chunk-based transcription."""
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "") -> str:
        return self.response_text


class TestRealTimeSTTPipeline(unittest.TestCase):

    def setUp(self):
        self.session = interview_engine.InterviewSession()
        self.session.clear()

    # ============================================================
    # TEST 1: NORMAL QUESTION PROGRESSIVE STREAM
    # ============================================================
    def test_01_normal_question_progression(self):
        # Step 1: "How..."
        p1 = compute_question_likelihood("How", session=self.session)
        self.assertEqual(p1["stability_state"], QuestionStabilityState.POSSIBLE_QUESTION)
        self.assertTrue(p1["is_question_stem"])
        self.assertFalse(p1["is_complete_enough"])

        # Step 2: "How did you..."
        p2 = compute_question_likelihood("How did you", session=self.session)
        self.assertEqual(p2["stability_state"], QuestionStabilityState.LIKELY_QUESTION)
        self.assertGreater(p2["likelihood"], p1["likelihood"])

        # Step 3: "How did you implement authentication..."
        p3 = compute_question_likelihood("How did you implement authentication", session=self.session)
        self.assertEqual(p3["stability_state"], QuestionStabilityState.STABLE_QUESTION)
        self.assertTrue(p3["is_complete_enough"])

        # Step 4: Finalized utterance
        analysis = classify_and_understand_utterance("How did you implement authentication?", session=self.session)
        self.assertTrue(analysis["should_generate_answer"])
        self.assertEqual(analysis["speaker"], "Interviewer")

    # ============================================================
    # TEST 2: NON-QUESTION STATEMENT
    # ============================================================
    def test_02_statement_protection(self):
        p = compute_question_likelihood("I live in St. Louis.", session=self.session)
        self.assertLessEqual(p["likelihood"], 0.20)
        self.assertFalse(p["is_question_stem"])
        self.assertEqual(p["stem_type"], "STATEMENT")

        analysis = classify_and_understand_utterance("I live in St. Louis.", session=self.session)
        self.assertFalse(analysis["should_generate_answer"])
        self.assertEqual(analysis["utterance_type"], "STATEMENT")

    # ============================================================
    # TEST 3: PROGRESSIVE FOLLOW-UP
    # ============================================================
    def test_03_followup_progression(self):
        self.session.active_topic = "MongoDB"
        self.session.add_transcript("Why did you choose MongoDB?", role="Interviewer", is_question=True)
        self.session.add_candidate_speech("Because of flexible schema.")

        # Partial 1: "What about..."
        p1 = compute_question_likelihood("What about", session=self.session)
        self.assertEqual(p1["stability_state"], QuestionStabilityState.POSSIBLE_QUESTION)
        self.assertEqual(p1["stem_type"], "FOLLOW_UP")

        # Partial 2: "What about scalability..."
        p2 = compute_question_likelihood("What about scalability", session=self.session)
        self.assertIn(p2["stability_state"], [QuestionStabilityState.LIKELY_QUESTION, QuestionStabilityState.STABLE_QUESTION])

        # Final
        analysis = classify_and_understand_utterance("What about scalability?", session=self.session)
        self.assertTrue(analysis["should_generate_answer"])
        self.assertEqual(analysis["utterance_type"], "FOLLOW_UP")

    # ============================================================
    # TEST 4: PROGRESSIVE CODING IMPERATIVE
    # ============================================================
    def test_04_coding_imperative_progression(self):
        p1 = compute_question_likelihood("Write a function", session=self.session)
        self.assertEqual(p1["stability_state"], QuestionStabilityState.LIKELY_QUESTION)
        self.assertEqual(p1["stem_type"], "CODING_IMPERATIVE")

        p2 = compute_question_likelihood("Write a function to reverse a linked list", session=self.session)
        self.assertEqual(p2["stability_state"], QuestionStabilityState.STABLE_QUESTION)

        analysis = classify_and_understand_utterance("Write a function to reverse a linked list", session=self.session)
        self.assertTrue(analysis["should_generate_answer"])
        self.assertEqual(analysis["utterance_type"], "CODING")

    # ============================================================
    # TEST 5: CANDIDATE SPEECH PROTECTION DURING LIVE STT
    # ============================================================
    def test_05_candidate_speech_protection(self):
        self.session.turn_state = InterviewTurnState.CANDIDATE_SPEAKING
        
        cand_text = "I used Django because it provides built-in ORM and rapid prototyping capabilities."
        p = compute_question_likelihood(cand_text, session=self.session)
        
        analysis = classify_and_understand_utterance(cand_text, session=self.session)
        self.assertEqual(analysis["speaker"], "Candidate")
        self.assertFalse(analysis["should_generate_answer"])

    # ============================================================
    # TEST 6: SPEAKER TRANSITION & INTERRUPTION
    # ============================================================
    def test_06_interviewer_interruption(self):
        # Candidate was speaking
        self.session.add_candidate_speech("I used Django for the backend.")
        self.assertEqual(self.session.turn_state, InterviewTurnState.CANDIDATE_SPEAKING)

        # Interviewer interrupts with a direct question
        interviewer_q = "Why didn't you use Flask?"
        analysis = classify_and_understand_utterance(interviewer_q, session=self.session)
        
        self.assertEqual(analysis["speaker"], "Interviewer")
        self.assertTrue(analysis["should_generate_answer"])
        self.assertEqual(self.session.turn_state, InterviewTurnState.INTERVIEWER_SPEAKING)

    # ============================================================
    # TEST 7: RAPID QUESTION SEQUENCE
    # ============================================================
    def test_07_rapid_question_sequence(self):
        q1 = "What is polymorphism?"
        analysis1 = classify_and_understand_utterance(q1, session=self.session)
        self.assertTrue(analysis1["should_generate_answer"])
        self.session.record_qa(q1, "Polymorphism allows objects to take multiple forms.", category="technical")

        q2 = "And what is encapsulation?"
        analysis2 = classify_and_understand_utterance(q2, session=self.session)
        self.assertTrue(analysis2["should_generate_answer"])
        self.assertNotEqual(analysis1["question_id"], analysis2["question_id"])

    # ============================================================
    # TEST 8: DEDUPLICATION OF PROGRESSIVE INTERIM UPDATES
    # ============================================================
    def test_08_deduplication_interim_to_final(self):
        # Simulate 4 interim updates followed by 1 final commit
        interim_steps = [
            "Why",
            "Why did",
            "Why did you",
            "Why did you choose React"
        ]
        likelihood_results = [compute_question_likelihood(step, session=self.session) for step in interim_steps]
        
        # All partial steps are classified correctly without committing to session QA
        self.assertEqual(len(self.session.qa_pairs), 0)
        self.assertEqual(len(self.session.transcript_log), 0)

        # Finalized utterance arrives
        final_q = "Why did you choose React?"
        analysis = classify_and_understand_utterance(final_q, session=self.session)
        self.session.add_transcript(final_q, role="Interviewer", is_question=True)
        self.session.record_qa(final_q, "Because of its component-based virtual DOM.", category="technical")

        # Exactly 1 transcript record and 1 QA pair
        self.assertEqual(len(self.session.transcript_log), 1)
        self.assertEqual(len(self.session.qa_pairs), 1)

    # ============================================================
    # TEST 9: STT PROVIDER ABSTRACTION
    # ============================================================
    def test_09_stt_provider_abstraction(self):
        mock_provider = MockSTTProvider("Explain indexing in PostgreSQL.")
        
        captured_chunks = []
        captured_interims = []
        
        def on_chunk(text, is_q):
            captured_chunks.append((text, is_q))
            
        def on_interim(text, info):
            captured_interims.append((text, info))

        transcriber = ContinuousTranscriber(
            client=None,
            on_transcript_chunk=on_chunk,
            on_interim_transcript=on_interim,
            provider=mock_provider,
            session=self.session
        )
        
        # Test direct transcription through provider
        result = transcriber.provider.transcribe(b"dummy_pcm_bytes_12345678901234567890", sample_rate=16000)
        self.assertEqual(result, "Explain indexing in PostgreSQL.")

    # ============================================================
    # TEST 10: PERFORMANCE BENCHMARK
    # ============================================================
    def test_10_question_likelihood_benchmark(self):
        start = time.perf_counter()
        iterations = 500
        for _ in range(iterations):
            compute_question_likelihood("How did you implement authentication in this project?", session=self.session)
        elapsed = (time.perf_counter() - start) * 1000.0 / iterations
        
        print(f"\n[Question Likelihood Benchmark] Average decision latency: {elapsed:.4f} ms")
        self.assertLess(elapsed, 1.0)  # Must be sub-millisecond


if __name__ == "__main__":
    unittest.main(verbosity=2)
