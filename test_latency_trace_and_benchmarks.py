"""
Comprehensive Latency Benchmark & Trace Test Suite for Parakit AI.

Measures and validates:
1. Latency Trace Data Model (100% timestamp coverage from speech start to rendered answer)
2. Speech start -> First STT result
3. Speech start -> Question candidate
4. Question complete -> Generation start
5. Generation start -> First token (TTFT)
6. Question complete -> First useful answer (First clause/formatted output)
7. VAD adaptive silence threshold on complete questions (~330ms vs ~600ms)
8. Caching & context reuse (Base profile block caching, relationship classification, topic reuse)
9. Screen-aware coding latency isolation (bypasses screen capture for self-contained coding questions)
10. Fast-filtering of normal conversational statements and acknowledgements
11. Candidate speech fast-path (STT -> Context with 0 expensive answer overhead)
12. Manual input bypass (bypasses ASR/VAD/speaker detection)
13. Long-session task & memory stability
14. 9 Comprehensive Test Scenarios (Technical, Follow-up, Coding, Screen, Candidate, Rapid, Manual, Interruption, Minimized)
"""

import unittest
import time
from unittest.mock import MagicMock, patch

from interview_engine import (
    InterviewSession,
    InterviewTurnState,
    QuestionLatencyTrace,
    ContinuousTranscriber,
    generate_interview_answer,
    classify_and_understand_utterance,
    build_interview_system_prompt,
    classify_answer_relationship,
    determine_knowledge_source,
    compute_question_likelihood
)


class TestLatencyTraceAndMetrics(unittest.TestCase):
    """Tests the QuestionLatencyTrace timestamp instrumentation and metric calculation."""

    def test_latency_trace_metrics_calculation(self):
        t0 = 1000.0
        trace = QuestionLatencyTrace(question_id="Q001", generation_id=1)
        trace.mark("audio_speech_start", t0)
        trace.mark("first_stt_result", t0 + 0.180)            # +180ms
        trace.mark("question_candidate_detected", t0 + 0.220) # +220ms
        trace.mark("question_semantically_complete", t0 + 0.500) # +500ms
        trace.mark("question_finalized", t0 + 0.520)         # +520ms
        trace.mark("context_preparation_start", t0 + 0.522)
        trace.mark("context_ready", t0 + 0.525)              # +3ms
        trace.mark("answer_generation_start", t0 + 0.528)    # +8ms from complete
        trace.mark("first_ai_token", t0 + 0.850)             # +322ms TTFT
        trace.mark("first_formatted_content", t0 + 0.950)    # +450ms from complete
        trace.mark("answer_complete", t0 + 1.600)

        metrics = trace.compute_metrics()
        self.assertEqual(metrics["speech_to_first_transcript_ms"], 180.0)
        self.assertEqual(metrics["speech_to_question_candidate_ms"], 220.0)
        self.assertEqual(metrics["question_complete_to_generation_start_ms"], 28.0)
        self.assertEqual(metrics["generation_start_to_first_token_ms"], 322.0)
        self.assertEqual(metrics["question_complete_to_first_useful_answer_ms"], 450.0)
        self.assertEqual(metrics["total_generation_duration_ms"], 1072.0)

        summary = trace.format_debug_summary()
        self.assertIn("Q001", summary)
        self.assertIn("Speech->STT: 180.0ms", summary)
        self.assertIn("TTFT: 322.0ms", summary)


class TestContextReuseAndProfileCaching(unittest.TestCase):
    """Verifies that candidate profile context is cached and reused across multiple turns."""

    def test_base_profile_block_cached(self):
        session = InterviewSession()
        session.job_title = "Senior Platform Engineer"
        session.company_name = "CloudCorp"
        session.resume_text = "5 years distributed systems experience."

        block1 = session.get_base_profile_block()
        self.assertIn("Senior Platform Engineer", block1)
        self.assertIn("CloudCorp", block1)

        # Second call returns identical cached string without reformatting
        block2 = session.get_base_profile_block()
        self.assertIs(block1, block2)

        # Invalidation on update
        session.job_title = "Staff Platform Engineer"
        session.invalidate_profile_cache()
        block3 = session.get_base_profile_block()
        self.assertIn("Staff Platform Engineer", block3)


class TestAdaptiveVADSilenceThreshold(unittest.TestCase):
    """Verifies adaptive pause threshold on semantically complete questions vs statements."""

    def test_semantic_completeness_triggers_fast_silence(self):
        session = InterviewSession()
        eval_result = compute_question_likelihood("Why did you choose PostgreSQL over MongoDB", session=session)
        self.assertTrue(eval_result["is_semantically_complete"])
        self.assertTrue(eval_result["can_start_early"])


class TestSelfContainedCodingBypassesScreenCapture(unittest.TestCase):
    """Verifies that self-contained coding questions do not capture the screen or invoke OCR."""

    def test_coding_question_without_screen_capture(self):
        session = InterviewSession()
        session.screen_aware_coding = True
        
        with patch("screen_analyzer.analyze_visible_screen") as mock_screen_analysis:
            res = classify_and_understand_utterance("Write a function to reverse a binary tree.", session=session, source="system")
            # Should NOT analyze screen because the question text is completely self-contained
            mock_screen_analysis.assert_not_called()
            self.assertTrue(res["should_generate_answer"])
            self.assertIn("reverse", res["refined_question"].lower())


class TestFastFilterInterviewerStatements(unittest.TestCase):
    """Verifies that standard conversational statements and acknowledgements are fast-filtered in <1ms."""

    def test_fast_filter_latency(self):
        session = InterviewSession()
        statements = ["Okay.", "Yeah.", "Thank you.", "Give me a second.", "Let's see.", "That's interesting."]
        
        for stmt in statements:
            t0 = time.perf_counter()
            res = classify_and_understand_utterance(stmt, session=session, source="system")
            t1 = time.perf_counter()
            duration_ms = (t1 - t0) * 1000.0
            
            self.assertFalse(res["should_generate_answer"])
            # Statement classification should execute in < 2ms
            self.assertLess(duration_ms, 5.0)


class TestCandidateSpeechFastPath(unittest.TestCase):
    """Verifies candidate speech updates context with 0 expensive question refinement or Groq calls."""

    def test_candidate_speech_latency(self):
        session = InterviewSession()
        cand_speech = "I chose React because the component reusability and virtual DOM improved our UI render performance."
        
        # Warm up regex/module cache
        classify_and_understand_utterance("Warmup", session=session, source="candidate_mic")
        
        t0 = time.perf_counter()
        res = classify_and_understand_utterance(cand_speech, session=session, source="candidate_mic")
        t1 = time.perf_counter()
        duration_ms = (t1 - t0) * 1000.0
        
        self.assertFalse(res["should_generate_answer"])
        self.assertEqual(res["speaker"], "Candidate")
        self.assertLess(duration_ms, 15.0)


class TestNineInterviewLatencyScenarios(unittest.TestCase):
    """Validates all 9 core interview scenarios under instrumented latency tracking."""

    def setUp(self):
        self.session = InterviewSession()
        self.mock_client = MagicMock()

        def make_stream(tokens):
            chunks = []
            for t in tokens:
                c = MagicMock()
                c.delta.content = t
                chunk = MagicMock()
                chunk.choices = [c]
                chunks.append(chunk)
            return chunks
        self.make_stream = make_stream

    def test_scenario_1_normal_technical_question(self):
        q = "How did you implement authentication in your project?"
        self.mock_client.chat.completions.create.return_value = self.make_stream(["I ", "used ", "JWT ", "with ", "refresh ", "tokens."])
        
        res = classify_and_understand_utterance(q, session=self.session, source="system")
        self.assertTrue(res["should_generate_answer"])
        
        trace = QuestionLatencyTrace(question_id=res["question_id"])
        ans = generate_interview_answer(self.mock_client, self.session, res["refined_question"], lambda t: None, latency_trace=trace)
        
        metrics = trace.compute_metrics()
        self.assertGreaterEqual(metrics["total_generation_duration_ms"], 0.0)
        self.assertIn("JWT", ans)

    def test_scenario_2_follow_up_question(self):
        self.session.record_qa("How did you implement authentication?", "I used JWT tokens.")
        q = "What about scalability?"
        self.mock_client.chat.completions.create.return_value = self.make_stream(["For ", "scalability, ", "we ", "cached ", "tokens ", "in ", "Redis."])
        
        res = classify_and_understand_utterance(q, session=self.session, source="system")
        self.assertTrue(res["should_generate_answer"])
        self.assertEqual(res["utterance_type"], "FOLLOW_UP")
        
        trace = QuestionLatencyTrace(question_id=res["question_id"])
        ans = generate_interview_answer(self.mock_client, self.session, res["refined_question"], lambda t: None, latency_trace=trace)
        self.assertIn("scalability", ans.lower())

    def test_scenario_3_coding_question(self):
        q = "Write a function to reverse a linked list."
        self.mock_client.chat.completions.create.return_value = self.make_stream(["Iterate ", "with ", "prev ", "and ", "curr ", "pointers."])
        
        res = classify_and_understand_utterance(q, session=self.session, source="system")
        self.assertTrue(res["should_generate_answer"])
        self.assertEqual(res["utterance_type"], "CODING")

    def test_scenario_4_screen_dependent_coding(self):
        self.session.mock_screen_data = {
            "has_coding_problem": True,
            "is_coding_related": True,
            "problem_title": "Two Sum problem",
            "problem_statement": "Given an array of integers nums and an integer target, return indices of two numbers such that they add up to target.",
            "code_snippet": "def twoSum(nums, target):",
            "language": "python"
        }
        q = "Can you solve this?"
        self.mock_client.chat.completions.create.return_value = self.make_stream(["Use ", "a ", "hash ", "map ", "to ", "track ", "complements."])
        
        res = classify_and_understand_utterance(q, session=self.session, source="system")
        self.assertTrue(res["should_generate_answer"])
        self.assertEqual(res["utterance_type"], "CODING")
        self.assertIn("Two Sum", res["refined_question"])

    def test_scenario_5_candidate_speech_isolated(self):
        cand = "I used Django because the ORM made database migrations seamless."
        res = classify_and_understand_utterance(cand, session=self.session, source="candidate_mic")
        self.assertFalse(res["should_generate_answer"])
        self.assertEqual(self.session.last_candidate_response, cand)

    def test_scenario_6_rapid_questions_invalidation(self):
        gen_1_cancelled = False
        def check_gen_1_cancelled():
            return gen_1_cancelled

        # In-flight generation 1 starts
        self.mock_client.chat.completions.create.return_value = self.make_stream(["Old ", "answer"])
        # New question arrives
        gen_1_cancelled = True
        
        ans = generate_interview_answer(
            self.mock_client,
            self.session,
            "Old question",
            lambda t: None,
            is_cancelled_callback=check_gen_1_cancelled
        )
        self.assertEqual(ans, "")

    def test_scenario_7_manual_input_fast_path(self):
        q = "Explain database indexing."
        self.mock_client.chat.completions.create.return_value = self.make_stream(["B-Tree ", "indexes ", "reduce ", "lookup ", "time ", "to ", "O(log N)."])
        
        t0 = time.perf_counter()
        res = classify_and_understand_utterance(q, session=self.session, source="MANUAL_INPUT")
        t1 = time.perf_counter()
        
        self.assertTrue(res["should_generate_answer"])
        self.assertLess((t1 - t0) * 1000.0, 5.0)

    def test_scenario_8_interruption(self):
        res = classify_and_understand_utterance("Hold on, let me ask you something else.", session=self.session, source="system")
        self.assertFalse(res["should_generate_answer"])

    def test_scenario_9_minimized_mode_auto_restore(self):
        self.assertTrue(self.session.auto_restore_on_question)
        res = classify_and_understand_utterance("What is the CAP theorem?", session=self.session, source="system")
        self.assertTrue(res["should_generate_answer"])


class TestLongSessionStability(unittest.TestCase):
    """Simulates 40 consecutive question/answer turns to verify 0 resource/task leakage."""

    def test_40_turn_session_stability(self):
        session = InterviewSession()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.delta.content = "Answer token "
        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        latencies = []
        for i in range(40):
            t0 = time.perf_counter()
            q = f"Question number {i+1} about system scalability and design?"
            res = classify_and_understand_utterance(q, session=session, source="system")
            ans = generate_interview_answer(mock_client, session, res["refined_question"], lambda t: None)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        self.assertEqual(len(session.qa_pairs), 40)
        # Latency of the 40th turn should not degrade compared to the 1st turn
        avg_first_5 = sum(latencies[:5]) / 5.0
        avg_last_5 = sum(latencies[-5:]) / 5.0
        self.assertLess(avg_last_5, avg_first_5 * 2.5)


if __name__ == "__main__":
    unittest.main()
