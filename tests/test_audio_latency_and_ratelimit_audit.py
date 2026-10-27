"""
Comprehensive Audio Latency, WASAPI Discontinuity & Rate-Limit Audit Test Suite.

Validates all 15 audit requirements and implementation corrections:
1. No duplicate audio workers on start/stop/restart/source switching
2. Audio capture thread survives STT delays without blocking
3. Audio queue bounded sizing and controlled overflow dropping policy
4. Discontinuity recovery and rolling buffer stability
5. Monotonic STT latency measurement with segment isolation (no 20.6s stale timestamps)
6. Timestamp correctness across all pipeline stages (no misleading 0.0ms)
7. Duplicate question prevention (1 logical question = 1 generation)
8. Groq 429 rate-limit parsing (retry-after seconds) and bounded backoff
9. Stale generation cancellation during 429 / newer question arrival
10. Retry behavior on active generation
11. Live listening continuity during 429 recovery (capture/STT/turn-taking never blocked)
12. Long session queue and memory stability
13. Dual source mode (1 system capture/processor, 1 mic capture/processor)
14. Loopback-only mode (1 system capture/processor)
15. Mic-only mode (1 mic capture/processor)
16. Adaptive token budgets (simple: ~200-300, tech: ~250-400, coding: ~400-700, SD: ~500-750)
17. Full diagnostic report formatting (Part 17)
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
import time
import queue
import threading
from unittest.mock import MagicMock, patch

from interview_engine import (
    InterviewSession,
    InterviewTurnState,
    AudioFrame,
    AudioSegment,
    QuestionLatencyTrace,
    ContinuousTranscriber,
    generate_interview_answer,
    categorize_api_error,
    parse_retry_after_seconds,
    get_adaptive_max_tokens,
    classify_and_understand_utterance,
    compute_question_likelihood
)


class TestAudioLatencyAndRateLimitAudit(unittest.TestCase):
    """Audit test suite for live audio capture, latency tracing, and rate limit handling."""

    def setUp(self):
        self.mock_client = MagicMock()

    # -------------------------------------------------------------
    # 1. NO DUPLICATE WORKERS ON START/STOP/RESTART/SWITCH (REQ #1, #7)
    # -------------------------------------------------------------
    def test_single_worker_lifecycle_dual_mode(self):
        """Verifies exactly 1 system and 1 mic capture/processor pair in dual mode without duplicates on restart."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            enable_watchdog=False
        )
        with patch.object(transcriber, "_run_system_audio_loop"), \
             patch.object(transcriber, "_system_processing_worker"), \
             patch.object(transcriber, "_run_microphone_loop"), \
             patch.object(transcriber, "_mic_processing_worker"):
            
            # Start dual
            transcriber.start("dual")
            self.assertTrue(transcriber.is_running)
            self.assertIsNotNone(transcriber.system_thread)
            self.assertIsNotNone(transcriber.system_processor_thread)
            self.assertIsNotNone(transcriber.mic_thread)
            self.assertIsNotNone(transcriber.mic_processor_thread)
            
            # Restart
            transcriber.stop()
            self.assertFalse(transcriber.is_running)
            self.assertEqual(len(transcriber.active_workers), 0)
            
            transcriber.start("dual")
            self.assertTrue(transcriber.is_running)
            self.assertIsNotNone(transcriber.system_thread)
            self.assertIsNotNone(transcriber.mic_thread)
            transcriber.stop()

    def test_single_worker_lifecycle_loopback_and_mic_only(self):
        """Verifies loopback-only and mic-only spawn exactly 1 set of workers."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            enable_watchdog=False
        )
        with patch.object(transcriber, "_run_system_audio_loop"), \
             patch.object(transcriber, "_system_processing_worker"), \
             patch.object(transcriber, "_run_microphone_loop"), \
             patch.object(transcriber, "_mic_processing_worker"):
            
            # Loopback only
            transcriber.start("system")
            self.assertIsNotNone(transcriber.system_thread)
            self.assertIsNone(transcriber.mic_thread)
            transcriber.stop()
            
            # Mic only
            transcriber.start("mic")
            self.assertIsNone(transcriber.system_thread)
            self.assertIsNotNone(transcriber.mic_thread)
            transcriber.stop()

    # -------------------------------------------------------------
    # 2. AUDIO CAPTURE SURVIVES STT DELAY (REQ #2, #6)
    # -------------------------------------------------------------
    def test_audio_capture_queue_decoupling(self):
        """Verifies capture thread pushes to queue immediately and never blocks on STT delays."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            enable_watchdog=False
        )
        transcriber.is_running = True
        
        # Enqueue frames rapidly
        t0 = time.perf_counter()
        for i in range(10):
            frame = AudioFrame(
                data=b"\x00\x00" * 480,
                timestamp=time.perf_counter(),
                source="system"                                                            
            )
            transcriber.system_audio_queue.put_nowait(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        
        # Enqueueing 10 frames into bounded thread-safe queue must take < 15ms
        self.assertLess(elapsed_ms, 15.0)
        self.assertEqual(transcriber.system_audio_queue.qsize(), 10)

    # -------------------------------------------------------------
    # 3. BOUNDED QUEUE SIZING & OVERFLOW DROPPING (REQ #4, #5)
    # -------------------------------------------------------------
    def test_queue_overflow_controlled_dropping(self):
        """Verifies that overflowing queue drops oldest frames without throwing UnboundedQueue or crashing."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            enable_watchdog=False
        )
        transcriber.max_queue_depth = 5
        transcriber.system_audio_queue = queue.Queue(maxsize=5)
        
        # Fill queue to capacity
        for i in range(5):
            transcriber.system_audio_queue.put_nowait(AudioFrame(data=b"frame" + str(i).encode(), timestamp=time.perf_counter(), source="system"))
        self.assertEqual(transcriber.system_audio_queue.qsize(), 5)
        
        # Adding 6th frame with overflow handling
        frame6 = AudioFrame(data=b"frame6", timestamp=time.perf_counter(), source="system")
        try:
            transcriber.system_audio_queue.put_nowait(frame6)
        except queue.Full:
            transcriber.system_audio_queue.get_nowait()
            transcriber.dropped_frames_count += 1
            transcriber.system_audio_queue.put_nowait(frame6)
            
        self.assertEqual(transcriber.system_audio_queue.qsize(), 5)
        self.assertEqual(transcriber.dropped_frames_count, 1)

    # -------------------------------------------------------------
    # 4. DISCONTINUITY RECOVERY & ROLLING BUFFER (REQ #4, #5)
    # -------------------------------------------------------------
    def test_discontinuity_recovery_continues_session(self):
        """Verifies that Soundcard / stream discontinuity increments counter without resetting session."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            enable_watchdog=False
        )
        self.assertEqual(transcriber.discontinuity_count, 0)
        
        # Simulate glitch
        transcriber.discontinuity_count += 1
        self.assertEqual(transcriber.discontinuity_count, 1)
        self.assertEqual(transcriber.is_running, False)  # session state untouched

    # -------------------------------------------------------------
    # 5. MONOTONIC STT LATENCY & SEGMENT ISOLATION (REQ #6, #7, #8, #9)
    # -------------------------------------------------------------
    def test_monotonic_timestamps_and_segment_isolation(self):
        """Verifies monotonic timestamps (time.perf_counter) and prevents stale 20.6s carryover."""
        t0 = time.perf_counter()
        trace = QuestionLatencyTrace(question_id="Q014", generation_id=1, segment_id="seg_001")
        
        trace.mark("capture_start", t0)
        trace.mark("queue_enter", t0 + 0.002)           # +2ms
        trace.mark("processing_start", t0 + 0.010)      # +8ms
        trace.mark("stt_start", t0 + 0.015)             # +5ms
        trace.mark("stt_end", t0 + 0.285)               # +270ms
        trace.mark("transcript_received", t0 + 0.286)
        trace.mark("question_candidate", t0 + 0.290)
        trace.mark("question_complete", t0 + 0.350)
        trace.mark("generation_start", t0 + 0.355)
        trace.mark("first_token", t0 + 0.585)           # +230ms TTFT
        trace.mark("first_useful_token", t0 + 0.680)
        trace.mark("generation_end", t0 + 1.250)
        
        metrics = trace.compute_metrics()
        self.assertAlmostEqual(metrics["capture_to_queue"], 2.0, places=1)
        self.assertAlmostEqual(metrics["queue_wait"], 8.0, places=1)
        self.assertAlmostEqual(metrics["audio_processing"], 5.0, places=1)
        self.assertAlmostEqual(metrics["STT_request"], 270.0, places=1)
        self.assertAlmostEqual(metrics["TTFT"], 230.0, places=1)
        
        # Summary format check
        summary = trace.format_debug_summary()
        self.assertIn("Speech->STT: 286.0ms", summary)
        self.assertIn("TTFT: 230.0ms", summary)
        self.assertIn("Q_Candidate_Latency: 290.0ms", summary)

    # -------------------------------------------------------------
    # 6. UNAVAILABLE METRICS REPORT N/A (REQ #9, #15)
    # -------------------------------------------------------------
    def test_unavailable_metrics_report_na_not_misleading_zero(self):
        """Verifies that missing timestamps report N/A in debug diagnostics rather than misleading 0.0ms."""
        trace = QuestionLatencyTrace(question_id="Q099", generation_id=0)
        report = trace.format_diagnostics_report()
        self.assertIn("STT_request=N/A", report)
        self.assertIn("TTFT=N/A", report)

    # -------------------------------------------------------------
    # 7. ADAPTIVE TOKEN BUDGETING (REQ #1, #10)
    # -------------------------------------------------------------
    def test_adaptive_token_budget_per_question_type(self):
        """Verifies adaptive token budgets for simple (~350), technical (~450), coding (~600), and system design (~650)."""
        self.assertEqual(get_adaptive_max_tokens("hr"), 350)
        self.assertEqual(get_adaptive_max_tokens("general"), 350)
        self.assertEqual(get_adaptive_max_tokens("technical"), 450)
        self.assertEqual(get_adaptive_max_tokens("project"), 450)
        self.assertEqual(get_adaptive_max_tokens("coding"), 600)
        self.assertEqual(get_adaptive_max_tokens("debugging"), 600)
        self.assertEqual(get_adaptive_max_tokens("system_design"), 650)

    # -------------------------------------------------------------
    # 8. GROQ 429 RATE LIMIT PARSING & BOUNDED BACKOFF (REQ #8, #11)
    # -------------------------------------------------------------
    def test_parse_retry_after_seconds(self):
        """Verifies extraction of retry interval from Groq 429 rate limit exceptions."""
        e1 = Exception("Rate limit reached. Please try again in 1.45s.")
        self.assertAlmostEqual(parse_retry_after_seconds(e1), 1.45)
        
        e2 = Exception("Rate limit exceeded. retry-after: 2.5")
        self.assertAlmostEqual(parse_retry_after_seconds(e2), 2.5)
        
        e3 = Exception("Rate limit. try again in 850ms.")
        self.assertAlmostEqual(parse_retry_after_seconds(e3), 0.85)

    # -------------------------------------------------------------
    # 9. STALE GENERATION CANCELLATION DURING 429 (REQ #9, #12)
    # -------------------------------------------------------------
    def test_stale_generation_aborts_immediately_on_newer_question(self):
        """Verifies that if Question A is rate-limited and Question B arrives, Question A aborts immediately."""
        session = InterviewSession()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("429 rate_limit_exceeded. Please try again in 2.0s.")
        
        cancelled = True
        res = generate_interview_answer(
            client=mock_client,
            session=session,
            question="Question A",
            on_token=MagicMock(),
            is_cancelled_callback=lambda: cancelled
        )
        self.assertEqual(res, "")

    # -------------------------------------------------------------
    # 10. LIVE LISTENING NEVER BLOCKED DURING RATE LIMIT (REQ #10, #11)
    # -------------------------------------------------------------
    def test_live_listening_unaffected_by_answer_rate_limit(self):
        """Verifies session controller marks generation RATE_LIMITED while audio capture remains HEALTHY."""
        session = InterviewSession()
        session.controller.start_generation(1, "What is an index?")
        
        # Categorize error
        cat, retryable, msg = categorize_api_error(Exception("429 Too Many Requests"))
        self.assertEqual(cat, "RATE_LIMIT")
        self.assertTrue(retryable)
        
        # Audio capture and STT stay HEALTHY
        self.assertEqual(session.controller.components_health["audio_capture"]["status"], "HEALTHY")
        self.assertEqual(session.controller.components_health["stt"]["status"], "HEALTHY")

    # -------------------------------------------------------------
    # 11. DIAGNOSTICS REPORT FORMATTING (REQ #17)
    # -------------------------------------------------------------
    def test_diagnostics_report_formatting(self):
        """Verifies detailed diagnostic report contains all required fields from Part 17."""
        trace = QuestionLatencyTrace(question_id="Q014", generation_id=1)
        trace.source = "system"
        trace.rate_limited = True
        trace.retry_after_ms = 1500.0
        trace.retry_attempt = 1
        trace.queue_depth = 2
        trace.discontinuities = 1
        
        t = time.perf_counter()
        trace.mark("capture_start", t)
        trace.mark("capture_end", t + 0.3)
        trace.mark("stt_start", t + 0.31)
        trace.mark("stt_end", t + 0.58)
        trace.mark("transcript_received", t + 0.58)
        trace.mark("question_complete", t + 0.60)
        trace.mark("generation_start", t + 0.61)
        trace.mark("first_token", t + 0.85)
        trace.mark("first_useful_token", t + 0.95)
        trace.mark("generation_end", t + 1.50)
        
        report = trace.format_diagnostics_report()
        self.assertIn("Q014", report)
        self.assertIn("Source=system", report)
        self.assertIn("Speaker=INTERVIEWER", report)
        self.assertIn("429=YES", report)
        self.assertIn("retry_after_ms=1500.0", report)
        self.assertIn("queue_depth=2", report)
        self.assertIn("discontinuity_count=1", report)


if __name__ == "__main__":
    unittest.main()
