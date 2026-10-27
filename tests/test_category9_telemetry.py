"""
Category 9 Unit & Integration Test Suite: Real-time Audio & Pipeline Telemetry
=============================================================================
Verifies:
1. High-Resolution Latency Percentiles (p50, p95, p99, min, max, mean) under 0, 1, and 25+ traces.
2. Low-Overhead Audio Quality Metrics Aggregation (speech hit rate %, dropout rate %, RMS, mute tracking).
3. Thread-Safe Structured In-Memory Error Ring Buffer (bounding at 100, categorization, overflow eviction).
4. Component Health Transitions & Overall System Status (HEALTHY, DEGRADED, ERROR).
5. Context-Aware Heartbeat Staleness & Automated Watchdog Recovery.
6. Centralized Telemetry Snapshot Sub-Millisecond Execution Budget (< 0.5ms).
7. Post-Session Analytics Backward Compatibility with Telemetry Enrichment.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
import time
import threading
from unittest.mock import MagicMock

from interview_engine import (
    InterviewSession,
    InterviewTurnState,
    SessionController,
    QuestionLatencyTrace,
    ContinuousTranscriber,
    calculate_session_analytics,
    categorize_api_error
)


class TestCategory9Telemetry(unittest.TestCase):
    """Test suite for Parakeet-Grade Category 9 Observability and Telemetry."""

    def setUp(self):
        self.session = InterviewSession()
        self.controller = self.session.controller

    def test_latency_percentiles_calculation(self):
        """Verify p50, p95, p99, mean, min, max calculation across 0, 1, and 25+ traces."""
        # 1. Zero traces
        summary = self.controller.get_latency_summary()
        self.assertEqual(summary["trace_count"], 0)
        self.assertEqual(summary["ttft_ms"]["count"], 0)
        self.assertEqual(summary["ttft_ms"]["p50"], 0.0)
        self.assertEqual(summary["ttft_ms"]["p95"], 0.0)

        # 2. Single trace
        tr1 = QuestionLatencyTrace(question_id="Q001", generation_id=1)
        tr1.mark("capture_start", 10.0)
        tr1.mark("transcript_received", 10.15)  # 150ms STT
        tr1.mark("generation_start", 10.20)
        tr1.mark("first_token", 10.45)           # 250ms TTFT
        tr1.mark("generation_end", 11.20)         # 1000ms total gen
        tr1.compute_metrics()
        self.session.latency_traces.append(tr1)

        summary1 = self.controller.get_latency_summary()
        self.assertEqual(summary1["trace_count"], 1)
        self.assertEqual(summary1["stt_latency_ms"]["count"], 1)
        self.assertAlmostEqual(summary1["stt_latency_ms"]["p50"], 150.0, delta=2.0)
        self.assertAlmostEqual(summary1["stt_latency_ms"]["p95"], 150.0, delta=2.0)
        self.assertAlmostEqual(summary1["ttft_ms"]["p50"], 250.0, delta=2.0)

        # 3. 25 traces with increasing TTFT (from 100ms to 580ms)
        self.session.latency_traces.clear()
        for i in range(25):
            tr = QuestionLatencyTrace(question_id=f"Q{i:03d}", generation_id=i)
            t_base = 100.0 + i * 10.0
            stt_dur = 120.0 + (i * 5.0)       # 120ms to 240ms
            ttft_dur = 150.0 + (i * 18.0)     # 150ms to 582ms
            
            tr.mark("capture_start", t_base)
            tr.mark("transcript_received", t_base + stt_dur / 1000.0)
            tr.mark("generation_start", t_base + stt_dur / 1000.0)
            tr.mark("first_token", t_base + (stt_dur + ttft_dur) / 1000.0)
            tr.mark("first_useful_token", t_base + (stt_dur + ttft_dur + 50.0) / 1000.0)
            tr.mark("generation_end", t_base + (stt_dur + ttft_dur + 800.0) / 1000.0)
            tr.compute_metrics()
            self.session.latency_traces.append(tr)

        summary25 = self.controller.get_latency_summary()
        self.assertEqual(summary25["trace_count"], 25)
        
        ttft_stats = summary25["ttft_ms"]
        self.assertEqual(ttft_stats["count"], 25)
        self.assertTrue(ttft_stats["min"] <= ttft_stats["p50"] <= ttft_stats["p95"] <= ttft_stats["p99"] <= ttft_stats["max"])
        self.assertGreater(ttft_stats["p95"], ttft_stats["p50"])
        self.assertGreater(ttft_stats["max"], ttft_stats["min"])

    def test_audio_quality_report_accumulation(self):
        """Verify audio quality report computes availability %, dropout %, VAD hit rate, and mute tracking."""
        mock_client = MagicMock()
        transcriber = ContinuousTranscriber(
            client=mock_client,
            on_transcript_chunk=MagicMock(),
            session=self.session
        )
        self.session.transcriber = transcriber

        # Simulate frame processing stats
        transcriber.total_system_frames = 800
        transcriber.system_speech_frames = 200
        transcriber.cumulative_system_rms = 800 * 450.0

        transcriber.total_mic_frames = 600
        transcriber.mic_speech_frames = 150
        transcriber.cumulative_mic_rms = 600 * 180.0

        transcriber.dropped_frames_count = 7
        transcriber.discontinuity_count = 2
        transcriber.is_mic_muted = True
        transcriber.mute_toggle_count = 3
        transcriber.audio_init_latency_ms = 42.5

        report = transcriber.get_audio_quality_report()
        self.assertEqual(report["total_frames"], 1400)
        self.assertEqual(report["total_system_frames"], 800)
        self.assertEqual(report["total_mic_frames"], 600)
        self.assertEqual(report["dropped_frames"], 7)
        self.assertAlmostEqual(report["system_speech_ratio_pct"], 25.0, delta=0.1)
        self.assertAlmostEqual(report["mic_speech_ratio_pct"], 25.0, delta=0.1)
        self.assertAlmostEqual(report["overall_speech_ratio_pct"], 25.0, delta=0.1)
        self.assertAlmostEqual(report["avg_system_rms"], 450.0, delta=0.1)
        self.assertAlmostEqual(report["avg_mic_rms"], 180.0, delta=0.1)
        self.assertTrue(report["is_mic_muted"])
        self.assertEqual(report["mute_toggle_count"], 3)
        self.assertAlmostEqual(report["audio_init_latency_ms"], 42.5, delta=0.1)
        self.assertGreater(report["availability_pct"], 99.0)

        # Also verify session proxy
        session_report = self.session.get_audio_quality_report()
        self.assertEqual(session_report["total_frames"], 1400)

    def test_structured_error_ring_buffer(self):
        """Verify thread-safe ring buffer bounding (maxlen=100), categorization, and summary aggregation."""
        self.controller._error_ring_buffer.clear()

        # 1. Record various error categories
        e1 = self.controller.record_error("audio_capture", "AUDCLNT_E_DEVICE_INVALIDATED", category="AUDIO_HARDWARE")
        self.assertEqual(e1["category"], "AUDIO_HARDWARE")
        self.assertEqual(e1["component"], "audio_capture")

        e2 = self.controller.record_error("stt", "429 Too Many Requests: Rate limit exceeded", category="RATE_LIMIT")
        self.assertEqual(e2["category"], "RATE_LIMIT")

        e3 = self.controller.record_error("answer_generation", Exception("Groq connection timeout"), category="TIMEOUT")
        self.assertEqual(e3["category"], "TIMEOUT")

        # 2. Check summary aggregation
        summary = self.controller.get_error_summary()
        self.assertEqual(summary["total_errors"], 3)
        self.assertEqual(summary["by_component"]["audio_capture"], 1)
        self.assertEqual(summary["by_component"]["stt"], 1)
        self.assertEqual(summary["by_component"]["answer_generation"], 1)
        self.assertEqual(summary["by_category"]["AUDIO_HARDWARE"], 1)
        self.assertEqual(summary["by_category"]["RATE_LIMIT"], 1)
        self.assertEqual(summary["by_category"]["TIMEOUT"], 1)
        self.assertEqual(len(summary["recent_errors"]), 3)

        # 3. Test ring buffer bounding (overflow eviction past 100 items)
        for i in range(120):
            self.controller.record_error("ui", f"Transient UI render glitch #{i}", category="UI")

        bounded_summary = self.controller.get_error_summary()
        self.assertEqual(bounded_summary["total_errors"], 100, "Ring buffer must be strictly bounded at 100 items")
        self.assertEqual(len(self.controller._error_ring_buffer), 100)
        # Most recent error must be the latest pushed item
        self.assertEqual(self.controller._error_ring_buffer[-1]["message"], "Transient UI render glitch #119")

    def test_component_health_and_overall_system_status(self):
        """Verify system status transitions between HEALTHY, DEGRADED, and ERROR."""
        self.controller.reset()
        self.assertEqual(self.controller.get_overall_system_status(), "HEALTHY")

        # Non-critical component error -> DEGRADED
        self.controller.update_component("screen_analysis", "ERROR", "Screen capture OCR failed")
        self.assertEqual(self.controller.get_overall_system_status(), "DEGRADED")

        # Critical component (audio_capture) in ERROR -> ERROR
        self.controller.update_component("audio_capture", "ERROR", "WASAPI hardware crashed")
        self.assertEqual(self.controller.get_overall_system_status(), "ERROR")

        # Critical component recovering -> DEGRADED
        self.controller.update_component("audio_capture", "RECOVERING", "Re-binding speaker loopback")
        self.assertEqual(self.controller.get_overall_system_status(), "DEGRADED")

        # All healthy -> HEALTHY
        self.controller.reset()
        self.assertEqual(self.controller.get_overall_system_status(), "HEALTHY")

    def test_heartbeat_staleness_and_recovery(self):
        """Verify context-aware staleness detection flags stalled audio and hung generation."""
        # 1. Test generation timeout recovery
        self.controller.start_generation(gen_id=99, question_id="Q999")
        self.controller.active_generation_info["start_time"] = time.time() - 30.0  # 30s elapsed (>25s timeout)
        
        anomalies = self.controller.check_health_and_staleness()
        self.assertTrue(any(a["type"] == "GENERATION_TIMEOUT" for a in anomalies))
        self.assertEqual(self.controller.state, InterviewTurnState.LISTENING)
        self.assertEqual(self.controller.active_generation_info["state"], "IDLE")

        # 2. Test audio stream stall check
        mock_client = MagicMock()
        transcriber = ContinuousTranscriber(
            client=mock_client,
            on_transcript_chunk=MagicMock(),
            session=self.session
        )
        self.session.transcriber = transcriber
        transcriber.is_running = True
        transcriber._last_system_frame_time = time.perf_counter() - 5.0  # 5s since last frame (>4.0s stall threshold)
        transcriber._last_mic_frame_time = 0.0

        audio_anomalies = self.controller.check_health_and_staleness()
        self.assertTrue(any(a["type"] == "AUDIO_STALL" for a in audio_anomalies))
        self.assertEqual(self.controller.components_health["audio_capture"]["status"], "DEGRADED")

    def test_telemetry_snapshot_budget(self):
        """Verify get_telemetry_snapshot executes in < 0.5ms and returns all schema keys."""
        # Populate with data
        tr = QuestionLatencyTrace(question_id="Q001", generation_id=1)
        tr.mark("capture_start", 1.0)
        tr.mark("generation_end", 2.0)
        self.session.latency_traces.append(tr)
        self.controller.record_error("stt", "Test warning", category="STT_API")

        # Measure latency over 100 calls
        t0 = time.perf_counter()
        iters = 100
        for _ in range(iters):
            snapshot = self.controller.get_telemetry_snapshot()
        avg_ms = (time.perf_counter() - t0) * 1000.0 / iters

        self.assertLess(avg_ms, 0.5, f"Telemetry snapshot must execute in < 0.5ms (measured {avg_ms:.4f}ms)")
        
        # Verify complete schema
        expected_keys = [
            "system_status", "session_state", "components",
            "audio_quality", "latency", "errors", "active_generation", "timestamp"
        ]
        for k in expected_keys:
            self.assertIn(k, snapshot, f"Snapshot missing schema key: {k}")

        self.assertIn(snapshot["system_status"], ("HEALTHY", "DEGRADED", "ERROR"))
        self.assertIn("trace_count", snapshot["latency"])
        self.assertIn("total_frames", snapshot["audio_quality"])
        self.assertIn("total_errors", snapshot["errors"])

    def test_session_analytics_telemetry_integration(self):
        """Verify calculate_session_analytics preserves all legacy keys while enriching with telemetry."""
        # Record sample question analytics
        self.session.question_analytics.append({
            "category": "technical_explanation",
            "generation_time_sec": 1.2,
            "word_count": 140,
            "topic": "Python Concurrency",
            "required_refinement": False,
            "required_clarification": False,
            "confidence": 0.92
        })
        self.session.qa_pairs.append({
            "question": "How does asyncio work?",
            "answer": "Asyncio uses an event loop..."
        })

        analytics = calculate_session_analytics(self.session)

        # 1. Verify all 14 legacy keys exist and match types
        legacy_keys = [
            "total_questions", "technical_count", "coding_count", "behavioral_count",
            "hr_count", "project_count", "system_design_count", "avg_generation_time_sec",
            "avg_answer_length_words", "questions_requiring_refinement",
            "questions_requiring_clarification", "low_confidence_count",
            "topics_discussed", "preparation_areas"
        ]
        for lk in legacy_keys:
            self.assertIn(lk, analytics, f"Legacy analytics key missing: {lk}")

        self.assertEqual(analytics["total_questions"], 1)
        self.assertEqual(analytics["technical_count"], 1)
        self.assertEqual(analytics["topics_discussed"], ["Python Concurrency"])

        # 2. Verify enriched Category 9 telemetry dictionaries
        self.assertIn("latency_summary", analytics)
        self.assertIn("audio_quality_summary", analytics)
        self.assertIn("error_summary", analytics)

        self.assertIsInstance(analytics["latency_summary"], dict)
        self.assertIsInstance(analytics["audio_quality_summary"], dict)
        self.assertIsInstance(analytics["error_summary"], dict)


if __name__ == "__main__":
    unittest.main()
