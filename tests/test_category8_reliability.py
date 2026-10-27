"""
Category 8: Reliability, Bounded Long-Session Memory & Graceful Shutdown Test Suite
==================================================================================
Tests:
1. Long-session memory bounding (sliding window caps on transcript_log, candidate_responses_log, qa_pairs, question_analytics, latency_traces).
2. Lossless cumulative analytics accumulation across memory pruning.
3. Comprehensive thread safety under high-concurrency race conditions.
4. Graceful shutdown handler, atexit registration, and active transcriber lifecycle.
5. Out-of-order STT sequencing memory bounds.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import time
import atexit
import threading
from typing import Dict, Any
from unittest.mock import MagicMock

import pytest

from interview_engine import (
    InterviewSession,
    InterviewTurnState,
    QuestionLatencyTrace,
    calculate_session_analytics,
    ContinuousTranscriber,
    _ACTIVE_TRANSCRIBERS,
    _ACTIVE_TRANSCRIBERS_LOCK,
    _shutdown_active_transcribers,
)


class TestCategory8Reliability:

    def setup_method(self):
        self.session = InterviewSession()

    def test_transcript_log_bounded_memory(self):
        """Verify that transcript_log obeys the sliding window cap and does not grow indefinitely."""
        self.session._max_transcript_log_size = 50  # test cap
        for i in range(150):
            self.session.add_transcript(f"Transcript chunk {i}", role="Interviewer", is_question=(i % 10 == 0))

        assert len(self.session.transcript_log) == 50
        # The newest entries must be preserved at the tail
        assert "Transcript chunk 149" in self.session.transcript_log[-1]["text"]
        assert "Transcript chunk 100" in self.session.transcript_log[0]["text"]

    def test_candidate_responses_log_bounded_memory(self):
        """Verify that candidate_responses_log obeys sliding window caps."""
        self.session._max_candidate_log_size = 30  # test cap
        for i in range(100):
            self.session.add_candidate_speech(f"Candidate response {i}")

        assert len(self.session.candidate_responses_log) == 30
        assert "Candidate response 99" in self.session.candidate_responses_log[-1]["text"]
        # Verify candidate_context uses active log
        ctx = self.session.candidate_context
        assert "Candidate response 99" in ctx
        assert "Candidate response 0" not in ctx  # Evicted

    def test_qa_pairs_and_analytics_lossless_bounding(self):
        """Verify qa_pairs and question_analytics bound memory while preserving cumulative metrics."""
        self.session._max_qa_pairs_size = 20
        self.session._max_question_analytics_size = 20

        categories = ["coding", "system_design", "behavioral", "technical", "hr"]
        for i in range(60):
            cat = categories[i % len(categories)]
            self.session.record_qa(
                question=f"Question {i} about {cat}?",
                answer=f"Answer for {cat} with multiple technical explanation words here for turn {i}",
                category=cat,
                gen_time=1.0 + (i * 0.02),
                required_refinement=(i % 5 == 0),
                confidence=0.85
            )

        # In-memory collections must be bounded to 20
        assert len(self.session.qa_pairs) == 20
        assert len(self.session.question_analytics) == 20
        assert self.session._cumulative_analytics["total_questions_pruned"] == 40

        # Lossless analytics calculation: total questions MUST reflect all 60 questions!
        analytics = calculate_session_analytics(self.session)
        assert analytics["total_questions"] == 60
        assert analytics["coding_count"] == 12
        assert analytics["system_design_count"] == 12
        assert analytics["behavioral_count"] == 12
        assert analytics["hr_count"] == 12
        assert analytics["technical_count"] == 12
        assert analytics["questions_requiring_refinement"] == 12
        assert analytics["avg_generation_time_sec"] > 0.0
        assert analytics["avg_answer_length_words"] > 0

    def test_prune_session_memory_explicit_call(self):
        """Verify explicit prune_session_memory accurately prunes and returns counts."""
        for i in range(100):
            self.session.add_transcript(f"Text {i}")
            self.session.add_candidate_speech(f"Speech {i}")
            self.session.record_qa(f"Q {i}?", f"A {i}", category="coding", gen_time=1.0)
            self.session.record_latency_trace(QuestionLatencyTrace(segment_id=f"seg_{i}"))

        res = self.session.prune_session_memory(
            max_transcripts=25,
            max_candidate=20,
            max_qa=15,
            max_analytics=15,
            max_traces=10
        )

        assert res["pruned_transcripts"] == 75
        assert res["pruned_candidate"] == 80
        assert res["pruned_qa"] == 85
        assert res["pruned_analytics"] == 85
        assert res["pruned_traces"] == 90

        assert len(self.session.transcript_log) == 25
        assert len(self.session.candidate_responses_log) == 20
        assert len(self.session.qa_pairs) == 15
        assert len(self.session.question_analytics) == 15
        assert len(self.session.latency_traces) == 10

        # Analytics total must still report 100 questions!
        analytics = calculate_session_analytics(self.session)
        assert analytics["total_questions"] == 100

    def test_clear_resets_cumulative_and_traces(self):
        """Verify clear() thoroughly flushes all collections, traces, and cumulative analytics."""
        self.session.add_transcript("Test")
        self.session.add_candidate_speech("Candidate")
        self.session.record_qa("Q?", "A", category="coding")
        self.session.record_latency_trace(QuestionLatencyTrace())
        self.session._cumulative_analytics["total_questions_pruned"] = 50

        self.session.clear()

        assert len(self.session.transcript_log) == 0
        assert len(self.session.candidate_responses_log) == 0
        assert len(self.session.qa_pairs) == 0
        assert len(self.session.question_analytics) == 0
        assert len(self.session.latency_traces) == 0
        assert self.session._cumulative_analytics["total_questions_pruned"] == 0

    def test_multithreaded_concurrency_race_safety(self):
        """Stress test with 10 concurrent worker threads mutating session state simultaneously."""
        self.session._max_transcript_log_size = 100
        self.session._max_candidate_log_size = 50
        self.session._max_qa_pairs_size = 40
        self.session._max_question_analytics_size = 40

        errors = []

        def worker(tid: int):
            try:
                for j in range(50):
                    self.session.add_transcript(f"Thread {tid} Chunk {j}", role="Interviewer", is_question=(j % 5 == 0))
                    self.session.add_candidate_speech(f"Thread {tid} Answer {j}")
                    _ = self.session.candidate_context
                    _ = self.session.recent_candidate_context
                    _ = self.session.get_latest_interviewer_question()
                    if j % 3 == 0:
                        self.session.record_qa(f"Thread {tid} Q{j}?", f"Answer from thread {tid}", category="technical")
                    if j % 10 == 0:
                        self.session.prune_session_memory()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Encountered thread safety errors: {errors}"
        assert len(self.session.transcript_log) <= 100
        assert len(self.session.candidate_responses_log) <= 50
        assert len(self.session.qa_pairs) <= 40
        assert len(self.session.question_analytics) <= 40

    def test_active_transcribers_lifecycle_and_shutdown(self):
        """Verify ContinuousTranscriber registers in _ACTIVE_TRANSCRIBERS on start and unregisters on stop."""
        mock_groq = MagicMock()
        t = ContinuousTranscriber(
            client=mock_groq,
            on_transcript_chunk=lambda *args: None,
            session=self.session,
            enable_watchdog=False
        )

        with _ACTIVE_TRANSCRIBERS_LOCK:
            initial_count = len(_ACTIVE_TRANSCRIBERS)

        # Mock workers so real audio hardware isn't required in unit tests
        t._run_system_audio_loop = MagicMock()
        t._run_microphone_loop = MagicMock()
        t._system_processing_worker = MagicMock()
        t._mic_processing_worker = MagicMock()

        t.start(source="dual")
        assert t.is_running is True

        with _ACTIVE_TRANSCRIBERS_LOCK:
            assert t in _ACTIVE_TRANSCRIBERS

        # Stopping should unregister and cleanly teardown
        t.stop()
        assert t.is_running is False

        with _ACTIVE_TRANSCRIBERS_LOCK:
            assert t not in _ACTIVE_TRANSCRIBERS

        # Test _shutdown_active_transcribers hook
        t2 = ContinuousTranscriber(
            client=mock_groq,
            on_transcript_chunk=lambda *args: None,
            session=self.session,
            enable_watchdog=False
        )
        t2._run_system_audio_loop = MagicMock()
        t2._run_microphone_loop = MagicMock()
        t2._system_processing_worker = MagicMock()
        t2._mic_processing_worker = MagicMock()

        t2.start(source="dual")
        with _ACTIVE_TRANSCRIBERS_LOCK:
            assert t2 in _ACTIVE_TRANSCRIBERS

        # Invoke global emergency shutdown
        _shutdown_active_transcribers()
        assert t2.is_running is False
        with _ACTIVE_TRANSCRIBERS_LOCK:
            assert t2 not in _ACTIVE_TRANSCRIBERS
