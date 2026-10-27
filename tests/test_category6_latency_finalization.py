"""
Test Suite: Category 6 - Latency, Adaptive Semantic Finalization & Stale Answer Protection
========================================================================================
Tests Parakeet Audit Matrix Category 6 and Checklist Items 6, 7, 26, 27, 28, 30, 33, 34:
1. Incomplete question trailing conjunction holds (premature answer prevention).
2. Complete question semantic finalization.
3. Grounded short follow-up handling ("Why?", "How?", "What about Redis?").
4. Rephrase / retraction cue detection and active generation invalidation.
5. In-flight stream cancellation with explicit socket/stream abort.
6. Generation timeout auto-recovery and telemetry error logging.
7. 429 rate limit Retry-After parsing and fast failover.
8. Monotonic latency trace metrics computation and sub-millisecond classification budget.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import time
import pytest
from unittest.mock import MagicMock
from typing import Dict, Any

from interview_engine import (
    InterviewSession,
    SessionController,
    InterviewTurnState,
    QuestionLatencyTrace,
    evaluate_question_completeness,
    classify_and_understand_utterance,
    generate_interview_answer,
    parse_retry_after_seconds
)


def test_incomplete_question_trailing_conjunction_hold():
    """Verify evolving questions with trailing conjunctions/stems are held back (<1% premature)."""
    incomplete_samples = [
        "Can you explain how you designed the database and",
        "What is the difference between TCP and",
        "How would you optimize the caching layer while",
        "Could you implement a function that takes an array or",
        "Why did you choose MongoDB instead of PostgreSQL because"
    ]
    
    for text in incomplete_samples:
        res = evaluate_question_completeness(text)
        assert res["is_semantically_complete"] is False, f"Must not finalize incomplete stem: '{text}'"
        assert res["can_start_early"] is False
        assert res["completeness"] < 0.5


def test_complete_question_semantic_finalization():
    """Verify complete interrogative and imperative questions finalize with high completeness."""
    complete_samples = [
        "Can you explain how you designed the database?",
        "What is the difference between a process and a thread?",
        "How would you optimize the caching layer in production?",
        "Implement a binary search algorithm in Python.",
        "Design a high throughput URL shortener service."
    ]

    for text in complete_samples:
        res = evaluate_question_completeness(text)
        assert res["is_semantically_complete"] is True, f"Must finalize complete question: '{text}'"
        assert res["can_start_early"] is True
        assert res["completeness"] >= 0.85


def test_short_followup_grounded_in_active_topic():
    """Verify short follow-ups finalize immediately when active conversation topic exists."""
    session = InterviewSession()
    session.active_topic = "Redis Distributed Caching"
    session.add_qa("Why did you use Redis?", "For sub-millisecond key-value lookups.")

    short_followups = ["Why?", "How?", "And why?", "And latency?", "What about MongoDB?"]
    for q in short_followups:
        res = evaluate_question_completeness(q, session=session)
        assert res["is_semantically_complete"] is True, f"Must finalize grounded follow-up: '{q}'"
        assert res["can_start_early"] is True
        assert res["completeness"] >= 0.90


def test_rephrase_detection_and_invalidation():
    """Verify interviewer retraction/rephrase cancels active generation and clears state."""
    session = InterviewSession()
    session.controller = SessionController(session)
    session.controller.start_generation(101, "q_old")
    assert session.controller.state == InterviewTurnState.GENERATING_ANSWER

    rephrase_utterance = "Actually, let me rephrase that question."
    result = classify_and_understand_utterance(
        utterance=rephrase_utterance,
        session=session,
        source="system"
    )

    assert result["utterance_type"] == "INTERVIEWER_REPHRASE"
    assert result["should_generate_answer"] is False
    assert session.controller.state == InterviewTurnState.WAITING_FOR_NEXT_QUESTION


def test_stale_generation_stream_abort():
    """Verify cancelled in-flight stream immediately closes underlying socket and aborts."""
    session = InterviewSession()

    class MockChunkDelta:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.delta = MockChunkDelta(content)

    class MockChunk:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    class MockStream:
        def __init__(self):
            self.closed = False
            self.chunks = [
                MockChunk("First "),
                MockChunk("second "),
                MockChunk("third "),
                MockChunk("fourth ")
            ]
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.idx >= len(self.chunks):
                raise StopIteration
            c = self.chunks[self.idx]
            self.idx += 1
            return c

        def close(self):
            self.closed = True

    mock_stream = MockStream()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_stream
    session.model = "llama-3.3-70b-versatile"
    session.model_name = "llama-3.3-70b-versatile"

    tokens_received = []
    # Cancel after 1st chunk is consumed
    def cancel_after_one():
        return mock_stream.idx >= 1

    def on_tok(tok):
        tokens_received.append(tok)

    # Patch get_openai_client to return None so mock_client (Groq) is used
    import interview_engine
    orig_get_openai = interview_engine.get_openai_client
    interview_engine.get_openai_client = lambda: None
    try:
        result = generate_interview_answer(
            client=mock_client,
            session=session,
            question="Explain indexing in PostgreSQL",
            on_token=on_tok,
            is_cancelled_callback=cancel_after_one
        )
    finally:
        interview_engine.get_openai_client = orig_get_openai

    assert result == ""
    assert mock_stream.closed is True, "Stream socket must be explicitly closed on cancellation"


def test_check_generation_timeout_telemetry_recording():
    """Verify generation timeout resets state, triggers recovery, and logs telemetry error."""
    session = InterviewSession()
    controller = SessionController(session)
    controller.start_generation(201, "q_timeout")
    controller.active_generation_info["start_time"] = time.time() - 30.0  # 30s ago (>25s timeout)

    timed_out = controller.check_generation_timeout()
    assert timed_out is True
    assert controller.state == InterviewTurnState.LISTENING

    # Check error telemetry in controller
    error_summary = controller.get_error_summary()
    assert error_summary["total_errors"] >= 1
    assert error_summary["by_category"].get("TIMEOUT", 0) >= 1
    assert any(e["category"] == "TIMEOUT" for e in error_summary["recent_errors"])


def test_parse_retry_after_and_bounded_backoff():
    """Verify Retry-After extraction and live call thresholds."""
    err_sec = Exception("Error 429: Rate limit exceeded, try again in 1.4s")
    assert parse_retry_after_seconds(err_sec) == 1.4

    err_ms = Exception("Error 429: Rate limit exceeded, try again in 800ms")
    assert parse_retry_after_seconds(err_ms) == 0.8

    err_hdr = Exception("Rate limit reached. retry-after: 3.5")
    assert parse_retry_after_seconds(err_hdr) == 3.5


def test_latency_trace_instrumentation_budget():
    """Verify latency trace metrics computation and fast-path classification budget."""
    trace = QuestionLatencyTrace(question_id="trace_test_1", generation_id=1)
    t0 = time.perf_counter()
    trace.mark("utterance_detected", t0)
    trace.mark("context_ready", t0 + 0.010)
    trace.mark("question_finalized", t0 + 0.025)
    trace.mark("generation_start", t0 + 0.030)
    trace.mark("first_token", t0 + 0.220)
    trace.mark("first_useful_token", t0 + 0.350)
    trace.mark("generation_end", t0 + 0.950)

    metrics = trace.compute_metrics()
    assert metrics["generation_start_to_first_token_ms"] == pytest.approx(190.0, abs=5.0)
    assert metrics["total_generation_duration_ms"] == pytest.approx(920.0, abs=10.0)
    assert metrics["total_question_to_displayed_answer_ms"] == pytest.approx(325.0, abs=5.0)

    # Fast-path classification latency benchmark (<0.2ms per call)
    iterations = 500
    bench_t0 = time.perf_counter()
    for _ in range(iterations):
        evaluate_question_completeness("Can you explain polymorphism in object-oriented programming?")
    avg_ms = ((time.perf_counter() - bench_t0) / iterations) * 1000.0

    print(f"\n[Classification Latency] Average evaluation time: {avg_ms:.4f} ms per call")
    assert avg_ms < 0.2, f"Question completeness evaluation must be < 0.2ms, took {avg_ms:.4f}ms"
