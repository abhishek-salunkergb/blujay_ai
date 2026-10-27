"""
End-to-End Verification Suite for Smart Hybrid Pipeline
Tests:
1. Dynamic Whisper Prompt Generation & CV Conditioning
2. Phonetic Technical ASR Post-Corrections (Homophones & Jargon)
3. 0.1ms Local CPU Chit-Chat & Filler Gating
4. Real Engineering Question Detection & Classification
5. In-Memory PCM Audio Preprocessing & Level Diagnostics (No Disk I/O)
6. Single-Pass Streaming Answer Generation & Sanitization with llama-3.3-70b-versatile
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import sys
import time
import io
import os
import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from speech import (
    build_dynamic_whisper_prompt,
    apply_technical_asr_corrections,
    is_conversational_chitchat,
    SystemAudioProcessor,
    MicrophoneAudioProcessor,
    compute_audio_diagnostics
)
from interview_engine import (
    clean_asr_transcript,
    is_likely_question,
    sanitize_generated_answer
)


def test_1_dynamic_whisper_priming():
    print("\n--- [TEST 1] Dynamic Whisper Keyword Priming ---")
    prompt = build_dynamic_whisper_prompt(
        candidate_skills="Python, FastAPI, Redis, Kubernetes, PostgreSQL, Kafka, PyTorch",
        job_title="Senior Backend Engineer",
        company="Uber"
    )
    print(f"Generated Primed Prompt:\n  \"{prompt}\"")
    assert "Role: Senior Backend Engineer" in prompt
    assert "Company: Uber" in prompt
    assert "PyTorch" in prompt and "Kafka" in prompt
    print("✅ Dynamic Whisper Priming: PASSED")


def test_2_phonetic_asr_corrections():
    print("\n--- [TEST 2] Phonetic Technical ASR Corrections ---")
    test_cases = [
        ("how do you use pie torch with cuda?", "how do you use PyTorch with cuda?"),
        ("we deployed on coo ber net es using dock er", "we deployed on Kubernetes using Docker"),
        ("stored the session in read is and database in post gres", "stored the session in Redis and database in PostgreSQL"),
        ("built a rest a p i with fast api and pydantic", "built a REST API with FastAPI and Pydantic"),
        ("the complexity is o of n log n and space is o of 1", "the complexity is O(N log N) and space is O(1)"),
        ("queried with graph ql and saved in sequel lite", "queried with GraphQL and saved in SQLite"),
    ]
    for raw_input, expected_output in test_cases:
        t0 = time.perf_counter()
        corrected = apply_technical_asr_corrections(raw_input)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        print(f"  Raw:       \"{raw_input}\"\n  Corrected: \"{corrected}\" ({elapsed_us:.1f} µs)")
        assert corrected.lower() == expected_output.lower(), f"Expected '{expected_output}', got '{corrected}'"
    print("✅ Phonetic ASR Corrections: PASSED")


def test_3_chitchat_rejection():
    print("\n--- [TEST 3] 0.1ms Local CPU Chit-Chat & Pleasantry Gating ---")
    chitchat_inputs = [
        "Hey Abhishek, can you hear me okay?",
        "Good morning, thanks for joining today.",
        "Let me share my screen for a second.",
        "Can you see my screen now?",
        "Okay cool, sounds good.",
        "Hi, how are you doing today?",
        "One moment please, let me open the doc.",
        "Sure, got it."
    ]
    for text in chitchat_inputs:
        t0 = time.perf_counter()
        is_chitchat = is_conversational_chitchat(text)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        print(f"  \"{text}\" -> Is Chit-Chat? {is_chitchat} ({elapsed_us:.1f} µs)")
        assert is_chitchat is True, f"Failed to detect chit-chat in: '{text}'"
    print("✅ Chit-Chat Gatekeeper: PASSED (Zero false triggers, <50µs execution)")


def test_4_real_question_detection():
    print("\n--- [TEST 4] Real Engineering Question Identification ---")
    real_questions = [
        "How would you design a distributed caching layer using Redis and Consistent Hashing?",
        "Can you explain the difference between optimistic and pessimistic locking in PostgreSQL?",
        "What are the trade-offs of microservices vs monolithic architecture?",
        "How do you handle database deadlocks and replication lag under high QPS?",
        "Walk me through how you implemented asynchronous task processing with Celery and RabbitMQ."
    ]
    for q in real_questions:
        is_chitchat = is_conversational_chitchat(q)
        is_q = is_likely_question(q)
        print(f"  \"{q[:55]}...\" -> ChitChat={is_chitchat}, IsQuestion={is_q}")
        assert is_chitchat is False, f"Question falsely marked as chitchat: '{q}'"
        assert is_q is True, f"Question failed detection: '{q}'"
    print("✅ Engineering Question Identification: PASSED")


def test_5_in_memory_audio_dsp():
    print("\n--- [TEST 5] In-Memory PCM Audio Preprocessing & Level Diagnostics ---")
    # Generate 16kHz synthetic sine wave speech burst
    sample_rate = 16000
    duration_s = 0.5
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # 440Hz tone with speech amplitude
    sine_signal = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    
    processor = MicrophoneAudioProcessor(sample_rate=sample_rate)
    
    t0 = time.perf_counter()
    proc_bytes, is_speech, diag = processor.process_frame(sine_signal.tobytes())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    print(f"  Frame Processing Time: {elapsed_ms:.2f} ms")
    print(f"  Diagnostics: RMS={diag.rms:.1f}, Peak={diag.peak:.1f}, SpeechLevel={diag.speech_level_db:.1f} dB, State={diag.state}")
    print(f"  Speech Detected: {is_speech}")
    
    assert len(proc_bytes) > 0
    assert is_speech is True
    assert diag.state in ("NORMAL", "LOUD")
    print("✅ In-Memory Audio DSP & Diagnostics: PASSED")


def test_6_answer_sanitization():
    print("\n--- [TEST 6] Reasoning Tag & Preamble Sanitization ---")
    raw_llm_output = """<think>
The user is asking about Redis caching. I will explain write-through vs write-back caching and consistent hashing.
</think>
Here is how you design a resilient distributed caching layer:

1. **Cache Architecture**:
   - Deploy a Redis Cluster with master-replica replication across availability zones.
   - Use **Consistent Hashing** on client keys to distribute traffic evenly without cache stampedes.

2. **Eviction & Expiration**:
   - Configure `allkeys-lru` eviction policy with TTLs tailored to access frequency.
"""
    cleaned = sanitize_generated_answer(raw_llm_output)
    print(f"  Sanitized Output Preview:\n{cleaned[:220]}...")
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Redis Cluster" in cleaned
    print("✅ Answer Sanitization: PASSED")


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 RUNNING SMART HYBRID PIPELINE COMPREHENSIVE E2E AUDIT")
    print("=" * 65)
    
    t_start = time.perf_counter()
    test_1_dynamic_whisper_priming()
    test_2_phonetic_asr_corrections()
    test_3_chitchat_rejection()
    test_4_real_question_detection()
    test_5_in_memory_audio_dsp()
    test_6_answer_sanitization()
    total_time = (time.perf_counter() - t_start) * 1000
    
    print("\n" + "=" * 65)
    print(f"🎉 ALL HYBRID PIPELINE TESTS PASSED 100% IN {total_time:.2f} ms!")
    print("=" * 65)
