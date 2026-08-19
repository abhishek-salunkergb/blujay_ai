"""
Audio Quality, Preprocessing, STT & Benchmark Test Suite
========================================================
Comprehensive verification of:
1. Signal diagnostics (RMS, Peak, SNR, Clipping Ratio, State)
2. DC offset removal & signal baseline
3. Safe Automatic Gain Control (AGC) & anti-clipping limiter
4. System & Microphone audio processor behavior
5. Pre-speech ring buffering & plosive consonant preservation
6. Safe rolling transcript overlap merging without duplicate words
7. Technical terminology normalization across 25 benchmark interview questions
8. Source-role isolation (System Loopback = Interviewer -> Answer, Mic = Candidate -> Context only)
"""

import unittest
import numpy as np
import time
from unittest.mock import MagicMock

from speech import (
    compute_audio_diagnostics,
    remove_dc_offset,
    safe_automatic_gain_control,
    SystemAudioProcessor,
    MicrophoneAudioProcessor,
    merge_overlapping_transcripts,
    AudioDiagnostics
)

from interview_engine import (
    clean_asr_transcript,
    ContinuousTranscriber,
    InterviewSession,
    STTProvider,
    is_likely_question
)


class MockSTTProvider(STTProvider):
    def __init__(self, transcript_return: str = "Why did you choose MongoDB instead of PostgreSQL?"):
        self.transcript_return = transcript_return
        self.call_count = 0

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "", language: str = None) -> str:
        self.call_count += 1
        return self.transcript_return


class TestAudioQualityAndSTTBenchmark(unittest.TestCase):

    # ============================================================
    # 1. AUDIO QUALITY DIAGNOSTICS & MEASUREMENT
    # ============================================================

    def test_audio_diagnostics_silent_signal(self):
        """Pure silence (RMS < 180) must be classified as SILENT."""
        silence = np.zeros(480, dtype=np.int16)
        diag = compute_audio_diagnostics(silence)
        self.assertEqual(diag.state, "SILENT")
        self.assertEqual(diag.rms, 0.0)
        self.assertEqual(diag.clipping_ratio, 0.0)

    def test_audio_diagnostics_low_signal(self):
        """Soft / quiet speech (RMS ~ 500) must be classified as LOW_SIGNAL."""
        low_signal = (np.sin(np.linspace(0, 10 * np.pi, 480)) * 500).astype(np.int16)
        diag = compute_audio_diagnostics(low_signal, noise_floor_rms=100.0)
        self.assertEqual(diag.state, "LOW_SIGNAL")
        self.assertTrue(300.0 < diag.rms < 600.0)

    def test_audio_diagnostics_normal_signal(self):
        """Normal conversational speech (RMS ~ 3000) must be classified as NORMAL."""
        normal_signal = (np.sin(np.linspace(0, 10 * np.pi, 480)) * 4000).astype(np.int16)
        diag = compute_audio_diagnostics(normal_signal, noise_floor_rms=200.0)
        self.assertEqual(diag.state, "NORMAL")
        self.assertTrue(2000.0 < diag.rms < 4000.0)

    def test_audio_diagnostics_clipping_detection(self):
        """Severely hot / saturated signal must be classified as CLIPPING."""
        clipped_signal = np.full(480, 32767, dtype=np.int16)
        diag = compute_audio_diagnostics(clipped_signal)
        self.assertEqual(diag.state, "CLIPPING")
        self.assertGreater(diag.clipping_ratio, 0.5)

    # ============================================================
    # 2. DC OFFSET REMOVAL
    # ============================================================

    def test_remove_dc_offset(self):
        """Removes constant electrical DC bias while preserving waveform variance."""
        biased = (np.sin(np.linspace(0, 10 * np.pi, 480)) * 1000 + 500).astype(np.int16)
        cleaned = remove_dc_offset(biased)
        self.assertAlmostEqual(float(np.mean(cleaned)), 0.0, delta=1.0)

    # ============================================================
    # 3. SAFE AUTOMATIC GAIN CONTROL (AGC) & LIMITER
    # ============================================================

    def test_safe_agc_boosts_low_signal(self):
        """Low-level speech (RMS 600) is cleanly boosted towards target (~2600) within +12dB limit."""
        low_signal = (np.sin(np.linspace(0, 10 * np.pi, 480)) * 800).astype(np.int16)
        boosted = safe_automatic_gain_control(low_signal, target_rms=2600.0, max_gain_db=12.0, noise_floor_rms=100.0)
        boosted_rms = float(np.sqrt(np.mean(boosted.astype(np.float32)**2)))
        self.assertGreater(boosted_rms, 1500.0)
        self.assertLessEqual(np.max(np.abs(boosted)), 32767)

    def test_safe_agc_does_not_boost_silence(self):
        """Pure silence/room noise floor is NOT amplified (prevents hiss pumping)."""
        silence = np.random.randint(-50, 50, size=480, dtype=np.int16)
        processed = safe_automatic_gain_control(silence, target_rms=2600.0, max_gain_db=12.0, noise_floor_rms=200.0)
        orig_peak = np.max(np.abs(silence))
        proc_peak = np.max(np.abs(processed))
        # Amplitude of silence must remain identical or unboosted
        self.assertLessEqual(proc_peak, orig_peak + 5)

    def test_safe_agc_anti_clipping_limiter(self):
        """Loud hot peaks approaching int16 limits are soft-compressed without digital wrap-around."""
        hot_signal = (np.sin(np.linspace(0, 10 * np.pi, 480)) * 32000).astype(np.int16)
        processed = safe_automatic_gain_control(hot_signal, target_rms=2600.0, max_gain_db=12.0, noise_floor_rms=200.0)
        self.assertLessEqual(np.max(np.abs(processed)), 32767)
        self.assertGreater(np.max(np.abs(processed)), 20000)

    # ============================================================
    # 4. SYSTEM & MICROPHONE AUDIO PROCESSORS
    # ============================================================

    def test_system_audio_processor_float32_input(self):
        """SystemAudioProcessor converts float32 loopback buffers [-1.0, 1.0] to leveled int16 PCM."""
        processor = SystemAudioProcessor(sample_rate=16000)
        float_speech = (np.sin(np.linspace(0, 10 * np.pi, 480)) * 0.15).astype(np.float32)
        pcm_bytes, is_speech, diag = processor.process_frame(float_speech)
        self.assertEqual(len(pcm_bytes), 480 * 2)
        self.assertTrue(is_speech)
        self.assertIsNotNone(diag)

    def test_microphone_audio_processor_noise_tracking(self):
        """MicrophoneAudioProcessor dynamically tracks ambient noise floor without false speech triggers."""
        processor = MicrophoneAudioProcessor(sample_rate=16000)
        fan_noise = np.random.randint(-150, 150, size=480, dtype=np.int16).tobytes()
        # Feed 10 frames of fan noise
        for _ in range(10):
            pcm_bytes, is_speech, diag = processor.process_frame(fan_noise)
        self.assertFalse(is_speech)
        self.assertLessEqual(processor.noise_floor_rms, 400.0)

    # ============================================================
    # 5. TRANSCRIPT OVERLAP & TOKEN DEDUPLICATION
    # ============================================================

    def test_merge_overlapping_transcripts_exact_overlap(self):
        """Merges overlapping rolling windows without duplicating shared words."""
        prev_text = "Why did you choose MongoDB"
        new_text = "MongoDB instead of PostgreSQL?"
        merged = merge_overlapping_transcripts(prev_text, new_text)
        self.assertEqual(merged, "Why did you choose MongoDB instead of PostgreSQL?")

    def test_merge_overlapping_transcripts_multi_word_overlap(self):
        """Merges 3-word overlap correctly."""
        prev_text = "How did you implement authentication using JWT"
        new_text = "authentication using JWT in your Django project?"
        merged = merge_overlapping_transcripts(prev_text, new_text)
        self.assertEqual(merged, "How did you implement authentication using JWT in your Django project?")

    def test_merge_overlapping_transcripts_contained_prefix(self):
        """Handles when new text starts with prev text."""
        prev_text = "Tell me about"
        new_text = "Tell me about your distributed architecture"
        merged = merge_overlapping_transcripts(prev_text, new_text)
        self.assertEqual(merged, "Tell me about your distributed architecture")

    # ============================================================
    # 6. TECHNICAL VOCABULARY & HOMOPHONE CORRECTION (25 BENCHMARK PHRASES)
    # ============================================================

    def test_technical_terminology_benchmark_25_phrases(self):
        """Validates 25 realistic interview phrases for technical normalization."""
        benchmark_cases = [
            ("how did you use pie torch in your ml pipeline", "PyTorch"),
            ("why did you choose post gres instead of mysql", "PostgreSQL"),
            ("we migrated our database to mongo db", "MongoDB"),
            ("i built a microservice using fast api", "FastAPI"),
            ("we used j w t tokens stored in cookies", "JWT"),
            ("real-time communication was done via web sockets", "WebSockets"),
            ("we designed a rest api with swagger docs", "REST API"),
            ("we queried our graph ql backend", "GraphQL"),
            ("the frontend was built using react js and redux", "React.js"),
            ("our server runs on node js with express", "Node.js"),
            ("we upgraded our dashboard to next js", "Next.js"),
            ("i wrote the frontend in type script", "TypeScript"),
            ("we trained an open ai model with fine tuning", "OpenAI"),
            ("our cluster runs on k8s in production", "Kubernetes"),
            ("we deployed our containers with docker and k8s", "Docker"),
            ("we used celery and rabbit mq for background jobs", "RabbitMQ"),
            ("we authenticated users via oauth2", "OAuth2"),
            ("our data pipeline uses num py and pan das", "NumPy"),
            ("we deployed microservices on a w s and gcp", "AWS"),
            ("what is the time complexity of binary search", "binary search"),
            ("can you explain cache invalidation strategies", "cache invalidation"),
            ("how do you handle distributed transactions", "distributed transactions"),
            ("write a function to reverse a linked list", "reverse a linked list"),
            ("tell me about a difficult technical challenge", "technical challenge"),
            ("what are the trade-offs of microservices", "trade-offs")
        ]

        passed = 0
        for raw_input, expected_term in benchmark_cases:
            cleaned = clean_asr_transcript(raw_input)
            self.assertIn(expected_term.lower(), cleaned.lower(), f"Failed to preserve/normalize {expected_term} in '{cleaned}'")
            passed += 1

        self.assertEqual(passed, 25)

    # ============================================================
    # 7. SOURCE-ROLE ISOLATION (SYSTEM = INTERVIEWER, MIC = CANDIDATE)
    # ============================================================

    def test_dual_source_roles_system_vs_candidate(self):
        """Verifies system loopback audio routes to interviewer while mic routes to candidate context."""
        session = InterviewSession()
        mock_provider = MockSTTProvider("Why did you choose PostgreSQL?")
        
        captured_chunks = []
        def on_chunk(text, is_q, source="system"):
            captured_chunks.append({"text": text, "is_q": is_q, "source": source})

        transcriber = ContinuousTranscriber(
            client=MagicMock(),
            on_transcript_chunk=on_chunk,
            provider=mock_provider,
            session=session
        )

        # 1. Dispatch from System Audio (Interviewer)
        frames = [b"\x00\x08" * 240 for _ in range(15)]
        transcriber._dispatch_final_transcription(frames, source="system")
        time.sleep(0.1)

        # 2. Dispatch from Candidate Microphone
        cand_provider = MockSTTProvider("I used PostgreSQL for its native JSONB support.")
        transcriber.provider = cand_provider
        transcriber._dispatch_final_transcription(frames, source="candidate_mic")
        time.sleep(0.1)

        self.assertEqual(len(captured_chunks), 2)
        self.assertEqual(captured_chunks[0]["source"], "system")
        self.assertEqual(captured_chunks[1]["source"], "candidate_mic")
        self.assertTrue(captured_chunks[0]["is_q"])


if __name__ == "__main__":
    unittest.main()
