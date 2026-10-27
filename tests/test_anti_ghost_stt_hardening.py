"""
Test Suite for Anti-Ghost STT & Hallucination Elimination
=========================================================
Tests:
1. 100% elimination of standalone ghost text generated on silence/noise:
   - "Hello?", "Yeah", "Right.", "Okay", "[coughing]", "(laughter)", "*sigh*", "Can you hear me?"
2. 100% preservation of genuine speech, questions, and conversational lead-ins:
   - "Hello, can you hear me properly?"
   - "Right, how do you handle optimistic concurrency?"
   - "Yeah, that makes sense. Can we discuss caching?"
   - "Why did you choose PostgreSQL over MySQL?"
3. VAD noise floor rejection vs. soft speech preservation.
4. Smart WebSocket bypass in _dispatch_final_transcription.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import time
import unittest
import numpy as np
from interview_engine import (
    _is_whisper_hallucination,
    clean_asr_transcript,
    ContinuousTranscriber,
    AudioSegment
)
from speech import (
    is_whisper_hallucination,
    compute_audio_diagnostics,
    SystemAudioProcessor,
    MicrophoneAudioProcessor,
    safe_automatic_gain_control
)


class TestAntiGhostSTTHardening(unittest.TestCase):

    # =========================================================================
    # 1. Standalone Ghost Words Elimination (Silence / Ambient Noise)
    # =========================================================================
    def test_standalone_ghost_words_eliminated(self):
        """Verifies that phantom tokens generated from silence or noise are 100% rejected."""
        ghost_samples = [
            "Hello",
            "hello?",
            "Hello.",
            "Yeah",
            "yeah.",
            "Yes",
            "yes.",
            "Right",
            "right.",
            "All right.",
            "alright",
            "Okay",
            "ok",
            "okay.",
            "um",
            "uh",
            "hmm",
            "ah",
            "oh",
            "you",
            "thank you",
            "thanks.",
            "bye",
            "goodbye",
            "can you hear me",
            "can you hear me?",
            "are you there?",
            "am i audible?",
            "sorry",
            "i'm sorry.",
            "welcome",
            "[coughing]",
            "[sigh]",
            "(laughter)",
            "*cough*",
            "[applause]",
            "(breathing)",
            "so",
            "and",
            "the",
        ]
        for ghost in ghost_samples:
            self.assertTrue(
                _is_whisper_hallucination(ghost),
                f"Ghost utterance '{ghost}' must be classified as hallucination in interview_engine"
            )
            self.assertEqual(
                clean_asr_transcript(ghost),
                "",
                f"clean_asr_transcript must return empty string for ghost '{ghost}'"
            )
            self.assertTrue(
                is_whisper_hallucination(ghost),
                f"Ghost utterance '{ghost}' must be classified as hallucination in speech.py"
            )

    # =========================================================================
    # 2. Genuine Speech & Multi-Word Questions Preservation (100% Sensitivity)
    # =========================================================================
    def test_genuine_speech_and_questions_preserved(self):
        """Verifies that real questions starting with conversational words are NEVER filtered."""
        genuine_questions = [
            "Hello, can you hear me properly?",
            "Right, how do you handle optimistic concurrency in PostgreSQL?",
            "Yeah, that makes sense. Can we discuss database sharding?",
            "Okay, what is the difference between a process and a thread?",
            "Why did you choose MongoDB over PostgreSQL?",
            "Explain how Raft consensus works in distributed systems.",
            "So, how would you design a rate limiter using Redis?",
            "Yes, I implemented JWT authentication with refresh token rotation.",
            "Can you explain the trade-offs of microservices vs monoliths?",
            "Hello everyone, welcome to the system design round."
        ]
        for speech in genuine_questions:
            self.assertFalse(
                _is_whisper_hallucination(speech),
                f"Genuine speech '{speech}' must NOT be classified as hallucination"
            )
            cleaned = clean_asr_transcript(speech)
            self.assertTrue(
                len(cleaned) > 0,
                f"clean_asr_transcript must preserve genuine speech '{speech}'"
            )
            self.assertFalse(
                is_whisper_hallucination(speech),
                f"is_whisper_hallucination must NOT filter genuine speech '{speech}'"
            )

    # =========================================================================
    # 3. AGC Noise Gate & Soft Voice Dynamics
    # =========================================================================
    def test_agc_does_not_amplify_silence_or_ambient_room_noise(self):
        """Verifies AGC does not amplify low baseline room noise (< 35 RMS)."""
        # Low ambient room noise (RMS ~20)
        ambient_noise = np.random.randint(-25, 25, 480, dtype=np.int16)
        diag = compute_audio_diagnostics(ambient_noise, noise_floor_rms=40.0)
        self.assertLess(diag.rms, 35.0)

        boosted = safe_automatic_gain_control(
            ambient_noise,
            target_rms=2800.0,
            max_gain_db=16.0,
            noise_floor_rms=40.0
        )
        # Must return unamplified samples
        np.testing.assert_array_equal(ambient_noise, boosted)

    def test_agc_levels_soft_voice_cleanly(self):
        """Verifies genuine soft speech (RMS ~55) above noise floor is leveled cleanly."""
        t = np.linspace(0, 480 / 16000, 480, endpoint=False)
        soft_voice = (np.sin(2 * np.pi * 180 * t) * 75).astype(np.int16)
        diag = compute_audio_diagnostics(soft_voice, noise_floor_rms=25.0)
        self.assertGreater(diag.rms, 45.0)

        boosted = safe_automatic_gain_control(
            soft_voice,
            target_rms=2600.0,
            max_gain_db=12.0,
            noise_floor_rms=25.0
        )
        boosted_diag = compute_audio_diagnostics(boosted, noise_floor_rms=25.0)
        self.assertGreater(boosted_diag.rms, diag.rms, "Soft voice must receive clean leveling")

    # =========================================================================
    # 4. Dispatch Final Transcription WebSocket Guard
    # =========================================================================
    def test_dispatch_final_transcription_bypasses_when_websocket_connected(self):
        """Verifies _dispatch_final_transcription skips redundant REST execution when WebSocket is connected."""
        class MockClient:
            pass

        class MockWS:
            is_running = True
            ws = object()  # Connected socket instance

        transcriber = ContinuousTranscriber(
            client=MockClient(),
            on_transcript_chunk=lambda text, is_q, src="system": None,
            provider=None
        )
        transcriber.live_dg_ws_system = MockWS()

        now = time.perf_counter()
        seg = AudioSegment(
            segment_id="seg_test_bypass",
            source="system",
            audio_bytes=b"\x00" * 3200,
            sample_rate=16000,
            capture_start=now - 0.5,
            capture_end=now,
            queue_enter=now - 0.5,
            processing_start=now - 0.4,
            speech_frames=10
        )

        # Calling _dispatch_final_transcription must return immediately without queuing to executor
        seq_before = transcriber._stt_seq_counter
        transcriber._dispatch_final_transcription(seg, source="system")
        seq_after = transcriber._stt_seq_counter
        self.assertEqual(seq_before, seq_after, "Must skip execution when WebSocket is actively connected")

    def test_dispatch_final_transcription_runs_when_websocket_offline(self):
        """Verifies _dispatch_final_transcription processes when WebSocket is offline (seamless failover)."""
        dispatched = []

        class MockClient:
            pass

        class MockProvider:
            def transcribe(self, audio_bytes, sample_rate=16000):
                return "Explain microservices architecture."

        transcriber = ContinuousTranscriber(
            client=MockClient(),
            on_transcript_chunk=lambda text, is_q, src="system": dispatched.append((text, is_q, src)),
            provider=MockProvider()
        )
        transcriber.live_dg_ws_system = None  # Offline / fallback mode

        # Synthesize real speech tone
        t = np.linspace(0, 4800 / 16000, 4800, endpoint=False)
        tone = (np.sin(2 * np.pi * 220 * t) * 5000).astype(np.int16)

        now = time.perf_counter()
        seg = AudioSegment(
            segment_id="seg_test_failover",
            source="system",
            audio_bytes=tone.tobytes(),
            sample_rate=16000,
            capture_start=now - 0.5,
            capture_end=now,
            queue_enter=now - 0.5,
            processing_start=now - 0.4,
            speech_frames=10
        )

        transcriber._dispatch_final_transcription(seg, source="system")
        time.sleep(0.15)  # Wait for worker thread

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0][0], "Explain microservices architecture.")
        self.assertEqual(dispatched[0][2], "system")


if __name__ == "__main__":
    unittest.main()
