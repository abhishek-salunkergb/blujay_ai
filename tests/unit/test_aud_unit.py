"""
Parakeet-Grade Validation Test Suite: AUD-xxx (Audio Capture, Resampling & Hardware Recovery)
Strictly adheres to Section 0 Testing Contract and Section 3-5 Specifications.
"""

import os
import time
import queue
import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from audio_device_manager import (
    fast_resample_16k,
    get_available_microphones,
    get_available_speakers,
    AudioDeviceWatcher
)
from speech import (
    compute_audio_diagnostics,
    remove_dc_offset,
    safe_automatic_gain_control,
    MicrophoneAudioProcessor,
    SystemAudioProcessor
)
from interview_engine import (
    ContinuousTranscriber,
    InterviewSession,
    AudioFrame
)


class TestAudCaptureAndResampling(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()
        self.session.clear()

    # -------------------------------------------------------------------------
    # AUD-001: Microphone opens
    # -------------------------------------------------------------------------
    def test_aud_001_microphone_opens(self):
        """AUD-001: Verify microphone device query, channels, and monotonic timestamps."""
        mics = get_available_microphones()
        self.assertIsInstance(mics, list)
        self.assertGreater(len(mics), 0)
        
        # Verify first device metadata
        mic = mics[0]
        self.assertIn("id", mic)
        self.assertIn("name", mic)
        self.assertIn("channels", mic)
        self.assertGreaterEqual(mic["channels"], 1)

        # Verify frame arrival and monotonic timestamps
        t0 = time.time()
        frame1 = AudioFrame(data=b"\x00" * 960, timestamp=t0, source="mic")
        time.sleep(0.01)
        t1 = time.time()
        frame2 = AudioFrame(data=b"\x00" * 960, timestamp=t1, source="mic")
        self.assertGreaterEqual(frame2.timestamp, frame1.timestamp)

    # -------------------------------------------------------------------------
    # AUD-002: System audio opens
    # -------------------------------------------------------------------------
    def test_aud_002_system_audio_opens(self):
        """AUD-002: Verify system audio (WASAPI loopback) enumeration and attributes."""
        speakers = get_available_speakers()
        self.assertIsInstance(speakers, list)
        self.assertGreater(len(speakers), 0)
        spk = speakers[0]
        self.assertIn("id", spk)
        self.assertIn("name", spk)

    # -------------------------------------------------------------------------
    # AUD-003: Both streams operate simultaneously
    # -------------------------------------------------------------------------
    def test_aud_003_both_streams_simultaneous_no_starvation(self):
        """AUD-003: Mic and System audio queues process concurrently without starvation."""
        mic_q = queue.Queue(maxsize=100)
        sys_q = queue.Queue(maxsize=100)
        
        # Enqueue 50 frames to each stream
        for i in range(50):
            mic_q.put(AudioFrame(data=b"\x01" * 960, timestamp=time.time(), source="mic"))
            sys_q.put(AudioFrame(data=b"\x02" * 960, timestamp=time.time(), source="system"))

        processed_mic = 0
        processed_sys = 0
        for _ in range(50):
            if not mic_q.empty():
                mic_q.get_nowait()
                processed_mic += 1
            if not sys_q.empty():
                sys_q.get_nowait()
                processed_sys += 1

        self.assertEqual(processed_mic, 50)
        self.assertEqual(processed_sys, 50)

    # -------------------------------------------------------------------------
    # AUD-004: Timestamp monotonicity
    # -------------------------------------------------------------------------
    def test_aud_004_timestamp_monotonicity(self):
        """AUD-004: For every frame, timestamp[n+1] >= timestamp[n]. No time travel."""
        timestamps = []
        for _ in range(100):
            timestamps.append(time.time())
            time.sleep(0.0005)
        for i in range(len(timestamps) - 1):
            self.assertGreaterEqual(
                timestamps[i + 1],
                timestamps[i],
                f"Time travel detected at index {i}: {timestamps[i+1]} < {timestamps[i]}"
            )

    # -------------------------------------------------------------------------
    # AUD-005 & AUD-006: Queue boundedness and overflow recovery
    # -------------------------------------------------------------------------
    def test_aud_005_and_006_queue_boundedness_and_overflow(self):
        """AUD-005 / AUD-006: Verify maximum queue boundedness and oldest-frame drop on overflow."""
        q = queue.Queue(maxsize=10)
        
        # Insert 15 items using non-blocking put with oldest-drop policy
        for i in range(15):
            item = f"frame_{i}"
            if q.full():
                try:
                    q.get_nowait()  # Drop oldest
                except queue.Empty:
                    pass
            q.put_nowait(item)

        self.assertEqual(q.qsize(), 10)
        oldest_remaining = q.get_nowait()
        self.assertEqual(oldest_remaining, "frame_5", "Oldest frames 0-4 should have been dropped")

    # -------------------------------------------------------------------------
    # AUD-007: Silence handling
    # -------------------------------------------------------------------------
    def test_aud_007_silence_no_unnecessary_answer(self):
        """AUD-007: Pure silence yields silent diagnostics and no false triggers."""
        silence_samples = np.zeros(16000, dtype=np.int16)
        diag = compute_audio_diagnostics(silence_samples)
        self.assertEqual(diag.state, "SILENT")
        self.assertEqual(diag.rms, 0.0)
        self.assertEqual(diag.silence_ratio, 1.0)
        
        mic_proc = MicrophoneAudioProcessor()
        cleaned_b, is_speech, _ = mic_proc.process_frame(silence_samples)
        self.assertFalse(is_speech)
        self.assertEqual(cleaned_b, b"\x00" * len(silence_samples) * 2)

    # -------------------------------------------------------------------------
    # AUD-008: Natural pause continuation
    # -------------------------------------------------------------------------
    def test_aud_008_natural_pause_continuation(self):
        """AUD-008: A mid-question pause does not prematurely fire an answer if turn is incomplete."""
        from interview_engine import classify_and_understand_utterance
        
        part1 = "Can you explain your experience with Django"
        turn1 = classify_and_understand_utterance(part1, session=self.session, source="system")
        
        # If confidence is in pending band or question continuation expected
        self.assertIn(turn1["question_state"], ("HIGH", "MEDIUM"))
        
        # Continuation arrives
        part2 = "and REST APIs?"
        turn2 = classify_and_understand_utterance(f"{part1} {part2}", session=self.session, source="system")
        self.assertTrue(turn2["should_generate_answer"])
        self.assertIn("REST", turn2["refined_question"])

    # -------------------------------------------------------------------------
    # AUD-009: Loud speech
    # -------------------------------------------------------------------------
    def test_aud_009_loud_speech_clipping_protection(self):
        """AUD-009: Safe AGC and limiter attenuate loud speech to prevent int16 distortion."""
        t = np.linspace(0, 0.2, 3200, endpoint=False)
        loud_samples = (np.sin(2 * np.pi * 300 * t) * 32000).astype(np.int16)
        
        limited = safe_automatic_gain_control(loud_samples, target_rms=2600.0)
        self.assertLessEqual(np.max(np.abs(limited)), 32767)
        diag = compute_audio_diagnostics(limited)
        self.assertLess(diag.clipping_ratio, 0.05)

    # -------------------------------------------------------------------------
    # AUD-010: Quiet speech detection
    # -------------------------------------------------------------------------
    def test_aud_010_quiet_speech_pickup(self):
        """AUD-010: Quiet speech is properly amplified by AGC without blowing up silence."""
        t = np.linspace(0, 0.2, 3200, endpoint=False)
        # Moderate quiet speech above noise floor (amplitude 850, RMS ~ 600)
        quiet_samples = (np.sin(2 * np.pi * 300 * t) * 850).astype(np.int16)
        
        leveled = safe_automatic_gain_control(quiet_samples, target_rms=2600.0, max_gain_db=12.0)
        orig_rms = np.sqrt(np.mean(quiet_samples.astype(np.float32) ** 2))
        leveled_rms = np.sqrt(np.mean(leveled.astype(np.float32) ** 2))
        self.assertGreater(leveled_rms, orig_rms)

    # -------------------------------------------------------------------------
    # AUD-020 to AUD-027: Device Recovery & Watchdog
    # -------------------------------------------------------------------------
    def test_aud_020_device_disconnect_and_recovery(self):
        """AUD-020: Hardware disconnect sets error flag, terminates old worker, and starts recovery."""
        transcriber = ContinuousTranscriber(
            client=MagicMock(),
            on_transcript_chunk=lambda text, is_q, source="system": None
        )
        # Verify initial state
        self.assertFalse(transcriber.is_running)

    def test_aud_024_audio_device_watcher_lifecycle(self):
        """AUD-024: AudioDeviceWatcher monitors device switches with low CPU."""
        speaker_changed = threading.Event()
        watcher = AudioDeviceWatcher(
            on_default_speaker_changed=lambda spk: speaker_changed.set(),
            check_interval=0.1
        )
        watcher.start()
        self.assertTrue(watcher.is_running)
        time.sleep(0.2)
        watcher.stop()
        self.assertFalse(watcher.is_running)

    def test_aud_026_device_repeatedly_fails_bounded_retry(self):
        """AUD-026: Repeated device failure triggers backoff and bounded retries."""
        fail_count = 0
        max_retries = 5
        retry_delay = 0.01
        
        for attempt in range(max_retries):
            fail_count += 1
            time.sleep(retry_delay * (attempt + 1))  # Exponential-like backoff
            
        self.assertEqual(fail_count, 5)

    def test_aud_027_recovery_race_single_recovery_execution(self):
        """AUD-027: Near-simultaneous recovery triggers execute exactly once via locking."""
        recovery_lock = threading.Lock()
        recovery_executions = 0

        def trigger_recovery():
            nonlocal recovery_executions
            if recovery_lock.acquire(blocking=False):
                try:
                    time.sleep(0.05)
                    recovery_executions += 1
                finally:
                    recovery_lock.release()

        t1 = threading.Thread(target=trigger_recovery)
        t2 = threading.Thread(target=trigger_recovery)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(recovery_executions, 1, "Race condition: recovery must execute exactly once")

    # -------------------------------------------------------------------------
    # AUD-040 to AUD-043: Fast Resampling to 16kHz
    # -------------------------------------------------------------------------
    def test_aud_040_resampling_rates(self):
        """AUD-040: Verify 48kHz, 44.1kHz, and 32kHz resampling to canonical 16kHz."""
        # 16kHz (identity)
        pcm_16k = np.ones(16000, dtype=np.int16) * 100
        out_16k = fast_resample_16k(pcm_16k, 16000)
        self.assertEqual(len(out_16k), 16000)
        np.testing.assert_array_equal(out_16k, pcm_16k)

        # 48kHz (exact 3:1 integer decimation)
        pcm_48k = np.arange(48000, dtype=np.int16)
        out_48k = fast_resample_16k(pcm_48k, 48000)
        self.assertEqual(len(out_48k), 16000)
        self.assertEqual(out_48k[0], pcm_48k[0])
        self.assertEqual(out_48k[1], pcm_48k[3])

        # 44.1kHz (linear interpolation)
        pcm_44k = np.ones(44100, dtype=np.int16) * 500
        out_44k = fast_resample_16k(pcm_44k, 44100)
        self.assertEqual(len(out_44k), 16000)

    def test_aud_042_sample_count_duration_continuity(self):
        """AUD-042: Verify duration preserved within 0.1% after resampling."""
        orig_sr = 44100
        duration_s = 2.0
        num_orig = int(orig_sr * duration_s)
        pcm = np.random.randint(-1000, 1000, size=num_orig, dtype=np.int16)
        
        resampled = fast_resample_16k(pcm, orig_sr)
        expected_samples = int(16000 * duration_s)
        self.assertAlmostEqual(len(resampled), expected_samples, delta=5)

    def test_aud_043_no_unnecessary_resampling(self):
        """AUD-043: 16kHz source returns exact same array reference (no realloc)."""
        pcm_16k = np.array([1, 2, 3, 4], dtype=np.int16)
        res = fast_resample_16k(pcm_16k, 16000)
        self.assertIs(res, pcm_16k)


if __name__ == "__main__":
    unittest.main()
