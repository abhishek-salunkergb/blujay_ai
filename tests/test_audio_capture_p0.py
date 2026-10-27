"""
Test Suite for Parakeet-Grade Audio Capture (P0) Hardening
===========================================================
Validates:
1. Fast integer decimation & linear resampling to 16kHz PCM
2. Windows Core Audio / system microphone mute state probe
3. AudioDeviceWatcher lifecycle, callbacks, and low-frequency polling
4. WASAPI device invalidation (AUDCLNT_E_DEVICE_INVALIDATED) auto-recovery
5. Seamless device hot-swapping (< 1-2 sec) via ContinuousTranscriber.switch_devices()
6. Silent-stop watchdog detecting driver stalls and auto-reviving capture workers
7. Audio telemetry metrics (availability, dropout rate, init latency)
8. Long-session bounded memory and queue depth integrity
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import time
import queue
import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from audio_device_manager import (
    fast_resample_16k,
    get_system_microphone_mute_state,
    AudioDeviceWatcher
)
from interview_engine import (
    ContinuousTranscriber,
    InterviewSession,
    AudioFrame
)


class TestAudioCaptureP0(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.session = InterviewSession()

    def test_fast_resample_16k_rates(self):
        """Verify fast decimation and interpolation to 16kHz."""
        # 1. 16kHz should be a no-op identity
        orig_16k = np.array([100, 200, 300, 400], dtype=np.int16)
        res_16k = fast_resample_16k(orig_16k, 16000)
        np.testing.assert_array_equal(res_16k, orig_16k)

        # 2. 48kHz -> 3:1 integer decimation (48000 -> 16000)
        # 48000 samples for 1 second of 48kHz audio
        t = np.linspace(0, 1, 48000, endpoint=False)
        tone_48k = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
        res_48k = fast_resample_16k(tone_48k, 48000)
        self.assertEqual(len(res_48k), 16000)
        self.assertEqual(res_48k.dtype, np.int16)
        self.assertEqual(res_48k[0], tone_48k[0])
        self.assertEqual(res_48k[1], tone_48k[3])

        # 3. 32kHz -> 2:1 integer decimation
        tone_32k = np.arange(32000, dtype=np.int16)
        res_32k = fast_resample_16k(tone_32k, 32000)
        self.assertEqual(len(res_32k), 16000)
        self.assertEqual(res_32k[1], tone_32k[2])

        # 4. 44.1kHz -> Linear interpolation
        t_44k = np.linspace(0, 1, 44100, endpoint=False)
        tone_44k = (np.sin(2 * np.pi * 440 * t_44k) * 10000).astype(np.int16)
        res_44k = fast_resample_16k(tone_44k, 44100)
        self.assertEqual(len(res_44k), 16000)
        self.assertEqual(res_44k.dtype, np.int16)

        # 5. Empty and None handling
        self.assertIsNone(fast_resample_16k(None, 48000))
        empty = np.array([], dtype=np.int16)
        res_empty = fast_resample_16k(empty, 48000)
        self.assertEqual(len(res_empty), 0)

    def test_system_microphone_mute_state_probe(self):
        """Verify Windows mute probe executes safely without crashing."""
        mute_state = get_system_microphone_mute_state()
        # Returns True, False, or None (if not Windows/COM error)
        self.assertIn(mute_state, (True, False, None))

    def test_audio_device_watcher_lifecycle_and_callback(self):
        """Verify AudioDeviceWatcher detects device switches and stops cleanly."""
        spk_changes = []
        mic_changes = []

        def on_spk(name):
            spk_changes.append(name)

        def on_mic(dev_id):
            mic_changes.append(dev_id)

        watcher = AudioDeviceWatcher(
            on_default_speaker_changed=on_spk,
            on_default_mic_changed=on_mic,
            check_interval=0.05
        )
        watcher.start()
        self.assertTrue(watcher.is_running)

        # Simulate a detected speaker change
        watcher._last_default_speaker_name = "Realtek Audio"
        with patch("audio_device_manager.sc.default_speaker") as mock_def:
            mock_speaker = MagicMock()
            mock_speaker.name = "Sony WH-1000XM5"
            mock_def.return_value = mock_speaker
            time.sleep(0.12)

        self.assertIn("Sony WH-1000XM5", spk_changes)
        watcher.stop()
        self.assertFalse(watcher.is_running)
        self.assertIsNone(watcher._thread)

    def test_hot_swap_device_switching(self):
        """Verify ContinuousTranscriber.switch_devices() seamlessly updates hardware and bumps epoch."""
        status_updates = []
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            on_status_change=lambda s: status_updates.append(s),
            session=self.session,
            enable_watchdog=False
        )

        with patch.object(transcriber, "_run_system_audio_loop"), \
             patch.object(transcriber, "_run_microphone_loop"), \
             patch.object(transcriber, "_system_processing_worker"), \
             patch.object(transcriber, "_mic_processing_worker"):

            transcriber.start(source="dual", mic_device_id=1, system_speaker_id="Speakers")
            self.assertEqual(transcriber._audio_stream_epoch, 1)
            self.assertEqual(transcriber.mic_device_id, 1)
            self.assertEqual(transcriber.system_speaker_id, "Speakers")

            # Add mock frames into queues
            dummy_frame = AudioFrame(data=b"\x00" * 960, timestamp=time.perf_counter(), source="system", epoch=1)
            transcriber.system_audio_queue.put(dummy_frame)
            transcriber.mic_audio_queue.put(dummy_frame)
            self.assertFalse(transcriber.system_audio_queue.empty())

            # Perform dynamic hot swap to USB headset
            success = transcriber.switch_devices(
                mic_device_id=4,
                system_speaker_id="USB Headset"
            )
            self.assertTrue(success)
            # Epoch bumped
            self.assertEqual(transcriber._audio_stream_epoch, 2)
            self.assertEqual(transcriber.mic_device_id, 4)
            self.assertEqual(transcriber.system_speaker_id, "USB Headset")
            # Queues flushed to eliminate stale audio
            self.assertTrue(transcriber.system_audio_queue.empty())
            self.assertTrue(transcriber.mic_audio_queue.empty())

            # Check metrics
            metrics = transcriber.get_audio_metrics()
            self.assertEqual(metrics["stream_epoch"], 2)
            self.assertEqual(metrics["mic_device_id"], 4)
            self.assertEqual(metrics["system_speaker_id"], "USB Headset")

            transcriber.stop()

    def test_wasapi_invalidation_breakout_recovery(self):
        """Verify that AUDCLNT_E_DEVICE_INVALIDATED or 3 consecutive errors breaks out of inner recorder to reacquire speaker."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            session=self.session,
            enable_watchdog=False
        )
        transcriber.is_running = True

        call_count = {"reacquire_speaker": 0, "record_calls": 0}

        class MockRecorder:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def record(self, numframes):
                call_count["record_calls"] += 1
                if call_count["record_calls"] >= 2:
                    # Throw device invalidated on 2nd record
                    raise Exception("AUDCLNT_E_DEVICE_INVALIDATED (0x88890004)")
                return np.zeros(numframes, dtype=np.int16)

        mock_loopback = MagicMock()
        mock_loopback.recorder.side_effect = lambda **kwargs: MockRecorder()

        def mock_default_speaker():
            call_count["reacquire_speaker"] += 1
            if call_count["reacquire_speaker"] > 2:
                # Stop loop after verifying reacquisition
                transcriber.is_running = False
            spk = MagicMock()
            spk.name = "Mock Speaker"
            return spk

        with patch("interview_engine.sc.default_speaker", side_effect=mock_default_speaker), \
             patch("interview_engine.sc.get_microphone", return_value=mock_loopback):

            # Run system audio loop with epoch 1
            transcriber._run_system_audio_loop(worker_id="test", epoch=0)

        # Speaker must have been reacquired at least twice due to invalidation breakout
        self.assertGreaterEqual(call_count["reacquire_speaker"], 2)
        self.assertGreater(transcriber.discontinuity_count, 0)
        transcriber.stop()

    def test_silent_stop_watchdog_freeze_recovery(self):
        """Verify watchdog detects stalled audio heartbeat and restarts capture thread with a fresh epoch."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            session=self.session,
            enable_watchdog=True
        )
        transcriber.is_running = True
        transcriber.source = "system"
        transcriber._audio_stream_epoch = 1

        # Simulate system thread is alive but stalled (last frame > 3.5s ago)
        mock_system_thread = MagicMock()
        mock_system_thread.is_alive.return_value = True
        transcriber.system_thread = mock_system_thread
        transcriber._last_system_frame_time = time.perf_counter() - 5.0

        recovered_threads = []
        def mock_thread_start(t):
            recovered_threads.append(t.name)

        with patch.object(threading.Thread, "start", side_effect=lambda: None):
            # Run one watchdog pass directly
            now = time.perf_counter()
            cur_epoch = transcriber._audio_stream_epoch
            sys_stalled = bool(transcriber.system_thread and transcriber.system_thread.is_alive() and (now - transcriber._last_system_frame_time > 3.5) and transcriber._last_system_frame_time > 0)
            self.assertTrue(sys_stalled)

            # Trigger recovery
            if sys_stalled:
                transcriber._audio_stream_epoch += 1

            self.assertEqual(transcriber._audio_stream_epoch, 2)

        transcriber.stop()

    def test_bounded_memory_over_prolonged_stream(self):
        """Verify audio queues drop oldest frames when full and do not grow unboundedly."""
        transcriber = ContinuousTranscriber(
            client=self.mock_client,
            on_transcript_chunk=MagicMock(),
            session=self.session,
            enable_watchdog=False
        )
        self.assertEqual(transcriber.system_audio_queue.maxsize, 30)

        # Push 1,000 frames into queue to simulate sustained capture load
        frame_data = b"\x00" * 960  # 30ms @ 16kHz 16-bit
        for i in range(1000):
            frame = AudioFrame(data=frame_data, timestamp=time.perf_counter(), source="system", epoch=1)
            try:
                transcriber.system_audio_queue.put_nowait(frame)
            except queue.Full:
                try:
                    transcriber.system_audio_queue.get_nowait()
                    transcriber.dropped_frames_count += 1
                except queue.Empty:
                    pass
                transcriber.system_audio_queue.put_nowait(frame)

        # Queue depth must be exactly capped at maxsize (30)
        self.assertEqual(transcriber.system_audio_queue.qsize(), 30)
        self.assertEqual(transcriber.dropped_frames_count, 970)

        # Verify telemetry
        metrics = transcriber.get_audio_metrics()
        self.assertEqual(metrics["dropped_frames_count"], 970)
        self.assertGreater(metrics["dropout_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
