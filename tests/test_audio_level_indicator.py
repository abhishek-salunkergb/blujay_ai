"""
Unit Test Suite for Live Audio VU Level Indicators (Feature 4)
==============================================================
Validates:
1. ContinuousTranscriber initialization with on_audio_level callback
2. System loopback audio level extraction and callback emission
3. Microphone audio level extraction and callback emission
4. Peak normalization across int16 and float32 buffers (0.0 to 1.0)
5. Robustness against exceptions in consumer level callback
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import time
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

from interview_engine import ContinuousTranscriber, AudioFrame


class MockSTTProvider:
    def __init__(self, name="mock"):
        self.name = name
    def transcribe(self, audio_data, prompt=None):
        return {"text": "mock transcript"}


class TestAudioLevelIndicator(unittest.TestCase):

    def test_01_init_transcriber_with_level_callback(self):
        """Verify ContinuousTranscriber stores on_audio_level callback cleanly."""
        levels_recorded = []
        transcriber = ContinuousTranscriber(
            client=MagicMock(),
            on_transcript_chunk=lambda text, is_q, source="system": None,
            on_audio_level=lambda src, lvl, rms: levels_recorded.append((src, lvl)),
            provider=MockSTTProvider()
        )
        self.assertIsNotNone(transcriber.on_audio_level)

    def test_02_system_audio_level_calculation(self):
        """Verify _run_system_audio_loop computes normalized level and emits on_audio_level."""
        levels_recorded = []
        transcriber = ContinuousTranscriber(
            client=MagicMock(),
            on_transcript_chunk=lambda text, is_q, source="system": None,
            on_audio_level=lambda src, lvl, rms: levels_recorded.append((src, lvl)),
            provider=MockSTTProvider()
        )

        # Generate 30ms of synthetic 16kHz sine wave audio (half peak = 0.5)
        t = np.linspace(0, 0.03, 480, False)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        mock_recorder = MagicMock()
        mock_recorder.record.return_value = sine_wave

        with patch("interview_engine.SOUNDCARD_AVAILABLE", True), \
             patch("interview_engine.sc.default_speaker", return_value=MagicMock()), \
             patch("interview_engine.sc.get_microphone") as mock_get_mic:
            
            mock_mic_ctx = MagicMock()
            mock_mic_ctx.recorder.return_value.__enter__.return_value = mock_recorder
            mock_get_mic.return_value = mock_mic_ctx

            transcriber.is_running = True
            
            # Run one iteration of system audio capture
            def _stop_after_frame(src, lvl, rms):
                levels_recorded.append((src, lvl))
                transcriber.is_running = False

            transcriber.on_audio_level = _stop_after_frame
            transcriber._run_system_audio_loop()

            self.assertGreaterEqual(len(levels_recorded), 1)
            src, lvl = levels_recorded[0]
            self.assertEqual(src, "system")
            self.assertAlmostEqual(lvl, 0.5, places=1)

    def test_03_microphone_audio_level_calculation(self):
        """Verify _run_microphone_loop computes normalized int16 level and emits on_audio_level."""
        levels_recorded = []
        transcriber = ContinuousTranscriber(
            client=MagicMock(),
            on_transcript_chunk=lambda text, is_q, source="system": None,
            on_audio_level=lambda src, lvl, rms: levels_recorded.append((src, lvl)),
            provider=MockSTTProvider()
        )

        # Generate int16 peak at ~16384 (approx 0.5 normalized)
        int16_samples = np.full(480, 16384, dtype=np.int16)
        
        mock_stream = MagicMock()
        mock_stream.read.return_value = (int16_samples, False)

        with patch("interview_engine.sd.RawInputStream") as mock_input_stream:
            mock_input_stream.return_value.__enter__.return_value = mock_stream

            transcriber.is_running = True

            def _stop_after_mic_frame(src, lvl, rms):
                levels_recorded.append((src, lvl))
                transcriber.is_running = False

            transcriber.on_audio_level = _stop_after_mic_frame
            transcriber._run_microphone_loop(channel_source="candidate_mic")

            self.assertGreaterEqual(len(levels_recorded), 1)
            src, lvl = levels_recorded[0]
            self.assertEqual(src, "mic")
            self.assertAlmostEqual(lvl, 0.5, places=1)

    def test_04_level_callback_exception_isolation(self):
        """Verify that an exception in on_audio_level does not crash the audio capture thread."""
        def _failing_callback(src, lvl, rms):
            raise RuntimeError("Test UI level meter failure")

        transcriber = ContinuousTranscriber(
            client=MagicMock(),
            on_transcript_chunk=lambda text, is_q, source="system": None,
            on_audio_level=_failing_callback,
            provider=MockSTTProvider()
        )

        int16_samples = np.full(480, 1000, dtype=np.int16)
        mock_stream = MagicMock()
        mock_stream.read.return_value = (int16_samples, False)

        with patch("interview_engine.sd.RawInputStream") as mock_input_stream:
            mock_input_stream.return_value.__enter__.return_value = mock_stream
            transcriber.is_running = True
            
            # Ensure it completes at least 1 loop cycle without raising exception
            call_count = [0]
            def _counted_read(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] >= 3:
                    transcriber.is_running = False
                return (int16_samples, False)
            mock_stream.read.side_effect = _counted_read

            # Should not raise
            transcriber._run_microphone_loop(channel_source="candidate_mic")
            self.assertGreaterEqual(call_count[0], 3)


if __name__ == "__main__":
    unittest.main()
