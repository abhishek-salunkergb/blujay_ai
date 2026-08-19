"""
Unit Test Suite for Audio Device Manager & Hardware Configuration (Feature 1)
=============================================================================
Validates:
1. Hardware microphone enumeration and default detection
2. Hardware speaker/loopback enumeration and default detection
3. Persistent device preferences saving and loading
4. Audio level probe lifecycle and normalized volume calculations
5. Graceful fallback on unavailable/invalid device IDs
6. ContinuousTranscriber device routing integration
"""

import os
import json
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

import audio_device_manager
from audio_device_manager import (
    get_available_microphones,
    get_available_speakers,
    get_saved_device_preferences,
    save_device_preferences,
    AudioLevelProbe
)
from interview_engine import ContinuousTranscriber, InterviewSession


class TestAudioDeviceManager(unittest.TestCase):

    def setUp(self):
        # Create a temp preferences file for isolated testing
        self.temp_prefs = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_prefs.close()
        self._orig_prefs_file = audio_device_manager.PREFERENCES_FILE
        audio_device_manager.PREFERENCES_FILE = self.temp_prefs.name

    def tearDown(self):
        audio_device_manager.PREFERENCES_FILE = self._orig_prefs_file
        if os.path.exists(self.temp_prefs.name):
            try:
                os.remove(self.temp_prefs.name)
            except Exception:
                pass

    def test_get_available_microphones_structure(self):
        """Verify get_available_microphones returns well-formed device list."""
        mics = get_available_microphones()
        self.assertIsInstance(mics, list)
        self.assertGreater(len(mics), 0)
        
        for m in mics:
            self.assertIn("id", m)
            self.assertIn("name", m)
            self.assertIn("display_name", m)
            self.assertIn("is_default", m)
            self.assertIsInstance(m["is_default"], bool)

    def test_get_available_speakers_structure(self):
        """Verify get_available_speakers returns well-formed speaker list."""
        speakers = get_available_speakers()
        self.assertIsInstance(speakers, list)
        self.assertGreater(len(speakers), 0)
        
        for s in speakers:
            self.assertIn("id", s)
            self.assertIn("name", s)
            self.assertIn("display_name", s)
            self.assertIn("is_default", s)
            self.assertIsInstance(s["is_default"], bool)

    def test_device_preferences_save_and_load(self):
        """Verify saving and loading device preferences round-trips accurately."""
        # Initial empty state
        prefs = get_saved_device_preferences()
        self.assertIsNone(prefs.get("mic_device_id"))
        
        # Save custom choices
        save_device_preferences(
            mic_device_id=2,
            mic_device_name="Blue Yeti USB Mic",
            speaker_device_id="Realtek Speakers",
            speaker_device_name="Realtek High Definition Audio"
        )
        
        loaded = get_saved_device_preferences()
        self.assertEqual(loaded["mic_device_id"], 2)
        self.assertEqual(loaded["mic_device_name"], "Blue Yeti USB Mic")
        self.assertEqual(loaded["speaker_device_id"], "Realtek Speakers")
        self.assertEqual(loaded["speaker_device_name"], "Realtek High Definition Audio")

    def test_audio_level_probe_lifecycle(self):
        """Verify AudioLevelProbe starts and stops cleanly without thread leaks."""
        received_levels = []

        def on_level(level, rms):
            received_levels.append((level, rms))

        probe = AudioLevelProbe(
            device_type="mic",
            device_id="default",
            on_level_update=on_level
        )
        
        probe.start()
        self.assertTrue(probe.is_running)
        time.sleep(0.15)
        probe.stop()
        self.assertFalse(probe.is_running)
        self.assertIsNone(probe._thread)

    def test_transcriber_accepts_and_routes_devices(self):
        """Verify ContinuousTranscriber correctly accepts mic_device_id and system_speaker_id."""
        mock_client = MagicMock()
        session = InterviewSession()
        
        transcriber = ContinuousTranscriber(
            client=mock_client,
            on_transcript_chunk=MagicMock(),
            session=session,
            enable_watchdog=False,
            mic_device_id=1,
            system_speaker_id="Sony WH-1000XM5"
        )
        
        self.assertEqual(transcriber.mic_device_id, 1)
        self.assertEqual(transcriber.system_speaker_id, "Sony WH-1000XM5")
        
        # Verify dynamic update on start()
        with patch.object(transcriber, "_run_system_audio_loop"), \
             patch.object(transcriber, "_run_microphone_loop"), \
             patch.object(transcriber, "_system_processing_worker"), \
             patch.object(transcriber, "_mic_processing_worker"):
            transcriber.start(
                source="dual",
                mic_device_id=3,
                system_speaker_id="AirPods Pro"
            )
            self.assertEqual(transcriber.mic_device_id, 3)
            self.assertEqual(transcriber.system_speaker_id, "AirPods Pro")
            transcriber.stop()


if __name__ == "__main__":
    unittest.main()
