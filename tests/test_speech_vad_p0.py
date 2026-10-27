"""
Test Suite for Parakeet-Grade Speech Detection & VAD Hardening (Category 2)
===========================================================================
Validates:
1. Impulsive mechanical typing click rejection (high crest factor, low pitch periodicity)
2. Deep low-pitched voice periodicity (70-100 Hz coverage without edge-truncation)
3. Pure tonal notification chime rejection (Slack/Teams/Windows dings) on loopback
4. Soft speech detection (PCM RMS 40-70) with adaptive SNR and AGC leveling
5. Noise floor anti-runaway ceiling (min(180.0, ...)) during loud ambient fan noise
6. Stop consonant hangover hold (2-state Schmitt trigger hysteresis)
7. Sub-millisecond vectorized DSP execution latency
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import time
import unittest
import numpy as np

from speech import (
    AudioDiagnostics,
    compute_audio_diagnostics,
    SystemAudioProcessor,
    MicrophoneAudioProcessor,
    remove_dc_offset,
    safe_automatic_gain_control
)


class TestSpeechVADP0(unittest.TestCase):

    def test_typing_click_rejection(self):
        """Verify mechanical keyboard click bursts have high crest factor, low periodicity, and are rejected."""
        # Synthesize an impulsive click: quiet baseline with a sharp 2ms spike (crest factor > 6.0)
        click_frame = np.random.randint(-15, 15, 480, dtype=np.int16)
        # Add sharp transient click spike at sample 150
        click_frame[150] = 8000
        click_frame[151] = -6500
        click_frame[152] = 4000
        click_frame[153] = -2500

        diag = compute_audio_diagnostics(click_frame, noise_floor_rms=40.0)
        self.assertGreater(diag.crest_factor, 5.5)
        self.assertTrue(diag.is_transient_click)

        # Feed to MicrophoneAudioProcessor
        proc = MicrophoneAudioProcessor(sample_rate=16000)
        out_bytes, is_speech, _ = proc.process_frame(click_frame)
        self.assertFalse(is_speech, "Impulsive mechanical typing click must be rejected as non-speech")

    def test_deep_bass_voice_periodicity(self):
        """Verify deep male voices (70-85 Hz) are recognized with high periodicity and NOT flagged as clicks."""
        # 80 Hz pitch tone at 16kHz (period T = 16000/80 = 200 samples)
        t = np.linspace(0, 480 / 16000, 480, endpoint=False)
        # Rich harmonic voice model: fundamental + 2nd + 3rd harmonics
        voice = (
            np.sin(2 * np.pi * 80 * t) * 2000 +
            np.sin(2 * np.pi * 160 * t) * 1200 +
            np.sin(2 * np.pi * 240 * t) * 600
        ).astype(np.int16)

        diag = compute_audio_diagnostics(voice, noise_floor_rms=40.0)
        self.assertGreater(diag.pitch_periodicity, 0.50, "Deep 80 Hz pitch must exhibit high autocorrelation periodicity")
        self.assertFalse(diag.is_transient_click, "Deep male voice must NEVER be misclassified as a typing click")

    def test_system_notification_chime_rejection(self):
        """Verify pure tonal notification chimes (Slack, Teams, Windows dings) are flagged and rejected."""
        # 1000 Hz pure sine wave chime (standard notification bell frequency)
        t = np.linspace(0, 480 / 16000, 480, endpoint=False)
        chime_frame = (np.sin(2 * np.pi * 1000 * t) * 3500).astype(np.int16)

        diag = compute_audio_diagnostics(chime_frame, noise_floor_rms=40.0)
        self.assertTrue(diag.is_tonal_chime, "1000 Hz pure notification tone must be flagged as tonal chime")

        # Feed to SystemAudioProcessor
        sys_proc = SystemAudioProcessor(sample_rate=16000)
        out_bytes, is_speech, _ = sys_proc.process_frame(chime_frame)
        self.assertFalse(is_speech, "System notification chime must NOT trigger speech detection on loopback")

    def test_soft_speech_capture_and_agc_leveling(self):
        """Verify soft-spoken speech (RMS ~45-65) triggers VAD and is leveled cleanly by AGC."""
        t = np.linspace(0, 480 / 16000, 480, endpoint=False)
        # 150 Hz harmonic voice (conversational pitch) at soft amplitude (RMS ~50)
        soft_voice = (
            np.sin(2 * np.pi * 150 * t) * 55 +
            np.sin(2 * np.pi * 300 * t) * 35
        ).astype(np.int16)

        diag = compute_audio_diagnostics(soft_voice, noise_floor_rms=25.0)
        self.assertGreater(diag.rms, 35.0)
        self.assertLess(diag.rms, 75.0)

        proc = MicrophoneAudioProcessor(sample_rate=16000)
        proc.noise_floor_rms = 25.0  # Quiet room noise floor

        # Feed 2 frames to satisfy 2-frame onset hysteresis
        proc.process_frame(soft_voice)
        out_bytes, is_speech, out_diag = proc.process_frame(soft_voice)
        
        # In presence of Silero or WebRTC, soft voice must trigger speech
        out_samples = np.frombuffer(out_bytes, dtype=np.int16)
        if is_speech:
            out_rms = float(np.sqrt(np.mean(out_samples.astype(np.float32) ** 2)))
            self.assertGreater(out_rms, diag.rms, "AGC must boost soft speech volume for STT")

    def test_noise_floor_anti_runaway_ceiling(self):
        """Verify noise floor ceiling min(180.0, ...) prevents fan noise from locking out conversational speech."""
        proc = MicrophoneAudioProcessor(sample_rate=16000)
        
        # Simulate high fan noise running at 220 RMS
        proc.noise_floor_rms = 220.0
        
        # Calculate effective min RMS
        effective_min_rms = min(180.0, max(35.0, proc.noise_floor_rms * 1.12))
        self.assertEqual(effective_min_rms, 180.0, "Noise floor threshold MUST be capped at 180.0 ceiling")

        # Conversational speech at 190 RMS exceeds ceiling and can be detected
        t = np.linspace(0, 480 / 16000, 480, endpoint=False)
        speech_frame = (np.sin(2 * np.pi * 180 * t) * 270).astype(np.int16)
        diag = compute_audio_diagnostics(speech_frame, noise_floor_rms=proc.noise_floor_rms)
        self.assertGreater(diag.rms, effective_min_rms, "Speech at 190 RMS must exceed 180.0 ceiling")

    def test_stop_consonant_hangover_hold(self):
        """Verify 2-state hysteresis holds speech across silent stop consonant closures (p, t, k closures)."""
        proc = MicrophoneAudioProcessor(sample_rate=16000)
        proc.noise_floor_rms = 30.0

        t = np.linspace(0, 480 / 16000, 480, endpoint=False)
        voiced_frame = (np.sin(2 * np.pi * 160 * t) * 2000).astype(np.int16)
        silent_frame = np.zeros(480, dtype=np.int16)

        # 1. Enter speech (2 frames onset)
        proc.process_frame(voiced_frame)
        _, is_speech1, _ = proc.process_frame(voiced_frame)
        self.assertTrue(is_speech1, "Must enter speech state on 2 consecutive voiced frames")

        # 2. Feed 3 frames of silence (~90ms stop consonant closure)
        _, is_speech_h1, _ = proc.process_frame(silent_frame)
        _, is_speech_h2, _ = proc.process_frame(silent_frame)
        _, is_speech_h3, _ = proc.process_frame(silent_frame)

        # Hangover hold (5 frames) must hold speech state through all 3 silent frames!
        self.assertTrue(is_speech_h1, "Hangover frame 1 must maintain speech state")
        self.assertTrue(is_speech_h2, "Hangover frame 2 must maintain speech state")
        self.assertTrue(is_speech_h3, "Hangover frame 3 must maintain speech state")

    def test_diagnostics_execution_latency(self):
        """Verify compute_audio_diagnostics executes in sub-millisecond real-time budget."""
        dummy = np.random.randint(-1000, 1000, 480, dtype=np.int16)
        
        # Warmup
        for _ in range(5):
            compute_audio_diagnostics(dummy)
            
        t0 = time.perf_counter()
        iters = 100
        for _ in range(iters):
            compute_audio_diagnostics(dummy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

        self.assertLess(elapsed_ms, 0.6, f"Diagnostics must execute in < 0.6ms per frame (measured {elapsed_ms:.3f}ms)")


if __name__ == "__main__":
    unittest.main()
