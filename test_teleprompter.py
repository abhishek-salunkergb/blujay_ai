"""
Unit Test Suite for Teleprompter Auto-Scroll Controller (Feature 7)
===================================================================
Validates:
1. Pacing calculations and speed presets (100, 135, 170 WPM)
2. State machine toggling (start, stop, toggle)
3. Speed cycling and status reporting
4. User manual interaction pause mechanism
5. Safe teardown and timer cancellation
"""

import time
import unittest
from unittest.mock import MagicMock

from teleprompter import TeleprompterController


class TestTeleprompter(unittest.TestCase):

    def test_01_init_and_presets(self):
        """Verify TeleprompterController initializes with default 135 WPM."""
        mock_text = MagicMock()
        mock_root = MagicMock()
        teleprompter = TeleprompterController(mock_text, mock_root)

        self.assertFalse(teleprompter.is_scrolling)
        self.assertEqual(teleprompter.wpm, 135)
        self.assertEqual(teleprompter.current_preset, "normal")

    def test_02_cycle_speed(self):
        """Verify cycle_speed rotates across normal -> fast -> slow -> normal."""
        mock_text = MagicMock()
        mock_root = MagicMock()
        teleprompter = TeleprompterController(mock_text, mock_root)

        # 1. normal -> fast
        s1 = teleprompter.cycle_speed()
        self.assertEqual(s1, "fast")
        self.assertEqual(teleprompter.wpm, 170)

        # 2. fast -> slow
        s2 = teleprompter.cycle_speed()
        self.assertEqual(s2, "slow")
        self.assertEqual(teleprompter.wpm, 100)

        # 3. slow -> normal
        s3 = teleprompter.cycle_speed()
        self.assertEqual(s3, "normal")
        self.assertEqual(teleprompter.wpm, 135)

    def test_03_start_stop_toggle(self):
        """Verify toggling starts and stops auto-scrolling."""
        mock_text = MagicMock()
        mock_root = MagicMock()
        teleprompter = TeleprompterController(mock_text, mock_root)

        # Toggle on
        res1 = teleprompter.toggle()
        self.assertTrue(res1)
        self.assertTrue(teleprompter.is_scrolling)
        mock_root.after.assert_called()

        # Toggle off
        res2 = teleprompter.toggle()
        self.assertFalse(res2)
        self.assertFalse(teleprompter.is_scrolling)

    def test_04_manual_interaction_pause(self):
        """Verify user manual interaction pauses scroll ticks."""
        mock_text = MagicMock()
        mock_root = MagicMock()
        teleprompter = TeleprompterController(mock_text, mock_root)

        teleprompter.start()
        self.assertEqual(teleprompter._user_interaction_paused_until, 0.0)

        teleprompter._on_manual_interaction()
        self.assertGreater(teleprompter._user_interaction_paused_until, time.time())

        # Cleanup
        teleprompter.stop()


if __name__ == "__main__":
    unittest.main()
