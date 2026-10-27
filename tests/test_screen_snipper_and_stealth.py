"""
Unit Test Suite for Screen Snipper and Stealth Capture Exclusion
===============================================================
Validates:
1. Multi-monitor virtual desktop geometry calculation.
2. Windows SetWindowDisplayAffinity (0x11) on InteractiveRegionSnipper window.
3. Coordinate calculation, bounding box normalization, and cancel/escape.
4. Integration with vision.capture_screen_image.
5. Backward compatibility of screen_analyzer.capture_screen_image.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import screen_snipper
from screen_snipper import InteractiveRegionSnipper
import screen_analyzer


class TestScreenSnipperAndStealth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def test_01_snipper_initialization_and_stealth_affinity(self):
        """Verify InteractiveRegionSnipper sets topmost, cross cursor, and applies WDA_EXCLUDEFROMCAPTURE on Windows."""
        completed_mock = MagicMock()
        cancel_mock = MagicMock()

        with patch("ctypes.windll.user32.SetWindowDisplayAffinity", return_value=1) as mock_affinity:
            snipper = InteractiveRegionSnipper(self.root, on_snip_completed=completed_mock, on_cancel=cancel_mock)
            self.assertIsNotNone(snipper.win)
            self.assertIsNotNone(snipper.canvas)
            if os.name == "nt":
                mock_affinity.assert_called()
                # Verify affinity value requested is 0x11 (WDA_EXCLUDEFROMCAPTURE)
                call_args = mock_affinity.call_args[0]
                self.assertEqual(call_args[1].value, 0x11)
            snipper._close()

    def test_02_coordinate_normalization(self):
        """Verify drag bounding box correctly handles dragging right-to-left and bottom-to-top."""
        completed_mock = MagicMock()
        snipper = InteractiveRegionSnipper(self.root, on_snip_completed=completed_mock)

        # Simulate drag from (500, 400) to (100, 200)
        snipper.start_x = 500
        snipper.start_y = 400

        with patch.object(snipper.win, "winfo_pointerx", return_value=100), \
             patch.object(snipper.win, "winfo_pointery", return_value=200), \
             patch.object(snipper, "_capture_and_dispatch") as mock_dispatch:
            event = MagicMock()
            snipper._on_release(event)
            import time
            time.sleep(0.03)
            self.root.update()
            mock_dispatch.assert_called_once_with((100, 200, 500, 400))

    def test_03_escape_key_triggers_cancel(self):
        """Verify pressing Escape closes window and invokes on_cancel callback."""
        completed_mock = MagicMock()
        cancel_mock = MagicMock()
        snipper = InteractiveRegionSnipper(self.root, on_snip_completed=completed_mock, on_cancel=cancel_mock)

        snipper._on_escape()
        cancel_mock.assert_called_once()
        completed_mock.assert_not_called()

    def test_04_capture_and_dispatch_calls_vision_capture(self):
        """Verify _capture_and_dispatch uses vision.capture_screen_image and passes result to callback."""
        completed_mock = MagicMock()
        snipper = InteractiveRegionSnipper(self.root, on_snip_completed=completed_mock)

        test_img_path = os.path.join(os.getcwd(), "test_snip.png")
        with patch("vision.capture_screen_image", return_value=test_img_path) as mock_vision_cap:
            snipper._capture_and_dispatch((50, 50, 250, 250))
            mock_vision_cap.assert_called_once_with(bbox=(50, 50, 250, 250))
            completed_mock.assert_called_once_with(test_img_path)

    def test_05_screen_analyzer_capture_screen_image_supports_parameters(self):
        """Verify screen_analyzer.capture_screen_image accepts monitor_id and bbox and falls back gracefully."""
        mock_img = Image.new("RGB", (300, 200), color=(10, 10, 10))
        with patch("vision.capture_screen_image", side_effect=Exception("Vision unavailable")), \
             patch("PIL.ImageGrab.grab", return_value=mock_img):
            path = screen_analyzer.capture_screen_image(monitor_id=1, bbox=(0, 0, 300, 200))
            self.assertTrue(os.path.exists(path))
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
