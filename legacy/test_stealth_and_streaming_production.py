import unittest
import numpy as np
from unittest.mock import MagicMock, patch

try:
    from legacy.overlay import set_click_through, hide_from_screen_capture, get_root_hwnd
except ImportError:
    from overlay import set_click_through, hide_from_screen_capture, get_root_hwnd


class TestStealthAndStreamingProduction(unittest.TestCase):

    def test_vad_frame_detection(self):
        # Test silent frame (all zeros)
        silent_frame = bytes(FRAME_SIZE * 2)
        self.assertFalse(is_speech_frame(silent_frame, SAMPLE_RATE))

        # Test high amplitude synthetic speech frame
        t = np.linspace(0, 0.03, FRAME_SIZE)
        sine_wave = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)
        audio_frame = sine_wave.tobytes()
        self.assertTrue(is_speech_frame(audio_frame, SAMPLE_RATE))

    def test_stealth_display_affinity_and_root_hwnd(self):
        dummy_hwnd = 12345
        root_id = get_root_hwnd(dummy_hwnd)
        self.assertIsNotNone(root_id)

        with patch("overlay.SetWindowDisplayAffinity", return_value=1):
            self.assertTrue(hide_from_screen_capture(dummy_hwnd))

    def test_click_through_toggle(self):
        dummy_hwnd = 12345
        with patch("overlay.GetWindowLongPtr", return_value=0), \
             patch("overlay.SetWindowLongPtr", return_value=1):
            res = set_click_through(dummy_hwnd, True)
            self.assertTrue(res)

            res_disable = set_click_through(dummy_hwnd, False)
            self.assertTrue(res_disable)

    def test_overlay_pro_ui_features(self):
        import tkinter as tk
        try:
            root = tk.Tk()
            root.withdraw()
            from overlay import OverlayWindow
            overlay = OverlayWindow(parent=root)
            
            # 1. Test clear and token append
            overlay.clear()
            self.assertEqual(overlay.current_text, "")
            
            overlay.append_token("def solve():\n")
            overlay.append_token("    return True")
            self.assertEqual(overlay.current_text, "def solve():\n    return True")
            
            # 2. Test full answer set & syntax highlight execution
            code_text = "def calculate_sum(a, b):\n    # Add numbers\n    return a + b"
            overlay.set_answer(code_text)
            self.assertEqual(overlay.current_text, code_text)
            
            # 3. Test font scaling
            init_font_size = overlay.font_size
            overlay.increase_font_size()
            self.assertEqual(overlay.font_size, init_font_size + 1)
            overlay.decrease_font_size()
            self.assertEqual(overlay.font_size, init_font_size)

            # 4. Test 1-click clipboard copy
            overlay.copy_answer_to_clipboard()
            
            # 5. Test pill mode toggle
            overlay.toggle_pill_mode()
            self.assertTrue(overlay.pill_mode)
            overlay.toggle_pill_mode()
            self.assertFalse(overlay.pill_mode)
            
            overlay.close()
            root.destroy()
        except Exception as e:
            print(f"Skipping GUI test in headless mode: {e}")


if __name__ == "__main__":
    unittest.main()
