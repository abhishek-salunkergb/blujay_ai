"""
Unit Test Suite for Multi-Monitor Screen Selector & Vision Solver (Feature 3)
=============================================================================
Validates:
1. Multi-monitor discovery and enumeration
2. Monitor preference loading and persistence
3. Targeted display and bounding box screenshot captures
4. Full backward-compatibility for legacy callers
5. Multimodal vision solver parameter routing (monitor_id, bbox, image_path)
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

import vision
from vision import (
    get_available_monitors,
    get_saved_monitor_preference,
    save_monitor_preference,
    capture_screen_image,
    encode_image_base64,
    solve_screen_coding_problem
)


class TestScreenVisionMultiMonitor(unittest.TestCase):

    def setUp(self):
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_config.close()
        self._orig_cfg = vision.VISION_CONFIG_FILE
        vision.VISION_CONFIG_FILE = self.temp_config.name

    def tearDown(self):
        vision.VISION_CONFIG_FILE = self._orig_cfg
        if os.path.exists(self.temp_config.name):
            try:
                os.remove(self.temp_config.name)
            except Exception:
                pass

    def test_01_get_available_monitors_structure(self):
        """Verify monitor discovery returns valid monitor dicts with required fields."""
        monitors = get_available_monitors()
        self.assertIsInstance(monitors, list)
        self.assertGreaterEqual(len(monitors), 1)

        for mon in monitors:
            self.assertIn("id", mon)
            self.assertIn("name", mon)
            self.assertIn("bbox", mon)
            self.assertIn("width", mon)
            self.assertIn("height", mon)
            self.assertIn("is_primary", mon)
            self.assertIsInstance(mon["bbox"], tuple)
            self.assertEqual(len(mon["bbox"]), 4)

    def test_02_monitor_preference_persistence(self):
        """Verify saving and loading preferred monitor ID round-trips accurately."""
        self.assertEqual(get_saved_monitor_preference(), 0)

        save_monitor_preference(1)
        self.assertEqual(get_saved_monitor_preference(), 1)

        save_monitor_preference(2)
        self.assertEqual(get_saved_monitor_preference(), 2)

    def test_03_capture_screen_image_default_backward_compat(self):
        """Verify capture_screen_image without arguments returns a valid existing PNG file."""
        mock_img = Image.new("RGB", (400, 300), color=(15, 23, 42))
        with patch("vision.ImageGrab.grab", return_value=mock_img):
            path = capture_screen_image()
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".png"))
            # Clean up
            try:
                os.remove(path)
            except Exception:
                pass

    def test_04_capture_screen_image_with_custom_bbox(self):
        """Verify capture_screen_image forwards custom crop bounding box to ImageGrab."""
        mock_img = Image.new("RGB", (200, 150), color=(2, 132, 199))
        with patch("vision.ImageGrab.grab", return_value=mock_img) as grab_mock:
            custom_bbox = (100, 100, 300, 250)
            path = capture_screen_image(bbox=custom_bbox)
            self.assertTrue(os.path.exists(path))
            grab_mock.assert_called()
            try:
                os.remove(path)
            except Exception:
                pass

    def test_05_capture_screen_image_with_monitor_id(self):
        """Verify capture_screen_image resolves monitor ID bbox."""
        fake_monitors = [
            {"id": 0, "name": "Mon 1", "bbox": (0, 0, 1920, 1080), "width": 1920, "height": 1080, "is_primary": True},
            {"id": 1, "name": "Mon 2", "bbox": (1920, 0, 3840, 1080), "width": 1920, "height": 1080, "is_primary": False}
        ]
        mock_img = Image.new("RGB", (1920, 1080), color=(10, 10, 10))
        with patch("vision.get_available_monitors", return_value=fake_monitors), \
             patch("vision.ImageGrab.grab", return_value=mock_img) as grab_mock:
            path = capture_screen_image(monitor_id=1)
            self.assertTrue(os.path.exists(path))
            grab_mock.assert_called()
            try:
                os.remove(path)
            except Exception:
                pass

    def test_06_solve_screen_coding_problem_with_image_path(self):
        """Verify solve_screen_coding_problem streams answer from existing snipped image."""
        # Create dummy image
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_img.close()
        im = Image.new("RGB", (100, 100), color=(255, 255, 255))
        im.save(temp_img.name, "PNG")

        tokens = []
        mock_client = MagicMock()
        
        # Mock streaming response
        mock_chunk = MagicMock()
        mock_choice = MagicMock()
        mock_choice.delta.content = "```python\ndef solve(): pass\n```"
        mock_chunk.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = [mock_chunk]

        result = solve_screen_coding_problem(
            client=mock_client,
            on_token=lambda t: tokens.append(t),
            custom_hint="Python 3",
            image_path=temp_img.name
        )

        self.assertIn("def solve()", result)
        self.assertGreaterEqual(len(tokens), 1)

        if os.path.exists(temp_img.name):
            try:
                os.remove(temp_img.name)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
