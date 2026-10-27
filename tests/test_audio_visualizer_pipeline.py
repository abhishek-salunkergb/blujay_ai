"""
Unit Test Suite for Live Audio Visualizer / Mic Health Pulse (Feature 2).
Validates:
1. DOM elements for System & Mic audio visualizer in frontend/index.html.
2. High-performance CSS styling & wave bar animations in frontend/style.css.
3. JavaScript real-time level listener in frontend/app.js.
4. Python bridge method _on_audio_level_update throttling and JS dispatch in web_copilot_view.py.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
from unittest.mock import MagicMock, patch
import os
import re

import web_copilot_view
from web_copilot_view import WebCopilotView


class TestAudioVisualizerPipeline(unittest.TestCase):

    def test_01_index_html_visualizer_elements_exist(self):
        """Verifies dual audio visualizer markup in frontend/index.html."""
        html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        self.assertIn('id="audio-health-visualizer"', html)
        self.assertIn('id="system-wave-bars"', html)
        self.assertIn('id="mic-wave-bars"', html)
        self.assertIn('class="wave-bar"', html)
        self.assertIn('class="audio-stream-pill system-stream"', html)
        self.assertIn('class="audio-stream-pill mic-stream"', html)

    def test_02_style_css_visualizer_rules_exist(self):
        """Verifies CSS styling for wave bars and active audio pill states in frontend/style.css."""
        css_path = os.path.join(os.path.dirname(__file__), "frontend", "style.css")
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
            
        self.assertIn('.audio-health-visualizer', css)
        self.assertIn('.audio-stream-pill', css)
        self.assertIn('.audio-stream-pill.active-speaking.mic-stream', css)
        self.assertIn('.audio-stream-pill.active-speaking.system-stream', css)
        self.assertIn('.audio-wave-bars', css)
        self.assertIn('.wave-bar', css)

    def test_03_app_js_on_audio_level_update_exists(self):
        """Verifies JavaScript visualizer animation engine in frontend/app.js."""
        js_path = os.path.join(os.path.dirname(__file__), "frontend", "app.js")
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
            
        self.assertIn('window.onAudioLevelUpdate', js)
        self.assertIn('renderAudioVisualizer', js)
        self.assertIn('mic-wave-bars', js)
        self.assertIn('system-wave-bars', js)

    @patch("web_copilot_view.ContinuousTranscriber")
    @patch("web_copilot_view.webview")
    def test_04_python_on_audio_level_update_dispatches_eval_js(self, mock_webview, mock_transcriber):
        """Verifies _on_audio_level_update formats and dispatches JS evaluate calls."""
        view = WebCopilotView(client=MagicMock())
        view._eval_js = MagicMock()
        
        # Simulate mic level
        view._on_audio_level_update("mic", level=0.75, rms=0.0)
        view._eval_js.assert_called_with("if(window.onAudioLevelUpdate) window.onAudioLevelUpdate('mic', 75);")
        
        # Simulate speaker loopback level
        view._last_audio_level_ts = 0  # reset throttle
        view._on_audio_level_update("system", level=0.42, rms=0.0)
        view._eval_js.assert_called_with("if(window.onAudioLevelUpdate) window.onAudioLevelUpdate('system', 42);")


if __name__ == "__main__":
    unittest.main()
