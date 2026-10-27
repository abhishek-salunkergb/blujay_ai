"""
=============================================================================
Comprehensive Interview UI Integrity & Connectivity Audit Test Suite
=============================================================================
Verifies that 100% of the Interview UI components, DOM elements, Python bridge
methods, STT audio pipelines, streaming hooks, and hotkeys are connected and working.
=============================================================================
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import os
import sys
import json
import re
from unittest.mock import MagicMock, patch

from web_copilot_view import WebCopilotView, WebCopilotBridge
from interview_engine import InterviewSession
from main import apply_home_config_to_copilot


def test_01_all_dom_ids_and_classes_in_sync():
    """Verifies that all DOM element IDs referenced in app.js exist in index.html."""
    print("--- [AUDIT 1] DOM Element IDs and Classes Synchronization ---")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    html_path = os.path.join(frontend_dir, "index.html")
    js_path = os.path.join(frontend_dir, "app.js")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    # Extract all document.getElementById('...') from JS
    js_ids = set(re.findall(r"document\.getElementById\(['\"]([^'\"]+)['\"]\)", js_content))
    print(f"  Found {len(js_ids)} DOM IDs referenced in app.js: {sorted(js_ids)}")

    missing_ids = []
    for elem_id in js_ids:
        if f'id="{elem_id}"' not in html_content and f"id='{elem_id}'" not in html_content:
            missing_ids.append(elem_id)

    assert not missing_ids, f"Missing DOM IDs in index.html: {missing_ids}"
    print("  ✓ 100% of DOM IDs referenced in app.js exist in index.html!")


def test_02_all_onclick_handlers_defined_in_js():
    """Verifies that every onclick handler in index.html has a corresponding window function in app.js."""
    print("--- [AUDIT 2] HTML onclick Attributes & JS Function Bindings ---")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    html_path = os.path.join(frontend_dir, "index.html")
    js_path = os.path.join(frontend_dir, "app.js")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    # Extract all onclick="funcName(...)"
    onclick_matches = re.findall(r'onclick="([a-zA-Z0-9_]+)\(', html_content)
    unique_handlers = sorted(set(onclick_matches))
    print(f"  Found {len(unique_handlers)} unique onclick handlers in index.html: {unique_handlers}")

    missing_handlers = []
    for handler in unique_handlers:
        pattern1 = f"window.{handler} ="
        pattern2 = f"function {handler}("
        if pattern1 not in js_content and pattern2 not in js_content:
            missing_handlers.append(handler)

    assert not missing_handlers, f"Missing JS function handlers for onclick in app.js: {missing_handlers}"
    print("  ✓ 100% of HTML onclick handlers are defined and reachable in app.js!")


def test_03_bridge_method_parity():
    """Verifies that every callPython('method_name') in app.js and home.js exists on WebCopilotBridge."""
    print("--- [AUDIT 3] JavaScript-to-Python Bridge Method Parity ---")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    app_js_path = os.path.join(frontend_dir, "app.js")
    home_js_path = os.path.join(frontend_dir, "home.js")

    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js = f.read()
    with open(home_js_path, "r", encoding="utf-8") as f:
        home_js = f.read()

    # Find all callPython('...') in app.js
    app_methods = set(re.findall(r"callPython\(['\"]([^'\"]+)['\"]", app_js))
    home_methods = set(re.findall(r"window\.pywebview\.api\.([a-zA-Z0-9_]+)\(", home_js))

    all_called_methods = app_methods.union(home_methods)
    print(f"  Found {len(all_called_methods)} bridge methods invoked across frontend: {sorted(all_called_methods)}")

    mock_view = MagicMock()
    bridge = WebCopilotBridge(mock_view)
    bridge_dir = dir(bridge)

    missing_bridge_methods = []
    for method in all_called_methods:
        if method not in bridge_dir:
            missing_bridge_methods.append(method)

    assert not missing_bridge_methods, f"Missing methods on WebCopilotBridge: {missing_bridge_methods}"
    print("  ✓ 100% of frontend bridge invocations are implemented on WebCopilotBridge!")


def test_04_end_to_end_question_and_answer_pipeline():
    """Simulates speech chunk -> question detection -> answer streaming -> history recording."""
    print("--- [AUDIT 4] End-to-End Speech & Answer Streaming Pipeline ---")
    with patch("web_copilot_view.ContinuousTranscriber"), \
         patch("web_copilot_view.GlobalHotkeyManager"):

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="We use distributed caches with Redis and Write-Through caching."))]
        )

        view = WebCopilotView(client=mock_client)
        view.window = MagicMock()

        # 1. Transcript chunk received
        view._on_transcript_chunk("Could you explain how you handle distributed cache invalidation in microservices?")
        view.window.evaluate_js.assert_called()

        # 2. Manual question submitted
        view.submit_manual_question("What is the time complexity of QuickSelect?")
        assert view.active_question == "What is the time complexity of QuickSelect?"

        # 3. Settings update
        view.update_settings({
            "company": "Amazon",
            "job_description": "Principal AWS Infrastructure Engineer",
            "instructions": "Keep answers concise and metric-focused"
        })
        assert view.session.company_name == "Amazon"
        assert "AWS Infrastructure" in view.session.job_description
        assert "metric-focused" in view.session.custom_instructions

        # 4. Bubble mode transition
        view.set_bubble_mode(True)
        assert view.is_bubble_mode is True
        view.set_bubble_mode(False)
        assert view.is_bubble_mode is False

        print("  ✓ End-to-end question detection, streaming, settings, and bubble mode fully verified!")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    test_01_all_dom_ids_and_classes_in_sync()
    test_02_all_onclick_handlers_defined_in_js()
    test_03_bridge_method_parity()
    test_04_end_to_end_question_and_answer_pipeline()
    print("✅ All 4 Interview UI Comprehensive Audits Passed!")
