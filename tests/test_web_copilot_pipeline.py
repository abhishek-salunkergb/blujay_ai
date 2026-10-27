"""
Automated Test Suite for WebCopilotView & PyWebView Native HUD Bridge
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

from web_copilot_view import WebCopilotView, WebCopilotBridge
from interview_engine import InterviewSession
from main import apply_home_config_to_copilot


def test_web_copilot_bridge_api_coverage():
    print("--- [TEST] WebCopilotBridge JS-API Coverage ---")
    mock_view = MagicMock()
    bridge = WebCopilotBridge(mock_view)

    # 1. Test all exposed bridge methods
    bridge.manual_generate_answer()
    mock_view.manual_generate_answer.assert_called_once()

    bridge.trigger_screenshot_solver()
    mock_view.trigger_code_screen_solver.assert_called_once()

    bridge.trigger_region_snip()
    mock_view.trigger_region_snip_solver.assert_called_once()

    bridge.submit_manual_question("What is concurrency?")
    mock_view.submit_manual_question.assert_called_once_with("What is concurrency?")

    bridge.toggle_transcription()
    mock_view.toggle_live_transcription.assert_called_once()

    bridge.clear_transcript()
    mock_view.clear_transcript.assert_called_once()

    bridge.refine_answer("30s Short", "Make it under 60 words.")
    mock_view.refine_answer.assert_called_once_with("30s Short", "Make it under 60 words.")

    bridge.trigger_system_design()
    mock_view.trigger_system_design.assert_called_once()

    bridge.trigger_star_story()
    mock_view.trigger_star_story.assert_called_once()

    bridge.trigger_run_code()
    mock_view.trigger_run_code.assert_called_once()

    bridge.toggle_teleprompter()
    mock_view.toggle_teleprompter.assert_called_once()

    bridge.resize_window(True)
    mock_view.resize_window.assert_called_once_with(True)

    bridge.set_bubble_mode(True)
    mock_view.set_bubble_mode.assert_called_once_with(True)

    bridge.analyze_jd_fit("JD text", "Resume text")
    mock_view.handle_analyze_jd_fit.assert_called_once_with("JD text", "Resume text")

    bridge.end_session()
    mock_view.handle_end_session.assert_called_once()

    bridge.export_session_report("# Report", "report.md")
    mock_view.handle_export_session_report.assert_called_once_with("# Report", "report.md")

    bridge.buy_credits("100")
    mock_view.handle_buy_credits.assert_called_once_with("100")

    bridge.toggle_app_expansion()
    mock_view.toggle_app_expansion.assert_called_once()

    print("  ✓ All WebCopilotBridge JS hooks verified!")


def test_web_copilot_end_and_export_handlers():
    print("--- [TEST] WebCopilotView end_session, export_session_report, and buy_credits handlers ---")
    with patch("web_copilot_view.ContinuousTranscriber") as mock_transcriber_cls, \
         patch("web_copilot_view.GlobalHotkeyManager") as mock_hotkey_cls:
        
        mock_client = MagicMock()
        view = WebCopilotView(client=mock_client)
        view.window = MagicMock()

        # 1. Test handle_end_session
        end_res = view.handle_end_session()
        assert end_res["status"] == "ended"
        assert view.current_page == "home"
        assert view.is_card_expanded is False
        view.window.resize.assert_called_with(620, 750)

        # 2. Test handle_export_session_report
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "test_report.md")
            export_res = view.handle_export_session_report("# Test Report", test_file)
            assert export_res["success"] is True
            assert os.path.exists(export_res["path"])
            with open(export_res["path"], "r", encoding="utf-8") as rf:
                content = rf.read()
            assert "# Test Report" in content

        # 3. Test handle_buy_credits
        with patch("session_client.SessionClient.mock_buy_product") as mock_mb:
            mock_mb.return_value = (True, "Purchased 200 units")
            buy_res = view.handle_buy_credits("100")
            assert buy_res["success"] is True
            assert "Purchased" in buy_res["message"]

        # 4. Test toggle_app_expansion
        exp_res = view.toggle_app_expansion()
        assert exp_res["status"] == "toggled"

        print("  ✓ WebCopilotView end_session, export_session_report, and buy_credits verified!")


def test_web_copilot_view_session_integration():
    print("--- [TEST] WebCopilotView Session & State Integration ---")
    
    with patch("web_copilot_view.ContinuousTranscriber") as mock_transcriber_cls, \
         patch("web_copilot_view.GlobalHotkeyManager") as mock_hotkey_cls:
        
        mock_client = MagicMock()
        view = WebCopilotView(client=mock_client)
        assert isinstance(view.session, InterviewSession)
        assert "BluJay AI" in view.WINDOW_TITLE
        assert view.is_card_expanded is False

        # Apply Home Config to WebCopilotView
        config = {
            "company": "Netflix",
            "role": "Senior Distributed Systems Engineer",
            "job_description": "Architect high-scale video delivery pipelines in Go/Python.",
            "instructions": "Emphasize fault tolerance and chaos engineering.",
            "style": "Direct & Highly Technical",
            "language": "English",
            "model": "llama-3.3-70b-versatile"
        }
        apply_home_config_to_copilot(view, config)

        assert view.session.company_name == "Netflix"
        assert view.session.job_title == "Senior Distributed Systems Engineer"
        assert "video delivery pipelines" in view.session.job_description
        assert "chaos engineering" in view.session.custom_instructions
        assert view.session.tone == "Direct & Highly Technical"
        
        # Verify base profile block for LLM prompt reflects all information
        profile_block = view.session.get_base_profile_block()
        assert "Company: Netflix" in profile_block
        assert "Senior Distributed Systems Engineer" in profile_block
        print("  ✓ WebCopilotView properly configured via apply_home_config_to_copilot")


def test_web_copilot_html_and_css_assets():
    print("--- [TEST] Frontend Assets Integrity ---")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    html_path = os.path.join(frontend_dir, "index.html")
    css_path = os.path.join(frontend_dir, "style.css")
    js_path = os.path.join(frontend_dir, "app.js")

    assert os.path.exists(html_path), "index.html must exist in frontend/"
    assert os.path.exists(css_path), "style.css must exist in frontend/"
    assert os.path.exists(js_path), "app.js must exist in frontend/"

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    # Verify key elements in HTML
    assert "BluJay AI" in html_content
    assert "capsule-header" in html_content
    assert "action-mode-bar" in html_content
    assert "glass-qa-card" in html_content
    assert "Summarized question:" in html_content
    assert "⭐ Answer:" in html_content
    assert "floating-app-bubble" in html_content

    # Verify CSS design system
    assert "--glass-bg" in css_content
    assert "backdrop-filter: blur" in css_content
    assert "code-block-wrapper" in css_content
    assert "floating-app-bubble" in css_content

    # Verify JS event hooks
    assert "window.onQuestionDetected" in js_content
    assert "window.onAnswerToken" in js_content
    assert "window.onAnswerComplete" in js_content
    assert "window.onTranscriptChunk" in js_content
    assert "window.restoreFromBubble" in js_content

    print("  ✓ All HTML5, CSS3, and JS assets verified and structurally intact!")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    test_web_copilot_bridge_api_coverage()
    test_web_copilot_end_and_export_handlers()
    test_web_copilot_view_session_integration()
    test_web_copilot_html_and_css_assets()
    print("✅ All WebCopilot Pipeline Tests Passed!")
