"""
Automated Test Suite for Parakeet Web Engine & Home UI Controller
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import sys
import tempfile
from unittest.mock import MagicMock

from home_ui import (
    ModernHomeDashboard,
    HomeDashboardBridge,
    SessionDetailsDialog,
    UserProfileDialog,
    BuyCreditsDialog,
    CreateSessionDialog
)
from interview_engine import InterviewSession
from main import apply_home_config_to_copilot


def test_home_dashboard_controller():
    print("--- [TEST] ModernHomeDashboard & PyWebView Bridge Integrity ---")
    dashboard = ModernHomeDashboard()
    assert dashboard.title_text == "✦ BluJay AI - Interview Copilot"
    assert "name" in dashboard.candidate_profile
    print("  ✓ ModernHomeDashboard initialized cleanly")

    bridge = HomeDashboardBridge(dashboard)
    assert hasattr(bridge, "launch_session")
    assert hasattr(bridge, "browse_document")
    assert hasattr(bridge, "save_profile")
    assert hasattr(bridge, "fetch_dashboard_data")
    assert hasattr(bridge, "buy_credits")
    print("  ✓ PyWebView Bridge contains all required Parakeet AI JS hooks")

    # Modals
    dummy_session = {
        "id": "sess_01",
        "job_title": "Senior AI Architect",
        "company_name": "OpenAI",
        "qa_pairs": [{"question": "Explain FlashAttention-2", "answer": "Tiling in SRAM..."}]
    }
    s_dlg = SessionDetailsDialog(None, dummy_session)
    assert s_dlg.session_data["id"] == "sess_01"
    print("  ✓ SessionDetailsDialog initialized cleanly")

    p_dlg = UserProfileDialog(None)
    assert isinstance(p_dlg.profile, dict)
    print("  ✓ UserProfileDialog initialized cleanly")

    c_dlg = CreateSessionDialog(None)
    assert isinstance(c_dlg.config, dict)
    print("  ✓ CreateSessionDialog initialized cleanly")


def test_home_ui_to_session_pipeline_integration():
    """Verify that Home UI configurations (Company, JD, Resume, Instructions, Tone) flow directly into InterviewSession."""
    print("--- [TEST] End-to-End Home UI -> Session -> LLM Prompt Pipeline ---")
    
    # 1. Create a dummy resume file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("Candidate: Abhishek Salunke\nExperience: 5 years in Python, FastAPI, and Distributed Systems.\nProjects: Built real-time high-throughput caching engine.")
        temp_resume_path = f.name

    try:
        # 2. Simulate Home UI launch payload
        config = {
            "company": "Stripe",
            "role": "Staff Backend Engineer",
            "job_description": "We are looking for a Staff Engineer to lead high-throughput financial infrastructure in Python and Go.",
            "instructions": "Be ultra-concise, focus on distributed idempotency, and keep responses under 90 seconds.",
            "style": "STAR Behavioral Framework",
            "language": "English",
            "model": "openai/gpt-oss-120b",
            "documents": [temp_resume_path],
            "audio_source": "dual",
            "auto_gen": True
        }

        # 3. Simulate App and Copilot View
        mock_app = MagicMock()
        mock_view = MagicMock()
        mock_session = InterviewSession()
        mock_view.session = mock_session
        mock_view.transcriber.is_running = True
        mock_app.copilot_view = mock_view

        # 4. Apply home config
        apply_home_config_to_copilot(mock_app, config)

        # 5. Assert all fields reached the session state
        assert mock_session.company_name == "Stripe"
        assert mock_session.job_title == "Staff Backend Engineer"
        assert "financial infrastructure" in mock_session.job_description
        assert "distributed idempotency" in mock_session.custom_instructions
        assert mock_session.tone == "STAR Behavioral Framework"
        assert mock_session.language == "English"
        assert mock_session.model_name == "openai/gpt-oss-120b"
        assert "high-throughput caching engine" in mock_session.resume_text

        # 6. Verify base profile block for LLM prompt reflects all information
        profile_block = mock_session.get_base_profile_block()
        assert "Company: Stripe" in profile_block
        assert "Staff Backend Engineer" in profile_block
        assert "financial infrastructure" in profile_block
        assert "high-throughput caching engine" in profile_block
        print("  ✓ Profile block successfully includes Company, JD, and Resume content")

    finally:
        if os.path.exists(temp_resume_path):
            try:
                os.remove(temp_resume_path)
            except Exception:
                pass

    print("✅ Home UI to Interview Session Integration: 100% Verified!")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    test_home_dashboard_controller()
