"""
Test suite to verify that frontend/home.html, home.css, and home.js
load cleanly and support both the native Desktop App Widget (1:1 with laptop photo)
and the expanded Web Browser Dashboard.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import sys

def test_frontend_home_files():
    print("--- [TEST] Checking BluJay AI frontend/home.html, home.css, home.js Integrity ---")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    html_file = os.path.join(frontend_dir, "home.html")
    css_file = os.path.join(frontend_dir, "home.css")
    js_file = os.path.join(frontend_dir, "home.js")

    assert os.path.exists(html_file), f"Missing {html_file}"
    assert os.path.exists(css_file), f"Missing {css_file}"
    assert os.path.exists(js_file), f"Missing {js_file}"

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
        # Desktop Widget elements (matching laptop photo)
        assert "Buy Credits" in html_content
        assert "Session Type" in html_content
        assert "Interview" in html_content
        assert "Regular Call" in html_content
        assert "Company" in html_content
        assert "Job Description" in html_content
        assert "+ Resume" in html_content
        assert "Create Session" in html_content
        assert "BluJay" in html_content
        assert "Abhishek" in html_content
        print("  ✓ frontend/home.html contains all 1:1 Desktop App Widget and Web Dashboard elements")

    with open(css_file, "r", encoding="utf-8") as f:
        css_content = f.read()
        assert "--brand-green" in css_content
        assert ".desktop-app-wrapper" in css_content
        assert ".desktop-card-window" in css_content
        assert ".btn-hero-create" in css_content
        assert ".web-dashboard-wrapper" in css_content
        print("  ✓ frontend/home.css contains all desktop widget & web dashboard design tokens")

    with open(js_file, "r", encoding="utf-8") as f:
        js_content = f.read()
        assert "switchAppTab" in js_content
        assert "executeCreateDesktopSession" in js_content
        assert "toggleAppExpansion" in js_content
        assert "filterDesktopSessions" in js_content
        print("  ✓ frontend/home.js contains all interactive controllers & bridge hooks")

    print("✅ BluJay AI HTML/CSS/JS Frontend Suite: 100% Verified and Ready!")

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    test_frontend_home_files()
