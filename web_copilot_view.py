"""
=============================================================================
BluJay AI - Unified Web Application & PyWebView Copilot Engine
=============================================================================
Provides a seamless, hardware-accelerated single-process lifecycle:
1. Opens the sleek modern Home Dashboard (home.html)
2. Transitions dynamically into the 1:1 Frosted Glass Copilot HUD (index.html)
3. Preserves 100% of Python audio STT, Groq LLM streaming, Vision OCR, and RAG
=============================================================================
"""

import os
import sys
import time
import json
import threading
import re
import uuid
import ctypes
import logging
import tkinter as tk
from typing import Optional, Any, List

logging.getLogger("pywebview").setLevel(logging.CRITICAL)

try:
    import webview
except ImportError:
    webview = None

from interview_engine import (
    InterviewSession,
    ContinuousTranscriber,
    SessionController,
    generate_interview_answer,
    extract_text_from_file
)
from hotkey_manager import GlobalHotkeyManager, get_saved_hotkey_preferences
from vision import solve_screen_coding_problem
from star_story_manager import find_best_star_story, format_star_spoken_answer
from system_design_generator import generate_system_design_diagram
from interview_exporter import export_session_to_markdown
from screen_snipper import InteractiveRegionSnipper
from session_client import client as session_client


# Windows Stealth Screen Capture Exclusion Constant
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def set_click_through(hwnd: int, enabled: bool) -> bool:
    """Toggles mouse events passing directly through the overlay window to underlying windows."""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_LAYERED = 0x00080000
        WS_EX_TOOLWINDOW = 0x00000080

        if hasattr(user32, "GetWindowLongPtrW"):
            get_long = user32.GetWindowLongPtrW
            set_long = user32.SetWindowLongPtrW
        else:
            get_long = user32.GetWindowLongW
            set_long = user32.SetWindowLongW

        style = get_long(wintypes.HWND(hwnd), GWL_EXSTYLE)
        if enabled:
            new_style = style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
        else:
            new_style = (style & ~WS_EX_TRANSPARENT) | WS_EX_LAYERED
        set_long(wintypes.HWND(hwnd), GWL_EXSTYLE, new_style)
        return True
    except Exception as e:
        print(f"Error setting click-through: {e}")
        return False


class WebCopilotBridge:
    """
    Unified JS-accessible API bridge exposed directly to window.pywebview.api.
    Handles events from both home.html/home.js and index.html/app.js.
    """
    def __init__(self, view: "WebCopilotView"):
        self._view = view

    # -------------------------------------------------------------------------
    # Home Dashboard Hooks (home.js)
    # -------------------------------------------------------------------------
    def launch_session(self, config: dict):
        return self._view.handle_launch_session(config)

    def browse_document(self):
        self._view.handle_browse_document()

    def upload_document_content(self, filename: str, content: str):
        self._view.handle_upload_document_content(filename, content)

    def save_profile(self, profile: dict):
        self._view.candidate_profile.update(profile)

    def fetch_dashboard_data(self) -> dict:
        try:
            return session_client.fetch_home_dashboard()
        except Exception:
            return {}

    def buy_credits(self, tier: str):
        return self._view.handle_buy_credits(tier)

    def close_window(self):
        self._view.close_window()

    def minimize_window(self):
        self._view.minimize_window()

    def minimize_to_floating_bubble(self):
        self._view.minimize_to_bubble()

    def toggle_app_expansion(self):
        return self._view.toggle_app_expansion()

    def analyze_jd_fit(self, jd_text: str, resume_text: str = "") -> dict:
        return self._view.handle_analyze_jd_fit(jd_text, resume_text)

    # -------------------------------------------------------------------------
    # Live Copilot HUD Hooks (app.js)
    # -------------------------------------------------------------------------
    def copilot_ready(self):
        return self._view.handle_copilot_ready()

    def manual_generate_answer(self):
        self._view.manual_generate_answer()

    def trigger_screenshot_solver(self):
        self._view.trigger_code_screen_solver()

    def trigger_region_snip(self):
        self._view.trigger_region_snip_solver()

    def submit_manual_question(self, text: str):
        self._view.submit_manual_question(text)

    def toggle_transcription(self):
        self._view.toggle_live_transcription()

    def clear_transcript(self):
        self._view.clear_transcript()

    def refine_answer(self, mode: str, prompt: str):
        self._view.refine_answer(mode, prompt)

    def trigger_system_design(self):
        self._view.trigger_system_design()

    def trigger_star_story(self):
        self._view.trigger_star_story()

    def trigger_run_code(self):
        self._view.trigger_run_code()

    def toggle_teleprompter(self):
        self._view.toggle_teleprompter()

    def toggle_ghost_mode(self):
        self._view.toggle_ghost_mode()

    def dock_to_camera(self):
        self._view.dock_to_camera()

    def trigger_mock_interview(self):
        self._view.trigger_mock_interview()

    def export_session_report(self, markdown_content: str = "", filename: str = ""):
        return self._view.handle_export_session_report(markdown_content, filename)

    def minimize_to_bubble(self):
        self._view.minimize_to_bubble()

    def update_audio_source(self, source: str):
        self._view.update_audio_source(source)

    def set_bubble_mode(self, is_bubble: bool):
        self._view.set_bubble_mode(is_bubble)

    def end_session(self):
        return self._view.handle_end_session()

    def copy_to_clipboard(self, text: str):
        self._view.copy_to_clipboard(text)

    def update_settings(self, config: dict):
        self._view.update_settings(config)

    def resize_window(self, expanded: bool):
        self._view.resize_window(expanded)


class WebCopilotView:
    """
    Unified Application Controller managing the PyWebView engine, Home Dashboard,
    and the 1:1 Frosted Glass Copilot HUD.
    """
    WINDOW_TITLE = "✦ BluJay AI - Interview Copilot"

    def __init__(self, client=None, document_context="", on_close_callback=None, main_app=None):
        self.client = client
        if self.client is None:
            api_k = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
            if api_k:
                try:
                    from groq import Groq
                    self.client = Groq(api_key=api_k)
                except Exception:
                    pass

        self.document_context = document_context
        self.on_close_callback = on_close_callback
        self.main_app = main_app

        # Session & Audio Pipeline
        self.session = InterviewSession()
        self.session_controller = getattr(self.session, "controller", None) or SessionController(self.session)
        self.transcriber = ContinuousTranscriber(
            client=self.client,
            on_transcript_chunk=self._on_transcript_chunk,
            on_status_change=self._on_audio_status,
            on_interim_transcript=self._on_interim_transcript,
            on_audio_level=self._on_audio_level_update,
            on_turn_resumed=self._on_turn_resumed,
            on_eager_eot=self._on_eager_eot,
            session=self.session
        )

        # Candidate Profile
        self.candidate_profile = {
            "name": "Abhishek Salunke",
            "skills": "Python, FastAPI, React, PostgreSQL, Docker, RAG, LLMs",
            "tone": "Direct & Highly Technical"
        }
        self.selected_documents: List[str] = []

        # State
        self.window: Optional[Any] = None
        self.is_card_expanded = False
        self.active_question = ""
        self.active_raw_answer = ""
        self.current_page = "home"  # "home" | "copilot"
        self._session_start_time = time.time()
        self.is_teleprompter_active = False
        self.teleprompter_speed = 1.0
        self.is_ghost_mode = False
        self.is_card_expanded = False

        # Hotkeys
        self.hotkey_manager = GlobalHotkeyManager()
        self._init_global_hotkeys()

        # Immediate pipeline pre-warming on application launch
        try:
            from interview_engine import prewarm_pipeline_connections
            prewarm_pipeline_connections(self.session)
        except Exception:
            pass

    def _init_global_hotkeys(self):
        try:
            prefs = get_saved_hotkey_preferences()
            combo_hide = prefs.get("panic_hide", "Ctrl+Alt+H")
            self.hotkey_manager.register("panic_hide", combo_hide, self.toggle_overlay_visibility)

            combo_solve = prefs.get("solve_screen", "Ctrl+Alt+S")
            self.hotkey_manager.register("solve_screen", combo_solve, self.trigger_code_screen_solver)

            combo_mute = prefs.get("mute_ai", "Ctrl+Alt+M")
            self.hotkey_manager.register("mute_ai", combo_mute, self.toggle_live_transcription)

            combo_regen = prefs.get("regenerate", "Ctrl+Alt+R")
            self.hotkey_manager.register("regenerate", combo_regen, self.manual_generate_answer)

            combo_tp = prefs.get("teleprompter", "Ctrl+Alt+P")
            self.hotkey_manager.register("teleprompter", combo_tp, self.toggle_teleprompter)

            combo_ghost = prefs.get("ghost_mode", "Ctrl+Alt+G")
            self.hotkey_manager.register("ghost_mode", combo_ghost, self.toggle_ghost_mode)

            combo_dock = prefs.get("dock_camera", "Ctrl+Alt+C")
            self.hotkey_manager.register("dock_camera", combo_dock, self.dock_to_camera)

            self.hotkey_manager.start()
        except Exception:
            pass

    def run(self):
        """Starts the unified PyWebView application on home.html."""
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
        html_path = os.path.join(frontend_dir, "home.html")

        bridge = WebCopilotBridge(self)

        if webview is not None and os.path.exists(html_path):
            norm_url = f"file:///{html_path.replace(os.sep, '/')}"
            self.window = webview.create_window(
                title=self.WINDOW_TITLE,
                url=norm_url,
                js_api=bridge,
                width=620,
                height=750,
                min_size=(50, 50),
                text_select=True,
                easy_drag=False,
                transparent=True,
                frameless=True,
                on_top=True
            )

            # Single process main loop - stealth watcher is started cleanly in _on_window_loaded
            webview.start(self._on_window_loaded, debug=False)
        else:
            print("pywebview not available, launching native fallback...")
            try:
                from legacy.interview_ui import InterviewCopilotView
            except ImportError:
                print("⚠️ PyWebView / Microsoft Edge WebView2 runtime is required to run BluJay AI.")
                return
            root = tk.Tk()
            root.title(self.WINDOW_TITLE)
            frame = tk.Frame(root)
            frame.pack(fill="both", expand=True)
            tk_view = InterviewCopilotView(
                parent_frame=frame,
                root=root,
                client=self.client,
                main_app_ref=self.main_app
            )
            root.mainloop()

    def _on_window_loaded(self):
        """Called when webview is loaded and ready."""
        if self.window:
            try:
                self.window.events.closing += self._on_window_closing
                self.window.events.closed += self._on_window_closing
            except Exception:
                pass
        
        # Start stealth loop in background after WebView2 render engine is initialized
        threading.Thread(target=self._stealth_loop, daemon=True, name="StealthWatcherThread").start()

    def _on_window_closing(self):
        self._is_closing = True
        self._closed = True
        def _cleanup():
            if getattr(self, "transcriber", None):
                try:
                    self.transcriber.stop()
                except Exception:
                    pass
            if getattr(self, "hotkey_manager", None):
                try:
                    self.hotkey_manager.stop()
                except Exception:
                    pass
            try:
                if getattr(self, "on_close_callback", None):
                    self.on_close_callback()
            except Exception as ce:
                print("Notice in on_close_callback:", ce)
            time.sleep(0.1)
            os._exit(0)
        threading.Thread(target=_cleanup, daemon=True, name="WindowClosingExitThread").start()

    def _get_overlay_hwnds(self) -> List[int]:
        """
        Discovers all valid top-level and child HWNDs belonging exclusively to the current process.
        Recursively discovers WebView2 render surfaces (Chrome_WidgetWin_0, Intermediate D3D Window)
        to guarantee 100% complete invisibility on all screen-sharing software.
        """
        if sys.platform != "win32":
            return []
        
        hwnds = []
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            current_pid = os.getpid()

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

            def enum_child_proc(ch, _):
                try:
                    c_pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(wintypes.HWND(ch), ctypes.byref(c_pid))
                    if c_pid.value == current_pid and user32.IsWindowVisible(wintypes.HWND(ch)):
                        hwnds.append(ch)
                except Exception:
                    pass
                return True

            child_callback = WNDENUMPROC(enum_child_proc)

            def enum_proc(h, _):
                try:
                    w_pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(wintypes.HWND(h), ctypes.byref(w_pid))
                    if w_pid.value == current_pid:
                        c_buf = ctypes.create_unicode_buffer(256)
                        user32.GetClassNameW(wintypes.HWND(h), c_buf, 256)
                        class_name = c_buf.value

                        # Filter out internal IME/GDI message-only helper windows and transient OS dialogs
                        if class_name in (
                            "MSCTFIME UI", "IME", "GDI+ Hook Window Class", 
                            "#32770", "tooltips_class32", "SysShadow", 
                            "WorkerW", "TclNotifier", "TtkMonitorClass", "Auto-Suggest Dropdown"
                        ) or ".NET-BroadcastEventWindow" in class_name:
                            return True

                        # Top-level or unparented popup/tool window
                        parent = user32.GetParent(wintypes.HWND(h))
                        if not parent or not user32.IsWindowVisible(parent):
                            hwnds.append(h)
                            # Also enumerate all child renderer controls (Edge WebView2, D3D surfaces)
                            try:
                                user32.EnumChildWindows(wintypes.HWND(h), child_callback, 0)
                            except Exception:
                                pass
                except Exception:
                    pass
                return True

            user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
        except Exception as e:
            print("[Capture Exclusion] Error enumerating process HWNDs:", e)

        # Deduplicate while preserving discovery order
        seen = set()
        unique_hwnds = []
        for x in hwnds:
            if x not in seen:
                seen.add(x)
                unique_hwnds.append(x)
        return unique_hwnds

    def _stealth_loop(self):
        """
        Lightweight periodic verifier checking tracked overlay and child HWNDs
        and detecting window recreation without high-frequency global overhead.
        """
        # Initial stabilization pause (0.3s) so WebView2 finishes DirectComposition binding
        time.sleep(0.3)
        self._apply_stealth_protection(force_log=True)

        while not getattr(self, "_is_closing", False):
            try:
                time.sleep(1.5)
                self._apply_stealth_protection(force_log=False)
            except Exception as e:
                pass

    def _apply_stealth_protection(self, force_log: bool = False):
        """
        Guaranteed 100% stealth capture exclusion for Windows capture mechanisms that respect WDA_EXCLUDEFROMCAPTURE (0x11).
        Targets all top-level process windows and child renderer HWNDs with immediate GetWindowDisplayAffinity validation.
        """
        if sys.platform != "win32":
            return
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            current_pid = os.getpid()

            target_hwnds = self._get_overlay_hwnds()
            if not hasattr(self, "_tracked_hwnds"):
                self._tracked_hwnds = set()

            # Detect HWND recreation / new window creation
            current_set = set(target_hwnds)
            new_hwnds = current_set - self._tracked_hwnds
            if new_hwnds:
                for nh in new_hwnds:
                    t_buf = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(wintypes.HWND(nh), t_buf, 256)
                    c_buf = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(wintypes.HWND(nh), c_buf, 256)
                    if t_buf.value or "WindowsForms" in c_buf.value:
                        print(f"[Capture Exclusion] New/Recreated HWND detected: {hex(nh)} | Class: '{c_buf.value}' | Title: '{t_buf.value}' (PID: {current_pid})")
                self._tracked_hwnds = current_set

            for h in target_hwnds:
                # 1. Apply SetWindowDisplayAffinity (0x11 = WDA_EXCLUDEFROMCAPTURE)
                set_success = user32.SetWindowDisplayAffinity(wintypes.HWND(h), wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE))
                err_code = kernel32.GetLastError() if not set_success else 0

                # 2. Immediately verify with GetWindowDisplayAffinity
                aff_val = wintypes.DWORD()
                get_success = user32.GetWindowDisplayAffinity(wintypes.HWND(h), ctypes.byref(aff_val))
                verified_affinity = aff_val.value if get_success else -1

                # 3. Log results if forced or on failure
                if not set_success or verified_affinity != WDA_EXCLUDEFROMCAPTURE:
                    t_buf = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(wintypes.HWND(h), t_buf, 256)
                    c_buf = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(wintypes.HWND(h), c_buf, 256)
                    if t_buf.value and ("BluJay" in t_buf.value or "WindowsForms" in c_buf.value):
                        print(
                            f"[Capture Exclusion Warning] SetWindowDisplayAffinity failed for HWND {hex(h)}!\n"
                            f"  LastError: {err_code} | Process PID: {current_pid}\n"
                            f"  Class: '{c_buf.value}' | Title: '{t_buf.value}'\n"
                            f"  Set Result: {bool(set_success)} | Verified Affinity: {hex(verified_affinity) if verified_affinity >= 0 else 'N/A'}"
                        )
                elif force_log:
                    t_buf = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(wintypes.HWND(h), t_buf, 256)
                    c_buf = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(wintypes.HWND(h), c_buf, 256)
                    if t_buf.value:
                        print(f"[Capture Exclusion Verified] HWND: {hex(h)} | Class: '{c_buf.value}' | Affinity: {hex(verified_affinity)} (WDA_EXCLUDEFROMCAPTURE)")
        except Exception as e:
            print("[Capture Exclusion] Exception in _apply_stealth_protection:", e)

    def _eval_js(self, js_code: str):
        """Safely dispatches JavaScript code to the webview window."""
        if getattr(self, "_is_closing", False):
            return
        if self.window:
            try:
                self.window.evaluate_js(js_code)
            except (Exception, BaseException):
                pass

    # =========================================================================
    # Home Dashboard Actions
    # =========================================================================

    def handle_browse_document(self):
        """Opens native file picker and synchronizes document with RAG without Tkinter thread errors."""
        def _picker():
            try:
                file_path = None
                if self.window and hasattr(self.window, "create_file_dialog"):
                    res = self.window.create_file_dialog(
                        webview.OPEN_DIALOG,
                        allow_multiple=False,
                        file_types=('Documents (*.pdf;*.docx;*.txt;*.md)', 'All files (*.*)')
                    )
                    if res and len(res) > 0:
                        file_path = res[0]

                if file_path and os.path.exists(file_path):
                    self.selected_documents = [file_path]
                    filename = os.path.basename(file_path)
                    r_text = extract_text_from_file(file_path)
                    if r_text:
                        self.session.resume_file_path = file_path
                        self.session.resume_text = r_text
                        self.session.invalidate_profile_cache()
                    self._eval_js(f"if(window.onDocumentSynced) window.onDocumentSynced('{filename}');")
                    print(f"[Document Sync] Selected document '{filename}' parsed successfully ({len(r_text or '')} characters).")
            except (Exception, BaseException) as pe:
                print("File picker notice:", pe)

        threading.Thread(target=_picker, daemon=True).start()

    def handle_upload_document_content(self, filename: str, content: str):
        """Processes directly uploaded or drag-and-dropped document content."""
        try:
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "uploads")
            os.makedirs(temp_dir, exist_ok=True)
            target_path = os.path.join(temp_dir, filename)

            if content.startswith("data:") and "," in content:
                import base64
                b64_data = content.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_data)
                with open(target_path, "wb") as f:
                    f.write(raw_bytes)
                r_text = extract_text_from_file(target_path)
            else:
                with open(target_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(content)
                r_text = content

            if r_text:
                self.session.resume_file_path = target_path
                self.session.resume_text = r_text
                self.session.invalidate_profile_cache()
                self.selected_documents = [target_path]
                self._eval_js(f"if(window.onDocumentSynced) window.onDocumentSynced('{filename}');")
                print(f"[Document Sync] Uploaded resume '{filename}' parsed successfully ({len(r_text)} characters).")
        except Exception as e:
            print("Error handling document upload:", e)

    def handle_analyze_jd_fit(self, jd_text: str, resume_text: str = "") -> dict:
        """Analyzes fit between Job Description and Resume using JD Matcher engine."""
        try:
            from jd_matcher import analyze_jd_resume_fit, save_jd_analysis_cache
            effective_resume = resume_text or getattr(self.session, "resume_text", "") or ""
            target_role = getattr(self.session, "job_title", None) or "Software Engineer"
            result = analyze_jd_resume_fit(
                jd_text=jd_text,
                resume_text=effective_resume,
                client=self.client,
                job_title=target_role
            )
            save_jd_analysis_cache(result)
            return result
        except Exception as e:
            print(f"Error in handle_analyze_jd_fit: {e}")
            from jd_matcher import heuristic_jd_match
            effective_resume = resume_text or getattr(self.session, "resume_text", "") or ""
            return heuristic_jd_match(jd_text, effective_resume)

    def handle_launch_session(self, config: dict):
        """Transitions the window from Home Dashboard to the Frosted Glass Copilot HUD."""
        # 1. Attach selected documents
        if not config.get("documents") and self.selected_documents:
            config["documents"] = list(self.selected_documents)

        # 2. Ingest configs into session
        if config.get("role"):
            self.session.job_title = config["role"]
        if config.get("company"):
            self.session.company_name = config["company"]
        if config.get("job_description"):
            self.session.job_description = config["job_description"]
        if config.get("instructions"):
            self.session.custom_instructions = config["instructions"]
        if config.get("style"):
            self.session.tone = config["style"]
        if config.get("language"):
            self.session.language = config["language"]
        if config.get("model"):
            self.session.model_name = config["model"]
        if config.get("audio_source"):
            src = str(config["audio_source"]).lower()
            if "dual" in src:
                self.session.audio_source = "dual"
            elif "loopback" in src or "system" in src or "speaker" in src:
                self.session.audio_source = "system"
            elif "mic" in src:
                self.session.audio_source = "mic"
            else:
                self.session.audio_source = "dual"
        if config.get("auto_gen") is not None:
            self.session.auto_generate = bool(config["auto_gen"])
        else:
            self.session.auto_generate = True

        # Parse resume document
        for doc_path in config.get("documents", []):
            if os.path.exists(doc_path):
                r_text = extract_text_from_file(doc_path)
                if r_text:
                    self.session.resume_file_path = doc_path
                    self.session.resume_text = r_text
                break

        self.session.invalidate_profile_cache()

        # 3. Transition window state to Copilot HUD
        self.current_page = "copilot"
        if self.window:
            try:
                self.window.resize(880, 160)
                self.window.on_top = True
                self._apply_stealth_protection()
            except Exception as e:
                print("Error resizing to copilot view:", e)

        # 4. Auto-start STT transcriber and pre-warm cloud STT/LLM connections in background
        try:
            from interview_engine import prewarm_pipeline_connections
            prewarm_pipeline_connections(self.session)
        except Exception:
            pass

        def _init_backend():
            try:
                from session_client import get_session_client
                sc = get_session_client()
                if sc and hasattr(sc, "start_session"):
                    sc.start_session(mode="copilot")
            except Exception as se:
                print(f"[Backend Session] Notice: {se}")

        threading.Thread(target=_init_backend, daemon=True).start()

        def _start_stt():
            time.sleep(0.2)
            if not self.transcriber.is_running:
                self.toggle_live_transcription()

        threading.Thread(target=_start_stt, daemon=True).start()
        return {"status": "ok"}

    def handle_end_session(self) -> dict:
        """Ends the active copilot session, terminates background STT, and returns window to home dashboard mode."""
        try:
            if self.transcriber and getattr(self.transcriber, "is_running", False):
                self.transcriber.stop()
        except Exception as e:
            print("Notice stopping transcriber on end session:", e)

        try:
            from session_client import get_session_client
            sc = get_session_client()
            if sc and hasattr(sc, "end_session"):
                v_session = getattr(self, "session", None)
                qa_pairs = getattr(v_session, "qa_pairs", []) or []
                transcript_log = getattr(v_session, "transcript_log", []) or []
                summary_md = getattr(v_session, "session_summary", None)
                if not summary_md and v_session:
                    try:
                        from interview_exporter import generate_session_markdown
                        summary_md = generate_session_markdown(v_session)
                    except Exception as ex:
                        print("Notice generating session summary markdown:", ex)
                
                job_title = getattr(v_session, "job_title", "Candidate") or "Candidate"
                company_name = getattr(v_session, "company_name", "Interview") or "Interview"
                sc.end_session(
                    qa_pairs=qa_pairs,
                    transcript_log=transcript_log,
                    summary_markdown=summary_md,
                    job_title=job_title,
                    company_name=company_name
                )
        except Exception as e:
            print("Notice ending backend session:", e)

        self.current_page = "home"
        self.is_card_expanded = False
        if self.window:
            try:
                self.window.resize(620, 750)
                self._apply_stealth_protection()
            except Exception as e:
                print("Notice resizing on end session:", e)

        print("[Session] Active copilot session ended cleanly. Navigating to Home.")
        return {"status": "ended"}

    def handle_export_session_report(self, markdown_content: str = "", filename: str = "") -> dict:
        """Exports session Q&A summary report as a markdown document."""
        try:
            if not markdown_content or not markdown_content.strip():
                try:
                    from interview_exporter import generate_session_markdown
                    if getattr(self, "session", None):
                        markdown_content = generate_session_markdown(self.session)
                except Exception as ex:
                    print("Notice generating export report fallback:", ex)

            if not filename:
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"blujay_interview_report_{ts}.md"

            docs_dir = os.path.expanduser("~/Documents")
            if not os.path.exists(docs_dir):
                docs_dir = os.path.dirname(os.path.abspath(__file__))
            export_path = os.path.join(docs_dir, filename)

            with open(export_path, "w", encoding="utf-8") as f:
                f.write(markdown_content or "# BluJay AI Interview Report\n\nNo Q&A recorded.")

            print(f"[Session Report] Saved to: {export_path}")
            self._eval_js(f"window.showToast?.('✓ Session report saved: {filename}');")
            return {"success": True, "path": export_path, "filename": filename}
        except Exception as e:
            print(f"Error saving session report: {e}")
            self._eval_js("window.showToast?.('⚠️ Failed to save report.');")
            return {"success": False, "error": str(e)}

    def end_session(self) -> dict:
        return self.handle_end_session()

    def export_session_report(self, markdown_content: str = "", filename: str = "") -> dict:
        return self.handle_export_session_report(markdown_content, filename)

    def handle_buy_credits(self, tier: str) -> dict:
        tier_map = {
            "100": "starter_pack",
            "starter": "starter_pack",
            "starter_pack": "starter_pack",
            "pro": "pro_pack",
            "pro_pack": "pro_pack",
            "power": "power_pack",
            "power_pack": "power_pack",
            "unlimited": "unlimited_monthly",
            "unlimited_monthly": "unlimited_monthly",
        }
        product_id = tier_map.get(str(tier).lower().strip(), "starter_pack")
        try:
            from session_client import get_session_client
            sc = get_session_client()
            success, msg = sc.mock_buy_product(product_id)
            if success:
                credits_label = "Unlimited" if sc.is_unlimited else f"{sc.credit_units / 100.0:.1f} Credits"
                self._eval_js(f"window.showToast?.('✓ Credits updated: {credits_label}');")
                return {
                    "success": True,
                    "message": msg,
                    "credit_units": sc.credit_units,
                    "is_unlimited": sc.is_unlimited,
                    "credits_label": credits_label
                }
            else:
                self._eval_js(f"window.showToast?.('Credit update notice: {msg}');")
                return {"success": False, "message": msg}
        except Exception as e:
            err_msg = f"Credit transaction error: {e}"
            self._eval_js(f"window.showToast?.('{err_msg}');")
            return {"success": False, "message": err_msg}

    def toggle_app_expansion(self) -> dict:
        """Toggles compact desktop widget / expanded web dashboard mode and resizes the window."""
        self.is_web_mode = not getattr(self, "is_web_mode", False)
        if self.window and getattr(self, "current_page", "home") == "home":
            try:
                if self.is_web_mode:
                    self.window.resize(1100, 780)
                else:
                    self.window.resize(620, 750)
            except Exception as e:
                print("Error resizing on toggle app expansion:", e)
        self._eval_js("if (typeof toggleAppExpansion === 'function') toggleAppExpansion();")
        return {"status": "toggled", "is_web_mode": self.is_web_mode}

    # =========================================================================
    # Audio & Transcription Handlers
    # =========================================================================

    def _is_duplicate_question(self, new_q: str, time_delta: float, window_sec: float = 1.8) -> bool:
        """
        Intelligent semantic duplicate detector:
        - If the exact or near-identical transcript arrives within window_sec, treat as duplicate.
        - If a genuinely new question arrives (different words or time_delta > window_sec), process immediately.
        """
        last_q = getattr(self, "_last_answered_question", "")
        if not last_q or time_delta > window_sec:
            return False
        
        nq = re.sub(r"[^\w\s]", "", new_q.lower()).strip()
        lq = re.sub(r"[^\w\s]", "", last_q.lower()).strip()
        
        if not nq or not lq:
            return False
        if nq == lq:
            return True
            
        nw = set(nq.split())
        lw = set(lq.split())
        if not nw or not lw:
            return False
            
        jaccard = len(nw.intersection(lw)) / len(nw.union(lw))
        return jaccard >= 0.92

    def _on_transcript_chunk(self, chunk: str, is_question: bool = False, source: str = "system"):
        if not chunk or not chunk.strip():
            return

        # 1. Update UI transcript log
        escaped_chunk = json.dumps(chunk)
        self._eval_js(f"window.onTranscriptChunk({escaped_chunk});")

        # 2. Process utterance through Question Understanding & Turn-Taking Layer
        try:
            from interview_engine import classify_and_understand_utterance
            # In Dual mode: 'candidate_mic' is authoritatively the candidate, 'system' is the interviewer.
            # In Mic-Only mode: 'mic' uses semantic turn-taking heuristics.
            # In System-Only mode: 'system' is loopback interviewer audio.
            if getattr(self.session, "audio_source", "dual") == "mic":
                eval_source = "mic"
            elif getattr(self.session, "audio_source", "dual") == "system":
                eval_source = "system"
            else:
                eval_source = source
            analysis = classify_and_understand_utterance(
                chunk,
                session=self.session,
                client=self.client,
                source=eval_source
            )
            
            speaker = analysis.get("speaker", "INTERVIEWER")
            role = "Candidate" if speaker == "CANDIDATE" else "Interviewer"
            is_q = bool(analysis.get("should_generate_answer", False))
            if hasattr(self.session, "add_transcript"):
                self.session.add_transcript(chunk, role=role, is_question=is_q, source=source)

            should_answer = analysis.get("should_generate_answer", False)
            hold_for_context = analysis.get("hold_for_context", False)

            # MEDIUM-CONFIDENCE FLUSH TIMER
            if hold_for_context and not should_answer:
                pmq = getattr(self.session, "pending_medium_question", None)
                if pmq and isinstance(pmq, dict):
                    captured_buf_id = pmq.get("buffer_id", "")
                    import threading as _threading
                    _threading.Timer(
                        0.8,
                        lambda _bid=captured_buf_id: self._flush_pending_medium_question(_bid)
                    ).start()
                return

            if not should_answer:
                return

            refined_q = analysis.get("refined_question") or chunk
            q_type = analysis.get("utterance_type", "technical")

            # 3. Intelligent duplicate detection (non-blocking for new questions)
            now = time.time()
            last_ts = getattr(self, "_last_answered_ts", 0.0)
            time_delta = now - last_ts

            if self._is_duplicate_question(refined_q, time_delta, window_sec=1.8):
                return

            if getattr(self.session, "auto_generate", True):
                self._last_answered_question = refined_q
                self._last_answered_ts = now
                self._handle_detected_question(refined_q, question_type=q_type)
            else:
                self.active_question = refined_q
                escaped_q = json.dumps({"question": refined_q, "type": q_type})
                self._eval_js(f"window.onQuestionDetected({escaped_q});")
        except Exception as e:
            print("Transcript question handler notice:", e)

    def _on_interim_transcript(self, raw_text: str, likelihood_info: Optional[dict] = None, source: str = "system"):
        if not raw_text or not raw_text.strip():
            return
            
        escaped = json.dumps(raw_text)
        self._eval_js(f"window.onTranscriptChunk({escaped});")

        if not likelihood_info or not isinstance(likelihood_info, dict):
            return

        # Candidate mic speech in dual mode must NEVER trigger speculative early generation
        if source in ("candidate_mic", "candidate_channel"):
            return

        # Early-Start Triggering: When interim text is a clear complete question, begin streaming before trailing pause
        can_start_early = likelihood_info.get("can_start_early", False)
        sem_key = likelihood_info.get("semantic_key", "")

        if can_start_early and getattr(self.session, "auto_generate", True) and sem_key:
            active_early_key = getattr(self, "_active_early_candidate_key", "")
            if sem_key != active_early_key:
                self._active_early_candidate_key = sem_key
                pred_type = likelihood_info.get("predicted_type", "technical")
                now = time.time()
                last_ts = getattr(self, "_last_answered_ts", 0.0)
                if not self._is_duplicate_question(raw_text, now - last_ts, window_sec=1.8):
                    self._last_answered_question = raw_text
                    self._last_answered_ts = now
                    self._handle_detected_question(raw_text, question_type=pred_type)

    def _on_turn_resumed(self, source: str = "system"):
        """Fired by Deepgram Flux TurnResumed/SpeechResumed when speaker pauses and resumes."""
        # Cancel any in-flight speculative generation immediately
        self._active_generation_id = uuid.uuid4().hex
        self.active_question = ""
        self.active_raw_answer = ""
        self._eval_js("if (window.onAnswerCancelled) window.onAnswerCancelled();")

    def _on_eager_eot(self, text: str, source: str = "system"):
        """Fired by Deepgram Flux EagerEndOfTurn for early speculative answer streaming."""
        if not text or not text.strip():
            return
        if source in ("candidate_mic", "candidate_channel"):
            return
        escaped_chunk = json.dumps(text)
        self._eval_js(f"window.onTranscriptChunk({escaped_chunk});")
        if not getattr(self.session, "auto_generate", True):
            return
        now = time.time()
        last_ts = getattr(self, "_last_answered_ts", 0.0)
        if not self._is_duplicate_question(text, now - last_ts, window_sec=1.8):
            self._last_answered_question = text
            self._last_answered_ts = now
            from classifier import classify_question
            q_type = classify_question(text)
            self._handle_detected_question(text, question_type=q_type)

    def _flush_pending_medium_question(self, expected_buf_id: str):
        """
        Deferred flush for a MEDIUM-confidence pending question (Web HUD).

        Called by threading.Timer ~1.0 s after the utterance entered the MEDIUM buffer.
        Race-safety contract (identical to the Desktop UI):
          - No-op if the buffer was already consumed by an incoming interviewer chunk.
          - No-op if a newer MEDIUM utterance replaced this one (different buffer_id).
          - Re-evaluates standalone when buffer_id still matches and buffer was not consumed.
        Candidate microphone speech never reaches this method.
        """
        try:
            pmq = getattr(self.session, "pending_medium_question", None)
            if not pmq or not isinstance(pmq, dict):
                return  # Already consumed.
            if pmq.get("buffer_id", "") != expected_buf_id:
                return  # Superseded by a newer MEDIUM utterance.

            pending_raw = pmq.get("raw_text", pmq.get("text", ""))
            self.session.pending_medium_question = None  # Consume before re-eval.

            if not pending_raw:
                return

            # Re-run through the full pipeline as a standalone interviewer utterance.
            self._on_transcript_chunk(pending_raw, source="system")
        except Exception as _e:
            print(f"Web-medium-flush notice: {_e}")
    def _on_audio_status(self, status: str):
        is_listening = "Listening" in status or bool(self.transcriber and self.transcriber.is_running)
        self._eval_js(f"window.onAudioStatusChange('{status}', {str(is_listening).lower()});")

    def _on_audio_level_update(self, source: str, level: float = 0.0, rms: float = 0.0):
        now = time.perf_counter()
        if now - getattr(self, "_last_audio_level_ts", 0) > 0.05:
            self._last_audio_level_ts = now
            lvl_pct = int(min(max(float(level) * 100.0, 0.0), 100.0))
            self._eval_js(f"if(window.onAudioLevelUpdate) window.onAudioLevelUpdate('{source}', {lvl_pct});")

    def toggle_live_transcription(self):
        if not self.transcriber.is_running:
            self.transcriber.start(
                source=self.session.audio_source,
                mic_device_id=getattr(self.session, "selected_mic_device", None),
                system_speaker_id=getattr(self.session, "selected_speaker_device", None)
            )
            self._eval_js("window.onAudioStatusChange('Listening', true);")
        else:
            self.transcriber.stop()
            self._eval_js("window.onAudioStatusChange('Paused', false);")

    def handle_copilot_ready(self):
        """Called immediately when index.html/app.js finishes loading in PyWebView."""
        self.current_page = "copilot"
        self._apply_stealth_protection()
        if not self.transcriber.is_running:
            self.toggle_live_transcription()
        else:
            self._eval_js("window.onAudioStatusChange('Listening', true);")
        return {"status": "ready", "audio_source": self.session.audio_source}

    def clear_transcript(self):
        self.session.clear_history()
        self.active_question = ""
        self.active_raw_answer = ""
        self.resize_window(False)

    # =========================================================================
    # AI Answer Generation & Actions
    # =========================================================================

    def _handle_detected_question(self, question: str, question_type: str = "technical"):
        self.active_question = question
        self.active_raw_answer = ""
        escaped_q = json.dumps({"question": question, "type": question_type})
        self._eval_js(f"window.onQuestionDetected({escaped_q});")
        self._generate_answer_stream(question, question_type=question_type)

    def manual_generate_answer(self):
        answered_questions = set()
        if hasattr(self.session, "qa_pairs") and self.session.qa_pairs:
            answered_questions = {qa.get("question", "").strip().lower() for qa in self.session.qa_pairs}

        question = ""
        # 1. Prioritize any new unanswered question from session
        if hasattr(self.session, "get_latest_interviewer_question"):
            cand_q = (self.session.get_latest_interviewer_question() or "").strip()
            if cand_q and cand_q.lower() not in answered_questions:
                question = cand_q

        # 2. Check active_question if not already answered
        if not question:
            act_q = getattr(self, "active_question", "").strip()
            if act_q and act_q.lower() not in answered_questions:
                question = act_q

        # 3. Fallback to recent unanswered speech in transcript log
        if not question and hasattr(self.session, "transcript_log"):
            for entry in reversed(self.session.transcript_log):
                t_text = entry.get("text", "").strip()
                if t_text and (t_text.lower() not in answered_questions):
                    question = t_text
                    break

        # 4. If nothing new, fallback to active_question or last qa question
        if not question:
            question = getattr(self, "active_question", "") or (self.session.qa_pairs[-1]["question"] if (hasattr(self.session, "qa_pairs") and self.session.qa_pairs) else "")

        if not question or not question.strip():
            self._eval_js("window.showToast?.('🎙️ Listening for speech... Speak into your mic or type a question.');")
            return
        from classifier import classify_question
        q_type = classify_question(question)
        self._handle_detected_question(question, question_type=q_type)

    def submit_manual_question(self, text: str):
        if not text:
            return
        from interview_engine import classify_and_understand_utterance
        analysis = classify_and_understand_utterance(
            text,
            session=self.session,
            client=self.client,
            source="manual",
            is_manual=True
        )
        refined_q = analysis.get("refined_question") or text
        q_type = (analysis.get("utterance_type") or "technical").lower()
        if hasattr(self.session, "add_transcript"):
            self.session.add_transcript(text, role="Interviewer", is_question=True, source="MANUAL_INPUT", is_manual=True)
        self._handle_detected_question(refined_q, question_type=q_type)

    def _generate_answer_stream(self, question: str, question_type: str = "technical"):
        """Streams Groq response directly to the Web HUD with instant cancellation of stale tasks."""
        gen_id = uuid.uuid4().hex
        self._active_generation_id = gen_id
        self.active_question = question
        self._eval_js("window.onAnswerStart();")

        generation_completed = False

        def _is_cancelled() -> bool:
            return getattr(self, "_active_generation_id", None) != gen_id

        def _watchdog():
            time.sleep(20.0)
            if not generation_completed and getattr(self, "_active_generation_id", "") == gen_id:
                if hasattr(self.session, "controller") and self.session.controller:
                    try:
                        self.session.controller.check_generation_timeout()
                    except Exception:
                        pass
                self._eval_js("window.onAnswerComplete?.({full_text: '⚠️ Generation timed out. Listening for next question...'});")

        threading.Thread(target=_watchdog, daemon=True).start()

        def _worker():
            nonlocal generation_completed
            try:
                # Ensure client is initialized
                if self.client is None:
                    api_k = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
                    if api_k:
                        from groq import Groq
                        self.client = Groq(api_key=api_k, timeout=12.0)

                def _token_cb(tok: str):
                    if not _is_cancelled():
                        escaped_tok = json.dumps({"token": tok})
                        self._eval_js(f"window.onAnswerToken({escaped_tok});")

                full_ans = generate_interview_answer(
                    client=self.client,
                    session=self.session,
                    question=question,
                    on_token=_token_cb,
                    question_type=question_type,
                    is_cancelled_callback=_is_cancelled
                )

                if _is_cancelled():
                    return
                
                # If cloud generation was empty, use grounded offline heuristic synthesis
                if not full_ans or not full_ans.strip():
                    from offline_fallback_manager import generate_offline_heuristic_answer
                    full_ans = generate_offline_heuristic_answer(
                        question=question,
                        job_title=self.session.job_title or "Software Engineer",
                        company_name=self.session.company_name or "",
                        resume_text=self.session.resume_text or "",
                        on_token=_token_cb
                    )

                if not _is_cancelled():
                    generation_completed = True
                    self.active_raw_answer = full_ans
                    self.session.record_qa(question, full_ans, category=question_type)
                    from interview_engine import InterviewTurnState
                    self.session.turn_state = InterviewTurnState.LISTENING
                    self.active_question = ""
                    if hasattr(self.session, "controller") and self.session.controller:
                        try:
                            self.session.controller.end_generation(getattr(self.session, "active_generation_id", 0), success=True)
                        except Exception:
                            pass
                    escaped_ans = json.dumps({"full_text": full_ans})
                    self._eval_js(f"window.onAnswerComplete({escaped_ans});")
            except Exception as e:
                print("Error in answer stream:", e)
                if not _is_cancelled():
                    try:
                        from offline_fallback_manager import generate_offline_heuristic_answer
                        fallback_ans = generate_offline_heuristic_answer(
                            question=question,
                            job_title=self.session.job_title or "Software Engineer",
                            company_name=self.session.company_name or "",
                            resume_text=self.session.resume_text or "",
                            on_token=lambda t: None
                        )
                        generation_completed = True
                        self.active_raw_answer = fallback_ans
                        from interview_engine import InterviewTurnState
                        self.session.turn_state = InterviewTurnState.LISTENING
                        self.active_question = ""
                        if hasattr(self.session, "controller") and self.session.controller:
                            try:
                                self.session.controller.end_generation(getattr(self.session, "active_generation_id", 0), success=True)
                            except Exception:
                                pass
                        escaped_ans = json.dumps({"full_text": fallback_ans})
                        self._eval_js(f"window.onAnswerComplete({escaped_ans});")
                    except Exception as fe:
                        print("Fallback generation error:", fe)

        threading.Thread(target=_worker, daemon=True).start()

    def refine_answer(self, mode: str, prompt_instruction: str):
        if not self.active_question:
            self.active_question = "Explain your technical background."
        refined_q = f"{self.active_question}\n\n[Refinement Instruction: {prompt_instruction}]"
        self._handle_detected_question(refined_q)

    def trigger_code_screen_solver(self, image_path: Optional[str] = None):
        """Captures screen code (or uses snipped image) and generates solution with real-time streaming."""
        self.is_card_expanded = True
        q_label = "✂️ Snipped Region Coding Analysis" if image_path else "🖥️ Screen Coding Problem Analysis"
        escaped_q = json.dumps({"question": q_label, "type": "CODING"})
        self._eval_js(f"window.onQuestionDetected({escaped_q});")
        self._eval_js("window.onAnswerStart({});")

        def _worker():
            try:
                mon_id = getattr(self.session, "selected_monitor_id", 0)

                def _on_token_stream(tok: str):
                    escaped_tok = json.dumps({"token": tok})
                    self._eval_js(f"window.onAnswerToken({escaped_tok});")

                res = solve_screen_coding_problem(
                    client=self.client,
                    on_token=_on_token_stream,
                    monitor_id=mon_id,
                    image_path=image_path
                )
                if isinstance(res, dict):
                    q = res.get("question", q_label)
                    ans = res.get("solution", "")
                else:
                    q = q_label
                    ans = str(res) if res else "No solution generated from screen."
                
                # Extract clean problem title from solution if available
                prob_title = None
                title_m = re.search(r"(?:Problem Summary|Problem Statement|Problem)[\*\:\s\-]+([^\n\r]+)", ans)
                if title_m:
                    cand_title = title_m.group(1).strip().strip("*: -#")
                    if cand_title and len(cand_title) < 80:
                        prob_title = cand_title
                if not prob_title:
                    fn_m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", ans)
                    if fn_m:
                        prob_title = f"Problem ({fn_m.group(1)})"
                if prob_title and q == q_label:
                    q = f"{q_label}: {prob_title}"

                # Update copilot and session state so follow-up turns are fully grounded in the solved code
                self.active_raw_answer = ans
                self.active_question = q
                is_valid_solution = bool(ans and not ans.startswith("⚠️") and "No solution" not in ans)
                if hasattr(self.session, "record_qa") and is_valid_solution:
                    self.session.record_qa(q, ans, category="coding")
                if is_valid_solution and hasattr(self.session, "active_topic"):
                    self.session.active_topic = "Coding"
                from interview_engine import InterviewTurnState
                self.session.turn_state = InterviewTurnState.LISTENING

                escaped_ans = json.dumps({"full_text": ans})
                self._eval_js(f"window.onAnswerComplete({escaped_ans});")
            except Exception as e:
                print(f"Notice in screen coding solver: {e}")
                escaped_err = json.dumps({"full_text": f"⚠️ Screen coding solver notice: {e}"})
                self._eval_js(f"window.onAnswerComplete({escaped_err});")

        threading.Thread(target=_worker, daemon=True).start()

    def trigger_region_snip_solver(self):
        """Launches screen snipper for targeted visual OCR."""
        def _launch():
            try:
                root = tk.Tk()
                root.withdraw()
                InteractiveRegionSnipper(root, on_snip_completed=self._on_snip_completed)
                root.mainloop()
            except Exception as e:
                print("Region snip notice:", e)
                self.trigger_code_screen_solver()

        threading.Thread(target=_launch, daemon=True).start()

    def _on_snip_completed(self, image_path: str):
        if image_path and os.path.exists(image_path):
            self.trigger_code_screen_solver(image_path=image_path)

    def trigger_system_design(self):
        topic = self.active_question or "High-Throughput Distributed Cache Architecture"
        def _worker():
            try:
                diag = generate_system_design_diagram(topic)
                self._handle_detected_question(f"System Design: {topic}")
            except Exception:
                self.refine_answer("System Design", "Structure as high-scale system design architecture.")
        threading.Thread(target=_worker, daemon=True).start()

    def trigger_star_story(self):
        story = find_best_star_story(self.active_question or "Overcoming technical adversity", self.session)
        formatted = format_star_spoken_answer(story)
        escaped_q = json.dumps({"question": self.active_question or "Behavioral STAR Story", "type": "STAR"})
        self._eval_js(f"window.onQuestionDetected({escaped_q});")
        escaped_ans = json.dumps({"full_text": formatted})
        self._eval_js(f"window.onAnswerComplete({escaped_ans});")

    def trigger_run_code(self):
        try:
            from code_sandbox import extract_code_from_markdown, execute_code_snippet
            current_ans = self.active_answer or ""
            snippets = extract_code_from_markdown(current_ans)
            if snippets:
                first = snippets[0]
                res = execute_code_snippet(first["code"], language=first["language"], timeout_sec=4.0)
                status_icon = "✅" if res["success"] else "❌"
                output_str = res["stdout"] if res["success"] else (res["stderr"] or "No output")
                result_md = f"### {status_icon} Sandbox Execution Result ({first['language']})\n```\n{output_str}\n```\n*Duration: {res.get('duration_sec', 0)}s*"
                escaped_q = json.dumps("▶️ Run Code Sandbox")
                escaped_ans = json.dumps(result_md)
                self._eval_js(f"window.onQuestionDetected({escaped_q});")
                self._eval_js(f"window.onAnswerComplete({escaped_ans});")
                return
        except Exception as e:
            print("Run code notice:", e)
        self.refine_answer("Run Code", "Execute and verify python code.")

    def toggle_teleprompter(self):
        self.is_teleprompter_active = not self.is_teleprompter_active
        self._eval_js(f"window.onTeleprompterStatus({str(self.is_teleprompter_active).lower()}, {self.teleprompter_speed});")
        return self.is_teleprompter_active

    def trigger_mock_interview(self):
        try:
            from mock_interviewer import MockInterviewSession
            role = "backend"
            if self.session and self.session.job_title:
                jt = self.session.job_title.lower()
                if "front" in jt:
                    role = "frontend"
                elif "full" in jt:
                    role = "fullstack"
                elif "system" in jt or "architect" in jt:
                    role = "system_design"
                elif "ml" in jt or "ai" in jt or "data" in jt:
                    role = "ml_ai"
                elif "manager" in jt or "lead" in jt:
                    role = "engineering_manager"
                elif "behavioral" in jt or "hr" in jt:
                    role = "behavioral"
            self.mock_session = MockInterviewSession(
                role=role,
                resume_text=getattr(self.session, "resume_text", "")
            )
            first_q = self.mock_session.generate_first_question()
            q_text = first_q.get("interviewer_text", first_q.get("question_text", ""))
            escaped_q = json.dumps(f"🎙️ Mock Interview ({self.mock_session.role.replace('_', ' ').title()})")
            escaped_ans = json.dumps(q_text)
            self._eval_js(f"window.onQuestionDetected({escaped_q});")
            self._eval_js(f"window.onAnswerComplete({escaped_ans});")
        except Exception as e:
            print("Mock interview trigger notice:", e)

    def export_session_report(self):
        try:
            out_path = f"interview_report_{int(time.time())}.md"
            export_session_to_markdown(self.session, out_path)
            print(f"Exported session report to {out_path}")
        except Exception as e:
            print("Export notice:", e)

    def minimize_window(self):
        """Minimizes the OS window to the taskbar."""
        if self.window:
            try:
                self.window.minimize()
            except Exception as e:
                print("Error minimizing window:", e)

    def minimize_to_bubble(self):
        if getattr(self, "current_page", "home") == "copilot":
            self._eval_js("if (typeof window.minimizeToBubble === 'function') window.minimizeToBubble();")
        else:
            self.minimize_window()

    def toggle_overlay_visibility(self):
        if getattr(self, "current_page", "home") == "copilot":
            if getattr(self, "is_bubble_mode", False):
                self._eval_js("if (typeof window.restoreFromBubble === 'function') window.restoreFromBubble();")
            else:
                self._eval_js("if (typeof window.minimizeToBubble === 'function') window.minimizeToBubble();")
        else:
            self.minimize_window()

    def toggle_ghost_mode(self):
        """Toggles ghost click-through mode using Win32 WS_EX_TRANSPARENT."""
        self.is_ghost_mode = not self.is_ghost_mode
        hwnds = self._get_overlay_hwnds()
        for h in hwnds:
            set_click_through(h, self.is_ghost_mode)
        status = "ON (Clicks pass to IDE)" if self.is_ghost_mode else "OFF (Normal mode)"
        self._eval_js(f"window.showToast?.('👻 Ghost Click-Through: {status}');")
        print(f"[Stealth] Ghost Mode: {status}")

    def dock_to_camera(self):
        """Docks the HUD directly under the primary camera (top-center) for natural eye contact."""
        if not self.window or self.current_page != "copilot":
            return
        try:
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            target_w = 880
            target_h = 560 if self.is_card_expanded else 145
            target_x = max(0, (screen_w - target_w) // 2)
            target_y = 10
            self.window.move(target_x, target_y)
            self.window.resize(target_w, target_h)
            self._apply_stealth_protection()
            self._eval_js("window.showToast?.('📷 Docked under webcam for eye contact');")
            print("[Window] HUD docked under webcam.")
        except Exception as e:
            print("Error docking to camera:", e)

    def set_bubble_mode(self, is_bubble: bool):
        self.is_bubble_mode = is_bubble
        if self.window:
            try:
                if is_bubble:
                    self.window.resize(50, 50)
                else:
                    target_w = 880 if self.current_page == "copilot" else (1100 if getattr(self, "is_web_mode", False) else 620)
                    target_h = (560 if self.is_card_expanded else 160) if self.current_page == "copilot" else 750
                    self.window.resize(target_w, target_h)
                self._apply_stealth_protection()
            except Exception:
                pass

    def update_audio_source(self, source: str):
        """Switches audio capture between 'dual' (system loopback + mic), 'system' (speaker loopback only), and 'mic' (microphone only)."""
        valid_source = source if source in ("dual", "both", "system", "loopback", "mic", "microphone") else "dual"
        self.session.audio_source = "dual" if valid_source in ("dual", "both") else ("system" if valid_source in ("system", "loopback") else "mic")
        if self.transcriber and self.transcriber.is_running:
            self.transcriber.stop()
            time.sleep(0.2)
            self.transcriber.start(
                source=self.session.audio_source,
                mic_device_id=getattr(self.session, "selected_mic_device", None),
                system_speaker_id=getattr(self.session, "selected_speaker_device", None)
            )
        print(f"[Audio Source] Switched live audio capture mode to: '{self.session.audio_source}'")

    def end_session(self):
        """Ends the active session and navigates back to Home Dashboard."""
        def _persist():
            try:
                if self.transcriber:
                    self.transcriber.stop()
                session_client.end_session(
                    qa_pairs=self.session.qa_pairs,
                    transcript_log=self.session.transcript_log,
                    summary_markdown=getattr(self.session, "session_summary", None),
                    job_title=self.session.job_title,
                    company_name=self.session.company_name
                )
            except Exception as se:
                print("Error persisting session:", se)

        threading.Thread(target=_persist, daemon=True).start()

        # Navigate back to Home
        self.current_page = "home"
        if self.window:
            try:
                self.window.resize(620, 750)
                self.window.on_top = False
                self._apply_stealth_protection()
            except Exception:
                pass

    def copy_to_clipboard(self, text: str):
        if not text:
            return
        if sys.platform == "win32":
            try:
                import subprocess
                subprocess.run(["clip"], input=text.encode("utf-16le"), check=True, creationflags=0x08000000)
                return
            except Exception:
                pass
            try:
                r = tk.Tk()
                r.withdraw()
                r.clipboard_clear()
                r.clipboard_append(text)
                r.update()
                r.destroy()
            except Exception:
                pass

    def update_settings(self, config: dict):
        if config.get("company"):
            self.session.company_name = config["company"]
        if config.get("job_description"):
            self.session.job_description = config["job_description"]
        if config.get("instructions"):
            self.session.custom_instructions = config["instructions"]
        if config.get("model"):
            self.session.model_name = config["model"]
        self.session.invalidate_profile_cache()

    # =========================================================================
    # Window Geometry & Controls
    # =========================================================================

    def resize_window(self, expanded: bool):
        self.is_card_expanded = expanded
        if self.window and not getattr(self, "is_bubble_mode", False):
            h = 750 if expanded else 160
            curr_w = getattr(self.window, "width", 620) or 620
            try:
                self.window.resize(curr_w, h)
                self._apply_stealth_protection()
            except Exception:
                pass

    def close_window(self):
        """Cleanly and asynchronously closes the window and terminates all background audio/threads."""
        self._is_closing = True
        self._closed = True

        def _do_close():
            try:
                if getattr(self, "transcriber", None):
                    self.transcriber.stop()
            except Exception:
                pass
            try:
                if getattr(self, "hotkey_manager", None):
                    self.hotkey_manager.stop()
            except Exception:
                pass
            try:
                if getattr(self, "on_close_callback", None):
                    self.on_close_callback()
            except Exception:
                pass
            try:
                if self.window:
                    self.window.destroy()
            except Exception:
                pass
            time.sleep(0.1)
            os._exit(0)

        threading.Thread(target=_do_close, daemon=True, name="WindowCloseThread").start()
        return {"status": "closing"}
