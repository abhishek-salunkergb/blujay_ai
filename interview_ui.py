"""
Parakeet AI Real-Time Interview Copilot Interface
=================================================
Ultra-modern, glassmorphism dual-pane interview copilot with live streaming
transcript, real-time question detection, one-click answer refinement,
Spacebar quick-trigger, coding problem solver, and post-session summaries.
"""

import os
import re
import time
import collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
from typing import Callable, Optional, List, Dict, Any, Tuple

from interview_engine import (
    InterviewSession,
    ContinuousTranscriber,
    generate_interview_answer,
    generate_session_summary,
    is_likely_question,
    refine_interview_question,
    clean_asr_transcript,
    analyze_conversation_flow,
    classify_and_understand_utterance,
    SessionController,
    ComponentHealthState,
    categorize_api_error
)
from vision import (
    solve_screen_coding_problem,
    get_available_monitors,
    get_saved_monitor_preference,
    save_monitor_preference
)
from screen_snipper import InteractiveRegionSnipper
from system_design_generator import generate_system_design_diagram
from teleprompter import TeleprompterController
from interview_exporter import (
    export_session_to_markdown,
    export_session_to_json,
    export_session_to_text_report
)
from star_story_manager import (
    find_best_star_story,
    format_star_spoken_answer,
    extract_star_stories_heuristic,
    extract_star_stories_llm,
    load_star_stories_from_disk
)
from code_sandbox import (
    extract_code_from_markdown,
    execute_code_snippet
)
from hotkey_manager import (
    GlobalHotkeyManager,
    get_saved_hotkey_preferences,
    save_hotkey_preferences
)
from mock_interviewer import (
    MockInterviewSession,
    ROLE_TRACKS,
    DIFFICULTY_LEVELS,
    PERSONAS
)


class InterviewCopilotView:
    """Parakeet AI Real-time Interview Copilot View component."""
    
    def __init__(self, parent_frame: tk.Frame, root: tk.Tk, client, main_app_ref):
        self.container = parent_frame
        self.root = root
        self.client = client
        self.main_app = main_app_ref
        
        # Session State & Audio Engine
        self.session = InterviewSession()
        self.session_controller = getattr(self.session, "controller", None) or SessionController(self.session)
        self.transcriber = ContinuousTranscriber(
            client=self.client,
            on_transcript_chunk=self._on_transcript_chunk,
            on_status_change=self._on_audio_status,
            on_interim_transcript=self._on_interim_transcript,
            on_audio_level=self._on_audio_level_update,
            session=self.session
        )
        self._interim_active = False
        
        self.is_streaming_answer = False
        self.active_question = ""
        self.active_raw_question = ""
        self.active_early_candidate_key = ""
        self.active_early_generation_id = 0
        
        # Deduplication & Question Intelligence
        self.recent_questions_cache = collections.deque(maxlen=30)
        self.last_question_time = 0.0
        
        # Theme Palette (Modern, Minimal, High-Contrast Dark Canvas)
        self.PALETTE = {
            "bg": "#090d16",
            "panel_bg": "#0f172a",
            "panel_border": "#1e293b",
            "panel_divider": "#1e293b",
            "card_bg": "#0f172a",
            "card_border": "#1e293b",
            "card_border_glow": "#2563eb",
            "text": "#f8fafc",
            "text_secondary": "#cbd5e1",
            "text_muted": "#94a3b8",
            "text_subtle": "#64748b",
            "accent_primary": "#2563eb",
            "accent_cyan": "#38bdf8",
            "accent_green": "#10b981",
            "accent_purple": "#a855f7",
            "accent_amber": "#f59e0b",
            "accent_rose": "#ef4444",
            "btn_bg": "#1e293b",
            "btn_hover": "#334155",
            "tag_bg": "#1e293b"
        }
        
        self.show_conversation_history = False
        self._is_overlay_hidden = False
        self._pulse_idx = 0
        self._pulse_colors = ["#34d399", "#10b981", "#059669", "#10b981"]
        self._build_ui()
        self.teleprompter = TeleprompterController(self.answer_box, self.root, on_status_update=self._on_teleprompter_status)
        self._bind_hotkeys()
        self.hotkey_manager = GlobalHotkeyManager()
        self._init_global_hotkeys()
        self._start_pulse_animation()

    def _bind_hover_fx(self, widget, normal_bg, hover_bg, normal_fg=None, hover_fg=None):
        """Binds smooth visual hover state animation to buttons and chips."""
        if not widget:
            return
        def _on_enter(e):
            try:
                if widget.winfo_exists():
                    widget.config(bg=hover_bg)
                    if hover_fg:
                        widget.config(fg=hover_fg)
            except Exception:
                pass
        def _on_leave(e):
            try:
                if widget.winfo_exists():
                    widget.config(bg=normal_bg)
                    if normal_fg:
                        widget.config(fg=normal_fg)
            except Exception:
                pass
        widget.bind("<Enter>", _on_enter, add="+")
        widget.bind("<Leave>", _on_leave, add="+")

    def _start_pulse_animation(self):
        """Smooth breathing glow animation on active listening badge."""
        if hasattr(self, "audio_status_badge") and self._is_alive(self.audio_status_badge):
            try:
                self._pulse_idx = (self._pulse_idx + 1) % len(self._pulse_colors)
                color = self._pulse_colors[self._pulse_idx]
                if "Listening" in self.audio_status_badge.cget("text"):
                    self.audio_status_badge.config(fg=color)
            except Exception:
                pass
        self.safe_after(450, self._start_pulse_animation)

    def _is_alive(self, widget=None) -> bool:
        """Checks if the Tkinter root and widget are alive and have not been destroyed."""
        try:
            if not hasattr(self, "root") or not self.root:
                return False
            if not self.root.winfo_exists():
                return False
            if widget is not None:
                if not hasattr(widget, "winfo_exists") or not widget.winfo_exists():
                    return False
            return True
        except Exception:
            return False

    def safe_after(self, delay_ms: int, callback: Callable):
        """Dispatches callback on Tkinter main thread safely if window/widgets are still alive."""
        if not self._is_alive():
            return
        def _safe_wrapper():
            try:
                if self._is_alive():
                    callback()
            except (tk.TclError, RuntimeError, Exception):
                pass
        try:
            self.root.after(delay_ms, _safe_wrapper)
        except (tk.TclError, RuntimeError, Exception):
            pass
        
    def _build_ui(self):
        """Constructs the authentic Parakeet AI interface matching live interview video calls."""
        self.main_frame = tk.Frame(self.container, bg="#18191d", highlightthickness=0)
        self.main_frame.pack(fill="both", expand=True)

        self._history_items = []
        self._history_idx = -1
        self._is_card_expanded = False
        self._session_start_time = time.time()
        self._drag_x = 0
        self._drag_y = 0

        # 1. Top Capsule Stealth Bar
        self._build_header()

        # 2. Rolling Speech Subtitle & Quick Action Strip
        self._build_action_strip()

        # 3. Dynamic Auto-Expanding Frosted Glass Answer Card
        self._build_answer_card()

        # Start with compact window (collapsed card)
        self._resize_overlay(expanded=False)
        self._start_session_timer()
        self.show_conversation_history = True

    def _start_window_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_window_drag(self, event):
        try:
            x = self.root.winfo_x() + (event.x - self._drag_x)
            y = self.root.winfo_y() + (event.y - self._drag_y)
            self.root.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _resize_overlay(self, expanded: bool):
        try:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            w = min(960, max(880, int(screen_w * 0.72)))
            h = min(640, max(540, int(screen_h * 0.65))) if expanded else 124
            x = max(0, int((screen_w - w) / 2))
            y = 12
            self.root.minsize(400, 30)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def show_answer_card(self):
        """Expands the floating frosted glass answer card smoothly."""
        if not self._is_card_expanded:
            self._is_card_expanded = True
            if hasattr(self, "answer_card_frame") and self.answer_card_frame:
                self.answer_card_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))
            self._resize_overlay(expanded=True)

    def hide_answer_card(self):
        """Collapses the answer card back to the compact top stealth bar."""
        self._is_card_expanded = False
        if hasattr(self, "answer_card_frame") and self.answer_card_frame:
            self.answer_card_frame.pack_forget()
        self._resize_overlay(expanded=False)

    def _start_session_timer(self):
        """Ticks session timer every second with active state indicator."""
        def _tick():
            try:
                if self._is_alive():
                    elapsed = int(time.time() - self._session_start_time)
                    mins = max(1, elapsed // 60)
                    is_running = hasattr(self, "transcriber") and getattr(self.transcriber, "is_running", False)
                    status_text = "Active" if is_running else "Idle"
                    if hasattr(self, "timer_badge") and self.timer_badge:
                        self.timer_badge.config(text=f"🟠 {mins} min ({status_text})")
                    self.safe_after(1000, _tick)
            except Exception:
                pass
        self.safe_after(1000, _tick)

    def _build_header(self):
        """Top Capsule Header matching authentic Parakeet AI."""
        header = tk.Frame(self.main_frame, bg="#1e1f24", height=36, padx=8, pady=4)
        header.pack(side="top", fill="x")

        # Allow dragging window from top bar
        header.bind("<ButtonPress-1>", self._start_window_drag)
        header.bind("<B1-Motion>", self._do_window_drag)

        # Left Box: Logo + Hide + More
        left_box = tk.Frame(header, bg="#1e1f24")
        left_box.pack(side="left", fill="y")
        left_box.bind("<ButtonPress-1>", self._start_window_drag)
        left_box.bind("<B1-Motion>", self._do_window_drag)

        self.brand_lbl = tk.Label(
            left_box,
            text="🦜 ParakeetAI",
            bg="#1e1f24",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        self.brand_lbl.pack(side="left", padx=(2, 6))
        self.brand_lbl.bind("<ButtonPress-1>", self._start_window_drag)
        self.brand_lbl.bind("<B1-Motion>", self._do_window_drag)

        self.hide_btn = tk.Button(
            left_box,
            text="Hide",
            command=self._minimize_window,
            bg="#2b2d35",
            fg="#ffffff",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=9,
            pady=2,
            cursor="hand2"
        )
        self.hide_btn.pack(side="left", padx=(0, 5))
        self._bind_hover_fx(self.hide_btn, "#2b2d35", "#383b45", "#ffffff", "#ffffff")

        self.hdr_more_btn = tk.Button(
            left_box,
            text="···",
            command=self._show_header_more_menu,
            bg="#2b2d35",
            fg="#cbd5e1",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
            padx=7,
            pady=2,
            cursor="hand2"
        )
        self.hdr_more_btn.pack(side="left", padx=(0, 6))
        self._bind_hover_fx(self.hdr_more_btn, "#2b2d35", "#383b45", "#cbd5e1", "#ffffff")

        # Center: Live Timer
        center_box = tk.Frame(header, bg="#1e1f24")
        center_box.pack(side="left", expand=True)
        center_box.bind("<ButtonPress-1>", self._start_window_drag)
        center_box.bind("<B1-Motion>", self._do_window_drag)

        self.timer_badge = tk.Label(
            center_box,
            text="🟠 1 min (Active)",
            bg="#1e1f24",
            fg="#e2e8f0",
            font=("Segoe UI", 8)
        )
        self.timer_badge.pack(side="left", padx=4)
        self.timer_badge.bind("<ButtonPress-1>", self._start_window_drag)
        self.timer_badge.bind("<B1-Motion>", self._do_window_drag)

        # Right Box: Stealth Icon + Start/Stop + Drawer + Clear + Close
        right_box = tk.Frame(header, bg="#1e1f24")
        right_box.pack(side="right", fill="y")
        right_box.bind("<ButtonPress-1>", self._start_window_drag)
        right_box.bind("<B1-Motion>", self._do_window_drag)

        self.stealth_indicator = tk.Label(
            right_box,
            text="🖥️ ⏺",
            bg="#2b2d35",
            fg="#e2e8f0",
            font=("Segoe UI", 8),
            padx=6,
            pady=2
        )
        self.stealth_indicator.pack(side="left", padx=(0, 5))
        self.live_indicator = self.stealth_indicator
        self.recording_dot = self.stealth_indicator

        self.audio_toggle_btn = tk.Button(
            right_box,
            text="Start ⚡",
            command=self.toggle_live_transcription,
            bg="#2b2d35",
            fg="#34d399",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=9,
            pady=2,
            cursor="hand2"
        )
        self.audio_toggle_btn.pack(side="left", padx=(0, 5))
        self._bind_hover_fx(self.audio_toggle_btn, "#2b2d35", "#383b45", "#34d399", "#ffffff")

        self.drawer_btn = tk.Button(
            right_box,
            text="≡",
            command=self._show_header_more_menu,
            bg="#2b2d35",
            fg="#cbd5e1",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
            padx=7,
            pady=1,
            cursor="hand2"
        )
        self.drawer_btn.pack(side="left", padx=(0, 4))
        self._bind_hover_fx(self.drawer_btn, "#2b2d35", "#383b45", "#cbd5e1", "#ffffff")

        self.clear_btn = tk.Button(
            right_box,
            text="Clear",
            command=self.clear_transcript,
            bg="#2b2d35",
            fg="#cbd5e1",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=7,
            pady=2,
            cursor="hand2"
        )
        self.clear_btn.pack(side="left", padx=(0, 5))
        self._bind_hover_fx(self.clear_btn, "#2b2d35", "#383b45", "#cbd5e1", "#ffffff")

        self.close_btn = tk.Button(
            right_box,
            text="✕",
            command=self._close_window,
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            cursor="hand2"
        )
        self.close_btn.pack(side="left")
        self._bind_hover_fx(self.close_btn, "#dc2626", "#b91c1c", "#ffffff", "#ffffff")

        # Compatibility handles
        self.min_btn = self.hide_btn
        self.header_more_btn = self.hdr_more_btn
        self.history_toggle_btn = self.hdr_more_btn
        self.copy_btn = tk.Button(header, text="📋 Copy", command=self.copy_answer_to_clipboard)
        self.settings_btn = tk.Button(header, text="⚙️", command=self.open_settings_modal)
        self.audio_status_badge = self.timer_badge
        self.role_badge = tk.Label(header, text="Tech Interview")
        self.resume_badge = tk.Button(header, text="Resume ✓", command=self.open_settings_modal)
        self.spk_vu_label = tk.Label(header, text="🔊 0%")
        self.mic_vu_label = tk.Label(header, text="🎙️ 0%")
        self.opacity_btn = tk.Button(header, text="👁️ 100%", command=self.cycle_opacity)
        self.shortcuts_hdr_btn = tk.Button(header, text="⌨", command=lambda: self.main_app.show_shortcuts_modal() if hasattr(self, "main_app") and self.main_app else None)
        self.end_session_btn = tk.Button(header, text="🏁 End", command=self.show_session_summary_modal)
        self.export_btn = tk.Button(header, text="📄 Export", command=self.open_export_dialog)
        self.mock_interview_btn = tk.Button(header, text="🎓 Mock", command=self.open_mock_interview_modal)
        self.source_btn = tk.Button(header, text="🔊 Source", command=self.toggle_audio_source)
        self.auto_mode_btn = tk.Button(header, text="⚡ Auto-Gen", command=self.toggle_auto_generate_mode)

    def _build_action_strip(self):
        """Action strip containing rolling subtitle line and quick actions: AI Answer, Analyse Screen, Manual message."""
        strip = tk.Frame(self.main_frame, bg="#18191d", padx=8, pady=2)
        strip.pack(side="top", fill="x", pady=(2, 4))

        # 1. Rolling subtitle line
        sub_frame = tk.Frame(strip, bg="#18191d")
        sub_frame.pack(fill="x", pady=(0, 2))

        self.subtitle_label = tk.Label(
            sub_frame,
            text="Listening for speech...",
            bg="#18191d",
            fg="#cbd5e1",
            font=("Segoe UI", 9),
            anchor="w",
            justify="left"
        )
        self.subtitle_label.pack(side="left", fill="x", expand=True, padx=4)

        # 2. Action row
        action_row = tk.Frame(strip, bg="#18191d")
        action_row.pack(fill="x")

        self.ai_answer_btn = tk.Button(
            action_row,
            text="✦ AI Answer",
            command=self.manual_generate_answer,
            bg="#262830",
            fg="#ffffff",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=11,
            pady=3,
            cursor="hand2"
        )
        self.ai_answer_btn.pack(side="left", padx=(0, 5))
        self._bind_hover_fx(self.ai_answer_btn, "#262830", "#383b45", "#ffffff", "#ffffff")

        self.code_solve_btn = tk.Button(
            action_row,
            text="Analyse Screen",
            command=self.trigger_code_screen_solver,
            bg="#262830",
            fg="#ffffff",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=11,
            pady=3,
            cursor="hand2"
        )
        self.code_solve_btn.pack(side="left", padx=(0, 5))
        self._bind_hover_fx(self.code_solve_btn, "#262830", "#383b45", "#ffffff", "#ffffff")

        self.snip_solve_btn = tk.Button(
            action_row,
            text="✂️",
            command=self.trigger_region_snip_solver,
            bg="#262830",
            fg="#cbd5e1",
            activebackground="#383b45",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=3,
            cursor="hand2"
        )
        self.snip_solve_btn.pack(side="left", padx=(0, 5))
        self._bind_hover_fx(self.snip_solve_btn, "#262830", "#383b45", "#cbd5e1", "#ffffff")

        # Type message container
        input_container = tk.Frame(action_row, bg="#141518", highlightbackground="#2b2d35", highlightthickness=1)
        input_container.pack(side="left", fill="both", expand=True)

        self.type_input = tk.Entry(
            input_container,
            bg="#141518",
            fg="#f8fafc",
            insertbackground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 9)
        )
        self.type_input.pack(side="left", fill="both", expand=True, padx=8, pady=3)
        self.type_input.insert(0, "Type a manual message...")
        self.type_entry = self.type_input

        self.type_submit_btn = tk.Button(
            input_container,
            text="Send",
            command=self.submit_manual_question,
            bg="#141518",
            fg="#94a3b8",
            activebackground="#141518",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            cursor="hand2"
        )
        self.type_submit_btn.pack(side="right", padx=4)

        self.type_input.bind("<FocusIn>", self._on_type_focus_in)
        self.type_input.bind("<FocusOut>", self._on_type_focus_out)
        self.type_input.bind("<Return>", lambda e: self.submit_manual_question())
        self.type_input.bind("<KP_Enter>", lambda e: self.submit_manual_question())

        # Compatibility aliases
        self.generate_btn = self.ai_answer_btn
        self.more_btn = self.type_submit_btn
        self.dock_more_btn = self.hdr_more_btn

    def _build_answer_card(self):
        """Floating frosted glass answer card matching authentic Parakeet AI."""
        self.answer_card_frame = tk.Frame(
            self.main_frame,
            bg="#111216",
            highlightbackground="#23252b",
            highlightthickness=1,
            padx=16,
            pady=10
        )
        # By default collapsed until an answer is ready/streaming!

        # 1. Top Card Control Bar (< > + ✕ close)
        card_top = tk.Frame(self.answer_card_frame, bg="#111216")
        card_top.pack(fill="x", pady=(0, 4))

        nav_box = tk.Frame(card_top, bg="#111216")
        nav_box.pack(side="left")

        self.prev_q_btn = tk.Button(
            nav_box,
            text="<",
            command=self._nav_prev_question,
            bg="#111216",
            fg="#94a3b8",
            activebackground="#111216",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        self.prev_q_btn.pack(side="left", padx=(0, 4))

        self.next_q_btn = tk.Button(
            nav_box,
            text=">",
            command=self._nav_next_question,
            bg="#111216",
            fg="#94a3b8",
            activebackground="#111216",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        self.next_q_btn.pack(side="left", padx=(0, 6))

        self.history_counter_lbl = tk.Label(
            nav_box,
            text="",
            bg="#111216",
            fg="#64748b",
            font=("Segoe UI", 9)
        )
        self.history_counter_lbl.pack(side="left")

        right_card_top = tk.Frame(card_top, bg="#111216")
        right_card_top.pack(side="right")

        self.card_copy_btn = tk.Button(
            right_card_top,
            text="📋 Copy",
            command=self.copy_answer_to_clipboard,
            bg="#202228",
            fg="#cbd5e1",
            activebackground="#333640",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=9,
            pady=2,
            cursor="hand2"
        )
        self.card_copy_btn.pack(side="left", padx=(0, 8))
        self._bind_hover_fx(self.card_copy_btn, "#202228", "#333640", "#cbd5e1", "#ffffff")

        self.card_close_btn = tk.Button(
            right_card_top,
            text="✕",
            command=self.hide_answer_card,
            bg="#111216",
            fg="#94a3b8",
            activebackground="#111216",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        self.card_close_btn.pack(side="left")

        # 2. Summarized Question Row
        q_frame = tk.Frame(self.answer_card_frame, bg="#111216")
        q_frame.pack(fill="x", pady=(2, 6))

        self.q_prefix_label = tk.Label(
            q_frame,
            text="💬 Summarized question: ",
            bg="#111216",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold")
        )
        self.q_prefix_label.pack(side="left", anchor="nw")

        self.understood_label = tk.Label(
            q_frame,
            text="",
            bg="#111216",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            wraplength=800,
            justify="left"
        )
        self.understood_label.pack(side="left", fill="x", expand=True)

        # 3. Answer Title Row
        ans_title_frame = tk.Frame(self.answer_card_frame, bg="#111216")
        ans_title_frame.pack(fill="x", pady=(3, 3))

        self.answer_title_label = tk.Label(
            ans_title_frame,
            text="⭐ Answer:",
            bg="#111216",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold")
        )
        self.answer_title_label.pack(side="left")

        self.answer_status_badge = tk.Label(
            ans_title_frame,
            text="",
            bg="#111216",
            fg="#34d399",
            font=("Segoe UI", 8, "bold")
        )
        self.answer_status_badge.pack(side="left", padx=8)

        # 4. Formatted Answer Box
        ans_box_frame = tk.Frame(self.answer_card_frame, bg="#111216")
        ans_box_frame.pack(fill="both", expand=True, pady=(2, 4))

        self.answer_box = ScrolledText(
            ans_box_frame,
            bg="#111216",
            fg="#f1f5f9",
            insertbackground="#ffffff",
            selectbackground="#334155",
            selectforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 11),
            wrap="word",
            padx=10,
            pady=6,
            height=12,
            spacing1=4,
            spacing2=4,
            spacing3=8
        )
        self.answer_box.pack(fill="both", expand=True)

        # High-Contrast Scannable Typography Tags
        self.answer_box.tag_config("heading", foreground="#38bdf8", font=("Segoe UI", 12, "bold"), spacing1=6, spacing3=3)
        self.answer_box.tag_config("solution_header", foreground="#ffffff", font=("Segoe UI", 12, "bold"), spacing1=8, spacing3=4)
        self.answer_box.tag_config("section_header", foreground="#38bdf8", font=("Segoe UI", 11, "bold"), spacing1=8, spacing3=3)
        self.answer_box.tag_config("key_header", foreground="#fbbf24", font=("Segoe UI", 11, "bold"), spacing1=8, spacing3=3)
        self.answer_box.tag_config("key_body", foreground="#fef08a", font=("Segoe UI", 11), spacing1=3, spacing3=5, lmargin1=12, lmargin2=12)
        self.answer_box.tag_config("bullet_symbol", foreground="#38bdf8", font=("Segoe UI", 11, "bold"))
        self.answer_box.tag_config("complexity_time_tag", foreground="#38bdf8", font=("Segoe UI", 11, "bold"))
        self.answer_box.tag_config("complexity_space_tag", foreground="#fbbf24", font=("Segoe UI", 11, "bold"))
        self.answer_box.tag_config("bullet_body", foreground="#f8fafc", font=("Segoe UI", 11), spacing1=3, spacing3=4, lmargin1=14, lmargin2=28)
        self.answer_box.tag_config("body", foreground="#f8fafc", font=("Segoe UI", 11), spacing1=4, spacing2=4, spacing3=8)
        self.answer_box.tag_config("code_kw", foreground="#f472b6", font=("Consolas", 10, "bold"))
        self.answer_box.tag_config("code_fn", foreground="#60a5fa", font=("Consolas", 10, "bold"))
        self.answer_box.tag_config("code_builtin", foreground="#38bdf8", font=("Consolas", 10))
        self.answer_box.tag_config("code_str", foreground="#a7f3d0", font=("Consolas", 10))
        self.answer_box.tag_config("code_num", foreground="#fbbf24", font=("Consolas", 10))
        self.answer_box.tag_config("code_comment", foreground="#64748b", font=("Consolas", 10, "italic"))
        self.answer_box.tag_config("code_ident", foreground="#f8fafc", font=("Consolas", 10))
        self.answer_box.tag_config("code_body", foreground="#f8fafc", background="#181a22", font=("Consolas", 10), spacing1=3, spacing3=3, lmargin1=12, lmargin2=12)

        # 5. Bottom Refinement Chips
        chips_frame = tk.Frame(self.answer_card_frame, bg="#111216")
        chips_frame.pack(fill="x", pady=(4, 0))

        chips = [
            ("⚡ 30s Short", "30s Short", "Make this response concise, direct, and under 60 words for a quick punchy delivery.", "#38bdf8"),
            ("🎯 STAR", "STAR Method", "Structure this answer strictly in conversational STAR format with clear metrics.", "#a855f7"),
            ("🛠️ Deep Tech", "Deep Tech", "Add deeper architectural details, trade-offs, and internal mechanics.", "#10b981"),
            ("✨ Simplify", "Simplify", "Explain this simply, clearly, and conversationally in plain English.", "#38bdf8"),
            ("🏛️ System Design", "System Design", "Structure as high-scale system design.", "#f59e0b"),
            ("🌟 STAR Story", "STAR Story", "Generate behavioral story grounded in resume.", "#c084fc"),
            ("▶️ Run Code", "Run Code", "Execute Python code sandbox.", "#2dd4bf"),
            ("🎓 Mock Practice", "Mock Practice", "Launch mock practice simulation.", "#fb7185"),
        ]
        for text, mode_name, prompt, color in chips:
            btn = tk.Button(
                chips_frame,
                text=text,
                command=lambda m=mode_name, p=prompt: self._on_chip_clicked(m, p),
                bg="#1a1c24",
                fg=color,
                activebackground="#2a2d3a",
                activeforeground="#ffffff",
                relief="flat",
                borderwidth=0,
                font=("Segoe UI", 8, "bold"),
                padx=8,
                pady=3,
                cursor="hand2"
            )
            btn.pack(side="left", padx=(0, 5))
            self._bind_hover_fx(btn, "#1a1c24", "#2a2d3a", color, "#ffffff")

        # Compatibility references
        self.right_panel = self.answer_card_frame
        self.left_panel = tk.Frame(self.main_frame, bg="#080c14")
        self.transcript_box = ScrolledText(self.left_panel, bg="#080c14", fg="#94a3b8", height=1)
        self.live_indicator = self.timer_badge
        self.understood_frame = self.answer_card_frame
        self.question_type_badge = self.q_prefix_label
        self.understood_tag = self.q_prefix_label
        self.teleprompter_btn = tk.Button(chips_frame, text="▶ Auto-Scroll", command=self.toggle_teleprompter)
        self.sys_design_btn = tk.Button(chips_frame, text="🏛️ System Design", command=self.trigger_system_design_generator)
        self.star_story_btn = tk.Button(chips_frame, text="🌟 STAR Story", command=self.trigger_star_story_generator)
        self.run_code_btn = tk.Button(chips_frame, text="▶️ Run Code", command=self.trigger_run_code_sandbox)

    def _on_chip_clicked(self, mode_name: str, prompt: str):
        if mode_name == "Mock Practice":
            self.open_mock_interview_modal()
        elif mode_name == "System Design":
            self.trigger_system_design_generator()
        elif mode_name == "STAR Story":
            self.trigger_star_story_generator()
        elif mode_name == "Run Code":
            self.trigger_run_code_sandbox()
        else:
            self._select_answer_mode(mode_name, prompt)

    def _nav_prev_question(self):
        if self._history_items and self._history_idx > 0:
            self._history_idx -= 1
            self._load_history_item(self._history_idx)

    def _nav_next_question(self):
        if self._history_items and self._history_idx < len(self._history_items) - 1:
            self._history_idx += 1
            self._load_history_item(self._history_idx)

    def _load_history_item(self, idx: int):
        if 0 <= idx < len(self._history_items):
            item = self._history_items[idx]
            q = item.get("question", "")
            a = item.get("answer", "")
            q_type = item.get("type", "general")
            self.understood_label.config(text=q)
            self.history_counter_lbl.config(text=f"{idx + 1}/{len(self._history_items)}")
            from answer_formatter import render_structured_answer_to_text_widget
            render_structured_answer_to_text_widget(self.answer_box, a, q_type)
            self.show_answer_card()

    def _record_history(self, question: str, answer: str, q_type: str = "general"):
        if not question or not answer:
            return
        if self._history_items and self._history_items[-1].get("question") == question:
            self._history_items[-1]["answer"] = answer
            self._history_items[-1]["type"] = q_type
        else:
            self._history_items.append({"question": question, "answer": answer, "type": q_type})
        self._history_idx = len(self._history_items) - 1
        if hasattr(self, "history_counter_lbl") and self.history_counter_lbl:
            self.history_counter_lbl.config(text=f"{self._history_idx + 1}/{len(self._history_items)}")

    def _apply_conversation_history_layout(self):
        pass

    def toggle_conversation_history(self):
        pass

    def _build_question_bar(self):
        pass

    def _build_left_pane(self):
        pass

    def _build_right_pane(self):
        pass

    def _build_bottom_dock(self):
        pass

    def cycle_opacity(self):
        """Cycles window transparency: 100% -> 90% -> 80% -> 70% -> 60% -> 100%."""
        levels = [1.0, 0.90, 0.80, 0.70, 0.60]
        self.current_opacity_idx = (getattr(self, "current_opacity_idx", 0) + 1) % len(levels)
        new_alpha = levels[self.current_opacity_idx]
        self.set_opacity(new_alpha)

    def set_opacity(self, alpha_val: float):
        """Sets the alpha opacity for the window."""
        try:
            self.root.attributes("-alpha", alpha_val)
            pct = int(alpha_val * 100)
            if hasattr(self, "opacity_btn") and self.opacity_btn:
                self.opacity_btn.config(text=f"👁️ {pct}%")
        except Exception as e:
            print("Opacity set notice:", e)

    def _show_opacity_menu(self):
        """Displays direct opacity selector dropdown on right click."""
        menu = tk.Menu(self.root, tearoff=0, bg="#1e293b", fg="#f8fafc", activebackground="#2563eb", activeforeground="#ffffff", font=("Segoe UI", 9))
        menu.add_command(label="100% (Solid)", command=lambda: self.set_opacity(1.0))
        menu.add_command(label="90% (Subtle Glass)", command=lambda: self.set_opacity(0.90))
        menu.add_command(label="80% (Glassmorphic)", command=lambda: self.set_opacity(0.80))
        menu.add_command(label="70% (High Transparency)", command=lambda: self.set_opacity(0.70))
        menu.add_command(label="60% (Stealth Ghost)", command=lambda: self.set_opacity(0.60))
        try:
            x = self.root.winfo_rootx() + 40
            y = self.root.winfo_rooty() + 38
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _minimize_window(self):
        if hasattr(self, "main_app") and self.main_app and hasattr(self.main_app, "minimize_window"):
            self.main_app.minimize_window()
        else:
            self.root.iconify()

    def _close_window(self):
        if hasattr(self, "main_app") and self.main_app and hasattr(self.main_app, "close_window"):
            self.main_app.close_window()
        else:
            self.root.destroy()

    def _show_header_more_menu(self):
        """Displays settings, shortcuts, practice, and session end."""
        menu = tk.Menu(self.root, tearoff=0, bg="#1e293b", fg="#f8fafc", activebackground="#2563eb", activeforeground="#ffffff", font=("Segoe UI", 9))
        menu.add_command(label="📋 Copy Answer to Clipboard (Alt+K)", command=self.copy_answer_to_clipboard)
        menu.add_command(label="⚙️ Job & Resume Settings", command=self.open_settings_modal)
        menu.add_command(label="⌨️ Keyboard Shortcuts", command=lambda: self.main_app.show_shortcuts_modal() if hasattr(self, "main_app") and self.main_app else None)
        menu.add_command(label="🎓 Mock Interview Practice", command=self.open_mock_interview_modal)
        menu.add_command(label="🏛️ System Design Generator", command=self.trigger_system_design_generator)
        menu.add_command(label="🌟 STAR Story Generator", command=self.trigger_star_story_generator)
        menu.add_command(label="▶️ Run Code Sandbox", command=self.trigger_run_code_sandbox)
        menu.add_command(label="▶ Auto-Scroll Teleprompter (Ctrl+Alt+P)", command=self.toggle_teleprompter)
        menu.add_command(label="🔊 Switch Audio Source", command=self.toggle_audio_source)
        menu.add_command(label="👁️ Opacity Adjustment", command=self._show_opacity_menu)
        menu.add_command(label="📄 Export Session Report", command=self.open_export_dialog)
        menu.add_separator()
        menu.add_command(label="🏁 End Session & Summary", command=self.show_session_summary_modal)
        self._header_menu = menu

        try:
            x = self.root.winfo_rootx() + 40
            y = self.root.winfo_rooty() + 38
            self._header_menu.tk_popup(x, y)
        finally:
            self._header_menu.grab_release()

    def _select_answer_mode(self, mode_name: str, prompt: str):
        """Updates the active mode label and triggers refinement."""
        if self._is_alive(self.more_btn):
            self.more_btn.config(text=f"Mode: {mode_name} ▾")
        self.refine_active_answer(prompt)

    def _show_more_modes_menu(self):
        """Displays the popup menu for refinement modes and secondary actions."""
        if not hasattr(self, "_modes_menu"):
            menu = tk.Menu(self.root, tearoff=0, bg="#1e293b", fg="#f8fafc", activebackground="#2563eb", activeforeground="#ffffff", font=("Segoe UI", 9))
            menu.add_command(label="⚡ 30s Short Refinement", command=lambda: self._select_answer_mode("30s Short", "Make this response extremely concise, direct, and under 60 words for a quick punchy delivery."))
            menu.add_command(label="🎯 STAR Method Refinement", command=lambda: self._select_answer_mode("STAR Method", "Structure this answer strictly in conversational STAR format (Situation, Task, Action, Result) with clear impact."))
            menu.add_command(label="🛠️ Deep Tech Refinement", command=lambda: self._select_answer_mode("Deep Tech", "Add deeper architectural details, trade-offs, scalability considerations, and internal mechanics."))
            menu.add_command(label="🏗️ System Design Refinement", command=lambda: self._select_answer_mode("System Design", "Structure as high-scale system design: Requirements, High-Level Architecture, APIs, Data Storage, Caching, Scaling & Failure Handling."))
            menu.add_command(label="💡 Real Example Refinement", command=lambda: self._select_answer_mode("Real Example", "Weave in a concrete real-world engineering example and metric from my verified experience."))
            menu.add_command(label="✨ Simplify Refinement", command=lambda: self._select_answer_mode("Simplify", "Explain this simply, clearly, and conversationally in plain English without unnecessary jargon."))
            menu.add_separator()
            menu.add_command(label="⚡ Automatic Mode", command=lambda: self._select_answer_mode("Automatic", "Provide an optimal, highly structured technical answer."))
            self._modes_menu = menu

        try:
            x = self.more_btn.winfo_rootx()
            y = self.more_btn.winfo_rooty() - self._modes_menu.winfo_reqheight() - 4
            self._modes_menu.tk_popup(x, y)
        finally:
            self._modes_menu.grab_release()

    def _on_type_focus_in(self, event):
        try:
            curr = self.type_input.get("1.0", "end-1c").strip()
            if curr == "Type or paste question here...":
                self.type_input.delete("1.0", "end")
                self.type_input.config(fg="#f8fafc")
        except Exception:
            pass

    def _on_type_focus_out(self, event):
        try:
            curr = self.type_input.get("1.0", "end-1c").strip()
            if not curr:
                self.type_input.delete("1.0", "end")
                self.type_input.insert("1.0", "Type or paste question here...")
                self.type_input.config(fg="#94a3b8")
        except Exception:
            pass

    def submit_manual_question(self, event=None):
        """
        Unified manual question submission handler invoked via Ctrl+Enter or [⚡ Ask AI] button.
        Routes the typed/pasted question through the unified semantic classifier,
        screen-aware coding fusion, refinement, and answer streaming pipelines.
        """
        if not self._is_alive():
            return "break"
            
        try:
            raw_query = self.type_input.get("1.0", "end-1c").strip()
        except Exception:
            return "break"
            
        PLACEHOLDER = "Type or paste question here..."
        if not raw_query or raw_query == PLACEHOLDER:
            return "break"  # Ignore empty or whitespace-only input
            
        # Deduplication check: prevent rapid double submission within 3.0s
        clean_key = re.sub(r'[^a-zA-Z0-9]', '', raw_query.lower())
        now = time.time()
        if clean_key in self.recent_questions_cache and (now - self.last_question_time < 3.0):
            return "break"
            
        self.recent_questions_cache.append(clean_key)
        self.last_question_time = now
        
        # Clear input upon successful acceptance
        try:
            self.type_input.delete("1.0", "end")
        except Exception:
            pass
            
        # Update compact status
        if self._is_alive(self.answer_status_badge):
            self.answer_status_badge.config(text="⌨️ Manual Question", fg=self.PALETTE["accent_cyan"])
            
        # Process through the Unified Question Processor in worker thread
        def _process_worker():
            try:
                analysis = classify_and_understand_utterance(
                    utterance=raw_query,
                    session=self.session,
                    client=self.client,
                    source="MANUAL_INPUT",
                    is_manual=True
                )
                self.safe_after(0, lambda: self._handle_manual_analysis_ui(analysis, raw_query))
            except Exception as e:
                print(f"Manual question understanding error: {e}")
                self.safe_after(0, lambda: self._handle_manual_fallback_ui(raw_query))
                
        threading.Thread(target=_process_worker, daemon=True).start()
        return "break"

    def _handle_manual_analysis_ui(self, analysis: dict, raw_query: str):
        if not self._is_alive():
            return
            
        refined_q = analysis.get("refined_question", raw_query)
        utterance_type = analysis.get("utterance_type", "TECHNICAL")
        
        # Invalidate any active stale generation (live or previous manual)
        self.current_generation_id = getattr(self, "current_generation_id", 0) + 1
        
        # Add to shared session transcript and QA history
        entry = self.session.add_transcript(raw_query, role="Interviewer", is_question=True, source="MANUAL_INPUT", is_manual=True)
        
        if self._is_alive(self.transcript_box):
            try:
                t_str = entry.get('timestamp', '')
                time_display = f"{t_str} " if t_str else ""
                self.transcript_box.insert("end", time_display, "timestamp")
                tag_label = "⌨️ Manual Question  " if utterance_type != "FOLLOW_UP" else "⌨️ Manual Follow-Up  "
                self.transcript_box.insert("end", tag_label, "question_tag")
                self.transcript_box.insert("end", "[ Prompt ]\n", "speaker_interviewer")
                self.transcript_box.insert("end", f"❓ {raw_query}\n\n", "question_text")
                self.transcript_box.see("end")
            except Exception:
                pass
                
        # Update understood label
        if self._is_alive(self.understood_label):
            label_text = f"\"{refined_q}\""
            if utterance_type == "FOLLOW_UP":
                label_text = f"🔗 (Follow-Up): \"{refined_q}\""
            self.understood_label.config(
                text=label_text,
                fg="#ffffff",
                font=("Segoe UI", 13, "bold")
            )
        if hasattr(self, "question_type_badge") and self._is_alive(self.question_type_badge):
            type_label = utterance_type.replace('_', ' ').title()
            self.question_type_badge.config(text=f"● {type_label}", fg="#38bdf8")
            
        self.active_question = refined_q
        self.active_raw_question = raw_query
        
        # Trigger answer streaming through unified AI answer generator
        self.trigger_ai_answer(
            question=refined_q,
            raw_transcript=raw_query,
            question_type=utterance_type,
            source="MANUAL_INPUT"
        )

    def _handle_manual_fallback_ui(self, raw_query: str):
        if not self._is_alive():
            return
        self.current_generation_id = getattr(self, "current_generation_id", 0) + 1
        entry = self.session.add_transcript(raw_query, role="Interviewer", is_question=True, source="MANUAL_INPUT", is_manual=True)
        if self._is_alive(self.transcript_box):
            try:
                t_str = entry.get('timestamp', '')
                time_display = f"{t_str} " if t_str else ""
                self.transcript_box.insert("end", time_display, "timestamp")
                self.transcript_box.insert("end", "⌨️ Manual Question\n", "question_tag")
                self.transcript_box.insert("end", f"❓ {raw_query}\n\n", "question_text")
                self.transcript_box.see("end")
            except Exception:
                pass
        self.trigger_ai_answer(question=raw_query, raw_transcript=raw_query, source="MANUAL_INPUT")

    def _select_answer_mode(self, mode_name: str, prompt: str):
        """Updates the active mode label and triggers refinement."""
        if hasattr(self, "more_btn") and self._is_alive(self.more_btn):
            self.more_btn.config(text=f"Mode: {mode_name} ▾")
        self.refine_active_answer(prompt)

    def _on_type_focus_in(self, event):
        try:
            curr = self.type_input.get().strip()
            if curr == "Type a manual message...":
                self.type_input.delete(0, "end")
                self.type_input.config(fg="#f8fafc")
        except Exception:
            pass

    def _on_type_focus_out(self, event):
        try:
            curr = self.type_input.get().strip()
            if not curr:
                self.type_input.delete(0, "end")
                self.type_input.insert(0, "Type a manual message...")
                self.type_input.config(fg="#94a3b8")
        except Exception:
            pass

    def submit_manual_question(self, event=None):
        """
        Unified manual question submission handler invoked via Enter, Ctrl+Enter or Send button.
        Routes the typed/pasted question through the unified semantic classifier,
        screen-aware coding fusion, refinement, and answer streaming pipelines.
        """
        if not self._is_alive():
            return "break"
            
        try:
            if isinstance(self.type_input, tk.Entry):
                raw_query = self.type_input.get().strip()
            else:
                raw_query = self.type_input.get("1.0", "end-1c").strip()
        except Exception:
            return "break"
            
        PLACEHOLDER = "Type a manual message..."
        if not raw_query or raw_query == PLACEHOLDER:
            return "break"
            
        clean_key = re.sub(r'[^a-zA-Z0-9]', '', raw_query.lower())
        now = time.time()
        if clean_key in self.recent_questions_cache and (now - self.last_question_time < 3.0):
            return "break"
            
        self.recent_questions_cache.append(clean_key)
        self.last_question_time = now
        
        try:
            if isinstance(self.type_input, tk.Entry):
                self.type_input.delete(0, "end")
            else:
                self.type_input.delete("1.0", "end")
        except Exception:
            pass
            
        if self._is_alive(self.answer_status_badge):
            self.answer_status_badge.config(text="⌨️ Manual Question", fg=self.PALETTE["accent_cyan"])
            
        def _process_worker():
            try:
                analysis = classify_and_understand_utterance(
                    utterance=raw_query,
                    session=self.session,
                    client=self.client,
                    source="MANUAL_INPUT",
                    is_manual=True
                )
                self.safe_after(0, lambda: self._handle_manual_analysis_ui(analysis, raw_query))
            except Exception as e:
                print(f"Manual question understanding error: {e}")
                self.safe_after(0, lambda: self._handle_manual_fallback_ui(raw_query))
                
        threading.Thread(target=_process_worker, daemon=True).start()
        return "break"

    def _handle_manual_analysis_ui(self, analysis: dict, raw_query: str):
        if not self._is_alive():
            return
            
        refined_q = analysis.get("refined_question", raw_query)
        utterance_type = analysis.get("utterance_type", "TECHNICAL")
        
        self.current_generation_id = getattr(self, "current_generation_id", 0) + 1
        entry = self.session.add_transcript(raw_query, role="Interviewer", is_question=True, source="MANUAL_INPUT", is_manual=True)
        
        if self._is_alive(self.transcript_box):
            try:
                t_str = entry.get('timestamp', '')
                time_display = f"{t_str} " if t_str else ""
                self.transcript_box.insert("end", time_display, "timestamp")
                tag_label = "⌨️ Manual Question  " if utterance_type != "FOLLOW_UP" else "⌨️ Manual Follow-Up  "
                self.transcript_box.insert("end", tag_label, "question_tag")
                self.transcript_box.insert("end", "[ Prompt ]\n", "speaker_interviewer")
                self.transcript_box.insert("end", f"❓ {raw_query}\n\n", "question_text")
                self.transcript_box.see("end")
            except Exception:
                pass
                
        if self._is_alive(self.understood_label):
            label_text = f"\"{refined_q}\""
            if utterance_type == "FOLLOW_UP":
                label_text = f"🔗 (Follow-Up): \"{refined_q}\""
            self.understood_label.config(
                text=label_text,
                fg="#ffffff",
                font=("Segoe UI", 10, "bold")
            )
        if hasattr(self, "question_type_badge") and self._is_alive(self.question_type_badge):
            type_label = utterance_type.replace('_', ' ').title()
            self.question_type_badge.config(text=f"● {type_label}", fg="#38bdf8")
            
        self.active_question = refined_q
        self.active_raw_question = raw_query
        self.show_answer_card()
        self._record_history(refined_q, "", utterance_type)
        
        self.trigger_ai_answer(
            question=refined_q,
            raw_transcript=raw_query,
            question_type=utterance_type,
            source="MANUAL_INPUT"
        )

    def _handle_manual_fallback_ui(self, raw_query: str):
        if not self._is_alive():
            return
        self.current_generation_id = getattr(self, "current_generation_id", 0) + 1
        entry = self.session.add_transcript(raw_query, role="Interviewer", is_question=True, source="MANUAL_INPUT", is_manual=True)
        if self._is_alive(self.transcript_box):
            try:
                t_str = entry.get('timestamp', '')
                time_display = f"{t_str} " if t_str else ""
                self.transcript_box.insert("end", time_display, "timestamp")
                self.transcript_box.insert("end", "⌨️ Manual Question\n", "question_tag")
                self.transcript_box.insert("end", f"❓ {raw_query}\n\n", "question_text")
                self.transcript_box.see("end")
            except Exception:
                pass
        self.show_answer_card()
        self._record_history(raw_query, "", "general")
        self.trigger_ai_answer(question=raw_query, raw_transcript=raw_query, source="MANUAL_INPUT")

    def _create_pill_btn(self, parent, text, command, color):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg="#1e293b",
            fg=color,
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=5,
            cursor="hand2"
        )
        return btn

    def _bind_hotkeys(self):
        """Binds Ctrl+Enter, Ctrl+K, Spacebar, Arrow key scrolling, and other shortcuts."""
        self.root.bind("<Control-Return>", lambda e: self.submit_manual_question())
        self.root.bind("<Control-KP_Enter>", lambda e: self.submit_manual_question())
        self.root.bind("<Control-k>", lambda e: self.toggle_conversation_history())
        self.root.bind("<Control-K>", lambda e: self.toggle_conversation_history())
        self.root.bind("<Up>", self._on_arrow_up)
        self.root.bind("<Down>", self._on_arrow_down)
        self.root.bind("<Prior>", lambda e: self.answer_box.yview_scroll(-5, "units"))
        self.root.bind("<Next>", lambda e: self.answer_box.yview_scroll(5, "units"))
        self.root.bind("<space>", self._on_space_pressed)
        self.root.bind("<Alt-s>", lambda e: self.trigger_code_screen_solver())
        self.root.bind("<Alt-S>", lambda e: self.trigger_code_screen_solver())
        self.root.bind("<Alt-o>", lambda e: self.cycle_opacity())
        self.root.bind("<Alt-O>", lambda e: self.cycle_opacity())

    def _init_global_hotkeys(self):
        """Registers system-wide OS hotkeys (works across Zoom, Teams, Meet, IDE, Browser)."""
        prefs = get_saved_hotkey_preferences()

        # 1. Panic Stealth Hide/Unhide (Default: Ctrl+Alt+H)
        combo_hide = prefs.get("panic_hide", "Ctrl+Alt+H")
        self.hotkey_manager.register("panic_hide", combo_hide, lambda: self.root.after(0, self.toggle_overlay_visibility))

        # 2. Snip & Solve Screen Coding Problem (Default: Ctrl+Alt+S)
        combo_solve = prefs.get("solve_screen", "Ctrl+Alt+S")
        self.hotkey_manager.register("solve_screen", combo_solve, lambda: self.root.after(0, self.trigger_code_screen_solver))

        # 3. Mute / Pause AI Listening (Default: Ctrl+Alt+M)
        combo_mute = prefs.get("mute_ai", "Ctrl+Alt+M")
        self.hotkey_manager.register("mute_ai", combo_mute, lambda: self.root.after(0, self.toggle_live_transcription))

        # 4. Quick Regenerate Answer (Default: Ctrl+Alt+R)
        combo_regen = prefs.get("regenerate", "Ctrl+Alt+R")
        self.hotkey_manager.register("regenerate", combo_regen, lambda: self.root.after(0, self.manual_generate_answer))

        # 5. Cycle Window Opacity (Default: Ctrl+Alt+T)
        combo_op = prefs.get("cycle_opacity", "Ctrl+Alt+T")
        self.hotkey_manager.register("cycle_opacity", combo_op, lambda: self.root.after(0, self.cycle_opacity))

        # 6. Teleprompter Auto-Scroll Toggle (Default: Ctrl+Alt+P)
        combo_tp = prefs.get("teleprompter", "Ctrl+Alt+P")
        self.hotkey_manager.register("teleprompter", combo_tp, lambda: self.root.after(0, self.toggle_teleprompter))

        # Start the OS message pump
        self.hotkey_manager.start()

    def toggle_teleprompter(self):
        """Toggles hands-free teleprompter auto-scroll on the answer pane."""
        if hasattr(self, "teleprompter") and self.teleprompter:
            is_active = self.teleprompter.toggle()
            if hasattr(self, "teleprompter_btn") and self._is_alive(self.teleprompter_btn):
                if is_active:
                    self.teleprompter_btn.config(text=f"⏸️ Pause ({self.teleprompter.wpm} WPM)", fg=self.PALETTE["accent_cyan"])
                else:
                    self.teleprompter_btn.config(text="▶ Auto-Scroll", fg=self.PALETTE["text_secondary"])

    def cycle_teleprompter_speed(self):
        """Cycles teleprompter reading speed (Normal -> Fast -> Slow)."""
        if hasattr(self, "teleprompter") and self.teleprompter:
            new_preset = self.teleprompter.cycle_speed()
            if hasattr(self, "teleprompter_btn") and self._is_alive(self.teleprompter_btn):
                self.teleprompter_btn.config(text=f"▶ {new_preset.title()} ({self.teleprompter.wpm} WPM)")

    def _on_teleprompter_status(self, status: str):
        """Callback from teleprompter on status change."""
        if hasattr(self, "teleprompter_btn") and self._is_alive(self.teleprompter_btn):
            if "Paused" in status or "End" in status:
                self.teleprompter_btn.config(text="▶ Auto-Scroll", fg=self.PALETTE["text_secondary"])

    def toggle_overlay_visibility(self):
        """Panic Stealth Hotkey: instantly toggles overlay visibility on/off across the OS."""
        if getattr(self, "_is_overlay_hidden", False):
            try:
                self.root.deiconify()
                self.root.lift()
                self.root.attributes("-topmost", True)
                self.root.focus_force()
            except Exception:
                pass
            self._is_overlay_hidden = False
        else:
            try:
                self.root.withdraw()
            except Exception:
                pass
            self._is_overlay_hidden = True

    def cleanup_resources(self):
        """Clean up background workers and hotkey listeners on session end."""
        if hasattr(self, "teleprompter") and self.teleprompter:
            self.teleprompter.stop()
        if hasattr(self, "hotkey_manager") and self.hotkey_manager:
            self.hotkey_manager.stop()
        if hasattr(self, "transcriber") and self.transcriber and self.transcriber.is_running:
            self.transcriber.stop()

    def _on_arrow_up(self, event):
        widget = self.root.focus_get()
        if isinstance(widget, (tk.Entry, tk.Text, ScrolledText)) and widget == self.type_input:
            return
        if self._is_alive(self.answer_box):
            self.answer_box.yview_scroll(-2, "units")
        return "break"

    def _on_arrow_down(self, event):
        widget = self.root.focus_get()
        if isinstance(widget, (tk.Entry, tk.Text, ScrolledText)) and widget == self.type_input:
            return
        if self._is_alive(self.answer_box):
            self.answer_box.yview_scroll(2, "units")
        return "break"

    def _on_space_pressed(self, event):
        """Spacebar triggers answer generation when not typing inside an Entry or Text widget."""
        widget = self.root.focus_get()
        if isinstance(widget, (tk.Entry, tk.Text, ScrolledText)):
            return  # Allow normal space typing in text fields
        self.manual_generate_answer()

    # =========================================================
    # LIVE AUDIO & TRANSCRIPTION HANDLING
    # =========================================================

    def toggle_live_transcription(self):
        if not self.transcriber.is_running:
            self.transcriber.start(
                source=self.session.audio_source,
                mic_device_id=getattr(self.session, "selected_mic_device", None),
                system_speaker_id=getattr(self.session, "selected_speaker_device", None)
            )
            if hasattr(self, "audio_toggle_btn") and self._is_alive(self.audio_toggle_btn):
                self.audio_toggle_btn.config(
                    text="Stop ⚡",
                    fg="#ef4444",
                    bg="#2b2d35"
                )
            if hasattr(self, "live_indicator") and self.live_indicator and self._is_alive(self.live_indicator):
                self.live_indicator.config(text="🖥️ ⏺", fg="#ef4444", bg="#2b2d35")
        else:
            self.transcriber.stop()
            if hasattr(self, "audio_toggle_btn") and self._is_alive(self.audio_toggle_btn):
                self.audio_toggle_btn.config(
                    text="Start ⚡",
                    fg="#34d399",
                    bg="#2b2d35"
                )
            if hasattr(self, "live_indicator") and self.live_indicator and self._is_alive(self.live_indicator):
                self.live_indicator.config(text="🖥️ ⏺", fg="#e2e8f0", bg="#2b2d35")

    def toggle_audio_source(self):
        if self.session.audio_source == "dual":
            self.session.audio_source = "system"
            self.source_btn.config(text="🔊 Source: Loopback (Interviewer)")
        elif self.session.audio_source == "system":
            self.session.audio_source = "mic"
            self.source_btn.config(text="🎙️ Source: Mic (Single Mic Heuristic)")
        else:
            self.session.audio_source = "dual"
            self.source_btn.config(text="🔊 Source: Dual (Loopback + Mic)")
            
        if self.transcriber.is_running:
            self.transcriber.stop()
            self.transcriber.start(
                source=self.session.audio_source,
                mic_device_id=getattr(self.session, "selected_mic_device", None),
                system_speaker_id=getattr(self.session, "selected_speaker_device", None)
            )

    def toggle_auto_generate_mode(self):
        self.session.auto_generate = not self.session.auto_generate
        if self._is_alive(self.auto_mode_btn):
            if self.session.auto_generate:
                self.auto_mode_btn.config(
                    text="⚡ Auto-Gen: ON",
                    fg=self.PALETTE["accent_cyan"]
                )
            else:
                self.auto_mode_btn.config(
                    text="⌨️ Auto-Gen: OFF",
                    fg=self.PALETTE["text_muted"]
                )

    def _on_audio_status(self, status: str):
        if not self._is_alive(self.audio_status_badge):
            return
        self.safe_after(0, lambda: self.audio_status_badge.config(text=status) if self._is_alive(self.audio_status_badge) else None)

    def _on_audio_level_update(self, source: str, level: float, rms: float):
        """Updates live dual-channel VU meters in the HUD header."""
        pct = min(100, int(level * 100))
        
        def _apply():
            try:
                if "system" in source or "speaker" in source:
                    if hasattr(self, "spk_vu_label") and self._is_alive(self.spk_vu_label):
                        if pct > 4:
                            self.spk_vu_label.config(text=f"🔊 {pct}%", fg="#38bdf8")
                        else:
                            self.spk_vu_label.config(text="🔊 0%", fg=self.PALETTE["text_subtle"])
                else:
                    if hasattr(self, "mic_vu_label") and self._is_alive(self.mic_vu_label):
                        if pct > 4:
                            self.mic_vu_label.config(text=f"🎙️ {pct}%", fg="#34d399")
                        else:
                            self.mic_vu_label.config(text="🎙️ 0%", fg=self.PALETTE["text_subtle"])
            except Exception:
                pass

        if self._is_alive(self.root):
            self.safe_after(0, _apply)

    def _on_interim_transcript(self, raw_text: str, likelihood_info: dict, source: str = "system"):
        """
        Dispatched from continuous transcriber when a live interim speech fragment is recognized.
        Updates question likelihood status and interim visual preview.
        """
        if not raw_text or not self._is_alive(self.transcript_box):
            return
        self.safe_after(0, lambda: self._render_interim_ui(raw_text, likelihood_info, source=source))

    def _render_interim_ui(self, raw_text: str, likelihood_info: dict, source: str = "system"):
        if not self._is_alive(self.transcript_box):
            return
        try:
            # Update live rolling subtitle strip
            if hasattr(self, "subtitle_label") and self._is_alive(self.subtitle_label):
                display_sub = raw_text.strip()
                if len(display_sub) > 130:
                    display_sub = "..." + display_sub[-127:]
                self.subtitle_label.config(text=display_sub)

            # Candidate interim speech: preview only, NEVER triggers early answer generation
            if source == "candidate_mic":
                if self._is_alive(self.live_indicator):
                    self.live_indicator.config(text="● CANDIDATE SPEAKING", fg=self.PALETTE["text_muted"])
                if self._interim_active and self.transcript_box.tag_ranges("interim_tag"):
                    try:
                        self.transcript_box.delete("interim_tag.first", "interim_text.last + 1 chars")
                    except Exception:
                        pass
                self.transcript_box.mark_set("interim_start", "end-1c")
                self.transcript_box.insert("end", "⏳ (Live) Candidate: ", "interim_tag")
                self.transcript_box.insert("end", f"{raw_text}\n", "interim_text")
                self.transcript_box.mark_set("interim_end", "end-1c")
                self._interim_active = True
                self.transcript_box.see("end")
                return

            state = likelihood_info.get("stability_state", "PARTIAL")
            state_val = state.value if hasattr(state, "value") else str(state)
            can_start_early = likelihood_info.get("can_start_early", False)
            sem_key = likelihood_info.get("semantic_key", "")
            
            # Update live indicators based on question likelihood state
            if can_start_early:
                if self._is_alive(self.live_indicator):
                    self.live_indicator.config(text="⚡ ANSWER GENERATING...", fg=self.PALETTE["accent_cyan"])
                if self._is_alive(self.understood_label):
                    self.understood_label.config(text=f"\"{raw_text}\"", fg="#f8fafc")
                self.show_answer_card()
            elif state_val in ("LIKELY_QUESTION", "STABLE_QUESTION"):
                if self._is_alive(self.live_indicator):
                    self.live_indicator.config(text="⚡ QUESTION DETECTING...", fg=self.PALETTE["accent_amber"])
                if self._is_alive(self.understood_label):
                    self.understood_label.config(text=f"\"{raw_text}\"", fg="#94a3b8")
            elif self._is_alive(self.live_indicator):
                self.live_indicator.config(text="● STREAMING", fg=self.PALETTE["accent_green"])

            # Render or update interim line in transcript_box
            if self._interim_active and self.transcript_box.tag_ranges("interim_tag"):
                try:
                    self.transcript_box.delete("interim_tag.first", "interim_text.last + 1 chars")
                except Exception:
                    pass

            self.transcript_box.mark_set("interim_start", "end-1c")
            self.transcript_box.insert("end", "● Speaking... ", "interim_tag")
            self.transcript_box.insert("end", f"{raw_text}\n", "interim_text")
            self.transcript_box.mark_set("interim_end", "end-1c")
            self._interim_active = True
            self.transcript_box.see("end")

            # Phase B: Early-Start Answer Triggering (When semantically complete without waiting for trailing silence)
            if can_start_early and self.session.auto_generate and sem_key:
                if sem_key != self.active_early_candidate_key and not self.is_streaming_answer:
                    self.active_early_candidate_key = sem_key
                    self.active_early_generation_id = getattr(self, "current_generation_id", 0) + 1
                    self.current_generation_id = self.active_early_generation_id
                    
                    pred_type = likelihood_info.get("predicted_type", "technical").upper()
                    self.active_question = raw_text
                    self.active_raw_question = raw_text
                    
                    # Check Auto-Restore if user minimized Parakit
                    if hasattr(self, "main_app") and self.main_app and getattr(self.main_app, "is_minimized", False):
                        if getattr(self.session, "auto_restore_on_question", True):
                            self.main_app.restore_window()
                            
                    self.trigger_ai_answer(
                        question=raw_text,
                        raw_transcript=raw_text,
                        question_type=pred_type,
                        source="LIVE_INTERVIEW"
                    )
        except Exception:
            pass

    def _on_transcript_chunk(self, raw_text: str, is_q_heuristic: bool = False, source: str = "system"):
        """
        Callback from continuous transcriber running in worker thread.
        Passes raw speech through the Question Understanding Layer with source attribution.
        """
        def _understand_worker():
            try:
                # 1. Run semantic classification & question understanding with source channel
                analysis = classify_and_understand_utterance(
                    utterance=raw_text,
                    session=self.session,
                    client=self.client,
                    source=source
                )
                
                # 2. Dispatch to UI for rendering & conditional answer generation
                self.safe_after(0, lambda: self._append_transcript_item_ui(analysis))
            except Exception as e:
                print(f"Transcript understanding notice: {e}")
                fallback_speaker = "Candidate" if source == "candidate_mic" else "Interviewer"
                fallback_analysis = {
                    "raw_text": raw_text,
                    "speaker": fallback_speaker,
                    "utterance_type": "STATEMENT" if fallback_speaker == "Interviewer" else "CANDIDATE_ANSWER",
                    "question_confidence": 0.3 if fallback_speaker == "Interviewer" else 0.0,
                    "should_generate_answer": False,
                    "refined_question": raw_text,
                    "context_topic": ""
                }
                self.safe_after(0, lambda: self._append_transcript_item_ui(fallback_analysis))
                
        threading.Thread(target=_understand_worker, daemon=True).start()

    def _append_transcript_item_ui(self, analysis: dict):
        """
        Appends the raw transcript to the UI while cleanly delegating
        only genuine interviewer questions to the answer generation pipeline.
        Candidate speech is captured for context without triggering answers.
        """
        if not self._is_alive(self.transcript_box):
            return
            
        try:
            # Clear any interim line before appending permanent row
            if self._interim_active:
                try:
                    if self.transcript_box.tag_ranges("interim_tag"):
                        self.transcript_box.delete("interim_tag.first", "interim_text.last + 1 chars")
                except Exception:
                    pass
                self._interim_active = False

            raw_text = analysis.get("raw_text", "").strip()
            if not raw_text:
                return
                
            # Update live rolling subtitle strip
            if hasattr(self, "subtitle_label") and self._is_alive(self.subtitle_label):
                display_sub = raw_text.strip()
                if len(display_sub) > 130:
                    display_sub = "..." + display_sub[-127:]
                self.subtitle_label.config(text=display_sub)

            speaker = analysis.get("speaker", "Interviewer")
            should_answer = analysis.get("should_generate_answer", False)
            refined_q = analysis.get("refined_question", raw_text)
            utterance_type = analysis.get("utterance_type", "STATEMENT")
            should_cancel_stale = analysis.get("should_cancel_stale", False)
            
            # 1. CANDIDATE TURN HANDLING: Render to transcript, store for context, DO NOT ANSWER
            if speaker == "Candidate":
                entry = self.session.add_transcript(raw_text, role="Candidate", is_question=False)
                t_str = entry.get('timestamp', '')
                time_display = f"{t_str} " if t_str else ""
                self.transcript_box.insert("end", time_display, "timestamp")
                self.transcript_box.insert("end", "🎙️ You\n", "speaker_candidate")
                self.transcript_box.insert("end", f"{raw_text}\n\n", "normal_text")
                self.transcript_box.see("end")
                if self._is_alive(self.live_indicator):
                    if self.transcriber.is_running:
                        self.live_indicator.config(text="● LISTENING", fg=self.PALETTE["accent_green"])
                    else:
                        self.live_indicator.config(text="⏹️ STOPPED", fg=self.PALETTE["text_muted"])
                return
                
            # 2. INTERVIEWER TURN HANDLING:
            from interview_engine import get_normalized_semantic_key
            final_sem_key = get_normalized_semantic_key(refined_q)
            
            was_early_started = False
            if self.active_early_candidate_key and final_sem_key:
                if (final_sem_key == self.active_early_candidate_key or
                    final_sem_key in self.active_early_candidate_key or
                    self.active_early_candidate_key in final_sem_key):
                    was_early_started = True

            # If not an early started continuation, increment generation id on correction/new question
            if not was_early_started and (should_cancel_stale or should_answer):
                self.current_generation_id = getattr(self, "current_generation_id", 0) + 1

            # Add to session history & render to Raw Transcript UI
            entry = self.session.add_transcript(raw_text, role="Interviewer", is_question=should_answer)
            t_str = entry.get('timestamp', '')
            time_display = f"{t_str} " if t_str else ""
            self.transcript_box.insert("end", time_display, "timestamp")
            
            if should_answer:
                tag_label = "👤 Interviewer  " if utterance_type != "FOLLOW_UP" else "👤 Interviewer  "
                q_pill = "[ Question ]\n" if utterance_type != "FOLLOW_UP" else "[ Follow-Up ]\n"
                self.transcript_box.insert("end", tag_label, "speaker_interviewer")
                self.transcript_box.insert("end", q_pill, "question_tag")
                self.transcript_box.insert("end", f"❓ {raw_text}\n\n", "question_text")
                
                # Deduplication check
                clean_key = re.sub(r'[^a-zA-Z0-9]', '', refined_q.lower())
                now = time.time()
                if clean_key in self.recent_questions_cache and (now - self.last_question_time < 5.0):
                    self.transcript_box.see("end")
                    return
                    
                self.recent_questions_cache.append(clean_key)
                self.last_question_time = now
                
                # Update Understood question HUD banner
                if self._is_alive(self.understood_label):
                    label_text = f"\"{refined_q}\""
                    if utterance_type == "FOLLOW_UP":
                        label_text = f"🔗 (Follow-Up): \"{refined_q}\""
                        
                    self.understood_label.config(
                        text=label_text,
                        fg="#ffffff",
                        font=("Segoe UI", 10, "bold")
                    )
                if hasattr(self, "question_type_badge") and self._is_alive(self.question_type_badge):
                    type_label = utterance_type.replace('_', ' ').title()
                    self.question_type_badge.config(text=f"● {type_label}", fg="#38bdf8")

                self.active_question = refined_q
                self.active_raw_question = raw_text
                self.show_answer_card()
                self._record_history(refined_q, "", utterance_type)
                
                # Check Auto-Restore if user minimized Parakit
                if hasattr(self, "main_app") and self.main_app and getattr(self.main_app, "is_minimized", False):
                    if getattr(self.session, "auto_restore_on_question", True):
                        self.main_app.restore_window()
                    else:
                        if hasattr(self.main_app, "floating_icon") and self.main_app.floating_icon:
                            self.main_app.floating_icon.set_status("answer_pending", "AI Thinking...")

                # If NOT already started early, trigger the answer pipeline now!
                if self.session.auto_generate and not was_early_started:
                    self.trigger_ai_answer(
                        question=refined_q,
                        raw_transcript=raw_text,
                        question_type=utterance_type,
                        source="LIVE_INTERVIEW"
                    )
            else:
                # Statements, greetings, acknowledgements, fillers -> Transcript Only!
                self.transcript_box.insert("end", "👤 Interviewer\n", "speaker_interviewer")
                self.transcript_box.insert("end", f"{raw_text}\n\n", "interviewer_text")
                
            self.transcript_box.see("end")
            if self._is_alive(self.live_indicator) and not self.is_streaming_answer:
                if self.transcriber.is_running:
                    self.live_indicator.config(text="● LISTENING", fg=self.PALETTE["accent_green"])
                else:
                    self.live_indicator.config(text="⏹️ STOPPED", fg=self.PALETTE["text_muted"])
        except (tk.TclError, RuntimeError, Exception):
            pass

    def clear_transcript(self):
        if not self._is_alive(self.transcript_box):
            return
        try:
            self.current_generation_id = getattr(self, "current_generation_id", 0) + 1
            self.session.clear()
            if hasattr(self, "session_controller") and self.session_controller:
                self.session_controller.reset()
            self.recent_questions_cache.clear()
            self.transcript_box.delete("1.0", "end")
            if self._is_alive(self.answer_box):
                self.answer_box.delete("1.0", "end")
            self.active_question = ""
            self.active_raw_question = ""
            if self._is_alive(self.understood_label):
                self.understood_label.config(text="Listening for interviewer question...", fg="#94a3b8", font=("Segoe UI", 9, "italic"))
            if hasattr(self, "subtitle_label") and self._is_alive(self.subtitle_label):
                self.subtitle_label.config(text="Listening for interviewer question...")
            if self._is_alive(self.answer_status_badge):
                self.answer_status_badge.config(text="● Ready", fg=self.PALETTE["accent_green"])
            self.hide_answer_card()
        except (tk.TclError, RuntimeError, Exception):
            pass

    # =========================================================
    # ANSWER GENERATION & STREAMING
    # =========================================================

    def manual_generate_answer(self):
        """Generates answer for the latest question or transcript buffer on demand."""
        target_q = self.active_question
        raw_t = self.active_raw_question
        if not target_q and self.session.transcript_log:
            # Pick the most recent utterance
            target_q = self.session.transcript_log[-1]["text"]
            raw_t = target_q
            
        if not target_q:
            self.set_answer_text("⚠️ No question or speech detected yet. Start listening or upload a question.")
            return
            
        self.trigger_ai_answer(question=target_q, raw_transcript=raw_t)

    def trigger_ai_answer(
        self,
        question: str,
        refinement: Optional[str] = None,
        raw_transcript: Optional[str] = None,
        question_type: Optional[str] = None,
        source: str = "LIVE_INTERVIEW"
    ):
        """Starts asynchronous streaming of the personalized answer with interruption handling, watchdog timeout & error isolation."""
        self.current_generation_id = getattr(self, "current_generation_id", 0) + 1
        gen_id = self.current_generation_id
        self.is_streaming_answer = True
        self.active_question = question
        self.show_answer_card()
        
        if hasattr(self, "session_controller") and self.session_controller:
            self.session_controller.start_generation(gen_id, question)
            
        # Generation Watchdog: check timeout after 25s
        self.safe_after(25000, lambda: self._check_generation_watchdog(gen_id))
        
        if self._is_alive(self.answer_box):
            try:
                self.answer_box.delete("1.0", "end")
            except Exception:
                pass
        if self._is_alive(self.answer_status_badge):
            try:
                status_txt = "⚡ Generating (Manual)..." if source == "MANUAL_INPUT" else "⚡ Generating..."
                self.answer_status_badge.config(text=status_txt, fg=self.PALETTE["accent_amber"])
            except Exception:
                pass
        
        def _worker():
            success = False
            try:
                final_answer = generate_interview_answer(
                    client=self.client,
                    session=self.session,
                    question=question,
                    on_token=self._stream_answer_token,
                    refinement_instruction=refinement,
                    raw_transcript=raw_transcript,
                    is_cancelled_callback=lambda: gen_id != getattr(self, "current_generation_id", 0),
                    question_type=question_type
                )
                if final_answer and gen_id == getattr(self, "current_generation_id", 0):
                    success = True
                    try:
                        from answer_formatter import render_structured_answer_to_text_widget
                        self.safe_after(0, lambda a=final_answer: render_structured_answer_to_text_widget(self.answer_box, a, question_type or "general") if self._is_alive(self.answer_box) else None)
                    except Exception:
                        self.safe_after(0, lambda a=final_answer: self.set_answer_text(a, question_type or "general"))
                    self._record_history(question, final_answer, question_type or "general")
            except Exception as e:
                cat, is_retryable, user_msg = categorize_api_error(e)
                print(f"Answer worker notice [{cat}]: {user_msg}")
                self.safe_after(0, lambda: self.set_answer_text(f"⚠️ {user_msg}"))
            finally:
                if gen_id == getattr(self, "current_generation_id", 0):
                    self.is_streaming_answer = False
                    if hasattr(self, "session_controller") and self.session_controller:
                        self.session_controller.end_generation(gen_id, success=success)
                    if success:
                        lat = self.session.latency_stats.get("total_sec", "")
                        lat_str = f"✓ Complete ({lat}s)" if lat else "✓ Complete"
                        self.safe_after(0, lambda: self.answer_status_badge.config(text=lat_str, fg=self.PALETTE["accent_green"]) if self._is_alive(self.answer_status_badge) else None)
                        if hasattr(self, "main_app") and self.main_app and getattr(self.main_app, "is_minimized", False):
                            if hasattr(self.main_app, "floating_icon") and self.main_app.floating_icon:
                                self.main_app.floating_icon.set_status("answer_ready", "Answer Ready! ⚡")
                    else:
                        self.safe_after(0, lambda: self.answer_status_badge.config(text="⚠️ Recovered (Listening)", fg=self.PALETTE["accent_amber"]) if self._is_alive(self.answer_status_badge) else None)
                
        threading.Thread(target=_worker, daemon=True).start()

    def _check_generation_watchdog(self, gen_id: int):
        """Watchdog check to cleanly cancel and recover hung or timed-out generation workers."""
        if gen_id == getattr(self, "current_generation_id", 0) and self.is_streaming_answer:
            if hasattr(self, "session_controller") and self.session_controller:
                self.session_controller.check_generation_timeout()
            self.is_streaming_answer = False
            if self._is_alive(self.answer_status_badge):
                self.answer_status_badge.config(text="⚠️ Timeout (Recovered)", fg=self.PALETTE["accent_amber"])
            if self._is_alive(self.answer_box) and not self.answer_box.get("1.0", "end").strip():
                self.set_answer_text("⚠️ Answer generation timed out. The session remains active and listening for the next question.")

    def _stream_answer_token(self, token: str):
        self.safe_after(0, lambda: self._append_answer_token_ui(token))

    def _append_answer_token_ui(self, token: str):
        if not self._is_alive(self.answer_box):
            return
        try:
            self.show_answer_card()
            # Auto-restore overlay on first streaming token if minimized
            current_gen = getattr(self, "current_generation_id", 0)
            if not hasattr(self, "_restored_generations"):
                self._restored_generations = set()
            if current_gen not in self._restored_generations:
                self._restored_generations.add(current_gen)
                if hasattr(self, "main_app") and self.main_app and getattr(self.main_app, "is_minimized", False):
                    if getattr(self.session, "auto_restore_on_question", True):
                        self.main_app.restore_window()

            # Check if viewport is near bottom before inserting (smooth auto-scroll)
            y_view = self.answer_box.yview()
            is_at_bottom = len(y_view) < 2 or y_view[1] >= 0.93

            self.answer_box.insert("end", token, "body")
            
            if is_at_bottom:
                self.answer_box.see("end")
        except (tk.TclError, RuntimeError, Exception):
            pass

    def set_answer_text(self, text: str, question_type: str = "general"):
        if not self._is_alive(self.answer_box):
            return
        try:
            self.show_answer_card()
            from answer_formatter import render_structured_answer_to_text_widget
            render_structured_answer_to_text_widget(self.answer_box, text, question_type)
        except Exception:
            try:
                self.answer_box.delete("1.0", "end")
                self.answer_box.insert("end", text, "body")
            except (tk.TclError, RuntimeError, Exception):
                pass

    def refine_active_answer(self, refinement_prompt: str):
        """Refines the current answer with specific format/tone."""
        if not self.active_question and not self.session.latest_answer:
            messagebox.showinfo("Interview Copilot", "No active answer to refine yet.")
            return
            
        q = self.active_question if self.active_question else (self.session.qa_pairs[-1]["question"] if self.session.qa_pairs else "Interview Question")
        self.trigger_ai_answer(q, refinement=refinement_prompt, raw_transcript=self.active_raw_question)

    def copy_answer_to_clipboard(self):
        text = self.answer_box.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.answer_status_badge.config(text="📋 Copied!", fg=self.PALETTE["accent_cyan"])
            self.root.after(1500, lambda: self.answer_status_badge.config(text="✓ Complete", fg=self.PALETTE["accent_green"]))

    # =========================================================
    # MOCK INTERVIEW MODE
    # =========================================================

    def open_mock_interview_modal(self):
        """Interactive Mock Interview Practice Drawer."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Mock Interview Simulator")
        dialog.geometry("640x700")
        dialog.configure(bg=self.PALETTE["bg"])
        dialog.attributes("-topmost", True)
        dialog.transient(self.root)

        tk.Label(
            dialog,
            text="🎓 AI Mock Interview Simulator",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text"],
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=20, pady=(16, 4))

        tk.Label(
            dialog,
            text="Practice realistic interviewer questions, speak/type your answer, and receive instant scoring & feedback.",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text_muted"],
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=20, pady=(0, 10))

        # Control Bar: Role & Difficulty
        ctrl_frame = tk.Frame(dialog, bg=self.PALETTE["bg"])
        ctrl_frame.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(ctrl_frame, text="Role:", bg=self.PALETTE["bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        role_entry = tk.Entry(ctrl_frame, bg=self.PALETTE["card_bg"], fg=self.PALETTE["text"], insertbackground=self.PALETTE["text"], relief="flat", font=("Segoe UI", 9), width=20)
        role_entry.insert(0, self.session.job_title or "Software Engineer")
        role_entry.pack(side="left", padx=(0, 12))

        tk.Label(ctrl_frame, text="Difficulty:", bg=self.PALETTE["bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        diff_var = tk.StringVar(value="Senior")
        diff_cb = ttk.Combobox(ctrl_frame, textvariable=diff_var, values=["Junior", "Mid-Level", "Senior", "Staff / Principal", "FAANG"], state="readonly", width=14)
        diff_cb.pack(side="left")

        mock_history = []

        # Question Frame
        q_frame = tk.Frame(dialog, bg=self.PALETTE["card_bg"], highlightbackground=self.PALETTE["card_border"], highlightthickness=1)
        q_frame.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(q_frame, text="INTERVIEW QUESTION:", bg=self.PALETTE["card_bg"], fg=self.PALETTE["accent_amber"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        
        q_label = tk.Label(
            q_frame,
            text="Click 'Generate Question' to begin your mock interview.",
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            font=("Segoe UI", 10, "bold"),
            wraplength=580,
            justify="left"
        )
        q_label.pack(anchor="w", padx=10, pady=(2, 10))

        def _generate_mock_q():
            q_label.config(text="Generating realistic question from your resume & target role...")
            def _w():
                from interview_engine import generate_mock_interview_question
                r = role_entry.get().strip() or "Software Engineer"
                d = diff_var.get()
                prev_q = [item["question"] for item in mock_history]
                q = generate_mock_interview_question(self.client, role=r, difficulty=d, session=self.session, previous_questions=prev_q)
                self.root.after(0, lambda: q_label.config(text=q))
            threading.Thread(target=_w, daemon=True).start()

        # Candidate Answer Entry
        tk.Label(dialog, text="YOUR SPOKEN / TYPED ANSWER:", bg=self.PALETTE["bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20, pady=(0, 2))
        ans_input = ScrolledText(
            dialog,
            height=5,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            insertbackground=self.PALETTE["text"],
            relief="flat",
            font=("Segoe UI", 10),
            wrap="word"
        )
        ans_input.pack(fill="x", padx=20, pady=(0, 10))

        # Feedback Display
        tk.Label(dialog, text="AI COACH EVALUATION & FEEDBACK:", bg=self.PALETTE["bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20, pady=(0, 2))
        feedback_box = ScrolledText(
            dialog,
            height=7,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            relief="flat",
            font=("Segoe UI", 9),
            wrap="word"
        )
        feedback_box.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        def _evaluate_answer():
            q_text = q_label.cget("text")
            a_text = ans_input.get("1.0", "end").strip()
            if not a_text or "Click" in q_text:
                messagebox.showinfo("Mock Interview", "Please generate a question and provide an answer first.")
                return
            feedback_box.delete("1.0", "end")
            feedback_box.insert("end", "Evaluating technical correctness, communication, structure, clarity...\n")
            
            def _w():
                from interview_engine import evaluate_mock_interview_answer
                d = diff_var.get()
                res = evaluate_mock_interview_answer(self.client, q_text, a_text, self.session, difficulty=d)
                mock_history.append({
                    "question": q_text,
                    "answer": a_text,
                    "evaluation": res
                })
                self.root.after(0, lambda: _show_feedback(res))
                
            threading.Thread(target=_w, daemon=True).start()

        def _show_feedback(res):
            feedback_box.delete("1.0", "end")
            score = res.get("overall_score", 80)
            feedback_box.insert("end", f"🎯 OVERALL SCORE: {score}/100\n", "score")
            feedback_box.insert("end", f"📊 Tech: {res.get('technical_correctness', 80)}/100 | Relevance: {res.get('relevance', 80)}/100 | Comm: {res.get('communication', 80)}/100 | Clarity: {res.get('clarity', 80)}/100\n\n")
            feedback_box.insert("end", f"💡 Summary: {res.get('summary_feedback', '')}\n\n")
            if res.get("strengths"):
                feedback_box.insert("end", "✅ Strengths:\n")
                for s in res.get("strengths", []):
                    feedback_box.insert("end", f"  • {s}\n")
                feedback_box.insert("end", "\n")
            if res.get("improvements"):
                feedback_box.insert("end", "⚠️ Areas to Improve:\n")
                for imp in res.get("improvements", []):
                    feedback_box.insert("end", f"  • {imp}\n")
                feedback_box.insert("end", "\n")
            if res.get("example_improved_answer"):
                feedback_box.insert("end", f"⭐ Model Answer:\n\"{res.get('example_improved_answer')}\"\n")

        def _finish_mock_report():
            if not mock_history:
                messagebox.showinfo("Mock Interview", "Please complete at least one question before viewing the final report.")
                return
            from interview_engine import generate_mock_interview_final_report
            r = role_entry.get().strip() or "Software Engineer"
            d = diff_var.get()
            rep = generate_mock_interview_final_report(self.client, self.session, mock_history, role=r, difficulty=d)
            
            rep_dialog = tk.Toplevel(dialog)
            rep_dialog.title("Mock Interview Report Card")
            rep_dialog.geometry("680x600")
            rep_dialog.configure(bg=self.PALETTE["bg"])
            rep_dialog.attributes("-topmost", True)
            
            rep_box = ScrolledText(rep_dialog, bg=self.PALETTE["card_bg"], fg=self.PALETTE["text"], relief="flat", font=("Segoe UI", 10), wrap="word")
            rep_box.pack(fill="both", expand=True, padx=20, pady=20)
            rep_box.insert("end", rep)

        # Action Buttons
        btn_row = tk.Frame(dialog, bg=self.PALETTE["bg"])
        btn_row.pack(fill="x", padx=20, pady=(0, 16))

        tk.Button(
            btn_row,
            text="🎲 Next Question",
            command=_generate_mock_q,
            bg="#1e293b",
            fg="#38bdf8",
            activebackground="#334155",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            cursor="hand2"
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_row,
            text="✓ Evaluate Answer",
            command=_evaluate_answer,
            bg=self.PALETTE["accent_green"],
            fg="#ffffff",
            activebackground="#059669",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2"
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_row,
            text="🏆 Finish & View Report",
            command=_finish_mock_report,
            bg="#4f46e5",
            fg="#ffffff",
            activebackground="#4338ca",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2"
        ).pack(side="right")

    def _stream_answer_token(self, token: str):
        self.root.after(0, lambda: self._append_answer_token_ui(token))

    def _append_answer_token_ui(self, token: str):
        self.answer_box.insert("end", token, "body")
        self.answer_box.see("end")

    def set_answer_text(self, text: str):
        self.answer_box.delete("1.0", "end")
        self.answer_box.insert("end", text, "body")

    def refine_active_answer(self, refinement_prompt: str):
        """Refines the current answer with specific format/tone."""
        if not self.active_question and not self.session.latest_answer:
            messagebox.showinfo("Interview Copilot", "No active answer to refine yet.")
            return
            
        q = self.active_question if self.active_question else (self.session.qa_pairs[-1]["question"] if self.session.qa_pairs else "Interview Question")
        self.trigger_ai_answer(q, refinement=refinement_prompt)

    def copy_answer_to_clipboard(self):
        text = self.answer_box.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.answer_status_badge.config(text="📋 Copied!", fg=self.PALETTE["accent_cyan"])
            self.root.after(1500, lambda: self.answer_status_badge.config(text="✓ Complete", fg=self.PALETTE["accent_green"]))

    # =========================================================
    # CODING INTERVIEW SCREEN PROBLEM SOLVER
    # =========================================================

    def trigger_code_screen_solver(self):
        """Captures problem from screen and streams optimal code solution."""
        self.answer_box.delete("1.0", "end")
        self.answer_status_badge.config(text="🔍 Analyzing Screen...", fg=self.PALETTE["accent_amber"])
        self.is_streaming_answer = True
        
        def _worker():
            try:
                ans = solve_screen_coding_problem(
                    client=self.client,
                    on_token=self._stream_answer_token,
                    custom_hint=f"Target role: {self.session.job_title}"
                )
                if ans:
                    try:
                        from answer_formatter import render_structured_answer_to_text_widget
                        self.safe_after(0, lambda a=ans: render_structured_answer_to_text_widget(self.answer_box, a, "coding") if self._is_alive(self.answer_box) else None)
                    except Exception:
                        pass
            except Exception as e:
                self.safe_after(0, lambda: self.set_answer_text(f"Vision coding solver error: {e}"))
            finally:
                self.is_streaming_answer = False
                self.safe_after(0, lambda: self.answer_status_badge.config(text="✓ Code Ready", fg=self.PALETTE["accent_cyan"]) if self._is_alive(self.answer_status_badge) else None)
                
        threading.Thread(target=_worker, daemon=True).start()

    def trigger_region_snip_solver(self):
        """Opens interactive drag-to-crop region selector and streams vision AI code solution."""
        self.answer_status_badge.config(text="✂️ Select Region on Screen...", fg=self.PALETTE["accent_cyan"])
        
        def _on_snip_done(image_path: str):
            self.answer_box.delete("1.0", "end")
            self.answer_status_badge.config(text="🔍 Analyzing Snipped Region...", fg=self.PALETTE["accent_amber"])
            self.is_streaming_answer = True
            
            def _worker():
                try:
                    ans = solve_screen_coding_problem(
                        client=self.client,
                        on_token=self._stream_answer_token,
                        custom_hint=f"Target role: {self.session.job_title}",
                        image_path=image_path
                    )
                    if ans:
                        try:
                            from answer_formatter import render_structured_answer_to_text_widget
                            self.safe_after(0, lambda a=ans: render_structured_answer_to_text_widget(self.answer_box, a, "coding") if self._is_alive(self.answer_box) else None)
                        except Exception:
                            pass
                except Exception as e:
                    self.safe_after(0, lambda: self.set_answer_text(f"Vision region solver error: {e}"))
                finally:
                    self.is_streaming_answer = False
                    self.safe_after(0, lambda: self.answer_status_badge.config(text="✓ Code Ready", fg=self.PALETTE["accent_cyan"]) if self._is_alive(self.answer_status_badge) else None)
                    
            threading.Thread(target=_worker, daemon=True).start()
            
        def _on_cancel():
            self.answer_status_badge.config(text="Ready", fg=self.PALETTE["text_muted"])
            
        InteractiveRegionSnipper(self.root, on_snip_completed=_on_snip_done, on_cancel=_on_cancel)

    def trigger_system_design_generator(self):
        """Generates architecture diagram, scale calculations, and trade-offs for active question or topic."""
        topic = self.active_question or self.type_input.get().strip() or "Distributed Web Architecture"
        self.answer_box.delete("1.0", "end")
        self.answer_status_badge.config(text="🏛️ Generating System Design...", fg=self.PALETTE["accent_amber"])
        self.is_streaming_answer = True
        
        def _worker():
            try:
                res = generate_system_design_diagram(
                    topic=topic,
                    client=self.client,
                    custom_requirements=f"Role: {self.session.job_title}"
                )
                
                scale = res.get("scale_estimations", {})
                trade_offs_text = "\n".join([f"• {t}" for t in res.get("key_trade_offs", [])])
                points_text = "\n".join([f"{p}" for p in res.get("interview_talking_points", [])])
                
                out_text = f"""🏛️ SYSTEM DESIGN BLUEPRINT: {res.get('topic', topic)}

============================================================
1. 📐 HIGH-LEVEL ARCHITECTURE DIAGRAM
============================================================
{res.get('ascii_diagram', '')}

============================================================
2. ⏱️ BACK-OF-THE-ENVELOPE SCALE ESTIMATIONS
============================================================
• Daily Active Users (DAU) : {scale.get('daily_active_users', '10M')}
• Read/Write Ratio         : {scale.get('read_write_ratio', '10:1')}
• Read Throughput (QPS)    : {scale.get('read_qps', '10k QPS')}
• Write Throughput (QPS)   : {scale.get('write_qps', '1k QPS')}
• Storage per Year         : {scale.get('storage_per_year', '30 TB')}

============================================================
3. ⚖️ CORE ARCHITECTURAL TRADE-OFFS & BOTTLENECKS
============================================================
{trade_offs_text}

============================================================
4. 🗣️ INTERVIEW PRESENTATION & TALKING POINTS
============================================================
{points_text}
"""
                self.root.after(0, lambda: self.set_answer_text(out_text))
            except Exception as e:
                self.root.after(0, lambda: self.set_answer_text(f"System Design generator error: {e}"))
            finally:
                self.is_streaming_answer = False
                self.root.after(0, lambda: self.answer_status_badge.config(text="✓ Architecture Ready", fg=self.PALETTE["accent_cyan"]))
                
        threading.Thread(target=_worker, daemon=True).start()

    def open_export_dialog(self):
        """Opens save file dialog to export interview session transcript, solutions, and debrief."""
        default_fname = f"Interview_Debrief_{self.session.job_title or 'Tech'}_{int(time.time())}.md"
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Interview Debrief & Solutions",
            initialfile=default_fname,
            filetypes=[
                ("Markdown Report (*.md)", "*.md"),
                ("JSON Data (*.json)", "*.json"),
                ("Text File (*.txt)", "*.txt"),
                ("All Files (*.*)", "*.*")
            ]
        )
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".json":
                export_session_to_json(self.session, file_path)
            elif ext == ".txt":
                export_session_to_text_report(self.session, file_path)
            else:
                export_session_to_markdown(self.session, file_path)
            messagebox.showinfo("Export Successful", f"Interview debrief report saved to:\n{file_path}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not export session report:\n{e}", parent=self.root)

    def trigger_star_story_generator(self):
        """Finds or generates optimal STAR method response from candidate's resume/story bank."""
        question = self.active_question or self.type_input.get().strip() or "Tell me about a time you faced a major technical challenge or conflict."
        self.answer_box.delete("1.0", "end")
        self.answer_status_badge.config(text="🌟 Structuring STAR Story...", fg=self.PALETTE["accent_amber"])
        self.is_streaming_answer = True

        def _worker():
            try:
                # 1. Check disk story bank
                stories = load_star_stories_from_disk()
                if not stories:
                    # Extract from active resume
                    resume_txt = self.session.resume_text or ""
                    if self.client and resume_txt:
                        stories = extract_star_stories_llm(resume_txt, client=self.client, custom_role=self.session.job_title)
                    else:
                        stories = extract_star_stories_heuristic(resume_txt)

                best_story = find_best_star_story(question, stories)
                formatted_ans = format_star_spoken_answer(question, best_story)
                self.root.after(0, lambda: self.set_answer_text(formatted_ans))
            except Exception as e:
                self.root.after(0, lambda: self.set_answer_text(f"STAR Story generator error: {e}"))
            finally:
                self.is_streaming_answer = False
                self.root.after(0, lambda: self.answer_status_badge.config(text="✓ STAR Ready", fg=self.PALETTE["accent_green"]))

        threading.Thread(target=_worker, daemon=True).start()

    def trigger_run_code_sandbox(self):
        """Extracts and safely executes code from the current answer box in an isolated subprocess."""
        current_text = self.answer_box.get("1.0", "end").strip()
        if not current_text:
            messagebox.showinfo("Code Sandbox", "No code in the answer pane to execute.")
            return

        blocks = extract_code_from_markdown(current_text)
        if blocks:
            target_code = blocks[0]["code"]
            lang = blocks[0]["language"]
        else:
            target_code = current_text
            lang = "python"

        self.answer_status_badge.config(text="▶️ Executing Code...", fg=self.PALETTE["accent_cyan"])

        def _worker():
            res = execute_code_snippet(target_code, language=lang, timeout_sec=4.0)
            
            def _apply():
                duration = res.get("duration_sec", 0.0)
                out_header = f"\n\n============================================================\n▶️ CODE EXECUTION RESULT ({lang.title()} • {duration}s)\n============================================================\n"
                if res.get("success"):
                    self.answer_status_badge.config(text=f"✓ Ran in {duration}s", fg=self.PALETTE["accent_green"])
                    out_body = f"STDOUT:\n{res.get('stdout') or '(Program finished with return code 0 and no output)'}\n"
                else:
                    self.answer_status_badge.config(text="⚠️ Execution Error", fg=self.PALETTE["accent_rose"])
                    err_msg = res.get("stderr") or "Unknown runtime error."
                    out_body = f"STDERR / ERROR:\n{err_msg}\n"

                self.answer_box.insert("end", out_header + out_body)
                self.answer_box.see("end")

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================
    # SETTINGS & CONTEXT MODAL
    # =========================================================

    def open_settings_modal(self):
        """Opens clean glassmorphic drawer/modal for resume, job details, and tone settings."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Job & Candidate Context Settings")
        dialog.geometry("580x690")
        dialog.configure(bg=self.PALETTE["bg"])
        dialog.attributes("-topmost", True)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Dialog Title
        title_frame = tk.Frame(dialog, bg=self.PALETTE["bg"])
        title_frame.pack(fill="x", padx=20, pady=(16, 12))
        
        tk.Label(
            title_frame,
            text="⚙️ Interview Copilot Context",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text"],
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        
        tk.Label(
            title_frame,
            text="Personalize AI answers to your target company, role, resume, and preferred tone.",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text_muted"],
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(2, 0))
        
        # Scrollable form container
        form = tk.Frame(dialog, bg=self.PALETTE["bg"])
        form.pack(fill="both", expand=True, padx=20, pady=4)
        
        # 1. Resume File Picker & Status Card
        self._create_field_label(form, "UPLOADED RESUME / CV (.PDF, .DOCX, .TXT)")
        resume_card = tk.Frame(form, bg=self.PALETTE["card_bg"], highlightbackground=self.PALETTE["card_border"], highlightthickness=1)
        resume_card.pack(fill="x", pady=(2, 12))
        
        processing_resume_path = None
        failed_resume_path = None
        
        def _format_size(num_bytes: int) -> str:
            if num_bytes < 1024:
                return f"{num_bytes} B"
            elif num_bytes < 1024 * 1024:
                return f"{num_bytes / 1024:.1f} KB"
            else:
                return f"{num_bytes / (1024 * 1024):.1f} MB"

        def _get_file_type_label(path: str) -> str:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                return "PDF"
            elif ext == ".docx":
                return "DOCX"
            elif ext in [".txt", ".md"]:
                return "TXT"
            return ext.upper().lstrip(".")

        def _render_resume_card():
            for child in resume_card.winfo_children():
                child.destroy()
                
            # STATE A: Background Processing
            if processing_resume_path:
                content_row = tk.Frame(resume_card, bg=self.PALETTE["card_bg"])
                content_row.pack(fill="x", padx=12, pady=10)
                
                fname = os.path.basename(processing_resume_path)
                fsize = ""
                try:
                    fsize = f" • {_format_size(os.path.getsize(processing_resume_path))}"
                except Exception:
                    pass
                    
                info_col = tk.Frame(content_row, bg=self.PALETTE["card_bg"])
                info_col.pack(side="left", fill="x", expand=True)
                
                tk.Label(
                    info_col,
                    text="⏳ Extracting and processing resume...",
                    bg=self.PALETTE["card_bg"],
                    fg=self.PALETTE["accent_cyan"],
                    font=("Segoe UI", 9, "bold")
                ).pack(anchor="w")
                
                tk.Label(
                    info_col,
                    text=f"{fname}{fsize}",
                    bg=self.PALETTE["card_bg"],
                    fg=self.PALETTE["text_muted"],
                    font=("Segoe UI", 9)
                ).pack(anchor="w", pady=(2, 0))
                return

            # STATE B: Active Resume Loaded (READY)
            if self.session.resume_path and os.path.exists(self.session.resume_path):
                content_row = tk.Frame(resume_card, bg=self.PALETTE["card_bg"])
                content_row.pack(fill="x", padx=12, pady=10)
                
                fname = os.path.basename(self.session.resume_path)
                ftype = _get_file_type_label(self.session.resume_path)
                fsize = ""
                try:
                    fsize = _format_size(os.path.getsize(self.session.resume_path))
                except Exception:
                    fsize = "Unknown size"
                    
                char_count = len(self.session.resume_text) if self.session.resume_text else 0
                meta_str = f"{ftype} • {fsize} • {char_count:,} characters ready"
                
                info_col = tk.Frame(content_row, bg=self.PALETTE["card_bg"])
                info_col.pack(side="left", fill="x", expand=True)
                
                status_hdr = tk.Label(
                    info_col,
                    text="✓ Resume Ready",
                    bg=self.PALETTE["card_bg"],
                    fg=self.PALETTE["accent_green"],
                    font=("Segoe UI", 9, "bold")
                )
                status_hdr.pack(anchor="w")
                
                file_lbl = tk.Label(
                    info_col,
                    text=fname,
                    bg=self.PALETTE["card_bg"],
                    fg=self.PALETTE["text"],
                    font=("Segoe UI", 10, "bold")
                )
                file_lbl.pack(anchor="w", pady=(1, 1))
                
                meta_lbl = tk.Label(
                    info_col,
                    text=meta_str,
                    bg=self.PALETTE["card_bg"],
                    fg=self.PALETTE["text_muted"],
                    font=("Segoe UI", 9)
                )
                meta_lbl.pack(anchor="w")
                
                btn_col = tk.Frame(content_row, bg=self.PALETTE["card_bg"])
                btn_col.pack(side="right", fill="y")
                
                change_btn = tk.Button(
                    btn_col,
                    text="Change File",
                    command=_handle_file_pick,
                    bg="#1e293b",
                    fg="#38bdf8",
                    activebackground="#334155",
                    activeforeground="#ffffff",
                    relief="flat",
                    borderwidth=0,
                    font=("Segoe UI", 9, "bold"),
                    padx=10,
                    pady=4,
                    cursor="hand2"
                )
                change_btn.pack(side="left", padx=(0, 6))
                
                remove_btn = tk.Button(
                    btn_col,
                    text="Remove",
                    command=_handle_remove,
                    bg="#3b1d1d",
                    fg="#fca5a5",
                    activebackground="#7f1d1d",
                    activeforeground="#ffffff",
                    relief="flat",
                    borderwidth=0,
                    font=("Segoe UI", 9, "bold"),
                    padx=10,
                    pady=4,
                    cursor="hand2"
                )
                remove_btn.pack(side="left")
                return

            # STATE C: Extraction Failed (ERROR)
            if failed_resume_path:
                content_row = tk.Frame(resume_card, bg=self.PALETTE["card_bg"])
                content_row.pack(fill="x", padx=12, pady=10)
                
                fname = os.path.basename(failed_resume_path)
                info_col = tk.Frame(content_row, bg=self.PALETTE["card_bg"])
                info_col.pack(side="left", fill="x", expand=True)
                
                tk.Label(
                    info_col,
                    text="⚠️ Resume selected, but processing failed.",
                    bg=self.PALETTE["card_bg"],
                    fg=self.PALETTE["accent_amber"],
                    font=("Segoe UI", 9, "bold")
                ).pack(anchor="w")
                
                tk.Label(
                    info_col,
                    text=f"Selected: {fname} (Could not extract readable text)",
                    bg=self.PALETTE["card_bg"],
                    fg=self.PALETTE["text_muted"],
                    font=("Segoe UI", 9)
                ).pack(anchor="w", pady=(1, 0))
                
                btn_col = tk.Frame(content_row, bg=self.PALETTE["card_bg"])
                btn_col.pack(side="right", fill="y")
                
                tk.Button(
                    btn_col,
                    text="Retry",
                    command=lambda: _process_resume_file(failed_resume_path),
                    bg="#1e293b",
                    fg="#fbbf24",
                    activebackground="#334155",
                    activeforeground="#ffffff",
                    relief="flat",
                    borderwidth=0,
                    font=("Segoe UI", 9, "bold"),
                    padx=8,
                    pady=4,
                    cursor="hand2"
                ).pack(side="left", padx=(0, 4))
                
                tk.Button(
                    btn_col,
                    text="Change File",
                    command=_handle_file_pick,
                    bg="#1e293b",
                    fg="#38bdf8",
                    activebackground="#334155",
                    activeforeground="#ffffff",
                    relief="flat",
                    borderwidth=0,
                    font=("Segoe UI", 9, "bold"),
                    padx=8,
                    pady=4,
                    cursor="hand2"
                ).pack(side="left", padx=(0, 4))
                
                tk.Button(
                    btn_col,
                    text="Remove",
                    command=_handle_remove,
                    bg="#3b1d1d",
                    fg="#fca5a5",
                    activebackground="#7f1d1d",
                    activeforeground="#ffffff",
                    relief="flat",
                    borderwidth=0,
                    font=("Segoe UI", 9, "bold"),
                    padx=8,
                    pady=4,
                    cursor="hand2"
                ).pack(side="left")
                return

            # STATE D: No Resume Loaded (EMPTY)
            content_row = tk.Frame(resume_card, bg=self.PALETTE["card_bg"])
            content_row.pack(fill="x", padx=12, pady=10)
            
            info_col = tk.Frame(content_row, bg=self.PALETTE["card_bg"])
            info_col.pack(side="left", fill="x", expand=True)
            
            tk.Label(
                info_col,
                text="No resume loaded",
                bg=self.PALETTE["card_bg"],
                fg=self.PALETTE["text_muted"],
                font=("Segoe UI", 10)
            ).pack(anchor="w")
            
            tk.Label(
                info_col,
                text="Supported: PDF, DOCX, TXT",
                bg=self.PALETTE["card_bg"],
                fg=self.PALETTE["text_subtle"],
                font=("Segoe UI", 9)
            ).pack(anchor="w", pady=(1, 0))
            
            pick_btn = tk.Button(
                content_row,
                text="Choose File",
                command=_handle_file_pick,
                bg="#1d4ed8",
                fg="#ffffff",
                activebackground="#2563eb",
                activeforeground="#ffffff",
                relief="flat",
                borderwidth=0,
                font=("Segoe UI", 9, "bold"),
                padx=12,
                pady=5,
                cursor="hand2"
            )
            pick_btn.pack(side="right")

        def _handle_file_pick():
            path = filedialog.askopenfilename(
                parent=dialog,
                title="Select Resume / CV (.PDF, .DOCX, .TXT)",
                filetypes=[
                    ("Supported Resumes (*.pdf, *.docx, *.txt)", "*.pdf;*.docx;*.txt"),
                    ("PDF Documents (*.pdf)", "*.pdf"),
                    ("Word Documents (*.docx)", "*.docx"),
                    ("Text Files (*.txt)", "*.txt"),
                    ("All Files (*.*)", "*.*")
                ]
            )
            if not path:
                return  # Clean cancellation
                
            ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}
            ext = os.path.splitext(path)[1].lower()
            if ext not in ALLOWED_EXTS:
                messagebox.showwarning(
                    "Unsupported File Type",
                    f"'{os.path.basename(path)}' is not a supported file type ({ext}).\n\nPlease select a PDF, DOCX, or TXT resume.",
                    parent=dialog
                )
                return
                
            _process_resume_file(path)

        def _process_resume_file(path: str):
            nonlocal processing_resume_path, failed_resume_path
            processing_resume_path = path
            failed_resume_path = None
            _render_resume_card()
            
            def _worker():
                success = self.session.set_resume(path)
                def _on_finish():
                    nonlocal processing_resume_path, failed_resume_path
                    processing_resume_path = None
                    if success and self.session.resume_text:
                        failed_resume_path = None
                        fname = os.path.basename(path)
                        disp_name = f"📄 Resume: {fname[:18]}..." if len(fname) > 18 else f"📄 Resume: {fname}"
                        self.resume_badge.config(text=disp_name, fg=self.PALETTE["accent_green"])
                    else:
                        failed_resume_path = path
                        self.resume_badge.config(text="📄 Resume: Error", fg=self.PALETTE["accent_amber"])
                    _render_resume_card()
                dialog.after(0, _on_finish)
                
            threading.Thread(target=_worker, daemon=True).start()

        def _handle_remove():
            nonlocal processing_resume_path, failed_resume_path
            processing_resume_path = None
            failed_resume_path = None
            self.session.remove_resume()
            self.resume_badge.config(text="📄 Resume: Not loaded", fg=self.PALETTE["text_muted"])
            _render_resume_card()

        _render_resume_card()
        
        # 2. Target Job Title
        self._create_field_label(form, "TARGET JOB TITLE")
        job_entry = tk.Entry(
            form,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            insertbackground=self.PALETTE["text"],
            relief="flat",
            highlightbackground=self.PALETTE["card_border"],
            highlightthickness=1,
            font=("Segoe UI", 10)
        )
        job_entry.pack(fill="x", pady=(2, 10), ipady=4)
        job_entry.insert(0, self.session.job_title)
        
        # 3. Target Company
        self._create_field_label(form, "TARGET COMPANY NAME")
        comp_entry = tk.Entry(
            form,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            insertbackground=self.PALETTE["text"],
            relief="flat",
            highlightbackground=self.PALETTE["card_border"],
            highlightthickness=1,
            font=("Segoe UI", 10)
        )
        comp_entry.pack(fill="x", pady=(2, 10), ipady=4)
        comp_entry.insert(0, self.session.company_name)
        
        # 4. Job Description Textarea
        self._create_field_label(form, "JOB DESCRIPTION / KEY REQUIREMENTS (OPTIONAL)")
        jd_box = ScrolledText(
            form,
            height=4,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            insertbackground=self.PALETTE["text"],
            relief="flat",
            highlightbackground=self.PALETTE["card_border"],
            highlightthickness=1,
            font=("Segoe UI", 9),
            wrap="word"
        )
        jd_box.pack(fill="x", pady=(2, 10))
        jd_box.insert("1.0", self.session.job_description)
        
        # 5. Tone & Language Dropdowns in 2 columns
        dropdown_row = tk.Frame(form, bg=self.PALETTE["bg"])
        dropdown_row.pack(fill="x", pady=(2, 10))
        dropdown_row.columnconfigure(0, weight=1)
        dropdown_row.columnconfigure(1, weight=1)
        
        # Tone
        tone_frame = tk.Frame(dropdown_row, bg=self.PALETTE["bg"])
        tone_frame.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._create_field_label(tone_frame, "INTERVIEW TONE")
        tone_var = tk.StringVar(value=self.session.tone)
        tone_cb = ttk.Combobox(
            tone_frame,
            textvariable=tone_var,
            values=[
                "Confident & Conversational",
                "Deep Technical / Senior",
                "Executive / Leadership",
                "Simple & Direct English",
                "Formal & Academic"
            ],
            state="readonly"
        )
        tone_cb.pack(fill="x", pady=2)
        
        # Language
        lang_frame = tk.Frame(dropdown_row, bg=self.PALETTE["bg"])
        lang_frame.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self._create_field_label(lang_frame, "ANSWER LANGUAGE")
        lang_var = tk.StringVar(value=self.session.language)
        lang_cb = ttk.Combobox(
            lang_frame,
            textvariable=lang_var,
            values=[
                "English",
                "Spanish",
                "French",
                "German",
                "Hindi",
                "Portuguese",
                "Mandarin"
            ],
            state="readonly"
        )
        lang_cb.pack(fill="x", pady=2)

        # 6. Auto-Restore on New Question Toggle
        self._create_field_label(form, "AUTO-RESTORE ON QUESTION (WHEN MINIMIZED)")
        auto_res_row = tk.Frame(form, bg=self.PALETTE["card_bg"], highlightbackground=self.PALETTE["card_border"], highlightthickness=1)
        auto_res_row.pack(fill="x", pady=(2, 10), ipady=4)

        auto_res_var = tk.BooleanVar(value=self.session.auto_restore_on_question)

        auto_res_lbl = tk.Label(
            auto_res_row,
            text="Auto-restore overlay when question arrives:",
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            font=("Segoe UI", 9)
        )
        auto_res_lbl.pack(side="left", padx=10)

        def _toggle_auto_res():
            new_val = not auto_res_var.get()
            auto_res_var.set(new_val)
            _update_auto_res_ui()

        def _update_auto_res_ui():
            if auto_res_var.get():
                auto_res_toggle_btn.config(text="ON  (Auto-Expands)", bg="#064e3b", fg="#34d399")
            else:
                auto_res_toggle_btn.config(text="OFF (Stays in Floating Icon)", bg="#1e293b", fg="#94a3b8")

        auto_res_toggle_btn = tk.Button(
            auto_res_row,
            text="",
            command=_toggle_auto_res,
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=10,
            pady=3,
            cursor="hand2"
        )
        auto_res_toggle_btn.pack(side="right", padx=10)
        _update_auto_res_ui()

        # Save Button
        def _save_settings():
            self.session.job_title = job_entry.get().strip()
            self.session.company_name = comp_entry.get().strip()
            self.session.job_description = jd_box.get("1.0", "end").strip()
            self.session.tone = tone_var.get()
            self.session.language = lang_var.get()
            self.session.set_auto_restore(auto_res_var.get())
            if hasattr(self, "main_app") and self.main_app:
                self.main_app.auto_restore_on_question = self.session.auto_restore_on_question
            
            role_text = f"🎯 Target: {self.session.job_title or 'Tech Role'}"
            if self.session.company_name:
                role_text += f" @ {self.session.company_name}"
            self.role_badge.config(text=role_text)
            
            dialog.destroy()

        save_btn = tk.Button(
            dialog,
            text="✓ Save Context & Continue",
            command=_save_settings,
            bg=self.PALETTE["accent_green"],
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=8,
            cursor="hand2"
        )
        save_btn.pack(fill="x", padx=20, pady=(4, 16))

    def _create_field_label(self, parent, text):
        tk.Label(
            parent,
            text=text,
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text_muted"],
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w")

    # =========================================================
    # SESSION SUMMARY MODAL
    # =========================================================

    def show_session_summary_modal(self):
        """Generates and displays executive interview summary with copy/save options."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Interview Summary & Key Takeaways")
        dialog.geometry("640x700")
        dialog.configure(bg=self.PALETTE["bg"])
        dialog.attributes("-topmost", True)
        dialog.transient(self.root)
        
        title_frame = tk.Frame(dialog, bg=self.PALETTE["bg"])
        title_frame.pack(fill="x", padx=20, pady=(16, 10))
        
        tk.Label(
            title_frame,
            text="🏁 Executive Interview Summary",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text"],
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        
        status_lbl = tk.Label(
            title_frame,
            text="🤖 Synthesizing interview answers and action items...",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["accent_amber"],
            font=("Segoe UI", 9)
        )
        status_lbl.pack(anchor="w", pady=(2, 0))
        
        summary_box = ScrolledText(
            dialog,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text"],
            insertbackground=self.PALETTE["text"],
            relief="flat",
            highlightbackground=self.PALETTE["card_border"],
            highlightthickness=1,
            font=("Segoe UI", 10),
            wrap="word",
            padx=14,
            pady=12
        )
        summary_box.pack(fill="both", expand=True, padx=20, pady=8)
        
        # Bottom button row
        btn_row = tk.Frame(dialog, bg=self.PALETTE["bg"])
        btn_row.pack(fill="x", padx=20, pady=(8, 16))
        
        def _copy_summary():
            content = summary_box.get("1.0", "end").strip()
            if content:
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                messagebox.showinfo("Copied", "Interview summary copied to clipboard!")
                
        def _save_summary():
            content = summary_box.get("1.0", "end").strip()
            if not content:
                return
            path = filedialog.asksaveasfilename(
                title="Save Interview Summary",
                defaultextension=".md",
                filetypes=[("Markdown", "*.md"), ("Text Files", "*.txt")]
            )
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Saved", f"Interview summary saved to:\n{path}")
                
        copy_btn = tk.Button(
            btn_row,
            text="📋 Copy Notes",
            command=_copy_summary,
            bg=self.PALETTE["btn_bg"],
            fg=self.PALETTE["accent_cyan"],
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            cursor="hand2"
        )
        copy_btn.pack(side="left", padx=(0, 8))
        
        save_btn = tk.Button(
            btn_row,
            text="💾 Save Markdown",
            command=_save_summary,
            bg=self.PALETTE["btn_bg"],
            fg=self.PALETTE["text"],
            relief="flat",
            font=("Segoe UI", 9),
            padx=12,
            pady=6,
            cursor="hand2"
        )
        save_btn.pack(side="left")
        
        close_btn = tk.Button(
            btn_row,
            text="Close",
            command=dialog.destroy,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text_muted"],
            relief="flat",
            font=("Segoe UI", 9),
            padx=12,
            pady=6,
            cursor="hand2"
        )
        close_btn.pack(side="right")
        
        # Generate summary in thread
        def _generate_report():
            report = generate_session_summary(self.client, self.session)
            def _update():
                summary_box.delete("1.0", "end")
                summary_box.insert("end", report)
                status_lbl.config(text="✓ Session report ready", fg=self.PALETTE["accent_green"])
            self.root.after(0, _update)
            
        threading.Thread(target=_generate_report, daemon=True).start()

    # =========================================================
    # AI MOCK INTERVIEWER & SIMULATION MODAL (FEATURE 9)
    # =========================================================

    def open_mock_interview_modal(self):
        """
        Launches the AI Mock Interview simulation dialog with customizable role,
        difficulty, persona, dynamic questioning, and comprehensive scorecard.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("AI Mock Interviewer & Simulation Practice")
        dialog.geometry("700x740")
        dialog.configure(bg=self.PALETTE["bg"])
        dialog.attributes("-topmost", True)
        dialog.transient(self.root)

        header = tk.Frame(dialog, bg=self.PALETTE["bg"])
        header.pack(fill="x", padx=24, pady=(20, 10))

        tk.Label(
            header,
            text="🎓 AI Mock Interview Simulator",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text"],
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Practice live verbal answering with an adaptive AI hiring manager and receive an instant rubric scorecard.",
            bg=self.PALETTE["bg"],
            fg=self.PALETTE["text_muted"],
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(2, 0))

        main_scroll = tk.Frame(dialog, bg=self.PALETTE["bg"])
        main_scroll.pack(fill="both", expand=True, padx=24, pady=8)

        # Simulation Setup Frame
        setup_frame = tk.Frame(main_scroll, bg=self.PALETTE["card_bg"], highlightbackground=self.PALETTE["card_border"], highlightthickness=1)
        setup_frame.pack(fill="x", pady=(0, 12), padx=2, ipady=8)

        # 1. Role Track
        tk.Label(setup_frame, text="ROLE / DOMAIN TRACK", bg=self.PALETTE["card_bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(10, 2))
        role_var = tk.StringVar(value="backend")
        role_cb = ttk.Combobox(
            setup_frame,
            textvariable=role_var,
            values=list(ROLE_TRACKS.keys()),
            state="readonly",
            font=("Segoe UI", 10)
        )
        role_cb.pack(fill="x", padx=16, pady=(0, 8))

        # 2. Difficulty & Questions
        row2 = tk.Frame(setup_frame, bg=self.PALETTE["card_bg"])
        row2.pack(fill="x", padx=16, pady=(0, 8))
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        diff_col = tk.Frame(row2, bg=self.PALETTE["card_bg"])
        diff_col.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(diff_col, text="DIFFICULTY LEVEL", bg=self.PALETTE["card_bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 2))
        diff_var = tk.StringVar(value="senior")
        diff_cb = ttk.Combobox(diff_col, textvariable=diff_var, values=list(DIFFICULTY_LEVELS.keys()), state="readonly")
        diff_cb.pack(fill="x")

        q_col = tk.Frame(row2, bg=self.PALETTE["card_bg"])
        q_col.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(q_col, text="QUESTIONS COUNT", bg=self.PALETTE["card_bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 2))
        q_count_var = tk.StringVar(value="3")
        q_count_cb = ttk.Combobox(q_col, textvariable=q_count_var, values=["3", "5", "7"], state="readonly")
        q_count_cb.pack(fill="x")

        # 3. Interviewer Persona
        tk.Label(setup_frame, text="INTERVIEWER PERSONA", bg=self.PALETTE["card_bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(4, 2))
        persona_var = tk.StringVar(value="bar_raiser")
        persona_cb = ttk.Combobox(setup_frame, textvariable=persona_var, values=list(PERSONAS.keys()), state="readonly")
        persona_cb.pack(fill="x", padx=16, pady=(0, 10))

        # Simulation Runtime Area
        runtime_frame = tk.Frame(main_scroll, bg=self.PALETTE["bg"])
        runtime_frame.pack(fill="both", expand=True, pady=(4, 0))

        interviewer_box = ScrolledText(
            runtime_frame,
            height=6,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["accent_cyan"],
            insertbackground=self.PALETTE["text"],
            relief="flat",
            highlightbackground=self.PALETTE["card_border"],
            highlightthickness=1,
            font=("Segoe UI", 10),
            wrap="word",
            padx=12,
            pady=10
        )
        interviewer_box.pack(fill="x", pady=(0, 8))
        interviewer_box.insert("1.0", "💡 Configure your mock interview preferences above and click [▶ Start Practice Simulation] to begin.")

        tk.Label(runtime_frame, text="YOUR SPOKEN / TYPED ANSWER:", bg=self.PALETTE["bg"], fg=self.PALETTE["text_muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(2, 2))
        answer_input = ScrolledText(
            runtime_frame,
            height=6,
            bg="#0d111c",
            fg=self.PALETTE["text"],
            insertbackground=self.PALETTE["text"],
            relief="flat",
            highlightbackground="#2563eb",
            highlightthickness=1,
            font=("Segoe UI", 10),
            wrap="word",
            padx=12,
            pady=10
        )
        answer_input.pack(fill="both", expand=True, pady=(0, 8))

        # Action Buttons Row
        action_bar = tk.Frame(dialog, bg=self.PALETTE["bg"])
        action_bar.pack(fill="x", padx=24, pady=(8, 16))

        mock_session_inst: List[Optional[MockInterviewSession]] = [None]

        def _start_sim():
            try:
                target_q = int(q_count_var.get())
            except Exception:
                target_q = 3
            mock_session_inst[0] = MockInterviewSession(
                role=role_var.get(),
                difficulty=diff_var.get(),
                persona=persona_var.get(),
                target_questions=target_q,
                resume_text=self.session.resume_text,
                jd_text=self.session.job_description
            )
            first_item = mock_session_inst[0].generate_first_question()
            interviewer_box.delete("1.0", "end")
            interviewer_box.insert("end", first_item["interviewer_text"])
            answer_input.delete("1.0", "end")
            start_btn.config(text="🔄 Restart Simulation")
            submit_btn.config(state="normal")

        def _submit_ans():
            if not mock_session_inst[0]:
                return
            cand_ans = answer_input.get("1.0", "end").strip()
            if not cand_ans:
                messagebox.showwarning("Input Required", "Please speak or type your answer before submitting.")
                return

            res = mock_session_inst[0].submit_answer(cand_ans)
            answer_input.delete("1.0", "end")

            if res["is_completed"]:
                sc = res["scorecard"]
                interviewer_box.delete("1.0", "end")
                report_lines = [
                    "══════════════════════════════════════════════════",
                    f"🏆 MOCK INTERVIEW SCORECARD: {sc['overall_score']}/100",
                    f"⭐ Recommendation: {sc['recommendation']}",
                    "══════════════════════════════════════════════════\n",
                    f"📋 Summary: {sc['summary']}\n",
                    "📊 Category Breakdown:"
                ]
                for cat, val in sc["category_scores"].items():
                    report_lines.append(f"  • {cat}: {val}/100")
                report_lines.append("\n✅ Key Strengths:")
                for st in sc["strengths"]:
                    report_lines.append(f"  + {st}")
                report_lines.append("\n🚀 Improvement Areas:")
                for imp in sc["improvement_areas"]:
                    report_lines.append(f"  - {imp}")

                interviewer_box.insert("end", "\n".join(report_lines))
                submit_btn.config(state="disabled")
            else:
                interviewer_box.delete("1.0", "end")
                nxt = res["next_question"]
                interviewer_box.insert("end", f"Previous Answer Score: {res['evaluation']['score']}/100\nFeedback: {res['evaluation']['feedback']}\n\n{nxt['interviewer_text']}")

        start_btn = tk.Button(
            action_bar,
            text="▶ Start Practice Simulation",
            command=_start_sim,
            bg=self.PALETTE["accent_green"],
            fg="#ffffff",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=6,
            cursor="hand2"
        )
        start_btn.pack(side="left", padx=(0, 8))

        submit_btn = tk.Button(
            action_bar,
            text="🎤 Submit Answer →",
            command=_submit_ans,
            state="disabled",
            bg=self.PALETTE["btn_bg"],
            fg=self.PALETTE["accent_cyan"],
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=6,
            cursor="hand2"
        )
        submit_btn.pack(side="left")

        close_btn = tk.Button(
            action_bar,
            text="Close",
            command=dialog.destroy,
            bg=self.PALETTE["card_bg"],
            fg=self.PALETTE["text_muted"],
            relief="flat",
            font=("Segoe UI", 9),
            padx=12,
            pady=6,
            cursor="hand2"
        )
        close_btn.pack(side="right")

