"""
=============================================================================
BluJay AI - Next-Gen Call Sessions Dashboard & Window Orchestrator
=============================================================================
Ultra-Modern, Dark Glassmorphic UI matching top-tier SaaS standards (Parakeet / Linear / Raycast):
 - Deep Obsidian & Titanium dark canvas (#080C14).
 - Modern sidebar navigation with luminous active indicator pills.
 - Sleek Pro / Free Plan card with live battery progress meter.
 - Quick-Start Template Presets with custom icons and glowing hover borders.
 - Fast Search & Filter Bar (All, Active, Completed).
 - Drag-and-Drop Resume RAG manager.
 - Stealth Screen Capture Exclusion (WDA_EXCLUDEFROMCAPTURE).
=============================================================================
"""

import os
import sys
import threading
from typing import Optional, List, Dict, Any
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import ctypes
from session_client import client as session_client
from audio_device_manager import (
    get_available_microphones,
    get_available_speakers,
    get_saved_device_preferences,
    save_device_preferences,
    AudioLevelProbe
)
from vision import (
    get_available_monitors,
    get_saved_monitor_preference,
    save_monitor_preference
)

# =============================================================================
# HIGH-DPI AWARENESS (WINDOWS)
# =============================================================================
try:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# =============================================================================
# SCREEN CAPTURE EXCLUSION (STEALTH OVERLAY PROTECTION)
# =============================================================================
from ctypes import wintypes

GA_ROOT = 2
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011

user32 = ctypes.windll.user32 if sys.platform == "win32" else None
SetWindowDisplayAffinity = getattr(user32, "SetWindowDisplayAffinity", None) if user32 else None
GetAncestor = getattr(user32, "GetAncestor", None) if user32 else None
GetParent = getattr(user32, "GetParent", None) if user32 else None


def get_root_hwnd(window_id):
    """Resolve true OS top-level HWND from Tkinter / CustomTkinter window id."""
    if not window_id:
        return None
    try:
        if GetAncestor:
            root_h = GetAncestor(wintypes.HWND(window_id), wintypes.UINT(GA_ROOT))
            if root_h:
                return root_h
        if GetParent:
            parent_h = GetParent(wintypes.HWND(window_id))
            if parent_h:
                return parent_h
    except Exception:
        pass
    return window_id


def exclude_from_screen_capture(hwnd):
    """
    Excludes the window from all screen sharing (Zoom, Google Meet, Microsoft Teams,
    Discord, Slack, OBS, Windows Game Bar, screenshots, Snipping Tool, etc.).
    The window remains 100% visible and interactive on the user's screen.
    """
    if SetWindowDisplayAffinity is None or not hwnd:
        return False

    actual_hwnd = get_root_hwnd(hwnd) or hwnd

    try:
        # 1. Primary: WDA_EXCLUDEFROMCAPTURE (0x11) - Windows 10 2004+ / Windows 11
        result = SetWindowDisplayAffinity(
            wintypes.HWND(actual_hwnd),
            wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE)
        )
        if result:
            return True

        # 2. Fallback: WDA_MONITOR (0x01) for legacy Windows builds
        result = SetWindowDisplayAffinity(
            wintypes.HWND(actual_hwnd),
            wintypes.DWORD(WDA_MONITOR)
        )
        return bool(result)

    except Exception as error:
        print("Screen capture exclusion note:", error)
        return False


# Configure CustomTkinter Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# =============================================================================
# ULTRA-MODERN DARK GLASS PALETTE (Obsidian Slate & Electric Accents)
# =============================================================================
PALETTE = {
    # Surfaces
    "bg_main": "#080C14",
    "bg_sidebar": "#0D1322",
    "sidebar_border": "#1E293B",
    "bg_subtle": "#111827",
    "bg_subtle_hover": "#1E293B",
    "bg_card": "#111827",
    "bg_card_hover": "#1F293D",

    # Borders
    "border_subtle": "#1E293B",
    "border_card": "#1F293D",
    "border_focus": "#38BDF8",

    # Typography
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "text_subtle": "#475569",

    # Brand Accent (Electric Cyan & Emerald Green)
    "brand_green": "#10B981",
    "brand_green_hover": "#059669",
    "brand_green_bg": "#064E3B",
    "brand_green_border": "#059669",
    "brand_green_text": "#34D399",

    "accent_cyan": "#38BDF8",
    "accent_blue": "#2563EB",
    "accent_blue_hover": "#1D4ED8",
    "accent_rose": "#F43F5E",
    "accent_rose_hover": "#E11D48",
    "accent_amber": "#F59E0B",
    "accent_amber_hover": "#D97706",

    # Nav Selection
    "nav_active_bg": "#1E293B",
    "nav_active_border": "#38BDF8",
    "nav_active_text": "#F8FAFC",
    "nav_inactive_text": "#94A3B8",
    "nav_hover_bg": "#172033",

    # Buttons
    "btn_dark": "#2563EB",
    "btn_dark_hover": "#1D4ED8",
    "btn_secondary_bg": "#1E293B",
    "btn_secondary_border": "#334155",
    "btn_secondary_hover": "#334155",
}


# =============================================================================
# BUY CREDITS / UPGRADE MODAL
# =============================================================================
class BuyCreditsDialog(ctk.CTkToplevel):
    def __init__(self, parent, on_purchase_success=None):
        super().__init__(parent)
        self.parent = parent
        self.on_purchase_success = on_purchase_success

        self.title("Upgrade Plan - BluJay AI")
        self.geometry("580x620")
        self.configure(fg_color=PALETTE["bg_sidebar"])
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()
        self._center_window(parent)
        self._build_ui()
        self._apply_stealth_capture()

    def _apply_stealth_capture(self):
        try:
            self.update_idletasks()
            raw_id = self.winfo_id()
            hwnd = get_root_hwnd(raw_id) or raw_id
            exclude_from_screen_capture(hwnd)
        except Exception:
            pass

    def _center_window(self, parent):
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        w, h = 580, 620
        x = px + max(0, (pw - w) // 2)
        y = py + max(0, (ph - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        pad = 28

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=pad, pady=(24, 16))

        ctk.CTkLabel(
            header,
            text="✨ Upgrade Your BluJay AI Plan",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Get unlimited full-length live AI copilot calls, screen vision solvers, and resume RAG.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(4, 0))

        # Products list
        cards_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        cards_scroll.pack(fill="both", expand=True, padx=pad, pady=(0, 20))

        products = [
            {
                "id": "starter_pack",
                "name": "Starter 5-Pack",
                "price": "$19",
                "credits": "5.0 Credits",
                "sessions": "5 full-length interview sessions",
                "tag": "Popular",
                "featured": False
            },
            {
                "id": "pro_bundle",
                "name": "Pro Interview Pass",
                "price": "$49",
                "credits": "15.0 Credits",
                "sessions": "15 full sessions + Vision OCR + RAG",
                "tag": "Best Value",
                "featured": True
            },
            {
                "id": "unlimited_season",
                "name": "Unlimited Season Pass",
                "price": "$99",
                "credits": "50.0 Credits",
                "sessions": "50 full sessions + Priority AI Latency",
                "tag": "Full Access",
                "featured": False
            }
        ]

        for p in products:
            card = ctk.CTkFrame(
                cards_scroll,
                fg_color=PALETTE["bg_card"],
                corner_radius=10,
                border_color=PALETTE["brand_green"] if p["featured"] else PALETTE["border_card"],
                border_width=2 if p["featured"] else 1
            )
            card.pack(fill="x", pady=6, ipady=4)

            left_box = ctk.CTkFrame(card, fg_color="transparent")
            left_box.pack(side="left", padx=16, pady=10)

            if p["tag"]:
                badge = ctk.CTkFrame(left_box, fg_color="#064E3B", corner_radius=4)
                badge.pack(anchor="w", pady=(0, 2))
                ctk.CTkLabel(badge, text=p["tag"], font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"), text_color="#34D399").pack(padx=6, pady=1)

            ctk.CTkLabel(left_box, text=p["name"], font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color=PALETTE["text_primary"]).pack(anchor="w")
            ctk.CTkLabel(left_box, text=f"{p['credits']} • {p['sessions']}", font=ctk.CTkFont(family="Segoe UI", size=11), text_color=PALETTE["text_secondary"]).pack(anchor="w")

            right_box = ctk.CTkFrame(card, fg_color="transparent")
            right_box.pack(side="right", padx=16, pady=10)

            ctk.CTkLabel(right_box, text=p["price"], font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"), text_color=PALETTE["text_primary"]).pack(side="left", padx=(0, 14))

            ctk.CTkButton(
                right_box,
                text="Purchase",
                command=lambda pid=p["id"]: self._buy(pid),
                width=90,
                height=32,
                corner_radius=6,
                fg_color=PALETTE["brand_green"] if p["featured"] else PALETTE["btn_dark"],
                hover_color=PALETTE["brand_green_hover"] if p["featured"] else PALETTE["btn_dark_hover"],
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
            ).pack(side="left")

    def _buy(self, product_id: str):
        success, msg = session_client.mock_buy_product(product_id)
        if success:
            messagebox.showinfo("Purchase Complete", f"Success! {msg}")
            if self.on_purchase_success:
                self.on_purchase_success()
            self.destroy()
        else:
            messagebox.showerror("Purchase Failed", msg)


# =============================================================================
# CANDIDATE PROFILE & PREFERENCES MODAL
# =============================================================================
class UserProfileDialog(ctk.CTkToplevel):
    def __init__(self, parent, on_save_callback=None, current_profile=None):
        super().__init__(parent)
        self.parent = parent
        self.on_save_callback = on_save_callback
        self.profile = current_profile or {
            "name": "Abhishek Salunke",
            "email": "abhisheksalunke7387@gmail.com",
            "role": "Senior Full Stack & AI Engineer",
            "skills": "Python, Django, FastAPI, React, PostgreSQL, Docker, RAG, LLM Copilots",
            "tone": "Direct & Highly Technical (Recommended)"
        }

        self.title("Candidate Profile & AI Grounding - BluJay AI")
        self.geometry("580x660")
        self.configure(fg_color=PALETTE["bg_sidebar"])
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()
        self._center_window(parent)
        self._build_ui()
        self._apply_stealth_capture()

    def _apply_stealth_capture(self):
        try:
            self.update_idletasks()
            raw_id = self.winfo_id()
            hwnd = get_root_hwnd(raw_id) or raw_id
            exclude_from_screen_capture(hwnd)
        except Exception:
            pass

    def _center_window(self, parent):
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        w, h = 580, 660
        x = px + max(0, (pw - w) // 2)
        y = py + max(0, (ph - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        pad = 28

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=pad, pady=(24, 12))

        # Avatar + Title
        top_row = ctk.CTkFrame(header, fg_color="transparent")
        top_row.pack(fill="x")

        avatar_box = ctk.CTkFrame(top_row, fg_color="#1E293B", width=44, height=44, corner_radius=22, border_color="#38BDF8", border_width=1)
        avatar_box.pack(side="left", padx=(0, 12))
        avatar_box.pack_propagate(False)
        ctk.CTkLabel(avatar_box, text="AS", font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"), text_color="#38BDF8").place(relx=0.5, rely=0.5, anchor="center")

        title_info = ctk.CTkFrame(top_row, fg_color="transparent")
        title_info.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            title_info,
            text="Candidate Profile & Preferences",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_info,
            text="Ground real-time answers in your verified background and persona.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(1, 0))

        # Scrollable form
        scroll_form = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_form.pack(fill="both", expand=True, padx=pad, pady=(0, 16))

        # 1. Full Name
        ctk.CTkLabel(scroll_form, text="Full Name", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=PALETTE["text_primary"]).pack(anchor="w", pady=(6, 2))
        self.name_entry = ctk.CTkEntry(scroll_form, height=36, corner_radius=8, border_color=PALETTE["border_card"], fg_color=PALETTE["bg_card"], text_color=PALETTE["text_primary"], font=ctk.CTkFont(family="Segoe UI", size=11))
        self.name_entry.pack(fill="x", pady=(0, 10))
        self.name_entry.insert(0, self.profile.get("name", "Abhishek Salunke"))

        # 2. Email Address
        ctk.CTkLabel(scroll_form, text="Email Address", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=PALETTE["text_primary"]).pack(anchor="w", pady=(0, 2))
        self.email_entry = ctk.CTkEntry(scroll_form, height=36, corner_radius=8, border_color=PALETTE["border_card"], fg_color=PALETTE["bg_card"], text_color=PALETTE["text_primary"], font=ctk.CTkFont(family="Segoe UI", size=11))
        self.email_entry.pack(fill="x", pady=(0, 10))
        self.email_entry.insert(0, self.profile.get("email", "abhisheksalunke7387@gmail.com"))

        # 3. Target Primary Title
        ctk.CTkLabel(scroll_form, text="Primary Engineering Role", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=PALETTE["text_primary"]).pack(anchor="w", pady=(0, 2))
        self.role_entry = ctk.CTkEntry(scroll_form, height=36, corner_radius=8, border_color=PALETTE["border_card"], fg_color=PALETTE["bg_card"], text_color=PALETTE["text_primary"], font=ctk.CTkFont(family="Segoe UI", size=11))
        self.role_entry.pack(fill="x", pady=(0, 10))
        self.role_entry.insert(0, self.profile.get("role", "Senior Full Stack & AI Engineer"))

        # 4. Core Verified Tech Stack
        ctk.CTkLabel(scroll_form, text="Verified Core Tech Stack (Used for Anti-Hallucination)", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=PALETTE["text_primary"]).pack(anchor="w", pady=(0, 2))
        self.skills_entry = ctk.CTkEntry(scroll_form, height=36, corner_radius=8, border_color=PALETTE["border_card"], fg_color=PALETTE["bg_card"], text_color=PALETTE["text_primary"], font=ctk.CTkFont(family="Segoe UI", size=11))
        self.skills_entry.pack(fill="x", pady=(0, 10))
        self.skills_entry.insert(0, self.profile.get("skills", "Python, Django, FastAPI, React, PostgreSQL, Docker, RAG, LLM Copilots"))

        # 5. Preferred Tone
        ctk.CTkLabel(scroll_form, text="Preferred Answer Delivery Style", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=PALETTE["text_primary"]).pack(anchor="w", pady=(0, 2))
        self.tone_opt = ctk.CTkOptionMenu(
            scroll_form,
            values=["Direct & Highly Technical (Recommended)", "STAR Behavioral Framework", "Executive & Concise (Under 60s)", "ELI5 / Plain English"],
            height=36,
            corner_radius=8,
            fg_color=PALETTE["bg_card"],
            button_color=PALETTE["btn_secondary_border"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.tone_opt.pack(fill="x", pady=(0, 14))
        self.tone_opt.set(self.profile.get("tone", "Direct & Highly Technical (Recommended)"))

        # Live Security & Engine Badges
        sec_card = ctk.CTkFrame(scroll_form, fg_color=PALETTE["bg_card"], corner_radius=8, border_color=PALETTE["border_card"], border_width=1)
        sec_card.pack(fill="x", pady=(0, 10), ipady=6)

        ctk.CTkLabel(sec_card, text="🛡️ System Security & Latency Status", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=12, pady=(6, 4))
        ctk.CTkLabel(sec_card, text="• Windows WDA_EXCLUDEFROMCAPTURE: 100% Screen Share Stealth ON\n• Low-Latency Groq LPU Pipeline: Sub-500ms TTFT Connected\n• Resume Document RAG: Local Vector Isolation Active", font=ctk.CTkFont(family="Segoe UI", size=10), text_color=PALETTE["text_secondary"], justify="left").pack(anchor="w", padx=12, pady=(0, 6))

        # Footer Actions
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=pad, pady=(0, 20))

        ctk.CTkButton(
            footer,
            text="Close",
            command=self.destroy,
            width=90,
            height=36,
            corner_radius=8,
            fg_color=PALETTE["btn_secondary_bg"],
            hover_color=PALETTE["bg_subtle_hover"],
            text_color=PALETTE["text_secondary"],
            border_color=PALETTE["border_card"],
            border_width=1,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        ).pack(side="left")

        ctk.CTkButton(
            footer,
            text="💾 Save Profile Preferences",
            command=self._save_profile,
            height=36,
            corner_radius=8,
            fg_color=PALETTE["brand_green"],
            hover_color=PALETTE["brand_green_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        ).pack(side="right")

    def _save_profile(self):
        new_profile = {
            "name": self.name_entry.get().strip() or "Candidate Profile",
            "email": self.email_entry.get().strip() or "candidate@ai-copilot.io",
            "role": self.role_entry.get().strip() or "Senior Software Engineer",
            "skills": self.skills_entry.get().strip() or "Python, Full Stack",
            "tone": self.tone_opt.get()
        }
        if self.on_save_callback:
            self.on_save_callback(new_profile)
        messagebox.showinfo("Profile Updated", "Candidate profile preferences successfully saved!")
        self.destroy()


# =============================================================================
# CREATE SESSION MODAL (LAUNCH CONFIGURATION & HARDWARE AUDIO SELECTION)
# =============================================================================
class CreateSessionDialog(ctk.CTkToplevel):
    def __init__(self, parent, on_launch_callback=None, default_config=None):
        super().__init__(parent)
        self.parent = parent
        self.on_launch_callback = on_launch_callback
        self.config = default_config or {}

        # Hardware audio discovery & preferences
        self.available_mics = get_available_microphones()
        self.available_speakers = get_available_speakers()
        self.available_monitors = get_available_monitors()
        self.saved_device_prefs = get_saved_device_preferences()
        self.saved_monitor_id = get_saved_monitor_preference()
        self.active_probe: Optional[AudioLevelProbe] = None
        self.active_probe_type: Optional[str] = None

        self.title("Create Call Session - BluJay AI")
        self.geometry("580x730")
        self.configure(fg_color=PALETTE["bg_sidebar"])
        self.resizable(False, False)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.transient(parent)
        self.grab_set()
        self._center_window(parent)
        self._build_ui()
        self._apply_stealth_capture()
        self.after(50, self._apply_stealth_capture)

    def _apply_stealth_capture(self):
        try:
            self.update_idletasks()
            raw_id = self.winfo_id()
            hwnd = get_root_hwnd(raw_id) or raw_id
            exclude_from_screen_capture(hwnd)
        except Exception:
            pass

    def _center_window(self, parent):
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        w, h = 580, 730
        x = px + max(0, (pw - w) // 2)
        y = py + max(0, (ph - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        pad = 26

        # Title & Subtitle Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=pad, pady=(20, 10))

        ctk.CTkLabel(
            header,
            text="✦ Create Call Session",
            font=ctk.CTkFont(family="Segoe UI", size=19, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Configure your session details, select audio devices, and test live levels.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(2, 0))

        # Main Scrollable Form
        form_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        form_scroll.pack(fill="both", expand=True, padx=pad, pady=(0, 14))

        # 1. Job Role / Position
        ctk.CTkLabel(
            form_scroll,
            text="Target Job Title / Role *",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w", pady=(4, 4))

        self.role_entry = ctk.CTkEntry(
            form_scroll,
            placeholder_text="e.g. Senior Software Engineer, Full Stack Lead",
            height=36,
            corner_radius=8,
            border_color=PALETTE["border_card"],
            fg_color=PALETTE["bg_card"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=12)
        )
        self.role_entry.pack(fill="x", pady=(0, 10))
        if self.config.get("role"):
            self.role_entry.insert(0, self.config["role"])

        # 2. Company Name
        ctk.CTkLabel(
            form_scroll,
            text="Target Company (Optional)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w", pady=(0, 4))

        self.company_entry = ctk.CTkEntry(
            form_scroll,
            placeholder_text="e.g. Google, Stripe, Meta, Tech Startup",
            height=36,
            corner_radius=8,
            border_color=PALETTE["border_card"],
            fg_color=PALETTE["bg_card"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=12)
        )
        self.company_entry.pack(fill="x", pady=(0, 10))
        if self.config.get("company"):
            self.company_entry.insert(0, self.config["company"])

        # 2b. Job Description (JD) / Requirements
        ctk.CTkLabel(
            form_scroll,
            text="📋 Job Description (JD) / Requirements (Optional)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        self.jd_textbox = ctk.CTkTextbox(
            form_scroll,
            height=70,
            corner_radius=8,
            border_color=PALETTE["border_card"],
            border_width=1,
            fg_color=PALETTE["bg_card"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.jd_textbox.pack(fill="x", pady=(0, 6))

        # Fit Analysis Button & Status Row
        jd_btn_row = ctk.CTkFrame(form_scroll, fg_color="transparent")
        jd_btn_row.pack(fill="x", pady=(0, 10))

        self.jd_analyze_btn = ctk.CTkButton(
            jd_btn_row,
            text="⚡ Analyze Match & Predict Questions",
            command=self._analyze_jd_fit,
            height=28,
            corner_radius=6,
            fg_color=PALETTE["btn_secondary_bg"],
            hover_color=PALETTE["bg_subtle_hover"],
            text_color=PALETTE["accent_cyan"],
            border_color=PALETTE["border_card"],
            border_width=1,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold")
        )
        self.jd_analyze_btn.pack(side="left")

        self.jd_fit_lbl = ctk.CTkLabel(
            jd_btn_row,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=PALETTE["brand_green_text"]
        )
        self.jd_fit_lbl.pack(side="left", padx=(8, 0))

        # 3. Audio Mode
        ctk.CTkLabel(
            form_scroll,
            text="Audio Capture Strategy",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w", pady=(0, 4))

        self.source_opt = ctk.CTkOptionMenu(
            form_scroll,
            values=["Dual Audio (System Loopback + Mic) [Recommended]", "Loopback Only (Interviewer Only)", "Mic Only (Candidate Only)"],
            height=36,
            corner_radius=8,
            fg_color=PALETTE["bg_card"],
            button_color=PALETTE["btn_secondary_border"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.source_opt.pack(fill="x", pady=(0, 14))

        # Divider: Hardware Audio Devices & Testing
        hw_divider = ctk.CTkFrame(form_scroll, height=1, fg_color=PALETTE["border_card"])
        hw_divider.pack(fill="x", pady=(4, 12))

        ctk.CTkLabel(
            form_scroll,
            text="🎧 Audio Hardware & Signal Verification",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["accent_cyan"]
        ).pack(anchor="w", pady=(0, 8))

        # 4. Microphone Selection (Candidate)
        ctk.CTkLabel(
            form_scroll,
            text="🎙️ Microphone (Candidate Input)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        mic_names = [m["display_name"] for m in self.available_mics]
        default_mic_name = mic_names[0] if mic_names else "Default Microphone"
        saved_mic_id = self.saved_device_prefs.get("mic_device_id")
        if saved_mic_id is not None:
            for m in self.available_mics:
                if str(m["id"]) == str(saved_mic_id):
                    default_mic_name = m["display_name"]
                    break

        self.mic_opt = ctk.CTkOptionMenu(
            form_scroll,
            values=mic_names or ["Default Microphone"],
            height=34,
            corner_radius=8,
            fg_color=PALETTE["bg_card"],
            button_color=PALETTE["btn_secondary_border"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.mic_opt.set(default_mic_name)
        self.mic_opt.pack(fill="x", pady=(0, 4))

        # Mic Test Row (Button + Live Level Bar)
        mic_test_row = ctk.CTkFrame(form_scroll, fg_color="transparent")
        mic_test_row.pack(fill="x", pady=(0, 12))

        self.mic_test_btn = ctk.CTkButton(
            mic_test_row,
            text="🎙️ Test Mic",
            command=self._toggle_test_mic,
            width=90,
            height=26,
            corner_radius=6,
            fg_color=PALETTE["btn_secondary_bg"],
            hover_color=PALETTE["bg_subtle_hover"],
            text_color=PALETTE["text_primary"],
            border_color=PALETTE["border_card"],
            border_width=1,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold")
        )
        self.mic_test_btn.pack(side="left", padx=(0, 8))

        self.mic_meter = ctk.CTkProgressBar(
            mic_test_row,
            height=10,
            corner_radius=5,
            progress_color=PALETTE["brand_green"],
            fg_color=PALETTE["bg_card"]
        )
        self.mic_meter.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.mic_meter.set(0.0)

        self.mic_status_lbl = ctk.CTkLabel(
            mic_test_row,
            text="Ready",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=PALETTE["text_muted"],
            width=60
        )
        self.mic_status_lbl.pack(side="right")

        # 5. Speaker / Loopback Selection (Interviewer)
        ctk.CTkLabel(
            form_scroll,
            text="🔊 Meeting Audio / Speaker Loopback (Interviewer Audio)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        spk_names = [s["display_name"] for s in self.available_speakers]
        default_spk_name = spk_names[0] if spk_names else "Default System Audio"
        saved_spk_id = self.saved_device_prefs.get("speaker_device_id")
        if saved_spk_id is not None:
            for s in self.available_speakers:
                if str(s["id"]) == str(saved_spk_id):
                    default_spk_name = s["display_name"]
                    break

        self.spk_opt = ctk.CTkOptionMenu(
            form_scroll,
            values=spk_names or ["Default System Audio"],
            height=34,
            corner_radius=8,
            fg_color=PALETTE["bg_card"],
            button_color=PALETTE["btn_secondary_border"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.spk_opt.set(default_spk_name)
        self.spk_opt.pack(fill="x", pady=(0, 4))

        # Speaker Test Row (Button + Live Level Bar)
        spk_test_row = ctk.CTkFrame(form_scroll, fg_color="transparent")
        spk_test_row.pack(fill="x", pady=(0, 14))

        self.spk_test_btn = ctk.CTkButton(
            spk_test_row,
            text="🔊 Test Loopback",
            command=self._toggle_test_speaker,
            width=110,
            height=26,
            corner_radius=6,
            fg_color=PALETTE["btn_secondary_bg"],
            hover_color=PALETTE["bg_subtle_hover"],
            text_color=PALETTE["text_primary"],
            border_color=PALETTE["border_card"],
            border_width=1,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold")
        )
        self.spk_test_btn.pack(side="left", padx=(0, 8))

        self.spk_meter = ctk.CTkProgressBar(
            spk_test_row,
            height=10,
            corner_radius=5,
            progress_color=PALETTE["accent_cyan"],
            fg_color=PALETTE["bg_card"]
        )
        self.spk_meter.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.spk_meter.set(0.0)

        self.spk_status_lbl = ctk.CTkLabel(
            spk_test_row,
            text="Ready",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=PALETTE["text_muted"],
            width=60
        )
        self.spk_status_lbl.pack(side="right")

        # 6. Target Display for Multimodal Vision (Coding Problems & Diagrams)
        ctk.CTkLabel(
            form_scroll,
            text="🖥️ Target Screen / Display for Vision AI",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        mon_names = [m["name"] for m in self.available_monitors]
        default_mon_name = mon_names[0] if mon_names else "Primary Display"
        for m in self.available_monitors:
            if m["id"] == self.saved_monitor_id:
                default_mon_name = m["name"]
                break

        self.mon_opt = ctk.CTkOptionMenu(
            form_scroll,
            values=mon_names or ["Primary Display"],
            height=34,
            corner_radius=8,
            fg_color=PALETTE["bg_card"],
            button_color=PALETTE["btn_secondary_border"],
            text_color=PALETTE["text_primary"],
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.mon_opt.set(default_mon_name)
        self.mon_opt.pack(fill="x", pady=(0, 14))

        # 7. Global Hotkeys Cheat-Sheet Card
        hk_card = ctk.CTkFrame(form_scroll, fg_color=PALETTE["bg_card"], corner_radius=8, border_color=PALETTE["border_card"], border_width=1)
        hk_card.pack(fill="x", pady=(0, 14), padx=2)

        hk_inner = ctk.CTkFrame(hk_card, fg_color="transparent")
        hk_inner.pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(
            hk_inner,
            text="⌨️ Active System-Wide Hotkeys (Work Across All Apps)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            hk_inner,
            text="• Ctrl + Alt + H : Panic Stealth Hide / Unhide Overlay\n• Ctrl + Alt + S : Snip & Solve Screen Coding Problem\n• Ctrl + Alt + M : Mute / Pause AI Speech-To-Text\n• Ctrl + Alt + R : Quick Regenerate / Elaborate Answer\n• Ctrl + Alt + T : Cycle Window Transparency (95% ➔ 80% ➔ 60%)",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=PALETTE["text_secondary"],
            justify="left"
        ).pack(anchor="w")

        # 7. Auto-Generate Switch
        self.auto_gen_switch = ctk.CTkSwitch(
            form_scroll,
            text="Auto-Generate AI Answers when interviewer finishes speaking",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_primary"],
            progress_color=PALETTE["brand_green"]
        )
        self.auto_gen_switch.pack(anchor="w", pady=(0, 14))
        self.auto_gen_switch.select()

        # Footer Actions
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=pad, pady=(0, 18))

        ctk.CTkButton(
            footer,
            text="Cancel",
            command=self._on_close,
            width=100,
            height=38,
            corner_radius=8,
            fg_color=PALETTE["btn_secondary_bg"],
            hover_color=PALETTE["bg_subtle_hover"],
            text_color=PALETTE["text_secondary"],
            border_color=PALETTE["border_card"],
            border_width=1,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        ).pack(side="left")

        ctk.CTkButton(
            footer,
            text="⚡ Launch Real-Time Copilot",
            command=self._launch,
            height=38,
            corner_radius=8,
            fg_color=PALETTE["brand_green"],
            hover_color=PALETTE["brand_green_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        ).pack(side="right")

    # =========================================================================
    # AUDIO PROBE TESTING
    # =========================================================================

    def _stop_any_probe(self):
        if self.active_probe:
            self.active_probe.stop()
            self.active_probe = None
        self.active_probe_type = None
        try:
            self.mic_test_btn.configure(text="🎙️ Test Mic", fg_color=PALETTE["btn_secondary_bg"])
            self.spk_test_btn.configure(text="🔊 Test Loopback", fg_color=PALETTE["btn_secondary_bg"])
            self.mic_meter.set(0.0)
            self.spk_meter.set(0.0)
            self.mic_status_lbl.configure(text="Ready", text_color=PALETTE["text_muted"])
            self.spk_status_lbl.configure(text="Ready", text_color=PALETTE["text_muted"])
        except Exception:
            pass

    def _toggle_test_mic(self):
        if self.active_probe_type == "mic":
            self._stop_any_probe()
            return

        self._stop_any_probe()
        selected_display = self.mic_opt.get()
        selected_id = "default"
        for m in self.available_mics:
            if m["display_name"] == selected_display:
                selected_id = m["id"]
                break

        self.active_probe_type = "mic"
        self.mic_test_btn.configure(text="⏹️ Stop Test", fg_color=PALETTE["accent_rose"])
        self.mic_status_lbl.configure(text="Testing...", text_color=PALETTE["brand_green_text"])

        self.active_probe = AudioLevelProbe(
            device_type="mic",
            device_id=selected_id,
            on_level_update=self._update_mic_meter
        )
        self.active_probe.start()

    def _toggle_test_speaker(self):
        if self.active_probe_type == "speaker":
            self._stop_any_probe()
            return

        self._stop_any_probe()
        selected_display = self.spk_opt.get()
        selected_id = "default"
        for s in self.available_speakers:
            if s["display_name"] == selected_display:
                selected_id = s["id"]
                break

        self.active_probe_type = "speaker"
        self.spk_test_btn.configure(text="⏹️ Stop Test", fg_color=PALETTE["accent_rose"])
        self.spk_status_lbl.configure(text="Testing...", text_color=PALETTE["accent_cyan"])

        self.active_probe = AudioLevelProbe(
            device_type="speaker",
            device_id=selected_id,
            on_level_update=self._update_speaker_meter
        )
        self.active_probe.start()

    def _update_mic_meter(self, level: float, rms: float):
        def _apply():
            try:
                self.mic_meter.set(level)
                if level > 0.05:
                    self.mic_status_lbl.configure(text=f"{int(level*100)}%", text_color=PALETTE["brand_green_text"])
                else:
                    self.mic_status_lbl.configure(text="Listening...", text_color=PALETTE["text_muted"])
            except Exception:
                pass
        self.after(0, _apply)

    def _update_speaker_meter(self, level: float, rms: float):
        def _apply():
            try:
                self.spk_meter.set(level)
                if level > 0.05:
                    self.spk_status_lbl.configure(text=f"{int(level*100)}%", text_color=PALETTE["accent_cyan"])
                else:
                    self.spk_status_lbl.configure(text="Listening...", text_color=PALETTE["text_muted"])
            except Exception:
                pass
        self.after(0, _apply)

    def _analyze_jd_fit(self):
        jd_text = self.jd_textbox.get("1.0", "end").strip()
        if not jd_text:
            self.jd_fit_lbl.configure(text="⚠️ Paste JD text first", text_color=PALETTE["accent_amber"])
            return

        self.jd_fit_lbl.configure(text="Analyzing...", text_color=PALETTE["accent_cyan"])
        
        def _worker():
            from jd_matcher import analyze_jd_resume_fit, save_jd_analysis_cache
            # Read active resume text if available
            resume_text = ""
            try:
                from rag import load_documents
                docs = load_documents()
                resume_text = "\n".join([d["text"] for d in docs])
            except Exception:
                pass
                
            res = analyze_jd_resume_fit(jd_text, resume_text, job_title=self.role_entry.get().strip())
            save_jd_analysis_cache(res)
            
            def _apply():
                score = res.get("match_score", 0)
                matched_count = len(res.get("matched_skills", []))
                gap_count = len(res.get("missing_skills", []))
                self.jd_fit_lbl.configure(
                    text=f"✓ Fit: {score}% | {matched_count} Match | {gap_count} Gaps",
                    text_color=PALETTE["brand_green_text"] if score >= 70 else PALETTE["accent_amber"]
                )
            self.after(0, _apply)
            
        threading.Thread(target=_worker, daemon=True).start()

    def _on_close(self):
        self._stop_any_probe()
        self.destroy()

    def _launch(self):
        self._stop_any_probe()

        role = self.role_entry.get().strip() or "Senior Software Engineer"
        company = self.company_entry.get().strip()
        jd_text = self.jd_textbox.get("1.0", "end").strip()
        source_raw = self.source_opt.get()
        source_key = "dual" if "Dual" in source_raw else ("system" if "Loopback" in source_raw else "mic")
        auto_gen = bool(self.auto_gen_switch.get())

        # Resolve selected mic
        selected_mic_display = self.mic_opt.get()
        selected_mic_id = "default"
        selected_mic_name = selected_mic_display
        for m in self.available_mics:
            if m["display_name"] == selected_mic_display:
                selected_mic_id = m["id"]
                selected_mic_name = m["name"]
                break

        # Resolve selected speaker
        selected_spk_display = self.spk_opt.get()
        selected_spk_id = "default"
        selected_spk_name = selected_spk_display
        for s in self.available_speakers:
            if s["display_name"] == selected_spk_display:
                selected_spk_id = s["id"]
                selected_spk_name = s["name"]
                break

        # Resolve selected monitor
        selected_mon_display = self.mon_opt.get()
        selected_mon_id = 0
        for m in self.available_monitors:
            if m["name"] == selected_mon_display:
                selected_mon_id = m["id"]
                break

        # Save preferences to disk
        save_device_preferences(
            mic_device_id=selected_mic_id,
            mic_device_name=selected_mic_name,
            speaker_device_id=selected_spk_id,
            speaker_device_name=selected_spk_name
        )
        save_monitor_preference(selected_mon_id)

        config = {
            "role": role,
            "company": company,
            "job_description": jd_text,
            "audio_source": source_key,
            "auto_gen": auto_gen,
            "mic_device_id": selected_mic_id,
            "mic_device_name": selected_mic_name,
            "speaker_device_id": selected_spk_id,
            "speaker_device_name": selected_spk_name,
            "monitor_id": selected_mon_id,
            "mode": "copilot"
        }

        self.destroy()
        if self.on_launch_callback:
            self.on_launch_callback(config)


# =============================================================================
# BLUJAY AI MAIN HOME DASHBOARD
# =============================================================================
class ModernHomeDashboard(ctk.CTk):
    def __init__(
        self,
        on_launch_copilot=None,
        on_launch_standard=None,
        on_launch_practice=None,
        on_documents_updated=None,
        **kwargs
    ):
        super().__init__()

        self.on_launch_copilot = on_launch_copilot
        self.on_launch_standard = on_launch_standard
        self.on_launch_practice = on_launch_practice
        self.on_documents_updated = on_documents_updated

        self.title("BluJay AI - Intelligent Interview Copilot")
        self.geometry("1180x760")
        self.minsize(1020, 660)
        self.configure(fg_color=PALETTE["bg_main"])

        self.current_tab = "sessions"
        self.current_filter = "all"

        self._apply_stealth_capture()
        self.after(100, self._apply_stealth_capture)
        self.after(1000, self._apply_stealth_capture)

        self._build_dashboard_layout()

    def _apply_stealth_capture(self):
        try:
            self.update_idletasks()
            raw_id = self.winfo_id()
            hwnd = get_root_hwnd(raw_id) or raw_id
            exclude_from_screen_capture(hwnd)
        except Exception:
            pass

    def _build_dashboard_layout(self):
        self.main_container = ctk.CTkFrame(self, fg_color=PALETTE["bg_main"], corner_radius=0)
        self.main_container.pack(fill="both", expand=True)

        # 1. LEFT SIDEBAR
        self.sidebar = ctk.CTkFrame(
            self.main_container,
            width=260,
            fg_color=PALETTE["bg_sidebar"],
            corner_radius=0,
            border_color=PALETTE["sidebar_border"],
            border_width=1
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # App Header / Logo
        logo_box = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=54)
        logo_box.pack(fill="x", padx=16, pady=(18, 12))
        logo_box.pack_propagate(False)

        logo_inner = ctk.CTkFrame(logo_box, fg_color="transparent")
        logo_inner.pack(side="left", fill="y")

        # Glowing Icon Badge
        icon_badge = ctk.CTkFrame(logo_inner, fg_color="#10B981", width=32, height=32, corner_radius=8)
        icon_badge.pack(side="left", padx=(0, 10))
        icon_badge.pack_propagate(False)
        ctk.CTkLabel(icon_badge, text="✦", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color="#FFFFFF").place(relx=0.5, rely=0.5, anchor="center")

        brand_text_frame = ctk.CTkFrame(logo_inner, fg_color="transparent")
        brand_text_frame.pack(side="left")

        title_row = ctk.CTkFrame(brand_text_frame, fg_color="transparent")
        title_row.pack(anchor="w")

        ctk.CTkLabel(
            title_row,
            text="BluJay",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(side="left", padx=(0, 4))

        ctk.CTkLabel(
            title_row,
            text="AI",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=PALETTE["brand_green"]
        ).pack(side="left", padx=(0, 6))

        pro_pill = ctk.CTkFrame(title_row, fg_color="#064E3B", corner_radius=4, border_color="#059669", border_width=1)
        pro_pill.pack(side="left")
        ctk.CTkLabel(pro_pill, text="PRO", font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"), text_color="#34D399").pack(padx=4, pady=0)

        ctk.CTkLabel(
            brand_text_frame,
            text="Live Interview Copilot",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=PALETTE["text_muted"]
        ).pack(anchor="w")

        # Workspace Navigation
        workspace_lbl = ctk.CTkLabel(
            self.sidebar,
            text="MAIN WORKSPACE",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=PALETTE["text_muted"]
        )
        workspace_lbl.pack(anchor="w", padx=20, pady=(16, 8))

        self.nav_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.nav_container.pack(fill="x", padx=12, pady=(0, 10))

        self.btn_nav_sessions = ctk.CTkButton(
            self.nav_container,
            text="  ⚡  Call Sessions",
            anchor="w",
            height=42,
            corner_radius=8,
            fg_color=PALETTE["nav_active_bg"],
            text_color=PALETTE["nav_active_text"],
            hover_color=PALETTE["nav_active_bg"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=lambda: self._switch_nav("sessions")
        )
        self.btn_nav_sessions.pack(fill="x", pady=2)

        self.btn_nav_cvs = ctk.CTkButton(
            self.nav_container,
            text="  📄  CVs & Resumes",
            anchor="w",
            height=42,
            corner_radius=8,
            fg_color="transparent",
            text_color=PALETTE["nav_inactive_text"],
            hover_color=PALETTE["nav_hover_bg"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=lambda: self._switch_nav("cvs")
        )
        self.btn_nav_cvs.pack(fill="x", pady=2)

        self.btn_nav_docs = ctk.CTkButton(
            self.nav_container,
            text="  📁  Knowledge & Docs",
            anchor="w",
            height=42,
            corner_radius=8,
            fg_color="transparent",
            text_color=PALETTE["nav_inactive_text"],
            hover_color=PALETTE["nav_hover_bg"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=lambda: self._switch_nav("documents")
        )
        self.btn_nav_docs.pack(fill="x", pady=2)

        # Spacer
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        # Sidebar Bottom Section
        bottom_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_box.pack(side="bottom", fill="x", padx=12, pady=(0, 16))

        # Plan Card
        self.plan_card = ctk.CTkFrame(
            bottom_box,
            fg_color="#091424",
            corner_radius=10,
            border_color="#1E3A5F",
            border_width=1
        )
        self.plan_card.pack(fill="x", pady=(0, 12), ipady=4)

        plan_top = ctk.CTkFrame(self.plan_card, fg_color="transparent")
        plan_top.pack(fill="x", padx=12, pady=(10, 2))

        plan_title_box = ctk.CTkFrame(plan_top, fg_color="transparent")
        plan_title_box.pack(side="left")

        ctk.CTkLabel(plan_title_box, text="⚡", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 6))
        self.plan_title_lbl = ctk.CTkLabel(
            plan_title_box,
            text="Pro Pass Active",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#38BDF8"
        )
        self.plan_title_lbl.pack(side="left")

        self.plan_desc_lbl = ctk.CTkLabel(
            self.plan_card,
            text="Full Stealth & Fast Groq LPU Active",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=PALETTE["text_secondary"],
            wraplength=210,
            justify="left"
        )
        self.plan_desc_lbl.pack(anchor="w", padx=12, pady=(2, 6))

        ctk.CTkButton(
            self.plan_card,
            text="✨ Upgrade Credits",
            command=self._open_buy_credits_modal,
            height=32,
            corner_radius=6,
            fg_color=PALETTE["btn_dark"],
            hover_color=PALETTE["btn_dark_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        ).pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkFrame(bottom_box, fg_color=PALETTE["border_card"], height=1).pack(fill="x", pady=8)

        # User Profile Row (Interactive Clickable Card)
        self.user_profile_box = ctk.CTkFrame(
            bottom_box,
            fg_color="transparent",
            corner_radius=8,
            cursor="hand2"
        )
        self.user_profile_box.pack(fill="x", pady=(2, 0))

        self.avatar_circle = ctk.CTkFrame(self.user_profile_box, fg_color="#1E293B", width=36, height=36, corner_radius=18, border_color="#38BDF8", border_width=1, cursor="hand2")
        self.avatar_circle.pack(side="left", padx=(0, 8))
        self.avatar_circle.pack_propagate(False)

        self.avatar_lbl = ctk.CTkLabel(
            self.avatar_circle,
            text="AS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#38BDF8",
            cursor="hand2"
        )
        self.avatar_lbl.place(relx=0.5, rely=0.5, anchor="center")

        user_info = ctk.CTkFrame(self.user_profile_box, fg_color="transparent", cursor="hand2")
        user_info.pack(side="left", fill="x", expand=True)

        self.user_name_lbl = ctk.CTkLabel(
            user_info,
            text="Abhishek Salunke",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["text_primary"],
            cursor="hand2"
        )
        self.user_name_lbl.pack(anchor="w")

        self.user_email_lbl = ctk.CTkLabel(
            user_info,
            text="⚙ Click to Edit Profile",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color="#38BDF8",
            cursor="hand2"
        )
        self.user_email_lbl.pack(anchor="w")

        # Bind hover and click events
        for widget in [self.user_profile_box, self.avatar_circle, self.avatar_lbl, user_info, self.user_name_lbl, self.user_email_lbl]:
            widget.bind("<Button-1>", lambda e: self._open_user_profile_modal())
            widget.bind("<Enter>", lambda e: self.user_profile_box.configure(fg_color="#131D31"))
            widget.bind("<Leave>", lambda e: self.user_profile_box.configure(fg_color="transparent"))

        # Candidate profile state
        self.candidate_profile = {
            "name": "Abhishek Salunke",
            "email": "abhisheksalunke7387@gmail.com",
            "role": "Senior Full Stack & AI Engineer",
            "skills": "Python, Django, FastAPI, React, PostgreSQL, Docker, RAG, LLM Copilots",
            "tone": "Direct & Highly Technical (Recommended)"
        }

        # 2. RIGHT MAIN CONTENT AREA
        self.content_area = ctk.CTkFrame(self.main_container, fg_color=PALETTE["bg_main"], corner_radius=0)
        self.content_area.pack(side="right", fill="both", expand=True)

        self._render_active_tab()

    def _open_user_profile_modal(self):
        """Opens the Candidate Profile & AI Grounding Preferences dialog."""
        UserProfileDialog(self, on_save_callback=self._on_profile_saved, current_profile=self.candidate_profile)

    def _on_profile_saved(self, new_profile: dict):
        """Updates candidate profile in state and UI."""
        self.candidate_profile = new_profile
        name = new_profile.get("name", "Candidate Profile")
        self.user_name_lbl.configure(text=name)

        # Compute initials
        parts = name.strip().split()
        initials = "".join([p[0].upper() for p in parts[:2]]) if parts else "AS"
        self.avatar_lbl.configure(text=initials)

    def _switch_nav(self, tab: str):
        self.current_tab = tab

        for btn, name in [
            (self.btn_nav_sessions, "sessions"),
            (self.btn_nav_cvs, "cvs"),
            (self.btn_nav_docs, "documents")
        ]:
            if name == tab:
                btn.configure(
                    fg_color=PALETTE["nav_active_bg"],
                    text_color=PALETTE["nav_active_text"],
                    font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=PALETTE["nav_inactive_text"],
                    font=ctk.CTkFont(family="Segoe UI", size=12)
                )

        self._render_active_tab()

    def _render_active_tab(self):
        for w in self.content_area.winfo_children():
            w.destroy()

        if self.current_tab == "sessions":
            self._render_call_sessions_view()
        elif self.current_tab == "cvs":
            self._render_cvs_view()
        elif self.current_tab == "documents":
            self._render_documents_view()

    # =========================================================================
    # CALL SESSIONS VIEW (NEXT-GEN DASHBOARD + QUICK START PRESETS)
    # =========================================================================
    def _render_call_sessions_view(self):
        header_row = ctk.CTkFrame(self.content_area, fg_color="transparent", height=60)
        header_row.pack(fill="x", padx=36, pady=(26, 8))

        title_box = ctk.CTkFrame(header_row, fg_color="transparent")
        title_box.pack(side="left")

        ctk.CTkLabel(
            title_box,
            text="Call Sessions & Copilot Launch",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Manage real-time interview sessions, LeetCode screen solvers, and AI talking points.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(2, 0))

        ctk.CTkButton(
            header_row,
            text="＋ Create Session",
            command=self._on_click_create_session,
            height=38,
            corner_radius=8,
            fg_color=PALETTE["brand_green"],
            hover_color=PALETTE["brand_green_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        ).pack(side="right")

        # Main Scrollable Canvas for Hero Banner & Presets
        canvas_scroll = ctk.CTkScrollableFrame(self.content_area, fg_color="transparent")
        canvas_scroll.pack(fill="both", expand=True, padx=36, pady=(10, 16))

        # Hero Banner Box
        hero_box = ctk.CTkFrame(canvas_scroll, fg_color=PALETTE["bg_card"], corner_radius=12, border_color=PALETTE["border_card"], border_width=1)
        hero_box.pack(fill="x", pady=(6, 24), ipady=12)

        badge_circle = ctk.CTkFrame(hero_box, fg_color="#064E3B", width=56, height=56, corner_radius=28, border_color="#059669", border_width=1)
        badge_circle.pack(pady=(12, 10))
        badge_circle.pack_propagate(False)

        ctk.CTkLabel(badge_circle, text="🎙️", font=ctk.CTkFont(size=22)).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            hero_box,
            text="Ready to launch a live interview session?",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(pady=(0, 4))

        ctk.CTkLabel(
            hero_box,
            text="Start real-time speech transcription, LeetCode screen vision solver, and resume-grounded answers.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"]
        ).pack(pady=(0, 16))

        ctk.CTkButton(
            hero_box,
            text="⚡ Launch Real-Time Copilot",
            command=self._on_click_create_session,
            height=40,
            width=210,
            corner_radius=8,
            fg_color=PALETTE["btn_dark"],
            hover_color=PALETTE["btn_dark_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        ).pack(pady=(0, 8))

        # Quick Templates Section
        ctk.CTkLabel(
            canvas_scroll,
            text="⚡ QUICK-START INTERVIEW PRESETS",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=PALETTE["accent_cyan"]
        ).pack(anchor="w", pady=(0, 12))

        presets_grid = ctk.CTkFrame(canvas_scroll, fg_color="transparent")
        presets_grid.pack(fill="x", pady=(0, 20))
        presets_grid.columnconfigure((0, 1, 2), weight=1, uniform="preset_col")

        templates = [
            {
                "icon": "💻",
                "title": "Coding & LeetCode",
                "role": "Software Engineer (Coding & DSA)",
                "desc": "Screen vision OCR solver, Big-O complexity analysis, and syntax-highlighted solutions.",
                "color": "#3B82F6"
            },
            {
                "icon": "🏛️",
                "title": "System Design",
                "role": "Senior Architect / System Design",
                "desc": "Distributed architecture, microservices, scaling trade-offs, and database designs.",
                "color": "#8B5CF6"
            },
            {
                "icon": "🎯",
                "title": "Behavioral & STAR",
                "role": "Technical Lead / Leadership",
                "desc": "STAR framework answers (Situation, Task, Action, Result) grounded in your resume.",
                "color": "#10B981"
            }
        ]

        for idx, t in enumerate(templates):
            card = ctk.CTkFrame(
                presets_grid,
                fg_color=PALETTE["bg_card"],
                corner_radius=10,
                border_color=PALETTE["border_card"],
                border_width=1
            )
            card.grid(row=0, column=idx, sticky="nsew", padx=6 if idx == 1 else (0 if idx == 0 else 0), pady=0)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="both", expand=True, padx=16, pady=16)

            icon_box = ctk.CTkFrame(inner, fg_color="#1E293B", width=36, height=36, corner_radius=8, border_color="#334155", border_width=1)
            icon_box.pack(anchor="w", pady=(0, 10))
            icon_box.pack_propagate(False)
            ctk.CTkLabel(icon_box, text=t["icon"], font=ctk.CTkFont(size=16)).place(relx=0.5, rely=0.5, anchor="center")

            ctk.CTkLabel(inner, text=t["title"], font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"), text_color=PALETTE["text_primary"]).pack(anchor="w")
            ctk.CTkLabel(inner, text=t["desc"], font=ctk.CTkFont(family="Segoe UI", size=10), text_color=PALETTE["text_secondary"], wraplength=200, justify="left").pack(anchor="w", pady=(4, 14))

            ctk.CTkButton(
                inner,
                text="Launch Preset →",
                command=lambda r=t["role"]: self._launch_preset(r),
                height=32,
                corner_radius=6,
                fg_color=PALETTE["btn_secondary_bg"],
                text_color=PALETTE["text_primary"],
                hover_color=PALETTE["bg_subtle_hover"],
                border_color=PALETTE["border_card"],
                border_width=1,
                font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold")
            ).pack(fill="x")

    def _launch_preset(self, role_title: str):
        config = {
            "role": role_title,
            "company": "Tech Company",
            "audio_source": "dual",
            "auto_gen": True,
            "mode": "copilot"
        }
        self._execute_launch(config)

    def _set_filter(self, filter_mode: str):
        self.current_filter = filter_mode
        self._render_call_sessions_view()

    # =========================================================================
    # CVS & RESUMES VIEW
    # =========================================================================
    def _render_cvs_view(self):
        header_row = ctk.CTkFrame(self.content_area, fg_color="transparent", height=60)
        header_row.pack(fill="x", padx=36, pady=(26, 16))

        title_box = ctk.CTkFrame(header_row, fg_color="transparent")
        title_box.pack(side="left")

        ctk.CTkLabel(
            title_box,
            text="CVs & Resumes",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Upload and manage resumes to ground AI responses in your verified background.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(2, 0))

        content_scroll = ctk.CTkScrollableFrame(self.content_area, fg_color="transparent")
        content_scroll.pack(fill="both", expand=True, padx=36, pady=(0, 20))

        # Upload Box
        upload_box = ctk.CTkFrame(
            content_scroll,
            fg_color=PALETTE["bg_card"],
            corner_radius=10,
            border_color=PALETTE["border_card"],
            border_width=1
        )
        upload_box.pack(fill="x", pady=(0, 20), ipady=12)

        ctk.CTkLabel(upload_box, text="📄 Upload New CV / Resume", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color=PALETTE["text_primary"]).pack(pady=(12, 4))
        ctk.CTkLabel(upload_box, text="Supported formats: PDF, DOCX, TXT. Automatically indexed for RAG retrieval.", font=ctk.CTkFont(family="Segoe UI", size=11), text_color=PALETTE["text_secondary"]).pack(pady=(0, 12))

        ctk.CTkButton(
            upload_box,
            text="Browse File & Sync RAG",
            command=self._upload_resume_file,
            height=36,
            corner_radius=8,
            fg_color=PALETTE["btn_dark"],
            hover_color=PALETTE["btn_dark_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        ).pack()

    def _upload_resume_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Resume Document",
            filetypes=[("Documents", "*.pdf;*.docx;*.txt;*.md"), ("All Files", "*.*")]
        )
        if file_path and os.path.exists(file_path):
            messagebox.showinfo("RAG Sync Complete", f"Resume successfully loaded & synced:\n{os.path.basename(file_path)}")
            if self.on_documents_updated:
                self.on_documents_updated([file_path])

    # =========================================================================
    # DOCUMENTS VIEW
    # =========================================================================
    def _render_documents_view(self):
        header_row = ctk.CTkFrame(self.content_area, fg_color="transparent", height=60)
        header_row.pack(fill="x", padx=36, pady=(26, 16))

        title_box = ctk.CTkFrame(header_row, fg_color="transparent")
        title_box.pack(side="left")

        ctk.CTkLabel(
            title_box,
            text="Knowledge & Documents",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=PALETTE["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Manage project portfolios, architecture specs, and STAR stories for RAG retrieval.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=PALETTE["text_secondary"]
        ).pack(anchor="w", pady=(2, 0))

        content_scroll = ctk.CTkScrollableFrame(self.content_area, fg_color="transparent")
        content_scroll.pack(fill="both", expand=True, padx=36, pady=(0, 20))

        doc_card = ctk.CTkFrame(
            content_scroll,
            fg_color=PALETTE["bg_card"],
            corner_radius=10,
            border_color=PALETTE["border_card"],
            border_width=1
        )
        doc_card.pack(fill="x", pady=(0, 20), ipady=12)

        ctk.CTkLabel(doc_card, text="📁 Ingest Technical Documents", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color=PALETTE["text_primary"]).pack(pady=(12, 4))
        ctk.CTkLabel(doc_card, text="Feed custom project details to prevent the AI from making up facts.", font=ctk.CTkFont(family="Segoe UI", size=11), text_color=PALETTE["text_secondary"]).pack(pady=(0, 12))

        ctk.CTkButton(
            doc_card,
            text="Add Knowledge Document",
            command=self._upload_resume_file,
            height=36,
            corner_radius=8,
            fg_color=PALETTE["btn_dark"],
            hover_color=PALETTE["btn_dark_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        ).pack()

    def _on_click_create_session(self):
        CreateSessionDialog(self, on_launch_callback=self._execute_launch)

    def _open_buy_credits_modal(self):
        BuyCreditsDialog(self)

    def _execute_launch(self, config):
        if self.on_launch_copilot:
            self.on_launch_copilot(config)
