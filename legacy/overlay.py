import re
import ctypes
from ctypes import wintypes
import tkinter as tk
from typing import Optional


# ============================================================
# WINDOWS CONSTANTS
# ============================================================

GWL_EXSTYLE = -20

WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_TOOLWINDOW = 0x00000080

GA_ROOT = 2
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011


# ============================================================
# WINDOWS API
# ============================================================

user32 = getattr(ctypes, "windll", None)
if user32 is not None:
    user32 = user32.user32
    SetWindowLongPtr = getattr(user32, "SetWindowLongPtrW", getattr(user32, "SetWindowLongW", None))
    GetWindowLongPtr = getattr(user32, "GetWindowLongPtrW", getattr(user32, "GetWindowLongW", None))
    SetWindowDisplayAffinity = getattr(user32, "SetWindowDisplayAffinity", None)
    GetAncestor = getattr(user32, "GetAncestor", None)
    GetParent = getattr(user32, "GetParent", None)
    EnumChildWindows = getattr(user32, "EnumChildWindows", None)
    EnumWindows = getattr(user32, "EnumWindows", None)
    GetWindowThreadProcessId = getattr(user32, "GetWindowThreadProcessId", None)
else:
    SetWindowLongPtr = None
    GetWindowLongPtr = None
    SetWindowDisplayAffinity = None
    GetAncestor = None
    GetParent = None
    EnumChildWindows = None
    EnumWindows = None
    GetWindowThreadProcessId = None


# ============================================================
# ROOT HWND RESOLUTION
# ============================================================

def get_root_hwnd(window_id: int) -> Optional[int]:
    """Resolve true OS top-level HWND from Tkinter window id."""
    if not window_id or user32 is None:
        return window_id
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


# ============================================================
# CLICK THROUGH
# ============================================================

def set_click_through(hwnd: int, enabled: bool) -> bool:
    """Toggles mouse events passing directly through the overlay window to IDEs/browsers."""
    if GetWindowLongPtr is None or SetWindowLongPtr is None or not hwnd:
        return False
    try:
        actual_hwnd = get_root_hwnd(hwnd) or hwnd
        style = GetWindowLongPtr(wintypes.HWND(actual_hwnd), GWL_EXSTYLE)
        if enabled:
            new_style = style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
        else:
            new_style = (style & ~WS_EX_TRANSPARENT) | WS_EX_LAYERED
        SetWindowLongPtr(wintypes.HWND(actual_hwnd), GWL_EXSTYLE, new_style)
        return True
    except Exception as e:
        print(f"Error setting click-through: {e}")
        return False


# ============================================================
# EXCLUDE FROM SCREEN CAPTURE (STEALTH - ZERO BLACK BOX)
# ============================================================

def hide_from_screen_capture(hwnd: int) -> bool:
    """
    Excludes the window from screen sharing (Zoom, Google Meet, MS Teams, OBS, Discord, etc.).
    Uses WDA_EXCLUDEFROMCAPTURE (0x11) so the window is 100% invisible on screen shares
    with NO black box / black space. The window remains completely visible to the candidate.
    """
    if SetWindowDisplayAffinity is None or not hwnd:
        return False
    actual_hwnd = get_root_hwnd(hwnd) or hwnd
    try:
        # Exclude completely from capture (0x11 = WDA_EXCLUDEFROMCAPTURE)
        # Note: We NEVER fall back to WDA_MONITOR (0x01) because WDA_MONITOR causes Windows
        # to render a solid black box on screen shares/recordings.
        res = SetWindowDisplayAffinity(wintypes.HWND(actual_hwnd), wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE))

        # Also apply to all child windows in the hierarchy
        if EnumChildWindows:
            def _child_cb(child_h, lparam):
                try:
                    SetWindowDisplayAffinity(wintypes.HWND(child_h), wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE))
                except Exception:
                    pass
                return True
            try:
                cb_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
                EnumChildWindows(wintypes.HWND(actual_hwnd), cb_type(_child_cb), 0)
            except Exception:
                pass

        return bool(res)
    except Exception as error:
        print("Screen capture exclusion error:", error)
        return False


# ============================================================
# STEALTH OVERLAY WINDOW WITH PRO UI ENHANCEMENTS
# ============================================================

class OverlayWindow:
    """
    Pro Stealth HUD & Eye-Contact Teleprompter with:
    - In-Overlay Code Syntax Highlighting (Keywords, Strings, Comments, Builtins)
    - 1-Click Code/Answer Copy to Clipboard (Alt+K) with status HUD indicator
    - Dynamic Audio Level / Waveform Visualizer
    - On-the-fly Font Scaling (Alt + Plus / Alt + Minus)
    - Webcam-Aligned Pill Mode (Alt + P) for natural eye contact
    - Click-Through Toggle (Alt + C) for typing through the overlay
    - Screen-Share Invisibility (WDA_EXCLUDEFROMCAPTURE)
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Abhishek Gen AI Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#0a0c10")

        # States
        self.click_through = True
        self.pill_mode = False
        self.opacity = 0.94
        self.font_size = 10
        self.current_text = ""
        self.current_question = ""
        self.audio_active = True
        self._audio_wave_idx = 0
        self._wave_frames = [" ▂▃", "▂▄▆", "▃▅█", " ▃▅", "  ▂"]

        # Dimensions & Positioning
        screen_width = self.root.winfo_screenwidth()
        self.default_width = 520
        self.default_height = 360
        self.default_x = screen_width - self.default_width - 30
        self.default_y = 60
        self.root.geometry(f"{self.default_width}x{self.default_height}+{self.default_x}+{self.default_y}")

        try:
            self.root.attributes("-alpha", self.opacity)
        except Exception:
            pass

        # Main UI Container
        self.frame = tk.Frame(
            self.root,
            bg="#0a0c10",
            highlightbackground="#1e222d",
            highlightthickness=1
        )
        self.frame.pack(fill="both", expand=True)

        # 1. Fixed Header Bar
        self.header_frame = tk.Frame(self.frame, bg="#0e1017", height=36, padx=8, pady=4)
        self.header_frame.pack(side="top", fill="x")

        self.title_label = tk.Label(
            self.header_frame,
            text="🟢 Abhishek Gen AI",
            bg="#0e1017",
            fg="#f8fafc",
            font=("Segoe UI", 9, "bold")
        )
        self.title_label.pack(side="left", padx=(4, 8))

        self.audio_indicator = tk.Label(
            self.header_frame,
            text="● [Mic: ▂▄▆]",
            bg="#0e1017",
            fg="#34d399",
            font=("Segoe UI", 8, "bold")
        )
        self.audio_indicator.pack(side="left", padx=4)

        self.status_label = tk.Label(
            self.header_frame,
            text="[Alt+C Click-Thru | Alt+T Scroll | Alt+K Copy]",
            bg="#0e1017",
            fg="#68d391",
            font=("Segoe UI", 8)
        )
        self.status_label.pack(side="right", padx=6)

        # 2. Content Container
        self.content_container = tk.Frame(self.frame, bg="#0a0c10", padx=8, pady=4)
        self.content_container.pack(fill="both", expand=True)

        # A. Separate Question Section
        self.q_section = tk.Frame(self.content_container, bg="#0a0c10")
        self.q_section.pack(side="top", fill="x", pady=(0, 4))

        self.q_title_lbl = tk.Label(
            self.q_section,
            text="🔵 Question:",
            bg="#0a0c10",
            fg="#38bdf8",
            font=("Segoe UI", 8, "bold")
        )
        self.q_title_lbl.pack(anchor="w", pady=(0, 2))

        self.q_card = tk.Frame(
            self.q_section,
            bg="#121622",
            highlightbackground="#1e2638",
            highlightthickness=1,
            padx=8,
            pady=4
        )
        self.q_card.pack(fill="x")

        self.question_label = tk.Label(
            self.q_card,
            text="Listening for interviewer...",
            bg="#121622",
            fg="#f8fafc",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            justify="left",
            wraplength=480
        )
        self.question_label.pack(fill="x")

        # B. Separate Answer Section
        self.ans_section = tk.Frame(self.content_container, bg="#0a0c10")
        self.ans_section.pack(side="top", fill="both", expand=True)

        self.ans_title_lbl = tk.Label(
            self.ans_section,
            text="⭐ Answer:",
            bg="#0a0c10",
            fg="#fbbf24",
            font=("Segoe UI", 8, "bold")
        )
        self.ans_title_lbl.pack(anchor="w", pady=(0, 2))

        self.ans_card = tk.Frame(
            self.ans_section,
            bg="#0f121a",
            highlightbackground="#1c202d",
            highlightthickness=1
        )
        self.ans_card.pack(fill="both", expand=True)

        self.text_widget = tk.Text(
            self.ans_card,
            bg="#0f121a",
            fg="#f8fafc",
            font=("Segoe UI", self.font_size),
            wrap="word",
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=6
        )
        self.text_widget.pack(fill="both", expand=True)

        # Configure Syntax & Highlight Tags
        self._configure_text_tags()

        self.text_widget.insert("1.0", "Copilot is ready and listening...")
        # Teleprompter Controller (Feature 7)
        try:
            from teleprompter import TeleprompterController
            self.teleprompter = TeleprompterController(self.text_widget, self.root, on_status_update=self._on_teleprompter_status)
        except Exception:
            self.teleprompter = None

        # Global Hotkey Bindings
        self.root.bind("<Alt-c>", self.toggle_click_through)
        self.root.bind("<Control-Shift-X>", self.toggle_click_through)
        self.root.bind("<Alt-k>", self.copy_answer_to_clipboard)
        self.root.bind("<Alt-p>", self.toggle_pill_mode)
        self.root.bind("<Alt-t>", self.toggle_teleprompter)
        self.root.bind("<Alt-s>", self.cycle_teleprompter_speed)
        self.root.bind("<Alt-h>", self.toggle_visibility)
        self.root.bind("<Escape>", lambda e: self.hide())
        self.root.bind("<Alt-Up>", self.increase_opacity)
        self.root.bind("<Alt-Down>", self.decrease_opacity)
        self.root.bind("<Alt-plus>", self.increase_font_size)
        self.root.bind("<Alt-equal>", self.increase_font_size)
        self.root.bind("<Alt-minus>", self.decrease_font_size)

        # Apply Windows API flags once rendered
        self.root.update_idletasks()
        self.hwnd = self.root.winfo_id()

        hide_from_screen_capture(self.hwnd)
        set_click_through(self.hwnd, True)

        # Start audio wave heartbeat
        self._animate_audio_wave()

    def set_question(self, question: str):
        """Sets interviewer question in the dedicated Question section."""
        self.current_question = question
        try:
            self.question_label.config(text=question)
        except Exception:
            pass

    def _on_teleprompter_status(self, status: str):
        try:
            self.status_label.config(text=f"[Teleprompter: {status}]", fg="#38bdf8")
            self.root.after(2000, lambda: self.status_label.config(text="[Alt+C Click-Thru | Alt+T Scroll | Alt+K Copy]", fg="#68d391"))
        except Exception:
            pass

    def toggle_teleprompter(self, event=None):
        """Toggles hands-free teleprompter auto-scrolling."""
        if hasattr(self, "teleprompter") and self.teleprompter:
            is_active = self.teleprompter.toggle()
            status = "ON" if is_active else "OFF"
            self.status_label.config(text=f"● Teleprompter {status} ({self.teleprompter.wpm} WPM)", fg="#38bdf8" if is_active else "#f6ad55")

    def cycle_teleprompter_speed(self, event=None):
        """Cycles teleprompter reading speed (normal -> fast -> slow)."""
        if hasattr(self, "teleprompter") and self.teleprompter:
            new_preset = self.teleprompter.cycle_speed()
            self.status_label.config(text=f"● Speed: {new_preset.upper()} ({self.teleprompter.wpm} WPM)", fg="#38bdf8")

    def update_audio_level(self, source: str, level: float, rms: float = 0.0):
        """Updates the live audio level indicator based on real-time acoustic input."""
        try:
            bars_count = max(1, min(5, int(level * 5) + 1)) if level > 0.05 else 1
            wave_str = " ▂▃▄▅"[:bars_count]
            src_label = "Mic" if "mic" in str(source).lower() else "Spk"
            color = "#4fd1c5" if "mic" in str(source).lower() else "#63b3ed"
            self.audio_indicator.config(text=f"● [{src_label}: {wave_str}]", fg=color)
        except Exception:
            pass

    def _configure_text_tags(self):
        """Sets up syntax highlighting & Parakeet AI scannable teleprompter tags."""
        base_f = ("Segoe UI", self.font_size)
        bold_f = ("Segoe UI", self.font_size, "bold")
        mono_f = ("Consolas", max(9, self.font_size - 1))
        mono_bold_f = ("Consolas", max(9, self.font_size - 1), "bold")

        # Teleprompter Structure Tags
        self.text_widget.tag_configure("bullet_symbol", foreground="#34D399", font=bold_f)
        self.text_widget.tag_configure("bold_lead", foreground="#38BDF8", font=bold_f)
        self.text_widget.tag_configure("complexity_time", foreground="#A78BFA", font=bold_f)
        self.text_widget.tag_configure("complexity_space", foreground="#FBBF24", font=bold_f)
        self.text_widget.tag_configure("section_hook", foreground="#F8FAFC", font=bold_f)

        # Code Syntax Highlighting Tags
        self.text_widget.tag_configure("keyword", foreground="#C678DD", font=mono_bold_f)
        self.text_widget.tag_configure("string", foreground="#98C379", font=mono_f)
        self.text_widget.tag_configure("comment", foreground="#64748B", font=("Consolas", max(9, self.font_size - 1), "italic"))
        self.text_widget.tag_configure("code_block", foreground="#E2E8F0", background="#1A2234", font=mono_f)
        self.text_widget.tag_configure("copied_banner", foreground="#FFFFFF", background="#059669", font=("Segoe UI", 9, "bold"))

        # Configure widget line spacing
        self.text_widget.config(spacing1=2, spacing2=2, spacing3=4)

    def _animate_audio_wave(self):
        """Gentle animation indicating active listening."""
        if self.audio_active:
            self._audio_wave_idx = (self._audio_wave_idx + 1) % len(self._wave_frames)
            current_wave = self._wave_frames[self._audio_wave_idx]
            self.audio_indicator.config(text=f"● [Mic: {current_wave}]")
        self.root.after(350, self._animate_audio_wave)

    def apply_syntax_highlighting(self):
        """
        Colorizes real-time answer content into Parakeet AI / ChatGPT scannable formatting:
        - Emerald bullet points (•)
        - Cyan bold lead-in concepts (**Concept**: ...)
        - Purple / Amber Big-O complexity tags
        - High-contrast syntax-highlighted code blocks
        """
        content = self.text_widget.get("1.0", tk.END)
        
        # Remove existing syntax tags
        for tag in ["keyword", "string", "comment", "bullet_symbol", "bold_lead", "complexity_time", "complexity_space"]:
            self.text_widget.tag_remove(tag, "1.0", tk.END)

        # 1. Highlight Bullet Symbols
        for match in re.finditer(r"^[ \t]*([•\-\*])[ \t]+", content, flags=re.MULTILINE):
            start = f"1.0 + {match.start(1)} chars"
            end = f"1.0 + {match.end(1)} chars"
            self.text_widget.tag_add("bullet_symbol", start, end)

        # 2. Highlight Bold Lead-in Headers (**Keyword / Topic**:)
        for match in re.finditer(r"(\*\*([^*]+)\*\*[:\s]?)", content):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            self.text_widget.tag_add("bold_lead", start, end)

        # 3. Highlight Big-O Complexity (Time: O(...) / Space: O(...))
        for match in re.finditer(r"\bTime(?:\s*Complexity)?\s*:\s*(O\([^\)]+\))", content, flags=re.IGNORECASE):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            self.text_widget.tag_add("complexity_time", start, end)

        for match in re.finditer(r"\bSpace(?:\s*Complexity)?\s*:\s*(O\([^\)]+\))", content, flags=re.IGNORECASE):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            self.text_widget.tag_add("complexity_space", start, end)

        # 4. Code Syntax Highlighting
        keywords = r"\b(def|class|return|import|from|if|else|elif|for|while|try|except|async|await|const|let|var|function|public|private|void|int|bool|string)\b"
        for match in re.finditer(keywords, content):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            self.text_widget.tag_add("keyword", start, end)

        # Strings regex
        for match in re.finditer(r'(".*?"|\'.*?\')', content):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            self.text_widget.tag_add("string", start, end)

        # Comments regex
        for match in re.finditer(r'(#.*?$|//.*?$)', content, flags=re.MULTILINE):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            self.text_widget.tag_add("comment", start, end)

    def copy_answer_to_clipboard(self, event=None):
        """Copies full answer or extracted code block to the clipboard with visual flash."""
        if not self.current_text:
            return
        
        text_to_copy = self.current_text.strip()
        # If code block ``` exists, extract clean code
        code_matches = re.findall(r"```(?:\w+)?\n?(.*?)```", text_to_copy, re.DOTALL)
        if code_matches:
            text_to_copy = "\n\n".join(code_matches).strip()

        self.root.clipboard_clear()
        self.root.clipboard_append(text_to_copy)

        # Flash visual banner in status
        prev_status = self.status_label.cget("text")
        prev_color = self.status_label.cget("fg")
        self.status_label.config(text="✓ Copied to Clipboard!", fg="#38b2ac")
        self.root.after(1500, lambda: self.status_label.config(text=prev_status, fg=prev_color))

    def increase_font_size(self, event=None):
        """Increases font size for larger 1440p/4K monitors."""
        self.font_size = min(18, self.font_size + 1)
        self.text_widget.config(font=("Segoe UI", self.font_size))
        self._configure_text_tags()
        self.apply_syntax_highlighting()

    def decrease_font_size(self, event=None):
        """Decreases font size for compact view."""
        self.font_size = max(8, self.font_size - 1)
        self.text_widget.config(font=("Segoe UI", self.font_size))
        self._configure_text_tags()
        self.apply_syntax_highlighting()

    def toggle_click_through(self, event=None):
        """Toggles click-through mode."""
        self.click_through = not self.click_through
        set_click_through(self.hwnd, self.click_through)
        status_text = "● Click-Through ON" if self.click_through else "● Interactive Mode"
        color = "#68d391" if self.click_through else "#f6ad55"
        self.status_label.config(text=status_text, fg=color)

    def toggle_pill_mode(self, event=None):
        """Toggles between compact webcam-aligned 'Pill' mode and normal HUD mode."""
        screen_width = self.root.winfo_screenwidth()
        self.pill_mode = not self.pill_mode

        if self.pill_mode:
            pill_width = 540
            pill_height = 90
            pill_x = int((screen_width - pill_width) / 2)
            pill_y = 10  # Directly beneath the webcam
            self.root.geometry(f"{pill_width}x{pill_height}+{pill_x}+{pill_y}")
            self.header_frame.pack_forget()
            self.text_widget.config(font=("Segoe UI", max(8, self.font_size - 1), "bold"))
        else:
            self.header_frame.pack(fill="x", before=self.text_widget)
            self.root.geometry(f"{self.default_width}x{self.default_height}+{self.default_x}+{self.default_y}")
            self.text_widget.config(font=("Segoe UI", self.font_size))
            self.apply_syntax_highlighting()

    def set_answer(self, text: str):
        """Replaces text with full formatted answer, applies syntax colors and auto-scrolls."""
        self.current_text = text
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, text)
        self.apply_syntax_highlighting()
        self.text_widget.see("1.0")
        self.text_widget.config(state="disabled")

    def append_token(self, token: str):
        """Appends streaming tokens in real time (<300ms perceived latency)."""
        self.current_text += token
        self.text_widget.config(state="normal")
        self.text_widget.insert(tk.END, token)
        if any(c in token for c in ("\n", ".", ":", "*", "`")):
            self.apply_syntax_highlighting()
        self.text_widget.see(tk.END)
        self.text_widget.config(state="disabled")

    def clear(self):
        """Clears text for new question."""
        self.current_text = ""
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.config(state="disabled")

    def toggle_visibility(self, event=None):
        """Quickly hides or restores the overlay window."""
        if self.root.winfo_viewable():
            self.hide()
        else:
            self.show()

    def increase_opacity(self, event=None):
        self.opacity = min(1.0, self.opacity + 0.05)
        try:
            self.root.attributes("-alpha", self.opacity)
        except Exception:
            pass

    def decrease_opacity(self, event=None):
        self.opacity = max(0.2, self.opacity - 0.05)
        try:
            self.root.attributes("-alpha", self.opacity)
        except Exception:
            pass

    def show(self):
        self.root.deiconify()

    def hide(self):
        self.root.withdraw()

    def close(self):
        self.root.destroy()