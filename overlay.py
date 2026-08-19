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
else:
    SetWindowLongPtr = None
    GetWindowLongPtr = None
    SetWindowDisplayAffinity = None
    GetAncestor = None
    GetParent = None


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
# EXCLUDE FROM SCREEN CAPTURE (STEALTH)
# ============================================================

def hide_from_screen_capture(hwnd: int) -> bool:
    """
    Excludes the window from screen sharing (Zoom, Google Meet, MS Teams, OBS, Discord, etc.).
    The window remains completely visible to the candidate.
    """
    if SetWindowDisplayAffinity is None or not hwnd:
        return False
    actual_hwnd = get_root_hwnd(hwnd) or hwnd
    try:
        # Primary: WDA_EXCLUDEFROMCAPTURE (0x11)
        res = SetWindowDisplayAffinity(wintypes.HWND(actual_hwnd), wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE))
        if res:
            return True
        # Fallback: WDA_MONITOR (0x01)
        res = SetWindowDisplayAffinity(wintypes.HWND(actual_hwnd), wintypes.DWORD(WDA_MONITOR))
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
        self.root.title("Copilot Stealth Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#121214")

        # States
        self.click_through = True
        self.pill_mode = False
        self.opacity = 0.92
        self.font_size = 10
        self.current_text = ""
        self.audio_active = True
        self._audio_wave_idx = 0
        self._wave_frames = [" ▂▃", "▂▄▆", "▃▅█", " ▃▅", "  ▂"]

        # Dimensions & Positioning
        screen_width = self.root.winfo_screenwidth()
        self.default_width = 480
        self.default_height = 230
        self.default_x = screen_width - self.default_width - 30
        self.default_y = 60
        self.root.geometry(f"{self.default_width}x{self.default_height}+{self.default_x}+{self.default_y}")

        try:
            self.root.attributes("-alpha", self.opacity)
        except Exception:
            pass

        # UI Container
        self.frame = tk.Frame(
            self.root,
            bg="#121214",
            highlightbackground="#2d3748",
            highlightthickness=1
        )
        self.frame.pack(fill="both", expand=True)

        # Header Bar
        self.header_frame = tk.Frame(self.frame, bg="#1a202c")
        self.header_frame.pack(fill="x", padx=0, pady=0)

        self.title_label = tk.Label(
            self.header_frame,
            text="⚡ Copilot HUD",
            bg="#1a202c",
            fg="#63b3ed",
            font=("Segoe UI", 9, "bold")
        )
        self.title_label.pack(side="left", padx=10, pady=4)

        self.audio_indicator = tk.Label(
            self.header_frame,
            text="● [Mic: ▂▄▆]",
            bg="#1a202c",
            fg="#4fd1c5",
            font=("Segoe UI", 8, "bold")
        )
        self.audio_indicator.pack(side="left", padx=5, pady=4)

        self.status_label = tk.Label(
            self.header_frame,
            text="[Alt+C Click-Thru | Alt+K Copy | Alt+P Pill]",
            bg="#1a202c",
            fg="#68d391",
            font=("Segoe UI", 8)
        )
        self.status_label.pack(side="right", padx=10, pady=4)

        # Content Text Widget with Styled Tags
        self.text_widget = tk.Text(
            self.frame,
            bg="#121214",
            fg="#e2e8f0",
            font=("Segoe UI", self.font_size),
            wrap="word",
            bd=0,
            highlightthickness=0,
            padx=10,
            pady=8
        )
        self.text_widget.pack(fill="both", expand=True)

        # Configure Syntax & Highlight Tags
        self._configure_text_tags()

        self.text_widget.insert("1.0", "Copilot is ready and listening...")
        self.text_widget.config(state="disabled")

        # Global Hotkey Bindings
        self.root.bind("<Alt-c>", self.toggle_click_through)
        self.root.bind("<Control-Shift-X>", self.toggle_click_through)
        self.root.bind("<Alt-k>", self.copy_answer_to_clipboard)
        self.root.bind("<Alt-p>", self.toggle_pill_mode)
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

    def _configure_text_tags(self):
        """Sets up syntax highlighting tags for code and structured responses."""
        self.text_widget.tag_configure("bullet", foreground="#63b3ed", font=("Segoe UI", self.font_size, "bold"))
        self.text_widget.tag_configure("keyword", foreground="#c678dd", font=("Consolas", self.font_size, "bold"))
        self.text_widget.tag_configure("string", foreground="#98c379", font=("Consolas", self.font_size))
        self.text_widget.tag_configure("comment", foreground="#718096", font=("Consolas", self.font_size, "italic"))
        self.text_widget.tag_configure("code_block", foreground="#e2e8f0", background="#1a202c", font=("Consolas", self.font_size))
        self.text_widget.tag_configure("copied_banner", foreground="#ffffff", background="#319795", font=("Segoe UI", 9, "bold"))

    def _animate_audio_wave(self):
        """Gentle animation indicating active listening."""
        if self.audio_active:
            self._audio_wave_idx = (self._audio_wave_idx + 1) % len(self._wave_frames)
            current_wave = self._wave_frames[self._audio_wave_idx]
            self.audio_indicator.config(text=f"● [Mic: {current_wave}]")
        self.root.after(350, self._animate_audio_wave)

    def apply_syntax_highlighting(self):
        """Colorizes code blocks and key terms in real time."""
        content = self.text_widget.get("1.0", tk.END)
        
        # Remove existing syntax tags
        for tag in ["keyword", "string", "comment", "bullet"]:
            self.text_widget.tag_remove(tag, "1.0", tk.END)

        # Keywords regex
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