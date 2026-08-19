import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
import os
import ctypes
from ctypes import wintypes


# ============================================================
# WINDOWS API CONSTANTS
# ============================================================

GWL_EXSTYLE = -20

WS_EX_LAYERED = 0x00080000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TRANSPARENT = 0x00000020

# Windows 10 version 2004+
GA_ROOT = 2
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011


# ============================================================
# WINDOWS API
# ============================================================

user32 = ctypes.windll.user32

SetWindowDisplayAffinity = getattr(
    user32,
    "SetWindowDisplayAffinity",
    None
)

GetAncestor = getattr(
    user32,
    "GetAncestor",
    None
)

GetParent = getattr(
    user32,
    "GetParent",
    None
)

GetWindowLongPtr = getattr(
    user32,
    "GetWindowLongPtrW",
    user32.GetWindowLongW
)

SetWindowLongPtr = getattr(
    user32,
    "SetWindowLongPtrW",
    user32.SetWindowLongW
)


# ============================================================
# SCREEN CAPTURE EXCLUSION
# ============================================================

def get_root_hwnd(window_id):
    """Resolve true OS top-level HWND from Tkinter window id."""
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
        print("SetWindowDisplayAffinity is not available.")
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
        print(
            "Screen capture exclusion error:",
            error
        )
        return False


from interview_ui import InterviewCopilotView
from dotenv import load_dotenv
from groq import Groq
from session_client import client as session_client


# ============================================================
# FLOATING MINIMIZED ICON WIDGET
# ============================================================

class FloatingIconWidget:
    """
    Compact, draggable, round floating circular icon shown when the overlay is minimized.
    Positioned at TOP + CENTER of the primary screen.
    Clicking restores the main Parakit AI overlay.
    """
    def __init__(self, parent_root, on_restore_callback):
        self.parent_root = parent_root
        self.on_restore_callback = on_restore_callback
        self.window = tk.Toplevel(parent_root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        
        # Transparent background for a 100% round circular badge on Windows
        self._transparent_color = "#000001"
        try:
            self.window.attributes("-transparentcolor", self._transparent_color)
            self.window.configure(bg=self._transparent_color)
        except Exception:
            self.window.configure(bg="#0b0d14")

        self.size = 54
        self.canvas = tk.Canvas(
            self.window,
            width=self.size,
            height=self.size,
            bg=self._transparent_color,
            highlightthickness=0,
            cursor="hand2"
        )
        self.canvas.pack(fill="both", expand=True)

        # Backwards compatibility handles
        self.frame = self.canvas
        self.title_lbl = None
        self.icon_lbl = None
        self.status_dot = None

        self.status = "listening"
        self._is_hovered = False
        self._draw_circle()

        # Initial geometry calculation (Top Center)
        self.recenter()

        # Drag and Click handling
        self._start_x = 0
        self._start_y = 0
        self._is_dragging = False

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)

    def _draw_circle(self):
        self.canvas.delete("all")
        # Color schemes based on state
        if self.status == "answer_pending":
            ring_color = "#f59e0b"
            glow_fill = "#1e1b4b" if not self._is_hovered else "#312e81"
            star_color = "#fef08a"
            dot_color = "#f59e0b"
        elif self.status == "answer_ready":
            ring_color = "#38bdf8"
            glow_fill = "#082f49" if not self._is_hovered else "#0369a1"
            star_color = "#38bdf8"
            dot_color = "#38bdf8"
        else: # listening
            ring_color = "#38bdf8" if not self._is_hovered else "#60a5fa"
            glow_fill = "#0f172a" if not self._is_hovered else "#1e293b"
            star_color = "#38bdf8"
            dot_color = "#10b981"

        # Outer glowing circle
        self.canvas.create_oval(2, 2, self.size - 2, self.size - 2, fill=glow_fill, outline=ring_color, width=2)
        # Inner circular highlight
        self.canvas.create_oval(5, 5, self.size - 5, self.size - 5, fill="", outline="#1e293b", width=1)
        # Central Star Icon
        self.canvas.create_text(self.size // 2, (self.size // 2) - 1, text="✦", fill=star_color, font=("Segoe UI", 16, "bold"))
        # Live status mini indicator dot
        self.canvas.create_oval(self.size - 17, self.size - 17, self.size - 5, self.size - 5, fill=dot_color, outline="#0b0d14", width=2)

    def _on_enter(self, event):
        self._is_hovered = True
        self._draw_circle()

    def _on_leave(self, event):
        self._is_hovered = False
        self._draw_circle()

    def calculate_position(self) -> tuple:
        """Dynamically calculates the TOP + CENTER position on the primary screen."""
        try:
            icon_w = self.size
            icon_h = self.size
            screen_w = self.parent_root.winfo_screenwidth()
            x = max(0, int((screen_w - icon_w) / 2))
            y = 12
            return x, y, icon_w, icon_h
        except Exception:
            return 500, 12, 54, 54

    def recenter(self):
        """Positions the minimized icon at the exact top-center of the primary monitor."""
        try:
            x, y, icon_w, icon_h = self.calculate_position()
            self.window.geometry(f"{icon_w}x{icon_h}+{x}+{y}")
        except Exception:
            self.window.geometry("54x54+500+12")

    def _on_press(self, event):
        self._start_x = event.x_root
        self._start_y = event.y_root
        self._win_x = self.window.winfo_x()
        self._win_y = self.window.winfo_y()
        self._is_dragging = False

    def _on_motion(self, event):
        dx = event.x_root - self._start_x
        dy = event.y_root - self._start_y
        if abs(dx) > 3 or abs(dy) > 3:
            self._is_dragging = True
            new_x = self._win_x + dx
            new_y = self._win_y + dy
            self.window.geometry(f"+{new_x}+{new_y}")

    def _on_release(self, event):
        if not self._is_dragging:
            self.on_restore_callback()

    def set_status(self, status_type: str, label_text: str = ""):
        try:
            self.status = status_type
            self._draw_circle()
        except Exception:
            pass

    def show(self):
        try:
            self.recenter()
            self.window.deiconify()
            self.window.lift()
            self.window.attributes("-topmost", True)
            self._apply_stealth()
        except Exception:
            pass

    def hide(self):
        try:
            self.window.withdraw()
        except Exception:
            pass

    def destroy(self):
        try:
            self.window.destroy()
        except Exception:
            pass

    def _apply_stealth(self):
        try:
            self.window.update_idletasks()
            raw_id = self.window.winfo_id()
            hwnd = get_root_hwnd(raw_id) or raw_id
            exclude_from_screen_capture(hwnd)
        except Exception:
            pass


# ============================================================
# ASSISTANT UI
# ============================================================

class AssistantUI:

    def __init__(self, root, start_callback, client=None, on_close_callback=None, initial_mode="copilot"):

        self.root = root
        self.start_callback = start_callback
        self.on_close_callback = on_close_callback

        if client is not None:
            self.client = client
        else:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
            self.client = Groq(api_key=api_key) if api_key else None

        self.resume_path = None

        self.maximized = False

        # Overlay states
        self.overlay_mode = True
        self.click_through = False

        # Minimized & Floating Icon states
        self.is_minimized = False
        self.user_minimized = False
        self.auto_restore_on_question = True
        self.floating_icon = None

        # Active UI Mode: "copilot" or "standard"
        self.active_mode = initial_mode

        # =====================================================
        # COLORS
        # =====================================================

        BG = "#090909"
        HEADER = "#0d0d0d"
        PANEL = "#121212"
        BORDER = "#242424"

        TEXT = "#f5f5f5"
        SECONDARY = "#8b8b8b"
        MUTED = "#606060"

        BUTTON = "#1c1c1c"
        BUTTON_HOVER = "#292929"

        GREEN = "#48d597"

        self.BG = BG
        self.HEADER = HEADER
        self.PANEL = PANEL
        self.BORDER = BORDER
        self.TEXT = TEXT
        self.SECONDARY = SECONDARY
        self.MUTED = MUTED
        self.BUTTON = BUTTON
        self.BUTTON_HOVER = BUTTON_HOVER
        self.GREEN = GREEN

        # =====================================================
        # WINDOW
        # =====================================================

        root.title("AI Interview Assistant")

        try:
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            win_w = min(1280, max(960, int(screen_w * 0.78)))
            win_h = min(880, max(700, int(screen_h * 0.82)))
            pos_x = max(0, (screen_w - win_w) // 2)
            pos_y = max(10, (screen_h - win_h) // 2 - 20)
            root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        except Exception:
            root.geometry("1160x760")

        root.minsize(
            880,
            620
        )

        root.configure(
            bg=BG
        )

        # Borderless window
        root.overrideredirect(True)

        # Always above normal windows
        root.attributes(
            "-topmost",
            True
        )

        # =====================================================
        # OUTER FRAME
        # =====================================================

        self.outer = tk.Frame(
            root,
            bg=BG,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        self.outer.pack(
            fill="both",
            expand=True
        )

        # =====================================================
        # TITLE BAR
        # =====================================================

        self.title_bar = tk.Frame(
            self.outer,
            bg=HEADER,
            height=58
        )

        self.title_bar.pack(
            fill="x"
        )

        self.title_bar.pack_propagate(False)

        # Dragging
        self.title_bar.bind(
            "<ButtonPress-1>",
            self.start_move
        )

        self.title_bar.bind(
            "<B1-Motion>",
            self.do_move
        )

        # =====================================================
        # LOGO
        # =====================================================

        logo = tk.Label(
            self.title_bar,
            text="✦",
            bg=HEADER,
            fg=TEXT,
            font=("Segoe UI", 19, "bold")
        )

        logo.pack(
            side="left",
            padx=(18, 8)
        )

        # =====================================================
        # TITLE
        # =====================================================

        title_container = tk.Frame(
            self.title_bar,
            bg=HEADER
        )

        title_container.pack(
            side="left"
        )

        title = tk.Label(
            title_container,
            text="BluJay AI",
            bg=HEADER,
            fg="#10b981",
            font=("Segoe UI", 12, "bold")
        )

        title.pack(
            anchor="w"
        )

        subtitle = tk.Label(
            title_container,
            text="Real-Time Stealth Copilot • Active",
            bg=HEADER,
            fg=MUTED,
            font=("Segoe UI", 8)
        )

        subtitle.pack(
            anchor="w"
        )

        # =====================================================
        # BACK TO DASHBOARD & MODE SWITCHER TABS
        # =====================================================

        self.tabs_frame = tk.Frame(
            self.title_bar,
            bg=HEADER
        )

        self.tabs_frame.pack(
            side="left",
            padx=(20, 10)
        )

        self.back_btn = tk.Button(
            self.tabs_frame,
            text="← Dashboard",
            command=self.close_window,
            bg="#1e293b",
            fg="#e2e8f0",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=5,
            cursor="hand2"
        )
        self.back_btn.pack(side="left", padx=(0, 10))

        self.copilot_tab = tk.Button(
            self.tabs_frame,
            text="⚡ Real-time Copilot",
            command=lambda: self.switch_mode("copilot"),
            bg="#1e293b" if self.active_mode == "copilot" else "#141414",
            fg="#38bdf8" if self.active_mode == "copilot" else "#94a3b8",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold" if self.active_mode == "copilot" else "normal"),
            padx=12,
            pady=5,
            cursor="hand2"
        )
        self.copilot_tab.pack(side="left", padx=(0, 6))

        self.standard_tab = tk.Button(
            self.tabs_frame,
            text="💬 Standard Assistant",
            command=lambda: self.switch_mode("standard"),
            bg="#1e293b" if self.active_mode == "standard" else "#141414",
            fg="#38bdf8" if self.active_mode == "standard" else "#94a3b8",
            activebackground="#262626",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold" if self.active_mode == "standard" else "normal"),
            padx=12,
            pady=5,
            cursor="hand2"
        )
        self.standard_tab.pack(side="left")

        # =====================================================
        # SESSION TIMER & CREDIT BADGE
        # =====================================================

        self.session_timer_lbl = tk.Label(
            self.title_bar,
            text="⏳ 30:00 / 30m",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=3
        )
        self.session_timer_lbl.pack(
            side="left",
            padx=(10, 6)
        )

        self.credit_badge_lbl = tk.Label(
            self.title_bar,
            text="⚡ Credits: Active",
            bg="#064e3b",
            fg="#34d399",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=3
        )
        self.credit_badge_lbl.pack(
            side="left",
            padx=(0, 8)
        )

        self.shortcuts_btn = tk.Button(
            self.title_bar,
            text="⌨ Shortcuts",
            command=self.show_shortcuts_modal,
            bg="#1e293b",
            fg="#94a3b8",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=3,
            cursor="hand2"
        )
        self.shortcuts_btn.pack(
            side="left",
            padx=(0, 10)
        )

        # =====================================================
        # STATUS
        # =====================================================

        self.status = tk.Label(
            self.title_bar,
            text="● Ready",
            bg=HEADER,
            fg=GREEN,
            font=("Segoe UI", 9)
        )

        self.status.pack(
            side="right",
            padx=(10, 15)
        )

        # =====================================================
        # WINDOW CONTROLS
        # =====================================================

        window_controls = tk.Frame(
            self.title_bar,
            bg=HEADER
        )

        window_controls.pack(
            side="right"
        )

        self.create_window_button(
            window_controls,
            "—",
            self.minimize_window,
            "#bdbdbd"
        ).pack(
            side="left"
        )

        self.create_window_button(
            window_controls,
            "□",
            self.toggle_maximize,
            "#bdbdbd"
        ).pack(
            side="left"
        )

        self.create_window_button(
            window_controls,
            "×",
            self.close_window,
            "#ff6b6b"
        ).pack(
            side="left"
        )

        # =====================================================
        # VIEWS CONTAINER
        # =====================================================

        self.views_container = tk.Frame(
            self.outer,
            bg=BG
        )

        self.views_container.pack(
            fill="both",
            expand=True
        )

        # 1. Parakeet Real-time Copilot View
        self.copilot_container = tk.Frame(
            self.views_container,
            bg=BG
        )
        self.copilot_view = InterviewCopilotView(
            self.copilot_container,
            self.root,
            self.client,
            self
        )

        # 2. Standard Assistant View
        self.standard_container = tk.Frame(
            self.views_container,
            bg=BG
        )

        if self.active_mode == "copilot":
            self.title_bar.pack_forget()
            self.root.minsize(400, 30)
            self.outer.configure(bg="#18191d", highlightthickness=0)
            self.views_container.configure(bg="#18191d")
            self.copilot_container.configure(bg="#18191d")
            self.copilot_container.pack(fill="both", expand=True)
            if hasattr(self.copilot_view, "_resize_overlay"):
                self.copilot_view._resize_overlay(expanded=False)
        else:
            self.standard_container.pack(fill="both", expand=True)

        # =====================================================
        # MAIN CONTENT (STANDARD VIEW)
        #
        # IMPORTANT:
        # Use GRID instead of pack() here.
        #
        # This keeps the input and buttons permanently visible.
        # =====================================================

        content = tk.Frame(
            self.standard_container,
            bg=BG
        )

        content.pack(
            fill="both",
            expand=True,
            padx=22,
            pady=18
        )

        # Four columns
        content.columnconfigure(
            0,
            weight=1
        )

        # Answer row expands
        content.rowconfigure(
            3,
            weight=1
        )

        # =====================================================
        # TOP ROW
        # =====================================================

        top_row = tk.Frame(
            content,
            bg=BG
        )

        top_row.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 14)
        )

        top_row.columnconfigure(
            0,
            weight=1
        )

        # =====================================================
        # RESUME CARD
        # =====================================================

        resume_card = tk.Frame(
            top_row,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        resume_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 7)
        )

        resume_info = tk.Frame(
            resume_card,
            bg=PANEL
        )

        resume_info.pack(
            side="left",
            fill="both",
            expand=True,
            padx=16,
            pady=12
        )

        tk.Label(
            resume_info,
            text="RESUME / CV",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(
            anchor="w"
        )

        self.resume_label = tk.Label(
            resume_info,
            text="No resume selected",
            bg=PANEL,
            fg=SECONDARY,
            font=("Segoe UI", 10)
        )

        self.resume_label.pack(
            anchor="w",
            pady=(4, 0)
        )

        self.resume_button = self.create_button(
            resume_card,
            "＋  Add Resume",
            self.select_resume,
            width=14
        )

        self.resume_button.pack(
            side="right",
            padx=12,
            pady=12
        )

        # =====================================================
        # STATUS CARD
        # =====================================================

        status_card = tk.Frame(
            top_row,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            width=190
        )

        status_card.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(7, 0)
        )

        status_card.pack_propagate(False)

        tk.Label(
            status_card,
            text="ASSISTANT STATUS",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(
            anchor="w",
            padx=15,
            pady=(12, 3)
        )

        self.status_small = tk.Label(
            status_card,
            text="● Ready",
            bg=PANEL,
            fg=GREEN,
            font=("Segoe UI", 10, "bold")
        )

        self.status_small.pack(
            anchor="w",
            padx=15
        )

        # =====================================================
        # CURRENT QUESTION LABEL
        # =====================================================

        self.create_section_label(
            content,
            "CURRENT QUESTION",
            row=1
        )

        # =====================================================
        # QUESTION BOX
        # =====================================================

        question_frame = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        question_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(6, 12)
        )

        self.question_box = ScrolledText(
            question_frame,
            height=4,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            selectbackground="#333333",
            selectforeground=TEXT,
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 11),
            wrap="word",
            padx=12,
            pady=10
        )

        self.question_box.pack(
            fill="both",
            expand=True
        )

        # =====================================================
        # ANSWER SECTION
        # =====================================================

        self.create_section_label(
            content,
            "AI RESPONSE",
            row=3
        )

        # Move answer frame to row 4
        content.rowconfigure(
            4,
            weight=1
        )

        answer_frame = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        answer_frame.grid(
            row=4,
            column=0,
            sticky="nsew",
            pady=(6, 12)
        )

        self.answer_box = ScrolledText(
            answer_frame,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            selectbackground="#333333",
            selectforeground=TEXT,
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 11),
            wrap="word",
            padx=12,
            pady=12
        )

        self.answer_box.pack(
            fill="both",
            expand=True
        )

        # Tags for Answer Box (Professional High-Contrast Typography & Scannability)
        self.answer_box.tag_config("section_header", foreground="#38bdf8", background="#0c233c", font=("Segoe UI", 10, "bold"), spacing1=8, spacing3=4)
        self.answer_box.tag_config("key_header", foreground="#fbbf24", background="#2b2310", font=("Segoe UI", 10, "bold"), spacing1=8, spacing3=4)
        self.answer_box.tag_config("code_header", foreground="#a78bfa", background="#1e1b4b", font=("Segoe UI", 9, "bold"), spacing1=8, spacing3=2)
        self.answer_box.tag_config("code_body", foreground="#67e8f9", background="#090d16", font=("Consolas", 10), spacing1=4, spacing3=4, lmargin1=14, lmargin2=14)
        self.answer_box.tag_config("key_body", foreground="#fef08a", font=("Segoe UI", 10, "bold"), spacing1=2, spacing3=4, lmargin1=10, lmargin2=10)
        self.answer_box.tag_config("bullet_body", foreground="#f1f5f9", font=("Segoe UI", 10), spacing1=2, spacing3=3, lmargin1=12, lmargin2=22)
        self.answer_box.tag_config("body", foreground=TEXT, font=("Segoe UI", 10), spacing1=2, spacing3=4)

        # =====================================================
        # TEXT INPUT
        # =====================================================

        input_frame = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        input_frame.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(0, 10)
        )

        self.placeholder = (
            "Ask an interview question..."
        )

        self.text_input = tk.Entry(
            input_frame,
            bg=PANEL,
            fg=MUTED,
            insertbackground=TEXT,
            selectbackground="#333333",
            selectforeground=TEXT,
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 11)
        )

        self.text_input.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(14, 8),
            pady=13
        )

        self.text_input.insert(
            0,
            self.placeholder
        )

        self.text_input.bind(
            "<FocusIn>",
            self.input_focus_in
        )

        self.text_input.bind(
            "<FocusOut>",
            self.input_focus_out
        )

        self.text_input.bind(
            "<Return>",
            lambda event: self.ask_typed()
        )

        self.type_button = self.create_button(
            input_frame,
            "Send  ➜",
            self.ask_typed,
            width=10
        )

        self.type_button.pack(
            side="right",
            padx=7,
            pady=7
        )

        # =====================================================
        # BOTTOM CONTROLS
        # =====================================================

        controls = tk.Frame(
            content,
            bg=BG
        )

        controls.grid(
            row=6,
            column=0,
            sticky="ew"
        )

        # -----------------------------------------------------
        # MICROPHONE
        # -----------------------------------------------------

        self.button = self.create_button(
            controls,
            "🎤  Start Listening",
            self.start_microphone
        )

        self.button.pack(
            side="left",
            padx=(0, 6)
        )

        # -----------------------------------------------------
        # SYSTEM AUDIO START
        # -----------------------------------------------------

        self.system_start_button = self.create_button(
            controls,
            "🔊  Start Audio",
            self.start_system_audio
        )

        self.system_start_button.pack(
            side="left",
            padx=6
        )

        # -----------------------------------------------------
        # SYSTEM AUDIO STOP
        # -----------------------------------------------------

        self.system_stop_button = self.create_button(
            controls,
            "■  Stop Audio",
            self.stop_system_audio
        )

        self.system_stop_button.pack(
            side="left",
            padx=6
        )

        self.system_stop_button.config(
            state=tk.DISABLED
        )

        # -----------------------------------------------------
        # OVERLAY BUTTON
        # -----------------------------------------------------

        self.overlay_button = self.create_button(
            controls,
            "◉  Overlay ON",
            self.toggle_overlay_mode
        )

        self.overlay_button.pack(
            side="right",
            padx=(6, 0)
        )

        # =====================================================
        # KEYBOARD SHORTCUTS
        # =====================================================

        root.bind(
            "<Control-q>",
            lambda event: self.close_window()
        )

        root.bind(
            "<Control-Q>",
            lambda event: self.close_window()
        )

        root.bind(
            "<Control-w>",
            lambda event: self.close_window()
        )

        root.bind(
            "<Control-W>",
            lambda event: self.close_window()
        )

        root.bind(
            "<Control-Shift-O>",
            lambda event: self.toggle_overlay_mode()
        )

        root.bind(
            "<Control-Shift-o>",
            lambda event: self.toggle_overlay_mode()
        )

        root.bind(
            "<Control-Shift-X>",
            lambda event: self.toggle_click_through()
        )

        root.bind(
            "<Control-Shift-x>",
            lambda event: self.toggle_click_through()
        )

        root.bind(
            "<Control-Shift-M>",
            lambda event: self.toggle_minimize()
        )

        root.bind(
            "<Control-Shift-m>",
            lambda event: self.toggle_minimize()
        )

        root.bind(
            "<Control-Shift-F>",
            lambda event: self.toggle_maximize()
        )

        root.bind(
            "<Control-Shift-f>",
            lambda event: self.toggle_maximize()
        )

        # =====================================================
        # APPLY OVERLAY
        # =====================================================

        self.root.update_idletasks()

        self.apply_overlay_settings()
        self._start_session_timer()

    # =========================================================
    # CREATE NORMAL BUTTON
    # =========================================================

    def create_button(
        self,
        parent,
        text,
        command,
        width=None
    ):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.BUTTON,
            fg=self.TEXT,
            activebackground=self.BUTTON_HOVER,
            activeforeground=self.TEXT,
            disabledforeground="#555555",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=8,
            cursor="hand2"
        )

        if width:
            button.config(
                width=width
            )

        button.bind(
            "<Enter>",
            lambda event: (
                button.config(
                    bg=self.BUTTON_HOVER
                )
                if button["state"] != tk.DISABLED
                else None
            )
        )

        button.bind(
            "<Leave>",
            lambda event: (
                button.config(
                    bg=self.BUTTON
                )
                if button["state"] != tk.DISABLED
                else None
            )
        )

        return button

    # =========================================================
    # WINDOW BUTTON
    # =========================================================

    def create_window_button(
        self,
        parent,
        text,
        command,
        color
    ):

        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#0d0d0d",
            fg=color,
            activebackground="#202020",
            activeforeground=color,
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 12),
            width=4,
            height=2,
            cursor="hand2"
        )

    # =========================================================
    # SECTION LABEL
    # =========================================================

    def create_section_label(
        self,
        parent,
        text,
        row
    ):

        label = tk.Label(
            parent,
            text=text,
            bg=self.BG,
            fg=self.MUTED,
            font=("Segoe UI", 8, "bold")
        )

        label.grid(
            row=row,
            column=0,
            sticky="w"
        )

    # =========================================================
    # MODE SWITCHER
    # =========================================================

    def switch_mode(self, mode):

        self.active_mode = mode

        if mode == "copilot":
            self.title_bar.pack_forget()
            self.standard_container.pack_forget()

            self.copilot_container.pack(
                fill="both",
                expand=True
            )

            self.copilot_tab.config(
                bg="#1e293b",
                fg="#38bdf8",
                font=("Segoe UI", 9, "bold")
            )

            self.standard_tab.config(
                bg="#141414",
                fg="#94a3b8",
                font=("Segoe UI", 9)
            )

            self.set_status(
                "Copilot Active"
            )

        else:
            self.title_bar.pack(fill="x", before=self.views_container)
            self.copilot_container.pack_forget()

            self.standard_container.pack(
                fill="both",
                expand=True
            )

            self.copilot_tab.config(
                bg="#141414",
                fg="#94a3b8",
                font=("Segoe UI", 9)
            )

            self.standard_tab.config(
                bg="#1e293b",
                fg="#38bdf8",
                font=("Segoe UI", 9, "bold")
            )

            self.set_status(
                "Standard Active"
            )

    # =========================================================
    # APPLY OVERLAY SETTINGS
    # =========================================================

    def apply_overlay_settings(self):

        self.root.update_idletasks()

        raw_id = self.root.winfo_id()
        hwnd = get_root_hwnd(raw_id) or raw_id

        # Always on top
        self.root.attributes(
            "-topmost",
            True
        )

        # Ask Windows to exclude this window from all screen capture APIs (Zoom, Meet, Teams, OBS, screenshots)
        capture_hidden = exclude_from_screen_capture(
            hwnd
        )

        print(
            "======================================"
        )

        print(
            "AI INTERVIEW ASSISTANT"
        )

        print(
            "Overlay mode:",
            self.overlay_mode
        )

        print(
            "Always on top: ON"
        )

        print(
            "Screen capture exclusion:",
            capture_hidden
        )

        print(
            "======================================"
        )

    # =========================================================
    # TOGGLE OVERLAY
    # =========================================================

    def toggle_overlay_mode(self):

        self.overlay_mode = not self.overlay_mode

        if self.overlay_mode:

            self.root.attributes(
                "-topmost",
                True
            )

            self.overlay_button.config(
                text="◉  Overlay ON"
            )

            self.set_status(
                "Overlay active"
            )

            print(
                "Overlay mode: ON"
            )

        else:

            self.root.attributes(
                "-topmost",
                False
            )

            self.overlay_button.config(
                text="○  Overlay OFF"
            )

            self.set_status(
                "Normal window"
            )

            print(
                "Overlay mode: OFF"
            )

    # =========================================================
    # CLICK THROUGH
    # =========================================================

    def toggle_click_through(self):

        self.click_through = not self.click_through

        hwnd = self.root.winfo_id()

        style = GetWindowLongPtr(
            wintypes.HWND(hwnd),
            GWL_EXSTYLE
        )

        if self.click_through:

            new_style = (
                style
                | WS_EX_TRANSPARENT
                | WS_EX_LAYERED
                | WS_EX_TOOLWINDOW
            )

            self.set_status(
                "Click-through ON"
            )

            print(
                "Click-through: ON"
            )

        else:

            new_style = (
                style
                & ~WS_EX_TRANSPARENT
            )

            self.set_status(
                "Click-through OFF"
            )

            print(
                "Click-through: OFF"
            )

        SetWindowLongPtr(
            wintypes.HWND(hwnd),
            GWL_EXSTYLE,
            new_style
        )

        self.root.attributes(
            "-topmost",
            self.overlay_mode
        )

    # =========================================================
    # WINDOW DRAGGING
    # =========================================================

    def start_move(self, event):

        self._drag_x = event.x
        self._drag_y = event.y

    def do_move(self, event):

        x = (
            self.root.winfo_x()
            + event.x
            - self._drag_x
        )

        y = (
            self.root.winfo_y()
            + event.y
            - self._drag_y
        )

        self.root.geometry(
            f"+{x}+{y}"
        )

    # =========================================================
    # MINIMIZE & RESTORE (FLOATING ICON)
    # =========================================================

    def minimize_window(self):
        """Hides the main overlay and displays the small floating Parakit icon."""
        if self.is_minimized:
            return

        self.is_minimized = True
        self.user_minimized = True
        self.root.withdraw()

        if self.floating_icon is None:
            self.floating_icon = FloatingIconWidget(self.root, on_restore_callback=self.restore_window)

        self.floating_icon.show()

    def restore_window(self):
        """Restores the main overlay from minimized state."""
        if not self.is_minimized:
            return

        self.is_minimized = False
        self.user_minimized = False

        if self.floating_icon is not None:
            self.floating_icon.hide()

        self.root.deiconify()
        self.root.overrideredirect(True)
        self.root.attributes(
            "-topmost",
            self.overlay_mode
        )
        self.root.lift()
        self.root.focus_force()

        # Re-apply capture setting after window state change
        self.root.after(
            50,
            self.apply_overlay_settings
        )

    def toggle_minimize(self):
        """Toggles between Minimized state (floating icon) and Restored state (full overlay)."""
        if self.is_minimized:
            self.restore_window()
        else:
            self.minimize_window()

    def restore_borderless(self):

        self.root.overrideredirect(
            True
        )

        self.root.attributes(
            "-topmost",
            self.overlay_mode
        )

        # Re-apply capture setting after window state change.
        self.root.after(
            50,
            self.apply_overlay_settings
        )

    # =========================================================
    # MAXIMIZE
    # =========================================================

    def toggle_maximize(self):

        if not self.maximized:

            self.root.overrideredirect(
                False
            )

            self.root.state(
                "zoomed"
            )

            self.root.after(
                100,
                self.restore_borderless
            )

            self.maximized = True

        else:

            self.root.overrideredirect(
                False
            )

            self.root.state(
                "normal"
            )

            self.root.geometry(
                "1020x720"
            )

            self.root.after(
                100,
                self.restore_borderless
            )

            self.maximized = False

    # =========================================================
    # SHORTCUTS MODAL
    # =========================================================

    def show_shortcuts_modal(self):
        """Displays small, elegant reference of active registered shortcuts."""
        dialog = tk.Toplevel(self.root)
        dialog.title("BluJay AI Shortcuts")
        dialog.geometry("450x400")
        dialog.configure(bg="#0b0d14")
        dialog.attributes("-topmost", True)
        dialog.transient(self.root)

        # Stealth capture protection for shortcuts dialog
        try:
            dialog.update_idletasks()
            raw_id = dialog.winfo_id()
            hwnd = get_root_hwnd(raw_id) or raw_id
            exclude_from_screen_capture(hwnd)
        except Exception:
            pass

        title_frame = tk.Frame(dialog, bg="#0b0d14")
        title_frame.pack(fill="x", padx=20, pady=(18, 10))

        tk.Label(
            title_frame,
            text="⌨️ BluJay AI Shortcuts",
            bg="#0b0d14",
            fg="#f8fafc",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="Active keyboard shortcuts for seamless operation.",
            bg="#0b0d14",
            fg="#94a3b8",
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(2, 0))

        shortcuts_card = tk.Frame(dialog, bg="#121624", highlightbackground="#28334e", highlightthickness=1)
        shortcuts_card.pack(fill="both", expand=True, padx=20, pady=6)

        shortcuts_list = [
            ("Ctrl + Shift + M", "Minimize to Floating Icon / Restore Overlay"),
            ("Ctrl + Shift + O", "Toggle Always-On-Top Mode"),
            ("Ctrl + Shift + X", "Toggle Stealth Click-Through Mode"),
            ("Ctrl + Shift + F", "Toggle Maximize Window Mode"),
            ("Spacebar", "Quick AI Answer Generation (Copilot)"),
            ("Alt + S", "Solve Screen Coding Problem (Vision AI)"),
            ("Alt + O", "Cycle Window Transparency / Opacity (100% - 60%)"),
            ("Ctrl + Q / W", "Close Assistant & Return to Dashboard")
        ]

        for key, desc in shortcuts_list:
            row = tk.Frame(shortcuts_card, bg="#121624")
            row.pack(fill="x", padx=12, pady=4)

            badge = tk.Label(
                row,
                text=key,
                bg="#1e293b",
                fg="#38bdf8",
                font=("Segoe UI", 9, "bold"),
                padx=8,
                pady=3
            )
            badge.pack(side="left")

            tk.Label(
                row,
                text=desc,
                bg="#121624",
                fg="#e2e8f0",
                font=("Segoe UI", 9)
            ).pack(side="left", padx=10)

        close_btn = tk.Button(
            dialog,
            text="Got It (Close)",
            command=dialog.destroy,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
            padx=16,
            pady=6,
            cursor="hand2"
        )
        close_btn.pack(pady=(8, 16))

    # =========================================================
    # CLOSE
    # =========================================================

    def close_window(self):
        if self.floating_icon is not None:
            try:
                self.floating_icon.destroy()
            except Exception:
                pass

        try:
            session_client.end_session()
        except Exception:
            pass

        if self.on_close_callback:
            try:
                self.on_close_callback()
            except Exception:
                pass

        self.root.destroy()

    def _start_session_timer(self):
        self._update_session_timer()

    def _update_session_timer(self):
        try:
            if not self.root.winfo_exists():
                return
            
            # 1. Update remaining time in 30-min block
            rem_sec = session_client.get_remaining_seconds()
            mins = rem_sec // 60
            secs = rem_sec % 60
            self.session_timer_lbl.config(text=f"⏳ {mins:02d}:{secs:02d} / 30m")

            # 2. Update credit badge
            if session_client.is_unlimited:
                self.credit_badge_lbl.config(text="👑 PRO UNLIMITED", fg="#facc15", bg="#713f12")
            else:
                formatted_credits = round(session_client.credit_units / 100.0, 2)
                self.credit_badge_lbl.config(text=f"⚡ {formatted_credits} Credits", fg="#34d399", bg="#064e3b")
        except Exception:
            pass
        
        # Ticker every 1000ms
        try:
            self.root.after(1000, self._update_session_timer)
        except Exception:
            pass

    # =========================================================
    # INPUT FOCUS
    # =========================================================

    def input_focus_in(self, event):

        if self.text_input.get() == self.placeholder:

            self.text_input.delete(
                0,
                tk.END
            )

            self.text_input.config(
                fg=self.TEXT
            )

    def input_focus_out(self, event):

        if not self.text_input.get().strip():

            self.text_input.insert(
                0,
                self.placeholder
            )

            self.text_input.config(
                fg=self.MUTED
            )

    # =========================================================
    # RESUME
    # =========================================================

    def select_resume(self):

        file_path = filedialog.askopenfilename(
            title="Select Resume / CV",
            filetypes=[
                (
                    "Resume files",
                    "*.pdf *.docx *.txt"
                ),
                (
                    "PDF files",
                    "*.pdf"
                ),
                (
                    "Word files",
                    "*.docx"
                ),
                (
                    "Text files",
                    "*.txt"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not file_path:
            return

        self.resume_path = file_path

        filename = os.path.basename(
            file_path
        )

        self.resume_label.config(
            text="✓ " + filename,
            fg=self.GREEN
        )

        self.set_status(
            "📄 Resume loaded"
        )

        self.start_callback(
            "__RESUME__:" + file_path
        )

    # =========================================================
    # STATUS
    # =========================================================

    def set_status(self, text):
        if hasattr(self, "status") and self.status:
            try:
                self.status.config(text="● " + text)
            except Exception:
                pass

        if hasattr(self, "status_small") and self.status_small:
            try:
                self.status_small.config(text="● " + text)
            except Exception:
                pass

        lower = text.lower()
        if "error" in lower:
            color = "#ff6b6b"
        elif (
            "thinking" in lower
            or "recording" in lower
            or "listening" in lower
            or "converting" in lower
            or "generating" in lower
            or "stopping" in lower
        ):
            color = "#f0c75e"
        else:
            color = self.GREEN

        if hasattr(self, "status") and self.status:
            try:
                self.status.config(fg=color)
            except Exception:
                pass

        if hasattr(self, "status_small") and self.status_small:
            try:
                self.status_small.config(fg=color)
            except Exception:
                pass

    # =========================================================
    # QUESTION
    # =========================================================

    def set_question(self, text):

        self.question_box.delete(
            "1.0",
            tk.END
        )

        self.question_box.insert(
            tk.END,
            text
        )

    # =========================================================
    # ANSWER
    # =========================================================

    def set_answer(self, text):

        self.answer_box.delete(
            "1.0",
            tk.END
        )

        self.answer_box.insert(
            tk.END,
            text
        )

    # =========================================================
    # TYPED QUESTION
    # =========================================================

    def ask_typed(self):

        question = (
            self.text_input
            .get()
            .strip()
        )

        if (
            not question
            or question == self.placeholder
        ):
            return

        # Show question immediately
        self.set_question(
            question
        )

        # Clear input
        self.text_input.delete(
            0,
            tk.END
        )

        # Status
        self.set_status(
            "Thinking..."
        )

        # Send to backend
        self.start_callback(
            question
        )

    # =========================================================
    # MICROPHONE
    # =========================================================

    def start_microphone(self):

        self.set_status(
            "🎤 Listening..."
        )

        self.start_callback()

    # =========================================================
    # SYSTEM AUDIO START
    # =========================================================

    def start_system_audio(self):

        self.system_start_button.config(
            state=tk.DISABLED
        )

        self.system_stop_button.config(
            state=tk.NORMAL
        )

        self.set_status(
            "🔊 Recording computer audio..."
        )

        self.start_callback(
            "__START_SYSTEM_AUDIO__"
        )

    # =========================================================
    # SYSTEM AUDIO STOP
    # =========================================================

    def stop_system_audio(self):

        self.system_stop_button.config(
            state=tk.DISABLED
        )

        self.set_status(
            "🛑 Stopping recording..."
        )

        self.start_callback(
            "__STOP_SYSTEM_AUDIO__"
        )

    # =========================================================
    # CLOSE / RETURN TO DASHBOARD
    # =========================================================

    def close_window(self):
        if hasattr(self, "copilot_view") and self.copilot_view:
            try:
                self.copilot_view.cleanup_resources()
            except Exception as ce:
                print("copilot_view cleanup error:", ce)
        if self.on_close_callback:
            try:
                self.on_close_callback()
            except Exception as e:
                print("on_close_callback error:", e)
        self.root.destroy()

    # =========================================================
    # VIEW MODE SWITCHER
    # =========================================================

    def switch_mode(self, mode):
        """Switch active view inside session between 'copilot' and 'standard'."""
        self.active_mode = mode

        if hasattr(self, "copilot_container") and self.copilot_container:
            self.copilot_container.pack_forget()
        if hasattr(self, "standard_container") and self.standard_container:
            self.standard_container.pack_forget()

        if mode == "copilot":
            if hasattr(self, "copilot_container") and self.copilot_container:
                self.copilot_container.pack(fill="both", expand=True)
            if hasattr(self, "copilot_tab") and self.copilot_tab:
                self.copilot_tab.config(bg="#1e293b", fg="#38bdf8", font=("Segoe UI", 9, "bold"))
            if hasattr(self, "standard_tab") and self.standard_tab:
                self.standard_tab.config(bg="#141414", fg="#94a3b8", font=("Segoe UI", 9))
            self.set_status("Session Active")
        elif mode == "standard":
            if hasattr(self, "standard_container") and self.standard_container:
                self.standard_container.pack(fill="both", expand=True)
            if hasattr(self, "copilot_tab") and self.copilot_tab:
                self.copilot_tab.config(bg="#141414", fg="#94a3b8", font=("Segoe UI", 9))
            if hasattr(self, "standard_tab") and self.standard_tab:
                self.standard_tab.config(bg="#1e293b", fg="#38bdf8", font=("Segoe UI", 9, "bold"))
            self.set_status("Standard Assistant")