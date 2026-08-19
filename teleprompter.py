"""
Teleprompter Auto-Scroll & Read-Pacing Guide (Feature 7)
========================================================
Provides hands-free smooth auto-scrolling for AI-generated answers at natural speaking pace,
allowing candidates to maintain steady eye contact with interviewers on webcam.
Includes adjustable speech pacing (100-180 WPM), hover-pause detection, and hotkey control.
"""

import time
import tkinter as tk
from typing import Optional, Callable


class TeleprompterController:
    """
    Manages smooth, non-blocking auto-scrolling for Tkinter Text / ScrolledText widgets.
    """

    SPEED_PRESETS = {
        "slow": 100,      # 100 words per minute (deliberate / complex system design)
        "normal": 135,    # 135 words per minute (natural interview pacing)
        "fast": 170       # 170 words per minute (rapid summary / overview)
    }

    def __init__(self, text_widget: tk.Text, root: tk.Tk, on_status_update: Optional[Callable[[str], None]] = None):
        self.text_widget = text_widget
        self.root = root
        self.on_status_update = on_status_update

        self.is_scrolling = False
        self.wpm = self.SPEED_PRESETS["normal"]
        self.current_preset = "normal"
        self._user_interaction_paused_until = 0.0
        self._after_job_id: Optional[str] = None
        self._scroll_interval_ms = 50  # 20 ticks per second for 60fps smooth scrolling

        # Bind mouse/interaction overrides to pause temporarily
        if self.text_widget:
            self.text_widget.bind("<MouseWheel>", self._on_manual_interaction, add="+")
            self.text_widget.bind("<Button-1>", self._on_manual_interaction, add="+")
            self.text_widget.bind("<B1-Motion>", self._on_manual_interaction, add="+")

    def toggle(self) -> bool:
        """Toggles play/pause state."""
        if self.is_scrolling:
            self.stop()
        else:
            self.start()
        return self.is_scrolling

    def start(self):
        """Starts smooth auto-scroll."""
        if self.is_scrolling:
            return
        self.is_scrolling = True
        if self.on_status_update:
            self.on_status_update(f"▶ Scrolling ({self.wpm} WPM)")
        self._tick()

    def stop(self):
        """Stops auto-scroll."""
        self.is_scrolling = False
        if self._after_job_id:
            try:
                self.root.after_cancel(self._after_job_id)
            except Exception:
                pass
            self._after_job_id = None
        if self.on_status_update:
            self.on_status_update("⏸️ Paused")

    def set_speed_preset(self, preset: str):
        """Sets speed preset ('slow', 'normal', 'fast')."""
        if preset in self.SPEED_PRESETS:
            self.current_preset = preset
            self.wpm = self.SPEED_PRESETS[preset]
            if self.is_scrolling and self.on_status_update:
                self.on_status_update(f"▶ Scrolling ({self.wpm} WPM)")

    def cycle_speed(self) -> str:
        """Cycles speed preset: normal -> fast -> slow -> normal."""
        order = ["normal", "fast", "slow"]
        curr_idx = order.index(self.current_preset) if self.current_preset in order else 0
        next_preset = order[(curr_idx + 1) % len(order)]
        self.set_speed_preset(next_preset)
        return next_preset

    def _on_manual_interaction(self, event=None):
        """Pauses auto-scroll for 2.5 seconds when user manually scrolls or clicks."""
        self._user_interaction_paused_until = time.time() + 2.5

    def _tick(self):
        """Scroll tick executing every _scroll_interval_ms."""
        if not self.is_scrolling:
            return

        now = time.time()
        if now < self._user_interaction_paused_until:
            # User temporarily paused interaction
            self._after_job_id = self.root.after(self._scroll_interval_ms, self._tick)
            return

        try:
            if self.text_widget and self.text_widget.winfo_exists():
                y_view = self.text_widget.yview()
                # y_view is (top_fraction, bottom_fraction) e.g. (0.0, 0.45)
                if y_view[1] >= 0.999:
                    # Reached end of text
                    self.stop()
                    if self.on_status_update:
                        self.on_status_update("✓ End of Answer")
                    return

                # Calculate vertical fraction step per tick based on WPM
                # ~135 words/min ~= 2.25 words/sec. Typical page is 150 words.
                step_fraction = (self.wpm / 150.0) * (self._scroll_interval_ms / 60000.0) * 0.4
                new_top = min(1.0, y_view[0] + step_fraction)
                self.text_widget.yview_moveto(new_top)
        except Exception:
            pass

        if self.is_scrolling:
            self._after_job_id = self.root.after(self._scroll_interval_ms, self._tick)
