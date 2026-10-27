"""
Interactive Screen Region Snipper Overlay (Feature 3)
=====================================================
Provides a sleek, semi-transparent fullscreen canvas allowing candidates
to click-and-drag to snip any coding question, diagram, or IDE region
and immediately forward it to the Multimodal Vision AI solver.
"""

import os
import tempfile
import tkinter as tk
from typing import Callable, Optional, Tuple
from PIL import ImageGrab


class InteractiveRegionSnipper:
    """
    Fullscreen overlay for interactive mouse drag-to-crop rectangle selection.
    """

    def __init__(self, root: tk.Tk, on_snip_completed: Callable[[str], None], on_cancel: Optional[Callable[[], None]] = None):
        self.parent_root = root
        self.on_snip_completed = on_snip_completed
        self.on_cancel = on_cancel
        
        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.rect_id: Optional[int] = None
        self.dim_text_id: Optional[int] = None
        
        # Build fullscreen snip window
        self.win = tk.Toplevel(root)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.35)
        self.win.configure(bg="#000000")
        self.win.overrideredirect(True)
        
        # Span across virtual screen (all connected monitors)
        screen_x, screen_y = 0, 0
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        if os.name == "nt":
            try:
                import ctypes
                u32 = ctypes.windll.user32
                vx = u32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
                vy = u32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
                vw = u32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
                vh = u32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
                if vw > 0 and vh > 0:
                    screen_x, screen_y, screen_w, screen_h = vx, vy, vw, vh
            except Exception:
                pass

        self.win.geometry(f"{screen_w}x{screen_h}+{screen_x}+{screen_y}")
        self.win.config(cursor="cross")

        # Apply OS-level capture exclusion (WDA_EXCLUDEFROMCAPTURE = 0x11) so snipper is invisible to screen shares
        if os.name == "nt":
            self.win.update_idletasks()
            try:
                import ctypes
                from ctypes import wintypes
                u32 = ctypes.windll.user32
                w_id = self.win.winfo_id()
                parent_hwnd = u32.GetParent(w_id)
                target_hwnd = parent_hwnd if parent_hwnd else w_id
                u32.SetWindowDisplayAffinity(wintypes.HWND(target_hwnd), wintypes.DWORD(0x11))
            except Exception:
                pass

        self.canvas = tk.Canvas(
            self.win,
            width=screen_w,
            height=screen_h,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Instructions banner
        self.canvas.create_text(
            screen_w // 2,
            40,
            text="✂️ Drag to select region to solve (Press ESC to cancel)",
            fill="#ffffff",
            font=("Segoe UI", 13, "bold")
        )

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.win.bind("<Escape>", self._on_escape)
        self.win.bind("<Button-3>", self._on_escape)  # Right click to cancel

    def _on_press(self, event):
        self.start_x = self.win.winfo_pointerx()
        self.start_y = self.win.winfo_pointery()
        # Canvas coordinates
        cx = event.x
        cy = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.dim_text_id:
            self.canvas.delete(self.dim_text_id)
            self.dim_text_id = None
        self.rect_id = self.canvas.create_rectangle(
            cx, cy, cx, cy,
            outline="#38bdf8",
            width=2,
            fill="#0284c7"
        )

    def _on_drag(self, event):
        if not self.rect_id or self.start_x is None:
            return
        # Calculate canvas coords
        cur_cx = event.x
        cur_cy = event.y
        
        # Start canvas coords
        start_cx = self.start_x - self.win.winfo_rootx()
        start_cy = self.start_y - self.win.winfo_rooty()
        
        self.canvas.coords(self.rect_id, start_cx, start_cy, cur_cx, cur_cy)
        
        # Live HUD badge
        w_px = abs(cur_cx - start_cx)
        h_px = abs(cur_cy - start_cy)
        dim_str = f" {w_px} × {h_px} px "
        badge_x = max(start_cx, cur_cx) - 40
        badge_y = max(start_cy, cur_cy) + 16
        
        if not self.dim_text_id:
            self.dim_text_id = self.canvas.create_text(
                badge_x, badge_y,
                text=dim_str,
                fill="#38bdf8",
                font=("Segoe UI", 10, "bold")
            )
        else:
            self.canvas.coords(self.dim_text_id, badge_x, badge_y)
            self.canvas.itemconfig(self.dim_text_id, text=dim_str)

    def _on_release(self, event):
        if self.start_x is None or self.start_y is None:
            self._close()
            return
            
        end_x = self.win.winfo_pointerx()
        end_y = self.win.winfo_pointery()
        
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        w = x2 - x1
        h = y2 - y1
        
        # Close snip overlay window before taking actual screenshot
        self._close()
        
        if w >= 20 and h >= 20:
            # Dispatch capture promptly
            self.parent_root.after(15, lambda: self._capture_and_dispatch((x1, y1, x2, y2)))
        else:
            if self.on_cancel:
                self.on_cancel()

    def _capture_and_dispatch(self, bbox: Tuple[int, int, int, int]):
        try:
            try:
                from vision import capture_screen_image as vision_capture
                temp_path = vision_capture(bbox=bbox)
            except Exception:
                try:
                    screenshot = ImageGrab.grab(bbox=bbox, all_screens=True)
                except Exception:
                    screenshot = ImageGrab.grab(bbox=bbox)
                    
                temp_path = os.path.join(tempfile.gettempdir(), f"interview_snip_{os.getpid()}.png")
                screenshot.save(temp_path, format="PNG")
            self.on_snip_completed(temp_path)
        except Exception as e:
            print(f"Error capturing region snip: {e}")
            if self.on_cancel:
                self.on_cancel()

    def _on_escape(self, event=None):
        self._close()
        if self.on_cancel:
            self.on_cancel()

    def _close(self):
        try:
            self.win.destroy()
        except Exception:
            pass
