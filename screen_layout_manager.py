"""
Screen Layout Manager & Responsive Auto-Sizing Engine
=====================================================
Detects actual Windows usable work area (excluding taskbars/docks) across single and
multi-monitor environments, handles DPI scaling, calculates safe margins, and dynamically
sizes the Live Interview UI and dialogs so content is never clipped or overflowing.
"""

import sys
import ctypes
from typing import Dict, Optional, Tuple


def enable_dpi_awareness():
    """
    Enables Windows Per-Monitor DPI awareness to prevent blurry scaling
    and ensure exact physical pixel to logical unit mapping.
    """
    if sys.platform == "win32":
        try:
            # Windows 10 1703+ Per-Monitor V2
            SetProcessDpiAwarenessContext = getattr(ctypes.windll.user32, "SetProcessDpiAwarenessContext", None)
            if SetProcessDpiAwarenessContext:
                # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
                SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
                return
        except Exception:
            pass

        try:
            # Windows 8.1+ Per-Monitor DPI
            SetProcessDpiAwareness = getattr(ctypes.windll.shcore, "SetProcessDpiAwareness", None)
            if SetProcessDpiAwareness:
                # 2 = PROCESS_PER_MONITOR_DPI_AWARE
                SetProcessDpiAwareness(2)
                return
        except Exception:
            pass

        try:
            # Windows Vista+ System DPI
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def get_usable_work_area(
    root=None,
    monitor_id: Optional[int] = None,
    hwnd: Optional[int] = None
) -> Dict[str, int]:
    """
    Returns exact work area bounds {x, y, width, height, is_primary} for the target monitor,
    excluding Windows taskbars (bottom, top, left, or right) and docked toolbars.
    """
    if sys.platform == "win32":
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32

            class MONITORINFOEXW(ctypes.Structure):
                _fields_ = [
                    ('cbSize', wintypes.DWORD),
                    ('rcMonitor', wintypes.RECT),
                    ('rcWork', wintypes.RECT),
                    ('dwFlags', wintypes.DWORD),
                    ('szDevice', wintypes.WCHAR * 32)
                ]

            monitors = []

            def _monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
                info = MONITORINFOEXW()
                info.cbSize = ctypes.sizeof(MONITORINFOEXW)
                if user32.GetMonitorInfoW(hMonitor, ctypes.byref(info)):
                    rw = info.rcWork
                    rm = info.rcMonitor
                    is_prim = bool(info.dwFlags & 1)
                    monitors.append({
                        "id": len(monitors),
                        "x": int(rw.left),
                        "y": int(rw.top),
                        "width": int(rw.right - rw.left),
                        "height": int(rw.bottom - rw.top),
                        "mon_x": int(rm.left),
                        "mon_y": int(rm.top),
                        "mon_width": int(rm.right - rm.left),
                        "mon_height": int(rm.bottom - rm.top),
                        "is_primary": is_prim,
                        "device": str(info.szDevice).strip()
                    })
                return True

            MonitorEnumProc = ctypes.WINFUNCTYPE(
                wintypes.BOOL,
                wintypes.HMONITOR,
                wintypes.HDC,
                ctypes.POINTER(wintypes.RECT),
                wintypes.LPARAM
            )
            cb = MonitorEnumProc(_monitor_enum_proc)
            user32.EnumDisplayMonitors(None, None, cb, 0)

            if monitors:
                # If a specific monitor index was requested and exists
                if monitor_id is not None and 0 <= monitor_id < len(monitors):
                    return monitors[monitor_id]

                # If hwnd is provided, find the monitor containing the window
                if hwnd:
                    MONITOR_DEFAULTTONEAREST = 2
                    h_mon = user32.MonitorFromWindow(wintypes.HWND(hwnd), MONITOR_DEFAULTTONEAREST)
                    info = MONITORINFOEXW()
                    info.cbSize = ctypes.sizeof(MONITORINFOEXW)
                    if user32.GetMonitorInfoW(h_mon, ctypes.byref(info)):
                        rw = info.rcWork
                        return {
                            "id": 0,
                            "x": int(rw.left),
                            "y": int(rw.top),
                            "width": int(rw.right - rw.left),
                            "height": int(rw.bottom - rw.top),
                            "is_primary": bool(info.dwFlags & 1)
                        }

                # Default to primary monitor
                for m in monitors:
                    if m["is_primary"]:
                        return m
                return monitors[0]

        except Exception as e:
            pass

        # Fallback to SPI_GETWORKAREA on Windows
        try:
            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long)
                ]

            SPI_GETWORKAREA = 0x0030
            rect = RECT()
            if ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
                w = int(rect.right - rect.left)
                h = int(rect.bottom - rect.top)
                if w > 100 and h > 100:
                    return {
                        "id": 0,
                        "x": int(rect.left),
                        "y": int(rect.top),
                        "width": w,
                        "height": h,
                        "is_primary": True
                    }
        except Exception:
            pass

    # Generic Tkinter fallback (safe taskbar assumption of ~48px)
    try:
        if root and hasattr(root, "winfo_screenwidth"):
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
        else:
            sw, sh = 1920, 1080
    except Exception:
        sw, sh = 1920, 1080

    return {
        "id": 0,
        "x": 0,
        "y": 0,
        "width": max(640, sw),
        "height": max(480, sh - 48),  # Subtract default taskbar height
        "is_primary": True
    }


def calculate_copilot_overlay_geometry(
    work_area: Dict[str, int],
    expanded: bool = True,
    custom_width_ratio: Optional[float] = None
) -> Tuple[int, int, int, int, str]:
    """
    Calculates responsive geometry (x, y, width, height, geom_string) for the Interview Copilot HUD.
    
    Rules:
    - Expanded:
      - Width: Scales cleanly (65-74% on standard displays, bounded between 840px and 1140px, never exceeding work_w - 20).
      - Height: Scales to 65-76% of work height on standard screens; on compact laptop screens (height <= 768),
        fits within work_h - 32 so bottom controls & input are NEVER clipped.
      - Pos: Centered horizontally on target display, top margin 8-16px.
    - Collapsed:
      - Height: ~94-98px compact capsule HUD.
      - Width: Matches the expanded width for smooth non-jittery transitions.
    """
    work_x = work_area.get("x", 0)
    work_y = work_area.get("y", 0)
    work_w = max(600, work_area.get("width", 1920))
    work_h = max(450, work_area.get("height", 1080))

    # 1. Responsive Width Calculation
    if custom_width_ratio:
        raw_w = int(work_w * custom_width_ratio)
    elif work_w >= 2560:
        # Ultrawide / 4K monitors: keep comfortable reading width
        raw_w = min(1180, int(work_w * 0.50))
    elif work_w >= 1920:
        # Standard 1080p / 1440p
        raw_w = min(1080, int(work_w * 0.65))
    elif work_w >= 1440:
        # 1600x900 or 1440x900
        raw_w = min(1020, int(work_w * 0.72))
    elif work_w >= 1280:
        # 1366x768 / 1280x720 laptops
        raw_w = min(960, int(work_w * 0.78))
    else:
        # Small sub-1280 displays
        raw_w = int(work_w * 0.90)

    # Safe bounds
    min_safe_w = min(780, work_w - 24)
    max_safe_w = max(min_safe_w, work_w - 24)
    win_w = max(min_safe_w, min(max_safe_w, raw_w))

    # 2. Responsive Height Calculation
    if not expanded:
        win_h = 210
    else:
        if work_h <= 600:
            # Very short screen (e.g. low-res display or vertical split)
            win_h = max(380, work_h - 24)
        elif work_h <= 740:
            # 1366x768 laptop screen (work_h ~680-720 after taskbar)
            win_h = max(520, int(work_h * 0.82))
            win_h = min(win_h, work_h - 30)
        elif work_h <= 900:
            # 1600x900 or 1080p with 125% scaling (work_h ~820-860)
            win_h = max(580, min(720, int(work_h * 0.75)))
        elif work_h <= 1100:
            # 1080p standard (work_h ~1040)
            win_h = max(620, min(760, int(work_h * 0.70)))
        else:
            # 1440p / 4K high-res display
            win_h = max(680, min(840, int(work_h * 0.65)))

    # 3. Position Calculation (Top-Centered on active display)
    pos_x = work_x + max(0, (work_w - win_w) // 2)
    top_margin = max(6, min(16, int(work_h * 0.015)))
    pos_y = work_y + top_margin

    geom_str = f"{win_w}x{win_h}+{pos_x}+{pos_y}"
    return pos_x, pos_y, win_w, win_h, geom_str


def calculate_dialog_geometry(
    work_area: Dict[str, int],
    target_w: int = 640,
    target_h: int = 680
) -> Tuple[int, int, int, int, str]:
    """
    Safely calculates centered geometry for modal dialogs so they never exceed the screen work area.
    """
    work_x = work_area.get("x", 0)
    work_y = work_area.get("y", 0)
    work_w = max(500, work_area.get("width", 1920))
    work_h = max(400, work_area.get("height", 1080))

    safe_w = min(target_w, work_w - 30)
    safe_h = min(target_h, work_h - 40)

    pos_x = work_x + max(0, (work_w - safe_w) // 2)
    pos_y = work_y + max(0, (work_h - safe_h) // 2)

    geom_str = f"{safe_w}x{safe_h}+{pos_x}+{pos_y}"
    return pos_x, pos_y, safe_w, safe_h, geom_str
