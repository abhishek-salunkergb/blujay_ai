"""
Screen & Coding Interview Problem Solver (Vision Integration)
============================================================
Captures screen / active window of coding problems (LeetCode, HackerRank, Codeforces, IDEs)
and generates optimal algorithmic solutions, clean code, and complexity analysis.
Supports multi-monitor discovery, targeted display selection, and custom region cropping.
"""

import os
import sys
import json
import base64
import tempfile
import ctypes
from typing import Callable, Optional, List, Dict, Tuple, Any
from PIL import Image, ImageGrab
from groq import Groq

VISION_CONFIG_FILE = os.path.expanduser("~/.ai_copilot_vision.json")


def get_available_monitors() -> List[Dict[str, Any]]:
    """
    Enumerates all connected physical displays with resolutions, bounding boxes, and primary flags.
    Uses Win32 EnumDisplayMonitors on Windows with PIL/pyautogui fallbacks.
    """
    monitors: List[Dict[str, Any]] = []
    
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

            def _monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
                r = lprcMonitor.contents
                info = MONITORINFOEXW()
                info.cbSize = ctypes.sizeof(MONITORINFOEXW)
                user32.GetMonitorInfoW(hMonitor, ctypes.byref(info))
                
                is_primary = bool(info.dwFlags & 1)
                left = int(r.left)
                top = int(r.top)
                right = int(r.right)
                bottom = int(r.bottom)
                width = right - left
                height = bottom - top
                
                dev_name = str(info.szDevice).strip()
                mon_id = len(monitors)
                primary_tag = " (Primary)" if is_primary else ""
                
                monitors.append({
                    "id": mon_id,
                    "name": f"Display {mon_id + 1}{primary_tag} - {width}x{height}",
                    "device": dev_name,
                    "bbox": (left, top, right, bottom),
                    "width": width,
                    "height": height,
                    "is_primary": is_primary
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
        except Exception as e:
            print(f"Notice during Win32 EnumDisplayMonitors: {e}")

    if not monitors:
        # Fallback using PIL.ImageGrab or virtual display
        try:
            im = ImageGrab.grab()
            w, h = im.size
        except Exception:
            w, h = 1920, 1080
            
        monitors.append({
            "id": 0,
            "name": f"Primary Display - {w}x{h}",
            "device": "DISPLAY1",
            "bbox": (0, 0, w, h),
            "width": w,
            "height": h,
            "is_primary": True
        })
        
    return monitors


def get_saved_monitor_preference() -> int:
    """Load saved monitor index preference (defaults to 0)."""
    if os.path.exists(VISION_CONFIG_FILE):
        try:
            with open(VISION_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("monitor_id", 0))
        except Exception:
            pass
    return 0


def save_monitor_preference(monitor_id: int):
    """Save monitor index preference to file."""
    try:
        with open(VISION_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"monitor_id": int(monitor_id)}, f, indent=2)
    except Exception as e:
        print(f"Notice saving vision monitor preferences: {e}")


def capture_screen_image(
    monitor_id: Optional[int] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> str:
    """
    Takes a screenshot of the specified monitor or subregion bounding box (x1, y1, x2, y2).
    Saves to a temporary PNG file and returns the file path.
    """
    screenshot = None
    target_bbox = bbox

    # If specific monitor is requested and no explicit crop bbox given, look up monitor's bbox
    if target_bbox is None and monitor_id is not None:
        mons = get_available_monitors()
        for m in mons:
            if m["id"] == monitor_id:
                target_bbox = m["bbox"]
                break

    # If neither was provided, check saved preference
    if target_bbox is None and monitor_id is None:
        saved_id = get_saved_monitor_preference()
        if saved_id > 0:
            mons = get_available_monitors()
            for m in mons:
                if m["id"] == saved_id:
                    target_bbox = m["bbox"]
                    break

    # Attempt capture with PIL ImageGrab
    try:
        if target_bbox:
            # On Windows, all_screens=True allows coordinates on secondary monitors
            try:
                screenshot = ImageGrab.grab(bbox=target_bbox, all_screens=True)
            except Exception:
                screenshot = ImageGrab.grab(bbox=target_bbox)
        else:
            screenshot = ImageGrab.grab(all_screens=False)
    except Exception as e:
        print(f"ImageGrab error: {e}")

    # Fallback with PyAutoGUI if ImageGrab failed
    if screenshot is None:
        try:
            import pyautogui
            if target_bbox:
                x1, y1, x2, y2 = target_bbox
                w = max(10, x2 - x1)
                h = max(10, y2 - y1)
                screenshot = pyautogui.screenshot(region=(x1, y1, w, h))
            else:
                screenshot = pyautogui.screenshot()
        except Exception as pe:
            print(f"PyAutoGUI screenshot fallback error: {pe}")
            screenshot = Image.new("RGB", (1280, 720), color=(15, 23, 42))

    temp_path = os.path.join(tempfile.gettempdir(), f"interview_screen_capture_{os.getpid()}.png")
    screenshot.save(temp_path, format="PNG")
    return temp_path


def encode_image_base64(image_path: str) -> str:
    """Encodes an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def solve_screen_coding_problem(
    client: Groq,
    on_token: Callable[[str], None],
    custom_hint: Optional[str] = None,
    monitor_id: Optional[int] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    image_path: Optional[str] = None
) -> str:
    """
    Captures on-screen coding problem (or uses provided image_path) and streams the optimal code solution.
    """
    created_temp = False
    if not image_path or not os.path.exists(image_path):
        image_path = capture_screen_image(monitor_id=monitor_id, bbox=bbox)
        created_temp = True
        
    base64_image = encode_image_base64(image_path)
    
    prompt = """
You are an expert Competitive Programmer and FAANG Coding Interview Specialist.
Analyze the coding problem displayed in this screenshot.

Structure your response clearly for an interview setting:
1. 💡 **Problem Summary & Core Edge Cases**
2. 🧠 **Optimal Approach & Intuition** (Why this data structure / algorithm was chosen)
3. 💻 **Clean, Production-Ready Code** (Python by default or language shown in screenshot, with clear variable names and comments)
4. ⏱️ **Complexity Analysis**:
   - Time Complexity: O(...)
   - Space Complexity: O(...)
5. 🗣️ **Interview Talking Points** (Key sentences candidate should say to explain this solution aloud)

Keep it crisp, accurate, and immediately actionable.
"""
    if custom_hint:
        prompt += f"\n\nCandidate's specific instruction/language preference: {custom_hint}"
        
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    answer = ""
    # Streaming filter to completely strip out <think>...</think> reasoning tags
    try:
        from interview_engine import StreamingReasoningFilter, sanitize_generated_answer
        stream_filter = StreamingReasoningFilter(on_token)
    except Exception:
        stream_filter = None

    def _safe_emit(tok: str):
        if stream_filter:
            stream_filter.push(tok)
        else:
            on_token(tok)

    # Try active multimodal vision models
    vision_models = ["qwen/qwen3.6-27b", "llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]
    for v_model in vision_models:
        try:
            stream = client.chat.completions.create(
                model=v_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                stream=True
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                tok = chunk.choices[0].delta.content
                if tok:
                    answer += tok
                    _safe_emit(tok)
            if stream_filter:
                stream_filter.flush()
            if answer.strip():
                break
        except Exception as e:
            print(f"Vision model notice on {v_model}: {e}")
            continue

    if not answer.strip():
        # Fallback to direct coding prompt if image_url isn't supported by the active model
        text_messages = [
            {"role": "system", "content": "You are a FAANG Senior Coding Interview Specialist. Provide structured optimal solutions with code, approach, and complexity. Never output think tags."},
            {"role": "user", "content": f"Please provide a complete, optimal interview solution with code and Big-O complexity for the coding problem on screen.\n[Context: {custom_hint or 'Coding Challenge'}]"}
        ]
        for fallback_model in ["openai/gpt-oss-20b", "groq/compound", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"]:
            try:
                curr_max = 2048 if ("qwen" in fallback_model or "120b" in fallback_model) else 1000
                stream = client.chat.completions.create(
                    model=fallback_model,
                    messages=text_messages,
                    temperature=0.3,
                    max_tokens=curr_max,
                    stream=True
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    tok = chunk.choices[0].delta.content
                    if tok:
                        answer += tok
                        _safe_emit(tok)
                if stream_filter:
                    stream_filter.flush()
                if answer.strip():
                    break
            except Exception:
                continue
            
    if created_temp:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass
        
    try:
        from interview_engine import sanitize_generated_answer
        clean_final = sanitize_generated_answer(answer)
    except Exception:
        import re
        clean_final = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    if not clean_final.strip():
        fallback = "⚠️ Could not analyze screenshot. Please make sure the coding problem is clearly visible on your screen."
        on_token(fallback)
        return fallback
        
    return clean_final
