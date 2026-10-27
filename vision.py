"""
Screen & Coding Interview Problem Solver (Dual Gemini Multimodal Vision & Windows OCR)
=====================================================================================
Captures screen / active window of coding problems (LeetCode, HackerRank, Codeforces, IDEs, SQL, Diagrams)
and generates optimal algorithmic solutions, clean code, and complexity analysis.
Features:
1. Direct Google Gemini 3.6 Flash Multimodal Vision (Pixel-level understanding of multiple code blocks)
2. Hardware-accelerated Windows Native OCR (winocr) + Groq Qwen 2.5 27B / GPT-OSS 120B
3. Multi-monitor discovery and interactive region cropping
"""

import os
import sys
import json
import base64
import tempfile
import uuid
import ctypes
import re
import asyncio
from typing import Callable, Optional, List, Dict, Tuple, Any
from PIL import Image, ImageGrab, ImageEnhance
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

VISION_CONFIG_FILE = os.path.expanduser("~/.ai_copilot_vision.json")

# Attempt to load winocr for hardware-accelerated Windows Native OCR
try:
    import winocr  # type: ignore
    HAS_WINOCR = True
except Exception:
    winocr = None  # type: ignore
    HAS_WINOCR = False

_CACHED_OPENAI_CLIENT = None
_CACHED_OPENAI_KEY = None
_CACHED_GEMINI_VISION_CLIENT = None
_CACHED_GEMINI_VISION_KEY = None
import concurrent.futures
_OCR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="WinOCR")


def get_openai_vision_client():
    """Returns or initializes a persistent keep-alive OpenAI client for Vision API calls."""
    global _CACHED_OPENAI_CLIENT, _CACHED_OPENAI_KEY
    openai_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if openai_key:
        if _CACHED_OPENAI_CLIENT is None or _CACHED_OPENAI_KEY != openai_key:
            import importlib
            openai_mod = importlib.import_module("openai")
            _CACHED_OPENAI_CLIENT = openai_mod.OpenAI(api_key=openai_key, max_retries=1, timeout=25.0)
            _CACHED_OPENAI_KEY = openai_key
        return _CACHED_OPENAI_CLIENT
    return None


def get_gemini_vision_client():
    """Returns or initializes a persistent Gemini client for Vision fallback."""
    global _CACHED_GEMINI_VISION_CLIENT, _CACHED_GEMINI_VISION_KEY
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        if _CACHED_GEMINI_VISION_CLIENT is None or _CACHED_GEMINI_VISION_KEY != gemini_key:
            from google import genai
            _CACHED_GEMINI_VISION_CLIENT = genai.Client(api_key=gemini_key)
            _CACHED_GEMINI_VISION_KEY = gemini_key
        return _CACHED_GEMINI_VISION_CLIENT
    return None


def get_available_monitors() -> List[Dict[str, Any]]:
    """
    Enumerates all connected physical displays with resolutions, bounding boxes, and primary flags.
    Uses Win32 EnumDisplayMonitors on Windows with PIL/pyautogui fallbacks.
    """
    monitors: List[Dict[str, Any]] = []
    
    if sys.platform == "win32":
        try:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
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

    if target_bbox is None and monitor_id is not None:
        mons = get_available_monitors()
        for m in mons:
            if m["id"] == monitor_id:
                target_bbox = m["bbox"]
                break

    if target_bbox is None and monitor_id is None:
        saved_id = get_saved_monitor_preference()
        if saved_id > 0:
            mons = get_available_monitors()
            for m in mons:
                if m["id"] == saved_id:
                    target_bbox = m["bbox"]
                    break

    try:
        if target_bbox:
            try:
                screenshot = ImageGrab.grab(bbox=target_bbox, all_screens=True)
            except Exception:
                screenshot = ImageGrab.grab(bbox=target_bbox)
        else:
            screenshot = ImageGrab.grab(all_screens=False)
    except Exception as e:
        print(f"ImageGrab error: {e}")

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

    unique_id = uuid.uuid4().hex[:8]
    temp_path = os.path.join(tempfile.gettempdir(), f"interview_screen_{os.getpid()}_{unique_id}.png")
    screenshot.save(temp_path, format="PNG")
    return temp_path


def detect_coding_language_from_context(text: str) -> str:
    """
    Detects the active programming language from extracted OCR text or problem description.
    Defaults to 'Python' if ambiguous or multiple candidates exist.
    """
    if not text:
        return "Python"
    t = text.lower()
    
    if "python" in t or "def " in text or "self." in text or "import sys" in text:
        return "Python"
    if "java" in t or "public class" in text or "system.out" in t or "string[] args" in t:
        return "Java"
    if "c++" in t or "cpp" in t or "#include" in text or "std::" in text or "vector<" in text:
        return "C++"
    if "typescript" in t or "interface " in text or ": string" in text or ": number" in text:
        return "TypeScript"
    if "javascript" in t or "const " in text or "let " in text or "console.log" in text:
        return "JavaScript"
    if "golang" in t or "func " in text or "package main" in text:
        return "Go"
    if "rust" in t or "fn " in text or "impl " in text or "mut " in text:
        return "Rust"
    if "sql" in t or ("select " in t and "from " in t):
        return "SQL"
        
    return "Python"


def extract_text_from_image(image_path: str) -> str:
    """
    Performs high-accuracy OCR on the captured screenshot using Windows Native OCR
    with adaptive contrast and luminance equalization for dark-mode and light-mode IDEs.
    """
    if not os.path.exists(image_path):
        return ""

    if not HAS_WINOCR:
        return ""

    try:
        img = Image.open(image_path)
        gray = img.convert("L")
        
        # Adaptive contrast & brightness normalization based on background luminance
        try:
            stat = gray.resize((64, 64))
            pixels = list(stat.getdata())
            avg_lum = sum(pixels) / len(pixels)
        except Exception:
            avg_lum = 128.0

        if avg_lum < 110.0:
            # Dark-mode IDE / Terminal (VS Code Dark, Dracula, Monokai)
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.9)
        else:
            # Light-mode Browser / LeetCode Light
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.6)

        async def _ocr():
            res = await winocr.recognize_pil(enhanced, "en")
            return res.text

        try:
            # Use persistent OCR background executor to eliminate COM apartment violations (RPC_E_WRONG_THREAD)
            try:
                loop = asyncio.get_running_loop()
                return _OCR_EXECUTOR.submit(lambda: asyncio.run(_ocr())).result(timeout=5.0)
            except RuntimeError:
                # No running loop on caller thread — execute directly or via worker
                return _OCR_EXECUTOR.submit(lambda: asyncio.run(_ocr())).result(timeout=5.0)
        except Exception:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_ocr())
            finally:
                loop.close()
    except Exception as e:
        print(f"OCR extraction notice: {e}")
        return ""


def encode_image_base64(image_path: str) -> str:
    """Encodes an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_optimized_image_base64(image_path: str, max_width: int = 1920) -> Tuple[str, str]:
    """
    Optimizes and downscales screenshot in-memory.
    Uses LOSSLESS PNG for code/dark-mode screenshots (preserves thin character strokes)
    and compressed JPEG for light/natural content.
    Returns (base64_string, mime_type).
    """
    try:
        with Image.open(image_path) as img:
            # Proportionally downscale if wider than max_width
            if img.width > max_width:
                ratio = max_width / float(img.width)
                new_height = int(float(img.height) * ratio)
                resized_img = img.resize((max_width, new_height), Image.Resampling.BILINEAR)
            else:
                resized_img = img

            rgb_img = resized_img.convert("RGB") if resized_img.mode in ("RGBA", "P") else resized_img

            # Detect code content via background luminance (dark-mode IDEs have avg lum < 100)
            try:
                thumb = rgb_img.resize((64, 64))
                gray_pixels = [sum(p) / 3 for p in thumb.getdata()]
                avg_lum = sum(gray_pixels) / len(gray_pixels)
            except Exception:
                avg_lum = 128.0

            import io
            buf = io.BytesIO()
            if avg_lum < 100.0:
                # Dark-mode code screenshot: use lossless PNG to preserve thin monospace strokes
                rgb_img.save(buf, format="PNG", optimize=True)
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                return b64_str, "image/png"
            else:
                # Light/natural content: use high-quality JPEG (quality=90)
                rgb_img.save(buf, format="JPEG", quality=90, optimize=True)
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                return b64_str, "image/jpeg"
    except Exception as e:
        print(f"Image optimization notice: {e}, using raw PNG fallback.")
        return encode_image_base64(image_path), "image/png"


def solve_screen_coding_problem(
    client: Optional[Groq] = None,
    on_token: Optional[Callable[[str], None]] = None,
    custom_hint: Optional[str] = None,
    monitor_id: Optional[int] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    image_path: Optional[str] = None
) -> str:
    """
    Captures on-screen coding problem (or uses provided image_path) and produces
    the 100% optimal, production-ready code solution with complexity analysis.
    Uses direct Google Gemini 3.6 Flash Native Vision + Windows OCR / Groq Qwen.
    """
    if on_token is None:
        on_token = lambda t: None

    created_temp = False
    if not image_path or not os.path.exists(image_path):
        image_path = capture_screen_image(monitor_id=monitor_id, bbox=bbox)
        created_temp = True

    prompt = """You are an elite Competitive Programming World Champion and FAANG Principal Engineer.
A coding problem was captured directly from the candidate's screen.
Extract the EXACT problem statement, ignoring any extraneous browser UI, tabs, or desktop chrome.

Provide a 100% mathematically correct, optimal, bug-free, and production-ready solution structured as follows:

1. 💡 **Problem Summary & Key Edge Cases**:
   - Restate the core problem, inputs, outputs, and constraints.
   - List key edge cases (e.g., empty array, single element, negative numbers, duplicates).

2. 🧠 **Optimal Approach & Intuition**:
   - 1-2 crisp sentences explaining the optimal algorithmic pattern (e.g., Two-Pointer In-Place, Sliding Window, Dynamic Programming, Hash Map).

3. 💻 **Clean, Production-Ready Code**:
   - You MUST write the complete, optimal, bug-free implementation enclosed in a single ```python (or detected target language) markdown code block.
   - Include type annotations, docstrings, and comprehensive edge-case handling.
   - ZERO placeholders, ZERO comments like "# implement here", ZERO "pass", and ZERO pseudo-code. Provide the complete runnable code.

4. ⏱️ **Complexity Analysis**:
   - • **Time Complexity**: O(...) with 1-line exact rationale
   - • **Space Complexity**: O(...) with 1-line exact rationale

5. 🗣️ **Candidate Talking Points**:
   - 3-4 crisp, high-impact bullet points with bold lead-ins that the candidate can read aloud to explain the solution to the interviewer.
"""
    if custom_hint:
        prompt += f"\n\nCandidate's preference / hint: {custom_hint.strip()}"

    answer = ""
    try:
        from interview_engine import StreamingReasoningFilter
        stream_filter = StreamingReasoningFilter(on_token)
    except Exception:
        stream_filter = None

    def _safe_emit(tok: str):
        if stream_filter:
            stream_filter.push(tok)
        else:
            on_token(tok)

    is_mock = client is not None and (hasattr(client, "_mock_return_value") or hasattr(client, "_mock_methods") or "Mock" in type(client).__name__)

    # =========================================================================
    # TIER 1: OPENAI GPT-4o MULTIMODAL VISION (Optimized Streaming) [PRIMARY]
    # =========================================================================
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not is_mock and openai_key and os.path.exists(image_path):
        try:
            o_client = get_openai_vision_client()
            b64_img, mime_type = encode_optimized_image_base64(image_path)
            v_model = os.getenv("DEFAULT_VISION_MODEL", "gpt-4o").strip()
            stream = o_client.chat.completions.create(
                model=v_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}}
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=0.1,
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
        except Exception as oe:
            print(f"OpenAI Vision streaming notice: {oe}, falling back to Gemini / Groq...")

    # =========================================================================
    # TIER 2: GOOGLE GEMINI MULTIMODAL VISION (Direct Image) [FALLBACK]
    # =========================================================================
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not is_mock and not answer.strip() and gemini_key and os.path.exists(image_path):
        primary_gemini = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        gemini_vision_models = [
            primary_gemini,
            "gemini-flash-latest",
            "gemini-3.6-flash",
            "gemini-2.5-flash"
        ]
        gemini_vision_models = list(dict.fromkeys(gemini_vision_models))
        for g_mod in gemini_vision_models:
            try:
                g_client = get_gemini_vision_client()
                if not g_client:
                    break
                with Image.open(image_path) as pil_img:
                    pil_img.load()
                    img_rgb = pil_img.convert("RGB") if pil_img.mode in ("RGBA", "P") else pil_img.copy()
                    g_resp = g_client.models.generate_content(
                        model=g_mod,
                        contents=[img_rgb, prompt]
                    )
                if g_resp and g_resp.text and g_resp.text.strip():
                    answer = g_resp.text
                    _safe_emit(answer)
                    break
            except Exception as ge:
                err_str = str(ge)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"Gemini vision quota limit reached on {g_mod} -> Trying next fallback.")
                else:
                    print(f"Gemini vision notice on {g_mod}: {ge}")
                continue

    # =========================================================================
    # TIER 3: WINDOWS OCR + GROQ REASONING MODELS (Llama 3.3 70B / Qwen 27B)
    # =========================================================================
    if not answer.strip():
        extracted_text = extract_text_from_image(image_path)
        detected_lang = detect_coding_language_from_context(extracted_text)
        
        ocr_prompt = prompt
        if detected_lang != "Python":
            ocr_prompt += f"\n\n[DETECTED TARGET LANGUAGE]: {detected_lang} (Provide production-ready {detected_lang} implementation)"
        if extracted_text.strip():
            ocr_prompt += f"\n\n### [EXACT TEXT EXTRACTED FROM SCREEN]:\n{extracted_text.strip()}"

        # Initialize Groq client if not passed
        if client is None:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                client = Groq(api_key=groq_key)

        if client is not None:
            reasoning_models = [
                "llama-3.3-70b-versatile",
                "qwen/qwen3.6-27b",
                "llama-3.1-8b-instant"
            ]

            messages = [
                {
                    "role": "system",
                    "content": "You are a FAANG Principal Interview Specialist. You MUST deliver complete, production-ready, runnable code blocks with optimal Big-O complexity and talking points. Never omit code."
                },
                {
                    "role": "user",
                    "content": ocr_prompt
                }
            ]

            for model_name in reasoning_models:
                try:
                    curr_max = 3500
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.1,
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
                except Exception as e:
                    print(f"Groq reasoning notice on {model_name}: {e}")
                    continue

    if stream_filter:
        stream_filter.flush()

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
        clean_final = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    if not clean_final.strip():
        fallback = "⚠️ Could not analyze screenshot. Please make sure the coding problem is clearly visible on your screen."
        on_token(fallback)
        return fallback

    return clean_final
