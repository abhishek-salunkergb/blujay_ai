"""
Screen Analyzer Service for Multimodal Coding Interview Detection
=================================================================
Combines spoken interviewer question with on-demand screen understanding to:
1. Detect visual coding problems (LeetCode, HackerRank, IDEs, terminal errors, SQL tables).
2. Structure screen context (problem statement, code, error message, language, confidence).
3. Fuse voice and screen signals for accurate question understanding and zero-click mode activation.
4. Strictly respects user privacy by avoiding continuous background screen captures.
"""

import os
import re
import time
import base64
import tempfile
import threading
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageGrab
from groq import Groq


# Cached screen context to prevent redundant captures
_SCREEN_CACHE: Dict[str, Any] = {
    "data": None,
    "timestamp": 0.0
}
CACHE_TTL_SECONDS = 20.0


def is_visual_reference_question(question: str) -> bool:
    """
    Checks whether a spoken question refers to something on-screen,
    indicating that screen analysis should be triggered.
    """
    if not question:
        return False
    q = question.lower().strip().rstrip(" .?!,")
    
    # If the spoken question already explicitly specifies a self-contained algorithmic concept, screen analysis is not required
    self_contained_indicators = [
        "dynamic programming", "binary search", "linked list", "sql query", "two sum",
        "merge sort", "quick sort", "tree traversal", "graph", "dijkstra", "hash map", "reverse a"
    ]
    if any(sci in q for sci in self_contained_indicators):
        return False
        
    visual_triggers = [
        r"\b(can you solve this|take a look at this|look at this)\b",
        r"\b(solve (this|it|this problem|this challenge|this code))\b",
        r"\b(implement (this|it|this problem|this challenge|the problem))\b",
        r"\b(code (this|it|this problem)|now code (this|it))\b",
        r"\b(solve (the )?(problem|question|challenge|code) (on|shown on|displayed on|given on|visible on) (the )?screen)\b",
        r"\b(implement (the )?(problem|question|challenge|code) (shown|on the screen|on screen|given on the screen|given on screen))\b",
        r"\b(explain (this|the) (problem|challenge|code|question|error|traceback))\b",
        r"\b(give (me )?(pseudo\s*code|the code) for this)\b",
        r"\b(optimize (the|this) (code|solution|problem) (on screen|on the screen|shown on screen))\b",
        r"\b(what('s| is) (the )?time complexity of this)\b",
        r"\b(what('s| is) wrong with this|why is this failing|fix this|find the bug)\b",
        r"\b(what does this (code )?output|what('s| is) the output|output of this)\b",
        r"\b(can you write the code|can you implement this|write code for this|write (the|a) solution|code the solution|write (it|this)( iteratively| recursively)?|now write (it|this)( iteratively| recursively)?)\b",
        r"\b(can you write a query for this|write a query for this|query for this|sql for this)\b",
        r"\b(how would you (get|find|calculate|query) this)\b",
        r"\b(can you modify this|how would you change this)\b",
        r"\b(look at (the|this) (screen|editor|problem|code|table|error))\b",
        r"^(solve this|solve it|implement this|implement it|code this|code it|write it|write this|fix this|what's wrong|output|write the solution)$"
    ]
    return any(re.search(p, q) for p in visual_triggers)


def is_spoken_coding_question(question: str) -> bool:
    """
    Checks if the spoken question is clearly a coding/algorithm/SQL/debugging problem
    from text alone without needing screen analysis.
    """
    if not question:
        return False
    q = question.lower().strip()
    
    coding_starters = [
        "write a function", "write a program", "write code", "write a sql query",
        "reverse a linked list", "two sum", "binary search", "find the second largest",
        "find the longest substring", "time complexity of", "space complexity of",
        "can you optimize it", "can we optimize this", "what is the time complexity",
        "what's the complexity", "give me the code", "give me code", "show me the code",
        "implement this", "now code it", "write the solution", "give me pseudocode",
        "optimize your solution"
    ]
    if any(q.startswith(s) or s in q for s in coding_starters):
        return True
        
    return False


def capture_screen_image(monitor_id: Optional[int] = None, bbox: Optional[Tuple[int, int, int, int]] = None) -> str:
    """Takes a screenshot of the preferred/active monitor or bounding box using vision.capture_screen_image with DPI awareness."""
    try:
        from vision import capture_screen_image as vision_capture
        return vision_capture(monitor_id=monitor_id, bbox=bbox)
    except Exception:
        pass

    try:
        if bbox:
            try:
                screenshot = ImageGrab.grab(bbox=bbox, all_screens=True)
            except Exception:
                screenshot = ImageGrab.grab(bbox=bbox)
        else:
            try:
                screenshot = ImageGrab.grab(all_screens=True)
            except Exception:
                screenshot = ImageGrab.grab()
    except Exception:
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
        except Exception as e:
            print(f"Screen capture notice: {e}")
            screenshot = Image.new("RGB", (800, 600), color=(0, 0, 0))
            
    import uuid
    temp_path = os.path.join(tempfile.gettempdir(), f"interview_screen_{os.getpid()}_{uuid.uuid4().hex[:8]}.png")
    screenshot.save(temp_path, format="PNG")
    return temp_path


def encode_image_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_visible_screen(
    client: Optional[Groq] = None,
    mock_screen_data: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
    session: Optional[Any] = None
) -> Dict[str, Any]:
    """
    On-demand screen analysis for coding interview tasks.
    Extracts structured coding context: language, problem statement, code, errors.
    """
    global _SCREEN_CACHE
    now = time.time()
    
    # Return mock screen data if provided (for automated testing / environments without display)
    if mock_screen_data:
        if session is not None and hasattr(session, "screen_context_cache"):
            session.screen_context_cache = {"data": mock_screen_data, "timestamp": now}
        else:
            _SCREEN_CACHE = {"data": mock_screen_data, "timestamp": now}
        return mock_screen_data
        
    # 1. If session is provided, strictly use session cache (prevent cross-session contamination)
    if session is not None and hasattr(session, "screen_context_cache"):
        s_cache = getattr(session, "screen_context_cache", None)
        if not force_refresh and s_cache and s_cache.get("data") and (now - s_cache.get("timestamp", 0) < CACHE_TTL_SECONDS):
            return s_cache["data"]
    elif session is None and not force_refresh and _SCREEN_CACHE.get("data") and (now - _SCREEN_CACHE.get("timestamp", 0) < CACHE_TTL_SECONDS):
        # 2. Standalone fallback (only when no session object exists)
        return _SCREEN_CACHE["data"]
        
    # Default fallback structured context
    empty_result: Dict[str, Any] = {
        "is_coding_related": False,
        "language": "Unknown",
        "problem_statement": "",
        "code": "",
        "error_message": "",
        "visible_question": "",
        "question_type": "general",
        "confidence": 0.0
    }
    
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not client and not openai_key:
        return empty_result
        
    image_path = ""
    try:
        mon_id = getattr(session, "selected_monitor_id", None) if session else None
        image_path = capture_screen_image(monitor_id=mon_id)
        
        # Extract on-screen text using hardware-accelerated Windows Native OCR
        extracted_text = ""
        try:
            from vision import extract_text_from_image
            extracted_text = extract_text_from_image(image_path)
        except Exception:
            pass

        prompt = f"""You are a Real-Time Technical Interview Screen Analyzer.
Analyze the following text and code extracted from the user's screen:

[EXTRACTED ON-SCREEN CONTENT]:
{extracted_text if extracted_text.strip() else "(Blank / minimal text visible on screen)"}

Determine if there is an active coding problem, code editor, SQL query, or error/traceback displayed.

OUTPUT STRICTLY VALID JSON ONLY:
{{
  "is_coding_related": true,
  "language": "Python|Java|C++|JavaScript|TypeScript|SQL|Go|Rust|Unknown",
  "problem_statement": "Exact text of the visible problem statement or coding prompt",
  "code": "Visible code snippet or initial function template",
  "error_message": "Visible traceback, compilation error, or failed test if present, else empty string",
  "visible_question": "Short 1-sentence summary of the visible coding challenge",
  "question_type": "coding|debugging|sql|none",
  "confidence": 0.95
}}
If no coding problem or code is visible, set is_coding_related: false, question_type: "none", and confidence: 0.0."""

        resp = None
        # Tier 1 (if OPENAI_API_KEY present): Direct OpenAI GPT-4o-mini structured JSON
        if openai_key:
            try:
                import importlib
                openai_mod = importlib.import_module("openai")
                o_client = openai_mod.OpenAI(api_key=openai_key)
                resp = o_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a JSON-only screen analysis parser. Always output valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
            except Exception as oe:
                print(f"OpenAI screen analysis notice: {oe}, trying Groq models...")

        # Tier 2: Groq models fallback
        if not resp and client:
            for v_model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                try:
                    resp = client.chat.completions.create(
                        model=v_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a JSON-only screen analysis parser. Always output valid JSON."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.0,
                        max_tokens=1000,
                        response_format={"type": "json_object"}
                    )
                    if resp:
                        break
                except Exception:
                    continue

        if not resp:
            return empty_result

        import json
        content_str = resp.choices[0].message.content or "{}"
        clean_json_str = re.sub(r"<think>.*?</think>", "", content_str, flags=re.DOTALL).strip()
        data = json.loads(clean_json_str)
        if session is not None and hasattr(session, "screen_context_cache"):
            session.screen_context_cache = {"data": data, "timestamp": now}
        else:
            _SCREEN_CACHE = {"data": data, "timestamp": now}
        return data
    except Exception as e:
        return empty_result
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass


def start_async_screen_refresh(
    client: Optional[Groq] = None,
    session: Optional[Any] = None
) -> None:
    """
    Refreshes screen context asynchronously on a background worker thread.
    Allows the voice speech processing loop to read screen context in 0ms without blocking.
    """
    def _async_worker():
        try:
            analyze_visible_screen(client=client, force_refresh=True, session=session)
        except Exception as e:
            print(f"[Async Screen Refresh] Notice: {e}")

    threading.Thread(target=_async_worker, daemon=True).start()


def fuse_voice_and_screen_question(
    spoken_question: str,
    screen_context: Optional[Dict[str, Any]] = None,
    session: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Combines spoken interviewer question with on-demand screen understanding.
    
    Rules:
    1. Spoken clear non-coding question (e.g. 'Tell me about yourself', 'Why did you choose this architecture?')
       has PRIORITY and is NOT overridden by visible code.
    2. Spoken visual reference (e.g. 'Can you solve this?', 'What's wrong with this?') is completed
       by the problem statement, code, or error extracted from the screen.
    3. Spoken coding question without visual references (e.g. 'Write a function to reverse a linked list')
       is classified as coding directly without requiring screen capture.
    """
    from classifier import classify_question
    
    spoken_q = (spoken_question or "").strip()
    spoken_lower = spoken_q.lower()
    
    # 1. Spoken Question Type Classification
    base_type = classify_question(spoken_q)
    
    # Check if spoken question is a clear behavioral, HR, resume, architecture, system design, or technical inquiry
    is_clear_non_coding = base_type in {"hr", "behavioral", "resume", "project", "system_design", "system_design_component", "technical", "database", "cloud", "ai_ml", "general"} and not is_visual_reference_question(spoken_q)
    if is_clear_non_coding:
        return {
            "refined_question": spoken_q,
            "question_type": base_type,
            "is_coding": False,
            "language": "English",
            "screen_used": False,
            "code_context": "",
            "error_context": ""
        }
        
    # Check if spoken question is already a complete coding inquiry
    if base_type in {"coding", "coding_complexity", "coding_optimization", "sql", "debugging"} and not is_visual_reference_question(spoken_q):
        return {
            "refined_question": spoken_q,
            "question_type": base_type,
            "is_coding": True,
            "language": "Python" if base_type != "sql" else "SQL",
            "screen_used": False,
            "code_context": "",
            "error_context": ""
        }
        
    # 2. Multimodal Fusion when Screen Context is Available
    is_coding_rel = bool(screen_context and (screen_context.get("is_coding_related") or screen_context.get("has_coding_problem")))
    if screen_context and is_coding_rel:
        lang = screen_context.get("language") or screen_context.get("detected_language", "Python")
        code = screen_context.get("code") or screen_context.get("code_snippet", "")
        error = screen_context.get("error_message", "")
        title = screen_context.get("problem_title", "").strip()
        desc = screen_context.get("problem_statement") or screen_context.get("problem_description") or screen_context.get("visible_question", "")
        if title and desc and title.lower() not in desc.lower():
            prob = f"{title}: {desc}"
        else:
            prob = desc or title
        screen_q_type = screen_context.get("question_type", "coding")
        
        # Case A: SQL Inquiry
        if screen_q_type == "sql" or "sql" in spoken_lower or "salary" in spoken_lower or "table" in spoken_lower or "SELECT" in code:
            ctx_parts = [p for p in [prob, code] if p]
            ctx_str = " | ".join(ctx_parts)
            refined = f"Write an optimized SQL query for: {spoken_q} (Schema/Context: {ctx_str})"
            return {
                "refined_question": refined,
                "question_type": "sql",
                "is_coding": True,
                "language": "SQL",
                "screen_used": True,
                "code_context": code,
                "error_context": error
            }
            
        # Case B: Debugging / Error Diagnosis
        if screen_q_type == "debugging" or any(w in spoken_lower for w in ["wrong", "bug", "failing", "error", "fix", "output"]):
            refined = f"Debug and explain the issue with this {lang} code:\n{code}\nError: {error or 'Logical flaw / unexpected output'}"
            return {
                "refined_question": refined,
                "question_type": "debugging",
                "is_coding": True,
                "language": lang,
                "screen_used": True,
                "code_context": code,
                "error_context": error
            }
            
        # Case C: Coding / DSA Problem
        if prob or code or is_visual_reference_question(spoken_q):
            target_prob = prob if prob else f"Problem displayed in {lang}: {code}"
            refined = f"Provide an optimal solution for the following coding problem in {lang}: {target_prob}"
            return {
                "refined_question": refined,
                "question_type": "coding",
                "is_coding": True,
                "language": lang,
                "screen_used": True,
                "code_context": code,
                "error_context": error
            }
            
    # Default fallback: use spoken classification
    return {
        "refined_question": spoken_q,
        "question_type": base_type,
        "is_coding": base_type in {"coding", "coding_complexity", "coding_optimization", "sql", "debugging"},
        "language": "Python" if base_type == "coding" else "English",
        "screen_used": False,
        "code_context": "",
        "error_context": ""
    }
