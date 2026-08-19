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
from typing import Dict, Any, Optional
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
    q = question.lower().strip()
    
    # If the spoken question already explicitly specifies a self-contained algorithmic concept, screen analysis is not required
    self_contained_indicators = [
        "dynamic programming", "binary search", "linked list", "sql query", "two sum",
        "merge sort", "quick sort", "tree traversal", "graph", "dijkstra", "hash map", "reverse a"
    ]
    if any(sci in q for sci in self_contained_indicators):
        return False
        
    visual_triggers = [
        r"\b(can you solve this|take a look at this|look at this)\b",
        r"\b(solve this\??|solve this problem\??|solve this challenge\??|solve this code\??)$",
        r"\b(what('s| is) wrong with this|why is this failing|fix this|find the bug)\b",
        r"\b(what does this (code )?output|what('s| is) the output)\b",
        r"\b(can you write the code|can you implement this|write code for this)\b",
        r"\b(how would you (get|find|calculate|query) this)\b",
        r"\b(can you modify this|how would you change this)\b",
        r"\b(look at (the|this) (screen|editor|problem|code|table|error))\b",
        r"^(solve this\??|fix this\??|what's wrong\??|output\??)$"
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
        "what's the complexity"
    ]
    if any(q.startswith(s) or s in q for s in coding_starters):
        return True
        
    return False


def capture_screen_image() -> str:
    """Takes a screenshot of the primary display, saving to a temp file."""
    try:
        screenshot = ImageGrab.grab()
    except Exception:
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
        except Exception as e:
            print(f"Screen capture notice: {e}")
            screenshot = Image.new("RGB", (800, 600), color=(0, 0, 0))
            
    temp_path = os.path.join(tempfile.gettempdir(), "parakit_interview_screen.png")
    screenshot.save(temp_path, format="PNG")
    return temp_path


def encode_image_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_visible_screen(
    client: Optional[Groq] = None,
    mock_screen_data: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    On-demand screen analysis for coding interview tasks.
    Extracts structured coding context: language, problem statement, code, errors.
    """
    global _SCREEN_CACHE
    now = time.time()
    
    # Return mock screen data if provided (for automated testing / environments without display)
    if mock_screen_data:
        _SCREEN_CACHE = {"data": mock_screen_data, "timestamp": now}
        return mock_screen_data
        
    # Check cache TTL
    if not force_refresh and _SCREEN_CACHE["data"] and (now - _SCREEN_CACHE["timestamp"] < CACHE_TTL_SECONDS):
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
    
    if not client:
        return empty_result
        
    image_path = ""
    try:
        image_path = capture_screen_image()
        b64_img = encode_image_base64(image_path)
        
        prompt = """You are a Real-Time Technical Interview Screen Analyzer.
Analyze the provided screenshot of the user's screen.
Determine if there is an active coding problem, code editor, SQL query, or error/traceback displayed.

OUTPUT STRICTLY VALID JSON ONLY:
{
  "is_coding_related": true,
  "language": "Python|Java|C++|JavaScript|TypeScript|SQL|Go|Rust|Unknown",
  "problem_statement": "Exact text of the visible problem statement or coding prompt",
  "code": "Visible code snippet or initial function template",
  "error_message": "Visible traceback, compilation error, or failed test if present, else empty string",
  "visible_question": "Short 1-sentence summary of the visible coding challenge",
  "question_type": "coding|debugging|sql|none",
  "confidence": 0.95
}
If no coding problem or code is visible, set is_coding_related: false, question_type: "none", and confidence: 0.0."""

        # Try vision model with fallback
        resp = None
        for v_model in ["qwen/qwen3.6-27b", "llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]:
            try:
                resp = client.chat.completions.create(
                    model=v_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                            ]
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
            refined = f"Write an optimized SQL query for: {spoken_q} (Schema/Context: {code or prob})"
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
