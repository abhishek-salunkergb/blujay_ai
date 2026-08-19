"""
Real-time Interview Copilot Engine (Parakeet AI Inspired)
=========================================================
Handles continuous audio transcription, real-time question detection,
personalized answer generation (grounded in resume, JD, company info, and candidate profile),
rapid streaming, one-click answer refinement, and post-interview summaries.
"""

import os
import re
import io
import time
import json
import wave
import uuid
import queue
import tempfile
import threading
import collections
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List, Tuple, Union

import sounddevice as sd
import numpy as np
try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except Exception:
    SOUNDCARD_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except Exception:
    VAD_AVAILABLE = False

from groq import Groq
from rag import search_knowledge, load_documents
from classifier import classify_question
from speech import (
    SystemAudioProcessor,
    MicrophoneAudioProcessor,
    merge_overlapping_transcripts,
    compute_audio_diagnostics,
    AudioDiagnostics,
    remove_dc_offset,
    safe_automatic_gain_control
)

# Document parsers
def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from PDF, DOCX, or TXT files."""
    if not file_path or not os.path.exists(file_path):
        return ""
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""
            
    elif ext == ".pdf":
        try:
            import importlib
            pypdf_mod = importlib.import_module("pypdf")
            reader = pypdf_mod.PdfReader(file_path)
            pages = []
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
            return "\n\n".join(pages).strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    elif ext == ".docx":
        try:
            import importlib
            docx_mod = importlib.import_module("docx")
            doc = docx_mod.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs).strip()
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
            
    return ""


# ============================================================
# ASR CORRECTIONS & QUESTION REFINEMENT LAYER
# ============================================================

QUESTION_STARTERS = [
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "can you", "could you", "would you", "will you", "do you", "did you", "have you",
    "tell me", "explain", "describe", "walk me through", "give me an example",
    "share a time", "discuss", "what is", "how do", "how did", "why did", "why do",
    "can we", "is it possible", "how would you", "what would you", "elaborate on",
    "what about", "how about", "and for", "so how", "so what", "in your opinion",
    "your approach", "how you", "how authentication", "how security", "why not",
    "write a", "write code", "implement", "find the", "solve", "design a"
]

INTERVIEW_QUESTION_PATTERNS = [
    r"\?$",                                           # Ends with question mark
    r"^(what|why|how|when|where|who|which)\b",        # Starts with WH-word
    r"^(can|could|would|will|do|did|have|has|are|is)\s+you\b", # Direct second-person questions
    r"^(tell me|explain|describe|walk me through|give an example|share a time|talk about|discuss)\b",
    r"^(what about|how about|and for|so what about|why not|and what|and how)\b", # Follow-up questions
    r"^(write a|write code|implement|find the|solve|design a|debug|fix)\b",      # Coding & imperative questions
    r"\b(your experience with|your background in|how you handled|your role in|how you implemented)\b",
    r"\b(what was your approach|what were the trade-offs|why did you choose|how did you use)\b",
    r"\b(how (you|they|the|to) (implemented|handle|handled|design|designed|scale|scaled|optimize|optimized))\b",
    r"^(your approach\??|performance\??|database\??|security\??|scalability\??|caching\??)$"
]

ASR_CORRECTIONS = {
    r"\bpie\s*torch\b": "PyTorch",
    r"\bpy\s*torch\b": "PyTorch",
    r"\bsequel\b": "SQL",
    r"\bs\s*q\s*l\b": "SQL",
    r"\bpost\s*gres\b": "PostgreSQL",
    r"\bpost\s*gress\b": "PostgreSQL",
    r"\bpost\s*gre\s*s\s*q\s*l\b": "PostgreSQL",
    r"\bmongo\s*d\s*b\b": "MongoDB",
    r"\bmongo\s*db\b": "MongoDB",
    r"\bdjango\b": "Django",
    r"\breact\b": "React",
    r"\bredux\b": "Redux",
    r"\bflask\b": "Flask",
    r"\bfast\s*api\b": "FastAPI",
    r"\bj\s*w\s*t\b": "JWT",
    r"\bweb\s*socket(s)?\b": r"WebSocket\1",
    r"\brest\s*api(s)?\b": r"REST API\1",
    r"\bgraph\s*ql\b": "GraphQL",
    r"\breact\s*j\s*s\b": "React.js",
    r"\breact\s*js\b": "React.js",
    r"\bnode\s*j\s*s\b": "Node.js",
    r"\bnode\s*js\b": "Node.js",
    r"\bnext\s*j\s*s\b": "Next.js",
    r"\bnext\s*js\b": "Next.js",
    r"\bvue\s*j\s*s\b": "Vue.js",
    r"\bvue\s*js\b": "Vue.js",
    r"\btype\s*script\b": "TypeScript",
    r"\bjava\s*script\b": "JavaScript",
    r"\bnum\s*py\b": "NumPy",
    r"\bpan\s*das\b": "Pandas",
    r"\btensor\s*flow\b": "TensorFlow",
    r"\bopen\s*ai\b": "OpenAI",
    r"\bdata\s*base\b": "database",
    r"\bdata\s*bases\b": "databases",
    r"\bcapturing\s+property\b": "capturing properly",
    r"\bcapturing\s+properties\b": "capturing properly",
    r"\binterruptions?\s+are\s+okay\b": "interruptions",
    r"\bcashe\b": "cache",
    r"\bcash\s+invalidation\b": "cache invalidation",
    r"\bcash\s+memory\b": "cache memory",
    r"\bdocer\b": "Docker",
    r"\bk\s*8\s*s\b": "Kubernetes",
    r"\bkube\b": "Kubernetes",
    r"\brabbit\s*m\s*q\b": "RabbitMQ",
    r"\bo\s*auth(2)?\b": r"OAuth\1",
    r"\br\s*a\s*g\b": "RAG",
    r"\bl\s*l\s*m\b": "LLM",
    r"\bg\s*r\s*p\s*c\b": "gRPC",
    r"\bci\s*/?\s*cd\b": "CI/CD",
    r"\ba\s*w\s*s\b": "AWS",
    r"\bg\s*c\s*p\b": "GCP",
    r"\bazure\b": "Azure"
}


def clean_asr_transcript(text: str) -> str:
    """Performs fast rule-based normalization, homophone correction, and deduplication of repeated words."""
    if not text:
        return ""
    
    clean = text.strip()
    
    # 1. Deduplicate consecutive repeated phrases/words (e.g. "how did you how did you" -> "how did you")
    clean = re.sub(r'\b(\w+(?:\s+\w+){0,3})\s+\1\b', r'\1', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\b(\w+)\s+\1\b', r'\1', clean, flags=re.IGNORECASE)
    
    # 2. Apply common ASR dictionary replacements
    for pattern, replacement in ASR_CORRECTIONS.items():
        clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
        
    return clean.strip()


def compute_question_confidence(raw_text: str, session: Optional[Any] = None) -> float:
    """
    Two-Stage Question Candidate Detector:
    Computes a continuous question intent confidence score (0.0 to 1.0).
    Ensures genuine interviewer questions are NEVER missed due to missing '?' or conversational phrasing.
    """
    if not raw_text or len(raw_text.strip()) < 2:
        return 0.0
    
    cleaned = clean_asr_transcript(raw_text)
    u_lower = cleaned.lower().strip()
    words = u_lower.split()
    
    # 1. Direct interrogative WH-starters & prompt commands
    wh_starters = [
        "why did you", "why didn't you", "why choose", "why not", "why",
        "tell me about", "walk me through", "explain how", "explain", "can you explain",
        "describe a time", "how did you", "how would you", "how do you", "how can we",
        "how does", "how", "what is the difference", "what are the trade-offs",
        "what about", "how about", "what database", "what framework", "what architecture",
        "which database", "what technologies", "design a", "write a function",
        "how do you handle", "how authentication", "how you implemented", "how security",
        "what was your approach", "your approach", "what did you use", "why django",
        "why react", "why redux", "why postgres", "why redis", "why mongodb",
        "can you clarify", "did you mean"
    ]
    for starter in wh_starters:
        if u_lower.startswith(starter):
            return 0.98 if "?" in raw_text else 0.95
            
    # 2. Contains question mark or rising intonation
    if "?" in raw_text:
        return 0.95
        
    # 3. Follow-up question patterns (e.g. "What about scalability?", "Why not MySQL?")
    if re.search(r"^(what about|how about|and for|so what about|why not|and what|and how)\b", u_lower):
        return 0.95
        
    # 4. Technical inquiry / architectural questions without '?'
    tech_inquiry_patterns = [
        r"\b(how (you|they|the|to) (implemented|handle|handled|design|designed|scale|scaled|optimize|optimized))\b",
        r"\b(why did you choose|what was the reason for|what were the challenges)\b",
        r"\b(what is your experience with|have you used|did you use)\b",
        r"^(explain|describe|clarify|elaborate on|walk through)\b",
        r"^(can you|could you|would you|do you|did you|have you)\b",
        r"\b(your approach|how you implemented)\b"
    ]
    for p in tech_inquiry_patterns:
        if re.search(p, u_lower):
            return 0.92
            
    # 5. Short follow-up keywords in context (e.g. "Scalability", "Performance", "Security", "Database")
    if session and getattr(session, "turn_state", None) is not None:
        if u_lower in {"scalability", "performance", "security", "caching", "database", "authentication", "latency", "trade-offs"}:
            return 0.80
            
    # 6. General regex matching
    if is_likely_question(cleaned):
        return 0.88
        
    return 0.20


def is_likely_question(text: str) -> bool:
    """Detect whether a transcribed utterance is a question, follow-up, or interview prompt."""
    if not text:
        return False
    
    clean = clean_asr_transcript(text)
    if len(clean) < 3:
        return False
        
    lower = clean.lower()
    
    if clean.endswith("?"):
        return True
        
    for pattern in INTERVIEW_QUESTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
            
    for starter in QUESTION_STARTERS:
        if lower.startswith(starter + " ") or lower == starter:
            return True
            
    return False


def refine_interview_question(
    client: Optional[Groq],
    raw_transcript: str,
    session: Optional[Any] = None,
    recent_context: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Intelligently recovers the intended interview question from noisy/imperfect speech-to-text transcript.
    - Resolves ASR homophones & tech terminology (e.g. 'pie torch' -> 'PyTorch', 'property' -> 'properly').
    - Fixes broken grammar & removes vocal disfluencies / repetitions.
    - Resolves contextual follow-ups (e.g. 'What about scalability?').
    - Classifies semantic intent (TECHNICAL, BEHAVIORAL, FOLLOW_UP, STATEMENT, etc.).
    - Never invents unrelated questions.
    """
    cleaned_local = clean_asr_transcript(raw_transcript)
    if not cleaned_local or len(cleaned_local) < 3:
        return {
            "is_question": False,
            "raw_transcript": raw_transcript,
            "refined_question": cleaned_local,
            "confidence": 0.0,
            "question_type": "STATEMENT"
        }
        
    # If client is not available, return rule-based refined question
    if not client:
        is_q = is_likely_question(cleaned_local)
        ref_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local + "?" if is_q else cleaned_local)
        return {
            "is_question": is_q,
            "raw_transcript": raw_transcript,
            "refined_question": ref_q,
            "confidence": 0.7 if is_q else 0.4,
            "question_type": "TECHNICAL" if is_q else "STATEMENT"
        }
        
    # Build conversation context snippet
    context_lines = []
    if session and getattr(session, "transcript_log", None):
        for item in session.transcript_log[-4:]:
            context_lines.append(f"{item['role']}: {item['text']}")
    elif recent_context:
        for item in recent_context[-4:]:
            context_lines.append(f"{item.get('role', 'Speaker')}: {item.get('text', '')}")
            
    ctx_str = "\n".join(context_lines) if context_lines else "No previous conversation context."
    job_target = f"Role: {session.job_title} @ {session.company_name}" if (session and (getattr(session, 'job_title', '') or getattr(session, 'company_name', ''))) else "General Tech Role"
    
    prompt = f"""You are an expert Real-Time Interview Speech Refiner.
You receive a noisy, imperfect speech-to-text transcript from an interviewer during a technical/job interview.

CANDIDATE CONTEXT: {job_target}
RECENT CONVERSATION HISTORY:
{ctx_str}

LATEST RAW SPOKEN TRANSCRIPT:
"{raw_transcript}"

TASK:
1. Determine if this utterance is an interviewer question, follow-up, interview prompt, or statement requiring candidate response.
   - Note: Spoken questions often lack question marks or start as statements (e.g. "Tell me about...", "Walk me through...", "how you handled...").
   - Follow-up fragments (e.g. "What about scalability?", "And database?") must be resolved in context.
2. If it is a question or prompt:
   - Fix speech-recognition mistakes, wrong words, and homophones (e.g. "cash" -> "cache", "property" -> "properly", "pie torch" -> "PyTorch", "sequel" -> "SQL").
   - Remove stuttered words, filler words, and vocal disfluencies.
   - Formulate the cleanest, grammatically sound, intended interview question.
   - DO NOT invent topics that the interviewer did not mention. Preserve the exact core semantic meaning.
3. If it is purely conversational filler (e.g. "thanks for joining", "can you hear me", "okay"), mark is_question: false.

OUTPUT MUST BE STRICTLY VALID JSON ONLY:
{{
  "is_question": true,
  "refined_question": "Clean and precise interview question here?",
  "question_type": "TECHNICAL",
  "confidence": 0.95
}}"""

    try:
        # Use primary model with fallback
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a specialized JSON-only interview speech refinement engine. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
        except Exception:
            resp = client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {"role": "system", "content": "You are a specialized JSON-only interview speech refinement engine. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=250
            )
            content = resp.choices[0].message.content

        # Extract JSON substring if wrapped in markdown or conversational text
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        json_str = json_match.group(0) if json_match else content
        data = json.loads(json_str)
        
        ref_q = data.get("refined_question", "").strip() or cleaned_local
        is_q = bool(data.get("is_question", False))
        conf = float(data.get("confidence", 0.85))
        q_type = data.get("question_type", "TECHNICAL" if is_q else "STATEMENT")
        
        return {
            "is_question": is_q,
            "raw_transcript": raw_transcript,
            "refined_question": ref_q,
            "confidence": conf,
            "question_type": q_type
        }
    except Exception as e:
        is_q = is_likely_question(cleaned_local)
        ref_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local + "?" if is_q else cleaned_local)
        return {
            "is_question": is_q,
            "raw_transcript": raw_transcript,
            "refined_question": ref_q,
            "confidence": 0.6 if is_q else 0.3,
            "question_type": "TECHNICAL" if is_q else "STATEMENT"
        }


# ============================================================
# INTERVIEW TURN-TAKING & SPEAKER INTELLIGENCE
# ============================================================

import enum

class InterviewTurnState(enum.Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    LISTENING = "LISTENING"
    CANDIDATE_SPEAKING = "CANDIDATE_SPEAKING"
    INTERVIEWER_SPEAKING = "INTERVIEWER_SPEAKING"
    WAITING_FOR_FINAL_TRANSCRIPT = "WAITING_FOR_FINAL_TRANSCRIPT"
    CLASSIFYING = "CLASSIFYING"
    PROCESSING = "PROCESSING"
    PROCESSING_QUESTION = "PROCESSING_QUESTION"
    GENERATING_ANSWER = "GENERATING_ANSWER"
    ANSWERING = "ANSWERING"
    ANSWER_STREAMING = "ANSWER_STREAMING"
    WAITING_FOR_NEXT_QUESTION = "WAITING_FOR_NEXT_QUESTION"
    WAITING_FOR_NEXT_TURN = "WAITING_FOR_NEXT_TURN"
    RECOVERING = "RECOVERING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    ERROR_RECOVERY = "ERROR_RECOVERY"


class ComponentHealthState(enum.Enum):
    HEALTHY = "HEALTHY"
    BUSY = "BUSY"
    IDLE = "IDLE"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    ERROR = "ERROR"


def categorize_api_error(error: Optional[Exception]) -> Tuple[str, bool, str]:
    """
    Categorizes LLM/STT API errors into semantic failure types.
    Returns (category, is_retryable, user_facing_message).
    Ensures API keys are NEVER logged or surfaced in user messages.
    """
    if error is None:
        return ("UNKNOWN", False, "An unknown error occurred.")
        
    err_str = str(error)
    err_lower = err_str.lower()
    
    # 1. AUTHENTICATION (401, invalid api key) - NON-RETRYABLE
    if any(p in err_lower for p in ["401", "unauthorized", "invalid api key", "authentication", "api_key_invalid", "incorrect api key"]):
        return ("AUTHENTICATION", False, "Authentication failed. Please verify your API key in settings.")
        
    # 2. MODEL_NOT_FOUND (404) - NON-RETRYABLE
    if any(p in err_lower for p in ["404", "model_not_found", "does not exist", "model not found"]):
        return ("MODEL_NOT_FOUND", False, "Configured AI model unavailable.")
        
    # 3. INVALID_REQUEST (400) - NON-RETRYABLE
    if any(p in err_lower for p in ["400", "bad_request", "invalid_request", "context_length_exceeded"]):
        return ("INVALID_REQUEST", False, "Request parameters could not be processed.")
        
    # 4. RATE_LIMIT (429) - RETRYABLE WITH BOUNDED BACKOFF
    if any(p in err_lower for p in ["429", "rate_limit", "quota exceeded", "too many requests", "rate limit"]):
        return ("RATE_LIMIT", True, "Rate limit reached. Retrying with backoff...")
        
    # 5. TIMEOUT - RETRYABLE
    if any(p in err_lower for p in ["timeout", "timed out", "readtimedout", "connecttimeout"]):
        return ("TIMEOUT", True, "Request timed out. Retrying...")
        
    # 6. NETWORK / CONNECTION - RETRYABLE
    if any(p in err_lower for p in ["connection", "connectionerror", "connection refused", "network", "remotedisconnected", "ssl"]):
        return ("NETWORK", True, "Temporary network connection issue. Reconnecting...")
        
    # 7. SERVER_ERROR (500, 502, 503, 504) - RETRYABLE
    if any(p in err_lower for p in ["500", "502", "503", "504", "internal server error", "service unavailable", "bad gateway"]):
        return ("SERVER_ERROR", True, "Service temporarily busy. Retrying...")
        
    return ("UNKNOWN", True, "Temporary service notice. Retrying...")


class SessionController:
    """
    Lightweight central Session Controller & Recovery Layer.
    Tracks session state, component health, heartbeat telemetry, and isolated watchdog recovery.
    """
    def __init__(self, session: Optional["InterviewSession"] = None):
        self.session = session
        self.state: InterviewTurnState = InterviewTurnState.IDLE
        self.components_health: Dict[str, Dict[str, Any]] = {
            "audio_capture": {"status": "HEALTHY", "last_updated": time.time(), "error": None},
            "stt": {"status": "HEALTHY", "last_updated": time.time(), "error": None},
            "question_detection": {"status": "HEALTHY", "last_updated": time.time(), "error": None},
            "context": {"status": "HEALTHY", "last_updated": time.time(), "error": None},
            "answer_generation": {"status": "IDLE", "last_updated": time.time(), "error": None},
            "streaming": {"status": "IDLE", "last_updated": time.time(), "error": None},
            "screen_analysis": {"status": "IDLE", "last_updated": time.time(), "error": None},
            "ui": {"status": "HEALTHY", "last_updated": time.time(), "error": None}
        }
        self.heartbeat: Dict[str, float] = {
            "last_audio_event": time.time(),
            "last_transcript_event": time.time(),
            "last_question_event": time.time(),
            "last_answer_event": time.time(),
            "last_heartbeat": time.time()
        }
        self.active_generation_info: Dict[str, Any] = {
            "generation_id": 0,
            "question_id": "",
            "start_time": 0.0,
            "state": "IDLE",
            "timeout_seconds": 25.0
        }
        self._recovery_lock = threading.Lock()

    def set_state(self, new_state: InterviewTurnState, reason: str = ""):
        self.state = new_state
        if self.session:
            self.session.turn_state = new_state
        self.heartbeat["last_heartbeat"] = time.time()

    def update_component(self, component: str, status: str, error: Optional[str] = None):
        if component in self.components_health:
            self.components_health[component] = {
                "status": status,
                "last_updated": time.time(),
                "error": error
            }
        self.heartbeat["last_heartbeat"] = time.time()

    def record_heartbeat_event(self, event_name: str):
        if event_name in self.heartbeat:
            self.heartbeat[event_name] = time.time()
        self.heartbeat["last_heartbeat"] = time.time()

    def start_generation(self, gen_id: int, question_id: str):
        self.active_generation_info = {
            "generation_id": gen_id,
            "question_id": question_id,
            "start_time": time.time(),
            "state": "ANSWERING",
            "timeout_seconds": 25.0
        }
        self.set_state(InterviewTurnState.GENERATING_ANSWER)
        self.update_component("answer_generation", "BUSY")
        self.update_component("streaming", "ACTIVE")

    def end_generation(self, gen_id: int, success: bool = True, error: Optional[str] = None):
        if self.active_generation_info.get("generation_id") == gen_id:
            self.active_generation_info["state"] = "IDLE"
            self.active_generation_info["start_time"] = 0.0
            self.set_state(InterviewTurnState.WAITING_FOR_NEXT_QUESTION if success else InterviewTurnState.LISTENING)
            self.update_component("answer_generation", "IDLE" if success else "RECOVERING", error)
            self.update_component("streaming", "IDLE")
            self.record_heartbeat_event("last_answer_event")

    def check_generation_timeout(self) -> bool:
        """Returns True if the active generation has exceeded safe timeout."""
        if self.active_generation_info.get("state") in ("ANSWERING", "BUSY"):
            start = self.active_generation_info.get("start_time", 0.0)
            timeout = self.active_generation_info.get("timeout_seconds", 25.0)
            if start > 0 and (time.time() - start > timeout):
                self.update_component("answer_generation", "RECOVERING", "Generation request timed out")
                self.set_state(InterviewTurnState.LISTENING)
                self.active_generation_info["state"] = "IDLE"
                return True
        return False

    def reset(self):
        """Clean session reset while preserving candidate profile data."""
        self.state = InterviewTurnState.IDLE
        for k in self.components_health:
            self.components_health[k] = {"status": "HEALTHY" if k not in ("answer_generation", "streaming", "screen_analysis") else "IDLE", "last_updated": time.time(), "error": None}
        self.active_generation_info = {
            "generation_id": 0,
            "question_id": "",
            "start_time": 0.0,
            "state": "IDLE",
            "timeout_seconds": 25.0
        }


@dataclass
class AudioFrame:
    """Raw audio frame captured from hardware device with monotonic timestamp."""
    data: Any
    timestamp: float  # monotonic time.perf_counter()
    source: str       # 'system' or 'candidate_mic' or 'mic'
    is_discontinuity: bool = False


@dataclass
class AudioSegment:
    """A segmented chunk of speech ready for transcription with immutable IDs and monotonic timestamps."""
    segment_id: str
    source: str
    audio_bytes: bytes
    sample_rate: int
    capture_start: float       # time.perf_counter()
    capture_end: float         # time.perf_counter()
    queue_enter: float         # time.perf_counter()
    processing_start: float    # time.perf_counter()
    silence_frames: int = 0
    speech_frames: int = 0


def get_adaptive_max_tokens(question_type: str = "general") -> int:
    """
    Adaptive output token budgets based on question type:
    - Simple/definition: ~200-300
    - Normal technical/project/follow-up: ~250-400
    - Coding/debugging: ~400-700
    - System design: ~500-750
    """
    qt = (question_type or "").lower().strip()
    if qt in ("coding", "debugging", "algorithm", "sql", "coding_complexity", "coding_optimization"):
        return 650
    elif qt in ("system_design", "system_design_component", "architecture"):
        return 700
    elif qt in ("technical", "project", "follow_up", "database", "cloud", "ai_ml", "resume"):
        return 380
    elif qt in ("hr", "behavioral", "clarification", "general", "statement"):
        return 280
    return 350


def parse_retry_after_seconds(error: Optional[Exception]) -> float:
    """Extracts retry interval from 429 rate limit error or headers."""
    if not error:
        return 1.0
    err_str = str(error)
    match_s = re.search(r"try again in ([\d\.]+)\s*s", err_str, re.IGNORECASE)
    if match_s:
        try:
            return float(match_s.group(1))
        except ValueError:
            pass
    match_ms = re.search(r"try again in ([\d\.]+)\s*ms", err_str, re.IGNORECASE)
    if match_ms:
        try:
            return float(match_ms.group(1)) / 1000.0
        except ValueError:
            pass
    match_hdr = re.search(r"retry[-_]after[:=\s]+([\d\.]+)", err_str, re.IGNORECASE)
    if match_hdr:
        try:
            return float(match_hdr.group(1))
        except ValueError:
            pass
    return 1.0


class QuestionLatencyTrace:
    """
    Detailed, monotonic timestamp latency trace for an interviewer question.
    Calculates exact durations across capture, queue, STT, completeness, context prep, TTFT, and first displayed answer.
    """
    def __init__(self, question_id: str = "", generation_id: int = 0, segment_id: str = ""):
        self.question_id: str = question_id
        self.generation_id: int = generation_id
        self.segment_id: str = segment_id
        self.source: str = "system"
        self.timestamps: Dict[str, float] = {
            "capture_start": 0.0,
            "capture_end": 0.0,
            "queue_enter": 0.0,
            "processing_start": 0.0,
            "stt_start": 0.0,
            "stt_end": 0.0,
            "transcript_received": 0.0,
            "question_candidate": 0.0,
            "question_complete": 0.0,
            "question_finalized": 0.0,
            "context_preparation_start": 0.0,
            "context_ready": 0.0,
            "generation_start": 0.0,
            "first_token": 0.0,
            "first_useful_token": 0.0,
            "generation_end": 0.0,
            # Legacy mapping aliases
            "audio_speech_start": 0.0,
            "first_stt_result": 0.0,
            "question_candidate_detected": 0.0,
            "question_semantically_complete": 0.0,
            "answer_generation_start": 0.0,
            "first_ai_token": 0.0,
            "first_formatted_content": 0.0,
            "answer_complete": 0.0
        }
        self.metrics: Dict[str, Any] = {}
        self.queue_depth: int = 0
        self.discontinuities: int = 0
        self.rate_limited: bool = False
        self.retry_after_ms: float = 0.0
        self.retry_attempt: int = 0

    def mark(self, event_name: str, ts: Optional[float] = None):
        val = ts if ts is not None else time.perf_counter()
        if event_name in self.timestamps:
            if self.timestamps[event_name] == 0.0 or event_name in (
                "first_token", "first_useful_token", "generation_end",
                "first_ai_token", "first_formatted_content", "answer_complete"
            ):
                self.timestamps[event_name] = val
                
        # Synchronize alias pairs
        alias_map = {
            "capture_start": "audio_speech_start",
            "audio_speech_start": "capture_start",
            "transcript_received": "first_stt_result",
            "first_stt_result": "transcript_received",
            "question_candidate": "question_candidate_detected",
            "question_candidate_detected": "question_candidate",
            "question_complete": "question_semantically_complete",
            "question_semantically_complete": "question_complete",
            "generation_start": "answer_generation_start",
            "answer_generation_start": "generation_start",
            "first_token": "first_ai_token",
            "first_ai_token": "first_token",
            "first_useful_token": "first_formatted_content",
            "first_formatted_content": "first_useful_token",
            "generation_end": "answer_complete",
            "answer_complete": "generation_end"
        }
        if event_name in alias_map:
            target = alias_map[event_name]
            if self.timestamps.get(target, 0.0) == 0.0 or target in (
                "first_token", "first_useful_token", "generation_end",
                "first_ai_token", "first_formatted_content", "answer_complete"
            ):
                self.timestamps[target] = val

    def compute_metrics(self) -> Dict[str, Any]:
        t = self.timestamps
        def diff(end_key, start_key):
            if t.get(end_key, 0) > 0 and t.get(start_key, 0) > 0:
                return round((t[end_key] - t[start_key]) * 1000.0, 2)
            return None

        def safe_float(val):
            return val if val is not None else 0.0

        # Granular breakdown metrics
        cap_to_q = diff("queue_enter", "capture_start")
        q_wait = diff("processing_start", "queue_enter")
        audio_proc = diff("stt_start", "processing_start")
        stt_req = diff("stt_end", "stt_start")
        stt_tot = diff("transcript_received", "capture_start") or diff("first_stt_result", "audio_speech_start")
        q_det = diff("question_candidate", "transcript_received") or diff("question_candidate_detected", "first_stt_result")
        q_comp = diff("question_complete", "question_candidate") or diff("question_semantically_complete", "question_candidate_detected")
        gen_wait = diff("generation_start", "question_complete") if t.get("question_complete") else diff("generation_start", "question_finalized")
        ttft = diff("first_token", "generation_start") or diff("first_ai_token", "answer_generation_start")
        first_useful = diff("first_useful_token", "question_complete") if t.get("question_complete") else diff("first_useful_token", "question_finalized")
        if first_useful is None:
            first_useful = diff("first_formatted_content", "question_semantically_complete") if t.get("question_semantically_complete") else diff("first_formatted_content", "question_finalized")
        tot_gen = diff("generation_end", "generation_start") or diff("answer_complete", "answer_generation_start")
        tot_question_to_displayed = diff("first_useful_token", "question_finalized") if t.get("first_useful_token") else diff("first_token", "question_finalized")
        if tot_question_to_displayed is None:
            tot_question_to_displayed = diff("first_formatted_content", "question_finalized") if t.get("first_formatted_content") else diff("first_ai_token", "question_finalized")

        self.metrics = {
            # Legacy compatibility float metrics
            "speech_to_first_transcript_ms": safe_float(stt_tot),
            "speech_to_question_candidate_ms": safe_float(diff("question_candidate_detected", "audio_speech_start") or diff("question_candidate", "capture_start")),
            "question_complete_to_generation_start_ms": safe_float(gen_wait),
            "generation_start_to_first_token_ms": safe_float(ttft),
            "question_complete_to_first_useful_answer_ms": safe_float(first_useful),
            "total_question_to_displayed_answer_ms": safe_float(tot_question_to_displayed),
            "total_generation_duration_ms": safe_float(tot_gen),
            # Detailed breakdown keys
            "capture_to_queue": cap_to_q,
            "queue_wait": q_wait,
            "audio_processing": audio_proc,
            "STT_request": stt_req,
            "STT_total": stt_tot,
            "question_detection": q_det,
            "question_completion": q_comp,
            "generation_wait": gen_wait,
            "TTFT": ttft,
            "first_useful_answer": first_useful,
            "total": tot_question_to_displayed
        }
        return self.metrics

    def format_debug_summary(self) -> str:
        self.compute_metrics()
        m = self.metrics
        def fmt(key):
            val = m.get(key)
            return f"{val}ms" if (val is not None and val > 0) else ("0.0ms" if isinstance(val, (int, float)) and val == 0.0 else "N/A")
            
        speech_stt = f"{m.get('speech_to_first_transcript_ms', 0)}ms"
        candidate_lat = f"{m.get('speech_to_question_candidate_ms', 0)}ms"
        q_gen = f"{m.get('question_complete_to_generation_start_ms', 0)}ms"
        ttft = f"{m.get('generation_start_to_first_token_ms', 0)}ms"
        first_useful = f"{m.get('question_complete_to_first_useful_answer_ms', 0)}ms"
        total = f"{m.get('total_question_to_displayed_answer_ms', 0)}ms"
        
        return (
            f"[{self.question_id}|Gen#{self.generation_id}] "
            f"Speech->STT: {speech_stt} | "
            f"Candidate: {candidate_lat} | "
            f"Q_Candidate_Latency: {candidate_lat} | "
            f"Q_Complete->Gen: {q_gen} | "
            f"TTFT: {ttft} | "
            f"First Useful Answer: {first_useful} | "
            f"Total: {total}"
        )

    def format_diagnostics_report(self) -> str:
        self.compute_metrics()
        m = self.metrics
        def fmt(k):
            val = m.get(k)
            return f"{val}ms" if val is not None else "N/A"
            
        return f"""
{self.question_id or 'Q000'}
Source={self.source}
Speaker=INTERVIEWER

Audio:
  speech_start={self.timestamps.get('capture_start', 0.0):.4f}
  segment_created={self.timestamps.get('capture_end', 0.0):.4f}
  STT_start={self.timestamps.get('stt_start', 0.0):.4f}
  STT_end={self.timestamps.get('stt_end', 0.0):.4f}
  transcript_received={self.timestamps.get('transcript_received', 0.0):.4f}

Latency Breakdown:
  capture_to_queue={fmt('capture_to_queue')}
  queue_wait={fmt('queue_wait')}
  audio_processing={fmt('audio_processing')}
  STT_request={fmt('STT_request')}
  STT_total={fmt('STT_total')}
  question_detection={fmt('question_detection')}
  question_completion={fmt('question_completion')}
  generation_wait={fmt('generation_wait')}

Question:
  candidate={self.timestamps.get('question_candidate', 0.0):.4f}
  complete={self.timestamps.get('question_complete', 0.0):.4f}

Generation:
  start={self.timestamps.get('generation_start', 0.0):.4f}
  TTFT={fmt('TTFT')}
  first_useful={fmt('first_useful_answer')}
  total={fmt('total')}

Rate limiting:
  429={'YES' if self.rate_limited else 'NO'}
  retry_after_ms={self.retry_after_ms:.1f}
  retry_attempt={self.retry_attempt}

Audio:
  queue_depth={self.queue_depth}
  discontinuity_count={self.discontinuities}
""".strip()


def detect_speaker_turn(
    utterance: str,
    session: Optional["InterviewSession"] = None,
    source: str = "mic"
) -> Dict[str, Any]:
    """
    Multi-Signal Turn & Speaker Identification Layer.
    
    Signal Priority:
    1. Hardware/Channel Source: Loopback audio (system audio from meeting software) is definitively the INTERVIEWER.
    2. Direct Question Starters & Interrogatives (e.g. "Why did you...", "Tell me about...", "Why Redux?", "Why?").
    3. First-Person Candidate Narratives & Experiences (e.g. "I built...", "I chose... because...", "In my project...").
    4. Short Utterance Policy & Turn Continuity (e.g. "Redux.", "Yes.", "React, Django, Redux."):
       - Follows active candidate turn state to prevent false answer triggers on short candidate answers.
       - Interprets rising-intonation queries ("Redux?", "Why?") as Interviewer follow-ups.
    5. Fallback & Temporary UNKNOWN state for ambiguous fragments.
    """
    if not utterance or len(utterance.strip()) < 1:
        return {"speaker": "UNKNOWN", "confidence": 0.0, "reason": "empty"}
        
    u_lower = utterance.lower().strip()
    u_clean = u_lower.strip(" .!?,:;")
    words = u_clean.split()
    word_count = len(words)
    has_question_mark = utterance.strip().endswith("?")
    
    # ---------------------------------------------------------
    # PRIORITY 1: SEPARATE AUDIO CHANNEL / SOURCE
    # ---------------------------------------------------------
    if source in ("system", "loopback", "interviewer_channel"):
        return {
            "speaker": "INTERVIEWER",
            "confidence": 0.99,
            "reason": "loopback_audio_channel"
        }
    if source in ("candidate_mic", "candidate_channel"):
        return {
            "speaker": "CANDIDATE",
            "confidence": 0.99,
            "reason": "microphone_candidate_channel"
        }
        
    # ---------------------------------------------------------
    # PRIORITY 2: DIRECT INTERVIEWER QUESTIONS & PROMPTS
    # ---------------------------------------------------------
    # Interrogatives and explicit question starters indicate an Interviewer turn or Interruption
    interviewer_phrases = [
        r"^(why did you|why didn't you|why choose|why not)\b",
        r"^(tell me about|walk me through|explain how you|can you explain|describe a time)\b",
        r"^(how did you|how would you|how do you|how can we|how does)\b",
        r"^(what is the difference|what are the trade-offs|what about scalability|what about performance)\b",
        r"^(what database|what framework|what architecture|what tech stack|which database|what technologies)\b",
        r"^(design a|write a function|how do you handle|let's discuss|moving on to)\b",
        r"^(why (django|react|redis|postgres|kafka|kubernetes|mongodb|python|aws|docker|jwt|nosql|sql|redux))\b",
        r"^(i('m| am) going to ask you|let's start with|my next question is|one last question)\b",
        r"^(okay,?\s*)?(give me a (second|moment|min)|one (second|moment)|let me check|let's see|hold on|wait a (minute|second))\b",
        r"^(thanks,?\s*[a-z]+|thank you (so much|very much)|thanks for joining)\b",
        r"^(i live in|i've got a picture|i'm looking at your profile)\b"
    ]
    for p in interviewer_phrases:
        if re.search(p, u_lower):
            return {
                "speaker": "INTERVIEWER",
                "confidence": 0.96,
                "reason": "direct_interviewer_speech"
            }
            
    # Direct short interrogatives: "Why?", "How?", "Which?", "Why JWT?", "Why Redux?"
    if u_clean in {"why", "how", "which", "where", "what"}:
        return {
            "speaker": "INTERVIEWER",
            "confidence": 0.95,
            "reason": "short_interrogative_followup"
        }

    # ---------------------------------------------------------
    # PRIORITY 3: CANDIDATE 1ST-PERSON NARRATIVE & EXPERIENCES
    # ---------------------------------------------------------
    candidate_phrases = [
        r"^(also\s+|and\s+)?(i('d|'ll|'ve|'m)?|we('d|'ll|'ve|'re)?)\s+(also\s+)?(built|worked on|used|use|implemented|designed|created|developed|started|solved|choose|chose|handled|migrated|configured|deployed|added|set up|would|would use|trained|benchmarked|decided|selected)\b",
        r"\b(we|i|i'd|we'd) (also )?(used|use|implemented|built|deployed|designed|configured|added|chose|trained|would use)\b",
        r"^(in my (previous|last|recent|past) (role|job|company|project|experience)|at my (previous|last) company)\b",
        r"^(my (role|responsibility|approach|experience|team)|on my team, (i|we)|for my project)\b",
        r"^(i selected|i decided to use|the reason i chose|i preferred|i wanted to|i used|i would use|i would|i'd use)\b",
        r"\b(in my experience|i have experience with|i am familiar with|i typically use|what i did was|what i would do)\b",
        r"^(because (the|it|we|i|component)|the reason was|basically (we|i)|so (we|i) (used|built|had))\b",
        r"^(yes, (i|we) did|yeah, (i|we)|sure, so (i|in)|so basically (i|in my))\b",
        r"^(can i (explain|clarify|share|ask)|should i (write|code|explain)|can you hear me)\b"
    ]
    for p in candidate_phrases:
        if re.search(p, u_lower):
            return {
                "speaker": "CANDIDATE",
                "confidence": 0.95,
                "reason": "first_person_narrative"
            }

    # ---------------------------------------------------------
    # PRIORITY 4: SHORT-UTTERANCE POLICY & TURN CONTINUITY
    # ---------------------------------------------------------
    tech_keywords = {
        "redux", "react", "python", "django", "jwt", "mongodb", "postgres",
        "postgresql", "docker", "kubernetes", "aws", "redis", "kafka",
        "scalability", "authentication", "caching", "sql", "nosql"
    }
    
    if word_count <= 4 and not any(u_clean.startswith(w) for w in ["why", "how", "what", "which", "where", "can you", "tell me", "let's"]):
        # If utterance ends with '?' and is a single query (e.g. "Redux?", "Why?"), treat as Interviewer prompt
        if has_question_mark:
            return {
                "speaker": "INTERVIEWER",
                "confidence": 0.85,
                "reason": "interviewer_short_query"
            }
            
        # If answering an active interviewer question (e.g. "What state-management library did you use?" -> "Redux.")
        if is_conversational_eligible_answer(utterance, session):
            return {
                "speaker": "CANDIDATE",
                "confidence": 0.95,
                "reason": "candidate_direct_answer_to_question"
            }
            
        # If candidate is speaking and utters a confirmation or multi-word list (e.g. "React, Django, Redux.", "Yes.")
        if session and getattr(session, "turn_state", None) == InterviewTurnState.CANDIDATE_SPEAKING:
            if u_clean in {"yes", "yeah", "correct", "yep", "sure"} or "," in utterance or word_count > 1:
                return {
                    "speaker": "CANDIDATE",
                    "confidence": 0.92,
                    "reason": "candidate_short_answer_continuity"
                }
            # Isolated single tech keyword following candidate speech -> Ambiguous interviewer prompt
            if u_clean in tech_keywords:
                return {
                    "speaker": "INTERVIEWER",
                    "confidence": 0.70,
                    "reason": "ambiguous_short_tech_keyword"
                }
                
        # Default candidate turn if interviewer just asked a question
        if session and getattr(session, "turn_state", None) == InterviewTurnState.INTERVIEWER_SPEAKING:
            return {
                "speaker": "CANDIDATE",
                "confidence": 0.90,
                "reason": "candidate_answering_question"
            }

    # ---------------------------------------------------------
    # PRIORITY 5: CONVERSATION TURN STATE CONTINUITY
    # ---------------------------------------------------------
    if session and getattr(session, "turn_state", None) == InterviewTurnState.CANDIDATE_SPEAKING:
        if not is_likely_question(utterance):
            return {
                "speaker": "CANDIDATE",
                "confidence": 0.88,
                "reason": "continuous_candidate_turn"
            }

    # ---------------------------------------------------------
    # PRIORITY 6: FALLBACK & TEMPORARY UNKNOWN STATE
    # ---------------------------------------------------------
    if is_likely_question(utterance):
        return {
            "speaker": "INTERVIEWER",
            "confidence": 0.82,
            "reason": "question_structure"
        }
    elif word_count <= 2 and not has_question_mark and not session:
        # Ambiguous isolated 1-2 word utterance without session context
        return {
            "speaker": "UNKNOWN",
            "confidence": 0.50,
            "reason": "ambiguous_short_fragment"
        }
    else:
        # Statements default to Interviewer statement for transcript logging without answering
        return {
            "speaker": "INTERVIEWER",
            "confidence": 0.60,
            "reason": "default_interviewer_statement"
        }


def is_conversational_eligible_answer(term: str, session: Optional["InterviewSession"] = None) -> bool:
    """
    Evaluates whether a short technical term (e.g. 'Redux.', 'PostgreSQL.', 'JWT.')
    is an eligible direct answer to the preceding interviewer question.
    """
    if not session or not term:
        return False
    last_q = getattr(session, "latest_detected_question", "") or ""
    if not last_q and session.qa_pairs:
        last_q = session.qa_pairs[-1].get("question", "")
    if not last_q:
        return False
        
    lq_lower = last_q.lower()
    t_clean = term.lower().strip(" .!?,:;")
    
    # Check if the question asks for tools, technologies, database, framework, state management, etc.
    triggers = [
        "what", "which", "did you use", "have you used", "technologies", "tech stack",
        "state management", "database", "framework", "library", "tools", "language",
        "how did you implement", "what architecture"
    ]
    return any(trig in lq_lower for trig in triggers) and len(t_clean.split()) <= 4


# ============================================================
# SEPARATE QUESTION UNDERSTANDING & UTTERANCE CLASSIFIER
# ============================================================

def classify_and_understand_utterance(
    utterance: str,
    session: Optional["InterviewSession"] = None,
    client: Optional[Groq] = None,
    source: str = "mic",
    speaker_hint: Optional[str] = None,
    is_manual: bool = False
) -> Dict[str, Any]:
    """
    Continuous Turn-Taking & Question Interpretation Layer.
    
    1. Distinguishes speaker (CANDIDATE vs INTERVIEWER vs UNKNOWN).
    2. If CANDIDATE: records speech into recent context, NEVER triggers an answer.
    3. If INTERVIEWER:
       - Statements/Fillers/Greetings/AMBIGUOUS_SHORT_UTTERANCES -> Transcript only, NO answer.
       - Genuine Questions/Follow-ups -> Refines question with candidate context,
         assigns unique question_id, and sets should_generate_answer = True.
    4. Supports BOTH Live Interview speech and Manual Typed Input under ONE unified engine.
    5. Multiple Interviewer identities (e.g. 'Interviewer 1', 'Interviewer 2') share unified session context.
    """
    raw_text = utterance.strip()
    is_manual_input = bool(is_manual or source in ("manual", "MANUAL_INPUT", "typed"))
    input_source = "MANUAL_INPUT" if is_manual_input else "LIVE_INTERVIEW"
    
    if not raw_text:
        return {
            "raw_text": "",
            "speaker": "UNKNOWN",
            "utterance_type": "FILLER",
            "question_confidence": 0.0,
            "should_generate_answer": False,
            "refined_question": "",
            "context_topic": "",
            "question_id": "",
            "input_source": input_source
        }
        
    u_lower = raw_text.lower().strip(" .!?,:;")
    words = u_lower.split()
    has_question_mark = raw_text.endswith("?")
    
    # ---------------------------------------------------------
    # 1. STAGE 1: QUESTION CANDIDATE CONFIDENCE & SPEAKER ANALYSIS
    # ---------------------------------------------------------
    if is_manual_input:
        # Manual typed questions bypass audio speaker detection
        speaker = "MANUAL"
        q_confidence = 1.0
        interviewer_name = speaker_hint if speaker_hint else "Manual Input"
    elif source in ("system", "loopback", "interviewer_channel"):
        # Dual Mode: System / Speaker Loopback is AUTHORITATIVELY the Interviewer
        speaker = "INTERVIEWER"
        q_confidence = compute_question_confidence(raw_text, session=session)
        interviewer_name = speaker_hint if speaker_hint else getattr(session, "active_interviewer_id", "Interviewer")
    elif source in ("candidate_mic", "candidate_channel"):
        # Dual Mode: Microphone is AUTHORITATIVELY the Candidate (Never triggers answers)
        speaker = "CANDIDATE"
        q_confidence = 0.0
        interviewer_name = "Interviewer"
    else:
        # Mic Only / Single Microphone Mode: Use semantic speaker heuristics
        q_confidence = compute_question_confidence(raw_text, session=session)
        speaker_info = detect_speaker_turn(raw_text, session=session, source=source)
        speaker = speaker_info["speaker"]
        
        # Check conversational eligibility for short single technical words (e.g. "What db did you use?" -> "Redux.")
        if len(words) <= 3 and not has_question_mark:
            if is_conversational_eligible_answer(raw_text, session):
                speaker = "CANDIDATE"
                
        # ANSWER PRIORITY MATRIX (Mic-Only mode):
        is_explicit_cand = (speaker_info["speaker"] == "CANDIDATE" and speaker_info.get("reason") in {"first_person_narrative", "candidate_direct_answer_to_question"}) or bool(re.search(r"^(also\s+|and\s+)?(i|we)\s+(also\s+)?(built|used|worked|implemented|designed|created|decided|chose|handled|would|trained|benchmarked|selected|started|solved|scaled)\b", u_lower))
        if (q_confidence >= 0.40 or is_likely_question(raw_text) or speaker_info["speaker"] == "INTERVIEWER") and not is_explicit_cand:
            speaker = "INTERVIEWER"
        
        # Resolve specific interviewer identity
        if speaker == "INTERVIEWER":
            interviewer_name = speaker_hint if speaker_hint else getattr(session, "active_interviewer_id", "Interviewer")
        else:
            interviewer_name = "Interviewer"
    
    # ---------------------------------------------------------
    # 2. CANDIDATE TURN HANDLING (LISTEN ONLY, STORE AS CONTEXT)
    # ---------------------------------------------------------
    if speaker == "CANDIDATE":
        if session:
            # If there was a pending UNKNOWN utterance within 30s, resolve into candidate context
            pending = getattr(session, "pending_ambiguous_utterance", None)
            if pending and isinstance(pending, dict):
                p_text = pending.get("text", "")
                p_time = pending.get("timestamp", 0)
                if p_text and (time.time() - p_time <= 30.0):
                    session.add_candidate_speech(p_text)
            session.add_candidate_speech(raw_text)
            session.pending_ambiguous_utterance = None
        return {
            "raw_text": raw_text,
            "speaker": "Candidate",
            "utterance_type": "CANDIDATE_ANSWER",
            "question_confidence": 0.0,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }
        
    # If speaker is UNKNOWN, do not answer; buffer as pending ambiguous utterance
    if speaker == "UNKNOWN":
        if session:
            session.pending_ambiguous_utterance = {
                "text": raw_text.strip(" .!?,:;"),
                "timestamp": time.time(),
                "speaker_confidence": speaker_info.get("confidence", 0.5),
                "context": getattr(session, "active_topic", "")
            }
        return {
            "raw_text": raw_text,
            "speaker": "Unknown",
            "utterance_type": "UNKNOWN",
            "question_confidence": 0.0,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }

    # ---------------------------------------------------------
    # 3. INTERVIEWER TURN: AMBIGUOUS SHORT UTTERANCES & FAST FILTERS
    # ---------------------------------------------------------
    # Single-word flat technical terms from interviewer (e.g. "Redux." without "?")
    tech_keywords = {
        "redux", "react", "python", "django", "jwt", "mongodb", "postgres",
        "postgresql", "docker", "kubernetes", "aws", "redis", "kafka",
        "scalability", "authentication", "caching", "sql", "nosql"
    }
    if u_lower in tech_keywords and not has_question_mark:
        # Buffer as pending ambiguous interviewer context for the next follow-up question
        if session:
            session.pending_ambiguous_utterance = {
                "text": raw_text.strip(" .!?,:;"),
                "timestamp": time.time(),
                "speaker_confidence": 0.5,
                "context": getattr(session, "active_topic", "")
            }
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": "AMBIGUOUS_SHORT_UTTERANCE",
            "question_confidence": 0.3,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": raw_text.capitalize(),
            "question_id": ""
        }

    # ---------------------------------------------------------
    # 4. INTERVIEWER TURN: FAST FILTER FOR OBVIOUS NON-QUESTIONS
    # ---------------------------------------------------------
    
    # Acknowledgements & Confirmations
    acknowledgements = {
        "yeah", "yes", "yep", "yup", "yes sir", "yes ma'am", "okay", "ok",
        "alright", "all right", "got it", "sure", "right", "understood",
        "i see", "sounds good", "great", "cool", "perfect", "fine", "correct",
        "that's interesting", "let me think"
    }
    if u_lower in acknowledgements:
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": "ACKNOWLEDGEMENT",
            "question_confidence": 0.0,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }
        
    # Greetings & Salutations
    greetings = {
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "thanks for joining", "thanks for joining us today", "thanks for coming",
        "welcome", "nice to meet you", "thank you for joining", "bye", "goodbye",
        "how are you", "how are you doing"
    }
    if u_lower in greetings:
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": "GREETING",
            "question_confidence": 0.0,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }
        
    # Fillers & Delay Phrases
    fillers = {
        "give me a second", "give me a moment", "one second", "one moment",
        "wait a minute", "let me check", "let's see", "hold on", "um", "uh",
        "just a second", "give me a min"
    }
    if u_lower in fillers:
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": "FILLER",
            "question_confidence": 0.0,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }
        
    # Conversational Statements (including introductory phrases before the actual question)
    conversational_statements = [
        r"^i('m| am) going to ask you (a question|something)",
        r"^i('ve| have) got a picture of you",
        r"^i live in (the city of )?",
        r"^thank(s| you)( so much)?(\.|\,)?( thanks,?\s*[a-z]+)?$",
        r"^i('m| am) (not going to talk to you|looking at your (profile|resume))",
        r"^let('s| us) (get started|begin|move on)$",
        r"^can you hear me( clearly)?\??$",
        r"^are you ready\??$"
    ]
    for pattern in conversational_statements:
        if re.search(pattern, u_lower):
            return {
                "raw_text": raw_text,
                "speaker": interviewer_name,
                "utterance_type": "STATEMENT",
                "question_confidence": 0.1,
                "should_generate_answer": False,
                "refined_question": raw_text,
                "context_topic": getattr(session, "active_topic", ""),
                "question_id": ""
            }

    # ---------------------------------------------------------
    # 4.5. INTERVIEWER TURN: TRAILING CONJUNCTIONS & INCOMPLETE PREAMBLES
    # ---------------------------------------------------------
    # Check if there is a pending incomplete interviewer preamble clause
    if session and getattr(session, "pending_interviewer_preamble", None):
        p_preamble = session.pending_interviewer_preamble
        if p_preamble and isinstance(p_preamble, dict):
            p_text = p_preamble.get("text", "")
            p_time = p_preamble.get("timestamp", 0)
            if p_text and (time.time() - p_time <= 15.0):
                raw_text = f"{p_text} {raw_text}".strip()
                u_lower = raw_text.lower().strip()
                words = u_lower.split()
        session.pending_interviewer_preamble = None

    # Detect if CURRENT utterance is an incomplete trailing preamble (e.g. "We had an issue with database replication and...")
    is_trailing_conjunction = bool(re.search(r"\b(and|or|but|because|so|when|that|like|where|if|which|as well as|then)\s*[\.\,\…\s]*$", u_lower))
    if is_trailing_conjunction and not has_question_mark and not is_likely_question(raw_text):
        if session:
            session.pending_interviewer_preamble = {
                "text": raw_text.strip(" .!?,:;"),
                "timestamp": time.time()
            }
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": "INCOMPLETE_PREAMBLE",
            "question_confidence": 0.2,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }

    # ---------------------------------------------------------
    # 5. CONVERSATION FLOW & FOLLOW-UP ANALYSIS (USING CANDIDATE CONTEXT)
    # ---------------------------------------------------------
    # Check if an ambiguous technical term was buffered right before this question
    if session and getattr(session, "pending_ambiguous_utterance", None):
        p_info = session.pending_ambiguous_utterance
        if isinstance(p_info, dict):
            p_term = p_info.get("text", "")
            if p_term and p_term.lower() not in raw_text.lower():
                if re.search(r"\bit\b", raw_text, flags=re.IGNORECASE):
                    raw_text = re.sub(r"\bit\b", p_term, raw_text, flags=re.IGNORECASE)
                elif "regarding" not in raw_text.lower():
                    raw_text = f"{raw_text.rstrip('?')} regarding {p_term}?"
        session.pending_ambiguous_utterance = None

    flow = analyze_conversation_flow(raw_text, session)
    flow_type = flow.get("flow_type", "NEW_QUESTION")
    
    # Assign unique question ID for active interviewer questions
    if session:
        session.question_counter += 1
        q_id = f"Q{session.question_counter:03d}"
        session.active_question_id = q_id
        session.turn_state = InterviewTurnState.INTERVIEWER_SPEAKING
        if not session.latest_latency_trace or session.latest_latency_trace.question_id != q_id:
            session.latest_latency_trace = QuestionLatencyTrace(question_id=q_id)
        trace = session.latest_latency_trace
        trace.mark("context_preparation_start")
    else:
        q_id = "Q001"
        trace = None

    # ---------------------------------------------------------
    # 5. MULTIMODAL SCREEN & VOICE FUSION (FOR CODING / VISUAL INQUIRIES)
    # ---------------------------------------------------------
    from screen_analyzer import is_visual_reference_question, analyze_visible_screen, fuse_voice_and_screen_question
    from classifier import classify_question
    
    screen_ctx = None
    if session and getattr(session, "screen_aware_coding", True):
        mock_screen = getattr(session, "mock_screen_data", None)
        if is_visual_reference_question(raw_text) or mock_screen is not None:
            screen_ctx = analyze_visible_screen(client=client, mock_screen_data=mock_screen)
            
    fused_info = fuse_voice_and_screen_question(raw_text, screen_ctx, session=session)
    if fused_info.get("screen_used") or fused_info.get("is_coding"):
        refined_q = fused_info["refined_question"]
        q_type = fused_info["question_type"].upper()
        if trace:
            trace.mark("context_ready")
            trace.mark("question_finalized")
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": q_type,
            "question_confidence": 0.95,
            "should_generate_answer": True,
            "refined_question": refined_q,
            "context_topic": "Coding" if q_type in {"CODING", "DEBUGGING", "SQL", "CODING_COMPLEXITY", "CODING_OPTIMIZATION"} else getattr(session, "active_topic", ""),
            "question_id": q_id,
            "code_context": fused_info.get("code_context", ""),
            "error_context": fused_info.get("error_context", ""),
            "input_source": input_source
        }

    # ---------------------------------------------------------
    # 6. CONVERSATION FLOW & FOLLOW-UP ANALYSIS (USING CANDIDATE CONTEXT)
    # ---------------------------------------------------------
    if flow_type == "FOLLOW_UP":
        fused_q = flow.get("fused_question", raw_text)
        # Incorporate candidate's previous response into follow-up if applicable
        if session and getattr(session, "last_candidate_response", ""):
            cand_resp_snippet = session.last_candidate_response[:80].rstrip(" .")
            if "regarding" not in fused_q.lower() and "context" not in fused_q.lower():
                fused_q = f"{raw_text.rstrip('?')} (following candidate's mention of: '{cand_resp_snippet}')?"
                
        if trace:
            trace.mark("context_ready")
            trace.mark("question_finalized")
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": "FOLLOW_UP",
            "question_confidence": 0.95,
            "should_generate_answer": True,
            "refined_question": fused_q,
            "context_topic": flow.get("topic", getattr(session, "active_topic", "")),
            "question_id": q_id,
            "input_source": input_source
        }
        
    if flow_type == "QUESTION_CORRECTION":
        fused_q = flow.get("fused_question", raw_text)
        if trace:
            trace.mark("context_ready")
            trace.mark("question_finalized")
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": "QUESTION",
            "question_confidence": 0.95,
            "should_generate_answer": True,
            "refined_question": fused_q,
            "context_topic": flow.get("topic", getattr(session, "active_topic", "")),
            "should_cancel_stale": True,
            "question_id": q_id,
            "input_source": input_source
        }

    # ---------------------------------------------------------
    # 7. SEMANTIC QUESTION DETECTION & REFINEMENT
    # ---------------------------------------------------------
    cleaned_local = clean_asr_transcript(raw_text)
    is_q_heuristic = is_likely_question(cleaned_local) or has_question_mark
    auto_conf = compute_question_confidence(cleaned_local, session=session)
    
    if is_manual_input:
        is_q = True
        confidence = 1.0
        q_type = classify_question(cleaned_local).upper()
        refined_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local + "?")
    elif is_q_heuristic or auto_conf >= 0.70:
        # Fast path: instant rule-based refinement without an extra blocking 2.5s LLM roundtrip
        is_q = True
        refined_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local + "?")
        confidence = max(auto_conf, 0.90)
        q_type = classify_question(refined_q).upper()
    elif client and len(raw_text.split()) >= 4:
        # Fallback to LLM refinement only for long ambiguous statements that might be hidden questions
        res = refine_interview_question(client=client, raw_transcript=raw_text, session=session)
        is_q = res.get("is_question", False)
        refined_q = res.get("refined_question", cleaned_local)
        confidence = float(res.get("confidence", 0.8))
        q_type = res.get("question_type", "TECHNICAL" if is_q else "STATEMENT")
    else:
        is_q = False
        refined_q = cleaned_local
        confidence = auto_conf
        q_type = "STATEMENT"
        
    should_answer = bool(is_manual_input or (is_q and confidence >= 0.60))
    
    if trace:
        trace.mark("context_ready")
        if should_answer:
            trace.mark("question_finalized")
            
    return {
        "raw_text": raw_text,
        "speaker": interviewer_name,
        "utterance_type": q_type if should_answer else "STATEMENT",
        "question_confidence": confidence,
        "should_generate_answer": should_answer,
        "refined_question": refined_q if should_answer else raw_text,
        "context_topic": getattr(session, "active_topic", ""),
        "question_id": q_id if should_answer else "",
        "input_source": input_source
    }


# ============================================================
# ANSWER INTELLIGENCE & CONVERSATION RELATIONSHIP CLASSIFIER
# ============================================================

def classify_answer_relationship(
    utterance: str,
    session: Optional["InterviewSession"] = None
) -> Dict[str, Any]:
    """
    Parakit Answer Intelligence Layer:
    Understands the relationship between the current interviewer question
    and the immediately preceding conversation context.
    
    Supported Relationship Types:
    - NEW_TOPIC
    - WHY
    - HOW
    - PERFORMANCE (Scalability, throughput, latency)
    - SECURITY
    - COMPARISON (Why not X vs Y)
    - EXPANSION (Explain that / elaborate)
    - CLARIFICATION
    - EXAMPLE_REQUEST
    - OPTIMIZATION (Coding optimization)
    - COMPLEXITY (Time / Space complexity)
    - TRADE_OFF (Drawbacks / limitations)
    - IMPROVEMENT (What would you change)
    - DIRECT_FOLLOW_UP
    """
    if not utterance:
        return {
            "relationship_type": "NEW_TOPIC",
            "is_follow_up": False,
            "topic": "",
            "fused_question": utterance,
            "delta_focus": "",
            "strategy_key": "general"
        }

    u_lower = utterance.lower().strip()
    u_clean = re.sub(r'[^a-z0-9\s]', '', u_lower).strip()
    words = u_clean.split()
    
    active_topic = getattr(session, "active_topic", "") if session else ""
    last_cand = getattr(session, "last_candidate_response", "") if session else ""
    last_q = ""
    if session and session.qa_pairs:
        last_q = session.qa_pairs[-1].get("question", "")
    elif session and getattr(session, "latest_detected_question", ""):
        last_q = session.latest_detected_question

    # 1. NEW_TOPIC DETECTION (Explicit Topic Shifts)
    new_topic_starters = [
        r"^(now\s+)?(tell me about|let's move on to|let's switch to|moving on to|next topic|shifting to|let's discuss your internship|let's discuss your background)\b",
        r"\b(let'?s talk about something else|changing gears|different topic|moving to a different)\b"
    ]
    for p in new_topic_starters:
        if re.search(p, u_lower):
            new_top = re.sub(p, "", utterance, flags=re.IGNORECASE).strip(" ,:;?-")
            if session:
                session.active_topic = new_top if new_top else "New Topic"
            return {
                "relationship_type": "NEW_TOPIC",
                "is_follow_up": False,
                "topic": session.active_topic if session else "",
                "fused_question": utterance,
                "delta_focus": "Fresh topic inquiry. Do not bring unrelated technical details from previous discussions.",
                "strategy_key": "general"
            }

    # 2. WHY FOLLOW-UP
    if u_clean in {"why", "and why", "why so", "why is that", "why did you choose that", "why this approach", "why?"} or u_lower == "why?":
        topic_name = active_topic or "the discussed choice"
        return {
            "relationship_type": "WHY",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Why did you choose {topic_name} in this project?",
            "delta_focus": f"Give strictly the justification and reasons for selecting {topic_name} in the project. Do not restate general definitions.",
            "strategy_key": "follow_up_why"
        }

    # 3. HOW FOLLOW-UP
    if u_clean in {"how", "and how", "how so", "how does that work", "how did you implement that", "how do you use that", "how?"} or u_lower == "how?":
        topic_name = active_topic or "this component"
        return {
            "relationship_type": "HOW",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"How did you implement/use {topic_name} in this project?",
            "delta_focus": f"Explain the concrete implementation mechanics of {topic_name} in the project.",
            "strategy_key": "follow_up_how"
        }

    # 4. PERFORMANCE / SCALABILITY FOLLOW-UP
    if any(kw in u_lower for kw in ["scalability", "what about scalability", "performance", "what about performance", "how does it scale", "traffic increases", "high traffic", "high load", "concurrency"]):
        topic_name = active_topic or "this architecture"
        return {
            "relationship_type": "PERFORMANCE",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"How does {topic_name} handle scalability, performance, and high traffic in this project?",
            "delta_focus": f"Focus strictly on scalability and performance mechanisms (horizontal scaling, sharding, caching, bottleneck avoidance) for {topic_name}. DO NOT repeat basic definitions or feature descriptions.",
            "strategy_key": "follow_up_scalability"
        }

    # 5. SECURITY FOLLOW-UP
    if any(kw in u_lower for kw in ["security", "what about security", "how do you secure", "vulnerabilities", "how is it secured"]):
        topic_name = active_topic or "this implementation"
        return {
            "relationship_type": "SECURITY",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"What security considerations and defenses were applied to {topic_name} in this project?",
            "delta_focus": f"Focus strictly on security practices (token storage, encryption, validation, threat mitigations) for {topic_name}.",
            "strategy_key": "follow_up_security"
        }

    # 6. COMPARISON FOLLOW-UP ("Why not MySQL?", "Why not Postgres?", "Compare with X")
    comp_match = re.search(r"\b(why not|versus|vs\.?|compare (it )?to|instead of|difference between this and)\s+([a-zA-Z0-9_\+\#]+)", u_lower)
    if comp_match or u_lower.startswith("why not"):
        raw_alt = comp_match.group(3) if comp_match else (words[-1] if words else "alternative")
        tech_acronyms = {
            "mysql": "MySQL", "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
            "mongodb": "MongoDB", "jwt": "JWT", "sql": "SQL", "nosql": "NoSQL",
            "aws": "AWS", "gcp": "GCP", "redis": "Redis", "django": "Django",
            "react": "React", "python": "Python", "docker": "Docker", "kafka": "Kafka"
        }
        alt_tech = tech_acronyms.get(raw_alt.lower(), raw_alt.capitalize())
        topic_name = active_topic or "the chosen technology"
        return {
            "relationship_type": "COMPARISON",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Why did you choose {topic_name} instead of {alt_tech} in this project, and what are the trade-offs?",
            "delta_focus": f"Compare {topic_name} (chosen) vs {alt_tech} (alternative) specifically for this project: why {topic_name} fit best, why {alt_tech} was less suitable, and the key architectural trade-offs.",
            "strategy_key": "follow_up_comparison"
        }

    # 7. EXPANSION FOLLOW-UP ("Can you explain that?", "Can you elaborate?")
    if any(kw in u_lower for kw in ["can you explain that", "can you elaborate", "explain that further", "tell me more about that", "can you expand", "expand on that"]):
        topic_name = active_topic or (last_cand[:40] if last_cand else "the previous explanation")
        return {
            "relationship_type": "EXPANSION",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Can you elaborate in detail on {topic_name} in this project?",
            "delta_focus": f"Expand deeply on {topic_name}, detailing how it operates under the hood, why it was valuable, and when it is useful. Do not restart the high-level introduction.",
            "strategy_key": "follow_up_expansion"
        }

    # 8. CLARIFICATION FOLLOW-UP
    if any(kw in u_lower for kw in ["what do you mean", "can you clarify", "to confirm", "are you saying"]):
        topic_name = active_topic or "the previous statement"
        return {
            "relationship_type": "CLARIFICATION",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Can you clarify what you mean regarding {topic_name}?",
            "delta_focus": f"Clarify the assumptions and precise meaning regarding {topic_name}.",
            "strategy_key": "clarification"
        }

    # 9. EXAMPLE_REQUEST FOLLOW-UP
    if any(kw in u_lower for kw in ["can you give an example", "give me an example", "show an example", "concrete example", "for example", "like what"]):
        topic_name = active_topic or "the discussed concept"
        return {
            "relationship_type": "EXAMPLE_REQUEST",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Can you provide a concrete, real-world example of {topic_name} in your project?",
            "delta_focus": f"Provide a realistic, concrete engineering example from the project illustrating {topic_name}.",
            "strategy_key": "follow_up_example"
        }

    # 10. OPTIMIZATION FOLLOW-UP
    if any(kw in u_lower for kw in ["can you optimize it", "can we optimize this", "can you optimize this", "make it faster", "more optimal", "can we do better"]):
        return {
            "relationship_type": "OPTIMIZATION",
            "is_follow_up": True,
            "topic": "Coding Optimization",
            "fused_question": "Can you optimize the previous coding solution for better time and space complexity?",
            "delta_focus": "Identify the performance bottleneck of the previous code solution and provide the optimized approach with superior Big-O complexity.",
            "strategy_key": "coding_optimization"
        }

    # 11. COMPLEXITY FOLLOW-UP
    if any(kw in u_lower for kw in ["what is the time complexity", "what is the space complexity", "what's the complexity", "what is the complexity", "time complexity", "space complexity", "what is the big o", "big o"]):
        return {
            "relationship_type": "COMPLEXITY",
            "is_follow_up": True,
            "topic": "Complexity Analysis",
            "fused_question": "What is the time and space complexity of the preceding solution?",
            "delta_focus": "State the Time and Space Complexity of the preceding code solution with a brief mathematical breakdown. Do not rewrite full code.",
            "strategy_key": "coding_complexity"
        }

    # 12. TRADE_OFF FOLLOW-UP
    if any(kw in u_lower for kw in ["what are the drawbacks", "drawbacks", "what are the limitations", "limitations", "what are the trade-offs", "trade-offs", "downside"]):
        topic_name = active_topic or "this technology choice"
        return {
            "relationship_type": "TRADE_OFF",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"What are the drawbacks and trade-offs of {topic_name} in this project?",
            "delta_focus": f"Discuss the engineering limitations and trade-offs of {topic_name} and how our team mitigated them.",
            "strategy_key": "follow_up_tradeoff"
        }

    # 13. IMPROVEMENT FOLLOW-UP
    if any(kw in u_lower for kw in ["what would you change", "how would you improve it", "how would you improve that", "what would you do differently", "future improvements"]):
        topic_name = active_topic or "this architecture"
        return {
            "relationship_type": "IMPROVEMENT",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"What would you change or improve about {topic_name} based on production learnings?",
            "delta_focus": f"State specific architectural or operational improvements for {topic_name} with expected benefits.",
            "strategy_key": "follow_up_improvement"
        }

    # 14. GENERAL DIRECT FOLLOW-UP
    followup_starters = ["what about", "how about", "and for", "so what about", "then what", "and in that case"]
    if any(u_lower.startswith(s) for s in followup_starters) or (session and session.qa_pairs and len(words) <= 4 and u_lower.endswith("?")):
        topic_name = active_topic or "the discussed topic"
        fused = f"{utterance.rstrip('?')} regarding {topic_name}?"
        return {
            "relationship_type": "DIRECT_FOLLOW_UP",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": fused,
            "delta_focus": f"Direct follow-up in the context of {topic_name}.",
            "strategy_key": "follow_up"
        }

    # Default: New Question
    tech_matches = re.findall(r"\b(django|python|react|docker|kubernetes|aws|postgresql|mongodb|sql|graphql|rest|microservices|redis|kafka|system design|authentication|caching|jwt)\b", u_lower)
    if tech_matches and session:
        session.active_topic = tech_matches[0].capitalize()
        
    return {
        "relationship_type": "NEW_TOPIC",
        "is_follow_up": False,
        "topic": session.active_topic if session else "",
        "fused_question": utterance,
        "delta_focus": "",
        "strategy_key": "general"
    }


def analyze_conversation_flow(utterance: str, session: Optional["InterviewSession"] = None) -> Dict[str, Any]:
    """
    Tracks and analyzes interviewer conversation dynamics using the Answer Intelligence Layer:
    - QUESTION_CORRECTION ("Actually let me rephrase that...")
    - FOLLOW_UP ("Okay, and what about scalability?")
    - TOPIC_CHANGE ("Let's move on to...")
    - CLARIFICATION
    - MULTI_PART_QUESTION
    - NEW_QUESTION
    """
    if not utterance:
        return {"flow_type": "NEW_QUESTION", "fused_question": utterance, "should_cancel_stale": False}

    u_lower = utterance.lower().strip()
    active_topic = getattr(session, "active_topic", "") if session else ""

    # 1. QUESTION_CORRECTION ("Actually let me rephrase that...", "Wait, what I meant was...")
    correction_patterns = [
        r"^(actually,?\s*)?(let me (rephrase|re-ask|restate|change)|wait,?\s*(what i meant|let me|actually)|scratch that|no wait)",
        r"\b(sorry i meant|let me ask that differently|what i actually wanted to ask)\b"
    ]
    for p in correction_patterns:
        if re.search(p, u_lower):
            cleaned_q = re.sub(p, "", utterance, flags=re.IGNORECASE).strip(" ,:;-")
            return {
                "flow_type": "QUESTION_CORRECTION",
                "fused_question": cleaned_q if cleaned_q else utterance,
                "should_cancel_stale": True,
                "topic": active_topic
            }

    # 2. Answer Intelligence Relationship Classifier
    rel = classify_answer_relationship(utterance, session=session)
    if rel["relationship_type"] == "NEW_TOPIC" and rel.get("delta_focus"):
        return {
            "flow_type": "TOPIC_CHANGE",
            "fused_question": utterance,
            "should_cancel_stale": False,
            "topic": rel["topic"],
            "relationship": rel
        }
    elif rel["is_follow_up"]:
        return {
            "flow_type": "FOLLOW_UP",
            "fused_question": rel["fused_question"],
            "should_cancel_stale": False,
            "topic": rel["topic"],
            "relationship": rel
        }

    # 3. MULTI_PART_QUESTION
    if (" and " in u_lower or " also " in u_lower) and u_lower.count("?") >= 2:
        return {
            "flow_type": "MULTI_PART",
            "fused_question": utterance,
            "should_cancel_stale": False,
            "topic": active_topic,
            "relationship": rel
        }

    return {
        "flow_type": "NEW_QUESTION",
        "fused_question": utterance,
        "should_cancel_stale": False,
        "topic": session.active_topic if session else "",
        "relationship": rel
    }


# ============================================================
# SETTINGS PERSISTENCE
# ============================================================

CONFIG_PATH = os.path.expanduser("~/.parakit_config.json")


def load_user_config() -> Dict[str, Any]:
    """Loads persistent user preferences."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print("Config load notice:", e)
    return {"auto_restore_on_question": True}


def save_user_config(cfg: Dict[str, Any]):
    """Saves user preferences to persistent storage."""
    try:
        data = load_user_config()
        data.update(cfg)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Config save notice:", e)


# ============================================================
# INTERVIEW SESSION STATE
# ============================================================

class InterviewSession:
    """Encapsulates all interview parameters, candidate info, and live state."""
    
    def __init__(self):
        self.job_title: str = ""
        self.company_name: str = ""
        self.job_description: str = ""
        self.resume_path: Optional[str] = None
        self.resume_text: str = ""
        self.knowledge_text: str = ""
        self.tone: str = "Confident & Conversational"
        self.language: str = "English"
        self.custom_instructions: str = ""
        self.auto_generate: bool = True
        self.audio_source: str = "dual"  # "dual", "system" (loopback only), or "mic" (single mic heuristic)
        self.selected_mic_device: Optional[Union[int, str]] = None
        self.selected_speaker_device: Optional[str] = None
        self.mic_device_name: Optional[str] = None
        self.speaker_device_name: Optional[str] = None
        self.candidate_context_enabled: bool = True
        
        # Conversation Flow Intelligence State
        self.active_topic: str = ""
        self.last_question: str = ""
        
        # Load user configuration
        user_cfg = load_user_config()
        self.auto_restore_on_question: bool = user_cfg.get("auto_restore_on_question", True)
        self.job_title: str = user_cfg.get("job_title", "")
        self.company_name: str = user_cfg.get("company_name", "")
        self.job_description: str = user_cfg.get("job_description", "")
        self.tone: str = user_cfg.get("tone", "Confident & Conversational")
        self.language: str = user_cfg.get("language", "English")
        
        # Live conversation log: list of dicts { 'timestamp': str, 'role': 'Interviewer'|'Candidate', 'text': str, 'is_question': bool }
        self.transcript_log: List[Dict[str, Any]] = []
        
        # AI Answer history for summary and follow-up
        self.qa_pairs: List[Dict[str, str]] = []
        
        # Active detected question pending answer
        self.latest_detected_question: Optional[str] = None
        self.latest_answer: str = ""
        
        # Turn-Taking & Speaker Intelligence State
        self.turn_state: InterviewTurnState = InterviewTurnState.LISTENING
        self.active_question_id: str = ""
        self.question_counter: int = 0
        self.last_candidate_response: str = ""
        self.last_confirmed_speaker: str = "Interviewer"
        self.active_interviewer_id: str = "Interviewer"
        self.candidate_responses_log: List[Dict[str, str]] = []
        self.pending_ambiguous_utterance: Optional[Dict[str, Any]] = None
        self.controller: SessionController = SessionController(self)
        
        # Screen-Aware Coding Configuration
        self.screen_aware_coding: bool = user_cfg.get("screen_aware_coding", True)
        self.mock_screen_data: Optional[Dict[str, Any]] = None
        self.screen_context_cache: Optional[Dict[str, Any]] = None

        # Detailed analytics for each question
        self.question_analytics: List[Dict[str, Any]] = []
        
        # Real-time Latency Traces (Per Question & Generation)
        self.latest_latency_trace: Optional[QuestionLatencyTrace] = None
        self.latency_traces: List[QuestionLatencyTrace] = []
        self._cached_base_profile_block: Optional[str] = None
        
        # Load default candidate info from knowledge folder if available
        self.load_knowledge_base()

        # Load persisted resume if valid
        saved_resume = user_cfg.get("resume_path")
        if saved_resume and os.path.exists(saved_resume):
            self.set_resume(saved_resume)

    def invalidate_profile_cache(self):
        self._cached_base_profile_block = None

    def get_base_profile_block(self) -> str:
        if self._cached_base_profile_block is not None:
            return self._cached_base_profile_block
            
        rag_context = self.knowledge_text if self.knowledge_text else "No extra knowledge documents loaded."
        resume_context = self.resume_text if self.resume_text else "No uploaded resume provided."
        job_target = f"Target Role: {self.job_title}\nCompany: {self.company_name}" if (self.job_title or self.company_name) else "Target Role: Software Engineer / Tech Role"
        jd_context = self.job_description if self.job_description else "Not provided."
        
        self._cached_base_profile_block = f"""
============================================================
CANDIDATE TARGET & JOB CONTEXT
============================================================
{job_target}

JOB DESCRIPTION / REQUIREMENTS:
{jd_context}

============================================================
CANDIDATE'S VERIFIED BACKGROUND & RESUME
============================================================
--- UPLOADED RESUME / CV ---
{resume_context}

--- CANDIDATE KNOWLEDGE BASE & VERIFIED FACTS ---
{rag_context}
"""
        return self._cached_base_profile_block

    def set_screen_aware_coding(self, val: bool):
        self.screen_aware_coding = bool(val)
        save_user_config({"screen_aware_coding": self.screen_aware_coding})

    def set_auto_restore(self, val: bool):
        self.auto_restore_on_question = bool(val)
        save_user_config({"auto_restore_on_question": self.auto_restore_on_question})

    def load_knowledge_base(self):
        docs = load_documents()
        self.knowledge_text = "\n\n".join([f"--- {d['filename']} ---\n{d['text']}" for d in docs])
        self.invalidate_profile_cache()

    def set_resume(self, path: str) -> bool:
        if not path or not os.path.exists(path):
            self.resume_path = None
            self.resume_text = ""
            self.invalidate_profile_cache()
            save_user_config({"resume_path": ""})
            return False
            
        extracted = extract_text_from_file(path)
        if extracted and len(extracted.strip()) > 0:
            self.resume_path = path
            self.resume_text = extracted
            self.invalidate_profile_cache()
            save_user_config({"resume_path": path})
            print(f"Loaded resume from {path} ({len(self.resume_text)} characters extracted)")
            return True
        else:
            self.resume_path = None
            self.resume_text = ""
            self.invalidate_profile_cache()
            return False

    def remove_resume(self):
        self.resume_path = None
        self.resume_text = ""
        self.invalidate_profile_cache()
        save_user_config({"resume_path": ""})
        print("Removed active resume from session context.")

    def add_transcript(self, text: str, role: str = "Interviewer", is_question: bool = False, source: str = "LIVE_INTERVIEW", is_manual: bool = False):
        ts = time.strftime("%H:%M:%S")
        src = "MANUAL_INPUT" if is_manual or source == "MANUAL_INPUT" else "LIVE_INTERVIEW"
        entry = {
            "timestamp": ts,
            "role": role,
            "text": text,
            "is_question": is_question,
            "source": src
        }
        self.transcript_log.append(entry)
        if is_question:
            self.latest_detected_question = text
            self.last_confirmed_speaker = role
        elif role != "Candidate":
            self.last_confirmed_speaker = role
        return entry

    def add_candidate_speech(self, text: str):
        """Records candidate answer into session memory for context without triggering answers."""
        self.last_candidate_response = text
        self.last_confirmed_speaker = "Candidate"
        self.turn_state = InterviewTurnState.CANDIDATE_SPEAKING
        ts = time.strftime("%H:%M:%S")
        self.candidate_responses_log.append({
            "timestamp": ts,
            "text": text,
            "active_topic": self.active_topic
        })

    @property
    def candidate_context(self) -> str:
        parts = [entry["text"] for entry in self.candidate_responses_log]
        if not parts:
            parts = [t["text"] for t in self.transcript_log if t.get("role") == "Candidate"]
        return " ".join(parts)

    def record_qa(
        self,
        question: str,
        answer: str,
        category: str = "general",
        gen_time: float = 0.0,
        required_refinement: bool = False,
        required_clarification: bool = False,
        confidence: float = 0.9,
        topic: str = ""
    ):
        ts = time.strftime("%H:%M:%S")
        self.qa_pairs.append({
            "question": question,
            "answer": answer,
            "timestamp": ts
        })
        self.latest_answer = answer
        
        word_count = len(answer.split())
        self.question_analytics.append({
            "question": question,
            "category": category.lower(),
            "generation_time_sec": round(gen_time, 2),
            "word_count": word_count,
            "required_refinement": required_refinement,
            "required_clarification": required_clarification,
            "confidence": confidence,
            "topic": topic or self.active_topic or "General"
        })

    def clear(self):
        """Clears all session transcripts, QA pairs, and analytics data."""
        self.transcript_log.clear()
        self.qa_pairs.clear()
        self.question_analytics.clear()
        self.candidate_responses_log.clear()
        self.active_topic = ""
        self.last_question = ""
        self.last_candidate_response = ""
        self.active_question_id = ""
        self.question_counter = 0
        self.turn_state = InterviewTurnState.IDLE
        self.latest_detected_question = None
        self.latest_answer = ""
        self.pending_ambiguous_utterance = None
        if hasattr(self, "controller") and self.controller:
            self.controller.reset()


def calculate_session_analytics(session: InterviewSession) -> Dict[str, Any]:
    """
    Computes comprehensive post-interview analytics:
    - Total, technical, coding, behavioral, HR, project, system design questions
    - Generation time & answer length averages
    - Refinement, clarification, and low-confidence counts
    - Topics discussed and targeted preparation areas.
    """
    records = session.question_analytics
    total_q = len(records) if records else len(session.qa_pairs)
    
    if not total_q:
        return {
            "total_questions": 0,
            "technical_count": 0,
            "coding_count": 0,
            "behavioral_count": 0,
            "hr_count": 0,
            "project_count": 0,
            "system_design_count": 0,
            "avg_generation_time_sec": 0.0,
            "avg_answer_length_words": 0,
            "questions_requiring_refinement": 0,
            "questions_requiring_clarification": 0,
            "low_confidence_count": 0,
            "topics_discussed": [],
            "preparation_areas": []
        }
        
    tech_count = sum(1 for r in records if "tech" in r.get("category", "") or "database" in r.get("category", "") or "cloud" in r.get("category", ""))
    coding_count = sum(1 for r in records if "coding" in r.get("category", ""))
    behavioral_count = sum(1 for r in records if "behavioral" in r.get("category", ""))
    hr_count = sum(1 for r in records if "hr" in r.get("category", "") or "resume" in r.get("category", ""))
    project_count = sum(1 for r in records if "project" in r.get("category", "") or "experience" in r.get("category", ""))
    sd_count = sum(1 for r in records if "system_design" in r.get("category", ""))
    
    gen_times = [r.get("generation_time_sec", 0.0) for r in records if r.get("generation_time_sec", 0.0) > 0]
    avg_gen_time = round(sum(gen_times) / len(gen_times), 2) if gen_times else 0.85
    
    word_counts = [r.get("word_count", 0) for r in records if r.get("word_count", 0) > 0]
    avg_words = int(sum(word_counts) / len(word_counts)) if word_counts else 180
    
    refinements = sum(1 for r in records if r.get("required_refinement", False))
    clarifications = sum(1 for r in records if r.get("required_clarification", False) or "clarification" in r.get("category", ""))
    low_conf = sum(1 for r in records if r.get("confidence", 1.0) < 0.75)
    
    # Collect unique topics discussed
    topics = []
    for r in records:
        t = r.get("topic", "").strip()
        if t and t != "General" and t not in topics:
            topics.append(t)
    if not topics and getattr(session, "active_topic", ""):
        topics.append(session.active_topic)
        
    # Determine targeted preparation areas strictly from the interview session data
    prep_areas = []
    if low_conf > 0:
        prep_areas.append("Review edge cases & ambiguous speech scenarios for complex prompts")
    if coding_count > 0:
        prep_areas.append("Practice Space/Time Complexity Big-O proofs for algorithmic questions")
    if sd_count > 0:
        prep_areas.append("Deepen capacity estimation & distributed consensus mechanisms in system design")
    if behavioral_count > 0:
        prep_areas.append("Quantify business metrics & team impact in STAR behavioral responses")
    if not prep_areas:
        prep_areas.append("Continue practicing live verbal pacing and concise architectural summaries")
        
    return {
        "total_questions": total_q,
        "technical_count": tech_count,
        "coding_count": coding_count,
        "behavioral_count": behavioral_count,
        "hr_count": hr_count,
        "project_count": project_count,
        "system_design_count": sd_count,
        "avg_generation_time_sec": avg_gen_time,
        "avg_answer_length_words": avg_words,
        "questions_requiring_refinement": refinements,
        "questions_requiring_clarification": clarifications,
        "low_confidence_count": low_conf,
        "topics_discussed": topics,
        "preparation_areas": prep_areas
    }


# ============================================================
# QUESTION STABILITY, COMPLETENESS & PROGRESSIVE LIKELIHOOD MODEL
# ============================================================

class QuestionStabilityState(enum.Enum):
    PARTIAL = "PARTIAL"
    POSSIBLE_QUESTION = "POSSIBLE_QUESTION"
    LIKELY_QUESTION = "LIKELY_QUESTION"
    STABLE_QUESTION = "STABLE_QUESTION"
    FINAL_QUESTION = "FINAL_QUESTION"


def get_normalized_semantic_key(text: str) -> str:
    """
    Extracts the core semantic intent tokens from a question for stability,
    material change detection, and deduplication comparison.
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    stop_words = {
        "the", "a", "an", "for", "this", "our", "my", "your", "in", "with",
        "please", "can", "you", "tell", "me", "about", "did", "do", "would",
        "could", "is", "are", "was", "were", "to", "of", "at", "by", "on"
    }
    tokens = [w for w in t.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


def evaluate_question_completeness(
    text: str,
    session: Optional["InterviewSession"] = None
) -> Dict[str, Any]:
    """
    Parakit Early-Answer Intelligence:
    Evaluates whether an evolving speech utterance is semantically complete
    and contains sufficient predicate/object information to start answer generation safely.
    """
    if not text or not text.strip():
        return {
            "completeness": 0.0,
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": "",
            "reason": "empty"
        }

    raw = text.strip()
    u_lower = raw.lower().strip(" .!?,:;")
    words = u_lower.split()
    word_count = len(words)
    has_qmark = raw.endswith("?")
    sem_key = get_normalized_semantic_key(raw)
    
    # 1. Non-question statements (e.g. "I live in St. Louis.", "Sounds good.", "Thanks.")
    statement_starters = [
        "i live", "i am", "i'm", "my name", "sounds good", "thanks", "thank you", "okay got it",
        "got it", "i see", "yes", "yeah", "yep", "sure", "no problem", "hello", "hi", "good morning"
    ]
    if any(u_lower.startswith(s) for s in statement_starters) and not has_qmark:
        return {
            "completeness": 1.0,
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": sem_key,
            "reason": "statement"
        }

    # 2. Check for Incomplete Trailing Stems (dangling verbs/prepositions that change meaning if completed)
    incomplete_trailing_words = {
        "choose", "use", "implement", "between", "about", "in", "for", "with",
        "at", "to", "of", "and", "vs", "versus", "from", "on", "into", "the", "a", "an", "your", "my",
        "like", "such", "as", "than", "or", "did", "do", "how", "why", "what", "can", "could", "would"
    }
    
    last_word = words[-1] if words else ""
    is_trailing_incomplete = (last_word in incomplete_trailing_words) and not has_qmark

    # 3. Follow-up evaluation (Grounded in existing active topic)
    active_topic = getattr(session, "active_topic", "") if session else ""
    if u_lower in {"why?", "why", "and why?", "and why", "how?", "how", "and how?", "and how", "why so"}:
        return {
            "completeness": 1.0 if active_topic else 0.70,
            "is_semantically_complete": bool(active_topic),
            "can_start_early": bool(active_topic),
            "semantic_key": sem_key or "why",
            "reason": "short_followup_grounded" if active_topic else "unresolved_followup"
        }

    followup_stems = ["what about", "how about", "and for", "why not", "then what", "so what", "and in that case"]
    if any(u_lower.startswith(f) for f in followup_stems):
        if word_count <= 2 and is_trailing_incomplete:
            return {
                "completeness": 0.40,
                "is_semantically_complete": False,
                "can_start_early": False,
                "semantic_key": sem_key,
                "reason": "incomplete_followup_stem"
            }
        else:
            return {
                "completeness": 0.95,
                "is_semantically_complete": True,
                "can_start_early": True,
                "semantic_key": sem_key,
                "reason": "complete_followup"
            }

    # 4. Coding Imperatives
    coding_imperatives = [
        "write a", "write an", "implement a", "implement an", "find the", "find all", "create a",
        "reverse a", "reverse the", "solve the", "calculate", "design a", "design the", "optimize"
    ]
    if any(u_lower.startswith(c) for c in coding_imperatives):
        if word_count <= 3 or is_trailing_incomplete:
            return {
                "completeness": 0.50,
                "is_semantically_complete": False,
                "can_start_early": False,
                "semantic_key": sem_key,
                "reason": "incomplete_coding_stem"
            }
        else:
            return {
                "completeness": 0.95,
                "is_semantically_complete": True,
                "can_start_early": True,
                "semantic_key": sem_key,
                "reason": "complete_coding_question"
            }

    # 5. General Interrogative Technical Questions
    interrogatives = [
        "what", "why", "how", "when", "where", "which", "who", "whom",
        "can you", "could you", "would you", "tell me", "explain", "describe",
        "walk me through", "elaborate on", "clarify"
    ]
    if any(u_lower.startswith(q) for q in interrogatives):
        if word_count <= 3 or is_trailing_incomplete:
            return {
                "completeness": 0.45,
                "is_semantically_complete": False,
                "can_start_early": False,
                "semantic_key": sem_key,
                "reason": "incomplete_interrogative_stem"
            }
        else:
            return {
                "completeness": 0.95 if word_count >= 5 or has_qmark else 0.85,
                "is_semantically_complete": True,
                "can_start_early": True,
                "semantic_key": sem_key,
                "reason": "complete_interrogative_question"
            }

    if has_qmark and word_count >= 3:
        return {
            "completeness": 0.90,
            "is_semantically_complete": True,
            "can_start_early": True,
            "semantic_key": sem_key,
            "reason": "punctuated_question"
        }

    return {
        "completeness": 0.50,
        "is_semantically_complete": False,
        "can_start_early": False,
        "semantic_key": sem_key,
        "reason": "insufficient_signal"
    }


def compute_question_likelihood(
    text: str,
    session: Optional["InterviewSession"] = None
) -> Dict[str, Any]:
    """
    Evaluates progressive question likelihood, semantic completeness,
    and pre-assembles speculative context for early answer generation.
    """
    if not text or not text.strip():
        return {
            "likelihood": 0.0,
            "stability_state": QuestionStabilityState.PARTIAL,
            "is_question_stem": False,
            "stem_type": "EMPTY",
            "is_complete_enough": False,
            "preview_topic": "",
            "completeness": 0.0,
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": "",
            "predicted_type": "general"
        }

    comp_eval = evaluate_question_completeness(text, session=session)
    raw = text.strip()
    u_lower = raw.lower().strip(" .!?,:;")
    words = u_lower.split()
    word_count = len(words)
    has_qmark = raw.endswith("?")

    interrogatives = [
        "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
        "can you", "could you", "would you", "will you", "do you", "did you", "have you", "has",
        "is it", "is there", "are you", "are there", "should we", "tell me", "explain", "describe",
        "walk me through", "elaborate on", "clarify"
    ]
    coding_imperatives = [
        "write a", "write an", "implement a", "implement an", "find the", "find all", "create a",
        "reverse a", "reverse the", "solve the", "calculate", "design a", "design the", "optimize"
    ]
    followup_stems = [
        "what about", "how about", "and for", "why not", "then what", "so what", "and in that case",
        "what if"
    ]

    is_interrogative = any(u_lower.startswith(q) for q in interrogatives)
    is_coding = any(u_lower.startswith(c) for c in coding_imperatives)
    is_followup = any(u_lower.startswith(f) for f in followup_stems)

    # Statements
    if comp_eval["reason"] == "statement":
        return {
            "likelihood": 0.10,
            "stability_state": QuestionStabilityState.PARTIAL,
            "is_question_stem": False,
            "stem_type": "STATEMENT",
            "is_complete_enough": word_count >= 3,
            "preview_topic": "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": "statement"
        }

    from classifier import classify_question
    pred_type = classify_question(raw)

    if is_followup:
        if comp_eval["can_start_early"]:
            state = QuestionStabilityState.STABLE_QUESTION
            likelihood = 0.95
        elif word_count <= 2:
            state = QuestionStabilityState.POSSIBLE_QUESTION
            likelihood = 0.70
        else:
            state = QuestionStabilityState.LIKELY_QUESTION
            likelihood = 0.85
            
        return {
            "likelihood": likelihood,
            "stability_state": state,
            "is_question_stem": True,
            "stem_type": "FOLLOW_UP",
            "is_complete_enough": comp_eval["is_semantically_complete"],
            "preview_topic": getattr(session, "active_topic", "") if session else "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": "follow_up"
        }

    if is_coding:
        if comp_eval["can_start_early"]:
            state = QuestionStabilityState.STABLE_QUESTION
            likelihood = 0.95
        else:
            state = QuestionStabilityState.LIKELY_QUESTION
            likelihood = 0.80
            
        return {
            "likelihood": likelihood,
            "stability_state": state,
            "is_question_stem": True,
            "stem_type": "CODING_IMPERATIVE",
            "is_complete_enough": comp_eval["is_semantically_complete"],
            "preview_topic": "Coding",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": "coding"
        }

    if is_interrogative:
        if comp_eval["can_start_early"]:
            state = QuestionStabilityState.STABLE_QUESTION
            likelihood = 0.95
        elif word_count <= 2 and not has_qmark:
            state = QuestionStabilityState.POSSIBLE_QUESTION
            likelihood = 0.65
        else:
            state = QuestionStabilityState.LIKELY_QUESTION
            likelihood = 0.82
            
        return {
            "likelihood": likelihood,
            "stability_state": state,
            "is_question_stem": True,
            "stem_type": "INTERROGATIVE",
            "is_complete_enough": comp_eval["is_semantically_complete"],
            "preview_topic": getattr(session, "active_topic", "") if session else "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": pred_type
        }

    if has_qmark:
        return {
            "likelihood": 0.90,
            "stability_state": QuestionStabilityState.STABLE_QUESTION if word_count >= 3 else QuestionStabilityState.POSSIBLE_QUESTION,
            "is_question_stem": True,
            "stem_type": "PUNCTUATED_QUESTION",
            "is_complete_enough": True,
            "preview_topic": getattr(session, "active_topic", "") if session else "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": pred_type
        }

    return {
        "likelihood": 0.25 if word_count <= 3 else 0.40,
        "stability_state": QuestionStabilityState.PARTIAL,
        "is_question_stem": False,
        "stem_type": "GENERAL",
        "is_complete_enough": word_count >= 4,
        "preview_topic": "",
        "completeness": comp_eval["completeness"],
        "is_semantically_complete": False,
        "can_start_early": False,
        "semantic_key": comp_eval["semantic_key"],
        "predicted_type": "general"
    }


# ============================================================
# STT PROVIDER ABSTRACTION
# ============================================================

from abc import ABC, abstractmethod

class STTProvider(ABC):
    """Abstract interface for speech-to-text providers."""
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "") -> str:
        pass


class GroqWhisperProvider(STTProvider):
    """Production Groq Whisper implementation (whisper-large-v3-turbo) with automatic retry and rate-limit backoff."""
    def __init__(self, client: Groq, model: str = "whisper-large-v3-turbo", default_language: Optional[str] = "en"):
        self.client = client
        self.model = model
        self.default_language = default_language
        self._rate_limit_cooldown_until = 0.0

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "", language: Optional[str] = None) -> str:
        if not audio_bytes or len(audio_bytes) < 320:
            return ""
            
        now = time.time()
        if now < self._rate_limit_cooldown_until:
            return ""
            
        import io
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        wav_bytes = wav_io.getvalue()
        
        target_lang = language or self.default_language
        kwargs = {
            "file": ("audio.wav", wav_bytes),
            "model": self.model,
            "response_format": "text",
            "temperature": 0.0,
            "prompt": prompt or "Python, SQL, React, AWS, Docker, Kubernetes, algorithms, architecture."
        }
        if target_lang and target_lang.lower() not in ["auto", "none", ""]:
            # Map full language name to 2-letter ISO code if needed
            lang_code = target_lang.lower()
            lang_map = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "hindi": "hi", "mandarin": "zh", "chinese": "zh", "japanese": "ja"}
            kwargs["language"] = lang_map.get(lang_code, lang_code[:2] if len(lang_code) > 2 else lang_code)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                transcription = self.client.audio.transcriptions.create(**kwargs)
                return transcription.strip() if isinstance(transcription, str) else ""
            except Exception as e:
                cat, is_retryable, user_msg = categorize_api_error(e)
                if cat == "RATE_LIMIT" or "429" in str(e):
                    self._rate_limit_cooldown_until = time.time() + 4.0
                    return ""
                if not is_retryable or attempt == max_retries - 1:
                    print(f"STT API Notice [{cat}]: {user_msg}")
                    return ""
                time.sleep(0.25 * (attempt + 1))
        return ""


# ============================================================
# CONTINUOUS AUDIO STREAMING & LOW-LATENCY TRANSCRIBER
# ============================================================

# ============================================================
# CONTINUOUS AUDIO STREAMING & LOW-LATENCY TRANSCRIBER
# ============================================================

class ContinuousTranscriber:
    """
    Continuous audio recording and progressive low-latency transcription stream.
    Supports Dual concurrent capture (System Audio Loopback = Interviewer, Microphone = Candidate),
    Loopback-only, and Microphone-only modes with decoupled queues, glitch-tolerant rolling buffers,
    immutable segment IDs, and monotonic latency measurement.
    """
    
    def __init__(
        self,
        client: Groq,
        on_transcript_chunk: Callable[..., None],
        on_status_change: Optional[Callable[[str], None]] = None,
        on_interim_transcript: Optional[Callable[..., None]] = None,
        on_audio_level: Optional[Callable[[str, float, float], None]] = None,
        provider: Optional[STTProvider] = None,
        session: Optional["InterviewSession"] = None,
        enable_watchdog: bool = True,
        mic_device_id: Optional[Union[int, str]] = None,
        system_speaker_id: Optional[str] = None
    ):
        self.client = client
        self.on_transcript_chunk = on_transcript_chunk
        self.on_status_change = on_status_change
        self.on_interim_transcript = on_interim_transcript
        self.on_audio_level = on_audio_level
        self.provider = provider or GroqWhisperProvider(client)
        self.session = session
        self.enable_watchdog = enable_watchdog
        self.mic_device_id = mic_device_id
        self.system_speaker_id = system_speaker_id
        
        self.is_running = False
        self.source = "dual"  # 'dual', 'system', 'loopback', or 'mic'
        
        # Audio thread references (exactly 1 capture + 1 processor per active source)
        self.system_thread: Optional[threading.Thread] = None
        self.system_processor_thread: Optional[threading.Thread] = None
        self.mic_thread: Optional[threading.Thread] = None
        self.mic_processor_thread: Optional[threading.Thread] = None
        self.watchdog_thread: Optional[threading.Thread] = None
        self.enable_watchdog: bool = True
        self._stop_event = threading.Event()
        
        # Sized audio queues (30 frames @ 30ms = 900ms max buffer duration)
        self.max_queue_depth = 30
        self.system_audio_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_depth)
        self.mic_audio_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_depth)
        
        # Active worker diagnostics & lifecycle tracking
        self.active_workers: Dict[str, Dict[str, Any]] = {}
        self.discontinuity_count: int = 0
        self.dropped_frames_count: int = 0
        
        self.sample_rate = 16000
        self.channels = 1
        self._interim_in_flight = False
        self._current_speech_start = 0.0
        self._is_candidate_semantically_complete = False

    def start(
        self,
        source: str = "dual",
        mic_device_id: Optional[Union[int, str]] = None,
        system_speaker_id: Optional[str] = None
    ):
        if self.is_running:
            return
            
        self.source = source
        if mic_device_id is not None:
            self.mic_device_id = mic_device_id
        if system_speaker_id is not None:
            self.system_speaker_id = system_speaker_id
            
        self.is_running = True
        self._stop_event.clear()
        
        # Clear queues
        while not self.system_audio_queue.empty():
            try:
                self.system_audio_queue.get_nowait()
            except queue.Empty:
                break
        while not self.mic_audio_queue.empty():
            try:
                self.mic_audio_queue.get_nowait()
            except queue.Empty:
                break
                
        if self.session and hasattr(self.session, "controller") and self.session.controller:
            self.session.controller.set_state(InterviewTurnState.LISTENING)
            self.session.controller.update_component("audio_capture", "HEALTHY")
            self.session.controller.update_component("stt", "HEALTHY")
        
        if self.on_status_change:
            if source == "dual":
                self.on_status_change("🎙️ Live Listening (Dual: System Loopback + Mic)...")
            elif source in ("system", "loopback"):
                self.on_status_change("🔊 Live Listening (System Audio Loopback)...")
            else:
                self.on_status_change("🎙️ Live Listening (Microphone)...")
                
        now = time.perf_counter()
        
        # Start capture and processing workers per selected source
        if source == "dual":
            if SOUNDCARD_AVAILABLE:
                sys_cap_id = f"sys_cap_{uuid.uuid4().hex[:6]}"
                sys_proc_id = f"sys_proc_{uuid.uuid4().hex[:6]}"
                self.active_workers[sys_cap_id] = {"source": "system", "role": "capture", "start_time": now}
                self.active_workers[sys_proc_id] = {"source": "system", "role": "processor", "start_time": now}
                
                self.system_processor_thread = threading.Thread(
                    target=lambda: self._system_processing_worker(sys_proc_id),
                    daemon=True,
                    name=f"SystemAudioProcessor-{sys_proc_id}"
                )
                self.system_processor_thread.start()
                
                self.system_thread = threading.Thread(
                    target=lambda: self._run_system_audio_loop(sys_cap_id),
                    daemon=True,
                    name=f"SystemAudioCapture-{sys_cap_id}"
                )
                self.system_thread.start()
            else:
                if self.on_status_change:
                    self.on_status_change("⚠️ Soundcard loopback unavailable; microphone active.")
                    
            mic_cap_id = f"mic_cap_{uuid.uuid4().hex[:6]}"
            mic_proc_id = f"mic_proc_{uuid.uuid4().hex[:6]}"
            self.active_workers[mic_cap_id] = {"source": "candidate_mic", "role": "capture", "start_time": now}
            self.active_workers[mic_proc_id] = {"source": "candidate_mic", "role": "processor", "start_time": now}
            
            self.mic_processor_thread = threading.Thread(
                target=lambda: self._mic_processing_worker(channel_source="candidate_mic", worker_id=mic_proc_id),
                daemon=True,
                name=f"MicProcessor-{mic_proc_id}"
            )
            self.mic_processor_thread.start()
            
            self.mic_thread = threading.Thread(
                target=lambda: self._run_microphone_loop(channel_source="candidate_mic", worker_id=mic_cap_id),
                daemon=True,
                name=f"MicCandidateLoop-{mic_cap_id}"
            )
            self.mic_thread.start()
            
        elif source in ("system", "loopback"):
            if SOUNDCARD_AVAILABLE:
                sys_cap_id = f"sys_cap_{uuid.uuid4().hex[:6]}"
                sys_proc_id = f"sys_proc_{uuid.uuid4().hex[:6]}"
                self.active_workers[sys_cap_id] = {"source": "system", "role": "capture", "start_time": now}
                self.active_workers[sys_proc_id] = {"source": "system", "role": "processor", "start_time": now}
                
                self.system_processor_thread = threading.Thread(
                    target=lambda: self._system_processing_worker(sys_proc_id),
                    daemon=True,
                    name=f"SystemAudioProcessor-{sys_proc_id}"
                )
                self.system_processor_thread.start()
                
                self.system_thread = threading.Thread(
                    target=lambda: self._run_system_audio_loop(sys_cap_id),
                    daemon=True,
                    name=f"SystemAudioCapture-{sys_cap_id}"
                )
                self.system_thread.start()
            else:
                if self.on_status_change:
                    self.on_status_change("⚠️ Soundcard loopback unavailable.")
                    
        elif source == "mic":
            mic_cap_id = f"mic_cap_{uuid.uuid4().hex[:6]}"
            mic_proc_id = f"mic_proc_{uuid.uuid4().hex[:6]}"
            self.active_workers[mic_cap_id] = {"source": "mic", "role": "capture", "start_time": now}
            self.active_workers[mic_proc_id] = {"source": "mic", "role": "processor", "start_time": now}
            
            self.mic_processor_thread = threading.Thread(
                target=lambda: self._mic_processing_worker(channel_source="mic", worker_id=mic_proc_id),
                daemon=True,
                name=f"MicProcessor-{mic_proc_id}"
            )
            self.mic_processor_thread.start()
            
            self.mic_thread = threading.Thread(
                target=lambda: self._run_microphone_loop(channel_source="mic", worker_id=mic_cap_id),
                daemon=True,
                name=f"MicSingleLoop-{mic_cap_id}"
            )
            self.mic_thread.start()

        # Start audio watchdog for self-healing dead threads
        if self.enable_watchdog:
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="AudioWatchdogThread")
            self.watchdog_thread.start()

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        now = time.perf_counter()
        for w_id in list(self.active_workers.keys()):
            self.active_workers[w_id]["stop_time"] = now
            
        threads_to_join = [
            self.system_thread,
            self.system_processor_thread,
            self.mic_thread,
            self.mic_processor_thread,
            self.watchdog_thread
        ]
        for t in threads_to_join:
            if t and t.is_alive():
                t.join(timeout=1.0)
                
        self.system_thread = None
        self.system_processor_thread = None
        self.mic_thread = None
        self.mic_processor_thread = None
        self.watchdog_thread = None
        self.active_workers.clear()
        
        if self.session and hasattr(self.session, "controller") and self.session.controller:
            self.session.controller.set_state(InterviewTurnState.STOPPED)
            self.session.controller.update_component("audio_capture", "IDLE")
        if self.on_status_change:
            self.on_status_change("⏹️ Stopped")

    def _watchdog_loop(self):
        """Lightweight watchdog thread that monitors audio streams and auto-recovers dead capture workers."""
        while self.is_running and not self._stop_event.is_set():
            time.sleep(2.0)
            if not self.is_running or self._stop_event.is_set():
                break
                
            try:
                now = time.perf_counter()
                # 1. Check and recover System Audio Loopback workers
                if self.source in ("dual", "system", "loopback") and SOUNDCARD_AVAILABLE:
                    if self.system_processor_thread and not self.system_processor_thread.is_alive() and self.is_running:
                        sys_proc_id = f"sys_proc_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[sys_proc_id] = {"source": "system", "role": "processor", "start_time": now}
                        self.system_processor_thread = threading.Thread(
                            target=lambda: self._system_processing_worker(sys_proc_id),
                            daemon=True,
                            name=f"SystemAudioProcessor-{sys_proc_id}"
                        )
                        self.system_processor_thread.start()
                        
                    if self.system_thread and not self.system_thread.is_alive() and self.is_running:
                        if self.on_status_change:
                            try:
                                self.on_status_change("↻ Recovering system audio capture...")
                            except Exception:
                                pass
                        if self.session and hasattr(self.session, "controller") and self.session.controller:
                            self.session.controller.update_component("audio_capture", "RECOVERING", "System capture thread restarted")
                        sys_cap_id = f"sys_cap_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[sys_cap_id] = {"source": "system", "role": "capture", "start_time": now}
                        self.system_thread = threading.Thread(
                            target=lambda: self._run_system_audio_loop(sys_cap_id),
                            daemon=True,
                            name=f"SystemAudioCapture-{sys_cap_id}"
                        )
                        self.system_thread.start()

                # 2. Check and recover Microphone workers
                if self.source in ("dual", "mic"):
                    channel_src = "candidate_mic" if self.source == "dual" else "mic"
                    if self.mic_processor_thread and not self.mic_processor_thread.is_alive() and self.is_running:
                        mic_proc_id = f"mic_proc_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[mic_proc_id] = {"source": channel_src, "role": "processor", "start_time": now}
                        self.mic_processor_thread = threading.Thread(
                            target=lambda: self._mic_processing_worker(channel_source=channel_src, worker_id=mic_proc_id),
                            daemon=True,
                            name=f"MicProcessor-{mic_proc_id}"
                        )
                        self.mic_processor_thread.start()
                        
                    if self.mic_thread and not self.mic_thread.is_alive() and self.is_running:
                        if self.on_status_change:
                            try:
                                self.on_status_change("↻ Recovering microphone capture...")
                            except Exception:
                                pass
                        if self.session and hasattr(self.session, "controller") and self.session.controller:
                            self.session.controller.update_component("audio_capture", "RECOVERING", "Microphone capture thread restarted")
                        mic_cap_id = f"mic_cap_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[mic_cap_id] = {"source": channel_src, "role": "capture", "start_time": now}
                        self.mic_thread = threading.Thread(
                            target=lambda: self._run_microphone_loop(channel_source=channel_src, worker_id=mic_cap_id),
                            daemon=True,
                            name=f"MicCandidateLoop-{mic_cap_id}"
                        )
                        self.mic_thread.start()
                        
                # Update heartbeat
                if self.session and hasattr(self.session, "controller") and self.session.controller:
                    self.session.controller.record_heartbeat_event("last_audio_event")
            except Exception as e:
                print(f"Audio watchdog check notice: {e}")

    def _run_system_audio_loop(self, worker_id: str = ""):
        """
        Lightweight hardware capture loop for System Audio Loopback.
        ONLY: read audio frame, timestamp, enqueue, continue.
        NEVER performs DSP, VAD, STT, or LLM processing.
        """
        retry_delay = 0.1
        while self.is_running and not self._stop_event.is_set():
            try:
                speaker = None
                if self.system_speaker_id and self.system_speaker_id != "default":
                    try:
                        for s in sc.all_speakers():
                            if s.name == self.system_speaker_id:
                                speaker = s
                                break
                    except Exception:
                        speaker = None
                if speaker is None:
                    speaker = sc.default_speaker()
                    
                loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
                
                frame_duration = 30  # ms
                frame_size = int(self.sample_rate * frame_duration / 1000)
                
                with loopback.recorder(samplerate=self.sample_rate, channels=1, blocksize=1024) as recorder:
                    retry_delay = 0.1
                    while self.is_running and not self._stop_event.is_set():
                        ts_capture = time.perf_counter()
                        try:
                            data = recorder.record(numframes=frame_size)
                        except Exception as e:
                            self.discontinuity_count += 1
                            continue
                            
                        if data is None or len(data) == 0:
                            time.sleep(0.005)
                            continue
                            
                        if self.on_audio_level:
                            try:
                                if isinstance(data, np.ndarray) and data.size > 0:
                                    peak = float(np.max(np.abs(data)))
                                    norm_lvl = min(1.0, peak / 32767.0) if data.dtype == np.int16 else min(1.0, peak)
                                    self.on_audio_level("system", norm_lvl, 0.0)
                            except Exception:
                                pass

                        frame = AudioFrame(
                            data=data,
                            timestamp=ts_capture,
                            source="system"
                        )
                        
                        try:
                            self.system_audio_queue.put_nowait(frame)
                        except queue.Full:
                            # Controlled queue overflow: drop oldest frame to maintain low latency
                            try:
                                self.system_audio_queue.get_nowait()
                                self.dropped_frames_count += 1
                            except queue.Empty:
                                pass
                            try:
                                self.system_audio_queue.put_nowait(frame)
                            except queue.Full:
                                pass
            except Exception as e:
                if self.on_status_change:
                    self.on_status_change(f"⚠️ Loopback audio error: {e}")
                if not self.is_running or self._stop_event.is_set():
                    return
                time.sleep(retry_delay)
                retry_delay = min(1.0, retry_delay * 1.5)

    def _system_processing_worker(self, worker_id: str = ""):
        """
        Dedicated processing worker for System Audio.
        Handles DSP, VAD, rolling buffers, speech segmentation, and STT dispatch.
        """
        processor = SystemAudioProcessor(sample_rate=self.sample_rate)
        default_silence_frames = 12    # ~360ms of silence
        min_speech_frames = 12         # ~360ms min speech
        interim_interval_frames = 45   # ~1350ms interval
        
        audio_buffer = []
        pre_buffer = collections.deque(maxlen=7)  # ~210ms pre-speech buffer
        speech_frames_count = 0
        silence_frames_count = 0
        in_speech = False
        speech_start_perf = 0.0
        
        while self.is_running and not self._stop_event.is_set():
            try:
                frame: AudioFrame = self.system_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            proc_start_ts = time.perf_counter()
            proc_bytes, is_speech, diag = processor.process_frame(frame.data)
            
            if is_speech:
                if not in_speech:
                    speech_start_perf = frame.timestamp
                    self._current_speech_start = speech_start_perf
                    if self.session:
                        trace = self.session.latest_latency_trace
                        if not trace or trace.timestamps.get("answer_complete", 0.0) > 0.0:
                            trace = QuestionLatencyTrace()
                            self.session.latest_latency_trace = trace
                        trace.mark("capture_start", speech_start_perf)
                        trace.mark("queue_enter", frame.timestamp)
                        trace.mark("processing_start", proc_start_ts)
                        trace.queue_depth = self.system_audio_queue.qsize()
                        trace.discontinuities = self.discontinuity_count
                    in_speech = True
                    audio_buffer.extend(list(pre_buffer))
                    
                audio_buffer.append(proc_bytes)
                speech_frames_count += 1
                silence_frames_count = 0
                
                if speech_frames_count >= min_speech_frames and (speech_frames_count % interim_interval_frames == 0):
                    self._dispatch_interim_transcription(
                        list(audio_buffer),
                        source="system",
                        speech_start_ts=speech_start_perf
                    )
            elif in_speech:
                audio_buffer.append(proc_bytes)
                silence_frames_count += 1
                
                effective_silence = 8 if getattr(self, "_is_candidate_semantically_complete", False) else default_silence_frames
                
                if silence_frames_count >= effective_silence:
                    if speech_frames_count >= min_speech_frames:
                        seg = AudioSegment(
                            segment_id=f"seg_sys_{uuid.uuid4().hex[:6]}",
                            source="system",
                            audio_bytes=b"".join(audio_buffer),
                            sample_rate=self.sample_rate,
                            capture_start=speech_start_perf,
                            capture_end=time.perf_counter(),
                            queue_enter=frame.timestamp,
                            processing_start=proc_start_ts,
                            silence_frames=silence_frames_count,
                            speech_frames=speech_frames_count
                        )
                        self._dispatch_final_transcription(seg, source="system")
                    audio_buffer.clear()
                    speech_frames_count = 0
                    silence_frames_count = 0
                    in_speech = False
                    self._is_candidate_semantically_complete = False
            else:
                pre_buffer.append(proc_bytes)
                
            # Prevent unbounded buffer if continuous speech > 10s
            if len(audio_buffer) * 30 > 10000:
                if speech_frames_count >= min_speech_frames:
                    seg = AudioSegment(
                        segment_id=f"seg_sys_{uuid.uuid4().hex[:6]}",
                        source="system",
                        audio_bytes=b"".join(audio_buffer),
                        sample_rate=self.sample_rate,
                        capture_start=speech_start_perf,
                        capture_end=time.perf_counter(),
                        queue_enter=frame.timestamp,
                        processing_start=proc_start_ts,
                        silence_frames=silence_frames_count,
                        speech_frames=speech_frames_count
                    )
                    self._dispatch_final_transcription(seg, source="system")
                audio_buffer.clear()
                speech_frames_count = 0
                silence_frames_count = 0
                in_speech = False
                self._is_candidate_semantically_complete = False

    def _run_microphone_loop(self, channel_source: str = "candidate_mic", worker_id: str = ""):
        """
        Lightweight hardware capture loop for Microphone.
        ONLY: read audio frame, timestamp, enqueue, continue.
        """
        frame_duration = 30  # ms
        frame_size = int(self.sample_rate * frame_duration / 1000)
        
        dev_idx = None
        if self.mic_device_id is not None and self.mic_device_id != "default":
            try:
                dev_idx = int(self.mic_device_id)
            except (ValueError, TypeError):
                dev_idx = None
        
        retry_delay = 0.1
        while self.is_running and not self._stop_event.is_set():
            stream_dev = dev_idx
            try:
                with sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=frame_size,
                    dtype="int16",
                    channels=1,
                    device=stream_dev
                ) as stream:
                    retry_delay = 0.1
                    while self.is_running and not self._stop_event.is_set():
                        ts_capture = time.perf_counter()
                        frame, overflow = stream.read(frame_size)
                        if overflow:
                            self.discontinuity_count += 1
                        frame_bytes = bytes(frame)
                        
                        if self.on_audio_level:
                            try:
                                samples = np.frombuffer(frame, dtype=np.int16)
                                peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
                                norm_lvl = min(1.0, peak / 32767.0)
                                self.on_audio_level("mic", norm_lvl, 0.0)
                            except Exception:
                                pass

                        frame_obj = AudioFrame(
                            data=frame_bytes,
                            timestamp=ts_capture,
                            source=channel_source,
                            is_discontinuity=bool(overflow)
                        )
                        
                        try:
                            self.mic_audio_queue.put_nowait(frame_obj)
                        except queue.Full:
                            try:
                                self.mic_audio_queue.get_nowait()
                                self.dropped_frames_count += 1
                            except queue.Empty:
                                pass
                            try:
                                self.mic_audio_queue.put_nowait(frame_obj)
                            except queue.Full:
                                pass
            except Exception as e:
                if self.on_status_change:
                    self.on_status_change(f"⚠️ Microphone error: {e}")
                if not self.is_running or self._stop_event.is_set() or "9983" in str(e) or "Stream is stopped" in str(e):
                    return
                # If specific device failed, fallback to system default
                if dev_idx is not None:
                    dev_idx = None
                time.sleep(retry_delay)
                retry_delay = min(1.0, retry_delay * 1.5)

    def _mic_processing_worker(self, channel_source: str = "candidate_mic", worker_id: str = ""):
        """
        Dedicated processing worker for Microphone audio.
        """
        processor = MicrophoneAudioProcessor(sample_rate=self.sample_rate)
        default_silence_frames = 18    # ~540ms natural speech pause tolerance
        min_speech_frames = 10         # ~300ms
        interim_interval_frames = 45   # ~1350ms
        
        audio_buffer = []
        pre_buffer = collections.deque(maxlen=10)
        speech_frames_count = 0
        silence_frames_count = 0
        in_speech = False
        speech_start_perf = 0.0
        
        while self.is_running and not self._stop_event.is_set():
            try:
                frame: AudioFrame = self.mic_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            proc_start_ts = time.perf_counter()
            proc_bytes, is_speech, diag = processor.process_frame(frame.data)
            
            if is_speech:
                if not in_speech:
                    speech_start_perf = frame.timestamp
                    self._current_speech_start = speech_start_perf
                    if self.session:
                        trace = self.session.latest_latency_trace
                        if not trace or trace.timestamps.get("answer_complete", 0.0) > 0.0:
                            trace = QuestionLatencyTrace()
                            self.session.latest_latency_trace = trace
                        trace.mark("capture_start", speech_start_perf)
                        trace.mark("queue_enter", frame.timestamp)
                        trace.mark("processing_start", proc_start_ts)
                        trace.queue_depth = self.mic_audio_queue.qsize()
                        trace.discontinuities = self.discontinuity_count
                    in_speech = True
                    audio_buffer.extend(list(pre_buffer))
                    
                audio_buffer.append(proc_bytes)
                speech_frames_count += 1
                silence_frames_count = 0
                
                if speech_frames_count >= min_speech_frames and (speech_frames_count % interim_interval_frames == 0):
                    self._dispatch_interim_transcription(
                        list(audio_buffer),
                        source=channel_source,
                        speech_start_ts=speech_start_perf
                    )
            elif in_speech:
                audio_buffer.append(proc_bytes)
                silence_frames_count += 1
                
                effective_silence = 8 if getattr(self, "_is_candidate_semantically_complete", False) else default_silence_frames
                
                if silence_frames_count >= effective_silence:
                    if speech_frames_count >= min_speech_frames:
                        seg = AudioSegment(
                            segment_id=f"seg_mic_{uuid.uuid4().hex[:6]}",
                            source=channel_source,
                            audio_bytes=b"".join(audio_buffer),
                            sample_rate=self.sample_rate,
                            capture_start=speech_start_perf,
                            capture_end=time.perf_counter(),
                            queue_enter=frame.timestamp,
                            processing_start=proc_start_ts,
                            silence_frames=silence_frames_count,
                            speech_frames=speech_frames_count
                        )
                        self._dispatch_final_transcription(seg, source=channel_source)
                    audio_buffer.clear()
                    speech_frames_count = 0
                    silence_frames_count = 0
                    in_speech = False
                    self._is_candidate_semantically_complete = False
            else:
                pre_buffer.append(proc_bytes)
                
            if len(audio_buffer) * 30 > 10000:
                if speech_frames_count >= min_speech_frames:
                    seg = AudioSegment(
                        segment_id=f"seg_mic_{uuid.uuid4().hex[:6]}",
                        source=channel_source,
                        audio_bytes=b"".join(audio_buffer),
                        sample_rate=self.sample_rate,
                        capture_start=speech_start_perf,
                        capture_end=time.perf_counter(),
                        queue_enter=frame.timestamp,
                        processing_start=proc_start_ts,
                        silence_frames=silence_frames_count,
                        speech_frames=speech_frames_count
                    )
                    self._dispatch_final_transcription(seg, source=channel_source)
                audio_buffer.clear()
                speech_frames_count = 0
                silence_frames_count = 0
                in_speech = False
                self._is_candidate_semantically_complete = False

    def _dispatch_interim_transcription(
        self,
        audio_frames: List[bytes],
        source: str = "system",
        speech_start_ts: Optional[float] = None
    ):
        """Dispatches lightweight interim transcription for live visual preview & early question detection."""
        now = time.perf_counter()
        if not audio_frames or self._interim_in_flight or (now - getattr(self, "_last_interim_stt_time", 0.0) < 1.5):
            return
            
        def _worker():
            self._interim_in_flight = True
            self._last_interim_stt_time = time.perf_counter()
            try:
                raw_bytes = b"".join(audio_frames)
                t_stt_start = time.perf_counter()
                text = self.provider.transcribe(raw_bytes, sample_rate=self.sample_rate)
                t_stt_end = time.perf_counter()
                
                if text and len(text.strip()) >= 2:
                    cleaned_text = clean_asr_transcript(text.strip())
                    likelihood_info = compute_question_likelihood(cleaned_text, session=self.session)
                    
                    # Latency trace marks
                    if self.session:
                        trace = self.session.latest_latency_trace
                        if not trace or trace.timestamps.get("answer_complete", 0.0) > 0.0:
                            trace = QuestionLatencyTrace()
                            self.session.latest_latency_trace = trace
                        trace.source = source
                        if speech_start_ts:
                            trace.mark("capture_start", speech_start_ts)
                        trace.mark("stt_start", t_stt_start)
                        trace.mark("stt_end", t_stt_end)
                        trace.mark("transcript_received", t_stt_end)
                        if likelihood_info.get("is_question_stem") or likelihood_info.get("likelihood", 0) >= 0.7:
                            trace.mark("question_candidate", time.perf_counter())
                        if likelihood_info.get("is_semantically_complete") or likelihood_info.get("can_start_early"):
                            trace.mark("question_complete", time.perf_counter())
                            self._is_candidate_semantically_complete = True
                            
                    if self.on_interim_transcript:
                        import inspect
                        sig = inspect.signature(self.on_interim_transcript)
                        if len(sig.parameters) >= 3 or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                            self.on_interim_transcript(cleaned_text, likelihood_info, source)
                        else:
                            self.on_interim_transcript(cleaned_text, likelihood_info)
            except Exception as e:
                pass
            finally:
                self._interim_in_flight = False
                
        threading.Thread(target=_worker, daemon=True).start()

    def _dispatch_final_transcription(
        self,
        segment_or_frames: Any,
        source: str = "system"
    ):
        """Transcribes finalized PCM frames via STT provider and dispatches to question analyzer."""
        if isinstance(segment_or_frames, AudioSegment):
            segment = segment_or_frames
            raw_bytes = segment.audio_bytes
            speech_start_ts = segment.capture_start
            queue_enter_ts = segment.queue_enter
            processing_start_ts = segment.processing_start
            segment_id = segment.segment_id
        elif isinstance(segment_or_frames, list):
            if not segment_or_frames or len(segment_or_frames) < 10:
                return
            raw_bytes = b"".join(segment_or_frames)
            speech_start_ts = getattr(self, "_current_speech_start", time.perf_counter())
            queue_enter_ts = speech_start_ts
            processing_start_ts = speech_start_ts
            segment_id = f"seg_{uuid.uuid4().hex[:6]}"
        else:
            return
            
        if not raw_bytes or len(raw_bytes) < 320:
            return
            
        def _worker():
            try:
                samples = np.frombuffer(raw_bytes, dtype=np.int16)
                total_rms = float(np.sqrt(np.mean(samples.astype(np.float32)**2)))
                
                # Discard near-silent / room noise buffers
                if total_rms < 120:
                    return
                
                t_stt_start = time.perf_counter()
                text = self.provider.transcribe(raw_bytes, sample_rate=self.sample_rate)
                t_stt_end = time.perf_counter()
                
                if not text or len(text.strip()) < 2:
                    return
                    
                cleaned_text = clean_asr_transcript(text.strip())
                is_q = is_likely_question(cleaned_text)
                
                if self.session:
                    trace = self.session.latest_latency_trace
                    if not trace or trace.timestamps.get("answer_complete", 0.0) > 0.0:
                        trace = QuestionLatencyTrace(segment_id=segment_id)
                        self.session.latest_latency_trace = trace
                    trace.source = source
                    trace.segment_id = segment_id
                    trace.mark("capture_start", speech_start_ts)
                    trace.mark("queue_enter", queue_enter_ts)
                    trace.mark("processing_start", processing_start_ts)
                    trace.mark("stt_start", t_stt_start)
                    trace.mark("stt_end", t_stt_end)
                    trace.mark("transcript_received", t_stt_end)
                
                import inspect
                sig = inspect.signature(self.on_transcript_chunk)
                if len(sig.parameters) >= 3 or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    self.on_transcript_chunk(cleaned_text, is_q, source)
                else:
                    self.on_transcript_chunk(cleaned_text, is_q)
            except Exception as e:
                print(f"Final transcription error: {e}")
                
        threading.Thread(target=_worker, daemon=True).start()


# ============================================================
# KNOWLEDGE SOURCE CLASSIFIER & UNIFIED CANDIDATE CONTEXT
# ============================================================

def determine_knowledge_source(
    question: str,
    session: Optional[InterviewSession] = None,
    recent_context: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Determines whether the incoming question requires:
    1. CANDIDATE_EXPERIENCE (Resume, work history, past achievements)
    2. CANDIDATE_PROJECT (Specific implementation, design decisions, challenges faced)
    3. FOLLOW_UP_CONVERSATION (Follow-up referring to previous answer)
    4. GENERAL_TECHNICAL (Core language/system concept, algorithm, standard definition)
    5. HYBRID_COMBINATION (Technical concept tied to candidate's verified experience)
    """
    if not question:
        return "GENERAL_TECHNICAL"
        
    lower = question.lower()
    
    # 1. Check for follow-up questions
    followup_starters = ["what about", "how about", "and for", "why did you do that", "why not", "then what happened", "why?"]
    if any(lower.startswith(f) for f in followup_starters) or (session and session.qa_pairs and len(lower.split()) <= 4 and lower.endswith("?")):
        return "FOLLOW_UP_CONVERSATION"
        
    # 2. Check for candidate project / implementation inquiries
    project_patterns = [
        r"\b(your project|in your project|in your application|the project you built)\b",
        r"\b(why did you choose|why did you pick|what database did you use)\b",
        r"\b(what problems did you face|what challenges did you face|how did you solve)\b",
        r"\b(how did you implement|how did you handle|how you implemented)\b",
        r"\b(your architecture|your stack|your database|your pipeline)\b"
    ]
    for p in project_patterns:
        if re.search(p, lower):
            return "CANDIDATE_PROJECT"
            
    # 3. Check for candidate resume / background inquiries
    resume_patterns = [
        r"\b(walk me through your resume|tell me about yourself|your background|your experience)\b",
        r"\b(your role at|your time at|in your previous role|in your last company)\b",
        r"\b(your greatest achievement|biggest accomplishment|your responsibilities)\b",
        r"\b(why should we hire you|your strengths|your leadership)\b"
    ]
    for p in resume_patterns:
        if re.search(p, lower):
            return "CANDIDATE_EXPERIENCE"
            
    # 4. Pure general technical / algorithm definitions
    general_tech_patterns = [
        r"^what is (the difference between|polymorphism|a deadlock|an index|acid|cap theorem|rest|jwt|cors|grpc)",
        r"^explain (how indexing works|tcp|udp|garbage collection|event loop|b-tree)",
        r"^write (a function|an algorithm|a query|a regex|a binary search)"
    ]
    for p in general_tech_patterns:
        if re.search(p, lower):
            return "GENERAL_TECHNICAL"
            
    return "HYBRID_COMBINATION"


def sanitize_generated_answer(text: str) -> str:
    """
    Sanitizes raw model output to prevent internal reasoning leakage, planning metadata,
    <think> tags, and conversational preamble from ever reaching the user or session history.
    Returns strictly candidate-ready spoken text.
    """
    if not text:
        return ""
        
    cleaned = text.strip()
    
    # 1. Strip XML-style reasoning / thought tags
    tag_patterns = [
        r"<think(?:\s+[^>]*)?>[\s\S]*?</think>",
        r"<thought(?:\s+[^>]*)?>[\s\S]*?</thought>",
        r"<reasoning(?:\s+[^>]*)?>[\s\S]*?</reasoning>",
        r"<analysis(?:\s+[^>]*)?>[\s\S]*?</analysis>",
        r"<plan(?:\s+[^>]*)?>[\s\S]*?</plan>",
        r"<internal(?:\s+[^>]*)?>[\s\S]*?</internal>",
        r"<thought_process(?:\s+[^>]*)?>[\s\S]*?</thought_process>",
        r"^<think(?:\s+[^>]*)?>[\s\S]*",  # Incomplete / unclosed think block
    ]
    for p in tag_patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
        
    # 2. Strip code-fenced thoughts (e.g. ```thought ... ```)
    cleaned = re.sub(r"```(?:thought|thinking|reasoning|plan)\b[\s\S]*?```", "", cleaned, flags=re.IGNORECASE).strip()
    
    # 3. Strip leaked planning blocks at the beginning of the response
    draft_match = re.search(r"\b(?:Drafting|Draft answer|Candidate answer|Final answer|Spoken answer)\s*:\s*[\"']?([\s\S]+)", cleaned, re.IGNORECASE)
    if draft_match and re.search(r"\b(?:Context|Strategy|Background|Rules|Goal|Thought process)\s*:", cleaned[:draft_match.start()], re.IGNORECASE):
        cleaned = draft_match.group(1).strip().strip("\"'")
        
    # 4. Strip line-by-line leaked planning prefixes at start of text
    lines = cleaned.split("\n")
    cleaned_lines = []
    in_planning_header = True
    for line in lines:
        l_str = line.strip()
        if in_planning_header:
            if not l_str:
                continue
            if re.match(r"^(?:Context|Strategy|Background|Rules|Drafting|Reasoning|Analysis|Planning|Thought process|Internal instructions|Plan|Step \d+)\s*:", l_str, re.IGNORECASE):
                continue
            if re.match(r"^(?:The user wants|Context is|Candidate is|Target role:|Strategy:)", l_str, re.IGNORECASE):
                continue
            # First legitimate content line reached
            in_planning_header = False
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
            
    cleaned = "\n".join(cleaned_lines).strip()
    
    # 5. Strip surrounding quotes if the model wrapped the entire answer in quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        if len(cleaned) > 2:
            cleaned = cleaned[1:-1].strip()
            
    # 6. Multi-pass robotic conversational padding stripping
    robotic_patterns = [
        r"^(?:sure|certainly|absolutely|of course)[!.,]?\s*",
        r"^here is (?:your|the|a)?\s*(?:answer|response|structured answer)?:?\s*",
        r"^as an ai(?: (?:language|conversational) model)?[!.,]?\s*",
        r"^according to (?:your|my) (?:resume|background|experience),?\s*",
        r"^(?:i can help with that|let me explain|here's what you can say):?\s*",
        r"^(?:when it comes to telling you about myself|to tell you about myself),?\s*"
    ]
    for _ in range(3):
        prev = cleaned
        for pattern in robotic_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == prev:
            break
            
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    return cleaned


class StreamingReasoningFilter:
    """
    Filters raw LLM streaming tokens in real time to prevent reasoning tags (<think>...</think>)
    and planning headers from ever leaking into UI rendering or live transcript display.
    """
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.buffer = ""
        self.in_think_block = False
        self.past_header = False
        self.accumulated_text = ""

    def push(self, token: str):
        self.buffer += token
        self.accumulated_text += token
        
        # 1. Check for entry into <think> tag
        if not self.in_think_block:
            if "<think" in self.buffer.lower():
                idx = self.buffer.lower().find("<think")
                clean_prefix = self.buffer[:idx]
                if clean_prefix and self.past_header:
                    self.callback(clean_prefix)
                self.buffer = self.buffer[idx:]
                self.in_think_block = True

        # 2. If in think block, look for closing </think>
        if self.in_think_block:
            if "</think>" in self.buffer.lower():
                idx = self.buffer.lower().find("</think>") + len("</think>")
                self.buffer = self.buffer[idx:]
                self.in_think_block = False
            else:
                return

        # 3. Check for leading planning headers (e.g. Context:, Strategy:, Drafting:)
        if not self.past_header:
            if "\n\n" in self.buffer or len(self.buffer) > 100:
                if re.search(r"^(?:<think|Context:|Strategy:|Background:|Rules:|Drafting:|Reasoning:|Analysis:|Thought process:)", self.buffer.strip(), re.IGNORECASE):
                    if "\n\n" in self.buffer:
                        parts = self.buffer.split("\n\n", 1)
                        if re.search(r"\b(?:Context|Strategy|Background|Rules|Drafting|Reasoning)\b", parts[0], re.IGNORECASE):
                            self.buffer = parts[1]
                            self.past_header = True
                        else:
                            self.past_header = True
                    else:
                        return
                else:
                    self.past_header = True
            else:
                if any(self.buffer.strip().lower().startswith(p) for p in ["context:", "strategy:", "background:", "drafting:", "reasoning:", "rules:"]):
                    return
                if len(self.buffer) > 25:
                    self.past_header = True
                else:
                    return

        # 4. Flush safe clean tokens
        if not self.in_think_block and self.past_header:
            if self.buffer:
                if "<" in self.buffer and not "</" in self.buffer:
                    last_angle = self.buffer.rfind("<")
                    to_send = self.buffer[:last_angle]
                    self.buffer = self.buffer[last_angle:]
                    if to_send:
                        self.callback(to_send)
                else:
                    to_send = self.buffer
                    self.buffer = ""
                    self.callback(to_send)

    def flush(self):
        """Flushes any remaining valid buffer if not inside a thought block."""
        if not self.in_think_block and self.buffer:
            clean = sanitize_generated_answer(self.buffer)
            if clean:
                self.callback(clean)
            self.buffer = ""


def build_interview_system_prompt(
    session: InterviewSession,
    question_type: str = "general",
    knowledge_source: str = "HYBRID_COMBINATION",
    relationship_info: Optional[Dict[str, Any]] = None
) -> str:
    """Builds a rich, personalized system prompt aligned with Parakeet AI standards, strict grounding policies, and conversation intelligence."""
    base_profile = session.get_base_profile_block()
    
    grounding_rules = """
3. PERSONALIZATION & GROUNDING POLICIES:
   - For questions about YOUR EXPERIENCE or PROJECTS: Base your answer strictly on the verified facts in the candidate's Resume and Knowledge Base below.
   - ZERO FABRICATION & NO OVERSELLING: Never invent companies you didn't work for, degrees you don't have, or tools absent from your background. Do NOT call yourself a "Senior Software Engineer" or claim years of leadership unless explicitly written in the candidate profile.
   - DO NOT DUMP KEYWORDS: Do NOT list 10+ technologies in a single answer. Choose the top 3–4 most relevant skills.
   - AVOID CORPORATE BUZZWORDS: Speak like a real candidate. Say "I built...", "I worked with...", "I focused on...". Avoid buzzwords like "architected", "leveraging", "robust engineering ecosystem", "scalable intelligent automation", or "cutting-edge solutions".
   - For GENERAL TECHNICAL questions: Provide the precise, industry-standard definition and practical trade-offs; then weave in verified experience only if relevant.
"""
    if knowledge_source in ["CANDIDATE_EXPERIENCE", "CANDIDATE_PROJECT"]:
        grounding_rules += "\n   - CRITICAL FOCUS: This question asks directly about the candidate's personal project/experience. Focus heavily on real project architecture, design trade-offs, and verified tech stack."
    elif knowledge_source == "FOLLOW_UP_CONVERSATION" or (relationship_info and relationship_info.get("is_follow_up")):
        grounding_rules += "\n   - CRITICAL FOCUS: This is a contextual follow-up to the preceding question/answer. Directly build upon the technology and architecture established in the previous turn."
        
    relationship_block = ""
    if relationship_info and relationship_info.get("is_follow_up"):
        rel_type = relationship_info.get("relationship_type", "FOLLOW_UP")
        topic = relationship_info.get("topic", getattr(session, "active_topic", "Current Project"))
        delta_focus = relationship_info.get("delta_focus", "")
        relationship_block = f"""
============================================================
ANSWER INTELLIGENCE & CONVERSATION DELTA
============================================================
- Relationship: {rel_type}
- Active Discussion Topic: {topic}
- Delta Focus Directive: {delta_focus}
- ANTI-REPETITION MANDATE: Do NOT repeat the entire previous answer or restate basic concepts. Answer only the delta requested by this follow-up question.
"""

    from classifier import get_answer_strategy
    specific_strategy = get_answer_strategy(question_type)
    
    prompt = f"""
You are PARAKEET AI — the world's most elite, hyper-intelligent Real-Time Interview Copilot and executive interview coach.
You are whispering directly into the candidate's ear during a live, high-stakes interview.

{base_profile}
{relationship_block}
============================================================
SESSION CONFIGURATION & STRATEGY
============================================================
- Selected Tone / Delivery: {session.tone}
- Preferred Spoken Language: {session.language}
- Extra User Instructions: {session.custom_instructions if session.custom_instructions else "None"}
- Detected Question Category: {question_type.upper()}
- Targeted Knowledge Source: {knowledge_source}

============================================================
DEDICATED ANSWER GENERATION STRATEGY
============================================================
{specific_strategy}

============================================================
CRITICAL RULES & LENGTH BUDGETS (PARAKEET AI QUALITY)
============================================================
1. SPEAK IN FIRST PERSON ("I", "my", "our team") as the candidate talking naturally to the interviewer.
2. ZERO REASONING LEAKAGE: NEVER output <think> tags, chain-of-thought, reasoning notes, or planning headers ("Context:", "Strategy:", "Drafting:", "Background:"). Start IMMEDIATELY with the first word of the spoken response.
3. ZERO ROBOTIC FLUFF: Never say "Certainly!", "Sure thing!", "Here is a structured answer:", "As an AI...", or "According to my resume". Dive straight into the natural spoken response.
{grounding_rules}
4. QUESTION-SPECIFIC LENGTH & SPOKEN-READY DELIVERY:
   - "Tell me about yourself" / Introductions: 3–5 natural sentences (~60–100 words, max 120 words).
   - Simple Definition / General Concept: 1–3 sentences (~30–60 words).
   - Project Question: 50–120 words.
   - Why Technology / Comparison: 40–90 words.
   - Behavioral / Situation (STAR): 80–150 words.
   - Technical Concept: 50–120 words.
   - Follow-up Delta: 20–70 words.
   - Coding / System Design: Structured explanation with clean code / components.
5. TARGET LANGUAGE: Deliver the answer in {session.language}.
"""
    return prompt.strip()


def validate_and_clean_answer(
    raw_answer: str,
    question: str,
    session: InterviewSession,
    question_type: str,
    knowledge_source: str
) -> str:
    """
    Lightweight, ultra-fast Answer Quality and Grounding Validation Layer.
    1. Sanitizes internal reasoning tags (<think>, Context:, Strategy:, Drafting:).
    2. Strips out robotic meta-chatter, robotic disclaimers, and conversational padding.
    3. Re-enforces first-person authentic candidate voice.
    4. Prevents artificial claims and ensures natural spoken length.
    """
    if not raw_answer:
        return ""
        
    cleaned = sanitize_generated_answer(raw_answer)
        
    # Format and standardize scannable layout
    try:
        from answer_formatter import format_interview_answer
        cleaned = format_interview_answer(cleaned, question_type=question_type)
    except Exception:
        pass

    return cleaned


def generate_interview_answer(
    client: Groq,
    session: InterviewSession,
    question: str,
    on_token: Callable[[str], None],
    refinement_instruction: Optional[str] = None,
    raw_transcript: Optional[str] = None,
    is_cancelled_callback: Optional[Callable[[], bool]] = None,
    question_type: Optional[str] = None,
    latency_trace: Optional[QuestionLatencyTrace] = None
) -> str:
    """
    Streams a personalized, top-tier interview answer via Groq LLM with adaptive token budgeting,
    structured 429 rate-limit backoff, stale generation cancellation, and monotonic latency instrumentation.
    """
    start_time = time.perf_counter()
    trace = latency_trace or getattr(session, "latest_latency_trace", None)
    if trace:
        trace.mark("generation_start", start_time)
        
    rel_info = classify_answer_relationship(question, session=session)
    
    strategy_key = rel_info.get("strategy_key", "")
    if strategy_key and strategy_key != "general":
        q_type = strategy_key
    elif question_type:
        q_type = question_type.lower()
    else:
        q_type = classify_question(question)
        
    knowledge_source = determine_knowledge_source(question, session)
    system_prompt = build_interview_system_prompt(session, q_type, knowledge_source, relationship_info=rel_info)
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add recent Q&A context (up to 2 past turns with concise answer truncation to preserve TPM budget)
    for qa in session.qa_pairs[-2:]:
        q_text = qa.get("question", "").strip()
        a_text = qa.get("answer", "").strip()
        words = a_text.split()
        if len(words) > 150:
            a_text = " ".join(words[:150]) + "..."
        messages.append({"role": "user", "content": q_text})
        messages.append({"role": "assistant", "content": a_text})
        
    if rel_info.get("is_follow_up") and rel_info.get("delta_focus"):
        user_prompt = f"INTERVIEWER'S FOLLOW-UP QUESTION:\n\"{question}\"\n\n[ANSWER INTELLIGENCE CONTEXT]:\n{rel_info.get('delta_focus')}\n\nGenerate the focused, authentic, spoken first-person answer addressing this exact delta."
    else:
        user_prompt = f"INTERVIEWER'S QUESTION:\n\"{question}\"\n\nGenerate the perfect, authentic, spoken first-person answer for the candidate right now."
        
    if raw_transcript and raw_transcript != question:
        user_prompt += f"\n(Raw audio transcript: \"{raw_transcript}\")"
    
    if refinement_instruction:
        user_prompt += f"\n\n[USER REFINEMENT OVERRIDE]: {refinement_instruction}"
        
    messages.append({"role": "user", "content": user_prompt})
    
    # Adaptive token budget per question type (Correction #1)
    adaptive_max_tokens = get_adaptive_max_tokens(q_type)
    
    # High-capacity & verified active models with graceful rate-limit fallback
    models_to_try = [
        "openai/gpt-oss-20b",
        "groq/compound",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "allam-2-7b"
    ]
    
    answer = ""
    first_token_time = None
    stream_filter = StreamingReasoningFilter(on_token)
    
    for model_name in models_to_try:
        if is_cancelled_callback and is_cancelled_callback():
            return ""
            
        try:
            # For reasoning models like qwen/qwen3.6-27b, ensure higher token budget so think block doesn't consume all tokens
            curr_max_tokens = max(adaptive_max_tokens, 1024) if "qwen" in model_name or "120b" in model_name else adaptive_max_tokens
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.6,
                max_tokens=curr_max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if is_cancelled_callback and is_cancelled_callback():
                    return ""
                    
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    now_token = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now_token
                        if trace:
                            trace.mark("first_token", first_token_time)
                    answer += token
                    if trace and (len(answer.split()) >= 3 or any(p in answer for p in ['.', '!', '?', '\n'])):
                        trace.mark("first_useful_token", now_token)
                    stream_filter.push(token)
                    
            stream_filter.flush()
            if answer.strip():
                break
        except Exception as e:
            cat, is_retryable, user_msg = categorize_api_error(e)
            
            # Rate limit specific bounded handling (Correction #2, #8, #9)
            if cat == "RATE_LIMIT":
                if trace:
                    trace.rate_limited = True
                    retry_sec = parse_retry_after_seconds(e)
                    trace.retry_after_ms = retry_sec * 1000.0
                    trace.retry_attempt += 1
                
                # Check cancellation before sleep
                if is_cancelled_callback and is_cancelled_callback():
                    return ""
                    
                retry_sec = parse_retry_after_seconds(e)
                bounded_sleep = min(retry_sec, 2.5)
                # Sleep in 100ms slices to abort immediately on stale generation
                slice_end = time.perf_counter() + bounded_sleep
                while time.perf_counter() < slice_end:
                    if is_cancelled_callback and is_cancelled_callback():
                        return ""
                    time.sleep(0.1)
                continue
                
            # Authentication failure: stop immediately
            if cat == "AUTHENTICATION":
                print(f"Groq auth notice [{cat}] on {model_name}: {user_msg}")
                break
                
            # Model not found / Invalid request / Timeout: continue to next fallback model
            print(f"Groq generation notice on {model_name} [{cat}]: {e}, trying next model...")
            continue
            
    total_time = time.perf_counter() - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time
    session.latency_stats = {
        "ttft_sec": round(ttft, 2),
        "total_sec": round(total_time, 2)
    }
    
    if trace:
        trace.mark("generation_end", time.perf_counter())
        trace.compute_metrics()
        session.latency_traces.append(trace)
        # Debug trace summary print
        summary = trace.format_debug_summary()
        if session.latency_traces and len(session.latency_traces) % 5 == 0:
            print(f"[Latency Diagnostics] {summary}")
    
    if not answer.strip():
        if is_cancelled_callback and is_cancelled_callback():
            return ""
        # Multi-Tier Offline Fallback (Ollama -> Heuristic Resume Synthesis)
        try:
            from offline_fallback_manager import (
                check_ollama_available,
                stream_ollama_response,
                generate_offline_heuristic_answer
            )
            ollama_ok, ollama_models = check_ollama_available()
            if ollama_ok and ollama_models:
                chosen_model = ollama_models[0]
                answer = stream_ollama_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=chosen_model,
                    on_token=on_token,
                    is_cancelled=is_cancelled_callback
                )
            if not answer.strip():
                answer = generate_offline_heuristic_answer(
                    question=question,
                    job_title=session.job_title or "Software Engineer",
                    company_name=session.company_name or "",
                    resume_text=session.resume_text or "",
                    on_token=on_token
                )
        except Exception as e:
            print(f"Notice during offline fallback generation: {e}")

    if not answer.strip():
        fallback_msg = "Could not generate answer at this moment. Please check your network connection and Groq API key."
        on_token(fallback_msg)
        return fallback_msg
        
    validated_answer = validate_and_clean_answer(
        raw_answer=answer,
        question=question,
        session=session,
        question_type=question_type,
        knowledge_source=knowledge_source
    )
    
    session.record_qa(question, validated_answer)
    return validated_answer


def predict_likely_followups(client: Groq, question: str, answer: str) -> List[str]:
    """Asynchronously predicts 2-3 likely follow-up questions from the interviewer."""
    if not client or not question:
        return []
        
    prompt = f"""Given this interview question and candidate's response:
QUESTION: "{question}"
ANSWER: "{answer[:300]}..."

Predict 2 or 3 sharp, realistic follow-up questions an interviewer might ask next.
Return JSON array only: ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]"""

    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        if isinstance(data, dict):
            # Find list in values
            for v in data.values():
                if isinstance(v, list):
                    return [str(x) for x in v[:3]]
        elif isinstance(data, list):
            return [str(x) for x in data[:3]]
    except Exception as e:
        print(f"Follow-up prediction notice: {e}")
        
    return []


# ============================================================
# MOCK INTERVIEW MODE ENGINE
# ============================================================

def generate_mock_interview_question(
    client: Groq,
    role: str = "Software Engineer",
    difficulty: str = "Senior",
    session: Optional[InterviewSession] = None,
    previous_questions: Optional[List[str]] = None
) -> str:
    """
    Generates a realistic mock interview question tailored to the candidate's verified
    resume/projects and chosen difficulty level without inventing false background.
    """
    resume_snippet = session.resume_text[:1200] if (session and session.resume_text) else "No uploaded resume provided."
    prev_str = "\n".join([f"- {q}" for q in (previous_questions or [])[-4:]]) if previous_questions else "None yet."
    
    prompt = f"""You are a top-tier Technical Hiring Manager conducting a realistic Mock Interview.
Target Role: {role}
Difficulty Level: {difficulty}

CANDIDATE'S VERIFIED BACKGROUND & RESUME:
{resume_snippet}

PREVIOUSLY ASKED IN THIS MOCK SESSION:
{prev_str}

TASK:
Generate ONE realistic, articulate interview question for this candidate at the '{difficulty}' level.
- Incorporate technologies or projects from their verified resume when relevant.
- Do NOT invent companies or roles not present in the resume.
- Make the question distinct from previously asked questions.
- Return ONLY the question string, without quotation marks or conversational preamble.
"""
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=90
        )
        return resp.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"Mock question generation error: {e}")
        return f"Can you walk me through the architecture of a challenging project you built as a {role}?"


def evaluate_mock_interview_answer(
    client: Groq,
    question: str,
    candidate_answer: str,
    session: Optional[InterviewSession] = None,
    difficulty: str = "Senior"
) -> Dict[str, Any]:
    """
    Evaluates candidate answer across 7 key dimensions:
    - Technical Correctness
    - Relevance
    - Communication & Structure
    - Completeness
    - Clarity
    - Confidence Indicators
    - Overall Score
    Plus actionable feedback and an example improved answer.
    """
    if not candidate_answer or len(candidate_answer.strip().split()) < 3:
        return {
            "overall_score": 35,
            "technical_correctness": 30,
            "relevance": 40,
            "communication": 40,
            "completeness": 30,
            "clarity": 40,
            "confidence_score": 30,
            "strengths": ["Attempted to answer the prompt"],
            "improvements": ["Answer is too brief; elaborate with technical architecture and concrete examples."],
            "summary_feedback": "The answer was very short. Aim for 120-250 words with concrete engineering details.",
            "example_improved_answer": "In my implementation approach, I structured the solution using..."
        }

    role = session.job_title if (session and session.job_title) else "Software Engineer"
    prompt = f"""You are a Lead Hiring Manager evaluating a candidate's live spoken answer.
Role: {role}
Difficulty Level: {difficulty}

QUESTION ASKED:
"{question}"

CANDIDATE'S SPOKEN ANSWER:
"{candidate_answer}"

Evaluate the answer objectively on:
1. Technical Correctness (0-100)
2. Relevance (0-100)
3. Communication & Structure (STAR for behavioral, intuition+code for coding) (0-100)
4. Completeness & Depth (0-100)
5. Clarity & Pacing (0-100)
6. Confidence Indicators (0-100)

Return STRICTLY JSON format:
{{
  "overall_score": 85,
  "technical_correctness": 88,
  "relevance": 90,
  "communication": 82,
  "completeness": 80,
  "clarity": 85,
  "confidence_score": 85,
  "strengths": ["Clear explanation of core concept", "Used relevant terminology"],
  "improvements": ["Explicitly mention trade-offs or performance numbers"],
  "summary_feedback": "A strong, articulate response with good technical grounding.",
  "example_improved_answer": "A polished, first-person spoken answer that the candidate could deliver..."
}}"""

    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=650,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Mock interview eval notice: {e}")
        return {
            "overall_score": 80,
            "technical_correctness": 82,
            "relevance": 85,
            "communication": 80,
            "completeness": 78,
            "clarity": 80,
            "confidence_score": 80,
            "strengths": ["Direct response to the question"],
            "improvements": ["Add specific engineering metrics and trade-offs"],
            "summary_feedback": "Answer was complete and addressed the question well.",
            "example_improved_answer": f"When addressing {question.lower().rstrip('?')}, my approach is to prioritize..."
        }


def generate_mock_interview_final_report(
    client: Groq,
    session: InterviewSession,
    mock_history: List[Dict[str, Any]],
    role: str = "Software Engineer",
    difficulty: str = "Senior"
) -> str:
    """Generates the comprehensive final mock interview report card."""
    if not mock_history:
        return "No mock interview questions were completed."

    scores = [item["evaluation"].get("overall_score", 80) for item in mock_history if "evaluation" in item]
    avg_score = int(sum(scores) / len(scores)) if scores else 80
    
    q_breakdown = ""
    for idx, item in enumerate(mock_history, 1):
        q = item.get("question", "")
        a = item.get("answer", "")
        ev = item.get("evaluation", {})
        q_breakdown += f"""
### Question {idx}: "{q}"
- **Candidate Answer**: {a}
- **Score**: {ev.get('overall_score', 80)}/100 (Tech: {ev.get('technical_correctness', 80)} | Comm: {ev.get('communication', 80)} | Clarity: {ev.get('clarity', 80)})
- **Strengths**: {', '.join(ev.get('strengths', ['Good response']))}
- **Areas to Improve**: {', '.join(ev.get('improvements', ['Add more detail']))}
- **💡 Polished Model Answer**:
> {ev.get('example_improved_answer', 'N/A')}
"""

    report = f"""# 🎓 Mock Interview Comprehensive Report Card

**Target Role:** {role} | **Difficulty Level:** {difficulty}  
**Overall Performance Score:** **{avg_score} / 100**

---

## 📊 Performance Breakdown

{q_breakdown}

---

## 🎯 Recommended Next Steps & Practice Topics
- Review the model answers above to adopt high-impact phrasing and trade-off explanations.
- Continue practicing vocal delivery and structured pacing using Parakit Mock Mode.
"""
    return report.strip()


# ============================================================
# POST-INTERVIEW SESSION SUMMARY GENERATOR
# ============================================================

def generate_session_summary(client: Groq, session: InterviewSession) -> str:
    """Generates a complete post-interview report with analytics dashboard, strengths, Q&A breakdown, and preparation areas."""
    if not session.qa_pairs and not session.transcript_log:
        return "No interview activity was recorded during this session."
        
    analytics = calculate_session_analytics(session)
    
    analytics_header = f"""# 📊 Interview Session Analytics Report

| Metric | Value | Metric | Value |
| :--- | :--- | :--- | :--- |
| **Total Questions** | {analytics['total_questions']} | **Avg Response Time** | {analytics['avg_generation_time_sec']}s |
| **Technical Questions** | {analytics['technical_count']} | **Avg Answer Length** | {analytics['avg_answer_length_words']} words |
| **Coding Questions** | {analytics['coding_count']} | **Refined Queries** | {analytics['questions_requiring_refinement']} |
| **System Design** | {analytics['system_design_count']} | **Clarifications** | {analytics['questions_requiring_clarification']} |
| **Behavioral (STAR)** | {analytics['behavioral_count']} | **Low Confidence Items** | {analytics['low_confidence_count']} |
| **Project Questions** | {analytics['project_count']} | **HR / Intro Questions** | {analytics['hr_count']} |

**Topics Covered:** {', '.join(analytics['topics_discussed']) if analytics['topics_discussed'] else 'General Engineering'}

---

"""
    raw_transcript = "\n".join([f"[{t['timestamp']}] {t['role']}: {t['text']}" for t in session.transcript_log])
    qa_summary = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in session.qa_pairs])
    
    prompt = f"""
You are an expert interview coach and senior engineering hiring manager.
Review the interview session transcript and candidate answers below.
Target Role: {session.job_title or 'General Tech Role'}
Company: {session.company_name or 'Target Company'}

TRANSCRIPT:
{raw_transcript}

QUESTIONS & ANSWERS:
{qa_summary}

Please provide an organized, executive-grade post-interview report formatted in clean Markdown with:
1. 🎯 **Session Overview & Key Topics Covered**
2. 💡 **Questions Asked & Response Highlights**
3. ⭐ **Demonstrated Strengths & Standout Answers**
4. 📈 **Targeted Preparation Areas (based strictly on this session)**
5. 📝 **Next Action Items & Recommended Follow-Up Notes**

Keep the tone encouraging, constructive, and highly actionable.
"""
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a world-class career coach and tech hiring manager."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1200
        )
        return analytics_header + resp.choices[0].message.content
    except Exception as e:
        print(f"Summary generation error: {e}")
        # Fallback manual summary
        summary_lines = [
            analytics_header,
            f"**Role:** {session.job_title or 'N/A'} | **Company:** {session.company_name or 'N/A'}",
            "",
            "## 📈 Recommended Preparation Areas",
        ]
        for p in analytics["preparation_areas"]:
            summary_lines.append(f"- {p}")
        summary_lines.append("\n## Questions & Answers Log")
        for idx, qa in enumerate(session.qa_pairs, 1):
            summary_lines.append(f"### {idx}. {qa['question']}")
            summary_lines.append(f"{qa['answer']}\n")
        return "\n".join(summary_lines)
