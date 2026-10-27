"""
BluJay AI - Session Models & State Tracking
===========================================
Encapsulates InterviewSession, SessionController, latency tracing, and telemetry.
"""

import os
import re
import time
import uuid
import enum
import threading
import collections
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List, Tuple, Union, Set

from engine.utils import load_user_config, save_user_config, extract_text_from_file
from rag import load_documents

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
    Lightweight central Session Controller, Telemetry & Recovery Layer.
    Tracks session state, component health, heartbeat telemetry, latency percentiles,
    audio quality aggregation, thread-safe structured error logging, and isolated watchdog recovery.
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
        self._telemetry_lock = threading.Lock()
        self._error_ring_buffer: collections.deque = collections.deque(maxlen=100)

    def set_state(self, new_state: InterviewTurnState, reason: str = ""):
        self.state = new_state
        if self.session:
            lock = getattr(self.session, "_state_lock", None)
            if lock:
                with lock:
                    self.session.turn_state = new_state
            else:
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

    def record_error(
        self,
        component: str,
        error: Union[str, Exception],
        category: Optional[str] = None,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Thread-safe structured error logging into circular in-memory ring buffer (maxlen=100).
        Automatically categorizes errors, updates component health, and updates heartbeat.
        """
        now = time.time()
        err_msg = str(error)
        
        # Determine category if not explicitly provided
        effective_category = category
        if not effective_category:
            if isinstance(error, Exception):
                cat, _, _ = categorize_api_error(error)
                effective_category = cat
            elif "audio" in component or "soundcard" in err_msg.lower() or "wasapi" in err_msg.lower():
                effective_category = "AUDIO_HARDWARE"
            elif "stt" in component or "deepgram" in err_msg.lower() or "whisper" in err_msg.lower():
                effective_category = "STT_API"
            elif "generation" in component or "groq" in err_msg.lower() or "gemini" in err_msg.lower():
                effective_category = "LLM_API"
            elif "screen" in component:
                effective_category = "SCREEN_CAPTURE"
            else:
                effective_category = "SYSTEM"

        entry = {
            "timestamp": now,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "component": component,
            "category": effective_category,
            "message": err_msg,
            "recoverable": recoverable,
            "context": context or {}
        }
        
        with self._telemetry_lock:
            self._error_ring_buffer.append(entry)
            
        status = "RECOVERING" if recoverable else "ERROR"
        self.update_component(component, status, err_msg)
        return entry

    def get_error_summary(self) -> Dict[str, Any]:
        """Returns structured aggregation of recorded error events."""
        with self._telemetry_lock:
            errors_list = list(self._error_ring_buffer)
            
        total_errors = len(errors_list)
        by_component: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for e in errors_list:
            comp = e.get("component", "unknown")
            cat = e.get("category", "UNKNOWN")
            by_component[comp] = by_component.get(comp, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1
            
        recent = errors_list[-10:] if errors_list else []
        return {
            "total_errors": total_errors,
            "by_component": by_component,
            "by_category": by_category,
            "recent_errors": recent
        }

    def get_latency_summary(self) -> Dict[str, Any]:
        """
        Computes robust session-wide latency statistics (count, min, max, mean, p50, p95, p99).
        Handles zero-sample and single-sample gracefully without throwing exceptions or generating NaN.
        """
        traces = getattr(self.session, "latency_traces", []) if self.session else []
        if not traces:
            empty_stat = {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
            return {
                "trace_count": 0,
                "stt_latency_ms": dict(empty_stat),
                "ttft_ms": dict(empty_stat),
                "first_useful_answer_ms": dict(empty_stat),
                "e2e_latency_ms": dict(empty_stat),
                "total_generation_duration_ms": dict(empty_stat)
            }

        def compute_stat_dict(values: List[float]) -> Dict[str, float]:
            clean_vals = sorted([float(v) for v in values if isinstance(v, (int, float)) and v > 0.0])
            n = len(clean_vals)
            if n == 0:
                return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
            
            mean_v = round(sum(clean_vals) / n, 2)
            min_v = round(clean_vals[0], 2)
            max_v = round(clean_vals[-1], 2)
            
            p50_idx = min(n - 1, int(n * 0.50))
            p95_idx = min(n - 1, int(n * 0.95))
            p99_idx = min(n - 1, int(n * 0.99))
            
            return {
                "count": n,
                "p50": round(clean_vals[p50_idx], 2),
                "p95": round(clean_vals[p95_idx], 2),
                "p99": round(clean_vals[p99_idx], 2),
                "mean": mean_v,
                "min": min_v,
                "max": max_v
            }

        stt_vals, ttft_vals, first_useful_vals, e2e_vals, gen_dur_vals = [], [], [], [], []
        for tr in traces:
            m = tr.metrics if tr.metrics else tr.compute_metrics()
            stt_vals.append(m.get("speech_to_first_transcript_ms") or m.get("STT_total") or 0.0)
            ttft_vals.append(m.get("generation_start_to_first_token_ms") or m.get("TTFT") or 0.0)
            first_useful_vals.append(m.get("question_complete_to_first_useful_answer_ms") or m.get("first_useful_answer") or 0.0)
            e2e_vals.append(m.get("total_question_to_displayed_answer_ms") or m.get("total") or 0.0)
            gen_dur_vals.append(m.get("total_generation_duration_ms") or 0.0)

        return {
            "trace_count": len(traces),
            "stt_latency_ms": compute_stat_dict(stt_vals),
            "ttft_ms": compute_stat_dict(ttft_vals),
            "first_useful_answer_ms": compute_stat_dict(first_useful_vals),
            "e2e_latency_ms": compute_stat_dict(e2e_vals),
            "total_generation_duration_ms": compute_stat_dict(gen_dur_vals)
        }

    def get_overall_system_status(self) -> str:
        """
        Derives overall system status:
        - 'ERROR': Any critical component (audio_capture, stt, question_detection) is in ERROR.
        - 'DEGRADED': Any component is in RECOVERING or non-critical component is in ERROR.
        - 'HEALTHY': All components HEALTHY, BUSY, or IDLE.
        """
        critical = ("audio_capture", "stt", "question_detection")
        has_error = False
        has_recovering = False
        for k, v in self.components_health.items():
            status = v.get("status", "HEALTHY")
            if status == "ERROR":
                if k in critical:
                    return "ERROR"
                has_error = True
            elif status == "RECOVERING":
                has_recovering = True
        if has_error or has_recovering:
            return "DEGRADED"
        return "HEALTHY"

    def get_audio_quality_report(self) -> Dict[str, Any]:
        """Fetches cumulative audio quality report from continuous transcriber if available."""
        transcriber = getattr(self.session, "transcriber", None) if self.session else None
        if transcriber and hasattr(transcriber, "get_audio_quality_report"):
            return transcriber.get_audio_quality_report()
        return {
            "total_frames": 0,
            "dropped_frames": 0,
            "dropout_rate_pct": 0.0,
            "availability_pct": 100.0,
            "discontinuity_count": 0,
            "is_mic_muted": False
        }

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """
        Atomic snapshot of session controller health, audio metrics, latency percentiles, and errors.
        Executes in < 0.1ms under lock.
        """
        with self._telemetry_lock:
            overall = self.get_overall_system_status()
            comp_copy = {k: dict(v) for k, v in self.components_health.items()}
            
        return {
            "system_status": overall,
            "session_state": self.state.value if hasattr(self.state, "value") else str(self.state),
            "components": comp_copy,
            "audio_quality": self.get_audio_quality_report(),
            "latency": self.get_latency_summary(),
            "errors": self.get_error_summary(),
            "active_generation": dict(self.active_generation_info),
            "timestamp": time.time()
        }

    def check_health_and_staleness(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Context-aware health and staleness check:
        - Detects audio freeze (>4.0s without frame when stream is active)
        - Detects hung answer generation (>25s timeout)
        Returns list of detected anomalies.
        """
        cur = now if now is not None else time.time()
        anomalies = []
        
        # 1. Answer generation timeout check
        if self.check_generation_timeout():
            anomalies.append({
                "type": "GENERATION_TIMEOUT",
                "component": "answer_generation",
                "detail": "Answer generation exceeded 25s timeout and was auto-recovered"
            })

        # 2. Audio stream stall check (context-aware: only when transcriber is running)
        transcriber = getattr(self.session, "transcriber", None) if self.session else None
        if transcriber and getattr(transcriber, "is_running", False):
            last_audio = getattr(transcriber, "_last_system_frame_time", 0.0) or getattr(transcriber, "_last_mic_frame_time", 0.0)
            if last_audio > 0 and (cur - last_audio > 4.0):
                anomalies.append({
                    "type": "AUDIO_STALL",
                    "component": "audio_capture",
                    "detail": f"No audio frames received for {cur - last_audio:.1f}s while stream is active"
                })
                self.update_component("audio_capture", "DEGRADED", "Audio frame feed stalled (>4s)")

        return anomalies

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
                self.record_error(
                    component="answer_generation",
                    error=f"Generation request timed out after {timeout:.1f}s",
                    category="TIMEOUT",
                    recoverable=True
                )
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
    epoch: int = 0


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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the trace, timestamps, and computed metrics into a JSON-friendly dict."""
        self.compute_metrics()
        return {
            "question_id": self.question_id,
            "generation_id": self.generation_id,
            "segment_id": self.segment_id,
            "source": self.source,
            "timestamps": dict(self.timestamps),
            "metrics": dict(self.metrics),
            "queue_depth": self.queue_depth,
            "discontinuities": self.discontinuities,
            "rate_limited": self.rate_limited,
            "retry_after_ms": self.retry_after_ms,
            "retry_attempt": self.retry_attempt
        }



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
        
        self._state_lock: threading.RLock = threading.RLock()
        self._medium_ttl_s: float = 0.35
        
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
        self.session_summary: Optional[str] = None
        
        # Active detected question pending answer
        self.latest_detected_question: Optional[str] = None
        self.latest_answer: str = ""
        
        # Turn-Taking & Speaker Intelligence State
        self.turn_state: InterviewTurnState = InterviewTurnState.LISTENING
        self.active_question_id: str = ""
        self.question_counter: int = 0
        self._candidate_responses: List[str] = []
        self.last_confirmed_speaker: str = "Interviewer"
        self.active_interviewer_id: str = "Interviewer"
        self.candidate_responses_log: List[Dict[str, str]] = []
        self.pending_ambiguous_utterance: Optional[Dict[str, Any]] = None

        # Category 8: Long-session memory bounding ceilings (configurable)
        self._max_transcript_log_size: int = 500
        self._max_candidate_log_size: int = 300
        self._max_qa_pairs_size: int = 100
        self._max_question_analytics_size: int = 100
        self._max_latency_traces_size: int = 100

        # Category 8: Lossless cumulative analytics accumulator
        self._cumulative_analytics: Dict[str, Any] = {
            "total_questions_pruned": 0,
            "category_counts": {},
            "total_generation_time": 0.0,
            "gen_time_count": 0,
            "total_words": 0,
            "word_count": 0,
            "refinements": 0,
            "clarifications": 0,
            "low_conf": 0,
            "topics": set(),
        }

        # Medium-confidence question hold buffer.
        # Holds interviewer utterances scoring MEDIUM confidence (0.50 <= conf < 0.75).
        # The UI layer flushes it after ~350ms if no follow-up chunk arrives to complete it.
        # Contains: {"text", "raw_text", "confidence", "timestamp", "source", "speaker", "buffer_id"}
        # buffer_id is a UUID hex string used to detect stale timer firings (race safety).
        self.pending_medium_question: Optional[Dict[str, Any]] = None
        self.controller: SessionController = SessionController(self)
        
        # Screen-Aware Coding Configuration
        self.screen_aware_coding: bool = user_cfg.get("screen_aware_coding", True)
        self.mock_screen_data: Optional[Dict[str, Any]] = None
        self.screen_context_cache: Optional[Dict[str, Any]] = None
        
        # Coding Solution & Multi-Turn Problem State
        self.active_coding_problem: str = ""
        self.active_coding_solution: str = ""
        self.active_coding_language: str = "Python"
        self.active_coding_approach: str = ""

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
        with getattr(self, "_state_lock", threading.RLock()):
            self._cached_base_profile_block = None

    def get_base_profile_block(self) -> str:
        with self._state_lock:
            if self._cached_base_profile_block is not None:
                return self._cached_base_profile_block
            
        job_target = f"Target Role: {self.job_title}\nCompany: {self.company_name}" if (self.job_title or self.company_name) else "Target Role: Software Engineer / Tech Role"
        jd_context = self.job_description if self.job_description else "Not provided."
        instructions_context = f"\nCUSTOM PREPARATION INSTRUCTIONS:\n{self.custom_instructions}\n" if self.custom_instructions else ""
        
        has_custom_resume = bool(self.resume_text and self.resume_text.strip())
        
        if has_custom_resume:
            candidate_background_section = f"""============================================================
--- UPLOADED RESUME / CV ---
CANDIDATE'S PRIMARY UPLOADED RESUME (PRIMARY SOURCE OF TRUTH)
============================================================
The candidate has explicitly uploaded the following resume. Always use the work history, companies, education, skills, metrics, and projects from this resume FIRST:

{self.resume_text.strip()}
"""
        else:
            rag_context = self.knowledge_text if self.knowledge_text else "No extra knowledge documents loaded."
            candidate_background_section = f"""============================================================
CANDIDATE VERIFIED BACKGROUND (DEFAULT KNOWLEDGE BASE)
============================================================
No custom resume was uploaded for this session. Using verified default candidate knowledge base:

{rag_context}
"""
        
        self._cached_base_profile_block = f"""
============================================================
CANDIDATE TARGET & JOB CONTEXT
============================================================
{job_target}

JOB DESCRIPTION / REQUIREMENTS:
{jd_context}
{instructions_context}
{candidate_background_section}
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

    def purge_old_transcripts(self, max_age_days: int = 7) -> int:
        """
        PII retention policy: purges local transcript files older than max_age_days.
        Returns count of purged files.
        """
        purged = 0
        now = time.time()
        cutoff = now - (max_age_days * 86400)
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        if os.path.exists(logs_dir):
            try:
                for f in os.listdir(logs_dir):
                    fp = os.path.join(logs_dir, f)
                    if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                        try:
                            os.remove(fp)
                            purged += 1
                        except OSError:
                            pass
            except Exception as e:
                print(f"[PII Retention] Error purging old transcripts: {e}")
        return purged

    def _fold_into_cumulative_analytics(self, rec: Dict[str, Any]):
        """Helper to accumulate metrics from pruned question analytics records."""
        if not rec:
            return
        self._cumulative_analytics["total_questions_pruned"] += 1
        cat = rec.get("category", "").lower()
        if cat:
            self._cumulative_analytics["category_counts"][cat] = (
                self._cumulative_analytics["category_counts"].get(cat, 0) + 1
            )
        gt = rec.get("generation_time_sec", 0.0)
        if gt > 0:
            self._cumulative_analytics["total_generation_time"] += gt
            self._cumulative_analytics["gen_time_count"] += 1
        wc = rec.get("word_count", 0)
        if wc > 0:
            self._cumulative_analytics["total_words"] += wc
            self._cumulative_analytics["word_count"] += 1
        if rec.get("required_refinement", False):
            self._cumulative_analytics["refinements"] += 1
        if rec.get("required_clarification", False) or "clarification" in cat:
            self._cumulative_analytics["clarifications"] += 1
        if rec.get("confidence", 1.0) < 0.75:
            self._cumulative_analytics["low_conf"] += 1
        top = rec.get("topic", "").strip()
        if top and top != "General":
            self._cumulative_analytics["topics"].add(top)

    def prune_session_memory(
        self,
        max_transcripts: Optional[int] = None,
        max_qa: Optional[int] = None,
        max_candidate: Optional[int] = None,
        max_analytics: Optional[int] = None,
        max_traces: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Actively prunes session memory structures to stay within bounded RAM limits.
        Lossless analytics guarantee: evicting question_analytics folds historical
        metrics into _cumulative_analytics so calculate_session_analytics remains 100% exact.
        """
        limit_transcripts = max_transcripts if max_transcripts is not None else self._max_transcript_log_size
        limit_candidate = max_candidate if max_candidate is not None else self._max_candidate_log_size
        limit_qa = max_qa if max_qa is not None else self._max_qa_pairs_size
        limit_analytics = max_analytics if max_analytics is not None else self._max_question_analytics_size
        limit_traces = max_traces if max_traces is not None else self._max_latency_traces_size

        pruned_t = 0
        pruned_c = 0
        pruned_q = 0
        pruned_a = 0
        pruned_tr = 0

        with self._state_lock:
            # 1. Prune transcript_log
            if len(self.transcript_log) > limit_transcripts:
                pruned_t = len(self.transcript_log) - limit_transcripts
                self.transcript_log = self.transcript_log[pruned_t:]

            # 2. Prune candidate_responses_log
            if len(self.candidate_responses_log) > limit_candidate:
                pruned_c = len(self.candidate_responses_log) - limit_candidate
                self.candidate_responses_log = self.candidate_responses_log[pruned_c:]

            # 3. Prune qa_pairs
            if len(self.qa_pairs) > limit_qa:
                pruned_q = len(self.qa_pairs) - limit_qa
                self.qa_pairs = self.qa_pairs[pruned_q:]

            # 4. Prune question_analytics (losslessly accumulating into _cumulative_analytics)
            if len(self.question_analytics) > limit_analytics:
                pruned_a = len(self.question_analytics) - limit_analytics
                evicted = self.question_analytics[:pruned_a]
                self.question_analytics = self.question_analytics[pruned_a:]
                for rec in evicted:
                    self._fold_into_cumulative_analytics(rec)

            # 5. Prune latency_traces
            if len(self.latency_traces) > limit_traces:
                pruned_tr = len(self.latency_traces) - limit_traces
                self.latency_traces = self.latency_traces[pruned_tr:]

        return {
            "pruned_transcripts": pruned_t,
            "pruned_candidate": pruned_c,
            "pruned_qa": pruned_q,
            "pruned_analytics": pruned_a,
            "pruned_traces": pruned_tr
        }

    def record_latency_trace(self, trace: QuestionLatencyTrace):
        """Thread-safely records a QuestionLatencyTrace with bounded capacity."""
        if not trace:
            return
        with self._state_lock:
            self.latency_traces.append(trace)
            if len(self.latency_traces) > self._max_latency_traces_size:
                self.latency_traces.pop(0)

    def set_pending_medium_question(self, data: Optional[Dict[str, Any]]):
        """Thread-safely stores pending medium-confidence question hold buffer."""
        with self._state_lock:
            self.pending_medium_question = data

    def get_and_clear_pending_medium_question(self, buffer_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Thread-safely retrieves and clears pending medium question if buffer_id matches."""
        with self._state_lock:
            if not self.pending_medium_question:
                return None
            if buffer_id and self.pending_medium_question.get("buffer_id") != buffer_id:
                return None
            item = self.pending_medium_question
            self.pending_medium_question = None
            return item

    def set_pending_ambiguous_utterance(self, data: Optional[Dict[str, Any]]):
        """Thread-safely stores pending ambiguous utterance."""
        with self._state_lock:
            self.pending_ambiguous_utterance = data

    def get_and_clear_pending_ambiguous_utterance(self) -> Optional[Dict[str, Any]]:
        """Thread-safely retrieves and clears pending ambiguous utterance."""
        with self._state_lock:
            item = self.pending_ambiguous_utterance
            self.pending_ambiguous_utterance = None
            return item

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
        with self._state_lock:
            self.transcript_log.append(entry)
            if len(self.transcript_log) > self._max_transcript_log_size:
                self.transcript_log.pop(0)
            if is_question:
                self.latest_detected_question = text
                self.last_confirmed_speaker = role
            elif role != "Candidate":
                self.last_confirmed_speaker = role
        return entry

    def get_latest_interviewer_question(self) -> str:
        """Retrieves the most recent question asked by the interviewer or manual prompt."""
        with self._state_lock:
            answered_questions = {qa.get("question", "").strip() for qa in self.qa_pairs}
            if self.latest_detected_question and self.latest_detected_question.strip() not in answered_questions:
                return self.latest_detected_question.strip()
            for item in reversed(self.transcript_log):
                txt = item.get("text", "").strip()
                if not txt:
                    continue
                if item.get("role") in ["Interviewer", "system", "loopback"] or item.get("is_question"):
                    if txt not in answered_questions:
                        return txt
            if self.latest_detected_question:
                return self.latest_detected_question.strip()
            if self.last_question:
                return self.last_question.strip()
            if self.qa_pairs:
                return self.qa_pairs[-1].get("question", "").strip()
            for item in reversed(self.transcript_log):
                if item.get("text"):
                    return item["text"].strip()
            return ""

    @property
    def last_candidate_response(self) -> str:
        """Returns the most recent candidate utterance (backward-compatible)."""
        with self._state_lock:
            return self._candidate_responses[-1] if self._candidate_responses else ""

    @last_candidate_response.setter
    def last_candidate_response(self, value: str):
        """Maintains a rolling window of max 3 candidate utterances."""
        with self._state_lock:
            if value:
                self._candidate_responses.append(value)
                if len(self._candidate_responses) > 3:
                    self._candidate_responses.pop(0)
            else:
                self._candidate_responses.clear()

    @property
    def recent_candidate_context(self) -> str:
        """Returns rolling window of last 3 candidate responses joined."""
        with self._state_lock:
            return " ... ".join(self._candidate_responses) if self._candidate_responses else ""

    def add_candidate_speech(self, text: str):
        """Records candidate answer into session memory for context without triggering answers."""
        ts = time.strftime("%H:%M:%S")
        with self._state_lock:
            self.last_candidate_response = text
            self.last_confirmed_speaker = "Candidate"
            self.turn_state = InterviewTurnState.CANDIDATE_SPEAKING
            self.candidate_responses_log.append({
                "timestamp": ts,
                "text": text,
                "active_topic": self.active_topic
            })
            if len(self.candidate_responses_log) > self._max_candidate_log_size:
                self.candidate_responses_log.pop(0)

    @property
    def candidate_context(self) -> str:
        with self._state_lock:
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
        word_count = len(answer.split())
        cat_clean = category.lower() if category else "general"
        analytics_entry = {
            "question": question,
            "category": cat_clean,
            "generation_time_sec": round(gen_time, 2),
            "word_count": word_count,
            "required_refinement": required_refinement,
            "required_clarification": required_clarification,
            "confidence": confidence,
            "topic": topic or getattr(self, "active_topic", "") or "General"
        }

        with self._state_lock:
            self.qa_pairs.append({
                "question": question,
                "answer": answer,
                "timestamp": ts
            })
            if len(self.qa_pairs) > self._max_qa_pairs_size:
                self.qa_pairs.pop(0)

            self.latest_answer = answer
            self.last_question = question
            self.last_question_ts = time.time()
            self.latest_detected_question = None
            self.active_question_id = ""
            self.turn_state = InterviewTurnState.LISTENING

            self.question_analytics.append(analytics_entry)
            if len(self.question_analytics) > self._max_question_analytics_size:
                evicted = self.question_analytics.pop(0)
                self._fold_into_cumulative_analytics(evicted)

            # Check if the answer contains a canonical code block and update active coding state
            cat_lower = (category or "").lower()
            code_match = re.search(r"```([a-zA-Z0-9_\+\#-]*)\r?\n([\s\S]*?)```", answer)
            if code_match and cat_lower in ("coding", "sql", "debugging", "coding_optimization", "algorithm"):
                lang = code_match.group(1).strip().lower()
                # Do not overwrite canonical solution with pseudocode or plain text blocks
                if lang not in ("text", "pseudocode", "plain"):
                    self.active_coding_solution = code_match.group(0).strip()
                    # Preserve existing descriptive active_coding_problem if this is an optimization turn or imperative command
                    q_clean = question.lower().strip().rstrip(".?!")
                    imperative_commands = {
                        "give me the code", "give me code", "show me the code", "show the code",
                        "implement this", "implement that", "implement it", "now code it", "now code this",
                        "code this", "code that", "code it", "code it up", "write the solution", "code the solution",
                        "write a solution", "write the code", "can you code this", "can you implement this",
                        "solve this", "solve it", "can you solve this", "optimize this", "optimize that",
                        "can you optimize it", "can we optimize this", "optimize your solution", "make this faster",
                        "optimize the approach", "optimize your approach", "can we improve the time complexity",
                        "can we improve the space complexity", "improve the time complexity", "improve time complexity",
                        "improve space complexity"
                    }
                    is_follow_up_coding = (cat_lower == "coding_optimization" or q_clean in imperative_commands)
                    if not (self.active_coding_problem and is_follow_up_coding):
                        self.active_coding_problem = question
                    if lang:
                        self.active_coding_language = lang.capitalize()

        if hasattr(self, "controller") and self.controller:
            try:
                self.controller.set_state(InterviewTurnState.LISTENING)
            except Exception:
                pass

    def reset_coding_state(self):
        """Resets the active coding problem and solution state for a new problem."""
        with self._state_lock:
            self.active_coding_problem = ""
            self.active_coding_solution = ""
            self.active_coding_language = "Python"
            self.active_coding_approach = ""

    def add_qa(self, question: str, answer: str, category: str = "general"):
        """Alias for record_qa for backward and bridge compatibility."""
        return self.record_qa(question=question, answer=answer, category=category)

    def clear(self):
        """Clears all session transcripts, QA pairs, and analytics data."""
        with self._state_lock:
            self.transcript_log.clear()
            self.qa_pairs.clear()
            self.question_analytics.clear()
            self.candidate_responses_log.clear()
            self.latency_traces.clear()
            self._candidate_responses.clear()
            self._cumulative_analytics = {
                "total_questions_pruned": 0,
                "category_counts": {},
                "total_generation_time": 0.0,
                "gen_time_count": 0,
                "total_words": 0,
                "word_count": 0,
                "refinements": 0,
                "clarifications": 0,
                "low_conf": 0,
                "topics": set(),
            }
            self.active_topic = ""
            self.last_question = ""
            self.active_coding_problem = ""
            self.active_coding_solution = ""
            self.active_coding_language = "Python"
            self.active_coding_approach = ""
            self.screen_context_cache = None
            self.mock_screen_data = None
            self.active_question_id = ""
            self.question_counter = 0
            self.turn_state = InterviewTurnState.IDLE
            self.latest_detected_question = None
            self.latest_answer = ""
            self.pending_ambiguous_utterance = None
            self.pending_medium_question = None
        if hasattr(self, "controller") and self.controller:
            self.controller.reset()

    def clear_history(self):
        """Alias for clear()."""
        return self.clear()

    def get_latency_summary(self) -> Dict[str, Any]:
        """Returns session latency percentiles summary from session controller."""
        return self.controller.get_latency_summary() if self.controller else {}

    def get_error_summary(self) -> Dict[str, Any]:
        """Returns structured error summary from session controller."""
        return self.controller.get_error_summary() if self.controller else {}

    def get_audio_quality_report(self) -> Dict[str, Any]:
        """Returns audio quality report from continuous transcriber."""
        return self.controller.get_audio_quality_report() if self.controller else {}

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Returns atomic system telemetry snapshot from session controller."""
        return self.controller.get_telemetry_snapshot() if self.controller else {}


def calculate_session_analytics(session: InterviewSession) -> Dict[str, Any]:
    """
    Computes comprehensive post-interview analytics:
    - Total, technical, coding, behavioral, HR, project, system design questions
    - Generation time & answer length averages
    - Refinement, clarification, and low-confidence counts
    - Topics discussed and targeted preparation areas
    - Category 9 Telemetry: Latency percentiles (p50/p95/p99), audio quality, error summary.
    - Category 8 Long-Session Resilience: Losslessly incorporates cumulative pruned metrics.
    """
    records = list(session.question_analytics) if getattr(session, "question_analytics", None) else []
    cum = getattr(session, "_cumulative_analytics", None) or {}
    pruned_q = cum.get("total_questions_pruned", 0)
    
    total_q = (len(records) if records else len(getattr(session, "qa_pairs", []))) + pruned_q
    
    latency_summary = session.get_latency_summary() if hasattr(session, "get_latency_summary") else {}
    audio_summary = session.get_audio_quality_report() if hasattr(session, "get_audio_quality_report") else {}
    error_summary = session.get_error_summary() if hasattr(session, "get_error_summary") else {}

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
            "preparation_areas": [],
            "latency_summary": latency_summary,
            "audio_quality_summary": audio_summary,
            "error_summary": error_summary
        }
        
    cum_cats = cum.get("category_counts", {})
    cum_tech = sum(v for k, v in cum_cats.items() if "tech" in k or "database" in k or "cloud" in k)
    cum_coding = sum(v for k, v in cum_cats.items() if "coding" in k)
    cum_behavioral = sum(v for k, v in cum_cats.items() if "behavioral" in k)
    cum_hr = sum(v for k, v in cum_cats.items() if "hr" in k or "resume" in k)
    cum_project = sum(v for k, v in cum_cats.items() if "project" in k or "experience" in k)
    cum_sd = sum(v for k, v in cum_cats.items() if "system_design" in k)

    tech_count = sum(1 for r in records if "tech" in r.get("category", "") or "database" in r.get("category", "") or "cloud" in r.get("category", "")) + cum_tech
    coding_count = sum(1 for r in records if "coding" in r.get("category", "")) + cum_coding
    behavioral_count = sum(1 for r in records if "behavioral" in r.get("category", "")) + cum_behavioral
    hr_count = sum(1 for r in records if "hr" in r.get("category", "") or "resume" in r.get("category", "")) + cum_hr
    project_count = sum(1 for r in records if "project" in r.get("category", "") or "experience" in r.get("category", "")) + cum_project
    sd_count = sum(1 for r in records if "system_design" in r.get("category", "")) + cum_sd
    
    gen_times = [r.get("generation_time_sec", 0.0) for r in records if r.get("generation_time_sec", 0.0) > 0]
    total_gen_time = sum(gen_times) + cum.get("total_generation_time", 0.0)
    total_gen_samples = len(gen_times) + cum.get("gen_time_count", 0)
    avg_gen_time = round(total_gen_time / total_gen_samples, 2) if total_gen_samples > 0 else 0.85
    
    word_counts = [r.get("word_count", 0) for r in records if r.get("word_count", 0) > 0]
    total_words = sum(word_counts) + cum.get("total_words", 0)
    total_word_samples = len(word_counts) + cum.get("word_count", 0)
    avg_words = int(total_words / total_word_samples) if total_word_samples > 0 else 180
    
    refinements = sum(1 for r in records if r.get("required_refinement", False)) + cum.get("refinements", 0)
    clarifications = sum(1 for r in records if r.get("required_clarification", False) or "clarification" in r.get("category", "")) + cum.get("clarifications", 0)
    low_conf = sum(1 for r in records if r.get("confidence", 1.0) < 0.75) + cum.get("low_conf", 0)
    
    # Collect unique topics discussed
    topics = []
    for r in records:
        t = r.get("topic", "").strip()
        if t and t != "General" and t not in topics:
            topics.append(t)
    for t in cum.get("topics", set()):
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
        "preparation_areas": prep_areas,
        "latency_summary": latency_summary,
        "audio_quality_summary": audio_summary,
        "error_summary": error_summary
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

