"""
BluJay AI - Continuous Audio Transcriber & STT Provider Pipeline
================================================================
Manages multi-source audio capture (system loopback & microphone),
real-time Groq Whisper transcription, turn segmentation, and streaming triggers.
"""

import os
import io
import time
import wave
import uuid
import queue
import threading
import collections
import concurrent.futures
import sys
import atexit
import signal
import weakref
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, List, Tuple, Union

import sounddevice as sd
import numpy as np
try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except Exception:
    SOUNDCARD_AVAILABLE = False

from groq import Groq
from speech import (
    SystemAudioProcessor,
    MicrophoneAudioProcessor,
    merge_overlapping_transcripts,
    apply_technical_asr_corrections,
    build_dynamic_whisper_prompt,
    is_conversational_chitchat
)
from engine.utils import get_openai_client, load_user_config
from engine.session_models import (
    InterviewSession,
    InterviewTurnState,
    ComponentHealthState,
    AudioFrame,
    AudioSegment,
    QuestionLatencyTrace,
    categorize_api_error
)
from engine.relationship_classifier import (
    clean_asr_transcript,
    compute_question_confidence,
    is_likely_question,
    refine_interview_question,
    classify_answer_relationship,
    evaluate_question_completeness,
    compute_question_likelihood
)
from engine.turn_taking import (
    detect_speaker_turn,
    classify_and_understand_utterance
)
from engine.llm_orchestrator import generate_interview_answer

class STTProvider(ABC):
    """Abstract interface for speech-to-text providers."""
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "") -> str:
        pass


class GroqWhisperProvider(STTProvider):
    """Production Groq Whisper implementation (whisper-large-v3-turbo) with automatic retry, rate-limit backoff, and fallback."""
    def __init__(self, client: Groq, model: str = "whisper-large-v3-turbo", default_language: Optional[str] = "en"):
        self.client = client
        self.model = model
        self.default_language = default_language
        self._rate_limit_cooldown_until = 0.0

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "", language: Optional[str] = None) -> str:
        # Skip ultra-short bursts (< 50ms = 800 samples at 16kHz = 1600 bytes int16)
        if not audio_bytes or len(audio_bytes) < 1600:
            return ""
            
        # Reject pure room ambient noise / fan static to prevent Whisper hallucination loops
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(samples) > 0:
            rms = float(np.sqrt(np.mean(samples.astype(np.float32)**2)))
            if rms < 110.0:
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
        effective_prompt = prompt or build_dynamic_whisper_prompt()
        kwargs = {
            "file": ("audio.wav", wav_bytes),
            "model": self.model,
            "response_format": "text",
            "temperature": 0.0,
            "prompt": effective_prompt
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
                raw_text = transcription.strip() if isinstance(transcription, str) else ""
                # Apply phonetic technical ASR post-correction and hallucination filtering
                return apply_technical_asr_corrections(clean_asr_transcript(raw_text))
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


class ResilientMultiSTTProvider(STTProvider):
    """
    Tiered multi-provider STT with transparent instant failover:
    - Tier 1 (PRIMARY): Deepgram Nova-3 (~120-150ms ultra-low latency)
    - Tier 2 (HARDWARE ACCELERATION): Groq Whisper (whisper-large-v3-turbo, ~150-250ms)
    - Tier 3 (FALLBACK): OpenAI Whisper (whisper-1) if Deepgram & Groq are unavailable
    - Tier 4 (FALLBACK): Google Gemini Multimodal Audio if GEMINI_API_KEY is available
    """
    def __init__(self, groq_client: Optional[Groq] = None, default_language: Optional[str] = "en"):
        if groq_client:
            self.groq_provider = GroqWhisperProvider(client=groq_client, default_language=default_language)
        else:
            api_k = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
            if api_k:
                try:
                    from groq import Groq
                    self.groq_provider = GroqWhisperProvider(client=Groq(api_key=api_k), default_language=default_language)
                except Exception:
                    self.groq_provider = None
            else:
                self.groq_provider = None

        self.default_language = default_language
        self._openai_client = None
        self._deepgram_session = None
        self._deepgram_key = None
        self._gemini_client = None
        self._gemini_cooldown_until = 0.0

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "", language: Optional[str] = None) -> str:
        if not audio_bytes or len(audio_bytes) < 1600:
            return ""

        # Reject pure room ambient noise / fan static to prevent Whisper hallucination loops
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(samples) > 0:
            rms = float(np.sqrt(np.mean(samples.astype(np.float32)**2)))
            if rms < 110.0:
                return ""

        # Tier 1 (PRIMARY): Deepgram Nova-3 (~120-150ms)
        deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if deepgram_key:
            try:
                if self._deepgram_session is None or self._deepgram_key != deepgram_key:
                    import requests
                    self._deepgram_session = requests.Session()
                    self._deepgram_session.headers.update({
                        "Authorization": f"Token {deepgram_key}",
                        "Content-Type": "audio/wav"
                    })
                    self._deepgram_key = deepgram_key

                dg_model = os.getenv("DEFAULT_STT_MODEL", "nova-3").strip()
                if "flux" in dg_model.lower():
                    dg_model = "nova-3"
                dg_url = f"https://api.deepgram.com/v1/listen?model={dg_model}&smart_format=true&punctuate=true"
                req_lang = language or self.default_language or "en"
                if req_lang and req_lang.lower() not in ["auto", "none", ""]:
                    dg_url += f"&language={req_lang.lower()}"
                
                keywords = ["Kubernetes:2", "FastAPI:2", "PostgreSQL:2", "Docker:2", "JWT:2", "gRPC:2", "Redis:2", "GraphQL:2"]
                for kw in keywords:
                    dg_url += f"&keywords={kw}"

                import io, requests
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_bytes)
                wav_bytes = wav_io.getvalue()
                try:
                    resp = self._deepgram_session.post(dg_url, data=wav_bytes, timeout=3.5)
                except Exception:
                    self._deepgram_session = requests.Session()
                    self._deepgram_session.headers.update({
                        "Authorization": f"Token {deepgram_key}",
                        "Content-Type": "audio/wav"
                    })
                    resp = self._deepgram_session.post(dg_url, data=wav_bytes, timeout=3.5)
                if resp.status_code == 200:
                    res_json = resp.json()
                    channels = res_json.get("results", {}).get("channels", [])
                    if channels and channels[0].get("alternatives"):
                        text = channels[0]["alternatives"][0].get("transcript", "").strip()
                        if text:
                            return apply_technical_asr_corrections(clean_asr_transcript(text))
                    return ""
            except Exception as dg_e:
                print(f"Deepgram STT notice: {dg_e}, falling back to Groq/OpenAI...")

        # Tier 2 (HARDWARE ACCELERATION): Groq Whisper (whisper-large-v3-turbo, ~150-250ms)
        if self.groq_provider:
            try:
                result = self.groq_provider.transcribe(audio_bytes, sample_rate=sample_rate, prompt=prompt, language=language)
                if result and result.strip():
                    return result
            except Exception as e:
                print(f"Groq STT notice: {e}, falling back to OpenAI/Gemini...")

        # Tier 2 (FALLBACK): OpenAI Whisper (whisper-1) when Groq is unavailable or fails
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            try:
                if self._openai_client is None:
                    import importlib
                    openai_mod = importlib.import_module("openai")
                    self._openai_client = openai_mod.OpenAI(api_key=openai_key)
                
                import io
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_bytes)
                wav_bytes = wav_io.getvalue()
                
                effective_prompt = prompt or build_dynamic_whisper_prompt()
                resp = self._openai_client.audio.transcriptions.create(
                    file=("audio.wav", wav_bytes),
                    model="whisper-1",
                    response_format="text",
                    temperature=0.0,
                    prompt=effective_prompt
                )
                text = resp.strip() if isinstance(resp, str) else getattr(resp, "text", "").strip()
                if text:
                    return apply_technical_asr_corrections(clean_asr_transcript(text))
            except Exception as e:
                print(f"OpenAI STT notice: {e}, falling back to Gemini...")

        # Tier 3 (FALLBACK): Gemini Audio Fallback
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key and time.time() > self._gemini_cooldown_until:
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore
                if self._gemini_client is None:
                    self._gemini_client = genai.Client(api_key=gemini_key)
                
                import io
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_bytes)
                wav_bytes = wav_io.getvalue()
                
                gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
                g_resp = self._gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=[
                        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                        "Transcribe the spoken speech in this audio accurately. Return ONLY the transcribed text verbatim, with no commentary or extra formatting."
                    ]
                )
                if g_resp and g_resp.text:
                    text = g_resp.text.strip()
                    if text:
                        return apply_technical_asr_corrections(clean_asr_transcript(text))
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    self._gemini_cooldown_until = time.time() + 60.0
                    print("Gemini STT free tier quota limit reached -> backing off for 60s.")
                else:
                    print(f"Tier 3 Gemini STT fallback notice: {e}")

        return ""


# ============================================================
# CONTINUOUS AUDIO STREAMING & LOW-LATENCY TRANSCRIBER
# ============================================================

# Global Active Continuous Transcribers Registry for guaranteed graceful teardown
_ACTIVE_TRANSCRIBERS: weakref.WeakSet = weakref.WeakSet()
_ACTIVE_TRANSCRIBERS_LOCK = threading.Lock()


def _shutdown_active_transcribers():
    """Shuts down all active ContinuousTranscriber instances to prevent orphaned audio threads/sockets."""
    with _ACTIVE_TRANSCRIBERS_LOCK:
        active = list(_ACTIVE_TRANSCRIBERS)
    for t in active:
        try:
            t.stop()
        except Exception:
            pass


atexit.register(_shutdown_active_transcribers)


def _install_graceful_signal_handlers():
    """Installs Windows/POSIX-compatible graceful signal handlers on the main interpreter thread."""
    if threading.current_thread() is not threading.main_thread():
        return
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            prev_handler = signal.getsignal(sig)

            def _handler(signum, frame, old_h=prev_handler):
                _shutdown_active_transcribers()
                if callable(old_h) and old_h not in (signal.SIG_DFL, signal.SIG_IGN):
                    old_h(signum, frame)
                elif old_h == signal.SIG_DFL:
                    sys.exit(0)

            signal.signal(sig, _handler)
        except (ValueError, OSError, AttributeError):
            pass


_install_graceful_signal_handlers()


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
        system_speaker_id: Optional[str] = None,
        on_turn_resumed: Optional[Callable[[str], None]] = None,
        on_eager_eot: Optional[Callable[[str, str], None]] = None
    ):
        self.client = client
        self.on_transcript_chunk = on_transcript_chunk
        self.on_status_change = on_status_change
        self.on_interim_transcript = on_interim_transcript
        self.on_audio_level = on_audio_level
        self.provider = provider or ResilientMultiSTTProvider(groq_client=client)
        self.session = session
        self.enable_watchdog = enable_watchdog
        self.mic_device_id = mic_device_id
        self.system_speaker_id = system_speaker_id
        self.on_turn_resumed = on_turn_resumed
        self.on_eager_eot = on_eager_eot
        
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
        self._last_system_frame_time: float = 0.0
        self._last_mic_frame_time: float = 0.0
        self.total_system_frames: int = 0
        self.total_mic_frames: int = 0
        self.system_speech_frames: int = 0
        self.mic_speech_frames: int = 0
        self.cumulative_system_rms: float = 0.0
        self.cumulative_mic_rms: float = 0.0
        self.mute_toggle_count: int = 0
        self._last_mute_state: Optional[bool] = None
        self.is_mic_muted: bool = False
        self._device_watcher = None
        self._init_start_time: float = 0.0
        self.audio_init_latency_ms: float = 0.0
        if self.session:
            self.session.transcriber = self
        
        self.sample_rate = 16000
        self.channels = 1
        self._interim_in_flight = False
        self._current_speech_start = 0.0
        self._last_candidate_speech_time = 0.0
        self._is_candidate_semantically_complete = False
        self._stt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="STTWorker")
        self.live_dg_ws = None
        self.live_dg_ws_system = None
        self.live_dg_ws_mic = None
        self._audio_stream_epoch: int = 0
        self._last_system_speech_time: float = 0.0
        self._last_system_speech_rms: float = 0.0
        self._last_interim_text: str = ""
        self._stt_seq_counter: int = 0
        self._stt_next_seq: int = 0
        self._stt_pending_results: Dict[int, Tuple[str, bool, str]] = {}
        self._stt_order_lock = threading.Lock()
        
        # Pre-warm shared ONNX VAD session during transcriber initialization so start() is instant (<15ms)
        try:
            from speech import get_silero_vad
            get_silero_vad()
        except Exception:
            pass

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
        with _ACTIVE_TRANSCRIBERS_LOCK:
            _ACTIVE_TRANSCRIBERS.add(self)
        self._audio_stream_epoch += 1
        current_epoch = self._audio_stream_epoch
        
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
            else:
                self.on_status_change("🎙️ Live Listening (Microphone)...")
                
        now = time.perf_counter()
        
        # Initialize and start Live Deepgram WebSocket client(s) for instant word-by-word streaming
        deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if deepgram_key:
            try:
                from speech import DeepgramLiveWebSocketClient
                if source == "dual":
                    self.live_dg_ws_system = DeepgramLiveWebSocketClient(
                        api_key=deepgram_key,
                        sample_rate=self.sample_rate,
                        on_interim=self._on_live_ws_interim,
                        on_final=self._on_live_ws_final,
                        on_eager_eot=self._on_live_ws_eager_eot,
                        on_turn_resumed=self._on_live_ws_turn_resumed,
                        source_label="system"
                    )
                    self.live_dg_ws_system.start()

                    self.live_dg_ws_mic = DeepgramLiveWebSocketClient(
                        api_key=deepgram_key,
                        sample_rate=self.sample_rate,
                        on_interim=self._on_live_ws_interim,
                        on_final=self._on_live_ws_final,
                        on_eager_eot=self._on_live_ws_eager_eot,
                        on_turn_resumed=self._on_live_ws_turn_resumed,
                        source_label="candidate_mic"
                    )
                    self.live_dg_ws_mic.start()
                    self.live_dg_ws = self.live_dg_ws_system
                elif source in ("system", "loopback"):
                    self.live_dg_ws_system = DeepgramLiveWebSocketClient(
                        api_key=deepgram_key,
                        sample_rate=self.sample_rate,
                        on_interim=self._on_live_ws_interim,
                        on_final=self._on_live_ws_final,
                        on_eager_eot=self._on_live_ws_eager_eot,
                        on_turn_resumed=self._on_live_ws_turn_resumed,
                        source_label="system"
                    )
                    self.live_dg_ws_system.start()
                    self.live_dg_ws_mic = None
                    self.live_dg_ws = self.live_dg_ws_system
                else:  # source == "mic"
                    self.live_dg_ws_mic = DeepgramLiveWebSocketClient(
                        api_key=deepgram_key,
                        sample_rate=self.sample_rate,
                        on_interim=self._on_live_ws_interim,
                        on_final=self._on_live_ws_final,
                        on_eager_eot=self._on_live_ws_eager_eot,
                        on_turn_resumed=self._on_live_ws_turn_resumed,
                        source_label="candidate_mic"
                    )
                    self.live_dg_ws_mic.start()
                    self.live_dg_ws_system = None
                    self.live_dg_ws = self.live_dg_ws_mic
            except Exception as e:
                print(f"Deepgram Live WebSocket start notice: {e}")
        
        # Start capture and processing workers per selected source with epoch tag
        if source == "dual":
            if SOUNDCARD_AVAILABLE:
                sys_cap_id = f"sys_cap_{uuid.uuid4().hex[:6]}"
                sys_proc_id = f"sys_proc_{uuid.uuid4().hex[:6]}"
                self.active_workers[sys_cap_id] = {"source": "system", "role": "capture", "start_time": now}
                self.active_workers[sys_proc_id] = {"source": "system", "role": "processor", "start_time": now}
                
                self.system_processor_thread = threading.Thread(
                    target=lambda epoch=current_epoch, pid=sys_proc_id: self._system_processing_worker(pid, epoch=epoch),
                    daemon=True,
                    name=f"SystemAudioProcessor-{sys_proc_id}"
                )
                self.system_processor_thread.start()
                
                self.system_thread = threading.Thread(
                    target=lambda epoch=current_epoch, cid=sys_cap_id: self._run_system_audio_loop(cid, epoch=epoch),
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
                target=lambda epoch=current_epoch, pid=mic_proc_id: self._mic_processing_worker(channel_source="candidate_mic", worker_id=pid, epoch=epoch),
                daemon=True,
                name=f"MicProcessor-{mic_proc_id}"
            )
            self.mic_processor_thread.start()
            
            self.mic_thread = threading.Thread(
                target=lambda epoch=current_epoch, cid=mic_cap_id: self._run_microphone_loop(channel_source="candidate_mic", worker_id=cid, epoch=epoch),
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
                    target=lambda epoch=current_epoch, pid=sys_proc_id: self._system_processing_worker(pid, epoch=epoch),
                    daemon=True,
                    name=f"SystemAudioProcessor-{sys_proc_id}"
                )
                self.system_processor_thread.start()
                
                self.system_thread = threading.Thread(
                    target=lambda epoch=current_epoch, cid=sys_cap_id: self._run_system_audio_loop(cid, epoch=epoch),
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
                target=lambda epoch=current_epoch, pid=mic_proc_id: self._mic_processing_worker(channel_source="mic", worker_id=pid, epoch=epoch),
                daemon=True,
                name=f"MicProcessor-{mic_proc_id}"
            )
            self.mic_processor_thread.start()
            
            self.mic_thread = threading.Thread(
                target=lambda epoch=current_epoch, cid=mic_cap_id: self._run_microphone_loop(channel_source="mic", worker_id=cid, epoch=epoch),
                daemon=True,
                name=f"MicSingleLoop-{mic_cap_id}"
            )
            self.mic_thread.start()

        # Start audio watchdog for self-healing dead threads
        if self.enable_watchdog:
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="AudioWatchdogThread")
            self.watchdog_thread.start()

        # Start device watcher for auto-recovery on Bluetooth / USB / default device switch.
        # Deferred to a background thread so start() returns immediately without device probing.
        def _start_device_watcher_deferred():
            try:
                from audio_device_manager import AudioDeviceWatcher
                def _on_def_speaker_change(new_spk_name):
                    if self.is_running and self.source in ("dual", "system", "loopback"):
                        if self.system_speaker_id in (None, "default"):
                            if self.on_status_change:
                                try:
                                    self.on_status_change(f"↻ Audio output switched: {new_spk_name}")
                                except Exception:
                                    pass
                            self.switch_devices(system_speaker_id="default")

                def _on_def_mic_change(new_mic_id):
                    if self.is_running and self.source in ("dual", "mic"):
                        if self.mic_device_id in (None, "default"):
                            if self.on_status_change:
                                try:
                                    self.on_status_change(f"↻ Microphone switched: {new_mic_id}")
                                except Exception:
                                    pass
                            self.switch_devices(mic_device_id="default")

                if not self.is_running or self._stop_event.is_set():
                    return
                watcher = AudioDeviceWatcher(
                    on_default_speaker_changed=_on_def_speaker_change,
                    on_default_mic_changed=_on_def_mic_change,
                    check_interval=2.0
                )
                if not self.is_running or self._stop_event.is_set():
                    return
                self._device_watcher = watcher
                watcher.start()
            except Exception:
                self._device_watcher = None

        threading.Thread(target=_start_device_watcher_deferred, daemon=True, name="DeviceWatcherDeferred").start()

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        with _ACTIVE_TRANSCRIBERS_LOCK:
            _ACTIVE_TRANSCRIBERS.discard(self)
        now = time.perf_counter()
        for w_id in list(self.active_workers.keys()):
            self.active_workers[w_id]["stop_time"] = now
            
        for ws_attr in ["live_dg_ws_system", "live_dg_ws_mic", "live_dg_ws"]:
            ws_inst = getattr(self, ws_attr, None)
            if ws_inst:
                try:
                    ws_inst.stop()
                except Exception:
                    pass
                setattr(self, ws_attr, None)

        if getattr(self, "_device_watcher", None):
            try:
                self._device_watcher.stop()
            except Exception:
                pass
            self._device_watcher = None
            
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
        
        if hasattr(self, "_stt_executor") and self._stt_executor:
            try:
                self._stt_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._stt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="STTWorker")
        with getattr(self, "_stt_order_lock", threading.Lock()):
            if hasattr(self, "_stt_pending_results"):
                self._stt_pending_results.clear()
            self._stt_seq_counter = 0
            self._stt_next_seq = 0
        self._last_interim_text = ""
        
        if self.session and hasattr(self.session, "controller") and self.session.controller:
            self.session.controller.set_state(InterviewTurnState.STOPPED)
            self.session.controller.update_component("audio_capture", "IDLE")
        if self.on_status_change:
            self.on_status_change("⏹️ Stopped")

    def switch_devices(
        self,
        mic_device_id: Optional[Union[int, str]] = None,
        system_speaker_id: Optional[str] = None
    ) -> bool:
        """
        Dynamically hot-swaps active audio capture devices (Bluetooth headphones, USB headset, etc.)
        in < 1-2 seconds without restarting the interview session, dropping transcripts, or losing state.
        """
        if not self.is_running:
            if mic_device_id is not None:
                self.mic_device_id = mic_device_id
            if system_speaker_id is not None:
                self.system_speaker_id = system_speaker_id
            return True

        now = time.perf_counter()
        if mic_device_id is not None:
            self.mic_device_id = mic_device_id
        if system_speaker_id is not None:
            self.system_speaker_id = system_speaker_id

        # Bump epoch to invalidate in-flight frames from previous devices
        self._audio_stream_epoch += 1
        new_epoch = self._audio_stream_epoch

        # Clear stale queued frames
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

        # Restart system capture if active
        if self.source in ("dual", "system", "loopback") and SOUNDCARD_AVAILABLE:
            self._last_system_frame_time = now
            sys_cap_id = f"sys_cap_sw_{uuid.uuid4().hex[:6]}"
            self.active_workers[sys_cap_id] = {"source": "system", "role": "capture", "start_time": now}
            self.system_thread = threading.Thread(
                target=lambda epoch=new_epoch, cid=sys_cap_id: self._run_system_audio_loop(cid, epoch=epoch),
                daemon=True,
                name=f"SystemAudioCapture-{sys_cap_id}"
            )
            self.system_thread.start()

        # Restart mic capture if active
        if self.source in ("dual", "mic"):
            self._last_mic_frame_time = now
            channel_src = "candidate_mic" if self.source == "dual" else "mic"
            mic_cap_id = f"mic_cap_sw_{uuid.uuid4().hex[:6]}"
            self.active_workers[mic_cap_id] = {"source": channel_src, "role": "capture", "start_time": now}
            self.mic_thread = threading.Thread(
                target=lambda epoch=new_epoch, cid=mic_cap_id: self._run_microphone_loop(channel_source=channel_src, worker_id=cid, epoch=epoch),
                daemon=True,
                name=f"MicHotSwapLoop-{mic_cap_id}"
            )
            self.mic_thread.start()

        if self.on_status_change:
            try:
                self.on_status_change("↻ Audio devices switched successfully")
            except Exception:
                pass
        return True

    def get_audio_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive real-time audio capture health metrics."""
        total_frames = self.total_system_frames + self.total_mic_frames
        dropout_pct = (self.dropped_frames_count / max(1, total_frames + self.dropped_frames_count)) * 100.0
        return {
            "is_running": self.is_running,
            "source": self.source,
            "stream_epoch": self._audio_stream_epoch,
            "total_system_frames": self.total_system_frames,
            "total_mic_frames": self.total_mic_frames,
            "total_frames": total_frames,
            "dropped_frames_count": self.dropped_frames_count,
            "discontinuity_count": self.discontinuity_count,
            "dropout_rate_pct": round(dropout_pct, 4),
            "audio_init_latency_ms": round(self.audio_init_latency_ms, 2),
            "is_mic_muted": self.is_mic_muted,
            "mic_device_id": self.mic_device_id,
            "system_speaker_id": self.system_speaker_id
        }

    def get_audio_quality_report(self) -> Dict[str, Any]:
        """
        Computes a comprehensive, session-wide audio quality report:
        - Total frames processed (system + mic)
        - Dropped frames and dropout rate percentage
        - System and mic capture availability percentage
        - VAD speech hit rate (speech frames vs total frames)
        - Hardware discontinuity count
        - Average RMS for mic and system
        - Microphone mute telemetry (current state and toggle count)
        """
        total_frames = self.total_system_frames + self.total_mic_frames
        total_dropouts = self.dropped_frames_count
        dropout_rate = (total_dropouts / max(1, total_frames + total_dropouts)) * 100.0
        
        sys_speech_ratio = (self.system_speech_frames / max(1, self.total_system_frames)) * 100.0
        mic_speech_ratio = (self.mic_speech_frames / max(1, self.total_mic_frames)) * 100.0
        overall_speech_ratio = ((self.system_speech_frames + self.mic_speech_frames) / max(1, total_frames)) * 100.0
        
        avg_sys_rms = round(self.cumulative_system_rms / max(1, self.total_system_frames), 2)
        avg_mic_rms = round(self.cumulative_mic_rms / max(1, self.total_mic_frames), 2)
        
        return {
            "total_frames": total_frames,
            "total_system_frames": self.total_system_frames,
            "total_mic_frames": self.total_mic_frames,
            "dropped_frames": total_dropouts,
            "dropout_rate_pct": round(dropout_rate, 4),
            "availability_pct": round(100.0 - dropout_rate, 2),
            "discontinuity_count": self.discontinuity_count,
            "system_speech_frames": self.system_speech_frames,
            "mic_speech_frames": self.mic_speech_frames,
            "system_speech_ratio_pct": round(sys_speech_ratio, 2),
            "mic_speech_ratio_pct": round(mic_speech_ratio, 2),
            "overall_speech_ratio_pct": round(overall_speech_ratio, 2),
            "avg_system_rms": avg_sys_rms,
            "avg_mic_rms": avg_mic_rms,
            "is_mic_muted": self.is_mic_muted,
            "mute_toggle_count": self.mute_toggle_count,
            "audio_init_latency_ms": round(self.audio_init_latency_ms, 2)
        }

    def _watchdog_loop(self):
        """Lightweight watchdog thread that monitors audio streams and auto-recovers dead capture workers."""
        while self.is_running and not self._stop_event.is_set():
            time.sleep(2.0)
            if not self.is_running or self._stop_event.is_set():
                break
                
            try:
                now = time.perf_counter()
                cur_epoch = self._audio_stream_epoch

                # 0. Check microphone mute state (Windows Core Audio & hardware mute)
                try:
                    from audio_device_manager import get_system_microphone_mute_state
                    sys_muted = get_system_microphone_mute_state()
                    if sys_muted is not None and sys_muted != self.is_mic_muted:
                        self.is_mic_muted = sys_muted
                        self.mute_toggle_count += 1
                        if self.on_status_change:
                            if self.is_mic_muted:
                                self.on_status_change("⚠️ Microphone is muted in system settings")
                            else:
                                self.on_status_change("🎙️ Microphone unmuted")
                except Exception:
                    pass

                # 1. Check and recover System Audio Loopback workers
                if self.source in ("dual", "system", "loopback") and SOUNDCARD_AVAILABLE:
                    sys_proc_dead = bool(self.system_processor_thread and not self.system_processor_thread.is_alive())
                    sys_cap_dead = bool(self.system_thread and not self.system_thread.is_alive())
                    # Silent-freeze detection: thread alive but no frame recorded for > 3.5s
                    sys_stalled = bool(self.system_thread and self.system_thread.is_alive() and (now - self._last_system_frame_time > 3.5) and self._last_system_frame_time > 0)

                    if sys_proc_dead and self.is_running:
                        sys_proc_id = f"sys_proc_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[sys_proc_id] = {"source": "system", "role": "processor", "start_time": now}
                        self.system_processor_thread = threading.Thread(
                            target=lambda epoch=cur_epoch, pid=sys_proc_id: self._system_processing_worker(pid, epoch=epoch),
                            daemon=True,
                            name=f"SystemAudioProcessor-{sys_proc_id}"
                        )
                        self.system_processor_thread.start()
                        
                    if (sys_cap_dead or sys_stalled) and self.is_running:
                        if self.on_status_change:
                            try:
                                self.on_status_change("↻ Recovering system audio capture...")
                            except Exception:
                                pass
                        if self.session and hasattr(self.session, "controller") and self.session.controller:
                            self.session.controller.update_component("audio_capture", "RECOVERING", "System capture thread restarted")
                        
                        if sys_stalled:
                            self._audio_stream_epoch += 1
                            cur_epoch = self._audio_stream_epoch

                        self._last_system_frame_time = now
                        sys_cap_id = f"sys_cap_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[sys_cap_id] = {"source": "system", "role": "capture", "start_time": now}
                        self.system_thread = threading.Thread(
                            target=lambda epoch=cur_epoch, cid=sys_cap_id: self._run_system_audio_loop(cid, epoch=epoch),
                            daemon=True,
                            name=f"SystemAudioCapture-{sys_cap_id}"
                        )
                        self.system_thread.start()

                # 2. Check and recover Microphone workers
                if self.source in ("dual", "mic"):
                    channel_src = "candidate_mic" if self.source == "dual" else "mic"
                    mic_proc_dead = bool(self.mic_processor_thread and not self.mic_processor_thread.is_alive())
                    mic_cap_dead = bool(self.mic_thread and not self.mic_thread.is_alive())
                    mic_stalled = bool(self.mic_thread and self.mic_thread.is_alive() and (now - self._last_mic_frame_time > 3.5) and self._last_mic_frame_time > 0)

                    if mic_proc_dead and self.is_running:
                        mic_proc_id = f"mic_proc_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[mic_proc_id] = {"source": channel_src, "role": "processor", "start_time": now}
                        self.mic_processor_thread = threading.Thread(
                            target=lambda epoch=cur_epoch, pid=mic_proc_id: self._mic_processing_worker(channel_source=channel_src, worker_id=pid, epoch=epoch),
                            daemon=True,
                            name=f"MicProcessor-{mic_proc_id}"
                        )
                        self.mic_processor_thread.start()
                        
                    if (mic_cap_dead or mic_stalled) and self.is_running:
                        if self.on_status_change:
                            try:
                                self.on_status_change("↻ Recovering microphone capture...")
                            except Exception:
                                pass
                        if self.session and hasattr(self.session, "controller") and self.session.controller:
                            self.session.controller.update_component("audio_capture", "RECOVERING", "Microphone capture thread restarted")
                        
                        if mic_stalled:
                            self._audio_stream_epoch += 1
                            cur_epoch = self._audio_stream_epoch

                        self._last_mic_frame_time = now
                        mic_cap_id = f"mic_cap_rec_{uuid.uuid4().hex[:6]}"
                        self.active_workers[mic_cap_id] = {"source": channel_src, "role": "capture", "start_time": now}
                        self.mic_thread = threading.Thread(
                            target=lambda epoch=cur_epoch, cid=mic_cap_id: self._run_microphone_loop(channel_source=channel_src, worker_id=cid, epoch=epoch),
                            daemon=True,
                            name=f"MicCandidateLoop-{mic_cap_id}"
                        )
                        self.mic_thread.start()
                        
                # Update heartbeat
                if self.session and hasattr(self.session, "controller") and self.session.controller:
                    self.session.controller.record_heartbeat_event("last_audio_event")
            except Exception as e:
                print(f"Audio watchdog check notice: {e}")

    def _run_system_audio_loop(self, worker_id: str = "", epoch: int = 0):
        """
        Lightweight hardware capture loop for System Audio Loopback.
        ONLY: read audio frame, timestamp, enqueue, continue.
        NEVER performs DSP, VAD, STT, or LLM processing.
        """
        retry_delay = 0.1
        while self.is_running and not self._stop_event.is_set():
            if epoch != 0 and epoch != self._audio_stream_epoch:
                return
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
                
                # Issue 29: blocksize aligned with frame_size (480) for 16kHz 30ms frames
                with loopback.recorder(samplerate=self.sample_rate, channels=1, blocksize=frame_size) as recorder:
                    retry_delay = 0.1
                    discontinuity_flag = False
                    consecutive_errors = 0
                    while self.is_running and not self._stop_event.is_set():
                        if epoch != 0 and epoch != self._audio_stream_epoch:
                            return
                        ts_capture = time.perf_counter()
                        try:
                            data = recorder.record(numframes=frame_size)
                            consecutive_errors = 0
                            discontinuity_flag = False
                        except Exception as e:
                            self.discontinuity_count += 1
                            consecutive_errors += 1
                            discontinuity_flag = True
                            if self.session and hasattr(self.session, "controller") and self.session.controller:
                                try:
                                    self.session.controller.record_error("audio_capture", e, category="AUDIO_HARDWARE", recoverable=True)
                                except Exception:
                                    pass
                            # If device was invalidated (0x88890004 or unplugged) or 3 consecutive failures:
                            if consecutive_errors >= 3 or "AUDCLNT_E_DEVICE_INVALIDATED" in str(e) or "0x88890004" in str(e):
                                break  # Break out of inner recorder to re-acquire speaker loopback recorder
                            time.sleep(0.01)
                            continue
                            
                        if data is None or len(data) == 0:
                            time.sleep(0.005)
                            continue
                            
                        if isinstance(data, np.ndarray):
                            data = data.flatten()
                            
                        # Update frame heartbeat & telemetry
                        self._last_system_frame_time = ts_capture
                        self.total_system_frames += 1
                        if self.total_system_frames == 1 and self.audio_init_latency_ms == 0.0:
                            self.audio_init_latency_ms = (ts_capture - self._init_start_time) * 1000.0

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
                            source="system",
                            is_discontinuity=discontinuity_flag,
                            epoch=epoch
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
                    self.on_status_change(f"⚠️ Loopback audio error — reconnecting: {e}")
                if not self.is_running or self._stop_event.is_set():
                    return
                time.sleep(retry_delay)
                retry_delay = min(1.0, retry_delay * 1.5)

    def _system_processing_worker(self, worker_id: str = "", epoch: int = 0):
        """
        Dedicated processing worker for System Audio.
        Handles DSP, VAD, rolling buffers, speech segmentation, and STT dispatch.
        """
        processor = SystemAudioProcessor(sample_rate=self.sample_rate)
        default_silence_frames = 10    # ~300ms of silence (ultra-low latency turn-taking)
        min_speech_frames = 10         # ~300ms min speech (guards against ambient clicks/noise)
        interim_interval_frames = 12   # ~360ms interval (stable interim preview)
        
        audio_buffer = []
        pre_buffer = collections.deque(maxlen=10)  # ~300ms pre-speech buffer
        speech_frames_count = 0
        silence_frames_count = 0
        in_speech = False
        speech_start_perf = 0.0
        
        while self.is_running and not self._stop_event.is_set():
            if epoch != 0 and epoch != self._audio_stream_epoch:
                return
            try:
                frame: AudioFrame = self.system_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Drop late frames from previous stream epochs
            if frame.epoch != 0 and frame.epoch != self._audio_stream_epoch:
                continue
                
            proc_start_ts = time.perf_counter()
            proc_bytes, is_speech, diag = processor.process_frame(frame.data)
            self.total_system_frames += 1
            if is_speech:
                self.system_speech_frames += 1
            self.cumulative_system_rms += diag.rms
            
            ws_sys = getattr(self, "live_dg_ws_system", None) or self.live_dg_ws
            if ws_sys and ws_sys.is_running:
                if getattr(ws_sys, "_is_flux", False) or is_speech:
                    ws_sys.send_frame(proc_bytes)
            
            # Acoustic cross-talk suppression: if candidate is actively speaking into mic, discard faint speaker loopback echo
            if is_speech and self.source == "dual" and not in_speech and (time.perf_counter() - getattr(self, "_last_candidate_speech_time", 0.0) < 0.25) and diag.rms < 120.0:
                is_speech = False

            if is_speech:
                self._last_system_speech_time = time.perf_counter()
                self._last_system_speech_rms = diag.rms
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
                
                # Zero-Wait Endpointing: 2 frames (~60ms) if question is syntactically complete, else default 10 frames (~300ms)
                effective_silence = 2 if getattr(self, "_is_candidate_semantically_complete", False) else default_silence_frames
                
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
                    self._last_interim_text = ""
            else:
                pre_buffer.append(proc_bytes)
                
            # Prevent unbounded buffer if continuous speech > 45s
            if len(audio_buffer) * 30 > 45000:
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
                self._last_interim_text = ""

    def _run_microphone_loop(self, channel_source: str = "candidate_mic", worker_id: str = "", epoch: int = 0):
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
            if epoch != 0 and epoch != self._audio_stream_epoch:
                return
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
                        if epoch != 0 and epoch != self._audio_stream_epoch:
                            return
                        ts_capture = time.perf_counter()
                        frame, overflow = stream.read(frame_size)
                        if overflow:
                            self.discontinuity_count += 1
                        frame_bytes = bytes(frame)
                        
                        # Update frame heartbeat & telemetry
                        self._last_mic_frame_time = ts_capture
                        self.total_mic_frames += 1
                        if self.total_mic_frames == 1 and self.audio_init_latency_ms == 0.0:
                            self.audio_init_latency_ms = (ts_capture - self._init_start_time) * 1000.0

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
                            is_discontinuity=bool(overflow),
                            epoch=epoch
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
                if self.session and hasattr(self.session, "controller") and self.session.controller:
                    try:
                        self.session.controller.record_error("audio_capture", e, category="AUDIO_HARDWARE", recoverable=True)
                    except Exception:
                        pass
                if self.on_status_change:
                    self.on_status_change(f"⚠️ Microphone error: {e}")
                if not self.is_running or self._stop_event.is_set():
                    return
                # If specific device failed, fallback to system default
                if dev_idx is not None:
                    dev_idx = None
                time.sleep(retry_delay)
                retry_delay = min(1.0, retry_delay * 1.5)

    def _mic_processing_worker(self, channel_source: str = "candidate_mic", worker_id: str = "", epoch: int = 0):
        """
        Dedicated processing worker for Microphone audio.
        """
        processor = MicrophoneAudioProcessor(sample_rate=self.sample_rate)
        default_silence_frames = 8     # ~240ms of silence (natural pause protection for slow speech)
        min_speech_frames = 6          # ~180ms min speech (fast pickup for brief questions)
        interim_interval_frames = 6    # ~180ms interval (instant visual subtitle feedback)
        
        audio_buffer = []
        pre_buffer = collections.deque(maxlen=8)
        speech_frames_count = 0
        silence_frames_count = 0
        in_speech = False
        speech_start_perf = 0.0
        
        while self.is_running and not self._stop_event.is_set():
            if epoch != 0 and epoch != self._audio_stream_epoch:
                return
            try:
                frame: AudioFrame = self.mic_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Drop late frames from previous stream epochs
            if frame.epoch != 0 and frame.epoch != self._audio_stream_epoch:
                continue
                
            proc_start_ts = time.perf_counter()
            proc_bytes, is_speech, diag = processor.process_frame(frame.data)
            self.total_mic_frames += 1
            if is_speech:
                self.mic_speech_frames += 1
            self.cumulative_mic_rms += diag.rms
            
            ws_mic = getattr(self, "live_dg_ws_mic", None) or (self.live_dg_ws if self.source == "mic" else None)
            if ws_mic and ws_mic.is_running:
                if getattr(ws_mic, "_is_flux", False) or is_speech:
                    ws_mic.send_frame(proc_bytes)
            
            # Acoustic cross-talk defense with proportional suppression (Issue 27):
            # If system loopback is actively outputting speech (> 220 RMS), suppress faint acoustic speaker bleed
            # while preserving genuine candidate double-talk / interruptions (> 380 RMS or high pitch periodicity)
            sys_recent = (time.perf_counter() - getattr(self, "_last_system_speech_time", 0.0) < 0.35)
            sys_rms = getattr(self, "_last_system_speech_rms", 0.0)
            if is_speech and self.source == "dual" and sys_recent and sys_rms > 220.0:
                is_double_talk = (diag.rms >= 380.0) or (getattr(diag, "pitch_periodicity", 0.0) > 0.55 and diag.rms > 200.0)
                if not is_double_talk and diag.rms < min(360.0, sys_rms * 0.85):
                    is_speech = False

            if is_speech:
                if diag.rms > 300.0:
                    self._last_candidate_speech_time = time.perf_counter()
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
                
                effective_silence = 2 if getattr(self, "_is_candidate_semantically_complete", False) else default_silence_frames
                
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
                    self._last_interim_text = ""
                    if processor.silero_vad:
                        processor.silero_vad.reset()
            else:
                pre_buffer.append(proc_bytes)
                
            if len(audio_buffer) * 30 > 45000:
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
                if processor.silero_vad:
                    processor.silero_vad.reset()

    def _dispatch_interim_transcription(
        self,
        audio_frames: List[bytes],
        source: str = "system",
        speech_start_ts: Optional[float] = None
    ):
        """Dispatches lightweight interim transcription for live visual preview & early question detection."""
        # Skip redundant batch STT when Deepgram WebSocket is already streaming word-by-word (~50ms)
        active_ws = getattr(self, "live_dg_ws_system", None) if source == "system" else getattr(self, "live_dg_ws_mic", None)
        active_ws = active_ws or self.live_dg_ws
        if active_ws and active_ws.is_running:
            return
        now = time.perf_counter()
        if not audio_frames or len(audio_frames) < 10 or self._interim_in_flight or (now - getattr(self, "_last_interim_stt_time", 0.0) < 0.35):
            return
            
        def _worker():
            self._interim_in_flight = True
            self._last_interim_stt_time = time.perf_counter()
            try:
                raw_bytes = b"".join(audio_frames)
                if len(raw_bytes) < 3200:
                    return
                samples = np.frombuffer(raw_bytes, dtype=np.int16)
                if len(samples) > 0:
                    rms = float(np.sqrt(np.mean(samples.astype(np.float32)**2)))
                    if rms < 130.0:
                        return
                t_stt_start = time.perf_counter()
                text = self.provider.transcribe(raw_bytes, sample_rate=self.sample_rate)
                t_stt_end = time.perf_counter()
                
                if text and len(text.strip()) >= 2:
                    cleaned_text = clean_asr_transcript(text.strip())
                    if hasattr(self, "_last_interim_text") and self._last_interim_text:
                        cleaned_text = merge_overlapping_transcripts(self._last_interim_text, cleaned_text)
                    self._last_interim_text = cleaned_text
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
                
        try:
            self._stt_executor.submit(_worker)
        except Exception:
            threading.Thread(target=_worker, daemon=True).start()

    def _on_live_ws_interim(self, text: str, source: str):
        if not text:
            return
        cleaned = clean_asr_transcript(text.strip())
        if not cleaned:
            return
        if hasattr(self, "_last_interim_text") and self._last_interim_text:
            cleaned = merge_overlapping_transcripts(self._last_interim_text, cleaned)
        self._last_interim_text = cleaned
        likelihood_info = compute_question_likelihood(cleaned, session=self.session)
        if self.session:
            trace = self.session.latest_latency_trace
            if trace and trace.timestamps.get("answer_complete", 0.0) == 0.0:
                trace.mark("transcript_received", time.perf_counter())
                if likelihood_info.get("is_question_stem") or likelihood_info.get("likelihood", 0) >= 0.7:
                    trace.mark("question_candidate", time.perf_counter())
                if likelihood_info.get("is_semantically_complete") or likelihood_info.get("can_start_early"):
                    trace.mark("question_complete", time.perf_counter())
                    self._is_candidate_semantically_complete = True
        if self.on_interim_transcript:
            import inspect
            sig = inspect.signature(self.on_interim_transcript)
            if len(sig.parameters) >= 3 or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                self.on_interim_transcript(cleaned, likelihood_info, source)
    def _on_live_ws_eager_eot(self, text: str, source: str):
        if not text:
            return
        cleaned = clean_asr_transcript(text.strip())
        if not cleaned or len(cleaned) < 2:
            return
        self._last_eager_text = cleaned
        if hasattr(self, "on_eager_eot") and self.on_eager_eot:
            try:
                self.on_eager_eot(cleaned, source)
                return
            except Exception as e:
                print("on_eager_eot notice:", e)

        likelihood_info = compute_question_likelihood(cleaned, session=self.session)
        likelihood_info["can_start_early"] = True
        likelihood_info["is_semantically_complete"] = True
        if self.on_interim_transcript:
            import inspect
            sig = inspect.signature(self.on_interim_transcript)
            if len(sig.parameters) >= 3 or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                self.on_interim_transcript(cleaned, likelihood_info, source)
            else:
                self.on_interim_transcript(cleaned, likelihood_info)

    def _on_live_ws_turn_resumed(self, source: str):
        if self.session:
            self.session.turn_state = InterviewTurnState.INTERVIEWER_SPEAKING
        if hasattr(self, "on_turn_resumed") and self.on_turn_resumed:
            try:
                self.on_turn_resumed(source)
            except Exception as e:
                print("on_turn_resumed notice:", e)

    def _on_live_ws_final(self, text: str, source: str):
        self._last_interim_text = ""
        if not text:
            return
        cleaned = clean_asr_transcript(text.strip())
        if not cleaned or len(cleaned) < 2:
            return
        is_q = False if is_conversational_chitchat(cleaned) else is_likely_question(cleaned)
        if self.session:
            trace = self.session.latest_latency_trace
            if trace and trace.timestamps.get("answer_complete", 0.0) == 0.0:
                trace.mark("transcript_received", time.perf_counter())
        self._emit_transcript(cleaned, is_q, source)

    def _emit_transcript(self, text: str, is_q: bool, source: str):
        if not text or not self.on_transcript_chunk:
            return
        import inspect
        sig = inspect.signature(self.on_transcript_chunk)
        if len(sig.parameters) >= 3 or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            self.on_transcript_chunk(text, is_q, source)
        else:
            self.on_transcript_chunk(text, is_q)

    def _dispatch_final_transcription(
        self,
        segment_or_frames: Any,
        source: str = "system"
    ):
        """Transcribes finalized PCM frames via STT provider and dispatches to question analyzer."""
        # Skip redundant batch STT when Deepgram WebSocket is actively connected and streaming
        active_ws = getattr(self, "live_dg_ws_system", None) if source == "system" else getattr(self, "live_dg_ws_mic", None)
        active_ws = active_ws or self.live_dg_ws
        if active_ws and getattr(active_ws, "is_running", False) and getattr(active_ws, "ws", None) is not None:
            return

        if isinstance(segment_or_frames, AudioSegment):
            segment = segment_or_frames
            if getattr(segment, "speech_frames", 0) < 6:
                return
            raw_bytes = segment.audio_bytes
            speech_start_ts = segment.capture_start
            queue_enter_ts = segment.queue_enter
            processing_start_ts = segment.processing_start
            segment_id = segment.segment_id
        elif isinstance(segment_or_frames, list):
            if not segment_or_frames or len(segment_or_frames) < 6:
                return
            raw_bytes = b"".join(segment_or_frames)
            speech_start_ts = getattr(self, "_current_speech_start", time.perf_counter())
            queue_enter_ts = speech_start_ts
            processing_start_ts = speech_start_ts
            segment_id = f"seg_{uuid.uuid4().hex[:6]}"
        else:
            return
            
        if not raw_bytes or len(raw_bytes) < 3200:
            return

        with getattr(self, "_stt_order_lock", threading.Lock()):
            seq_id = getattr(self, "_stt_seq_counter", 0)
            self._stt_seq_counter = seq_id + 1

        def _skip_seq():
            with self._stt_order_lock:
                if len(self._stt_pending_results) > 50:
                    for k in sorted(self._stt_pending_results.keys())[:25]:
                        self._stt_pending_results.pop(k, None)
                if seq_id == self._stt_next_seq:
                    self._stt_next_seq += 1
                    while self._stt_next_seq in self._stt_pending_results:
                        p_text, p_is_q, p_src = self._stt_pending_results.pop(self._stt_next_seq)
                        self._stt_next_seq += 1
                        if p_text:
                            self._emit_transcript(p_text, p_is_q, p_src)
                else:
                    self._stt_pending_results[seq_id] = ("", False, "")
                    while self._stt_next_seq in self._stt_pending_results:
                        p_text, p_is_q, p_src = self._stt_pending_results.pop(self._stt_next_seq)
                        self._stt_next_seq += 1
                        if p_text:
                            self._emit_transcript(p_text, p_is_q, p_src)
            
        def _worker():
            try:
                samples = np.frombuffer(raw_bytes, dtype=np.int16)
                total_rms = float(np.sqrt(np.mean(samples.astype(np.float32)**2)))
                
                # Discard near-silent / room noise buffers (speech is > 120 RMS)
                if total_rms < 120.0:
                    _skip_seq()
                    return
                
                # Stream directly to ultra-low-latency STT provider (Deepgram Nova-3 / Groq Whisper)
                proc_bytes = raw_bytes
                
                t_stt_start = time.perf_counter()
                text = self.provider.transcribe(proc_bytes, sample_rate=self.sample_rate)
                t_stt_end = time.perf_counter()
                
                if not text or len(text.strip()) < 2:
                    _skip_seq()
                    return
                    
                cleaned_text = clean_asr_transcript(text.strip())
                self._last_interim_text = ""
                # Fast heuristic chit-chat check (0.1ms CPU gatekeeper)
                if is_conversational_chitchat(cleaned_text):
                    is_q = False
                else:
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
                
                with self._stt_order_lock:
                    if len(self._stt_pending_results) > 50:
                        for k in sorted(self._stt_pending_results.keys())[:25]:
                            self._stt_pending_results.pop(k, None)
                    if seq_id == self._stt_next_seq:
                        self._stt_next_seq += 1
                        self._emit_transcript(cleaned_text, is_q, source)
                        while self._stt_next_seq in self._stt_pending_results:
                            p_text, p_is_q, p_src = self._stt_pending_results.pop(self._stt_next_seq)
                            self._stt_next_seq += 1
                            if p_text:
                                self._emit_transcript(p_text, p_is_q, p_src)
                    else:
                        self._stt_pending_results[seq_id] = (cleaned_text, is_q, source)
            except Exception as e:
                print(f"Final transcription error: {e}")
                _skip_seq()
                
        try:
            self._stt_executor.submit(_worker)
        except Exception:
            threading.Thread(target=_worker, daemon=True).start()


# ============================================================
# KNOWLEDGE SOURCE CLASSIFIER & UNIFIED CANDIDATE CONTEXT
# ============================================================

