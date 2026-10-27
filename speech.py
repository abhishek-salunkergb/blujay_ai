"""
High-Fidelity Audio Engineering, Diagnostics, Preprocessing & STT Pipeline
========================================================================
Source-specific DSP processors for System Loopback (Interviewer) and Microphone (Candidate),
featuring safe AGC, clipping protection, noise floor gating, and transcript overlap merging.

Microphone path includes ML-based Silero VAD (ONNX) and spectral noise reduction for
high-accuracy speech detection and clean audio delivery to Whisper STT.
"""

import io
import os
import re
import wave
import json
import time
import queue
import collections
import warnings
import threading
import asyncio
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")

try:
    import webrtcvad  # type: ignore
    VAD_AVAILABLE = True
except Exception:
    webrtcvad = None
    VAD_AVAILABLE = False

# Prevent OpenMP and native threading collisions on Windows with ONNX Runtime
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import onnxruntime as _ort_mod  # type: ignore
    _ORT_MODULE = _ort_mod
    _ORT_TRIED = True
except (Exception, BaseException):
    _ORT_MODULE = None
    _ORT_TRIED = True

def _get_ort():
    return _ORT_MODULE

try:
    import noisereduce as nr  # type: ignore
    NOISEREDUCE_AVAILABLE = True
except Exception:
    nr = None
    NOISEREDUCE_AVAILABLE = False

try:
    import websockets.sync.client as _ws_sync_client  # type: ignore
except Exception:
    _ws_sync_client = None


SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

vad = webrtcvad.Vad(2) if VAD_AVAILABLE else None


# ============================================================
# 0. SILERO VAD (ML-BASED) & NOISE REDUCTION
# ============================================================

# Resolve the silero_vad.onnx model path relative to this file's directory
_SILERO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx")


_ORT_INIT_LOCK = threading.Lock()
_SHARED_ORT_SESSIONS = {}


def _get_shared_ort_session(model_path: str):
    """
    Returns a shared, compiled ONNX Runtime InferenceSession for model_path.
    InferenceSession.run() is thread-safe on CPU across multiple inference calls.
    Caching this eliminates the ~400ms-600ms disk read and ONNX graph compilation penalty
    on detector initialization.
    """
    with _ORT_INIT_LOCK:
        if model_path in _SHARED_ORT_SESSIONS:
            return _SHARED_ORT_SESSIONS[model_path]
        ort = _get_ort()
        if ort is not None and os.path.isfile(model_path):
            try:
                sess_opts = ort.SessionOptions()
                sess_opts.inter_op_num_threads = 1
                sess_opts.intra_op_num_threads = 1
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"]
                )
                _SHARED_ORT_SESSIONS[model_path] = session
                return session
            except (Exception, BaseException):
                return None
        return None

_GLOBAL_OPENAI_CLIENT = None
_GLOBAL_OPENAI_KEY = None
_DEEPGRAM_SESSION = None
_DEEPGRAM_KEY = None


def get_deepgram_session():
    """Returns or initializes the persistent keep-alive session for Deepgram."""
    global _DEEPGRAM_SESSION, _DEEPGRAM_KEY
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if deepgram_key:
        if _DEEPGRAM_SESSION is None or _DEEPGRAM_KEY != deepgram_key:
            import requests
            _DEEPGRAM_SESSION = requests.Session()
            _DEEPGRAM_SESSION.headers.update({
                "Authorization": f"Token {deepgram_key}",
                "Content-Type": "audio/wav"
            })
            _DEEPGRAM_KEY = deepgram_key
        return _DEEPGRAM_SESSION
    return None


class DeepgramLiveWebSocketClient:
    """
    True Full-Duplex Real-Time Live Streaming WebSocket Client for Deepgram Flux & Nova-3.
    - If model is 'flux-general-en' (or starts with 'flux'): connects via Deepgram /v2/listen endpoint
      with explicit eager_eot_threshold=0.5 and eot_threshold=0.8, sample-level 80ms PCM chunking,
      and native TurnResumed/EagerEndOfTurn/EndOfTurn conversational turn events.
    - If model is 'nova-3': connects via /v1/listen with smart formatting and interim transcripts.
    - Resilient Auto-Fallback: If /v2/listen encounters any connection error, automatically and silently
      falls back to /v1/listen (Nova-3) for 100% interview session uptime.
    """
    CHUNK_BYTES = 2560  # Exactly 80ms @ 16kHz 16-bit mono (1280 samples * 2 bytes)

    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        model: Optional[str] = None,
        language: str = "en",
        on_interim: Optional[Callable[[str, str], None]] = None,
        on_final: Optional[Callable[[str, str], None]] = None,
        on_eager_eot: Optional[Callable[[str, str], None]] = None,
        on_turn_resumed: Optional[Callable[[str], None]] = None,
        source_label: str = "mic"
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "").strip()
        self.sample_rate = sample_rate
        raw_model = model or os.getenv("DEFAULT_STT_MODEL", "flux-general-en")
        if raw_model.strip().lower() == "flux":
            raw_model = "flux-general-en"
        self.model = raw_model.strip()
        self._is_flux = "flux" in self.model.lower()
        self.language = language
        self.on_interim = on_interim
        self.on_final = on_final
        self.on_eager_eot = on_eager_eot
        self.on_turn_resumed = on_turn_resumed
        self.source_label = source_label
        
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self._stop_event = threading.Event()
        self.ws = None
        self._worker_thread = None
        self._receiver_thread = None
        self._last_interim_text = ""
        self._last_eager_text = ""
        self._flux_pcm_buffer = bytearray()

    def start(self):
        if not self.api_key:
            return False
        self.is_running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name=f"DeepgramLiveWS-{self.source_label}")
        self._worker_thread.start()
        return True

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        self._flux_pcm_buffer.clear()

    def send_frame(self, pcm_bytes: bytes):
        if not self.is_running or not pcm_bytes:
            return
        
        if self._is_flux:
            # Sample-level byte accumulation: slice off exact 2560-byte (80ms) chunks
            self._flux_pcm_buffer.extend(pcm_bytes)
            while len(self._flux_pcm_buffer) >= self.CHUNK_BYTES:
                chunk = bytes(self._flux_pcm_buffer[:self.CHUNK_BYTES])
                del self._flux_pcm_buffer[:self.CHUNK_BYTES]
                self._enqueue_packet(chunk)
        else:
            self._enqueue_packet(pcm_bytes)

    def _enqueue_packet(self, data: bytes):
        try:
            self.audio_queue.put_nowait(data)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(data)
            except queue.Full:
                pass

    def _build_url(self) -> str:
        if self._is_flux:
            url = (
                f"wss://api.deepgram.com/v2/listen?model={self.model}"
                f"&eager_eot_threshold=0.5&eot_threshold=0.8"
                f"&encoding=linear16&sample_rate={self.sample_rate}&channels=1"
            )
            if self.language and self.language.lower() not in ["auto", "none", "", "en"]:
                url += f"&language_hint={self.language.lower()}"
        else:
            url = (
                f"wss://api.deepgram.com/v1/listen?model={self.model}"
                f"&encoding=linear16&sample_rate={self.sample_rate}&channels=1"
                f"&interim_results=true&smart_format=true&punctuate=true&endpointing=300"
                f"&keywords=Kubernetes:2&keywords=FastAPI:2&keywords=PostgreSQL:2&keywords=Docker:2&keywords=Redis:2&keywords=JWT:2&keywords=gRPC:2&keywords=GraphQL:2"
            )
            if self.language and self.language.lower() not in ["auto", "none", ""]:
                url += f"&language={self.language.lower()}"
        return url

    def _run_loop(self):
        ws_client = _ws_sync_client
        if ws_client is None:
            import websockets.sync.client as ws_client
        headers = {"Authorization": f"Token {self.api_key}"}

        while self.is_running and not self._stop_event.is_set():
            url = self._build_url()
            try:
                self.ws = ws_client.connect(url, additional_headers=headers, open_timeout=5.0)
                
                # Start background receiver thread
                self._receiver_thread = threading.Thread(target=self._receiver_loop, args=(self.ws,), daemon=True)
                self._receiver_thread.start()

                # Stream audio chunks continuously
                while self.is_running and not self._stop_event.is_set():
                    try:
                        frame = self.audio_queue.get(timeout=0.05)
                        if frame:
                            self.ws.send(frame)
                    except queue.Empty:
                        continue
            except Exception as e:
                if not self.is_running or self._stop_event.is_set():
                    break
                # Resilient Fallback: If Flux v2 rejected or failed, silently fallback to Nova-3 v1
                if self._is_flux:
                    print(f"Deepgram Flux v2 connection notice ({e}). Auto-falling back to Nova-3...")
                    self.model = "nova-3"
                    self._is_flux = False
                    time.sleep(0.5)
                    continue
                time.sleep(1.0)
            finally:
                if self.ws:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None

    def _receiver_loop(self, ws):
        while self.is_running and not self._stop_event.is_set():
            try:
                msg = ws.recv(timeout=1.0)
                if not msg:
                    continue
                data = json.loads(msg)
                
                # -------------------------------------------------------------
                # Deepgram v2 Conversational Turn Events (Flux)
                # -------------------------------------------------------------
                msg_type = data.get("type", "")
                event_name = data.get("event") or (msg_type if msg_type in ("EagerEndOfTurn", "TurnResumed", "SpeechResumed", "EndOfTurn") else None)

                if msg_type == "TurnInfo" or event_name:
                    event = event_name or data.get("event", "")
                    transcript = data.get("transcript", "").strip()

                    if event == "EagerEndOfTurn":
                        if transcript:
                            self._last_eager_text = transcript
                            if self.on_eager_eot:
                                self.on_eager_eot(transcript, self.source_label)
                            elif self.on_interim:
                                self.on_interim(transcript, self.source_label)

                    elif event in ("TurnResumed", "SpeechResumed"):
                        # Speaker paused and resumed speech: cancel in-flight speculative answer immediately
                        if self.on_turn_resumed:
                            self.on_turn_resumed(self.source_label)

                    elif event == "EndOfTurn":
                        if transcript:
                            if self.on_final:
                                self.on_final(transcript, self.source_label)
                            self._last_interim_text = ""
                            self._last_eager_text = ""

                # -------------------------------------------------------------
                # Deepgram v1 / Standard Results (Nova-3 & streaming interims)
                # -------------------------------------------------------------
                elif msg_type == "Results":
                    channel = data.get("channel", {})
                    alts = channel.get("alternatives", [])
                    if alts:
                        text = alts[0].get("transcript", "").strip()
                        if not text:
                            continue
                        is_final = bool(data.get("is_final", False))
                        speech_final = bool(data.get("speech_final", False))

                        if speech_final or (is_final and len(text.split()) >= 3):
                            if self.on_final:
                                self.on_final(text, self.source_label)
                            self._last_interim_text = ""
                        else:
                            if text != self._last_interim_text:
                                self._last_interim_text = text
                                if self.on_interim:
                                    self.on_interim(text, self.source_label)
            except Exception:
                break


class SileroVADDetector:
    """
    ML-based Voice Activity Detection using Silero VAD ONNX model.
    Runs on CPU via onnxruntime (~1ms per frame). No PyTorch required.
    Falls back to WebRTC VAD if the ONNX model file is missing.
    Thread-safe LSTM state management protected by internal lock.
    """

    def __init__(self, model_path: str = _SILERO_MODEL_PATH, threshold: float = 0.45):
        self.threshold = threshold
        self._session = None
        self._h = None
        self._c = None
        self._sr = None
        self._available = False
        self._lock = threading.Lock()

        session = _get_shared_ort_session(model_path)
        if session is not None:
            self._session = session
            self._reset_state()
            self._available = True
        else:
            self._available = False

    def _reset_state(self):
        """Initializes / resets the LSTM hidden state for the Silero model."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array([16000], dtype=np.int64)

    @property
    def available(self) -> bool:
        return self._available

    def is_speech(self, audio_int16: np.ndarray) -> Tuple[bool, float]:
        """
        Detects speech in an int16 PCM audio chunk.
        Returns (is_speech: bool, probability: float).
        Accepts any chunk size; internally resamples to 512-sample windows.
        Thread-safe: LSTM state operations are locked.
        """
        if not self._available or audio_int16 is None or len(audio_int16) == 0:
            return False, 0.0

        try:
            # Convert int16 to float32 normalized [-1, 1]
            audio_f32 = audio_int16.astype(np.float32) / 32768.0

            # Silero VAD expects 512-sample chunks at 16kHz
            # Process in 512-sample windows, return max probability
            window_size = 512
            max_prob = 0.0

            if len(audio_f32) < window_size:
                # Pad short frames with zeros
                padded = np.zeros(window_size, dtype=np.float32)
                padded[:len(audio_f32)] = audio_f32
                audio_f32 = padded

            with self._lock:
                for start in range(0, len(audio_f32) - window_size + 1, window_size):
                    chunk = audio_f32[start:start + window_size].reshape(1, -1)

                    ort_inputs = {
                        "input": chunk,
                        "h": self._h,
                        "c": self._c,
                        "sr": self._sr
                    }
                    ort_outputs = self._session.run(None, ort_inputs)

                    prob = float(ort_outputs[0].item())
                    self._h = ort_outputs[1]
                    self._c = ort_outputs[2]
                    max_prob = max(max_prob, prob)

                return max_prob >= self.threshold, max_prob

        except Exception:
            return False, 0.0

    def reset(self):
        """Resets hidden state between utterances. Thread-safe."""
        if self._available:
            with self._lock:
                self._reset_state()


# Module-level singleton for the Silero VAD detector (preserved for backward compatibility)
_silero_vad = None


def get_silero_vad() -> SileroVADDetector:
    global _silero_vad
    if _silero_vad is None:
        _silero_vad = SileroVADDetector()
    return _silero_vad


def reduce_mic_noise(samples_int16: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Applies gentle spectral noise reduction to microphone audio using noisereduce.
    Uses conservative reduction (prop_decrease=0.6) to keep speech formants and consonants
    crisp and intact for Whisper STT.
    Returns cleaned int16 PCM samples.
    """
    if not NOISEREDUCE_AVAILABLE or samples_int16 is None or len(samples_int16) == 0:
        return samples_int16

    try:
        # Convert to float for noisereduce
        audio_float = samples_int16.astype(np.float32) / 32768.0

        # Apply gentle stationary noise reduction (prop_decrease=0.6 preserves natural voice)
        reduced = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.6,
            n_fft=512
        )

        # Convert back to int16
        return np.clip(reduced * 32768.0, -32768.0, 32767.0).astype(np.int16)

    except Exception:
        # On any error, return original audio unchanged
        return samples_int16


# ============================================================
# 1. AUDIO QUALITY DIAGNOSTICS & MEASUREMENT
# ============================================================

@dataclass
class AudioDiagnostics:
    """Real-time signal diagnostics computed before STT segmentation."""
    rms: float
    peak: float
    speech_level_db: float
    silence_ratio: float
    clipping_ratio: float
    snr_estimate_db: float
    state: str  # 'SILENT', 'LOW_SIGNAL', 'NORMAL', 'LOUD', 'CLIPPING', 'NOISY'
    crest_factor: float = 0.0
    zero_crossing_rate: float = 0.0
    pitch_periodicity: float = 0.0
    is_transient_click: bool = False
    is_tonal_chime: bool = False


def compute_audio_diagnostics(
    samples: np.ndarray,
    noise_floor_rms: float = 350.0
) -> AudioDiagnostics:
    """
    Computes lightweight DSP diagnostics for audio quality monitoring, adaptive processing,
    impulsive typing click rejection, and tonal notification chime detection.
    """
    if samples is None or len(samples) == 0:
        return AudioDiagnostics(
            rms=0.0,
            peak=0.0,
            speech_level_db=-96.0,
            silence_ratio=1.0,
            clipping_ratio=0.0,
            snr_estimate_db=0.0,
            state="SILENT",
            crest_factor=0.0,
            zero_crossing_rate=0.0,
            pitch_periodicity=0.0,
            is_transient_click=False,
            is_tonal_chime=False
        )
        
    # Clean NaN / Inf values
    if not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=32767.0, neginf=-32768.0)
        
    abs_samples = np.abs(samples.astype(np.float32))
    peak = float(np.max(abs_samples)) if len(abs_samples) > 0 else 0.0
    rms = float(np.sqrt(np.mean(abs_samples ** 2))) if len(abs_samples) > 0 else 0.0
    
    # Calculate dBFS relative to max 16-bit PCM (32768)
    speech_level_db = 20.0 * np.log10(max(rms, 1.0) / 32768.0)
    
    # Clipping detection: ratio of samples within 98% of peak (32100 / 32767)
    clipping_threshold = 32100.0
    clipping_count = np.sum(abs_samples >= clipping_threshold)
    clipping_ratio = float(clipping_count / len(samples)) if len(samples) > 0 else 0.0
    
    # Silence ratio: samples below baseline noise floor
    silence_count = np.sum(abs_samples < max(100.0, noise_floor_rms * 1.2))
    silence_ratio = float(silence_count / len(samples)) if len(samples) > 0 else 1.0
    
    # Signal-to-noise ratio estimate
    safe_noise = max(50.0, noise_floor_rms)
    snr_estimate_db = float(20.0 * np.log10(max(rms, 1.0) / safe_noise))
    
    # Classify state
    if clipping_ratio > 0.005 or peak >= 32700.0:
        state = "CLIPPING"
    elif rms < 180.0:
        state = "SILENT"
    elif snr_estimate_db < 3.0 and rms > 600.0:
        state = "NOISY"
    elif rms < 850.0:
        state = "LOW_SIGNAL"
    elif rms > 14000.0:
        state = "LOUD"
    else:
        state = "NORMAL"

    # 1. Crest Factor (peak / RMS): sharp mechanical clicks produce > 5.5 (often 7.0-14.0)
    crest_factor = float(peak / max(rms, 1.0))

    # 2. Zero-Crossing Rate (ZCR): plastic/metal impact produces high broadband crossings (> 0.20)
    zcr = float(np.mean(samples[:-1] * samples[1:] < 0)) if len(samples) > 1 else 0.0

    # 3. Vectorized FFT Autocorrelation & Spectral Peakiness via Wiener-Khinchin Theorem
    # Executes in ~0.05ms (C-level numpy vectorization) covering pitch lags 32 to 230 (70Hz - 500Hz)
    pitch_periodicity = 0.0
    is_tonal_chime = False
    if len(samples) >= 256 and rms > 25.0:
        s = samples.astype(np.float32)
        n_fft = 1024
        fft_val = np.fft.rfft(s, n=n_fft)
        power = np.abs(fft_val) ** 2
        autocorr = np.fft.irfft(power)[:len(s)]
        energy = autocorr[0] + 1e-6
        max_lag = min(230, len(s) - 40)
        if max_lag > 32:
            pitch_periodicity = max(0.0, float(np.max(autocorr[32:max_lag])) / energy)

        # Spectral Peakiness from power spectrum (pure tonal notification chime gate)
        mean_power = float(np.mean(power))
        if mean_power > 1e-4 and rms > 50.0:
            peakiness = float(np.max(power) / (mean_power + 1e-6))
            if peakiness > 30.0:
                peak_bin = int(np.argmax(power))
                peak_freq = peak_bin * (16000.0 / n_fft)
                if peak_freq >= 550.0:
                    is_tonal_chime = True

    # 4. Impulsive typing transient click detection
    is_transient_click = False
    if crest_factor > 5.5 and zcr > 0.20 and pitch_periodicity < 0.25:
        is_transient_click = True

    return AudioDiagnostics(
        rms=rms,
        peak=peak,
        speech_level_db=speech_level_db,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        snr_estimate_db=snr_estimate_db,
        state=state,
        crest_factor=crest_factor,
        zero_crossing_rate=zcr,
        pitch_periodicity=pitch_periodicity,
        is_transient_click=is_transient_click,
        is_tonal_chime=is_tonal_chime
    )


# ============================================================
# 2. SIGNAL PREPROCESSING: DC OFFSET, AGC & LIMITER
# ============================================================

def remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    """Removes DC bias and ultra-low frequency sub-audible rumble from audio samples, preserving int16 PCM format."""
    if samples is None or len(samples) == 0:
        return samples
    mean_val = np.mean(samples.astype(np.float32))
    cleaned = samples.astype(np.float32) - mean_val
    return np.clip(cleaned, -32768.0, 32767.0).astype(np.int16)


def safe_automatic_gain_control(
    samples: np.ndarray,
    target_rms: float = 2600.0,
    max_gain_db: float = 12.0,
    noise_floor_rms: float = 350.0
) -> np.ndarray:
    """
    Safe adaptive leveler.
    - Boosts low-volume interviewer speech cleanly (up to +12dB).
    - Preserves normal speech dynamics (1.0x gain).
    - Soft-compresses loud audio to prevent clipping.
    - NEVER amplifies baseline room silence or fan noise.
    """
    if samples is None or len(samples) == 0:
        return samples
        
    samples_float = samples.astype(np.float32)
    diag = compute_audio_diagnostics(samples, noise_floor_rms=noise_floor_rms)
    
    # If pure silence or near noise floor, return unboosted to prevent amplifying noise floor into speech
    if diag.rms < max(35.0, noise_floor_rms * 1.25):
        return samples
        
    # Calculate required linear gain
    current_rms = max(diag.rms, 100.0)
    ideal_gain = target_rms / current_rms
    
    # Cap maximum boost to max_gain_db (e.g. +12dB ~ 3.98x)
    max_linear_gain = 10.0 ** (max_gain_db / 20.0)
    applied_gain = min(max_linear_gain, max(0.65, ideal_gain))
    
    # If loud or clipping, attenuate gently
    if diag.peak > 30000.0:
        attenuation = 29000.0 / diag.peak
        applied_gain = min(applied_gain, attenuation)
        
    processed = samples_float * applied_gain
    
    # Soft limiter knee when approaching int16 extremes
    limit_threshold = 31000.0
    over_idx = np.abs(processed) > limit_threshold
    if np.any(over_idx):
        sign = np.sign(processed[over_idx])
        mag = np.abs(processed[over_idx])
        # Smooth compression curve
        compressed = limit_threshold + (32767.0 - limit_threshold) * np.tanh((mag - limit_threshold) / 4000.0)
        processed[over_idx] = sign * compressed
        
    return np.clip(processed, -32768.0, 32767.0).astype(np.int16)


# ============================================================
# 3. SOURCE-SPECIFIC AUDIO PROCESSORS
# ============================================================

class SystemAudioProcessor:
    """
    DSP Processor for System Loopback (Interviewer Audio).
    Optimized for digital meeting output (Zoom, Teams, Google Meet, Browser).
    - DC offset removal.
    - Adaptive volume leveling for soft interviewers.
    - Zero noise pumping on silence.
    - Anti-clipping soft limiter.
    - Dedicated Neural Silero VAD + Tonal Chime Gate (rejects Slack/Teams/Windows dings).
    - 2-State Schmitt Trigger Hysteresis with Hangover hold.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_floor_rms = 40.0  # System loopback has very low electrical noise floor
        self.last_diagnostics: Optional[AudioDiagnostics] = None
        # Dedicated Neural VAD instance for system loopback
        try:
            self.silero_vad = SileroVADDetector(threshold=0.38)
            if not self.silero_vad.available:
                self.silero_vad = get_silero_vad()
        except Exception:
            self.silero_vad = get_silero_vad()
        self.webrtc_vad = webrtcvad.Vad(2) if VAD_AVAILABLE else None

        # 2-State Hysteresis State Machine & Hangover Hold (Schmitt Trigger)
        self._in_speech: bool = False
        self._onset_count: int = 0
        self._hangover_count: int = 0

    def process_frame(self, frame_float_or_int: Any) -> Tuple[bytes, bool, AudioDiagnostics]:
        """
        Processes a raw frame, computes diagnostics, and returns 16-bit PCM bytes.
        Handles float32, float64, int16, multi-channel 2D shapes, and raw bytes.
        """
        if frame_float_or_int is None:
            diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
            return b"", False, diag
            
        if isinstance(frame_float_or_int, bytes):
            if len(frame_float_or_int) == 0:
                diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
                return b"", False, diag
            int_samples = np.frombuffer(frame_float_or_int, dtype=np.int16)
        elif isinstance(frame_float_or_int, np.ndarray):
            if frame_float_or_int.size == 0:
                diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
                return b"", False, diag
            flat = frame_float_or_int.flatten()
            if not np.isfinite(flat).all():
                flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)
            if flat.dtype in (np.float32, np.float64):
                clipped = np.clip(flat, -1.0, 1.0)
                int_samples = (clipped * 32767.0).astype(np.int16)
            else:
                int_samples = flat.astype(np.int16)
        else:
            try:
                int_samples = np.array(frame_float_or_int, dtype=np.int16).flatten()
            except Exception:
                diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
                return b"", False, diag
            
        # Remove DC offset
        dc_clean = remove_dc_offset(int_samples)
        
        # Calculate diagnostics
        diag = compute_audio_diagnostics(dc_clean, noise_floor_rms=self.noise_floor_rms)
        self.last_diagnostics = diag
        
        # Determine speech presence with Neural Silero VAD + WebRTC VAD + Chime Gate
        silero_detected = False
        silero_prob = 0.0
        silero_active = bool(self.silero_vad and getattr(self.silero_vad, "available", False))
        if silero_active:
            try:
                silero_detected, silero_prob = self.silero_vad.is_speech(dc_clean)
            except Exception:
                silero_detected = False

        webrtc_detected = False
        if not silero_detected and self.webrtc_vad and VAD_AVAILABLE:
            try:
                if len(dc_clean) in (160, 320, 480):
                    webrtc_detected = self.webrtc_vad.is_speech(dc_clean.tobytes(), self.sample_rate)
            except Exception:
                webrtc_detected = False

        # Pure tonal chime / notification ding rejection (Slack knock, Teams ding, OS alert)
        is_chime = diag.is_tonal_chime and silero_prob < 0.45

        # Instantaneous raw speech decision
        # Requires: not a notification chime, above minimum floor, and neural speech (or WebRTC fallback)
        if silero_active:
            speech_detected = (silero_detected and silero_prob >= 0.35) or (webrtc_detected and diag.rms > 150.0)
        else:
            speech_detected = webrtc_detected and diag.rms > 75.0

        raw_speech = (not is_chime) and (diag.rms > 35.0) and (diag.state != "SILENT") and speech_detected

        # 2-State Hysteresis Schmitt Trigger with Hangover Hold
        # - Instant onset on clean system loopback to prevent clipping initial syllables / plosives
        # - Hangover: holds state for 4 frames (~120ms) across silent stop consonant closures
        if raw_speech:
            self._in_speech = True
            self._hangover_count = 4
        else:
            if self._hangover_count > 0 and self._in_speech:
                self._hangover_count -= 1
            else:
                self._in_speech = False

        is_speech = self._in_speech
        
        # Apply dynamic noise gating & safe AGC
        if not is_speech:
            leveled = np.zeros_like(dc_clean)
        else:
            leveled = safe_automatic_gain_control(
                dc_clean,
                target_rms=2800.0,
                max_gain_db=12.0,
                noise_floor_rms=self.noise_floor_rms
            )
        
        return leveled.tobytes(), is_speech, diag


def apply_fan_highpass_filter(samples: np.ndarray, sample_rate: int = 16000, cutoff_hz: float = 85.0) -> np.ndarray:
    """
    Strips low-frequency mechanical fan motor rumble, desk vibrations, and AC hum (< 85 Hz).
    Uses lightweight scipy.signal butterworth filter if available, or direct fast IIR filter.
    """
    if samples is None or len(samples) < 4:
        return samples
    try:
        from scipy import signal
        sos = signal.butter(2, cutoff_hz, 'hp', fs=sample_rate, output='sos')
        filtered = signal.sosfilt(sos, samples.astype(np.float32))
        return np.clip(filtered, -32768.0, 32767.0).astype(np.int16)
    except Exception:
        # Fast 1st-order IIR fallback: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        dt = 1.0 / sample_rate
        alpha = rc / (rc + dt)
        x = samples.astype(np.float32)
        y = np.zeros_like(x)
        for i in range(1, len(x)):
            y[i] = alpha * (y[i-1] + x[i] - x[i-1])
        return np.clip(y, -32768.0, 32767.0).astype(np.int16)


class MicrophoneAudioProcessor:
    """
    DSP Processor for Microphone (Candidate & In-Person Speech).
    Optimized for physical environment capture (laptop mic, fan noise rejection, room reverb).
    - Lightweight DC-offset removal and 85Hz High-Pass Fan Rumble Filter.
    - Strict Multi-stage Neural VAD: Silero VAD ONNX + WebRTC VAD.
    - Parakeet-grade fan and AC noise rejection (energy-only false triggers eliminated).
    - Mechanical typing click & impulsive transient rejection.
    - Bounded dynamic floor with anti-runaway ceiling min(180.0, max(35.0, floor * 1.12)).
    - 2-State Hysteresis Schmitt Trigger with 5-frame hangover hold.
    - Responsive AGC (+16dB) for soft and slow voices.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_floor_rms = 40.0
        # Dedicated per-processor Silero instance for isolated thread state; falls back gracefully
        try:
            self.silero_vad = SileroVADDetector(threshold=0.40)
            if not self.silero_vad.available:
                self.silero_vad = get_silero_vad()
        except Exception:
            self.silero_vad = get_silero_vad()
        self.webrtc_vad = webrtcvad.Vad(3) if VAD_AVAILABLE else None  # Mode 3: aggressive noise rejection
        self.last_diagnostics: Optional[AudioDiagnostics] = None
        self._was_speaking = False

        # 2-State Hysteresis State Machine & Hangover Hold (Schmitt Trigger)
        self._in_speech: bool = False
        self._onset_count: int = 0
        self._hangover_count: int = 0

    def process_frame(self, frame_bytes: Any) -> Tuple[bytes, bool, AudioDiagnostics]:
        """
        Processes microphone frame with DC offset removal, fan filter, adaptive noise tracking, and robust VAD.
        Flow: frame → DC offset + Fan Highpass → diagnostics → Neural VAD → Dynamic Noise Gate + AGC → output
        """
        if frame_bytes is None:
            diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
            return b"", False, diag
            
        if isinstance(frame_bytes, bytes):
            if len(frame_bytes) == 0:
                diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
                return b"", False, diag
            samples = np.frombuffer(frame_bytes, dtype=np.int16)
        elif isinstance(frame_bytes, np.ndarray):
            if frame_bytes.size == 0:
                diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
                return b"", False, diag
            flat = frame_bytes.flatten()
            if not np.isfinite(flat).all():
                flat = np.nan_to_num(flat, nan=0.0, posinf=32767.0, neginf=-32768.0)
            if flat.dtype in (np.float32, np.float64):
                clipped = np.clip(flat, -1.0, 1.0)
                samples = (clipped * 32767.0).astype(np.int16)
            else:
                samples = flat.astype(np.int16)
        else:
            try:
                samples = np.array(frame_bytes, dtype=np.int16).flatten()
            except Exception:
                diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
                return b"", False, diag
        
        # 1. Remove DC offset and apply 85Hz Highpass Rumble Filter (strips fan motor & AC hum)
        dc_clean = remove_dc_offset(samples)
        filtered = apply_fan_highpass_filter(dc_clean, self.sample_rate, cutoff_hz=85.0)
        
        # 2. Dynamic noise floor update on clean signal
        diag = compute_audio_diagnostics(filtered, noise_floor_rms=self.noise_floor_rms)
        self.last_diagnostics = diag
        
        # Track background ambient noise (e.g. fan hum)
        if 15.0 < diag.rms < 350.0 and diag.state in ("SILENT", "LOW_SIGNAL", "NOISY"):
            self.noise_floor_rms = min(120.0, max(15.0, 0.95 * self.noise_floor_rms + 0.05 * diag.rms))
            
        # 3. Neural VAD detection (Silero VAD + WebRTC VAD Mode 3)
        vad_detected = False
        silero_prob = 0.0
        silero_active = bool(self.silero_vad is not None and getattr(self.silero_vad, "available", False))
        if silero_active:
            try:
                vad_detected, silero_prob = self.silero_vad.is_speech(filtered)
            except Exception:
                vad_detected = False
                
        # Only fallback to WebRTC VAD if Silero is unavailable, in ambiguous boundary zone (prob >= 0.20),
        # or audio has strong signal energy (rms > 150.0).
        # This rejects faint ambient fan/room noise (rms < 150 with prob < 0.20) while preserving voiced speech.
        if not vad_detected and self.webrtc_vad and VAD_AVAILABLE:
            if not silero_active or silero_prob >= 0.20 or diag.rms > 150.0:
                try:
                    if len(filtered) in (160, 320, 480):
                        vad_detected = self.webrtc_vad.is_speech(filtered.tobytes(), self.sample_rate)
                    else:
                        raw_b = filtered.tobytes()
                        # Check first few 30ms frames
                        for off in range(0, min(len(filtered) - 480 + 1, 480 * 5), 480):
                            if self.webrtc_vad.is_speech(raw_b[off * 2 : (off + 480) * 2], self.sample_rate):
                                vad_detected = True
                                break
                except Exception:
                    vad_detected = False

        # Mechanical keyboard click & impulsive transient rejection
        is_typing_click = diag.is_transient_click and silero_prob < 0.50

        # Bounded dynamic floor with anti-runaway ceiling:
        # Lower bound (35.0) enables soft speech pickup; upper bound (180.0) prevents fan noise runaway
        effective_min_rms = min(180.0, max(35.0, self.noise_floor_rms * 1.12))

        # Raw frame voice detection decision
        raw_speech = (not is_typing_click) and (vad_detected and diag.rms > effective_min_rms)

        # 2-State Hysteresis Schmitt Trigger with 5-frame (~150ms) Hangover Hold
        if raw_speech:
            self._onset_count += 1
            # If buffer is multi-frame (>= 60ms) or we've reached 2 consecutive frames, lock into speech
            if len(filtered) >= 960 or self._onset_count >= 2 or self._in_speech:
                self._in_speech = True
                self._hangover_count = 5
        else:
            self._onset_count = 0
            if self._hangover_count > 0 and self._in_speech:
                self._hangover_count -= 1
            else:
                self._in_speech = False

        is_speech = self._in_speech

        # Auto-reset Silero hidden state when voice ends to prevent state accumulation/drift
        if self._was_speaking and not is_speech:
            if self.silero_vad:
                self.silero_vad.reset()
        self._was_speaking = is_speech
            
        # 4. Apply dynamic noise gate & responsive AGC for microphone (+16dB boost for soft/slow voices)
        # If no active speech is detected, clamp to pure silence to eliminate fan/AC/typing noise
        if not is_speech:
            processed = np.zeros_like(filtered)
        else:
            processed = safe_automatic_gain_control(
                filtered,
                target_rms=2800.0,
                max_gain_db=16.0,
                noise_floor_rms=self.noise_floor_rms
            )
        
        return processed.tobytes(), is_speech, diag


# ============================================================
# 4. TRANSCRIPT OVERLAP & TOKEN DEDUPLICATION
# ============================================================

def merge_overlapping_transcripts(prev_text: str, new_text: str) -> str:
    """
    Safely merges rolling STT windows by finding the longest overlapping word sequence.
    
    Example:
      prev: "Why did you choose MongoDB"
      new:  "MongoDB for this project"
      -> "Why did you choose MongoDB for this project"
    """
    if not prev_text or not prev_text.strip():
        return new_text.strip() if new_text else ""
    if not new_text or not new_text.strip():
        return prev_text.strip()
        
    prev_words = prev_text.strip().split()
    new_words = new_text.strip().split()
    
    # Check for prefix containment
    if new_text.strip().lower().startswith(prev_text.strip().lower()):
        return new_text.strip()
    if prev_text.strip().lower().endswith(new_text.strip().lower()):
        return prev_text.strip()
        
    # Search for largest overlapping token block (from min 1 word up to len)
    max_overlap = min(len(prev_words), len(new_words))
    best_overlap_len = 0
    
    for k in range(max_overlap, 0, -1):
        prev_suffix = [re.sub(r'[^\w]', '', w.lower()) for w in prev_words[-k:]]
        new_prefix = [re.sub(r'[^\w]', '', w.lower()) for w in new_words[:k]]
        
        if prev_suffix == new_prefix and all(len(w) > 0 for w in prev_suffix):
            best_overlap_len = k
            break
            
    if best_overlap_len > 0:
        merged_words = prev_words + new_words[best_overlap_len:]
        return " ".join(merged_words).strip()
        
    # No direct overlap found: join cleanly if both form continuous speech
    return f"{prev_text.strip()} {new_text.strip()}".strip()


# ============================================================
# 5. CORE SPEECH RECOGNITION INTERFACE
# ============================================================

def is_speech_frame(frame_bytes: bytes, sample_rate: int = SAMPLE_RATE, aggressiveness: int = 2) -> bool:
    """Detects speech using WebRTC VAD with fallback to root-mean-square energy."""
    if VAD_AVAILABLE and vad is not None:
        try:
            return vad.is_speech(frame_bytes, sample_rate)
        except Exception:
            pass
    # Fallback to audio amplitude energy check
    audio_data = np.frombuffer(frame_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) if len(audio_data) > 0 else 0
    return rms > 300.0


def record_until_silence(
    silence_ms: int = 650,
    max_initial_silence_s: float = 4.0,
    max_duration_s: float = 60.0,
    pre_speech_buffer_ms: int = 400
) -> bytes:
    """
    High-precision speech recording with pre-speech ring buffer and fast natural pause cut (~650ms).
    Returns b"" if no speech is spoken within max_initial_silence_s, preventing phantom hallucinations.
    """
    print("\n🎤 Listening for speech...")

    audio_frames = []
    pre_buffer_len = max(1, int(pre_speech_buffer_ms / FRAME_DURATION))
    pre_buffer = collections.deque(maxlen=pre_buffer_len)

    max_silence_frames = max(5, int(silence_ms / FRAME_DURATION))
    max_initial_frames = int((max_initial_silence_s * 1000) / FRAME_DURATION)
    max_total_frames = int((max_duration_s * 1000) / FRAME_DURATION)

    silence_frames = 0
    consecutive_speech_frames = 0
    total_speech_frames = 0
    started_speaking = False
    total_frames = 0
    processor = MicrophoneAudioProcessor(sample_rate=SAMPLE_RATE)
    if sd is None:
        raise RuntimeError("sounddevice is not available. Please install sounddevice.")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1
    ) as stream:
        while total_frames < max_total_frames:
            total_frames += 1
            frame, overflow = stream.read(FRAME_SIZE)
            frame_bytes = bytes(frame)

            proc_bytes, speech_present, _ = processor.process_frame(frame_bytes)

            if speech_present:
                consecutive_speech_frames += 1
                total_speech_frames += 1
                if not started_speaking:
                    if consecutive_speech_frames >= 2:  # 60ms confirms speech start
                        started_speaking = True
                        audio_frames.extend(list(pre_buffer))
                        audio_frames.append(proc_bytes)
                    else:
                        pre_buffer.append(proc_bytes)
                else:
                    silence_frames = 0
                    audio_frames.append(proc_bytes)
            elif started_speaking:
                consecutive_speech_frames = 0
                audio_frames.append(proc_bytes)
                silence_frames += 1
                if silence_frames >= max_silence_frames:
                    break
            else:
                consecutive_speech_frames = 0
                pre_buffer.append(proc_bytes)
                # If user hasn't started speaking within timeout, stop cleanly
                if total_frames >= max_initial_frames:
                    print("⚠️ No speech detected within initial timeout.")
                    return b""

    # Must have recorded meaningful speech duration (at least ~300ms)
    if not started_speaking or total_speech_frames < 10 or not audio_frames:
        return b""

    print("✅ Speech boundary detected.")

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(audio_frames))

    return wav_io.getvalue()


async def async_record_until_silence(
    silence_ms: int = 1200,
    max_initial_silence_s: float = 4.0,
    max_duration_s: float = 60.0,
    pre_speech_buffer_ms: int = 800
) -> bytes:
    """
    Non-blocking async wrapper around record_until_silence.
    Executes synchronous audio recording on a thread-pool worker to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: record_until_silence(
            silence_ms=silence_ms,
            max_initial_silence_s=max_initial_silence_s,
            max_duration_s=max_duration_s,
            pre_speech_buffer_ms=pre_speech_buffer_ms
        )
    )


# ============================================================
# 5. FAST HEURISTIC CHIT-CHAT FILTER & PHONETIC CORRECTIONS
# ============================================================

CONVERSATIONAL_CHITCHAT_PATTERNS: List[str] = [
    r"^(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
    r"^(can\s+you\s+hear\s+me|am\s+i\s+audible|can\s+you\s+see\s+my\s+screen|are\s+you\s+there)\b",
    r"^(thanks\s+for\s+joining|thank\s+you\s+for\s+joining|nice\s+to\s+meet\s+you)\b",
    r"^(let\s+me\s+open|let\s+me\s+share|give\s+me\s+a\s+second|give\s+me\s+one\s+second|one\s+moment|just\s+a\s+sec)\b",
    r"^(how\s+are\s+you|how's\s+it\s+going|how\s+is\s+your\s+day)\b",
    r"^(okay\s+cool|sounds\s+good|got\s+it|makes\s+sense|alright\s+cool|perfect\s+thanks)\b",
]

CHITCHAT_TOKENS = {
    "ok", "okay", "yeah", "yes", "sure", "cool", "got", "it", "hello", "hi", "hey",
    "thanks", "thank", "you", "bye", "sounds", "good", "alright", "awesome", "perfect",
    "fine", "great", "yep", "nope", "ya", "so", "um", "uh", "hmm", "ah"
}


def is_conversational_chitchat(text: str) -> bool:
    """
    Ultra-fast (0.1ms) CPU regex classifier to detect and discard pleasantries,
    mic checks, screen sharing notices, and conversational filler without LLM delay.
    """
    if not text:
        return True
    cleaned = text.strip().lower()
    stripped_punct = re.sub(r"[,\.!\?]", " ", cleaned).strip()
    words = [w for w in stripped_punct.split() if w]
    
    # If all tokens belong to common conversational acknowledgements / filler (strictly short fillers)
    if words and len(words) <= 3 and all(w in CHITCHAT_TOKENS for w in words):
        return True
        
    for pattern in CONVERSATIONAL_CHITCHAT_PATTERNS:
        if re.search(pattern, cleaned):
            return True
    return False


TECHNICAL_PHONETIC_CORRECTIONS: Dict[str, str] = {
    # Core Languages & Runtimes
    r"\btype\s*script\b": "TypeScript",
    r"\bjava\s*script\b": "JavaScript",
    r"\bnode\s*j\s*s\b": "Node.js",
    r"\bnode\s*js\b": "Node.js",
    r"\bnext\s*j\s*s\b": "Next.js",
    r"\bnext\s*js\b": "Next.js",
    r"\bvue\s*j\s*s\b": "Vue.js",
    r"\bvue\s*js\b": "Vue.js",
    r"\breact\s*j\s*s\b": "React.js",
    r"\breact\s*js\b": "React.js",
    r"\breact\b": "React",
    r"\bredux\b": "Redux",
    r"\bnum\s*py\b": "NumPy",
    r"\bpan\s*das\b": "Pandas",
    r"\basync\s*io\b": "asyncio",
    r"\ba\s*sync\b": "async",

    # Frameworks & Backend
    r"\bfast\s*api\b": "FastAPI",
    r"\bfast\s*a\s*p\s*i\b": "FastAPI",
    r"\bdjango\b": "Django",
    r"\bflask\b": "Flask",
    r"\bcell\s*ery\b": "Celery",
    r"\bpie\s*dantic\b": "Pydantic",
    r"\bspring\s*boot\b": "Spring Boot",
    r"\buvicorn\b": "Uvicorn",
    r"\bgunicorn\b": "Gunicorn",
    r"\bsql\s*alchemy\b": "SQLAlchemy",

    # Databases, Caching & Data Stores
    r"\bpost\s*gres\b": "PostgreSQL",
    r"\bpost\s*grace\b": "PostgreSQL",
    r"\bpost\s*grass\b": "PostgreSQL",
    r"\bpost\s*gre\s*s\s*q\s*l\b": "PostgreSQL",
    r"\bmy\s*s\s*q\s*l\b": "MySQL",
    r"\bmy\s*sequel\b": "MySQL",
    r"\bmongo\s*d\s*b\b": "MongoDB",
    r"\bmongo\s*db\b": "MongoDB",
    r"\bread\s*is\b": "Redis",
    r"\bred\s*is\b": "Redis",
    r"\bredis\b": "Redis",
    r"\bsql\s*light\b": "SQLite",
    r"\bsequel\s*lite\b": "SQLite",
    r"\bno\s*sequel\b": "NoSQL",
    r"\bcashe\b": "cache",
    r"\bcash\s+invalidation\b": "cache invalidation",
    r"\bcash\s+memory\b": "cache memory",
    r"\bcassandra\b": "Cassandra",
    r"\bclick\s*house\b": "ClickHouse",
    r"\bsnow\s*flake\b": "Snowflake",
    r"\bdynamo\s*db\b": "DynamoDB",
    r"\bcockroach\s*db\b": "CockroachDB",
    r"\bsupabase\b": "Supabase",
    r"\belastic\s*search\b": "Elasticsearch",

    # AI, ML & LLMs
    r"\bpie\s*torch\b": "PyTorch",
    r"\bpi\s*torch\b": "PyTorch",
    r"\btensor\s*flow\b": "TensorFlow",
    r"\blang\s*chain\b": "LangChain",
    r"\bllama\s*index\b": "LlamaIndex",
    r"\bhugging\s*face\b": "HuggingFace",
    r"\bv\s*l\s*l\s*m\b": "vLLM",
    r"\br\s*a\s*g\b": "RAG",
    r"\brag\b": "RAG",
    r"\bl\s*l\s*m\b": "LLM",
    r"\bl\s*l\s*m\s*s\b": "LLMs",
    r"\bllm\b": "LLM",
    r"\bllms\b": "LLMs",
    r"\bn\s*l\s*p\b": "NLP",
    r"\bopen\s*ai\b": "OpenAI",
    r"\bgroq\b": "Groq",
    r"\bgrock\b": "Groq",
    r"\bvector\s*db\b": "Vector DB",
    r"\bvector\s*database\b": "vector database",
    r"\bfaiss\b": "FAISS",
    r"\bchroma\s*db\b": "ChromaDB",
    r"\bqdrant\b": "Qdrant",
    r"\bmilvus\b": "Milvus",

    # Cloud, DevOps & Infrastructure
    r"\bcoo\s*ber\s*net\s*es\b": "Kubernetes",
    r"\bcoo\s*ber\s*netes\b": "Kubernetes",
    r"\bcoobernetes\b": "Kubernetes",
    r"\bk\s*8\s*s\b": "Kubernetes",
    r"\bkube\b": "Kubernetes",
    r"\bkubernetes\b": "Kubernetes",
    r"\bdocer\b": "Docker",
    r"\bdock\s*er\b": "Docker",
    r"\bdocker\b": "Docker",
    r"\bci\s*/?\s*cd\b": "CI/CD",
    r"\ba\s*w\s*s\b": "AWS",
    r"\bg\s*c\s*p\b": "GCP",
    r"\bazure\b": "Azure",
    r"\bterra\s*form\b": "Terraform",
    r"\bansible\b": "Ansible",
    r"\bprometheus\b": "Prometheus",
    r"\bgrafana\b": "Grafana",
    r"\bkaf\s*ka\b": "Kafka",
    r"\brabbit\s*m\s*q\b": "RabbitMQ",

    # APIs, Protocols & Auth
    r"\bg\s*rpc\b": "gRPC",
    r"\bg\s*r\s*p\s*c\b": "gRPC",
    r"\bgraph\s*ql\b": "GraphQL",
    r"\bgraph\s*q\s*l\b": "GraphQL",
    r"\brest\s*api(s)?\b": r"REST API\1",
    r"\brest\s*a\s*p\s*i(s)?\b": r"REST API\1",
    r"\bweb\s*socket(s)?\b": r"WebSocket\1",
    r"\bj\s*w\s*t\b": "JWT",
    r"\bjay\s*son\b": "JSON",
    r"\bo\s*auth(2)?\b": r"OAuth\1",
    r"\bsaml\b": "SAML",

    # Algorithms, Complexity & DSA
    r"\bo\s*of\s*n\s*log\s*n\b": "O(N log N)",
    r"\bo\s*of\s*n\s*squared\b": "O(N^2)",
    r"\bo\s*of\s*n\b": "O(N)",
    r"\bo\s*of\s*1\b": "O(1)",
    r"\bbig\s*o\b": "Big-O",
    r"\bleet\s*code\b": "LeetCode",
    r"\bdijkstra\b": "Dijkstra",
    r"\bb\s*f\s*s\b": "BFS",
    r"\bd\s*f\s*s\b": "DFS",
    r"\bunion\s*find\b": "Union-Find",
    r"\btrie\b": "Trie",
    r"\bcap\s*theorem\b": "CAP theorem",
    r"\bacid\b": "ACID",

    # General normalizations
    r"\bs\s*q\s*l\b": "SQL",
    r"\bsequel\b": "SQL",
    r"\bpostgress\b": "PostgreSQL",
    r"\bpostgres\b": "PostgreSQL",
    r"\bdata\s*base\b": "database",
    r"\bdata\s*bases\b": "databases",
    r"\bcapturing\s+property\b": "capturing properly",
    r"\bcapturing\s+properties\b": "capturing properly",
    r"\binterruptions?\s+are\s+okay\b": "interruptions",
}


def apply_technical_asr_corrections(text: str) -> str:
    """
    Applies regex-based phonetic corrections to fix common Whisper ASR misspellings
    of technical frameworks, libraries, and algorithm complexity terms.
    """
    if not text:
        return ""
    result = text
    for pattern, replacement in TECHNICAL_PHONETIC_CORRECTIONS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def build_dynamic_whisper_prompt(
    candidate_skills: Optional[str] = None,
    job_title: Optional[str] = None,
    company: Optional[str] = None
) -> str:
    """
    Dynamically generates a clean, natural conversational Whisper prompt that conditions
    the speech decoder on professional technical dialogue without triggering keyword echo hallucinations on silence.
    """
    base_prompt = "Live professional technical interview conversation."
    parts = []
    if job_title:
        parts.append(f"Role: {job_title}")
    if company:
        parts.append(f"Company: {company}")
    if candidate_skills:
        parts.append(f"Skills: {candidate_skills}")
    if parts:
        return f"{base_prompt} {', '.join(parts)}."
    return base_prompt


WHISPER_HALLUCINATION_PATTERNS: List[str] = [
    r"^thank(s| you)?\s*(to|for)\b",
    r"^(special\s+)?thanks?\s+to\b",
    r"^thank(s| you)?\s*(for)?\s*(watching|listening|viewing)",
    r"^please\s*(subscribe|like|share)",
    r"^(sub(scribe)?|like|share|comment)\s*(and)?\s*(subscribe|like|share|comment)?",
    r"^\(?[♪♫]+.*[♪♫]+\)?$",
    r"^[\(\[\*]?(music|applause|laughter|silence|coughing|cough|sighing|sigh|breathing|breath|throat-clearing|throat clearing|cheering|applause|gasp|snort|snicker|yawn|groan|chime|ding)[\)\]\*]?$",
    r"^\.+$",
    r"^[\s\.\,\!\?]+$",
    r"^([a-z]\s+){3,}[a-z]\.?$",
    r"^you$",
    r"^(okay|ok|um|uh|hmm|ah|oh|huh|mhm|mm-hmm)\.?\??$",
    r"^(hello|hi|hey|howdy)\.?\??$",
    r"^(yeah|yes|yep|yup|nah|no)\.?$",
    r"^(right|all right|alright)\.?$",
    r"^(sure|fine|cool|great)\.?$",
    r"^(sorry|i'm sorry|excuse me)\.?$",
    r"^(welcome|you're welcome)\.?$",
    r"^(can you hear me|are you there|am i audible)\??$",
    r"^(i|we|they|he|she|it|that|this)\.?$",
    r"^(well|now|like)\.?$",
    r"^bye\.?$",
    r"^(so|and|but|the|a|an)\.?$",
    r"^thank you\.?$",
    r"^thanks\.?$",
    r"^(subtitles by|translated by|transcribed by|captions by)\b",
    r"^(amara\.org|mbc|mbc news|community subtitles)\b",
    r"^(see you in the next video|see you next time|bye bye|bye)\b",
    r"^(copyright|all rights reserved)\b",
]

_STANDALONE_GHOST_WORDS_SPEECH = {
    "", "you", "thank you", "thanks", "bye", "goodbye", "bye bye",
    "okay", "ok", "k", "yeah", "yes", "yep", "yup", "nah", "no",
    "hello", "hi", "hey", "howdy",
    "right", "alright", "all right",
    "so", "subscribe", "uh", "um", "ah", "oh", "hmm", "huh", "er", "mhm", "mm-hmm",
    "sorry", "im sorry", "i'm sorry", "excuse me",
    "welcome", "youre welcome", "you're welcome",
    "can you hear me", "are you there", "am i audible",
    "i", "we", "they", "he", "she", "it", "that", "this",
    "well", "now", "like"
}


def is_whisper_hallucination(text: str) -> bool:
    """Detects and rejects phantom transcriptions generated by Whisper on silence/ambient noise."""
    if not text or len(text.strip()) == 0:
        return True
    cleaned = text.strip().lower().rstrip(".,!?")
    # Check standalone single-word ghost utterances (ignoring punctuation)
    cleaned_word = re.sub(r"[^\w\s]", "", cleaned).strip()
    if cleaned_word in _STANDALONE_GHOST_WORDS_SPEECH:
        return True
    for p in WHISPER_HALLUCINATION_PATTERNS:
        if re.search(p, cleaned, re.IGNORECASE):
            return True
            
    # Check for prompt echo / keyword dumping on silence
    prompt_echo_phrases = [
        "technical engineering interview",
        "discussing python",
        "fastapi docker kubernetes",
        "big-o complexity",
        "algorithms system design",
        "live professional technical interview conversation"
    ]
    if any(pe in cleaned for pe in prompt_echo_phrases):
        return True

    # Check for repetitive stutter loops (e.g., "you you you you" or "thank you thank you")
    words = cleaned.split()
    if len(words) >= 4:
        # Check single repeated token
        if len(set(words)) == 1:
            return True
        # Check repeated 2-word phrase
        if len(words) % 2 == 0:
            half = len(words) // 2
            if words[:half] == words[half:]:
                return True
    return False


def speech_to_text(
    client,
    audio_source: Union[str, bytes],
    prompt: Optional[str] = None,
    candidate_skills: Optional[str] = None,
    job_title: Optional[str] = None,
    company: Optional[str] = None,
    language: Optional[str] = "en"
) -> str:
    """
    Transcribes audio using Whisper-large-v3-turbo with dynamic prompt priming,
    audio energy verification, hallucination rejection, multi-provider failover, and phonetic corrections.
    """
    if isinstance(audio_source, str):
        with open(audio_source, "rb") as file:
            data = file.read()
        try:
            os.remove(audio_source)
        except Exception:
            pass
        filename = os.path.basename(audio_source)
    else:
        data = audio_source
        filename = "audio.wav"

    # Audio length & silence check (must have at least ~0.2s of audio and non-zero RMS)
    if not data or len(data) < 3200:
        return ""

    try:
        raw_pcm = data[44:] if data.startswith(b"RIFF") else data
        samples = np.frombuffer(raw_pcm, dtype=np.int16)
        if len(samples) > 0:
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            if rms < 45.0:  # Baseline electrical noise, ignore
                return ""
    except Exception:
        pass

    # Generate dynamic primed prompt if custom keywords provided
    effective_prompt = prompt or build_dynamic_whisper_prompt(
        candidate_skills=candidate_skills,
        job_title=job_title,
        company=company
    )

    kwargs = {
        "file": (filename, data),
        "model": "whisper-large-v3-turbo",
        "response_format": "text",
        "temperature": 0.0,
        "prompt": effective_prompt
    }
    if language and language.lower() not in ["auto", "none", ""]:
        kwargs["language"] = language.lower()

    raw_text = ""
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_tried = False

    # Tier 1 (PRIMARY): Deepgram Nova-3 (~120-150ms ultra-low latency)
    if deepgram_key:
        try:
            global _DEEPGRAM_SESSION, _DEEPGRAM_KEY
            if _DEEPGRAM_SESSION is None or _DEEPGRAM_KEY != deepgram_key:
                import requests
                _DEEPGRAM_SESSION = requests.Session()
                _DEEPGRAM_SESSION.headers.update({
                    "Authorization": f"Token {deepgram_key}",
                    "Content-Type": "audio/wav"
                })
                _DEEPGRAM_KEY = deepgram_key

            dg_model = os.getenv("DEFAULT_STT_MODEL", "nova-3").strip()
            if "flux" in dg_model.lower():
                dg_model = "nova-3"
            dg_url = f"https://api.deepgram.com/v1/listen?model={dg_model}&smart_format=true&punctuate=true"
            if language and language.lower() not in ["auto", "none", ""]:
                dg_url += f"&language={language.lower()}"
            
            # Boost core technical frameworks
            keywords = ["Kubernetes:2", "FastAPI:2", "PostgreSQL:2", "Docker:2", "JWT:2", "gRPC:2", "Redis:2", "GraphQL:2"]
            for kw in keywords:
                dg_url += f"&keywords={kw}"

            # Prepare in-memory WAV buffer
            wav_bytes = data
            if not data.startswith(b"RIFF"):
                import io, wave
                bio = io.BytesIO()
                with wave.open(bio, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(data)
                wav_bytes = bio.getvalue()

            try:
                dg_resp = _DEEPGRAM_SESSION.post(dg_url, data=wav_bytes, timeout=4.0)
            except Exception:
                _DEEPGRAM_SESSION = requests.Session()
                _DEEPGRAM_SESSION.headers.update({
                    "Authorization": f"Token {deepgram_key}",
                    "Content-Type": "audio/wav"
                })
                dg_resp = _DEEPGRAM_SESSION.post(dg_url, data=wav_bytes, timeout=4.0)
            if dg_resp.status_code == 200:
                res_json = dg_resp.json()
                channels = res_json.get("results", {}).get("channels", [])
                if channels and channels[0].get("alternatives"):
                    raw_text = channels[0]["alternatives"][0].get("transcript", "").strip()
        except Exception as dg_err:
            print(f"Deepgram Nova-3 notice: {dg_err}, falling back to Groq/OpenAI...")

    # Tier 2 (HARDWARE ACCELERATION): Groq Whisper (whisper-large-v3-turbo, ~150-250ms)
    if not raw_text and client:
        try:
            groq_tried = True
            transcription = client.audio.transcriptions.create(**kwargs)
            if isinstance(transcription, str):
                raw_text = transcription.strip()
            elif hasattr(transcription, "text"):
                raw_text = transcription.text.strip()
            else:
                raw_text = str(transcription).strip()
        except Exception as groq_err:
            print(f"Primary Groq STT notice: {groq_err}, falling back to OpenAI...")

    # Tier 3 (FALLBACK): OpenAI Whisper (whisper-1) when Deepgram & Groq fail or client is not present
    if not raw_text and openai_key:
        try:
            global _GLOBAL_OPENAI_CLIENT, _GLOBAL_OPENAI_KEY
            if '_GLOBAL_OPENAI_CLIENT' not in globals() or _GLOBAL_OPENAI_CLIENT is None or _GLOBAL_OPENAI_KEY != openai_key:
                import importlib
                openai_mod = importlib.import_module("openai")
                _GLOBAL_OPENAI_CLIENT = openai_mod.OpenAI(api_key=openai_key, max_retries=1, timeout=10.0)
                _GLOBAL_OPENAI_KEY = openai_key
            oai_client = _GLOBAL_OPENAI_CLIENT
            oai_kwargs = dict(kwargs)
            oai_kwargs["model"] = "whisper-1"
            resp = oai_client.audio.transcriptions.create(**oai_kwargs)
            raw_text = resp.strip() if isinstance(resp, str) else getattr(resp, "text", "").strip()
        except Exception as oai_err:
            print(f"OpenAI STT fallback notice: {oai_err}")

    # Reject Whisper / STT silence hallucinations
    if is_whisper_hallucination(raw_text):
        return ""

    # Apply phonetic ASR post-correction
    return apply_technical_asr_corrections(raw_text)