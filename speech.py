"""
High-Fidelity Audio Engineering, Diagnostics, Preprocessing & STT Pipeline
========================================================================
Source-specific DSP processors for System Loopback (Interviewer) and Microphone (Candidate),
featuring safe AGC, clipping protection, noise floor gating, and transcript overlap merging.
"""

import io
import os
import re
import wave
import tempfile
import collections
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict, Any
import sounddevice as sd
import numpy as np

try:
    import webrtcvad
    VAD_AVAILABLE = True
except Exception:
    webrtcvad = None
    VAD_AVAILABLE = False


SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

vad = webrtcvad.Vad(2) if VAD_AVAILABLE else None


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


def compute_audio_diagnostics(
    samples: np.ndarray,
    noise_floor_rms: float = 350.0
) -> AudioDiagnostics:
    """
    Computes lightweight DSP diagnostics for audio quality monitoring and adaptive processing.
    """
    if samples is None or len(samples) == 0:
        return AudioDiagnostics(
            rms=0.0,
            peak=0.0,
            speech_level_db=-96.0,
            silence_ratio=1.0,
            clipping_ratio=0.0,
            snr_estimate_db=0.0,
            state="SILENT"
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
        
    return AudioDiagnostics(
        rms=rms,
        peak=peak,
        speech_level_db=speech_level_db,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        snr_estimate_db=snr_estimate_db,
        state=state
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
    
    # If pure silence, return unboosted to prevent amplifying noise floor
    if diag.state == "SILENT" or diag.rms < noise_floor_rms * 1.25:
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
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_floor_rms = 80.0  # System loopback has very low electrical noise floor
        self.last_diagnostics: Optional[AudioDiagnostics] = None

    def process_frame(self, frame_float_or_int: np.ndarray) -> Tuple[bytes, bool, AudioDiagnostics]:
        """
        Processes a raw frame, computes diagnostics, and returns 16-bit PCM bytes.
        """
        if frame_float_or_int is None or len(frame_float_or_int) == 0:
            diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
            return b"", False, diag
            
        # Convert float32 [-1.0, 1.0] to int16 if needed
        if frame_float_or_int.dtype == np.float32 or frame_float_or_int.dtype == np.float64:
            clipped = np.clip(frame_float_or_int, -1.0, 1.0)
            int_samples = (clipped * 32767.0).astype(np.int16)
        else:
            int_samples = frame_float_or_int.astype(np.int16)
            
        # Remove DC offset
        dc_clean = remove_dc_offset(int_samples)
        
        # Calculate diagnostics
        diag = compute_audio_diagnostics(dc_clean, noise_floor_rms=self.noise_floor_rms)
        self.last_diagnostics = diag
        
        # Determine speech presence
        is_speech = (diag.rms > 150.0) and (diag.state != "SILENT")
        
        # Apply safe AGC
        leveled = safe_automatic_gain_control(
            dc_clean,
            target_rms=2800.0,
            max_gain_db=12.0,
            noise_floor_rms=self.noise_floor_rms
        )
        
        return leveled.tobytes(), is_speech, diag


class MicrophoneAudioProcessor:
    """
    DSP Processor for Microphone (Candidate Audio).
    Optimized for physical environment capture (laptop mic, headset, room reverb).
    - Dynamic noise floor tracking (laptop fans, room AC, hum).
    - Spectral noise gating with high sensitivity for quiet speech.
    - Plosive consonant pre-buffering.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_floor_rms = 120.0
        self.vad = webrtcvad.Vad(1) if VAD_AVAILABLE else None  # Mode 1: Low aggressiveness, high sensitivity
        self.last_diagnostics: Optional[AudioDiagnostics] = None

    def process_frame(self, frame_bytes: bytes) -> Tuple[bytes, bool, AudioDiagnostics]:
        """
        Processes microphone frame with dynamic noise estimation and sensitive VAD.
        """
        if not frame_bytes:
            diag = compute_audio_diagnostics(np.array([], dtype=np.int16))
            return b"", False, diag
            
        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        
        # Remove DC offset
        dc_clean = remove_dc_offset(samples)
        
        # Dynamic noise floor update
        diag = compute_audio_diagnostics(dc_clean, noise_floor_rms=self.noise_floor_rms)
        self.last_diagnostics = diag
        
        # Track background noise during silence
        if diag.rms > 20.0 and diag.rms < 400.0 and diag.state in ("SILENT", "LOW_SIGNAL"):
            self.noise_floor_rms = 0.95 * self.noise_floor_rms + 0.05 * diag.rms
            
        speech_thresh = max(110.0, self.noise_floor_rms * 1.20)
        
        if self.vad and VAD_AVAILABLE:
            try:
                vad_detected = self.vad.is_speech(dc_clean.tobytes(), self.sample_rate)
            except Exception:
                vad_detected = False
            is_speech = vad_detected or (diag.rms > speech_thresh)
        else:
            is_speech = diag.rms > speech_thresh
            
        # Apply responsive AGC for microphone (+16dB boost for soft mics)
        processed = safe_automatic_gain_control(
            dc_clean,
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
    silence_ms: int = 550,
    max_duration_s: float = 30.0,
    pre_speech_buffer_ms: int = 360
) -> str:
    """
    High-precision speech recording with pre-speech ring buffer and sub-second silence cutoff.
    Pre-speech buffer ensures the beginning of the first word is never clipped.
    """
    print("\n🎤 Listening with low-latency VAD...")

    audio_frames = []
    pre_buffer_len = max(1, int(pre_speech_buffer_ms / FRAME_DURATION))
    pre_buffer = collections.deque(maxlen=pre_buffer_len)

    max_silence_frames = max(3, int(silence_ms / FRAME_DURATION))
    max_total_frames = int((max_duration_s * 1000) / FRAME_DURATION)

    silence_frames = 0
    started_speaking = False
    total_frames = 0
    processor = MicrophoneAudioProcessor(sample_rate=SAMPLE_RATE)

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
                if not started_speaking:
                    started_speaking = True
                    # Flush pre-speech context so the first consonant is intact
                    audio_frames.extend(list(pre_buffer))
                silence_frames = 0
                audio_frames.append(proc_bytes)
            elif started_speaking:
                audio_frames.append(proc_bytes)
                silence_frames += 1
                if silence_frames >= max_silence_frames:
                    break
            else:
                pre_buffer.append(proc_bytes)

    print("✅ Speech boundary detected.")

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    with wave.open(temp_file.name, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(audio_frames))

    return temp_file.name


def speech_to_text(
    client,
    audio_source: Union[str, bytes],
    prompt: str = "Technical interview question: Python, SQL, React, AWS, Docker, Kubernetes, algorithms.",
    language: Optional[str] = "en"
) -> str:
    """Transcribes audio using Whisper-large-v3-turbo with high accuracy and speed."""
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

    if not data or len(data) < 100:
        return ""

    kwargs = {
        "file": (filename, data),
        "model": "whisper-large-v3-turbo",
        "response_format": "text",
        "temperature": 0.0,
        "prompt": prompt
    }
    if language and language.lower() not in ["auto", "none", ""]:
        kwargs["language"] = language.lower()

    transcription = client.audio.transcriptions.create(**kwargs)

    if isinstance(transcription, str):
        return transcription.strip()
    elif hasattr(transcription, "text"):
        return transcription.text.strip()
    return str(transcription).strip()