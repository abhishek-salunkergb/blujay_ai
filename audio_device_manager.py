"""
Audio Device Discovery, Hardware Management & Level Meter Service
=================================================================
Discovers available microphones (candidate input) and speakers/loopback
devices (interviewer audio from Zoom/Meet/Teams), provides live level meters,
and manages persistent hardware preferences.
"""

import os
import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except Exception:
    sc = None
    SOUNDCARD_AVAILABLE = False


PREFERENCES_FILE = os.path.expanduser("~/.ai_copilot_devices.json")


def get_available_microphones() -> List[Dict[str, Any]]:
    """
    Enumerate all available audio input devices (microphones).
    Returns a list of dicts with: id, name, hostapi, channels, is_default.
    """
    devices = []
    default_input_idx = -1
    
    if not SOUNDDEVICE_AVAILABLE:
        return [{"id": "default", "name": "Default Microphone (System Default)", "display_name": "Default Microphone (System Default)", "is_default": True}]

    try:
        device_list = sd.query_devices()
        default_devices = sd.default.device
        default_input_idx = default_devices[0] if isinstance(default_devices, (list, tuple)) else -1
        hostapis = sd.query_hostapis()
    except Exception as e:
        print(f"Notice querying sounddevice inputs: {e}")
        return [{"id": "default", "name": "Default Microphone (System Default)", "display_name": "Default Microphone (System Default)", "is_default": True}]

    for idx, d in enumerate(device_list):
        max_inputs = d.get("max_input_channels", 0)
        if max_inputs <= 0:
            continue
            
        dev_name = d.get("name", f"Microphone {idx}")
        hostapi_idx = d.get("hostapi", 0)
        hostapi_name = hostapis[hostapi_idx].get("name", "") if hostapi_idx < len(hostapis) else ""
        
        # Format a user-friendly display name
        display_name = f"{dev_name} ({hostapi_name})" if hostapi_name and hostapi_name not in dev_name else dev_name
        is_default = (idx == default_input_idx)
        
        devices.append({
            "id": idx,
            "name": dev_name,
            "display_name": display_name,
            "channels": max_inputs,
            "hostapi": hostapi_name,
            "is_default": is_default
        })

    # Always ensure a default option exists
    if not devices:
        devices.append({"id": "default", "name": "Default Microphone (System Default)", "display_name": "Default Microphone (System Default)", "is_default": True})
        
    return devices


def get_available_speakers() -> List[Dict[str, Any]]:
    """
    Enumerate all available audio output devices (speakers/headphones) for WASAPI loopback.
    Returns a list of dicts with: id, name, display_name, is_default.
    """
    speakers = []
    
    # 1. First attempt soundcard (authoritative for WASAPI loopback on Windows)
    if SOUNDCARD_AVAILABLE:
        try:
            default_spk = None
            try:
                default_spk = sc.default_speaker()
            except Exception:
                pass
            default_name = default_spk.name if default_spk else ""

            all_spks = sc.all_speakers()
            for spk in all_spks:
                spk_name = getattr(spk, "name", str(spk))
                is_def = (spk_name == default_name)
                speakers.append({
                    "id": spk_name,
                    "name": spk_name,
                    "display_name": f"{spk_name} (Default)" if is_def else spk_name,
                    "is_default": is_def
                })
        except Exception as e:
            print(f"Notice querying soundcard speakers: {e}")

    # 2. Fallback to sounddevice if soundcard found nothing
    if not speakers and SOUNDDEVICE_AVAILABLE:
        try:
            device_list = sd.query_devices()
            default_output_idx = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else -1
            for idx, d in enumerate(device_list):
                if d.get("max_output_channels", 0) > 0:
                    name = d.get("name", f"Speaker {idx}")
                    is_def = (idx == default_output_idx)
                    speakers.append({
                        "id": str(idx),
                        "name": name,
                        "display_name": f"{name} (Default)" if is_def else name,
                        "is_default": is_def
                    })
        except Exception as e:
            print(f"Notice querying sounddevice outputs: {e}")

    if not speakers:
        speakers.append({
            "id": "default",
            "name": "Default System Audio (Loopback)",
            "display_name": "Default System Audio (Loopback)",
            "is_default": True
        })
        
    return speakers


# ============================================================
# DEVICE PREFERENCES PERSISTENCE
# ============================================================

def get_saved_device_preferences() -> Dict[str, Any]:
    """Retrieve saved hardware device preferences."""
    if os.path.exists(PREFERENCES_FILE):
        try:
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "mic_device_id": None,
        "mic_device_name": None,
        "speaker_device_id": None,
        "speaker_device_name": None
    }


def save_device_preferences(
    mic_device_id: Optional[Union[int, str]] = None,
    mic_device_name: Optional[str] = None,
    speaker_device_id: Optional[str] = None,
    speaker_device_name: Optional[str] = None
):
    """Save user preferred audio devices to disk."""
    prefs = get_saved_device_preferences()
    if mic_device_id is not None:
        prefs["mic_device_id"] = mic_device_id
    if mic_device_name is not None:
        prefs["mic_device_name"] = mic_device_name
    if speaker_device_id is not None:
        prefs["speaker_device_id"] = speaker_device_id
    if speaker_device_name is not None:
        prefs["speaker_device_name"] = speaker_device_name
        
    try:
        with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"Notice saving device preferences: {e}")


# ============================================================
# LIVE HARDWARE LEVEL METER PROBE
# ============================================================

class AudioLevelProbe:
    """
    Background worker that continuously captures short audio frames
    from a specified device and reports normalized RMS level (0.0 to 1.0).
    """

    def __init__(
        self,
        device_type: str,  # 'mic' or 'speaker'
        device_id: Optional[Union[int, str]] = None,
        on_level_update: Optional[Any] = None
    ):
        self.device_type = device_type
        self.device_id = device_id
        self.on_level_update = on_level_update
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._probe_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None

    def _probe_loop(self):
        sample_rate = 16000
        block_size = 800  # 50ms chunks

        if self.device_type == "mic" and SOUNDDEVICE_AVAILABLE:
            dev_idx = None
            if self.device_id is not None and self.device_id != "default":
                try:
                    dev_idx = int(self.device_id)
                except (ValueError, TypeError):
                    dev_idx = None

            try:
                with sd.InputStream(
                    samplerate=sample_rate,
                    blocksize=block_size,
                    channels=1,
                    dtype="int16",
                    device=dev_idx
                ) as stream:
                    while self.is_running:
                        data, overflow = stream.read(block_size)
                        if data is not None and len(data) > 0:
                            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                            rms = float(np.sqrt(np.mean(samples ** 2)))
                            # Normalize RMS (silence ~100, moderate ~3000, loud ~18000)
                            normalized = min(1.0, max(0.0, (rms - 100.0) / 6000.0))
                            if self.on_level_update:
                                self.on_level_update(normalized, rms)
                        time.sleep(0.04)
            except Exception as e:
                print(f"Mic level probe notice: {e}")
                
        elif self.device_type == "speaker" and SOUNDCARD_AVAILABLE:
            try:
                spk = None
                if self.device_id and self.device_id != "default":
                    try:
                        all_s = sc.all_speakers()
                        for s in all_s:
                            if s.name == self.device_id:
                                spk = s
                                break
                    except Exception:
                        pass
                if spk is None:
                    spk = sc.default_speaker()

                loopback = sc.get_microphone(id=str(spk.name), include_loopback=True)
                with loopback.recorder(samplerate=sample_rate, channels=1, blocksize=block_size) as rec:
                    while self.is_running:
                        data = rec.record(numframes=block_size)
                        if data is not None and len(data) > 0:
                            samples = (data * 32767.0).astype(np.float32)
                            rms = float(np.sqrt(np.mean(samples ** 2)))
                            normalized = min(1.0, max(0.0, (rms - 80.0) / 5000.0))
                            if self.on_level_update:
                                self.on_level_update(normalized, rms)
                        time.sleep(0.04)
            except Exception as e:
                print(f"Speaker loopback level probe notice: {e}")
