import soundcard as sc
import numpy as np
import wave
import tempfile
import threading


SAMPLE_RATE = 16000
CHANNELS = 1

_recording = False
_audio_chunks = []
_recording_thread = None


def _record_loop():

    global _recording
    global _audio_chunks

    speaker = sc.default_speaker()

    loopback = sc.get_microphone(
        id=str(speaker.name),
        include_loopback=True
    )

    _audio_chunks = []

    with loopback.recorder(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS
    ) as recorder:

        while _recording:

            data = recorder.record(
                numframes=1024
            )

            _audio_chunks.append(
                data.copy()
            )


def start_system_audio():

    global _recording
    global _recording_thread

    if _recording:
        return

    _recording = True

    _recording_thread = threading.Thread(
        target=_record_loop,
        daemon=True
    )

    _recording_thread.start()


def stop_system_audio():

    global _recording
    global _audio_chunks

    if not _recording:
        return None

    _recording = False

    if _recording_thread:
        _recording_thread.join(
            timeout=2
        )

    if not _audio_chunks:
        return None

    audio = np.concatenate(
        _audio_chunks,
        axis=0
    )

    audio = np.clip(
        audio,
        -1,
        1
    )

    audio = (
        audio * 32767
    ).astype(np.int16)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp_file.close()

    with wave.open(
        temp_file.name,
        "wb"
    ) as wav:

        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(
            audio.tobytes()
        )

    return temp_file.name