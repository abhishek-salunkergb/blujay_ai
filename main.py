
import os
import threading
import warnings
import logging
logging.getLogger("pywebview").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="soundcard")

import tkinter as tk
from classifier import classify_question
from dotenv import load_dotenv
from groq import Groq
from rag import search_knowledge
from speech import record_until_silence, speech_to_text


# ============================================================
# SYSTEM AUDIO CAPTURE HELPERS
# ============================================================

_system_audio_recording = False
_system_audio_chunks = []
_system_audio_thread = None


def start_system_audio():
    """Start recording system audio via loopback capture."""
    global _system_audio_recording, _system_audio_chunks, _system_audio_thread
    if _system_audio_recording:
        return
    _system_audio_recording = True
    _system_audio_chunks = []

    def _record_loop():
        global _system_audio_recording, _system_audio_chunks
        try:
            import soundcard as sc
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            with loopback.recorder(samplerate=16000, channels=1) as recorder:
                while _system_audio_recording:
                    data = recorder.record(numframes=1024)
                    _system_audio_chunks.append(data.copy())
        except Exception as e:
            print(f"[SystemAudio] Loopback capture notice: {e}")

    _system_audio_thread = threading.Thread(target=_record_loop, daemon=True)
    _system_audio_thread.start()


def stop_system_audio():
    """Stop recording system audio and export to WAV file."""
    global _system_audio_recording, _system_audio_chunks, _system_audio_thread
    if not _system_audio_recording:
        return None
    _system_audio_recording = False
    if _system_audio_thread:
        _system_audio_thread.join(timeout=2)
    if not _system_audio_chunks:
        return None
    try:
        import numpy as np
        import wave
        import tempfile
        audio = np.concatenate(_system_audio_chunks, axis=0)
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        with wave.open(temp_file.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio.tobytes())
        return temp_file.name
    except Exception as e:
        print(f"[SystemAudio] Error encoding system audio: {e}")
        return None


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")

# ============================================================
# GROQ CLIENT
# ============================================================

client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print("Warning initializing Groq client:", e)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []
active_topic = ""  # tracks current discussion topic for follow-up resolution


# ============================================================
# GLOBAL UI REFERENCE
# ============================================================

app = None


# ============================================================
# ASK AI (CANONICAL INTERVIEW ENGINE PIPELINE)
# ============================================================


def ask_ai(question, on_token):
    """
    Unified answer generation pipeline utilizing interview_engine.generate_interview_answer.
    Maintains 100% prompt, RAG, and relationship parity across both WebView and Tkinter runtimes.
    Enforces INTELLIGENT OUT-OF-CONTEXT INTERVIEW QUESTION STRATEGY with the
    THREE-LAYER ANSWER STRATEGY:
    "DO NOT refuse to answer and DO NOT pretend the candidate has experience".
    """
    global app, client

    v_session = None
    if app:
        if hasattr(app, "session") and app.session:
            v_session = app.session
        elif hasattr(app, "copilot_view") and app.copilot_view and hasattr(app.copilot_view, "session"):
            v_session = app.copilot_view.session

    if v_session is None:
        from interview_engine import InterviewSession
        v_session = InterviewSession()

    from interview_engine import generate_interview_answer
    return generate_interview_answer(
        client=client,
        session=v_session,
        question=question,
        on_token=on_token
    )


# ============================================================
# UPDATE ANSWER BOX
# ============================================================

def update_answer(token):

    app.root.after(
        0,
        lambda: add_token_to_answer(token)
    )


def add_token_to_answer(token):

    app.answer_box.insert(
        "end",
        token
    )

    app.answer_box.see(
        "end"
    )


# ============================================================
# START LISTENING / TYPED QUESTION
# ============================================================

def start_listening(question=None):

    thread = threading.Thread(
        target=process_question,
        args=(question,),
        daemon=True
    )

    thread.start()


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(question=None):

    
    try:

        # ==========================================
        # RESUME / CV
        # ==========================================

        if question and question.startswith("__RESUME__:"):

            resume_path = question[len("__RESUME__:"):]
            print("Resume selected:", resume_path)

            try:
                from interview_engine import extract_text_from_file
                extracted_text = extract_text_from_file(resume_path)
                
                # Copy or write to knowledge folder for RAG
                os.makedirs("knowledge", exist_ok=True)
                base_name = os.path.basename(resume_path)
                target_dest = os.path.join("knowledge", base_name)
                
                if extracted_text and not os.path.exists(target_dest):
                    try:
                        import shutil
                        shutil.copy2(resume_path, target_dest)
                    except Exception:
                        with open(target_dest, "w", encoding="utf-8", errors="ignore") as kf:
                            kf.write(extracted_text)

                if hasattr(app, "copilot_view") and app.copilot_view:
                    app.copilot_view.session.resume_file_path = resume_path
                    app.copilot_view.session.resume_text = extracted_text
                    if hasattr(app.copilot_view, "resume_chip") and app.copilot_view.resume_chip:
                        app.copilot_view.resume_chip.config(text=f"📄 {base_name[:18]}")

                # Refresh RAG cache
                try:
                    import rag
                    rag.load_documents()
                except Exception:
                    pass

                app.root.after(
                    0,
                    lambda: app.set_status(
                        f"📄 Resume loaded: {base_name[:20]}"
                    )
                )
            except Exception as re:
                print(f"Error parsing resume: {re}")
                app.root.after(
                    0,
                    lambda: app.set_status(
                        "⚠️ Error parsing resume"
                    )
                )

            return

    

        # ----------------------------------------------------
        # START SYSTEM AUDIO
        # ----------------------------------------------------

        if question == "__START_SYSTEM_AUDIO__":

            start_system_audio()

            app.root.after(
                0,
                lambda: app.set_status(
                    "🔴 Recording computer audio..."
                )
            )

            return


        # ----------------------------------------------------
        # STOP SYSTEM AUDIO
        # ----------------------------------------------------

        if question == "__STOP_SYSTEM_AUDIO__":

            app.root.after(
                0,
                lambda: app.set_status(
                    "📝 Converting audio to text..."
                )
            )

            audio_file = stop_system_audio()

            if not audio_file:

                app.root.after(
                    0,
                    lambda: app.set_status(
                        "⚠️ No audio recorded"
                    )
                )

                return

            question = speech_to_text(
                client,
                audio_file
            )


        # ----------------------------------------------------
        # MICROPHONE INPUT
        # ----------------------------------------------------

        elif question is None:

            app.root.after(
                0,
                lambda: app.set_status(
                    "🎤 Listening..."
                )
            )

            audio_file = record_until_silence()

            app.root.after(
                0,
                lambda: app.set_status(
                    "📝 Converting speech to text..."
                )
            )

            question = speech_to_text(
                client,
                audio_file
            )


        # ----------------------------------------------------
        # VALIDATE QUESTION
        # ----------------------------------------------------

        if not question or not question.strip():
            app.root.after(
                0,
                lambda: app.set_status(
                    "⚠️ No speech detected (Click mic to speak)"
                )
            )
            app.root.after(
                2500,
                lambda: app.set_status("✅ Ready")
            )
            return


        # ----------------------------------------------------
        # SHOW QUESTION
        # ----------------------------------------------------

        app.root.after(
            0,
            lambda q=question:
            app.set_question(q)
        )


        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        app.root.after(
            0,
            lambda: app.set_status(
                "🤖 Generating answer..."
            )
        )

        app.root.after(
            0,
            lambda: app.set_answer("")
        )


        ask_ai(
            question,
            update_answer
        )


        # ----------------------------------------------------
        # READY
        # ----------------------------------------------------

        app.root.after(
            0,
            lambda: app.set_status(
                "✅ Ready"
            )
        )


    except Exception as e:

        error_message = str(e)

        app.root.after(
            0,
            lambda: app.set_status(
                "❌ Error"
            )
        )

        app.root.after(
            0,
            lambda msg=error_message:
            app.set_answer(
                f"Something went wrong:\n\n{msg}"
            )
        )


# ============================================================
# WINDOW LAUNCH & ORCHESTRATION HANDLERS
# ============================================================

assistant_window = None


def apply_home_config_to_copilot(app_instance, config: dict):
    """Applies session settings from the Parakeet Home Dashboard to the live copilot overlay."""
    try:
        view = None
        if hasattr(app_instance, "copilot_view") and app_instance.copilot_view:
            view = app_instance.copilot_view
        elif hasattr(app_instance, "session"):
            view = app_instance

        if view:
            if config.get("role"):
                view.session.job_title = config["role"]
            if config.get("company"):
                view.session.company_name = config["company"]
            if config.get("job_description"):
                view.session.job_description = config["job_description"]
            if config.get("instructions"):
                view.session.custom_instructions = config["instructions"]
            if config.get("style"):
                view.session.tone = config["style"]
            if config.get("language"):
                view.session.language = config["language"]
            if config.get("model"):
                view.session.model_name = config["model"]
            if config.get("audio_source"):
                src = config["audio_source"]
                if "dual" in src.lower():
                    view.session.audio_source = "dual"
                elif "loopback" in src.lower() or "system" in src.lower():
                    view.session.audio_source = "system"
                elif "mic" in src.lower():
                    view.session.audio_source = "mic"
            if "mic_device_id" in config:
                view.session.selected_mic_device = config["mic_device_id"]
                view.session.mic_device_name = config.get("mic_device_name")
            if "speaker_device_id" in config:
                view.session.selected_speaker_device = config["speaker_device_id"]
                view.session.speaker_device_name = config.get("speaker_device_name")
            if "monitor_id" in config:
                view.session.selected_monitor_id = config["monitor_id"]
            if "auto_generate" in config:
                view.session.auto_generate = bool(config["auto_generate"])
            elif "auto_gen" in config:
                view.session.auto_generate = bool(config["auto_gen"])
            if "auto_restore" in config:
                view.session.auto_restore_on_question = bool(config["auto_restore"])
                app_instance.auto_restore_on_question = bool(config["auto_restore"])

            # Ingest first valid document directly to active session
            docs = config.get("documents", [])
            for doc_path in docs:
                if os.path.exists(doc_path):
                    if hasattr(view.session, "set_resume"):
                        view.session.set_resume(doc_path)
                    else:
                        from interview_engine import extract_text_from_file
                        r_text = extract_text_from_file(doc_path)
                        if r_text:
                            view.session.resume_file_path = doc_path
                            view.session.resume_text = r_text
                    b_name = os.path.basename(doc_path)
                    if hasattr(view, "resume_chip") and view.resume_chip:
                        view.resume_chip.config(text=f"📄 {b_name[:18]}")
                    break

            # Invalidate base profile cache so company, JD, resume, and instructions are immediately live
            if hasattr(view.session, "invalidate_profile_cache"):
                view.session.invalidate_profile_cache()

            # Refresh UI labels to reflect home settings
            role_txt = f"🎯 {view.session.job_title or 'Tech Interview'}"
            if view.session.company_name:
                role_txt += f" @ {view.session.company_name}"
            if hasattr(view, "role_badge") and view.role_badge:
                view.role_badge.config(text=role_txt)
            if hasattr(view, "source_btn") and view.source_btn:
                s_label = "🔊 Source: Dual" if view.session.audio_source == "dual" else ("🔊 Source: Loopback" if view.session.audio_source == "system" else "🎙️ Source: Mic")
                view.source_btn.config(text=s_label)
            if hasattr(view, "auto_mode_btn") and view.auto_mode_btn:
                a_label = "⚡ Auto-Gen: ON" if view.session.auto_generate else "⌨️ Auto-Gen: OFF"
                view.auto_mode_btn.config(text=a_label)

            # Auto-start Live STT on launch with selected hardware devices
            if hasattr(view, "transcriber") and not view.transcriber.is_running:
                view.transcriber.start(
                    source=view.session.audio_source,
                    mic_device_id=getattr(view.session, "selected_mic_device", None),
                    system_speaker_id=getattr(view.session, "selected_speaker_device", None)
                )
                if hasattr(view, "recording_dot") and view.recording_dot:
                    try:
                        view.recording_dot.config(text="🔴", fg="#ef4444")
                    except Exception:
                        pass
    except Exception as e:
        print("Error applying home settings to copilot view:", e)


def on_session_closed():
    global app, assistant_window
    
    # Save session history to backend before closing
    try:
        v_session = None
        if app:
            if hasattr(app, "session") and app.session:
                v_session = app.session
            elif hasattr(app, "copilot_view") and app.copilot_view and hasattr(app.copilot_view, "session"):
                v_session = app.copilot_view.session

        if v_session:
            from session_client import get_session_client
            sc = get_session_client()
            summary_md = getattr(v_session, "session_summary", None)
            if not summary_md:
                try:
                    from interview_exporter import generate_session_markdown
                    summary_md = generate_session_markdown(v_session)
                except Exception:
                    pass
            sc.end_session(
                qa_pairs=getattr(v_session, "qa_pairs", []),
                transcript_log=getattr(v_session, "transcript_log", []),
                summary_markdown=summary_md,
                job_title=getattr(v_session, "job_title", "Candidate"),
                company_name=getattr(v_session, "company_name", "Interview")
            )
    except Exception as se:
        print("Error persisting session history on close:", se)

    app = None
    assistant_window = None


def on_documents_updated(docs):
    for doc in docs:
        if os.path.exists(doc):
            try:
                start_listening(f"__RESUME__:{doc}")
            except Exception as e:
                print("Error syncing document:", e)


import time
import requests


def ensure_backend_running():
    """Ensure the authoritative FastAPI billing & session backend is running."""
    try:
        resp = requests.get("http://127.0.0.1:8000/", timeout=0.8)
        if resp.status_code == 200:
            return
    except Exception:
        pass

    def run_server():
        try:
            import uvicorn
            from backend.main import app as fastapi_app
            uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
        except Exception as e:
            print("Backend auto-start notice:", e)

    t = threading.Thread(target=run_server, daemon=True, name="BackendServerThread")
    t.start()
    time.sleep(1.0)


if __name__ == "__main__":
    ensure_backend_running()

    # ============================================================
    # APPLICATION ORCHESTRATION LIFECYCLE
    # ============================================================
    use_web = False
    try:
        import webview
        use_web = webview is not None
    except Exception:
        use_web = False

    if use_web:
        from web_copilot_view import WebCopilotView
        app = WebCopilotView(
            client=client,
            on_close_callback=on_session_closed
        )
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        except Exception as we:
            print("WebCopilot exception:", we)
        finally:
            import sys
            sys.exit(0)
    else:
        try:
            from home_ui import ModernHomeDashboard
            from legacy.ui import AssistantUI
        except ImportError:
            print("⚠️ Microsoft Edge WebView2 runtime is recommended to run BluJay AI.")
            import sys
            sys.exit(1)
        while True:
            dashboard = ModernHomeDashboard(
                on_documents_updated=on_documents_updated
            )
            launch_config = dashboard.mainloop()

            if not launch_config:
                break

            # Fallback Tkinter runtime if pywebview is unavailable
            root = tk.Tk()
            assistant_window = root

            def _handle_session_close():
                global app, assistant_window
                try:
                    on_session_closed()
                except Exception as se:
                    print("Error saving session on close:", se)
                try:
                    root.destroy()
                except Exception:
                    pass

            app = AssistantUI(
                root,
                start_listening,
                client=client,
                on_close_callback=_handle_session_close,
                initial_mode=launch_config.get("session_type", "copilot")
            )

            apply_home_config_to_copilot(app, launch_config)

            try:
                root.mainloop()
            except Exception as te:
                print("Interview UI exception:", te)
