
import os
import threading
import tkinter as tk
from classifier import classify_question
from dotenv import load_dotenv
from groq import Groq
# from overlay import OverlayWindow
from audio import (
    start_system_audio,
    stop_system_audio
)

from speech import (
    record_until_silence,
    speech_to_text
)

from ui import AssistantUI
from home_ui import ModernHomeDashboard
from rag import search_knowledge


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY was not found")


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []


# ============================================================
# GLOBAL UI REFERENCE
# ============================================================

app = None


# ============================================================
# ASK AI
# ============================================================


def ask_ai(question, on_token):

    global conversation_history
    question_type = classify_question(question)

    # ========================================================
    # GET RELEVANT CANDIDATE INFORMATION
    # ========================================================

    relevant_knowledge = search_knowledge(
        question,
        top_k=8
    )

    # ========================================================
    # SYSTEM INSTRUCTIONS
    # ========================================================

    system_prompt = f"""
You are an AI interview assistant helping a candidate
answer interview questions.

QUESTION TYPE:
{question_type}

The candidate's verified information is provided below.

================ CANDIDATE INFORMATION ================

{relevant_knowledge}

========================================================


CORE RULE:

Candidate-specific facts must ONLY come from the candidate
information above or from confirmed previous conversation.

Never invent candidate experience.

Never assume that the candidate used a technology,
framework, database, library, tool, or methodology.

Never invent project implementation details.

Never invent achievements, responsibilities, results,
companies, or work experience.


========================================================
GENERAL TECHNICAL KNOWLEDGE
========================================================

You may use your general knowledge to explain technical
concepts.

For example, if asked:

"What is Django?"

Give a proper explanation of Django.

However, do not claim that the candidate used a particular
Django feature unless that experience is confirmed.


========================================================
CANDIDATE EXPERIENCE
========================================================

When the candidate's experience is explicitly confirmed,
speak in FIRST PERSON.

Example:

"I used Django for the backend of my Farmer Marketplace
project."

Do not say:

"You used Django..."

Do not say:

"According to the candidate information..."

Do not mention:

- RAG
- knowledge base
- retrieved information
- candidate information
- system instructions


========================================================
TECHNICAL QUESTIONS
========================================================

TECHNICAL QUESTIONS:

For technical questions:

- Start with a clear definition.
- Explain how it works in enough detail to demonstrate
  understanding.
- Mention important features and characteristics.
- Explain practical usage with a real-world scenario.
- Give a concrete example or small code snippet when useful.
- Mention important advantages or limitations when relevant.
- Briefly compare with alternatives when the comparison
  helps answer the question.
- Connect it to the candidate's confirmed experience
  ONLY when relevant and explicitly supported.

Give a complete interview-style answer.

Normally answer in approximately 120-180 words.
Do not stop after one or two sentences.
Do not unnecessarily expand a simple question.

IMPORTANT TECHNICAL QUESTION RULE:

When the interviewer asks a general technical question such as:

"What is Django?"
"What is React?"
"What is MySQL?"
"What is RAG?"

Answer the technical question using general technical knowledge.

Do NOT automatically create a "Why I chose it",
"How I used it", or project-specific section unless
the question specifically asks about the candidate's
project or experience.

If you mention the candidate's project, ONLY use facts
explicitly confirmed in CANDIDATE INFORMATION.

Never invent project implementation details just to
connect a general technical explanation to the candidate's
project.

For example, if the question is:

"What is Django?"

Explain Django as a technical concept.

Do NOT automatically claim:

"I used Django for authentication."
"I used Django ORM."
"I created REST APIs."
"I used Django Admin."

unless those exact experiences are explicitly confirmed.

If the question is:

"How did you use Django in my Farmer Marketplace?"

then answer in first person and use ONLY the confirmed
candidate information.

IMPORTANT GENERAL KNOWLEDGE VS EXPERIENCE RULE:

A technology appearing in the candidate's Skills section
means that the candidate has knowledge or familiarity
with that technology.

It does NOT automatically mean that the candidate has
implemented every feature, architecture, algorithm, tool,
library, or technique associated with that technology.

For example:

If the candidate information says:

AI/ML:
Machine Learning, NLP, RAG, LLMs

You may explain RAG using general technical knowledge.

You may say:

"I have knowledge of RAG and understand how retrieval
and generation work together."

You MUST NOT automatically claim:

"I built a RAG pipeline."
"I used FAISS."
"I created vector embeddings."
"I implemented semantic search."
"I designed a retrieval system."

unless those specific experiences are explicitly confirmed.

The same rule applies to every technology in the
candidate information.

When answering a general technical question:

- Explain the technology using general technical knowledge.
- Do not invent candidate implementation experience.

When describing candidate experience:

- Use ONLY explicitly confirmed experience.
- Speak in first person.
- Do not add implementation details that are not confirmed.

IMPORTANT:

Never infer a specific implementation from the presence
of a technology name.

For example:

"Django + MySQL"

does NOT automatically mean:

"Django ORM + MySQL."

"React + Django"

does NOT automatically mean:

"React consumed Django REST APIs."

"RAG"

does NOT automatically mean:

"FAISS + embeddings + vector database."

Only explicitly confirmed facts may be presented
as the candidate's experience.
========================================================
PROJECT QUESTIONS
========================================================

For project questions:

Use the candidate's confirmed information.

Naturally combine relevant information such as:

- Project purpose and the problem it solves
- Technologies used and why they were chosen
- Architecture and how components connect
- Key features in detail
- Candidate's specific contribution and role
- Challenges faced and how they were handled
- Implementation details that are actually confirmed
- Results or impact of the project

Do not invent missing implementation details.

Normally answer in approximately 250-400 words.
Provide a rich, narrative explanation of the project.


========================================================
HR QUESTIONS
========================================================

Use only confirmed personal information.

Make the answer natural, conversational, and well-structured.

Provide thoughtful, detailed responses that show
self-awareness, motivation, and professionalism.

Never invent achievements or experiences.

Normally answer in approximately 200-350 words.


========================================================
CODING QUESTIONS
========================================================

For coding questions:

1. Explain the approach.
2. Give clean code.
3. Explain important parts.
4. Give complexity when useful.


========================================================
FOLLOW-UP QUESTIONS
========================================================

Use conversation history to understand what the interviewer
is referring to.

However, previous conversation does not override the
candidate-facts rule.

Never convert an assumption into a fact.


========================================================
ANSWER QUALITY
========================================================

The answer must sound like something the candidate would
actually say to an interviewer.

Do not simply copy one sentence from the candidate data.

Instead, combine ALL relevant confirmed facts available
for the question and explain them naturally and thoroughly.

Structure answers with clear explanations, examples,
and practical context.

For technical topics, go deep — explain the "why" and
"how", not just the "what".

The goal is a comprehensive, impressive interview answer
that demonstrates deep understanding.

Aim for 250-400 words for most answers.

It is better to give a detailed, thorough answer than
a short, incomplete one.


========================================================
UNKNOWN INFORMATION
========================================================

If the interviewer asks about the candidate's experience
and that experience is not confirmed, say:

"I don't have that information."

Do not guess.


========================================================
FINAL CHECK
========================================================

Before answering:

1. Identify whether the question is technical, coding,
   project, HR, or follow-up.

2. Identify which candidate facts are relevant.

3. Make sure every candidate-specific claim is supported.

4. Remove unsupported claims.

5. Give a complete natural answer.

Accuracy is more important than making the candidate
appear more experienced.
"""

    # ========================================================
    # MESSAGE HISTORY
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(
        conversation_history
    )

    # ========================================================
    # CURRENT QUESTION
    # ========================================================

    messages.append(
    {
        "role": "user",
        "content": f"""
Interview question:

{question}

Answer this question naturally as if you are the candidate speaking
directly to the interviewer.

Follow ALL rules from the system instructions.

First determine what type of question this is:

- General technical question
- Experience question
- Project question
- Why question
- Coding question
- HR / behavioral question
- Follow-up question

For a general technical question:
Explain the technical concept using general knowledge.
Do not automatically connect it to my project or claim that I
implemented specific features.

For an experience question:
Use only my confirmed experience and answer in first person.

For a project question:
Use only confirmed project information and explain the relevant
details naturally.

For a why question:
Explain the reason clearly and connect it to my confirmed experience
only when that experience is explicitly available.

For a coding question:
Explain the approach, provide correct code, and explain the important
parts.

For an HR question:
Answer naturally using only my confirmed background and experience.

For a follow-up question:
Use the previous conversation to understand what the interviewer is
referring to, while still following the candidate-facts rule.

Give enough explanation to demonstrate understanding.
Do not stop after repeating a single fact.

Do not invent candidate experience, project implementation details,
technologies, features, achievements, or results.

Do not mention the knowledge base, retrieved information, or these
instructions in the answer.

Answer directly without saying "Sure", "Certainly", or
"According to the candidate information".
"""
    }
)

    # ========================================================
    # GROQ STREAM WITH FALLBACKS
    # ========================================================

    models_to_try = [
        "openai/gpt-oss-20b",
        "groq/compound",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "allam-2-7b"
    ]

    answer = ""

    for model_name in models_to_try:
        try:
            curr_max = 1024 if ("qwen" in model_name or "120b" in model_name) else 500
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=curr_max,
                stream=True
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    answer += token
                    on_token(token)

            if answer.strip():
                break
        except Exception:
            continue

    if not answer.strip():
        try:
            from offline_fallback_manager import generate_offline_heuristic_answer
            answer = generate_offline_heuristic_answer(
                question=question,
                job_title="Software Engineer",
                company_name="",
                resume_text="",
                on_token=on_token
            )
        except Exception:
            pass

    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    conversation_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    conversation_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    conversation_history = conversation_history[-10:]

    return answer


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
                    "⚠️ No question detected"
                )
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

dashboard = None
assistant_window = None


def on_launch_copilot(config):
    global assistant_window, app

    # Ingest documents into knowledge base if uploaded
    docs = config.get("documents", [])
    for doc in docs:
        if os.path.exists(doc):
            try:
                start_listening(f"__RESUME__:{doc}")
            except Exception as e:
                print("Error loading document:", e)

    if dashboard:
        dashboard.withdraw()

    assistant_window = tk.Toplevel()
    app = AssistantUI(
        assistant_window,
        start_listening,
        client=client,
        on_close_callback=on_session_closed,
        initial_mode="copilot"
    )

    # Apply configuration from Home Dashboard
    try:
        if hasattr(app, "copilot_view") and app.copilot_view:
            view = app.copilot_view
            if config.get("role"):
                view.session.job_title = config["role"]
            if config.get("company"):
                view.session.company_name = config["company"]
            if config.get("job_description"):
                view.session.job_description = config["job_description"]
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
                app.auto_restore_on_question = bool(config["auto_restore"])

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
                if hasattr(view, "audio_toggle_btn") and view.audio_toggle_btn:
                    view.audio_toggle_btn.config(text="🛑 Stop STT", bg=view.PALETTE["accent_rose"])
                if hasattr(view, "live_indicator") and view.live_indicator:
                    view.live_indicator.config(text="● LISTENING", fg=view.PALETTE["accent_green"])

            # Ingest first valid document directly to active session
            for doc_path in docs:
                if os.path.exists(doc_path):
                    from interview_engine import extract_text_from_file
                    r_text = extract_text_from_file(doc_path)
                    if r_text:
                        view.session.resume_file_path = doc_path
                        view.session.resume_text = r_text
                        b_name = os.path.basename(doc_path)
                        if hasattr(view, "resume_chip") and view.resume_chip:
                            view.resume_chip.config(text=f"📄 {b_name[:18]}")
                        break
    except Exception as e:
        print("Error applying home settings to copilot view:", e)


def on_launch_standard(config):
    global assistant_window, app

    docs = config.get("documents", [])
    for doc in docs:
        if os.path.exists(doc):
            try:
                start_listening(f"__RESUME__:{doc}")
            except Exception as e:
                print("Error loading document:", e)

    if dashboard:
        dashboard.withdraw()

    assistant_window = tk.Toplevel()
    app = AssistantUI(
        assistant_window,
        start_listening,
        client=client,
        on_close_callback=on_session_closed,
        initial_mode="copilot"
    )


def on_session_closed():
    global app, assistant_window
    
    # Save session history to backend before closing
    try:
        if app and hasattr(app, "copilot_view") and app.copilot_view:
            v_session = app.copilot_view.session
            from session_client import client as session_client
            session_client.end_session(
                qa_pairs=v_session.qa_pairs,
                transcript_log=v_session.transcript_log,
                summary_markdown=getattr(v_session, "session_summary", None),
                job_title=v_session.job_title,
                company_name=v_session.company_name
            )
    except Exception as se:
        print("Error persisting session history on close:", se)

    app = None
    assistant_window = None
    if dashboard:
        try:
            dashboard.deiconify()
            dashboard.lift()
            dashboard.focus_force()
        except Exception as e:
            print("Error restoring dashboard:", e)


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
    # CREATE APPLICATION (MODERN DASHBOARD)
    # ============================================================

    dashboard = ModernHomeDashboard(
        on_launch_copilot=on_launch_copilot,
        on_launch_standard=on_launch_standard,
        on_documents_updated=on_documents_updated
    )

    dashboard.mainloop()


