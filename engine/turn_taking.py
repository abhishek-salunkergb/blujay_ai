"""
BluJay AI - Turn Taking & Utterance Understanding
==================================================
Handles speaker turn detection, utterance segmentation, ambiguity buffering,
and classifies interview utterances (coding, behavioral, system design, technical).
"""

import re
import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Tuple, Union

from groq import Groq
from classifier import classify_question
from speech import is_conversational_chitchat
from engine.session_models import (
    InterviewSession,
    InterviewTurnState,
    QuestionStabilityState,
    QuestionLatencyTrace
)
from engine.relationship_classifier import (
    clean_asr_transcript,
    compute_question_confidence,
    is_likely_question,
    refine_interview_question,
    classify_answer_relationship,
    analyze_conversation_flow,
    evaluate_question_completeness,
    compute_question_likelihood
)

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
    original_raw_text = raw_text
    is_manual_input = bool(is_manual or source in ("manual", "MANUAL_INPUT", "typed"))
    input_source = "MANUAL_INPUT" if is_manual_input else "LIVE_INTERVIEW"
    
    if not raw_text:
        return {
            "raw_text": "",
            "speaker": "UNKNOWN",
            "utterance_type": "FILLER",
            "question_confidence": 0.0,
            "question_state": "LOW",
            "hold_for_context": False,
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
        # CHANNEL-FIRST ATTRIBUTION: candidate_mic is ALWAYS the Candidate channel.
        # Physical audio source is ground truth. Interrogative heuristics must NOT
        # override the physical channel — a candidate asking a clarifying question
        # must never trigger answer generation.
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
            "question_state": "NON_QUESTION",
            "hold_for_context": False,
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
            "question_state": "NON_QUESTION",
            "hold_for_context": False,
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
        "scalability", "authentication", "caching", "sql", "nosql",
        # Additional standalone declarative words that are never standalone questions:
        "performance", "security", "concurrency", "latency", "throughput",
        "microservices", "api", "database", "optimization", "sharding",
        "replication", "logging", "monitoring", "deployment", "ci/cd",
    }
    if u_lower in tech_keywords and not has_question_mark:
        # Single-word technical term without '?'.
        # Promote to MEDIUM: put it in the pending_medium_question buffer
        # so if the next interviewer chunk completes it into a real question
        # (e.g. "Redis" + "How did you use Redis?") the merge fires.
        # If no follow-up arrives within the TTL, the standalone word is
        # also stored in pending_ambiguous_utterance for context injection.
        if session:
            import uuid as _uuid_mod_kw
            _kw_buf_id = _uuid_mod_kw.uuid4().hex
            session.pending_medium_question = {
                "text":       raw_text,
                "raw_text":   raw_text,
                "confidence": 0.50,
                "timestamp":  time.time(),
                "source":     source,
                "speaker":    "INTERVIEWER",
                "buffer_id":  _kw_buf_id,
            }
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
            "question_confidence": 0.50,
            "question_state": "MEDIUM",
            "hold_for_context": True,
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
        "that's interesting", "let me think", "no problem", "sure thing",
        "fair enough", "that's fine", "that is fine", "that's right",
        "that is correct", "that's correct", "that makes sense",
        "i understand", "i got it", "exactly", "sir", "ma'am", "no sir", "no ma'am"
    }

    # Extended acknowledgement & conversational lead-in prefix normalization
    _ack_lead_in_pattern = re.compile(
        r"^(yeah|yep|yup|yes|okay|ok|alright|all right|right|sure|i see|got it|perfect|great|cool|fine)"
        r"(\s*,\s*|\s+)"
        r"(that's fine|that makes sense|sounds good|no problem|exactly|i understand|that is correct|that's correct|i got it|sir|ma'am)?"
        r"(\s*,\s*|\s+)?"
        r"(so|and|but|also|then)?"
        r"(\s*,\s*|\s+)?",
        re.IGNORECASE
    )

    _clean_u = u_lower.rstrip(".!, ")
    _is_exact_ack = _clean_u in acknowledgements

    _ack_match = _ack_lead_in_pattern.match(raw_text)
    if _ack_match:
        _rem_text = raw_text[_ack_match.end():].strip()
        _rem_clean = _rem_text.lower().strip().rstrip(".!, ")
        _is_rem_ack = not _rem_clean or _rem_clean in acknowledgements
    else:
        _rem_text = raw_text
        _rem_clean = _clean_u
        _is_rem_ack = False

    # Standalone acknowledgement: suppress immediately
    if _is_exact_ack or (_ack_match is not None and _is_rem_ack):
        return {
            "raw_text": original_raw_text,
            "speaker": interviewer_name,
            "utterance_type": "ACKNOWLEDGEMENT",
            "question_confidence": 0.0,
            "question_state": "NON_QUESTION",
            "hold_for_context": False,
            "should_generate_answer": False,
            "refined_question": original_raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }

    # ---------------------------------------------------------
    # 4b. INTERVIEWER REPHRASE / RETRACTION DETECTION
    # ---------------------------------------------------------
    rephrase_patterns = [
        r"\b(?:actually,?\s+let\s+me\s+rephrase|let\s+me\s+re-?ask|wait,?\s+scratch\s+that|forget\s+that,?\s+let's\s+talk\s+about|nevermind,?\s+tell\s+me\s+about|wait,?\s+let\s+me\s+ask\s+that\s+differently|let\s+me\s+restate\s+that)\b"
    ]
    is_rephrase = any(re.search(pat, u_lower) for pat in rephrase_patterns)
    if is_rephrase:
        # Invalidate active in-flight generation if session is currently answering
        if session and hasattr(session, "controller") and session.controller:
            try:
                session.controller.active_generation_id = getattr(session.controller, "active_generation_id", 0) + 1
                session.controller.set_state(InterviewTurnState.WAITING_FOR_NEXT_QUESTION, reason="Interviewer rephrased question")
            except Exception:
                pass
        if session and hasattr(session, "pending_medium_question"):
            session.pending_medium_question = None
        return {
            "raw_text": original_raw_text,
            "speaker": interviewer_name,
            "utterance_type": "INTERVIEWER_REPHRASE",
            "question_confidence": 0.20,
            "question_state": "NON_QUESTION",
            "hold_for_context": True,
            "should_generate_answer": False,
            "refined_question": original_raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }

    # Conversational lead-in with remainder: normalize raw_text and proceed to existing pipeline
    if _ack_match and _rem_text:
        raw_text = _rem_text
        u_lower = raw_text.lower().strip()
        words = u_lower.split()
        cleaned_local = clean_asr_transcript(raw_text)
        
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
            "question_state": "NON_QUESTION",
            "hold_for_context": False,
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
            "question_state": "NON_QUESTION",
            "hold_for_context": False,
            "should_generate_answer": False,
            "refined_question": raw_text,
            "context_topic": getattr(session, "active_topic", ""),
            "question_id": ""
        }

    # Social & Conversational Logistics Filtering (including compound small talk & mixed utterances)
    _social_patterns = [
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening|day))[\s\.\,\!\?]*$",
        r"^how are you( doing)?( today)?\??$",
        r"^(can|could) you hear me( okay| clearly| alright)?\??$",
        r"^(can|could) you see my screen\??$",
        r"^(can|could) you see the screen\??$",
        r"^are you ready( to start)?\??$",
        r"^is everything (okay|alright|fine)\??$",
        r"^(can|could) you give me a (second|moment|min)\??$",
        r"^let('s| us) (get started|begin|move on)$",
        r"^thank(s| you)( so much)?(\.|\,)?( thanks,?\s*[a-z]+)?$",
        r"^(give|just) (me )?(a )?(second|moment|min|sec)(.*)?$",
        r"^(hold on|wait a (minute|second|moment)|let me check)(.*)?$",
        r"^let('s| us) see(.*)?$",
        r"^let me ask you (a question|something|something else)(.*)?$",
        r"^i('m| am) going to ask you (a question|something|something else)(.*)?$"
    ]
    
    # Check if utterance contains multiple clauses (e.g. "How are you doing today? Can you hear me okay?")
    _raw_clauses = [c.strip() for c in re.split(r"[\?\!\.\,]+", raw_text) if c.strip()]
    if _raw_clauses:
        _clause_is_social = []
        for cl in _raw_clauses:
            cl_lower = cl.lower()
            is_soc = (
                cl_lower in greetings
                or cl_lower in fillers
                or any(re.search(pat, cl_lower) for pat in _social_patterns)
            )
            _clause_is_social.append(is_soc)
        
        # If ALL clauses are social/logistical, ignore entire utterance
        if all(_clause_is_social):
            return {
                "raw_text": raw_text,
                "speaker": interviewer_name,
                "utterance_type": "STATEMENT",
                "question_confidence": 0.10,
                "question_state": "NON_QUESTION",
                "hold_for_context": False,
                "should_generate_answer": False,
                "refined_question": raw_text,
                "context_topic": getattr(session, "active_topic", ""),
                "question_id": ""
            }
        # If mixed (e.g. "How are you doing today? Can you explain how you designed the API?"), strip social lead-in
        elif any(_clause_is_social):
            non_social_clauses = [cl for cl, is_soc in zip(_raw_clauses, _clause_is_social) if not is_soc]
            if non_social_clauses:
                raw_text = " ".join(non_social_clauses)
                if not raw_text.endswith("?"):
                    raw_text += "?"
                u_lower = raw_text.lower().strip()
                words = u_lower.split()
                cleaned_local = clean_asr_transcript(raw_text)

    # Conversational Statements (including introductory phrases before the actual question)
    conversational_statements = [
        r"^i('m| am) going to ask you (a question|something)",
        r"^i('ve| have) got a picture of you",
        r"^i live in (the city of )?",
        r"^thank(s| you)( so much)?(\.|\,)?( thanks,?\s*[a-z]+)?$",
        r"^i('m| am) (not going to talk to you|looking at your (profile|resume))",
        r"^let('s| us) (get started|begin|move on)$",
        r"^can you hear me( clearly)?\??$",
        r"^are you ready\??$",
        r"^(actually,?\s*)?let me (rephrase|reword|restate|ask that|ask this)(.*)?$",
        r"^(actually,?\s*)?(scratch that|never mind|ignore that|disregard that)(.*)?$",
        r"^(so )?how we (handle|handled|do|manage|built|use|deploy|run|solve) this (in|for|at|on) .*",
        r"^(so )?why we (chose|picked|used|went with|decided on) .* (was|is) because.*"
    ]
    for pattern in conversational_statements:
        if re.search(pattern, u_lower):
            return {
                "raw_text": raw_text,
                "speaker": interviewer_name,
                "utterance_type": "STATEMENT",
                "question_confidence": 0.1,
                "question_state": "LOW",
                "hold_for_context": False,
                "should_generate_answer": False,
                "refined_question": raw_text,
                "context_topic": getattr(session, "active_topic", ""),
                "question_id": ""
            }

    # ---------------------------------------------------------
    # DUPLICATE QUESTION DETECTION & SUPPRESSION (TIME-BOUNDED)
    # ---------------------------------------------------------
    if session and (getattr(session, "qa_pairs", None) or getattr(session, "last_question", "")):
        last_q_ts = getattr(session, "last_question_ts", 0.0)
        time_since_last_q = time.time() - last_q_ts if last_q_ts > 0 else 999.0
        # Only suppress duplicates within an active 8-second time window
        if time_since_last_q <= 8.0:
            norm_current = re.sub(r"[^a-z0-9]", "", u_lower)
            last_q = getattr(session, "last_question", "")
            if not last_q and getattr(session, "qa_pairs", None):
                last_q = session.qa_pairs[-1].get("question", "")
            norm_last = re.sub(r"[^a-z0-9]", "", (last_q or "").lower())
            is_dup = (
                norm_current
                and norm_last
                and (
                    norm_current == norm_last
                    or (len(norm_current) >= 15 and (norm_current in norm_last or norm_last.endswith(norm_current)))
                )
            )
            if is_dup and not is_manual_input:
                return {
                    "raw_text": raw_text,
                    "speaker": interviewer_name,
                    "utterance_type": "DUPLICATE",
                    "question_confidence": 0.0,
                    "question_state": "NON_QUESTION",
                    "hold_for_context": False,
                    "should_generate_answer": False,
                    "refined_question": raw_text,
                    "reason": "DUPLICATE_QUESTION_SUPPRESSED",
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

    # ---------------------------------------------------------
    # 4.6. MEDIUM-CONFIDENCE BUFFER MERGE
    # If a previous MEDIUM-confidence interviewer utterance is pending and the current
    # chunk is ALSO from the interviewer (not the candidate), merge them and re-evaluate
    # the combined text through the full pipeline.
    #
    # Safety rules:
    #   - Only merge when source is interviewer/system (never candidate_mic).
    #   - The buffer is cleared BEFORE the recursive call to prevent infinite loops.
    #   - The buffer_id from the timer is checked by the UI flush; the recursive call
    #     here does not need to check it because it consumes the buffer unconditionally.
    #   - If the buffer has already expired (TTL > 0.35 s), discard and process current chunk fresh.
    # ---------------------------------------------------------
    _MEDIUM_TTL_S = 0.35  # seconds; calibrated for 350ms low-latency response
    if (
        session is not None
        and not is_manual_input
        and source not in ("candidate_mic", "candidate_channel", "mic", "candidate")
        and speaker in ("INTERVIEWER", "MANUAL")
    ):
        with getattr(session, "_state_lock", threading.RLock()):
            pmq = getattr(session, "pending_medium_question", None)
            if pmq and isinstance(pmq, dict):
                pmq_text = pmq.get("raw_text", pmq.get("text", ""))
                pmq_time = pmq.get("timestamp", 0)
                if pmq_text and (time.time() - pmq_time) <= _MEDIUM_TTL_S:
                    # Within TTL: merge pending + current and re-evaluate.
                    combined_raw = f"{pmq_text.rstrip('?., ')} {raw_text}".strip()
                    # Clear buffer BEFORE recursive call (prevents infinite recursion).
                    session.pending_medium_question = None
                    return classify_and_understand_utterance(
                        combined_raw,
                        session=session,
                        client=client,
                        source=source,
                        speaker_hint=speaker_hint,
                        is_manual=is_manual
                    )
                else:
                    # TTL expired before a matching chunk arrived — discard the stale buffer
                    # silently; the UI timer's race-safety check will handle the flush result.
                    session.pending_medium_question = None

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
            "question_state": "LOW",
            "hold_for_context": False,
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
            p_time = p_info.get("timestamp", 0.0)
            # Only fuse if the ambiguous term was uttered recently (<= 10s) and current text is an elliptical query
            if time.time() - p_time <= 10.0:
                p_term = p_info.get("text", "")
                if p_term and p_term.lower() not in raw_text.lower():
                    if re.search(r"\bit\b", raw_text, flags=re.IGNORECASE):
                        raw_text = re.sub(r"\bit\b", p_term, raw_text, flags=re.IGNORECASE)
                    elif re.search(r"^(what about|how about|and how about|why not)\b", raw_text, flags=re.IGNORECASE):
                        raw_text = f"{raw_text.rstrip('?')} regarding {p_term}?"
        session.pending_ambiguous_utterance = None

    flow = analyze_conversation_flow(raw_text, session)
    flow_type = flow.get("flow_type", "NEW_QUESTION")
    
    # Assign unique question ID for active interviewer questions
    if session:
        with getattr(session, "_state_lock", threading.RLock()):
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
    from screen_analyzer import is_visual_reference_question, fuse_voice_and_screen_question
    from classifier import classify_question
    
    screen_ctx = None
    if session and getattr(session, "screen_aware_coding", True):
        mock_screen = getattr(session, "mock_screen_data", None)
        if mock_screen is not None:
            screen_ctx = mock_screen
            session.screen_context_cache = {"data": mock_screen, "timestamp": time.time()}
        elif is_visual_reference_question(raw_text):
            # Read cached session screen context in 0ms without blocking the voice loop
            s_cache = getattr(session, "screen_context_cache", None)
            now = time.time()
            if s_cache and s_cache.get("data") and (now - s_cache.get("timestamp", 0) < 15.0):
                screen_ctx = s_cache["data"]
            else:
                from screen_analyzer import start_async_screen_refresh
                start_async_screen_refresh(client=client, session=session)
                screen_ctx = s_cache.get("data") if s_cache else None

        if screen_ctx and (screen_ctx.get("is_coding_related") or screen_ctx.get("has_coding_problem") or screen_ctx.get("problem_statement") or screen_ctx.get("code")):
            title = screen_ctx.get("problem_title", "").strip()
            desc = screen_ctx.get("problem_statement") or screen_ctx.get("problem_description") or screen_ctx.get("visible_question", "")
            if title and desc and title.lower() not in desc.lower():
                prob_desc = f"{title}: {desc}"
            else:
                prob_desc = desc or title
            if prob_desc:
                session.active_coding_problem = prob_desc
            lang = screen_ctx.get("language") or screen_ctx.get("detected_language")
            if lang and lang.lower() not in ("unknown", "none"):
                session.active_coding_language = lang.capitalize()
            
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
            "question_state": "HIGH",
            "hold_for_context": False,
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
        rel = flow.get("relationship") or {}
        strat_key = rel.get("strategy_key", "")
        is_coding_follow_up = strat_key in ("code_explanation", "pseudocode", "coding_complexity", "coding_optimization", "coding")
        # Incorporate candidate's recent response context into follow-up if applicable (non-coding only)
        if session and not is_coding_follow_up:
            cand_ctx = getattr(session, "recent_candidate_context", "") or getattr(session, "last_candidate_response", "")
            if cand_ctx:
                cand_resp_snippet = cand_ctx[:120].rstrip(" .")
                if "regarding" not in fused_q.lower() and "context" not in fused_q.lower():
                    fused_q = f"{raw_text.rstrip('?')} (following candidate's mention of: '{cand_resp_snippet}')?"
                
        if trace:
            trace.mark("context_ready")
            trace.mark("question_finalized")
        u_type = strat_key.upper() if (is_coding_follow_up and strat_key) else "FOLLOW_UP"
        return {
            "raw_text": raw_text,
            "speaker": interviewer_name,
            "utterance_type": u_type,
            "question_confidence": 0.95,
            "question_state": "HIGH",
            "hold_for_context": False,
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
            "question_state": "HIGH",
            "hold_for_context": False,
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

    # INCOMPLETE FRAGMENT / TRAILING COMPLETENESS GUARD:
    # Trailing fragments ("...", "..") or incomplete question starters without clause completion ("what if...")
    # are held in pending_medium_question.
    # CRITICAL: Genuine short questions ("What is indexing", "Explain sharding", "Define polymorphism")
    # MUST NOT be treated as incomplete fragments.
    _has_trailing_ellipsis = bool(re.search(r"(\.\.\.|\.\.|\…)\s*$", raw_text.strip()))
    _word_count = len(cleaned_local.split())
    _is_question_intent = (
        has_question_mark
        or is_likely_question(cleaned_local)
        or auto_conf >= 0.70
        or bool(re.search(r"^(what|why|how|explain|describe|define|tell me|walk me|can you|could you)\b", u_lower))
    )
    _is_incomplete_starter = (
        bool(re.search(r"^(what if|suppose we|what about|how do we)\s+[\w\s]{1,20}$", u_lower))
        and not has_question_mark
        and not any(sci in u_lower for sci in [" empty", " null", " none", " zero", " works", " works?", " scaled", " handled", " choose", " chose", " implement", " design", " database", " table", " indexing", " sharding", " replication", " caching", " api", " event loop", " polymorphism"])
    )
    _is_incomplete_fragment = (
        not is_manual_input
        and (
            _has_trailing_ellipsis
            or (
                not _is_question_intent
                and (
                    _is_incomplete_starter
                    or (_word_count <= 3 and not has_question_mark and u_lower not in {"why", "how", "explain", "and why", "and how", "why so", "how so"})
                )
            )
        )
    )
    if _is_incomplete_fragment and (is_q_heuristic or auto_conf >= 0.70 or _has_trailing_ellipsis):
        # Treat as MEDIUM regardless of raw confidence score.
        # The next chunk will complete it into a proper question (or the TTL will expire).
        if session is not None:
            import uuid as _uuid_mod_sf
            _sf_buf_id = _uuid_mod_sf.uuid4().hex
            session.pending_medium_question = {
                "text":       cleaned_local,
                "raw_text":   raw_text,
                "confidence": min(auto_conf, 0.70),   # cap at top of MEDIUM band
                "timestamp":  time.time(),
                "source":     source,
                "speaker":    "INTERVIEWER",
                "buffer_id":  _sf_buf_id,
            }
        if trace:
            trace.mark("context_ready")
        return {
            "raw_text":            raw_text,
            "speaker":             interviewer_name,
            "utterance_type":      "INCOMPLETE_FRAGMENT",
            "question_confidence": min(auto_conf, 0.70),
            "question_state":      "MEDIUM",
            "hold_for_context":    True,
            "should_generate_answer": False,
            "refined_question":    raw_text,
            "context_topic":       getattr(session, "active_topic", ""),
            "question_id":         "",
            "input_source":        input_source
        }
    
    if is_manual_input:
        is_q = True
        confidence = 1.0
        q_type = classify_question(cleaned_local).upper()
        refined_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local.rstrip(". ") + "?")
    elif is_q_heuristic or auto_conf >= 0.70:
        # Fast path: instant rule-based refinement without an extra blocking 2.5s LLM roundtrip
        is_q = True
        refined_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local.rstrip(". ") + "?")
        # Use the actual computed confidence instead of artificially boosting to 0.90.
        # This lets the threshold gate (>= 0.60 below) remain meaningful.
        confidence = auto_conf
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
        
    # ---------------------------------------------------------
    # 3-LEVEL CONFIDENCE GATE
    #
    # HIGH  (>= 0.75): Strong evidence this is a question → generate immediately.
    # MEDIUM (0.50 – 0.74): Ambiguous → hold in pending buffer, wait for next chunk.
    # LOW   (< 0.50): Unlikely question → transcript only, no generation.
    #
    # Threshold rationale:
    #   compute_question_confidence() outputs cluster around 0.92–0.98 (strong questions),
    #   0.80 (single technical keyword in context), and 0.20 (default fallback).
    #   The old 0.60 gate caused 0.80 (e.g. lone word "Scalability") to trigger answers.
    #   MEDIUM band [0.50, 0.75) captures the 0.80 single-keyword case plus any LLM
    #   fallback scores returned between 0.50 and 0.74, holding them for one more chunk.
    # ---------------------------------------------------------
    _CONF_HIGH   = 0.75
    _CONF_MEDIUM = 0.50

    if is_manual_input:
        # Manual typed questions always generate immediately — bypass confidence gate.
        should_answer     = True
        hold_for_context  = False
        question_state    = "HIGH"
    elif is_q and confidence >= _CONF_HIGH:
        # HIGH: Generate immediately.
        should_answer     = True
        hold_for_context  = False
        question_state    = "HIGH"
    elif is_q and confidence >= _CONF_MEDIUM:
        # Check if this question starts with an unambiguous clean syntactic question stem
        # e.g., "Can you explain...", "How does...", "What is the difference between..."
        # If it is clean and has a complete stem and sufficient length, promote to HIGH immediately
        clean_stem_pattern = r"^(can you|could you|how do|how does|how would|what is|what are|why do|why does|explain|describe|tell me about)\b"
        if re.search(clean_stem_pattern, raw_text.strip(), flags=re.IGNORECASE) and len(raw_text.strip().split()) >= 4:
            should_answer     = True
            hold_for_context  = False
            question_state    = "HIGH"
        else:
            # MEDIUM: Hold briefly; wait for the next interviewer chunk to complete it.
            should_answer     = False
            hold_for_context  = True
            question_state    = "MEDIUM"
            if session is not None:
                with getattr(session, "_state_lock", threading.RLock()):
                    existing_pmq = getattr(session, "pending_medium_question", None)
                    if existing_pmq and isinstance(existing_pmq, dict):
                        # Merge in-place so earlier utterance is preserved rather than clobbered
                        existing_pmq["text"] = f"{existing_pmq.get('text', '').rstrip('?., ')} {refined_q}".strip()
                        existing_pmq["raw_text"] = f"{existing_pmq.get('raw_text', '').rstrip('?., ')} {raw_text}".strip()
                        existing_pmq["confidence"] = max(existing_pmq.get("confidence", confidence), confidence)
                        existing_pmq["timestamp"] = time.time()
                    else:
                        import uuid as _uuid_mod
                        _buf_id = _uuid_mod.uuid4().hex
                        session.pending_medium_question = {
                            "text":       refined_q,
                            "raw_text":   raw_text,
                            "confidence": confidence,
                            "timestamp":  time.time(),
                            "source":     source,
                            "speaker":    "INTERVIEWER",
                            "buffer_id":  _buf_id,
                        }
    else:
        # LOW: Discard.
        should_answer     = False
        hold_for_context  = False
        question_state    = "LOW"

    if trace:
        trace.mark("context_ready")
        if should_answer:
            trace.mark("question_finalized")

    return {
        "raw_text":              raw_text,
        "speaker":               interviewer_name,
        "utterance_type":        q_type if should_answer else "STATEMENT",
        "question_confidence":   confidence,
        "question_state":        question_state,      # NEW: "HIGH" | "MEDIUM" | "LOW"
        "hold_for_context":      hold_for_context,    # NEW: True when entering MEDIUM buffer
        "should_generate_answer": should_answer,
        "refined_question":      refined_q if should_answer else raw_text,
        "context_topic":         getattr(session, "active_topic", ""),
        "question_id":           q_id if should_answer else "",
        "input_source":          input_source
    }

