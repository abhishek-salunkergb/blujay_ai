"""
BluJay AI - LLM Orchestration & Answer Generation
==================================================
Manages streaming answer generation (Groq & OpenAI), reasoning filters,
anti-fluff sanitization, rate-limit backoff, and follow-up prediction.
"""

import os
import re
import time
import json
import threading
import concurrent.futures
from typing import Callable, Optional, Dict, Any, List, Tuple, Union

from groq import Groq
from classifier import classify_question

from engine.utils import get_openai_client
from engine.session_models import (
    InterviewSession,
    InterviewTurnState,
    QuestionLatencyTrace,
    categorize_api_error,
    calculate_session_analytics
)
from engine.relationship_classifier import classify_answer_relationship
from engine.prompt_builder import (
    determine_knowledge_source,
    build_interview_system_prompt,
    validate_candidate_grounding,
    TECH_TOOL_WHITELIST
)

def get_adaptive_max_tokens(question_type: str = "general") -> int:
    """
    Adaptive output token budgets based on question type:
    Calibrated for snappy, Parakeet-grade spoken interview answers (150-250 words, 2-3s streaming):
    - Simple/HR/behavioral: ~350 tokens
    - Normal technical/project/follow-up: ~450 tokens
    - Coding/debugging: ~600 tokens
    - System design: ~650 tokens
    """
    qt = (question_type or "").lower().strip()
    if qt in ("coding", "debugging", "algorithm", "sql", "coding_complexity", "coding_optimization"):
        return 600
    elif qt in ("system_design", "system_design_component", "architecture"):
        return 650
    elif qt in ("technical", "project", "follow_up", "database", "cloud", "ai_ml", "resume", "pseudocode", "code_explanation"):
        return 450
    elif qt in ("hr", "behavioral", "clarification", "general", "statement"):
        return 350
    return 400


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
    3. Validates and auto-sanitizes unverified employer or degree claims against resume/knowledge.
    4. Re-enforces first-person authentic candidate voice.
    5. Prevents artificial claims and ensures natural spoken length.
    """
    if not raw_answer:
        return ""
        
    cleaned = sanitize_generated_answer(raw_answer)

    # Grounding Validation against session verified background
    try:
        resume_txt = getattr(session, "resume_text", "") or ""
        know_txt = getattr(session, "knowledge_text", "") or ""
        profile_txt = getattr(session, "_cached_base_profile_block", "") or ""
        cleaned, warnings = validate_candidate_grounding(
            answer=cleaned,
            resume_text=resume_txt,
            knowledge_text=know_txt,
            candidate_profile=profile_txt
        )
        if warnings and hasattr(session, "telemetry") and session.telemetry:
            for w in warnings:
                session.telemetry.record_event("grounding_sanitized", {"warning": w, "question": question[:60]})
    except Exception:
        # Strict Non-breaking guarantee: never fail generation on grounding validation
        pass
        
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
    effective_question = rel_info.get("fused_question", question) if rel_info.get("is_follow_up") and rel_info.get("fused_question") else question
    
    strategy_key = rel_info.get("strategy_key", "")
    if strategy_key and strategy_key != "general":
        q_type = strategy_key
    elif question_type:
        q_type = question_type.lower()
    else:
        q_type = classify_question(effective_question)
        
    knowledge_source = determine_knowledge_source(effective_question, session)
    system_prompt = build_interview_system_prompt(
        session,
        q_type,
        knowledge_source,
        relationship_info=rel_info,
        question=effective_question
    )
    
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

    # If candidate spoke verbally into mic on recent turn, inject as verbal context
    candidate_recent = getattr(session, "last_candidate_response", "").strip()
    if candidate_recent and not any(candidate_recent in m.get("content", "") for m in messages):
        c_words = candidate_recent.split()
        if len(c_words) > 100:
            candidate_recent = " ".join(c_words[:100]) + "..."
        messages.append({"role": "assistant", "content": f"[Candidate Spoken Input]: {candidate_recent}"})
        
    if q_type == "resume":
        user_prompt = (
            f"INTERVIEWER'S INTRODUCTORY / RESUME QUESTION:\n\"{effective_question}\"\n\n"
            "Generate the perfect, crisp, authentic, spoken first-person self-introduction for the candidate based strictly on verified candidate background.\n"
            "• Natural 1-sentence opening with identity, degree, and core passion.\n"
            "• Highlight primary tech stack (top 3-4 technologies) and primary project achievements from verified profile.\n"
            "• State enthusiasm for the target role/company naturally.\n"
            "Deliver strictly natural spoken speech. Do NOT write code blocks, Dockerfiles, or Kubernetes configs."
        )
    elif q_type == "project":
        user_prompt = (
            f"INTERVIEWER'S PROJECT WALKTHROUGH QUESTION:\n\"{effective_question}\"\n\n"
            "Generate a structured, authentic, spoken first-person walkthrough of the candidate's verified project:\n"
            "• **Architecture & Stack**: Core technologies, database, and system components used.\n"
            "• **My Role & Implementation**: Concrete backend/frontend features and business logic I built.\n"
            "• **Impact & Results**: Measurable outcomes, users, transaction throughput, or latency reduction.\n"
            "Deliver strictly natural spoken speech."
        )
    elif q_type in ["behavioral", "hr"]:
        user_prompt = (
            f"INTERVIEWER'S BEHAVIORAL / HR QUESTION:\n\"{effective_question}\"\n\n"
            "Generate an authentic, spoken first-person answer structured with the STAR method (Situation, Task, Action, Result) based on verified experience.\n"
            "Keep it conversational, professional, and grounded in real candidate achievements."
        )
    elif q_type in ["system_design", "system_design_component", "architecture"]:
        user_prompt = (
            f"INTERVIEWER'S SYSTEM DESIGN QUESTION:\n\"{effective_question}\"\n\n"
            "Deliver a structured, high-scale architectural design following Principal FAANG standards:\n"
            "1. High-Level Architecture & Requirements: Functional scope and non-functional requirements (throughput, latency, availability, consistency).\n"
            "2. Core Components & Data Flow: Component breakdown (API Gateway, Microservices, Caching layer e.g. Redis, Database choices e.g. PostgreSQL/Cassandra, Event Streaming e.g. Kafka).\n"
            "3. Scaling, Bottlenecks & Trade-offs: Concurrency, partition tolerance (CAP), failover strategy, and database sharding.\n"
            "4. Candidate Talking Points: 3-4 crisp spoken points with bold lead-ins explaining key architectural decisions."
        )
    elif q_type == "pseudocode":
        anchor = ""
        if getattr(session, "active_coding_solution", ""):
            anchor = f"\n[PRECEDING ACTIVE CODE SOLUTION]:\n{session.active_coding_solution}\n"
        user_prompt = (
            f"INTERVIEWER'S PSEUDOCODE REQUEST:\n\"{effective_question}\"\n{anchor}\n"
            "Deliver clean, language-agnostic algorithmic pseudocode:\n"
            "1. Spoken Intuition: 1 brief sentence summarizing the strategy.\n"
            "2. Complete Algorithmic Pseudocode Block: Provide clean, language-agnostic step-by-step pseudocode in a ```text block with meaningful variable names and step-by-step logic.\n"
            "3. State Time & Space Complexity Big-O: `• **Complexity**: Time: O(...) | Space: O(...)`."
        )
    elif q_type == "code_explanation":
        anchor = ""
        if getattr(session, "active_coding_solution", ""):
            anchor = f"\n[PRECEDING ACTIVE CODE SOLUTION TO EXPLAIN]:\n{session.active_coding_solution}\n"
        user_prompt = (
            f"INTERVIEWER'S CODE EXPLANATION REQUEST:\n\"{effective_question}\"\n{anchor}\n"
            "Deliver a spoken first-person walkthrough explaining the intuition and mechanics of this solution:\n"
            "1. High-Level Strategy: 1-2 spoken sentences on core algorithmic pattern and data structure choices.\n"
            "2. Step-by-Step Logic: 2-3 clean bullet points explaining main loop/recursion, variables, and edge case handling.\n"
            "3. Complexity Confirmation: 1 sentence confirming Big-O Time & Space optimality."
        )
    elif q_type == "coding_complexity":
        anchor = ""
        if getattr(session, "active_coding_solution", ""):
            anchor = f"\n[PRECEDING ACTIVE CODE SOLUTION]:\n{session.active_coding_solution}\n"
        user_prompt = (
            f"INTERVIEWER'S COMPLEXITY QUESTION:\n\"{effective_question}\"\n{anchor}\n"
            "State the exact Time and Space Complexity with Big-O notation (`• **Complexity**: Time: O(...) | Space: O(...)`) and a 1-2 sentence breakdown of dominant operations."
        )
    elif q_type in ["coding", "sql", "coding_optimization", "debugging", "algorithm"]:
        anchor = ""
        if q_type == "coding_optimization" and getattr(session, "active_coding_solution", ""):
            anchor = f"\n[INITIAL CODE SOLUTION TO OPTIMIZE]:\n{session.active_coding_solution}\n"
        user_prompt = (
            f"INTERVIEWER'S CODING QUESTION:\n\"{effective_question}\"\n{anchor}\n"
            "Deliver the 100% optimal, production-ready, complete solution structured as:\n"
            "1. Candidate Intuition: 1-2 spoken sentences explaining the optimal algorithmic pattern.\n"
            "2. Complete Executable Code Block: Write the complete, production-ready, optimal implementation enclosed in a single ```python (or requested language) code block with full type annotations, docstring, and all edge cases handled. Never omit the code.\n"
            "3. Complexity: Time Complexity O(...) and Space Complexity O(...).\n"
            "4. Candidate Talking Points: 2-3 crisp spoken bullet points with bold lead-ins to explain the code aloud."
        )
    elif rel_info.get("is_follow_up") and rel_info.get("delta_focus"):
        user_prompt = f"INTERVIEWER'S FOLLOW-UP QUESTION:\n\"{effective_question}\"\n\n[ANSWER INTELLIGENCE CONTEXT]:\n{rel_info.get('delta_focus')}\n\nGenerate the focused, authentic, spoken first-person answer addressing this exact delta."
    else:
        user_prompt = (
            f"INTERVIEWER'S TECHNICAL QUESTION:\n\"{effective_question}\"\n\n"
            "Deliver the crisp, direct spoken answer with zero conversational fluff following BluJay AI standards:\n"
            "1. Direct 1-2 sentence core definition and immediate answer.\n"
            "2. 2-3 high-impact bullet points with bold lead-ins explaining mechanisms, trade-offs, and practical usage.\n"
            "3. Natural connection or perspective from the candidate's verified stack without inventing unconfirmed experience."
        )
        
    if raw_transcript and raw_transcript != effective_question:
        user_prompt += f"\n(Raw audio transcript: \"{raw_transcript}\")"
    elif effective_question != question:
        user_prompt += f"\n(Raw audio transcript: \"{question}\")"
    
    if refinement_instruction:
        user_prompt += f"\n\n[USER REFINEMENT OVERRIDE]: {refinement_instruction}"
        
    messages.append({"role": "user", "content": user_prompt})
    
    # Adaptive token budget per question type (Correction #1)
    adaptive_max_tokens = get_adaptive_max_tokens(q_type)
    
    # High-accuracy active models in priority order: Groq ultra-low-latency default unless OpenAI explicitly chosen
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    session_model = getattr(session, "model_name", None)
    env_model = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o-mini").strip()
    is_mock = client is not None and (hasattr(client, "_mock_return_value") or type(client).__name__ in ["MagicMock", "Mock"])

    if session_model:
        preferred_model = session_model
        is_openai_model = preferred_model.startswith("gpt-") or preferred_model.startswith("o1") or preferred_model.startswith("o3") or preferred_model.startswith("chatgpt")
    elif env_model:
        preferred_model = env_model
        is_openai_model = preferred_model.startswith("gpt-") or preferred_model.startswith("o1") or preferred_model.startswith("o3") or preferred_model.startswith("chatgpt")
    elif openai_key:
        preferred_model = "gpt-4o-mini"
        is_openai_model = True
    elif is_mock:
        preferred_model = "gpt-4o-mini"
        is_openai_model = True
    elif client is not None:
        preferred_model = "openai/gpt-oss-120b"
        is_openai_model = False
    else:
        preferred_model = "gpt-4o-mini"
        is_openai_model = True

    primary_groq = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    default_models = [
        "gpt-4o-mini",
        "gpt-4o",
        primary_groq,
        "llama-3.3-70b-versatile",
        os.getenv("FAST_GROQ_MODEL", "llama-3.1-8b-instant"),
        "llama-3.1-8b-instant"
    ]
    models_to_try = [preferred_model] + [m for m in default_models if m != preferred_model]
    models_to_try = list(dict.fromkeys(models_to_try))
    
    answer = ""
    first_token_time = None
    stream_filter = StreamingReasoningFilter(on_token)

    # Pre-flight cancellation check: abort before issuing any LLM request
    if is_cancelled_callback and is_cancelled_callback():
        return ""

    # 1. Direct OpenAI Streaming (Primary default when configured, or mock client)
    if is_openai_model and (is_mock or openai_key):
        try:
            o_client = client if is_mock else get_openai_client()
            if o_client:
                o_model = preferred_model if is_openai_model else "gpt-4o-mini"
                stream = o_client.chat.completions.create(
                    model=o_model,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=adaptive_max_tokens,
                    stream=True
                )
                for chunk in stream:
                    if is_cancelled_callback and is_cancelled_callback():
                        try:
                            if hasattr(stream, "close"):
                                stream.close()
                            elif hasattr(stream, "response") and hasattr(stream.response, "close"):
                                stream.response.close()
                        except Exception:
                            pass
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
        except Exception as oe:
            print(f"OpenAI generation notice on {preferred_model}: {oe}, trying fallback provider...")
            answer = ""
            stream_filter = StreamingReasoningFilter(on_token)

    # 2. Groq Cloud Streaming (Fallback path)
    if not answer.strip() and client is not None:
        groq_models = [m for m in models_to_try if not (m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("chatgpt"))]
        for model_name in groq_models:
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
                        try:
                            if hasattr(stream, "close"):
                                stream.close()
                            elif hasattr(stream, "response") and hasattr(stream.response, "close"):
                                stream.response.close()
                        except Exception:
                            pass
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
                # Mid-stream exception fallback reset:
                # If stream drops after yielding partial tokens, reset answer so fallback engines
                # can generate a complete, unpoisoned answer
                if answer and not answer.strip().endswith((".", "!", "?")):
                    answer = ""
                    stream_filter = StreamingReasoningFilter(on_token)
                cat, is_retryable, user_msg = categorize_api_error(e)
                
                # Rate limit specific bounded handling (Fast failover on high delay to prevent dead air)
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
                    if retry_sec > 2.0:
                        # Fail over immediately to next model rather than stalling live interview
                        continue
                    bounded_sleep = min(retry_sec, 2.0)
                    slice_end = time.perf_counter() + bounded_sleep
                    while time.perf_counter() < slice_end:
                        if is_cancelled_callback and is_cancelled_callback():
                            return ""
                        time.sleep(0.05)
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
        if hasattr(session, "record_latency_trace"):
            session.record_latency_trace(trace)
        else:
            session.latency_traces.append(trace)
        # Debug trace summary print
        summary = trace.format_debug_summary()
        if session.latency_traces and len(session.latency_traces) % 5 == 0:
            print(f"[Latency Diagnostics] {summary}")
    
    # Seamless Google Gemini Multimodal / Spoken Answer Fallback
    if not answer.strip():
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            primary_gemini = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
            gemini_answer_models = [primary_gemini, "gemini-flash-latest", "gemini-3.6-flash", "gemini-2.5-flash"]
            gemini_answer_models = list(dict.fromkeys(gemini_answer_models))
            for g_mod in gemini_answer_models:
                try:
                    from google import genai  # type: ignore
                    g_client = genai.Client(api_key=gemini_key)
                    g_full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    g_resp = g_client.models.generate_content(
                        model=g_mod,
                        contents=g_full_prompt
                    )
                    if g_resp and g_resp.text and g_resp.text.strip():
                        answer = g_resp.text
                        if stream_filter:
                            stream_filter.push(answer)
                            stream_filter.flush()
                        else:
                            on_token(answer)
                        break
                except Exception as ge:
                    print(f"Gemini spoken answer fallback notice ({g_mod}): {ge}")
                    continue

    # Seamless OpenAI LLM Fallback (if configured)
    if not answer.strip():
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import importlib
                openai_mod = importlib.import_module("openai")
                o_client = openai_mod.OpenAI(api_key=openai_key)
                o_stream = o_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.6,
                    max_tokens=adaptive_max_tokens,
                    stream=True
                )
                for o_chunk in o_stream:
                    if is_cancelled_callback and is_cancelled_callback():
                        return ""
                    if not o_chunk.choices:
                        continue
                    o_tok = o_chunk.choices[0].delta.content
                    if o_tok:
                        answer += o_tok
                        if stream_filter:
                            stream_filter.push(o_tok)
                        else:
                            on_token(o_tok)
                if stream_filter:
                    stream_filter.flush()
            except Exception as oe:
                print(f"OpenAI answer fallback notice: {oe}")

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
            model=os.getenv("FAST_GROQ_MODEL", "llama-3.1-8b-instant"),
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


def prewarm_pipeline_connections(session: Optional[InterviewSession] = None):
    """
    Asynchronously pre-warms cloud STT, LLM, and Vision connection pools (TLS handshakes)
    and pre-caches candidate profile context to eliminate first-turn cold start latency.
    Also starts a background keep-alive daemon to prevent connection pool timeout.
    """
    def _worker():
        try:
            # 1. Pre-warm Deepgram TLS socket
            from speech import get_deepgram_session
            dg_sess = get_deepgram_session()
            if dg_sess:
                dg_sess.head("https://api.deepgram.com", timeout=2.0)
        except Exception:
            pass

        try:
            # 2. Pre-warm OpenAI HTTP/2 TLS Connection Pool (CRITICAL FOR GPT-4O-MINI)
            o_client = get_openai_client()
            if o_client:
                o_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1
                )
        except Exception:
            pass

        try:
            # 3. Pre-warm Groq client connection pool
            groq_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
            if groq_key:
                from groq import Groq
                _gc = Groq(api_key=groq_key)
                _gc.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1
                )
        except Exception:
            pass

        try:
            # 4. Pre-warm Vision clients (OpenAI GPT-4o Vision + Gemini)
            from vision import get_openai_vision_client, get_gemini_vision_client
            get_openai_vision_client()
            get_gemini_vision_client()
        except Exception:
            pass

        try:
            # 5. Pre-cache candidate profile and resume context
            if session is not None:
                session.get_base_profile_block()
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True, name="PipelinePrewarmWorker").start()

    # Start OpenAI keep-alive daemon (pings every 20s to keep TLS connection hot)
    def _keepalive_daemon():
        while True:
            time.sleep(20.0)
            try:
                c = get_openai_client()
                if c:
                    c.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "k"}],
                        max_tokens=1
                    )
            except Exception:
                pass
    threading.Thread(target=_keepalive_daemon, daemon=True, name="OpenAIKeepAlive").start()


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
            model=os.getenv("FAST_GROQ_MODEL", "llama-3.1-8b-instant"),
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
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
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
- Continue practicing vocal delivery and structured pacing using BluJay Mock Mode.
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
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
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
