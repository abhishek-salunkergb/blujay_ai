"""
BluJay AI - Answer Intelligence & Relationship Classifier
=========================================================
Implements Parakeet AI candidate grounding, ghost topic bleed prevention,
semantic question fusion, question stability, and flow analysis.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple, Set

from groq import Groq
from engine.session_models import InterviewSession, InterviewTurnState, QuestionStabilityState


QUESTION_STARTERS = [
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "can you", "could you", "would you", "will you", "do you", "did you", "have you",
    "tell me", "explain", "describe", "walk me through", "give me an example",
    "share a time", "discuss", "what is", "how do", "how did", "why did", "why do",
    "can we", "is it possible", "how would you", "what would you", "elaborate on",
    "what about", "how about", "and for", "so how", "so what", "in your opinion",
    "your approach", "how you", "how authentication", "how security", "why not",
    "write a", "write code", "implement", "find the", "solve", "design a"
]

INTERVIEW_QUESTION_PATTERNS = [
    r"\?$",                                           # Ends with question mark
    r"^(what|why|how|when|where|who|which)\b",        # Starts with WH-word
    r"^(can|could|would|will|do|did|have|has|are|is)\s+you\b", # Direct second-person questions
    r"^(tell me|explain|describe|define|compare|contrast|walk me through|give an example|share a time|talk about|discuss)\b",
    r"^(what about|how about|and for|so what about|why not|and what|and how)\b", # Follow-up questions
    r"^(write a|write code|implement|find the|solve|design a|debug|fix)\b",      # Coding & imperative questions
    r"\b(your experience with|your background in|how you handled|your role in|how you implemented)\b",
    r"\b(what was your approach|what were the trade-offs|why did you choose|how did you use)\b",
    r"\b(how (you|they|the|to) (implemented|handle|handled|design|designed|scale|scaled|optimize|optimized))\b",
    r"^(your approach\??|performance\??|database\??|security\??|scalability\??|caching\??)$"
]

ASR_CORRECTIONS = {
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
    r"\bpost\s*gress\b": "PostgreSQL",
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
    r"\bpy\s*torch\b": "PyTorch",
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
    r"\bsequel\b": "SQL",
    r"\bpostgress\b": "PostgreSQL",
    r"\bpostgres\b": "PostgreSQL",
    r"\bs\s*q\s*l\b": "SQL",
    r"\bdata\s*base\b": "database",
    r"\bdata\s*bases\b": "databases",
    r"\bcapturing\s+property\b": "capturing properly",
    r"\bcapturing\s+properties\b": "capturing properly",
    r"\binterruptions?\s+are\s+okay\b": "interruptions",
}


# Common Whisper hallucination patterns (generated from noise/silence input)
# Common Whisper hallucination patterns (generated from noise/silence input)
WHISPER_HALLUCINATIONS = [
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

_STANDALONE_GHOST_WORDS = {
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


def _is_whisper_hallucination(text: str) -> bool:
    """Returns True if the text matches a known Whisper hallucination pattern."""
    if not text:
        return True
    t = text.strip().lower()
    if len(t) < 2:
        return True
        
    # Check standalone single-word ghost utterances (ignoring punctuation)
    cleaned_word = re.sub(r"[^\w\s]", "", t).strip()
    if cleaned_word in _STANDALONE_GHOST_WORDS:
        return True
        
    for pattern in WHISPER_HALLUCINATIONS:
        if re.search(pattern, t, re.IGNORECASE):
            return True

    # Check for prompt echo / keyword dumping on silence
    prompt_echo_phrases = [
        "technical engineering interview",
        "discussing python",
        "fastapi docker kubernetes",
        "big-o complexity",
        "algorithms system design",
        "live professional technical interview conversation",
        "interview discussion regarding"
    ]
    if any(pe in t for pe in prompt_echo_phrases):
        return True
            
    # Check for repetitive stutter loops (e.g., "you you you you" or "thank you thank you")
    words = t.split()
    if len(words) >= 4:
        if len(set(words)) == 1:
            return True
        if len(words) % 2 == 0:
            half = len(words) // 2
            if words[:half] == words[half:]:
                return True
    return False


def clean_asr_transcript(text: str) -> str:
    """Performs fast rule-based normalization, homophone correction, hallucination filtering, and deduplication of repeated words."""
    if not text:
        return ""
    
    clean = text.strip()
    
    # 0. Filter Whisper hallucinations (ghost text from noise/silence)
    if _is_whisper_hallucination(clean):
        return ""
    
    # 1. Apply technical ASR dictionary replacements first (preserves acronyms like 'l l m' -> 'LLM')
    for pattern, replacement in ASR_CORRECTIONS.items():
        clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
        
    # 2. Deduplicate consecutive repeated phrases/words (e.g. "how did you how did you" -> "how did you")
    clean = re.sub(r'\b([A-Za-z0-9_]{2,}(?:\s+[A-Za-z0-9_]+){0,3})\s+\1\b', r'\1', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\b([A-Za-z0-9_]{2,})\s+\1\b', r'\1', clean, flags=re.IGNORECASE)
    
    return clean.strip()


def compute_question_confidence(raw_text: str, session: Optional[Any] = None) -> float:
    """
    Two-Stage Question Candidate Detector:
    Computes a continuous question intent confidence score (0.0 to 1.0).
    Ensures genuine interviewer questions are NEVER missed due to missing '?' or conversational phrasing.
    """
    if not raw_text or len(raw_text.strip()) < 2:
        return 0.0
    
    cleaned = clean_asr_transcript(raw_text)
    u_lower = cleaned.lower().strip()
    words = u_lower.split()
    
    # 0. Fast-filter known non-question declarative prefixes
    _non_question_prefixes = (
        # Meeting / setup logistics
        "can you hear me", "can you see my screen", "can you see the screen",
        "can you wait", "can you hold on",
        # Suggestions framed as questions
        "why don't we", "why not just", "why not start", "why not take",
        # Declarative how/what sentences ("What we normally do is...")
        "how i like to", "how we work", "how our team", "how our process",
        "how the interview", "how this interview",
        "how we handle", "how we handled", "how we do", "how we manage", "how we built", "how we use", "how we deploy",
        "so how we", "so why we",
        "why we chose", "why we picked", "why we used", "why we went with", "why we decided",
        "actually let me", "actually, let me", "let me rephrase", "let me reword", "let me ask that",
        "what we do is", "what we'll do", "what our team",
        "what we normally", "what we usually", "what we typically",
        "what we have", "what we've", "what we use",
        "what i do is", "what i normally", "what i usually",
        "what they do", "what they use", "what they normally",
        # Politeness / casual offers
        "tell me if you", "tell me if there",
    )
    if not raw_text.endswith("?"):
        for excl in _non_question_prefixes:
            if u_lower.startswith(excl):
                return 0.15

    # 1. Direct interrogative WH-starters & prompt commands
    wh_starters = [
        "why did you", "why didn't you", "why choose", "why not", "why",
        "tell me about", "walk me through", "explain how", "explain", "can you explain",
        "describe a time", "how did you", "how would you", "how do you", "how can we",
        "how does", "how", "what is the difference", "what are the trade-offs",
        "what about", "how about", "what database", "what framework", "what architecture",
        "which database", "what technologies", "design a", "write a function",
        "how do you handle", "how authentication", "how you implemented", "how security",
        "what was your approach", "your approach", "what did you use", "why django",
        "why react", "why redux", "why postgres", "why redis", "why mongodb",
        "can you clarify", "did you mean"
    ]
    for starter in wh_starters:
        if u_lower.startswith(starter):
            return 0.98 if "?" in raw_text else 0.95
            
    # 2. Contains question mark or rising intonation
    if "?" in raw_text:
        return 0.95
        
    # 3. Follow-up question patterns (e.g. "What about scalability?", "Why not MySQL?")
    if re.search(r"^(what about|how about|and for|so what about|why not|and what|and how)\b", u_lower):
        return 0.95
        
    # 4. Technical inquiry / architectural questions without '?'
    tech_inquiry_patterns = [
        r"\b(how (you|they|the|to) (implemented|handle|handled|design|designed|scale|scaled|optimize|optimized))\b",
        r"\b(why did you choose|what was the reason for|what were the challenges)\b",
        r"\b(what is your experience with|have you used|did you use)\b",
        r"^(explain|describe|define|compare|contrast|clarify|elaborate on|walk through)\b",
        r"^(can you|could you|would you|do you|did you|have you)\b",
        r"\b(your approach|how you implemented)\b"
    ]
    for p in tech_inquiry_patterns:
        if re.search(p, u_lower):
            return 0.92
            
    # 5. Short follow-up keywords in context (e.g. "Scalability", "Performance", "Security", "Database")
    if session and getattr(session, "turn_state", None) is not None:
        if u_lower in {"scalability", "performance", "security", "caching", "database", "authentication", "latency", "trade-offs"}:
            return 0.80
            
    # 6. General regex matching
    if is_likely_question(cleaned):
        return 0.88
        
    return 0.20


def is_likely_question(text: str) -> bool:
    """Detect whether a transcribed utterance is a question, follow-up, or interview prompt."""
    if not text:
        return False
    
    clean = clean_asr_transcript(text)
    if len(clean) < 3:
        return False
        
    lower = clean.lower()
    
    if clean.endswith("?"):
        return True
        
    # Explicit exclusion list: common non-question phrases that start with a question word.
    # These MUST be checked BEFORE pattern matching and starters.
    _non_question_prefixes = (
        # Meeting / setup logistics
        "can you hear me", "can you see my screen", "can you see the screen",
        "can you wait", "can you hold on",
        # Suggestions framed as questions
        "why don't we", "why not just", "why not start", "why not take",
        # Declarative how/what sentences ("What we normally do is...")
        "how i like to", "how we work", "how our team", "how our process",
        "how the interview", "how this interview",
        "how we handle", "how we handled", "how we do", "how we manage", "how we built", "how we use", "how we deploy",
        "so how we", "so why we",
        "why we chose", "why we picked", "why we used", "why we went with", "why we decided",
        "actually let me", "actually, let me", "let me rephrase", "let me reword", "let me ask that",
        "what we do is", "what we'll do", "what our team",
        "what we normally", "what we usually", "what we typically",
        "what we have", "what we've", "what we use",
        "what i do is", "what i normally", "what i usually",
        "what they do", "what they use", "what they normally",
        # Politeness / casual offers
        "tell me if you", "tell me if there",
    )
    for excl in _non_question_prefixes:
        if lower.startswith(excl):
            return False

    for pattern in INTERVIEW_QUESTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True

    for starter in QUESTION_STARTERS:
        if lower.startswith(starter + " ") or lower == starter:
            return True

    return False


def refine_interview_question(
    client: Optional[Groq],
    raw_transcript: str,
    session: Optional[Any] = None,
    recent_context: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Intelligently recovers the intended interview question from noisy/imperfect speech-to-text transcript.
    - Resolves ASR homophones & tech terminology (e.g. 'pie torch' -> 'PyTorch', 'property' -> 'properly').
    - Fixes broken grammar & removes vocal disfluencies / repetitions.
    - Resolves contextual follow-ups (e.g. 'What about scalability?').
    - Classifies semantic intent (TECHNICAL, BEHAVIORAL, FOLLOW_UP, STATEMENT, etc.).
    - Never invents unrelated questions.
    """
    cleaned_local = clean_asr_transcript(raw_transcript)
    if not cleaned_local or len(cleaned_local) < 3:
        return {
            "is_question": False,
            "raw_transcript": raw_transcript,
            "refined_question": cleaned_local,
            "confidence": 0.0,
            "question_type": "STATEMENT"
        }
        
    # If client is not available, return rule-based refined question
    if not client:
        is_q = is_likely_question(cleaned_local)
        ref_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local + "?" if is_q else cleaned_local)
        return {
            "is_question": is_q,
            "raw_transcript": raw_transcript,
            "refined_question": ref_q,
            "confidence": 0.7 if is_q else 0.4,
            "question_type": "TECHNICAL" if is_q else "STATEMENT"
        }
        
    # Build conversation context snippet
    context_lines = []
    if session and getattr(session, "transcript_log", None):
        for item in session.transcript_log[-4:]:
            context_lines.append(f"{item['role']}: {item['text']}")
    elif recent_context:
        for item in recent_context[-4:]:
            context_lines.append(f"{item.get('role', 'Speaker')}: {item.get('text', '')}")
            
    ctx_str = "\n".join(context_lines) if context_lines else "No previous conversation context."
    job_target = f"Role: {session.job_title} @ {session.company_name}" if (session and (getattr(session, 'job_title', '') or getattr(session, 'company_name', ''))) else "General Tech Role"
    
    prompt = f"""You are an expert Real-Time Interview Speech Refiner.
You receive a noisy, imperfect speech-to-text transcript from an interviewer during a technical/job interview.

CANDIDATE CONTEXT: {job_target}
RECENT CONVERSATION HISTORY:
{ctx_str}

LATEST RAW SPOKEN TRANSCRIPT:
"{raw_transcript}"

TASK:
1. Determine if this utterance is an interviewer question, follow-up, interview prompt, or statement requiring candidate response.
   - Note: Spoken questions often lack question marks or start as statements (e.g. "Tell me about...", "Walk me through...", "how you handled...").
   - Follow-up fragments (e.g. "What about scalability?", "And database?") must be resolved in context.
2. If it is a question or prompt:
   - Fix speech-recognition mistakes, wrong words, and homophones (e.g. "cash" -> "cache", "property" -> "properly", "pie torch" -> "PyTorch", "sequel" -> "SQL").
   - Remove stuttered words, filler words, and vocal disfluencies.
   - Formulate the cleanest, grammatically sound, intended interview question.
   - DO NOT invent topics that the interviewer did not mention. Preserve the exact core semantic meaning.
3. If it is purely conversational filler (e.g. "thanks for joining", "can you hear me", "okay"), mark is_question: false.

OUTPUT MUST BE STRICTLY VALID JSON ONLY:
{{
  "is_question": true,
  "refined_question": "Clean and precise interview question here?",
  "question_type": "TECHNICAL",
  "confidence": 0.95
}}"""

    try:
        # 1. Primary: Direct OpenAI gpt-4o-mini speech refiner when configured
        openai_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
        content = ""
        if openai_key:
            try:
                o_client = get_openai_client()
                if o_client:
                    resp = o_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a specialized JSON-only interview speech refinement engine. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=250,
                        response_format={"type": "json_object"}
                    )
                    if resp.choices and resp.choices[0].message.content:
                        content = resp.choices[0].message.content
            except Exception as o_err:
                content = ""

        # 2. Fallback: Groq chat models
        if not content and client is not None:
            models_to_try = [
                os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "llama-3.3-70b-versatile",
                os.getenv("FAST_GROQ_MODEL", "llama-3.1-8b-instant"),
                "llama-3.1-8b-instant"
            ]
            models_to_try = list(dict.fromkeys(models_to_try))
            for m in models_to_try:
                try:
                    resp = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": "You are a specialized JSON-only interview speech refinement engine. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=250,
                        response_format={"type": "json_object"}
                    )
                    content = resp.choices[0].message.content
                    if content:
                        break
                except Exception:
                    continue

        # Extract JSON substring if wrapped in markdown or conversational text
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        json_str = json_match.group(0) if json_match else content
        data = json.loads(json_str)
        
        ref_q = data.get("refined_question", "").strip() or cleaned_local
        is_q = bool(data.get("is_question", False))
        conf = float(data.get("confidence", 0.85))
        q_type = data.get("question_type", "TECHNICAL" if is_q else "STATEMENT")
        
        return {
            "is_question": is_q,
            "raw_transcript": raw_transcript,
            "refined_question": ref_q,
            "confidence": conf,
            "question_type": q_type
        }
    except Exception as e:
        is_q = is_likely_question(cleaned_local)
        ref_q = cleaned_local if cleaned_local.endswith("?") else (cleaned_local + "?" if is_q else cleaned_local)
        return {
            "is_question": is_q,
            "raw_transcript": raw_transcript,
            "refined_question": ref_q,
            "confidence": 0.6 if is_q else 0.3,
            "question_type": "TECHNICAL" if is_q else "STATEMENT"
        }


# ============================================================
# INTERVIEW TURN-TAKING & SPEAKER INTELLIGENCE
# ============================================================

# ============================================================
# ANSWER INTELLIGENCE & CONVERSATION RELATIONSHIP CLASSIFIER
# ============================================================

PRIMARY_TECH_PLATFORMS: Dict[str, str] = {
    "django": "Django", "python": "Python", "react": "React", "docker": "Docker",
    "kubernetes": "Kubernetes", "aws": "AWS", "gcp": "GCP", "azure": "Azure",
    "postgresql": "PostgreSQL", "postgres": "PostgreSQL", "mongodb": "MongoDB",
    "sql": "SQL", "nosql": "NoSQL", "graphql": "GraphQL", "rest": "REST",
    "microservices": "Microservices", "redis": "Redis", "kafka": "Kafka",
    "celery": "Celery", "rabbitmq": "RabbitMQ", "elasticsearch": "Elasticsearch",
    "cassandra": "Cassandra", "grpc": "gRPC", "fastapi": "FastAPI", "flask": "Flask"
}


def extract_primary_tech_entity(text: str) -> Optional[str]:
    """Extracts the primary concrete technology framework, database, or message broker from text."""
    if not text:
        return None
    t_lower = text.lower()
    for k in sorted(PRIMARY_TECH_PLATFORMS.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", t_lower):
            return PRIMARY_TECH_PLATFORMS[k]
    return None


def classify_answer_relationship(
    utterance: str,
    session: Optional["InterviewSession"] = None
) -> Dict[str, Any]:
    """
    Parakit Answer Intelligence Layer:
    Understands the relationship between the current interviewer question
    and the immediately preceding conversation context.
    
    Supported Relationship Types:
    - NEW_TOPIC
    - WHY
    - HOW
    - PERFORMANCE (Scalability, throughput, latency)
    - SECURITY
    - COMPARISON (Why not X vs Y)
    - EXPANSION (Explain that / elaborate)
    - CLARIFICATION
    - EXAMPLE_REQUEST
    - OPTIMIZATION (Coding optimization)
    - COMPLEXITY (Time / Space complexity)
    - TRADE_OFF (Drawbacks / limitations)
    - IMPROVEMENT (What would you change)
    - DIRECT_FOLLOW_UP
    """
    if not utterance:
        return {
            "relationship_type": "NEW_TOPIC",
            "is_follow_up": False,
            "topic": "",
            "fused_question": utterance,
            "delta_focus": "",
            "strategy_key": "general"
        }

    u_lower = utterance.lower().strip()
    u_clean = re.sub(r'[^a-z0-9\s]', '', u_lower).strip()
    words = u_clean.split()
    
    active_topic = getattr(session, "active_topic", "") if session else ""
    last_cand = getattr(session, "last_candidate_response", "") if session else ""
    last_q = ""
    if session and session.qa_pairs:
        last_q = session.qa_pairs[-1].get("question", "")
    elif session and getattr(session, "latest_detected_question", ""):
        last_q = session.latest_detected_question

    # Candidate Voice Grounding:
    # If candidate recently spoke and introduced a concrete technology that differs from active_topic,
    # the candidate's spoken utterance is the latest authoritative truth of what was discussed.
    cand_tech = extract_primary_tech_entity(last_cand) if last_cand else None
    if cand_tech and active_topic and cand_tech.lower() not in active_topic.lower():
        active_topic = cand_tech
        if session:
            session.active_topic = cand_tech
    elif not active_topic and cand_tech:
        active_topic = cand_tech
        if session:
            session.active_topic = cand_tech

    # Guard against incomplete trailing fragments with ellipses ("how would you...", "what if...")
    _has_trailing_ellipsis = bool(re.search(r"(\.\.\.|\.\.|\…)\s*$", utterance.strip()))
    if _has_trailing_ellipsis:
        return {
            "relationship_type": "NEW_TOPIC",
            "is_follow_up": False,
            "topic": active_topic,
            "fused_question": utterance,
            "delta_focus": "",
            "strategy_key": "general"
        }

    # 1. NEW_TOPIC DETECTION (Explicit Topic Shifts)
    new_topic_starters = [
        r"^(now\s+)?(tell me about|let's move on to|let's switch to|moving on to|next topic|shifting to|let's discuss your internship|let's discuss your background)\b",
        r"\b(let'?s talk about something else|changing gears|different topic|moving to a different)\b"
    ]
    for p in new_topic_starters:
        if re.search(p, u_lower):
            new_top = re.sub(p, "", utterance, flags=re.IGNORECASE).strip(" ,:;?-")
            if session:
                session.active_topic = new_top if new_top else "New Topic"
            return {
                "relationship_type": "NEW_TOPIC",
                "is_follow_up": False,
                "topic": session.active_topic if session else "",
                "fused_question": utterance,
                "delta_focus": "Fresh topic inquiry. Do not bring unrelated technical details from previous discussions.",
                "strategy_key": "general"
            }

    # Guard: declarative statement prefixes that should NEVER be treated as interrogative follow-ups
    is_declarative_statement = bool(re.search(r"^(i|we|he|she|they)\s+(know|knew|discussed|explained|saw|noticed|think|thought|understand|understood)\s+(how|why)\b", u_lower))
    is_social_greeting = bool(re.search(r"^how('s|\s+is|\s+are|\s+was|\s+have)\s+(you|your|things|everything|it going|the day|the weekend)\b", u_lower))

    # 2. COMPARISON FOLLOW-UP ("Why not MySQL?", "Why not Postgres?", "Compare with X")
    comp_match = re.search(r"\b(why not|versus|vs\.?|compare (it )?to|instead of|difference between this and)\s+([a-zA-Z0-9_\+\#]+)", u_lower)
    if comp_match or u_lower.startswith("why not"):
        raw_alt = comp_match.group(3) if comp_match else (words[-1] if words else "alternative")
        tech_acronyms = {
            "mysql": "MySQL", "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
            "mongodb": "MongoDB", "jwt": "JWT", "sql": "SQL", "nosql": "NoSQL",
            "aws": "AWS", "gcp": "GCP", "redis": "Redis", "django": "Django",
            "react": "React", "python": "Python", "docker": "Docker", "kafka": "Kafka"
        }
        alt_tech = tech_acronyms.get(raw_alt.lower(), raw_alt.capitalize())
        topic_name = active_topic or "the chosen technology"
        return {
            "relationship_type": "COMPARISON",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Why did you choose {topic_name} instead of {alt_tech} in this project, and what are the trade-offs?",
            "delta_focus": f"Compare {topic_name} (chosen) vs {alt_tech} (alternative) specifically for this project: why {topic_name} fit best, why {alt_tech} was less suitable, and the key architectural trade-offs.",
            "strategy_key": "follow_up_comparison"
        }

    # 3. WHY FOLLOW-UP
    _why_match = (
        (u_clean in {"why", "and why", "why so", "why is that", "why did you choose that", "why this approach", "why?"} or u_lower.startswith("why?") or u_lower == "why")
        or (
            (u_lower.endswith("?") or is_likely_question(utterance))
            and bool(re.search(r"^(why|and why|so why|but why)\b|\bwhy\s+(did you|would you|do you|don't you|not|is that|is it|is this|is that better|was that|choose|use|pick|decide)\b|\bwhy\s+(this|that|it|the approach|your choice|this approach|this decision)\b", u_lower))
        )
    )
    if _why_match and not is_declarative_statement:
        # Check if the utterance introduces a new distinct platform technology (e.g. "Why do we use Kafka?")
        utterance_tech = extract_primary_tech_entity(utterance)
        is_distinct_platform = bool(
            utterance_tech 
            and active_topic 
            and utterance_tech.lower() not in active_topic.lower()
            and utterance_tech.lower() in PRIMARY_TECH_PLATFORMS
        )
        if is_distinct_platform:
            topic_name = utterance_tech
            if session:
                session.active_topic = utterance_tech
            fused = utterance if utterance.endswith("?") else (utterance + "?")
            return {
                "relationship_type": "WHY",
                "is_follow_up": True,
                "topic": utterance_tech,
                "fused_question": fused,
                "delta_focus": f"Give strictly the technical justification and reasons for selecting {utterance_tech}. Do not repeat basic definitions.",
                "strategy_key": "follow_up_why"
            }

        topic_name = active_topic or "the discussed choice"
        if topic_name and topic_name.lower() not in u_lower:
            if re.search(r"\b(it|that|this|the approach|this approach|that approach|this choice)\b", utterance, re.IGNORECASE):
                fused = re.sub(r"\b(it|that|this|the approach|this approach|that approach|this choice)\b", topic_name, utterance, count=1, flags=re.IGNORECASE)
            else:
                fused = f"{utterance.rstrip('?')} regarding {topic_name}?"
        else:
            fused = utterance if utterance.endswith("?") else (utterance + "?")
        return {
            "relationship_type": "WHY",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": fused,
            "delta_focus": f"Give strictly the justification and reasons for selecting {topic_name} in the project. Do not restate general definitions.",
            "strategy_key": "follow_up_why"
        }

    # 4. HOW FOLLOW-UP
    _how_match = (
        (u_clean in {"how", "and how", "how so", "how does that work", "how did you implement that", "how do you use that", "how?"} or u_lower.startswith("how?") or u_lower == "how")
        or (
            (u_lower.endswith("?") or is_likely_question(utterance))
            and bool(re.search(r"^(how|and how|so how|but how)\b|\bhow\s+(did you|do you|would you|does it|does that|does this|is it|is that|was it|was that|to handle|to implement)\b|\bhow\s+(this|that|it|the approach|your solution|this system|this works|this is done)\b", u_lower))
        )
    )
    if _how_match and not is_declarative_statement and not is_social_greeting:
        # Check if the utterance introduces a new distinct platform technology (e.g. "How do you implement caching in Redis?")
        utterance_tech = extract_primary_tech_entity(utterance)
        is_distinct_platform = bool(
            utterance_tech 
            and active_topic 
            and utterance_tech.lower() not in active_topic.lower()
            and utterance_tech.lower() in PRIMARY_TECH_PLATFORMS
        )
        if is_distinct_platform:
            topic_name = utterance_tech
            if session:
                session.active_topic = utterance_tech
            fused = utterance if utterance.endswith("?") else (utterance + "?")
            return {
                "relationship_type": "HOW",
                "is_follow_up": True,
                "topic": utterance_tech,
                "fused_question": fused,
                "delta_focus": f"Explain the concrete implementation mechanics of {utterance_tech}.",
                "strategy_key": "follow_up_how"
            }

        topic_name = active_topic or "this component"
        if topic_name and topic_name.lower() not in u_lower:
            if re.search(r"\b(it|that|this|the approach|your solution|this system|this)\b", utterance, re.IGNORECASE):
                fused = re.sub(r"\b(it|that|this|the approach|your solution|this system|this)\b", topic_name, utterance, count=1, flags=re.IGNORECASE)
            else:
                fused = f"{utterance.rstrip('?')} regarding {topic_name}?"
        else:
            fused = utterance if utterance.endswith("?") else (utterance + "?")
        return {
            "relationship_type": "HOW",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": fused,
            "delta_focus": f"Explain the concrete implementation mechanics of {topic_name} in the project.",
            "strategy_key": "follow_up_how"
        }

    # 5. PERFORMANCE / SCALABILITY FOLLOW-UP
    # Guard: keyword match alone is insufficient — the utterance must also be interrogative.
    # Declarative statements like "I noticed good performance" must NOT trigger a follow-up.
    _perf_kw_match = any(kw in u_lower for kw in [
        "scalability", "what about scalability", "performance", "what about performance",
        "how does it scale", "traffic increases", "high traffic", "high load", "concurrency"
    ])
    _perf_is_question = u_lower.endswith("?") or re.search(
        r"^(what|how|why|can you|could you|would you|tell me|explain|describe)", u_lower
    )
    if _perf_kw_match and _perf_is_question:
        tech_explicit = re.findall(r"\b(django|python|react|docker|kubernetes|aws|postgresql|postgres|mongodb|sql|nosql|graphql|rest|microservices|redis|kafka|system design|authentication|caching|jwt)\b", u_lower)
        topic_name = tech_explicit[0].capitalize() if tech_explicit else (active_topic or "this architecture")
        if tech_explicit and session:
            session.active_topic = topic_name
        return {
            "relationship_type": "PERFORMANCE",
            "is_follow_up": True,
            "topic": topic_name,
            "fused_question": f"How does {topic_name} handle scalability, performance, and high traffic in this project?",
            "delta_focus": f"Focus strictly on scalability and performance mechanisms (horizontal scaling, sharding, caching, bottleneck avoidance) for {topic_name}. DO NOT repeat basic definitions or feature descriptions.",
            "strategy_key": "follow_up_scalability"
        }

    # 6. SECURITY FOLLOW-UP
    # Guard: keyword match alone is insufficient — the utterance must also be interrogative.
    # Declarative statements like "Security is very important here" must NOT trigger a follow-up.
    _sec_kw_match = any(kw in u_lower for kw in [
        "security", "what about security", "how do you secure", "vulnerabilities", "how is it secured"
    ])
    _sec_is_question = u_lower.endswith("?") or re.search(
        r"^(what|how|why|can you|could you|would you|tell me|explain|describe)", u_lower
    )
    if _sec_kw_match and _sec_is_question:
        tech_explicit = re.findall(r"\b(django|python|react|docker|kubernetes|aws|postgresql|postgres|mongodb|sql|nosql|graphql|rest|microservices|redis|kafka|system design|authentication|caching|jwt)\b", u_lower)
        topic_name = tech_explicit[0].capitalize() if tech_explicit else (active_topic or "this implementation")
        if tech_explicit and session:
            session.active_topic = topic_name
        return {
            "relationship_type": "SECURITY",
            "is_follow_up": True,
            "topic": topic_name,
            "fused_question": f"What security considerations and defenses were applied to {topic_name} in this project?",
            "delta_focus": f"Focus strictly on security practices (token storage, encryption, validation, threat mitigations) for {topic_name}.",
            "strategy_key": "follow_up_security"
        }

    # 7. EXPANSION FOLLOW-UP ("Can you explain that?", "Can you elaborate?")
    if any(kw in u_lower for kw in ["can you explain that", "can you elaborate", "explain that further", "tell me more about that", "can you expand", "expand on that"]):
        topic_name = active_topic or (last_cand[:40] if last_cand else "the previous explanation")
        return {
            "relationship_type": "EXPANSION",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Can you elaborate in detail on {topic_name} in this project?",
            "delta_focus": f"Expand deeply on {topic_name}, detailing how it operates under the hood, why it was valuable, and when it is useful. Do not restart the high-level introduction.",
            "strategy_key": "follow_up_expansion"
        }

    # 8. CLARIFICATION FOLLOW-UP
    if any(kw in u_lower for kw in ["what do you mean", "can you clarify", "to confirm", "are you saying"]):
        topic_name = active_topic or "the previous statement"
        return {
            "relationship_type": "CLARIFICATION",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Can you clarify what you mean regarding {topic_name}?",
            "delta_focus": f"Clarify the assumptions and precise meaning regarding {topic_name}.",
            "strategy_key": "clarification"
        }

    # 9. EXAMPLE_REQUEST FOLLOW-UP
    if any(kw in u_lower for kw in ["can you give an example", "give me an example", "show an example", "concrete example", "for example", "like what"]):
        topic_name = active_topic or "the discussed concept"
        return {
            "relationship_type": "EXAMPLE_REQUEST",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"Can you provide a concrete, real-world example of {topic_name} in your project?",
            "delta_focus": f"Provide a realistic, concrete engineering example from the project illustrating {topic_name}.",
            "strategy_key": "follow_up_example"
        }

    # Coding State Context
    has_active_coding = bool(session and (getattr(session, "active_coding_solution", "") or getattr(session, "active_coding_problem", "") or getattr(session, "active_topic", "").lower() in ("coding", "coding optimization", "complexity analysis", "algorithm", "sql", "debugging", "pseudocode", "code explanation")))
    active_prob = getattr(session, "active_coding_problem", "") if session else ""

    # 10. PSEUDOCODE FOLLOW-UP
    if any(kw in u_lower for kw in ["give me pseudocode", "write pseudocode", "show me pseudocode", "pseudocode for this", "give pseudo code", "show pseudocode", "give pseudocode", "write pseudo code", "pseudo code", "pseudocode"]):
        prob_desc = active_prob or active_topic or "the coding problem"
        return {
            "relationship_type": "PSEUDOCODE",
            "is_follow_up": True if has_active_coding else False,
            "topic": "Pseudocode",
            "fused_question": f"Provide clean algorithmic pseudocode for: {prob_desc}",
            "delta_focus": f"Generate language-agnostic step-by-step algorithmic pseudocode for {prob_desc}.",
            "strategy_key": "pseudocode"
        }

    # 11. CODE EXPLANATION FOLLOW-UP
    _code_explain_kw = [
        "explain your approach", "explain the approach", "explain your code", "walk me through your code",
        "explain the solution", "explain your solution", "how does your code work", "walk me through the solution",
        "explain this code", "explain the code", "explain the algorithmic approach", "explain the algorithmic approach and logic"
    ]
    if any(kw in u_lower for kw in _code_explain_kw) or (has_active_coding and any(kw in u_lower for kw in ["explain it", "explain this", "walk me through it", "explain the logic"])):
        prob_desc = active_prob or active_topic or "the preceding solution"
        return {
            "relationship_type": "CODE_EXPLANATION",
            "is_follow_up": True if has_active_coding else False,
            "topic": "Code Explanation",
            "fused_question": f"Explain the algorithmic approach and logic for {prob_desc}",
            "delta_focus": f"Deliver a spoken walkthrough explaining the intuition, data structures, and line-by-line mechanics of {prob_desc}. Do not re-output full raw code.",
            "strategy_key": "code_explanation"
        }

    # 12. CODE IMPLEMENTATION FOLLOW-UP ("Give me the code", "Implement this", "Now code it", "Write the code")
    _code_impl_kw = ["give me the code", "give me code", "show me the code", "show the code", "now code it", "now code this", "code this", "code it", "code it up", "write the solution", "code the solution", "write the code", "implement this", "implement that", "implement it"]
    if any(u_clean == kw or u_lower.startswith(kw) for kw in _code_impl_kw):
        prob_desc = active_prob or active_topic or "the discussed challenge"
        return {
            "relationship_type": "CODE_IMPLEMENTATION",
            "is_follow_up": True if has_active_coding else False,
            "topic": "Coding Solution",
            "fused_question": f"Provide the complete production code implementation for {prob_desc}",
            "delta_focus": f"Deliver the optimal, complete runnable code implementation for {prob_desc}.",
            "strategy_key": "coding"
        }

    # 13. OPTIMIZATION FOLLOW-UP
    _is_coding_opt_phrase = any(kw in u_lower for kw in [
        "can you optimize it", "can we optimize this", "can you optimize this", "can we do better",
        "optimize the solution", "optimize your solution", "more optimal approach", "make it faster",
        "optimize this", "optimize that", "optimize it", "make this faster", "make that faster",
        "optimize the approach", "optimize your approach", "optimize approach", "optimize solution",
        "can we improve the time complexity", "can we improve the space complexity",
        "improve time complexity", "improve the time complexity", "improve space complexity"
    ])
    if _is_coding_opt_phrase or (has_active_coding and "optimize" in u_lower and any(w in u_lower for w in ["it", "this", "solution", "approach", "code", "faster"])):
        prob_desc = active_prob or "the previous coding solution"
        return {
            "relationship_type": "OPTIMIZATION",
            "is_follow_up": True,
            "topic": "Coding Optimization",
            "fused_question": f"Can you optimize the solution for {prob_desc} for better time and space complexity?",
            "delta_focus": f"Identify the performance bottleneck of the solution for {prob_desc} and provide the optimized approach with superior Big-O complexity.",
            "strategy_key": "coding_optimization"
        }

    # 14. COMPLEXITY FOLLOW-UP
    if any(kw in u_lower for kw in ["what is the time complexity", "what is the space complexity", "what's the complexity", "what is the complexity", "time complexity", "space complexity", "what is the big o", "big o", "tell me the complexity", "complexity of this"]):
        prob_desc = active_prob or "the preceding solution"
        return {
            "relationship_type": "COMPLEXITY",
            "is_follow_up": True,
            "topic": "Complexity Analysis",
            "fused_question": f"What is the time and space complexity of the solution for {prob_desc}?",
            "delta_focus": f"State the Time and Space Complexity of {prob_desc} with Big-O notation and mathematical breakdown. Do not rewrite full code.",
            "strategy_key": "coding_complexity"
        }

    # 12. TRADE_OFF FOLLOW-UP
    if any(kw in u_lower for kw in ["what are the drawbacks", "drawbacks", "what are the limitations", "limitations", "what are the trade-offs", "trade-offs", "downside"]):
        topic_name = active_topic or "this technology choice"
        return {
            "relationship_type": "TRADE_OFF",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"What are the drawbacks and trade-offs of {topic_name} in this project?",
            "delta_focus": f"Discuss the engineering limitations and trade-offs of {topic_name} and how our team mitigated them.",
            "strategy_key": "follow_up_tradeoff"
        }

    # 13. IMPROVEMENT FOLLOW-UP
    if any(kw in u_lower for kw in ["what would you change", "how would you improve it", "how would you improve that", "what would you do differently", "future improvements"]):
        topic_name = active_topic or "this architecture"
        return {
            "relationship_type": "IMPROVEMENT",
            "is_follow_up": True,
            "topic": active_topic,
            "fused_question": f"What would you change or improve about {topic_name} based on production learnings?",
            "delta_focus": f"State specific architectural or operational improvements for {topic_name} with expected benefits.",
            "strategy_key": "follow_up_improvement"
        }

    # 14. GENERAL DIRECT FOLLOW-UP
    followup_starters = ["what about", "how about", "and for", "so what about", "then what", "and in that case"]
    is_standalone_query = bool(re.search(
        r"^(what\s+is|what\s+are|define|explain|describe|tell me about|how\s+does|how\s+do|why\s+do|why\s+does)\b",
        u_lower
    ))
    has_followup_starter = any(u_lower.startswith(s) for s in followup_starters)
    is_short_elliptical = (
        session is not None
        and bool(session.qa_pairs)
        and len(words) <= 4
        and u_lower.endswith("?")
        and not is_standalone_query
        and not bool(re.findall(r"\b(django|python|react|docker|kubernetes|aws|postgresql|postgres|mongodb|sql|nosql|graphql|rest|microservices|redis|kafka|system design|authentication|caching|jwt|indexing|sharding|replication|polymorphism|database|api)\b", u_lower))
    )
    if has_followup_starter or is_short_elliptical:
        found_new_tech = re.findall(r"\b(django|python|react|docker|kubernetes|aws|postgresql|postgres|mongodb|sql|nosql|graphql|rest|microservices|redis|kafka|system design|authentication|caching|jwt|indexing|sharding|replication|polymorphism|database|api)\b", u_lower)
        if found_new_tech:
            topic_name = found_new_tech[0].capitalize()
            fused = utterance if utterance.endswith("?") else (utterance + "?")
        else:
            topic_name = active_topic or "the discussed topic"
            fused = f"{utterance.rstrip('?')} regarding {topic_name}?"
        if found_new_tech and session:
            session.active_topic = topic_name
        return {
            "relationship_type": "DIRECT_FOLLOW_UP",
            "is_follow_up": True,
            "topic": topic_name,
            "fused_question": fused,
            "delta_focus": f"Direct follow-up in the context of {topic_name}.",
            "strategy_key": "follow_up"
        }

    # Default: New Question
    _old_topic = session.active_topic if session else ""
    tech_matches = re.findall(r"\b(django|python|react|docker|kubernetes|aws|postgresql|postgres|mongodb|sql|nosql|graphql|rest|microservices|redis|kafka|system design|authentication|caching|jwt|indexing|sharding|replication|polymorphism|database|api)\b", u_lower)
    new_topic = tech_matches[0].capitalize() if tech_matches else _old_topic

    result = {
        "relationship_type": "NEW_TOPIC",
        "is_follow_up": False,
        "topic": new_topic,
        "fused_question": utterance,
        "delta_focus": "",
        "strategy_key": "general"
    }
    if tech_matches and session:
        session.active_topic = new_topic
    return result


def analyze_conversation_flow(utterance: str, session: Optional["InterviewSession"] = None) -> Dict[str, Any]:
    """
    Tracks and analyzes interviewer conversation dynamics using the Answer Intelligence Layer:
    - QUESTION_CORRECTION ("Actually let me rephrase that...")
    - FOLLOW_UP ("Okay, and what about scalability?")
    - TOPIC_CHANGE ("Let's move on to...")
    - CLARIFICATION
    - MULTI_PART_QUESTION
    - NEW_QUESTION
    """
    if not utterance:
        return {"flow_type": "NEW_QUESTION", "fused_question": utterance, "should_cancel_stale": False}

    u_lower = utterance.lower().strip()
    active_topic = getattr(session, "active_topic", "") if session else ""

    # 1. QUESTION_CORRECTION ("Actually let me rephrase that...", "Wait, what I meant was...")
    correction_patterns = [
        r"^(actually,?\s*)?(let me (rephrase|re-ask|restate|change)|wait,?\s*(what i meant|let me|actually)|scratch that|no wait)",
        r"\b(sorry i meant|let me ask that differently|what i actually wanted to ask)\b"
    ]
    for p in correction_patterns:
        if re.search(p, u_lower):
            cleaned_q = re.sub(p, "", utterance, flags=re.IGNORECASE).strip(" ,:;-")
            return {
                "flow_type": "QUESTION_CORRECTION",
                "fused_question": cleaned_q if cleaned_q else utterance,
                "should_cancel_stale": True,
                "topic": active_topic
            }

    # 2. Answer Intelligence Relationship Classifier
    rel = classify_answer_relationship(utterance, session=session)
    if rel["relationship_type"] == "NEW_TOPIC" and rel.get("delta_focus"):
        return {
            "flow_type": "TOPIC_CHANGE",
            "fused_question": utterance,
            "should_cancel_stale": False,
            "topic": rel["topic"],
            "relationship": rel
        }
    elif rel["is_follow_up"]:
        return {
            "flow_type": "FOLLOW_UP",
            "fused_question": rel["fused_question"],
            "should_cancel_stale": False,
            "topic": rel["topic"],
            "relationship": rel
        }

    # 3. MULTI_PART_QUESTION
    if (" and " in u_lower or " also " in u_lower) and u_lower.count("?") >= 2:
        return {
            "flow_type": "MULTI_PART",
            "fused_question": utterance,
            "should_cancel_stale": False,
            "topic": active_topic,
            "relationship": rel
        }

    return {
        "flow_type": "NEW_QUESTION",
        "fused_question": utterance,
        "should_cancel_stale": False,
        "topic": session.active_topic if session else "",
        "relationship": rel
    }



# ============================================================

def get_normalized_semantic_key(text: str) -> str:
    """
    Extracts the core semantic intent tokens from a question for stability,
    material change detection, and deduplication comparison.
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    stop_words = {
        "the", "a", "an", "for", "this", "our", "my", "your", "in", "with",
        "please", "can", "you", "tell", "me", "about", "did", "do", "would",
        "could", "is", "are", "was", "were", "to", "of", "at", "by", "on"
    }
    tokens = [w for w in t.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


def evaluate_question_completeness(
    text: str,
    session: Optional["InterviewSession"] = None
) -> Dict[str, Any]:
    """
    Parakit Early-Answer Intelligence:
    Evaluates whether an evolving speech utterance is semantically complete
    and contains sufficient predicate/object information to start answer generation safely.
    """
    if not text or not text.strip():
        return {
            "completeness": 0.0,
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": "",
            "reason": "empty"
        }

    raw = text.strip()
    u_lower = raw.lower().strip(" .!?,:;")
    words = u_lower.split()
    word_count = len(words)
    has_qmark = raw.endswith("?")
    sem_key = get_normalized_semantic_key(raw)
    
    # 1. Non-question statements (e.g. "I live in St. Louis.", "Sounds good.", "Thanks.")
    statement_starters = [
        "i live", "i am", "i'm", "my name", "sounds good", "thanks", "thank you", "okay got it",
        "got it", "i see", "yes", "yeah", "yep", "sure", "no problem", "hello", "hi", "good morning"
    ]
    if any(u_lower.startswith(s) for s in statement_starters) and not has_qmark:
        return {
            "completeness": 1.0,
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": sem_key,
            "reason": "statement"
        }

    # 2. Check for Incomplete Trailing Stems (dangling verbs/prepositions that change meaning if completed)
    incomplete_trailing_words = {
        "choose", "use", "implement", "between", "about", "in", "for", "with",
        "at", "to", "of", "and", "vs", "versus", "from", "on", "into", "the", "a", "an", "your", "my",
        "like", "such", "as", "than", "or", "did", "do", "how", "why", "what", "can", "could", "would",
        "while", "whether", "because", "so", "but", "where", "when", "which", "depending", "assuming", "given"
    }
    
    last_word = words[-1] if words else ""
    is_trailing_incomplete = (last_word in incomplete_trailing_words) and not has_qmark

    # Early rejection if trailing conjunction or dangling stem (prevents premature answer trigger < 1%)
    if is_trailing_incomplete and word_count >= 2:
        return {
            "completeness": 0.35,
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": sem_key,
            "reason": f"incomplete_trailing_{last_word}"
        }

    # 3. Follow-up evaluation (Grounded in existing active topic or recent QA history)
    active_topic = getattr(session, "active_topic", "") if session else ""
    has_recent_qa = bool(session and getattr(session, "qa_pairs", []))
    is_grounded_context = bool(active_topic or has_recent_qa)

    if u_lower in {"why?", "why", "and why?", "and why", "how?", "how", "and how?", "and how", "why so"}:
        return {
            "completeness": 1.0 if is_grounded_context else 0.70,
            "is_semantically_complete": is_grounded_context,
            "can_start_early": is_grounded_context,
            "semantic_key": sem_key or "why",
            "reason": "short_followup_grounded" if is_grounded_context else "unresolved_followup"
        }

    followup_stems = ["what about", "how about", "and for", "why not", "then what", "so what", "and in that case", "what of", "and"]
    if any(u_lower.startswith(f) for f in followup_stems):
        if word_count <= 2 and is_trailing_incomplete:
            return {
                "completeness": 0.40,
                "is_semantically_complete": False,
                "can_start_early": False,
                "semantic_key": sem_key,
                "reason": "incomplete_followup_stem"
            }
        elif is_grounded_context or word_count >= 3 or has_qmark:
            return {
                "completeness": 0.95,
                "is_semantically_complete": True,
                "can_start_early": True,
                "semantic_key": sem_key,
                "reason": "complete_followup"
            }

    # 4. Coding Imperatives
    coding_imperatives = [
        "write a", "write an", "implement a", "implement an", "find the", "find all", "create a",
        "reverse a", "reverse the", "solve the", "calculate", "design a", "design the", "optimize",
        "build a", "build an", "construct a", "construct an"
    ]
    if any(u_lower.startswith(c) for c in coding_imperatives):
        if word_count <= 3 or is_trailing_incomplete:
            return {
                "completeness": 0.50,
                "is_semantically_complete": False,
                "can_start_early": False,
                "semantic_key": sem_key,
                "reason": "incomplete_coding_stem"
            }
        else:
            return {
                "completeness": 0.95,
                "is_semantically_complete": True,
                "can_start_early": True,
                "semantic_key": sem_key,
                "reason": "complete_coding_question"
            }

    # 5. General Interrogative Technical Questions
    interrogatives = [
        "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
        "can you", "could you", "would you", "will you", "do you", "did you", "have you", "has",
        "is it", "is there", "are you", "are there", "should we", "tell me", "explain", "describe",
        "walk me through", "elaborate on", "clarify", "discuss", "compare", "give an example"
    ]
    if any(u_lower.startswith(q) for q in interrogatives):
        if word_count <= 3 or is_trailing_incomplete:
            return {
                "completeness": 0.45,
                "is_semantically_complete": False,
                "can_start_early": False,
                "semantic_key": sem_key,
                "reason": "incomplete_interrogative_stem"
            }
        else:
            return {
                "completeness": 0.95 if word_count >= 5 or has_qmark else 0.85,
                "is_semantically_complete": True,
                "can_start_early": True,
                "semantic_key": sem_key,
                "reason": "complete_interrogative_question"
            }

    if has_qmark and word_count >= 3:
        return {
            "completeness": 0.90,
            "is_semantically_complete": True,
            "can_start_early": True,
            "semantic_key": sem_key,
            "reason": "punctuated_question"
        }

    return {
        "completeness": 0.50,
        "is_semantically_complete": False,
        "can_start_early": False,
        "semantic_key": sem_key,
        "reason": "insufficient_signal"
    }


def compute_question_likelihood(
    text: str,
    session: Optional["InterviewSession"] = None
) -> Dict[str, Any]:
    """
    Evaluates progressive question likelihood, semantic completeness,
    and pre-assembles speculative context for early answer generation.
    """
    if not text or not text.strip():
        return {
            "likelihood": 0.0,
            "stability_state": QuestionStabilityState.PARTIAL,
            "is_question_stem": False,
            "stem_type": "EMPTY",
            "is_complete_enough": False,
            "preview_topic": "",
            "completeness": 0.0,
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": "",
            "predicted_type": "general"
        }

    comp_eval = evaluate_question_completeness(text, session=session)
    raw = text.strip()
    u_lower = raw.lower().strip(" .!?,:;")
    words = u_lower.split()
    word_count = len(words)
    has_qmark = raw.endswith("?")

    interrogatives = [
        "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
        "can you", "could you", "would you", "will you", "do you", "did you", "have you", "has",
        "is it", "is there", "are you", "are there", "should we", "tell me", "explain", "describe",
        "walk me through", "elaborate on", "clarify"
    ]
    coding_imperatives = [
        "write a", "write an", "implement a", "implement an", "find the", "find all", "create a",
        "reverse a", "reverse the", "solve the", "calculate", "design a", "design the", "optimize"
    ]
    followup_stems = [
        "what about", "how about", "and for", "why not", "then what", "so what", "and in that case",
        "what if"
    ]

    is_interrogative = any(u_lower.startswith(q) for q in interrogatives)
    is_coding = any(u_lower.startswith(c) for c in coding_imperatives)
    is_followup = any(u_lower.startswith(f) for f in followup_stems)

    # Statements
    if comp_eval["reason"] == "statement":
        return {
            "likelihood": 0.10,
            "stability_state": QuestionStabilityState.PARTIAL,
            "is_question_stem": False,
            "stem_type": "STATEMENT",
            "is_complete_enough": word_count >= 3,
            "preview_topic": "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": False,
            "can_start_early": False,
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": "statement"
        }

    from classifier import classify_question
    pred_type = classify_question(raw)

    if is_followup:
        if comp_eval["can_start_early"]:
            state = QuestionStabilityState.STABLE_QUESTION
            likelihood = 0.95
        elif word_count <= 2:
            state = QuestionStabilityState.POSSIBLE_QUESTION
            likelihood = 0.70
        else:
            state = QuestionStabilityState.LIKELY_QUESTION
            likelihood = 0.85
            
        return {
            "likelihood": likelihood,
            "stability_state": state,
            "is_question_stem": True,
            "stem_type": "FOLLOW_UP",
            "is_complete_enough": comp_eval["is_semantically_complete"],
            "preview_topic": getattr(session, "active_topic", "") if session else "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": "follow_up"
        }

    if is_coding:
        if comp_eval["can_start_early"]:
            state = QuestionStabilityState.STABLE_QUESTION
            likelihood = 0.95
        else:
            state = QuestionStabilityState.LIKELY_QUESTION
            likelihood = 0.80
            
        return {
            "likelihood": likelihood,
            "stability_state": state,
            "is_question_stem": True,
            "stem_type": "CODING_IMPERATIVE",
            "is_complete_enough": comp_eval["is_semantically_complete"],
            "preview_topic": "Coding",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": "coding"
        }

    if is_interrogative:
        if comp_eval["can_start_early"]:
            state = QuestionStabilityState.STABLE_QUESTION
            likelihood = 0.95
        elif word_count <= 2 and not has_qmark:
            state = QuestionStabilityState.POSSIBLE_QUESTION
            likelihood = 0.65
        else:
            state = QuestionStabilityState.LIKELY_QUESTION
            likelihood = 0.82
            
        return {
            "likelihood": likelihood,
            "stability_state": state,
            "is_question_stem": True,
            "stem_type": "INTERROGATIVE",
            "is_complete_enough": comp_eval["is_semantically_complete"],
            "preview_topic": getattr(session, "active_topic", "") if session else "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": pred_type
        }

    if has_qmark:
        return {
            "likelihood": 0.90,
            "stability_state": QuestionStabilityState.STABLE_QUESTION if word_count >= 3 else QuestionStabilityState.POSSIBLE_QUESTION,
            "is_question_stem": True,
            "stem_type": "PUNCTUATED_QUESTION",
            "is_complete_enough": True,
            "preview_topic": getattr(session, "active_topic", "") if session else "",
            "completeness": comp_eval["completeness"],
            "is_semantically_complete": comp_eval["is_semantically_complete"],
            "can_start_early": comp_eval["can_start_early"],
            "semantic_key": comp_eval["semantic_key"],
            "predicted_type": pred_type
        }

    return {
        "likelihood": 0.25 if word_count <= 3 else 0.40,
        "stability_state": QuestionStabilityState.PARTIAL,
        "is_question_stem": False,
        "stem_type": "GENERAL",
        "is_complete_enough": word_count >= 4,
        "preview_topic": "",
        "completeness": comp_eval["completeness"],
        "is_semantically_complete": False,
        "can_start_early": False,
        "semantic_key": comp_eval["semantic_key"],
        "predicted_type": "general"
    }


# ============================================================
# STT PROVIDER ABSTRACTION
# ============================================================

from abc import ABC, abstractmethod

