"""
Local Offline Fallback Engine (Feature 12)
===========================================
Provides resilient multi-tier AI inference failover for live interviews:
Tier 1: High-Speed Groq Cloud API (Default)
Tier 2: Local Ollama LLM Service (Zero internet / Rate limit failover)
Tier 3: Local Heuristic Resume RAG Synthesizer (Deterministic offline fallback)

Ensures candidates never experience dead air or app crashes if WiFi drops mid-interview.
"""

import os
import json
import time
import urllib.request
from typing import List, Optional, Callable, Tuple

OFFLINE_CONFIG_FILE = os.getenv("BLUJAY_PROVIDER_PATH", os.path.expanduser("~/.blujay_provider.json"))
LEGACY_OFFLINE_CONFIG_FILE = os.path.expanduser("~/.ai_copilot_provider.json")
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def check_ollama_available(host: str = DEFAULT_OLLAMA_HOST, timeout_sec: float = 0.6) -> Tuple[bool, List[str]]:
    """
    Checks if a local Ollama server is running and returns list of installed models.
    """
    url = f"{host.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Interview-Copilot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return True, models
    except Exception:
        pass
    return False, []


def stream_ollama_response(
    prompt: str,
    system_prompt: str = "",
    model: str = "llama3.2",
    host: str = DEFAULT_OLLAMA_HOST,
    on_token: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    timeout_sec: float = 10.0
) -> str:
    """
    Streams response tokens from local Ollama HTTP API endpoint.
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": 0.5,
            "num_predict": 400
        }
    }
    
    full_text = ""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "AI-Interview-Copilot/1.0"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            for line in resp:
                if is_cancelled and is_cancelled():
                    break
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    token = data.get("response", "")
                    if token:
                        full_text += token
                        if on_token:
                            on_token(token)
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Notice: Local Ollama stream error: {e}")
        
    return full_text


def stream_gemini_response(
    prompt: str,
    system_prompt: str = "",
    on_token: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None
) -> str:
    """
    Streams response from Google Gemini 3.6 Flash fallback provider.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return ""
    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            contents=full_prompt
        )
        if resp and resp.text:
            ans = resp.text
            if on_token:
                on_token(ans)
            return ans
    except Exception as ge:
        print(f"Notice: Gemini stream error: {ge}")
    return ""


def generate_offline_heuristic_answer(
    question: str,
    job_title: str = "Software Engineer",
    company_name: str = "",
    resume_text: str = "",
    on_token: Optional[Callable[[str], None]] = None
) -> str:
    """
    Deterministic offline answer generator grounded on resume and question keywords
    when neither cloud nor local Ollama LLM is reachable.
    """
    q_lower = question.lower()
    company_clause = f" at {company_name}" if company_name else ""
    matched_context = ""
    if resume_text:
        lines = [l.strip() for l in resume_text.splitlines() if len(l.strip()) > 20]
        relevant_line = ""
        for line in lines:
            ll = line.lower()
            if any(k in ll for k in ["built", "developed", "architected", "implemented", "engineered", "designed", "created", "stack", "python", "fastapi", "react", "sql"]):
                relevant_line = line
                break
        if not relevant_line and lines:
            relevant_line = lines[0]
        if relevant_line:
            matched_context = f"\nRelevant Background: {relevant_line}"
    
    # 1. System Design question
    if any(k in q_lower for k in ["design", "architecture", "scale", "throughput", "sharding", "microservices"]):
        ans = f"""[Offline Synthesis - System Design]
For designing this system as a {job_title}{company_clause}:
1. Requirements & Scale: Clarify functional requirements, read/write ratios (e.g. 10:1 read-heavy), and SLA latency targets.
2. High-Level Architecture: Client -> DNS/CDN -> API Gateway / Load Balancer -> Stateless Application Services.
3. Data Layer & Caching: Primary relational/NoSQL DB with read replicas, backed by Redis for hot queries and Kafka for async event processing.
4. Scalability & Resilience: Horizontal auto-scaling, database sharding by user partition hash, and multi-region failover.{matched_context}
"""
    # 2. Behavioral STAR question
    elif any(k in q_lower for k in ["tell me about a time", "conflict", "challenge", "failure", "leadership", "proud"]):
        if resume_text:
            ans = f"""[Offline Synthesis - Behavioral STAR]
• Situation: In my verified background as a {job_title}{company_clause}, our team handled core platform delivery requirements.
• Task: I owned the implementation and ensured alignment with project goals.
• Action: I broke the initiative into phased deliverables, introduced automated testing, and proactively conducted code reviews.{matched_context}
• Result: Delivered the feature reliably and maintained high code quality.
"""
        else:
            ans = f"""[Offline Synthesis - Behavioral STAR Framework]
• Situation: Structure your response around a real project challenge from your engineering background.
• Task: Clearly articulate your direct responsibility and technical ownership.
• Action: Detail the specific technical decisions, debugging steps, and collaboration you performed.
• Result: State the concrete, measurable outcome and lessons learned without unconfirmed claims.
"""
    # 3. Technical / General Engineering question
    else:
        ans = f"""[Offline Synthesis - Direct Technical Answer]
In approaching '{question}' for the {job_title} role{company_clause}:
1. Core Concept: Focus on the fundamental trade-offs between computational complexity, maintainability, and reliability.
2. Best Practices: Leverage standard design patterns, strict type-safety, robust error handling, and end-to-end integration tests.
3. Architecture: Prioritize observable metrics (latency, error rates) and clean modular structure.{matched_context}
"""

    ans = ans.strip()
    if on_token:
        # Simulate gentle token streaming
        words = ans.split(" ")
        for i, w in enumerate(words):
            token = w + (" " if i < len(words) - 1 else "")
            on_token(token)
            time.sleep(0.01)

    return ans


class OfflineFallbackManager:
    """
    Coordinates multi-tier AI inference failover between Cloud, Local Ollama, and Heuristic.
    """

    def __init__(self, ollama_host: str = DEFAULT_OLLAMA_HOST):
        self.ollama_host = ollama_host
        self.preferred_provider = self._load_preference()
        self._cached_ollama_status: Optional[Tuple[bool, List[str]]] = None
        self._last_ollama_check_time = 0.0

    def get_preferred_provider(self) -> str:
        return self.preferred_provider

    def set_preferred_provider(self, provider: str):
        if provider in ["auto", "cloud", "ollama", "heuristic"]:
            self.preferred_provider = provider
            self._save_preference()

    def get_ollama_status(self, force_refresh: bool = False) -> Tuple[bool, List[str]]:
        now = time.time()
        if not force_refresh and self._cached_ollama_status is not None and (now - self._last_ollama_check_time < 3.0):
            return self._cached_ollama_status
        status = check_ollama_available(self.ollama_host)
        self._cached_ollama_status = status
        self._last_ollama_check_time = now
        return status

    def _load_preference(self) -> str:
        for p in (OFFLINE_CONFIG_FILE, LEGACY_OFFLINE_CONFIG_FILE):
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("preferred_provider", "auto")
                except Exception:
                    pass
        return "auto"

    def _save_preference(self):
        try:
            with open(OFFLINE_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump({"preferred_provider": self.preferred_provider}, f, indent=2)
        except Exception:
            pass
