"""
BluJay AI - Prompt Builder & Knowledge Grounding
================================================
Constructs high-precision system prompts conditioned on candidate resume,
job description, RAG retrieved knowledge, question type, and Parakeet follow-up fusion.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from engine.session_models import InterviewSession

def determine_knowledge_source(
    question: str,
    session: Optional[InterviewSession] = None,
    recent_context: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Determines whether the incoming question requires:
    1. CANDIDATE_EXPERIENCE (Resume, work history, past achievements)
    2. CANDIDATE_PROJECT (Specific implementation, design decisions, challenges faced)
    3. FOLLOW_UP_CONVERSATION (Follow-up referring to previous answer)
    4. GENERAL_TECHNICAL (Core language/system concept, algorithm, standard definition)
    5. HYBRID_COMBINATION (Technical concept tied to candidate's verified experience)
    """
    if not question:
        return "GENERAL_TECHNICAL"
        
    lower = question.lower()
    
    # 1. Check for follow-up questions
    followup_starters = ["what about", "how about", "and for", "why did you do that", "why not", "then what happened", "why?"]
    if any(lower.startswith(f) for f in followup_starters) or (session and session.qa_pairs and len(lower.split()) <= 4 and lower.endswith("?")):
        return "FOLLOW_UP_CONVERSATION"
        
    # 2. Check for candidate project / implementation inquiries
    project_patterns = [
        r"\b(your project|in your project|in your application|the project you built)\b",
        r"\b(why did you choose|why did you pick|what database did you use)\b",
        r"\b(what problems did you face|what challenges did you face|how did you solve)\b",
        r"\b(how did you implement|how did you handle|how you implemented)\b",
        r"\b(your architecture|your stack|your database|your pipeline)\b"
    ]
    for p in project_patterns:
        if re.search(p, lower):
            return "CANDIDATE_PROJECT"
            
    # 3. Check for candidate resume / background inquiries
    resume_patterns = [
        r"\b(walk me through your resume|tell me about yourself|your background|your experience)\b",
        r"\b(your role at|your time at|in your previous role|in your last company)\b",
        r"\b(your greatest achievement|biggest accomplishment|your responsibilities)\b",
        r"\b(why should we hire you|your strengths|your leadership)\b"
    ]
    for p in resume_patterns:
        if re.search(p, lower):
            return "CANDIDATE_EXPERIENCE"
            
    # 4. Pure general technical / algorithm definitions
    general_tech_patterns = [
        r"^what is (the difference between|polymorphism|a deadlock|an index|acid|cap theorem|rest|jwt|cors|grpc)",
        r"^explain (how indexing works|tcp|udp|garbage collection|event loop|b-tree)",
        r"^write (a function|an algorithm|a query|a regex|a binary search)"
    ]
    for p in general_tech_patterns:
        if re.search(p, lower):
            return "GENERAL_TECHNICAL"

    # 5. Check for completely unrelated domain questions (e.g. quantum physics, specialized hardware)
    unrelated_patterns = [
        r"\b(quantum|qubit|decoherence|error correction|quantum gate|particle physics|superconductivity)\b",
        r"\b(asic design|fpga verilog|vhdl|pcb layout|rf circuit|vlsi)\b"
    ]
    for p in unrelated_patterns:
        if re.search(p, lower):
            return "UNRELATED_TECHNICAL"

    # 6. Check for out-of-context technical questions (e.g. DevOps/Infrastructure asked to backend/frontend devs)
    out_of_context_tech_patterns = [
        r"\b(kubernetes|k8s|cni|pod networking|service mesh|istio|terraform|ansible|helm|ci/cd|jenkins|argocd|prometheus|grafana)\b",
        r"\b(kernel module|ebpf|device driver|bare metal|bgp routing|ospf)\b"
    ]
    for p in out_of_context_tech_patterns:
        if re.search(p, lower):
            resume_text = getattr(session, "resume_text", "") or ""
            if not re.search(p, resume_text.lower()):
                return "OUT_OF_CONTEXT_TECHNICAL"
            
    return "HYBRID_COMBINATION"




def build_interview_system_prompt(
    session: InterviewSession,
    question_type: str = "general",
    knowledge_source: str = "HYBRID_COMBINATION",
    relationship_info: Optional[Dict[str, Any]] = None,
    question: Optional[str] = None
) -> str:
    """Builds a rich, personalized system prompt aligned with BluJay AI standards, strict grounding policies, dynamic RAG retrieval, and conversation intelligence."""
    base_profile = session.get_base_profile_block()
    
    # Dynamic Semantic Knowledge Retrieval (RAG)
    rag_retrieval_section = ""
    effective_rag_query = ""
    if relationship_info and relationship_info.get("is_follow_up") and relationship_info.get("fused_question"):
        effective_rag_query = relationship_info["fused_question"].strip()
    elif question:
        effective_rag_query = question.strip()
    elif getattr(session, "latest_detected_question", ""):
        effective_rag_query = getattr(session, "latest_detected_question", "").strip()

    if effective_rag_query:
        try:
            import rag
            rag_output = rag.search_knowledge(effective_rag_query, top_k=5)
            if rag_output and rag_output.strip():
                rag_retrieval_section = f"""
============================================================
SEMANTIC RETRIEVED KNOWLEDGE (DYNAMIC RAG CONTEXT)
============================================================
{rag_output.strip()}
- RAG GROUNDING MANDATE: Ground technical specifics, system metrics, and architectural trade-offs using the semantically retrieved chunks above.
"""
        except Exception:
            pass

    has_custom_resume = bool(session.resume_text and session.resume_text.strip())
    
    if has_custom_resume:
        grounding_rules = """
3. PERSONALIZATION & GROUNDING POLICIES (CUSTOM RESUME PRIORITY #1):
   - HIGHEST PRIORITY SINGLE SOURCE OF TRUTH: The candidate has uploaded a custom resume. Derive all work experience, project names, previous employers, degrees, tools, and achievements EXCLUSIVELY from the uploaded resume above.
   - ZERO DEFAULT OVERRIDE: Do NOT invent or pull default projects not present in the uploaded resume.
   - ZERO FABRICATION & NO OVERSELLING: Never invent companies, roles, or degrees absent from the uploaded resume. Speak in natural first-person candidate voice ("I built...", "In my role at...").
   - OUT-OF-SCOPE HONESTY MANDATE: If asked about an unconfirmed project, employer, or degree, explicitly state that your direct experience is in your confirmed areas, but provide the technical approach or how you would solve it.
   - DO NOT DUMP KEYWORDS: Choose the top 3–4 most relevant skills.
   - AVOID CORPORATE BUZZWORDS: Speak like a real candidate. Say "I built...", "I worked with...", "I focused on...".
   - For GENERAL TECHNICAL questions: Provide the precise, industry-standard definition and practical trade-offs; then weave in verified experience from the uploaded resume only if relevant.
"""
    else:
        grounding_rules = """
3. PERSONALIZATION & GROUNDING POLICIES (DEFAULT KNOWLEDGE BASE):
   - For questions about YOUR EXPERIENCE or PROJECTS: Base your answer strictly on the verified facts in the candidate's default Knowledge Base above.
   - ZERO FABRICATION & NO OVERSELLING: Never invent companies you didn't work for, degrees you don't have, or tools absent from your background.
   - OUT-OF-SCOPE HONESTY MANDATE: If asked about an unconfirmed project, employer, or degree, explicitly state that your direct experience is in your confirmed areas, but provide the technical approach or how you would solve it.
   - DO NOT DUMP KEYWORDS: Choose the top 3–4 most relevant skills.
   - AVOID CORPORATE BUZZWORDS: Speak like a real candidate. Say "I built...", "I worked with...", "I focused on...".
   - For GENERAL TECHNICAL questions: Provide the precise, industry-standard definition and practical trade-offs; then weave in verified experience only if relevant.
"""
    if knowledge_source in ["CANDIDATE_EXPERIENCE", "CANDIDATE_PROJECT"]:
        grounding_rules += "\n   - CRITICAL FOCUS: This question asks directly about the candidate's personal project/experience. Focus heavily on real project architecture, design trade-offs, and verified tech stack."
    elif knowledge_source == "FOLLOW_UP_CONVERSATION" or (relationship_info and relationship_info.get("is_follow_up")):
        grounding_rules += "\n   - CRITICAL FOCUS: This is a contextual follow-up to the preceding question/answer. Directly build upon the technology and architecture established in the previous turn."
        
    relationship_block = ""
    if relationship_info and relationship_info.get("is_follow_up"):
        rel_type = relationship_info.get("relationship_type", "FOLLOW_UP")
        topic = relationship_info.get("topic", getattr(session, "active_topic", "Current Project"))
        delta_focus = relationship_info.get("delta_focus", "")
        relationship_block = f"""
============================================================
ANSWER INTELLIGENCE & CONVERSATION DELTA
============================================================
- Relationship: {rel_type}
- Active Discussion Topic: {topic}
- Delta Focus Directive: {delta_focus}
- ANTI-REPETITION MANDATE: Do NOT repeat the entire previous answer or restate basic concepts. Answer only the delta requested by this follow-up question.
"""

    from classifier import get_answer_strategy
    specific_strategy = get_answer_strategy(question_type)
    
    prompt = f"""
You are BluJay AI — the world's most elite, hyper-intelligent Real-Time Interview Copilot and executive interview coach.
You are whispering directly into the candidate's ear during a live, high-stakes interview.

{base_profile}
{rag_retrieval_section}
{relationship_block}
============================================================
SESSION CONFIGURATION & STRATEGY
============================================================
- Selected Tone / Delivery: {session.tone}
- Preferred Spoken Language: {session.language}
- Extra User Instructions: {session.custom_instructions if session.custom_instructions else "None"}
- Detected Question Category: {question_type.upper()}
- Targeted Knowledge Source: {knowledge_source}

============================================================
INTELLIGENT OUT-OF-CONTEXT INTERVIEW QUESTION STRATEGY
============================================================
PRIMARY OBJECTIVE:
When the interviewer asks a question that is outside the candidate's primary role, skills, or confirmed experience:
1. DO NOT refuse to answer and DO NOT pretend the candidate has experience that is not confirmed.
2. Answer the interviewer's actual question accurately using general technical knowledge.
3. Determine whether there is a genuine and useful connection between the question and the candidate's role, skills, projects, or confirmed experience.
4. If a genuine connection exists, naturally bridge the explanation to the candidate's perspective.
5. If no meaningful connection exists, simply provide the correct technical answer without forcing a bridge.
6. Never convert general technical knowledge into a claim about the candidate's personal experience.

1. INTERNAL CLASSIFICATION (DO NOT EXPOSE):
Determine whether the question is:
- Directly related to the candidate's role
- Related to the candidate's skills
- Related to a confirmed project
- Related to a technology the candidate knows
- Outside the candidate's primary role but technically relevant (e.g., DevOps/SRE/Infrastructure asked to a Python developer)
- Completely unrelated to the candidate's known background (e.g., quantum error correction, specialized ASIC hardware)
- HR / behavioral
- Coding
- Follow-up

2. OUT-OF-CONTEXT TECHNICAL QUESTIONS:
If the interviewer asks a technical question outside the candidate's primary role, still answer the question directly.
Example: Candidate is a Python Developer, and the interviewer asks "How do Kubernetes pods communicate across nodes?"
- The answer must first explain Kubernetes networking accurately (Pod networking, Pod IP addresses, CNI plugins, Node routing, cluster networking).
- Use general technical knowledge.
- Do NOT say "I configured Kubernetes networking" unless this experience is explicitly confirmed.

3. INTELLIGENT BRIDGE (NATURAL & OPTIONAL):
Is there a natural and technically meaningful connection to the candidate's role, skills, or confirmed experience?
- IF YES: Create a short natural bridge.
  Example: "From a Python backend perspective, the important part is that the application generally relies on the platform's networking layer for service-to-service communication, while the application itself focuses on handling requests and business logic."
- IF NO: Do NOT force a bridge. Simply finish the technical explanation cleanly. (e.g. For quantum error correction or specialized hardware with no meaningful relationship to Python development, answer the technical question directly and stop).

4. CRITICAL RULE — NEVER INVENT EXPERIENCE:
General technical knowledge must NEVER become candidate experience.
If asked "How would you design a Kubernetes deployment?":
- Do NOT automatically say "I designed Kubernetes deployments", "I managed Kubernetes clusters", or "I configured CNI networking".
- Instead, use hypothetical/perspective language: "From a Python developer's perspective, I would focus on making the application container-ready and ensuring that configuration, health checks, logging, and tests are suitable for deployment."
- The phrase "I would..." describes an approach or perspective. It does NOT claim that the candidate has already performed that work.

5. TECHNOLOGY KNOWLEDGE != IMPLEMENTATION EXPERIENCE:
Presence of a technology in candidate's background (e.g. Python, Docker, AWS) means familiarity, NOT an automatic claim to having implemented every associated technology (Kubernetes, Terraform, AWS EKS, Lambda, ECS, Celery, FAISS, Pinecone). Only use those as general technical explanations unless confirmed.

6. THREE-LAYER ANSWER STRATEGY (DECISION FRAMEWORK):
- Layer 1 — Direct Technical Explanation: Answer what was asked (what it is, how it works, important components, why it is used, practical example).
- Layer 2 — Natural Role Connection: Only if a genuine connection exists ("From a Python developer's perspective...", "At the application level...", "The part I would be most concerned with as a backend developer is...").
- Layer 3 — Truth Validation: Check every first-person statement. If confirmed: "I used...", "I implemented...". If unconfirmed: "I would...", "From a developer's perspective...", or explain generally.

7. DO NOT FORCE THE THREE-PART STRUCTURE:
Use the bridge only when it improves the answer. The interviewer should hear a natural human response, not a predefined template.

8. EXAMPLES:
- CI/CD: "A CI/CD pipeline automates the process from code changes through testing and deployment. In Continuous Integration, automated checks run on every commit. Once passed, deployment packages the Docker image and promotes it to target environments. From a Python developer's perspective, I would focus on making the application reliable for that pipeline by maintaining clean dependencies, automated tests, environment-specific configuration, and appropriate health checks." ("I would focus..." does NOT falsely claim "I built the company's CI/CD infrastructure".)
- KUBERNETES: "Kubernetes networking is designed so that pods can communicate across nodes using pod IP addresses via a CNI plugin... From a Python backend perspective, the important point is that the application focuses on serving requests and communicating with services while Kubernetes provides the underlying infrastructure connectivity."

9. FOLLOW-UP QUESTIONS ("How did you implement that?"):
If the previous answer discussed a general concept that is unconfirmed in candidate's past work, say:
"I haven't personally implemented that based on the experience I have available, but the approach would generally be..." then explain the approach if appropriate.

10. "WHY" QUESTIONS ("Why did you choose Kubernetes?"):
If unconfirmed: "If I were choosing a platform for this type of application, I would consider Kubernetes because..."

11. CODING QUESTIONS ON UNFAMILIAR TECH:
Provide correct code, explanation, and complexity without claiming previous production usage of unfamiliar tools.

12. ANSWER NATURALNESS & ZERO META-LEAKAGE:
Never say "According to the Intelligent Bridge Strategy", "Based on retrieved context", "Although this is outside my domain", or "Based on candidate information".

============================================================
DEDICATED ANSWER GENERATION STRATEGY
============================================================
{specific_strategy}

============================================================
CRITICAL RULES & LENGTH BUDGETS (BLUJAY AI DIRECT SPOKEN DELIVERY)
============================================================
1. SPEAK IN FIRST PERSON ("I", "my", "our team") as the candidate talking naturally to the interviewer.
2. ZERO REASONING LEAKAGE & ZERO META-HEADERS:
   - NEVER output <think> tags, chain-of-thought, or internal reasoning notes.
   - NEVER output synthetic meta-labels or headers like "ANSWER:", "APPROACH:", "WHY:", "KEY POINT:", "SITUATION:", "TASK:", "ACTION:", "RESULT:".
   - Start IMMEDIATELY with the actual spoken words you say to the interviewer.
3. ZERO ROBOTIC FLUFF: Never say "Certainly!", "Sure thing!", "Here is an answer:", "As an AI...", or "According to my resume". Dive straight into the natural spoken response.
{grounding_rules}
4. BLUJAY AI TELEPROMPTER FORMATTING (MAXIMUM GLANCEABILITY):
   - Always structure answers so the candidate can read them seamlessly while looking at the camera.
   - Start with a clear, confident 1–2 sentence spoken direct answer/hook.
   - Use concise, high-impact bullet points with **bold lead-ins** (`• **Core Step / Architecture**: ...`, `• **Trade-off / Choice**: ...`).
   - AVOID solid dense walls of paragraph text.
5. QUESTION-SPECIFIC DEPTH:
   - Technical Concepts: Crisp spoken hook, 2–4 bulleted mechanisms with bold lead-ins, and 1-sentence wrap-up (~120–220 words).
   - Project & Architecture: Project overview hook, 3–4 bulleted architecture/implementation steps, and measurable impact (~150–260 words).
   - Behavioral: Natural narrative hook, bulleted actions taken with bold lead-ins, and clear quantifiable result (~150–250 words).
   - Coding / Algorithm: MUST ALWAYS write the complete, optimal, bug-free implementation enclosed in a markdown ```python code block (or requested language). NEVER explain the algorithm without providing the complete code block. Follow the code with `• **Complexity**: Time: O(...) | Space: O(...)` and `• **Edge Cases**: ...`.
   - 30s Short Mode: Direct, punchy 2–3 sentence reply under 50 words.
6. TARGET LANGUAGE: Deliver the answer in {session.language}.
"""
    return prompt.strip()


# ============================================================
# GROUNDING VALIDATION (BLUJAY ANTI-FABRICATION GATEWAY)
# ============================================================

TECH_TOOL_WHITELIST = {
    "aws", "amazon web services", "gcp", "google cloud", "google cloud platform",
    "azure", "microsoft azure", "docker", "kubernetes", "k8s", "stripe", "mongodb",
    "postgresql", "postgres", "redis", "kafka", "rabbitmq", "react", "next.js", "nextjs",
    "vue", "angular", "node.js", "nodejs", "node", "python", "java", "go", "golang",
    "rust", "c++", "c#", ".net", "dotnet", "linux", "unix", "ubuntu", "debian", "redhat",
    "centos", "github", "gitlab", "bitbucket", "jira", "confluence", "slack", "discord",
    "zoom", "teams", "cloud", "platform", "database", "server", "api", "apis", "sdk", "sdks",
    "engine", "framework", "service", "services", "microservice", "microservices",
    "elasticsearch", "opensearch", "openai", "groq", "anthropic", "claude", "gemini",
    "llama", "whisper", "vercel", "supabase", "firebase", "dynamodb", "cassandra",
    "mysql", "sqlite", "graphql", "rest", "grpc", "protobuf", "nginx", "apache",
    "terraform", "ansible", "helm", "prometheus", "grafana", "datadog", "splunk",
    "new relic", "sentry", "celery", "airflow", "spark", "hadoop", "flink",
    "snowflake", "bigquery", "redshift", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "fastapi", "flask", "django", "spring", "express", "webrtc",
    "websocket", "websockets", "jwt", "oauth", "iam", "ci/cd", "cicd", "git", "sqs", "sns"
}


def validate_candidate_grounding(
    answer: str,
    resume_text: str = "",
    knowledge_text: str = "",
    candidate_profile: str = ""
) -> Tuple[str, List[str]]:
    """
    Evaluates generated answer against verified candidate ground truth.
    Deterministic, sub-millisecond execution (< 0.3ms) with zero external network overhead.
    
    1. Extracts claimed employers and degree institutions from first-person phrases.
    2. Whitelists technical tools, services, protocols, platforms, and verified entities.
    3. Auto-sanitizes unverified employer or degree claims into natural, grammar-preserving
       candidate phrasing (e.g. "when I was at Netflix" -> "in my previous engineering experience").
    4. Returns (sanitized_answer, warnings).
    """
    if not answer or not answer.strip():
        return answer, []

    combined_verified = f"{resume_text}\n{knowledge_text}\n{candidate_profile}".lower()
    warnings_list: List[str] = []
    sanitized = answer

    def is_entity_allowed(ent: str) -> bool:
        clean_ent = ent.strip().strip(".,;:!?\"'()[]{}")
        low = clean_ent.lower()
        if not low or len(low) < 2:
            return True
        if low in TECH_TOOL_WHITELIST:
            return True
        parts = low.split()
        if any(p in TECH_TOOL_WHITELIST for p in parts if len(p) > 2):
            return True
        if low in combined_verified:
            return True
        return False

    # 1. Employer patterns:
    def _sanitize_when_worked_at(m):
        ent = m.group(1).strip()
        if is_entity_allowed(ent):
            return m.group(0)
        warnings_list.append(f"Unverified employer claim '{ent}' sanitized to generic phrasing.")
        return "in my previous engineering experience"

    sanitized = re.sub(
        r'(?i)\bwhen\s+I\s+(?:was|worked)\s+at\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?)',
        _sanitize_when_worked_at,
        sanitized
    )

    def _sanitize_in_role_at(m):
        ent = m.group(1).strip()
        if is_entity_allowed(ent):
            return m.group(0)
        warnings_list.append(f"Unverified employer claim '{ent}' sanitized to generic phrasing.")
        return "in my previous role"

    sanitized = re.sub(
        r'(?i)\bin\s+my\s+(?:role|position|job)\s+at\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?)',
        _sanitize_in_role_at,
        sanitized
    )

    def _sanitize_at_company_i(m):
        ent = m.group(1).strip()
        verb = m.group(2)
        if is_entity_allowed(ent):
            return m.group(0)
        warnings_list.append(f"Unverified employer claim '{ent}' sanitized to generic phrasing.")
        return f"In my previous role, I {verb}"

    sanitized = re.sub(
        r'(?i)\bat\s+([A-Z][A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?),\s+I\s+([a-z]+)',
        _sanitize_at_company_i,
        sanitized
    )

    def _sanitize_my_team_at(m):
        ent = m.group(1).strip()
        if is_entity_allowed(ent):
            return m.group(0)
        warnings_list.append(f"Unverified employer claim '{ent}' sanitized to generic phrasing.")
        return "my previous engineering team"

    sanitized = re.sub(
        r'(?i)\bmy\s+team\s+at\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?)',
        _sanitize_my_team_at,
        sanitized
    )

    def _sanitize_while_at(m):
        ent = m.group(1).strip()
        if is_entity_allowed(ent):
            return m.group(0)
        warnings_list.append(f"Unverified employer claim '{ent}' sanitized to generic phrasing.")
        return "in my previous experience"

    sanitized = re.sub(
        r'(?i)\bwhile\s+at\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?)',
        _sanitize_while_at,
        sanitized
    )

    # 2. Degree / Education patterns:
    def _sanitize_graduated_from(m):
        ent = m.group(1).strip()
        if is_entity_allowed(ent):
            return m.group(0)
        warnings_list.append(f"Unverified degree claim '{ent}' sanitized to generic phrasing.")
        return "through my academic training"

    sanitized = re.sub(
        r'(?i)\b(?:graduated\s+from|studied\s+at)\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,3})',
        _sanitize_graduated_from,
        sanitized
    )

    def _sanitize_my_degree_at(m):
        ent = m.group(1).strip()
        if is_entity_allowed(ent):
            return m.group(0)
        warnings_list.append(f"Unverified degree claim '{ent}' sanitized to generic phrasing.")
        return "my degree"

    sanitized = re.sub(
        r'(?i)\bmy\s+(?:degree|bachelor[\'’]?s?|master[\'’]?s?|phd|doctorate)\s+(?:at|from)\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,3})',
        _sanitize_my_degree_at,
        sanitized
    )

    return sanitized, warnings_list


