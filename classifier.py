import re
from typing import Dict, Any


# ============================================================
# 13 QUESTION TYPE CLASSIFICATION RULES
# ============================================================

def classify_question(question: str) -> str:
    """
    Intelligently classifies an interview question into one of 13 specific types:
    - clarification
    - follow_up
    - coding
    - system_design
    - database
    - cloud
    - ai_ml
    - resume
    - project
    - behavioral
    - hr
    - technical
    - general
    """
    if not question:
        return "general"

    q = question.lower().strip()

    # 1. CLARIFICATION
    clarification_patterns = [
        r"^(did you mean|are you asking|can you clarify|to confirm|just to clarify|are you referring to)",
        r"\b(what do you mean by|could you rephrase|can you repeat)\b"
    ]
    for pattern in clarification_patterns:
        if re.search(pattern, q):
            return "clarification"

    # 2. FOLLOW-UP & CODING SPECIFIC INQUIRIES
    if any(kw in q for kw in ["what's the time complexity", "what's the space complexity", "what's the complexity", "what is the complexity", "what is the time complexity", "what is the space complexity", "what is the big o", "what's the big o", "time complexity", "space complexity", "complexity of this", "tell me the complexity"]):
        return "coding_complexity"

    if any(kw in q for kw in ["can you optimize it", "can we optimize this", "can you optimize this", "can we do better", "optimize the solution", "more optimal approach", "make it faster"]):
        return "coding_optimization"

    follow_up_starters = [
        "what about", "how about", "and for", "why not", "then what",
        "can you explain that further", "tell me more about that", "can you elaborate", "why?", "why redux?", "why react?", "why jwt?"
    ]
    if any(q.startswith(starter) for starter in follow_up_starters) or (q in {"why?", "how?", "redux?", "scalability?", "caching?", "why not?"}):
        return "follow_up"

    # 3. PROJECT / PRACTICAL IMPLEMENTATION (Prioritize candidate project inquiries)
    project_keywords = [
        "your project", "in your project", "the project you built", "farmer marketplace",
        "project architecture", "project features", "why did you choose this stack",
        "what challenges did you face in your project", "how did you implement in your project"
    ]
    if any(kw in q for kw in project_keywords) or ("project" in q and "your" in q):
        return "project"

    # 4. DEBUGGING & CODE REVIEW
    debugging_keywords = [
        "what's wrong with this", "what is wrong with this", "why is this failing",
        "fix this code", "debug this", "find the bug", "what does this code output",
        "what does this output", "what is the output of", "traceback", "syntaxerror",
        "typeerror", "why is this error", "compilation error", "failed test"
    ]
    if any(kw in q for kw in debugging_keywords):
        return "debugging"

    # 5. SQL & RELATIONAL QUERIES
    sql_keywords = [
        "write a sql query", "sql query to", "write sql to", "second highest salary",
        "nth highest salary", "select query", "join query", "group by query",
        "window function in sql", "dense_rank", "sql query", "write a query"
    ]
    if any(kw in q for kw in sql_keywords):
        return "sql"

    # 6. CODING / ALGORITHMS / DSA
    coding_keywords = [
        "write a program", "write code", "code for", "function to",
        "program to", "solve this", "algorithm", "time complexity", "space complexity",
        "reverse a", "binary search", "two sum", "linked list", "leetcode", "data structure",
        "second largest element", "longest substring", "valid parentheses", "merge intervals",
        "given an array", "given a string", "given a list", "array of integers", "subarray",
        "target sum", "maximum subarray", "indices of the two numbers", "add up to target"
    ]
    if any(kw in q for kw in coding_keywords):
        return "coding"

    # 7. SYSTEM DESIGN & ARCHITECTURE
    system_design_component_patterns = [
        r"\b(how would you shard|sharding strategy for|cache invalidation for|how do you handle backpressure)\b",
        r"\b(how would you secure the api|caching strategy for this|load balancing for this|monitoring for this)\b",
        r"\b(failover for this system|how does the queue handle|database replication for this)\b"
    ]
    for pattern in system_design_component_patterns:
        if re.search(pattern, q):
            return "system_design_component"

    system_design_keywords = [
        "system design", "design a", "design the", "architecture of", "architecture",
        "how would you design", "scale to", "high throughput", "rate limiter",
        "load balancer", "microservices architecture", "event driven", "message broker"
    ]
    if any(kw in q for kw in system_design_keywords):
        return "system_design"

    # 8. DATABASE / STORAGE
    database_keywords = [
        "database", "sql vs nosql", "indexing", "acid", "sharding",
        "replication", "query optimization", "mongodb", "postgresql", "mysql",
        "redis", "cassandra", "orm", "database transaction", "schema design"
    ]
    if any(kw in q for kw in database_keywords):
        return "database"

    # 7. CLOUD & DEVOPS
    cloud_keywords = [
        "aws", "gcp", "azure", "kubernetes", "k8s", "docker", "terraform",
        "ci/cd", "serverless", "lambda", "ec2", "s3", "containerization",
        "devops", "cloud deployment", "infrastructure as code"
    ]
    if any(kw in q for kw in cloud_keywords):
        return "cloud"

    # 8. AI / MACHINE LEARNING
    ai_ml_keywords = [
        "machine learning", "deep learning", "neural network", "transformer",
        "rag", "llm", "fine-tuning", "pytorch", "tensorflow", "embeddings",
        "vector database", "model evaluation", "prompt engineering", "langchain",
        "loss function", "optimizer", "ml model", "nlp", "computer vision"
    ]
    if any(kw in q for kw in ai_ml_keywords):
        return "ai_ml"

    # 9. RESUME / BACKGROUND
    resume_keywords = [
        "walk me through your resume", "tell me about your background",
        "your journey", "your career", "your previous role", "at your last company",
        "why are you leaving", "summarize your experience"
    ]
    if any(kw in q for kw in resume_keywords):
        return "resume"

    # 10. BEHAVIORAL
    behavioral_keywords = [
        "tell me about a time", "conflict", "disagreed", "difficult situation",
        "under pressure", "tight deadline", "failure", "mistake you made",
        "handled feedback", "difficult stakeholder", "proudest accomplishment"
    ]
    if any(kw in q for kw in behavioral_keywords):
        return "behavioral"

    # 11. HR / CULTURE
    hr_keywords = [
        "tell me about yourself", "why should we hire you", "why do you want to work here",
        "where do you see yourself in 5 years", "your greatest strength",
        "your weakness", "salary expectations", "company culture", "work environment"
    ]
    if any(kw in q for kw in hr_keywords):
        return "hr"

    # 12. TECHNICAL CONCEPTS
    technical_keywords = [
        "what is", "how does", "difference between", "explain",
        "polymorphism", "inheritance", "rest api", "graphql", "jwt",
        "concurrency", "multithreading", "garbage collection", "tcp vs udp"
    ]
    if any(kw in q for kw in technical_keywords):
        return "technical"

    # 13. GENERAL
    return "general"


# ============================================================
# STRATEGY SELECTOR FOR EACH QUESTION TYPE
# ============================================================

STRATEGY_MAP: Dict[str, str] = {
    "technical": (
        "TECHNICAL STRATEGY:\n"
        "Format with clean, scannable uppercase section titles:\n"
        "ANSWER\n"
        "[Crisp 1-2 sentence core direct response/definition in natural first-person spoken tone]\n\n"
        "APPROACH\n"
        "• [Key step or architectural mechanic 1]\n"
        "• [Key step or architectural mechanic 2]\n"
        "• [Key step or architectural mechanic 3]\n\n"
        "WHY\n"
        "• [Primary benefit or architectural trade-off]\n"
        "• [Why this choice over alternatives]\n\n"
        "KEY POINT\n"
        "[Memorable 1-sentence takeaway or project metric]"
    ),
    "coding": (
        "CODING INTERVIEW STRATEGY:\n"
        "Format with scannable uppercase section titles:\n"
        "APPROACH\n"
        "[Core intuition, algorithm logic, data structure choice, and trade-offs]\n\n"
        "CODE\n"
        "```python\n"
        "[Clean, idiomatic, fully working code with helpful comments]\n"
        "```\n\n"
        "COMPLEXITY\n"
        "Time: O(...) with brief derivation\n"
        "Space: O(...) auxiliary memory\n\n"
        "EDGE CASES\n"
        "• Empty input / null checks\n"
        "• Duplicate values or negative numbers\n"
        "• Single element / boundary constraints"
    ),
    "coding_complexity": (
        "CODING COMPLEXITY ONLY STRATEGY:\n"
        "The interviewer is asking exclusively about the Time and Space Complexity of the preceding code solution.\n"
        "- DO NOT regenerate code or full algorithm from scratch.\n"
        "COMPLEXITY\n"
        "Time: O(...) - [1 sentence mathematical breakdown]\n"
        "Space: O(...) - [Auxiliary memory breakdown]\n\n"
        "KEY POINT\n"
        "[Bottleneck operation or memory trade-off]"
    ),
    "coding_optimization": (
        "CODING OPTIMIZATION STRATEGY:\n"
        "The interviewer asked to optimize the previous solution.\n"
        "APPROACH\n"
        "[Identify previous bottleneck (e.g. O(N^2) brute force) and present the optimized approach (e.g. O(N) Hash Map / Two Pointers)]\n\n"
        "CODE\n"
        "```python\n"
        "[Clean, optimized production code]\n"
        "```\n\n"
        "COMPLEXITY\n"
        "Time: O(...) (improved from previous version)\n"
        "Space: O(...)\n\n"
        "TRADE-OFFS\n"
        "[Time-space trade-offs of the optimized approach]"
    ),
    "debugging": (
        "DEBUGGING & CODE REVIEW STRATEGY:\n"
        "Format with clear scannable sections:\n"
        "PROBLEM\n"
        "[State exact bug/error and line location]\n\n"
        "CAUSE\n"
        "[Explain the root cause / logical flaw / edge case failure]\n\n"
        "FIX\n"
        "```python\n"
        "[Clean, corrected code]\n"
        "```\n\n"
        "WHY\n"
        "[Why this fix resolves the issue and prevents regressions]"
    ),
    "sql": (
        "SQL QUERY STRATEGY:\n"
        "Format with scannable sections:\n"
        "SQL\n"
        "```sql\n"
        "[Optimized SQL query with Window Functions, CTEs, or JOINs]\n"
        "```\n\n"
        "EXPLANATION\n"
        "• [Step 1: clause evaluation]\n"
        "• [Step 2: window partition / filtering logic]\n\n"
        "COMPLEXITY / PERFORMANCE\n"
        "[Recommended indexes (e.g. B-Tree on target column) and execution considerations]"
    ),
    "behavioral": (
        "BEHAVIORAL (STAR) STRATEGY:\n"
        "Format strictly in conversational STAR format:\n"
        "SITUATION\n"
        "[Brief 1-2 sentence context, project, and stake]\n\n"
        "TASK\n"
        "[What specific engineering/leadership responsibility fell on me]\n\n"
        "ACTION\n"
        "• [Specific action 1: architectural or technical decision taken]\n"
        "• [Specific action 2: implementation / collaboration]\n\n"
        "RESULT\n"
        "[Quantifiable outcome, metric improvement, or positive team impact]"
    ),
    "project": (
        "PROJECT STRATEGY:\n"
        "Grounded strictly in verified candidate facts:\n"
        "ANSWER\n"
        "• Project: [Verified Project Name]\n"
        "• Stack: [Verified Technologies]\n"
        "• Purpose: [Core objective]\n"
        "• Key Feature: [Main feature implemented]\n"
        "• My Role: [Responsibility]\n\n"
        "HOW IT WORKED\n"
        "1. [Workflow step 1]\n"
        "2. [Workflow step 2]\n"
        "3. [Workflow step 3]\n\n"
        "RESULT\n"
        "[Quantifiable metric or verified impact]"
    ),
    "system_design": (
        "SYSTEM DESIGN INTERVIEW STRATEGY:\n"
        "REQUIREMENTS\n"
        "• Functional: [Core APIs and user workflows]\n"
        "• Non-Functional: [High availability, low latency <50ms, data consistency]\n\n"
        "ARCHITECTURE\n"
        "Client -> CDN -> Load Balancer -> API Gateway -> Microservices -> DB / Cache\n\n"
        "COMPONENTS\n"
        "• Ingestion / API Layer: [Stateless services]\n"
        "• Storage & Cache: [SQL/NoSQL database + Redis caching]\n"
        "• Asynchronous Pipeline: [Kafka / RabbitMQ event stream]\n\n"
        "SCALABILITY\n"
        "• Horizontal auto-scaling, database read-replicas, and table sharding\n\n"
        "TRADE-OFFS\n"
        "• [CAP trade-offs, consistency vs availability]"
    ),
    "system_design_component": (
        "SYSTEM DESIGN COMPONENT DEEP-DIVE STRATEGY:\n"
        "ANSWER\n"
        "[Direct explanation of the component implementation within active architecture]\n\n"
        "APPROACH\n"
        "• [Implementation detail 1]\n"
        "• [Implementation detail 2]\n\n"
        "TRADE-OFFS\n"
        "[Failure modes, backpressure, or cache eviction trade-offs]"
    ),
    "database": (
        "DATABASE STRATEGY:\n"
        "ANSWER\n"
        "[Direct schema/storage architecture answer]\n\n"
        "APPROACH\n"
        "• [Indexing strategy or schema normalization/denormalization]\n"
        "• [Query optimization technique]\n\n"
        "WHY\n"
        "[Consistency vs latency and ACID compliance trade-offs]\n\n"
        "KEY POINT\n"
        "[Production performance takeaway]"
    ),
    "cloud": (
        "CLOUD & DEVOPS STRATEGY:\n"
        "ANSWER\n"
        "[Cloud architecture and infrastructure deployment answer]\n\n"
        "APPROACH\n"
        "• [Containerization / Kubernetes orchestration]\n"
        "• [CI/CD automated pipeline]\n\n"
        "KEY POINT\n"
        "[High availability and infrastructure reliability]"
    ),
    "ai_ml": (
        "AI / ML STRATEGY:\n"
        "ANSWER\n"
        "[Model architecture and machine learning intuition]\n\n"
        "APPROACH\n"
        "• [Data processing & training/fine-tuning approach]\n"
        "• [Inference latency and deployment pipeline]\n\n"
        "KEY POINT\n"
        "[Evaluation metrics and production reliability]"
    ),
    "resume": (
        "RESUME / INTRODUCTORY STRATEGY:\n"
        "Deliver a natural, spoken 3–5 sentence intro (~60–100 words) in first-person.\n"
        "1. Current role / education background.\n"
        "2. Key technical skills (mention top 3–4 relevant technologies only, do not dump full resume).\n"
        "3. Core project or practical experience.\n"
        "4. What role you are looking forward to.\n"
        "Speak conversationally as a real person. Do NOT output headings, planning notes, or bullet points."
    ),
    "hr": (
        "HR & BEHAVIORAL STRATEGY:\n"
        "Deliver a natural, spoken response (~60–100 words) demonstrating genuine alignment, cultural fit, and motivation.\n"
        "Speak directly and authentically in first person. Do NOT output headings or meta-commentary."
    ),
    "follow_up": (
        "FOLLOW-UP DELTA STRATEGY:\n"
        "Direct, concise spoken answer (~30–70 words) focused strictly on the delta/specific query without repeating previous full explanations.\n"
        "Do NOT repeat the entire question or output rigid section headers."
    ),
    "follow_up_why": (
        "FOLLOW-UP (WHY) STRATEGY:\n"
        "Direct architectural or technical justification (~40–80 words) for this specific choice in the discussed project.\n"
        "Highlight the primary practical benefit (performance, simplicity, or reliability) naturally."
    ),
    "follow_up_how": (
        "FOLLOW-UP (HOW) STRATEGY:\n"
        "Crisp, step-by-step spoken explanation (~40–90 words) of implementation mechanics in the discussed project."
    ),
    "follow_up_scalability": (
        "SCALABILITY & PERFORMANCE STRATEGY:\n"
        "Direct spoken explanation (~50–100 words) of how this component scales (caching, sharding, replicas) without unnecessary buzzwords."
    ),
    "follow_up_security": (
        "SECURITY CONSIDERATIONS STRATEGY:\n"
        "Direct spoken explanation (~40–80 words) of security defenses (JWT, encryption, token handling) applied to this component."
    ),
    "follow_up_comparison": (
        "TECHNOLOGY COMPARISON STRATEGY:\n"
        "Crisp spoken comparison (~50–90 words) explaining why the chosen technology fit the requirements over alternatives."
    ),
    "follow_up_expansion": (
        "DETAILED EXPANSION STRATEGY:\n"
        "Deeper technical explanation (~50–100 words) expanding the specific point with practical details."
    ),
    "follow_up_example": (
        "CONCRETE EXAMPLE STRATEGY:\n"
        "Realistic spoken scenario (~50–90 words) from the discussed project illustrating the concept in practice."
    ),
    "follow_up_tradeoff": (
        "TRADE-OFFS & DRAWBACKS STRATEGY:\n"
        "Honest engineering trade-offs (~40–80 words) and how your team mitigated them in practice."
    ),
    "follow_up_improvement": (
        "IMPROVEMENTS & FUTURE CHANGES STRATEGY:\n"
        "Specific improvement (~40–80 words) you would make based on production learnings."
    ),
    "clarification": (
        "CLARIFICATION STRATEGY:\n"
        "Crisp 1–2 sentence clarification (~20–40 words) confirming scope and assumptions directly."
    ),
    "general": (
        "GENERAL CONVERSATIONAL STRATEGY:\n"
        "Natural, spoken first-person response (~40–90 words). Direct, concise, and grounded in candidate facts without rigid headings."
    )
}


def get_answer_strategy(question_type: str) -> str:
    """Returns the dedicated answer generation strategy for the given question type."""
    return STRATEGY_MAP.get(question_type.lower(), STRATEGY_MAP["general"])


if __name__ == "__main__":
    test_q = [
        "What is polymorphism in Java?",
        "Write a Python program to reverse a linked list.",
        "Tell me about a time you handled a difficult conflict.",
        "Explain your Farmer Marketplace project.",
        "How would you design Twitter?",
        "Explain indexing in PostgreSQL.",
        "How do you deploy microservices on AWS Kubernetes?",
        "What loss function did you use for the ML model?",
        "Walk me through your resume.",
        "Why should we hire you?",
        "What about scalability?",
        "Can you clarify if you mean SQL or NoSQL?",
        "Thanks for joining today."
    ]

    for q in test_q:
        q_type = classify_question(q)
        print(f"[{q_type.upper():<14}] {q}")