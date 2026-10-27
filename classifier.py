import re
from typing import Dict


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
    # Coding Optimization (Checked before complexity to capture 'improve the time complexity')
    if any(kw in q for kw in [
        "can you optimize", "can we optimize", "can you optimize it", "can we optimize this",
        "can you optimize this", "can we do better", "optimize the solution", "optimize your solution",
        "more optimal approach", "make it faster", "optimize this", "optimize that", "optimize it",
        "make this faster", "make that faster", "optimize the approach", "optimize your approach",
        "optimize approach", "optimize solution", "optimize the time", "optimize the space",
        "optimize time complexity", "optimize space complexity",
        "can we improve the time complexity", "can we improve the space complexity",
        "improve time complexity", "improve the time complexity", "improve space complexity"
    ]):
        return "coding_optimization"

    # Coding Complexity Inquiries
    if any(kw in q for kw in [
        "what's the time complexity", "what's the space complexity", "what's the complexity",
        "what is the complexity", "what is the time complexity", "what is the space complexity",
        "what is the big o", "what's the big o", "time complexity", "space complexity",
        "complexity of this", "tell me the complexity", "big o complexity", "big o of this"
    ]):
        return "coding_complexity"

    # Pseudocode
    if any(kw in q for kw in [
        "give me pseudocode", "write pseudocode", "show me pseudocode", "pseudocode for this",
        "give pseudo code", "show pseudocode", "give pseudocode", "write pseudo code",
        "pseudo code", "pseudocode"
    ]):
        return "pseudocode"

    # Code Explanation
    if any(kw in q for kw in [
        "explain your approach", "explain the approach", "explain your code", "walk me through your code",
        "explain the solution", "explain your solution", "how does your code work", "walk me through the solution",
        "explain this code", "explain the code", "walk me through this code"
    ]):
        return "code_explanation"

    follow_up_starters = [
        "what about", "how about", "and for", "why not", "then what",
        "can you explain that further", "tell me more about that", "can you elaborate", "why?", "why redux?", "why react?", "why jwt?"
    ]
    if any(q.startswith(starter) for starter in follow_up_starters) or (q in {"why?", "how?", "redux?", "scalability?", "caching?", "why not?"}):
        return "follow_up"

    # 3. PROJECT / PRACTICAL IMPLEMENTATION (Prioritize candidate project inquiries)
    project_keywords = [
        "your project", "in your project", "the project you built", "e-commerce application",
        "marketplace platform", "project architecture", "project features", "why did you choose this stack",
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
        "write a program", "write code", "code for", "function to", "write a function",
        "program to", "solve this", "algorithm", "time complexity", "space complexity",
        "reverse a", "binary search", "two sum", "linked list", "leetcode", "data structure",
        "second largest element", "longest substring", "valid parentheses", "merge intervals",
        "given an array", "given a string", "given a list", "array of integers", "subarray",
        "target sum", "maximum subarray", "indices of the two numbers", "add up to target",
        "how to reverse", "how do you reverse", "how to implement", "implement a", "implement the",
        "write python", "write java", "write c++", "write javascript", "find the maximum",
        "find the minimum", "sort an array", "tree traversal", "dfs", "bfs", "dynamic programming",
        "sliding window", "two pointers", "stack", "queue", "binary tree", "graph", "hashmap", "heap", "recursion",
        # Natural imperative coding commands
        "give me the code", "give me code", "show me the code", "show the code",
        "implement this", "implement that", "implement it", "now code it", "now code this",
        "code this", "code that", "code it", "code it up", "write the solution", "code the solution",
        "write a solution", "write the code", "can you code this", "can you implement this", "solve the problem",
        "write it iteratively", "now write it iteratively", "write it recursively", "now write it", "write it"
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

    # 9. CLOUD / INFRASTRUCTURE
    cloud_keywords = [
        "docker", "kubernetes", "k8s", "aws", "gcp", "azure", "ci/cd",
        "terraform", "serverless", "lambda", "ecs", "eks", "helm"
    ]
    if any(kw in q for kw in cloud_keywords):
        return "cloud"

    # 10. AI / ML
    ai_ml_keywords = [
        "machine learning", "deep learning", "neural network", "llm", "rag",
        "transformer", "fine-tuning", "embedding", "vector database", "pytorch", "tensorflow"
    ]
    if any(kw in q for kw in ai_ml_keywords):
        return "ai_ml"

    # 11. RESUME / EXPERIENCE
    resume_keywords = [
        "tell me about yourself", "walk me through your resume", "your experience",
        "your background", "introduce yourself", "years of experience", "current role"
    ]
    if any(kw in q for kw in resume_keywords):
        return "resume"

    # 12. BEHAVIORAL
    behavioral_keywords = [
        "tell me about a time", "conflict with", "difficult situation", "failed project",
        "leadership", "disagreement", "under pressure", "tight deadline", "proudest achievement"
    ]
    if any(kw in q for kw in behavioral_keywords):
        return "behavioral"

    # 13. HR / LOGISTICS
    hr_keywords = [
        "salary expectation", "notice period", "why this company", "why leave",
        "relocation", "remote work", "joining date", "strengths and weaknesses"
    ]
    if any(kw in q for kw in hr_keywords):
        return "hr"

    return "technical"


# ============================================================
# STRATEGY SELECTOR FOR EACH QUESTION TYPE (BLUJAY AI DIRECT SPOKEN DELIVERY)
# ============================================================

STRATEGY_MAP: Dict[str, str] = {
    "technical": (
        "TECHNICAL STRATEGY:\n"
        "Deliver the exact, direct spoken answer with zero meta-headers (do NOT write 'ANSWER:', 'APPROACH:', 'WHY:', 'KEY POINT:').\n"
        "1. Start immediately with a clear, confident 1-2 sentence spoken direct answer.\n"
        "2. Follow with 2-4 scannable bullet points using bold lead-ins for key mechanisms and choices (`• **Key Step / Architecture**: detail`, `• **Trade-off / Why**: choice vs alternatives`).\n"
        "3. Conclude with a punchy 1-sentence wrap-up."
    ),
    "coding": (
        "CODING INTERVIEW STRATEGY (MANDATORY COMPLETE CODE BLOCK):\n"
        "1. Spoken Approach: Start with a brief 1-2 sentence spoken explanation of the algorithmic intuition.\n"
        "2. Complexity: Provide the exact Big-O on its own line: `• **Complexity**: Time: O(...) | Space: O(...)`.\n"
        "3. Complete Production Code: Provide the complete, optimal, runnable code in a single ```<language> block (e.g. ```python, ```javascript, ```java, ```cpp, ```sql). Never omit the code block or provide only pseudo-code.\n"
        "4. Candidate Spoken Talking Points & Edge Cases: 2-3 crisp bullet points with bold lead-ins for key mechanisms and edge cases."
    ),
    "coding_complexity": (
        "CODING COMPLEXITY ONLY STRATEGY:\n"
        "State the exact Time and Space complexity in 1-2 direct spoken sentences with Big-O notation (`• **Complexity**: Time: O(...) | Space: O(...)`) and the dominant operation."
    ),
    "coding_optimization": (
        "CODING OPTIMIZATION STRATEGY:\n"
        "1. State the optimization intuition in 1 sentence.\n"
        "2. Provide the improved Big-O: `• **Complexity**: Time: O(...) | Space: O(...)`.\n"
        "3. Provide the optimized ```<language> code block.\n"
        "4. State why this is faster/more memory efficient."
    ),
    "pseudocode": (
        "PSEUDOCODE STRATEGY:\n"
        "1. Spoken Intuition: 1 brief sentence summarizing the strategy.\n"
        "2. Complete Algorithmic Pseudocode Block: Provide clean, language-agnostic step-by-step pseudocode in a ```text block with meaningful variable names and clear logic.\n"
        "3. State Time & Space Complexity Big-O on its own line: `• **Complexity**: Time: O(...) | Space: O(...)`."
    ),
    "code_explanation": (
        "CODE EXPLANATION STRATEGY:\n"
        "Deliver a crisp, spoken first-person walkthrough of the code solution without repeating full raw code:\n"
        "1. High-Level Strategy: 1-2 spoken sentences explaining the core algorithmic approach and data structures chosen.\n"
        "2. Step-by-Step Logic: 2-3 clean bullet points explaining the primary loop/recursion, state transitions, and edge case handling.\n"
        "3. Complexity Justification: 1 sentence confirming why this approach achieves optimal Time and Space complexity."
    ),
    "debugging": (
        "DEBUGGING & CODE REVIEW STRATEGY:\n"
        "1. Directly state the bug and root cause in 1-2 sentences.\n"
        "2. Provide the corrected ```<language> code block.\n"
        "3. Briefly explain why this fix resolves the issue."
    ),
    "sql": (
        "SQL QUERY STRATEGY:\n"
        "1. Brief 1-sentence spoken explanation of the query approach.\n"
        "2. Provide the Big-O / Execution complexity if applicable.\n"
        "3. Clean, production-ready ```sql query.\n"
        "4. Key performance considerations or indexing recommendation (`• **Indexing**: ...`)."
    ),
    "behavioral": (
        "BEHAVIORAL (STAR) STRATEGY:\n"
        "Deliver the direct spoken story in first-person without writing 'SITUATION:', 'TASK:', 'ACTION:', 'RESULT:'.\n"
        "Structure as a natural narrative:\n"
        "• **Context & Challenge**: [1-2 sentences setting the real scenario and stake]\n"
        "• **My Actions**: [2-3 concrete technical/leadership actions taken]\n"
        "• **Outcome & Impact**: [Quantifiable metric improvement and final result]"
    ),
    "project": (
        "PROJECT STRATEGY:\n"
        "Grounded strictly in verified candidate facts:\n"
        "1. Direct 1-2 sentence spoken overview of the project and core objective.\n"
        "2. 3-4 bullet points on the architecture, technical challenges solved, and key features (`• **Architecture / Stack**: ...`, `• **Key Implementation**: ...`).\n"
        "3. Measurable impact or production metric."
    ),
    "system_design": (
        "SYSTEM DESIGN INTERVIEW STRATEGY:\n"
        "1. Direct 1-2 sentence high-level architecture hook.\n"
        "2. Core components in clean bullets (`• **API / Ingestion Layer**: ...`, `• **Storage & Caching**: ...`, `• **Event Streaming**: ...`).\n"
        "3. Scalability, high-availability, and trade-offs in 2-3 concise bullets."
    ),
    "system_design_component": (
        "SYSTEM DESIGN COMPONENT DEEP-DIVE STRATEGY:\n"
        "Direct spoken explanation of how that specific component operates, failure modes, and trade-offs in 3-4 clean bullet points."
    ),
    "database": (
        "DATABASE STRATEGY:\n"
        "Direct spoken response comparing schemas, indexing, ACID vs eventual consistency, and practical production trade-offs in 3-4 clean bullet points."
    ),
    "cloud": (
        "CLOUD & DEVOPS STRATEGY:\n"
        "Direct spoken explanation of the cloud architecture, CI/CD pipeline, container orchestration, and reliability in 3-4 clean bullet points."
    ),
    "ai_ml": (
        "AI / ML STRATEGY:\n"
        "Direct spoken response explaining model architecture, training/fine-tuning pipelines, inference latency, and evaluation in 3-4 clean bullet points."
    ),
    "resume": (
        "RESUME / INTRODUCTORY STRATEGY:\n"
        "Deliver a natural, spoken 3–5 sentence intro in first-person (~60–100 words) covering current role, top 3 technical strengths, key project highlight, and enthusiasm."
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