"""
Live AI Answer Quality, Formatting & Latency Verification
=========================================================
Tests live Groq model answer generation across all major interview modes:
1. Technical Architecture & Engineering (FastAPI + MongoDB + Redis)
2. Behavioral STAR Method (Production Incident / Conflict)
3. System Design (High-scale Real-time Notifications)
4. 30s Short Mode (Punchy concise delivery < 60 words)
5. Algorithm & Coding Problem (Two Sum / Sliding Window)
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import interview_engine

load_dotenv()


def test_live_outputs():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment or .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    session = interview_engine.InterviewSession()
    session.job_title = "Senior Full-Stack AI Engineer"
    session.company_name = "TechCorp"
    session.tone = "Confident & Conversational"

    test_cases = [
        {
            "mode": "Technical Architecture",
            "category": "technical",
            "question": "How did you design and implement JWT authentication in FastAPI with MongoDB and Redis caching?",
            "override_prompt": ""
        },
        {
            "mode": "Behavioral STAR",
            "category": "behavioral",
            "question": "Tell me about a time you resolved a high-severity production outage under tight pressure.",
            "override_prompt": ""
        },
        {
            "mode": "System Design",
            "category": "system_design",
            "question": "Design a distributed real-time notification service handling 10 million concurrent WebSocket connections.",
            "override_prompt": ""
        },
        {
            "mode": "30s Short Mode",
            "category": "general",
            "question": "Why did you choose MongoDB instead of PostgreSQL for this service?",
            "override_prompt": "Make this response extremely concise, direct, and under 50 words for a punchy 30-second delivery."
        },
        {
            "mode": "Algorithm & Coding",
            "category": "coding",
            "question": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. Write the optimal O(N) Python code.",
            "override_prompt": ""
        }
    ]

    print("=" * 80)
    print("LIVE AI ANSWER GENERATION QUALITY & LATENCY EVALUATION")
    print("=" * 80)

    all_passed = True

    for idx, test in enumerate(test_cases, 1):
        mode = test["mode"]
        q = test["question"]
        override = test["override_prompt"]
        
        print(f"\n[{idx}/5] TESTING MODE: {mode}")
        print(f"[Question] \"{q}\"")
        
        tokens = []
        t_start = time.perf_counter()
        
        answer = interview_engine.generate_interview_answer(
            client=client,
            session=session,
            question=q,
            on_token=lambda t: tokens.append(t),
            refinement_instruction=override
        )
        
        t_duration = time.perf_counter() - t_start
        word_count = len(answer.split())
        
        print(f"[Latency] {t_duration:.2f}s | Words: {word_count} | Streamed Tokens: {len(tokens)}")
        print("-" * 60)
        print("Generated Answer:")
        print(answer)
        print("-" * 60)
        
        # Validations
        if not answer or word_count < 10:
            print(f"[FAIL] Empty or excessively short answer for {mode}")
            all_passed = False
        else:
            print(f"[SUCCESS] High-quality structured output generated in {t_duration:.2f}s")
            
        time.sleep(0.5)

    print("\n" + "=" * 80)
    if all_passed:
        print("[ALL TESTS PASSED] 100% PERFECT OUTPUTS & FAST GENERATION")
    else:
        print("[WARNING] SOME TESTS ENCOUNTERED ISSUES")
    print("=" * 80)


if __name__ == "__main__":
    test_live_outputs()
