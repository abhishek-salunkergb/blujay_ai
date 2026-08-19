"""
Job Description (JD) Matcher & Fit Analysis Engine (Feature 5)
==============================================================
Analyzes Job Descriptions against Candidate Resumes using hybrid semantic analysis & LLM reasoning.
Calculates skill match percentage, matched strengths, skill gaps, bridging talking points,
and predicted technical & behavioral interview questions tailored specifically to the role.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional
from groq import Groq

JD_ANALYSIS_CACHE_FILE = os.path.expanduser("~/.ai_copilot_jd_analysis.json")


def extract_keywords_heuristic(text: str) -> List[str]:
    """Extracts core programming languages, frameworks, and technical keywords."""
    COMMON_TECH_TERMS = {
        "python", "javascript", "typescript", "java", "c++", "c#", "golang", "go", "rust",
        "react", "next.js", "vue", "angular", "node.js", "express", "fastapi", "django", "flask",
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform", "ci/cd", "git",
        "graphql", "rest", "grpc", "microservices", "system design", "distributed systems",
        "kafka", "rabbitmq", "webrtc", "llm", "rag", "pytorch", "tensorflow", "agile", "scrum"
    }
    found = set()
    cleaned = re.sub(r"[^\w\s\+\#\/\.]", " ", text.lower())
    words = cleaned.split()
    for w in words:
        clean_w = w.strip(".,;:!?()")
        if clean_w in COMMON_TECH_TERMS:
            found.add(clean_w)
    for term in ["system design", "distributed systems", "ci/cd", "next.js", "node.js"]:
        if term in text.lower():
            found.add(term)
    return sorted(list(found))


def heuristic_jd_match(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """Fast deterministic heuristic match between JD and Resume without external API."""
    jd_keywords = extract_keywords_heuristic(jd_text)
    resume_keywords = extract_keywords_heuristic(resume_text)
    
    if not jd_keywords:
        # Fallback if no specific tech keywords identified
        return {
            "match_score": 75,
            "matched_skills": resume_keywords[:6],
            "missing_skills": ["Domain specific tooling", "System scale specifics"],
            "bridging_talking_points": [
                "Highlight deep problem-solving velocity and architectural fundamentals.",
                "Emphasize strong cross-functional communication and fast technology adoption."
            ],
            "predicted_questions": [
                "Can you walk us through the most technically challenging system you have designed?",
                "How do you handle production outages or performance bottlenecks under scale?",
                "Describe a time you had to learn a new framework or domain quickly."
            ]
        }
        
    matched = [k for k in jd_keywords if k in resume_keywords]
    missing = [k for k in jd_keywords if k not in resume_keywords]
    
    score = int((len(matched) / max(1, len(jd_keywords))) * 100)
    score = min(98, max(35, score))
    
    bridging = []
    if missing:
        for m in missing[:3]:
            bridging.append(f"When asked about {m.upper()}, emphasize related foundational experience and fast ramp-up capability.")
    else:
        bridging.append("Your skills align closely with the core requirements—emphasize depth of impact and measurable metrics.")
        
    predicted = [
        f"How have you utilized {matched[0].upper() if matched else 'core technologies'} in high-concurrency production environments?",
        "Tell me about a complex trade-off you made when designing a distributed service.",
        f"How would you approach scaling or optimizing a service built with {jd_keywords[0].upper() if jd_keywords else 'modern frameworks'}?",
        "Describe a time you resolved a difficult cross-team technical disagreement.",
        "What is your philosophy on writing maintainable, well-tested code?"
    ]

    return {
        "match_score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "bridging_talking_points": bridging,
        "predicted_questions": predicted
    }


def analyze_jd_resume_fit(
    jd_text: str,
    resume_text: str,
    client: Optional[Groq] = None,
    job_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes JD and Resume fit using Groq LLM reasoning with fast heuristic fallback.
    Returns structured match score, matched skills, gaps, bridging points, and predicted questions.
    """
    if not jd_text.strip():
        return {
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "bridging_talking_points": ["Please paste the job description text to compute fit analysis."],
            "predicted_questions": []
        }
        
    if not client:
        return heuristic_jd_match(jd_text, resume_text)
        
    prompt = f"""
You are a Principal Technical Recruiter and Senior Engineering Hiring Manager at FAANG.
Analyze the fit between this Job Description (JD) and Candidate Resume.

TARGET ROLE: {job_title or "Software Engineering"}

=== JOB DESCRIPTION ===
{jd_text[:4000]}

=== CANDIDATE RESUME ===
{resume_text[:4000] if resume_text else "No uploaded resume provided."}

Return a valid JSON object strictly matching this schema:
{{
  "match_score": <integer 0-100 representing realistic candidate qualification percentage>,
  "matched_skills": [<list of strings of confirmed overlapping skills and requirements>],
  "missing_skills": [<list of strings of JD requirements not evident in resume>],
  "bridging_talking_points": [<list of 2-4 actionable sentences advising how candidate can bridge missing skills during the interview using their existing experience>],
  "predicted_questions": [<list of 4-5 high-probability technical and behavioral interview questions tailored specifically to this role and company requirements>]
}}
Only return valid JSON. Do not include markdown code fence formatting or explanations.
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900
        )
        content = completion.choices[0].message.content.strip()
        # Strip potential markdown formatting if returned
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        
        # Validate keys
        if "match_score" in data and "matched_skills" in data:
            return data
    except Exception as e:
        print(f"Notice during LLM JD analysis: {e}")
        
    return heuristic_jd_match(jd_text, resume_text)


def save_jd_analysis_cache(analysis: Dict[str, Any]):
    """Saves latest JD analysis result to local cache."""
    try:
        with open(JD_ANALYSIS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
    except Exception as e:
        print(f"Notice saving JD analysis cache: {e}")


def get_cached_jd_analysis() -> Optional[Dict[str, Any]]:
    """Loads cached JD analysis if available."""
    if os.path.exists(JD_ANALYSIS_CACHE_FILE):
        try:
            with open(JD_ANALYSIS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None
