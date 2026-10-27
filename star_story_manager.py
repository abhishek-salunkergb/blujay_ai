"""
Behavioral STAR Story Bank & Answer Matrix Engine (Feature 14)
==============================================================
Extracts, indexes, and structures candidate experiences into the STAR Method
(Situation -> Task -> Action -> Result -> Key Learnings) for FAANG Leadership Principles
and behavioral interview rounds.
"""

import os
import re
import json
from typing import Dict, Any, List
from groq import Groq

STAR_STORIES_STORAGE_FILE = os.path.expanduser("~/.ai_copilot_star_stories.json")

COMMON_BEHAVIORAL_CATEGORIES = [
    "Technical Challenge / Incident Outage",
    "Leadership & Ownership",
    "Conflict Resolution & Disagreement",
    "Overcoming Failure / Tough Deadline",
    "Mentorship & Cross-Functional Collaboration",
    "Innovation & Process Optimization"
]


def extract_star_stories_heuristic(resume_text: str) -> List[Dict[str, Any]]:
    """Generates clean heuristic STAR stories from resume text when offline."""
    lines = [l.strip() for l in resume_text.splitlines() if len(l.strip()) > 15]
    summary_hint = lines[0] if lines else "Core system scaling & reliability"
    project_hint = lines[1] if len(lines) > 1 else "Microservices Migration"

    stories = [
        {
            "id": "story_tech_challenge",
            "title": "Resolving High-Severity Outage & Distributed Scaling",
            "category": "Technical Challenge / Incident Outage",
            "applicable_keywords": ["challenge", "outage", "scale", "bottleneck", "incident", "latency", "failure", "debug"],
            "situation": f"At my previous company, our production services faced unexpected latency spikes and throughput degradation during peak traffic. Context: {summary_hint[:100]}",
            "task": "I was the designated technical lead responsible for triaging the root cause, mitigating immediate customer impact, and implementing long-term architectural safeguards.",
            "action": "I led an emergency war room, analyzed distributed traces to pinpoint database connection pool exhaustion, added Redis rate limiting, and horizontally scaled application pods.",
            "result": "Restored 99.99% service availability within 35 minutes and prevented an estimated $120k in lost transaction volume.",
            "key_learnings": "Emphasized the importance of automated circuit breakers, comprehensive observability, and blameless post-mortems."
        },
        {
            "id": "story_conflict_disagreement",
            "title": "Architectural Alignment on Technology Migration",
            "category": "Conflict Resolution & Disagreement",
            "applicable_keywords": ["conflict", "disagreement", "differ", "pushback", "alignment", "convince", "stakeholder"],
            "situation": f"During the planning of our next-generation architecture ({project_hint[:80]}), team members had conflicting opinions between adopting a GraphQL gateway vs extending existing REST APIs.",
            "task": "My goal was to de-escalate emotional attachment to tools, establish data-driven evaluation criteria, and reach cross-team consensus without stalling delivery timelines.",
            "action": "I organized a time-boxed 3-day prototyping spike, benchmarked payload size, mobile client performance, and developer velocity metrics for both approaches, and presented an objective trade-off matrix.",
            "result": "The team unanimously agreed on a hybrid federated approach, completing the migration 2 weeks ahead of quarterly schedule.",
            "key_learnings": "Focusing on measurable business and engineering outcomes rather than dogmatic preferences creates psychological safety and team alignment."
        },
        {
            "id": "story_leadership_ownership",
            "title": "Mentoring Junior Engineers and Raising Code Quality Standards",
            "category": "Leadership & Ownership",
            "applicable_keywords": ["leadership", "ownership", "mentor", "initiative", "lead", "champion", "standard", "culture"],
            "situation": "Our engineering org grew rapidly, resulting in disparate coding practices, increased test flake rates, and onboarding bottlenecks for new hires.",
            "task": "I took initiative outside my core project scope to modernize our engineering standards and streamline engineering onboarding.",
            "action": "I established automated CI/CD linting & security scanning, created comprehensive RFC architecture templates, and instituted weekly peer code review workshops.",
            "result": "Reduced new engineer ramp-up time from 5 weeks to 2 weeks, and reduced production regression escaped bugs by 45%.",
            "key_learnings": "True engineering leadership is about multiplying team effectiveness and building scalable systems and cultures."
        }
    ]
    return stories


def extract_star_stories_llm(
    resume_text: str,
    client: Groq,
    custom_role: str = "Software Engineer"
) -> List[Dict[str, Any]]:
    """Extracts personalized STAR stories using Groq LLM reasoning."""
    if not resume_text:
        return extract_star_stories_heuristic(resume_text)

    prompt = f"""
You are a Principal Technical Recruiter and Staff Engineering Hiring Manager.
Analyze this candidate's resume and extract 3 to 4 distinct, highly compelling behavioral STAR stories
tailored for FAANG / Top-Tier Behavioral and Leadership interviews for a {custom_role}.

RESUME:
{resume_text[:4000]}

Return a valid JSON list of objects matching this exact schema:
[
  {{
    "id": "<short_snake_case_id>",
    "title": "<Concise story title>",
    "category": "<One of: Technical Challenge / Incident Outage, Leadership & Ownership, Conflict Resolution & Disagreement, Overcoming Failure / Tough Deadline, Mentorship & Cross-Functional Collaboration>",
    "applicable_keywords": ["<list of 6-8 relevant question keywords, e.g. conflict, challenge, leadership, failure, deadline>"],
    "situation": "<1-2 spoken sentences setting the context and high stakes>",
    "task": "<1 sentence stating the candidate's specific ownership and objective>",
    "action": "<2-3 clear first-person sentences detailing technical/interpersonal actions taken>",
    "result": "<1-2 sentences with concrete metrics, business impact, and measurable success>",
    "key_learnings": "<1 sentence summarizing key professional takeaway>"
  }}
]
Only return valid JSON without markdown code fences.
"""
    models_to_try = [
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "llama-3.3-70b-versatile",
        os.getenv("FAST_GROQ_MODEL", "llama-3.1-8b-instant"),
        "llama-3.1-8b-instant"
    ]
    models_to_try = list(dict.fromkeys(models_to_try))
    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            content = completion.choices[0].message.content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            stories = json.loads(content)
            if isinstance(stories, list) and len(stories) > 0 and "situation" in stories[0]:
                return stories
        except Exception as e:
            continue

    return extract_star_stories_heuristic(resume_text)


def find_best_star_story(question: str, stories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finds the most relevant STAR story for a given behavioral interview question."""
    if not stories:
        return extract_star_stories_heuristic("")[0]

    q_lower = question.lower()
    best_story = stories[0]
    best_score = -1

    for story in stories:
        score = 0
        keywords = story.get("applicable_keywords", [])
        for kw in keywords:
            if kw.lower() in q_lower:
                score += 3
        # Check category matches
        cat = story.get("category", "").lower()
        if any(c in q_lower for c in cat.split()):
            score += 2

        if score > best_score:
            best_score = score
            best_story = story

    return best_story


def format_star_spoken_answer(question: str, story: Dict[str, Any]) -> str:
    """Formats a STAR story into a spoken, natural 1st-person interview response."""
    title = story.get("title", "Key Experience")
    cat = story.get("category", "Behavioral Context")
    sit = story.get("situation", "")
    tsk = story.get("task", "")
    act = story.get("action", "")
    res = story.get("result", "")
    lrn = story.get("key_learnings", "")

    formatted = f"""🌟 BEHAVIORAL STAR RESPONSE: {title}
Category: {cat}

============================================================
1. 📍 SITUATION
============================================================
{sit}

============================================================
2. 🎯 TASK & OWNERSHIP
============================================================
{tsk}

============================================================
3. ⚡ ACTION TAKEN
============================================================
{act}

============================================================
4. 📈 MEASURABLE RESULT
============================================================
{res}

============================================================
5. 💡 KEY TAKEAWAY & REFLECTION
============================================================
{lrn}
"""
    return formatted.strip()


def format_star_natural_spoken_answer(question: str, story: Dict[str, Any]) -> str:
    """Formats a STAR story into a natural, teleprompter-ready conversational response without rigid meta-headers."""
    sit = story.get("situation", "").strip()
    act = story.get("action", "").strip()
    res = story.get("result", "").strip()
    lrn = story.get("key_learnings", "").strip()

    spoken = (
        f"{sit} "
        f"{act} "
        f"{res}"
    )
    if lrn:
        prefix = lrn[0].lower() + lrn[1:] if len(lrn) > 1 else lrn
        spoken += f" Looking back, {prefix}"
    return spoken.strip()


def load_star_stories_from_disk() -> List[Dict[str, Any]]:
    """Loads saved STAR story bank from disk."""
    if os.path.exists(STAR_STORIES_STORAGE_FILE):
        try:
            with open(STAR_STORIES_STORAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_star_stories_to_disk(stories: List[Dict[str, Any]]):
    """Persists STAR stories to disk."""
    try:
        with open(STAR_STORIES_STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(stories, f, indent=2)
    except Exception as e:
        print(f"Notice: Failed to save STAR stories: {e}")
