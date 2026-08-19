"""
Post-Interview Session Exporter (Feature 8)
===========================================
Exports live interview sessions into structured Markdown (.md), JSON (.json),
and formatted PDF/HTML reports for post-interview debriefs and preparation.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from interview_engine import InterviewSession


def generate_session_markdown(session: InterviewSession) -> str:
    """Generates structured Markdown report from an InterviewSession."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    role_str = session.job_title or "Software Engineering"
    company_str = session.company_name or "Tech Company"
    qa_list = getattr(session, "qa_pairs", []) or []
    
    md_lines = [
        f"# 🎙️ Interview Debrief Report - {role_str} @ {company_str}",
        f"**Date Generated:** {now_str}  ",
        f"**Session Mode:** {session.audio_source.title()} Audio Capture  ",
        f"**Total Questions Handled:** {len(qa_list)}  ",
        "",
        "---",
        "",
        "## 📊 Executive Summary & Performance Insights",
        session.session_summary or "Session completed successfully. Review detailed Q&A transcripts below.",
        "",
        "---",
        "",
        "## 💬 Chronological Questions & AI Solutions Breakdown",
        ""
    ]

    if not qa_list:
        md_lines.append("_No Q&A interactions recorded during this session._\n")
    else:
        for idx, qa in enumerate(qa_list, 1):
            q_text = qa.get("question", "Interview Question")
            ans_text = qa.get("answer", "No answer recorded.")
            ts = qa.get("timestamp", "")
            time_tag = f" ({ts})" if ts else ""
            
            md_lines.append(f"### Question {idx}{time_tag}")
            md_lines.append(f"**❓ Interviewer:** {q_text}\n")
            md_lines.append("**💡 Suggested Response / Architectural Solution:**")
            md_lines.append(f"```\n{ans_text}\n```\n")
            md_lines.append("---\n")

    if getattr(session, "job_description", ""):
        md_lines.append("## 📋 Targeted Job Description")
        md_lines.append(f"```\n{session.job_description}\n```\n")

    return "\n".join(md_lines)


def export_session_to_markdown(session: InterviewSession, output_path: str) -> str:
    """Exports session report to a Markdown file."""
    content = generate_session_markdown(session)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def export_session_to_json(session: InterviewSession, output_path: str) -> str:
    """Exports complete raw session data to a structured JSON file."""
    data = {
        "job_title": session.job_title,
        "company_name": session.company_name,
        "job_description": session.job_description,
        "audio_source": session.audio_source,
        "timestamp": datetime.now().isoformat(),
        "session_summary": session.session_summary,
        "qa_pairs": getattr(session, "qa_pairs", []),
        "raw_transcript_lines": getattr(session, "transcript_lines", [])
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output_path


def export_session_to_text_report(session: InterviewSession, output_path: str) -> str:
    """Exports session report to clean plain text format."""
    content = generate_session_markdown(session)
    # Strip markdown syntax for plain text
    plain_text = content.replace("# ", "").replace("## ", "").replace("### ", "").replace("**", "").replace("```", "")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plain_text)
    return output_path
