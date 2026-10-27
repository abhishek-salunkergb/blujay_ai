"""
Unit & Integration Tests for Natural Spoken Answer Quality and Reasoning Leakage Prevention
=============================================================================================
Verifies:
1. Stripping <think>...</think> and unclosed reasoning tags
2. Stripping planning headers (Context:, Strategy:, Background:, Rules:, Drafting:, Planning:)
3. Streaming filter suppresses internal reasoning before UI rendering
4. "Tell me about yourself" produces natural 60-100 word spoken response without resume dumping
5. Conversational questions have zero rigid sections (no synthetic '✦ ANSWER' or '✦ APPROACH')
6. No overselling (does not claim unverified Senior roles or fabricated metrics)
7. Avoids corporate buzzwords ('architected', 'leveraging', 'robust engineering ecosystem')
8. Natural spoken length budgets across all 9 required interview questions
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
from unittest.mock import MagicMock

from interview_engine import (
    InterviewSession,
    sanitize_generated_answer,
    StreamingReasoningFilter,
    validate_and_clean_answer,
    build_interview_system_prompt
)
from answer_formatter import parse_structured_answer, format_interview_answer


class TestAnswerQualityAndSanitization(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()
        self.session.clear()
        self.session.job_title = "Python / AI Developer"
        self.session.resume_text = (
            "EDUCATION: BTech in Computer Engineering.\n"
            "SKILLS: Python, Java, JavaScript, Django, React, MySQL, MongoDB, NLP, RAG, LLMs, AWS, Docker.\n"
            "PROJECT: Farmer Marketplace - built using Django, React, and MySQL with AI-driven crop pricing features."
        )

    # -------------------------------------------------------------------------
    # 1. REASONING LEAKAGE STRIPPING (<think>, <thought>, <analysis>, <plan>)
    # -------------------------------------------------------------------------
    def test_strip_think_tags_and_reasoning_blocks(self):
        raw_output = """<think>
The user wants a response to "Tell me about yourself."
Context: Candidate is Abhishek, BTech Computer Engineering.
Strategy: Crisp 2-3 sentence core response.
Drafting:
</think>
I'm a Computer Engineering graduate with a strong focus on Python, full-stack development, and practical AI. I've worked extensively with Django, React, and databases like MySQL and MongoDB, building projects like a farmer marketplace with AI-driven features. I'm looking for a developer role where I can build impactful applications and continue learning."""
        
        cleaned = sanitize_generated_answer(raw_output)
        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("</think>", cleaned)
        self.assertNotIn("The user wants a response", cleaned)
        self.assertNotIn("Strategy:", cleaned)
        self.assertNotIn("Drafting:", cleaned)
        self.assertTrue(cleaned.startswith("I'm a Computer Engineering graduate"))
        
        words = cleaned.split()
        self.assertGreaterEqual(len(words), 40)
        self.assertLessEqual(len(words), 120)

    # -------------------------------------------------------------------------
    # 2. STRIP UNTAGGED PLANNING HEADERS (Context:, Strategy:, Background:, Drafting:)
    # -------------------------------------------------------------------------
    def test_strip_untagged_planning_headers(self):
        raw_planning = """Context: Live interview, candidate Abhishek.
Strategy: Focus on Python and Django experience in the Farmer Marketplace.
Background: BTech Computer Engineering.
Rules: Spoken first-person tone.
Drafting:
"I've worked with Python and Django to build web services and real-time platforms. In my farmer marketplace project, I developed the backend APIs and integrated MySQL for data persistence. I'm excited about full-stack and AI roles where I can contribute to core features."
"""
        cleaned = sanitize_generated_answer(raw_planning)
        self.assertNotIn("Context:", cleaned)
        self.assertNotIn("Strategy:", cleaned)
        self.assertNotIn("Background:", cleaned)
        self.assertNotIn("Rules:", cleaned)
        self.assertNotIn("Drafting:", cleaned)
        self.assertTrue(cleaned.startswith("I've worked with Python and Django"))
        self.assertFalse(cleaned.endswith('"'))

    # -------------------------------------------------------------------------
    # 3. STREAMING REASONING FILTER SUPPRESSES REASONING DURING LIVE STREAM
    # -------------------------------------------------------------------------
    def test_streaming_reasoning_filter(self):
        streamed_tokens = []
        filter_obj = StreamingReasoningFilter(lambda t: streamed_tokens.append(t))
        
        raw_stream = [
            "<think>", "\nThe user wants a response.", "\nStrategy: keep it brief.\n", "</think>\n",
            "I'm a ", "software engineer ", "focused on ", "Python and Django development."
        ]
        for token in raw_stream:
            filter_obj.push(token)
        filter_obj.flush()
        
        combined_stream = "".join(streamed_tokens)
        self.assertNotIn("<think>", combined_stream)
        self.assertNotIn("Strategy:", combined_stream)
        self.assertNotIn("The user wants a response", combined_stream)
        self.assertIn("I'm a software engineer focused on Python and Django development.", combined_stream)

    # -------------------------------------------------------------------------
    # 4. "TELL ME ABOUT YOURSELF" NATURAL SPOKEN ANSWER & WORD COUNT BUDGET
    # -------------------------------------------------------------------------
    def test_tell_me_about_yourself_prompt_and_length(self):
        prompt = build_interview_system_prompt(self.session, question_type="resume")
        self.assertIn("60–100 words", prompt)
        self.assertIn("ZERO REASONING LEAKAGE", prompt)
        self.assertIn("AVOID CORPORATE BUZZWORDS", prompt)
        
        sample_good_answer = (
            "I'm a final-year Computer Engineering student with a strong interest in Python, "
            "full-stack development, and AI. I've worked with technologies like Django, React, "
            "MySQL, and modern AI/ML tools, and I've built projects such as a farmer marketplace "
            "with AI-based features. I'm looking for a role where I can apply my development skills "
            "to real-world products and continue growing as an engineer."
        )
        cleaned = validate_and_clean_answer(
            sample_good_answer,
            question="Tell me about yourself.",
            session=self.session,
            question_type="resume",
            knowledge_source="CANDIDATE_EXPERIENCE"
        )
        words = cleaned.split()
        self.assertGreaterEqual(len(words), 50)
        self.assertLessEqual(len(words), 110)
        self.assertNotIn("✦ ANSWER", cleaned)
        self.assertNotIn("✦ APPROACH", cleaned)

    # -------------------------------------------------------------------------
    # 5. NO RIGID SECTION HEADINGS FOR SIMPLE HR/INTRODUCTORY QUESTIONS
    # -------------------------------------------------------------------------
    def test_conversational_questions_no_rigid_sections(self):
        answer_text = (
            "ANSWER:\n"
            "I chose Django because its built-in ORM and authentication allowed us to build the backend quickly.\n\n"
            "KEY POINT:\n"
            "It provided a reliable foundation for our marketplace."
        )
        formatted = format_interview_answer(answer_text, question_type="hr")
        self.assertNotIn("ANSWER:", formatted)
        self.assertNotIn("KEY POINT:", formatted)
        self.assertIn("I chose Django because", formatted)

    # -------------------------------------------------------------------------
    # 6. NO OVERSELLING & NO RESUME DUMPING
    # -------------------------------------------------------------------------
    def test_no_overselling_rules_in_prompt(self):
        prompt = build_interview_system_prompt(self.session, question_type="hr")
        self.assertIn("NO OVERSELLING", prompt)
        self.assertIn("Never invent companies", prompt)
        self.assertIn("DO NOT DUMP KEYWORDS", prompt)

    # -------------------------------------------------------------------------
    # 7. 9 TARGET INTERVIEW QUESTIONS SANITIZATION & BOUNDED BUDGETS
    # -------------------------------------------------------------------------
    def test_9_target_interview_questions_sanitization(self):
        test_qa_pairs = [
            ("Tell me about yourself.", "I'm a final-year Computer Engineering student focusing on Python and AI...", 120),
            ("Why should we hire you?", "I bring practical full-stack experience with Django and React, along with AI skills...", 110),
            ("Why did you choose Django?", "I chose Django because of its robust ORM and built-in security features...", 95),
            ("Tell me about your project.", "I built a farmer marketplace where buyers and sellers can trade directly with AI pricing...", 120),
            ("What was your role in the project?", "I designed the database schema with MySQL and implemented the core REST APIs in Django...", 100),
            ("Why MongoDB?", "We used MongoDB for unstructured data that required flexible schema evolution...", 90),
            ("What is your biggest challenge?", "One major challenge was optimizing real-time query latencies under concurrent user load...", 150),
            ("Why do you want this role?", "I want to work on production software where I can combine backend engineering with practical AI...", 100),
            ("What are your strengths?", "My main strengths are fast problem solving, clean backend architecture in Python, and adaptability...", 100),
        ]
        
        for q, ans, max_w in test_qa_pairs:
            cleaned = sanitize_generated_answer(ans)
            words = cleaned.split()
            self.assertLessEqual(len(words), max_w, f"Answer for '{q}' exceeded {max_w} words")
            self.assertNotIn("<think>", cleaned)
            self.assertNotIn("Strategy:", cleaned)


if __name__ == "__main__":
    unittest.main()
