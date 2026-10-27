"""
Parakeet-Grade Follow-Up Hardening Test Suite
============================================
Validates the 4 core pillars of the Parakeet AI follow-up architecture:
1. Tech entity shift detection (Zero Ghost Topic Bleed)
2. Candidate voice grounding (Candidate speech overrides stale topics)
3. Dynamic RAG retrieval using fused questions (Zero RAG starvation on "Why?")
4. Dual-layer prompt construction (effective_question + raw audio transcript)
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interview_engine import (
    InterviewSession,
    classify_answer_relationship,
    build_interview_system_prompt,
    generate_interview_answer
)


class TestParakeetFollowUpHardening(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()

    def test_01_tech_shift_prevents_ghost_topic_bleed(self):
        """When an interviewer asks about a new platform tech (Redis), do NOT append old topic (Django)."""
        self.session.active_topic = "Django Project"
        rel = classify_answer_relationship("How do you implement caching in Redis?", session=self.session)
        
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "HOW")
        self.assertEqual(rel["topic"], "Redis")
        self.assertEqual(self.session.active_topic, "Redis")
        self.assertNotIn("Django", rel["fused_question"])
        self.assertEqual(rel["fused_question"], "How do you implement caching in Redis?")
        self.assertIn("Redis", rel["delta_focus"])

    def test_02_why_kafka_shift_prevents_ghost_topic_bleed(self):
        """When asking 'Why do we use Kafka?', cleanly shift from MongoDB to Kafka without 'regarding MongoDB'."""
        self.session.active_topic = "MongoDB"
        rel = classify_answer_relationship("Why do we use Kafka for real-time streaming?", session=self.session)
        
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "WHY")
        self.assertEqual(rel["topic"], "Kafka")
        self.assertEqual(self.session.active_topic, "Kafka")
        self.assertNotIn("MongoDB", rel["fused_question"])
        self.assertEqual(rel["fused_question"], "Why do we use Kafka for real-time streaming?")
        self.assertIn("Kafka", rel["delta_focus"])

    def test_03_candidate_voice_grounding_overrides_stale_topic(self):
        """When candidate states they used PostgreSQL, follow-up resolves to PostgreSQL, NOT old MongoDB."""
        self.session.active_topic = "MongoDB"
        self.session.add_candidate_speech("For our database we went with PostgreSQL and partitioned the orders table by year.")
        
        rel = classify_answer_relationship("Why did you choose that partition scheme?", session=self.session)
        
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "WHY")
        self.assertEqual(rel["topic"], "PostgreSQL")
        self.assertEqual(self.session.active_topic, "PostgreSQL")
        self.assertIn("PostgreSQL", rel["fused_question"])
        self.assertNotIn("MongoDB", rel["fused_question"])

    @patch("rag.search_knowledge")
    def test_04_rag_retrieval_queries_fused_question_not_bare_why(self, mock_search):
        """RAG must be queried with the synthesized canonical question, not the bare word 'Why?'."""
        mock_search.return_value = "• Redis Caching: Latency < 2ms, TTL 3600s."
        
        rel_info = {
            "relationship_type": "WHY",
            "is_follow_up": True,
            "topic": "Redis",
            "fused_question": "Why did you choose Redis over Memcached for session management?",
            "delta_focus": "Justification only."
        }
        
        prompt = build_interview_system_prompt(
            session=self.session,
            question_type="follow_up_why",
            knowledge_source="FOLLOW_UP_CONVERSATION",
            relationship_info=rel_info,
            question="Why?"
        )
        
        mock_search.assert_called_once_with(
            "Why did you choose Redis over Memcached for session management?",
            top_k=5
        )
        self.assertIn("Redis Caching: Latency < 2ms", prompt)
        self.assertIn("ANSWER INTELLIGENCE & CONVERSATION DELTA", prompt)

    def test_05_generate_interview_answer_injects_effective_question_and_raw_transcript(self):
        """generate_interview_answer must present the fused question to the LLM and include raw audio."""
        mock_client = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="1. Direct justification: Redis was chosen for in-memory latency."))]
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        self.session.active_topic = "Redis Caching"
        
        ans = generate_interview_answer(
            client=mock_client,
            session=self.session,
            question="Why?",
            on_token=lambda tok: None,
            raw_transcript="Why?",
            question_type="technical"
        )
        
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        
        user_msg = [m["content"] for m in messages if m["role"] == "user"][-1]
        self.assertIn("Why regarding Redis Caching?", user_msg)
        self.assertIn('(Raw audio transcript: "Why?")', user_msg)
        self.assertIn("ANSWER INTELLIGENCE CONTEXT", user_msg)


if __name__ == "__main__":
    unittest.main()
