"""
Automated Test Suite for Parakit Answer Intelligence Layer
==========================================================
Tests:
1. Test 1: Why Follow-up
2. Test 2: Scalability & Performance Follow-up
3. Test 3: Technology Comparison Follow-up
4. Test 4: Expansion & Deepening Follow-up
5. Test 5: Concrete Example Request Follow-up
6. Test 6: Coding Optimization Follow-up
7. Test 7: Complexity Analysis Follow-up
8. Test 8: Security Considerations Follow-up
9. Test 9: Clean New Topic Transition (No Context Leakage)
10. Test 10: Ambiguity & Safest Interpretation
11. Test 11: Multi-turn Session Isolation
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine
from classifier import get_answer_strategy


class TestAnswerIntelligenceLayer(unittest.TestCase):

    def setUp(self):
        self.session = interview_engine.InterviewSession()
        self.session.clear()

    # ============================================================
    # TEST 1: WHY FOLLOW-UP
    # ============================================================
    def test_01_why_followup(self):
        self.session.active_topic = "MongoDB"
        self.session.add_transcript("Why did you choose MongoDB?", role="Interviewer", is_question=True)
        self.session.add_candidate_speech("Because our document schema was dynamic and rapidly evolving.")
        
        rel = interview_engine.classify_answer_relationship("Why?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "WHY")
        self.assertIn("MongoDB", rel["fused_question"])
        self.assertEqual(rel["strategy_key"], "follow_up_why")

    # ============================================================
    # TEST 2: SCALABILITY & PERFORMANCE FOLLOW-UP (DELTA FOCUS)
    # ============================================================
    def test_02_scalability_followup(self):
        self.session.active_topic = "MongoDB"
        self.session.add_transcript("Why did you choose MongoDB?", role="Interviewer", is_question=True)
        self.session.record_qa("Why did you choose MongoDB?", "MongoDB provided document flexibility for our JSON data.", category="database")
        self.session.add_candidate_speech("Because of flexibility.")
        
        rel = interview_engine.classify_answer_relationship("What about scalability?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "PERFORMANCE")
        self.assertIn("scalability", rel["fused_question"].lower())
        self.assertIn("sharding", rel["delta_focus"].lower())
        self.assertEqual(rel["strategy_key"], "follow_up_scalability")
        
        # Verify prompt construction includes anti-repetition delta
        prompt = interview_engine.build_interview_system_prompt(self.session, "follow_up_scalability", relationship_info=rel)
        self.assertIn("ANSWER INTELLIGENCE & CONVERSATION DELTA", prompt)
        self.assertIn("ANTI-REPETITION MANDATE", prompt)

    # ============================================================
    # TEST 3: TECHNOLOGY COMPARISON (WHY NOT MYSQL?)
    # ============================================================
    def test_03_comparison_followup(self):
        self.session.active_topic = "MongoDB"
        self.session.add_transcript("Why MongoDB?", role="Interviewer", is_question=True)
        
        rel = interview_engine.classify_answer_relationship("Why not MySQL?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "COMPARISON")
        self.assertIn("MySQL", rel["fused_question"])
        self.assertIn("MongoDB", rel["fused_question"])
        self.assertEqual(rel["strategy_key"], "follow_up_comparison")

    # ============================================================
    # TEST 4: EXPANSION & DEEPENING (CAN YOU EXPLAIN THAT?)
    # ============================================================
    def test_04_expansion_followup(self):
        self.session.active_topic = "Redis"
        self.session.add_transcript("Why did you use Redis?", role="Interviewer", is_question=True)
        self.session.add_candidate_speech("Redis was used for caching frequently accessed user profile data.")
        
        rel = interview_engine.classify_answer_relationship("Can you explain that?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "EXPANSION")
        self.assertIn("Redis", rel["fused_question"])
        self.assertEqual(rel["strategy_key"], "follow_up_expansion")

    # ============================================================
    # TEST 5: CONCRETE EXAMPLE REQUEST
    # ============================================================
    def test_05_example_request_followup(self):
        self.session.active_topic = "Redis Caching"
        self.session.add_transcript("How does caching help?", role="Interviewer", is_question=True)
        
        rel = interview_engine.classify_answer_relationship("Can you give me an example?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "EXAMPLE_REQUEST")
        self.assertIn("example", rel["delta_focus"].lower())
        self.assertEqual(rel["strategy_key"], "follow_up_example")

    # ============================================================
    # TEST 6: CODING OPTIMIZATION
    # ============================================================
    def test_06_coding_optimization_followup(self):
        self.session.active_topic = "Coding"
        self.session.add_transcript("Find duplicates in an array.", role="Interviewer", is_question=True)
        
        rel = interview_engine.classify_answer_relationship("Can you optimize it?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "OPTIMIZATION")
        self.assertEqual(rel["strategy_key"], "coding_optimization")

    # ============================================================
    # TEST 7: TIME & SPACE COMPLEXITY
    # ============================================================
    def test_07_complexity_followup(self):
        self.session.active_topic = "Coding"
        self.session.add_transcript("Find duplicates in an array.", role="Interviewer", is_question=True)
        
        rel = interview_engine.classify_answer_relationship("What is the time complexity?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "COMPLEXITY")
        self.assertEqual(rel["strategy_key"], "coding_complexity")

    # ============================================================
    # TEST 8: SECURITY CONSIDERATIONS
    # ============================================================
    def test_08_security_followup(self):
        self.session.active_topic = "JWT Authentication"
        self.session.add_transcript("How did you implement authentication?", role="Interviewer", is_question=True)
        self.session.add_candidate_speech("Using JSON Web Tokens stored in HttpOnly cookies.")
        
        rel = interview_engine.classify_answer_relationship("What about security?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "SECURITY")
        self.assertIn("JWT Authentication", rel["fused_question"])
        self.assertEqual(rel["strategy_key"], "follow_up_security")

    # ============================================================
    # TEST 9: NEW TOPIC SHIFT (NO CONTEXT LEAKAGE)
    # ============================================================
    def test_09_new_topic_no_leakage(self):
        self.session.active_topic = "Django Backend"
        self.session.add_transcript("Why did you choose Django?", role="Interviewer", is_question=True)
        
        # Interviewer switches topics
        rel = interview_engine.classify_answer_relationship("Now tell me about your internship.", session=self.session)
        self.assertFalse(rel["is_follow_up"])
        self.assertEqual(rel["relationship_type"], "NEW_TOPIC")
        self.assertNotIn("Django", rel["fused_question"])

    # ============================================================
    # TEST 10: AMBIGUITY SAFEGUARD
    # ============================================================
    def test_10_ambiguity_safeguard(self):
        self.session.active_topic = "Distributed Caching"
        self.session.add_transcript("Explain your caching approach.", role="Interviewer", is_question=True)
        
        rel = interview_engine.classify_answer_relationship("Can you explain that?", session=self.session)
        self.assertTrue(rel["is_follow_up"])
        self.assertIn("Distributed Caching", rel["fused_question"])

    # ============================================================
    # TEST 11: MULTI-SESSION ISOLATION
    # ============================================================
    def test_11_session_isolation(self):
        session_a = interview_engine.InterviewSession()
        session_b = interview_engine.InterviewSession()
        
        session_a.active_topic = "PostgreSQL"
        session_b.active_topic = "Kafka Streaming"
        
        rel_a = interview_engine.classify_answer_relationship("What about scalability?", session=session_a)
        rel_b = interview_engine.classify_answer_relationship("What about scalability?", session=session_b)
        
        self.assertIn("PostgreSQL", rel_a["fused_question"])
        self.assertNotIn("Kafka", rel_a["fused_question"])
        
        self.assertIn("Kafka", rel_b["fused_question"])
        self.assertNotIn("PostgreSQL", rel_b["fused_question"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
