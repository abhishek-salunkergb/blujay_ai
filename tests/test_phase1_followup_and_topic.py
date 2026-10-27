import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interview_engine import InterviewSession, classify_answer_relationship, classify_and_understand_utterance, detect_speaker_turn


class TestPhase1FollowUpAndTopic(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()

    def test_01_why_did_you_choose_django(self):
        """Test sentence-level WHY question preserves/references topic and classifies as WHY."""
        self.session.active_topic = "Django Project"
        res = classify_answer_relationship("Why did you choose Django?", session=self.session)
        self.assertTrue(res["is_follow_up"])
        self.assertEqual(res["relationship_type"], "WHY")
        self.assertEqual(res["topic"], "Django Project")
        self.assertIn("Why did you choose Django", res["fused_question"])

    def test_02_how_did_you_handle_authentication(self):
        """Test sentence-level HOW question classifies as HOW and connects to active topic."""
        self.session.active_topic = "Django REST Backend"
        res = classify_answer_relationship("How did you handle authentication?", session=self.session)
        self.assertTrue(res["is_follow_up"])
        self.assertEqual(res["relationship_type"], "HOW")
        self.assertEqual(res["topic"], "Django REST Backend")
        self.assertIn("How did you handle authentication", res["fused_question"])

    def test_03_why_is_that_better_pronoun_resolution(self):
        """Test pronoun resolution for 'that' -> active topic."""
        self.session.active_topic = "Redis Caching"
        res = classify_answer_relationship("Why is that better?", session=self.session)
        self.assertTrue(res["is_follow_up"])
        self.assertEqual(res["relationship_type"], "WHY")
        self.assertEqual(res["topic"], "Redis Caching")
        self.assertIn("Redis Caching", res["fused_question"])
        self.assertNotIn("that", res["fused_question"].lower())

    def test_04_how_does_it_work_pronoun_resolution(self):
        """Test pronoun resolution for 'it' -> active topic."""
        self.session.active_topic = "Kafka Stream Pipeline"
        res = classify_answer_relationship("How does it work?", session=self.session)
        self.assertTrue(res["is_follow_up"])
        self.assertEqual(res["relationship_type"], "HOW")
        self.assertEqual(res["topic"], "Kafka Stream Pipeline")
        self.assertIn("Kafka Stream Pipeline", res["fused_question"])

    def test_05_statement_containing_how_should_not_trigger_followup(self):
        """Declarative statement 'I know how Django works.' should NOT trigger follow-up."""
        self.session.active_topic = "Django"
        res = classify_answer_relationship("I know how Django works.", session=self.session)
        self.assertFalse(res["is_follow_up"])
        self.assertEqual(res["relationship_type"], "NEW_TOPIC")

    def test_06_statement_containing_why_should_not_trigger_followup(self):
        """Declarative statement 'We discussed why performance matters.' should NOT trigger follow-up."""
        self.session.active_topic = "Architecture"
        res = classify_answer_relationship("We discussed why performance matters.", session=self.session)
        self.assertFalse(res["is_follow_up"])
        self.assertEqual(res["relationship_type"], "NEW_TOPIC")

    def test_07_social_greeting_how_should_not_trigger_followup(self):
        """Social greeting 'How are you today?' should NOT be treated as a technical HOW follow-up."""
        self.session.active_topic = "Django"
        res = classify_answer_relationship("How are you today?", session=self.session)
        self.assertFalse(res["is_follow_up"])

    def test_08_active_topic_preservation_during_followup(self):
        """Verify active_topic is NOT overwritten by new utterance before follow-up resolution."""
        # Turn 1: Discuss Django architecture -> sets active_topic to Django
        res1 = classify_answer_relationship("Explain your Django project architecture.", session=self.session)
        self.assertEqual(self.session.active_topic, "Django")
        
        # Turn 2: Follow-up question with pronoun 'it'
        res2 = classify_answer_relationship("Why did you choose it?", session=self.session)
        self.assertTrue(res2["is_follow_up"])
        self.assertEqual(res2["relationship_type"], "WHY")
        self.assertEqual(self.session.active_topic, "Django")
        self.assertEqual(res2["fused_question"], "Why did you choose Django?")

    def test_09_rolling_3_candidate_utterances_window(self):
        """Test that last_candidate_response maintains a rolling window of max 3 items."""
        # Empty at start
        self.assertEqual(self.session.last_candidate_response, "")
        self.assertEqual(self.session.recent_candidate_context, "")
        
        # Utterance 1
        self.session.add_candidate_speech("I used Django REST Framework for the backend.")
        self.assertEqual(self.session.last_candidate_response, "I used Django REST Framework for the backend.")
        self.assertEqual(len(self.session._candidate_responses), 1)
        
        # Utterance 2
        self.session.add_candidate_speech("For authentication, I implemented JWT tokens.")
        self.assertEqual(self.session.last_candidate_response, "For authentication, I implemented JWT tokens.")
        self.assertEqual(len(self.session._candidate_responses), 2)
        
        # Utterance 3
        self.session.add_candidate_speech("We deployed everything on AWS ECS with Docker.")
        self.assertEqual(self.session.last_candidate_response, "We deployed everything on AWS ECS with Docker.")
        self.assertEqual(len(self.session._candidate_responses), 3)
        self.assertIn("Django REST", self.session.recent_candidate_context)
        self.assertIn("JWT tokens", self.session.recent_candidate_context)
        self.assertIn("AWS ECS", self.session.recent_candidate_context)
        
        # Utterance 4 (Window evicts first item)
        self.session.add_candidate_speech("Database queries are cached using Redis.")
        self.assertEqual(self.session.last_candidate_response, "Database queries are cached using Redis.")
        self.assertEqual(len(self.session._candidate_responses), 3)
        self.assertNotIn("Django REST", self.session.recent_candidate_context)
        self.assertIn("JWT tokens", self.session.recent_candidate_context)
        self.assertIn("Redis", self.session.recent_candidate_context)

        # Clear wipes the rolling window
        self.session.clear()
        self.assertEqual(self.session.last_candidate_response, "")
        self.assertEqual(self.session.recent_candidate_context, "")
        self.assertEqual(len(self.session._candidate_responses), 0)


if __name__ == "__main__":
    unittest.main()
