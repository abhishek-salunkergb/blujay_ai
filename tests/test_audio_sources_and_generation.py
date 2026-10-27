import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interview_engine import InterviewSession
from web_copilot_view import WebCopilotView


class TestAudioSourceRoutingAndGeneration(unittest.TestCase):
    def setUp(self):
        self.view = WebCopilotView.__new__(WebCopilotView)
        self.view.session = InterviewSession()
        self.view.client = MagicMock()
        self.view.window = None
        self.view.transcriber = MagicMock()
        self.view.transcriber.is_running = False
        self.view.current_page = "copilot"
        self.view.is_card_expanded = False
        self.view._eval_js = MagicMock()
        self.view._handle_detected_question = MagicMock()

    def test_mic_only_mode_answers_question_from_mic(self):
        """In Mic-Only mode, questions spoken into the mic must trigger answer generation."""
        self.view.session.audio_source = "mic"
        self.view.session.auto_generate = True
        
        # Interviewer asks question into mic
        self.view._on_transcript_chunk(
            "What is the difference between processes and threads in Python?",
            source="mic"
        )
        
        self.view._handle_detected_question.assert_called_once()
        args, kwargs = self.view._handle_detected_question.call_args
        self.assertIn("processes and threads", args[0].lower())

    def test_mic_only_mode_ignores_candidate_narrative(self):
        """In Mic-Only mode, candidate first-person answers must not trigger answers."""
        self.view.session.audio_source = "mic"
        self.view.session.auto_generate = True
        
        # Candidate speaks answer
        self.view._on_transcript_chunk(
            "I built a high-throughput pipeline using Celery and Redis.",
            source="mic"
        )
        
        self.view._handle_detected_question.assert_not_called()

    def test_system_only_mode_answers_speaker_question(self):
        """In System-Only mode, questions received on speaker loopback trigger answer generation."""
        self.view.session.audio_source = "system"
        self.view.session.auto_generate = True
        
        self.view._on_transcript_chunk(
            "How do you handle database failover in PostgreSQL?",
            source="system"
        )
        
        self.view._handle_detected_question.assert_called_once()
        args, kwargs = self.view._handle_detected_question.call_args
        self.assertIn("postgresql", args[0].lower())

    def test_dual_mode_answers_speaker_question(self):
        """In Dual mode, questions received on speaker loopback (interviewer) trigger answer generation."""
        self.view.session.audio_source = "dual"
        self.view.session.auto_generate = True
        
        self.view._on_transcript_chunk(
            "Can you explain how Kafka consumer groups rebalance?",
            source="system"
        )
        
        self.view._handle_detected_question.assert_called_once()

    def test_dual_mode_candidate_mic_never_answers_even_interrogative(self):
        """In Dual mode, candidate speech into mic (even if interrogative) must NEVER trigger answer generation."""
        self.view.session.audio_source = "dual"
        self.view.session.auto_generate = True
        
        # Candidate asks interviewer a clarifying question into their mic
        self.view._on_transcript_chunk(
            "Should I write the optimal solution or start with a brute force approach?",
            source="candidate_mic"
        )
        
        # Must NOT trigger answer generation
        self.view._handle_detected_question.assert_not_called()
        
        # Candidate speech must be preserved in candidate responses
        cand_texts = [r["text"] for r in self.view.session.candidate_responses_log]
        self.assertTrue(any("brute force" in t for t in cand_texts))

    def test_dual_mode_candidate_interim_and_eager_eot_never_trigger(self):
        """In Dual mode, candidate interim or eager EOT on candidate_mic must NEVER trigger early answers."""
        self.view.session.audio_source = "dual"
        self.view.session.auto_generate = True
        
        # Interim transcript on candidate_mic
        self.view._on_interim_transcript(
            "What about edge cases?",
            likelihood_info={"can_start_early": True, "semantic_key": "test_key"},
            source="candidate_mic"
        )
        self.view._handle_detected_question.assert_not_called()

        # Eager EOT on candidate_mic
        self.view._on_eager_eot("What about edge cases?", source="candidate_mic")
        self.view._handle_detected_question.assert_not_called()


if __name__ == "__main__":
    unittest.main()
