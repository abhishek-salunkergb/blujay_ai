import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import interview_engine
from interview_engine import InterviewSession, build_interview_system_prompt
import main
from web_copilot_view import WebCopilotView
import session_client


class TestWiringAndDisconnections(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()
        self.session.job_title = "Senior Python Engineer"
        self.session.company_name = "Acme Cloud"
        self.session.qa_pairs = [
            {"question": "Tell me about your Python background.", "answer": "I have built distributed microservices.", "timestamp": "12:00:00"}
        ]
        self.session.transcript_log = [
            {"speaker": "Interviewer", "text": "Tell me about your Python background.", "timestamp": "12:00:00"}
        ]

    def test_rag_connected_to_system_prompt(self):
        """Verifies that RAG search is dynamically queried and injected into the system prompt."""
        question = "What distributed database did you use for high-throughput messaging?"
        
        # Test with question parameter
        prompt = build_interview_system_prompt(
            self.session,
            question_type="technical",
            question=question
        )
        
        self.assertIn("SEMANTIC RETRIEVED KNOWLEDGE (DYNAMIC RAG CONTEXT)", prompt)
        self.assertIn("RAG GROUNDING MANDATE", prompt)

    def test_main_ask_ai_unification(self):
        """Verifies ask_ai in main.py delegates cleanly to interview_engine without legacy prompt logic."""
        mock_view = MagicMock()
        mock_view.session = self.session
        main.app = mock_view
        
        tokens_received = []
        def on_token(tok):
            tokens_received.append(tok)
            
        with patch("interview_engine.generate_interview_answer") as mock_gen:
            mock_gen.return_value = "Mocked canonical answer."
            ans = main.ask_ai("Explain Python GIL trade-offs", on_token)
            
            mock_gen.assert_called_once()
            args, kwargs = mock_gen.call_args
            self.assertEqual(kwargs.get("question"), "Explain Python GIL trade-offs")
            self.assertEqual(kwargs.get("session"), self.session)
            self.assertEqual(ans, "Mocked canonical answer.")

    def test_web_copilot_view_session_end_persistence(self):
        """Verifies handle_end_session passes complete qa_pairs, transcript_log, and summary_markdown to sc.end_session."""
        view = WebCopilotView.__new__(WebCopilotView)
        view.session = self.session
        view.transcriber = None
        view.window = None
        view.current_page = "copilot"
        view.is_card_expanded = True
        view._is_closing = False
        view._closed = False
        
        with patch("session_client.get_session_client") as mock_get_sc:
            mock_sc = MagicMock()
            mock_get_sc.return_value = mock_sc
            
            res = view.handle_end_session()
            self.assertEqual(res, {"status": "ended"})
            
            mock_sc.end_session.assert_called_once()
            _, kwargs = mock_sc.end_session.call_args
            
            self.assertEqual(kwargs.get("qa_pairs"), self.session.qa_pairs)
            self.assertEqual(kwargs.get("transcript_log"), self.session.transcript_log)
            self.assertEqual(kwargs.get("job_title"), "Senior Python Engineer")
            self.assertEqual(kwargs.get("company_name"), "Acme Cloud")
            self.assertIsNotNone(kwargs.get("summary_markdown"))
            self.assertIn("Interview Debrief Report", kwargs.get("summary_markdown"))

    def test_main_on_session_closed_persistence(self):
        """Verifies main.py on_session_closed properly resolves app.session and persists history."""
        view = MagicMock()
        view.session = self.session
        main.app = view
        
        with patch("session_client.get_session_client") as mock_get_sc:
            mock_sc = MagicMock()
            mock_get_sc.return_value = mock_sc
            
            main.on_session_closed()
            
            mock_sc.end_session.assert_called_once()
            _, kwargs = mock_sc.end_session.call_args
            self.assertEqual(kwargs.get("qa_pairs"), self.session.qa_pairs)
            self.assertEqual(kwargs.get("transcript_log"), self.session.transcript_log)
            self.assertEqual(kwargs.get("job_title"), "Senior Python Engineer")
            self.assertEqual(kwargs.get("company_name"), "Acme Cloud")
            self.assertIn("Interview Debrief Report", kwargs.get("summary_markdown"))


if __name__ == "__main__":
    unittest.main()
