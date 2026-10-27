import os
import unittest
from unittest.mock import MagicMock, patch

from interview_engine import (
    InterviewSession,
    ResilientMultiSTTProvider,
    GroqWhisperProvider,
    generate_interview_answer
)
from web_copilot_view import WebCopilotView


class TestOpenAIAndSystemHealth(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()
        self.session.clear()
        self.session.job_title = "Senior Python & Distributed Systems Engineer"
        self.session.company_name = "Tech Corp"
        self.session.resume_text = "Experience: 5 years building scalable microservices, FastAPI, Redis, Docker, and PostgreSQL."

    def test_resilient_stt_fallbacks_to_openai_when_groq_unavailable(self):
        """Verify that when Groq is unavailable, ResilientMultiSTTProvider seamlessly falls back to OpenAI Whisper."""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key-12345", "GROQ_API_KEY": ""}):
            provider = ResilientMultiSTTProvider()
            mock_openai = MagicMock()
            mock_openai.audio.transcriptions.create.return_value = "What is the difference between processes and threads?"
            provider._openai_client = mock_openai

            # Simulate 16000Hz 16-bit PCM audio with speech energy (0.1 sec = 3200 bytes)
            dummy_audio = b"\x50\x20" * 1600
            result = provider.transcribe(dummy_audio, sample_rate=16000)

            self.assertIn("difference between processes and threads", result)
            mock_openai.audio.transcriptions.create.assert_called_once()

    def test_generate_interview_answer_openai_streaming(self):
        """Verify that generate_interview_answer streams tokens directly using OpenAI when model starts with gpt-."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key-12345"}):
            self.session.model_name = "gpt-4o-mini"
            
            # Mock chunk response stream
            mock_chunk1 = MagicMock()
            mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Processes have separate memory spaces,"))]
            mock_chunk2 = MagicMock()
            mock_chunk2.choices = [MagicMock(delta=MagicMock(content=" while threads share the same address space."))]

            mock_openai = MagicMock()
            mock_openai.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]

            collected_tokens = []
            def on_token(token):
                collected_tokens.append(token)

            with patch("importlib.import_module") as mock_import:
                mock_mod = MagicMock()
                mock_mod.OpenAI.return_value = mock_openai
                mock_import.return_value = mock_mod

                generate_interview_answer(
                    session=self.session,
                    question="What is the difference between a process and a thread?",
                    client=None, # Groq client is None
                    on_token=on_token
                )

            full_answer = "".join(collected_tokens)
            self.assertIn("Processes have separate memory spaces", full_answer)
            self.assertIn("while threads share the same address space", full_answer)

    def test_candidate_audio_records_in_memory_safely(self):
        """Verify candidate speech is recorded into memory without throwing errors."""
        self.session.add_candidate_speech("I used Redis for caching frequent database queries.")
        self.assertEqual(self.session.last_candidate_response, "I used Redis for caching frequent database queries.")

    def test_handle_launch_session_config_ingestion(self):
        """Verify WebCopilotView correctly ingests OpenAI config without exceptions."""
        view = WebCopilotView()
        config = {
            "role": "Senior Cloud Architect",
            "company": "Amazon AWS",
            "model": "gpt-4o-mini",
            "style": "STAR Behavioral Framework",
            "audio_source": "dual",
            "save_transcript": True
        }
        try:
            res = view.handle_launch_session(config)
            self.assertEqual(res["status"], "ok")
            self.assertEqual(view.session.job_title, "Senior Cloud Architect")
            self.assertEqual(view.session.company_name, "Amazon AWS")
            self.assertEqual(view.session.model_name, "gpt-4o-mini")
            self.assertEqual(view.session.tone, "STAR Behavioral Framework")
            self.assertEqual(view.session.audio_source, "dual")
        finally:
            if view.transcriber and view.transcriber.is_running:
                view.transcriber.stop()


if __name__ == "__main__":
    unittest.main()
