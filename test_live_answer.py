"""
Test live answer generation and token streaming with Groq
"""
import sys
import os
import unittest
from dotenv import load_dotenv
from groq import Groq
import interview_engine

load_dotenv()


class TestLiveGroqAnswer(unittest.TestCase):
    @unittest.skipUnless(os.getenv("LIVE_TEST") == "1", "Live manual test only; set LIVE_TEST=1 to run")
    def test_live_answer_stream_and_refinement(self):
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        session = interview_engine.InterviewSession()
        session.job_title = "AI Engineer"
        session.company_name = "TechCorp"
        session.tone = "Confident & Conversational"

        collected_tokens = []
        q = "Tell me about a time you handled a complex production incident with high traffic."
        answer = interview_engine.generate_interview_answer(
            client=client,
            session=session,
            question=q,
            on_token=lambda t: collected_tokens.append(t)
        )
        self.assertTrue(len(answer) > 0)


if __name__ == "__main__":
    unittest.main()
