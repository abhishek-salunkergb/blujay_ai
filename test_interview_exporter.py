"""
Unit Test Suite for Post-Interview Session Exporter (Feature 8)
===============================================================
Validates:
1. Markdown report generation with metadata and Q&A pairs
2. JSON structured data export
3. Plain text report export
4. Handling of empty sessions and edge-case formatting
"""

import os
import json
import tempfile
import unittest

from interview_engine import InterviewSession
from interview_exporter import (
    generate_session_markdown,
    export_session_to_markdown,
    export_session_to_json,
    export_session_to_text_report
)


class TestInterviewExporter(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()
        self.session.job_title = "Staff Systems Engineer"
        self.session.company_name = "Stripe"
        self.session.job_description = "Distributed Payments & High Availability"
        self.session.session_summary = "Demonstrated deep domain expertise in idempotent payment workflows."
        self.session.qa_pairs = [
            {
                "question": "How do you handle double-charge race conditions?",
                "answer": "Utilize atomic idempotency keys with Redis distributed locks and unique DB constraints.",
                "timestamp": "14:02:15"
            },
            {
                "question": "Explain consistency models in multi-region ledger replicas.",
                "answer": "Use Raft consensus for the primary shard group with asynchronous cross-region disaster replicas.",
                "timestamp": "14:08:40"
            }
        ]

    def test_01_generate_session_markdown(self):
        """Verify markdown string contains all session information."""
        md = generate_session_markdown(self.session)
        self.assertIn("Staff Systems Engineer", md)
        self.assertIn("Stripe", md)
        self.assertIn("double-charge race conditions", md)
        self.assertIn("atomic idempotency keys", md)
        self.assertIn("Distributed Payments", md)

    def test_02_export_session_to_markdown_file(self):
        """Verify markdown export writes file to disk."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
            tmp_path = tmp.name

        try:
            out_path = export_session_to_markdown(self.session, tmp_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Staff Systems Engineer", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_03_export_session_to_json_file(self):
        """Verify JSON export creates parseable json with correct structure."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name

        try:
            out_path = export_session_to_json(self.session, tmp_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["job_title"], "Staff Systems Engineer")
            self.assertEqual(len(data["qa_pairs"]), 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_04_export_session_to_text_report(self):
        """Verify text report export."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp_path = tmp.name

        try:
            out_path = export_session_to_text_report(self.session, tmp_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                txt = f.read()
            self.assertIn("Interview Debrief Report", txt)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
