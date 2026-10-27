"""
Unit Test Suite for System Design Architecture Generator (Feature 6)
====================================================================
Validates:
1. Heuristic architecture generator output structure and ASCII layout
2. Scale calculations (DAU, QPS, Storage) format
3. Trade-offs and talking points extraction
4. LLM structured generation and fallback robustness
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import json
import unittest
from unittest.mock import MagicMock

from system_design_generator import (
    generate_heuristic_architecture,
    generate_system_design_diagram
)


class TestSystemDesignGenerator(unittest.TestCase):

    def test_01_heuristic_architecture_structure(self):
        """Verify heuristic architecture generator returns all required components."""
        res = generate_heuristic_architecture("URL Shortener (TinyURL)")
        self.assertEqual(res["topic"], "URL Shortener (TinyURL)")
        self.assertIn("ascii_diagram", res)
        self.assertIn("mermaid_diagram", res)
        self.assertIn("scale_estimations", res)
        self.assertIn("key_trade_offs", res)
        self.assertIn("interview_talking_points", res)

        self.assertIn("Global Users", res["ascii_diagram"])
        self.assertIn("Redis", res["ascii_diagram"])
        self.assertGreaterEqual(len(res["key_trade_offs"]), 2)

    def test_02_scale_estimations_structure(self):
        """Verify scale estimations contain DAU, QPS, and storage metrics."""
        res = generate_heuristic_architecture("Chat Application")
        scale = res["scale_estimations"]
        self.assertIn("daily_active_users", scale)
        self.assertIn("read_qps", scale)
        self.assertIn("write_qps", scale)
        self.assertIn("storage_per_year", scale)

    def test_03_generate_with_llm_mock(self):
        """Verify generate_system_design_diagram parses structured LLM output."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "topic": "Distributed Cache",
                "ascii_diagram": "+--- Client ---+ -> +--- Consistent Hash Ring ---+",
                "mermaid_diagram": "graph TD\nClient --> Ring",
                "scale_estimations": {
                    "daily_active_users": "100M",
                    "read_write_ratio": "50:1",
                    "read_qps": "500,000 QPS",
                    "write_qps": "10,000 QPS",
                    "storage_per_year": "10 TB"
                },
                "key_trade_offs": ["LRU vs LFU eviction policy"],
                "interview_talking_points": ["Discuss cache invalidation strategies."]
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        res = generate_system_design_diagram("Distributed Cache", client=mock_client)
        self.assertEqual(res["topic"], "Distributed Cache")
        self.assertIn("Consistent Hash Ring", res["ascii_diagram"])
        self.assertIn("LRU vs LFU", res["key_trade_offs"][0])


if __name__ == "__main__":
    unittest.main()
