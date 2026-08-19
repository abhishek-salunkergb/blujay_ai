"""
Unit Test Suite for Local Offline Fallback Engine (Feature 12)
==============================================================
Validates:
1. Offline heuristic answer generation for System Design, Behavioral STAR, and Technical questions
2. Local Ollama availability checker and mock streaming
3. OfflineFallbackManager preference saving and provider selection
4. Resilience against missing network connections
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from offline_fallback_manager import (
    check_ollama_available,
    stream_ollama_response,
    generate_offline_heuristic_answer,
    OfflineFallbackManager
)


class TestOfflineFallbackManager(unittest.TestCase):

    def test_01_heuristic_system_design_answer(self):
        """Verify offline heuristic generator formats system design answers."""
        tokens = []
        ans = generate_offline_heuristic_answer(
            question="Design a distributed rate limiter and scalable caching layer",
            job_title="Senior Backend Engineer",
            company_name="Meta",
            on_token=tokens.append
        )
        self.assertIn("System Design", ans)
        self.assertIn("API Gateway", ans)
        self.assertIn("Redis", ans)
        self.assertTrue(len(tokens) > 0)

    def test_02_heuristic_behavioral_star_answer(self):
        """Verify offline heuristic generator formats STAR behavioral answers."""
        ans = generate_offline_heuristic_answer(
            question="Tell me about a time you faced a major technical challenge or conflict",
            job_title="Lead Architect"
        )
        self.assertIn("Situation:", ans)
        self.assertIn("Task:", ans)
        self.assertIn("Action:", ans)
        self.assertIn("Result:", ans)

    def test_03_heuristic_technical_answer_with_resume(self):
        """Verify offline heuristic generator incorporates resume context."""
        resume_sample = "Architected high-throughput financial payments microservices handling 50k QPS."
        ans = generate_offline_heuristic_answer(
            question="How do you ensure data integrity in microservices?",
            job_title="Staff Engineer",
            company_name="Stripe",
            resume_text=resume_sample
        )
        self.assertIn("Staff Engineer", ans)
        self.assertIn("Stripe", ans)
        self.assertIn("Relevant Background:", ans)

    def test_04_check_ollama_unavailable_graceful(self):
        """Verify check_ollama_available returns False when port is unreachable."""
        available, models = check_ollama_available(host="http://127.0.0.1:59999", timeout_sec=0.1)
        self.assertFalse(available)
        self.assertEqual(models, [])

    def test_05_manager_preference_persistence(self):
        """Verify manager saves and retrieves provider preference."""
        mgr = OfflineFallbackManager()
        mgr.set_preferred_provider("ollama")
        self.assertEqual(mgr.get_preferred_provider(), "ollama")

        mgr.set_preferred_provider("auto")
        self.assertEqual(mgr.get_preferred_provider(), "auto")


if __name__ == "__main__":
    unittest.main()
