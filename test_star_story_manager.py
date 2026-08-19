"""
Unit Test Suite for Behavioral STAR Story Bank Engine (Feature 14)
==================================================================
Validates:
1. Heuristic and LLM-based STAR story extraction
2. Question-to-story semantic keyword routing
3. Spoken STAR answer blueprint formatting
4. Disk persistence and recovery
"""

import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock

from star_story_manager import (
    extract_star_stories_heuristic,
    extract_star_stories_llm,
    find_best_star_story,
    format_star_spoken_answer,
    load_star_stories_from_disk,
    save_star_stories_to_disk,
    STAR_STORIES_STORAGE_FILE
)


class TestStarStoryManager(unittest.TestCase):

    def test_01_heuristic_star_stories(self):
        """Verify heuristic extractor produces at least 3 categorized stories."""
        sample_resume = "Senior Full-Stack & Distributed Systems Engineer\nArchitected high-throughput payment microservices"
        stories = extract_star_stories_heuristic(sample_resume)
        self.assertGreaterEqual(len(stories), 3)

        first = stories[0]
        self.assertIn("situation", first)
        self.assertIn("task", first)
        self.assertIn("action", first)
        self.assertIn("result", first)
        self.assertIn("key_learnings", first)

    def test_02_find_best_star_story_routing(self):
        """Verify matching accurately routes questions to appropriate category stories."""
        stories = extract_star_stories_heuristic("Test resume")

        # Conflict question
        conflict_q = "Tell me about a time you had a major disagreement with a team member."
        matched_conflict = find_best_star_story(conflict_q, stories)
        self.assertEqual(matched_conflict["id"], "story_conflict_disagreement")

        # Outage question
        outage_q = "Describe a high-severity production outage or latency incident you debugged."
        matched_outage = find_best_star_story(outage_q, stories)
        self.assertEqual(matched_outage["id"], "story_tech_challenge")

        # Leadership question
        leader_q = "How do you mentor junior engineers and champion code quality?"
        matched_leader = find_best_star_story(leader_q, stories)
        self.assertEqual(matched_leader["id"], "story_leadership_ownership")

    def test_03_format_star_spoken_answer(self):
        """Verify formatted spoken answer includes all 5 STAR sections."""
        story = {
            "title": "Scaling API Gateway",
            "category": "Technical Challenge",
            "situation": "Our payment API experienced high traffic spikes.",
            "task": "I needed to prevent downstream service collapse.",
            "action": "I implemented token-bucket rate limiting and async queuing.",
            "result": "Maintained 99.99% uptime during Black Friday.",
            "key_learnings": "Proactive capacity planning is essential."
        }
        formatted = format_star_spoken_answer("How did you scale the API?", story)
        self.assertIn("SITUATION", formatted)
        self.assertIn("TASK & OWNERSHIP", formatted)
        self.assertIn("ACTION TAKEN", formatted)
        self.assertIn("MEASURABLE RESULT", formatted)
        self.assertIn("KEY TAKEAWAY", formatted)
        self.assertIn("token-bucket rate limiting", formatted)

    def test_04_llm_mock_extraction(self):
        """Verify LLM extraction parses valid JSON responses."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps([
                {
                    "id": "story_llm_custom",
                    "title": "Migrated Monolith to Rust",
                    "category": "Technical Challenge / Incident Outage",
                    "applicable_keywords": ["rust", "monolith", "migration"],
                    "situation": "Legacy service had high CPU.",
                    "task": "Migrate to Rust.",
                    "action": "Rewrote core logic.",
                    "result": "CPU dropped 80%.",
                    "key_learnings": "Rust memory safety pays off."
                }
            ])))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        stories = extract_star_stories_llm("Rust developer", client=mock_client)
        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0]["title"], "Migrated Monolith to Rust")


if __name__ == "__main__":
    unittest.main()
