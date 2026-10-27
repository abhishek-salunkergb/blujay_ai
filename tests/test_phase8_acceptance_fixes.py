"""
Comprehensive Regression Test Suite for Phase 8 — Acceptance Fixes
===================================================================
Covers the 3 surgical acceptance failure fixes:
1. Small talk & mixed utterances (Logistics vs genuine questions)
2. Duplicate question suppression vs legitimate modified questions
3. Incomplete STT fragment buffering & continuation merging
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
import time
from interview_engine import (
    InterviewSession,
    classify_and_understand_utterance
)


class TestPhase8AcceptanceFixes(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()

    # -------------------------------------------------------------------------
    # 1. SMALL TALK & LOGISTICAL QUESTIONS (NO ANSWER)
    # -------------------------------------------------------------------------
    def test_01_small_talk_and_audio_logistics_filtered(self):
        """Verify conversational small talk and audio setup phrases are ignored."""
        small_talk_cases = [
            "How are you doing today? Can you hear me okay?",
            "Good morning! How are you doing today?",
            "Can you hear me? Can you see my screen?",
            "Are you ready to start?",
            "Is everything okay?",
            "Can you give me a second?"
        ]
        for utterance in small_talk_cases:
            with self.subTest(utterance=utterance):
                turn = classify_and_understand_utterance(utterance, session=self.session, source="system")
                self.assertFalse(
                    turn["should_generate_answer"],
                    f"Small talk '{utterance}' falsely triggered an answer!"
                )
                self.assertIn(turn["question_state"], {"LOW", "NON_QUESTION"})

    # -------------------------------------------------------------------------
    # 2. MIXED UTTERANCES (PRESERVE TECHNICAL QUESTION)
    # -------------------------------------------------------------------------
    def test_02_mixed_utterance_preserves_technical_question(self):
        """Verify mixed small-talk + technical questions strip the lead-in and answer."""
        mixed = "How are you doing today? Can you explain how you designed the API?"
        turn = classify_and_understand_utterance(mixed, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertEqual(turn["question_state"], "HIGH")
        self.assertIn("designed the API", turn["refined_question"])

    # -------------------------------------------------------------------------
    # 3. GENUINE HOW & WHY QUESTIONS (HIGH CONFIDENCE)
    # -------------------------------------------------------------------------
    def test_03_genuine_how_and_why_questions(self):
        """Verify genuine candidate-directed questions generate answers immediately."""
        genuine_questions = [
            "How would you handle authentication?",
            "Why did you choose Django?",
            "How does Django middleware work?",
            "Why did you choose Redis?",
            "Can you explain your approach?"
        ]
        for query in genuine_questions:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertTrue(
                    turn["should_generate_answer"],
                    f"Genuine question '{query}' failed to generate answer!"
                )
                self.assertEqual(turn["question_state"], "HIGH")

    # -------------------------------------------------------------------------
    # 4. DUPLICATE QUESTION SUPPRESSION & PUNCTUATION VARIATION
    # -------------------------------------------------------------------------
    def test_04_duplicate_question_suppression_and_normalization(self):
        """Verify identical repeated questions with punctuation variations are suppressed."""
        q1 = "Write a function to invert a binary tree."
        turn1 = classify_and_understand_utterance(q1, session=self.session, source="system")
        self.assertTrue(turn1["should_generate_answer"])
        self.session.record_qa(q1, "def invertTree(root): ...", category="coding")

        # Exact repeat
        turn2 = classify_and_understand_utterance(q1, session=self.session, source="system")
        self.assertFalse(turn2["should_generate_answer"])
        self.assertEqual(turn2["utterance_type"], "DUPLICATE")

        # Punctuation / casing variation
        q_variation = "write a function to invert a binary tree?"
        turn3 = classify_and_understand_utterance(q_variation, session=self.session, source="system")
        self.assertFalse(turn3["should_generate_answer"])
        self.assertEqual(turn3["utterance_type"], "DUPLICATE")

    # -------------------------------------------------------------------------
    # 5. LEGITIMATE MODIFIED QUESTIONS (NOT SUPPRESSED)
    # -------------------------------------------------------------------------
    def test_05_legitimate_modified_question_not_suppressed(self):
        """Verify legitimate follow-up / modified questions are NOT considered duplicates."""
        q1 = "Write a function to invert a binary tree."
        self.session.record_qa(q1, "def invertTree(root): ...", category="coding")

        modified_q = "Now write it iteratively."
        turn = classify_and_understand_utterance(modified_q, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertNotEqual(turn["utterance_type"], "DUPLICATE")

    # -------------------------------------------------------------------------
    # 6. INCOMPLETE FRAGMENT BUFFERING & CONTINUATION MERGING
    # -------------------------------------------------------------------------
    def test_06_incomplete_fragment_held_and_merged_with_continuation(self):
        """Verify trailing fragment is held in buffer and merged with subsequent continuation."""
        # Chunk 1: Incomplete trailing fragment
        chunk1 = "What if the tree..."
        turn1 = classify_and_understand_utterance(chunk1, session=self.session, source="system")
        self.assertFalse(turn1["should_generate_answer"])
        self.assertEqual(turn1["utterance_type"], "INCOMPLETE_FRAGMENT")
        self.assertEqual(turn1["question_state"], "MEDIUM")
        self.assertTrue(turn1["hold_for_context"])
        self.assertIsNotNone(self.session.pending_medium_question)

        # Chunk 2: Continuation
        chunk2 = "is empty?"
        turn2 = classify_and_understand_utterance(chunk2, session=self.session, source="system")
        self.assertTrue(turn2["should_generate_answer"])
        self.assertEqual(turn2["question_state"], "HIGH")
        self.assertIn("What if the tree is empty", turn2["refined_question"])
        self.assertIsNone(self.session.pending_medium_question)

    # -------------------------------------------------------------------------
    # 7. COMPLETE QUESTION ANSWERS IMMEDIATELY
    # -------------------------------------------------------------------------
    def test_07_complete_question_answers_immediately(self):
        """Verify complete question is answered immediately without buffering."""
        complete_q = "What if the tree is empty?"
        turn = classify_and_understand_utterance(complete_q, session=self.session, source="system")
        self.assertTrue(turn["should_generate_answer"])
        self.assertEqual(turn["question_state"], "HIGH")
        self.assertFalse(turn["hold_for_context"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
